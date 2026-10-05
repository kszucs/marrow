# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro object container files.

A file is a header -- the magic `Obj\\x01`, a `map<bytes>` of metadata holding
the writer's schema (`avro.schema`) and codec (`avro.codec`), and a 16-byte
sync marker -- followed by blocks: a record count, a byte size, that many bytes
of records compressed with the codec, and the sync marker again.

Metadata keys outside the `avro.` namespace are the file's own -- Iceberg keeps
its table schema and partition spec there -- and travel as the Arrow schema's
metadata: `AvroFile.schema()` carries them and `write_avro` writes the table
schema's back out.
"""

from std.random import random_ui64

from ..errors import CorruptError, InvalidError, NotImplementedError
from ..io import (
    BufferSource,
    BufferedSink,
    ByteSink,
    ByteSource,
    DynSink,
    DynSource,
    FileSink,
    StorageOptions,
)
from ..io.core import FOOTER_READ_SIZE
from ..schema import Schema
from ..tabular import RecordBatch, Table
from ..utils import (
    BigEndian,
    CompressionLibs,
    Crc32,
    LittleEndian,
    Snappy,
    Zstd,
)
from .binary import AvroBytes, AvroCursor
from .decoder import RecordDecoder
from .encoder import RecordEncoder
from .mapping import from_arrow, to_arrow
from .schema import AvroSchema

comptime MAGIC = "Obj\x01"
comptime SCHEMA_KEY = "avro.schema"
comptime CODEC_KEY = "avro.codec"
comptime SYNC_SIZE = 16
# Java's `DataFileWriter` default.
comptime DEFAULT_SYNC_INTERVAL = 64_000
"""Bytes of encoded records per block before the writer starts a new one."""


@fieldwise_init
struct AvroCodec(Equatable, ImplicitlyCopyable, Movable, Writable):
    """A block codec, by its `avro.codec` name."""

    var code: Int

    comptime NULL = Self(0)
    comptime DEFLATE = Self(1)
    comptime SNAPPY = Self(2)
    comptime ZSTANDARD = Self(3)
    comptime BZIP2 = Self(4)
    comptime XZ = Self(5)

    @staticmethod
    def parse(name: StringSlice) raises -> Self:
        for i in range(6):
            if name == Self(i).name():
                return Self(i)
        raise NotImplementedError(t"avro: unknown codec '{name}'")

    def name(self) -> StaticString:
        var c = self.code
        if c == 0:
            return "null"
        elif c == 1:
            return "deflate"
        elif c == 2:
            return "snappy"
        elif c == 3:
            return "zstandard"
        elif c == 4:
            return "bzip2"
        return "xz"

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.name())

    def _unsupported(self) -> NotImplementedError:
        return NotImplementedError(t"avro: the {self} codec is not supported")

    def decompress(self, src: Span[UInt8, _]) raises -> List[UInt8]:
        """A block's records, uncompressed."""
        if self == Self.NULL:
            return List[UInt8](src)
        elif self == Self.DEFLATE:
            return CompressionLibs.deflate_decompress(src)
        elif self == Self.ZSTANDARD:
            return CompressionLibs.zstd_decompress_unsized(src)
        elif self == Self.SNAPPY:
            # A raw snappy block, then the big-endian CRC-32 of the
            # uncompressed bytes.
            if len(src) < 4:
                raise CorruptError("avro: snappy block without its checksum")
            var body = src[: len(src) - 4]
            var out = List[UInt8]()
            Snappy.decompress(body, out)
            var expected = BigEndian.u32(src, len(src) - 4)
            if Crc32.compute(out) != expected:
                raise CorruptError("avro: snappy block checksum mismatch")
            return out^
        raise self._unsupported()

    def compress(self, src: Span[UInt8, _]) raises -> List[UInt8]:
        """A block's records, compressed."""
        if self == Self.NULL:
            return List[UInt8](src)
        elif self == Self.DEFLATE:
            return CompressionLibs.deflate_compress(src)
        elif self == Self.ZSTANDARD:
            var out = List[UInt8]()
            Zstd.compress(src, out)
            return out^
        elif self == Self.SNAPPY:
            var out = List[UInt8]()
            Snappy.compress(src, out)
            BigEndian.put_u32(out, Crc32.compute(src))
            return out^
        raise self._unsupported()


struct _Header(Movable):
    var metadata: Dict[String, List[UInt8]]
    var sync: List[UInt8]
    var size: Int
    """Bytes the header takes; the first block starts here."""

    def __init__(out self, data: Span[UInt8, _]) raises:
        var cur = AvroCursor(data, MAGIC.byte_length())
        self.metadata = Dict[String, List[UInt8]]()
        while True:
            var count = cur.block()[0]
            if count == 0:
                break
            for _ in range(count):
                var key = String(from_utf8=cur.bytes())
                self.metadata[key^] = List[UInt8](cur.bytes())
        self.sync = List[UInt8](cur.fixed(SYNC_SIZE))
        self.size = cur.pos


def _read_header[S: ByteSource](ref source: S) raises -> _Header:
    """The header is at the front and its length is not known before it is
    parsed: read 64 KiB, and widen the read while the metadata runs past it."""
    var size = source.size()
    var window = min(size, FOOTER_READ_SIZE)
    var head = source.read_at(0, window)
    if (
        len(head) < MAGIC.byte_length()
        or head[: MAGIC.byte_length()] != MAGIC.as_bytes()
    ):
        raise CorruptError("avro: not an object container file")
    while window < size:
        try:
            return _Header(head)
        except:
            window = min(size, 4 * window)
        head = source.read_at(0, window)
    return _Header(head)


struct AvroFile[S: ByteSource = BufferSource](Movable):
    """An Avro object container file, opened: its header is read, the blocks
    are not. `read` decodes them into a `Table`.

    Reads the writer's schema and nothing else -- there is no reader schema to
    resolve against. Columns are selected by name with `read(columns=...)`.
    """

    var _source: Self.S
    var _header: _Header
    var _avro: AvroSchema
    var _codec: AvroCodec

    def __init__(out self: AvroFile[BufferSource], path: String) raises:
        self = AvroFile[BufferSource](BufferSource(path))

    def __init__(out self, var source: Self.S) raises:
        var header = _read_header(source)
        var schema = header.metadata.get(SCHEMA_KEY)
        if not schema:
            raise CorruptError("avro: file has no avro.schema")
        self._avro = AvroSchema.parse(
            StringSlice(from_utf8=Span(schema.value()))
        )
        var codec = header.metadata.get(CODEC_KEY)
        self._codec = AvroCodec.NULL
        if codec:
            self._codec = AvroCodec.parse(
                StringSlice(from_utf8=Span(codec.value()))
            )
        self._header = header^
        self._source = source^

    def avro_schema(self) -> AvroSchema:
        """The writer's schema, as written."""
        return self._avro.copy()

    def codec(self) -> AvroCodec:
        return self._codec

    def metadata(self) -> Dict[String, List[UInt8]]:
        """Every header entry, `avro.schema` and `avro.codec` included."""
        return self._header.metadata.copy()

    def schema(self) raises -> Schema:
        """The Arrow schema `read` produces."""
        var schema = to_arrow(self._avro)
        schema.metadata = self._user_metadata()
        return schema^

    def _user_metadata(self) -> Dict[String, String]:
        """The header's entries outside the `avro.` namespace whose values are
        UTF-8 -- the Arrow schema's metadata."""
        var metadata = Dict[String, String]()
        for entry in self._header.metadata.items():
            if entry.key.startswith("avro."):
                continue
            try:
                metadata[entry.key] = String(from_utf8=Span(entry.value))
            except:
                pass
        return metadata^

    def read(
        self,
        columns: Optional[List[String]] = None,
        batch_size: Int = 65_536,
    ) raises -> Table:
        """Decode every block. Blocks are gathered into batches of about
        `batch_size` rows; a block is never split, so a batch holds at least
        one."""
        var decoder = RecordDecoder(self._avro, columns)
        var batches = List[RecordBatch]()
        var start = self._header.size
        var blocks = _Blocks(
            self._source.read_at(start, self._source.size() - start),
            self._header.sync,
            start,
        )
        while blocks.next():
            var records = self._codec.decompress(blocks.data)
            _decode_block(decoder, records, blocks.count)
            if decoder.rows() >= batch_size:
                batches.append(decoder.finish())
        if decoder.rows() > 0 or len(batches) == 0:
            batches.append(decoder.finish())
        var schema = decoder.schema()
        schema.metadata = self._user_metadata()
        return Table.from_batches(schema, batches)


struct _Blocks[origin: ImmOrigin, sync_origin: ImmOrigin](Movable):
    """The blocks after a header, in order: each a record count, a byte size,
    that many bytes of records and the file's sync marker, which is
    checked."""

    var _cur: AvroCursor[Self.origin]
    var _sync: Span[UInt8, Self.sync_origin]
    var _start: Int
    """Where the blocks start in the file, for error messages."""
    var count: Int
    var data: Span[UInt8, Self.origin]
    """The current block's records, still compressed."""

    def __init__(
        out self,
        body: Span[UInt8, Self.origin],
        sync: Span[UInt8, Self.sync_origin],
        start: Int,
    ):
        self._cur = AvroCursor(body)
        self._sync = sync
        self._start = start
        self.count = 0
        self.data = body[:0]

    def next(mut self) raises -> Bool:
        """Move to the next block; False at the end of the file."""
        if self._cur.remaining() == 0:
            return False
        self.count = self._cur.length()
        self.data = self._cur.fixed(self._cur.length())
        if self._cur.fixed(SYNC_SIZE) != self._sync:
            raise CorruptError(
                t"avro: sync marker mismatch at offset"
                t" {self._start + self._cur.pos - SYNC_SIZE}"
            )
        return True


def _decode_block(
    mut decoder: RecordDecoder, data: Span[UInt8, _], count: Int
) raises:
    var cur = AvroCursor(data)
    decoder.decode(cur, count)
    if cur.remaining() != 0:
        raise CorruptError(
            t"avro: block of {count} records has {cur.remaining()} bytes left"
            t" over"
        )


struct AvroWriter[S: ByteSink = FileSink](Movable):
    """Writes an Avro object container file, batch by batch; `close` finishes
    it.

    Records go into a block until it reaches `sync_interval` bytes, and each
    block is compressed with `codec` on its own.
    """

    var out: BufferedSink[Self.S]
    """Where the file goes; a `MemorySink`'s bytes are `out.sink().bytes()`."""
    var _encoder: RecordEncoder
    var _codec: AvroCodec
    var _sync: List[UInt8]
    var _block: AvroBytes
    var _rows: Int
    var _sync_interval: Int
    var _closed: Bool

    def __init__(
        out self: AvroWriter[FileSink],
        path: String,
        schema: Schema,
        codec: AvroCodec = AvroCodec.DEFLATE,
    ) raises:
        """A file at `path` with the Avro form of `schema`."""
        self = AvroWriter[FileSink](FileSink(path), schema, codec)

    def __init__(
        out self,
        var sink: Self.S,
        schema: Schema,
        codec: AvroCodec = AvroCodec.DEFLATE,
    ) raises:
        """A file in `sink` with the Avro form of `schema`, whose metadata is
        written into the header."""
        self = Self(sink^, from_arrow(schema), codec, schema.metadata)

    def __init__(
        out self,
        var sink: Self.S,
        avro: AvroSchema,
        codec: AvroCodec = AvroCodec.DEFLATE,
        metadata: Dict[String, String] = {},
        sync_interval: Int = DEFAULT_SYNC_INTERVAL,
    ) raises:
        """A file in `sink` of `avro` records, with `metadata` in its header
        beside `avro.schema` and `avro.codec`."""
        if codec == AvroCodec.BZIP2 or codec == AvroCodec.XZ:
            raise codec._unsupported()
        self.out = BufferedSink(sink^)
        self._encoder = RecordEncoder(avro)
        self._codec = codec
        self._sync = List[UInt8](capacity=SYNC_SIZE)
        for _ in range(SYNC_SIZE // 8):
            LittleEndian.append[DType.uint64](
                self._sync, random_ui64(0, UInt64.MAX)
            )
        self._block = AvroBytes()
        self._rows = 0
        self._sync_interval = sync_interval
        self._closed = False
        var header = AvroBytes()
        header.fixed(MAGIC.as_bytes())
        for key in metadata:
            if key.startswith("avro."):
                raise InvalidError(t"avro: metadata key '{key}' is reserved")
        header.long(Int64(len(metadata) + 2))
        header.bytes(StringSlice(SCHEMA_KEY).as_bytes())
        header.bytes(avro.to_json().as_bytes())
        header.bytes(StringSlice(CODEC_KEY).as_bytes())
        header.bytes(codec.name().as_bytes())
        for entry in metadata.items():
            header.bytes(entry.key.as_bytes())
            header.bytes(entry.value.as_bytes())
        header.long(0)
        header.fixed(self._sync)
        self.out.write(header.written())

    def write(mut self, batch: RecordBatch) raises:
        self._encoder.bind(batch)
        for row in range(batch.num_rows()):
            self._encoder.encode(row, self._block)
            self._rows += 1
            if len(self._block) >= self._sync_interval:
                self._flush()

    def write(mut self, table: Table) raises:
        for ref batch in table.to_batches():
            self.write(batch)

    def _flush(mut self) raises:
        if self._rows == 0:
            return
        # A block: its record count, its byte size, the bytes, the sync marker.
        var body = self._codec.compress(self._block.written())
        var head = AvroBytes()
        head.long(Int64(self._rows))
        head.long(Int64(len(body)))
        self.out.write(head.written())
        self.out.write(body)
        self.out.write(self._sync)
        self._block.clear()
        self._rows = 0

    def close(mut self) raises:
        """Write the last block and commit the file. Idempotent."""
        if self._closed:
            return
        self._flush()
        self.out.close()
        self._closed = True


def read_avro(
    uri: String,
    columns: Optional[List[String]] = None,
    options: StorageOptions = StorageOptions(),
) raises -> Table:
    """Read an Avro object container file at `uri` -- a local path or any URI
    `DynSource` opens -- into a `Table`, optionally just `columns`."""
    var f = AvroFile[DynSource](DynSource.open(uri, options))
    return f.read(columns)


def write_avro(
    table: Table,
    uri: String,
    codec: AvroCodec = AvroCodec.DEFLATE,
    options: StorageOptions = StorageOptions(),
) raises:
    """Write `table` to `uri` as an Avro object container file. The table's
    schema metadata goes into the file header."""
    var w = AvroWriter[DynSink](DynSink.open(uri, options), table.schema, codec)
    w.write(table)
    w.close()
