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
from ..io.core import FOOTER_READ_SIZE, Fetched
from ..execution import ExecContext
from ..formats import BatchReader, BatchWriter, Format
from ..schema import Schema
from ..tabular import RecordBatch, Table
from ..compress import Method
from ..utils import Crc32
from .binary import AvroBytes, AvroCursor
from .decoder import RecordDecoder
from .encoder import RecordEncoder
from .mapping import from_arrow, to_arrow
from .schema import AvroSchema
from ..utils.byteorder import BigEndian, LittleEndian

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

    def method(self) raises NotImplementedError -> Method:
        """The method this codec's blocks are compressed with."""
        if self == Self.NULL:
            return Method.NONE
        elif self == Self.DEFLATE:
            return Method.DEFLATE
        elif self == Self.SNAPPY:
            return Method.SNAPPY
        elif self == Self.ZSTANDARD:
            return Method.ZSTD
        raise self._unsupported()

    def decompress(
        self, src: Span[UInt8, _], native: Bool = True
    ) raises -> List[UInt8]:
        """A block's records, uncompressed. Without `native`, libsnappy or
        libzstd decompresses them."""
        var body = src
        if self == Self.SNAPPY:
            # A raw snappy block, then the big-endian CRC-32 of the
            # uncompressed bytes.
            if len(src) < 4:
                raise CorruptError("avro: snappy block without its checksum")
            body = src[: len(src) - 4]
        var out = self.method().decompress_unsized[
            Method.NONE, Method.DEFLATE, Method.SNAPPY, Method.ZSTD
        ](native, body)
        if self == Self.SNAPPY:
            var expected = BigEndian.fixed[DType.uint32](src, len(src) - 4)
            if Crc32.compute(out) != expected:
                raise CorruptError("avro: snappy block checksum mismatch")
        return out^

    def compress(
        self, src: Span[UInt8, _], native: Bool = True
    ) raises -> List[UInt8]:
        """A block's records, compressed. Without `native`, libsnappy or
        libzstd compresses them."""
        var out = List[UInt8]()
        self.method().compress[
            Method.NONE, Method.DEFLATE, Method.SNAPPY, Method.ZSTD
        ](native, src, out)
        if self == Self.SNAPPY:
            BigEndian.append[DType.uint32](out, Crc32.compute(src))
        return out^


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


def _fetch[
    S: ByteSource
](ref source: S, offset: Int, length: Int) raises -> Fetched:
    """One range, owned by the caller: `read_at` would keep every block a
    fetching source returns alive for as long as the source."""
    return source.read_ranges([(offset, length)], ExecContext.serial())


def _read_header[S: ByteSource](ref source: S) raises -> _Header:
    """The header is at the front and its length is not known before it is
    parsed: read 64 KiB, and widen the read while the metadata runs past it."""
    var size = source.size()
    var window = min(size, FOOTER_READ_SIZE)
    var head = _fetch(source, 0, window)
    if (
        len(head.span(0)) < MAGIC.byte_length()
        or head.span(0)[: MAGIC.byte_length()] != MAGIC.as_bytes()
    ):
        raise CorruptError("avro: not an object container file")
    while window < size:
        try:
            return _Header(head.span(0))
        except:
            window = min(size, 4 * window)
        head = _fetch(source, 0, window)
    return _Header(head.span(0))


struct AvroFile[S: ByteSource = BufferSource](Movable):
    """An Avro object container file, opened: its header is read, the blocks
    are not. `read` decodes them into a `Table`, `batches` one batch at a
    time.

    Reads the writer's schema and nothing else -- there is no reader schema to
    resolve against. Columns are selected by name with `read(columns=...)`.
    """

    var _source: Self.S
    var _header: _Header
    var _avro: AvroSchema
    var _codec: AvroCodec
    var _native_codecs: Bool
    """Whether compressed blocks decode in Mojo, the default, or through
    libsnappy and libzstd."""
    var _schema: Schema

    def __init__(
        out self: AvroFile[BufferSource],
        path: String,
        native_codecs: Bool = True,
    ) raises:
        self = AvroFile[BufferSource](BufferSource(path), native_codecs)

    def __init__(
        out self, var source: Self.S, native_codecs: Bool = True
    ) raises:
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
        self._native_codecs = native_codecs
        self._header = header^
        self._source = source^
        self._schema = to_arrow(self._avro)
        self._schema.metadata = self._user_metadata()

    def avro_schema(self) -> AvroSchema:
        """The writer's schema, as written."""
        return self._avro.copy()

    def codec(self) -> AvroCodec:
        return self._codec

    def metadata(self) -> Dict[String, List[UInt8]]:
        """Every header entry, `avro.schema` and `avro.codec` included."""
        return self._header.metadata.copy()

    def schema(self) -> Schema:
        """The Arrow schema `read` produces."""
        return self._schema.copy()

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
        var pos = self._header.size
        while pos < self._source.size():
            self._fill(decoder, pos, batch_size)
            if decoder.rows() > 0:
                batches.append(decoder.finish())
        if len(batches) == 0:
            batches.append(decoder.finish())
        var schema = decoder.schema()
        schema.metadata = self._schema.metadata.copy()
        return Table.from_batches(schema, batches)

    def batches(
        var self,
        columns: Optional[List[String]] = None,
        batch_size: Int = 65_536,
    ) raises -> AvroReader[Self.S]:
        """The file as a stream of batches of about `batch_size` rows, holding
        one block's bytes at a time."""
        return AvroReader(self^, columns, batch_size)

    def _fill(
        self, mut decoder: RecordDecoder, mut pos: Int, batch_size: Int
    ) raises:
        """Decode the blocks from `pos` on until `decoder` holds `batch_size`
        rows or the file ends, leaving `pos` after the last one.

        A block is a record count, a byte size, that many bytes of records and
        the file's sync marker, which is checked. Each is fetched on its own --
        its two varints first, then its bytes and marker -- so only one block is
        held at a time."""
        var size = self._source.size()
        while pos < size and decoder.rows() < batch_size:
            # Two varints take at most ten bytes each.
            var head = _fetch(self._source, pos, min(20, size - pos))
            var cur = AvroCursor(head.span(0))
            var count = cur.length()
            var length = cur.length()
            var start = pos + cur.pos
            var block = _fetch(self._source, start, length + SYNC_SIZE)
            var data = block.span(0)
            if data[length:] != Span(self._header.sync):
                raise CorruptError(
                    t"avro: sync marker mismatch at offset {start + length}"
                )
            var records = self._codec.decompress(
                data[:length], self._native_codecs
            )
            _decode_block(decoder, records, count)
            pos = start + length + SYNC_SIZE


struct AvroReader[S: ByteSource = BufferSource](BatchReader):
    """An Avro file read one batch at a time: each batch gathers blocks until
    it holds `batch_size` rows, and only the block being decoded is in
    memory."""

    var _file: AvroFile[Self.S]
    var _decoder: RecordDecoder
    var _schema: Schema
    var _pos: Int
    var _batch_size: Int

    def __init__(
        out self,
        var file: AvroFile[Self.S],
        columns: Optional[List[String]] = None,
        batch_size: Int = 65_536,
    ) raises:
        self._decoder = RecordDecoder(file._avro, columns)
        self._schema = self._decoder.schema()
        self._schema.metadata = file._schema.metadata.copy()
        self._pos = file._header.size
        self._batch_size = batch_size
        self._file = file^

    def schema(self) -> Schema:
        """The projected columns, with the file's metadata."""
        return self._schema.copy()

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        self._file._fill(self._decoder, self._pos, self._batch_size)
        if self._decoder.rows() == 0:
            return None
        return self._decoder.finish()


struct AvroFormat(Format):
    """Avro object container files, as a scan opens them."""

    comptime Reader = AvroReader[DynSource]

    @staticmethod
    def name() -> StaticString:
        return "Avro"

    @staticmethod
    def open(path: String, schema: Schema) raises -> Self.Reader:
        return AvroFile[DynSource](DynSource.open(path)).batches(schema.names())


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


struct AvroWriter[S: ByteSink = FileSink](BatchWriter):
    """Writes an Avro object container file, batch by batch; `close` finishes
    it.

    Records go into a block until it reaches `sync_interval` bytes, and each
    block is compressed with `codec` on its own.
    """

    var out: BufferedSink[Self.S]
    """Where the file goes; a `MemorySink`'s bytes are `out.sink().bytes()`."""
    var _encoder: RecordEncoder
    var _codec: AvroCodec
    var _native_codecs: Bool
    """Whether blocks compress in Mojo, the default, or through libsnappy and
    libzstd."""
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
        native_codecs: Bool = True,
    ) raises:
        """A file at `path` with the Avro form of `schema`."""
        self = AvroWriter[FileSink](
            FileSink(path), schema, codec, native_codecs
        )

    def __init__(
        out self,
        var sink: Self.S,
        schema: Schema,
        codec: AvroCodec = AvroCodec.DEFLATE,
        native_codecs: Bool = True,
    ) raises:
        """A file in `sink` with the Avro form of `schema`, whose metadata is
        written into the header."""
        self = Self(
            sink^,
            from_arrow(schema),
            codec,
            schema.metadata,
            native_codecs=native_codecs,
        )

    def __init__(
        out self,
        var sink: Self.S,
        avro: AvroSchema,
        codec: AvroCodec = AvroCodec.DEFLATE,
        metadata: Dict[String, String] = {},
        sync_interval: Int = DEFAULT_SYNC_INTERVAL,
        native_codecs: Bool = True,
    ) raises:
        """A file in `sink` of `avro` records, with `metadata` in its header
        beside `avro.schema` and `avro.codec`."""
        if codec == AvroCodec.BZIP2 or codec == AvroCodec.XZ:
            raise codec._unsupported()
        self.out = BufferedSink(sink^)
        self._encoder = RecordEncoder(avro)
        self._codec = codec
        self._native_codecs = native_codecs
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
        var body = self._codec.compress(
            self._block.written(), self._native_codecs
        )
        var head = AvroBytes()
        head.long(Int64(self._rows))
        head.long(Int64(len(body)))
        self.out.write(head.written())
        self.out.write(body)
        self.out.write(self._sync)
        # Avro records no offsets, so a block can leave memory once written.
        self.out.flush()
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
    native_codecs: Bool = True,
) raises -> Table:
    """Read an Avro object container file at `uri` -- a local path or any URI
    `DynSource` opens -- into a `Table`, optionally just `columns`. Without
    `native_codecs`, libsnappy and libzstd decompress the blocks."""
    var f = AvroFile[DynSource](DynSource.open(uri, options), native_codecs)
    return f.read(columns)


def write_avro(
    table: Table,
    uri: String,
    codec: AvroCodec = AvroCodec.DEFLATE,
    options: StorageOptions = StorageOptions(),
    native_codecs: Bool = True,
) raises:
    """Write `table` to `uri` as an Avro object container file. The table's
    schema metadata goes into the file header. Without `native_codecs`,
    libsnappy and libzstd compress the blocks."""
    var w = AvroWriter[DynSink](
        DynSink.open(uri, options), table.schema, codec, native_codecs
    )
    w.write(table)
    w.close()
