# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Reading Arrow IPC files and streams, a record batch at a time."""

from ...errors import CorruptError, IndexError
from ...arrays import DynArray, NullArray
from ...execution import ExecContext
from ...io import (
    BufferSource,
    ByteSource,
    DynSource,
    FOOTER_READ_SIZE,
    StorageOptions,
)
from ...schema import Schema
from ...tabular import RecordBatch
from ...builders import DynBuilder
from ...utils.byteorder import LittleEndian
from ..core import BatchReader, Format
from .format import (
    Block,
    FieldIpcInfo,
    HEADER_DICTIONARY_BATCH,
    HEADER_RECORD_BATCH,
    magic,
)
from .metadata import IpcDecoder


# ---------------------------------------------------------------------------
# IPC framing helpers and framed message reader
# ---------------------------------------------------------------------------


def _read_message[
    S: ByteSource
](
    ref src: S,
    mut pos: Int,
    mut meta: List[UInt8],
    mut body: List[UInt8],
) raises -> Bool:
    """Parse one framed IPC message (continuation + length + metadata + body)
    at `pos`, advancing it past the message. Returns False at end-of-stream.

    A free generic function rather than a method on a `_MessageReader[S]`
    struct: the readers already hold the source and the cursor, and a struct
    whose only job was to pair them added a nested generic instantiation for
    nothing.

    Three ranged reads per message — the frame prefix, the metadata, the body.
    On a memory map those are three `Buffer.view` calls; on a remote source
    they are three round trips, and a caller that knows a message's extent from
    a footer `Block` could ask for it in one. That optimisation is not here.

    The metadata and the body go through `read_ranges`, not `read_at`, and that
    is not a style choice: `read_at` hands back a span borrowed from storage
    the source keeps alive for *its* whole lifetime, so a fetching source
    retains every byte it ever served. A message body is the bulk of the file,
    so reading a 5 GB stream that way would end holding 5 GB. `read_ranges`
    returns a batch this function owns and drops. The 8-byte frame prefix goes
    the same way: it is tiny, but one arena entry *per message* still grows
    with the stream, and "bounded per call" is not the same as bounded.
    """
    var n = src.size()
    if pos + 4 > n:
        return False
    # Every fetch below is one range, so there is nothing to fan out.
    var one = ExecContext.serial()

    # The frame prefix is 4 bytes, or 8 when the continuation marker is
    # present. Ask for as much of it as the source has left.
    var pre_fetched = src.read_ranges([(pos, min(8, n - pos))], one)
    var pre = pre_fetched.span(0)
    var marker = LittleEndian.checked[DType.int32](pre, 0)
    var metadata_len: Int
    var meta_start: Int
    if UInt32(marker) == UInt32(0xFFFFFFFF):
        if pos + 8 > n:
            return False
        metadata_len = Int(LittleEndian.checked[DType.int32](pre, 4))
        meta_start = pos + 8
    else:
        metadata_len = Int(marker)
        meta_start = pos + 4

    if metadata_len == 0:
        pos = meta_start
        return False
    if metadata_len < 0 or meta_start + metadata_len > n:
        raise CorruptError(
            t"IPC: message metadata at {meta_start} runs past the end of a "
            t"{n}-byte source"
        )
    var meta_fetched = src.read_ranges([(meta_start, metadata_len)], one)
    meta.extend(meta_fetched.span(0))

    var raw_end = meta_start + metadata_len
    var meta_end = raw_end + (8 - raw_end % 8) % 8

    var dec = IpcDecoder(meta.copy())
    var body_len = Int(dec.body_length())
    if body_len < 0 or meta_end + body_len > n:
        raise CorruptError(
            t"IPC: message body at {meta_end} runs past the end of a {n}-byte "
            t"source"
        )
    var body_fetched = src.read_ranges([(meta_end, body_len)], one)
    body.extend(body_fetched.span(0))

    pos = meta_end + body_len
    return True


# ---------------------------------------------------------------------------
# Public: file and stream readers
# ---------------------------------------------------------------------------


struct RecordBatchFileReader[S: ByteSource = BufferSource](BatchReader):
    """Reader for the Arrow IPC file format with random-access batch reads.

    Opens a file by reading its **tail**, not its whole extent: the footer sits
    at the end, so one speculative `FOOTER_READ_SIZE` read finds it, and the
    read is repeated only when the footer overruns that window.
    """

    var _schema: Schema
    var _ipc_infos: List[FieldIpcInfo]
    var _blocks: List[Block]
    var _next: Int
    var _src: Self.S
    var _dict_values: List[DynArray]
    var _native_codecs: Bool
    """Whether compressed bodies decode in Mojo, the default, or through
    liblz4 and libzstd."""

    def __init__(
        out self: RecordBatchFileReader[BufferSource],
        path: String,
        native_codecs: Bool = True,
    ) raises:
        """Open a local file as a memory map — the convenience that pins
        `S == BufferSource`."""
        self = RecordBatchFileReader[BufferSource](
            BufferSource(path), native_codecs
        )

    def __init__(
        out self, var source: Self.S, native_codecs: Bool = True
    ) raises:
        self._native_codecs = native_codecs
        var n = source.size()
        if n < 14:
            raise CorruptError("IPC file too short")
        var magic = magic()

        var head = source.read_at(0, 8)
        for i in range(8):
            if head[i] != magic[i]:
                raise CorruptError("IPC file: bad magic bytes")

        # One tail read serves the trailing magic, the footer length and, all
        # but always, the footer itself.
        var want = min(FOOTER_READ_SIZE, n)
        var tail = source.read_at(n - want, want)
        for i in range(6):
            if tail[want - 6 + i] != magic[i]:
                raise CorruptError("IPC file: bad trailing magic")

        var footer_size = Int(
            LittleEndian.checked[DType.int32](tail, want - 10)
        )
        if footer_size < 0 or footer_size + 10 > n:
            raise CorruptError(t"IPC file: bad footer length {footer_size}")

        var footer_bytes: List[UInt8]
        if footer_size + 10 <= want:
            var at = want - 10 - footer_size
            footer_bytes = List[UInt8](tail[at : at + footer_size])
        else:
            # The speculative window was too small; ask for exactly the footer.
            var start = n - 10 - footer_size
            footer_bytes = List[UInt8](source.read_at(start, footer_size))

        var dec = IpcDecoder(footer_bytes^)
        var dict_blocks = List[Block]()
        var blocks = List[Block]()
        var ipc_infos = List[FieldIpcInfo]()
        self._schema = dec.read_footer(dict_blocks, blocks, ipc_infos)
        self._ipc_infos = ipc_infos^
        self._blocks = blocks^
        self._next = 0
        self._src = source^
        self._dict_values = List[DynArray]()

        # Load dictionary values from their footer-registered blocks.
        # dict_values is indexed by dict_id; pass partial list to decode_dict_batch
        # so that nested dicts (already loaded at lower ids) can be resolved.
        for di in range(len(dict_blocks)):
            var pos = Int(dict_blocks[di].offset)
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not _read_message(self._src, pos, meta, body):
                break
            var dict_id = IpcDecoder(meta.copy()).peek_dict_id()
            var lkup = FieldIpcInfo.find_in_schema(
                self._schema.fields, self._ipc_infos, dict_id
            )
            if lkup:
                var dec = IpcDecoder(meta^)
                var values = dec.decode_dict_batch(
                    lkup.value().value_type,
                    lkup.value().value_ipc_info,
                    body^,
                    self._dict_values,
                    self._native_codecs,
                )
                while len(self._dict_values) <= dict_id:
                    self._dict_values.append(NullArray(0))
                self._dict_values[dict_id] = values^

    def schema(self) -> Schema:
        return self._schema.copy()

    def num_record_batches(self) -> Int:
        return len(self._blocks)

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        """The batch after the last one this returned, in file order, or
        `None` after the last; `read_batch` is the random-access path."""
        if self._next >= len(self._blocks):
            return None
        self._next += 1
        return self.read_batch(self._next - 1)

    def read_batch(ref self, i: Int) raises -> RecordBatch:
        """One batch, by index. `ref self` rather than `mut self`: this format is
        random-access -- the footer's `Block` table holds every offset -- so the
        cursor is a local, and several readers of one file do not contend."""
        if i < 0 or i >= len(self._blocks):
            raise IndexError("RecordBatchFileReader: batch index out of range")
        var pos = Int(self._blocks[i].offset)
        var meta = List[UInt8]()
        var body = List[UInt8]()
        var _ok = _read_message(self._src, pos, meta, body)
        var dec = IpcDecoder(meta^)
        return dec.decode_record_batch(
            self._schema,
            self._ipc_infos,
            body^,
            self._dict_values,
            self._native_codecs,
        )

    def read_all(mut self) raises -> List[RecordBatch]:
        # Footer only lists record-batch blocks, so no header check needed.
        var batches = List[RecordBatch]()
        for i in range(len(self._blocks)):
            batches.append(self.read_batch(i))
        return batches^


struct RecordBatchStreamReader[S: ByteSource = BufferSource](BatchReader):
    """Reader for the Arrow IPC stream format.

    Framing stays strictly sequential, because a stream carries no index. That
    is honest but not ideal over an object store, which wants a prefetch window
    rather than a round trip per message; the seam admits such a source, and
    this reader does not add one.
    """

    var _schema: Schema
    var _ipc_infos: List[FieldIpcInfo]
    var _src: Self.S
    var _pos: Int
    var _dict_values: List[DynArray]
    """Every dictionary read so far, by id, deltas applied."""
    var _native_codecs: Bool
    """Whether compressed bodies decode in Mojo, the default, or through
    liblz4 and libzstd."""

    def __init__(
        out self: RecordBatchStreamReader[BufferSource],
        path: String,
        native_codecs: Bool = True,
    ) raises:
        """Open a local file as a memory map — the convenience that pins
        `S == BufferSource`."""
        self = RecordBatchStreamReader[BufferSource](
            BufferSource(path), native_codecs
        )

    def __init__(
        out self, var source: Self.S, native_codecs: Bool = True
    ) raises:
        self._native_codecs = native_codecs
        self._src = source^
        self._pos = 0
        self._dict_values = List[DynArray]()
        var meta = List[UInt8]()
        var body = List[UInt8]()
        if not _read_message(self._src, self._pos, meta, body):
            raise CorruptError(
                "RecordBatchStreamReader: missing schema message"
            )
        var ipc_infos = List[FieldIpcInfo]()
        var dec = IpcDecoder(meta^)
        self._schema = dec.decode_schema(ipc_infos)
        self._ipc_infos = ipc_infos^

    def schema(self) -> Schema:
        return self._schema.copy()

    def read_next_batch(mut self) raises -> Optional[RecordBatch]:
        """The next record batch, applying the dictionary messages before it,
        or `None` at the end of the stream."""
        while True:
            var meta = List[UInt8]()
            var body = List[UInt8]()
            if not _read_message(self._src, self._pos, meta, body):
                return None
            var header_type = IpcDecoder(meta.copy()).peek_header()
            if Int(header_type) == Int(HEADER_DICTIONARY_BATCH):
                var dict_id = IpcDecoder(meta.copy()).peek_dict_id()
                var is_delta = IpcDecoder(meta.copy()).peek_is_delta()
                var lkup = FieldIpcInfo.find_in_schema(
                    self._schema.fields, self._ipc_infos, dict_id
                )
                if lkup:
                    var dec = IpcDecoder(meta^)
                    var values = dec.decode_dict_batch(
                        lkup.value().value_type,
                        lkup.value().value_ipc_info,
                        body^,
                        self._dict_values,
                        self._native_codecs,
                    )
                    while len(self._dict_values) <= dict_id:
                        self._dict_values.append(NullArray(0))
                    if is_delta and len(self._dict_values[dict_id]) > 0:
                        # a delta carries only the newly-appended values
                        var merged = List[DynArray]()
                        merged.append(self._dict_values[dict_id].copy())
                        merged.append(values^)
                        self._dict_values[dict_id] = DynBuilder.concat(merged)
                    else:
                        self._dict_values[dict_id] = values^
            elif Int(header_type) == Int(HEADER_RECORD_BATCH):
                var dec = IpcDecoder(meta^)
                return dec.decode_record_batch(
                    self._schema,
                    self._ipc_infos,
                    body^,
                    self._dict_values,
                    self._native_codecs,
                )

    def read_all(mut self) raises -> List[RecordBatch]:
        var batches = List[RecordBatch]()
        while True:
            var batch = self.read_next_batch()
            if not batch:
                break
            batches.append(batch.take())
        return batches^


struct IpcFormat(Format):
    """The Arrow IPC file format, read by `RecordBatchFileReader`. A batch
    holds whole columns, so the projection reads nothing less."""

    comptime Reader = RecordBatchFileReader[DynSource]

    @staticmethod
    def name() -> StaticString:
        return "Ipc"

    @staticmethod
    def open(path: String, schema: Schema) raises -> Self.Reader:
        return RecordBatchFileReader[DynSource](DynSource.open(path))


def read_ipc_file(
    uri: String,
    options: StorageOptions = StorageOptions(),
    native_codecs: Bool = True,
) raises -> List[RecordBatch]:
    """Read an Arrow IPC file and return all RecordBatches; compressed
    bodies decode in Mojo, or with `native_codecs=False` through liblz4 and
    libzstd."""
    var r = RecordBatchFileReader(DynSource.open(uri, options), native_codecs)
    return r.read_all()


def read_ipc_stream(
    uri: String,
    options: StorageOptions = StorageOptions(),
    native_codecs: Bool = True,
) raises -> List[RecordBatch]:
    """Read an Arrow IPC stream and return all RecordBatches; compressed
    bodies decode as `read_ipc_file` says."""
    var r = RecordBatchStreamReader(DynSource.open(uri, options), native_codecs)
    return r.read_all()


def read_ipc_file_schema(
    uri: String, options: StorageOptions = StorageOptions()
) raises -> RecordBatch:
    """Read the schema from an Arrow IPC file; return a 0-row RecordBatch."""
    var r = RecordBatchFileReader(DynSource.open(uri, options))
    return RecordBatch.empty(r.schema())


def read_ipc_stream_schema(
    uri: String, options: StorageOptions = StorageOptions()
) raises -> RecordBatch:
    """Read the schema from an Arrow IPC stream; return a 0-row RecordBatch."""
    var r = RecordBatchStreamReader(DynSource.open(uri, options))
    return RecordBatch.empty(r.schema())
