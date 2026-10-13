# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Writing Arrow IPC files and streams, a record batch at a time."""

from ...errors import InvalidError
from ...arrays import DynArray
from ...io import BufferedSink, ByteSink, DynSink, FileSink, StorageOptions
from ...schema import Schema
from ...tabular import RecordBatch
from ...kernels.hashing import KeyCompare
from ...utils.byteorder import LittleEndian
from ..core import BatchWriter
from .body import BatchEncoder, DictPair
from .compression import BodyCompression
from .format import Block, magic, pad_to
from .metadata import IpcEncoder


# ---------------------------------------------------------------------------
# Public: incremental file and stream writers
# ---------------------------------------------------------------------------


struct RecordBatchFileWriter[S: ByteSink = FileSink](BatchWriter):
    """Incremental writer for the Arrow IPC file format.

    Write batches with `write`, then call `close()` to append the footer
    and commit. Nothing is published until `close()` succeeds.

    Batches are handed to the sink as they are written, so peak residency is
    one message rather than the whole file. That is safe because the only
    absolute offsets this format records are the `Block` positions below,
    which come from `tell()`, while an encoded message is self-contained.
    """

    var _out: BufferedSink[Self.S]
    var _schema: Schema
    var _dict_blocks: List[Block]
    var _blocks: List[Block]
    var _enc: BatchEncoder
    var _dicts: List[DynArray]
    var _closed: Bool

    def __init__(
        out self: RecordBatchFileWriter[FileSink],
        path: String,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """Write to a local file — the convenience that pins `S == FileSink`."""
        self = RecordBatchFileWriter[FileSink](
            FileSink(path), schema, compression, native_codecs
        )

    def __init__(
        out self,
        var sink: Self.S,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """`compression` compresses every body buffer, dictionaries'
        included -- in Mojo, or with `native_codecs=False` through liblz4
        and libzstd."""
        self._out = BufferedSink(sink^)
        self._schema = Schema(copy=schema)
        self._dict_blocks = List[Block]()
        self._blocks = List[Block]()
        self._enc = BatchEncoder(compression, native_codecs)
        self._dicts = []
        self._closed = False

        for b in magic():
            self._out.buffer().append(b)
        var schema_msg = IpcEncoder.frame_message(
            IpcEncoder.encode_schema_message(self._schema), List[UInt8]()
        )
        self._out.write(Span(schema_msg))

    def write(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise InvalidError("RecordBatchFileWriter: writer is closed")
        # Collect all (dict_id, values) pairs in DFS inner-first order; ids
        # are dense, so `_dicts[id]` is what was written for an id. The FILE
        # format holds one dictionary per id and this writer emits no deltas,
        # so a later batch must carry the same dictionary.
        var pairs = List[DictPair]()
        var next_id = 0
        for col in batch.columns:
            BatchEncoder.collect_dict_pairs(col.to_data(), pairs, next_id)
        for ref pair in pairs:
            var did = pair.dict_id
            if did < len(self._dicts):
                if not KeyCompare.equals(self._dicts[did], pair.values):
                    raise InvalidError(
                        t"RecordBatchFileWriter: dictionary {did} differs from"
                        t" the one already written"
                    )
                continue
            var dict_blk_start = Int64(self._out.tell())
            var eb = self._enc.encode_dict_message(Int64(did), pair.values)
            self._out.write(Span(eb.msg))
            self._dict_blocks.append(
                Block(dict_blk_start, eb.metadata_length, eb.body_length)
            )
            self._dicts.append(pair.values.copy())
        var blk_start = Int64(self._out.tell())
        var eb = self._enc.encode(batch)
        self._out.write(Span(eb.msg))
        self._blocks.append(
            Block(blk_start, eb.metadata_length, eb.body_length)
        )
        # One message resident at a time. `tell()` stays absolute across this,
        # which is what keeps the `Block` offsets above meaningful.
        self._out.flush()

    def close(mut self) raises:
        if self._closed:
            return
        self._out.pad_to(8)
        var footer_bytes = IpcEncoder.encode_footer(
            self._schema, self._dict_blocks, self._blocks
        )
        self._out.write(Span(footer_bytes))
        LittleEndian.append[DType.int32](
            self._out.buffer(), Int32(len(footer_bytes))
        )
        var magic = magic()
        for i in range(6):
            self._out.buffer().append(magic[i])
        self._out.close()
        self._closed = True


struct RecordBatchStreamWriter[S: ByteSink = FileSink](BatchWriter):
    """Incremental writer for the Arrow IPC stream format.

    Write batches with `write`, then call `close()` to write the EOS
    marker and commit. A stream records no offsets at all, so each message goes
    to the sink as soon as it is encoded.
    """

    var _out: BufferedSink[Self.S]
    var _enc: BatchEncoder
    var _closed: Bool

    def __init__(
        out self: RecordBatchStreamWriter[FileSink],
        path: String,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """Write to a local file — the convenience that pins `S == FileSink`."""
        self = RecordBatchStreamWriter[FileSink](
            FileSink(path), schema, compression, native_codecs
        )

    def __init__(
        out self,
        var sink: Self.S,
        schema: Schema,
        compression: Optional[BodyCompression] = None,
        native_codecs: Bool = True,
    ) raises:
        """`compression` compresses every body buffer, dictionaries'
        included -- in Mojo, or with `native_codecs=False` through liblz4
        and libzstd."""
        self._out = BufferedSink(sink^)
        self._enc = BatchEncoder(compression, native_codecs)
        self._closed = False

        var schema_msg = IpcEncoder.frame_message(
            IpcEncoder.encode_schema_message(schema),
            List[UInt8](),
        )
        self._out.write(Span(schema_msg))

    def write(mut self, batch: RecordBatch) raises:
        if self._closed:
            raise InvalidError("RecordBatchStreamWriter: writer is closed")
        # Stream format sends all dicts before each record batch.
        var pairs = List[DictPair]()
        var next_id = 0
        for col in batch.columns:
            BatchEncoder.collect_dict_pairs(col.to_data(), pairs, next_id)
        for j in range(len(pairs)):
            var eb = self._enc.encode_dict_message(
                Int64(pairs[j].dict_id), pairs[j].values
            )
            self._out.write(Span(eb.msg))
        self._out.write(Span(self._enc.encode(batch).msg))
        self._out.flush()

    def close(mut self) raises:
        if self._closed:
            return
        LittleEndian.append[DType.uint32](
            self._out.buffer(), UInt32(0xFFFFFFFF)
        )
        LittleEndian.append[DType.int32](self._out.buffer(), Int32(0))
        self._out.close()
        self._closed = True


# ---------------------------------------------------------------------------
# Public top-level functions
# ---------------------------------------------------------------------------


def write_ipc_file(
    uri: String,
    schema: Schema,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC file with an explicit schema;
    `compression` runs in Mojo, or with `native_codecs=False` through liblz4
    and libzstd."""
    var w = RecordBatchFileWriter(
        DynSink.open(uri, options), schema, compression, native_codecs
    )
    for batch in batches:
        w.write(batch)
    w.close()


def write_ipc_file(
    uri: String,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC file."""
    if len(batches) == 0:
        raise InvalidError(
            "write_ipc_file: no batches; use write_ipc_file(path, schema, "
            "batches) for schema-only files"
        )
    write_ipc_file(
        uri, batches[0].schema, batches, options, compression, native_codecs
    )


def write_ipc_stream(
    uri: String,
    schema: Schema,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC stream with an explicit schema;
    `compression` runs as `write_ipc_file` says."""
    var w = RecordBatchStreamWriter(
        DynSink.open(uri, options), schema, compression, native_codecs
    )
    for batch in batches:
        w.write(batch)
    w.close()


def write_ipc_stream(
    uri: String,
    batches: List[RecordBatch],
    options: StorageOptions = StorageOptions(),
    compression: Optional[BodyCompression] = None,
    native_codecs: Bool = True,
) raises:
    """Write RecordBatches to an Arrow IPC stream."""
    if len(batches) == 0:
        raise InvalidError(
            "write_ipc_stream: no batches; use write_ipc_stream(path, schema, "
            "batches) for schema-only streams"
        )
    write_ipc_stream(
        uri, batches[0].schema, batches, options, compression, native_codecs
    )
