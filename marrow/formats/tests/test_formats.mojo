# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every format through the one protocol: what a `BatchWriter` writes, the
format's `BatchReader` reads back batch for batch, and the bytes reach the sink
as each batch is written rather than at `close`."""

from std.memory import ArcPointer
from std.os.path import join
from std.python import Python
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...arrays import DynArray
from ..avro import AvroFile, AvroFormat, AvroWriter
from ...builders import array
from ...dtypes import int64
from ...execution import ExecContext
from ...io import BufferSource, ByteSink, ByteSource, Fetched, FileSink
from ..ipc import (
    IpcFormat,
    RecordBatchFileReader,
    RecordBatchFileWriter,
    RecordBatchStreamReader,
    RecordBatchStreamWriter,
)
from ..json import JsonFormat, JsonReader, JsonWriter
from ..parquet import ParquetFile, ParquetFormat, ParquetReader, read_table
from ..parquet.compression import Compression
from ..parquet.writer import FileWriter
from ...schema import Schema
from ...tabular import RecordBatch, record_batch
from ...utils.testing import ScratchDir
from ..core import BatchReader, BatchWriter, Format


comptime Writes = ArcPointer[List[Int]]
"""The length of every `write` a sink received, in order; `-1` marks `close`."""


struct _Tap(ByteSink):
    """A `FileSink` that remembers when bytes reached it."""

    var _inner: FileSink
    var _writes: Writes

    def __init__(out self, path: String, var writes: Writes) raises:
        self._inner = FileSink(path)
        self._writes = writes^

    def write[o: Origin[mut=False]](mut self, data: Span[UInt8, o]) raises:
        self._writes[].append(len(data))
        self._inner.write(data)

    def close(mut self) raises:
        self._writes[].append(-1)
        self._inner.close()


comptime Reads = ArcPointer[List[Tuple[Int, Int]]]
"""Every `(offset, length)` a source was asked for, in order."""


struct _Recorder(ByteSource):
    """A `BufferSource` that remembers what was read through it."""

    var _inner: BufferSource
    var _reads: Reads

    def __init__(out self, path: String, var reads: Reads) raises:
        self._inner = BufferSource(path)
        self._reads = reads^

    def size(self) -> Int:
        return self._inner.size()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        self._reads[].append((offset, length))
        return rebind[Span[UInt8, origin_of(self)]](
            self._inner.read_at(offset, length)
        )

    def read_ranges(
        ref self, ranges: List[Tuple[Int, Int]], ctx: ExecContext
    ) raises -> Fetched:
        for ref r in ranges:
            self._reads[].append((r[0], r[1]))
        return self._inner.read_ranges(ranges, ctx)


def _largest_read(reads: Reads) -> Int:
    var largest = 0
    for ref r in reads[]:
        largest = max(largest, r[1])
    return largest


def _write_parquet(path: String, row_group_size: Int) raises:
    """Nine rows of `n: int64, s: string` through pyarrow, in row groups of
    `row_group_size`."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var table = pa.table(
        Python.dict(
            n=Python.list(0, 1, 2, 3, 4, 5, 6, 7, 8),
            s=Python.list("a", "b", "c", "d", "e", "f", "g", "h", "i"),
        )
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _batches() raises -> List[RecordBatch]:
    var out = List[RecordBatch]()
    for i in range(3):
        var strings: List[Optional[String]] = [
            String("a") + String(i),
            None,
            String("c"),
        ]
        out.append(
            record_batch(
                [
                    array([i * 3, i * 3 + 1, i * 3 + 2], int64).to_dyn(),
                    array(strings^),
                ],
                names=["n", "s"],
            )
        )
    return out^


def _write_all[
    W: BatchWriter
](mut writer: W, batches: List[RecordBatch]) raises:
    for ref batch in batches:
        writer.write(batch)


def _drain[R: BatchReader](mut reader: R) raises -> List[RecordBatch]:
    var out = List[RecordBatch]()
    while True:
        var batch = reader.read_next_batch()
        if not batch:
            break
        out.append(batch.take())
    return out^


def _assert_same(got: List[RecordBatch], expected: List[RecordBatch]) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        assert_true(got[i] == expected[i])


def _bytes_before_close(writes: Writes) -> Int:
    """How many bytes reached the sink before it was closed."""
    var total = 0
    for n in writes[]:
        if n < 0:
            break
        total += n
    return total


def _close_twice_then_write[
    W: BatchWriter
](mut writer: W, batch: RecordBatch) raises:
    writer.close()
    writer.close()
    with assert_raises():
        writer.write(batch)


def test_ipc_file_reads_back_batch_by_batch() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var path = join(dir, "t.arrow")
        var writer = RecordBatchFileWriter(path, batches[0].schema)
        _write_all(writer, batches)
        writer.close()
        var reader = RecordBatchFileReader(path)
        assert_true(reader.schema() == batches[0].schema)
        _assert_same(_drain(reader), batches)
        assert_false(reader.read_next_batch())


def test_ipc_stream_reads_back_batch_by_batch() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var path = join(dir, "t.arrows")
        var writer = RecordBatchStreamWriter(path, batches[0].schema)
        _write_all(writer, batches)
        writer.close()
        var reader = RecordBatchStreamReader(path)
        assert_true(reader.schema() == batches[0].schema)
        _assert_same(_drain(reader), batches)


def test_json_reads_back_batch_by_batch() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var path = join(dir, "t.jsonl")
        var writer = JsonWriter(FileSink(path))
        _write_all(writer, batches)
        writer.close()
        var reader = JsonReader(BufferSource(path))
        var got = _drain(reader)
        # One block holds the whole file: the batches come back as one.
        assert_equal(len(got), 1)
        assert_equal(got[0].num_rows(), 9)
        assert_true(got[0].schema == batches[0].schema)


def test_ipc_file_writer_streams_to_its_sink() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var writes = Writes(List[Int]())
        var writer = RecordBatchFileWriter(
            _Tap(join(dir, "t.arrow"), writes), batches[0].schema
        )
        _write_all(writer, batches)
        assert_true(_bytes_before_close(writes) > 0)
        assert_true(len(writes[]) >= len(batches))
        writer.close()


def test_ipc_stream_writer_streams_to_its_sink() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var writes = Writes(List[Int]())
        var writer = RecordBatchStreamWriter(
            _Tap(join(dir, "t.arrows"), writes), batches[0].schema
        )
        _write_all(writer, batches)
        assert_true(len(writes[]) >= len(batches))
        writer.close()


def test_json_writer_streams_to_its_sink() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var writes = Writes(List[Int]())
        var writer = JsonWriter(_Tap(join(dir, "t.jsonl"), writes))
        _write_all(writer, batches)
        assert_equal(len(writes[]), len(batches))
        writer.close()


def test_writers_close_once() raises:
    var batch = _batches()[0].copy()
    with ScratchDir() as dir:
        var ipc_file = RecordBatchFileWriter(join(dir, "a.arrow"), batch.schema)
        _close_twice_then_write(ipc_file, batch)
        var ipc_stream = RecordBatchStreamWriter(
            join(dir, "a.arrows"), batch.schema
        )
        _close_twice_then_write(ipc_stream, batch)
        var json = JsonWriter(FileSink(join(dir, "a.jsonl")))
        _close_twice_then_write(json, batch)
        var avro = AvroWriter(join(dir, "a.avro"), batch.schema)
        _close_twice_then_write(avro, batch)
        var parquet = FileWriter(
            FileSink(join(dir, "a.parquet")), batch.schema, Compression.SNAPPY
        )
        _close_twice_then_write(parquet, batch)


def _scan[F: Format](path: String, schema: Schema) raises -> List[RecordBatch]:
    var reader = F.open(path, schema)
    return _drain(reader)


def test_formats_open_a_file_for_a_scan() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var arrow = join(dir, "t.arrow")
        var writer = RecordBatchFileWriter(arrow, batches[0].schema)
        _write_all(writer, batches)
        writer.close()
        assert_equal(IpcFormat.name(), "Ipc")
        _assert_same(_scan[IpcFormat](arrow, batches[0].schema), batches)

        var jsonl = join(dir, "t.jsonl")
        var json = JsonWriter(FileSink(jsonl))
        _write_all(json, batches)
        json.close()
        # Projected to `n`: the reader skips `s` rather than parsing it.
        var n_only = Schema(fields=[batches[0].schema.fields[0].copy()])
        var got = _scan[JsonFormat](jsonl, n_only)
        assert_equal(len(got[0].schema.fields), 1)
        assert_equal(got[0].num_rows(), 9)


def test_parquet_reads_back_one_row_group_per_batch() raises:
    with ScratchDir() as dir:
        var path = join(dir, "t.parquet")
        _write_parquet(path, row_group_size=3)
        var expected = read_table(path).to_batches()
        var reads = Reads(List[Tuple[Int, Int]]())
        var file = ParquetFile(_Recorder(path, reads))
        # Opening reads the footer's speculative tail, all of a file this
        # small; what follows is row group by row group.
        reads[].clear()
        var reader = ParquetReader(file^)
        assert_true(reader.schema() == expected[0].schema)
        var got = _drain(reader)
        assert_equal(len(got), 3)
        _assert_same(got, expected)
        assert_true(len(reads[]) > 0)
        assert_true(_largest_read(reads) < BufferSource(path).size() // 3)


def test_parquet_reader_follows_a_read_plan() raises:
    with ScratchDir() as dir:
        var path = join(dir, "t.parquet")
        _write_parquet(path, row_group_size=3)
        var reader = ParquetReader(
            ParquetFile(path),
            columns=Optional[List[String]](["s"]),
            row_groups=Optional[List[Int]]([2, 0]),
        )
        assert_equal(len(reader.schema().fields), 1)
        var got = _drain(reader)
        assert_equal(len(got), 2)
        assert_equal(got[0].num_rows(), 3)
        assert_equal(got[0].schema.fields[0].name, "s")


def test_parquet_format_opens_a_projection() raises:
    with ScratchDir() as dir:
        var path = join(dir, "t.parquet")
        _write_parquet(path, row_group_size=4)
        var full = read_table(path)
        var n_only = Schema(fields=[full.schema.fields[0].copy()])
        assert_equal(ParquetFormat.name(), "Parquet")
        var got = _scan[ParquetFormat](path, n_only)
        assert_equal(len(got), 3)
        assert_true(got[0].schema == n_only)


def test_avro_and_parquet_read_back_what_they_wrote() raises:
    var batches = _batches()
    with ScratchDir() as dir:
        var avro_path = join(dir, "t.avro")
        var avro = AvroWriter(avro_path, batches[0].schema)
        _write_all(avro, batches)
        avro.close()
        var avro_reader = AvroFile(avro_path).batches(batch_size=4)
        var from_avro = _drain(avro_reader)
        var rows = 0
        for ref b in from_avro:
            rows += b.num_rows()
        assert_equal(rows, 9)
        assert_equal(AvroFormat.name(), "Avro")
        assert_equal(
            _scan[AvroFormat](avro_path, batches[0].schema)[0].num_rows(), 9
        )

        var parquet_path = join(dir, "t.parquet")
        var parquet = FileWriter(
            FileSink(parquet_path),
            batches[0].schema,
            Compression.SNAPPY,
            row_group_size=3,
        )
        _write_all(parquet, batches)
        parquet.close()
        var parquet_reader = ParquetReader(ParquetFile(parquet_path))
        _assert_same(_drain(parquet_reader), batches)
