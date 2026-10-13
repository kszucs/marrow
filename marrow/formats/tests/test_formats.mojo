# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Every format through the one protocol: what a `BatchWriter` writes, the
format's `BatchReader` reads back batch for batch, and the bytes reach the sink
as each batch is written rather than at `close`."""

from std.memory import ArcPointer
from std.os.path import join
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import int64
from ...io import BufferSource, ByteSink, FileSink
from ...ipc import (
    IpcFormat,
    RecordBatchFileReader,
    RecordBatchFileWriter,
    RecordBatchStreamReader,
    RecordBatchStreamWriter,
)
from ...json import JsonFormat, JsonReader, JsonWriter
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
