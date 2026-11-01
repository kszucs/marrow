# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""IPC file and stream round-trips through marrow's Python bindings, checked
against pyarrow."""

import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

import marrow


def _make_pa_batch() -> pa.RecordBatch:
    return pa.record_batch(
        {
            "a": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "b": pa.array([1.1, 2.2, 3.3, 4.4, 5.5], type=pa.float64()),
        }
    )


class TestIPCRoundtrip:
    """IPC file and stream round-trips using marrow.read/write_ipc_*."""

    def _roundtrip_file(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
            marrow.write_ipc_file(f.name, batches=[marrow.record_batch(batch)])
            result_batches = marrow.read_ipc_file(f.name)
        assert len(result_batches) == 1
        return pa.record_batch(result_batches[0])

    def _roundtrip_stream(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        with tempfile.NamedTemporaryFile(suffix=".arrows") as f:
            marrow.write_ipc_stream(f.name, batches=[marrow.record_batch(batch)])
            result_batches = marrow.read_ipc_stream(f.name)
        assert len(result_batches) == 1
        return pa.record_batch(result_batches[0])

    def test_file_primitives(self):
        batch = _make_pa_batch()
        assert batch.equals(self._roundtrip_file(batch))

    def test_stream_primitives(self):
        batch = _make_pa_batch()
        assert batch.equals(self._roundtrip_stream(batch))

    def test_file_multi_batch(self):
        b1 = _make_pa_batch()
        b2 = _make_pa_batch()
        with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
            marrow.write_ipc_file(
                f.name,
                batches=[marrow.record_batch(b1), marrow.record_batch(b2)],
            )
            result_batches = marrow.read_ipc_file(f.name)
        assert len(result_batches) == 2
        assert b1.equals(pa.record_batch(result_batches[0]))
        assert b2.equals(pa.record_batch(result_batches[1]))

    def test_file_bool_and_string(self):
        batch = pa.record_batch(
            {
                "flags": pa.array([True, False, True], type=pa.bool_()),
                "name": pa.array(["x", "y", "z"], type=pa.utf8()),
            }
        )
        assert batch.equals(self._roundtrip_file(batch))

    def test_file_nullable(self):
        batch = pa.record_batch({"x": pa.array([10, None, 30, None], type=pa.int32())})
        result = self._roundtrip_file(batch)
        assert batch.equals(result)
        assert result.column("x").null_count == 2

    def test_stream_to_file(self):
        """write_ipc_stream + read_ipc_stream + write_ipc_file + read_ipc_file round-trip."""
        batch = _make_pa_batch()
        with (
            tempfile.NamedTemporaryFile(suffix=".arrows") as sf,
            tempfile.NamedTemporaryFile(suffix=".arrow") as ff,
        ):
            marrow.write_ipc_stream(sf.name, batches=[marrow.record_batch(batch)])
            ma_batches = marrow.read_ipc_stream(sf.name)
            marrow.write_ipc_file(ff.name, batches=list(ma_batches))
            result_batches = marrow.read_ipc_file(ff.name)
        assert len(result_batches) == 1
        assert batch.equals(pa.record_batch(result_batches[0]))

    @pytest.mark.parametrize("codec", ["lz4", "zstd"])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("native_codecs", [True, False])
    def test_compressed(self, codec, stream, native_codecs, tmp_path):
        """Marrow writes compressed bodies, in Mojo or through liblz4 and
        libzstd; pyarrow and marrow read them, marrow either way."""
        batch = pa.record_batch(
            {
                "a": pa.array(range(10_000), type=pa.int64()),
                "s": pa.array([f"v{i % 37}" for i in range(10_000)]),
            }
        )
        write = marrow.write_ipc_stream if stream else marrow.write_ipc_file
        read = marrow.read_ipc_stream if stream else marrow.read_ipc_file
        open_pa = pa.ipc.open_stream if stream else pa.ipc.open_file
        path = str(tmp_path / "compressed.arrow")
        write(
            path,
            batches=[marrow.record_batch(batch)],
            compression=codec,
            native_codecs=native_codecs,
        )
        assert open_pa(path).read_all().to_batches()[0].equals(batch)
        for read_native in [True, False]:
            got = read(path, native_codecs=read_native)
            assert batch.equals(pa.record_batch(got[0]))

    def test_pyarrow_writes_marrow_reads_file(self):
        """PyArrow writes IPC file, Marrow reads it."""
        batch = _make_pa_batch()
        with tempfile.NamedTemporaryFile(suffix=".arrow") as f:
            with pa.ipc.new_file(f.name, batch.schema) as writer:
                writer.write_batch(batch)
            result_batches = marrow.read_ipc_file(f.name)
        assert len(result_batches) == 1
        assert batch.equals(pa.record_batch(result_batches[0]))

    def test_pyarrow_writes_marrow_reads_stream(self):
        """PyArrow writes IPC stream, Marrow reads it."""
        batch = _make_pa_batch()
        with tempfile.NamedTemporaryFile(suffix=".arrows") as f:
            with pa.ipc.new_stream(f.name, batch.schema) as writer:
                writer.write_batch(batch)
            result_batches = marrow.read_ipc_stream(f.name)
        assert len(result_batches) == 1
        assert batch.equals(pa.record_batch(result_batches[0]))

    def test_file_dictionary_column(self):
        """Dictionary-encoded column survives an IPC file round-trip."""
        batch = pa.record_batch(
            {"cat": pa.array(["x", "y", "x", "z"]).dictionary_encode()}
        )
        assert batch.equals(self._roundtrip_file(batch))


# arrow-testing's gold files: the same cases in both byte orders.
GOLD = Path(__file__).resolve().parents[3] / "testing/data/arrow-ipc-stream/integration"


@pytest.mark.skipif(not GOLD.is_dir(), reason="arrow-testing is not checked out")
@pytest.mark.parametrize(
    "read, suffix",
    [(marrow.read_ipc_file, "arrow_file"), (marrow.read_ipc_stream, "stream")],
)
def test_big_endian_ipc_is_refused(read, suffix):
    little = GOLD / "1.0.0-littleendian" / f"generated_primitive.{suffix}"
    assert len(read(str(little))) > 0
    big = GOLD / "1.0.0-bigendian" / f"generated_primitive.{suffix}"
    with pytest.raises(marrow.ArrowNotImplementedError, match="big-endian"):
        read(str(big))


class TestIPCObjects:
    """`marrow.ipc.new_file` and friends against pyarrow."""

    def test_file_metadata_read_by_pyarrow(self, tmp_path):
        path = tmp_path / "m.arrow"
        batch = _make_pa_batch()
        with marrow.ipc.new_file(
            path, batch.schema, metadata={"origin": b"marrow"}
        ) as w:
            w.write_batch(marrow.record_batch(batch), custom_metadata={"k": "v"})
            w.write(marrow.record_batch(batch))
        reader = pa.ipc.open_file(path)
        assert reader.metadata == {b"origin": b"marrow"}
        first = reader.get_batch_with_custom_metadata(0)
        assert first.batch.equals(batch)
        assert first.custom_metadata.to_dict() == {b"k": b"v"}
        assert reader.get_batch_with_custom_metadata(1).custom_metadata is None

    def test_stream_metadata_read_by_pyarrow(self, tmp_path):
        path = tmp_path / "m.arrows"
        batch = _make_pa_batch()
        with marrow.ipc.new_stream(path, batch.schema) as w:
            w.write_batch(marrow.record_batch(batch), {"k": "v"})
        got = list(pa.ipc.open_stream(path).iter_batches_with_custom_metadata())
        assert len(got) == 1
        assert got[0].batch.equals(batch)
        assert got[0].custom_metadata.to_dict() == {b"k": b"v"}

    def test_file_metadata_written_by_pyarrow(self, tmp_path):
        path = str(tmp_path / "p.arrow")
        batch = _make_pa_batch()
        with pa.ipc.new_file(path, batch.schema, metadata={"origin": "pyarrow"}) as w:
            w.write_batch(batch, custom_metadata={"k": "v"})
            w.write_batch(batch)
        reader = marrow.ipc.open_file(path)
        assert reader.metadata == {b"origin": b"pyarrow"}
        assert reader.num_record_batches == 2
        first = reader.get_batch_with_custom_metadata(0)
        assert pa.record_batch(first.batch).equals(batch)
        assert first.custom_metadata == {b"k": b"v"}
        assert reader.get_batch_with_custom_metadata(1).custom_metadata is None
        assert pa.table(reader.read_all()).equals(pa.Table.from_batches([batch, batch]))

    def test_stream_metadata_written_by_pyarrow(self, tmp_path):
        path = str(tmp_path / "p.arrows")
        batch = _make_pa_batch()
        with pa.ipc.new_stream(path, batch.schema) as w:
            w.write_batch(batch, custom_metadata={"k": "v"})
            w.write_batch(batch)
        reader = marrow.ipc.open_stream(path)
        got = list(reader.iter_batches_with_custom_metadata())
        assert [g.custom_metadata for g in got] == [{b"k": b"v"}, None]
        assert pa.record_batch(got[0].batch).equals(batch)

    def test_write_table_iterate_and_read_all(self, tmp_path):
        path = tmp_path / "t.arrows"
        batch = _make_pa_batch()
        table = pa.Table.from_batches([batch, batch])
        with marrow.ipc.new_stream(path, table.schema) as w:
            w.write_table(marrow.table(table))
        assert len(list(marrow.ipc.open_stream(path))) == 2
        assert pa.table(marrow.ipc.open_stream(path).read_all()).equals(table)
        reader = marrow.ipc.open_stream(path)
        assert pa.record_batch(reader.read_next_batch()).equals(batch)

    def test_compression_option_and_sink_type(self, tmp_path):
        batch = _make_pa_batch()
        path = tmp_path / "c.arrow"
        options = marrow.ipc.IpcWriteOptions(compression="zstd")
        with marrow.ipc.new_file(path, batch.schema, options=options) as w:
            w.write(marrow.record_batch(batch))
        assert pa.ipc.open_file(path).get_batch(0).equals(batch)
        with pytest.raises(TypeError):
            marrow.ipc.new_file(object(), batch.schema)

    def test_write_accepts_batches_and_tables_only(self, tmp_path):
        batch = _make_pa_batch()
        table = pa.Table.from_batches([batch])
        path = tmp_path / "w.arrow"
        with marrow.ipc.new_file(path, batch.schema) as w:
            w.write(batch)
            w.write(table)
            with pytest.raises(TypeError):
                w.write(pa.array([1, 2]))
        assert (
            pa.ipc.open_file(path)
            .read_all()
            .equals(pa.Table.from_batches([batch, batch]))
        )
