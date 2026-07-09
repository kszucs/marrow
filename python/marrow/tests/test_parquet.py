"""Parquet reader/writer bindings, verified against PyArrow as the oracle."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow.parquet as mpq


def _to_pa(marrow_table):
    """marrow Table -> pyarrow Table via the Arrow C stream interface."""
    return pa.RecordBatchReader.from_stream(marrow_table).read_all()


def _sample():
    return pa.table(
        {
            "i": pa.array([1, 2, None, 4, 5], pa.int64()),
            "f": pa.array([1.5, 2.5, 3.5, 4.5, 5.5], pa.float64()),
            "b": pa.array([True, False, None, True, False], pa.bool_()),
            "s": pa.array(["a", "bb", None, "dddd", "e"]),
        }
    )


def _assert_equiv(got, want):
    assert got.column_names == want.column_names
    for i in range(want.num_columns):
        # Parquet drops the large_* Arrow distinction
        gt = str(got.column(i).type).replace("large_", "")
        wt = str(want.column(i).type).replace("large_", "")
        assert gt == wt
        assert got.column(i).to_pylist() == want.column(i).to_pylist()


def test_marrow_reads_pyarrow(tmp_path):
    p = tmp_path / "t.parquet"
    want = _sample()
    pq.write_table(want, p)
    got = _to_pa(mpq.read_table(p))
    _assert_equiv(got, pq.read_table(p))


@pytest.mark.parametrize("compression", ["none", "snappy", "zstd", "lz4"])
def test_pyarrow_reads_marrow(tmp_path, compression):
    src = tmp_path / "src.parquet"
    want = _sample()
    pq.write_table(want, src)
    mt = mpq.read_table(src)  # marrow Table
    out = tmp_path / "out.parquet"
    mpq.write_table(mt, out, compression=compression)
    _assert_equiv(pq.read_table(out), want)


def test_read_returns_marrow_table(tmp_path):
    p = tmp_path / "t.parquet"
    pq.write_table(_sample(), p)
    t = mpq.read_table(p)
    assert t.num_rows() == 5
    assert t.num_columns() == 4
    assert t.column_names() == ["i", "f", "b", "s"]


def test_unsupported_compression(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(_sample(), src)
    mt = mpq.read_table(src)
    with pytest.raises(Exception):
        mpq.write_table(mt, tmp_path / "o.parquet", compression="brotli")
