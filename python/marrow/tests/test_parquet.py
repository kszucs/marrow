"""Parquet reader/writer bindings, verified against PyArrow as the oracle."""

import os

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


@pytest.mark.parametrize("compression", ["snappy", "zstd", "lz4", "brotli", "gzip"])
def test_marrow_reads_every_codec(tmp_path, compression):
    """Each page codec is a library marrow opens at runtime -- from a wheel,
    the copy staged beside the extension -- so a missing or mis-staged one
    fails here and nowhere else: a Linux wheel once shipped a brotli decoder
    that could not load."""
    p = tmp_path / "t.parquet"
    want = _sample()
    pq.write_table(want, p, compression=compression)
    _assert_equiv(_to_pa(mpq.read_table(p)), want)


def test_marrow_reads_through_opendal(tmp_path):
    """`fs://` goes through `libopendal_c`. A checkout that never built it
    skips; a run that must have it -- MARROW_REQUIRE_OPENDAL, set when the
    wheel under test ships OpenDAL -- fails."""
    p = tmp_path / "t.parquet"
    want = _sample()
    pq.write_table(want, p)
    try:
        got = mpq.read_table(f"fs://{p}")
    except Exception as e:
        if os.environ.get("MARROW_REQUIRE_OPENDAL") or "opendal" not in str(e):
            raise
        pytest.skip(f"libopendal_c is not available: {e}")
    _assert_equiv(_to_pa(got), want)


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
    assert t.num_rows == 5
    assert t.num_columns == 4
    assert t.column_names == ["i", "f", "b", "s"]


def test_column_projection(tmp_path):
    p = tmp_path / "t.parquet"
    pq.write_table(_sample(), p)
    t = mpq.read_table(p, columns=["s", "i"])
    assert t.column_names == ["s", "i"]
    got = _to_pa(t)
    assert got.column("i").to_pylist() == [1, 2, None, 4, 5]
    assert got.column("s").to_pylist() == ["a", "bb", None, "dddd", "e"]


def test_unsupported_compression(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(_sample(), src)
    mt = mpq.read_table(src)
    with pytest.raises(Exception):
        mpq.write_table(mt, tmp_path / "o.parquet", compression="brotli")


def _int64_table(n):
    return pa.table({"a": pa.array(range(n), pa.int64())})


def _string_table(n):
    return pa.table({"s": pa.array([f"row-{i}-" + "x" * (i % 40) for i in range(n)])})


def test_content_defined_chunking_changes_page_layout(tmp_path):
    src = tmp_path / "src.parquet"
    want = _int64_table(200_000)
    pq.write_table(want, src)
    mt = mpq.read_table(src)

    plain = tmp_path / "plain.parquet"
    cdc = tmp_path / "cdc.parquet"
    mpq.write_table(mt, plain, compression="none")
    mpq.write_table(mt, cdc, compression="none", use_content_defined_chunking=True)

    assert plain.read_bytes() != cdc.read_bytes()
    _assert_equiv(pq.read_table(plain), want)
    _assert_equiv(pq.read_table(cdc), want)


def test_content_defined_chunking_string_column(tmp_path):
    """A non-int64 leaf -- the specific failure a fixed-width-only value
    dispatch would hide."""
    src = tmp_path / "src.parquet"
    want = _string_table(20_000)
    pq.write_table(want, src)
    mt = mpq.read_table(src)

    plain = tmp_path / "plain.parquet"
    cdc = tmp_path / "cdc.parquet"
    mpq.write_table(mt, plain, compression="none")
    mpq.write_table(
        mt,
        cdc,
        compression="none",
        use_content_defined_chunking={
            "min_chunk_size": 1024,
            "max_chunk_size": 4096,
        },
    )

    assert plain.read_bytes() != cdc.read_bytes()
    _assert_equiv(pq.read_table(plain), want)
    _assert_equiv(pq.read_table(cdc), want)


def test_content_defined_chunking_rejects_bad_sizes(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(_sample(), src)
    mt = mpq.read_table(src)
    with pytest.raises(Exception):
        mpq.write_table(
            mt,
            tmp_path / "bad.parquet",
            use_content_defined_chunking={
                "min_chunk_size": 1024,
                "max_chunk_size": 1024,
            },
        )


def test_content_defined_chunking_dict_requires_min_and_max(tmp_path):
    """PyArrow's shape: a dict must supply both `min_chunk_size` and
    `max_chunk_size` -- there is no fallback to the `True` defaults for a
    dict missing one, and an unrecognized key is rejected too."""
    src = tmp_path / "src.parquet"
    pq.write_table(_sample(), src)
    mt = mpq.read_table(src)
    with pytest.raises(ValueError, match="Missing options"):
        mpq.write_table(
            mt,
            tmp_path / "missing.parquet",
            use_content_defined_chunking={"min_chunk_size": 1024},
        )
    with pytest.raises(ValueError, match="Unknown options"):
        mpq.write_table(
            mt,
            tmp_path / "unknown.parquet",
            use_content_defined_chunking={
                "min_chunk_size": 1024,
                "max_chunk_size": 4096,
                "bogus": 1,
            },
        )


def test_content_defined_chunking_rejects_bad_type(tmp_path):
    src = tmp_path / "src.parquet"
    pq.write_table(_sample(), src)
    mt = mpq.read_table(src)
    with pytest.raises(TypeError):
        mpq.write_table(
            mt, tmp_path / "bad.parquet", use_content_defined_chunking="yes"
        )
