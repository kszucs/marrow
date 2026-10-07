# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet reader/writer bindings, verified against PyArrow as the oracle."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import marrow as ma
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


@pytest.mark.parametrize("compression", ["zstd", "lz4"])
@pytest.mark.parametrize("native_codecs", [True, False])
def test_library_codecs(tmp_path, compression, native_codecs):
    """LZ4 and ZSTD pages written in Mojo or through liblz4 and libzstd,
    and read back either way."""
    want = _sample()
    src = tmp_path / "src.parquet"
    pq.write_table(want, src)
    out = tmp_path / "out.parquet"
    mpq.write_table(
        mpq.read_table(src), out, compression=compression, native_codecs=native_codecs
    )
    _assert_equiv(pq.read_table(out), want)
    for read_native in [True, False]:
        got = _to_pa(mpq.read_table(out, native_codecs=read_native))
        _assert_equiv(got, want)


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
    with pytest.raises(ma.ArrowInvalid):
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
    with pytest.raises(ma.ArrowInvalid):
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


def test_read_binary_type_view(tmp_path):
    """`binary_type=binary_view()` reads strings as `string_view`, as
    pyarrow's option of the same name does."""
    import marrow as ma

    p = tmp_path / "t.parquet"
    pq.write_table(_sample(), p, store_schema=False)
    got = _to_pa(mpq.read_table(p, binary_type=ma.binary_view()))
    want = pq.read_table(p, binary_type=pa.binary_view())
    assert got.schema.field("s").type == pa.string_view()
    assert got.equals(want)


def test_write_string_view(tmp_path):
    """A string_view column writes as an ordinary string column."""
    import marrow as ma

    src = tmp_path / "src.parquet"
    pq.write_table(_sample(), src, store_schema=False)
    views = mpq.read_table(src, binary_type=ma.binary_view())
    assert _to_pa(views).schema.field("s").type == pa.string_view()
    out = tmp_path / "out.parquet"
    mpq.write_table(views, out)
    _assert_equiv(pq.read_table(out), _sample())


def _ids_table():
    def fid(n):
        return {b"PARQUET:field_id": str(n).encode()}

    schema = pa.schema(
        [
            pa.field("i", pa.int64(), metadata=fid(1)),
            pa.field(
                "s",
                pa.struct([pa.field("a", pa.int64(), metadata=fid(3))]),
                metadata=fid(2),
            ),
            pa.field(
                "l",
                pa.list_(pa.field("element", pa.int64(), metadata=fid(5))),
                metadata=fid(4),
            ),
            pa.field(
                "m",
                pa.map_(
                    pa.field("key", pa.string(), nullable=False, metadata=fid(7)),
                    pa.field("value", pa.int64(), metadata=fid(8)),
                ),
                metadata=fid(6),
            ),
        ]
    )
    data = {"i": [1], "s": [{"a": 1}], "l": [[1, 2]], "m": [{"k": 1}]}
    return pa.Table.from_pydict(data, schema=schema)


def _field_ids(schema):
    """Every field id in `schema`, keyed by a dotted path."""

    def fid(f):
        return int((f.metadata or {})[b"PARQUET:field_id"])

    s, l, m = schema.field("s"), schema.field("l"), schema.field("m")
    return {
        "i": fid(schema.field("i")),
        "s": fid(s),
        "s.a": fid(s.type.field("a")),
        "l": fid(l),
        "l.element": fid(l.type.value_field),
        "m": fid(m),
        "m.key": fid(m.type.key_field),
        "m.value": fid(m.type.item_field),
    }


def test_field_ids_round_trip(tmp_path):
    # marrow's Python Field/Schema expose no metadata; the C stream carries it
    src, dst = tmp_path / "src.parquet", tmp_path / "dst.parquet"
    want = _field_ids(_ids_table().schema)
    pq.write_table(_ids_table(), src)
    mt = mpq.read_table(src)
    assert _field_ids(_to_pa(mt).schema) == want
    mpq.write_table(mt, dst)
    assert _field_ids(pq.read_schema(dst)) == want
