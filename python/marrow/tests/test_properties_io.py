# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Round-trip properties for Parquet and Arrow IPC, with PyArrow as the
reference reader and writer.

Each format is crossed three ways — marrow writes and PyArrow reads, PyArrow
writes and marrow reads, marrow writes and reads back — over generated tables
whose columns may be sliced (a nonzero offset the writer must honour).

Divergences are pinned as strict ``xfail`` cases at the bottom of the file.
"""

import math
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import marrow as ma
import marrow.parquet as mpq
from marrow.tests.strategies import (
    BINARY_TYPES,
    FLOAT_TYPES,
    INTEGER_TYPES,
    STRING_TYPES,
    any_arrays,
    assert_same,
    dictionary_types,
    flat_types,
    lengths,
    nested_types,
)

# ── tables ─────────────────────────────────────────────────────────────────


@st.composite
def tables(draw, dtypes, max_columns=3, max_rows=150):
    """A PyArrow table of one to `max_columns` columns of drawn types, every
    column possibly a slice."""
    n = draw(lengths(max_rows))
    count = draw(st.integers(1, max_columns))
    columns = {f"c{i}": draw(any_arrays(dtypes, size=st.just(n))) for i in range(count)}
    return pa.table(columns)


def assert_tables_same(got, want):
    assert got.column_names == want.column_names
    for name in want.column_names:
        assert_same(got.column(name), want.column(name))


def to_pa_table(marrow_table):
    return pa.table(marrow_table)


# ── Parquet ────────────────────────────────────────────────────────────────

# The leaf types both libraries write to Parquet. Left out: float16, and the
# types marrow cannot write (date64, duration, interval, null, dictionary).
_PARQUET_LEAVES = st.sampled_from(
    [pa.bool_()]
    + INTEGER_TYPES
    + FLOAT_TYPES[1:]
    + STRING_TYPES[:2]
    + BINARY_TYPES[:2]
    + [pa.binary(3)]
    + [
        pa.date32(),
        pa.time32("s"),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
        pa.timestamp("s"),
        pa.timestamp("ms"),
        pa.timestamp("us", "UTC"),
        pa.timestamp("ns"),
    ]
    + [pa.decimal128(9, 2), pa.decimal128(20, 4), pa.decimal128(38, 10)]
)
# What PyArrow writes and marrow is asked to read, dictionary columns included.
# No fixed_size_list: PyArrow cannot write one holding nulls, nor always read
# back one it wrote.
_PARQUET_TYPES = st.one_of(
    _PARQUET_LEAVES, nested_types(_PARQUET_LEAVES, fixed=False), dictionary_types
)
# What marrow is asked to write.
_PARQUET_WRITE_TYPES = st.one_of(
    _PARQUET_LEAVES, nested_types(_PARQUET_LEAVES, fixed=False)
)


def _contains(t, predicate):
    if predicate(t):
        return True
    return any(_contains(t.field(i).type, predicate) for i in range(t.num_fields))


def _is_list(t):
    return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_map(t)


def _empty_list_column(table):
    """A zero-row table with a list or map column, which marrow's reader
    returns as one empty list (pinned below)."""
    return table.num_rows == 0 and any(
        _contains(f.type, _is_list) for f in table.schema
    )


def _pyarrow_read(path):
    try:
        return pq.read_table(path)
    except (pa.ArrowInvalid, OSError):
        # PyArrow cannot always read back its own all-null fixed_size_list.
        return None


_PARQUET_CODECS = ["none", "snappy", "zstd", "lz4"]


@given(
    tables(_PARQUET_TYPES),
    st.sampled_from(_PARQUET_CODECS),
    st.sampled_from(["1.0", "2.0"]),
    st.booleans(),
    st.sampled_from([None, 1, 64, 1000]),
)
def test_parquet_pyarrow_writes_marrow_reads(
    tmp_path, table, codec, page_version, dictionary, row_group_size
):
    """marrow reads what PyArrow wrote — every codec, both page versions,
    dictionary encoding on and off, and several row groups — and sees what
    PyArrow itself reads back.

    marrow does not read the ``ARROW:schema`` PyArrow stores, so some types
    come back in their Parquet form (large_string as string, decimal32 as
    decimal128, a time zone name as UTC); those compare after a cast to
    PyArrow's type, which is lossless for every such pair."""
    # A v2 page under a list can make marrow decode past the page's values
    # and abort (pinned below), so v2 is exercised on flat columns only.
    nested = any(_contains(f.type, _is_list) for f in table.schema)
    assume(page_version == "1.0" or not nested)
    path = tmp_path / "t.parquet"
    pq.write_table(
        table,
        path,
        compression=codec,
        data_page_version=page_version,
        use_dictionary=dictionary,
        row_group_size=row_group_size,
    )
    want = _pyarrow_read(path)
    assume(want is not None and not _empty_list_column(table))
    got = to_pa_table(mpq.read_table(path))
    assert got.num_rows == want.num_rows
    assert_tables_same(got.cast(want.schema), want)


@given(
    tables(_PARQUET_WRITE_TYPES),
    st.sampled_from(_PARQUET_CODECS),
    st.sampled_from(["1.0", "2.0"]),
)
def test_parquet_marrow_writes_pyarrow_reads(tmp_path, table, codec, page_version):
    """PyArrow reads back the values marrow wrote."""
    path = tmp_path / "t.parquet"
    mpq.write_table(
        ma.table(table), path, compression=codec, data_page_version=page_version
    )
    got = pq.read_table(path)
    assert got.num_rows == table.num_rows
    assert_tables_same(got.cast(table.schema), table)


@given(tables(_PARQUET_WRITE_TYPES), st.sampled_from(_PARQUET_CODECS), st.booleans())
def test_parquet_marrow_roundtrip(tmp_path, table, codec, cdc):
    """marrow reads back what marrow wrote, content-defined chunking or not."""
    assume(not _empty_list_column(table))
    path = tmp_path / "t.parquet"
    mpq.write_table(
        ma.table(table), path, compression=codec, use_content_defined_chunking=cdc
    )
    got = to_pa_table(mpq.read_table(path))
    assert_tables_same(got.cast(table.schema), table)


# ── IPC ────────────────────────────────────────────────────────────────────

_IPC_LEAVES = flat_types
_IPC_TYPES = st.one_of(_IPC_LEAVES, nested_types(_IPC_LEAVES), dictionary_types)


@st.composite
def batch_lists(draw):
    """One to three record batches sharing a schema. A dictionary column keeps
    its first batch's dictionary throughout, which is all the IPC file format
    allows; replacing it is pinned below."""
    first = draw(tables(_IPC_TYPES, max_rows=100))
    batches = [first]
    for _ in range(draw(st.integers(0, 2))):
        n = draw(lengths(100))
        columns = {}
        for name in first.column_names:
            t = first.schema.field(name).type
            if pa.types.is_dictionary(t):
                dictionary = first.column(name).combine_chunks().dictionary
                index = st.none()
                if len(dictionary):
                    index = st.one_of(st.none(), st.integers(0, len(dictionary) - 1))
                indices = pa.array(
                    draw(st.lists(index, min_size=n, max_size=n)), t.index_type
                )
                columns[name] = pa.DictionaryArray.from_arrays(indices, dictionary)
            else:
                columns[name] = draw(any_arrays(st.just(t), size=st.just(n)))
        batches.append(pa.table(columns))
    return [
        pa.record_batch([c.combine_chunks() for c in b.columns], schema=b.schema)
        for b in batches
    ]


def assert_batches_same(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.schema.names == w.schema.names
        for i in range(w.num_columns):
            assert_same(g.column(i), w.column(i))


_FORMATS = {
    "file": (ma.write_ipc_file, ma.read_ipc_file, ipc.new_file, ipc.open_file),
    "stream": (
        ma.write_ipc_stream,
        ma.read_ipc_stream,
        ipc.new_stream,
        ipc.open_stream,
    ),
}


@given(batch_lists(), st.sampled_from(sorted(_FORMATS)))
def test_ipc_marrow_writes_pyarrow_reads(tmp_path, batches, fmt):
    write, _, _, open_ = _FORMATS[fmt]
    path = str(tmp_path / "t.arrow")
    write(path, [ma.record_batch(b) for b in batches])
    with pa.OSFile(path) as f:
        reader = open_(f)
        if fmt == "file":
            got = [reader.get_batch(i) for i in range(reader.num_record_batches)]
        else:
            got = list(reader)
    assert_batches_same(got, batches)


@given(batch_lists(), st.sampled_from(sorted(_FORMATS)))
def test_ipc_pyarrow_writes_marrow_reads(tmp_path, batches, fmt):
    _, read, new, _ = _FORMATS[fmt]
    path = str(tmp_path / "t.arrow")
    with pa.OSFile(path, "wb") as sink, new(sink, batches[0].schema) as writer:
        for b in batches:
            writer.write_batch(b)
    got = [pa.record_batch(b) for b in read(path)]
    assert_batches_same(got, batches)


@given(batch_lists(), st.sampled_from(sorted(_FORMATS)))
def test_ipc_marrow_roundtrip(tmp_path, batches, fmt):
    write, read, _, _ = _FORMATS[fmt]
    path = str(tmp_path / "t.arrow")
    write(path, [ma.record_batch(b) for b in batches])
    got = [pa.record_batch(b) for b in read(path)]
    assert_batches_same(got, batches)


# ── pinned divergences ─────────────────────────────────────────────────────


def _in_subprocess(code, *args):
    """Run `code` in a child Python (it may abort) and return its exit code:
    0, or the negated signal that killed it.

    A child that exits with a Python error raised rather than crashed, which
    is not what the pinned cases expect; that is a `RuntimeError` here, so an
    `xfail(raises=AssertionError)` reports it instead of counting it.
    """
    proc = subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    if proc.returncode > 0:
        raise RuntimeError(f"the child raised:\n{proc.stderr[-2000:]}")
    return proc.returncode


def test_parquet_write_large_list(tmp_path):
    arr = pa.array([[1, None], None, [], [2]], pa.large_list(pa.int32()))
    path = tmp_path / "t.parquet"
    mpq.write_table(ma.table(pa.table({"c": arr})), path)
    assert pq.read_table(path).column("c").to_pylist() == arr.to_pylist()


@pytest.mark.parametrize(
    "arr",
    [
        pa.array([7], pa.int64()),
        pa.array([1, 1, None], pa.int32()),
        pa.array(["a", "a"]),
        pa.array([[1], [1]], pa.list_(pa.int32())),
    ],
    ids=["one-row", "repeated", "string", "list"],
)
def test_parquet_write_single_valued_column(tmp_path, arr):
    path = tmp_path / "t.parquet"
    mpq.write_table(ma.table(pa.table({"c": arr})), path)
    assert pq.read_table(path).column("c").to_pylist() == arr.to_pylist()


@pytest.mark.parametrize("dtype", [pa.timestamp("s", "UTC"), pa.time32("s")])
def test_parquet_write_seconds_unit(tmp_path, dtype):
    """Parquet has no second unit: both writers store milliseconds. PyArrow
    restores the unit from its stored Arrow schema; marrow writes none, so
    PyArrow -- and marrow -- read the column back in milliseconds."""
    arr = pa.array([1, None, 2], dtype)
    path = tmp_path / "t.parquet"
    mpq.write_table(ma.table(pa.table({"c": arr})), path)
    want = pq.read_table(path).column("c")
    assert want.type.unit == "ms"
    assert want.cast(dtype).to_pylist() == arr.to_pylist()
    got = pa.table(mpq.read_table(path)).column("c")
    assert got.equals(want)


def test_parquet_write_seconds_overflow(tmp_path):
    """A timestamp[s] too large for milliseconds is refused. Arrow C++ means
    to refuse it too, but PyArrow 23 writes the wrapped product."""
    table = pa.table({"c": pa.array([2**62], pa.timestamp("s"))})
    with pytest.raises(ma.ArrowInvalid):
        mpq.write_table(ma.table(table), tmp_path / "ma.parquet")


def test_ipc_file_dictionary_replacement(tmp_path):
    b1 = pa.record_batch({"c": pa.array(["x", "x"]).dictionary_encode()})
    b2 = pa.record_batch({"c": pa.array(["y", "z"]).dictionary_encode()})
    path = str(tmp_path / "t.arrow")
    try:
        ma.write_ipc_file(path, [ma.record_batch(b1), ma.record_batch(b2)])
    except ma.ArrowException:
        return  # refusing is as good as PyArrow
    got = [pa.record_batch(b).column(0).to_pylist() for b in ma.read_ipc_file(path)]
    assert got == [["x", "x"], ["y", "z"]]


@pytest.mark.parametrize(
    "dtype",
    [pa.list_(pa.int32()), pa.map_(pa.string(), pa.int32())],
    ids=["list", "map"],
)
def test_parquet_read_zero_row_list(tmp_path, dtype):
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"c": pa.array([], dtype)}), path)
    assert mpq.read_table(path).num_rows == 0


_LIST = pa.list_(pa.bool_())
_STRUCT_OF_LIST = pa.struct([("f0", _LIST)])


@pytest.mark.parametrize(
    "arr",
    [
        pa.array([None, {"f0": [True, False]}], _STRUCT_OF_LIST),
        pa.array(
            [None, {"s": None}, {"s": {"f0": None}}, {"s": {"f0": [True]}}],
            pa.struct([("s", _STRUCT_OF_LIST)]),
        ),
        pa.array(
            [None, [], [None, {"f0": []}, {"f0": [False, None]}], [None]],
            pa.list_(_STRUCT_OF_LIST),
        ),
        pa.array(
            [[("k", None), ("j", {"f0": [True]})], None],
            pa.map_(pa.string(), _STRUCT_OF_LIST),
        ),
    ],
    ids=["struct-list", "struct-struct-list", "list-struct-list", "map-struct-list"],
)
def test_parquet_write_null_struct_over_list(tmp_path, arr):
    path = tmp_path / "t.parquet"
    mpq.write_table(ma.table(pa.table({"c": arr})), path)
    assert pq.read_table(path).column("c").to_pylist() == arr.to_pylist()
    got = pa.table(mpq.read_table(path)).column("c")
    assert got.to_pylist() == arr.to_pylist()


@pytest.mark.parametrize(
    "dtype",
    [
        pa.list_(pa.struct([("f", pa.decimal128(9, 2))])),
        pa.list_(pa.struct([("f", pa.binary(3))])),
    ],
    ids=["decimal", "fixed_size_binary"],
)
def test_parquet_read_v2_page_under_empty_list(tmp_path, dtype):
    path = tmp_path / "t.parquet"
    pq.write_table(
        pa.table({"c": pa.array([[]], dtype)}), path, data_page_version="2.0"
    )
    code = "import sys, marrow.parquet as mpq; mpq.read_table(sys.argv[1])"
    assert _in_subprocess(code, path) == 0


def test_parquet_write_keeps_signed_zero(tmp_path):
    arr = pa.array([0.0, -0.0, 1.0])
    path = tmp_path / "t.parquet"
    mpq.write_table(ma.table(pa.table({"c": arr})), path)
    got = pa.table(mpq.read_table(path)).column("c").to_pylist()
    assert [math.copysign(1, v) for v in got] == [1.0, -1.0, 1.0]


@pytest.mark.parametrize("fmt", sorted(_FORMATS))
def test_ipc_read_float16(tmp_path, fmt):
    _, read, new, _ = _FORMATS[fmt]
    batch = pa.record_batch({"c": pa.array([1.5, None], pa.float16())})
    path = str(tmp_path / "t.arrow")
    with pa.OSFile(path, "wb") as sink, new(sink, batch.schema) as writer:
        writer.write_batch(batch)
    got = pa.record_batch(read(path)[0])
    assert got.schema.field("c").type == pa.float16()
    assert got.column("c").to_pylist() == [1.5, None]
