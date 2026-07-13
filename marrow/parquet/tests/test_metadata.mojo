"""Reading file metadata. Column statistics: `read_metadata` (raw footer) and
`read_statistics` (decoded typed min/max) — marrow reads the bounds PyArrow
writes, and round-trips its own (the write side is covered against PyArrow in
test_writer.mojo). Page index: `read_page_index` parses the OffsetIndex +
ColumnIndex PyArrow writes for a multi-page column, and the per-page bounds must
match the (sorted) data; marrow's own writer emits no page index yet, so those
chunks read back absent."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import (
    read_metadata,
    read_statistics,
    read_page_index,
    read_page_bounds,
    write_table,
)
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _write_pa(code: String, use_dictionary: Bool = True) raises -> String:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_readstats.parquet")
    pq.write_table(
        Python.evaluate(code),
        path,
        compression="none",
        use_dictionary=use_dictionary,
    )
    return path


def test_read_metadata_shape() raises:
    var path = _write_pa(
        "__import__('pyarrow').table({'a':"
        " __import__('pyarrow').array(list(range(2500)),"
        " type=__import__('pyarrow').int64())})"
    )
    var pq = Python.import_module("pyarrow.parquet")
    var meta = read_metadata(path)
    # matches pyarrow's own view of num_rows and column count
    assert_equal(meta.num_rows, 2500)
    assert_equal(len(meta.row_groups[0].columns), 1)
    # per-column-chunk null_count is surfaced without decoding data
    ref cm = meta.row_groups[0].columns[0].meta_data
    assert_equal(cm.null_count, 0)
    assert_true(cm.has_min_max)
    remove(path)


def test_read_int_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([5, -1, None, 9, -3],"
            " type=__import__('pyarrow').int64())}, )"
        ),
        use_dictionary=False,
    )
    var st = read_statistics(path)
    assert_equal(len(st), 1)  # one row group
    ref cs = st[0][0]
    assert_true(Bool(cs.min))
    assert_equal(cs.null_count, 1)
    assert_equal(cs.min.value().as_int64().value(), -3)
    assert_equal(cs.max.value().as_int64().value(), 9)
    remove(path)


def test_read_uint_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([1, 3000000000, 2],"
            " type=__import__('pyarrow').uint32())})"
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_equal(cs.min.value().as_uint32().value(), UInt32(1))
    assert_equal(
        cs.max.value().as_uint32().value(), UInt32(3000000000)
    )  # unsigned order
    remove(path)


def test_read_float_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([1.5, -2.5, 3.25, None],"
            " type=__import__('pyarrow').float64())})"
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(cs.min.value().as_float64().value() == -2.5)
    assert_true(cs.max.value().as_float64().value() == 3.25)
    remove(path)


def test_read_string_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'s':"
            " __import__('pyarrow').array(['banana', 'apple', 'cherry',"
            " None])})"
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(Bool(cs.min))
    assert_equal(cs.min.value().as_string().to_string(), "apple")
    assert_equal(cs.max.value().as_string().to_string(), "cherry")
    remove(path)


def test_read_stats_multiple_row_groups() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.evaluate("list(range(2500))"), type=pa.int64())
        )
    )
    var path = String("/tmp/marrow_readstats_rg.parquet")
    pq.write_table(
        tbl, path, row_group_size=1000, use_dictionary=False, compression="none"
    )
    var st = read_statistics(path)
    assert_equal(len(st), 3)  # 3 row groups
    # first row group covers rows [0, 1000)
    assert_equal(st[0][0].min.value().as_int64().value(), 0)
    assert_equal(st[0][0].max.value().as_int64().value(), 999)
    # last row group covers [2000, 2500)
    assert_equal(st[2][0].min.value().as_int64().value(), 2000)
    assert_equal(st[2][0].max.value().as_int64().value(), 2499)
    remove(path)


def test_roundtrip_own_stats() raises:
    # marrow writes -> marrow reads its own min/max back (closes the loop with
    # the PyArrow-oracle write tests in test_writer.mojo)
    var caps = Python.evaluate(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([7, 2, None, 11],"
        " type=__import__('pyarrow').int64()), 's':"
        " __import__('pyarrow').array(['m', 'a', 'z', None])})"
    ).__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_stats_rt.parquet")
    write_table(t, path)

    var st = read_statistics(path)
    assert_equal(st[0][0].min.value().as_int64().value(), 2)
    assert_equal(st[0][0].max.value().as_int64().value(), 11)
    assert_equal(st[0][0].null_count, 1)
    assert_equal(st[0][1].min.value().as_string().to_string(), "a")
    assert_equal(st[0][1].max.value().as_string().to_string(), "z")
    remove(path)


# ---------------------------------------------------------------------------
# Page index (OffsetIndex + ColumnIndex)
# ---------------------------------------------------------------------------


def _le_i64(b: List[UInt8]) -> Int:
    var v = 0
    for i in range(len(b)):
        v |= Int(b[i]) << (i * 8)
    return v


def test_page_index_sorted_int() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.evaluate("list(range(10000))"), type=pa.int64())
        )
    )
    var path = String("/tmp/marrow_pageidx.parquet")
    # tiny pages -> many pages in one chunk; page index on
    pq.write_table(
        tbl,
        path,
        data_page_size=256,
        use_dictionary=False,
        compression="none",
        write_page_index=True,
    )

    var pi = read_page_index(path)
    assert_equal(len(pi), 1)  # one row group
    ref col = pi[0][0]
    assert_true(Bool(col.offset_index))
    assert_true(Bool(col.column_index))
    ref oi = col.offset_index.value()
    ref cix = col.column_index.value()

    var npages = len(oi.page_locations)
    assert_true(npages > 1)
    assert_equal(len(cix.min_values), npages)
    assert_equal(len(cix.max_values), npages)
    assert_equal(len(cix.null_pages), npages)

    # first_row_index starts at 0 and strictly increases
    assert_equal(oi.page_locations[0].first_row_index, 0)
    for i in range(1, npages):
        assert_true(
            oi.page_locations[i].first_row_index
            > oi.page_locations[i - 1].first_row_index
        )

    # ascending column -> BoundaryOrder.ASCENDING (1); no null pages
    assert_equal(cix.boundary_order, 1)
    # global bounds
    assert_equal(_le_i64(cix.min_values[0]), 0)
    assert_equal(_le_i64(cix.max_values[npages - 1]), 9999)
    # value == row index here, so each page's min == its first_row_index
    for i in range(npages):
        assert_false(cix.null_pages[i])
        assert_equal(
            _le_i64(cix.min_values[i]), oi.page_locations[i].first_row_index
        )
    remove(path)


def test_marrow_file_has_page_index() raises:
    # marrow's writer now emits an OffsetIndex + ColumnIndex per column chunk
    var caps = Python.evaluate(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([5, 1, None, 9, 3],"
        " type=__import__('pyarrow').int64())})"
    ).__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_pageidx.parquet")
    write_table(t, path)
    var pi = read_page_index(path)
    assert_equal(len(pi), 1)
    assert_true(Bool(pi[0][0].offset_index))
    assert_true(Bool(pi[0][0].column_index))
    # one data page covering all rows, with the correct bounds and null count
    var pb = read_page_bounds(path)
    assert_equal(len(pb[0][0]), 1)
    var pg = pb[0][0][0].copy()
    assert_equal(pg.num_rows, 5)
    assert_equal(pg.min.value().as_int64().value(), 1)
    assert_equal(pg.max.value().as_int64().value(), 9)
    remove(path)


def test_marrow_page_index_pyarrow_reads() raises:
    # a marrow-written page index is spec-valid: PyArrow prunes with it and
    # returns exactly the matching rows.
    var pa = Python.import_module("pyarrow")
    var ds = Python.import_module("pyarrow.dataset")
    var ints = Python.list()
    for i in range(300):
        ints.append(i)
    var caps = pa.table(
        Python.dict(i=pa.array(ints, type=pa.int32()))
    ).__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_pageidx_pa.parquet")
    write_table(t, path, use_dictionary=False)
    var d = ds.dataset(path, format="parquet")
    var got = d.to_table(filter=(ds.field("i") >= 295))
    assert_equal(Int(py=got.num_rows), 5)
    assert_equal(Int(py=got.column(0)[0]), 295)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
