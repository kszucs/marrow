"""Parsing the page index (OffsetIndex + ColumnIndex). PyArrow writes a
multi-page column with the page index; marrow parses it and the per-page bounds
must match the (sorted) data. Marrow's own writer emits no page index yet, so
those chunks read back absent."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_page_index, write_table
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


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


def test_marrow_file_has_no_page_index() raises:
    # marrow's writer does not emit a page index yet -> both are absent
    var caps = Python.evaluate(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([1, 2, 3],"
        " type=__import__('pyarrow').int64())})"
    ).__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_nopageidx.parquet")
    write_table(t, path)
    var pi = read_page_index(path)
    assert_equal(len(pi), 1)
    assert_false(Bool(pi[0][0].offset_index))
    assert_false(Bool(pi[0][0].column_index))
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
