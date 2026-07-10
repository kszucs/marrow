"""RowSelection: the unit-level per-row keep/skip within a row group (built from
per-page keep flags and combined with intersect) plus the decode path — reading a
file with a RowSelection must yield exactly the selected rows, and must match a
full read filtered to the same rows. Uses tiny data pages so a selection
genuinely spans skip / keep / partial pages across every builder (primitive,
nullable, string/dict)."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table
from marrow.parquet.reader import RowSelection
from marrow.tabular import Table


def _write(code: String) raises -> String:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_pageskip.parquet")
    # tiny pages -> many pages per chunk; single row group
    pq.write_table(
        Python.evaluate(code),
        path,
        data_page_size=128,
        row_group_size=1000000,
        compression="none",
    )
    return path


def _selection(n: Int, keep_from: Int, keep_to: Int) -> RowSelection:
    var s = List[Bool](capacity=n)
    for i in range(n):
        s.append(i >= keep_from and i < keep_to)
    return RowSelection(s^)


def _strided(n: Int, m: Int, r: Int) -> RowSelection:
    # keep rows where i % m < r (scattered across page boundaries)
    var s = List[Bool](capacity=n)
    for i in range(n):
        s.append((i % m) < r)
    return RowSelection(s^)


def _assert_matches_full(
    var got: Table, var full: Table, sel: RowSelection
) raises:
    """`got` (read with `sel`) must equal `full` filtered to the selected rows.
    """
    assert_equal(got.num_rows(), sel.num_selected())
    var gb = got.to_batches()[0].copy()
    var fb = full.to_batches()[0].copy()
    var ncols = gb.num_columns()
    for c in range(ncols):
        var gcol = gb.columns[c].copy()
        var fcol = fb.columns[c].copy()
        var k = 0  # index into the selected (got) rows
        for i in range(sel.total_rows()):
            if sel.selected(i):
                assert_equal(gcol.is_valid(k), fcol.is_valid(i))
                if gcol.is_valid(k):
                    assert_equal(String(gcol[k]), String(fcol[i]))
                k += 1


def _check(code: String, sel: RowSelection) raises:
    var path = _write(code)
    var full = read_table(path)
    var rs = List[RowSelection]()
    rs.append(sel.copy())
    var got = read_table(path, row_selections=rs^)
    _assert_matches_full(got^, full^, sel)
    remove(path)


def test_contiguous_int() raises:
    _check(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array(list(range(10000)),"
            " type=__import__('pyarrow').int64())})"
        ),
        _selection(10000, 2500, 7500),
    )


def test_scattered_int() raises:
    _check(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array(list(range(10000)),"
            " type=__import__('pyarrow').int64())})"
        ),
        _strided(10000, 7, 3),
    )


def test_nullable_int() raises:
    _check(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([None if i % 4 == 0 else i for i in"
            " range(5000)], type=__import__('pyarrow').int64())})"
        ),
        _strided(5000, 5, 2),
    )


def test_string_dict() raises:
    # low cardinality -> dictionary-encoded data pages
    _check(
        (
            "__import__('pyarrow').table({'s':"
            " __import__('pyarrow').array([['red','green','blue'][i % 3] for i"
            " in range(6000)])})"
        ),
        _selection(6000, 1000, 4000),
    )


def test_string_plain_nullable() raises:
    _check(
        (
            "__import__('pyarrow').table({'s':"
            " __import__('pyarrow').array([None if i % 6 == 0 else 'v%d' % i"
            " for i in range(4000)], type=__import__('pyarrow').string())})"
        ),
        _strided(4000, 3, 1),
    )


def test_select_none() raises:
    # an empty selection reads zero rows
    var path = _write(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array(list(range(2000)),"
        " type=__import__('pyarrow').int64())})"
    )
    var s = List[Bool](capacity=2000)
    for _ in range(2000):
        s.append(False)
    var rs = List[RowSelection]()
    rs.append(RowSelection(s^))
    var got = read_table(path, row_selections=rs^)
    assert_equal(got.num_rows(), 0)
    remove(path)


# ---------------------------------------------------------------------------
# RowSelection unit tests: per-row keep/skip built from per-page keep flags and
# combined with intersect.
# ---------------------------------------------------------------------------


def test_all() raises:
    var s = RowSelection.all(5)
    assert_equal(s.total_rows(), 5)
    assert_equal(s.num_selected(), 5)
    assert_true(s.selects_all())
    assert_true(s.selects_any())


def test_from_pages() raises:
    # 3 pages of 2/3/2 rows; keep pages 0 and 2
    var keep: List[Bool] = [True, False, True]
    var rows: List[Int] = [2, 3, 2]
    var s = RowSelection.from_pages(keep, rows)
    assert_equal(s.total_rows(), 7)
    assert_equal(s.num_selected(), 4)
    assert_true(s.selected(0))
    assert_true(s.selected(1))
    assert_false(s.selected(2))  # page 1 skipped
    assert_false(s.selected(4))
    assert_true(s.selected(5))  # page 2 kept
    assert_false(s.selects_all())
    assert_true(s.selects_any())


def test_intersect() raises:
    var av: List[Bool] = [True, True, False, True]
    var a = RowSelection(av^)
    var bv: List[Bool] = [True, False, True, True]
    var b = RowSelection(bv^)
    var c = a.intersect(b)
    assert_equal(c.num_selected(), 2)  # rows 0 and 3
    assert_true(c.selected(0))
    assert_false(c.selected(1))
    assert_false(c.selected(2))
    assert_true(c.selected(3))


def test_intersect_size_mismatch() raises:
    var a = RowSelection.all(3)
    var b = RowSelection.all(4)
    with assert_raises():
        _ = a.intersect(b)


def test_selected_in() raises:
    var sv: List[Bool] = [True, False, True, True, False]
    var s = RowSelection(sv^)
    assert_equal(s.selected_in(0, 5), 3)
    assert_equal(s.selected_in(1, 2), 1)  # rows 1,2 -> only 2
    assert_equal(s.selected_in(3, 2), 1)  # rows 3,4 -> only 3


def test_none_selected() raises:
    var sv: List[Bool] = [False, False, False]
    var s = RowSelection(sv^)
    assert_false(s.selects_any())
    assert_equal(s.num_selected(), 0)


def main() raises:
    TestSuite.run[__functions_in_module()]()
