"""RowSelection: per-row keep/skip within a row group, built from per-page keep
flags and combined with intersect."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from marrow.testing import TestSuite
from marrow.parquet.reader import RowSelection


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
