"""Tests for the comptime-typed boolean expression nodes (``BoolValue``,
``Lt``, ``Gt``, ``Eq`` in ``marrow.aot.values``).

These tests exercise the comptime layer directly: single fused vectorize
pass, bit-packed ``BoolArray`` result, composition with ``NumericValue``
children (plain ``Column`` and ``Add``). See ``test_aot_injection.mojo`` for
``BoolValue`` nodes boxed into the runtime ``Expr`` via the ``FUSED`` tag.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.aot.values import Column, Add, Lt, Gt, Eq


def test_lt_fuses() raises:
    """Lt(Column, Column) produces the correct bit-packed BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var lt = Lt(Column[Int64Type](0), Column[Int64Type](1))
    var result = lt.execute(batch)
    assert_true(result[0].value())
    assert_true(not result[1].value())
    assert_true(result[2].value())
    assert_true(not result[3].value())
    assert_true(result[4].value())


def test_gt_fuses() raises:
    """Gt(Column, Column) produces the correct bit-packed BoolArray."""
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var gt = Gt(Column[Int64Type](0), Column[Int64Type](1))
    var result = gt.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_eq_fuses() raises:
    """Eq(Column, Column) produces the correct bit-packed BoolArray."""
    var a = array([1, 4, 3, 4, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var eq = Eq(Column[Int64Type](0), Column[Int64Type](1))
    var result = eq.execute(batch)
    assert_true(not result[0].value())
    assert_true(result[1].value())
    assert_true(not result[2].value())
    assert_true(result[3].value())
    assert_true(not result[4].value())


def test_comparison_composes_with_add() raises:
    """Gt((a + b), b) composes NumericValue and BoolValue nodes in one fused
    pass -- no intermediate arrays for either the add or the comparison.
    """
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var added = Add(Column[Int64Type](0), Column[Int64Type](1))
    var pred = Gt(added, Column[Int64Type](1))
    var result = pred.execute(batch)
    # a + b = [5, 9, 7, 12, 6]; compared to b = [4, 4, 4, 4, 4] -> all True
    for i in range(5):
        assert_true(result[i].value())


def test_comparison_write_to() raises:
    """Lt/Gt/Eq.write_to() display the expected nested structure."""
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    assert_equal(String(Lt(col_a, col_b)), "Lt(Col[0], Col[1])")
    assert_equal(String(Gt(col_a, col_b)), "Gt(Col[0], Col[1])")
    assert_equal(String(Eq(col_a, col_b)), "Eq(Col[0], Col[1])")


def test_comparison_dtype_is_bool() raises:
    """Lt/Gt/Eq.dtype() reports bool_, matching BoolValue's default impl."""
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var lt = Lt(col_a, col_b)
    assert_true(lt.dtype().value().is_bool())


def main() raises:
    TestSuite.run[__functions_in_module()]()
