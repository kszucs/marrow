"""Tests for comptime-typed expression execution.

These tests verify that ``Column``/``Add``/``Sub`` (the default comptime-typed
expression nodes) correctly evaluate expression trees via ``execute(batch)``,
running a single fused vectorize loop with zero intermediate arrays.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray, UInt32Array
from marrow.builders import array
from marrow.dtypes import Int64Type, int64, uint32
from marrow.aot import (
    Column,
    Add,
    Sub,
    NumericValue,
    StringColumn,
    Length,
)
from marrow.tabular import RecordBatch, record_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fused_add_exec(
    col_a: Int, col_b: Int, batch: RecordBatch
) raises -> PrimitiveArray[Int64Type]:
    """Build an Add expression and evaluate against the batch."""
    var a = Column[Int64Type](col_a)
    var b = Column[Int64Type](col_b)
    var expr = Add(a, b)
    return expr.execute(batch)


def _fused_sub_exec(
    col_a: Int, col_b: Int, batch: RecordBatch
) raises -> PrimitiveArray[Int64Type]:
    """Build a Sub expression and evaluate against the batch."""
    var a = Column[Int64Type](col_a)
    var b = Column[Int64Type](col_b)
    var expr = Sub(a, b)
    return expr.execute(batch)


# ---------------------------------------------------------------------------
# Basic fused add
# ---------------------------------------------------------------------------


def test_fused_add_basic() raises:
    """Add(col(0), col(1)) produces element-wise addition."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([11, 22, 33, 44, 55], int64))


def test_fused_add_single_element() raises:
    """Add works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_equal(result[0].value(), 50)


def test_fused_add_non_aligned() raises:
    """Add works with non-SIMD-aligned lengths."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([11, 22, 33, 44, 55, 66, 77], int64))


def test_fused_add_negative_values() raises:
    """Add handles negative values correctly."""
    var a = array([-1, 2, -3, 4, -5], int64)
    var b = array([1, -2, 3, -4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([0, 0, 0, 0, 0], int64))


def test_fused_add_large_values() raises:
    """Add handles large int64 values."""
    var a = array([9_223_372_036_854_775_806], int64)  # near max
    var b = array([1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_equal(result[0].value(), 9_223_372_036_854_775_807)  # max int64


def test_fused_add_same_column() raises:
    """Add(col(0), col(0)) doubles the values."""
    var a = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _fused_add_exec(0, 0, batch)
    assert_true(result == array([2, 4, 6, 8, 10], int64))


def test_fused_add_zero() raises:
    """Add with zeros produces all zeros."""
    var a = array([0, 0, 0], int64)
    var b = array([0, 0, 0], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([0, 0, 0], int64))


# ---------------------------------------------------------------------------
# Sub
# ---------------------------------------------------------------------------


def test_fused_sub_basic() raises:
    """Sub(col(0), col(1)) produces element-wise subtraction."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_true(result == array([9, 18, 27, 36, 45], int64))


def test_fused_sub_single_element() raises:
    """Sub works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_equal(result[0].value(), 34)


def test_fused_sub_negative_result() raises:
    """Sub handles negative results."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_true(result == array([-9, -18, -27], int64))


# ---------------------------------------------------------------------------
# Length (string)
# ---------------------------------------------------------------------------


def test_fused_length_basic() raises:
    """Length(StringColumn(0)) produces per-element byte lengths."""
    var a = array(["ab", "cde", "", "f"])
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(StringColumn(0))
    var result = expr.execute(batch)
    assert_true(result == array([2, 3, 0, 1], uint32))


def test_fused_length_non_aligned() raises:
    """Length works with non-SIMD-aligned lengths."""
    var a = array(["a", "bb", "ccc", "dddd", "e", "ff", "ggg"])
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(StringColumn(0))
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3, 4, 1, 2, 3], uint32))


def test_fused_length_sliced() raises:
    """Length matches kernels.string.string_lengths on a sliced array."""
    var full = array(["aa", "b", "ccc", "dddd", "e"])
    var a = full.slice(1, 3)
    var batch = record_batch([a^], names=["c0"])
    var expr = Length(StringColumn(0))
    var result = expr.execute(batch)
    assert_true(result == array([1, 3, 4], uint32))


def test_fused_length_write_to() raises:
    """Length.write_to produces nested readable output."""
    var expr = Length(StringColumn(2))
    assert_equal(String(expr), "Length(StrCol[2])")


# ---------------------------------------------------------------------------
# Trait conformance
# ---------------------------------------------------------------------------


def test_fused_column_implements_numeric_typed_value() raises:
    """Column[Int64Type] satisfies NumericValue."""
    # The fact that .execute() compiles on this type proves trait conformance.
    var expr = Column[Int64Type](0)
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = expr.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_fused_column_write_to() raises:
    """Column.write_to produces readable output."""
    var col = Column[Int64Type](3)
    assert_equal(String(col), "Col[3]")


def test_fused_add_write_to() raises:
    """Add.write_to produces nested readable output."""
    var a = Column[Int64Type](0)
    var b = Column[Int64Type](1)
    var expr = Add(a, b)
    assert_equal(String(expr), "Add(Col[0], Col[1])")


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------


def test_fused_column_copy() raises:
    """Column can be copied."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var col = Column[Int64Type](0)
    var copy = col.copy()
    var result = copy.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


def test_fused_add_copy() raises:
    """Add can be copied and re-executed."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var expr = Add(col_a, col_b)
    var copy = expr.copy()
    var result = copy.execute(batch)
    assert_true(result == array([11, 22, 33], int64))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    TestSuite.run[__functions_in_module()]()
