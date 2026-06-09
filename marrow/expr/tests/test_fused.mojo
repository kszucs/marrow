"""Tests for comptime-fused expression execution through the executor pipeline.

These tests verify that ``FusedProcessor[T]`` correctly evaluates comptime-fused
expression trees (``FusedColumn``, ``FusedAdd``) by binding them against a
``RecordBatch`` and running the single fused vectorize loop.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray
from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.expr import (
    FusedColumn,
    FusedAdd,
    FusedSub,
    FusedProcessor,
    NumericTypedValue,
    TypedValue,
)
from marrow.tabular import RecordBatch, record_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fused_add_exec(col_a: Int, col_b: Int, batch: RecordBatch) raises -> PrimitiveArray[Int64Type]:
    """Build a FusedAdd processor and evaluate against the batch."""
    var a = FusedColumn[Int64Type](col_a)
    var b = FusedColumn[Int64Type](col_b)
    var expr = FusedAdd(a, b)
    var proc = FusedProcessor(expr)
    ref result = proc.eval(batch).as_int64()
    return result.copy()


def _fused_sub_exec(col_a: Int, col_b: Int, batch: RecordBatch) raises -> PrimitiveArray[Int64Type]:
    """Build a FusedSub processor and evaluate against the batch."""
    var a = FusedColumn[Int64Type](col_a)
    var b = FusedColumn[Int64Type](col_b)
    var expr = FusedSub(a, b)
    var proc = FusedProcessor(expr)
    ref result = proc.eval(batch).as_int64()
    return result.copy()


# ---------------------------------------------------------------------------
# Basic fused add
# ---------------------------------------------------------------------------


def test_fused_add_basic() raises:
    """FusedAdd(col(0), col(1)) produces element-wise addition."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([11, 22, 33, 44, 55], int64))


def test_fused_add_single_element() raises:
    """FusedAdd works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_equal(result[0].value(), 50)


def test_fused_add_non_aligned() raises:
    """FusedAdd works with non-SIMD-aligned lengths."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([11, 22, 33, 44, 55, 66, 77], int64))


def test_fused_add_negative_values() raises:
    """FusedAdd handles negative values correctly."""
    var a = array([-1, 2, -3, 4, -5], int64)
    var b = array([1, -2, 3, -4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([0, 0, 0, 0, 0], int64))


def test_fused_add_large_values() raises:
    """FusedAdd handles large int64 values."""
    var a = array([9_223_372_036_854_775_806], int64)  # near max
    var b = array([1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_equal(result[0].value(), 9_223_372_036_854_775_807)  # max int64


def test_fused_add_same_column() raises:
    """FusedAdd(col(0), col(0)) doubles the values."""
    var a = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _fused_add_exec(0, 0, batch)
    assert_true(result == array([2, 4, 6, 8, 10], int64))


def test_fused_add_zero() raises:
    """FusedAdd with zeros produces all zeros."""
    var a = array([0, 0, 0], int64)
    var b = array([0, 0, 0], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_add_exec(0, 1, batch)
    assert_true(result == array([0, 0, 0], int64))


# ---------------------------------------------------------------------------
# FusedSub
# ---------------------------------------------------------------------------


def test_fused_sub_basic() raises:
    """FusedSub(col(0), col(1)) produces element-wise subtraction."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_true(result == array([9, 18, 27, 36, 45], int64))


def test_fused_sub_single_element() raises:
    """FusedSub works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_equal(result[0].value(), 34)


def test_fused_sub_negative_result() raises:
    """FusedSub handles negative results."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _fused_sub_exec(0, 1, batch)
    assert_true(result == array([-9, -18, -27], int64))


# ---------------------------------------------------------------------------
# Trait conformance
# ---------------------------------------------------------------------------


def test_fused_column_implements_numeric_typed_value() raises:
    """FusedColumn[Int64Type] satisfies NumericTypedValue."""
    var col = FusedColumn[Int64Type](0)
    # The fact that FusedProcessor compiles with this type proves trait conformance
    var expr = FusedColumn[Int64Type](0)
    var proc = FusedProcessor(expr)
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = proc.eval(batch)
    assert_true(result == array([1, 2, 3], int64))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_fused_column_write_to() raises:
    """FusedColumn.write_to produces readable output."""
    var col = FusedColumn[Int64Type](3)
    assert_equal(String(col), "FusedCol[3]")


def test_fused_add_write_to() raises:
    """FusedAdd.write_to produces nested readable output."""
    var a = FusedColumn[Int64Type](0)
    var b = FusedColumn[Int64Type](1)
    var expr = FusedAdd(a, b)
    assert_equal(String(expr), "FusedAdd(FusedCol[0], FusedCol[1])")


# ---------------------------------------------------------------------------
# Copy semantics
# ---------------------------------------------------------------------------


def test_fused_column_copy() raises:
    """FusedColumn can be copied."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var col = FusedColumn[Int64Type](0)
    var copy = col.copy()
    var result = copy.execute(batch)
    assert_true(result == array([1, 2, 3], int64))


def test_fused_add_copy() raises:
    """FusedAdd can be copied and re-executed."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var expr = FusedAdd(col_a, col_b)
    var copy = expr.copy()
    var result = copy.execute(batch)
    assert_true(result == array([11, 22, 33], int64))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    TestSuite.run[__functions_in_module()]()
