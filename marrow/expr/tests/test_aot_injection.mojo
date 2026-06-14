"""Tests for AOT-compiled value expression injection into runtime expressions.

These tests verify that comptime-fused expressions (from ``faszom.mojo`` and
``fused_values.mojo``) can be properly boxed into runtime ``Expr`` nodes via
the ``to_expr()`` method.

The key pattern is:
1. Create a comptime-fused expression (e.g., FusedAdd[FusedColumn, FusedColumn])
2. Box it into a runtime Expr via to_expr()
3. The boxed Expr carries the fused expression in its _fused slot
4. Fused expressions are executed via FusedProcessor[T], not through Planner

This test verifies that:
- to_expr() produces valid Expr with FUSED tag
- Fused expressions can be executed via FusedProcessor[T]
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray, Int64Array
from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.expr import Expr, Planner, FUSED
from marrow.expr.fused_values import (
    FusedColumn,
    FusedAdd,
    FusedSub,
    NumericTypedValue,
)
from marrow.expr.executor import FusedProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_fused_processor[T: NumericTypedValue](
    expr: T, batch: RecordBatch
) raises -> Int64Array:
    """Build a FusedProcessor and evaluate against the batch."""
    var proc = FusedProcessor[T](expr)
    ref result = proc.eval(batch).as_int64()
    return result.copy()


# ---------------------------------------------------------------------------
# Basic AOT expression injection tests
# ---------------------------------------------------------------------------


def test_fused_column_to_expr() raises:
    """FusedColumn.to_expr() produces a valid runtime Expr with FUSED tag."""
    var a = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy()], names=["c0"])

    # Create a fused column expression
    var fused = FusedColumn[Int64Type](0)

    # Box it into a runtime Expr
    var expr: Expr = fused.to_expr()

    # Verify the expression is tagged as FUSED
    assert_equal(expr.kind(), FUSED)

    # Execute via FusedProcessor (direct path)
    var direct_result = _exec_fused_processor(fused, batch)

    # Results should match
    assert_true(direct_result == array([1, 2, 3, 4, 5], int64))


def test_fused_add_to_expr() raises:
    """FusedAdd.to_expr() produces a valid runtime Expr with FUSED tag."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    # Create a fused add expression
    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var fused = FusedAdd(col_a, col_b)

    # Box it into a runtime Expr
    var expr: Expr = fused.to_expr()

    # Verify the expression is tagged as FUSED
    assert_equal(expr.kind(), FUSED)

    # Execute via FusedProcessor (direct path)
    var direct_result = _exec_fused_processor(fused, batch)

    # Results should match
    assert_true(direct_result == array([11, 22, 33, 44, 55], int64))


def test_fused_sub_to_expr() raises:
    """FusedSub.to_expr() produces a valid runtime Expr with FUSED tag."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    # Create a fused sub expression
    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var fused = FusedSub(col_a, col_b)

    # Box it into a runtime Expr
    var expr: Expr = fused.to_expr()

    # Verify the expression is tagged as FUSED
    assert_equal(expr.kind(), FUSED)

    # Execute via FusedProcessor (direct path)
    var direct_result = _exec_fused_processor(fused, batch)

    # Results should match
    assert_true(direct_result == array([9, 18, 27, 36, 45], int64))


# ---------------------------------------------------------------------------
# Nested expression injection tests
# ---------------------------------------------------------------------------


def test_nested_fused_add_sub_to_expr() raises:
    """Nested FusedAdd/FusedSub expressions can be boxed and evaluated."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var c = array([100, 200, 300, 400, 500], int64)
    var batch = record_batch([a.copy(), b.copy(), c.copy()], names=["c0", "c1", "c2"])

    # Create a nested expression: (a + b) - c
    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var col_c = FusedColumn[Int64Type](2)
    var add_expr = FusedAdd(col_a, col_b)
    var fused = FusedSub(add_expr, col_c)

    # Box it into a runtime Expr
    var expr: Expr = fused.to_expr()
    assert_equal(expr.kind(), FUSED)

    # Execute via FusedProcessor (direct path)
    var direct_result = _exec_fused_processor(fused, batch)

    # Results should match: (a + b) - c = [11-100, 22-200, 33-300, 44-400, 55-500]
    assert_true(direct_result == array([-89, -178, -267, -356, -445], int64))


def test_chained_fused_adds_to_expr() raises:
    """Chained FusedAdd expressions can be boxed and evaluated."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var c = array([100, 200, 300], int64)
    var d = array([1000, 2000, 3000], int64)
    var batch = record_batch([a.copy(), b.copy(), c.copy(), d.copy()], names=["c0", "c1", "c2", "c3"])

    # Create a chained expression: a + b + c + d
    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var col_c = FusedColumn[Int64Type](2)
    var col_d = FusedColumn[Int64Type](3)
    var add_ab = FusedAdd(col_a, col_b)
    var add_abc = FusedAdd(add_ab, col_c)
    var fused = FusedAdd(add_abc, col_d)

    # Box it into a runtime Expr
    var expr: Expr = fused.to_expr()
    assert_equal(expr.kind(), FUSED)

    # Execute via FusedProcessor (direct path)
    var direct_result = _exec_fused_processor(fused, batch)

    # Results should match: a + b + c + d = [1111, 2222, 3333]
    assert_true(direct_result == array([1111, 2222, 3333], int64))


# ---------------------------------------------------------------------------
# Type safety tests
# ---------------------------------------------------------------------------


def test_fused_column_different_types_to_expr() raises:
    """FusedColumn with different numeric types can be boxed."""
    # Test Int64
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])

    var col_i64 = FusedColumn[Int64Type](0)
    var expr_i64: Expr = col_i64.to_expr()
    assert_equal(expr_i64.kind(), FUSED)

    var result_i64 = _exec_fused_processor(col_i64, batch)
    assert_true(result_i64 == array([1, 2, 3], int64))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element_fused_to_expr() raises:
    """Single-element fused expressions can be boxed and evaluated."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])

    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var fused = FusedAdd(col_a, col_b)

    var expr: Expr = fused.to_expr()
    assert_equal(expr.kind(), FUSED)

    var result = _exec_fused_processor(fused, batch)
    assert_equal(result[0].value(), 50)


def test_negative_values_fused_to_expr() raises:
    """Fused expressions with negative values work correctly when boxed."""
    var a = array([-1, 2, -3, 4], int64)
    var b = array([1, -2, 3, -4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var fused = FusedAdd(col_a, col_b)

    var expr: Expr = fused.to_expr()
    assert_equal(expr.kind(), FUSED)

    var result = _exec_fused_processor(fused, batch)
    assert_true(result == array([0, 0, 0, 0], int64))


def test_large_values_fused_to_expr() raises:
    """Fused expressions with large int64 values work correctly when boxed."""
    var a = array([9_223_372_036_854_775_806], int64)  # near max int64
    var b = array([1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])

    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var fused = FusedAdd(col_a, col_b)

    var expr: Expr = fused.to_expr()
    assert_equal(expr.kind(), FUSED)

    var result = _exec_fused_processor(fused, batch)
    assert_equal(result[0].value(), 9_223_372_036_854_775_807)  # max int64


# ---------------------------------------------------------------------------
# Non-alignment tests
# ---------------------------------------------------------------------------


def test_non_aligned_length_fused_to_expr() raises:
    """Fused expressions work with non-SIMD-aligned lengths when boxed."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)  # 7 elements, not SIMD-aligned
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    var col_a = FusedColumn[Int64Type](0)
    var col_b = FusedColumn[Int64Type](1)
    var fused = FusedAdd(col_a, col_b)

    var expr: Expr = fused.to_expr()
    assert_equal(expr.kind(), FUSED)

    var result = _exec_fused_processor(fused, batch)
    assert_true(result == array([11, 22, 33, 44, 55, 66, 77], int64))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    TestSuite.run[__functions_in_module()]()
