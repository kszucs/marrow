"""Tests for AOT-compiled value expression injection into runtime expressions.

These tests verify that comptime-typed expressions (``marrow.aot.values``)
can be boxed into runtime ``Expr`` nodes via the ``Expr(value)`` constructor,
and executed end-to-end through the type-erased runtime path.

The key pattern is:
1. Create a comptime-typed expression (e.g., Add[Column, Column])
2. Box it into a runtime Expr via Expr(value)
3. The boxed Expr carries the comptime node in its _fused slot
4. Both the direct path (``expr.execute(batch)``) and the boxed runtime path
   (``boxed.eval(batch)``) run the same single fused vectorize loop, since
   ``Expr.eval()`` delegates FUSED nodes straight back to ``execute()``

This test verifies that:
- Expr(value) produces a valid Expr with FUSED tag
- The boxed Expr can be executed directly via Expr.eval(), matching the
  typed direct-execution path exactly — the runtime and comptime layers are
  a genuine two-way bridge, not just a one-way introspection box.
"""

from std.testing import assert_equal, assert_true

from marrow.testing import TestSuite

from marrow.arrays import PrimitiveArray, Int64Array
from marrow.builders import array
from marrow.dtypes import Int64Type, int64
from marrow.tabular import RecordBatch, record_batch
from marrow.dyn import Expr, FUSED
from marrow.aot.values import (
    Column,
    Add,
    Sub,
    Gt,
    NumericValue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_typed[
    T: NumericValue
](expr: T, batch: RecordBatch) raises -> Int64Array:
    """Evaluate a comptime-typed expression directly against the batch."""
    ref result = expr.execute(batch).to_any().as_int64()
    return result.copy()


# ---------------------------------------------------------------------------
# Basic AOT expression injection tests
# ---------------------------------------------------------------------------


def test_fused_column_to_expr() raises:
    """Column.to_expr() produces a valid runtime Expr with FUSED tag."""
    var a = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy()], names=["c0"])

    # Create a typed column expression
    var fused = Column[Int64Type](0)

    # Box it into a runtime Expr
    var expr = Expr(fused)

    # Verify the expression is tagged as FUSED
    assert_equal(expr.kind(), FUSED)

    # Execute via the direct typed path
    var direct_result = _exec_typed(fused, batch)
    assert_true(direct_result == array([1, 2, 3, 4, 5], int64))

    # Execute via the boxed runtime path — same fused pass, same result
    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == direct_result.copy().to_any())


def test_fused_add_to_expr() raises:
    """Add.to_expr() produces a valid runtime Expr with FUSED tag."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    # Create a typed add expression
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var fused = Add(col_a, col_b)

    # Box it into a runtime Expr
    var expr = Expr(fused)

    # Verify the expression is tagged as FUSED
    assert_equal(expr.kind(), FUSED)

    # Execute via the direct typed path
    var direct_result = _exec_typed(fused, batch)
    assert_true(direct_result == array([11, 22, 33, 44, 55], int64))

    # Execute via the boxed runtime path — same fused pass, same result
    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == direct_result.copy().to_any())


def test_fused_sub_to_expr() raises:
    """Sub.to_expr() produces a valid runtime Expr with FUSED tag."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    # Create a typed sub expression
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var fused = Sub(col_a, col_b)

    # Box it into a runtime Expr
    var expr = Expr(fused)

    # Verify the expression is tagged as FUSED
    assert_equal(expr.kind(), FUSED)

    # Execute via the direct typed path
    var direct_result = _exec_typed(fused, batch)
    assert_true(direct_result == array([9, 18, 27, 36, 45], int64))

    # Execute via the boxed runtime path — same fused pass, same result
    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == direct_result.copy().to_any())


# ---------------------------------------------------------------------------
# Nested expression injection tests
# ---------------------------------------------------------------------------


def test_nested_fused_add_sub_to_expr() raises:
    """Nested Add/Sub expressions can be boxed and evaluated."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var c = array([100, 200, 300, 400, 500], int64)
    var batch = record_batch(
        [a.copy(), b.copy(), c.copy()], names=["c0", "c1", "c2"]
    )

    # Create a nested expression: (a + b) - c
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var col_c = Column[Int64Type](2)
    var add_expr = Add(col_a, col_b)
    var fused = Sub(add_expr, col_c)

    # Box it into a runtime Expr
    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    # Execute via the direct typed path
    var direct_result = _exec_typed(fused, batch)

    # Results should match: (a + b) - c = [11-100, 22-200, 33-300, 44-400, 55-500]
    assert_true(direct_result == array([-89, -178, -267, -356, -445], int64))

    # Execute via the boxed runtime path — same fused pass, same result
    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == direct_result.copy().to_any())


def test_chained_fused_adds_to_expr() raises:
    """Chained Add expressions can be boxed and evaluated."""
    var a = array([1, 2, 3], int64)
    var b = array([10, 20, 30], int64)
    var c = array([100, 200, 300], int64)
    var d = array([1000, 2000, 3000], int64)
    var batch = record_batch(
        [a.copy(), b.copy(), c.copy(), d.copy()], names=["c0", "c1", "c2", "c3"]
    )

    # Create a chained expression: a + b + c + d
    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var col_c = Column[Int64Type](2)
    var col_d = Column[Int64Type](3)
    var add_ab = Add(col_a, col_b)
    var add_abc = Add(add_ab, col_c)
    var fused = Add(add_abc, col_d)

    # Box it into a runtime Expr
    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    # Execute via the direct typed path
    var direct_result = _exec_typed(fused, batch)

    # Results should match: a + b + c + d = [1111, 2222, 3333]
    assert_true(direct_result == array([1111, 2222, 3333], int64))

    # Execute via the boxed runtime path — same fused pass, same result
    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == direct_result.copy().to_any())


# ---------------------------------------------------------------------------
# Type safety tests
# ---------------------------------------------------------------------------


def test_fused_column_different_types_to_expr() raises:
    """Column with different numeric types can be boxed."""
    # Test Int64
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])

    var col_i64 = Column[Int64Type](0)
    var expr_i64 = Expr(col_i64)
    assert_equal(expr_i64.kind(), FUSED)

    var result_i64 = _exec_typed(col_i64, batch)
    assert_true(result_i64 == array([1, 2, 3], int64))

    var boxed_result = expr_i64.eval(batch)
    assert_true(boxed_result == result_i64.copy().to_any())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element_fused_to_expr() raises:
    """Single-element fused expressions can be boxed and evaluated."""
    var a = array([42], int64)
    var b = array([8], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])

    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var fused = Add(col_a, col_b)

    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    var result = _exec_typed(fused, batch)
    assert_equal(result[0].value(), 50)

    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == result.copy().to_any())


def test_negative_values_fused_to_expr() raises:
    """Fused expressions with negative values work correctly when boxed."""
    var a = array([-1, 2, -3, 4], int64)
    var b = array([1, -2, 3, -4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var fused = Add(col_a, col_b)

    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    var result = _exec_typed(fused, batch)
    assert_true(result == array([0, 0, 0, 0], int64))

    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == result.copy().to_any())


def test_large_values_fused_to_expr() raises:
    """Fused expressions with large int64 values work correctly when boxed."""
    var a = array([9_223_372_036_854_775_806], int64)  # near max int64
    var b = array([1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])

    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var fused = Add(col_a, col_b)

    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    var result = _exec_typed(fused, batch)
    assert_equal(result[0].value(), 9_223_372_036_854_775_807)  # max int64

    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == result.copy().to_any())


# ---------------------------------------------------------------------------
# Non-alignment tests
# ---------------------------------------------------------------------------


def test_non_aligned_length_fused_to_expr() raises:
    """Fused expressions work with non-SIMD-aligned lengths when boxed."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)  # 7 elements, not SIMD-aligned
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    var col_a = Column[Int64Type](0)
    var col_b = Column[Int64Type](1)
    var fused = Add(col_a, col_b)

    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    var result = _exec_typed(fused, batch)
    assert_true(result == array([11, 22, 33, 44, 55, 66, 77], int64))

    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == result.copy().to_any())


# ---------------------------------------------------------------------------
# BoolValue injection tests (Lt/Gt/Eq boxed into Expr via FUSED)
# ---------------------------------------------------------------------------


def test_fused_gt_to_expr() raises:
    """Expr(Gt(Column, Column)) produces a valid runtime Expr with FUSED tag,
    and both the direct and boxed paths agree.
    """
    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])

    var fused = Gt(Column[Int64Type](0), Column[Int64Type](1))
    var expr = Expr(fused)
    assert_equal(expr.kind(), FUSED)

    var direct_result = fused.execute(batch)
    var boxed_result = expr.eval(batch)
    assert_true(boxed_result == direct_result.copy().to_any())

    assert_true(not direct_result[0].value())
    assert_true(direct_result[1].value())
    assert_true(not direct_result[2].value())
    assert_true(direct_result[3].value())
    assert_true(not direct_result[4].value())


def test_fused_bool_value_drives_runtime_relational_plan() raises:
    """A boxed BoolValue predicate can drive AnyRelation.filter() -- the
    'hybrid' pattern: runtime relational structure, AOT-fused predicate.
    """
    from marrow.dyn import in_memory_table, execute

    var a = array([1, 5, 3, 8, 2], int64)
    var b = array([4, 4, 4, 4, 4], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["a", "b"])

    var predicate = Expr(Gt(Column[Int64Type](0), Column[Int64Type](1)))
    var plan = in_memory_table(batch).filter(predicate)
    var result = execute(plan)

    assert_equal(result.num_rows(), 2)
    ref col_a = result.columns[0].as_int64()
    assert_true(col_a.copy() == array([5, 8], int64))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    TestSuite.run[__functions_in_module()]()
