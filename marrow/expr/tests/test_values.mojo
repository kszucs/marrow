from std.testing import assert_equal, assert_true, TestSuite

from marrow.arrays import array, PrimitiveArray, AnyArray
from marrow.dtypes import int64, float64, bool_ as bool_dt
from marrow.kernels.arithmetic import add, sub, abs_ as k_abs, neg as k_neg
from marrow.tabular import RecordBatch, record_batch
from marrow.expr import (
    AnyValue,
    Column,
    Binary,
    Planner,
    col,
    lit,
    if_else,
    DISPATCH_AUTO,
    DISPATCH_CPU,
    DISPATCH_GPU,
)


def _exec(expr: AnyValue, batch: RecordBatch) raises -> PrimitiveArray[int64]:
    """Helper: build a value processor and evaluate against the batch."""
    var tmp = Planner().build(expr).eval(batch)
    ref result = tmp.as_primitive[int64]()
    return result.copy()


def _exec_pred(
    expr: AnyValue, batch: RecordBatch
) raises -> PrimitiveArray[bool_dt]:
    """Helper: build a value processor and evaluate predicate against the batch.
    """
    var tmp = Planner().build(expr).eval(batch)
    ref result = tmp.as_primitive[bool_dt]()
    return result.copy()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_add_expr() raises:
    """Operator + matches kernels.add."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([10, 20, 30, 40, 50])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) + col(1), batch)
    assert_true(result == add[int64](a, b))


def test_sub_expr() raises:
    """Operator - matches kernels.sub."""
    var a = array[int64]([10, 20, 30, 40, 50])
    var b = array[int64]([1, 2, 3, 4, 5])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) - col(1), batch)
    assert_true(result == sub[int64](a, b))


def test_neg_expr() raises:
    """Operator -x matches kernels.neg."""
    var a = array[int64]([1, -2, 3, -4, 5])
    var result = _exec(-col(0), record_batch([a.copy()], names=["c0"]))
    assert_true(result == k_neg[int64](a))


def test_abs_expr() raises:
    """Method .abs() matches kernels.abs_."""
    var a = array[int64]([-1, -2, 3, -4, 5])
    var result = _exec(col(0).abs(), record_batch([a.copy()], names=["c0"]))
    assert_true(result == k_abs[int64](a))


# ---------------------------------------------------------------------------
# Chained expressions
# ---------------------------------------------------------------------------


def test_abs_of_sub() raises:
    """Expression abs(a - b) matches abs_(sub(a, b))."""
    var a = array[int64]([1, 5, 3, 10, 2])
    var b = array[int64]([5, 1, 3, 2, 10])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec((col(0) - col(1)).abs(), batch)
    assert_true(result == k_abs[int64](sub[int64](a, b)))


def test_diff_of_squares() raises:
    """Expression (a + b) * (a - b) matches manual computation."""
    var a = array[int64]([3, 5, 7, 9, 11])
    var b = array[int64]([1, 2, 3, 4, 5])
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec((col(0) + col(1)) * (col(0) - col(1)), batch)
    assert_true(result == array[int64]([8, 21, 40, 65, 96]))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element() raises:
    """Expression works with a single-element array."""
    var a = array[int64]([42])
    var b = array[int64]([8])
    var result = _exec(
        col(0) + col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0], 50)


def test_non_aligned_length() raises:
    """Expression works with non-SIMD-aligned lengths."""
    var a = array[int64]([1, 2, 3, 4, 5, 6, 7])
    var b = array[int64]([10, 20, 30, 40, 50, 60, 70])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) + col(1), batch)
    assert_true(result == add[int64](a, b))


def test_write_to() raises:
    """AnyValue.write_to produces readable expression strings."""
    var expr = (col(0) - col(1)).abs()
    assert_equal(String(expr), "abs(sub(input(0), input(1)))")


# ---------------------------------------------------------------------------
# LITERAL node
# ---------------------------------------------------------------------------


def test_literal_int64() raises:
    """``lit()`` fills the array with the constant value."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var result = _exec(lit[int64](10), record_batch([a^], names=["c0"]))
    assert_true(result == array[int64]([10, 10, 10, 10, 10]))


def test_add_literal() raises:
    """Adds a + literal(7) == [8, 9, 10, 11, 12]."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var result = _exec(col(0) + lit[int64](7), record_batch([a^], names=["c0"]))
    assert_true(result == array[int64]([8, 9, 10, 11, 12]))


# ---------------------------------------------------------------------------
# Comparison (predicates)
# ---------------------------------------------------------------------------


def test_equal_pred() raises:
    """EQ returns True where a == b."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([1, 0, 3, 0, 5])
    var result = _exec_pred(
        col(0) == col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0], 1)
    assert_equal(result[1], 0)
    assert_equal(result[2], 1)
    assert_equal(result[3], 0)
    assert_equal(result[4], 1)


def test_less_pred() raises:
    """LT returns True where a < b."""
    var a = array[int64]([1, 5, 3, 10])
    var b = array[int64]([5, 1, 3, 20])
    var result = _exec_pred(
        col(0) < col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0], 1)
    assert_equal(result[1], 0)
    assert_equal(result[2], 0)
    assert_equal(result[3], 1)


def test_greater_equal_pred() raises:
    """GE returns True where a >= b."""
    var a = array[int64]([5, 1, 3, 20])
    var b = array[int64]([1, 5, 3, 10])
    var result = _exec_pred(
        col(0) >= col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0], 1)
    assert_equal(result[1], 0)
    assert_equal(result[2], 1)
    assert_equal(result[3], 1)


# ---------------------------------------------------------------------------
# Boolean (AND / OR / NOT)
# ---------------------------------------------------------------------------


def test_and_pred() raises:
    """AND: True only where both sides are True."""
    var a = array[int64]([1, 2, 3, 4])
    var b = array[int64]([2, 2, 2, 2])
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec_pred(
        (col(0) < col(1)) & (col(0) != lit[int64](3)), batch
    )
    assert_equal(result[0], 1)
    assert_equal(result[1], 0)
    assert_equal(result[2], 0)
    assert_equal(result[3], 0)


def test_not_pred() raises:
    """NOT inverts a boolean expression."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([3, 3, 3, 3, 3])
    var result = _exec_pred(
        ~(col(0) == col(1)), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0], 1)
    assert_equal(result[1], 1)
    assert_equal(result[2], 0)
    assert_equal(result[3], 1)
    assert_equal(result[4], 1)


# ---------------------------------------------------------------------------
# IF_ELSE
# ---------------------------------------------------------------------------


def test_if_else() raises:
    """``if_else`` selects from two arrays based on a bool condition."""
    var a = array[int64]([1, 5, 3, 10])
    var b = array[int64]([9, 2, 3, 1])
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec(if_else(col(0) > col(1), col(0), col(1)), batch)
    assert_equal(result[0], 9)
    assert_equal(result[1], 5)
    assert_equal(result[2], 3)
    assert_equal(result[3], 10)


# ---------------------------------------------------------------------------
# IS_NULL
# ---------------------------------------------------------------------------


def test_is_null() raises:
    """``is_null()`` is True for null elements, False for valid ones."""
    var a = array[int64]([1, 2, 3])
    var result = _exec_pred(col(0).is_null(), record_batch([a^], names=["c0"]))
    assert_true(result == array([False, False, False]))


# ---------------------------------------------------------------------------
# Dispatch hint
# ---------------------------------------------------------------------------


def test_dispatch_hint_default() raises:
    """Default dispatch is DISPATCH_AUTO."""
    var expr = col(0) + col(1)
    assert_equal(expr.dispatch, DISPATCH_AUTO)


def test_dispatch_hint_cpu() raises:
    """``with_dispatch(DISPATCH_CPU)`` returns a copy with CPU hint."""
    var expr = (col(0) + col(1)).with_dispatch(DISPATCH_CPU)
    assert_equal(expr.dispatch, DISPATCH_CPU)

    var a = array[int64]([1, 2, 3])
    var b = array[int64]([10, 20, 30])
    var result = _exec(expr, record_batch([a^, b^], names=["c0", "c1"]))
    assert_equal(result[0], 11)
    assert_equal(result[1], 22)
    assert_equal(result[2], 33)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_write_to_literal() raises:
    assert_equal(String(lit[int64](5)), "literal(...)")


def test_write_to_equal() raises:
    assert_equal(
        String(col(0) == col(1)),
        "equal(input(0), input(1))",
    )


def test_write_to_if_else() raises:
    var expr = if_else(col(0) < col(1), col(0), col(1))
    assert_equal(
        String(expr),
        "if_else(less(input(0), input(1)), input(0), input(1))",
    )


# ---------------------------------------------------------------------------
# Kind / downcast
# ---------------------------------------------------------------------------


def test_kind_column() raises:
    """Column node reports LOAD kind."""
    from marrow.expr import LOAD

    var expr = col(0)
    assert_equal(expr.kind(), LOAD)
    assert_equal(expr.downcast[Column]()[].index, 0)


def test_kind_literal() raises:
    """Literal node reports LITERAL kind."""
    from marrow.expr import LITERAL

    var expr = lit[int64](42)
    assert_equal(expr.kind(), LITERAL)


def test_kind_binary() raises:
    """Binary node reports its op as kind."""
    from marrow.expr import ADD

    var expr = col(0) + col(1)
    assert_equal(expr.kind(), ADD)
    assert_equal(expr.downcast[Binary]()[].op, ADD)


def test_inputs_binary() raises:
    """Binary.inputs() returns two children."""
    var expr = col(0) - col(1)
    var children = expr.inputs()
    assert_equal(len(children), 2)


def test_inputs_leaf() raises:
    """Column.inputs() returns empty list."""
    var expr = col(0)
    assert_equal(len(expr.inputs()), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
