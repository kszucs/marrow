from std.testing import assert_equal, TestSuite

from marrow.arrays import array, PrimitiveArray, Array
from marrow.dtypes import int64, float64, bool_ as bool_dt
from marrow.kernels.arithmetic import add, sub, abs_ as k_abs, neg as k_neg
from marrow.expr import (
    AnyValue,
    Column,
    Binary,
    col,
    lit,
    if_else,
    DISPATCH_AUTO,
    DISPATCH_CPU,
    DISPATCH_GPU,
)
from marrow.expr.executor import execute


fn _make(a: PrimitiveArray[int64]) -> List[Array]:
    var inputs = List[Array]()
    inputs.append(Array(a))
    return inputs^


fn _make(a: PrimitiveArray[int64], b: PrimitiveArray[int64]) -> List[Array]:
    var inputs = List[Array]()
    inputs.append(Array(a))
    inputs.append(Array(b))
    return inputs^


fn _exec(expr: AnyValue, inputs: List[Array]) raises -> PrimitiveArray[int64]:
    """Helper: execute and convert result to typed array."""
    return PrimitiveArray[int64](data=execute(expr, inputs))


fn _exec_pred(
    expr: AnyValue, inputs: List[Array]
) raises -> PrimitiveArray[bool_dt]:
    """Helper: execute predicate and convert result to typed bool array."""
    return PrimitiveArray[bool_dt](data=execute(expr, inputs))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_add_expr() raises:
    """Operator + matches kernels.add."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([10, 20, 30, 40, 50])

    var expr = col(0) + col(1)
    var result = _exec(expr, _make(a, b))
    var expected = add[int64](a, b)

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), expected.unsafe_get(i))


def test_sub_expr() raises:
    """Operator - matches kernels.sub."""
    var a = array[int64]([10, 20, 30, 40, 50])
    var b = array[int64]([1, 2, 3, 4, 5])

    var expr = col(0) - col(1)
    var result = _exec(expr, _make(a, b))
    var expected = sub[int64](a, b)

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), expected.unsafe_get(i))


def test_neg_expr() raises:
    """Operator -x matches kernels.neg."""
    var a = array[int64]([1, -2, 3, -4, 5])

    var expr = -col(0)
    var result = _exec(expr, _make(a))
    var expected = k_neg[int64](a)

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), expected.unsafe_get(i))


def test_abs_expr() raises:
    """Method .abs() matches kernels.abs_."""
    var a = array[int64]([-1, -2, 3, -4, 5])

    var expr = col(0).abs()
    var result = _exec(expr, _make(a))
    var expected = k_abs[int64](a)

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), expected.unsafe_get(i))


# ---------------------------------------------------------------------------
# Chained expressions
# ---------------------------------------------------------------------------


def test_abs_of_sub() raises:
    """Expression abs(a - b) matches abs_(sub(a, b))."""
    var a = array[int64]([1, 5, 3, 10, 2])
    var b = array[int64]([5, 1, 3, 2, 10])

    var expr = (col(0) - col(1)).abs()
    var result = _exec(expr, _make(a, b))
    var expected = k_abs[int64](sub[int64](a, b))

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), expected.unsafe_get(i))


def test_diff_of_squares() raises:
    """Expression (a + b) * (a - b) matches manual computation."""
    var a = array[int64]([3, 5, 7, 9, 11])
    var b = array[int64]([1, 2, 3, 4, 5])

    var expr = (col(0) + col(1)) * (col(0) - col(1))
    var result = _exec(expr, _make(a, b))

    for i in range(len(result)):
        var ai = a.unsafe_get(i)
        var bi = b.unsafe_get(i)
        assert_equal(result.unsafe_get(i), ai * ai - bi * bi)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element() raises:
    """Expression works with a single-element array."""
    var a = array[int64]([42])
    var b = array[int64]([8])

    var expr = col(0) + col(1)
    var result = _exec(expr, _make(a, b))
    assert_equal(result.unsafe_get(0), 50)


def test_non_aligned_length() raises:
    """Expression works with non-SIMD-aligned lengths."""
    var a = array[int64]([1, 2, 3, 4, 5, 6, 7])
    var b = array[int64]([10, 20, 30, 40, 50, 60, 70])

    var expr = col(0) + col(1)
    var result = _exec(expr, _make(a, b))
    var expected = add[int64](a, b)

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), expected.unsafe_get(i))


def test_write_to() raises:
    """AnyValue.write_to produces readable expression strings."""
    var expr = (col(0) - col(1)).abs()
    assert_equal(String(expr), "abs(sub(input(0), input(1)))")


# ---------------------------------------------------------------------------
# LITERAL node
# ---------------------------------------------------------------------------


def test_literal_int64() raises:
    """lit() fills the array with the constant value."""
    var a = array[int64]([1, 2, 3, 4, 5])

    var expr = lit[int64](10)
    var result = _exec(expr, _make(a))

    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), 10)


def test_add_literal() raises:
    """Adds a + literal(7) == [8, 9, 10, 11, 12]."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var expr = col(0) + lit[int64](7)
    var result = _exec(expr, _make(a))
    for i in range(len(result)):
        assert_equal(result.unsafe_get(i), a.unsafe_get(i) + 7)


# ---------------------------------------------------------------------------
# Comparison (predicates)
# ---------------------------------------------------------------------------


def test_equal_pred() raises:
    """EQ returns True where a == b."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([1, 0, 3, 0, 5])

    var expr = col(0) == col(1)
    var result = _exec_pred(expr, _make(a, b))

    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 0)
    assert_equal(result.unsafe_get(2), 1)
    assert_equal(result.unsafe_get(3), 0)
    assert_equal(result.unsafe_get(4), 1)


def test_less_pred() raises:
    """LT returns True where a < b."""
    var a = array[int64]([1, 5, 3, 10])
    var b = array[int64]([5, 1, 3, 20])

    var expr = col(0) < col(1)
    var result = _exec_pred(expr, _make(a, b))

    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 0)
    assert_equal(result.unsafe_get(2), 0)
    assert_equal(result.unsafe_get(3), 1)


def test_greater_equal_pred() raises:
    """GE returns True where a >= b."""
    var a = array[int64]([5, 1, 3, 20])
    var b = array[int64]([1, 5, 3, 10])

    var expr = col(0) >= col(1)
    var result = _exec_pred(expr, _make(a, b))

    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 0)
    assert_equal(result.unsafe_get(2), 1)
    assert_equal(result.unsafe_get(3), 1)


# ---------------------------------------------------------------------------
# Boolean (AND / OR / NOT)
# ---------------------------------------------------------------------------


def test_and_pred() raises:
    """AND: True only where both sides are True."""
    var a = array[int64]([1, 2, 3, 4])
    var b = array[int64]([2, 2, 2, 2])

    var less_expr = col(0) < col(1)
    var ne_expr = col(0) != lit[int64](3)
    var expr = less_expr & ne_expr
    var result = _exec_pred(expr, _make(a, b))

    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 0)
    assert_equal(result.unsafe_get(2), 0)
    assert_equal(result.unsafe_get(3), 0)


def test_not_pred() raises:
    """NOT inverts a boolean expression."""
    var a = array[int64]([1, 2, 3, 4, 5])
    var b = array[int64]([3, 3, 3, 3, 3])

    var expr = ~(col(0) == col(1))
    var result = _exec_pred(expr, _make(a, b))

    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(1), 1)
    assert_equal(result.unsafe_get(2), 0)
    assert_equal(result.unsafe_get(3), 1)
    assert_equal(result.unsafe_get(4), 1)


# ---------------------------------------------------------------------------
# IF_ELSE
# ---------------------------------------------------------------------------


def test_if_else() raises:
    """if_else selects from two arrays based on a bool condition."""
    var a = array[int64]([1, 5, 3, 10])
    var b = array[int64]([9, 2, 3, 1])

    var cond = col(0) > col(1)
    var expr = if_else(cond, col(0), col(1))
    var result = _exec(expr, _make(a, b))

    assert_equal(result.unsafe_get(0), 9)
    assert_equal(result.unsafe_get(1), 5)
    assert_equal(result.unsafe_get(2), 3)
    assert_equal(result.unsafe_get(3), 10)


# ---------------------------------------------------------------------------
# IS_NULL
# ---------------------------------------------------------------------------


def test_is_null() raises:
    """is_null() is True for null elements, False for valid ones."""
    var a = array[int64]([1, 2, 3])
    var expr = col(0).is_null()
    var result = _exec_pred(expr, _make(a))

    for i in range(3):
        assert_equal(result.unsafe_get(i), 0)


# ---------------------------------------------------------------------------
# Dispatch hint
# ---------------------------------------------------------------------------


def test_dispatch_hint_default() raises:
    """Default dispatch is DISPATCH_AUTO."""
    var expr = col(0) + col(1)
    assert_equal(expr.dispatch, DISPATCH_AUTO)


def test_dispatch_hint_cpu() raises:
    """with_dispatch(DISPATCH_CPU) returns a copy with CPU hint."""
    var expr = (col(0) + col(1)).with_dispatch(DISPATCH_CPU)
    assert_equal(expr.dispatch, DISPATCH_CPU)

    var a = array[int64]([1, 2, 3])
    var b = array[int64]([10, 20, 30])
    var result = _exec(expr, _make(a, b))
    assert_equal(result.unsafe_get(0), 11)
    assert_equal(result.unsafe_get(1), 22)
    assert_equal(result.unsafe_get(2), 33)


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
