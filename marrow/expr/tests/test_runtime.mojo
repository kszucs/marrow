from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import (
    PrimitiveArray,
    BoolArray,
    AnyArray,
    Int64Array,
    UInt32Array,
)
from marrow.builders import array
from marrow.dtypes import int64, float64, bool_ as bool_dt, uint32, Int64Type
from marrow.kernels.arithmetic import (
    AddKernel,
    SubKernel,
    AbsKernel,
    NegKernel,
)
from marrow.kernels.string import string_lengths
from marrow.tabular import RecordBatch, record_batch
from marrow.expr import (
    DynValue,
    col,
    lit,
    if_else,
    LOAD,
    LITERAL,
    ADD,
    SUB,
    MUL,
    DIV,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    AND,
    OR,
    NEG,
    ABS,
    NOT,
    IS_NULL,
    IF_ELSE,
    LENGTH,
    CAST,
)


def _exec(expr: DynValue, batch: RecordBatch) raises -> Int64Array:
    """Helper: evaluate an expression against the batch."""
    var tmp = expr.eval(batch)
    ref result = tmp.as_int64()
    return result.copy()


def _exec_length(expr: DynValue, batch: RecordBatch) raises -> UInt32Array:
    """Helper: evaluate a length expression against the batch."""
    var tmp = expr.eval(batch)
    ref result = tmp.as_uint32()
    return result.copy()


def _exec_pred(expr: DynValue, batch: RecordBatch) raises -> BoolArray:
    """Helper: evaluate a predicate expression against the batch."""
    var tmp = expr.eval(batch)
    return tmp.as_bool().copy()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_add_expr() raises:
    """Operator + matches kernels.add."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) + col(1), batch)
    assert_true(result == AddKernel.apply[Int64Type](a, b))


def test_sub_expr() raises:
    """Operator - matches kernels.sub."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) - col(1), batch)
    assert_true(result == SubKernel.apply[Int64Type](a, b))


def test_neg_expr() raises:
    """Operator -x matches kernels.neg."""
    var a = array([1, -2, 3, -4, 5], int64)
    var result = _exec(-col(0), record_batch([a.copy()], names=["c0"]))
    assert_true(result == NegKernel.apply[Int64Type](a))


def test_abs_expr() raises:
    """Method .abs() matches kernels.abs_."""
    var a = array([-1, -2, 3, -4, 5], int64)
    var result = _exec(col(0).abs(), record_batch([a.copy()], names=["c0"]))
    assert_true(result == AbsKernel.apply[Int64Type](a))


# ---------------------------------------------------------------------------
# Chained expressions
# ---------------------------------------------------------------------------


def test_abs_of_sub() raises:
    """Expression abs(a - b) matches abs_(sub(a, b))."""
    var a = array([1, 5, 3, 10, 2], int64)
    var b = array([5, 1, 3, 2, 10], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec((col(0) - col(1)).abs(), batch)
    assert_true(
        result == AbsKernel.apply[Int64Type](SubKernel.apply[Int64Type](a, b))
    )


def test_diff_of_squares() raises:
    """Expression (a + b) * (a - b) matches manual computation."""
    var a = array([3, 5, 7, 9, 11], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec((col(0) + col(1)) * (col(0) - col(1)), batch)
    assert_true(result == array([8, 21, 40, 65, 96], int64))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element() raises:
    """Expression works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var result = _exec(
        col(0) + col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0].value(), 50)


def test_non_aligned_length() raises:
    """Expression works with non-SIMD-aligned lengths."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) + col(1), batch)
    assert_true(result == AddKernel.apply[Int64Type](a, b))


def test_write_to() raises:
    """AnyValue.write_to produces readable expression strings."""
    var expr = (col(0) - col(1)).abs()
    assert_equal(String(expr), "abs(sub(input(0), input(1)))")


# ---------------------------------------------------------------------------
# LITERAL node
# ---------------------------------------------------------------------------


def test_literal_int64() raises:
    """``lit()`` fills the array with the constant value."""
    var a = array([1, 2, 3, 4, 5], int64)
    var result = _exec(lit[Int64Type](10), record_batch([a^], names=["c0"]))
    assert_true(result == array([10, 10, 10, 10, 10], int64))


def test_add_literal() raises:
    """Adds a + literal(7) == [8, 9, 10, 11, 12]."""
    var a = array([1, 2, 3, 4, 5], int64)
    var result = _exec(
        col(0) + lit[Int64Type](7), record_batch([a^], names=["c0"])
    )
    assert_true(result == array([8, 9, 10, 11, 12], int64))


# ---------------------------------------------------------------------------
# Comparison (predicates)
# ---------------------------------------------------------------------------


def test_equal_pred() raises:
    """EQ returns True where a == b."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([1, 0, 3, 0, 5], int64)
    var result = _exec_pred(
        col(0) == col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_false(result[3].value())
    assert_true(result[4].value())


def test_less_pred() raises:
    """LT returns True where a < b."""
    var a = array([1, 5, 3, 10], int64)
    var b = array([5, 1, 3, 20], int64)
    var result = _exec_pred(
        col(0) < col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())


def test_greater_equal_pred() raises:
    """GE returns True where a >= b."""
    var a = array([5, 1, 3, 20], int64)
    var b = array([1, 5, 3, 10], int64)
    var result = _exec_pred(
        col(0) >= col(1), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_true(result[2].value())
    assert_true(result[3].value())


# ---------------------------------------------------------------------------
# Boolean (AND / OR / NOT)
# ---------------------------------------------------------------------------


def test_and_pred() raises:
    """AND: True only where both sides are True."""
    var a = array([1, 2, 3, 4], int64)
    var b = array([2, 2, 2, 2], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec_pred(
        (col(0) < col(1)) & (col(0) != lit[Int64Type](3)), batch
    )
    assert_true(result[0].value())
    assert_false(result[1].value())
    assert_false(result[2].value())
    assert_false(result[3].value())


def test_not_pred() raises:
    """NOT inverts a boolean expression."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([3, 3, 3, 3, 3], int64)
    var result = _exec_pred(
        ~(col(0) == col(1)), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_true(result[0].value())
    assert_true(result[1].value())
    assert_false(result[2].value())
    assert_true(result[3].value())
    assert_true(result[4].value())


# ---------------------------------------------------------------------------
# IF_ELSE
# ---------------------------------------------------------------------------


def test_if_else() raises:
    """``if_else`` selects from two arrays based on a bool condition."""
    var a = array([1, 5, 3, 10], int64)
    var b = array([9, 2, 3, 1], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec(if_else(col(0) > col(1), col(0), col(1)), batch)
    assert_equal(result[0].value(), 9)
    assert_equal(result[1].value(), 5)
    assert_equal(result[2].value(), 3)
    assert_equal(result[3].value(), 10)


# ---------------------------------------------------------------------------
# IS_NULL
# ---------------------------------------------------------------------------


def test_is_null() raises:
    """``is_null()`` is True for null elements, False for valid ones."""
    var a = array([1, 2, 3], int64)
    var result = _exec_pred(col(0).is_null(), record_batch([a^], names=["c0"]))
    assert_true(result == array([False, False, False]))


# ---------------------------------------------------------------------------
# LENGTH (string)
# ---------------------------------------------------------------------------


def test_length_expr() raises:
    """``.length()`` matches kernels.string.string_lengths."""
    var a = array(["ab", "cde", "", "f"])
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _exec_length(col(0).length(), batch)
    assert_true(result == string_lengths(a))


def test_kind_length() raises:
    """``.length()`` node reports LENGTH kind."""
    var expr = col(0).length()
    assert_equal(expr.kind(), LENGTH)


def test_length_write_to() raises:
    """``.length()`` formats as length(...)."""
    var expr = col(0).length()
    assert_equal(String(expr), "length(input(0))")


# ---------------------------------------------------------------------------
# Kind / inputs
# ---------------------------------------------------------------------------


def test_kind_column() raises:
    """Column node reports LOAD kind."""
    var expr = col(0)
    assert_equal(expr.kind(), LOAD)


def test_kind_literal() raises:
    """Literal node reports LITERAL kind."""
    var expr = lit[Int64Type](42)
    assert_equal(expr.kind(), LITERAL)


def test_kind_binary() raises:
    """Binary node reports its op as kind."""
    var expr = col(0) + col(1)
    assert_equal(expr.kind(), ADD)


def test_dyn_cast_eval() raises:
    """A cast node evaluates via the router and matches the eager kernel."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var tmp = col(0).cast(float64).eval(batch)
    assert_true(tmp.dtype() == float64)
    assert_true(tmp.as_float64() == array([1.0, 2.0, 3.0], float64))


def test_dyn_cast_dtype_and_kind() raises:
    var expr = col(0).cast(float64)
    assert_equal(expr.kind(), CAST)
    assert_true(expr.dtype().value() == float64)


def main() raises:
    TestSuite.run[__functions_in_module()]()
