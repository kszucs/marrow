from std.testing import assert_equal, assert_true, assert_false

from ...arrays import (
    PrimitiveArray,
    BoolArray,
    AnyArray,
    Int64Array,
    Int32Array,
    TimestampArray,
)
from ...builders import array, PrimitiveBuilder
from ...dtypes import (
    int64,
    float64,
    bool_ as bool_dt,
    Int64Type,
    int32,
    string,
    timestamp,
    second,
    TimestampType,
)
from ...kernels.arithmetic import (
    AddKernel,
    SubKernel,
    AbsKernel,
    NegKernel,
    ModKernel,
    FloordivKernel,
)
from ...kernels.boolean import XorKernel
from ...kernels.string import LengthKernel, LikeKernel, ILikeKernel
from ...kernels.compare import LtKernel, LeKernel, GtKernel, GeKernel
from ...kernels.membership import is_in
from ...kernels.temporal import (
    YearKernel,
    MonthKernel,
    DayKernel,
    HourKernel,
    MinuteKernel,
    SecondKernel,
    DayOfWeekKernel,
    QuarterKernel,
    DayOfYearKernel,
    date_trunc,
)
from ...tabular import RecordBatch, record_batch
from ...expr import (
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

# The op tags added in this task are not re-exported from ``marrow.expr`` yet
# (that's a sibling packaging task), so import them from the module directly.
from ...expr.dynamic import (
    MOD,
    FLOORDIV,
    XOR,
    NOT_NULL,
    LIKE,
    ILIKE,
    IS_IN,
    COALESCE,
    NULLIF,
    CASE_WHEN,
    YEAR,
    MONTH,
    DAY,
    HOUR,
    MINUTE,
    SECOND,
    DAY_OF_WEEK,
    QUARTER,
    DAY_OF_YEAR,
    DATE_TRUNC,
    coalesce,
    case_when,
)


def _exec(expr: DynValue, batch: RecordBatch) raises -> Int64Array:
    """Helper: evaluate an expression against the batch."""
    var tmp = expr.eval(batch)
    ref result = tmp.as_int64()
    return result.copy()


def _exec_length(expr: DynValue, batch: RecordBatch) raises -> Int32Array:
    """Helper: evaluate a length expression against the batch."""
    var tmp = expr.eval(batch)
    ref result = tmp.as_int32()
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
    """``.length()`` matches kernels.string.LengthKernel."""
    var a = array(["ab", "cde", "", "f"])
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _exec_length(col(0).length(), batch)
    assert_true(result == LengthKernel.apply(a))


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


# ---------------------------------------------------------------------------
# New op tags: mod / floordiv / xor / not_null
# ---------------------------------------------------------------------------


def test_mod_expr() raises:
    """Operator % matches kernels.mod."""
    var a = array([10, 21, 33, 47, 5], int64)
    var b = array([3, 5, 4, 10, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) % col(1), batch)
    assert_true(result == ModKernel.apply[Int64Type](a, b))


def test_floordiv_expr() raises:
    """Operator // matches kernels.floordiv."""
    var a = array([10, 21, 33, 47, 5], int64)
    var b = array([3, 5, 4, 10, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col(0) // col(1), batch)
    assert_true(result == FloordivKernel.apply[Int64Type](a, b))


def test_xor_pred() raises:
    """Operator ^ matches kernels.boolean.xor over two bool masks."""
    var a = array([True, True, False, False])
    var b = array([True, False, True, False])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec_pred(col(0) ^ col(1), batch)
    assert_true(result == XorKernel.apply(a, b))


def test_not_null_pred() raises:
    """``not_null()`` is True for valid elements (all-valid input -> all True).
    """
    var a = array([1, 2, 3], int64)
    var result = _exec_pred(col(0).not_null(), record_batch([a^], names=["c0"]))
    assert_true(result == array([True, True, True]))


def test_kind_mod() raises:
    """``%`` node reports MOD kind and prints as mod(...)."""
    var expr = col(0) % col(1)
    assert_equal(expr.kind(), MOD)
    assert_equal(String(expr), "mod(input(0), input(1))")


def test_kind_floordiv() raises:
    var expr = col(0) // col(1)
    assert_equal(expr.kind(), FLOORDIV)
    assert_equal(String(expr), "floordiv(input(0), input(1))")


def test_kind_xor() raises:
    var expr = col(0) ^ col(1)
    assert_equal(expr.kind(), XOR)
    assert_equal(String(expr), "xor(input(0), input(1))")


def test_kind_not_null() raises:
    var expr = col(0).not_null()
    assert_equal(expr.kind(), NOT_NULL)
    assert_equal(String(expr), "not_null(input(0))")


# ---------------------------------------------------------------------------
# Plan-analysis metadata
# ---------------------------------------------------------------------------


def test_referenced_columns_named() raises:
    """Named LOAD leaves are collected in first-seen order, deduped."""
    var expr = (col("a") + col("b")) * col("a")
    var cols = expr.referenced_columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "a")
    assert_equal(cols[1], "b")


def test_referenced_columns_positional() raises:
    """Positional LOAD leaves render their index as a string, deduped."""
    var expr = (col(0) + col(2)) - col(0)
    var cols = expr.referenced_columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "0")
    assert_equal(cols[1], "2")


def test_referenced_columns_literal_only() raises:
    """An expression over only literals references no columns."""
    var expr = lit[Int64Type](1) + lit[Int64Type](2)
    assert_equal(len(expr.referenced_columns()), 0)


def test_is_deterministic() raises:
    """All currently supported tags are deterministic."""
    assert_true((col(0) % col(1)).is_deterministic())
    assert_true(if_else(col(0) > col(1), col(0), col(1)).is_deterministic())
    assert_true(col(0).cast(float64).is_deterministic())


# ---------------------------------------------------------------------------
# String ordering comparisons (< <= > >= route to compare.mojo string dispatch)
# ---------------------------------------------------------------------------


def test_string_less_pred() raises:
    """LT on string columns compares lexicographically via compare dispatch."""
    var a = array(["apple", "banana", "cherry", "date"])
    var b = array(["apricot", "banana", "blueberry", "durian"])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec_pred(col(0) < col(1), batch)
    assert_true(result == LtKernel.StringKernel.apply(a, b))
    assert_true(result == array([True, False, False, True]))


def test_string_all_compares() raises:
    """<=, >, >= all evaluate on strings and match the compare kernels."""
    var a = array(["a", "bb", "c"])
    var b = array(["a", "b", "cc"])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    assert_true(
        _exec_pred(col(0) <= col(1), batch) == LeKernel.StringKernel.apply(a, b)
    )
    assert_true(
        _exec_pred(col(0) > col(1), batch) == GtKernel.StringKernel.apply(a, b)
    )
    assert_true(
        _exec_pred(col(0) >= col(1), batch) == GeKernel.StringKernel.apply(a, b)
    )


def test_string_equal_pred() raises:
    """EQ on strings routes through the string equality free function."""
    var a = array(["x", "y", "z"])
    var b = array(["x", "Y", "z"])
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec_pred(col(0) == col(1), batch)
    assert_true(result == array([True, False, True]))


# ---------------------------------------------------------------------------
# like / ilike  (kernels.string.LikeKernel / ILikeKernel)
# ---------------------------------------------------------------------------


def test_like_expr() raises:
    """``.like`` matches SQL LIKE semantics (case-sensitive)."""
    var a = array(["apple", "banana", "apricot", "cherry"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col(0).like("a%"), batch)
    assert_true(result == array([True, False, True, False]))


def test_like_underscore() raises:
    """``_`` matches exactly one character."""
    var a = array(["cat", "cot", "cart", "ct"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col(0).like("c_t"), batch)
    assert_true(result == array([True, True, False, False]))


def test_ilike_expr() raises:
    """``.ilike`` is case-insensitive."""
    var a = array(["Apple", "BANANA", "apricot", "Cherry"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col(0).ilike("a%"), batch)
    assert_true(result == array([True, False, True, False]))


def test_like_kind_and_write_to() raises:
    var expr = col(0).like("a%")
    assert_equal(expr.kind(), LIKE)
    assert_equal(String(expr), "match_like(input(0), a%)")
    var iexpr = col(0).ilike("a%")
    assert_equal(iexpr.kind(), ILIKE)
    assert_equal(String(iexpr), "match_like_ci(input(0), a%)")


def test_like_referenced_columns() raises:
    """The pattern is a literal, so only the operand column is referenced."""
    var cols = col("s").like("a%").referenced_columns()
    assert_equal(len(cols), 1)
    assert_equal(cols[0], "s")


# ---------------------------------------------------------------------------
# is_in  (kernels.membership.is_in)
# ---------------------------------------------------------------------------


def test_isin_int_expr() raises:
    """ClickBench ``x IN (-1, 6)`` shape via ``.isin``."""
    var a = array([-1, 0, 6, 3, 6], int64)
    var value_set: AnyArray = array([-1, 6], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _exec_pred(col(0).isin(value_set), batch)
    assert_true(result == array([True, False, True, False, True]))
    assert_true(result == is_in(a, array([-1, 6], int64)))


def test_isin_string_expr() raises:
    var a = array(["apple", "banana", "cherry", "apple"])
    var value_set: AnyArray = array(["apple", "cherry"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col(0).isin(value_set), batch)
    assert_true(result == array([True, False, True, True]))


def test_isin_kind_and_metadata() raises:
    var value_set: AnyArray = array([1, 2], int64)
    var expr = col(0).isin(value_set)
    assert_equal(expr.kind(), IS_IN)
    assert_equal(String(expr), "is_in(input(0), value_set)")
    # referenced_columns resolves named LOADs (write_to renders the position).
    var cols = col("k").isin(value_set).referenced_columns()
    assert_equal(len(cols), 1)
    assert_equal(cols[0], "k")


# ---------------------------------------------------------------------------
# coalesce / nullif / case_when  (kernels.conditional)
# ---------------------------------------------------------------------------


def _with_nulls(values: List[Int], valid: List[Bool]) raises -> Int64Array:
    var b = PrimitiveBuilder[Int64Type](int64, capacity=len(values))
    for i in range(len(values)):
        if valid[i]:
            b.append(Int64(values[i]))
        else:
            b.append_null()
    return b.finish()


def test_coalesce_expr() raises:
    """First non-null across columns."""
    var a = _with_nulls([1, 0, 0, 4], [True, False, False, True])
    var b = _with_nulls([0, 2, 0, 5], [False, True, False, True])
    var c = _with_nulls([0, 0, 3, 6], [False, False, True, True])
    var batch = record_batch([a^, b^, c^], names=["c0", "c1", "c2"])
    var result = _exec(coalesce([col(0), col(1), col(2)]), batch)
    assert_true(result == array([1, 2, 3, 4], int64))


def test_nullif_expr() raises:
    """``a.nullif(b)`` nulls elements where a == b."""
    var a = array([1, 2, 3, 4], int64)
    var b = array([9, 2, 9, 4], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var tmp = col(0).nullif(col(1)).eval(batch)
    ref result = tmp.as_int64()
    assert_true(result[0].value() == 1)
    assert_false(result[1].is_valid())
    assert_true(result[2].value() == 3)
    assert_false(result[3].is_valid())


def test_case_when_expr() raises:
    """Two-branch CASE WHEN with an else fallback."""
    var x = array([1, 5, 10, 15], int64)
    var lo = array([100, 100, 100, 100], int64)
    var hi = array([200, 200, 200, 200], int64)
    var els = array([300, 300, 300, 300], int64)
    var batch = record_batch(
        [x^, lo^, hi^, els^], names=["x", "lo", "hi", "els"]
    )
    # x < 5 -> lo ; x < 12 -> hi ; else -> els
    var conds = [col(0) < lit[Int64Type](5), col(0) < lit[Int64Type](12)]
    var vals = [col(1), col(2)]
    var result = _exec(case_when(conds, vals, col(3)), batch)
    assert_true(result == array([100, 200, 200, 300], int64))


def test_case_when_no_else_nulls() raises:
    """Rows matching no branch and no else become null."""
    var x = array([1, 9], int64)
    var v = array([100, 100], int64)
    var batch = record_batch([x^, v^], names=["x", "v"])
    var conds = [col(0) < lit[Int64Type](5)]
    var vals = [col(1)]
    var tmp = case_when(conds, vals).eval(batch)
    ref result = tmp.as_int64()
    assert_true(result[0].value() == 100)
    assert_false(result[1].is_valid())


def test_conditional_kinds_and_columns() raises:
    var coa = coalesce([col("a"), col("b")])
    assert_equal(coa.kind(), COALESCE)
    var ccols = coa.referenced_columns()
    assert_equal(len(ccols), 2)
    assert_equal(ccols[0], "a")
    assert_equal(ccols[1], "b")

    var nif = col(0).nullif(col(1))
    assert_equal(nif.kind(), NULLIF)
    assert_equal(String(nif), "nullif(input(0), input(1))")

    var cw = case_when([col("p") > col("q")], [col("r")], col("s"))
    assert_equal(cw.kind(), CASE_WHEN)
    var cwcols = cw.referenced_columns()
    assert_equal(len(cwcols), 4)  # p, q, r, s


# ---------------------------------------------------------------------------
# Temporal extraction + date_trunc  (kernels.temporal)
# ---------------------------------------------------------------------------


def _ts(values: List[Int]) raises -> TimestampArray:
    var b = PrimitiveBuilder[TimestampType](
        timestamp(second), capacity=len(values)
    )
    for v in values:
        b.append(Int64(v))
    return b.finish()


def test_year_month_day_expr() raises:
    # 2019-06-15 12:30:45, 2020-02-29 00:00:00, 1970-01-01 00:00:00
    var a = _ts([1_560_601_845, 1_582_934_400, 0])
    var batch = record_batch([a.copy()], names=["c0"])
    assert_true(
        _exec_length(col(0).year(), batch) == array([2019, 2020, 1970], int32)
    )
    assert_true(_exec_length(col(0).month(), batch) == array([6, 2, 1], int32))
    assert_true(_exec_length(col(0).day(), batch) == array([15, 29, 1], int32))
    # wiring cross-check against the kernel directly
    assert_true(_exec_length(col(0).year(), batch) == YearKernel.apply(a))


def test_hour_minute_second_expr() raises:
    var a = _ts([1_560_601_845])  # 12:30:45 UTC
    var batch = record_batch([a^], names=["c0"])
    assert_true(_exec_length(col(0).hour(), batch) == array([12], int32))
    assert_true(_exec_length(col(0).minute(), batch) == array([30], int32))
    assert_true(_exec_length(col(0).second(), batch) == array([45], int32))


def test_quarter_dow_doy_expr() raises:
    # 2019-01-01 (Tue), day-of-year 1, quarter 1
    var a = _ts([1_546_300_800])
    var batch = record_batch([a^], names=["c0"])
    assert_true(_exec_length(col(0).quarter(), batch) == array([1], int32))
    assert_true(_exec_length(col(0).day_of_week(), batch) == array([1], int32))
    assert_true(_exec_length(col(0).day_of_year(), batch) == array([1], int32))


def test_date_trunc_expr() raises:
    var a = _ts([1_560_601_845])  # 2019-06-15 12:30:45
    var batch = record_batch([a.copy()], names=["c0"])
    var tmp = col(0).date_trunc("hour").eval(batch)
    assert_true(tmp.dtype() == timestamp(second))
    # 12:30:45 floored to the hour -> 12:00:00 == 1_560_600_000
    assert_true(tmp.as_timestamp() == _ts([1_560_600_000]))
    assert_true(
        tmp.as_timestamp() == date_trunc(a.copy(), "hour").as_timestamp()
    )


def test_temporal_kinds_and_write_to() raises:
    assert_equal(col(0).year().kind(), YEAR)
    assert_equal(col(0).month().kind(), MONTH)
    assert_equal(col(0).day().kind(), DAY)
    assert_equal(col(0).hour().kind(), HOUR)
    assert_equal(col(0).minute().kind(), MINUTE)
    assert_equal(col(0).second().kind(), SECOND)
    assert_equal(col(0).day_of_week().kind(), DAY_OF_WEEK)
    assert_equal(col(0).quarter().kind(), QUARTER)
    assert_equal(col(0).day_of_year().kind(), DAY_OF_YEAR)
    assert_equal(String(col(0).year()), "year(input(0))")
    var dt = col(0).date_trunc("day")
    assert_equal(dt.kind(), DATE_TRUNC)
    assert_equal(String(dt), "date_trunc(input(0), day)")


def test_temporal_referenced_columns() raises:
    var cols = col("ts").year().referenced_columns()
    assert_equal(len(cols), 1)
    assert_equal(cols[0], "ts")
