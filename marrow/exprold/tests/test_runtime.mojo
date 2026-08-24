from std.testing import assert_equal, assert_true, assert_false
from std.utils.numerics import nan

from ...arrays import (
    PrimitiveArray,
    BoolArray,
    DynArray,
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
    Int32Type,
    Float64Type,
    int32,
    string,
    timestamp,
    second,
    TimestampType,
)
from ...kernels.numeric import (
    AddKernel,
    SubKernel,
    AbsKernel,
    NegKernel,
    ModKernel,
    FloordivKernel,
)
from ...kernels.boolean import XorKernel
from ...kernels.string import (
    LengthKernel,
    LikeKernel,
    ILikeKernel,
    StringLtKernel,
    StringLeKernel,
    StringGtKernel,
    StringGeKernel,
)
from ...kernels.numeric import LtKernel, LeKernel, GtKernel, GeKernel
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
    DateTruncKernel,
    unit_hour,
)
from ...tabular import RecordBatch, record_batch
from ...exprold import (
    BoxedValue,
    DynValue,
    col,
    lit,
    if_else,
)

from ...exprold.builders import coalesce, case_when


def _exec(expr: DynValue, batch: RecordBatch) raises -> Int64Array:
    """Helper: evaluate an expression against the batch."""
    var tmp = expr.execute(batch)
    ref result = tmp.as_int64()
    return result.copy()


def _exec_length(expr: DynValue, batch: RecordBatch) raises -> Int32Array:
    """Helper: evaluate a length expression against the batch."""
    var tmp = expr.execute(batch)
    ref result = tmp.as_int32()
    return result.copy()


def _exec_pred(expr: BoxedValue, batch: RecordBatch) raises -> BoolArray:
    """Helper: evaluate a predicate expression against the batch.

    Takes the box rather than `DynValue` so a fused predicate and a tag
    predicate can both be handed to it. `is_null`/`is_valid`/`is_nan` are
    `DynValue`'s own methods and stay in the erased lane; the fused
    `NullPredicate` (bound `A: Value`) still accepts an erased operand and
    reaches this helper through the same implicit box."""
    var tmp = expr.execute(batch)
    return tmp.as_bool().copy()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_add_expr() raises:
    """Operator + matches kernels.add."""
    var a = array([1, 2, 3, 4, 5], int64)
    var b = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col("c0") + col("c1"), batch)
    assert_true(result == AddKernel.apply[Int64Type](a, b))


def test_sub_expr() raises:
    """Operator - matches kernels.sub."""
    var a = array([10, 20, 30, 40, 50], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col("c0") - col("c1"), batch)
    assert_true(result == SubKernel.apply[Int64Type](a, b))


def test_neg_expr() raises:
    """Operator -x matches kernels.neg."""
    var a = array([1, -2, 3, -4, 5], int64)
    var result = _exec(-col("c0"), record_batch([a.copy()], names=["c0"]))
    assert_true(result == NegKernel.apply[Int64Type](a))


def test_abs_expr() raises:
    """Method .abs() matches kernels.abs_."""
    var a = array([-1, -2, 3, -4, 5], int64)
    var result = _exec(col("c0").abs(), record_batch([a.copy()], names=["c0"]))
    assert_true(result == AbsKernel.apply[Int64Type](a))


# ---------------------------------------------------------------------------
# Chained expressions
# ---------------------------------------------------------------------------


def test_abs_of_sub() raises:
    """Expression abs(a - b) matches abs_(sub(a, b))."""
    var a = array([1, 5, 3, 10, 2], int64)
    var b = array([5, 1, 3, 2, 10], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec((col("c0") - col("c1")).abs(), batch)
    assert_true(
        result == AbsKernel.apply[Int64Type](SubKernel.apply[Int64Type](a, b))
    )


def test_diff_of_squares() raises:
    """Expression (a + b) * (a - b) matches manual computation."""
    var a = array([3, 5, 7, 9, 11], int64)
    var b = array([1, 2, 3, 4, 5], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec((col("c0") + col("c1")) * (col("c0") - col("c1")), batch)
    assert_true(result == array([8, 21, 40, 65, 96], int64))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_element() raises:
    """Expression works with a single-element array."""
    var a = array([42], int64)
    var b = array([8], int64)
    var result = _exec(
        col("c0") + col("c1"), record_batch([a^, b^], names=["c0", "c1"])
    )
    assert_equal(result[0].value(), 50)


def test_non_aligned_length() raises:
    """Expression works with non-SIMD-aligned lengths."""
    var a = array([1, 2, 3, 4, 5, 6, 7], int64)
    var b = array([10, 20, 30, 40, 50, 60, 70], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col("c0") + col("c1"), batch)
    assert_true(result == AddKernel.apply[Int64Type](a, b))


def test_literal_int64() raises:
    """``lit()`` fills the array with the constant value."""
    var a = array([1, 2, 3, 4, 5], int64)
    var result = _exec(lit[Int64Type](10), record_batch([a^], names=["c0"]))
    assert_true(result == array([10, 10, 10, 10, 10], int64))


def test_add_literal() raises:
    """Adds a + literal(7) == [8, 9, 10, 11, 12]."""
    var a = array([1, 2, 3, 4, 5], int64)
    var result = _exec(
        col("c0") + lit[Int64Type](7), record_batch([a^], names=["c0"])
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
        col("c0") == col("c1"), record_batch([a^, b^], names=["c0", "c1"])
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
        col("c0") < col("c1"), record_batch([a^, b^], names=["c0", "c1"])
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
        col("c0") >= col("c1"), record_batch([a^, b^], names=["c0", "c1"])
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
        (col("c0") < col("c1")) & (col("c0") != lit[Int64Type](3)), batch
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
        ~(col("c0") == col("c1")), record_batch([a^, b^], names=["c0", "c1"])
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
    var result = _exec(
        if_else(col("c0") > col("c1"), col("c0"), col("c1")), batch
    )
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
    var result = _exec_pred(
        col("c0").is_null(), record_batch([a^], names=["c0"])
    )
    assert_true(result == array([False, False, False]))


def _nullable_c0() raises -> RecordBatch:
    """One int64 column ``c0`` with nulls in positions 1 and 3."""
    var a = array([1, None, 3, None], int64)
    return record_batch([a^], names=["c0"])


def test_dyn_is_null_sees_actual_nulls() raises:
    """The tag node reads the validity bitmap, not the values."""
    var result = _exec_pred(col("c0").is_null(), _nullable_c0())
    assert_true(result == array([False, True, False, True]))


def test_dyn_is_valid_is_the_complement() raises:
    var result = _exec_pred(col("c0").is_valid(), _nullable_c0())
    assert_true(result == array([True, False, True, False]))


def test_dyn_null_predicate_stays_in_the_erased_lane() raises:
    """The point of overriding the `Value` defaults.

    A predicate that returned a fused `NullPredicate` could not be combined with
    a predicate that stayed erased: `BoolValue.__or__` takes a `BoolValue` and a
    `DynValue` is not one, so this expression would not compile at all. It is a
    compile-time property, so merely building the tree is the assertion; the
    result is checked to keep the test honest about what it ran."""
    var expr = col("c0").is_null() | (col("c0") > lit[Int64Type](2))
    var result = _exec_pred(expr, _nullable_c0())
    # null -> True (is_null), 3 > 2 -> True; 1 is neither.
    assert_true(result == array([False, True, True, True]))


def test_dyn_is_nan_over_floats() raises:
    """``is_nan()`` scans values, so it needs a real NaN to be worth anything.
    """
    var fb = PrimitiveBuilder[Float64Type](float64, 3)
    fb.append(Float64(1.0))
    fb.append(nan[DType.float64]())
    fb.append(Float64(3.0))
    var result = _exec_pred(
        col("c0").is_nan(), record_batch([fb.finish().to_dyn()], names=["c0"])
    )
    assert_true(result == array([False, True, False]))


def test_dyn_fill_null_from_literal() raises:
    """Nulls take the literal, valid elements are untouched."""
    var out = col("c0").fill_null(lit[Int64Type](0)).execute(_nullable_c0())
    assert_true(out == array([1, 0, 3, 0], int64).to_dyn())


def test_dyn_fill_null_widens_mixed_numeric_operands() raises:
    """The column is int32 and the replacement int64.

    `FillNullKernel` pins both operands to one dtype, so without the promotion
    in `_fill_null` this raises on a mismatch the caller never wrote — the same
    widening `_binary` and `_compare` already do."""
    var b = PrimitiveBuilder[Int32Type](int32, 3)
    b.append(Int32(1))
    b.append_null()
    b.append(Int32(3))
    var out = (
        col("c0")
        .fill_null(lit[Int64Type](7))
        .execute(record_batch([b.finish().to_dyn()], names=["c0"]))
    )
    assert_true(out == array([1, 7, 3], int64).to_dyn())


# ---------------------------------------------------------------------------
# LENGTH (string)
# ---------------------------------------------------------------------------


def test_length_expr() raises:
    """``.length()`` matches kernels.string.LengthKernel."""
    var a = array(["ab", "cde", "", "f"])
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _exec_length(col("c0").length(), batch)
    assert_true(result == LengthKernel.apply(a))


def test_dyn_cast_eval() raises:
    """A cast node evaluates via the router and matches the eager kernel."""
    var a = array([1, 2, 3], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var tmp = col("c0").cast(float64).execute(batch)
    assert_true(tmp.dtype() == float64)
    assert_true(tmp.as_float64() == array([1.0, 2.0, 3.0], float64))


def test_mod_expr() raises:
    """Operator % matches kernels.mod."""
    var a = array([10, 21, 33, 47, 5], int64)
    var b = array([3, 5, 4, 10, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col("c0") % col("c1"), batch)
    assert_true(result == ModKernel.apply[Int64Type](a, b))


def test_floordiv_expr() raises:
    """Operator // matches kernels.floordiv."""
    var a = array([10, 21, 33, 47, 5], int64)
    var b = array([3, 5, 4, 10, 5], int64)
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec(col("c0") // col("c1"), batch)
    assert_true(result == FloordivKernel.apply[Int64Type](a, b))


def test_xor_pred() raises:
    """Operator ^ matches kernels.boolean.xor over two bool masks."""
    var a = array([True, True, False, False])
    var b = array([True, False, True, False])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec_pred(col("c0") ^ col("c1"), batch)
    assert_true(result == XorKernel.apply(a, b))


def test_not_null_pred() raises:
    """``not_null()`` is True for valid elements (all-valid input -> all True).
    """
    var a = array([1, 2, 3], int64)
    var result = _exec_pred(
        col("c0").is_valid(), record_batch([a^], names=["c0"])
    )
    assert_true(result == array([True, True, True]))


def test_referenced_columns_named() raises:
    """Named LOAD leaves are collected in first-seen order, deduped."""
    var expr = (col("a") + col("b")) * col("a")
    var cols = expr.referenced_columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "a")
    assert_equal(cols[1], "b")


def test_referenced_columns_positional() raises:
    """A column referenced twice appears once, in first-seen order."""
    var expr = (col("c0") + col("c2")) - col("c0")
    var cols = expr.referenced_columns()
    assert_equal(len(cols), 2)
    assert_equal(cols[0], "c0")
    assert_equal(cols[1], "c2")


def test_referenced_columns_literal_only() raises:
    """An expression over only literals references no columns."""
    var expr = lit[Int64Type](1) + lit[Int64Type](2)
    assert_equal(len(expr.referenced_columns()), 0)


# ---------------------------------------------------------------------------
# String ordering comparisons (< <= > >= route to compare.mojo string dispatch)
# ---------------------------------------------------------------------------


def test_string_less_pred() raises:
    """LT on string columns compares lexicographically via compare dispatch."""
    var a = array(["apple", "banana", "cherry", "date"])
    var b = array(["apricot", "banana", "blueberry", "durian"])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    var result = _exec_pred(col("c0") < col("c1"), batch)
    assert_true(result == StringLtKernel.apply(a, b))
    assert_true(result == array([True, False, False, True]))


def test_string_all_compares() raises:
    """<=, >, >= all evaluate on strings and match the compare kernels."""
    var a = array(["a", "bb", "c"])
    var b = array(["a", "b", "cc"])
    var batch = record_batch([a.copy(), b.copy()], names=["c0", "c1"])
    assert_true(
        _exec_pred(col("c0") <= col("c1"), batch) == StringLeKernel.apply(a, b)
    )
    assert_true(
        _exec_pred(col("c0") > col("c1"), batch) == StringGtKernel.apply(a, b)
    )
    assert_true(
        _exec_pred(col("c0") >= col("c1"), batch) == StringGeKernel.apply(a, b)
    )


def test_string_equal_pred() raises:
    """EQ on strings routes through the string equality free function."""
    var a = array(["x", "y", "z"])
    var b = array(["x", "Y", "z"])
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var result = _exec_pred(col("c0") == col("c1"), batch)
    assert_true(result == array([True, False, True]))


# ---------------------------------------------------------------------------
# like / ilike  (kernels.string.LikeKernel / ILikeKernel)
# ---------------------------------------------------------------------------


def test_like_expr() raises:
    """``.like`` matches SQL LIKE semantics (case-sensitive)."""
    var a = array(["apple", "banana", "apricot", "cherry"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col("c0").like("a%"), batch)
    assert_true(result == array([True, False, True, False]))


def test_like_underscore() raises:
    """``_`` matches exactly one character."""
    var a = array(["cat", "cot", "cart", "ct"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col("c0").like("c_t"), batch)
    assert_true(result == array([True, True, False, False]))


def test_ilike_expr() raises:
    """``.ilike`` is case-insensitive."""
    var a = array(["Apple", "BANANA", "apricot", "Cherry"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col("c0").ilike("a%"), batch)
    assert_true(result == array([True, False, True, False]))


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
    var value_set: DynArray = array([-1, 6], int64)
    var batch = record_batch([a.copy()], names=["c0"])
    var result = _exec_pred(col("c0").isin(value_set), batch)
    assert_true(result == array([True, False, True, False, True]))
    assert_true(result == is_in(a, array([-1, 6], int64)))


def test_isin_string_expr() raises:
    var a = array(["apple", "banana", "cherry", "apple"])
    var value_set: DynArray = array(["apple", "cherry"])
    var batch = record_batch([a^], names=["c0"])
    var result = _exec_pred(col("c0").isin(value_set), batch)
    assert_true(result == array([True, False, True, True]))


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
    var result = _exec(coalesce([col("c0"), col("c1"), col("c2")]), batch)
    assert_true(result == array([1, 2, 3, 4], int64))


def test_nullif_expr() raises:
    """``a.nullif(b)`` nulls elements where a == b."""
    var a = array([1, 2, 3, 4], int64)
    var b = array([9, 2, 9, 4], int64)
    var batch = record_batch([a^, b^], names=["c0", "c1"])
    var tmp = col("c0").nullif(col("c1")).execute(batch)
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
    var conds = List[DynValue]()
    conds.append(col("x") < lit[Int64Type](5))
    conds.append(col("x") < lit[Int64Type](12))
    var vals = List[DynValue]()
    vals.append(col("lo"))
    vals.append(col("hi"))
    var result = _exec(case_when(conds, vals, col("els")), batch)
    assert_true(result == array([100, 200, 200, 300], int64))


def test_case_when_no_else_nulls() raises:
    """Rows matching no branch and no else become null."""
    var x = array([1, 9], int64)
    var v = array([100, 100], int64)
    var batch = record_batch([x^, v^], names=["x", "v"])
    var conds = List[DynValue]()
    conds.append(col("x") < lit[Int64Type](5))
    var vals = List[DynValue]()
    vals.append(col("v"))
    var tmp = case_when(conds, vals).execute(batch)
    ref result = tmp.as_int64()
    assert_true(result[0].value() == 100)
    assert_false(result[1].is_valid())


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
        _exec_length(col("c0").year(), batch)
        == array([2019, 2020, 1970], int32)
    )
    assert_true(
        _exec_length(col("c0").month(), batch) == array([6, 2, 1], int32)
    )
    assert_true(
        _exec_length(col("c0").day(), batch) == array([15, 29, 1], int32)
    )
    # wiring cross-check against the kernel directly
    assert_true(_exec_length(col("c0").year(), batch) == YearKernel.apply(a))


def test_hour_minute_second_expr() raises:
    var a = _ts([1_560_601_845])  # 12:30:45 UTC
    var batch = record_batch([a^], names=["c0"])
    assert_true(_exec_length(col("c0").hour(), batch) == array([12], int32))
    assert_true(_exec_length(col("c0").minute(), batch) == array([30], int32))
    assert_true(_exec_length(col("c0").second(), batch) == array([45], int32))


def test_quarter_dow_doy_expr() raises:
    # 2019-01-01 (Tue), day-of-year 1, quarter 1
    var a = _ts([1_546_300_800])
    var batch = record_batch([a^], names=["c0"])
    assert_true(_exec_length(col("c0").quarter(), batch) == array([1], int32))
    assert_true(
        _exec_length(col("c0").day_of_week(), batch) == array([1], int32)
    )
    assert_true(
        _exec_length(col("c0").day_of_year(), batch) == array([1], int32)
    )


def test_date_trunc_expr() raises:
    var a = _ts([1_560_601_845])  # 2019-06-15 12:30:45
    var batch = record_batch([a.copy()], names=["c0"])
    var trunc: DynValue = col("c0").date_trunc("hour")
    var tmp = trunc.execute(batch)
    assert_true(tmp.dtype() == timestamp(second))
    # 12:30:45 floored to the hour -> 12:00:00 == 1_560_600_000
    assert_true(tmp.as_timestamp() == _ts([1_560_600_000]))
    assert_true(
        tmp.as_timestamp()
        == DateTruncKernel.apply(a.copy(), unit_hour).as_timestamp()
    )


def test_temporal_referenced_columns() raises:
    var cols = col("ts").year().referenced_columns()
    assert_equal(len(cols), 1)
    assert_equal(cols[0], "ts")
