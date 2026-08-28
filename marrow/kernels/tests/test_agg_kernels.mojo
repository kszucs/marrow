"""`AggKernel` — the aggregates that consume columns, at both slot counts.

Every case here exists because of one hazard: the one-slot assignment carries
**no ids**, and every per-group body in this package is a
`for i in range(len(groups.ids))` loop. Handed `Groups.single(n)` such a loop
does not execute and answers `[0]` or `[null]` — a wrong answer, not a crash.
So each `AggregateFn` is exercised at one slot *and* at many, and the one-slot
expectation is always the whole-input answer rather than the identity.
"""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray, Int32Array
from ...builders import (
    StringBuilder,
    TimestampBuilder,
    array,
)
from ...dtypes import (
    DynType,
    Int64Type,
    TimestampType,
    float64,
    int32,
    int64,
    microsecond,
    timestamp,
)
from ..aggregate import (
    Dispersion,
    DistinctCount,
    MaxKernel,
    MaxOp,
    MinKernel,
    MinOp,
    Fold,
    StringExtremum,
)
from ..core import Groups


def _ids(values: List[Optional[Int]]) raises -> Int32Array:
    return array(values, int32).copy()


def _one(var column: DynArray) -> List[DynArray]:
    """A `AggregateFn` takes a *list* of inputs; every fold today takes one."""
    var out = List[DynArray](capacity=1)
    out.append(column^)
    return out^


def _strings(var values: List[Optional[String]]) raises -> DynArray:
    var b = StringBuilder(len(values))
    for ref v in values:
        if v:
            b.append(v.value())
        else:
            b.append_null()
    return b.finish().to_dyn()


def _timestamps(var values: List[Int]) raises -> DynArray:
    var b = TimestampBuilder(timestamp(microsecond, "UTC"), len(values))
    for v in values:
        b.append(Scalar[int64.native](v))
    return b.finish().to_dyn()


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def test_agg_groups_single_holds_no_ids() raises:
    """The convention every fold branches on: empty ids, one slot.

    Materialising `n` zeros to say "everything is group 0" is exactly the cost
    `ScalarGrouping` exists to avoid, so `single` must not do it.
    """
    var g = Groups.single(1000)
    assert_true(g.is_single())
    assert_equal(len(g.ids), 0)
    assert_equal(g.num_groups, 1)


def test_agg_groups_assignment_is_not_single() raises:
    var g = Groups(_ids([0, 0, 1]), 2)
    assert_true(not g.is_single())


# ---------------------------------------------------------------------------
# count_distinct
# ---------------------------------------------------------------------------
def test_agg_count_distinct_one_slot_numeric() raises:
    """The path that silently answers `[0]` if the `is_single` branch is
    missing — `count_distinct_grouped` loops over ids there are none of."""
    var out = DistinctCount[True].grouped(
        Groups.single(5), _one(array([10, 20, 10, 30, 20], int64).to_dyn())
    )
    assert_true(out.as_int64() == array([3], int64))


def test_agg_count_distinct_one_slot_string() raises:
    var out = DistinctCount[True].grouped(
        Groups.single(5),
        _one(_strings(["a", "b", "a", "c", "b"])),
    )
    assert_true(out.as_int64() == array([3], int64))


def test_agg_count_distinct_grouped_string() raises:
    """Group 0 sees a, b, a; group 1 sees c, c."""
    var out = DistinctCount[True].grouped(
        Groups(_ids([0, 0, 0, 1, 1]), 2),
        _one(_strings(["a", "b", "a", "c", "c"])),
    )
    assert_true(out.as_int64() == array([2, 1], int64))


def test_agg_count_distinct_grouped_numeric() raises:
    var out = DistinctCount[True].grouped(
        Groups(_ids([0, 1, 0, 1, 0]), 2),
        _one(array([7, 5, 7, 6, 8], int64).to_dyn()),
    )
    assert_true(out.as_int64() == array([2, 2], int64))


def test_agg_count_distinct_excludes_nulls_at_one_slot() raises:
    """SQL `COUNT(DISTINCT x)` / PyArrow `only_valid`: a null is not a value."""
    var out = DistinctCount[True].grouped(
        Groups.single(4), _one(_strings(["a", None, "a", "b"]))
    )
    assert_true(out.as_int64() == array([2], int64))


def test_agg_count_distinct_excludes_nulls_when_grouped() raises:
    var out = DistinctCount[True].grouped(
        Groups(_ids([0, 0, 1, 1]), 2),
        _one(_strings(["a", None, None, "b"])),
    )
    assert_true(out.as_int64() == array([1, 1], int64))


def test_agg_approx_count_distinct_one_slot() raises:
    """A HyperLogLog is exact at these cardinalities — linear counting takes
    over well below the register count."""
    var out = DistinctCount[False].grouped(
        Groups.single(5), _one(array([10, 20, 10, 30, 20], int64).to_dyn())
    )
    assert_true(out.as_int64() == array([3], int64))


def test_agg_approx_count_distinct_grouped() raises:
    var out = DistinctCount[False].grouped(
        Groups(_ids([0, 0, 0, 1, 1]), 2),
        _one(_strings(["a", "b", "a", "c", "c"])),
    )
    assert_true(out.as_int64() == array([2, 1], int64))


# ---------------------------------------------------------------------------
# string min / max
# ---------------------------------------------------------------------------
def test_agg_string_min_max_one_slot() raises:
    """`StringMinMax` has no `whole` of its own — this branch is the only
    thing standing between the whole-input answer and an empty loop."""
    var values = _strings(["b", "a", "c"])
    var lo = StringExtremum[MinOp].grouped(
        Groups.single(3), _one(values.copy())
    )
    var hi = StringExtremum[MaxOp].grouped(Groups.single(3), _one(values^))
    assert_true(lo.as_string() == array(["a"]))
    assert_true(hi.as_string() == array(["c"]))


def test_agg_string_min_max_grouped() raises:
    var values = _strings(["b", "d", "a", "c"])
    var lo = StringExtremum[MinOp].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(values.copy())
    )
    var hi = StringExtremum[MaxOp].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(values^)
    )
    assert_true(lo.as_string() == array(["b", "a"]))
    assert_true(hi.as_string() == array(["d", "c"]))


def test_agg_string_min_skips_nulls_at_one_slot() raises:
    var out = StringExtremum[MinOp].grouped(
        Groups.single(3), _one(_strings([None, "b", "a"]))
    )
    assert_true(out.as_string() == array(["a"]))


def test_agg_string_min_of_all_nulls_is_null() raises:
    var out = StringExtremum[MinOp].grouped(
        Groups.single(2), _one(_strings([None, None]))
    )
    assert_equal(len(out), 1)
    assert_true(out.is_null(0))


# ---------------------------------------------------------------------------
# fold_column — a fold algebra over a fixed-width column
# ---------------------------------------------------------------------------
def test_agg_temporal_min_max_keeps_unit_and_timezone() raises:
    """The capability the materialising path exists to add. `MinMax.acc_dtype`
    returns the *input* dtype, so a timestamp's unit and timezone must survive
    — and a dtype that disagreed with the schema would be a `Variant`
    misaccess at emit rather than a raise."""
    var values = _timestamps([30, 10, 20])
    var lo = Fold[MinKernel, TimestampType].grouped(
        Groups.single(3), _one(values.copy())
    )
    var hi = Fold[MaxKernel, TimestampType].grouped(
        Groups.single(3), _one(values^)
    )
    assert_true(lo.dtype() == DynType(timestamp(microsecond, "UTC")))
    assert_true(hi.dtype() == DynType(timestamp(microsecond, "UTC")))
    assert_equal(Int(lo.as_timestamp()[0].value()), 10)
    assert_equal(Int(hi.as_timestamp()[0].value()), 30)


def test_agg_temporal_min_grouped() raises:
    var out = Fold[MinKernel, TimestampType].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(_timestamps([30, 10, 50, 40]))
    )
    assert_true(out.dtype() == DynType(timestamp(microsecond, "UTC")))
    assert_equal(len(out), 2)
    assert_equal(Int(out.as_timestamp()[0].value()), 10)
    assert_equal(Int(out.as_timestamp()[1].value()), 40)


def test_agg_numeric_min_one_slot_and_grouped() raises:
    var values = array([3, 1, 4, 1], int64).to_dyn()
    var whole = Fold[MinKernel, Int64Type].grouped(
        Groups.single(4), _one(values.copy())
    )
    var grouped = Fold[MinKernel, Int64Type].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(values^)
    )
    assert_true(whole.as_int64() == array([1], int64))
    assert_true(grouped.as_int64() == array([1, 1], int64))


def test_agg_rejects_a_dtype_it_has_no_arm_for() raises:
    var raised = False
    try:
        _ = Fold[MinKernel, Int64Type].grouped(
            Groups.single(2), _one(_strings(["a", "b"]))
        )
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# dtype must agree with what grouped produces
# ---------------------------------------------------------------------------
def test_agg_out_dtype_agrees_with_the_column_produced() raises:
    """The pairing the compiler used to enforce and no longer does.

    `Aggregation.dtype` had to agree with `grouped`'s *return type* or the
    build failed. `AggKernel.grouped` returns `DynArray`, so a
    disagreement is a `Variant` misaccess at emit rather than a raise — one
    case per (aggregate, dtype-family) pair is what replaces the check.
    """
    var strings = _strings(["b", "a", "c"])
    var stamps = _timestamps([30, 10, 20])
    var numbers = array([3, 1, 4], int64).to_dyn()

    var string_dtypes = List[DynType](capacity=1)
    string_dtypes.append(strings.dtype())
    var stamp_dtypes = List[DynType](capacity=1)
    stamp_dtypes.append(stamps.dtype())
    var number_dtypes = List[DynType](capacity=1)
    number_dtypes.append(numbers.dtype())

    assert_true(
        DistinctCount[True].dtype(string_dtypes)
        == DistinctCount[True]
        .grouped(Groups.single(3), _one(strings.copy()))
        .dtype()
    )
    assert_true(
        StringExtremum[MinOp].dtype(string_dtypes)
        == StringExtremum[MinOp]
        .grouped(Groups.single(3), _one(strings^))
        .dtype()
    )
    assert_true(
        Fold[MaxKernel, TimestampType].dtype(stamp_dtypes)
        == Fold[MaxKernel, TimestampType]
        .grouped(Groups.single(3), _one(stamps^))
        .dtype()
    )
    assert_true(
        Fold[MinKernel, Int64Type].dtype(number_dtypes)
        == Fold[MinKernel, Int64Type]
        .grouped(Groups.single(3), _one(numbers^))
        .dtype()
    )


def test_agg_erased_face_answers_the_same() raises:
    """`Agg.grouped` is what the runtime lane holds — the same static
    method behind a thin pointer, so it must answer identically."""
    var fold = DistinctCount[True].grouped
    var erased = fold(Groups.single(3), _one(_strings(["a", "b", "a"])))
    var direct = DistinctCount[True].grouped(
        Groups.single(3), _one(_strings(["a", "b", "a"]))
    )
    assert_true(erased.as_int64() == direct.as_int64())


def test_agg_over_no_input_only_the_counts_answer() raises:
    """An input that produced no column at all: a cardinality is still 0, an
    extremum has no dtype to be null of and declines."""
    var counted = DistinctCount[True].empty()
    assert_true(Bool(counted))
    assert_true(counted.value().as_int64() == array([0], int64))
    assert_true(not StringExtremum[MinOp].empty())
    assert_true(not Fold[MaxKernel, Int64Type].empty())


# ---------------------------------------------------------------------------
# Dispersion — variance / stddev, the composite-accumulator aggregate
#
# Every expectation here was read off PyArrow (`pc.variance` / `pc.stddev`),
# which is where the `n - ddof <= 0 -> null` rule and the `ddof=0` default come
# from. Comparisons are tolerance-based: Welford's recurrence and Arrow's
# implementation agree to floating point, not bit-for-bit.
# ---------------------------------------------------------------------------
def _close(got: Float64, want: Float64) -> Bool:
    return abs(got - want) < 1e-9


def test_agg_variance_one_slot_population() raises:
    """`pc.variance([1,2,3,4])` is 1.25 at the default ddof=0."""
    var out = Dispersion[0, False].grouped(
        Groups.single(4), _one(array([1, 2, 3, 4], int64).to_dyn())
    )
    assert_equal(out.dtype(), DynType(float64))
    assert_equal(len(out), 1)
    assert_true(_close(out.as_float64()[0].value(), 1.25))


def test_agg_variance_one_slot_sample() raises:
    """`pc.variance([1,2,3,4], ddof=1)` is 5/3 — a different *type*, not a
    different argument: `Dispersion[1, False]` is its own instantiation."""
    var out = Dispersion[1, False].grouped(
        Groups.single(4), _one(array([1, 2, 3, 4], int64).to_dyn())
    )
    assert_true(_close(out.as_float64()[0].value(), 5.0 / 3.0))


def test_agg_stddev_is_the_root_of_the_variance() raises:
    var out = Dispersion[0, True].grouped(
        Groups.single(4), _one(array([1, 2, 3, 4], int64).to_dyn())
    )
    assert_true(_close(out.as_float64()[0].value(), 1.118033988749895))


def test_agg_variance_skips_nulls() raises:
    """`pc.variance([1, None, 3])` is 1.0 — the null is not a zero, and it is
    not counted in `n` either."""
    var values: List[Optional[Int]] = [1, None, 3]
    var out = Dispersion[0, False].grouped(
        Groups.single(3), _one(array(values, int64).to_dyn())
    )
    assert_true(_close(out.as_float64()[0].value(), 1.0))


def test_agg_variance_of_one_value_is_zero_but_null_when_sampled() raises:
    """The `n - ddof <= 0` rule, and the case that makes `ddof` visible:
    PyArrow answers 0.0 at ddof=0 and None at ddof=1."""
    var col = array([5], int64).to_dyn()
    var pop = Dispersion[0, False].grouped(Groups.single(1), _one(col.copy()))
    assert_true(_close(pop.as_float64()[0].value(), 0.0))

    var sample = Dispersion[1, False].grouped(Groups.single(1), _one(col^))
    assert_true(sample.is_null(0))


def test_agg_variance_of_an_all_null_slot_is_null() raises:
    var values: List[Optional[Int]] = [None, None]
    var out = Dispersion[0, False].grouped(
        Groups.single(2), _one(array(values, int64).to_dyn())
    )
    assert_true(out.is_null(0))


def test_agg_variance_grouped() raises:
    """The hazard this file exists for: a per-group body over `Groups.single`
    would answer `[null]`. Group 0 is [1,3] and group 1 is [10,20,30]."""
    var ids: List[Optional[Int]] = [0, 1, 0, 1, 1]
    var col = array([1, 10, 3, 20, 30], int64).to_dyn()
    var out = Dispersion[0, False].grouped(Groups(_ids(ids), 2), _one(col^))
    assert_equal(len(out), 2)
    assert_true(_close(out.as_float64()[0].value(), 1.0))
    var third = out.as_float64()[1].value()
    assert_true(_close(third, 200.0 / 3.0))


def test_agg_variance_grouped_slot_with_one_row_is_null_when_sampled() raises:
    var ids: List[Optional[Int]] = [0, 1, 1]
    var col = array([7, 2, 4], int64).to_dyn()
    var out = Dispersion[1, False].grouped(Groups(_ids(ids), 2), _one(col^))
    assert_true(out.is_null(0), "one row, ddof=1 -> no answer")
    assert_true(_close(out.as_float64()[1].value(), 2.0))


def test_agg_variance_over_no_input_is_null() raises:
    """Unlike an extremum, this can answer without a schema: the output dtype
    is float64 whatever was aggregated."""
    var answer = Dispersion[0, False].empty()
    assert_true(Bool(answer))
    assert_true(answer.value().is_null(0))


def test_agg_variance_is_numerically_stable() raises:
    """The reason for Welford rather than `E[x^2] - E[x]^2`.

    These four values differ by 1 around 1e9, so the naive form subtracts two
    numbers that agree to ~19 significant digits and returns garbage — often a
    small *negative* variance. The true population variance is 1.25.
    """
    var big = 1_000_000_000
    var col = array([big + 1, big + 2, big + 3, big + 4], int64).to_dyn()
    var out = Dispersion[0, False].grouped(Groups.single(4), _one(col^))
    var got = out.as_float64()[0].value()
    assert_true(got >= 0.0, "a variance is never negative")
    assert_true(_close(got, 1.25))
