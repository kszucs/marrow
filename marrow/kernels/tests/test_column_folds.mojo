"""`ColumnAggregation` — the aggregates that consume columns, at both slot counts.

Every case here exists because of one hazard: the one-slot assignment carries
**no ids**, and every per-group body in this package is a
`for i in range(len(groups.ids))` loop. Handed `Groups.single(n)` such a loop
does not execute and answers `[0]` or `[null]` — a wrong answer, not a crash.
So each `ColumnFold` is exercised at one slot *and* at many, and the one-slot
expectation is always the whole-input answer rather than the identity.
"""

from std.testing import assert_equal, assert_true

from ...arrays import DynArray, Int32Array
from ...builders import (
    StringBuilder,
    TimestampBuilder,
    array,
)
from ...dtypes import DynType, int32, int64, microsecond, timestamp
from ..aggregate import (
    DistinctCount,
    MaxKernel,
    MaxOp,
    MinKernel,
    MinOp,
    PrimitiveFold,
    StringExtremum,
    column_fold,
)
from ..core import Groups


def _ids(values: List[Optional[Int]]) raises -> Int32Array:
    return array(values, int32).copy()


def _one(var column: DynArray) -> List[DynArray]:
    """A `ColumnFold` takes a *list* of inputs; every fold today takes one."""
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
def test_column_fold_groups_single_holds_no_ids() raises:
    """The convention every fold branches on: empty ids, one slot.

    Materialising `n` zeros to say "everything is group 0" is exactly the cost
    `ScalarGrouping` exists to avoid, so `single` must not do it.
    """
    var g = Groups.single(1000)
    assert_true(g.is_single())
    assert_equal(len(g.ids), 0)
    assert_equal(g.num_groups, 1)


def test_column_fold_groups_assignment_is_not_single() raises:
    var g = Groups(_ids([0, 0, 1]), 2)
    assert_true(not g.is_single())


# ---------------------------------------------------------------------------
# count_distinct
# ---------------------------------------------------------------------------
def test_column_fold_count_distinct_one_slot_numeric() raises:
    """The path that silently answers `[0]` if the `is_single` branch is
    missing — `count_distinct_grouped` loops over ids there are none of."""
    var out = DistinctCount[True].grouped(
        Groups.single(5), _one(array([10, 20, 10, 30, 20], int64).to_dyn())
    )
    assert_true(out.as_int64() == array([3], int64))


def test_column_fold_count_distinct_one_slot_string() raises:
    var out = DistinctCount[True].grouped(
        Groups.single(5),
        _one(_strings(["a", "b", "a", "c", "b"])),
    )
    assert_true(out.as_int64() == array([3], int64))


def test_column_fold_count_distinct_grouped_string() raises:
    """Group 0 sees a, b, a; group 1 sees c, c."""
    var out = DistinctCount[True].grouped(
        Groups(_ids([0, 0, 0, 1, 1]), 2),
        _one(_strings(["a", "b", "a", "c", "c"])),
    )
    assert_true(out.as_int64() == array([2, 1], int64))


def test_column_fold_count_distinct_grouped_numeric() raises:
    var out = DistinctCount[True].grouped(
        Groups(_ids([0, 1, 0, 1, 0]), 2),
        _one(array([7, 5, 7, 6, 8], int64).to_dyn()),
    )
    assert_true(out.as_int64() == array([2, 2], int64))


def test_column_fold_count_distinct_excludes_nulls_at_one_slot() raises:
    """SQL `COUNT(DISTINCT x)` / PyArrow `only_valid`: a null is not a value."""
    var out = DistinctCount[True].grouped(
        Groups.single(4), _one(_strings(["a", None, "a", "b"]))
    )
    assert_true(out.as_int64() == array([2], int64))


def test_column_fold_count_distinct_excludes_nulls_when_grouped() raises:
    var out = DistinctCount[True].grouped(
        Groups(_ids([0, 0, 1, 1]), 2),
        _one(_strings(["a", None, None, "b"])),
    )
    assert_true(out.as_int64() == array([1, 1], int64))


def test_column_fold_approx_count_distinct_one_slot() raises:
    """A HyperLogLog is exact at these cardinalities — linear counting takes
    over well below the register count."""
    var out = DistinctCount[False].grouped(
        Groups.single(5), _one(array([10, 20, 10, 30, 20], int64).to_dyn())
    )
    assert_true(out.as_int64() == array([3], int64))


def test_column_fold_approx_count_distinct_grouped() raises:
    var out = DistinctCount[False].grouped(
        Groups(_ids([0, 0, 0, 1, 1]), 2),
        _one(_strings(["a", "b", "a", "c", "c"])),
    )
    assert_true(out.as_int64() == array([2, 1], int64))


# ---------------------------------------------------------------------------
# string min / max
# ---------------------------------------------------------------------------
def test_column_fold_string_min_max_one_slot() raises:
    """`StringMinMax` has no `whole` of its own — this branch is the only
    thing standing between the whole-input answer and an empty loop."""
    var values = _strings(["b", "a", "c"])
    var lo = StringExtremum[MinOp].grouped(
        Groups.single(3), _one(values.copy())
    )
    var hi = StringExtremum[MaxOp].grouped(Groups.single(3), _one(values^))
    assert_true(lo.as_string() == array(["a"]))
    assert_true(hi.as_string() == array(["c"]))


def test_column_fold_string_min_max_grouped() raises:
    var values = _strings(["b", "d", "a", "c"])
    var lo = StringExtremum[MinOp].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(values.copy())
    )
    var hi = StringExtremum[MaxOp].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(values^)
    )
    assert_true(lo.as_string() == array(["b", "a"]))
    assert_true(hi.as_string() == array(["d", "c"]))


def test_column_fold_string_min_skips_nulls_at_one_slot() raises:
    var out = StringExtremum[MinOp].grouped(
        Groups.single(3), _one(_strings([None, "b", "a"]))
    )
    assert_true(out.as_string() == array(["a"]))


def test_column_fold_string_min_of_all_nulls_is_null() raises:
    var out = StringExtremum[MinOp].grouped(
        Groups.single(2), _one(_strings([None, None]))
    )
    assert_equal(len(out), 1)
    assert_true(out.is_null(0))


# ---------------------------------------------------------------------------
# fold_column — a fold algebra over a fixed-width column
# ---------------------------------------------------------------------------
def test_column_fold_temporal_min_max_keeps_unit_and_timezone() raises:
    """The capability the materialising path exists to add. `MinMax.acc_dtype`
    returns the *input* dtype, so a timestamp's unit and timezone must survive
    — and a dtype that disagreed with the schema would be a `Variant`
    misaccess at emit rather than a raise."""
    var values = _timestamps([30, 10, 20])
    var lo = PrimitiveFold[MinKernel].grouped(
        Groups.single(3), _one(values.copy())
    )
    var hi = PrimitiveFold[MaxKernel].grouped(Groups.single(3), _one(values^))
    assert_true(lo.dtype() == DynType(timestamp(microsecond, "UTC")))
    assert_true(hi.dtype() == DynType(timestamp(microsecond, "UTC")))
    assert_equal(Int(lo.as_timestamp()[0].value()), 10)
    assert_equal(Int(hi.as_timestamp()[0].value()), 30)


def test_column_fold_temporal_min_grouped() raises:
    var out = PrimitiveFold[MinKernel].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(_timestamps([30, 10, 50, 40]))
    )
    assert_true(out.dtype() == DynType(timestamp(microsecond, "UTC")))
    assert_equal(len(out), 2)
    assert_equal(Int(out.as_timestamp()[0].value()), 10)
    assert_equal(Int(out.as_timestamp()[1].value()), 40)


def test_column_fold_numeric_min_one_slot_and_grouped() raises:
    var values = array([3, 1, 4, 1], int64).to_dyn()
    var whole = PrimitiveFold[MinKernel].grouped(
        Groups.single(4), _one(values.copy())
    )
    var grouped = PrimitiveFold[MinKernel].grouped(
        Groups(_ids([0, 0, 1, 1]), 2), _one(values^)
    )
    assert_true(whole.as_int64() == array([1], int64))
    assert_true(grouped.as_int64() == array([1, 1], int64))


def test_column_fold_rejects_a_dtype_it_has_no_arm_for() raises:
    var raised = False
    try:
        _ = PrimitiveFold[MinKernel].grouped(
            Groups.single(2), _one(_strings(["a", "b"]))
        )
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# out_dtype must agree with what grouped produces
# ---------------------------------------------------------------------------
def test_column_fold_out_dtype_agrees_with_the_column_produced() raises:
    """The pairing the compiler used to enforce and no longer does.

    `Aggregation.out_dtype` had to agree with `grouped`'s *return type* or the
    build failed. `ColumnAggregation.grouped` returns `DynArray`, so a
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
        DistinctCount[True].out_dtype(string_dtypes)
        == DistinctCount[True]
        .grouped(Groups.single(3), _one(strings.copy()))
        .dtype()
    )
    assert_true(
        StringExtremum[MinOp].out_dtype(string_dtypes)
        == StringExtremum[MinOp]
        .grouped(Groups.single(3), _one(strings^))
        .dtype()
    )
    assert_true(
        PrimitiveFold[MaxKernel].out_dtype(stamp_dtypes)
        == PrimitiveFold[MaxKernel]
        .grouped(Groups.single(3), _one(stamps^))
        .dtype()
    )
    assert_true(
        PrimitiveFold[MinKernel].out_dtype(number_dtypes)
        == PrimitiveFold[MinKernel]
        .grouped(Groups.single(3), _one(numbers^))
        .dtype()
    )


def test_column_fold_erased_face_answers_the_same() raises:
    """`column_fold[Agg]()` is what the runtime lane holds — the same static
    method behind a thin pointer, so it must answer identically."""
    var fold = column_fold[DistinctCount[True]]()
    var erased = fold(Groups.single(3), _one(_strings(["a", "b", "a"])))
    var direct = DistinctCount[True].grouped(
        Groups.single(3), _one(_strings(["a", "b", "a"]))
    )
    assert_true(erased.as_int64() == direct.as_int64())


def test_column_fold_over_no_input_only_the_counts_answer() raises:
    """An input that produced no column at all: a cardinality is still 0, an
    extremum has no dtype to be null of and declines."""
    var counted = DistinctCount[True].over_no_input()
    assert_true(Bool(counted))
    assert_true(counted.value().as_int64() == array([0], int64))
    assert_true(not StringExtremum[MinOp].over_no_input())
    assert_true(not PrimitiveFold[MaxKernel].over_no_input())
