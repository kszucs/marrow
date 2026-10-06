# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from std.math import nan
from std.testing import assert_equal, assert_true, assert_false

from ...arrays import DynArray, Int32Array, PrimitiveArray, StringArray
from ...scalars import BoolScalar, DynScalar
from ...builders import (
    array,
    nulls,
    Float64Builder,
    PrimitiveBuilder,
    Int32Builder,
    StringBuilder,
    Date32Builder,
    TimestampBuilder,
)
from ...dtypes import (
    StringType,
    int32,
    int64,
    float64,
    string,
    date32,
    timestamp,
    second,
    Date32Type,
    Float64Type,
    Int32Type,
    Int64Type,
    TimestampType,
)
from ...kernels.groupby import Groups
from ...kernels.window import WindowExtents, WindowFrame
from ...kernels.aggregate import (
    AggKernel,
    Fold,
    MaxFold,
    MaxOp,
    MeanFold,
    MinFold,
    MinOp,
    LexicalExtremum,
    SumFold,
    CountFold,
    Windowable,
    ValidCount,
    AnyKernel,
    AllKernel,
)


def whole[A: AggKernel](value: DynArray) raises -> DynScalar:
    """The whole-table aggregate of an erased column, as one scalar.

    `Groups.single` is the no-`GROUP BY` assignment: one slot, no ids. Every
    `AggKernel` branches on it first, so this reaches each one's whole-column
    fast path rather than a scatter loop over an id array that does not exist.
    """
    return A.grouped(
        Groups.single(len(value)), A.InArray(value.to_data())
    ).to_dyn()[0]


def test_sumtyped() raises:
    var a: DynArray = array([1, 2, 3, 4, 5], int64)
    var result = whole[Fold[SumFold, Int64Type]](a)
    assert_equal(result.as_int64().value(), 15)


def test_sumwith_nulls() raises:
    """Sum skips null values."""
    var a = Int32Builder(3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var col: DynArray = a.finish()
    var result = whole[Fold[SumFold, Int32Type]](col)
    assert_equal(result.as_int64().value(), 30)


def test_sumall_nulls() raises:
    var a: DynArray = nulls(5, int64)
    var result = whole[Fold[SumFold, Int64Type]](a)
    assert_equal(result.as_int64().value(), 0)


def test_sumempty() raises:
    var a: DynArray = array(int32)
    var result = whole[Fold[SumFold, Int32Type]](a)
    assert_equal(result.as_int64().value(), 0)


def test_sumuntyped() raises:
    var a: DynArray = array([1, 2, 3], int64)
    var result = whole[Fold[SumFold, Int64Type]](a)
    assert_equal(result.as_int64().value(), 6)


def test_mean_int() raises:
    """Mean of integers is a float64 scalar."""
    var a: DynArray = array([1, 2, 3, 4], int32)
    var result = whole[Fold[MeanFold, Int32Type]](a)
    assert_true(result.type() == float64)
    assert_equal(result.as_float64().value(), 2.5)


def test_mean_float() raises:
    var a: DynArray = array([1.0, 2.0, 6.0], float64)
    var result = whole[Fold[MeanFold, Float64Type]](a)
    assert_equal(result.as_float64().value(), 3.0)


def test_mean_skips_nulls() raises:
    """Nulls are excluded from both the sum and the divisor."""
    var a = Int32Builder(4)
    a.append(10)
    a.append(20)
    a.append_null()
    a.append(30)
    var arr: DynArray = a.finish()
    var result = whole[Fold[MeanFold, Int32Type]](arr)
    assert_equal(result.as_float64().value(), 20.0)  # (10+20+30)/3


def test_mean_all_null_is_null() raises:
    var a: DynArray = nulls(4, int64)
    var result = whole[Fold[MeanFold, Int64Type]](a)
    assert_false(result.is_valid())


def test_mean_empty_is_null() raises:
    var a: DynArray = array(int32)
    var result = whole[Fold[MeanFold, Int32Type]](a)
    assert_false(result.is_valid())


# ---------------------------------------------------------------------------
# whole-table min / max over strings (lexicographic / bytewise)
# ---------------------------------------------------------------------------


def _strings(var items: List[String]) raises -> DynArray:
    var b = StringBuilder(len(items))
    for it in items:
        b.append(it)
    return b.finish()


def test_min_string() raises:
    var a = _strings(["banana", "apple", "cherry"])
    var r = whole[LexicalExtremum[MinOp, StringArray]](a)
    assert_true(r.type() == string)
    assert_equal(r.as_string().to_string(), "apple")


def test_max_string() raises:
    var a = _strings(["banana", "apple", "cherry"])
    var r = whole[LexicalExtremum[MaxOp, StringArray]](a)
    assert_true(r.type() == string)
    assert_equal(r.as_string().to_string(), "cherry")


def test_min_string_skips_nulls() raises:
    var b = StringBuilder(4)
    b.append("m")
    b.append_null()
    b.append("a")
    b.append_null()
    var a: DynArray = b.finish()
    assert_equal(
        whole[LexicalExtremum[MinOp, StringArray]](a).as_string().to_string(),
        "a",
    )
    assert_equal(
        whole[LexicalExtremum[MaxOp, StringArray]](a).as_string().to_string(),
        "m",
    )


def test_min_string_all_null_is_null() raises:
    var b = StringBuilder(2)
    b.append_null()
    b.append_null()
    var a: DynArray = b.finish()
    assert_false(whole[LexicalExtremum[MinOp, StringArray]](a).is_valid())
    assert_false(whole[LexicalExtremum[MaxOp, StringArray]](a).is_valid())


def test_min_string_empty_is_null() raises:
    var b = StringBuilder(0)
    var a: DynArray = b.finish()
    assert_false(whole[LexicalExtremum[MinOp, StringArray]](a).is_valid())


# ---------------------------------------------------------------------------
# whole-table min / max over temporal (integer backing, dtype preserved)
# ---------------------------------------------------------------------------


def _date32(var days: List[Int]) raises -> DynArray:
    var b = Date32Builder(date32(), len(days))
    for d in days:
        b.append(Scalar[int32.native](d))
    return b.finish()


def test_min_max_date32() raises:
    var a = _date32([19000, 18500, 19000, 18800])
    var mn = whole[Fold[MinFold, Date32Type]](a)
    var mx = whole[Fold[MaxFold, Date32Type]](a)
    assert_true(mn.type() == date32().to_dyn())  # dtype preserved
    assert_true(mx.type() == date32().to_dyn())
    assert_equal(mn.as_date32().value(), 18500)
    assert_equal(mx.as_date32().value(), 19000)


def test_min_max_date32_skips_nulls() raises:
    var b = Date32Builder(date32(), 3)
    b.append(Scalar[int32.native](19000))
    b.append_null()
    b.append(Scalar[int32.native](18500))
    var a: DynArray = b.finish()
    assert_equal(whole[Fold[MinFold, Date32Type]](a).as_date32().value(), 18500)
    assert_equal(whole[Fold[MaxFold, Date32Type]](a).as_date32().value(), 19000)


def test_min_max_date32_all_null_is_null() raises:
    var b = Date32Builder(date32(), 2)
    b.append_null()
    b.append_null()
    var a: DynArray = b.finish()
    assert_false(whole[Fold[MinFold, Date32Type]](a).is_valid())
    assert_false(whole[Fold[MaxFold, Date32Type]](a).is_valid())


def test_min_max_timestamp_preserves_unit_tz() raises:
    var b = TimestampBuilder(timestamp(second, "UTC"), 3)
    b.append(Scalar[int64.native](3000))
    b.append(Scalar[int64.native](1000))
    b.append(Scalar[int64.native](2000))
    var a: DynArray = b.finish()
    var mn = whole[Fold[MinFold, TimestampType]](a)
    assert_true(mn.type() == timestamp(second, "UTC").to_dyn())
    assert_equal(mn.as_timestamp().value(), 1000)
    assert_equal(
        whole[Fold[MaxFold, TimestampType]](a).as_timestamp().value(), 3000
    )


# -- Windowable.over: one answer per frame, against the aggregate over each --


def _check_over[A: Windowable](value: DynArray) raises:
    """Every frame shape a window produces — the default `RANGE` frame over
    tied peers, sliding, whole-partition, empty, past the partition and
    two-sided `ROWS` frames — over three partitions, each row checked against
    `A` over that row's slice."""
    var n = len(value)
    var new_partition = List[Bool](length=n, fill=False)
    var new_peer = List[Bool](length=n, fill=False)
    for j in range(n):
        new_partition[j] = j == 0 or j == 7 or j == 15
        new_peer[j] = new_partition[j] or j % 3 == 0
    var extents = WindowExtents(new_partition^, new_peer^)
    var frames = [
        WindowFrame.default(),
        WindowFrame(True, -3, 0),
        WindowFrame(True, -1000, 1000),
        WindowFrame(True, -3, -1),
        WindowFrame(True, 5, 10),
        WindowFrame(True, 0, 0),
        WindowFrame(True, -1, 2),
    ]
    for frame in frames:
        var got = A.over(A.InArray(value.to_data()), extents, frame).to_dyn()
        assert_equal(len(got), n)
        for j in range(n):
            var lo, hi = extents.frame(j, frame)
            var expected = whole[A](value.slice(lo, hi - lo))
            assert_equal(
                String(got[j]), String(expected), String(frame, " row ", j)
            )


def test_over_matches_the_aggregate_over_each_frame() raises:
    var b = Int32Builder(23)
    for i in range(23):
        if i % 5 == 2:
            b.append_null()
        else:
            b.append(Int32((i * 7919) % 41 - 20))
    var col: DynArray = b.finish()
    _check_over[Fold[SumFold, Int32Type]](col)
    _check_over[Fold[MinFold, Int32Type]](col)
    _check_over[Fold[MaxFold, Int32Type]](col)
    _check_over[Fold[MeanFold, Int32Type]](col)
    _check_over[Fold[CountFold, Int32Type]](col)
    _check_over[ValidCount[Int32Array]](col)


def test_over_float_min_max_with_nan() raises:
    var b = Float64Builder(23)
    for i in range(23):
        if i % 7 == 3:
            b.append(nan[DType.float64]())
        elif i % 6 == 1:
            b.append_null()
        else:
            b.append(Float64((i * 31) % 17) - 8.5)
    var col: DynArray = b.finish()
    _check_over[Fold[MinFold, Float64Type]](col)
    _check_over[Fold[MaxFold, Float64Type]](col)
    _check_over[Fold[SumFold, Float64Type]](col)


# ---------------------------------------------------------------------------
# any / all under PyArrow's ScalarAggregateOptions
# ---------------------------------------------------------------------------


def _is(got: BoolScalar, expected: Optional[Bool]) raises:
    if expected:
        assert_true(got.is_valid())
        assert_equal(got.value(), expected.value())
    else:
        assert_false(got.is_valid())


def test_any_all_empty_or_all_null_is_null() raises:
    """The default `min_count=1` answers null without a valid element."""
    var all_null: List[Optional[Bool]] = [None, None]
    for values in [array(List[Optional[Bool]]()), array(all_null)]:
        _is(AnyKernel.aggregate(values), None)
        _is(AllKernel.aggregate(values), None)
        _is(AnyKernel.aggregate(values, min_count=0), False)
        _is(AllKernel.aggregate(values, min_count=0), True)


def test_any_all_skip_nulls() raises:
    var false_null = array([False, None])
    var true_null = array([True, None])
    _is(AnyKernel.aggregate(false_null), False)
    _is(AnyKernel.aggregate(true_null), True)
    _is(AllKernel.aggregate(false_null), False)
    _is(AllKernel.aggregate(true_null), True)


def test_any_all_without_skip_nulls_is_kleene() raises:
    """A null is unknown: the answer is null unless the fold already decided
    it — a True for `any`, a False for `all`."""
    var false_null = array([False, None])
    var true_null = array([True, None])
    _is(AnyKernel.aggregate(false_null, skip_nulls=False), None)
    _is(AnyKernel.aggregate(true_null, skip_nulls=False), True)
    _is(AllKernel.aggregate(false_null, skip_nulls=False), False)
    _is(AllKernel.aggregate(true_null, skip_nulls=False), None)
    _is(AnyKernel.aggregate(array([False, True]), skip_nulls=False), True)


def test_any_all_min_count_counts_valid_elements() raises:
    var values = array([True, None, True])
    _is(AnyKernel.aggregate(values, min_count=2), True)
    _is(AnyKernel.aggregate(values, min_count=3), None)
    _is(AllKernel.aggregate(values, skip_nulls=False, min_count=3), None)
