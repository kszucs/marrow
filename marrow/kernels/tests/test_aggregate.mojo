# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from std.math import iota, isnan, nan
from std.sys.info import simd_width_of
from std.testing import assert_equal, assert_true, assert_false

from ...arrays import (
    DynArray,
    Float64Array,
    Int32Array,
    Int64Array,
    PrimitiveArray,
    StringArray,
)
from ...buffers import Bitmap, Buffer
from ...errors import InvalidError
from ...scalars import BoolScalar, DynScalar
from ...views import BitmapView, BufferView
from ...builders import (
    array,
    nulls,
    Float64Builder,
    PrimitiveBuilder,
    Decimal128Builder,
    Int32Builder,
    Int64Builder,
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
    decimal128,
    timestamp,
    second,
    Date32Type,
    Decimal128Type,
    Float64Type,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    NumericType,
    PrimitiveType,
    TimestampType,
)
from ...kernels.groupby import Groups
from ...kernels.numeric import AddKernel, MinKernel
from ...kernels.window import WindowFrame, WindowSegments
from ...kernels.aggregate import (
    AggKernel,
    Fold,
    FoldKernel,
    MaxFold,
    MeanFold,
    MinFold,
    MinMax,
    LexicalExtremum,
    SumFold,
    Widening,
    MIN,
    SUM,
    CountFold,
    DecimalSumFold,
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
    var r = whole[LexicalExtremum[True, StringArray]](a)
    assert_true(r.type() == string)
    assert_equal(r.as_string().to_string(), "apple")


def test_max_string() raises:
    var a = _strings(["banana", "apple", "cherry"])
    var r = whole[LexicalExtremum[False, StringArray]](a)
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
        whole[LexicalExtremum[True, StringArray]](a).as_string().to_string(),
        "a",
    )
    assert_equal(
        whole[LexicalExtremum[False, StringArray]](a).as_string().to_string(),
        "m",
    )


def test_min_string_all_null_is_null() raises:
    var b = StringBuilder(2)
    b.append_null()
    b.append_null()
    var a: DynArray = b.finish()
    assert_false(whole[LexicalExtremum[True, StringArray]](a).is_valid())
    assert_false(whole[LexicalExtremum[False, StringArray]](a).is_valid())


def test_min_string_empty_is_null() raises:
    var b = StringBuilder(0)
    var a: DynArray = b.finish()
    assert_false(whole[LexicalExtremum[True, StringArray]](a).is_valid())


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


def _check_each_frame[
    A: Windowable
](value: DynArray, segments: WindowSegments, frames: List[WindowFrame]) raises:
    """`A.over` every frame, each row checked against `A` over that row's
    slice."""
    var n = len(value)
    for frame in frames:
        var got = A.over(A.InArray(value.to_data()), segments, frame).to_dyn()
        assert_equal(len(got), n)
        var bounds = segments.frames(frame)
        for j in range(n):
            var lo = Int(bounds[0].values()[j])
            var hi = Int(bounds[1].values()[j])
            var expected = whole[A](value.slice(lo, hi - lo))
            assert_equal(
                String(got[j]), String(expected), String(frame, " row ", j)
            )


def _check_over[A: Windowable](value: DynArray) raises:
    """Every frame shape a window produces — the default `RANGE` frame over
    tied peers, sliding, whole-partition, empty, past the partition and
    two-sided `ROWS` frames — over three partitions, each row checked against
    `A` over that row's slice."""
    var n = len(value)
    var peer_ends = Int64Builder(n + 1)
    for j in range(n):
        if j == 0 or j == 7 or j == 15 or j % 3 == 0:
            peer_ends.append(Int64(j))
    peer_ends.append(Int64(n))
    var segments = WindowSegments(
        array([0, 7, 15, n], int64), peer_ends.finish()
    )
    var frames: List[WindowFrame] = [
        WindowFrame.default(),
        WindowFrame(True, -3, 0),
        WindowFrame(True, -1000, 1000),
        WindowFrame(True, -3, -1),
        WindowFrame(True, 5, 10),
        WindowFrame(True, 0, 0),
        WindowFrame(True, -1, 2),
    ]
    _check_each_frame[A](value, segments, frames)


def _int32_with_nulls(n: Int) raises -> DynArray:
    var b = Int32Builder(n)
    for i in range(n):
        if i % 5 == 2:
            b.append_null()
        else:
            b.append(Int32((i * 7919) % 41 - 20))
    return b.finish()


def _decimals_with_nulls(n: Int) raises -> DynArray:
    var b = Decimal128Builder(decimal128(10, 2), n)
    for i in range(n):
        if i % 5 == 2:
            b.append_null()
        else:
            b.append(Scalar[Decimal128Type.native]((i * 7919) % 41 - 20))
    return b.finish()


def _float64_with_nans(n: Int) raises -> DynArray:
    var b = Float64Builder(n)
    for i in range(n):
        if i % 7 == 3 or (i >= 40 and i < 52):
            b.append(nan[DType.float64]())
        elif i % 6 == 1:
            b.append_null()
        else:
            b.append(Float64((i * 31) % 17) - 8.5)
    return b.finish()


def _two_partitions(n: Int) raises -> WindowSegments:
    """`n` rows cut at row 27, every row its own peer group."""
    var peer_ends = Int64Builder(n + 1)
    for j in range(n + 1):
        peer_ends.append(Int64(j))
    return WindowSegments(array([0, 27, n], int64), peer_ends.finish())


def test_over_matches_the_aggregate_over_each_frame() raises:
    var col = _int32_with_nulls(23)
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


def test_over_decimal_sum_matches_each_frame() raises:
    """A decimal sum folds every frame in its 128-bit accumulator."""
    var col = _decimals_with_nulls(23)
    _check_over[Fold[DecimalSumFold, Decimal128Type]](col)


def test_over_rows_frames_wider_than_a_block() raises:
    """`ROWS` frames wider than eight rows inside longer partitions: a row
    inside its partition folds a block's suffix and the next block's prefix,
    a register at a time."""
    comptime n = 60
    var segments = _two_partitions(n)
    var frames: List[WindowFrame] = [
        WindowFrame(True, -10, 0),
        WindowFrame(True, 0, 12),
        WindowFrame(True, -6, 5),
    ]
    var ints = _int32_with_nulls(n)
    _check_each_frame[Fold[SumFold, Int32Type]](ints, segments, frames)
    var floats = _float64_with_nans(n)
    _check_each_frame[Fold[MinFold, Float64Type]](floats, segments, frames)
    _check_each_frame[Fold[MaxFold, Float64Type]](floats, segments, frames)
    var decimals = _decimals_with_nulls(n)
    _check_each_frame[Fold[DecimalSumFold, Decimal128Type]](
        decimals, segments, frames
    )


def test_over_rows_offsets_past_every_partition() raises:
    """`Int.MIN`, `Int.MAX` and `2^62` reach past any partition, so each is
    the partition's edge."""
    comptime n = 60
    var segments = _two_partitions(n)
    var frames: List[WindowFrame] = [
        WindowFrame(True, Int.MIN, 0),
        WindowFrame(True, -(1 << 62), 1 << 62),
        WindowFrame(True, 0, Int.MAX),
    ]
    var ints = _int32_with_nulls(n)
    _check_each_frame[Fold[SumFold, Int32Type]](ints, segments, frames)
    _check_each_frame[ValidCount[Int32Array]](ints, segments, frames)
    var floats = _float64_with_nans(n)
    _check_each_frame[Fold[MinFold, Float64Type]](floats, segments, frames)


def test_over_a_frame_ending_before_it_starts_is_empty() raises:
    """`ROWS BETWEEN 3 FOLLOWING AND 1 FOLLOWING` holds no row for any row:
    the sum is null and both counts are 0."""
    var values = array([1, 2, 3, 4, 5], int32)
    var whole_column = array([0, 5], int64)
    var segments = WindowSegments(whole_column.copy(), whole_column.copy())
    var frame = WindowFrame(True, 3, 1)
    var none: List[Optional[Int]] = [None, None, None, None, None]
    assert_true(
        Fold[SumFold, Int32Type].over(values, segments, frame)
        == array(none, int64)
    )
    var zeros = array([0, 0, 0, 0, 0], int64)
    assert_true(
        Fold[CountFold, Int32Type].over(values, segments, frame) == zeros
    )
    assert_true(ValidCount[Int32Array].over(values, segments, frame) == zeros)


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


# -- min / max over floats: NaN, zeros, empty --


def _check_zero_extremes(zeros: DynArray) raises:
    """`min` and `max` of `0.0` and `-0.0` are a zero. Which one is not
    specified, so only the value is checked."""
    var lo = whole[Fold[MinFold, Float64Type]](zeros).as_float64().value()
    var hi = whole[Fold[MaxFold, Float64Type]](zeros).as_float64().value()
    assert_equal(lo, 0.0)
    assert_equal(hi, 0.0)


def _by_group[
    A: AggKernel
](value: DynArray, ids: List[Optional[Int]]) raises -> DynArray:
    """`A` over `value`, one answer per group id in `ids`."""
    var groups = Groups(array(ids, int32), 2)
    return A.grouped(groups, A.InArray(value.to_data())).to_dyn()


def test_min_max_float_nan_zeros_and_empty() raises:
    """NaN is skipped unless nothing else is there, the extremes of `0.0` and
    `-0.0` are a zero in either order, and an empty column has no extremum."""
    var q = nan[DType.float64]()
    var neg_zero = -Float64(0.0)
    var mixed: DynArray = array([q, 3.0, -1.0, q], float64)
    var all_nan: DynArray = array([q, q], float64)
    var zeros: DynArray = array([Float64(0.0), neg_zero], float64)
    var zeros_reversed: DynArray = array([neg_zero, Float64(0.0)], float64)
    var empty: DynArray = array(float64)

    assert_equal(
        whole[Fold[MinFold, Float64Type]](mixed).as_float64().value(), -1.0
    )
    assert_equal(
        whole[Fold[MaxFold, Float64Type]](mixed).as_float64().value(), 3.0
    )
    assert_true(
        isnan(whole[Fold[MinFold, Float64Type]](all_nan).as_float64().value())
    )
    assert_true(
        isnan(whole[Fold[MaxFold, Float64Type]](all_nan).as_float64().value())
    )
    _check_zero_extremes(zeros)
    _check_zero_extremes(zeros_reversed)
    assert_false(whole[Fold[MinFold, Float64Type]](empty).is_valid())
    assert_false(whole[Fold[MaxFold, Float64Type]](empty).is_valid())

    # Group 0 holds `NaN, 3.0`, group 1 holds `-1.0, NaN`.
    var ids: List[Optional[Int]] = [0, 0, 1, 1]
    var lo = _by_group[Fold[MinFold, Float64Type]](mixed, ids)
    var hi = _by_group[Fold[MaxFold, Float64Type]](mixed, ids)
    assert_equal(lo[0].as_float64().value(), 3.0)
    assert_equal(lo[1].as_float64().value(), -1.0)
    assert_equal(hi[0].as_float64().value(), 3.0)
    assert_equal(hi[1].as_float64().value(), -1.0)

    # A fold names itself by the label it is given.
    assert_equal(MinMax[MinKernel, MIN].name, "min")
    assert_equal(Widening[AddKernel, SUM].name, "sum")


# ---------------------------------------------------------------------------
# Fold.reduce / Fold.scatter — the lane loops a fold owns
# ---------------------------------------------------------------------------


def _check_slot(
    got: DynScalar, expected: Optional[Int64], label: String
) raises:
    """`got` is `expected`, or NULL when there is none."""
    if expected:
        assert_true(got.is_valid(), label)
        assert_equal(got.as_int64().value(), expected.value(), label)
    else:
        assert_false(got.is_valid(), label)


def _bits(bits: List[Bool]) raises -> Bitmap[mut=False]:
    """A validity bitmap holding `bits`."""
    var bitmap = Bitmap.alloc_zeroed(len(bits))
    for i in range(len(bits)):
        if bits[i]:
            bitmap.set(i)
    return bitmap^.to_immutable()


def _every_third_clear(n: Int) raises -> Bitmap[mut=False]:
    """`n` validity bits with every third one, `i % 3 == 2`, clear."""
    var bits = List[Bool](capacity=n)
    for i in range(n):
        bits.append(i % 3 != 2)
    return _bits(bits)


def _fold_rows[
    K: FoldKernel
](n: Int, valid: Optional[Bitmap[mut=False]]) raises -> DynScalar:
    """`Fold[K, Int64Type]` over rows `1..n` from a lane closure, through the
    masked `reduce` when `valid` is given."""
    comptime Acc = Fold[K, Int64Type].Acc

    @always_inline
    def lanes[W: Int](i: Int) -> SIMD[Acc, W]:
        return (iota[DType.int64, W]() + Int64(i + 1)).cast[Acc]()

    var fold = Fold[K, Int64Type](int64, False)
    if valid:
        fold.reduce(n, lanes, valid.value().view())
    else:
        fold.reduce(n, lanes)
    return fold.finish().to_dyn()[0]


def _reference_rows(n: Int, skip_third: Bool, minimum: Bool) -> Optional[Int64]:
    """The left fold, `min` or `sum`, of rows `1..n`, skipping `i % 3 == 2`
    when asked; `None` when no row is left."""
    var acc: Optional[Int64] = None
    for i in range(n):
        if not (skip_third and i % 3 == 2):
            var x = Int64(i + 1)
            if not acc:
                acc = x
            elif minimum:
                acc = min(acc.value(), x)
            else:
                acc = acc.value() + x
    return acc


def test_fold_lanes_every_tail_length() raises:
    """`Fold.reduce` over a lane closure at every row count around the SIMD
    width — none, part of a register, whole registers and a tail — masked and
    not; a fold with no valid row is NULL."""
    comptime width = simd_width_of[Scalar[DType.int64]]()
    for n in [0, 1, width - 1, width, width + 1, 3 * width + 5]:
        var valid = _every_third_clear(n)
        var label = String("n=", n)
        _check_slot(
            _fold_rows[SumFold](n, None),
            _reference_rows(n, False, False),
            label,
        )
        _check_slot(
            _fold_rows[MinFold](n, None), _reference_rows(n, False, True), label
        )
        _check_slot(
            _fold_rows[SumFold](n, valid.copy()),
            _reference_rows(n, True, False),
            label + " masked",
        )
        _check_slot(
            _fold_rows[MinFold](n, valid.copy()),
            _reference_rows(n, True, True),
            label + " masked",
        )
    var none_valid = Bitmap.alloc_zeroed(3 * width + 5).to_immutable()
    _check_slot(
        _fold_rows[SumFold](3 * width + 5, none_valid.copy()), None, "all clear"
    )
    _check_slot(
        _fold_rows[MinFold](3 * width + 5, none_valid.copy()), None, "all clear"
    )


def test_fold_lanes_sliced_with_nulls() raises:
    """An offset array with nulls through `Fold.update`, whole and grouped:
    the lanes read `values()` and the mask reads `validity()`, both with the
    offset applied. A null's payload is 0, below every valid value, so `min`
    answers 0 if one leaks."""
    var ints = List[Optional[Int]]()
    for i in range(100):
        if i % 3 == 0:
            ints.append(None)
        else:
            ints.append(i)
    var sliced = array(ints, int64).slice(1)
    var total = Int64(0)
    var per_group = List[Int64](length=4, fill=0)
    var least = List[Int64](length=4, fill=Int64.MAX)
    for j in range(99):
        if (j + 1) % 3 != 0:
            total += Int64(j + 1)
            per_group[j % 4] += Int64(j + 1)
            least[j % 4] = min(least[j % 4], Int64(j + 1))

    var single = Groups.single(99)
    var one = Fold[SumFold, Int64Type].grouped(single, sliced)
    assert_equal(one[0].value(), total)
    var lowest = Fold[MinFold, Int64Type].grouped(single, sliced)
    assert_equal(lowest[0].value(), 1)

    var ids = List[Optional[Int]]()
    for j in range(99):
        ids.append(j % 4)
    var by_four = Groups(array(ids, int32), 4)
    var sums = Fold[SumFold, Int64Type].grouped(by_four, sliced)
    var mins = Fold[MinFold, Int64Type].grouped(by_four, sliced)
    for g in range(4):
        assert_equal(sums[g].value(), per_group[g], String("sum ", g))
        assert_equal(mins[g].value(), least[g], String("min ", g))


def _scatter_lanes[
    K: FoldKernel
](groups: Groups, column: Int64Array) raises -> DynArray:
    """`Fold.scatter` over a lane closure reading `column`, nulls skipped."""
    comptime Acc = Fold[K, Int64Type].Acc
    var values = column.values()

    @always_inline
    def lanes[W: Int](i: Int) {var values} -> SIMD[Acc, W]:
        return values.load[W](i).cast[Acc]()

    var fold = Fold[K, Int64Type](int64, True)
    fold.scatter(groups, lanes, column.validity().value())
    return fold.finish().to_dyn()


def test_fold_lanes_scatter_matches_the_scalar_fold() raises:
    """`Fold.scatter` folds row `i` into slot `ids[i]`, skipping nulls: each
    slot's `sum`, `min` and `mean` equal a scalar fold of its rows, and a
    slot no row names is NULL."""
    comptime width = simd_width_of[Scalar[DType.int64]]()
    var n = 3 * width + 5
    var pattern = [0, 2, 1, 2, 0]
    var ids = List[Optional[Int]]()
    var values = List[Optional[Int]]()
    var sums = List[Int64](length=4, fill=0)
    var mins = List[Int64](length=4, fill=Int64.MAX)
    var counts = List[Int](length=4, fill=0)
    for i in range(n):
        var g = pattern[i % 5]
        ids.append(g)
        if i % 4 == 3:
            values.append(None)
        else:
            var x = i * 7 - 20
            values.append(x)
            sums[g] += Int64(x)
            mins[g] = min(mins[g], Int64(x))
            counts[g] += 1
    # Four slots, and no row names slot 3.
    var groups = Groups(array(ids, int32), 4)
    var column = array(values, int64)
    var got_sum = _scatter_lanes[SumFold](groups, column)
    var got_min = _scatter_lanes[MinFold](groups, column)
    var got_mean = _scatter_lanes[MeanFold](groups, column)
    for g in range(3):
        var label = String("slot ", g)
        assert_equal(got_sum[g].as_int64().value(), sums[g], label)
        assert_equal(got_min[g].as_int64().value(), mins[g], label)
        assert_equal(
            got_mean[g].as_float64().value(),
            Float64(sums[g]) / Float64(counts[g]),
            label,
        )
    assert_false(got_sum[3].is_valid())
    assert_false(got_min[3].is_valid())
    assert_false(got_mean[3].is_valid())


def test_fold_lanes_integer_wrap_is_grouping_free() raises:
    """`sum` over int64 values near `Int64.MAX` wraps, and folding several
    registers at once answers exactly the left fold: integer `+` is exact
    under any grouping."""
    comptime width = simd_width_of[Scalar[DType.int64]]()
    var n = 4 * width + 3
    var expected = Int64(0)
    for k in range(n):
        expected += Int64.MAX - Int64(k)

    @always_inline
    def lanes[W: Int](i: Int) -> SIMD[DType.int64, W]:
        return SIMD[DType.int64, W](Int64.MAX) - (
            iota[DType.int64, W]() + Int64(i)
        )

    var fold = Fold[SumFold, Int64Type](int64, False)
    fold.reduce(n, lanes)
    assert_equal(fold.finish()[0].value(), expected)


def _masked_fold[
    K: FoldKernel, V: NumericType
](values: BufferView[V.native, _], valid: BitmapView[_]) raises -> DynScalar:
    """`Fold[K, V]` over a lane closure reading `values`, through the masked
    `reduce`."""
    comptime Acc = Fold[K, V].Acc

    @always_inline
    def lanes[W: Int](i: Int) {var values} -> SIMD[Acc, W]:
        return values.load[W](i).cast[Acc]()

    var fold = Fold[K, V](V(), False)
    fold.reduce(len(values), lanes, valid)
    return fold.finish().to_dyn()[0]


def test_fold_lanes_sum_with_bitmap() raises:
    """The masked `reduce` folds only the rows whose bit is set."""
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    buf.unsafe_set[DType.int32](0, Int32(10))
    buf.unsafe_set[DType.int32](1, Int32(20))  # null
    buf.unsafe_set[DType.int32](2, Int32(30))
    buf.unsafe_set[DType.int32](3, Int32(40))  # null
    var bm = Bitmap.alloc_zeroed(4)
    bm.set(0)
    bm.set(2)
    var got = _masked_fold[SumFold, Int32Type](
        buf.view[DType.int32](0, 4), bm.view()
    )
    assert_equal(got.as_int64().value(), 40)


def test_fold_lanes_all_null_is_null() raises:
    """With every bit clear no row is folded, and the answer is NULL."""
    var buf = Buffer.alloc_zeroed[DType.int32](3)
    for i in range(3):
        buf.unsafe_set[DType.int32](i, Int32(99))
    var bm = Bitmap.alloc_zeroed(3)  # all bits clear = all null
    var got = _masked_fold[SumFold, Int32Type](
        buf.view[DType.int32](0, 3), bm.view()
    )
    assert_false(got.is_valid())


struct _NarrowSum(FoldKernel):
    """`sum` accumulating, wrapping, in its input's own type, so a fold over
    a narrow integer reads as many lanes per step as its registers hold."""

    comptime name = "narrow_sum"
    comptime AccType[V: PrimitiveType] = V

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        return Scalar[T](0)

    @staticmethod
    def acc_dtype[
        V: PrimitiveType
    ](dtype: V) raises InvalidError -> Self.AccType[V]:
        return dtype.copy()

    @always_inline
    @staticmethod
    def combine[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        return a + b

    @always_inline
    @staticmethod
    def finalize[A: DType](acc: Scalar[A], count: Int) -> Scalar[A]:
        return acc


def _check_masked_narrow_sum[V: NumericType]() raises:
    comptime T = V.native
    comptime N = 300
    var buf = Buffer.alloc_zeroed[T](N)
    var bm = Bitmap.alloc_zeroed(N)
    for i in range(N):
        buf.unsafe_set[T](i, Scalar[T](i % 5 + 1))
        if i % 3 != 0:
            bm.set(i)
    for offset in [0, 1, 5, 7, 13]:
        var n = N - offset
        var expected = Scalar[T](0)
        var widened = Int64(0)
        for i in range(offset, N):
            if i % 3 != 0:
                expected += Scalar[T](i % 5 + 1)
                widened += Int64(i % 5 + 1)
        var label = String(T, " offset=", offset)
        var narrow = _masked_fold[_NarrowSum, V](
            buf.view[T](offset, n), bm.view(offset, n)
        )
        assert_equal(narrow.as_primitive[V]().value(), expected, label)
        var wide = _masked_fold[SumFold, V](
            buf.view[T](offset, n), bm.view(offset, n)
        )
        assert_equal(wide.as_int64().value(), widened, label)


def test_fold_lanes_masked_narrow_ints_at_bit_offsets() raises:
    """A masked fold reads many lanes per step from one bitmap load, so the
    lanes must stay inside its window at every bit offset, narrow types
    included, where a step covers the most lanes."""
    _check_masked_narrow_sum[Int8Type]()
    _check_masked_narrow_sum[Int16Type]()
    _check_masked_narrow_sum[Int32Type]()
    _check_masked_narrow_sum[Int64Type]()


def _check_float_zeros(zeros: Float64Array, valid: Bitmap[mut=False]) raises:
    """`min` and `max` of `0.0` and `-0.0` are a zero (which one is not
    specified); the masked-out row holds a value either would answer if it
    leaked."""
    var lo = _masked_fold[MinFold, Float64Type](zeros.values(), valid.view())
    var hi = _masked_fold[MaxFold, Float64Type](zeros.values(), valid.view())
    assert_equal(lo.as_float64().value(), 0.0)
    assert_equal(hi.as_float64().value(), 0.0)


def _float_scatter[
    K: FoldKernel
](
    column: Float64Array, valid: Bitmap[mut=False], groups: Groups
) raises -> DynArray:
    """`Fold[K, Float64Type]` over a lane closure reading `column`, through
    the masked `scatter`."""
    comptime Acc = Fold[K, Float64Type].Acc
    var values = column.values()

    @always_inline
    def lanes[W: Int](i: Int) {var values} -> SIMD[Acc, W]:
        return values.load[W](i).cast[Acc]()

    var fold = Fold[K, Float64Type](float64, True)
    fold.scatter(groups, lanes, valid.view())
    return fold.finish().to_dyn()


def test_fold_lanes_float_min_max_nan() raises:
    """Through the masked lane entries: NaN is skipped unless nothing else is
    there, the extremes of two zeros are a zero, a masked-out row is never
    read, and a fold with no valid row is NULL."""
    var q = nan[DType.float64]()
    var neg_zero = -Float64(0.0)

    # The masked-out rows hold the extremes a leaked row would answer.
    var mixed = array([q, 3.0, -100.0, -1.0, 100.0, q], float64)
    var mixed_valid = _bits([True, True, False, True, False, True])
    var lo = _masked_fold[MinFold, Float64Type](
        mixed.values(), mixed_valid.view()
    )
    var hi = _masked_fold[MaxFold, Float64Type](
        mixed.values(), mixed_valid.view()
    )
    assert_equal(lo.as_float64().value(), -1.0)
    assert_equal(hi.as_float64().value(), 3.0)

    var all_nan = array([q, 7.0, q], float64)
    var all_nan_valid = _bits([True, False, True])
    lo = _masked_fold[MinFold, Float64Type](
        all_nan.values(), all_nan_valid.view()
    )
    hi = _masked_fold[MaxFold, Float64Type](
        all_nan.values(), all_nan_valid.view()
    )
    assert_true(isnan(lo.as_float64().value()))
    assert_true(isnan(hi.as_float64().value()))

    _check_float_zeros(
        array([Float64(0.0), neg_zero, -5.0], float64),
        _bits([True, True, False]),
    )
    _check_float_zeros(
        array([neg_zero, Float64(0.0), 5.0], float64),
        _bits([True, True, False]),
    )

    var empty = array(float64)
    var no_bits = _bits(List[Bool]())
    var none_valid = _bits([False, False, False, False, False, False])
    assert_false(
        _masked_fold[MinFold, Float64Type](
            empty.values(), no_bits.view()
        ).is_valid()
    )
    assert_false(
        _masked_fold[MaxFold, Float64Type](
            mixed.values(), none_valid.view()
        ).is_valid()
    )

    # Group 0 is all NaN, group 1 mixes NaN with `3.0` and a masked-out `9.0`,
    # group 2 holds `-1.0` and `2.0`.
    var grouped = array([q, q, 3.0, q, -1.0, 9.0, 2.0], float64)
    var grouped_valid = _bits([True, True, True, True, True, False, True])
    var ids: List[Optional[Int]] = [0, 0, 1, 1, 2, 1, 2]
    var groups = Groups(array(ids, int32), 3)
    var lows = _float_scatter[MinFold](grouped, grouped_valid, groups)
    var highs = _float_scatter[MaxFold](grouped, grouped_valid, groups)
    assert_true(isnan(lows[0].as_float64().value()))
    assert_true(isnan(highs[0].as_float64().value()))
    assert_equal(lows[1].as_float64().value(), 3.0)
    assert_equal(highs[1].as_float64().value(), 3.0)
    assert_equal(lows[2].as_float64().value(), -1.0)
    assert_equal(highs[2].as_float64().value(), 2.0)
