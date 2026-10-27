# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The order kernels in `kernels/scan.mojo`, against scalar references over
every segment layout."""

from std.math import isnan, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...arrays import DynArray, Float64Array, Int64Array, PrimitiveArray
from ...buffers import Buffer
from ...builders import (
    Int32Builder,
    Int64Builder,
    ListBuilder,
    TimestampBuilder,
    array,
)
from ...dtypes import (
    float64,
    int8,
    int32,
    int64,
    second,
    timestamp,
    uint8,
)
from ...views import BufferView, cpu_lanes
from ...kernels.aggregate import SumFold
from ...kernels.core import BinaryKernel, Identity
from ...kernels.filter import TakeKernel
from ...kernels.nested import ArrayLengthKernel, ListReduceKernel
from ...kernels.numeric import AddKernel, MaxKernel, MinKernel, MulKernel
from ...kernels.scan import (
    CumulativeKernel,
    PairwiseDiffKernel,
    Scan,
    ShiftKernel,
    SumScan,
)
from ...kernels.segments import ByEnds, BySize, Segments


def _same_value[T: DType](a: Scalar[T], b: Scalar[T]) -> Bool:
    """Equal as values: `==`, or both NaN."""
    comptime if T.is_floating_point():
        return a == b or (isnan(a) and isnan(b))
    else:
        return a == b


# ---------------------------------------------------------------------------
# Scan — the running sum, its tests ported from `test_views.mojo`
# ---------------------------------------------------------------------------


def test_prefix_sum_register() raises:
    var state = Int32(10)
    var v = SIMD[DType.int32, 4](1, 2, 3, 4)
    var out = SumScan().core[DType.int32, 4](v, state)
    assert_true(out == SIMD[DType.int32, 4](11, 13, 16, 20))
    assert_equal(state, 20)


def test_prefix_sum_wraps() raises:
    var state = Int8(127)
    var v = SIMD[DType.int8, 2](1, 1)
    assert_true(
        SumScan().core[DType.int8, 2](v, state)
        == SIMD[DType.int8, 2](-128, -127)
    )


def _check_scan[T: DType](n: Int) raises:
    var buf = Buffer.alloc_zeroed[T](n)
    var sums = Buffer.alloc_zeroed[T](n)
    var expected = List[Scalar[T]]()
    var p = Scalar[T](0)
    for i in range(n):
        var x = Scalar[T]((i * 7 + 3) % 11) - Scalar[T](5)
        buf.unsafe_set[T](i, x)
        p += x
        expected.append(p)
    var view = sums.view[T](0, n)
    SumScan().run(buf.view[T](0, n), view, BySize.whole(n))
    for i in range(n):
        assert_equal(view[i], expected[i])


def test_scan_lengths() raises:
    for n in [0, 1, 7, 8, 63, 64, 65, 130, 1000]:
        _check_scan[DType.int64](n)
        _check_scan[DType.int32](n)
        _check_scan[DType.float64](n)


def _core_over(
    view: BufferView[mut=True, DType.int64, _],
    lo: Int,
    hi: Int,
    mut state: Int64,
):
    """`SumScan.core` over rows `[lo, hi)`, four lanes at a time, then one."""
    var i = lo
    while i + 4 <= hi:
        view.store[4](i, SumScan().core[DType.int64, 4](view.load[4](i), state))
        i += 4
    while i < hi:
        view.store[1](i, SumScan().core[DType.int64, 1](view.load[1](i), state))
        i += 1


def test_scan_chains_carry() raises:
    var chained = Buffer.alloc_zeroed[DType.int64](100)
    var whole = Buffer.alloc_zeroed[DType.int64](100)
    for i in range(100):
        chained.unsafe_set[DType.int64](i, 1)
        whole.unsafe_set[DType.int64](i, 1)
    var view = chained.view[DType.int64](0, 100)
    var state = SumScan().start[DType.int64]()
    _core_over(view, 0, 37, state)
    _core_over(view, 37, 100, state)
    assert_equal(state, 100)
    var sums = Buffer.alloc_zeroed[DType.int64](100)
    var once = sums.view[DType.int64](0, 100)
    SumScan().run(whole.view[DType.int64](0, 100), once, BySize.whole(100))
    for i in range(100):
        assert_equal(view[i], once[i])


def test_scan_empty_returns_carry() raises:
    assert_equal(SumScan().start[DType.int32](), 0)
    var src = Buffer.alloc_zeroed[DType.int32](4)
    var buf = Buffer.alloc_zeroed[DType.int32](4)
    for i in range(4):
        src.unsafe_set[DType.int32](i, Int32(1))
        buf.unsafe_set[DType.int32](i, Int32(42))
    var view = buf.view[DType.int32](0, 4)
    var ends = array([2, 2], int32)
    SumScan().run(src.view[DType.int32](0, 4), view, ByEnds(ends.values()))
    SumScan().run(src.view[DType.int32](0, 0), view, BySize.whole(0))
    for i in range(4):
        assert_equal(view[i], 42)


# ---------------------------------------------------------------------------
# Scan over segments
# ---------------------------------------------------------------------------


def _check_scan_segments[
    K: BinaryKernel & Identity, T: DType, S: Segments
](src: BufferView[T, _], segments: S, label: String) raises:
    """`Scan[K].run` against a left fold of each segment; rows in no segment
    keep their sentinel."""
    var n = len(src)
    var buf = Buffer.alloc_zeroed[T](n)
    var dst = buf.view[T](0, n)
    var sentinel = Scalar[T](77)
    for i in range(n):
        dst.store[1](i, sentinel)
    Scan[K]().run(src, dst, segments)
    var covered = List[Bool](length=n, fill=False)
    for s in range(segments.count()):
        var acc = K.identity[T]()
        for i in range(segments.start(s), segments.end(s)):
            acc = K.core[T, 1](acc, src[i])
            covered[i] = True
            assert_true(
                _same_value(dst[i], acc),
                String(label, " segment ", s, " row ", i, ": ", dst[i]),
            )
    for i in range(n):
        if not covered[i]:
            assert_equal(dst[i], sentinel, String(label, " row ", i))


def _check_scan_layouts[
    K: BinaryKernel & Identity, T: DType
](src: BufferView[T, _]) raises:
    comptime W = cpu_lanes[T]
    var n = len(src)
    for k in [1, 3, max(W - 1, 1), W, W + 1, 64, n]:
        _check_scan_segments[K](src, BySize(n, k), String(K.name, " size ", k))
    # Empty, single-row and unaligned segments; rows 0-4 and 290 on are in
    # none.
    var ends = array([5, 5, 6, 6, 9, 20, 21, 64, 65, 131, 290], int32)
    _check_scan_segments[K](src, ByEnds(ends.values()), String(K.name, " ends"))


def _check_scan_kernels[T: DType](src: BufferView[T, _]) raises:
    _check_scan_layouts[AddKernel](src)
    _check_scan_layouts[MulKernel](src)
    _check_scan_layouts[MinKernel](src)
    _check_scan_layouts[MaxKernel](src)


def test_scan_matches_reference_over_segments() raises:
    comptime n = 301
    var ints = Buffer.alloc_zeroed[DType.int64](n)
    var floats = Buffer.alloc_zeroed[DType.float64](n)
    var state = UInt64(7)
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        ints.unsafe_set[DType.int64](i, Int64(state))
        floats.unsafe_set[DType.float64](i, Float64(Int(state >> 61) - 3))
    _check_scan_kernels(ints.view[DType.int64](0, n))
    _check_scan_kernels(floats.view[DType.float64](0, n))


# ---------------------------------------------------------------------------
# CumulativeKernel
# ---------------------------------------------------------------------------


def test_cumulative_skip_nulls() raises:
    """PyArrow's answers, over a column and over a slice whose validity
    starts mid-byte."""
    var whole = BySize.whole(5)
    for values in [
        array([1, 2, None, 4, 5], int64),
        array([9, None, 9, 1, 2, None, 4, 5], int64).slice(3),
    ]:
        assert_true(
            CumulativeKernel[AddKernel].apply(values, False, whole)
            == array([1, 3, None, None, None], int64)
        )
        assert_true(
            CumulativeKernel[AddKernel].apply(values, True, whole)
            == array([1, 3, None, 7, 12], int64)
        )


def _payloads_zero_at_nulls(result: Int64Array) raises:
    for i in range(len(result)):
        if not result.is_valid(i):
            assert_equal(result.unsafe_get(i), 0, String("payload ", i))


def test_cumulative_resets_per_segment() raises:
    var values = array([1, None, 3, 4, 5, 6], int64)
    var strict = CumulativeKernel[AddKernel].apply(values, False, BySize(6, 3))
    assert_true(strict == array([1, None, None, 4, 9, 15], int64))
    _payloads_zero_at_nulls(strict)
    var skipping = CumulativeKernel[AddKernel].apply(values, True, BySize(6, 3))
    assert_true(skipping == array([1, None, 4, 4, 9, 15], int64))
    _payloads_zero_at_nulls(skipping)


def test_cumulative_sliced_input() raises:
    var values = array([1, 2, 3, 4, 5], int64).slice(2)
    assert_true(
        CumulativeKernel[AddKernel].apply(values, False, BySize.whole(3))
        == array([3, 7, 12], int64)
    )


def _assert_floats(got: Float64Array, expected: List[Float64]) raises:
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        assert_true(
            _same_value(got.unsafe_get(i), expected[i]),
            String("row ", i, ": ", got.unsafe_get(i), " is not ", expected[i]),
        )


def test_cumulative_min_leading_nan_answers_nan() raises:
    """Marrow's `min`: NaN is skipped unless every row so far is NaN, where
    PyArrow answers the float's greatest value."""
    var n = nan[DType.float64]()
    _assert_floats(
        CumulativeKernel[MinKernel].apply(
            array([n, 2.0, 0.5], float64), False, BySize.whole(3)
        ),
        [n, 2.0, 0.5],
    )
    _assert_floats(
        CumulativeKernel[MinKernel].apply(
            array([1.0, n, 0.5, 2.0], float64), False, BySize.whole(4)
        ),
        [1.0, 1.0, 0.5, 0.5],
    )


def test_cumulative_dispatch_rejects_non_numeric() raises:
    var strings: DynArray = array(["a", "b"])
    with assert_raises(contains="not numeric"):
        _ = CumulativeKernel[AddKernel].dispatch(strings, True)


# ---------------------------------------------------------------------------
# ShiftKernel
# ---------------------------------------------------------------------------


def test_shift_offset_copy_matches_take() raises:
    var b = Int64Builder(40)
    for i in range(40):
        if i % 5 == 0:
            b.append_null()
        else:
            b.append(Int64(i * 3 - 7))
    var values = b.finish().slice(3)
    var n = len(values)
    for offset in range(-n - 1, n + 2):
        var idx = Int32Builder(n)
        for i in range(n):
            var src = i - offset
            if src >= 0 and src < n:
                idx.append(Int32(src))
            else:
                idx.append_null()
        var expected = TakeKernel.apply(values, idx.finish())
        var got = ShiftKernel.apply(values, offset, BySize.whole(n))
        assert_true(got == expected, String("offset ", offset))
        _payloads_zero_at_nulls(got)


def _null_rows(result: Int64Array) -> List[Int]:
    var rows = List[Int]()
    for i in range(len(result)):
        if not result.is_valid(i):
            rows.append(i)
    return rows^


def test_shift_fills_each_segment_edge() raises:
    var values = array([1, 2, 3, 4, 5, 6, 7], int64)
    var lag = ShiftKernel.apply(values, 1, BySize(7, 3))
    assert_true(_null_rows(lag) == [0, 3, 6])
    assert_true(lag == array([None, 1, 2, None, 4, 5, None], int64))
    var lead = ShiftKernel.apply(values, -1, BySize(7, 3))
    assert_true(_null_rows(lead) == [2, 5, 6])
    assert_true(lead == array([2, 3, None, 5, 6, None, None], int64))


def test_shift_non_primitive_layouts() raises:
    """Polars' answers."""
    var strings: DynArray = array(["a", "b", None])
    assert_true(
        ShiftKernel.dispatch(strings, 1).as_string() == array([None, "a", "b"])
    )
    var bools: DynArray = array([True, False, None])
    assert_true(
        ShiftKernel.dispatch(bools, 1).as_bool() == array([None, True, False])
    )
    var b = ListBuilder(Int64Builder())
    var child_any = b.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    b.append_valid()  # [1, 2]
    child.append(3)
    b.append_valid()  # [3]
    child.append(4)
    child.append(5)
    child.append(6)
    b.append_valid()  # [4, 5, 6]
    var lists: DynArray = b.finish()
    var lead = ShiftKernel.dispatch(lists, -1)  # [[3], [4, 5, 6], null]
    assert_true(
        ArrayLengthKernel.dispatch(lead).as_int32()
        == array([1, 3, None], int32)
    )
    assert_true(
        ListReduceKernel[SumFold].dispatch(lead).as_int64()
        == array([3, 15, None], int64)
    )


# ---------------------------------------------------------------------------
# PairwiseDiffKernel
# ---------------------------------------------------------------------------


def test_pairwise_diff_periods() raises:
    """Over a column and over a slice whose validity starts mid-byte."""
    for values in [
        array([5, 1, None, 7, 2], int64),
        array([0, None, 3, 5, 1, None, 7, 2], int64).slice(3),
    ]:
        var n = len(values)
        for period in range(-6, 7):
            var expected = Int64Builder(n)
            for i in range(n):
                var j = i - period
                if (
                    j >= 0
                    and j < n
                    and values.is_valid(i)
                    and values.is_valid(j)
                ):
                    expected.append(values.unsafe_get(i) - values.unsafe_get(j))
                else:
                    expected.append_null()
            var got = PairwiseDiffKernel.apply(values, period, BySize.whole(n))
            assert_true(got == expected.finish(), String("period ", period))
            _payloads_zero_at_nulls(got)


def test_pairwise_diff_wraps() raises:
    assert_true(
        PairwiseDiffKernel.apply(array([1, 0], uint8), 1, BySize.whole(2))
        == array([None, 255], uint8)
    )
    assert_true(
        PairwiseDiffKernel.apply(array([-128, 127], int8), 1, BySize.whole(2))
        == array([None, -1], int8)
    )


def test_pairwise_diff_per_segment() raises:
    var values = array([1, 3, 6, 10, 15, 21], int64)
    var got = PairwiseDiffKernel.apply(values, 1, BySize(6, 3))
    assert_true(got == array([None, 2, 3, None, 5, 6], int64))


def test_pairwise_diff_rejects_temporal() raises:
    var t = TimestampBuilder(timestamp(second, "UTC"), 2)
    t.append(Int64(1))
    t.append(Int64(5))
    var stamps: DynArray = t.finish()
    with assert_raises(contains="not numeric"):
        _ = PairwiseDiffKernel.dispatch(stamps, 1)
