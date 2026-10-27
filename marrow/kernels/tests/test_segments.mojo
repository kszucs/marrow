# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Segments and `Segmented.reduce`, against a scalar fold per segment."""

from std.math import isnan, nan
from std.testing import assert_equal, assert_false, assert_true

from ...arrays import Int32Array, PrimitiveArray
from ...builders import Int64Builder, PrimitiveBuilder, array
from ...dtypes import (
    Float64Type,
    Int8Type,
    Int32Type,
    Int64Type,
    NumericType,
    float64,
    int8,
    int32,
    int64,
)
from ...views import cpu_lanes
from ...kernels.aggregate import (
    CountFold,
    FoldKernel,
    MaxFold,
    MeanFold,
    MinFold,
    SumFold,
)
from ...kernels.segments import ByEnds, BySize, Segmented, Segments


def _bounds[S: Segments](segments: S) -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]]()
    for s in range(segments.count()):
        out.append((segments.start(s), segments.end(s)))
    return out^


def _assert_segments[
    S: Segments
](segments: S, expected: List[Tuple[Int, Int]]) raises:
    var got = _bounds(segments)
    assert_equal(len(got), len(expected))
    for s in range(len(got)):
        assert_equal(got[s][0], expected[s][0], String("start of ", s))
        assert_equal(got[s][1], expected[s][1], String("end of ", s))


def test_by_size_cuts_the_last_segment_short() raises:
    _assert_segments(BySize(10, 4), [(0, 4), (4, 8), (8, 10)])
    assert_equal(BySize(8, 4).count(), 2)
    assert_equal(BySize(0, 4).count(), 0)
    assert_equal(BySize.whole(0).count(), 0)
    _assert_segments(BySize.whole(5), [(0, 5)])


def test_by_ends_reads_sliced_offsets() raises:
    var ends = array([3, 3, 5, 9], int32)
    _assert_segments(ByEnds(ends.values()), [(3, 3), (3, 5), (5, 9)])


def _same_value[T: DType](a: Scalar[T], b: Scalar[T]) -> Bool:
    """Equal as values: `==`, or both NaN."""
    comptime if T.is_floating_point():
        return a == b or (isnan(a) and isnan(b))
    else:
        return a == b


def _column[V: NumericType](n: Int) raises -> PrimitiveArray[V]:
    """`n` small values, every 7th row null and, for a float, every 11th row
    NaN: every sum of them is exact."""
    comptime T = V.native
    var b = PrimitiveBuilder[V](n)
    for i in range(n):
        if i % 7 == 3:
            b.append_null()
        else:
            comptime if T.is_floating_point():
                if i % 11 == 5:
                    b.append(nan[T]())
                else:
                    b.append(Scalar[T]((i * 37) % 101 - 50))
            else:
                b.append(Scalar[T]((i * 37) % 101 - 50))
    return b.finish()


def _check_reduce[
    F: FoldKernel, V: NumericType, S: Segments
](values: PrimitiveArray[V], segments: S, label: String) raises:
    """`Segmented[F].reduce` against a scalar left fold of each segment's
    valid values."""
    comptime A = F.AccType[V].native
    var got = Segmented[F].reduce(values, segments)
    assert_equal(len(got), segments.count(), label)
    for s in range(segments.count()):
        var acc = F.identity[A]()
        var count = 0
        for i in range(segments.start(s), segments.end(s)):
            if values.is_valid(i):
                acc = F.combine[A, 1](acc, values.unsafe_get(i).cast[A]())
                count += 1
        var at = String(label, " segment ", s)
        if count == 0 and F.empty_is_null:
            assert_false(got.is_valid(s), at)
            assert_equal(got.unsafe_get(s), Scalar[A](0), at + " payload")
        else:
            assert_true(got.is_valid(s), at)
            var expected = F.finalize[A](acc, count)
            assert_true(
                _same_value(got.unsafe_get(s), expected),
                String(at, ": ", got.unsafe_get(s), " is not ", expected),
            )


def _check_layouts[F: FoldKernel, V: NumericType]() raises:
    comptime Wv = cpu_lanes[V.native]
    comptime Wa = cpu_lanes[F.AccType[V].native]
    var values = _column[V](300)
    for k in [1, 3, 7, Wv, Wv + 1, Wa, Wa + 1, 64, 300]:
        _check_reduce[F](
            values, BySize(300, k), String(F.name, " ", V(), " size ", k)
        )
    # Empty, single-row and unaligned segments, starting at row 5.
    var ends = array([5, 5, 6, 6, 9, 20, 21, 64, 65, 131, 300], int32)
    _check_reduce[F](
        values, ByEnds(ends.values()), String(F.name, " ", V(), " ends")
    )
    # A sliced column: positions index it as it stands.
    var sliced = values.slice(13, 250)
    _check_reduce[F](
        sliced, BySize(250, Wa + 1), String(F.name, " ", V(), " sliced")
    )


def _check_types[F: FoldKernel]() raises:
    _check_layouts[F, Int32Type]()
    _check_layouts[F, Int64Type]()
    _check_layouts[F, Float64Type]()


def test_segmented_reduce_matches_reference() raises:
    _check_types[SumFold]()
    _check_types[MinFold]()
    _check_types[MaxFold]()
    _check_types[MeanFold]()
    _check_types[CountFold]()


def test_segmented_reduce_every_tail_length() raises:
    comptime W = cpu_lanes[DType.int64]
    for n in [0, 1, W - 1, W, W + 1, 3 * W + 5]:
        var b = Int64Builder(n)
        for i in range(n):
            b.append(Int64(i + 1))
        var values = b.finish()
        var ends = array([0, n], int32)
        var sums = Segmented[SumFold].reduce(values, ByEnds(ends.values()))
        var counts = Segmented[CountFold].reduce(values, ByEnds(ends.values()))
        assert_equal(len(sums), 1)
        if n == 0:
            assert_false(sums.is_valid(0))
        else:
            assert_equal(sums.unsafe_get(0), Int64(n * (n + 1) // 2))
        assert_true(counts.is_valid(0))
        assert_equal(counts.unsafe_get(0), Int64(n))


def test_segmented_reduce_widens_narrow_sums() raises:
    var values = array([100, 100, 100], int8)
    var sums = Segmented[SumFold].reduce(values, BySize.whole(3))
    assert_true(sums == array([300], int64))


def test_segmented_reduce_float_min_max_nan_and_signed_zero() raises:
    var n = nan[DType.float64]()
    var some_nan = array([n, 3.0, -1.0, n], float64)
    var mins = Segmented[MinFold].reduce(some_nan, BySize.whole(4))
    var maxs = Segmented[MaxFold].reduce(some_nan, BySize.whole(4))
    assert_equal(mins.unsafe_get(0), -1.0)
    assert_equal(maxs.unsafe_get(0), 3.0)

    var all_nan = array([n, n], float64)
    assert_true(
        isnan(Segmented[MinFold].reduce(all_nan, BySize.whole(2)).unsafe_get(0))
    )
    assert_true(
        isnan(Segmented[MaxFold].reduce(all_nan, BySize.whole(2)).unsafe_get(0))
    )

    var z = Float64(0)
    var nz = -z
    for zeros in [array([z, nz], float64), array([nz, z], float64)]:
        var low = Segmented[MinFold].reduce(zeros, BySize.whole(2))
        var high = Segmented[MaxFold].reduce(zeros, BySize.whole(2))
        # Which zero answers is the target's choice: compare by value.
        assert_equal(low.unsafe_get(0), 0.0)
        assert_equal(high.unsafe_get(0), 0.0)

    var empty = array(float64)
    var ends = array([0, 0], int32)
    var none = Segmented[MinFold].reduce(empty, ByEnds(ends.values()))
    assert_equal(len(none), 1)
    assert_false(none.is_valid(0))
