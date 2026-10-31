# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Segments and `Segmented.reduce`, against a scalar fold per segment; runs,
and the run-end encoding built on them."""

from std.math import isnan, nan
from std.memory import bitcast
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...arrays import (
    DictionaryArray,
    DynArray,
    Float64Array,
    Int32Array,
    Int64Array,
    PrimitiveArray,
    RunEndEncodedArray,
    StructArray,
)
from ...buffers import Buffer
from ...builders import (
    Decimal128Builder,
    Float64Builder,
    Int64Builder,
    ListBuilder,
    PrimitiveBuilder,
    arange,
    array,
    nulls,
)
from ...dtypes import (
    DynType,
    Field,
    Float64Type,
    Int8Type,
    Int32Type,
    Int64Type,
    NumericType,
    decimal128,
    float64,
    int8,
    int16,
    int32,
    int64,
    null,
    run_end_encoded,
    struct_,
    uint32,
)
from ...execution import ExecContext
from ...views import cpu_lanes
from ...kernels.aggregate import (
    CountFold,
    FoldKernel,
    MaxFold,
    MeanFold,
    MinFold,
    SumFold,
)
from ...kernels.runs import (
    RunIndexKernel,
    RunEndDecodeKernel,
    RunEndEncodeKernel,
    RunsKernel,
)
from ...kernels.segments import ByEnds, BySize, PerSegment, Segmented, Segments
from ...kernels.window import WindowFrame, WindowSegments


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


# ---------------------------------------------------------------------------
# RunsKernel
# ---------------------------------------------------------------------------


def test_runs_end_where_any_key_changes() raises:
    var a: DynArray = array([1, 1, 2, 2, 2, 3], int64)
    assert_true(
        RunsKernel.dispatch([a.copy()], 6) == array([0, 2, 5, 6], int64)
    )
    var b: DynArray = array(["a", "b", "b", "b", "b", "b"])
    assert_true(
        RunsKernel.dispatch([a.copy(), b.copy()], 6)
        == array([0, 1, 2, 5, 6], int64)
    )


def test_runs_hold_nulls_and_nans_together() raises:
    """Nulls are one key, and so is every NaN whatever its sign."""
    var q = nan[DType.float64]()
    var b = Float64Builder(6)
    b.append_null()
    b.append_null()
    b.append(q)
    b.append(-q)
    b.append(1.0)
    b.append(1.0)
    var key: DynArray = b.finish()
    assert_true(RunsKernel.dispatch([key^], 6) == array([0, 2, 4, 6], int64))


def test_runs_edges() raises:
    assert_true(
        RunsKernel.dispatch(List[DynArray](), 4) == array([0, 4], int64)
    )
    var empty: DynArray = array(int64)
    assert_true(RunsKernel.dispatch([empty^], 0) == array([0], int64))
    var one: DynArray = array([7], int64)
    assert_true(RunsKernel.dispatch([one^], 1) == array([0, 1], int64))
    var sliced: DynArray = array([9, 1, 1, 2], int64).slice(1)
    assert_true(RunsKernel.dispatch([sliced^], 3) == array([0, 2, 3], int64))
    var short: DynArray = array([1, 2, 3], int64)
    with assert_raises(contains="InvalidError: runs: a key holds 3 rows"):
        _ = RunsKernel.dispatch([short^], 4)


def test_runs_over_513_rows() raises:
    """513 rows compare in 512 pairs, whose bits fill 64 bytes exactly; a
    run starts at the last row."""
    var b = Int64Builder(513)
    for i in range(512):
        b.append(Int64(i // 100))
    b.append(Int64(-1))
    var key: DynArray = b.finish()
    assert_true(
        RunsKernel.dispatch([key^], 513)
        == array([0, 100, 200, 300, 400, 500, 512, 513], int64)
    )


# ---------------------------------------------------------------------------
# RunEndEncodeKernel, RunEndDecodeKernel, RunIndexKernel
# ---------------------------------------------------------------------------


def _encodes(values: DynArray, run_ends: Int32Array, runs: DynArray) raises:
    """`values` encodes to `runs`, ending at `run_ends`, and decodes back to
    its rows."""
    var encoded = RunEndEncodeKernel.dispatch(values, int32)
    assert_true(
        encoded.type() == DynType(run_end_encoded(int32, values.dtype()))
    )
    assert_equal(len(encoded), len(values))
    assert_true(encoded.run_ends().as_int32() == run_ends)
    assert_equal(String(encoded.values()), String(runs))
    assert_equal(String(RunEndDecodeKernel.dispatch(encoded^)), String(values))


def test_run_end_encode_finds_runs_of_equal_bits() raises:
    _encodes(
        array([5, 5, 5, 1, 2, 2], int64),
        array([3, 4, 6], int32),
        array([5, 1, 2], int64),
    )
    # Consecutive nulls are one run, of a null value.
    _encodes(
        array([1, None, None, 1], int64),
        array([1, 3, 4], int32),
        array([1, None, 1], int64),
    )
    _encodes(
        array(["a", "a", None, "b"]),
        array([2, 3, 4], int32),
        array(["a", None, "b"]),
    )
    _encodes(
        array([True, True, None, False]),
        array([2, 3, 4], int32),
        array([True, None, False]),
    )
    var decimals = Decimal128Builder(decimal128(5, 2), 3)
    decimals.append(Scalar[DType.int128](1))
    decimals.append(Scalar[DType.int128](1))
    decimals.append(Scalar[DType.int128](1) << Scalar[DType.int128](70))
    var two = Decimal128Builder(decimal128(5, 2), 2)
    two.append(Scalar[DType.int128](1))
    two.append(Scalar[DType.int128](1) << Scalar[DType.int128](70))
    _encodes(decimals.finish(), array([2, 3], int32), two.finish())
    _encodes(
        nulls(3, DynType(null)), array([3], int32), nulls(1, DynType(null))
    )
    _encodes(array(int64), array(int32), array(int64))
    _encodes(
        array([9, 5, 5, 1], int64).slice(1),
        array([2, 3], int32),
        array([5, 1], int64),
    )

    # A float is compared by its bits: NaNs with two payloads are two runs,
    # and so are -0.0 and 0.0.
    var floats = Float64Builder(5)
    floats.append(nan[DType.float64]())
    floats.append(bitcast[DType.float64](UInt64(0x7FF8000000000001)))
    floats.append(-0.0)
    floats.append(0.0)
    floats.append(0.0)
    var column: DynArray = floats.finish()
    var bits = RunEndEncodeKernel.dispatch(column, int32)
    assert_true(bits.run_ends().as_int32() == array([1, 2, 3, 5], int32))
    var runs = bits.values()
    ref kept = runs.as_float64()
    assert_equal(
        bitcast[DType.uint64](kept.unsafe_get(1)), UInt64(0x7FF8000000000001)
    )
    assert_equal(
        bitcast[DType.uint64](kept.unsafe_get(2)), UInt64(0x8000000000000000)
    )
    var decoded = RunEndDecodeKernel.dispatch(bits^)
    assert_equal(
        bitcast[DType.uint64](decoded.as_float64().unsafe_get(1)),
        UInt64(0x7FF8000000000001),
    )

    var five: DynArray = array([5, 5, 5, 1, 2, 2], int64)
    var narrow = RunEndEncodeKernel.dispatch(five, int16)
    assert_true(narrow.run_ends().as_int16() == array([3, 4, 6], int16))
    var wide = RunEndEncodeKernel.dispatch(five, int64)
    assert_true(wide.run_ends().as_int64() == array([3, 4, 6], int64))


def test_run_end_encode_refuses() raises:
    var lists = ListBuilder(Int64Builder())
    lists.values().as_int64().append(1)
    lists.append_valid()
    var refused = List[DynArray]()
    refused.append(lists.finish())
    var children = List[DynArray]()
    children.append(array([1], int64))
    refused.append(
        StructArray(
            dtype=struct_(Field("a", int64)), length=1, children=children^
        )
    )
    refused.append(
        DictionaryArray.from_arrays(array([0, 0], int32), array(["a"]))
    )
    refused.append(
        RunEndEncodedArray.from_arrays(array([2], int32), array([1], int64))
    )
    for ref values in refused:
        with assert_raises(contains="TypeError: run_end_encode: cannot encode"):
            _ = RunEndEncodeKernel.dispatch(values, int32)
    var ints: DynArray = array([1], int64)
    with assert_raises(contains="TypeError: run_end_encode: run ends must be"):
        _ = RunEndEncodeKernel.dispatch(ints, int8)
    var full: DynArray = arange[Int64Type](0, 32767)
    assert_equal(len(RunEndEncodeKernel.dispatch(full, int16)), 32767)
    var over: DynArray = arange[Int64Type](0, 32768)
    with assert_raises(contains="InvalidError"):
        _ = RunEndEncodeKernel.dispatch(over, int16)


def test_run_end_decode_restores_the_rows() raises:
    """A window starting and ending inside runs decodes to its rows only;
    fixed-width and boolean values are repeated, the rest gathered."""
    var five = RunEndEncodedArray.from_arrays(
        array([3, 4, 6], int32), array([5, 1, 2], int64)
    )
    assert_true(
        RunEndDecodeKernel.apply(five.slice(1, 4)).as_int64()
        == array([5, 5, 1, 2], int64)
    )
    var gapped = RunEndEncodedArray.from_arrays(
        array([2, 3, 5], int32), array([1, None, 2], int64)
    )
    assert_true(
        RunEndDecodeKernel.apply(gapped.slice(1, 3)).as_int64()
        == array([1, None, 2], int64)
    )
    var flags = RunEndEncodedArray.from_arrays(
        array([2, 3, 5], int16), array([True, None, False])
    )
    assert_true(
        RunEndDecodeKernel.apply(flags.slice(1, 4)).as_bool()
        == array([True, None, False, False])
    )
    var words = RunEndEncodedArray.from_arrays(
        array([1, 3, 4], int64), array(["a", "b", None])
    )
    assert_equal(
        String(RunEndDecodeKernel.apply(words.slice(1, 3))),
        String(array(["b", "b", None])),
    )
    assert_equal(len(RunEndDecodeKernel.apply(five.slice(6, 0))), 0)

    var lists = ListBuilder(Int64Builder())
    lists.values().as_int64().append(1)
    lists.append_valid()
    lists.values().as_int64().append(2)
    lists.values().as_int64().append(3)
    lists.append_valid()
    var listed = RunEndEncodedArray.from_arrays(
        array([2, 3], int32), lists.finish()
    )
    var want = ListBuilder(Int64Builder())
    for _ in range(2):
        want.values().as_int64().append(1)
        want.append_valid()
    want.values().as_int64().append(2)
    want.values().as_int64().append(3)
    want.append_valid()
    assert_equal(
        String(RunEndDecodeKernel.apply(listed)), String(want.finish())
    )

    var children = List[DynArray]()
    children.append(array([1, 2], int64))
    var pairs = StructArray(
        dtype=struct_(Field("a", int64)), length=2, children=children^
    )
    var structs = RunEndEncodedArray.from_arrays(array([1, 3], int32), pairs^)
    var expected = List[DynArray]()
    expected.append(array([1, 2, 2], int64))
    assert_equal(
        String(RunEndDecodeKernel.apply(structs)),
        String(
            StructArray(
                dtype=struct_(Field("a", int64)),
                length=3,
                children=expected^,
            )
        ),
    )

    var plain: DynArray = array([1], int64)
    with assert_raises(contains="TypeError: run_end_decode"):
        _ = RunEndDecodeKernel.dispatch(plain)


def test_run_end_decode_core_over_empty_segments() raises:
    """A segment may be empty: the multiset's groups with no copies."""
    var ends = array([0, 2, 2, 5], int64)
    assert_true(
        RunEndDecodeKernel.apply(arange[Int32Type](0, 3), ByEnds(ends.values()))
        == array([0, 0, 2, 2, 2], int32)
    )
    var none = array([0], int64)
    assert_equal(
        len(
            RunEndDecodeKernel.apply(
                arange[Int32Type](0, 0), ByEnds(none.values())
            )
        ),
        0,
    )
    with assert_raises(contains="InvalidError: run_end_decode: 2 values"):
        _ = RunEndDecodeKernel.apply(
            arange[Int32Type](0, 2), ByEnds(ends.values())
        )


def test_run_index_numbers_each_run() raises:
    """Rows are numbered by run under the group-by identity, as `uint32`."""
    var ints: DynArray = array([1, 1, None, None, 2, 1], int64)
    var ids = RunIndexKernel.dispatch(ints)
    assert_true(ids.type() == DynType(uint32))
    assert_true(ids == array([0, 0, 1, 1, 2, 3], uint32))

    var floats = Float64Builder(5)
    floats.append(nan[DType.float64]())
    floats.append(bitcast[DType.float64](UInt64(0x7FF8000000000001)))
    floats.append(-0.0)
    floats.append(0.0)
    floats.append(1.0)
    var signed: DynArray = floats.finish()
    assert_true(
        RunIndexKernel.dispatch(signed) == array([0, 0, 1, 1, 2], uint32)
    )

    var lists = ListBuilder(Int64Builder())
    for item in [1, 1, 2]:
        lists.values().as_int64().append(Int64(item))
        lists.append_valid()
    var listed: DynArray = lists.finish()
    assert_true(RunIndexKernel.dispatch(listed) == array([0, 0, 1], uint32))

    var children = List[DynArray]()
    children.append(array([1, 1, 2], int64))
    var structs: DynArray = StructArray(
        dtype=struct_(Field("a", int64)), length=3, children=children^
    )
    assert_true(RunIndexKernel.dispatch(structs) == array([0, 0, 1], uint32))

    var empty: DynArray = array(int64)
    assert_equal(len(RunIndexKernel.dispatch(empty)), 0)


# ---------------------------------------------------------------------------
# PerSegment
# ---------------------------------------------------------------------------


def test_per_segment_broadcast_fills_each_segment() raises:
    comptime I = DType.int64
    comptime F = DType.float64
    var ends = array([0, 2, 2, 5], int64)
    var five = Buffer.alloc_zeroed[I](5)
    PerSegment(array([7, 8, 9], int64).buffer, 3).broadcast(
        ByEnds(ends.values()), five.view[I](0, 5)
    )
    assert_true(
        Int64Array(length=5, buffer=five^.to_immutable())
        == array([7, 7, 9, 9, 9], int64)
    )

    var three = Buffer.alloc_zeroed[F](3)
    PerSegment(array([0.5, 1.5], float64).buffer, 2).broadcast(
        BySize(3, 2), three.view[F](0, 3)
    )
    assert_true(
        Float64Array(length=3, buffer=three^.to_immutable())
        == array([0.5, 0.5, 1.5], float64)
    )

    var inner = array([1, 3], int64)
    var four = Buffer.alloc_zeroed[I](4)
    PerSegment(array([4], int64).buffer, 1).broadcast(
        ByEnds(inner.values()), four.view[I](0, 4)
    )
    assert_true(
        Int64Array(length=4, buffer=four^.to_immutable())
        == array([0, 4, 4, 0], int64)
    )


def test_by_size_finds_each_row_s_segment() raises:
    var sixteen = BySize(100, 16)
    for i in range(100):
        assert_equal(sixteen.of(i), i // 16)
    assert_equal(BySize(100, 12).of(37), 3)
    var two = PerSegment(array([7, 9], int64).buffer, 2)
    assert_equal(len(two), 2)
    assert_true(two.get[DType.int64](1) == 9)


# ---------------------------------------------------------------------------
# WindowSegments
# ---------------------------------------------------------------------------


def _window_keys() raises -> List[DynArray]:
    """The partition key, then the order key, already sorted."""
    var k: DynArray = array(["a", "a", "a", "b", "b"])
    var o: DynArray = array([1, 2, 2, 5, 5], int64)
    return [k^, o^]


def test_window_segments_of_sorted() raises:
    var ctx = ExecContext.serial()
    var both = WindowSegments.of_sorted(_window_keys(), 1, 5, ctx)
    assert_true(both.partition_ends == array([0, 3, 5], int64))
    assert_true(both.peer_ends == array([0, 1, 3, 5], int64))
    assert_true(both.partition_peers == array([0, 2, 3], int64))
    assert_equal(len(both), 5)

    var order = _window_keys()[1].copy()
    var ordered = WindowSegments.of_sorted([order^], 0, 5, ctx)
    assert_true(ordered.partition_ends == array([0, 5], int64))
    assert_true(ordered.peer_ends == array([0, 1, 3, 5], int64))
    assert_true(ordered.partition_peers == array([0, 3], int64))
    assert_equal(len(ordered), 5)

    var none = WindowSegments.of_sorted(List[DynArray](), 0, 5, ctx)
    assert_true(none.partition_ends == array([0, 5], int64))
    assert_true(none.peer_ends == array([0, 5], int64))
    assert_true(none.partition_peers == array([0, 1], int64))
    assert_equal(len(none), 5)


def _check_frames(
    segments: WindowSegments,
    frame: WindowFrame,
    starts: Int32Array,
    stops: Int32Array,
) raises:
    var got = segments.frames(frame)
    assert_true(got[0] == starts, String(frame, " starts"))
    assert_true(got[1] == stops, String(frame, " stops"))


def test_window_segments_frames() raises:
    var segments = WindowSegments.of_sorted(
        _window_keys(), 1, 5, ExecContext.serial()
    )
    _check_frames(
        segments,
        WindowFrame.default(),
        array([0, 0, 0, 3, 3], int32),
        array([1, 3, 3, 5, 5], int32),
    )
    _check_frames(
        segments,
        WindowFrame(True, -1, 0),
        array([0, 0, 1, 3, 3], int32),
        array([1, 2, 3, 4, 5], int32),
    )
    _check_frames(
        segments,
        WindowFrame(True, 0, 1),
        array([0, 1, 2, 3, 4], int32),
        array([2, 3, 3, 5, 5], int32),
    )
    _check_frames(
        segments,
        WindowFrame(True, 5, 10),
        array([3, 3, 3, 5, 5], int32),
        array([3, 3, 3, 5, 5], int32),
    )
    _check_frames(
        segments,
        WindowFrame(True, 3, 1),
        array([3, 3, 3, 5, 5], int32),
        array([3, 3, 3, 5, 5], int32),
    )
