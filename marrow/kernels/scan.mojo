# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Order-dependent kernels: each row's answer depends on the rows before it in
its segment.

`Scan[K]` is the running fold of a binary kernel, one register at a time, and
restarts at every segment; `SumScan` is `Scan[AddKernel]`.
`CumulativeKernel[K]` is `cumulative_sum`, `cumulative_prod`,
`cumulative_min` and `cumulative_max` over it. `ShiftKernel` reads each row
some rows back (`shift`), and `PairwiseDiffKernel` subtracts that row
(`pairwise_diff`). `RollingKernel[K]` folds each row's frame
`ROWS BETWEEN lo AND hi` (`rolling_sum`, `rolling_min`, `rolling_max`), and
`RollingMeanKernel` averages it (`rolling_mean`).

Every kernel here takes `Segments` and computes within each segment, so a
column cut into partitions is one call. Over a whole column the segments are
`BySize.whole(n)`.
"""

from std.math import align_up, iota

from ..arrays import DynArray, Float64Array, PrimitiveArray
from ..buffers import Bitmap, Buffer
from ..builders import arange
from ..dtypes import Float64Type, Int32Type, NumericType, PrimitiveType
from ..errors import InvalidError
from ..execution import ExecContext
from ..views import BufferView, apply, cpu_lanes, valid_or_zero
from .core import Associative, BinaryKernel, Identity, Kernel
from .filter import TakeKernel
from .numeric import AddKernel, SubKernel
from .segments import BySize, Segments


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


struct Scan[K: BinaryKernel & Identity](Copyable, Defaultable, Movable):
    """The running fold of `K`: row `i` of a segment is `K` over the
    segment's rows up to and including `i`, folded left to right.

    The state is the fold so far. `core` folds one register into it and
    answers every lane's running value, so a caller may carry the state
    across calls; `run` drives `core` over a column's segments.
    """

    comptime State[T: DType] = Scalar[T]

    def __init__(out self):
        pass

    def start[T: DType](self) -> Self.State[T]:
        """The state before the first row: `K`'s identity."""
        return Self.K.identity[T]()

    def reset[T: DType](self, mut state: Self.State[T]):
        """Back to the state before the first row, at a segment start."""
        state = Self.K.identity[T]()

    @always_inline
    def core[
        T: DType, W: Int
    ](self, v: SIMD[T, W], mut state: Self.State[T]) -> SIMD[T, W]:
        """Lane `j` is `K(state, v[0], ..., v[j])`, folded left to right; the
        state becomes lane `W - 1`. Integers wrap."""
        # Lane by lane: a log-step of shifted adds lowers each zero-filling
        # shift to a shuffle and measured slower.
        var out = SIMD[T, W]()
        var p = state
        comptime for j in range(W):
            p = Self.K.core[T, 1](p, v[j])
            out[j] = p
        state = p
        return out

    def run[
        T: DType, S: Segments
    ](
        self,
        src: BufferView[T, _],
        dst: BufferView[mut=True, T, _],
        segments: S,
    ):
        """`dst[i]` is `K` over `src[start(s) .. i]` for every row `i` of
        segment `s`. Rows in no segment are left as they are."""
        comptime W = cpu_lanes[T]
        var state = self.start[T]()
        for s in range(segments.count()):
            self.reset[T](state)
            var i = segments.start(s)
            var end = segments.end(s)
            # One row at a time up to a register boundary, so whole registers
            # stay aligned to the column.
            var head = min(end, align_up(i, W))
            while i < head:
                dst.store[1](i, self.core[T, 1](src.load[1](i), state))
                i += 1
            while i + W <= end:
                dst.store[W](i, self.core[T, W](src.load[W](i), state))
                i += W
            while i < end:
                dst.store[1](i, self.core[T, 1](src.load[1](i), state))
                i += 1


comptime SumScan = Scan[AddKernel]
"""Running sums."""


# ---------------------------------------------------------------------------
# CumulativeKernel
# ---------------------------------------------------------------------------


struct CumulativeKernel[K: BinaryKernel & Identity](Kernel):
    """The running `K` of each segment, in the input's dtype (integers
    wrap): `cumulative_sum`, `cumulative_prod`, `cumulative_min` and
    `cumulative_max` with `K = AddKernel`, `MulKernel`, `MinKernel` and
    `MaxKernel`.

    A null row stays null and folds as `K`'s identity. With
    `skip_nulls=False`, every row of a segment from its first null on is null.
    A floating-point minimum or maximum skips NaN, as `min` and `max` do, so it
    is NaN only while every row so far is.
    """

    comptime name = "cumulative"

    @staticmethod
    def apply[
        T: NumericType, S: Segments
    ](
        values: PrimitiveArray[T],
        skip_nulls: Bool,
        segments: S,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """`values`' running `K`, restarting at every segment; the segments
        partition the rows."""
        comptime N = T.native
        var n = len(values)
        var out = Buffer.alloc_uninit[N](n)
        var bitmap: Optional[Bitmap[mut=False]] = None
        var nulls = 0
        if values.null_count() == 0:
            Scan[Self.K]().run(values.values(), out.view[N](0, n), segments)
        else:
            var valid = values.validity().value()
            var scanned = Buffer.alloc_uninit[N](n)
            # A null folds as the identity; `out` holds the folded input
            # first, then the answer with a zero under every null.
            apply[N, N, Self._identity_where_null[N, _]](
                values.values(), valid, out.view[N](0, n)
            )
            Scan[Self.K]().run(
                out.view[N](0, n), scanned.view[N](0, n), segments
            )
            var kept = Bitmap.alloc_zeroed(n)
            kept.extend(valid, 0, n)
            if not skip_nulls:
                for s in range(segments.count()):
                    var lo = segments.start(s)
                    var hi = segments.end(s)
                    if valid.slice(lo, hi - lo).count_set_bits() < hi - lo:
                        var first = lo
                        while valid[first]:
                            first += 1
                        kept.set_range(first, hi - first, False)
            apply[N, N, valid_or_zero[N, _]](
                scanned.view[N](0, n), kept.view(0, n), out.view[N](0, n)
            )
            nulls = kept.view(0, n).unset_count()
            bitmap = kept^.to_immutable(length=n)
        return PrimitiveArray[T](
            values.dtype.copy(),
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=bitmap,
            buffer=out^.to_immutable(),
        )

    @staticmethod
    def dispatch(
        values: DynArray,
        skip_nulls: Bool,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """`apply` over the whole column, for a numeric dtype."""

        def leaf[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                values.as_primitive[T](),
                skip_nulls,
                BySize.whole(len(values)),
                ctx,
            ).to_dyn()

        return values.dtype().try_visit[DynArray, NumericType](
            leaf, "dtype is not numeric"
        )

    @always_inline
    @staticmethod
    def _identity_where_null[
        T: DType, W: Int
    ](values: SIMD[T, W], valid: SIMD[DType.bool, W]) -> SIMD[T, W]:
        return valid.select(values, SIMD[T, W](Self.K.identity[T]()))


# ---------------------------------------------------------------------------
# ShiftKernel, PairwiseDiffKernel
# ---------------------------------------------------------------------------


struct ShiftKernel(Kernel):
    """Each row takes the value `offset` rows before it in its segment, and is
    null where that row is outside the segment: `shift`, and over partitions
    `lag` (a positive offset) and `lead` (a negative one). Every layout."""

    comptime name = "shift"

    @staticmethod
    def apply[
        T: PrimitiveType, S: Segments
    ](
        values: PrimitiveArray[T],
        offset: Int,
        segments: S,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Row `i` of segment `s` is row `i - offset` when that row is in `s`,
        and null otherwise: one copy per segment."""
        comptime N = T.native
        var n = len(values)
        var source = values.buffer
        var first = values.offset
        if values.null_count() > 0:
            # A copied null keeps its payload, so copy from values whose nulls
            # are zero.
            var clean = Buffer.alloc_uninit[N](n)
            apply[N, N, valid_or_zero[N, _]](
                values.values(),
                values.validity().value(),
                clean.view[N](0, n),
            )
            source = clean^.to_immutable()
            first = 0
        var src = source.view[N](first, n)
        var out = Buffer.alloc_zeroed[N](n)
        var dst = out.view[N](0, n)
        var bits = Bitmap.alloc_zeroed(n)
        for s in range(segments.count()):
            var lo = segments.start(s)
            var hi = segments.end(s)
            var edge = min(abs(offset), hi - lo)
            var count = hi - lo - edge
            var to = lo + edge if offset >= 0 else lo
            var start = lo if offset >= 0 else lo + edge
            if count > 0:
                dst.slice(to, count).copy_from(src.slice(start, count), count)
                if values.null_count() > 0:
                    bits.extend(
                        values.validity().value().slice(start, count),
                        to,
                        count,
                    )
                else:
                    bits.set_range(to, count, True)
        var nulls = bits.view(0, n).unset_count()
        var bitmap: Optional[Bitmap[mut=False]] = None
        if nulls > 0:
            bitmap = bits^.to_immutable(length=n)
        return PrimitiveArray[T](
            values.dtype.copy(),
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=bitmap,
            buffer=out^.to_immutable(),
        )

    @staticmethod
    def dispatch[
        S: Segments
    ](
        values: DynArray,
        offset: Int,
        segments: S,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """Row `i` of segment `s` is row `i - offset` when that row is in `s`,
        and null otherwise; rows in no segment are null. `apply` for a
        fixed-width dtype; any other layout is gathered by index, each row's
        index shifted the same way."""
        if values.dtype().is_primitive():

            def leaf[T: PrimitiveType](d: T) raises {imm} -> DynArray:
                return Self.apply(
                    values.as_primitive[T](), offset, segments, ctx
                ).to_dyn()

            return values.dtype().try_visit[DynArray, PrimitiveType](
                leaf, "dtype is not primitive"
            )
        else:
            var rows = arange[Int32Type](0, len(values))
            var indices = Self.apply(rows, offset, segments, ctx)
            return TakeKernel.dispatch(values, indices, ctx)


struct PairwiseDiffKernel(Kernel):
    """Each row minus the row `period` rows before it in its segment, in the
    input's dtype (integers wrap): `pairwise_diff`. A negative period reads
    ahead."""

    comptime name = "pairwise_diff"

    @staticmethod
    def apply[
        T: NumericType, S: Segments
    ](
        values: PrimitiveArray[T],
        period: Int,
        segments: S,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """Row `i` of segment `s` is `values[i] - values[i - period]` when both
        rows are in `s` and valid, and null otherwise: one subtraction of two
        offset views per segment."""
        comptime N = T.native
        var n = len(values)
        var src = values.values()
        var valid = values.validity()
        var diff = Buffer.alloc_zeroed[N](n)
        var dst = diff.view[N](0, n)
        var bits = Bitmap.alloc_zeroed(n)
        # The buffers are on the host whatever device `ctx` names.
        var host = ExecContext.serial() if ctx.is_gpu() else ctx.copy()
        for s in range(segments.count()):
            var lo = segments.start(s)
            var hi = segments.end(s)
            var edge = min(abs(period), hi - lo)
            var count = hi - lo - edge
            var at = lo + edge if period >= 0 else lo
            var before = lo if period >= 0 else lo + edge
            if count > 0:
                apply[N, N, SubKernel.core[N, _]](
                    src.slice(at, count),
                    src.slice(before, count),
                    dst.slice(at, count),
                    host,
                )
                if valid:
                    var both = valid.value().slice(
                        at, count
                    ) & valid.value().slice(before, count)
                    bits.extend(both.view(0, count), at, count)
                else:
                    bits.set_range(at, count, True)
        var nulls = bits.view(0, n).unset_count()
        var bitmap: Optional[Bitmap[mut=False]] = None
        var buffer: Buffer[mut=False]
        if valid:
            # A row beside a null was subtracted all the same: zero it.
            var out = Buffer.alloc_uninit[N](n)
            apply[N, N, valid_or_zero[N, _]](
                dst, bits.view(0, n), out.view[N](0, n)
            )
            buffer = out^.to_immutable()
        else:
            buffer = diff^.to_immutable()
        if nulls > 0:
            bitmap = bits^.to_immutable(length=n)
        return PrimitiveArray[T](
            values.dtype.copy(),
            length=n,
            nulls=nulls,
            offset=0,
            bitmap=bitmap,
            buffer=buffer,
        )

    @staticmethod
    def dispatch(
        values: DynArray,
        period: Int,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """`apply` over the whole column, for a numeric dtype."""

        def leaf[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                values.as_primitive[T](),
                period,
                BySize.whole(len(values)),
                ctx,
            ).to_dyn()

        return values.dtype().try_visit[DynArray, NumericType](
            leaf, "dtype is not numeric"
        )


# ---------------------------------------------------------------------------
# RollingKernel, RollingMeanKernel
# ---------------------------------------------------------------------------


def _check_frame[K: Kernel](lo: Int, hi: Int, min_samples: Int) raises:
    """The frame `ROWS BETWEEN lo AND hi` holds a row, and `min_samples` is
    one of its sizes."""
    if lo > hi:
        raise K.error[InvalidError](
            t"the frame [{lo}, {hi}] ends before it starts"
        )
    # In 128 bits: a frame may hold more rows than an `Int` counts.
    var rows = Int128(hi) - Int128(lo) + 1
    if min_samples < 1 or Int128(min_samples) > rows:
        raise K.error[InvalidError](
            t"min_samples must be between 1 and the frame's {rows}"
            t" rows, got {min_samples}"
        )


def _valid_counts[
    T: PrimitiveType, S: Segments
](
    values: PrimitiveArray[T], segments: S, var lo: Int, var hi: Int
) raises -> Buffer[mut=False]:
    """How many valid rows each row's frame holds, clipped to its segment."""
    comptime C = DType.int32
    var n = len(values)
    # An offset past `n` rows is the same frame as `n`, and keeps `i + lo`
    # and `i + hi` in range.
    lo = min(max(lo, -n), n)
    hi = min(max(hi, -n), n)
    var counts = Buffer.alloc_zeroed[C](n)
    var dst = counts.view[C](0, n)
    if values.null_count() == 0:
        for s in range(segments.count()):
            var s0 = segments.start(s)
            var s1 = segments.end(s)
            for i in range(s0, s1):
                var width = min(i + hi, s1 - 1) - max(i + lo, s0) + 1
                dst.store[1](i, Int32(max(width, 0)))
    else:
        var valid = values.validity().value()
        var ones = Buffer.alloc_uninit[C](n)
        var src = ones.view[C](0, n)

        @always_inline
        def one[W: Int](i: Int) {var valid} -> SIMD[C, W]:
            return valid.load[W](i).cast[C]()

        apply(src, one)
        RollingKernel[AddKernel].frames(src, dst, segments, lo, hi)
    return counts^.to_immutable()


def _at_least(
    counts: Buffer[mut=False], n: Int, min_samples: Int
) raises -> Tuple[Optional[Bitmap[mut=False]], Int]:
    """The rows of `n` whose count is at least `min_samples`, and how many are
    not: a validity bitmap, `None` when every row is."""
    comptime C = DType.int32
    var c = counts.view[C](0, n)
    var bits = Bitmap.alloc_zeroed(n)

    @always_inline
    def enough[W: Int](i: Int) {var c, var min_samples} -> SIMD[DType.bool, W]:
        return c.load[W](i).ge(SIMD[C, W](min_samples))

    apply[C](bits.view(0, n), enough)
    var nulls = bits.view(0, n).unset_count()
    var bitmap: Optional[Bitmap[mut=False]] = None
    if nulls > 0:
        bitmap = bits^.to_immutable(length=n)
    return (bitmap, nulls)


struct RollingKernel[K: BinaryKernel & Associative & Identity](Kernel):
    """`K` over each row's frame `ROWS BETWEEN lo AND hi`, clipped to its
    segment: `rolling_sum`, `rolling_min` and `rolling_max` with
    `K = AddKernel`, `MinKernel` and `MaxKernel`, in the input's dtype.

    A frame of at most eight rows is folded directly, several rows at a time:
    from its first row where it lies inside its segment, and from `K`'s
    identity where the segment clips it. A wider one is folded by van Herk and
    Gil-Werman, from a prefix and a suffix of fixed blocks, and so costs the
    same at every width. So a floating-point sum may round differently from a
    left-to-right fold, and a sum of negative zeros may answer `+0.0` on some
    rows and `-0.0` on others.
    """

    comptime name = "rolling"

    @staticmethod
    def frames[
        T: DType, S: Segments
    ](
        src: BufferView[T, _],
        dst: BufferView[mut=True, T, _],
        segments: S,
        var lo: Int,
        var hi: Int,
    ) raises InvalidError:
        """`dst[i]` is `K` over `src[l .. r]` for every row `i` of segment
        `s`, with `l = max(i + lo, start(s))` and `r = min(i + hi, end(s) - 1)`,
        and `K`'s identity where `l > r`. Rows in no segment are left as they
        are."""
        if lo > hi:
            raise Self.error[InvalidError](
                t"the frame [{lo}, {hi}] ends before it starts"
            )
        var n = len(src)
        # An offset past `n` rows reaches past every segment, so it is the
        # same frame as `n`, and keeps the width and every row index in range.
        lo = min(max(lo, -n), n)
        hi = min(max(hi, -n), n)
        comptime W = cpu_lanes[T]
        var w = hi - lo + 1
        var direct = w <= 8
        var identity = Self.K.identity[T]()
        # van Herk-Gil-Werman: the fold of each block up to each row, and
        # from each row to the block's end.
        var scratch = 0 if direct else n
        var prefix_buffer = Buffer.alloc_uninit[T](scratch)
        var suffix_buffer = Buffer.alloc_uninit[T](scratch)
        var prefix = prefix_buffer.view[T](0, scratch)
        var suffix = suffix_buffer.view[T](0, scratch)
        for s in range(segments.count()):
            var s0 = segments.start(s)
            var s1 = segments.end(s)
            if not direct and s1 > s0:
                var m = s1 - s0
                Scan[Self.K]().run(
                    src.slice(s0, m), prefix.slice(s0, m), BySize(m, w)
                )
                var block_end = s1
                var block = s0 + ((m - 1) // w) * w
                while block >= s0:
                    var j = block_end - 1
                    suffix.store[1](j, src.load[1](j))
                    while j > block:
                        j -= 1
                        suffix.store[1](
                            j,
                            Self.K.core[T, 1](
                                src.load[1](j), suffix.load[1](j + 1)
                            ),
                        )
                    block_end = block
                    block -= w
            # Interior rows, whose frame lies inside the segment.
            var first = min(max(s0, s0 - lo), s1)
            var last = max(min(s1, s1 - hi), first)
            var i = s0
            while i < s1:
                if i == first and first < last:
                    i = last
                else:
                    var l = max(i + lo, s0)
                    var r = min(i + hi, s1 - 1)
                    # An empty frame answers the identity.
                    var acc = identity
                    if l <= r:
                        if direct:
                            for j in range(l, r + 1):
                                acc = Self.K.core[T, 1](acc, src.load[1](j))
                        elif (l - s0) // w == (r - s0) // w:
                            # One block: the frame starts it, or is clipped
                            # at the segment's end, which ends it.
                            if (l - s0) % w == 0:
                                acc = prefix.load[1](r)
                            else:
                                acc = suffix.load[1](l)
                        else:
                            acc = Self.K.core[T, 1](
                                suffix.load[1](l), prefix.load[1](r)
                            )
                    dst.store[1](i, acc)
                    i += 1
            i = first
            if direct:
                while i < last:
                    var stop = i + W if i + W <= last else i + 1
                    if stop == i + W:
                        var acc = src.load[W](i + lo)
                        for d in range(1, w):
                            acc = Self.K.core[T, W](
                                acc, src.load[W](i + lo + d)
                            )
                        dst.store[W](i, acc)
                    else:
                        var acc = src.load[1](i + lo)
                        for d in range(1, w):
                            acc = Self.K.core[T, 1](
                                acc, src.load[1](i + lo + d)
                            )
                        dst.store[1](i, acc)
                    i = stop
            else:
                # Each frame is the suffix of one block and the prefix of the
                # next, or the whole block when it starts one.
                comptime Phase = DType.int64
                var lanes = iota[Phase, W]()
                while i < last:
                    var stop = i + W if i + W <= last else i + 1
                    if stop == i + W:
                        var starts = (
                            (lanes + Scalar[Phase](i + lo - s0))
                            % Scalar[Phase](w)
                        ).eq(0)
                        var p = prefix.load[W](i + hi)
                        dst.store[W](
                            i,
                            starts.select(
                                p, Self.K.core[T, W](suffix.load[W](i + lo), p)
                            ),
                        )
                    else:
                        var p = prefix.load[1](i + hi)
                        if (i + lo - s0) % w == 0:
                            dst.store[1](i, p)
                        else:
                            dst.store[1](
                                i, Self.K.core[T, 1](suffix.load[1](i + lo), p)
                            )
                    i = stop

    @staticmethod
    def apply[
        T: NumericType, S: Segments
    ](
        values: PrimitiveArray[T],
        lo: Int,
        hi: Int,
        min_samples: Int,
        segments: S,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> PrimitiveArray[T]:
        """`frames` over `values`, a null folding as `K`'s identity; a row is
        null when its clipped frame holds fewer than `min_samples` valid
        rows."""
        _check_frame[Self](lo, hi, min_samples)
        comptime N = T.native
        var n = len(values)
        # A `min_samples` past `n` is never met, and `n + 1` says so in the
        # 32-bit counts.
        var need = min(min_samples, n + 1)
        var folded = Buffer.alloc_uninit[N](n)
        var acc = folded.view[N](0, n)
        if values.null_count() == 0:
            Self.frames(values.values(), acc, segments, lo, hi)
        else:
            var data = values.values()
            var valid = values.validity().value()
            var identity = Self.K.identity[N]()
            var filled = Buffer.alloc_uninit[N](n)
            var src = filled.view[N](0, n)

            @always_inline
            def lane[
                W: Int
            ](i: Int) {var data, var valid, var identity} -> SIMD[N, W]:
                return valid.load[W](i).select(
                    data.load[W](i), SIMD[N, W](identity)
                )

            apply(src, lane)
            Self.frames(src, acc, segments, lo, hi)
        var validity = _at_least(
            _valid_counts(values, segments, lo, hi), n, need
        )
        var buffer: Buffer[mut=False]
        if validity[0]:
            var out = Buffer.alloc_uninit[N](n)
            apply[N, N, valid_or_zero[N, _]](
                acc, validity[0].value().view(0, n), out.view[N](0, n)
            )
            buffer = out^.to_immutable()
        else:
            buffer = folded^.to_immutable()
        return PrimitiveArray[T](
            values.dtype.copy(),
            length=n,
            nulls=validity[1],
            offset=0,
            bitmap=validity[0],
            buffer=buffer,
        )

    @staticmethod
    def dispatch(
        values: DynArray,
        lo: Int,
        hi: Int,
        min_samples: Int,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """`apply` over the whole column, for a numeric dtype."""

        def leaf[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                values.as_primitive[T](),
                lo,
                hi,
                min_samples,
                BySize.whole(len(values)),
                ctx,
            ).to_dyn()

        return values.dtype().try_visit[DynArray, NumericType](
            leaf, "dtype is not numeric"
        )


struct RollingMeanKernel(Kernel):
    """The mean of each row's frame `ROWS BETWEEN lo AND hi`, clipped to its
    segment, as float64: `rolling_mean`. Nulls are skipped, and a row is null
    when its frame holds fewer than `min_samples` valid rows."""

    comptime name = "rolling_mean"

    @staticmethod
    def apply[
        T: NumericType, S: Segments
    ](
        values: PrimitiveArray[T],
        lo: Int,
        hi: Int,
        min_samples: Int,
        segments: S,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> Float64Array:
        """The sum of each frame's valid values over their count."""
        _check_frame[Self](lo, hi, min_samples)
        comptime F = DType.float64
        var n = len(values)
        # A `min_samples` past `n` is never met, and `n + 1` says so in the
        # 32-bit counts.
        var need = min(min_samples, n + 1)
        var data = values.values()
        var cast = Buffer.alloc_uninit[F](n)
        var x = cast.view[F](0, n)
        if values.null_count() == 0:

            @always_inline
            def lane[W: Int](i: Int) {var data} -> SIMD[F, W]:
                return data.load[W](i).cast[F]()

            apply(x, lane)
        else:
            var valid = values.validity().value()

            @always_inline
            def masked[W: Int](i: Int) {var data, var valid} -> SIMD[F, W]:
                return valid.load[W](i).select(
                    data.load[W](i).cast[F](), SIMD[F, W](0)
                )

            apply(x, masked)
        var summed = Buffer.alloc_uninit[F](n)
        var sums = summed.view[F](0, n)
        RollingKernel[AddKernel].frames(x, sums, segments, lo, hi)
        var counted = _valid_counts(values, segments, lo, hi)
        var counts = counted.view[DType.int32](0, n)
        var out = Buffer.alloc_uninit[F](n)

        @always_inline
        def mean[W: Int](i: Int) {var sums, var counts, var need} -> SIMD[F, W]:
            var c = counts.load[W](i)
            return c.ge(SIMD[DType.int32, W](need)).select(
                sums.load[W](i) / c.cast[F](), SIMD[F, W](0)
            )

        apply(out.view[F](0, n), mean)
        var validity = _at_least(counted, n, need)
        return Float64Array(
            Float64Type(),
            length=n,
            nulls=validity[1],
            offset=0,
            bitmap=validity[0],
            buffer=out^.to_immutable(),
        )

    @staticmethod
    def dispatch(
        values: DynArray,
        lo: Int,
        hi: Int,
        min_samples: Int,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """`apply` over the whole column, for a numeric dtype."""

        def leaf[T: NumericType](d: T) raises {imm} -> DynArray:
            return Self.apply(
                values.as_primitive[T](),
                lo,
                hi,
                min_samples,
                BySize.whole(len(values)),
                ctx,
            ).to_dyn()

        return values.dtype().try_visit[DynArray, NumericType](
            leaf, "dtype is not numeric"
        )
