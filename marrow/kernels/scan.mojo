# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Order-dependent kernels: each row's answer depends on the rows before it in
its segment.

`Scan[K]` is the running fold of a binary kernel, one register at a time, and
restarts at every segment; `SumScan` is `Scan[AddKernel]`.
`CumulativeKernel[K]` is `cumulative_sum`, `cumulative_prod`,
`cumulative_min` and `cumulative_max` over it. `ShiftKernel` reads each row
some rows back (`shift`), and `PairwiseDiffKernel` subtracts that row
(`pairwise_diff`).

Every kernel here takes `Segments` and computes within each segment, so a
column cut into partitions is one call. Over a whole column the segments are
`BySize.whole(n)`.
"""

from std.math import align_up

from ..arrays import DynArray, PrimitiveArray
from ..buffers import Bitmap, Buffer
from ..builders import Int32Builder
from ..dtypes import NumericType, PrimitiveType
from ..execution import ExecContext
from ..views import BufferView, apply, cpu_lanes, valid_or_zero
from .core import BinaryKernel, Identity, Kernel
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
    def dispatch(
        values: DynArray,
        offset: Int,
        ctx: ExecContext = ExecContext.serial(),
    ) raises -> DynArray:
        """`apply` over the whole column for a fixed-width dtype; any other
        layout is gathered by index."""
        var n = len(values)
        if values.dtype().is_primitive():

            def leaf[T: PrimitiveType](d: T) raises {imm} -> DynArray:
                return Self.apply(
                    values.as_primitive[T](), offset, BySize.whole(n), ctx
                ).to_dyn()

            return values.dtype().try_visit[DynArray, PrimitiveType](
                leaf, "dtype is not primitive"
            )
        else:
            var indices = Int32Builder(n)
            for i in range(n):
                var j = i - offset
                if j >= 0 and j < n:
                    indices.append(Int32(j))
                else:
                    indices.append_null()
            return TakeKernel.dispatch(values, indices.finish(), ctx)


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
