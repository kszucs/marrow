# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Order-dependent kernels: each row's answer depends on the rows before it in
its segment.

`Scan[K]` is the running fold of a binary kernel, one register at a time, and
restarts at every segment; `SumScan` is `Scan[AddKernel]`.
`CumulativeKernel[K]` is `cumulative_sum`, `cumulative_prod`,
`cumulative_min` and `cumulative_max` over it.

Every kernel here takes `Segments` and computes within each segment, so a
column cut into partitions is one call. Over a whole column the segments are
`BySize.whole(n)`.
"""

from std.math import align_up

from ..arrays import DynArray, PrimitiveArray
from ..buffers import Bitmap, Buffer
from ..dtypes import NumericType
from ..execution import ExecContext
from ..views import BufferView, apply, cpu_lanes, valid_or_zero
from .core import BinaryKernel, Identity, Kernel
from .numeric import AddKernel
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
