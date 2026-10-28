# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Segments: a column cut into consecutive runs of rows, and folds over them.

A **segment** is a consecutive run of rows, `[start, end)`. `Segments` says
how a column is cut: `BySize` into runs of one size (the last one shorter),
`ByEnds` at the positions an offsets buffer lists. Both are views, cheap to
copy and owning nothing.

`Segmented[F].reduce` folds every segment of a column with a fold kernel,
one answer per segment. A list array's offsets are segments of its child
(`ListReduceKernel`), and so are fixed-size morsels of a batch, the column a
bound is computed over (`BoundsKernel`), and the partitions the order kernels
in `scan.mojo` restart at. `PerSegment` is the other direction: one value per
segment, broadcast to every row of it.
"""

from std.math import ceildiv

from ..arrays import PrimitiveArray
from ..buffers import Buffer
from ..dtypes import PrimitiveType
from ..views import BufferView, cpu_lanes
from .aggregate import AggState, FoldKernel


trait Segments(Copyable, Deinitable, Movable):
    """How a column is cut into consecutive runs of rows.

    Segment `s` is rows `[start(s), end(s))`, and the segments are in row
    order: `start(s) <= end(s) <= start(s + 1)`. They need not cover every
    row, and a segment may be empty.
    """

    def count(self) -> Int:
        """How many segments there are."""
        ...

    def start(self, s: Int) -> Int:
        """The first row of segment `s`."""
        ...

    def end(self, s: Int) -> Int:
        """One past the last row of segment `s`."""
        ...


struct BySize(Segments, TrivialRegisterPassable):
    """`length` rows cut into segments of `size` rows, the last one shorter
    when `size` does not divide `length`."""

    var length: Int
    var size: Int

    def __init__(out self, length: Int, size: Int):
        debug_assert(size >= 1, "BySize: a segment holds at least one row")
        self.length = length
        self.size = size

    @staticmethod
    def whole(length: Int) -> Self:
        """All `length` rows as one segment, and no segment when there is no
        row."""
        return Self(length, max(length, 1))

    def count(self) -> Int:
        return ceildiv(self.length, self.size)

    def start(self, s: Int) -> Int:
        return s * self.size

    def end(self, s: Int) -> Int:
        return min(self.start(s) + self.size, self.length)


struct ByEnds[mut: Bool, //, E: DType, origin: Origin[mut=mut]](
    Segments, TrivialRegisterPassable
):
    """Segments at the positions `ends` lists: segment `s` is
    `[ends[s], ends[s + 1])`.

    `ends` holds `count() + 1` non-decreasing positions, the first of which
    may exceed 0: a sliced list array's offsets are segments of its child as
    they stand.
    """

    var ends: BufferView[Self.E, Self.origin]

    def __init__(out self, ends: BufferView[Self.E, Self.origin]):
        debug_assert(len(ends) >= 1, "ByEnds: count + 1 positions")
        self.ends = ends

    def count(self) -> Int:
        return len(self.ends) - 1

    def start(self, s: Int) -> Int:
        return Int(self.ends.load[1](s))

    def end(self, s: Int) -> Int:
        return Int(self.ends.load[1](s + 1))


struct Segmented[F: FoldKernel]:
    """A fold kernel applied to every segment of a column."""

    @staticmethod
    def reduce[
        V: PrimitiveType, S: Segments
    ](values: PrimitiveArray[V], segments: S) raises -> PrimitiveArray[
        Self.F.AccType[V]
    ]:
        """One row per segment: `F` over the segment's valid values.

        Positions index `values` as it stands, its offset applied, and the
        segments need not cover it. A segment with no valid value answers
        NULL, or `F`'s answer for nothing when `F` has one (`count` is 0).
        """
        comptime A = Self.F.AccType[V].native
        var data = values.values()
        var state = AggState[Self.F, V](Self.F.acc_dtype[V](values.dtype))
        state.reserve(segments.count())

        @always_inline
        def lanes[W: Int](i: Int) {var data} -> SIMD[A, W]:
            return data.load[W](i).cast[A]()

        if values.validity():
            var valid = values.validity().value()
            var identity = Self.F.identity[A]()

            # A null slot's payload is arbitrary, so it folds as the identity.
            @always_inline
            def masked[
                W: Int
            ](i: Int) {var valid, var identity, imm lanes} -> SIMD[A, W]:
                return valid.load[W](i).select(
                    lanes[W](i), SIMD[A, W](identity)
                )

            for s in range(segments.count()):
                var lo = segments.start(s)
                var hi = segments.end(s)
                state.combine_reserved(
                    s,
                    Self._fold[A](masked, lo, hi),
                    valid.slice(lo, hi - lo).count_set_bits(),
                )
        else:
            for s in range(segments.count()):
                var lo = segments.start(s)
                var hi = segments.end(s)
                state.combine_reserved(s, Self._fold[A](lanes, lo, hi), hi - lo)
        return state.finish()

    @always_inline
    @staticmethod
    def _fold[
        A: DType, L: def[W: Int](Int) -> SIMD[A, W]
    ](lanes: L, lo: Int, hi: Int) -> Scalar[A]:
        """Rows `[lo, hi)` of `lanes`: whole registers into one accumulator
        seeded with the identity, folded across its lanes, then the tail."""
        comptime W = cpu_lanes[A]
        var acc = SIMD[A, W](Self.F.identity[A]())
        var i = lo
        while i + W <= hi:
            acc = Self.F.combine[A, W](acc, lanes[W](i))
            i += W
        var folded = Self.F.fold_lanes[A, W](acc)
        comptime if W > 1:
            while i < hi:
                folded = Self.F.combine[A, 1](folded, lanes[1](i))
                i += 1
        return folded


struct PerSegment(Movable):
    """One value per segment, read at a dtype."""

    var values: Buffer[mut=False]
    """Segment `s`'s value at position `s`."""

    def __init__(out self, var values: Buffer[mut=False]):
        self.values = values^

    def broadcast[
        T: DType, S: Segments
    ](self, segments: S, dst: BufferView[mut=True, T, _]):
        """`dst[i]` is segment `s`'s value for every row `i` of `s`. Rows in
        no segment are left as they are."""
        var count = segments.count()
        debug_assert(
            self.values.length[T]() >= count,
            "PerSegment.broadcast: one value per segment",
        )
        comptime W = cpu_lanes[T]
        var src = self.values.view[T](0, count)
        for s in range(count):
            var value = src.load[1](s)
            var i = segments.start(s)
            var end = segments.end(s)
            while i + W <= end:
                dst.store[W](i, SIMD[T, W](value))
                i += W
            while i < end:
                dst.store[1](i, value)
                i += 1
