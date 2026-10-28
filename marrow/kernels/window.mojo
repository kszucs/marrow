# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Window-function kernels — partitions and peer groups as segments, ranking,
and frame gathers.

A window function is evaluated over a *sorted* input: `PARTITION BY k ORDER BY
v` is one ordering, `[k..., v...]`, so the partitioning is a **prefix of the
sort key** rather than a second mechanism. `marrow/expr/physical.mojo` sorts
once with `SortIndices.multi` and hands the sorted key columns here; everything
below reads only the boundaries that ordering induces.

Two boundaries, and the distinction between them is the whole of tie handling:

- a **partition** boundary — the `PARTITION BY` prefix changes;
- a **peer** boundary — the `ORDER BY` suffix changes as well. Rows between two
  peer boundaries are *peers*: equal under the ordering, and therefore
  indistinguishable to `RANK` and to the default `RANGE` frame.

Both are runs of equal rows (`RunsKernel`), so `WindowSegments` holds them as
segments: one position per partition and one per peer group, not one per row.

`ROW_NUMBER` ignores peers, `RANK` numbers by the peer group's first position
and `DENSE_RANK` by the peer group's ordinal. Those three lines are the only
difference between the three ranking functions. The ranking and distribution
functions compute one value per peer group and broadcast it to the group's
rows (`PerSegment`).

`LAG` and `LEAD` are `ShiftKernel` over the partitions. The frame-edge
functions (`FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`) answer with an
`Int32Array` of *source row indices*, null where the frame holds no such row,
and gather with `take` — which maps a null index to a null element. That is
why this module needs no per-dtype arm at all: `shift` and `take` already have
one.
"""

from ..arrays import DynArray, Float64Array, Int32Array, Int64Array
from ..buffers import Buffer
from ..builders import Float64Builder, Int32Builder, Int64Builder
from ..dtypes import DynType, float64, int64
from ..execution import ExecContext
from ..errors import InvalidError
from ..views import BufferView, apply
from .core import Associative, BinaryKernel, Identity, Kernel
from .filter import TakeKernel
from .runs import RunsKernel
from .scan import RollingKernel, Scan, ShiftKernel
from .segments import ByEnds, PerSegment


struct WindowFrame(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Which rows of the partition an aggregate window function sees.

    Two forms, and the difference between them is the one thing about frames
    that reliably surprises:

    - the **default** — `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`,
      which SQL applies whenever a window has an `ORDER BY` and no explicit
      frame. `RANGE` counts *peers*, so "current row" means the end of the
      current row's peer group, and tied rows all see the same frame.
    - **`ROWS`**, which counts rows, so tied rows see different frames.

    The two agree only when the `ORDER BY` key has no duplicates, which is why
    `window_explicit_rows_frame` exists as a separate golden case from
    `window_partitioned_running_sum`.

    With no `ORDER BY` at all the default frame is the whole partition. That
    falls out here rather than being special-cased: with no order key every row
    is a peer of every other, so the peer group *is* the partition.

    A `ROWS` frame has a bounded start and folds through `RollingKernel`: a
    narrow frame lying inside its partition from its first row rather than
    from the identity, a wide one in fixed blocks. So a floating-point sum,
    mean or product may round differently from a left-to-right fold, and a sum
    of negative zeros may answer `+0.0` on some rows and `-0.0` on others.
    `RANGE` frames, which start `UNBOUNDED PRECEDING`, fold left to right from
    the identity, through `Scan`.
    """

    var is_rows: Bool
    """`True` for `ROWS`, `False` for the default `RANGE`."""

    var preceding: Int
    """`ROWS` only: rows before the current one, as a non-positive offset."""

    var following: Int
    """`ROWS` only: rows after the current one, as a non-negative offset."""

    def __init__(out self, is_rows: Bool, preceding: Int, following: Int):
        self.is_rows = is_rows
        self.preceding = preceding
        self.following = following

    @staticmethod
    def default() -> WindowFrame:
        """`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`."""
        return WindowFrame(False, 0, 0)

    def __eq__(self, other: Self) -> Bool:
        return (
            self.is_rows == other.is_rows
            and self.preceding == other.preceding
            and self.following == other.following
        )

    def write_to[W: Writer](self, mut writer: W):
        if self.is_rows:
            writer.write("rows ", self.preceding, "..", self.following)
        else:
            writer.write("range unbounded..current")


struct WindowSegments(Copyable, Movable, Sized):
    """A sorted batch cut into partitions, and each partition into peer
    groups.

    Three columns of positions, each O(segments) rather than O(rows), read
    as `ByEnds`: the partitions, the peer groups, and which peer groups each
    partition holds. Every partition end is a peer group end, because the
    `ORDER BY` keys are compared after the `PARTITION BY` ones.
    """

    var partition_ends: Int64Array
    """Where each partition starts, then the row count."""

    var peer_ends: Int64Array
    """Where each peer group starts, then the row count."""

    var partition_peers: Int64Array
    """Partition `p`'s peer groups are
    `[partition_peers[p], partition_peers[p + 1])`."""

    def __init__(
        out self, var partition_ends: Int64Array, var peer_ends: Int64Array
    ) raises:
        """The segments of partitions and peer groups at these positions;
        each partition's peer groups are found in one walk."""
        var parts = partition_ends.values()
        var peers = peer_ends.values()
        var first = Buffer.alloc_uninit[DType.int64](len(parts))
        var dst = first.view[DType.int64](0, len(parts))
        var g = 0
        for p in range(len(parts)):
            while g < len(peers) - 1 and peers[g] < parts[p]:
                g += 1
            debug_assert(
                peers[g] == parts[p],
                "WindowSegments: every partition end is a peer group end",
            )
            dst.store[1](p, Int64(g))
        self.partition_peers = Int64Array(
            length=len(parts), buffer=first^.to_immutable()
        )
        self.partition_ends = partition_ends^
        self.peer_ends = peer_ends^

    @staticmethod
    def of_sorted(
        sorted_keys: List[DynArray],
        partition_keys: Int,
        length: Int,
        ctx: ExecContext,
    ) raises -> WindowSegments:
        """The segments of `length` rows already sorted by `sorted_keys`,
        whose first `partition_keys` columns are the `PARTITION BY` keys and
        the rest the `ORDER BY` keys.

        A partition is a run of rows equal on every partition key, and a peer
        group a run equal on every key. With no key at all, the rows are one
        partition of one peer group.
        """
        var by = List[DynArray](capacity=partition_keys)
        for i in range(partition_keys):
            by.append(sorted_keys[i].copy())
        var partitions = RunsKernel.dispatch(by, length, ctx)
        var peers: Int64Array
        if partition_keys == len(sorted_keys):
            peers = partitions.copy()
        else:
            peers = RunsKernel.dispatch(sorted_keys, length, ctx)
        return WindowSegments(partitions^, peers^)

    def __len__(self) -> Int:
        """The rows: the last partition end."""
        return Int(self.partition_ends.values()[len(self.partition_ends) - 1])

    def partitions(
        self,
    ) -> ByEnds[DType.int64, origin_of(self.partition_ends.buffer)]:
        """The partitions, as segments."""
        return ByEnds(self.partition_ends.values())

    def peers(self) -> ByEnds[DType.int64, origin_of(self.peer_ends.buffer)]:
        """The peer groups, as segments."""
        return ByEnds(self.peer_ends.values())

    def frames(
        self, frame: WindowFrame
    ) raises -> Tuple[Int32Array, Int32Array]:
        """Each row's frame as a half-open `[start, stop)` of sorted rows.

        Under `ROWS` the offsets are clipped to the partition. Under `RANGE`
        the frame runs from the partition start to the end of the current
        row's peer group, not to the current row — which is why `LAST_VALUE`
        is not the partition's last value.

        An empty frame answers `start == stop`, never `start > stop`: a frame
        lying wholly past its partition (`rows=(5, 10)` on a 3-row partition)
        is pulled back to the partition end, so the range can be sliced as is.
        """
        comptime I = DType.int32
        var n = len(self)
        var starts = Buffer.alloc_uninit[I](n)
        var stops = Buffer.alloc_uninit[I](n)
        var lo = starts.view[I](0, n)
        var hi = stops.view[I](0, n)
        var parts = self.partitions()
        var peers = self.peers()
        var first = self.partition_peers.values()
        # An offset past `n` rows reaches past every partition, so it is the
        # same frame as `n`, and keeps `j + offset` in range.
        var preceding = min(max(frame.preceding, -n), n)
        var following = min(max(frame.following, -n), n)
        for p in range(parts.count()):
            var s0 = parts.start(p)
            var s1 = parts.end(p)
            if frame.is_rows:
                for j in range(s0, s1):
                    var start = min(max(s0, j + preceding), s1)
                    var stop = max(start, min(s1, j + following + 1))
                    lo.store[1](j, Int32(start))
                    hi.store[1](j, Int32(stop))
            else:
                for g in range(Int(first[p]), Int(first[p + 1])):
                    var stop = peers.end(g)
                    for j in range(peers.start(g), stop):
                        lo.store[1](j, Int32(s0))
                        hi.store[1](j, Int32(stop))
        return (
            Int32Array(length=n, buffer=starts^.to_immutable()),
            Int32Array(length=n, buffer=stops^.to_immutable()),
        )

    def fold[
        K: BinaryKernel & Associative & Identity, T: DType
    ](
        self,
        src: BufferView[T, _],
        dst: BufferView[mut=True, T, _],
        frame: WindowFrame,
    ) raises:
        """`dst[j]` is `K` over `src`'s rows in row `j`'s frame, and `K`'s
        identity where the frame is empty.

        A `ROWS` frame is a rolling window over the partitions. The `RANGE`
        frame is a running scan over the partitions, then each peer group's
        last row broadcast to the group, whose rows share a frame; with no
        `ORDER BY` the peer group is the partition, so every row sees all of
        it.
        """
        if frame.is_rows:
            if frame.preceding <= frame.following:
                RollingKernel[K].frames(
                    src,
                    dst,
                    self.partitions(),
                    frame.preceding,
                    frame.following,
                )
            else:
                # The frame ends before it starts, so it is empty for every row.
                var identity = K.identity[T]()

                @always_inline
                def empty[W: Int](i: Int) {var identity} -> SIMD[T, W]:
                    return SIMD[T, W](identity)

                apply(dst, empty)
        else:
            Scan[K]().run(src, dst, self.partitions())
            var peers = self.peers()
            var last = Buffer.alloc_uninit[T](peers.count())
            var lasts = last.view[T](0, peers.count())
            for g in range(peers.count()):
                lasts.store[1](g, dst.load[1](peers.end(g) - 1))
            PerSegment(last^.to_immutable()).broadcast(peers, dst)


trait WindowKernel(Copyable, Deinitable, Movable):
    """A non-aggregate window function: a ranking, distribution, offset or
    frame-edge kernel over a sorted partition.

    A kernel's parameters are its fields — `lag`'s offset, `ntile`'s bucket
    count, `nth_value`'s `n` — validated when it is built, so a plan cannot
    hold one that will fail at execution. A plan renders them after the
    operand, in field order.

    `argument` is the operand column, already evaluated over the sorted batch;
    a kernel that reads none (`reads_argument()` is `False`) is handed a
    column of nulls and ignores it.

    `name`, `reads_argument` and `dtype` are **static methods rather than
    `comptime` members**: a `comptime` requirement does not resolve reliably
    off an externally bound parameter (CLAUDE.md, "Associated types"), and all
    three are read through the `K` a `WindowCall` names.
    """

    @staticmethod
    def name() -> String:
        """How this renders in a plan."""
        ...

    @staticmethod
    def reads_argument() -> Bool:
        """Whether the kernel reads an operand column."""
        ...

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        """The output type, given the operand's — fixed for the kernels that
        read none, the operand's own for the gathers."""
        ...

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        """This kernel's column over the sorted batch the segments describe."""
        ...


@fieldwise_init
struct RowNumber(WindowKernel):
    """`ROW_NUMBER()` — position within the partition, ties broken by order."""

    @staticmethod
    def name() -> String:
        return String("row_number")

    @staticmethod
    def reads_argument() -> Bool:
        return False

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return int64.to_dyn()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(segments))
        var parts = segments.partitions()
        for p in range(parts.count()):
            var lo = parts.start(p)
            for j in range(lo, parts.end(p)):
                out.append(Int64(j - lo + 1))
        return out.finish().to_dyn()


@fieldwise_init
struct Rank(WindowKernel):
    """`RANK()` — the peer group's first position, so ties leave gaps."""

    @staticmethod
    def name() -> String:
        return String("rank")

    @staticmethod
    def reads_argument() -> Bool:
        return False

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return int64.to_dyn()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        comptime I = DType.int64
        var n = len(segments)
        var parts = segments.partitions()
        var peers = segments.peers()
        var first = segments.partition_peers.values()
        var ranks = Int64Builder(peers.count())
        for p in range(parts.count()):
            for g in range(Int(first[p]), Int(first[p + 1])):
                ranks.append(Int64(peers.start(g) - parts.start(p) + 1))
        var out = Buffer.alloc_uninit[I](n)
        PerSegment(ranks.finish().buffer).broadcast(peers, out.view[I](0, n))
        return Int64Array(length=n, buffer=out^.to_immutable()).to_dyn()


@fieldwise_init
struct DenseRank(WindowKernel):
    """`DENSE_RANK()` — the peer group's ordinal, so ties leave no gap."""

    @staticmethod
    def name() -> String:
        return String("dense_rank")

    @staticmethod
    def reads_argument() -> Bool:
        return False

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return int64.to_dyn()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        comptime I = DType.int64
        var n = len(segments)
        var parts = segments.partitions()
        var peers = segments.peers()
        var first = segments.partition_peers.values()
        var ranks = Int64Builder(peers.count())
        for p in range(parts.count()):
            for g in range(Int(first[p]), Int(first[p + 1])):
                ranks.append(Int64(g - Int(first[p]) + 1))
        var out = Buffer.alloc_uninit[I](n)
        PerSegment(ranks.finish().buffer).broadcast(peers, out.view[I](0, n))
        return Int64Array(length=n, buffer=out^.to_immutable()).to_dyn()


struct Offset[lead: Bool](WindowKernel):
    """`LAG(v, n)` and `LEAD(v, n)` — the same gather, opposite directions.

    One body rather than two because `LAG(v, n)` *is* `LEAD(v, -n)`; the
    parameter carries which one the caller wrote, so a plan renders the
    function they asked for with the offset they wrote.
    """

    var offset: Int
    """Rows to read away from the current one, in this kernel's direction."""

    def __init__(out self, offset: Int):
        self.offset = offset

    @staticmethod
    def name() -> String:
        return String("lead") if Self.lead else String("lag")

    @staticmethod
    def reads_argument() -> Bool:
        return True

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return argument.copy()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        # `shift` reads `offset` rows back; `lead` reads forward.
        var back = -self.offset if Self.lead else self.offset
        return ShiftKernel.dispatch(argument, back, segments.partitions(), ctx)


@fieldwise_init
struct Edge[first: Bool](WindowKernel):
    """`FIRST_VALUE` and `LAST_VALUE` — the two ends of the frame.

    The end is a **comptime parameter**, the shape `Pad[left]` uses in
    `kernels/string.mojo`: a runtime flag would put a per-row branch inside a
    gather that cannot vary within a call, and would link both readings into a
    binary that names one.
    """

    @staticmethod
    def name() -> String:
        return String("first_value") if Self.first else String("last_value")

    @staticmethod
    def reads_argument() -> Bool:
        return True

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return argument.copy()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var bounds = segments.frames(frame)
        var starts = bounds[0].values()
        var stops = bounds[1].values()
        var idx = Int32Builder(len(segments))
        for j in range(len(segments)):
            var start = Int(starts[j])
            var stop = Int(stops[j])
            if stop == start:
                idx.append_null()
            elif Self.first:
                idx.append(Int32(start))
            else:
                idx.append(Int32(stop - 1))
        return TakeKernel.dispatch(argument.copy(), idx.finish(), ctx)


comptime Lag = Offset[False]
comptime Lead = Offset[True]
comptime FirstValue = Edge[True]
comptime LastValue = Edge[False]


@fieldwise_init
struct PercentRank(WindowKernel):
    """`PERCENT_RANK()` — `(rank - 1) / (rows - 1)`, `float64`."""

    @staticmethod
    def name() -> String:
        return String("percent_rank")

    @staticmethod
    def reads_argument() -> Bool:
        return False

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return float64.to_dyn()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        comptime F = DType.float64
        var n = len(segments)
        var parts = segments.partitions()
        var peers = segments.peers()
        var first = segments.partition_peers.values()
        var ranks = Float64Builder(peers.count())
        for p in range(parts.count()):
            var lo = parts.start(p)
            var rows = parts.end(p) - lo
            for g in range(Int(first[p]), Int(first[p + 1])):
                var rank = peers.start(g) - lo
                ranks.append(
                    0.0 if rows <= 1 else Float64(rank) / Float64(rows - 1)
                )
        var out = Buffer.alloc_uninit[F](n)
        PerSegment(ranks.finish().buffer).broadcast(peers, out.view[F](0, n))
        return Float64Array(length=n, buffer=out^.to_immutable()).to_dyn()


@fieldwise_init
struct CumeDist(WindowKernel):
    """`CUME_DIST()` — rows through this peer group over partition rows."""

    @staticmethod
    def name() -> String:
        return String("cume_dist")

    @staticmethod
    def reads_argument() -> Bool:
        return False

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return float64.to_dyn()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        comptime F = DType.float64
        var n = len(segments)
        var parts = segments.partitions()
        var peers = segments.peers()
        var first = segments.partition_peers.values()
        var dists = Float64Builder(peers.count())
        for p in range(parts.count()):
            var lo = parts.start(p)
            var rows = parts.end(p) - lo
            for g in range(Int(first[p]), Int(first[p + 1])):
                dists.append(Float64(peers.end(g) - lo) / Float64(rows))
        var out = Buffer.alloc_uninit[F](n)
        PerSegment(dists.finish().buffer).broadcast(peers, out.view[F](0, n))
        return Float64Array(length=n, buffer=out^.to_immutable()).to_dyn()


struct NTile(WindowKernel):
    """`NTILE(n)` — the partition in `n` buckets."""

    var buckets: Int
    """How many buckets; positive, checked when the kernel is built."""

    def __init__(out self, buckets: Int) raises:
        if buckets < 1:
            raise InvalidError(
                t"ntile: bucket count must be positive, got {buckets}"
            )
        self.buckets = buckets

    @staticmethod
    def name() -> String:
        return String("ntile")

    @staticmethod
    def reads_argument() -> Bool:
        return False

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return int64.to_dyn()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(segments))
        var parts = segments.partitions()
        for p in range(parts.count()):
            var rows = parts.end(p) - parts.start(p)
            var small = rows // self.buckets
            var extra = rows % self.buckets
            for i in range(rows):
                if small == 0:
                    out.append(Int64(i + 1))
                elif i < extra * (small + 1):
                    out.append(Int64(i // (small + 1) + 1))
                else:
                    var rest = i - extra * (small + 1)
                    out.append(Int64(extra + rest // small + 1))
        return out.finish().to_dyn()


struct NthValue(WindowKernel):
    """`NTH_VALUE(v, n)` — the `n`-th row of the frame, 1-based.

    `FIRST_VALUE` is `NTH_VALUE(v, 1)` and answers through `Edge` instead,
    because a frame's first row is an edge `frames` already names; an
    arbitrary `n` has to be counted from that edge.
    """

    var n: Int
    """Which row of the frame, 1-based; positive, checked when built."""

    def __init__(out self, n: Int) raises:
        if n < 1:
            raise InvalidError(t"nth_value: n must be positive, got {n}")
        self.n = n

    @staticmethod
    def name() -> String:
        return String("nth_value")

    @staticmethod
    def reads_argument() -> Bool:
        return True

    @staticmethod
    def dtype(argument: DynType) -> DynType:
        return argument.copy()

    def compute(
        self,
        segments: WindowSegments,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var bounds = segments.frames(frame)
        var starts = bounds[0].values()
        var stops = bounds[1].values()
        var idx = Int32Builder(len(segments))
        for j in range(len(segments)):
            var lo = Int(starts[j])
            var hi = Int(stops[j])
            # Against the frame's row count, so a huge `n` cannot overflow
            # `lo + n`.
            if self.n <= hi - lo:
                idx.append(Int32(lo + self.n - 1))
            else:
                idx.append_null()
        return TakeKernel.dispatch(argument.copy(), idx.finish(), ctx)
