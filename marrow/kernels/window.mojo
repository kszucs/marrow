# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Window-function kernels — partition extents, ranking, and frame gathers.

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

`ROW_NUMBER` ignores peers, `RANK` numbers by the peer group's first position
and `DENSE_RANK` by the peer group's ordinal. Those three lines are the only
difference between the three ranking functions.

The offset and frame-edge functions (`LAG`, `LEAD`, `FIRST_VALUE`,
`LAST_VALUE`) allocate nothing of their own: each answers with an
`Int32Array` of *source row indices*, null where the row it would read falls
outside the partition, and the caller gathers with `take` — which maps a null
index to a null element. That is why this module needs no per-dtype arm at
all: `take` already has one.
"""

from ..arrays import (
    BoolArray,
    DynArray,
    Float64Array,
    Int32Array,
    Int64Array,
)
from ..buffers import Bitmap
from ..builders import Float64Builder, Int32Builder, Int64Builder, arange
from ..dtypes import DynType, Int32Type, float64, int64
from ..execution import ExecContext
from ..errors import InvalidError
from .core import Kernel
from .filter import TakeKernel
from .hashing import KeyCompare


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


struct WindowExtents(Copyable, Movable, Sized):
    """Where each row's partition and peer group begin and end.

    Four parallel arrays over the *sorted* row positions, all half-open
    (`start` inclusive, `end` exclusive). They are computed once per `Window`
    node and read by every window expression on it, because every one of them
    asks the same two questions and only differs in what it does with the
    answer.

    Stored as `List[Int]` rather than as Arrow arrays: this is operator
    bookkeeping, read scalar-wise by the per-row loops below and never handed
    to a kernel or to a user. Materialising four `Int32Array`s would buy
    nothing and cost four allocations plus offset arithmetic on every read.
    """

    var partition_start: List[Int]
    """First sorted row of this row's partition."""

    var partition_end: List[Int]
    """One past this row's partition's last sorted row."""

    var peer_start: List[Int]
    """First sorted row equal to this one under the `ORDER BY`."""

    var peer_end: List[Int]
    """One past this row's peer group's last sorted row."""

    var peer_ordinal: List[Int]
    """How many peer groups precede this one *within its partition*, 0-based.

    `DENSE_RANK` is this plus one. Kept here rather than recomputed because it
    is the one boundary fact that a backwards scan cannot recover in O(1) — it
    counts groups rather than naming a position.
    """

    def __init__(
        out self,
        var new_partition: List[Bool],
        var new_peer: List[Bool],
    ) raises:
        """Turn two boundary flags per row into the four extents.

        `new_partition[j]` says row `j` starts a new partition, `new_peer[j]`
        that it starts a new peer group. A partition boundary is always a peer
        boundary too — the caller guarantees it, since the `ORDER BY` keys are
        compared *after* the `PARTITION BY` ones and a changed prefix makes the
        whole key different.
        """
        var n = len(new_partition)
        self.partition_start = List[Int](length=n, fill=0)
        self.partition_end = List[Int](length=n, fill=0)
        self.peer_start = List[Int](length=n, fill=0)
        self.peer_end = List[Int](length=n, fill=0)
        self.peer_ordinal = List[Int](length=n, fill=0)

        # Forward scan: a start is carried down until the next boundary.
        var p_start = 0
        var g_start = 0
        var ordinal = 0
        for j in range(n):
            if new_partition[j]:
                p_start = j
                ordinal = 0
            elif new_peer[j]:
                ordinal += 1
            if new_peer[j]:
                g_start = j
            self.partition_start[j] = p_start
            self.peer_start[j] = g_start
            self.peer_ordinal[j] = ordinal

        # Backward scan: an end is carried up from the row after the last one.
        var p_end = n
        var g_end = n
        for j in reversed(range(n)):
            if j + 1 < n and new_partition[j + 1]:
                p_end = j + 1
            if j + 1 < n and new_peer[j + 1]:
                g_end = j + 1
            self.partition_end[j] = p_end
            self.peer_end[j] = g_end

    @staticmethod
    def of_sorted(
        sorted_keys: List[DynArray], partition_keys: Int, ctx: ExecContext
    ) raises -> WindowExtents:
        """The extents of a batch already sorted by `sorted_keys`, whose first
        `partition_keys` columns are the `PARTITION BY` keys and the rest the
        `ORDER BY` keys.

        A row starts a partition where any partition key changes, and a peer
        group where any key at all does — the `ORDER BY` keys are compared
        after the `PARTITION BY` ones, so a partition boundary is always a peer
        boundary too.
        """
        var n = len(sorted_keys[0]) if len(sorted_keys) > 0 else 0
        var new_partition = List[Bool](length=n, fill=False)
        if n > 0:
            new_partition[0] = True
        for i in range(partition_keys):
            Self._mark_changes(sorted_keys[i], new_partition, ctx)
        var new_peer = new_partition.copy()
        for i in range(partition_keys, len(sorted_keys)):
            Self._mark_changes(sorted_keys[i], new_peer, ctx)
        return WindowExtents(new_partition^, new_peer^)

    @staticmethod
    def _mark_changes(
        key: DynArray, mut flags: List[Bool], ctx: ExecContext
    ) raises:
        """Set `flags[j]` where sorted `key` differs between rows `j-1` and
        `j`.

        ORs into `flags`, so a compound key changes wherever *any* of its
        columns does. "Differs" is `IS DISTINCT FROM`, not `=`: neither null
        nor NaN is distinct from itself here. That is what `PARTITION BY` and
        `ORDER BY` both mean — the same key identity `GROUP BY` groups by — so
        it is the same kernel, `KeyCompare`, comparing each row with the one
        before it. Nested keys compare structurally for the same reason.
        """
        var n = len(key)
        if n >= 2:
            var same = Bitmap.alloc_zeroed(n - 1)
            same.set_range(0, n - 1, True)
            var rows = arange[Int32Type](0, n - 1)
            KeyCompare.apply(
                key.slice(1, n - 1), rows, key.slice(0, n - 1), rows, same, ctx
            )
            for j in range(1, n):
                if not same.test(j - 1):
                    flags[j] = True

    def __len__(self) -> Int:
        return len(self.partition_start)

    def frame(self, j: Int, frame: WindowFrame) -> Tuple[Int, Int]:
        """Row `j`'s frame as a half-open `[start, stop)` of sorted rows.

        Under `ROWS` the offsets are clipped to the partition. Under `RANGE`
        the frame runs from the partition start to the end of the current
        row's peer group, not to the current row — which is why `LAST_VALUE`
        is not the partition's last value.

        An empty frame answers `start == stop`, never `start > stop`: a frame
        lying wholly past its partition (`rows=(5, 10)` on a 3-row partition)
        is pulled back to the partition end, so the range can be sliced as is.
        """
        if frame.is_rows:
            var start = min(
                max(self.partition_start[j], j + frame.preceding),
                self.partition_end[j],
            )
            var stop = max(
                start, min(self.partition_end[j], j + frame.following + 1)
            )
            return (start, stop)
        return (self.partition_start[j], self.peer_end[j])


trait WindowKernel(Copyable, Deinitable, Movable):
    """A non-aggregate window function: a ranking, distribution, offset or
    frame-edge kernel over a sorted partition.

    A kernel carries its own parameters as fields — `lag`'s offset,
    `ntile`'s bucket count, `nth_value`'s `n` — validated when it is built, so
    a plan cannot hold one that will fail at execution.

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

    def params(self) -> List[Int]:
        """The parameters a plan renders after the operand, e.g. `[2]` for
        `lag(v, 2)`."""
        ...

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        """This kernel's column over the sorted batch the extents describe."""
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

    def params(self) -> List[Int]:
        return List[Int]()

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            out.append(Int64(j - extents.partition_start[j] + 1))
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

    def params(self) -> List[Int]:
        return List[Int]()

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            out.append(
                Int64(extents.peer_start[j] - extents.partition_start[j] + 1)
            )
        return out.finish().to_dyn()


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

    def params(self) -> List[Int]:
        return List[Int]()

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            out.append(Int64(extents.peer_ordinal[j] + 1))
        return out.finish().to_dyn()


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

    def params(self) -> List[Int]:
        return [self.offset]

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var step = self.offset if Self.lead else -self.offset
        var idx = Int32Builder(len(extents))
        for j in range(len(extents)):
            var src = j + step
            if (
                src < extents.partition_start[j]
                or src >= extents.partition_end[j]
            ):
                idx.append_null()
            else:
                idx.append(Int32(src))
        return TakeKernel.dispatch(argument.copy(), idx.finish(), ctx)


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

    def params(self) -> List[Int]:
        return List[Int]()

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var idx = Int32Builder(len(extents))
        for j in range(len(extents)):
            var start, stop = extents.frame(j, frame)
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

    def params(self) -> List[Int]:
        return List[Int]()

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Float64Builder(len(extents))
        for j in range(len(extents)):
            var lo = extents.partition_start[j]
            var rows = extents.partition_end[j] - lo
            var rank = extents.peer_start[j] - lo
            out.append(0.0 if rows <= 1 else Float64(rank) / Float64(rows - 1))
        return out.finish().to_dyn()


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

    def params(self) -> List[Int]:
        return List[Int]()

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Float64Builder(len(extents))
        for j in range(len(extents)):
            var lo = extents.partition_start[j]
            var rows = extents.partition_end[j] - lo
            out.append(Float64(extents.peer_end[j] - lo) / Float64(rows))
        return out.finish().to_dyn()


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

    def params(self) -> List[Int]:
        return [self.buckets]

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            var lo = extents.partition_start[j]
            var rows = extents.partition_end[j] - lo
            var i = j - lo
            var small = rows // self.buckets
            var extra = rows % self.buckets
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
    because a frame's first row is an edge the extents already name; an
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

    def params(self) -> List[Int]:
        return [self.n]

    def compute(
        self,
        extents: WindowExtents,
        argument: DynArray,
        frame: WindowFrame,
        ctx: ExecContext,
    ) raises -> DynArray:
        var idx = Int32Builder(len(extents))
        for j in range(len(extents)):
            var lo, hi = extents.frame(j, frame)
            var at = lo + self.n - 1
            if at < hi:
                idx.append(Int32(at))
            else:
                idx.append_null()
        return TakeKernel.dispatch(argument.copy(), idx.finish(), ctx)
