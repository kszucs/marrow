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
from ..builders import Float64Builder, Int32Builder, Int64Builder
from ..dtypes import DynType, float64, int64
from ..execution import ExecContext
from .core import Kernel
from .filter import TakeKernel
from .numeric import equal


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

    def __len__(self) -> Int:
        return len(self.partition_start)


def mark_changes(key: DynArray, mut flags: List[Bool], ctx: ExecContext) raises:
    """Set `flags[j]` where sorted `key` differs between rows `j-1` and `j`.

    ORs into `flags`, so a caller marks a whole key list by calling this once
    per column: a compound key changes wherever *any* of its columns does.

    **Null is not distinct from null here**, which is `IS NOT DISTINCT FROM`
    and not `=`. That is what `PARTITION BY` and `ORDER BY` both mean — the
    same rule `GROUP BY` uses, where all nulls land in one group — and it is
    why this cannot simply read `equal`'s output: `equal` propagates null, so
    a null-versus-null comparison answers *null* rather than true, and reading
    that as "not equal" would give every null row its own partition. The
    three-way split below is the whole correction, and it needs no per-dtype
    arm because validity is on `DynArray` and the value comparison is
    `equal`'s job.
    """
    var n = len(key)
    if n < 2:
        return
    var eq = equal(key.slice(1, n - 1), key.slice(0, n - 1), ctx)
    var values = eq.values()
    for j in range(1, n):
        if flags[j]:
            continue
        var cur = key.is_valid(j)
        var prev = key.is_valid(j - 1)
        if cur != prev:
            # Exactly one side is null: distinct, and `equal` said null.
            flags[j] = True
        elif cur:
            # Both non-null, so `equal` is non-null too and decides.
            if not values.test(j - 1):
                flags[j] = True
        # Both null: not distinct. Leave the flag as it was.


trait WindowFunction:
    """One window function, as a type.

    `argument` is the operand column, already evaluated by the caller — the
    ranking functions ignore it and take no argument at all.

    `name` and `ranks` are **methods rather than `comptime` members**: a
    `comptime name: T` requirement does not resolve reliably as `F.name` off
    an externally-bound parameter (CLAUDE.md, "Associated types"), and both are
    read exactly that way, through the `F` a verb names.
    """

    @staticmethod
    def name() -> String:
        """How this renders in a plan."""
        ...

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        """This function's output type, or `None` for "the argument's".

        **Also says whether the function takes an argument**, because the two
        are the same fact: a function answering from position alone has no
        operand to take a type from, so it must state one; a function reading a
        column answers in that column's type. All nine agree, and a separate
        flag would be a second place to get it wrong.

        It does have to be a type and not a flag: `percent_rank` and
        `cume_dist` read no column, like `row_number`, but answer `float64`
        rather than `int64`.
        """
        ...

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        ...


struct RowNumber(WindowFunction):
    """`ROW_NUMBER()` — position within the partition, ties broken by order."""

    @staticmethod
    def name() -> String:
        return String("row_number")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return int64.to_dyn()

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            out.append(Int64(j - extents.partition_start[j] + 1))
        return out.finish().to_dyn()


struct Rank(WindowFunction):
    """`RANK()` — the peer group's first position, so ties leave gaps."""

    @staticmethod
    def name() -> String:
        return String("rank")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return int64.to_dyn()

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            out.append(
                Int64(extents.peer_start[j] - extents.partition_start[j] + 1)
            )
        return out.finish().to_dyn()


struct DenseRank(WindowFunction):
    """`DENSE_RANK()` — the peer group's ordinal, so ties leave no gap."""

    @staticmethod
    def name() -> String:
        return String("dense_rank")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return int64.to_dyn()

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            out.append(Int64(extents.peer_ordinal[j] + 1))
        return out.finish().to_dyn()


struct Offset[lead: Bool](WindowFunction):
    """`LAG` and `LEAD` — the same gather, opposite directions.

    One body rather than two because `LAG(v, n)` *is* `LEAD(v, -n)`; the verb
    negates at construction and this parameter carries only which name the
    caller wrote, so a plan renders the function they asked for.
    """

    @staticmethod
    def name() -> String:
        return String("lead") if Self.lead else String("lag")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return None

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var idx = Int32Builder(len(extents))
        for j in range(len(extents)):
            var src = j + offset
            if (
                src < extents.partition_start[j]
                or src >= extents.partition_end[j]
            ):
                idx.append_null()
            else:
                idx.append(Int32(src))
        return TakeKernel.dispatch(argument.value().copy(), idx.finish(), ctx)


struct Edge[first: Bool](WindowFunction):
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
    def fixed_dtype() -> Optional[DynType]:
        return None

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var idx = Int32Builder(len(extents))
        for j in range(len(extents)):
            var start: Int
            var stop: Int
            if is_rows:
                start = max(extents.partition_start[j], j + preceding)
                stop = min(extents.partition_end[j], j + following + 1)
            else:
                start = extents.partition_start[j]
                stop = extents.peer_end[j]
            if stop <= start:
                idx.append_null()
            elif Self.first:
                idx.append(Int32(start))
            else:
                idx.append(Int32(stop - 1))
        return TakeKernel.dispatch(argument.value().copy(), idx.finish(), ctx)


comptime Lag = Offset[False]
comptime Lead = Offset[True]
comptime FirstValue = Edge[True]
comptime LastValue = Edge[False]


struct PercentRank(WindowFunction):
    """`PERCENT_RANK()` — `(rank - 1) / (rows - 1)`, `float64`."""

    @staticmethod
    def name() -> String:
        return String("percent_rank")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return float64.to_dyn()

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Float64Builder(len(extents))
        for j in range(len(extents)):
            var lo = extents.partition_start[j]
            var rows = extents.partition_end[j] - lo
            var rank = extents.peer_start[j] - lo
            out.append(0.0 if rows <= 1 else Float64(rank) / Float64(rows - 1))
        return out.finish().to_dyn()


struct CumeDist(WindowFunction):
    """`CUME_DIST()` — rows through this peer group over partition rows."""

    @staticmethod
    def name() -> String:
        return String("cume_dist")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return float64.to_dyn()

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        var out = Float64Builder(len(extents))
        for j in range(len(extents)):
            var lo = extents.partition_start[j]
            var rows = extents.partition_end[j] - lo
            out.append(Float64(extents.peer_end[j] - lo) / Float64(rows))
        return out.finish().to_dyn()


struct NTile(WindowFunction):
    """`NTILE(n)` — the partition in `n` buckets, the count carried in
    `offset`.

    The bucket count rides the `offset` slot rather than being an operand: it
    is a constant of the *window*, not a column, so there is nothing per-row
    for an argument to hold. `lag`'s distance uses that slot for the same
    reason.
    """

    @staticmethod
    def name() -> String:
        return String("ntile")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return int64.to_dyn()

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        if offset < 1:
            raise Error("ntile: bucket count must be positive, got ", offset)
        var out = Int64Builder(len(extents))
        for j in range(len(extents)):
            var lo = extents.partition_start[j]
            var rows = extents.partition_end[j] - lo
            var i = j - lo
            var small = rows // offset
            var extra = rows % offset
            if small == 0:
                out.append(Int64(i + 1))
            elif i < extra * (small + 1):
                out.append(Int64(i // (small + 1) + 1))
            else:
                var rest = i - extra * (small + 1)
                out.append(Int64(extra + rest // small + 1))
        return out.finish().to_dyn()


struct NthValue(WindowFunction):
    """`NTH_VALUE(v, n)` — the `n`-th row of the frame, `n` in `offset`.

    `FIRST_VALUE` is `NTH_VALUE(v, 1)` and answers through `Edge` instead,
    because a frame's first row is an edge the extents already name; an
    arbitrary `n` has to be counted from that edge.
    """

    @staticmethod
    def name() -> String:
        return String("nth_value")

    @staticmethod
    def fixed_dtype() -> Optional[DynType]:
        return None

    @staticmethod
    def compute(
        extents: WindowExtents,
        argument: Optional[DynArray],
        offset: Int,
        is_rows: Bool,
        preceding: Int,
        following: Int,
        ctx: ExecContext,
    ) raises -> DynArray:
        if offset < 1:
            raise Error("nth_value: n must be positive, got ", offset)
        var idx = Int32Builder(len(extents))
        for j in range(len(extents)):
            var lo: Int
            var hi: Int
            if is_rows:
                lo = max(extents.partition_start[j], j + preceding)
                hi = min(extents.partition_end[j], j + following + 1)
            else:
                lo = extents.partition_start[j]
                hi = extents.peer_end[j]
            var at = lo + offset - 1
            if at < hi:
                idx.append(Int32(at))
            else:
                idx.append_null()
        return TakeKernel.dispatch(argument.value().copy(), idx.finish(), ctx)
