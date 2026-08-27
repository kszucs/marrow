"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouper`` hashes the key columns and resolves every row
     to a dense group index, storing the unique key rows.
  2. **Phase 2** — aggregate accumulation, layered on top by the caller through
     an ``AggKernel`` (``aggregate.mojo``).

The grouper itself is **aggregate-agnostic**: aggregates are ``AggKernel``
types, and mapping a runtime function *name* onto one lives in the expression
layer (``marrow/expr``). The ``GroupBy`` type below ties the two together:
``aggregate[A]`` runs one aggregate, ``aggregate_all`` runs a whole
``AggregateSet``, and the serial / thread-local / radix execution strategy is
picked from row count + cardinality.
"""

from max.algorithm.functional import sync_parallelize

from ..arrays import (
    StructArray,
    DynArray,
    UInt64Array,
    Int32Array,
    Int64Array,
)
from ..builders import DynBuilder, Int32Builder
from ..dtypes import DynType, Field, struct_
from .core import Groups
from .hashtable import SwissHashTable
from .partition import RadixPartitioner
from .hashing import RapidHashKernel
from ..utils import RapidHash64
from ..execution import ExecContext
from .filter import Take, take
from .concat import concat
from .aggregate import AggKernel


# ---------------------------------------------------------------------------
# HashGrouper — keys-only hash grouping
# ---------------------------------------------------------------------------


struct HashGrouper(Movable):
    """Keys-only hash grouper (ClickHouse-style, ``SwissHashTable``-backed).

    ``consume_keys`` hashes a batch of key rows, returns their dense group ids,
    and appends newly-seen key rows to a per-column builder. Call it repeatedly
    to accumulate groups across batches. NULL keys are treated as equal (same
    group), matching SQL GROUP BY semantics (unlike join, where NULL != NULL).

    Aggregate state is owned by the caller, not the grouper — see ``GroupBy``
    (typed/AOT path) and the expression layer (runtime path).
    """

    var _table: SwissHashTable[RapidHash64]
    var _key_builders: List[DynBuilder]

    def __init__(out self):
        self._table = SwissHashTable[RapidHash64]()
        self._key_builders = List[DynBuilder]()

    def num_groups(self) -> Int:
        return self._table.num_keys()

    def consume_hashes(
        mut self, hashes: UInt64Array, grow_adaptively: Bool = True
    ) raises -> Tuple[Int32Array, Int32Array]:
        """Resolve a batch of key hashes to dense group ids, and report where
        each *newly created* group first appeared.

        The core both grouping paths share. Returns ``(group ids per row, the
        rows — indices into this batch — at which the new groups first showed
        up)``. What a caller does with those rows is what distinguishes them:
        ``consume_keys`` gathers the key values immediately, because it groups
        batch after batch; a one-shot partitioned grouping keeps them and
        gathers every partition's keys in a single pass at the end.

        Bucket ids are dense and assigned in row order, so first occurrences
        appear in increasing id order — one forward scan collects them all and
        stops as soon as the last new group is found (near-instant when the
        groups all appear early, which is the low-cardinality case)."""
        var prev = self._table.num_keys()
        var bids = self._table.insert_hashes(
            hashes, grow_adaptively=grow_adaptively
        )
        var num_now = self._table.num_keys()

        var first_rows = Int32Builder(capacity=num_now - prev, zeroed=False)
        var next_new = prev
        if num_now > prev:
            for i in range(len(bids)):
                if Int(bids.unsafe_get(i)) == next_new:
                    first_rows.unsafe_append(Int32(i))
                    next_new += 1
                    if next_new == num_now:
                        break
        return (bids^, first_rows.finish())

    def consume_keys(
        mut self, keys: StructArray, hashes: Optional[UInt64Array] = None
    ) raises -> Int32Array:
        """Hash keys and resolve group indices, materializing the unique key
        rows as it goes. Returns the per-row group ids.

        New keys get new (dense, contiguous) group ids; existing keys return
        their previous id. Safe to call across multiple batches — this is the
        incremental entry point, which is why it gathers each batch's new key
        rows straight away rather than deferring. Pass ``hashes`` when the
        caller already computed them to skip the re-hash."""
        var n = len(keys)
        if n == 0:
            var empty = Int32Builder(0)
            return empty.finish()

        var batch_hashes = (
            hashes.value().copy() if hashes else RapidHashKernel.apply(keys)
        )
        var grouped = self.consume_hashes(batch_hashes)
        if len(grouped[1]) > 0:
            self._register_new_groups(keys, grouped[1])
        return grouped[0].copy()

    @staticmethod
    def key_fields(keys: StructArray) -> List[Field]:
        """The key columns' fields, read off the keys struct's own dtype."""
        var fields = List[Field]()
        ref st = keys.dtype.as_struct()
        for k in range(len(st.fields)):
            fields.append(Field(st.fields[k].name, st.fields[k].dtype.copy()))
        return fields^

    def key_columns(mut self, key_fields: List[Field]) raises -> List[DynArray]:
        """The unique group-key columns (empty arrays when no groups yet).
        Finishes the per-column key builders — call once, at emit time."""
        var cols = List[DynArray]()
        for k in range(len(key_fields)):
            if len(self._key_builders) == 0:
                var empty = DynBuilder(key_fields[k].dtype)
                cols.append(empty.finish())
            else:
                cols.append(self._key_builders[k].finish())
        return cols^

    def _register_new_groups(
        mut self, keys: StructArray, rows: Int32Array
    ) raises:
        """Append the key rows for newly-created groups to the per-column
        builders. Gathers all new rows in one ``take`` + one bulk ``extend`` per
        column instead of a slice/extend per group."""
        if len(self._key_builders) == 0:
            for k in range(len(keys.children)):
                self._key_builders.append(DynBuilder(keys.children[k].dtype()))
        var gathered = Take.apply(keys, rows)
        for k in range(len(keys.children)):
            self._key_builders[k].extend(gathered.children[k])


# ---------------------------------------------------------------------------
# Grouping — the placement axis
# ---------------------------------------------------------------------------
trait Grouping(Deinitable, Movable):
    """Which slot does a row contribute to?

    The *strategy*; `Groups` is the assignment it produces. One of the four
    axes an aggregation composes from — algebra x input x placement x emission
    — and the one that decides whether a fold scatters at all.

    A trait rather than a flag, so window partitions and a sorted or radix
    placement arrive as **conformers** rather than as further branches inside
    `GroupBy`. Placement is a comptime type parameter of a fold, so the loop a
    placement implies is chosen when the plan is built, not per batch.

    Takes **already-evaluated key columns**, never a `RecordBatch`: `kernels`
    must not depend on the expression layer, and evaluating a key expression is
    the caller's job. That is also what lets one grouping serve every aggregate
    in a query — the keys are hashed once, not once per aggregate.
    """

    comptime scatters: Bool
    """Whether a fold must scatter into per-slot accumulators.

    False for a single implicit slot, which lets a fold accumulate in registers
    and reduce once at the end. Not a micro-optimisation: scattering at one
    group measured **14.6x** worse than the register fold, which is the whole
    reason `ScalarGrouping` is its own conformer rather than `HashGrouping`
    with one key.
    """

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        """Place this batch's rows, extending the grouping with any new slots.

        Ids are dense and stable across calls, so an accumulator that folded an
        earlier batch keeps its slots when a later one introduces new groups.
        """
        ...

    def num_groups(self) -> Int:
        """How many slots exist so far — the size of a per-slot accumulator."""
        ...

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        """One column per key field, one row per slot.

        Empty when there are no keys. Call once, at emit time — it finishes the
        key builders.
        """
        ...


struct ScalarGrouping(Grouping):
    """One slot for every row — `SELECT sum(x) FROM t`, with no `GROUP BY`.

    Holds nothing and allocates nothing per batch. In particular it does **not**
    build a zero vector to say "every row is group 0": a fold whose `scatters`
    is False never reads the ids, and materialising one `Int32` per row to
    communicate a constant is exactly the cost this conformer exists to avoid.
    """

    comptime scatters = False

    def __init__(out self):
        pass

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        return Groups.single(num_rows)

    def num_groups(self) -> Int:
        return 1

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        return List[DynArray]()


struct HashGrouping(Grouping):
    """Dense ids from a keys-only hash table, accumulated across batches.

    Wraps `HashGrouper`, which already owns the hashing, the dense-id
    assignment and the unique-key materialisation. This adds the `Grouping`
    surface over it so a fold can be parameterised on placement.
    """

    comptime scatters = True

    var _grouper: HashGrouper

    def __init__(out self):
        self._grouper = HashGrouper()

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        """Hash the key columns and resolve dense ids.

        The field names built here are positional and deliberately arbitrary —
        `HashGrouper` keys on the *values*, and the caller supplies the real
        fields at `key_columns` time, where they reach the output schema.
        """
        var fields = List[Field](capacity=len(keys))
        for i in range(len(keys)):
            fields.append(Field("k" + String(i), keys[i].dtype()))
        var st = StructArray(
            dtype=struct_(fields^),
            length=num_rows,
            nulls=0,
            offset=0,
            bitmap=None,
            children=keys.copy(),
        )
        var ids = self._grouper.consume_keys(st)
        return Groups(ids^, self._grouper.num_groups())

    def num_groups(self) -> Int:
        return self._grouper.num_groups()

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        return self._grouper.key_columns(fields)


trait AggregateSet(Copyable, Deinitable, Movable):
    """What to compute per value column, for a grouping this layer drives.

    The grouper knows how to split rows and resolve them to group ids; it does
    not know what an aggregate is. A caller with a *runtime* set of aggregates
    (N different ones, chosen when the query was built) implements this and
    hands it over, so the choice of strategy — including the thread-local fold,
    which needs the partial state below — stays where the strategy is chosen.

    ``mergeable`` is the caller's answer to "can every column be folded per
    thread and merged?"; the grouper will not pick a strategy it cannot run."""

    def num_columns(self) -> Int:
        ...

    def mergeable(self) -> Bool:
        """Whether every column implements ``partials``/``merge``."""
        ...

    def grouped(
        self, column: Int, groups: Groups, values: DynArray
    ) raises -> DynArray:
        """Aggregate one value column over precomputed group ids."""
        ...

    def partials(
        self, column: Int, groups: Groups, values: DynArray
    ) raises -> Tuple[DynArray, Int64Array]:
        """One thread's raw per-group accumulator + valid counts."""
        ...

    def merge(
        self,
        column: Int,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        """Fold every thread's partials at remapped group ids and finalize."""
        ...


struct OneAggregate[A: AggKernel](AggregateSet):
    """A single statically-known aggregate, as a one-column set.

    The one-aggregate entry point (`GroupBy.aggregate[A]`) and the N-aggregate
    one want the same three strategies, so they run the same driver; this is
    what lets a single `A` in. `A` stays comptime inside every method, so
    nothing is interpreted — the column index is the only thing that became a
    runtime value, and there is exactly one.

    It holds the input dtype because `merge` needs it and only ever sees
    accumulators: a widening fold loses the input type on the way out, so
    `sum(int32)` hands back an int64 column that cannot say what it was folded
    from."""

    var _in_dtype: DynType

    def __init__(out self, var in_dtype: DynType):
        self._in_dtype = in_dtype^

    def num_columns(self) -> Int:
        return 1

    def mergeable(self) -> Bool:
        return Self.A.mergeable

    def grouped(
        self, column: Int, groups: Groups, values: DynArray
    ) raises -> DynArray:
        return Self.A.grouped(groups, [values.copy()])

    def partials(
        self, column: Int, groups: Groups, values: DynArray
    ) raises -> Tuple[DynArray, Int64Array]:
        return Self.A.partials(self._in_dtype, groups, [values.copy()])

    def merge(
        self,
        column: Int,
        remap: List[Int32Array],
        accs: List[DynArray],
        cnts: List[Int64Array],
        num_groups: Int,
    ) raises -> DynArray:
        return Self.A.merge(self._in_dtype, remap, accs, cnts, num_groups)


struct ThreadPartials(Copyable, Movable):
    """One worker's contribution to a thread-local aggregation: the unique keys
    it saw, and the raw (non-finalized) accumulator + valid counts per column
    over those keys."""

    var keys: StructArray
    var accs: List[DynArray]
    var cnts: List[Int64Array]

    def __init__(
        out self,
        var keys: StructArray,
        var accs: List[DynArray],
        var cnts: List[Int64Array],
    ):
        self.keys = keys^
        self.accs = accs^
        self.cnts = cnts^


struct GroupedColumns(Copyable, Movable):
    """The result of a grouped aggregation: the unique key columns, and one
    column per aggregate over them.

    Columns, not a table — naming the outputs and assembling a schema is the
    caller's business, and the caller is the only one who knows what the
    aggregates were called."""

    var keys: List[DynArray]
    var aggregates: List[DynArray]

    def __init__(
        out self, var keys: List[DynArray], var aggregates: List[DynArray]
    ):
        self.keys = keys^
        self.aggregates = aggregates^

    def num_rows(self) -> Int:
        """The group count — every column has one row per group."""
        if len(self.keys) > 0:
            return self.keys[0].length()
        elif len(self.aggregates) > 0:
            return self.aggregates[0].length()
        else:
            return 0


# ---------------------------------------------------------------------------
# GroupBy — grouped aggregation over a fixed set of key columns.
# ---------------------------------------------------------------------------


comptime _PARALLEL_MIN_ROWS = 60_000
"""Below this row count the serial path always wins — partitioning + dispatch
overhead dominates even for high-cardinality input."""

comptime _PARALLEL_ALWAYS_ROWS = 200_000
"""At or above this the parallel path wins for *any* cardinality, so the
cardinality probe is skipped."""

comptime RADIX_BITS = 6
"""Radix fanout for the high-cardinality parallel path (2**6 = 64 partitions)."""

comptime _CARD_SAMPLE_ROWS = 4096
"""Rows sampled (strided) to estimate cardinality on the dispatch boundary."""


comptime GROUP_SERIAL: UInt8 = 0
comptime GROUP_THREAD_LOCAL: UInt8 = 1
comptime GROUP_RADIX: UInt8 = 2
"""Groups execution strategies — see `GroupBy` for what each trades off.

Public so a driver layered on top (the expression layer's runtime, multi-
aggregate group-by) can reuse the same strategy decision instead of making its
own. These name a *grouping* strategy; no aggregate identity is involved."""


struct GroupBy(Movable):
    """Grouped aggregation over a fixed set of key columns.

    Mirrors PyArrow's ``table.group_by(keys)``: build once from the key columns,
    then aggregate. ``aggregate[A]`` runs one statically-known ``AggKernel``;
    ``aggregate_all`` runs a whole ``AggregateSet``. ``aggregate_columns`` is
    the open-coded counterpart: it groups once and emits one column per value
    column through a caller-supplied *comptime* lane. No aggregate name or tag
    ever reaches this module — mapping one onto an ``AggKernel`` is the
    expression layer's job (``marrow.expr``).

    The execution **strategy** is picked once at construction — from the row
    count, the worker budget (``ctx``), and a cheap one-time cardinality estimate
    — and reused across ``aggregate`` calls, since what the strategy trades off
    is the *grouping* cost, which is independent of the value column:

    - **serial** — small inputs, and low-/mid-cardinality inputs below
      ``_PARALLEL_ALWAYS_ROWS`` (fewer, larger groups → the single-table scatter
      beats any parallel overhead).
    - **thread-local partial aggregation** (`_thread_local`) — large
      low-/mid-cardinality inputs. Every core aggregates an equal contiguous
      chunk into its own table, then a cheap serial merge folds the partials.
      Scales with cores regardless of how few groups there are (unlike radix,
      which can't use more threads than there are distinct keys).
    - **radix-partition-parallel** — high-cardinality inputs,
      where a key lands in one partition so groups never span threads and the
      thread-local merge would instead become an O(N) serial bottleneck.

    Serial and radix are the same code with partitioning off or on; only the
    thread-local path differs, because it splits by row range rather than by key
    and therefore has to merge.
    """

    var _keys: StructArray
    var _ctx: ExecContext
    """How this grouping executes — held whole rather than destructured to a
    worker count. It used to be a bare `_num_threads: Int`, which two internal
    sites then rebuilt into `ExecContext.parallel(n)`; both silently dropped the
    caller's GPU device, since that factory sets `device=None`. Same defect
    `HashJoin` fixed and documented at `join.mojo:338`."""
    var _strategy: UInt8

    def __init__(
        out self,
        keys: StructArray,
        ctx: ExecContext = ExecContext.auto(),
        strategy: Optional[UInt8] = None,
    ) raises:
        """Group by a struct of key columns (multi-key GROUP BY).

        ``strategy`` forces one of ``GROUP_SERIAL`` / ``GROUP_THREAD_LOCAL`` /
        ``GROUP_RADIX`` instead of picking from row count and cardinality —
        the escape hatch tests and benchmarks need to compare the paths against
        each other on the same input."""
        self._keys = keys.copy()
        self._ctx = ctx.copy()
        self._strategy = (
            strategy.value() if strategy else Self._choose_strategy(
                self._keys, self._ctx
            )
        )

    def __init__(
        out self,
        key: DynArray,
        ctx: ExecContext = ExecContext.auto(),
        strategy: Optional[UInt8] = None,
    ) raises:
        """Group by a single key column."""
        var children = List[DynArray]()
        children.append(key.copy())
        var kd = key.to_data()
        self = Self(
            StructArray(
                dtype=struct_(Field("key", kd.dtype.copy())),
                length=kd.length,
                nulls=kd.nulls,
                offset=kd.offset,
                bitmap=kd.bitmap,
                children=children^,
            ),
            ctx,
            strategy,
        )

    @staticmethod
    def _choose_strategy(keys: StructArray, ctx: ExecContext) raises -> UInt8:
        var n = len(keys)
        if not ctx.worth_parallel(n, _PARALLEL_MIN_ROWS):
            return GROUP_SERIAL
        var high_card = Self._is_high_cardinality(keys, n)
        if n < _PARALLEL_ALWAYS_ROWS and not high_card:
            return GROUP_SERIAL
        if high_card:
            return GROUP_RADIX
        return GROUP_THREAD_LOCAL

    def keys(self) -> StructArray:
        """The key columns this grouping was built from."""
        return self._keys.copy()

    def num_threads(self) -> Int:
        """The resolved worker budget."""
        return self._ctx.resolved_num_threads()

    def strategy(self) -> UInt8:
        """The grouping strategy chosen at construction (``GROUP_SERIAL`` /
        ``GROUP_THREAD_LOCAL`` / ``GROUP_RADIX``)."""
        return self._strategy

    def aggregate[A: AggKernel](self, value: DynArray) raises -> GroupedColumns:
        """Aggregate one ``value`` column per group with aggregate ``A``.

        Returns the unique key columns and the aggregate column. ``A`` is fixed
        at compile time, so the fold it wraps is monomorphized after one
        dispatch; the strategy choice is the shared one (`aggregate_all`),
        because there is no reason for a single aggregate to pick differently
        from a set of them.

        This used to be two methods — a typed one taking `A.InArray` and an
        erased `apply[F]` that resolved a *name* first. `AggKernel` takes an
        erased column already, and no kernel in this package turns a name into
        behaviour, so both collapsed into this.
        """
        var values = List[DynArray]()
        values.append(value.copy())
        return self.aggregate_all(OneAggregate[A](value.dtype()), values)

    @staticmethod
    def _is_high_cardinality(keys: StructArray, n: Int) raises -> Bool:
        """Estimate whether the key set is high-cardinality from a strided sample.

        Samples ``_CARD_SAMPLE_ROWS`` rows evenly spread across the input (strided,
        so sorted/clustered keys stay representative), hashes them into a throwaway
        table, and reports high cardinality when more than half the sampled rows are
        distinct. Cost is a few thousand hashes — negligible next to the group-by,
        and only paid in the ``[_PARALLEL_MIN_ROWS, _PARALLEL_ALWAYS_ROWS)`` band.
        """
        var s = min(_CARD_SAMPLE_ROWS, n)
        var stride = n // s
        var idx = Int32Builder(capacity=s, zeroed=False)
        for i in range(s):
            idx.unsafe_append(Int32(i * stride))
        var sample = Take.apply(keys, idx.finish())
        var table = SwissHashTable[RapidHash64]()
        _ = table.insert(sample, grow_adaptively=True)
        return table.num_keys() * 2 > s

    def aggregate_all[
        C: AggregateSet
    ](self, agg: C, values: List[DynArray]) raises -> GroupedColumns:
        """Group the keys once and apply ``agg`` to every value column.

        The multi-aggregate entry point, and the only one that reaches all three
        strategies: thread-local partial folds when the caller says every column
        is mergeable, otherwise the key-partitioned driver (serial being its
        single-partition case). Which one ran is not the caller's business."""
        if len(values) != agg.num_columns():
            raise Error(
                "aggregate_all: one value column per aggregate is required"
            )
        if self._strategy == GROUP_THREAD_LOCAL and agg.mergeable():
            return Self._thread_local_columns[C](
                self._keys, agg, values, self._ctx
            )

        def by_column(
            j: Int, groups: Groups, value: DynArray
        ) raises {imm} -> DynArray:
            return agg.grouped(j, groups, value)

        return self.aggregate_columns(values, by_column)

    @staticmethod
    def _thread_local_columns[
        C: AggregateSet
    ](
        keys: StructArray, agg: C, values: List[DynArray], ctx: ExecContext
    ) raises -> GroupedColumns:
        """Thread-local partial aggregation for N columns at once.

        Every worker groups an equal contiguous chunk *once* — the grouping is
        what the columns share — and folds each column into its own partial
        state; a serial merge then re-keys the chunks into a global grouper and
        folds the partials at the global ids. The multi-column counterpart of
        `_thread_local[A]`, and what lets a runtime aggregate set take this path
        at all."""
        var n = len(keys)
        var na = agg.num_columns()
        var num_threads = ctx.resolved_num_threads()
        var chunk = (n + num_threads - 1) // num_threads

        # Pre-sized per-thread slots — no races on list growth.
        var partials = List[Optional[ThreadPartials]](
            length=num_threads, fill=None
        )

        # One slot per worker, like the result slots above: a single shared
        # `Optional[Error]` would be written by every failing thread at once.
        var worker_errs = List[Optional[Error]](length=num_threads, fill=None)

        def worker(t: Int) {mut worker_errs, mut partials, imm}:
            # `sync_parallelize`'s value form takes a non-raising worker. The
            # body still unwinds at its first error; the other workers cannot be
            # cancelled, so their errors are collected and raised after the join.
            try:
                var start = t * chunk
                if start >= n:
                    return
                var length = min(chunk, n - start)
                var kchunk = keys.slice(start, length)

                var grouper = HashGrouper()
                var gids = grouper.consume_keys(kchunk)  # group this chunk ONCE
                var ng = grouper.num_groups()

                var kcols = grouper.key_columns(grouper.key_fields(kchunk))
                var accs = List[DynArray]()
                var cnts = List[Int64Array]()
                for j in range(na):
                    var parts = agg.partials(
                        j,
                        Groups(gids.copy(), ng),
                        values[j].slice(start, length),
                    )
                    accs.append(parts[0].copy())
                    cnts.append(parts[1].copy())

                partials[t] = ThreadPartials(
                    StructArray(
                        dtype=keys.dtype.copy(),
                        length=ng,
                        nulls=0,
                        offset=0,
                        bitmap=None,
                        children=kcols^,
                    ),
                    accs^,
                    cnts^,
                )

            except e:
                worker_errs[t] = e

        # Hand-rolled rather than `ctx.stripe`, and it has to stay that way:
        # this worker **raises** (it hashes keys inside the stripe), and
        # `stripe`'s body is typed non-raising. Widening it was tried and
        # reverted — see the note on `ExecContext.stripe`; the parameter
        # form of `sync_parallelize` that accepts a raising worker needs an
        # implicitly-capturing closure, which miscompiles there.
        sync_parallelize(worker, num_threads)
        for err in worker_errs:
            if err:
                raise err.value()

        # Merge — re-key every chunk into the global grouper ONCE (shared across
        # columns), then fold each column's partials at the global ids.
        var gg = HashGrouper()
        var live = List[Int]()
        var remap = List[Int32Array]()
        for t in range(num_threads):
            if partials[t]:
                live.append(t)
                remap.append(gg.consume_keys(partials[t].value().keys))
        var ngg = gg.num_groups()

        var key_cols = gg.key_columns(gg.key_fields(keys))
        var agg_cols = List[DynArray]()
        for j in range(na):
            var accs = List[DynArray]()
            var cnts = List[Int64Array]()
            for i in range(len(live)):
                ref part = partials[live[i]].value()
                accs.append(part.accs[j].copy())
                cnts.append(part.cnts[j].copy())
            agg_cols.append(agg.merge(j, remap, accs, cnts, ngg))

        return GroupedColumns(key_cols^, agg_cols^)

    def aggregate_columns[
        ColAgg: def(Int, Groups, DynArray) raises -> DynArray
    ](self, values: List[DynArray], col_agg: ColAgg) raises -> GroupedColumns:
        """Group the keys once, then emit ``col_agg(j, gids, values[j], ng)`` as
        output column ``j`` — the multi-aggregate driver.

        Never thread-local: the aggregator is opaque here, so its per-thread
        partial state can't be merged. (The thread-local *fold* path is
        ``aggregate[A]``, where the aggregation — and therefore its ``merge`` —
        is statically known.)"""
        return Self._by_partition(
            self._keys,
            values,
            col_agg,
            self._ctx,
            partition=self._strategy != GROUP_SERIAL,
        )

    @staticmethod
    def _by_partition[
        ColAgg: def(Int, Groups, DynArray) raises -> DynArray,
    ](
        keys: StructArray,
        values: List[DynArray],
        col_agg: ColAgg,
        outer: ExecContext,
        partition: Bool,
    ) raises -> GroupedColumns:
        """The grouped-aggregation driver: group and aggregate one partition at
        a time, then stitch the partitions back together.

        ``partition`` picks between the two ways to split the rows, which are
        the same algorithm over a different partition count:

        - **off** — one partition holding every row in order. Hashing is serial
          (there is nothing to overlap it with), rows are not renumbered, and
          value columns need no gather.
        - **on** — radix partitioning on the top ``RADIX_BITS`` of the key hash,
          in parallel. A key lands in exactly one partition, so partitions are
          key-disjoint, and each one's table is pre-sized to its row count (at
          the high cardinality this path is chosen for, most rows are distinct
          keys, so growing adaptively would rehash ~log2(groups) times).

        Either way a partition's groups are final, so combining is a
        concatenation and never a merge. Each partition records each new group's
        *first-occurrence original row* instead of materializing key columns, so
        the keys are gathered in a single ``take`` over ``num_groups`` rows at
        the end rather than ``N`` rows per partition."""
        # One partition means one thread: gathering keys and concatenating a
        # single chunk under a parallel context would pay thread dispatch for a
        # `num_groups`-row `take`, which at small inputs is most of the query.
        #
        # Both arms resolve the worker count rather than passing `outer`
        # through, which reproduces the `ExecContext.parallel(n)` / `serial()`
        # pair this replaced exactly — minus dropping the device. Passing a bare
        # `auto()` down instead would be a *different* policy: every downstream
        # `wants_parallel` would re-consult its own 32768-row threshold, where a
        # forced count bypasses it. That may well be the better policy, but it
        # is a measurable change to the group-by hot path and does not belong in
        # a plumbing fix.
        var ctx = outer.with_threads(
            outer.resolved_num_threads()
        ) if partition else outer.with_threads(1)
        var na = len(values)

        def group_partition(
            rows: Int32Array, part_hashes: UInt64Array
        ) raises {imm} -> Tuple[Int32Array, List[DynArray]]:
            var grouper = HashGrouper()
            var grouped = grouper.consume_hashes(
                part_hashes, grow_adaptively=not partition
            )
            ref gids = grouped[0]
            var ng = grouper.num_groups()

            # The scan reports rows within *this partition*; a partitioned one
            # holds a subset in a different order, so translate to original row
            # numbers. Unpartitioned rows already are their own row numbers.
            var first = grouped[1].copy()
            if partition:
                first = take(rows.copy().to_dyn(), first).as_int32().copy()

            var agg_cols = List[DynArray]()
            for j in range(na):
                # Values in partition order, aligned with `gids`. A single
                # partition *is* the whole input, already in order — no gather.
                if partition:
                    agg_cols.append(
                        col_agg(
                            j,
                            Groups(gids.copy(), ng),
                            take(values[j], rows),
                        )
                    )
                else:
                    agg_cols.append(
                        col_agg(j, Groups(gids.copy(), ng), values[j])
                    )
            return (first^, agg_cols^)

        var parts = List[Tuple[Int32Array, List[DynArray]]]()
        if partition:

            def radix_partition(
                _pi: Int, rows: Int32Array, part_hashes: UInt64Array
            ) raises {imm} -> Tuple[Int32Array, List[DynArray]]:
                return group_partition(rows, part_hashes)

            # `ctx` is already the forced-count context here — this branch only
            # runs when `partition` is true.
            parts = RadixPartitioner(
                num_bits=RADIX_BITS, ctx=ctx.copy()
            ).map_partitions[Tuple[Int32Array, List[DynArray]]](
                RapidHashKernel.apply(keys, ctx), radix_partition
            )
        else:
            var no_rows = Int32Builder(0)
            parts.append(
                group_partition(
                    no_rows.finish(), RapidHashKernel.apply(keys, ctx)
                )
            )

        # The global unique-key set is the union of the partitions': concatenate
        # their first-occurrence rows and gather the key columns once.
        var first_chunks = List[DynArray]()
        for i in range(len(parts)):
            first_chunks.append(parts[i][0].copy())
        var first_any = concat(first_chunks, ctx)
        ref first_rows = first_any.as_int32()

        var key_cols = List[DynArray]()
        for k in range(len(keys.children)):
            key_cols.append(
                take(
                    keys.children[k].slice(keys.offset, len(keys)),
                    first_rows,
                    ctx,
                )
            )

        var agg_cols = List[DynArray]()
        for j in range(na):
            var chunks = List[DynArray]()
            for i in range(len(parts)):
                chunks.append(parts[i][1][j].copy())
            agg_cols.append(concat(chunks, ctx))

        return GroupedColumns(key_cols^, agg_cols^)
