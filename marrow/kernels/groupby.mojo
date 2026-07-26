"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouper`` hashes the key columns and resolves every row
     to a dense group index, storing the unique key rows.
  2. **Phase 2** — aggregate accumulation, layered on top by the caller through
     an ``Aggregation`` (``aggregate.mojo``).

The grouper itself is **aggregate-agnostic**: aggregates are ``Aggregation``
types — a kernel already bound to its input type — and mapping a runtime
function *name* onto one lives in the expression layer (``marrow/expr``). The
``GroupBy`` type below ties the two together: ``aggregate[A]`` takes a typed
column and is fully monomorphized end to end, ``apply[F]`` is the runtime-dtype
convenience on top of it, and the serial / thread-local / radix execution
strategy is picked from row count + cardinality.
"""

from std.algorithm.functional import sync_parallelize

from ..arrays import (
    StructArray,
    AnyArray,
    UInt64Array,
    Int32Array,
    Int64Array,
)
from ..builders import AnyBuilder, Int32Builder
from ..dtypes import Field, struct_
from .hashtable import SwissHashTable
from .partition import RadixPartitioner
from .hashing import rapidhash
from .execution import ExecutionContext
from .filter import take
from .concat import concat
from .aggregate import Aggregation, AggFunction


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

    var _table: SwissHashTable[rapidhash]
    var _key_builders: List[AnyBuilder]

    def __init__(out self):
        self._table = SwissHashTable[rapidhash]()
        self._key_builders = List[AnyBuilder]()

    def num_groups(self) -> Int:
        return self._table.num_keys()

    def consume_keys(
        mut self, keys: StructArray, hashes: Optional[UInt64Array] = None
    ) raises -> Int32Array:
        """Hash keys and resolve group indices. Returns the per-row group ids.

        New keys get new (dense, contiguous) group ids; existing keys return
        their previous id. The group ids are exactly the table's bucket ids, so
        they're returned as-is — no separate conversion pass. Safe to call
        across multiple batches. Pass ``hashes`` when the caller already
        computed them (e.g. the parallel path reuses the partitioner's hashes)
        to skip the re-hash.
        """
        var n = len(keys)
        if n == 0:
            var empty = Int32Builder(0)
            return empty.finish()

        var prev = self._table.num_keys()
        var bids = self._table.insert_hashes(
            hashes.value(), grow_adaptively=True
        ) if hashes else self._table.insert(keys, grow_adaptively=True)
        var num_now = self._table.num_keys()
        var new_groups = num_now - prev

        # Materialize the key rows for the newly-created groups. Bucket ids are
        # assigned densely in row order, so each new group's first occurrence
        # appears in increasing bid order — one forward scan collects them all,
        # stopping as soon as the last new group is found (near-instant for the
        # low-cardinality case where all groups appear early).
        if new_groups > 0:
            var first_rows = Int32Builder(capacity=new_groups, zeroed=False)
            var next_new = prev
            for i in range(n):
                if Int(bids.unsafe_get(i)) == next_new:
                    first_rows.unsafe_append(Int32(i))
                    next_new += 1
                    if next_new == num_now:
                        break
            self._register_new_groups(keys, first_rows.finish())

        return bids^

    def key_fields(self, keys: StructArray) -> List[Field]:
        """The key columns' fields, taken from a keys struct's dtype."""
        var fields = List[Field]()
        ref st = keys.dtype.as_struct()
        for k in range(len(st.fields)):
            fields.append(Field(st.fields[k].name, st.fields[k].dtype.copy()))
        return fields^

    def key_columns(mut self, key_fields: List[Field]) raises -> List[AnyArray]:
        """The unique group-key columns (empty arrays when no groups yet).
        Finishes the per-column key builders — call once, at emit time."""
        var cols = List[AnyArray]()
        for k in range(len(key_fields)):
            if len(self._key_builders) == 0:
                var empty = AnyBuilder(key_fields[k].dtype)
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
                self._key_builders.append(AnyBuilder(keys.children[k].dtype()))
        var gathered = take(keys, rows)
        for k in range(len(keys.children)):
            self._key_builders[k].extend(gathered.children[k])


struct GroupedColumns(Copyable, Movable):
    """The result of a grouped aggregation: the unique key columns, and one
    column per aggregate over them.

    Columns, not a table — naming the outputs and assembling a schema is the
    caller's business, and the caller is the only one who knows what the
    aggregates were called."""

    var keys: List[AnyArray]
    var aggregates: List[AnyArray]

    def __init__(
        out self, var keys: List[AnyArray], var aggregates: List[AnyArray]
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
"""Grouping execution strategies — see `GroupBy` for what each trades off.

Public so a driver layered on top (the expression layer's runtime, multi-
aggregate group-by) can reuse the same strategy decision instead of making its
own. These name a *grouping* strategy; no aggregate identity is involved."""


struct GroupBy(Movable):
    """Grouped aggregation over a fixed set of key columns.

    Mirrors PyArrow's ``table.group_by(keys)``: build once from the key columns,
    then aggregate. ``aggregate[A]`` takes a statically-known ``Aggregation`` and
    a typed column, so the whole path is monomorphized; ``apply[F]`` resolves a
    column's runtime dtype to that ``Aggregation`` first. ``aggregate_columns``
    is the multi-column counterpart: it groups once and emits one column per
    value column through a caller-supplied *comptime* aggregator. No aggregate
    name or tag ever reaches this module — mapping one onto an ``Aggregation``
    is the expression layer's job (``marrow.expr.aggregates``).

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
    var _num_threads: Int
    var _strategy: UInt8

    def __init__(
        out self,
        keys: StructArray,
        ctx: ExecutionContext = ExecutionContext.auto(),
        strategy: Optional[UInt8] = None,
    ) raises:
        """Group by a struct of key columns (multi-key GROUP BY).

        ``strategy`` forces one of ``GROUP_SERIAL`` / ``GROUP_THREAD_LOCAL`` /
        ``GROUP_RADIX`` instead of picking from row count and cardinality —
        the escape hatch tests and benchmarks need to compare the paths against
        each other on the same input."""
        self._keys = keys.copy()
        self._num_threads = ctx.resolved_num_threads()
        self._strategy = (
            strategy.value() if strategy else Self._choose_strategy(
                self._keys, self._num_threads
            )
        )

    def __init__(
        out self,
        key: AnyArray,
        ctx: ExecutionContext = ExecutionContext.auto(),
        strategy: Optional[UInt8] = None,
    ) raises:
        """Group by a single key column."""
        var children = List[AnyArray]()
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
    def _choose_strategy(keys: StructArray, num_threads: Int) raises -> UInt8:
        var n = len(keys)
        if num_threads <= 1 or n < _PARALLEL_MIN_ROWS:
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
        return self._num_threads

    def strategy(self) -> UInt8:
        """The grouping strategy chosen at construction (``GROUP_SERIAL`` /
        ``GROUP_THREAD_LOCAL`` / ``GROUP_RADIX``)."""
        return self._strategy

    def aggregate[
        A: Aggregation
    ](self, value: A.InArray) raises -> GroupedColumns:
        """Aggregate a typed ``value`` column per group with aggregation ``A``.

        Returns the unique key columns and the aggregate column. ``A`` fixes the kernel *and* the input type, so the whole path is
        monomorphized: no dtype is resolved, no aggregate identity is compared,
        and erasure only reappears where the grouping machinery shuffles rows
        (``take`` / ``slice`` / ``concat``), which is dtype-generic by nature.

        A mergeable aggregation can take the thread-local path — row chunks
        folded into per-thread partials, then merged. Everything else groups by
        key partition, where each partition's groups are final on their own."""
        var values = List[AnyArray]()
        values.append(A.to_any(value))

        @parameter
        def one_column(
            _j: Int, gids: Int32Array, column: AnyArray, ng: Int
        ) raises -> AnyArray:
            return A.grouped(gids, A.from_any(column), ng).to_any()

        comptime if A.is_mergeable:
            if self._strategy == GROUP_THREAD_LOCAL:
                return Self._thread_local[A](
                    self._keys, value, self._num_threads
                )
            else:
                return self.aggregate_columns[one_column](values)
        else:
            return self.aggregate_columns[one_column](values)

    def apply[F: AggFunction](self, value: AnyArray) raises -> GroupedColumns:
        """Aggregate an erased ``value`` column with function ``F``: resolve the
        column's dtype to the ``Aggregation`` implementing ``F`` over it, then
        run the typed path above. The runtime-dtype entry point; the AOT path
        names its ``Aggregation`` directly and calls ``aggregate``."""
        var box = List[GroupedColumns]()

        @parameter
        def run[A: Aggregation]() raises:
            box.append(self.aggregate[A](A.from_any(value)))

        F.resolve[run](value.dtype())
        return box[0].copy()

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
        var sample = take(keys, idx.finish())
        var table = SwissHashTable[rapidhash]()
        _ = table.insert(sample, grow_adaptively=True)
        return table.num_keys() * 2 > s

    @staticmethod
    def _thread_local[
        A: Aggregation
    ](
        keys: StructArray, value: A.InArray, num_threads: Int
    ) raises -> GroupedColumns:
        """Thread-local partial aggregation — the low-/mid-cardinality parallel path.

        Every worker aggregates an equal contiguous chunk of rows into its *own*
        grouper + `AggState`, producing per-thread partial `(unique keys, acc, cnt)`.
        A serial merge then re-keys each thread's local groups into a global grouper
        and folds the partials with `A.merge` (exact for every kernel — the
        accumulator is the raw fold and the count is carried separately, so `mean`
        merges as (Σsum, Σcount)).

        Unlike the radix path, this scales with core count no matter how few distinct
        keys there are: with 10 groups on 16 cores, all 16 cores still do 1/16 of the
        scan, and the merge touches only ``num_threads × groups`` rows. Its weakness
        is very high cardinality (the serial merge grows toward O(N)) — that case
        goes to `_radix` instead.
        """
        var n = len(keys)
        var chunk = (n + num_threads - 1) // num_threads
        # Row slicing is dtype-generic, so it runs on the erased handle; every
        # chunk is narrowed straight back to `A.InArray` for the fold.
        var erased = A.to_any(value)

        # Pre-sized per-thread partial slots — no races on list growth.
        var part_keys = List[Optional[StructArray]](
            length=num_threads, fill=None
        )
        var part_acc = List[Optional[A.OutArray]](length=num_threads, fill=None)
        var part_cnt = List[Optional[Int64Array]](length=num_threads, fill=None)

        @parameter
        def worker(t: Int) raises:
            var start = t * chunk
            if start >= n:
                return
            var length = min(chunk, n - start)
            var kchunk = keys.slice(start, length)

            var grouper = HashGrouper()
            var gids = grouper.consume_keys(kchunk)
            var ng = grouper.num_groups()

            var kfields = grouper.key_fields(kchunk)
            var kcols = grouper.key_columns(kfields)
            part_keys[t] = StructArray(
                dtype=keys.dtype.copy(),
                length=ng,
                nulls=0,
                offset=0,
                bitmap=None,
                children=kcols^,
            )

            var parts = A.partials(
                gids, A.from_any(erased.slice(start, length)), ng
            )
            part_acc[t] = parts[0].copy()
            part_cnt[t] = parts[1].copy()

        sync_parallelize[worker](num_threads)

        # Merge — serial, but touches only `num_threads × groups` rows.
        var gg = HashGrouper()
        var remap = List[Int32Array]()
        var accs = List[A.OutArray]()
        var cnts = List[Int64Array]()
        for t in range(num_threads):
            if not part_keys[t]:
                continue
            remap.append(gg.consume_keys(part_keys[t].value()))
            accs.append(part_acc[t].value().copy())
            cnts.append(part_cnt[t].value().copy())
        var agg_col = A.merge(remap, accs, cnts, gg.num_groups()).to_any()

        var key_cols = gg.key_columns(gg.key_fields(keys))
        var agg_cols = List[AnyArray]()
        agg_cols.append(agg_col^)
        return GroupedColumns(key_cols^, agg_cols^)

    def aggregate_columns[
        col_agg: def(Int, Int32Array, AnyArray, Int) raises capturing[
            _
        ] -> AnyArray
    ](self, values: List[AnyArray]) raises -> GroupedColumns:
        """Group the keys once, then emit ``col_agg(j, gids, values[j], ng)`` as
        output column ``j`` — the multi-aggregate driver.

        Never thread-local: the aggregator is opaque here, so its per-thread
        partial state can't be merged. (The thread-local *fold* path is
        ``aggregate[A]``, where the aggregation — and therefore its ``merge`` —
        is statically known.)"""
        return Self._by_partition[col_agg](
            self._keys,
            values,
            self._num_threads,
            partition=self._strategy != GROUP_SERIAL,
        )

    @staticmethod
    def _by_partition[
        col_agg: def(Int, Int32Array, AnyArray, Int) raises capturing[
            _
        ] -> AnyArray,
    ](
        keys: StructArray,
        values: List[AnyArray],
        num_threads: Int,
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
        var ctx = ExecutionContext.parallel(
            num_threads
        ) if partition else ExecutionContext.serial()
        var na = len(values)

        @parameter
        def group_partition(
            rows: Int32Array, part_hashes: UInt64Array
        ) raises -> Tuple[Int32Array, List[AnyArray]]:
            var n = len(part_hashes)
            var table = SwissHashTable[rapidhash]()
            var gids = table.insert_hashes(
                part_hashes, grow_adaptively=not partition
            )
            var ng = table.num_keys()

            # First-occurrence original row per new group (bucket ids are dense
            # and assigned in row order, so first occurrences appear in bid
            # order). Unpartitioned rows are their own row numbers.
            var first = Int32Builder(capacity=ng, zeroed=False)
            var next_new = 0
            for i in range(n):
                if Int(gids.unsafe_get(i)) == next_new:
                    first.unsafe_append(
                        rows.unsafe_get(i) if partition else Int32(i)
                    )
                    next_new += 1
                    if next_new == ng:
                        break

            var agg_cols = List[AnyArray]()
            for j in range(na):
                # Values in partition order, aligned with `gids`. A single
                # partition *is* the whole input, already in order — no gather.
                if partition:
                    agg_cols.append(col_agg(j, gids, take(values[j], rows), ng))
                else:
                    agg_cols.append(col_agg(j, gids, values[j], ng))
            return (first.finish(), agg_cols^)

        var parts = List[Tuple[Int32Array, List[AnyArray]]]()
        if partition:

            @parameter
            def radix_partition(
                _pi: Int, rows: Int32Array, part_hashes: UInt64Array
            ) raises -> Tuple[Int32Array, List[AnyArray]]:
                return group_partition(rows, part_hashes)

            parts = RadixPartitioner(
                num_bits=RADIX_BITS, num_threads=num_threads
            ).map_partitions[
                Tuple[Int32Array, List[AnyArray]], radix_partition
            ](
                rapidhash(keys, ctx)
            )
        else:
            var no_rows = Int32Builder(0)
            parts.append(
                group_partition(no_rows.finish(), rapidhash(keys, ctx))
            )

        # The global unique-key set is the union of the partitions': concatenate
        # their first-occurrence rows and gather the key columns once.
        var first_chunks = List[AnyArray]()
        for i in range(len(parts)):
            first_chunks.append(parts[i][0].copy())
        var first_any = concat(first_chunks, ctx)
        ref first_rows = first_any.as_int32()

        var key_cols = List[AnyArray]()
        for k in range(len(keys.children)):
            key_cols.append(
                take(
                    keys.children[k].slice(keys.offset, len(keys)),
                    first_rows,
                    ctx,
                )
            )

        var agg_cols = List[AnyArray]()
        for j in range(na):
            var chunks = List[AnyArray]()
            for i in range(len(parts)):
                chunks.append(parts[i][1][j].copy())
            agg_cols.append(concat(chunks, ctx))

        return GroupedColumns(key_cols^, agg_cols^)
