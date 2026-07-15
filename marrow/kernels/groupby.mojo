"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouper`` hashes the key columns and resolves every row
     to a dense group index, storing the unique key rows.
  2. **Phase 2** — aggregate accumulation, layered on top by the caller using an
     ``AggKernel``'s grouped methods (``scatter``/``grow``/``finish``).

The grouper itself is **aggregate-agnostic**: aggregates are ``AggKernel``
types (``aggregate.mojo``), and any runtime ``name -> kernel`` selection lives in
the expression layer (``marrow/expr``). The ``GroupBy`` type below ties the two
together for the compile-time / AOT path (one statically-known aggregate),
fully monomorphized with no runtime kernel dispatch, and picks the serial /
thread-local / radix execution strategy from row count + cardinality.
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
from ..dtypes import Field, AnyDataType, struct_, NumericType
from ..schema import Schema
from ..tabular import RecordBatch
from .hashtable import SwissHashTable, RadixPartitioner
from .hashing import rapidhash
from .execution import ExecutionContext
from .filter import take
from .concat import concat
from .aggregate import (
    AggKernel,
    AggState,
    for_value_dtype,
    SumKernel,
    ProductKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    MeanKernel,
)


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


# ---------------------------------------------------------------------------
# GroupBy — grouped aggregation over a fixed set of key columns.
# ---------------------------------------------------------------------------


comptime _PARALLEL_MIN_ROWS = 60_000
"""Below this row count the serial path always wins — partitioning + dispatch
overhead dominates even for high-cardinality input."""

comptime _PARALLEL_ALWAYS_ROWS = 200_000
"""At or above this the parallel path wins for *any* cardinality, so the
cardinality probe is skipped."""

comptime _RADIX_BITS = 6
"""Radix fanout for the high-cardinality parallel path (2**6 = 64 partitions)."""

comptime _CARD_SAMPLE_ROWS = 4096
"""Rows sampled (strided) to estimate cardinality on the dispatch boundary."""


struct GroupBy(Movable):
    """Grouped aggregation over a fixed set of key columns.

    Mirrors PyArrow's ``table.group_by(keys)``: build once from the key columns,
    then apply an aggregate with ``aggregate[K]`` — or the ``sum`` / ``product``
    / ``min`` / ``max`` / ``count`` / ``mean`` shorthands. Each aggregate is a
    statically-known ``AggKernel``, so the result is fully monomorphized (the
    input dtype ``V`` is resolved once at the boundary and the typed
    ``AggState[K, V]`` does the work); runtime, plan-driven multi-aggregate
    queries live in the expression layer instead.

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
    - **radix-partition-parallel** (`_radix`) — high-cardinality inputs,
      where a key lands in one partition so groups never span threads and the
      thread-local merge would instead become an O(N) serial bottleneck.
    """

    comptime _SERIAL: UInt8 = 0
    comptime _THREAD_LOCAL: UInt8 = 1
    comptime _RADIX: UInt8 = 2

    var _keys: StructArray
    var _num_threads: Int
    var _strategy: UInt8

    def __init__(
        out self,
        keys: StructArray,
        ctx: ExecutionContext = ExecutionContext.auto(),
    ) raises:
        """Group by a struct of key columns (multi-key GROUP BY)."""
        self._keys = keys.copy()
        self._num_threads = ctx.resolved_num_threads()
        self._strategy = Self._choose_strategy(self._keys, self._num_threads)

    def __init__(
        out self,
        key: AnyArray,
        ctx: ExecutionContext = ExecutionContext.auto(),
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
        )

    @staticmethod
    def _choose_strategy(keys: StructArray, num_threads: Int) raises -> UInt8:
        var n = len(keys)
        if num_threads <= 1 or n < _PARALLEL_MIN_ROWS:
            return Self._SERIAL
        var high_card = Self._is_high_cardinality(keys, n)
        if n < _PARALLEL_ALWAYS_ROWS and not high_card:
            return Self._SERIAL
        if high_card:
            return Self._RADIX
        return Self._THREAD_LOCAL

    def aggregate[K: AggKernel](self, value: AnyArray) raises -> RecordBatch:
        """Aggregate ``value`` per group with kernel ``K``. Returns a batch of
        the unique key columns followed by the aggregate column."""
        if self._strategy == Self._THREAD_LOCAL:
            return Self._thread_local[K](self._keys, value, self._num_threads)
        elif self._strategy == Self._RADIX:
            return Self._radix[K](self._keys, value, self._num_threads)
        return Self._serial[K](self._keys, value)

    def sum(self, value: AnyArray) raises -> RecordBatch:
        """Per-group sum (integers widen to int64, floats stay float64)."""
        return self.aggregate[SumKernel](value)

    def product(self, value: AnyArray) raises -> RecordBatch:
        """Per-group product."""
        return self.aggregate[ProductKernel](value)

    def min(self, value: AnyArray) raises -> RecordBatch:
        """Per-group minimum (preserves the input dtype)."""
        return self.aggregate[MinKernel](value)

    def max(self, value: AnyArray) raises -> RecordBatch:
        """Per-group maximum (preserves the input dtype)."""
        return self.aggregate[MaxKernel](value)

    def count(self, value: AnyArray) raises -> RecordBatch:
        """Per-group count of valid (non-null) values, as int64."""
        return self.aggregate[CountKernel](value)

    def mean(self, value: AnyArray) raises -> RecordBatch:
        """Per-group arithmetic mean, as float64."""
        return self.aggregate[MeanKernel](value)

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
    def _serial[
        K: AggKernel
    ](keys: StructArray, value: AnyArray) raises -> RecordBatch:
        """One `HashGrouper` + one typed `AggState[K, V]` over the whole input.
        """
        var grouper = HashGrouper()
        var gids = grouper.consume_keys(keys)
        var num_groups = grouper.num_groups()

        var box = List[AnyArray]()

        @parameter
        def by_value[V: NumericType]() raises:
            var state = AggState[K, V]()
            state.update(gids, value.as_primitive[V](), num_groups)
            box.append(state.finish(num_groups).to_any())

        for_value_dtype[by_value](value.dtype())
        var agg_col = box[0].copy()

        var kfields = grouper.key_fields(keys)
        var result_fields = List[Field]()
        for k in range(len(kfields)):
            result_fields.append(kfields[k].copy())
        result_fields.append(Field(K.name, agg_col.dtype().copy()))

        var result_cols = grouper.key_columns(kfields)
        result_cols.append(agg_col^)
        return RecordBatch(
            schema=Schema(fields=result_fields^), columns=result_cols^
        )

    @staticmethod
    def _slice_struct(
        keys: StructArray, start: Int, length: Int
    ) raises -> StructArray:
        """Zero-copy row-range slice of a keys struct — slices each child column so
        the per-column hashers see exactly ``[start, start+length)`` (the struct's
        own offset/length aren't propagated to children by the hasher)."""
        var children = List[AnyArray]()
        for k in range(len(keys.children)):
            children.append(keys.children[k].slice(start, length))
        return StructArray(
            dtype=keys.dtype.copy(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            children=children^,
        )

    @staticmethod
    def _thread_local[
        K: AggKernel
    ](
        keys: StructArray, value: AnyArray, num_threads: Int
    ) raises -> RecordBatch:
        """Thread-local partial aggregation — the low-/mid-cardinality parallel path.

        Every worker aggregates an equal contiguous chunk of rows into its *own*
        grouper + `AggState`, producing per-thread partial `(unique keys, acc, cnt)`.
        A serial merge then re-keys each thread's local groups into a global grouper
        and folds the partials with `AggState.merge` (exact for every kernel — the
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

        # Pre-sized per-thread partial slots — no races on list growth.
        var part_keys = List[Optional[StructArray]](
            length=num_threads, fill=None
        )
        var part_acc = List[Optional[AnyArray]](length=num_threads, fill=None)
        var part_cnt = List[Optional[Int64Array]](length=num_threads, fill=None)

        @parameter
        def worker(t: Int) raises:
            var start = t * chunk
            if start >= n:
                return
            var length = min(chunk, n - start)
            var kchunk = Self._slice_struct(keys, start, length)
            var vchunk = value.slice(start, length)

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

            @parameter
            def by_value[V: NumericType]() raises:
                var state = AggState[K, V]()
                state.update(gids, vchunk.as_primitive[V](), ng)
                var parts = state.into_partials()
                part_acc[t] = parts[0].copy().to_any()
                part_cnt[t] = parts[1].copy()

            for_value_dtype[by_value](value.dtype())

        sync_parallelize[worker](num_threads)

        # Merge — serial, but touches only `num_threads × groups` rows.
        var gg = HashGrouper()
        var box = List[AnyArray]()

        @parameter
        def merge_value[V: NumericType]() raises:
            var gstate = AggState[K, V]()
            for t in range(num_threads):
                if not part_keys[t]:
                    continue
                var lg = gg.consume_keys(part_keys[t].value())
                var gng = gg.num_groups()
                gstate.merge(
                    lg,
                    part_acc[t].value().as_primitive[K.AccType[V]](),
                    part_cnt[t].value(),
                    gng,
                )
            box.append(gstate.finish(gg.num_groups()).to_any())

        for_value_dtype[merge_value](value.dtype())
        var agg_col = box[0].copy()

        var kfields = gg.key_fields(keys)
        var out_fields = List[Field]()
        for k in range(len(kfields)):
            out_fields.append(kfields[k].copy())
        out_fields.append(Field(K.name, agg_col.dtype().copy()))

        var out_cols = gg.key_columns(kfields)
        out_cols.append(agg_col^)
        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    @staticmethod
    def _radix[
        K: AggKernel
    ](
        keys: StructArray, value: AnyArray, num_threads: Int
    ) raises -> RecordBatch:
        """Radix-partition-parallel grouped aggregation.

        Hash once, then split rows into ``2**_RADIX_BITS`` partitions by the top
        hash bits. A key lands in exactly one partition, so per-partition groups are
        disjoint and no cross-partition combine is needed. Each worker groups its
        partition by reusing the partition hashes (no re-hash) and aggregates,
        recording each new group's first-occurrence *original* row rather than
        materializing the key columns.

        The merge is cheap and touches only ``num_groups`` rows, not ``N``:
        concatenate the per-partition first-occurrence rows, gather the unique key
        columns from the original ``keys`` in one ``take``, and concatenate the
        per-partition aggregate columns. This avoids the two full-``N`` per-partition
        ``take`` gathers (keys + finished key columns) the previous version paid.
        """
        var ctx = ExecutionContext.parallel(num_threads)
        var partitioner = RadixPartitioner(
            num_bits=_RADIX_BITS, num_threads=num_threads
        )
        var partitions = partitioner.partition(rapidhash(keys, ctx))
        var p = len(partitions)

        # Pre-sized slots — workers assign by index without racing on list growth.
        var first_slots = List[Optional[Int32Array]](length=p, fill=None)
        var agg_slots = List[Optional[AnyArray]](length=p, fill=None)

        @parameter
        def worker(pi: Int) raises:
            ref rows = partitions[pi].row_indices.value()
            var n = len(rows)

            # Group this partition's rows by their (already computed) hashes.
            var table = SwissHashTable[rapidhash]()
            var gids = table.insert_hashes(
                partitions[pi].hashes, grow_adaptively=True
            )
            var ng = table.num_keys()

            # First-occurrence original row per new group (bids are dense and
            # assigned in row order, so first occurrences appear in bid order).
            var first = Int32Builder(capacity=ng, zeroed=False)
            var next_new = 0
            for i in range(n):
                if Int(gids.unsafe_get(i)) == next_new:
                    first.unsafe_append(rows.unsafe_get(i))
                    next_new += 1
                    if next_new == ng:
                        break

            # Values in partition order, aligned with `gids`, for the scatter.
            var pvals = take(value, rows)

            @parameter
            def by_value[V: NumericType]() raises:
                var state = AggState[K, V]()
                state.update(gids, pvals.as_primitive[V](), ng)
                agg_slots[pi] = state.finish(ng).to_any()

            for_value_dtype[by_value](value.dtype())
            first_slots[pi] = first.finish()

        sync_parallelize[worker](p)

        # Merge — groups are disjoint across partitions, so the global unique-key set
        # is just their union. Concatenate the first-occurrence rows, gather the key
        # columns from the original keys once, and concatenate the aggregates.
        var first_chunks = List[AnyArray]()
        var agg_chunks = List[AnyArray]()
        for i in range(p):
            first_chunks.append(first_slots[i].value().copy())
            agg_chunks.append(agg_slots[i].value().copy())
        var first_any = concat(first_chunks, ctx)
        ref first_rows = first_any.as_int32()

        ref kstruct = keys.dtype.as_struct()
        var out_fields = List[Field]()
        var out_cols = List[AnyArray]()
        for k in range(len(kstruct.fields)):
            out_fields.append(kstruct.fields[k].copy())
            out_cols.append(take(keys.children[k], first_rows, ctx))
        out_cols.append(concat(agg_chunks, ctx))

        # Aggregate output field dtype.
        var agg_box = List[AnyDataType]()

        @parameter
        def agg_dtype[V: NumericType]() raises:
            agg_box.append(AnyDataType(K.AccType[V]()))

        for_value_dtype[agg_dtype](value.dtype())
        out_fields.append(Field(K.name, agg_box[0].copy()))

        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)
