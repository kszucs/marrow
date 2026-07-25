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
from ..dtypes import Field, AnyDataType, struct_, NumericType, int32, int64
from ..schema import Schema
from ..tabular import RecordBatch
from .hashtable import SwissHashTable
from .partition import RadixPartitioner
from .hashing import rapidhash
from .execution import ExecutionContext
from .filter import take
from .concat import concat
from .aggregate import (
    AggKernel,
    AggState,
    SumKernel,
    ProductKernel,
    MinKernel,
    MaxKernel,
    CountKernel,
    MeanKernel,
    min_max_string_grouped,
    reinterpret_array,
    temporal_backing_dtype,
)
from .distinct import (
    count_distinct_grouped,
    approx_count_distinct_grouped,
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


def slice_struct(
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


struct GroupBy(Movable):
    """Grouped aggregation over a fixed set of key columns.

    Mirrors PyArrow's ``table.group_by(keys)``: build once from the key columns,
    then apply an aggregate with ``aggregate[K]`` — or the ``sum`` / ``product``
    / ``min`` / ``max`` / ``count`` / ``mean`` shorthands. Each aggregate is a
    statically-known ``AggKernel``, so the result is fully monomorphized (the
    input dtype ``V`` is resolved once at the boundary and the typed
    ``AggState[K, V]`` does the work). ``aggregate_columns`` is the multi-column
    counterpart: it groups once and emits one column per value column through a
    caller-supplied *comptime* aggregator — still no runtime kernel dispatch
    here. Resolving a runtime function *name* to a kernel is the expression
    layer's job (``marrow.expr.aggregates``), which builds exactly such an
    aggregator; this module never sees an aggregate name or tag.

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

    def aggregate[K: AggKernel](self, value: AnyArray) raises -> RecordBatch:
        """Aggregate ``value`` per group with kernel ``K``. Returns a batch of
        the unique key columns followed by the aggregate column."""
        if self._strategy == GROUP_THREAD_LOCAL:
            return Self._thread_local[K](self._keys, value, self._num_threads)
        elif self._strategy == GROUP_RADIX:
            return Self._radix[K](self._keys, value, self._num_threads)
        return Self._serial[K](self._keys, value)

    def sum(self, value: AnyArray) raises -> RecordBatch:
        """Per-group sum (integers widen to int64, floats stay float64)."""
        return self.aggregate[SumKernel](value)

    def product(self, value: AnyArray) raises -> RecordBatch:
        """Per-group product."""
        return self.aggregate[ProductKernel](value)

    def min(self, value: AnyArray) raises -> RecordBatch:
        """Per-group minimum (preserves the input dtype).

        Numeric columns take the fully-typed `AggState` fast path. Temporal
        columns fold over their (order-preserving) signed-integer backing and are
        relabelled back on the way out. String columns aren't an `AggKernel`
        fold at all, so they use the dedicated bytewise per-group scan."""
        return self._min_max[MinKernel](value, is_min=True)

    def max(self, value: AnyArray) raises -> RecordBatch:
        """Per-group maximum (preserves the input dtype). See `min` for how
        string / temporal columns are routed."""
        return self._min_max[MaxKernel](value, is_min=False)

    def _min_max[
        K: AggKernel
    ](self, value: AnyArray, *, is_min: Bool) raises -> RecordBatch:
        """Shared `min`/`max` routing — kernel picked by the caller at compile
        time, never from a runtime tag."""
        var vdt = value.dtype()
        if vdt.is_string() or vdt.is_large_string():

            @parameter
            def bytewise(
                _j: Int, gids: Int32Array, col: AnyArray, ng: Int
            ) raises -> AnyArray:
                return min_max_string_grouped(gids, col, ng, is_min)

            return self._one_column[bytewise](value, K.name)
        elif vdt.is_temporal():
            var backing = temporal_backing_dtype(vdt)
            var folded = self.aggregate[K](reinterpret_array(value, backing))
            return Self._relabel_last(folded^, vdt)
        else:
            return self.aggregate[K](value)

    @staticmethod
    def _relabel_last(
        var result: RecordBatch, dtype: AnyDataType
    ) raises -> RecordBatch:
        """Relabel the trailing (aggregate) column of a group-by result to
        ``dtype`` — the inverse of the temporal-backing reinterpret."""
        var last = result.num_columns() - 1
        var out_fields = List[Field]()
        var out_cols = List[AnyArray]()
        for c in range(result.num_columns()):
            if c == last:
                out_fields.append(
                    Field(result.schema.fields[c].name, dtype.copy())
                )
                out_cols.append(reinterpret_array(result.column(c), dtype))
            else:
                out_fields.append(result.schema.fields[c].copy())
                out_cols.append(result.column(c).copy())
        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    def count(self, value: AnyArray) raises -> RecordBatch:
        """Per-group count of valid (non-null) values, as int64."""
        return self.aggregate[CountKernel](value)

    def mean(self, value: AnyArray) raises -> RecordBatch:
        """Per-group arithmetic mean, as float64."""
        return self.aggregate[MeanKernel](value)

    def count_distinct(self, value: AnyArray) raises -> RecordBatch:
        """Per-group exact count of distinct non-null values, as int64."""

        @parameter
        def exact(
            _j: Int, gids: Int32Array, col: AnyArray, ng: Int
        ) raises -> AnyArray:
            return count_distinct_grouped(gids, col, ng)

        return self._one_column[exact](value, "count_distinct")

    def approx_count_distinct(self, value: AnyArray) raises -> RecordBatch:
        """Per-group approximate distinct count (HyperLogLog), as int64."""

        @parameter
        def approx(
            _j: Int, gids: Int32Array, col: AnyArray, ng: Int
        ) raises -> AnyArray:
            return approx_count_distinct_grouped(gids, col, ng)

        return self._one_column[approx](value, "approx_count_distinct")

    def _one_column[
        col_agg: def(Int, Int32Array, AnyArray, Int) raises capturing[
            _
        ] -> AnyArray
    ](self, value: AnyArray, name: String) raises -> RecordBatch:
        """`aggregate_columns` for a single value column."""
        var values = List[AnyArray]()
        values.append(value.copy())
        var names = List[String]()
        names.append(name)
        return self.aggregate_columns[col_agg](values, names)

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

        @parameter
        def by_value[V: NumericType](d: V) raises -> AnyArray:
            var state = AggState[K, V]()
            state.update(gids, value.as_primitive[V](), num_groups)
            return state.finish(num_groups).to_any()

        var agg_col = value.dtype().dispatch_numeric[by_value]()

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
            var kchunk = slice_struct(keys, start, length)
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
            def by_value[V: NumericType](d: V) raises:
                var state = AggState[K, V]()
                state.update(gids, vchunk.as_primitive[V](), ng)
                var parts = state.into_partials()
                part_acc[t] = parts[0].copy().to_any()
                part_cnt[t] = parts[1].copy()

            value.dtype().dispatch_numeric[by_value]()

        sync_parallelize[worker](num_threads)

        # Merge — serial, but touches only `num_threads × groups` rows.
        var gg = HashGrouper()

        @parameter
        def merge_value[V: NumericType](d: V) raises -> AnyArray:
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
            return gstate.finish(gg.num_groups()).to_any()

        var agg_col = value.dtype().dispatch_numeric[merge_value]()

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

        # Per-partition: group by the (already computed) hashes + aggregate,
        # returning (first-occurrence original rows, aggregate column). Groups
        # are disjoint across partitions, so no cross-partition combine is needed.
        @parameter
        def aggregate_partition(
            _pi: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises -> Tuple[Int32Array, AnyArray]:
            var n = len(rows)
            # Pre-size the table to the partition's row count (an upper bound on
            # its group count) instead of growing adaptively: this path only runs
            # for high cardinality, so most rows are distinct keys and the table
            # would otherwise rehash ~log2(groups) times during the build. The
            # over-allocation is bounded (rows/groups is small at high card) and
            # the extra ctrl memset is bandwidth-cheap — net ~5% faster.
            var table = SwissHashTable[rapidhash]()
            var gids = table.insert_hashes(part_hashes, grow_adaptively=False)
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
            def by_value[V: NumericType](d: V) raises -> AnyArray:
                var state = AggState[K, V]()
                state.update(gids, pvals.as_primitive[V](), ng)
                return state.finish(ng).to_any()

            return (
                first.finish(),
                value.dtype().dispatch_numeric[by_value](),
            )

        var hashes = rapidhash(keys, ctx)
        var parts = RadixPartitioner(
            num_bits=RADIX_BITS, num_threads=num_threads
        ).map_partitions[Tuple[Int32Array, AnyArray], aggregate_partition](
            hashes^
        )

        # Merge — the global unique-key set is the union of the partitions'.
        # Concatenate the first-occurrence rows, gather the key columns from the
        # original keys once, and concatenate the aggregate columns.
        var first_chunks = List[AnyArray]()
        var agg_chunks = List[AnyArray]()
        for i in range(len(parts)):
            first_chunks.append(parts[i][0].copy())
            agg_chunks.append(parts[i][1].copy())
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
        @parameter
        def agg_dtype[V: NumericType](d: V) raises -> AnyDataType:
            return AnyDataType(K.AccType[V]())

        out_fields.append(
            Field(K.name, value.dtype().dispatch_numeric[agg_dtype]())
        )

        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    # -----------------------------------------------------------------------
    # Generic multi-column driver — group ONCE, then produce one output column
    # per value column through a caller-supplied, *comptime* aggregator. The
    # aggregator is opaque to the grouper, so no aggregate identity (name, tag)
    # ever reaches this layer; the expression layer builds one that routes a
    # runtime function name to its kernel.
    # -----------------------------------------------------------------------

    def aggregate_columns[
        col_agg: def(Int, Int32Array, AnyArray, Int) raises capturing[
            _
        ] -> AnyArray
    ](self, values: List[AnyArray], names: List[String]) raises -> RecordBatch:
        """Group the keys once, then emit ``col_agg(j, gids, values[j], ng)`` as
        output column ``j``, named ``names[j]``.

        Returns the unique key columns followed by one aggregate column each.
        Runs serially or radix-partition-parallel — never thread-local: the
        aggregator is opaque, so its per-thread partial state can't be merged.
        (The thread-local *fold* path is `aggregate[K]`, where the kernel — and
        therefore `AggState.merge` — is statically known.)"""
        if len(values) != len(names):
            raise Error("aggregate_columns: len(values) != len(names)")
        if self._strategy == GROUP_SERIAL:
            return Self._serial_columns[col_agg](self._keys, values, names)
        else:
            return Self._radix_columns[col_agg](
                self._keys, values, names, self._num_threads
            )

    @staticmethod
    def _serial_columns[
        col_agg: def(Int, Int32Array, AnyArray, Int) raises capturing[
            _
        ] -> AnyArray
    ](
        keys: StructArray, values: List[AnyArray], names: List[String]
    ) raises -> RecordBatch:
        var grouper = HashGrouper()
        var gids = grouper.consume_keys(keys)
        var ng = grouper.num_groups()

        var kfields = grouper.key_fields(keys)
        var out_fields = List[Field]()
        for k in range(len(kfields)):
            out_fields.append(kfields[k].copy())
        var out_cols = grouper.key_columns(kfields)

        for j in range(len(values)):
            var col = col_agg(j, gids, values[j], ng)
            out_fields.append(Field(names[j], col.dtype().copy()))
            out_cols.append(col^)
        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)

    @staticmethod
    def _radix_columns[
        col_agg: def(Int, Int32Array, AnyArray, Int) raises capturing[
            _
        ] -> AnyArray
    ](
        keys: StructArray,
        values: List[AnyArray],
        names: List[String],
        num_threads: Int,
    ) raises -> RecordBatch:
        """Radix-partition-parallel counterpart of `_serial_columns`. A key lands
        in exactly one partition, so per-partition groups are disjoint and every
        partition's column is final — concatenated, never merged."""
        var ctx = ExecutionContext.parallel(num_threads)
        var na = len(values)

        @parameter
        def agg_partition(
            _pi: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises -> Tuple[Int32Array, List[AnyArray]]:
            var nrows = len(rows)
            var table = SwissHashTable[rapidhash]()
            var gids = table.insert_hashes(part_hashes, grow_adaptively=False)
            var ng = table.num_keys()

            var first = Int32Builder(capacity=ng, zeroed=False)
            var next_new = 0
            for i in range(nrows):
                if Int(gids.unsafe_get(i)) == next_new:
                    first.unsafe_append(rows.unsafe_get(i))
                    next_new += 1
                    if next_new == ng:
                        break

            var agg_cols = List[AnyArray]()
            for j in range(na):
                agg_cols.append(col_agg(j, gids, take(values[j], rows), ng))
            return (first.finish(), agg_cols^)

        var hashes = rapidhash(keys, ctx)
        var parts = RadixPartitioner(
            num_bits=RADIX_BITS, num_threads=num_threads
        ).map_partitions[Tuple[Int32Array, List[AnyArray]], agg_partition](
            hashes^
        )

        var first_chunks = List[AnyArray]()
        for i in range(len(parts)):
            first_chunks.append(parts[i][0].copy())
        var first_any = concat(first_chunks, ctx)
        ref first_rows = first_any.as_int32()

        ref kstruct = keys.dtype.as_struct()
        var out_fields = List[Field]()
        var out_cols = List[AnyArray]()
        for k in range(len(kstruct.fields)):
            out_fields.append(kstruct.fields[k].copy())
            out_cols.append(take(keys.children[k], first_rows, ctx))

        for j in range(na):
            var chunks = List[AnyArray]()
            for i in range(len(parts)):
                chunks.append(parts[i][1][j].copy())
            var col = concat(chunks, ctx)
            out_fields.append(Field(names[j], col.dtype().copy()))
            out_cols.append(col^)

        return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)
