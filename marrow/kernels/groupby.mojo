"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouper`` hashes the key columns and resolves every row
     to a dense group index, storing the unique key rows.
  2. **Phase 2** — aggregate accumulation, layered on top by the caller using an
     ``AggKernel``'s grouped methods (``scatter``/``grow``/``finish``).

The grouper itself is **aggregate-agnostic**: aggregates are ``AggKernel``
types (``aggregate.mojo``), and any runtime ``name -> kernel`` selection lives in
the expression layer (``marrow/expr``). The typed ``group_by[K]`` convenience
below ties the two together for the compile-time / AOT path (one aggregate,
known statically), fully monomorphized with no runtime kernel dispatch.
"""

from std.algorithm.functional import sync_parallelize
from std.sys.info import num_physical_cores

from ..arrays import (
    StructArray,
    AnyArray,
    UInt32Array,
    UInt64Array,
    Int32Array,
)
from ..builders import AnyBuilder, UInt32Builder
from ..dtypes import Field, AnyDataType, struct_, uint32, NumericType
from ..schema import Schema
from ..tabular import RecordBatch
from .hashtable import SwissHashTable, RadixPartitioner
from .hashing import rapidhash
from .execution import ExecutionContext
from .filter import take
from .concat import concat
from .aggregate import AggKernel, AggState, for_value_dtype


# ---------------------------------------------------------------------------
# HashGrouper — keys-only hash grouping
# ---------------------------------------------------------------------------


struct HashGrouper(Movable):
    """Keys-only hash grouper (ClickHouse-style, ``SwissHashTable``-backed).

    ``consume_keys`` hashes a batch of key rows, returns their dense group ids,
    and appends newly-seen key rows to a per-column builder. Call it repeatedly
    to accumulate groups across batches. NULL keys are treated as equal (same
    group), matching SQL GROUP BY semantics (unlike join, where NULL != NULL).

    Aggregate state is owned by the caller, not the grouper — see ``group_by``
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
    ) raises -> UInt32Array:
        """Hash keys and resolve group indices. Returns the per-row group ids.

        New keys get new (dense, contiguous) group ids; existing keys return
        their previous id. Safe to call across multiple batches. Pass
        ``hashes`` when the caller already computed them (e.g. the parallel
        path reuses the partitioner's hashes) to skip the re-hash.
        """
        var n = len(keys)
        if n == 0:
            var empty = UInt32Builder(0)
            return empty.finish()

        var prev = self._table.num_keys()
        var bids = self._table.insert_hashes(
            hashes.value()
        ) if hashes else self._table.insert(keys)
        var new_groups = self._table.num_keys() - prev

        if new_groups > 0:
            var seen = List[Bool](length=new_groups, fill=False)
            for i in range(n):
                var gid = Int(bids.unsafe_get(i))
                if gid >= prev and not seen[gid - prev]:
                    seen[gid - prev] = True
                    self._register_new_group(keys, i)

        # Convert int32 bucket ids → uint32 group ids.
        var gid_builder = UInt32Builder(capacity=n)
        for i in range(n):
            gid_builder.unsafe_append(
                Scalar[uint32.native](Int(bids.unsafe_get(i)))
            )
        return gid_builder.finish()

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

    def _register_new_group(mut self, keys: StructArray, row: Int) raises:
        """Append the key row for a newly created group to the per-column
        builders (O(1) amortized — no per-group column rebuild)."""
        if len(self._key_builders) == 0:
            for k in range(len(keys.children)):
                self._key_builders.append(AnyBuilder(keys.children[k].dtype()))
        for k in range(len(keys.children)):
            self._key_builders[k].extend(keys.children[k].slice(row, 1))


# ---------------------------------------------------------------------------
# group_by — typed single-aggregate grouped aggregation.
#
# Serial for small inputs; radix-partition-parallel for large ones (same
# pattern as `HashJoin`): hash keys once, split rows by the top hash bits into
# independent partitions, then group+aggregate each partition on its own thread.
# A key hashes to exactly one partition, so groups never span partitions — the
# merge is a plain per-column `concat`, no cross-partition combine.
# ---------------------------------------------------------------------------


comptime _PARALLEL_MIN_ROWS = 100_000
"""Below this row count the serial path wins — partitioning + dispatch overhead
dominates on small inputs (matches `HashJoin`'s threshold)."""

comptime _RADIX_BITS = 6
"""Radix fanout for the parallel path (2**6 = 64 partitions)."""


def group_by[
    K: AggKernel
](
    keys: StructArray,
    value: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> RecordBatch:
    """Grouped aggregation with a single, statically-known aggregate kernel.

    Monomorphized on ``K``; the input dtype ``V`` is resolved once at the
    boundary, then the fully typed ``AggState[K, V]`` does the work. Dispatches
    to a radix-partition-parallel path for large inputs. For runtime,
    multi-aggregate queries (kernels chosen from a plan), the expression layer
    drives the same ``AggState`` behind its own tag dispatch.
    """
    var nt = num_physical_cores()
    if len(keys) < _PARALLEL_MIN_ROWS or nt <= 1:
        return _group_by_serial[K](keys, value)
    return _group_by_parallel[K](keys, value, nt)


def _group_by_serial[
    K: AggKernel
](keys: StructArray, value: AnyArray) raises -> RecordBatch:
    """One `HashGrouper` + one typed `AggState[K, V]` over the whole input."""
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


def _partition_columns[
    op: def(Int32Array, UInt64Array) raises capturing[_] -> List[AnyArray]
](keys: StructArray, num_threads: Int) raises -> List[List[AnyArray]]:
    """Reusable radix-partition-parallel driver.

    Hash ``keys`` once, split rows into ``2**_RADIX_BITS`` partitions by the top
    hash bits, then run ``op(row_indices, hashes)`` for each partition on its own
    worker (partitions are independent — no locks). Returns each partition's
    output columns, in partition order. Encapsulates the hash/partition/dispatch/
    collect boilerplate shared by partition-parallel kernels (cf. ``HashJoin``).
    """
    var ctx = ExecutionContext.parallel(num_threads)
    var partitioner = RadixPartitioner(
        num_bits=_RADIX_BITS, num_threads=num_threads
    )
    var partitions = partitioner.partition(rapidhash(keys, ctx))
    var p = len(partitions)

    # Pre-sized slots — workers assign by index without racing on list growth.
    var slots = List[Optional[List[AnyArray]]](length=p, fill=None)

    @parameter
    def worker(i: Int) raises:
        slots[i] = op(
            partitions[i].row_indices.value().copy(),
            partitions[i].hashes.copy(),
        )

    sync_parallelize[worker](p)

    var out = List[List[AnyArray]]()
    for i in range(p):
        out.append(slots[i].value().copy())
    return out^


def _group_by_parallel[
    K: AggKernel
](keys: StructArray, value: AnyArray, num_threads: Int) raises -> RecordBatch:
    """Radix-partition-parallel grouped aggregation.

    A key lands in exactly one partition, so per-partition groups are disjoint —
    each partition groups + aggregates independently and the merge is a plain
    per-column ``concat``. Reuses the partitioner's hashes for grouping (no
    re-hash). The partition/parallel plumbing lives in ``_partition_columns``.
    """

    # Per-partition: group (reusing the partition hashes) + aggregate, returning
    # [key columns..., aggregate column].
    @parameter
    def aggregate_partition(
        rows: Int32Array, hashes: UInt64Array
    ) raises -> List[AnyArray]:
        var pkeys = take(keys, rows)
        var pvals = take(value, rows)

        var grouper = HashGrouper()
        var gids = grouper.consume_keys(pkeys, Optional(hashes.copy()))
        var ng = grouper.num_groups()

        var box = List[AnyArray]()

        @parameter
        def by_value[V: NumericType]() raises:
            var state = AggState[K, V]()
            state.update(gids, pvals.as_primitive[V](), ng)
            box.append(state.finish(ng).to_any())

        for_value_dtype[by_value](pvals.dtype())

        var cols = grouper.key_columns(grouper.key_fields(pkeys))
        cols.append(box[0].copy())
        return cols^

    var parts = _partition_columns[aggregate_partition](keys, num_threads)

    # Output schema: keys then the aggregate.
    ref kstruct = keys.dtype.as_struct()
    var num_cols = len(kstruct.fields) + 1
    var out_fields = List[Field]()
    for k in range(len(kstruct.fields)):
        out_fields.append(kstruct.fields[k].copy())
    var agg_box = List[AnyDataType]()

    @parameter
    def agg_dtype[V: NumericType]() raises:
        agg_box.append(AnyDataType(K.AccType[V]()))

    for_value_dtype[agg_dtype](value.dtype())
    out_fields.append(Field(K.name, agg_box[0].copy()))

    # Merge: concat each output column across partitions (groups are disjoint).
    var ctx = ExecutionContext.parallel(num_threads)
    var out_cols = List[AnyArray]()
    for c in range(num_cols):
        var chunks = List[AnyArray]()
        for i in range(len(parts)):
            chunks.append(parts[i][c].copy())
        out_cols.append(concat(chunks, ctx))

    return RecordBatch(schema=Schema(fields=out_fields^), columns=out_cols^)


def group_by[
    K: AggKernel
](
    key: AnyArray,
    value: AnyArray,
    ctx: ExecutionContext = ExecutionContext.serial(),
) raises -> RecordBatch:
    """``group_by[K]`` on a single key column."""
    var children = List[AnyArray]()
    children.append(key.copy())
    var key_data = key.to_data()
    var sa = StructArray(
        dtype=struct_(Field("key", key_data.dtype.copy())),
        length=key_data.length,
        nulls=key_data.nulls,
        offset=key_data.offset,
        bitmap=key_data.bitmap,
        children=children^,
    )
    return group_by[K](sa, value, ctx)
