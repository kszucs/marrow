"""Fused group-by + aggregation kernel.

ClickHouse-style two-phase architecture:
  1. **Phase 1** — hash keys, resolve every row to a group index
  2. **Phase 2** — each ``AggregateFunction`` scatter-updates per-group
     state in a single O(N) pass

Abstractions:
  - ``AggregateState``: per-group data held in an ``AnyBuilder``
    (one element per group). Type-erased — can hold any builder type.
  - ``AggregateFunction``: logic (create/add_batch/merge/insert_result_into)
    operating on ``AggregateState``. Defines how states are merged.
  - ``HashGrouper``: hash table + two-phase pipeline orchestration.
"""

from ..arrays import PrimitiveArray, StructArray, AnyArray
from ..builders import (
    PrimitiveBuilder,
    AnyBuilder,
    make_builder,
)
from ..dtypes import (
    DataType,
    Field,
    int64,
    uint32,
    uint64,
    float64,
    bool_,
    numeric_dtypes,
    primitive_dtypes,
    struct_,
)
from ..schema import Schema
from ..tabular import RecordBatch
from .hash_table import DictHashTable
from .hashing import hash_


# ---------------------------------------------------------------------------
# AggregateState — per-group data in a type-erased builder
# ---------------------------------------------------------------------------


struct AggregateState(Movable):
    """Per-group accumulator data stored as builder elements.

    Each group occupies one slot in the builder. ``AggregateFunction``
    creates new slots, scatter-updates existing ones, and finalizes
    the builder into the result column.

    Type-erased via ``AnyBuilder`` — the same abstraction works for
    float64 values (sum/min/max), int64 counts, or future list/string
    builders (collect/first/last).
    """

    var builder: AnyBuilder

    @implicit
    def __init__[T: DataType](out self, var builder: PrimitiveBuilder[T]):
        self.builder = AnyBuilder(builder^)

    def length(self) -> Int:
        return self.builder.length()

    def dtype(self) -> DataType:
        return self.builder.dtype()

    def finish(mut self) raises -> AnyArray:
        return self.builder.finish()


# ---------------------------------------------------------------------------
# AggregateFunction — logic operating on AggregateState
# ---------------------------------------------------------------------------


def _read_as_float64(col: AnyArray, row: Int) raises -> Float64:
    """Read any numeric element as Float64."""
    comptime for dt in primitive_dtypes:
        if col.dtype() == dt:
            return Float64(col.as_primitive[dt]().unsafe_get(row))
    raise Error("unsupported dtype for aggregation: ", col.dtype())


struct AggregateFunction(Copyable, Movable):
    """Logic for one aggregation, operating on ``AggregateState``.

    Follows ClickHouse's ``IAggregateFunction`` naming:
      - ``create()``             — init state for a new group
      - ``add_batch()``          — scatter-update from a batch (O(N))
      - ``merge()``              — combine partial states
      - ``insert_result_into()`` — finalize one group into output

    The function defines HOW states are merged. The state is just data.
    """

    var name: String
    var values: AggregateState  # per-group running value (float64)
    var counts: AggregateState  # per-group valid count (int64)

    def __init__(out self, name: String):
        self.name = name
        self.values = PrimitiveBuilder[float64]()
        self.counts = PrimitiveBuilder[int64]()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.values = PrimitiveBuilder[float64]()
        self.counts = PrimitiveBuilder[int64]()

    def num_groups(self) -> Int:
        return self.values.length()

    def create(mut self) raises:
        """Initialize state for a newly created group."""
        self.values.builder.as_primitive[float64]().append(
            Scalar[float64.native](0)
        )
        self.counts.builder.as_primitive[int64]().append(
            Scalar[int64.native](0)
        )

    def add_batch(
        mut self,
        group_ids: PrimitiveArray[uint32],
        input_col: AnyArray,
    ) raises:
        """Scatter-update: single O(N) pass over the batch."""
        var n = len(group_ids)
        var has_bitmap = input_col.null_count() > 0
        ref val_ptr = self.values.builder.as_primitive[float64]()
        ref cnt_ptr = self.counts.builder.as_primitive[int64]()

        for i in range(n):
            if has_bitmap and not input_col.is_valid(i):
                continue

            var g = Int(group_ids.unsafe_get(i))
            var cnt = Int(cnt_ptr._buffer.unsafe_get[int64.native](g))

            if self.name == "count":
                cnt_ptr._buffer.unsafe_set[int64.native](
                    g, Scalar[int64.native](cnt + 1)
                )
                continue

            var val = _read_as_float64(input_col, i)
            var cur = Float64(val_ptr._buffer.unsafe_get[float64.native](g))

            if self.name == "sum" or self.name == "mean":
                val_ptr._buffer.unsafe_set[float64.native](
                    g, Scalar[float64.native](cur + val)
                )
            elif self.name == "min":
                if cnt == 0 or val < cur:
                    val_ptr._buffer.unsafe_set[float64.native](
                        g, Scalar[float64.native](val)
                    )
            elif self.name == "max":
                if cnt == 0 or val > cur:
                    val_ptr._buffer.unsafe_set[float64.native](
                        g, Scalar[float64.native](val)
                    )

            cnt_ptr._buffer.unsafe_set[int64.native](
                g, Scalar[int64.native](cnt + 1)
            )

    def finish(
        mut self, col_name: String, num_groups: Int
    ) raises -> Tuple[Field, AnyArray]:
        """Finalize state into a result (field, column) pair."""
        if self.name == "count":
            return (
                Field(col_name, int64),
                self.counts.finish(),
            )

        if self.name == "mean":
            # Compute value / count for each group.
            var b = PrimitiveBuilder[float64](capacity=num_groups)
            ref val_ptr = self.values.builder.as_primitive[float64]()
            ref cnt_ptr = self.counts.builder.as_primitive[int64]()
            for g in range(num_groups):
                var c = Int(cnt_ptr._buffer.unsafe_get[int64.native](g))
                if c > 0:
                    var v = Float64(
                        val_ptr._buffer.unsafe_get[float64.native](g)
                    )
                    b.append(Scalar[float64.native](v / Float64(c)))
                else:
                    b.append_null()
            return (
                Field(col_name, float64),
                b.finish().to_any(),
            )

        # sum, min, max — emit value if count > 0, else null.
        var b = PrimitiveBuilder[float64](capacity=num_groups)
        ref val_ptr = self.values.builder.as_primitive[float64]()
        ref cnt_ptr = self.counts.builder.as_primitive[int64]()
        for g in range(num_groups):
            var c = Int(cnt_ptr._buffer.unsafe_get[int64.native](g))
            if c > 0:
                b.append(val_ptr._buffer.unsafe_get[float64.native](g))
            else:
                b.append_null()
        return (
            Field(col_name, float64),
            b.finish().to_any(),
        )


# ---------------------------------------------------------------------------
# HashGrouper — hash table + aggregate orchestration
# ---------------------------------------------------------------------------



def _concat_single(existing: AnyArray, single: AnyArray) raises -> AnyArray:
    """Append a length-1 array slice to an existing array."""
    var ab = make_builder(existing.dtype(), existing.length() + 1)
    ab.extend(existing)
    ab.extend(single)
    return ab.finish()


struct HashGrouper(Movable):
    """Hash-based grouped aggregation engine (ClickHouse-style).

    Two-phase batch consumption:
      Phase 1: hash keys → resolve per-row group indices, call
               ``create()`` on each ``AggregateFunction`` for new groups
      Phase 2: each ``AggregateFunction.add_batch()`` scatter-updates

    Supports incremental consumption: ``consume()`` can be called
    multiple times with successive batches. State accumulates.

    Uses ``DictHashTable`` — the same hash table struct used by ``HashJoin``.
    NULL keys are treated as equal (same group), consistent with SQL
    GROUP BY semantics (unlike join where NULL != NULL).
    """

    var _table: DictHashTable[hash_]
    var _group_keys: List[AnyArray]
    var _functions: List[AggregateFunction]

    def __init__(out self, agg_names: List[String]):
        self._table = DictHashTable[hash_]()
        self._group_keys = List[AnyArray]()
        self._functions = List[AggregateFunction]()
        for i in range(len(agg_names)):
            self._functions.append(AggregateFunction(agg_names[i]))

    def num_groups(self) -> Int:
        return self._table.num_buckets()

    def consume_keys(
        mut self, keys: StructArray
    ) raises -> PrimitiveArray[uint32]:
        """Hash keys and resolve group indices. Returns group_ids array.

        Can be called multiple times — groups accumulate across calls.
        New keys get new group IDs; existing keys return their previous ID.
        """
        var n = len(keys)
        if n == 0:
            var empty = PrimitiveBuilder[uint32](0)
            return empty.finish()

        var hashes = self._table.hash_keys(keys)

        var gid_builder = PrimitiveBuilder[uint32](capacity=n)
        for i in range(n):
            var h = UInt64(hashes.unsafe_get(i))
            var gid = self._probe_or_insert(keys, i, h)
            gid_builder.append(Scalar[uint32.native](gid))
        return gid_builder.finish()

    def consume_values(
        mut self,
        group_ids: PrimitiveArray[uint32],
        values: List[AnyArray],
    ) raises:
        """Scatter-update aggregate state using pre-resolved group_ids.

        Each AggregateFunction does a single O(N) pass over the batch.
        Values are processed and discarded — not stored.
        """
        for a in range(len(self._functions)):
            self._functions[a].add_batch(group_ids, values[a])

    def consume(mut self, keys: StructArray, values: List[AnyArray]) raises:
        """Convenience: consume_keys + consume_values in one call."""
        self.consume_values(self.consume_keys(keys), values)

    def finish(mut self, key_fields: List[Field]) raises -> RecordBatch:
        """Build result RecordBatch from key columns + finalized states."""
        var num_groups = self._table.num_buckets()
        var result_fields = List[Field]()
        var result_cols = List[AnyArray]()

        # Key columns.
        for k in range(len(key_fields)):
            result_fields.append(
                Field(key_fields[k].name, key_fields[k].dtype.copy())
            )
            if num_groups == 0:
                var empty = make_builder(key_fields[k].dtype.copy())
                result_cols.append(empty.finish())
            else:
                result_cols.append(self._group_keys[k].copy())

        # Aggregate columns.
        for a in range(len(self._functions)):
            var col_name = (
                String("col") + String(a) + "_" + self._functions[a].name
            )
            var pair = self._functions[a].finish(col_name, num_groups)
            result_fields.append(pair[0])
            result_cols.append(pair[1].copy())

        return RecordBatch(
            schema=Schema(fields=result_fields^),
            columns=result_cols^,
        )

    # --- hash table internals ---

    def _probe_or_insert(
        mut self, keys: StructArray, row: Int, h: UInt64
    ) raises -> Int:
        """Find existing group or insert new one.

        Uses DictHashTable.find_or_insert() — bucket_id IS the group_id.
        """
        var prev = self._table.num_buckets()
        var gid = self._table.find_or_insert(h)

        if gid < prev:
            return gid  # existing group

        # New group: store key row (one slice per column).
        if len(self._group_keys) == 0:
            for k in range(len(keys.children)):
                self._group_keys.append(keys.children[k].slice(row, 1).copy())
        else:
            for k in range(len(keys.children)):
                var new_val = _concat_single(
                    self._group_keys[k], keys.children[k].slice(row, 1)
                )
                self._group_keys[k] = new_val^

        for a in range(len(self._functions)):
            self._functions[a].create()

        return gid


# ---------------------------------------------------------------------------
# groupby — public API
# ---------------------------------------------------------------------------


def groupby(
    keys: StructArray,
    values: List[AnyArray],
    aggregations: List[String],
) raises -> RecordBatch:
    """Fused grouped aggregation on a struct array of keys.

    Supported aggregations: ``"sum"``, ``"min"``, ``"max"``,
    ``"count"``, ``"mean"``.

    Returns:
        RecordBatch with unique key columns + aggregated value columns.
    """
    if len(values) != len(aggregations):
        raise Error("groupby: len(values) != len(aggregations)")

    var grouper = HashGrouper(aggregations)
    grouper.consume(keys, values)

    var key_fields = List[Field]()
    for k in range(len(keys.dtype.fields)):
        key_fields.append(
            Field(
                keys.dtype.fields[k].name,
                keys.dtype.fields[k].dtype.copy(),
            )
        )

    return grouper.finish(key_fields)


def groupby(
    key: AnyArray,
    values: List[AnyArray],
    aggregations: List[String],
) raises -> RecordBatch:
    """Fused grouped aggregation on a single key column."""
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
    return groupby(sa, values, aggregations)
