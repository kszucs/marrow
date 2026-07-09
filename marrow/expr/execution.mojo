"""The execution layer for ``marrow.expr`` relational plans.

``Relation`` nodes (in ``relations.mojo``) are pure, immutable descriptions;
this module is where they actually *run*. ``Relation.to_processor(ctx)`` builds a
``Processor`` — the executing counterpart of a node — that owns all mutable
execution state (scan offset, built hash index, grouper, child processors) and
yields morsels through ``pull()``.

- ``Processor``     — the trait each ``*Processor`` implements (``schema``/``pull``).
- ``AnyProcessor``  — the type-erased, move-only container; also the pull driver
                      (``collect()`` drains it into one ``RecordBatch``).
- ``InMemoryTableProcessor`` / ``ParquetScanProcessor`` / ``FilterProcessor`` /
  ``ProjectProcessor`` / ``AggregateProcessor`` / ``JoinProcessor`` — one per
  node kind.

This layer depends only on the value box (``AnyValue``) and the kernels; it does
**not** import the ``Relation`` nodes, so the dependency is one-way
(``relations`` → ``execution``).
"""

from std.memory import ArcPointer

from ..arrays import AnyArray, StructArray
from .. import dtypes as dt
from ..schema import Schema
from ..tabular import RecordBatch
from ..kernels.concat import concat
from ..kernels.filter import filter
from ..kernels.groupby import HashGrouper
from ..kernels.join import HashJoin
from ..kernels.hashing import rapidhash
from ..parquet import read_table, read_statistics
from ..scalars import AnyScalar
from .values import AnyValue
from .pruning import PruneStats


comptime DEFAULT_MORSEL_SIZE: Int = 65_536


# ---------------------------------------------------------------------------
# Exhausted — signals a processor has no more morsels
# ---------------------------------------------------------------------------


struct Exhausted(TrivialRegisterPassable, Writable):
    """Raised by ``pull()`` when a processor has no more batches to yield."""

    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Exhausted")


# ---------------------------------------------------------------------------
# Processor trait + AnyProcessor
# ---------------------------------------------------------------------------


trait Processor(ImplicitlyDeletable, Movable):
    """The executing counterpart of a ``Relation`` node.

    Created by ``Relation.to_processor(ctx)``; owns all mutable execution state — scan
    offset, built hash index, grouper, child processors. ``pull()`` yields the
    next morsel or raises ``Exhausted``. Processors are single-use and move-only;
    the descriptive plan they were opened from is never touched."""

    def schema(self) -> Schema:
        ...

    def pull(mut self) raises -> RecordBatch:
        """Yield the next morsel, or raise ``Exhausted`` when done."""
        ...


struct AnyProcessor(Movable):
    """Type-erased, move-only processor behind an ``ArcPointer``; also the pull
    driver (``collect()`` drains it into one ``RecordBatch``)."""

    var _data: ArcPointer[NoneType]
    var _virt_schema: def(ArcPointer[NoneType]) thin -> Schema
    var _virt_pull: def(ArcPointer[NoneType]) thin raises -> RecordBatch
    var _virt_drop: def(var ArcPointer[NoneType]) thin

    @staticmethod
    def _tramp_schema[T: Processor](ptr: ArcPointer[NoneType]) -> Schema:
        return rebind[ArcPointer[T]](ptr)[].schema()

    @staticmethod
    def _tramp_pull[
        T: Processor
    ](ptr: ArcPointer[NoneType]) raises -> RecordBatch:
        return rebind[ArcPointer[T]](ptr)[].pull()

    @staticmethod
    def _tramp_drop[T: Processor](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[T]](ptr^)
        _ = typed^

    @implicit
    def __init__[T: Processor](out self, var value: T):
        var ptr = ArcPointer(value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_schema = Self._tramp_schema[T]
        self._virt_pull = Self._tramp_pull[T]
        self._virt_drop = Self._tramp_drop[T]

    def __del__(deinit self):
        self._virt_drop(self._data^)

    def schema(self) -> Schema:
        return self._virt_schema(self._data)

    def pull(mut self) raises -> RecordBatch:
        return self._virt_pull(self._data)

    def collect(mut self) raises -> RecordBatch:
        """Drain all morsels into a single RecordBatch."""
        var batches = List[RecordBatch]()
        while True:
            try:
                batches.append(self.pull())
            except Exhausted:
                break
        if len(batches) == 0:
            return RecordBatch(schema=self.schema(), columns=List[AnyArray]())
        if len(batches) == 1:
            return RecordBatch(copy=batches[0])
        var schema = batches[0].schema
        var num_cols = batches[0].num_columns()
        var result_cols = List[AnyArray](capacity=num_cols)
        for c in range(num_cols):
            var col_arrays = List[AnyArray](capacity=len(batches))
            for b in range(len(batches)):
                col_arrays.append(batches[b].columns[c].copy())
            result_cols.append(concat(col_arrays))
        return RecordBatch(schema=Schema(copy=schema), columns=result_cols^)


# ---------------------------------------------------------------------------
# Leaf processors
# ---------------------------------------------------------------------------


struct InMemoryTableProcessor(Processor):
    """Yields morsel-sized slices of an in-memory batch."""

    var batch: RecordBatch
    var morsel_size: Int
    var _offset: Int

    def __init__(out self, *, var batch: RecordBatch, morsel_size: Int):
        self.batch = batch^
        self.morsel_size = morsel_size
        self._offset = 0

    def schema(self) -> Schema:
        return Schema(copy=self.batch.schema)

    def pull(mut self) raises -> RecordBatch:
        if self._offset >= self.batch.num_rows():
            raise Exhausted()
        var remaining = self.batch.num_rows() - self._offset
        var length = (
            self.morsel_size if self.morsel_size < remaining else remaining
        )
        var result = self.batch.slice(self._offset, length)
        self._offset += length
        return result^


struct ParquetScanProcessor(Processor):
    """Reads the Parquet file on first pull, then yields morsels.

    When a `predicate` is pushed down, row groups whose column statistics prove
    the predicate can never match are skipped (never decoded). The predicate is
    only a pruning hint — a `Filter` above the scan still applies it exactly — so
    pushdown only ever reduces I/O, never changes results."""

    var path: String
    var _schema: Schema
    var morsel_size: Int
    var _predicate: Optional[AnyValue]
    var _batch: Optional[RecordBatch]
    var _offset: Int

    def __init__(
        out self,
        *,
        var path: String,
        var schema: Schema,
        morsel_size: Int,
        var predicate: Optional[AnyValue] = None,
    ):
        self.path = path^
        self._schema = schema^
        self.morsel_size = morsel_size
        self._predicate = predicate^
        self._batch = None
        self._offset = 0

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def _surviving_row_groups(self) raises -> Optional[List[Int]]:
        """Row groups that may contain a matching row, by pruning on per-column
        min/max statistics. Returns None (read everything) when there is no
        predicate or the file is nested (leaf count != top-level column count),
        which this row-group pruner does not map."""
        if not self._predicate:
            return None
        var stats = read_statistics(self.path)
        var ncols = len(self._schema.fields)
        var keep = List[Int]()
        for rg in range(len(stats)):
            ref rg_stats = stats[rg]
            if len(rg_stats) != ncols:
                keep.append(rg)  # nested layout: cannot prune -> keep
                continue
            var mins = List[Optional[AnyScalar]]()
            var maxs = List[Optional[AnyScalar]]()
            for c in range(ncols):
                mins.append(rg_stats[c].min.copy())
                maxs.append(rg_stats[c].max.copy())
            var pstats = PruneStats(Schema(copy=self._schema), mins^, maxs^)
            if self._predicate.value().prune_bound(pstats).maybe_true:
                keep.append(rg)
        return keep^

    def pull(mut self) raises -> RecordBatch:
        if not self._batch:
            var rgs = self._surviving_row_groups()
            var table = read_table(self.path, row_groups=rgs^)
            var batches = table.to_batches()
            if len(batches) == 0:
                self._batch = RecordBatch(
                    schema=table.schema, columns=List[AnyArray]()
                )
            elif len(batches) == 1:
                self._batch = RecordBatch(copy=batches[0])
            else:
                var num_cols = batches[0].num_columns()
                var cols = List[AnyArray](capacity=num_cols)
                for c in range(num_cols):
                    var col_arrays = List[AnyArray](capacity=len(batches))
                    for b in range(len(batches)):
                        col_arrays.append(batches[b].columns[c].copy())
                    cols.append(concat(col_arrays))
                self._batch = RecordBatch(
                    schema=Schema(copy=batches[0].schema), columns=cols^
                )
        ref batch = self._batch.value()
        if self._offset >= batch.num_rows():
            raise Exhausted()
        var remaining = batch.num_rows() - self._offset
        var length = (
            self.morsel_size if self.morsel_size < remaining else remaining
        )
        var result = batch.slice(self._offset, length)
        self._offset += length
        return result^


# ---------------------------------------------------------------------------
# Streaming processors
# ---------------------------------------------------------------------------


struct FilterProcessor(Processor):
    """Keeps rows where the predicate is True; skips empty morsels."""

    var input: AnyProcessor
    var predicate: AnyValue

    def __init__(out self, *, var input: AnyProcessor, var predicate: AnyValue):
        self.input = input^
        self.predicate = predicate^

    def schema(self) -> Schema:
        return self.input.schema()

    def pull(mut self) raises -> RecordBatch:
        # Skip morsels that filter to 0 rows; Exhausted propagates from input.
        while True:
            var batch = self.input.pull()
            var mask = self.predicate.to_array(batch)
            var cols = List[AnyArray]()
            for i in range(batch.num_columns()):
                cols.append(filter(batch.columns[i].copy(), mask.copy()))
            var result = RecordBatch(schema=batch.schema.copy(), columns=cols^)
            if result.num_rows() > 0:
                return result^


struct ProjectProcessor(Processor):
    """Evaluates each projected value against every input morsel."""

    var input: AnyProcessor
    var values: List[AnyValue]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: AnyProcessor,
        var values: List[AnyValue],
        var schema: Schema,
    ):
        self.input = input^
        self.values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        var batch = self.input.pull()  # raises Exhausted when done
        var cols = List[AnyArray]()
        for ref v in self.values:
            cols.append(v.to_array(batch))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


# ---------------------------------------------------------------------------
# Blocking processors
# ---------------------------------------------------------------------------


struct AggregateProcessor(Processor):
    """Blocking: consume all input into a grouper, then emit once."""

    var input: AnyProcessor
    var keys: List[AnyValue]
    var aggs: List[AnyValue]
    var _schema: Schema
    var _grouper: HashGrouper
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: AnyProcessor,
        var keys: List[AnyValue],
        var aggs: List[AnyValue],
        var funcs: List[String],
        var value_dtypes: List[dt.AnyDataType],
        var schema: Schema,
    ):
        self.input = input^
        self.keys = keys^
        self.aggs = aggs^
        self._schema = schema^
        self._grouper = HashGrouper(funcs^, value_dtypes^)
        self._emitted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def _key_fields(self) -> List[dt.Field]:
        # The output schema is key fields followed by aggregate fields, so the
        # first len(keys) fields are the group keys.
        var fields = List[dt.Field]()
        for i in range(len(self.keys)):
            fields.append(self._schema.fields[i].copy())
        return fields^

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()
        while True:
            try:
                var batch = self.input.pull()
                var key_children = List[AnyArray]()
                for i in range(len(self.keys)):
                    key_children.append(self.keys[i].to_array(batch))
                var key_struct = StructArray(
                    dtype=dt.struct_(self._key_fields()),
                    length=batch.num_rows(),
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    children=key_children^,
                )
                var gids = self._grouper.consume_keys(key_struct)
                var val_arrays = List[AnyArray]()
                for i in range(len(self.aggs)):
                    val_arrays.append(self.aggs[i].to_array(batch))
                self._grouper.consume_values(gids, val_arrays)
            except Exhausted:
                break
        self._emitted = True
        # The grouper generates its own aggregate column names (col{i}_{func}).
        # Re-label with the plan's declared schema so plan.schema() matches the
        # executed output exactly (columns are in the same key-then-agg order).
        var raw = self._grouper.finish(self._key_fields())
        var cols = List[AnyArray]()
        for ref c in raw.columns:
            cols.append(c.copy())
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


struct JoinProcessor(Processor):
    """Builds the left side fully on first pull, then streams the right side."""

    var left: AnyProcessor
    var right: AnyProcessor
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: UInt8
    var strictness: UInt8
    var _schema: Schema
    var _index: Optional[HashJoin[rapidhash]]
    var _exhausted: Bool

    def __init__(
        out self,
        *,
        var left: AnyProcessor,
        var right: AnyProcessor,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        join_kind: UInt8,
        strictness: UInt8,
        var schema: Schema,
    ):
        self.left = left^
        self.right = right^
        self.left_key_indices = left_key_indices^
        self.right_key_indices = right_key_indices^
        self.join_kind = join_kind
        self.strictness = strictness
        self._schema = schema^
        self._index = None
        self._exhausted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        if self._exhausted:
            raise Exhausted()
        if not self._index:
            var left_struct = self.left.collect().to_struct_array()
            var index = HashJoin[rapidhash]()
            index.build(left_struct, self.left_key_indices)
            self._index = index^
        try:
            var right_morsel = self.right.pull()
            var result = self._index.value().probe(
                right_morsel.to_struct_array(),
                self.right_key_indices,
                self.join_kind,
                self.strictness,
            )
            return RecordBatch(
                schema=self._schema.copy(), columns=result.children.copy()
            )
        except Exhausted:
            self._exhausted = True
        raise Exhausted()
