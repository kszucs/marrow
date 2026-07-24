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

from ..arrays import AnyArray, StructArray, Int32Array
from .. import dtypes as dt
from ..schema import Schema
from ..tabular import RecordBatch
from ..kernels.concat import concat
from ..kernels.filter import filter
from ..kernels.groupby import HashGrouper
from ..kernels.aggregate import (
    AggKernel,
    AggState,
    for_agg_tag,
    agg_tag_from_name,
)
from ..dtypes import NumericType
from ..utils import dispatch_over_numeric
from ..kernels.join import HashJoin
from ..kernels.hashing import rapidhash
from ..parquet import (
    read_table,
    read_metadata,
    read_statistics,
    read_page_bounds,
    RowSelection,
    ColumnStatistics,
    PageBounds,
)
from ..scalars import AnyScalar
from .lane import AnyValue
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

    def _bounds_of(
        self, mins: List[Optional[AnyScalar]], maxs: List[Optional[AnyScalar]]
    ) raises -> PruneStats:
        """A `PruneStats` over this scan's schema from parallel min/max lists.
        """
        return PruneStats(Schema(copy=self._schema), mins.copy(), maxs.copy())

    def _row_group_survives(
        self, rg_stats: List[ColumnStatistics]
    ) raises -> Bool:
        """Whether the predicate might match some row of a group, from its
        per-column chunk statistics."""
        var mins = List[Optional[AnyScalar]]()
        var maxs = List[Optional[AnyScalar]]()
        for c in range(len(rg_stats)):
            mins.append(rg_stats[c].min.copy())
            maxs.append(rg_stats[c].max.copy())
        return (
            self._predicate.value()
            .prune(self._bounds_of(mins, maxs))
            .maybe_true
        )

    def _page_selection(
        self, rg_pages: List[List[PageBounds]], num_rows: Int
    ) raises -> RowSelection:
        """Rows of one group that survive page-level pruning: for each column
        with a page index, keep a page iff the predicate might match given that
        page's bounds (other columns unknown), then intersect the per-column
        selections. Pages of an unindexed column impose no restriction."""
        var ncols = len(self._schema.fields)
        var sel = RowSelection.all(num_rows)
        for c in range(ncols):
            ref pages = rg_pages[c]
            if len(pages) == 0:
                continue  # no page index for this column
            var keep = List[Bool]()
            var page_rows = List[Int]()
            for p in range(len(pages)):
                var mins = List[Optional[AnyScalar]]()
                var maxs = List[Optional[AnyScalar]]()
                for cc in range(ncols):
                    if cc == c:
                        mins.append(pages[p].min.copy())
                        maxs.append(pages[p].max.copy())
                    else:
                        mins.append(None)
                        maxs.append(None)
                keep.append(
                    self._predicate.value()
                    .prune(self._bounds_of(mins, maxs))
                    .maybe_true
                )
                page_rows.append(pages[p].num_rows)
            sel = sel.intersect(RowSelection.from_pages(keep, page_rows))
        return sel^

    def _read_plan(
        self,
    ) raises -> Tuple[Optional[List[Int]], Optional[List[RowSelection]]]:
        """The pushdown plan: which row groups to read and, when the page index
        lets the reader skip pages, a per-group row selection. Returns
        `(None, None)` with no predicate, and `(groups, None)` when only
        row-group skipping applies. Pruning is skipped for a nested file (leaf
        count != top-level column count) — those groups are kept whole."""
        if not self._predicate:
            return (None, None)
        var meta = read_metadata(self.path)
        var stats = read_statistics(self.path)
        var pages = read_page_bounds(self.path)
        var ncols = len(self._schema.fields)
        var groups = List[Int]()
        var selections = List[RowSelection]()
        var any_page_skip = False
        for rg in range(len(meta.row_groups)):
            var num_rows = meta.row_groups[rg].num_rows
            if len(stats[rg]) != ncols:
                groups.append(rg)
                selections.append(RowSelection.all(num_rows))
                continue
            if not self._row_group_survives(stats[rg]):
                continue
            groups.append(rg)
            var sel = self._page_selection(pages[rg], num_rows)
            if not sel.selects_all():
                any_page_skip = True
            selections.append(sel^)
        if any_page_skip:
            return (Optional(groups^), Optional(selections^))
        return (Optional(groups^), None)

    def pull(mut self) raises -> RecordBatch:
        if not self._batch:
            var plan = self._read_plan()
            var table = read_table(
                self.path,
                row_groups=plan[0].copy(),
                row_selections=plan[1].copy(),
            )
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
            var mask = self.predicate.execute(batch)
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
            cols.append(v.execute(batch))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


# ---------------------------------------------------------------------------
# Blocking processors
# ---------------------------------------------------------------------------


struct AggregateProcessor(Processor):
    """Blocking: drain all input, then aggregate each column once.

    Keys are grouped by a keys-only ``HashGrouper``. Because ``AggState[K, V]``
    is fully typed (no ``AnyBuilder``), it can't be stored across the runtime,
    heterogeneous aggregate set — so this blocking node buffers the per-batch
    group ids and value columns, then for each aggregate resolves ``(K, V)`` via
    the tag + input dtype and drives one typed ``AggState[K, V]`` over all
    batches. The typed hot loop is the trade for buffering the (already fully
    consumed) input.

    Runtime aggregate dispatch (the dynamic plan's ``name -> kernel`` selection)
    uses the shared ``agg_tag_from_name`` + ``for_agg_tag`` tag switch from
    ``marrow.kernels.aggregate`` — mirroring ``DynValue`` — so the typed
    ``AggState[K, V]`` hot loop carries no dispatch."""

    @staticmethod
    def out_dtype(
        tag: UInt8, value_dtype: dt.AnyDataType
    ) raises -> dt.AnyDataType:
        """Output/accumulator dtype for an aggregate tag on a given input dtype.
        """
        var box = List[dt.AnyDataType]()

        @parameter
        def by_kind[K: AggKernel]() raises:
            @parameter
            def by_value[V: NumericType](d: V) raises:
                box.append(dt.AnyDataType(K.AccType[V]()))

            dispatch_over_numeric[by_value](value_dtype)

        for_agg_tag[by_kind](tag)
        return box[0].copy()

    var input: AnyProcessor
    var keys: List[AnyValue]
    var aggs: List[AnyValue]
    var _schema: Schema
    var _grouper: HashGrouper
    var _tags: List[UInt8]
    var _value_dtypes: List[dt.AnyDataType]
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
    ) raises:
        self.input = input^
        self.keys = keys^
        self.aggs = aggs^
        self._schema = schema^
        self._grouper = HashGrouper()
        self._tags = List[UInt8]()
        self._value_dtypes = List[dt.AnyDataType]()
        for i in range(len(funcs)):
            self._tags.append(agg_tag_from_name(funcs[i]))
            self._value_dtypes.append(value_dtypes[i].copy())
        self._emitted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def _key_fields(self) -> List[dt.Field]:
        # Output schema is key fields then aggregate fields; the first
        # len(keys) fields are the group keys.
        var fields = List[dt.Field]()
        for i in range(len(self.keys)):
            fields.append(self._schema.fields[i].copy())
        return fields^

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()

        # Phase 1 — drain input, buffering per-batch group ids + value columns.
        var gids_per_batch = List[Int32Array]()
        var values_per_batch = List[List[AnyArray]]()
        while True:
            try:
                var batch = self.input.pull()
                var key_children = List[AnyArray]()
                for i in range(len(self.keys)):
                    key_children.append(self.keys[i].execute(batch))
                var key_struct = StructArray(
                    dtype=dt.struct_(self._key_fields()),
                    length=batch.num_rows(),
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    children=key_children^,
                )
                gids_per_batch.append(self._grouper.consume_keys(key_struct))
                var vals = List[AnyArray]()
                for i in range(len(self.aggs)):
                    vals.append(self.aggs[i].execute(batch))
                values_per_batch.append(vals^)
            except Exhausted:
                break
        self._emitted = True

        # Phase 2 — key columns + one typed AggState per aggregate.
        var kfields = self._key_fields()
        var cols = self._grouper.key_columns(kfields)
        var num_groups = self._grouper.num_groups()
        for i in range(len(self._tags)):
            cols.append(
                self._aggregate(i, gids_per_batch, values_per_batch, num_groups)
            )
        return RecordBatch(schema=self._schema.copy(), columns=cols^)

    def _aggregate(
        self,
        i: Int,
        gids_per_batch: List[Int32Array],
        values_per_batch: List[List[AnyArray]],
        num_groups: Int,
    ) raises -> AnyArray:
        """Drive one typed `AggState[K, V]` over all buffered batches for
        aggregate `i`, resolving `(K, V)` from its tag + input dtype."""
        var box = List[AnyArray]()

        @parameter
        def by_kind[K: AggKernel]() raises:
            @parameter
            def by_value[V: NumericType](d: V) raises:
                var state = AggState[K, V]()
                for b in range(len(gids_per_batch)):
                    state.update(
                        gids_per_batch[b],
                        values_per_batch[b][i].as_primitive[V](),
                        num_groups,
                    )
                box.append(state.finish(num_groups).to_any())

            dispatch_over_numeric[by_value](self._value_dtypes[i])

        for_agg_tag[by_kind](self._tags[i])
        return box[0].copy()


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
