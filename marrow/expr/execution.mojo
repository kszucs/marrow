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
from ..kernels.sort import sort as sort_by_keys
from ..kernels.groupby import HashGrouper
from ..kernels.aggregate import (
    reinterpret_array,
    temporal_backing_dtype,
)
from .aggregates import agg_tag_from_name, aggregate_column
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
from .values import AnyValue
from .pruning import PruneStats
from ..kernels.execution import ExecutionContext


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


struct LimitProcessor(Processor):
    """Streaming row limit/offset: skip ``offset`` rows, then pass through at
    most ``length`` rows and stop.

    Slices morsels at the input boundary — a morsel that straddles the offset or
    the length boundary is sliced, so the total emitted row count is exactly
    ``min(length, available - offset)`` regardless of morsel size."""

    var input: AnyProcessor
    var _schema: Schema
    var _skip: Int
    var _remaining: Int

    def __init__(
        out self, *, var input: AnyProcessor, offset: Int, length: Int
    ):
        self.input = input^
        self._schema = self.input.schema()
        self._skip = offset
        self._remaining = length

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        # Loop over input morsels, dropping those fully inside the offset window;
        # Exhausted from the input propagates out.
        while True:
            if self._remaining <= 0:
                raise Exhausted()
            var batch = self.input.pull()
            var nrows = batch.num_rows()
            if self._skip >= nrows:
                self._skip -= nrows
            else:
                var start = self._skip
                self._skip = 0
                var avail = nrows - start
                var length = (
                    avail if avail < self._remaining else self._remaining
                )
                self._remaining -= length
                return batch.slice(start, length)


# ---------------------------------------------------------------------------
# Blocking processors
# ---------------------------------------------------------------------------


struct SortProcessor(Processor):
    """Blocking: buffer all input, sort by the key expressions, emit once.

    A pipeline breaker. It drains the child fully (``collect``), evaluates each
    key expression over the whole input, and prepends the resulting key columns
    to the data columns in a single ``StructArray``. ``kernels.sort.sort`` then
    orders that struct by the key field indices, gathering *every* field (keys
    and data) with ``take`` — so the trailing data fields come back permuted into
    sorted order. When ``limit`` is set the kernel's top-K path (a truncated
    permutation) runs instead of a full sort followed by a slice."""

    var input: AnyProcessor
    var keys: List[AnyValue]
    var ascending: List[Bool]
    var nulls_first: Bool
    var stable: Bool
    var limit: Optional[Int]
    var _schema: Schema
    var _ctx: ExecutionContext
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: AnyProcessor,
        var keys: List[AnyValue],
        var ascending: List[Bool],
        nulls_first: Bool,
        stable: Bool,
        var limit: Optional[Int],
        var schema: Schema,
        var ctx: ExecutionContext,
    ):
        self.input = input^
        self.keys = keys^
        self.ascending = ascending^
        self.nulls_first = nulls_first
        self.stable = stable
        self.limit = limit^
        self._schema = schema^
        self._ctx = ctx^
        self._emitted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()
        self._emitted = True

        var full = self.input.collect()
        var nrows = full.num_rows()
        if nrows == 0:
            # Nothing to order; return the (already schema-correct) empty batch.
            return full^

        # Build [key0, key1, ..., data0, data1, ...] in one StructArray. The key
        # columns are evaluated over the fully-drained input; the data columns
        # are the input columns unchanged.
        var num_keys = len(self.keys)
        var fields = List[dt.Field]()
        var children = List[AnyArray]()
        for i in range(num_keys):
            var kcol = self.keys[i].execute(full)
            fields.append(dt.Field("__sort_key" + String(i), kcol.dtype()))
            children.append(kcol^)
        for i in range(full.num_columns()):
            fields.append(self._schema.fields[i].copy())
            children.append(full.columns[i].copy())

        var key_struct = StructArray(
            dtype=dt.struct_(fields^),
            length=nrows,
            nulls=0,
            offset=0,
            bitmap=None,
            children=children^,
        )

        var key_indices = List[Int]()
        for i in range(num_keys):
            key_indices.append(i)

        var ordered = sort_by_keys(
            key_struct,
            key_indices=key_indices^,
            ascending=self.ascending.copy(),
            nulls_first=self.nulls_first,
            stable=self.stable,
            limit=self.limit.copy(),
            ctx=self._ctx,
        )

        # The trailing fields are the sorted data columns.
        var out_cols = List[AnyArray]()
        for i in range(full.num_columns()):
            out_cols.append(ordered.field(num_keys + i))
        return RecordBatch(schema=self._schema.copy(), columns=out_cols^)


struct AggregateProcessor(Processor):
    """Blocking: drain all input, then compute each aggregate column once.

    Keys are grouped by a keys-only ``HashGrouper`` *as morsels arrive*, so the
    grouping is incremental; only the per-batch group ids and the evaluated value
    columns are buffered. On emit, each aggregate's chunks are ``concat``-ed once
    and handed to ``aggregate_column`` — the same per-column entry point the
    runtime multi-aggregate driver uses — so the whole routing (distinct
    kernels, string/temporal min/max, non-numeric ``count``, typed ``AggState``
    folds) is shared rather than duplicated here. Keys and aggregate inputs are
    arbitrary ``AnyValue`` expressions, evaluated per morsel.

    ``HAVING`` needs no node of its own: a ``Filter`` on top of the ``Aggregate``
    relation evaluates its predicate against the aggregate's *output* batch, so
    ``rel.aggregate(...).filter(col("total") > lit(10))`` is exactly a
    post-aggregate filter over the aggregate output schema."""

    var input: AnyProcessor
    var keys: List[AnyValue]
    var aggs: List[AnyValue]
    var _schema: Schema
    var _grouper: HashGrouper
    var _tags: List[UInt8]
    var _ctx: ExecutionContext
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: AnyProcessor,
        var keys: List[AnyValue],
        var aggs: List[AnyValue],
        var funcs: List[String],
        var schema: Schema,
        var ctx: ExecutionContext,
    ) raises:
        self.input = input^
        self.keys = keys^
        self.aggs = aggs^
        self._schema = schema^
        self._grouper = HashGrouper()
        self._tags = List[UInt8]()
        for i in range(len(funcs)):
            self._tags.append(agg_tag_from_name(funcs[i]))
        self._ctx = ctx^
        self._emitted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def _key_fields(self) raises -> List[dt.Field]:
        """The group-key fields *as grouped*.

        Output schema is key fields then aggregate fields, so the first
        ``len(keys)`` fields are the group keys. A temporal key is grouped
        through its signed-integer backing (the same reinterpret idiom the
        aggregate kernels use for temporal min/max): key values are only ever
        compared for equality, the reinterpretation is exact and free, and the
        hash/scatter path is then fully numeric. ``_as_declared`` relabels the
        unique key column back to the schema's temporal dtype on emit."""
        var fields = List[dt.Field]()
        for i in range(len(self.keys)):
            ref f = self._schema.fields[i]
            if f.dtype.is_temporal():
                fields.append(dt.Field(f.name, temporal_backing_dtype(f.dtype)))
            else:
                fields.append(f.copy())
        return fields^

    @staticmethod
    def _as_grouped(var value: AnyArray) raises -> AnyArray:
        """An evaluated key column in the dtype it is grouped under — temporal
        reinterpreted to its integer backing, everything else unchanged."""
        var vdt = value.dtype()
        if vdt.is_temporal():
            return reinterpret_array(value, temporal_backing_dtype(vdt))
        return value^

    def _as_declared(self, i: Int, var value: AnyArray) raises -> AnyArray:
        """Inverse of ``_as_grouped`` — the unique key column ``i`` relabelled
        back to the output schema's dtype."""
        ref target = self._schema.fields[i].dtype
        if target.is_temporal():
            return reinterpret_array(value, target)
        return value^

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()
        self._emitted = True

        # Phase 1 — drain the input, grouping the keys morsel by morsel and
        # buffering the group ids + the evaluated value columns.
        var gid_chunks = List[AnyArray]()
        var value_chunks = List[List[AnyArray]]()
        for _ in range(len(self.aggs)):
            value_chunks.append(List[AnyArray]())
        while True:
            try:
                var batch = self.input.pull()
                var key_children = List[AnyArray]()
                for i in range(len(self.keys)):
                    key_children.append(
                        Self._as_grouped(self.keys[i].execute(batch))
                    )
                var key_struct = StructArray(
                    dtype=dt.struct_(self._key_fields()),
                    length=batch.num_rows(),
                    nulls=0,
                    offset=0,
                    bitmap=None,
                    children=key_children^,
                )
                gid_chunks.append(self._grouper.consume_keys(key_struct))
                for i in range(len(self.aggs)):
                    value_chunks[i].append(self.aggs[i].execute(batch))
            except Exhausted:
                break

        if len(gid_chunks) == 0:
            return RecordBatch.empty(self._schema)

        # Phase 2 — the unique key columns, then one shared per-column aggregate
        # each. An aggregate's buffered chunks are released as soon as they are
        # concatenated, so only one aggregate's contiguous copy is ever live.
        var gids_any = concat(gid_chunks, self._ctx)
        var gids = gids_any.as_int32().copy()
        var num_groups = self._grouper.num_groups()
        var grouped_keys = self._grouper.key_columns(self._key_fields())
        var cols = List[AnyArray]()
        for i in range(len(grouped_keys)):
            cols.append(self._as_declared(i, grouped_keys[i].copy()))
        for i in range(len(self._tags)):
            var value = concat(value_chunks[i], self._ctx)
            value_chunks[i].clear()
            cols.append(
                aggregate_column(gids, value, num_groups, self._tags[i])
            )
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
