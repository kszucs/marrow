"""The execution layer for ``marrow.expr`` relational plans.

``Relation`` nodes (in ``relations.mojo``) are pure, immutable descriptions;
this module is where they actually *run*. ``Relation.to_processor(ctx)`` builds a
``Processor`` — the executing counterpart of a node — that owns all mutable
execution state (scan offset, built hash index, grouper, child processors) and
yields morsels through ``pull()``.

- ``Processor``     — the trait each ``*Processor`` implements (``schema``/``pull``).
- ``DynProcessor``  — the type-erased, move-only container; also the pull driver
                      (``collect()`` drains it into one ``RecordBatch``).
- ``InMemoryTableProcessor`` / ``ParquetScanProcessor`` / ``FilterProcessor`` /
  ``ProjectProcessor`` / ``AggregateProcessor`` / ``JoinProcessor`` — one per
  node kind.

This layer depends only on the value box (``DynValue``) and the kernels; it does
**not** import the ``Relation`` nodes, so the dependency is one-way
(``relations`` → ``execution``).
"""

from std.memory import ArcPointer


from ..arrays import DynArray, StructArray
from .. import dtypes as dt
from ..schema import Schema
from ..builders import Int32Builder
from ..tabular import RecordBatch
from ..kernels.concat import concat
from ..kernels.filter import filter
from ..kernels.sort import sort as sort_by_keys
from ..kernels.groupby import HashGrouper
from .aggregates import AggFunc
from ..kernels.join import (
    HashJoin,
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_LEFT,
    JOIN_SEMI,
    JoinKind,
)
from ..kernels.hashing import rapidhash
from ..parquet.source import MappedFile
from ..parquet import (
    LeafSet,
    ParquetFile,
    RowSelection,
    ColumnStatistics,
    PageBounds,
)
from ..scalars import DynScalar
from .relations import BoxedValue
from ..kernels.core import Grouping
from .pruning import PruneStats
from ..execution import ExecContext


comptime DEFAULT_MORSEL_SIZE: Int = 65_536

# How much decoded row-group data a Parquet scan may hold at once. This is the
# constant that makes the scan's memory independent of file size — see
# `ParquetScanProcessor._window_end` for why it is not one row group.
comptime _WINDOW_BYTES: Int = 64 * 1024 * 1024


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
# Processor trait + DynProcessor
# ---------------------------------------------------------------------------


trait Processor(Deinitable, Movable):
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


struct DynProcessor(Movable):
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

    def __deinit__(deinit self):
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
            return RecordBatch(schema=self.schema(), columns=List[DynArray]())
        if len(batches) == 1:
            return RecordBatch(copy=batches[0])
        var schema = batches[0].schema
        var num_cols = batches[0].num_columns()
        var result_cols = List[DynArray](capacity=num_cols)
        for c in range(num_cols):
            var col_arrays = List[DynArray](capacity=len(batches))
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


struct ParquetScanProcessor[leaves: LeafSet = LeafSet.all()](Processor):
    """Streams a Parquet file a **bounded window of row groups** at a time.

    The file is opened once, on the first pull, and stays open for the life of
    the scan; a window of row groups is decoded only when its rows are asked
    for, and is dropped as soon as the next window is needed. So the decoded
    data resident at any moment is bounded by `window x row_group_size` — a
    function of the machine and the writer's group size, **not of the file
    size**, which is the point.

    The window exists purely for throughput: `ParquetFile.read` parallelizes
    over the (row group x leaf) grid, and reading a single group offers only as
    many slots as the scan has columns (see `_window_size`). The groups are
    still handed out one at a time, so a morsel never straddles a row-group
    boundary. None of this changes the rows produced — `morsel_size` and the
    window are only how the pipeline is driven.

    The scan reads **only the columns in its own schema**: those names are pushed
    into the read as a projection, so unselected column chunks are never even
    touched. A scan's schema is therefore the projection, and that is what an
    optimizer rewrites to push one down.

    When a `predicate` is pushed down, row groups whose column statistics prove
    the predicate can never match are skipped (never decoded). The predicate is
    only a pruning hint — a `Filter` above the scan still applies it exactly — so
    pushdown only ever reduces I/O, never changes results."""

    var path: String
    var _schema: Schema
    var morsel_size: Int
    var _predicate: Optional[BoxedValue]
    var _file: Optional[ParquetFile[MappedFile, Self.leaves]]
    # The read plan, computed once when the file is opened: the row groups to
    # decode in order, and (only when page-level skipping applies) one row
    # selection per entry.
    var _groups: List[Int]
    var _group_bytes: List[Int]  # uncompressed size of each, for the window
    var _selections: Optional[List[RowSelection]]
    var _next_group: Int
    # The decoded row groups of the current window, which one is being handed
    # out, and how far into it we are.
    var _batches: List[RecordBatch]
    var _batch_idx: Int
    var _offset: Int

    def __init__(
        out self,
        *,
        var path: String,
        var schema: Schema,
        morsel_size: Int,
        var predicate: Optional[BoxedValue] = None,
    ):
        self.path = path^
        self._schema = schema^
        self.morsel_size = morsel_size
        self._predicate = predicate^
        self._file = None
        self._groups = List[Int]()
        self._group_bytes = List[Int]()
        self._selections = None
        self._next_group = 0
        self._batches = List[RecordBatch]()
        self._batch_idx = 0
        self._offset = 0

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def _columns(self) -> List[String]:
        """The projection pushed into the read: this scan's own column names."""
        var out = List[String](capacity=len(self._schema.fields))
        for ref f in self._schema.fields:
            out.append(f.name)
        return out^

    def _bounds_of(
        self, mins: List[Optional[DynScalar]], maxs: List[Optional[DynScalar]]
    ) raises -> PruneStats:
        """A `PruneStats` over this scan's schema from parallel min/max lists.
        """
        return PruneStats(Schema(copy=self._schema), mins.copy(), maxs.copy())

    def _leaf_map(self, file_schema: Schema) -> List[Int]:
        """This scan's columns as *file leaf* indices — which is what
        `statistics()` and `page_bounds()` are indexed by, while the predicate
        resolves against the scan's own schema. Empty when a column has no leaf
        of its own name, which is how a nested file turns pruning off rather
        than misaligning it against a projected schema."""
        var out = List[Int](capacity=len(self._schema.fields))
        for ref f in self._schema.fields:
            var i = file_schema.get_field_index(f.name)
            if i < 0:
                return List[Int]()
            out.append(i)
        return out^

    def _row_group_survives(
        self, rg_stats: List[ColumnStatistics], leaf_of: List[Int]
    ) raises -> Bool:
        """Whether the predicate might match some row of a group, from its
        per-column chunk statistics."""
        var mins = List[Optional[DynScalar]]()
        var maxs = List[Optional[DynScalar]]()
        for c in range(len(leaf_of)):
            mins.append(rg_stats[leaf_of[c]].min.copy())
            maxs.append(rg_stats[leaf_of[c]].max.copy())
        return (
            self._predicate.value()
            .prune(self._bounds_of(mins, maxs))
            .maybe_true
        )

    def _page_selection(
        self,
        rg_pages: List[List[PageBounds]],
        num_rows: Int,
        leaf_of: List[Int],
    ) raises -> RowSelection:
        """Rows of one group that survive page-level pruning: for each column
        with a page index, keep a page iff the predicate might match given that
        page's bounds (other columns unknown), then intersect the per-column
        selections. Pages of an unindexed column impose no restriction."""
        var ncols = len(leaf_of)
        var sel = RowSelection.all(num_rows)
        for c in range(ncols):
            ref pages = rg_pages[leaf_of[c]]
            if len(pages) == 0:
                continue  # no page index for this column
            var keep = List[Bool]()
            var page_rows = List[Int]()
            for p in range(len(pages)):
                var mins = List[Optional[DynScalar]]()
                var maxs = List[Optional[DynScalar]]()
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
        self, pf: ParquetFile[MappedFile, Self.leaves]
    ) raises -> Tuple[List[Int], Optional[List[RowSelection]]]:
        """The pushdown plan: which row groups to read, in order, and — when the
        page index lets the reader skip pages — a per-group row selection.
        Without a predicate every group is read whole. Pruning is skipped for a
        nested file (leaf count != top-level column count) and for a column the
        file has no leaf for; those groups are kept whole."""
        var meta = pf.metadata()
        var groups = List[Int]()
        if not self._predicate:
            for rg in range(len(meta.row_groups)):
                groups.append(rg)
            return (groups^, None)
        var file_schema = pf.schema()
        var leaf_of = self._leaf_map(file_schema)
        var stats = pf.statistics()
        var pages = pf.page_bounds()
        var nleaves = len(file_schema.fields)
        var selections = List[RowSelection]()
        var any_page_skip = False
        for rg in range(len(meta.row_groups)):
            var num_rows = meta.row_groups[rg].num_rows
            if len(leaf_of) == 0 or len(stats[rg]) != nleaves:
                groups.append(rg)
                selections.append(RowSelection.all(num_rows))
                continue
            if not self._row_group_survives(stats[rg], leaf_of):
                continue
            groups.append(rg)
            var sel = self._page_selection(pages[rg], num_rows, leaf_of)
            if not sel.selects_all():
                any_page_skip = True
            selections.append(sel^)
        if any_page_skip:
            return (groups^, Optional(selections^))
        return (groups^, None)

    def _open(mut self) raises:
        """Open the file and fix the read plan — once per scan, on first pull.

        Each of these used to construct its own `ParquetFile`: four memory maps
        and four footer parses for a single logical read. Now the file stays
        open for the life of the processor, which is also what lets row groups
        be decoded one at a time."""
        var pf = ParquetFile[MappedFile, Self.leaves](self.path)
        var plan = self._read_plan(pf)
        self._groups = plan[0].copy()
        self._selections = plan[1].copy()
        var meta = pf.metadata()
        self._group_bytes = List[Int](capacity=len(self._groups))
        for g in self._groups:
            self._group_bytes.append(meta.row_groups[g].total_byte_size)
        self._next_group = 0
        self._file = pf^

    def _window_end(self, start: Int) -> Int:
        """One past the last row group of the window beginning at `start`.

        As many groups as fit in `_WINDOW_BYTES` of decoded data, and always at
        least one — so a single group larger than the budget is still read,
        rather than the scan deadlocking on its own limit.

        Reading exactly one group per call was the obvious reading of "stream
        it", and it is 1.6x-4.7x slower than reading the file whole: every call
        re-pays the fixed cost of a `read` (plan, dispatch, join), and each one
        ends on a barrier whose stragglers cannot be filled by the next group's
        work. `ParquetFile.read` also parallelizes over the (row group x leaf)
        grid, so a one-group read of a 2-column projection has two slots to
        spread over every core.

        A byte budget is the honest form of the bound this scan exists for:
        resident decoded data is capped by a constant, independent of the file
        size. `total_byte_size` is the group's *uncompressed* size across all
        columns, so for a projection it over-estimates — the window is then
        smaller than it needed to be, which is the safe direction."""
        var used = 0
        var end = start
        while end < len(self._groups):
            var next = used + self._group_bytes[end]
            if end > start and next > _WINDOW_BYTES:
                break
            used = next
            end += 1
        return end

    def _load_next_window(mut self) raises:
        """Decode the next window of selected row groups, dropping the previous
        one. Raises `Exhausted` once every group has been handed out.

        The decoded groups are kept as *separate* batches and handed out one at
        a time, so a morsel still never straddles a row-group boundary — the
        window is purely how much is decoded per `read`, not how much is
        concatenated. Groups that decode to no rows (every page pruned) are
        dropped here so `pull` never yields an empty morsel."""
        while self._next_group < len(self._groups):
            var start = self._next_group
            var stop = self._window_end(start)
            self._next_group = stop
            var window = List[Int](capacity=stop - start)
            var selections = List[RowSelection]()
            for g in range(start, stop):
                window.append(self._groups[g])
                if self._selections:
                    selections.append(self._selections.value()[g].copy())
            var selection: Optional[List[RowSelection]] = None
            if self._selections:
                selection = Optional(selections^)
            var table = self._file.value().read(
                self._columns(), Optional(window^), selection^
            )
            var batches = List[RecordBatch]()
            for ref b in table.to_batches():
                if b.num_rows() > 0:
                    batches.append(RecordBatch(copy=b))
            if len(batches) == 0:
                continue
            self._batches = batches^
            self._batch_idx = 0
            self._offset = 0
            return
        raise Exhausted()

    def pull(mut self) raises -> RecordBatch:
        if not self._file:
            self._open()
        # Advance to the next decoded row group, then the next window, until
        # something has rows left or the file is drained (Exhausted propagates).
        while True:
            if self._batch_idx >= len(self._batches):
                self._load_next_window()
            elif self._offset >= self._batches[self._batch_idx].num_rows():
                self._batch_idx += 1
                self._offset = 0
            else:
                break
        ref batch = self._batches[self._batch_idx]
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

    var input: DynProcessor
    var predicate: BoxedValue

    def __init__(
        out self, *, var input: DynProcessor, var predicate: BoxedValue
    ):
        self.input = input^
        self.predicate = predicate^

    def schema(self) -> Schema:
        return self.input.schema()

    def pull(mut self) raises -> RecordBatch:
        # Skip morsels that filter to 0 rows; Exhausted propagates from input.
        while True:
            var batch = self.input.pull()
            var mask = self.predicate.execute(batch)
            var cols = List[DynArray]()
            for i in range(batch.num_columns()):
                cols.append(filter(batch.columns[i].copy(), mask.copy()))
            var result = RecordBatch(schema=batch.schema.copy(), columns=cols^)
            if result.num_rows() > 0:
                return result^


struct ProjectProcessor(Processor):
    """Evaluates each projected value against every input morsel."""

    var input: DynProcessor
    var values: List[BoxedValue]
    var _schema: Schema

    def __init__(
        out self,
        *,
        var input: DynProcessor,
        var values: List[BoxedValue],
        var schema: Schema,
    ):
        self.input = input^
        self.values = values^
        self._schema = schema^

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def pull(mut self) raises -> RecordBatch:
        var batch = self.input.pull()  # raises Exhausted when done
        var cols = List[DynArray]()
        for ref v in self.values:
            cols.append(v.execute(batch))
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


struct LimitProcessor(Processor):
    """Streaming row limit/offset: skip ``offset`` rows, then pass through at
    most ``length`` rows and stop.

    Slices morsels at the input boundary — a morsel that straddles the offset or
    the length boundary is sliced, so the total emitted row count is exactly
    ``min(length, available - offset)`` regardless of morsel size."""

    var input: DynProcessor
    var _schema: Schema
    var _skip: Int
    var _remaining: Int

    def __init__(
        out self, *, var input: DynProcessor, offset: Int, length: Int
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

    var input: DynProcessor
    var keys: List[BoxedValue]
    var ascending: List[Bool]
    var nulls_first: Bool
    var stable: Bool
    var limit: Optional[Int]
    var _schema: Schema
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: DynProcessor,
        var keys: List[BoxedValue],
        var ascending: List[Bool],
        nulls_first: Bool,
        stable: Bool,
        var limit: Optional[Int],
        var schema: Schema,
        var ctx: ExecContext,
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
        var children = List[DynArray]()
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
        var out_cols = List[DynArray]()
        for i in range(full.num_columns()):
            out_cols.append(ordered.field(num_keys + i))
        return RecordBatch(schema=self._schema.copy(), columns=out_cols^)


struct AggregateProcessor(Processor):
    """Blocking: drain all input, then compute each aggregate column once.

    Keys are grouped by a keys-only ``HashGrouper`` *as morsels arrive*, so the
    grouping is incremental; only the per-batch group ids and the evaluated value
    columns are buffered. On emit, each aggregate's chunks are ``concat``-ed once
    and handed to its ``Aggregation``, which already knows what it is: the
    routing that used to be a name comparison (bytewise string min/max, the
    temporal fold, the validity-only count, the distinct sketches) was decided
    when the plan was built. A fused plan therefore reaches ``AggState[K, V]``
    with nothing interpreted in between. Keys (``keys``) and aggregate inputs
    (``inputs``) are arbitrary ``DynValue`` expressions, evaluated per morsel.

    ``HAVING`` needs no node of its own: a ``Filter`` on top of the ``Aggregate``
    relation evaluates its predicate against the aggregate's *output* batch, so
    ``rel.aggregate(...).filter(col("total") > lit(10))`` is exactly a
    post-aggregate filter over the aggregate output schema."""

    var input: DynProcessor
    var keys: List[BoxedValue]
    var inputs: List[BoxedValue]
    var aggs: List[AggFunc]
    var _schema: Schema
    var _grouper: HashGrouper
    var _ctx: ExecContext
    var _emitted: Bool

    def __init__(
        out self,
        *,
        var input: DynProcessor,
        var keys: List[BoxedValue],
        var inputs: List[BoxedValue],
        var aggs: List[AggFunc],
        var schema: Schema,
        var ctx: ExecContext,
    ) raises:
        self.input = input^
        self.keys = keys^
        self.inputs = inputs^
        self.aggs = aggs^
        self._schema = schema^
        self._grouper = HashGrouper()
        self._ctx = ctx^
        self._emitted = False

    def schema(self) -> Schema:
        return Schema(copy=self._schema)

    def _key_fields(self) raises -> List[dt.Field]:
        """The group-key fields. The output schema is key fields then aggregate
        fields, so the first ``len(keys)`` fields are the group keys."""
        var fields = List[dt.Field]()
        for i in range(len(self.keys)):
            fields.append(self._schema.fields[i].copy())
        return fields^

    def pull(mut self) raises -> RecordBatch:
        if self._emitted:
            raise Exhausted()
        self._emitted = True

        # Phase 1 — drain the input, grouping the keys morsel by morsel and
        # buffering the group ids + the evaluated value columns.
        var keyless = len(self.keys) == 0
        var gid_chunks = List[DynArray]()
        var value_chunks = List[List[DynArray]]()
        for _ in range(len(self.inputs)):
            value_chunks.append(List[DynArray]())
        var morsels = 0
        while True:
            try:
                var batch = self.input.pull()
                morsels += 1
                if not keyless:
                    var key_children = List[DynArray]()
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
                    gid_chunks.append(self._grouper.consume_keys(key_struct))
                for i in range(len(self.inputs)):
                    value_chunks[i].append(self.inputs[i].execute(batch))
            except Exhausted:
                break

        if morsels == 0:
            return RecordBatch.empty(self._schema)

        if keyless:
            # ``SELECT agg(x), ...`` with no GROUP BY — one implicit group, so
            # there is nothing to hash: every row is group 0 and the aggregates
            # run through the same per-column entry point as a GROUP BY.
            #
            # Not `FoldedAggregates.whole`, which would be faster here (a vectorized
            # SIMD reduce rather than a scatter): reaching it from a *plan*
            # makes the whole name→aggregate catalog reachable from every plan,
            # and that measured +13% on the fused binary-size gate for a path a
            # keyed query never runs. The eager `RecordBatch.aggregate` binding
            # still takes the fast route.
            var values = List[DynArray]()
            for i in range(len(self.inputs)):
                values.append(concat(value_chunks[i], self._ctx))
                value_chunks[i].clear()

            var zeros = Int32Builder(values[0].length())
            for _ in range(values[0].length()):
                zeros.append(Int32(0))
            var group_zero = zeros.finish()

            var cols = List[DynArray]()
            for i in range(len(self.aggs)):
                cols.append(
                    self.aggs[i].grouped(
                        Grouping(group_zero.copy(), 1), values[i]
                    )
                )
            return RecordBatch(schema=self._schema.copy(), columns=cols^)

        # Phase 2 — the unique key columns, then one shared per-column aggregate
        # each. An aggregate's buffered chunks are released as soon as they are
        # concatenated, so only one aggregate's contiguous copy is ever live.
        var gids_any = concat(gid_chunks, self._ctx)
        var gids = gids_any.as_int32().copy()
        var num_groups = self._grouper.num_groups()
        var grouped_keys = self._grouper.key_columns(self._key_fields())
        var cols = List[DynArray]()
        for i in range(len(grouped_keys)):
            cols.append(grouped_keys[i].copy())
        for i in range(len(self.aggs)):
            var value = concat(value_chunks[i], self._ctx)
            value_chunks[i].clear()
            cols.append(
                self.aggs[i].grouped(Grouping(gids.copy(), num_groups), value)
            )
        return RecordBatch(schema=self._schema.copy(), columns=cols^)


struct JoinProcessor(Processor):
    """Builds the left side fully on first pull, then streams the right side."""

    var left: DynProcessor
    var right: DynProcessor
    var left_key_indices: List[Int]
    var right_key_indices: List[Int]
    var join_kind: JoinKind
    var strictness: UInt8
    var _schema: Schema
    var _index: Optional[HashJoin[rapidhash]]
    var _exhausted: Bool

    def __init__(
        out self,
        *,
        var left: DynProcessor,
        var right: DynProcessor,
        var left_key_indices: List[Int],
        var right_key_indices: List[Int],
        join_kind: JoinKind,
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

    @staticmethod
    def _blocks_on_probe_side(kind: JoinKind) -> Bool:
        """Whether this join kind's output depends on the *whole* probe side.

        LEFT/FULL emit build rows that no probe row matched, SEMI emits the
        build rows that some probe row matched, and ANTI emits the complement —
        all three are properties of every probe row taken together, not of one
        morsel. Probing morsel-by-morsel recomputes them per morsel, so LEFT,
        FULL and ANTI re-emit their tail once per morsel and SEMI emits a build
        row once per morsel that matches it.

        RIGHT is not here: its extra rows are unmatched *probe* rows, and each
        probe row belongs to exactly one morsel, so streaming is correct.
        """
        return (
            kind == JOIN_LEFT
            or kind == JOIN_FULL
            or kind == JOIN_SEMI
            or kind == JOIN_ANTI
        )

    def pull(mut self) raises -> RecordBatch:
        if self._exhausted:
            raise Exhausted()
        if not self._index:
            var left_struct = self.left.collect().to_struct_array()
            var index = HashJoin[rapidhash]()
            index.build(left_struct, self.left_key_indices)
            self._index = index^

        if JoinProcessor._blocks_on_probe_side(self.join_kind):
            # One probe over the whole probe side. The streaming alternative
            # needs the kernel to accumulate build-side matches across probes
            # and emit the tail on drain; until it does, this is the shape that
            # is correct.
            self._exhausted = True
            var right_all = self.right.collect()
            var blocked = self._index.value().probe(
                right_all.to_struct_array(),
                self.right_key_indices,
                self.join_kind,
                self.strictness,
            )
            return RecordBatch(
                schema=self._schema.copy(), columns=blocked.children.copy()
            )

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
