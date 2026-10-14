# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Where a phase's morsels come from.

A `Source` declares how many morsels it has and reads one when asked. It never
decides which worker reads which: that is the `Driver`'s strategy. Sources are
erased by `DynSource`, because the kind of a plan's sources is chosen at run
time from the plan, and a closed `Variant` would link every reader into every
binary.
"""

from ..utils.stats import count
from std.atomic import Atomic
from std.memory import ArcPointer
from std.time import perf_counter_ns

from ..arrays import StructArray
from ..errors import DynError
from ..execution import ExecContext
from ..expr.index import Index, page_selections
from ..expr.logical import Bindings, DynValue
from ..expr.physical import DynOperator, JoinBuild, TraceEvent, Trail
from ..utils.threads import Thread
from ..io import DynSource as DynByteSource, StorageOptions
from ..formats.ipc import RecordBatchFileReader
from ..formats.parquet.reader import LeafSet, ParquetFile, RowSelection
from ..compress import Codecs
from ..schema import Schema


trait Source(Deinitable, Movable):
    """The morsels of one phase: how many there are, and each one's batches."""

    def count(self) -> Int:
        """How many morsels there are, or -1 when only reading to the end
        tells. A source of unknown count is read in order, on one thread."""
        ...

    def rows(self) -> Int:
        """How many rows the morsels hold in all, or -1 when unknown."""
        ...

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        """Morsel `i`'s batches, read on `ctx`. A source of unknown count
        answers its next morsel, whatever `i` is, and nothing past the last."""
        ...

    def describe(self) -> String:
        """One line naming the source, for a plan report."""
        ...


struct DynSource(Movable):
    """A `Source` of any kind, erased."""

    var _data: ArcPointer[NoneType]
    var _virt_count: def(ArcPointer[NoneType]) thin -> Int
    var _virt_rows: def(ArcPointer[NoneType]) thin -> Int
    var _virt_read: def(
        ArcPointer[NoneType], Int, ExecContext
    ) thin raises -> List[StructArray]
    var _virt_describe: def(ArcPointer[NoneType]) thin -> String
    var _virt_drop: def(var ArcPointer[NoneType]) thin
    """Releases the box's reference at the source's true type, so its
    destructor runs: an erased `ArcPointer` forgets it."""

    @staticmethod
    def _count_tramp[S: Source](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[S]](ptr)[].count()

    @staticmethod
    def _rows_tramp[S: Source](ptr: ArcPointer[NoneType]) -> Int:
        return rebind[ArcPointer[S]](ptr)[].rows()

    @staticmethod
    def _read_tramp[
        S: Source
    ](ptr: ArcPointer[NoneType], i: Int, ctx: ExecContext) raises -> List[
        StructArray
    ]:
        return rebind[ArcPointer[S]](ptr)[].read(i, ctx)

    @staticmethod
    def _describe_tramp[S: Source](ptr: ArcPointer[NoneType]) -> String:
        return rebind[ArcPointer[S]](ptr)[].describe()

    @staticmethod
    def _drop_tramp[S: Source](var ptr: ArcPointer[NoneType]):
        var typed = rebind[ArcPointer[S]](ptr)
        _ = ptr^
        _ = typed^

    @implicit
    def __init__[S: Source](out self, var value: S):
        var ptr = ArcPointer[S](value^)
        self._data = rebind[ArcPointer[NoneType]](ptr^)
        self._virt_count = Self._count_tramp[S]
        self._virt_rows = Self._rows_tramp[S]
        self._virt_read = Self._read_tramp[S]
        self._virt_describe = Self._describe_tramp[S]
        self._virt_drop = Self._drop_tramp[S]

    def __deinit__(deinit self):
        self._virt_drop(self._data^)

    def count(self) -> Int:
        return self._virt_count(self._data)

    def rows(self) -> Int:
        return self._virt_rows(self._data)

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return self._virt_read(self._data, i, ctx)

    def describe(self) -> String:
        return self._virt_describe(self._data)


struct TableSource(Source):
    """Batches already in memory, a few at a time: each morsel is some of
    them, or a slice of one."""

    var _morsels: List[List[StructArray]]

    def __init__(out self, var morsels: List[List[StructArray]]):
        self._morsels = morsels^

    @staticmethod
    def of(var batches: List[StructArray]) -> Self:
        """One morsel per batch."""
        var morsels = List[List[StructArray]](capacity=len(batches))
        for ref b in batches:
            morsels.append([b.copy()])
        return Self(morsels^)

    @staticmethod
    def sliced(batch: StructArray, morsel_rows: Int) -> Self:
        """`batch` in morsels of `morsel_rows` rows, the last one shorter.
        A slice shares the batch's buffers. An empty batch is one empty
        morsel, so its schema still reaches the stages that read it."""
        var n = len(batch)
        var morsels = List[List[StructArray]]()
        if n <= morsel_rows:
            morsels.append([batch.copy()])
        else:
            var start = 0
            while start < n:
                var length = min(morsel_rows, n - start)
                morsels.append([batch.slice(start, length)])
                start += length
        return Self(morsels^)

    def count(self) -> Int:
        return len(self._morsels)

    def rows(self) -> Int:
        var n = 0
        for ref m in self._morsels:
            for ref b in m:
                n += len(b)
        return n

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return self._morsels[i].copy()

    def describe(self) -> String:
        return String("TableSource(", self.rows(), " rows)")


struct ObservedSource(Source):
    """A source whose reads are timed, into a total several workers add to
    at once, and with a `trail` recorded as timeline slices: one slot per
    morsel, so the workers reading at once never write the same one."""

    var _source: DynSource
    var _ns: ArcPointer[Atomic[Int64]]
    var _trail: Optional[Trail]

    def __init__(
        out self,
        var source: DynSource,
        ns: ArcPointer[Atomic[Int64]],
        trail: Optional[Trail] = None,
    ):
        self._source = source^
        self._ns = ns
        self._trail = trail
        if self._trail:
            var count = self._source.count()
            for _ in range(max(0, count)):
                self._trail.value()[].append(TraceEvent("", 0, 0, -1, -1))

    def count(self) -> Int:
        return self._source.count()

    def rows(self) -> Int:
        return self._source.rows()

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var since = Int(perf_counter_ns())
        var batches = self._source.read(i, ctx)
        var ns = Int(perf_counter_ns()) - since
        _ = self._ns[].fetch_add(Int64(ns))
        if self._trail:
            var rows = 0
            for ref b in batches:
                rows += len(b)
            var event = TraceEvent(
                String("read morsel ", i), Thread.current_id(), since, ns, rows
            )
            ref trail = self._trail.value()[]
            if i < len(trail) and self._source.count() >= 0:
                trail[i] = event^
            else:
                # A source of unknown count is read on one thread, in order.
                trail.append(event^)
        return batches^

    def describe(self) -> String:
        return self._source.describe()


struct PlannedSource(Source):
    """Rows a dry run did not produce, where a phase would read them: no
    morsel holds anything, and the rows are an estimate."""

    var _name: String
    var _unit: String
    var _morsels: Int
    var _rows: Int

    def __init__(
        out self, var name: String, var unit: String, morsels: Int, rows: Int
    ):
        self._name = name^
        self._unit = unit^
        self._morsels = morsels
        self._rows = rows

    def count(self) -> Int:
        return self._morsels

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        return []

    def describe(self) -> String:
        if self._rows < 0:
            return String(self._name, "(", self._unit, " unknown)")
        return String(self._name, "(~", self._rows, " ", self._unit, " est)")


struct OperatorSource(Source):
    """A source operator — a scan — read in order: each morsel is the batch it
    drains next. Its count is unknown, so it is read on one thread."""

    var _op: ArcPointer[DynOperator]
    var _name: String

    def __init__(out self, var op: DynOperator, var name: String):
        self._op = ArcPointer(op^)
        self._name = name^

    def count(self) -> Int:
        return -1

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var batch = self._op[].drain()
        if batch:
            return [batch.value().struct_array()]
        return []

    def describe(self) -> String:
        return self._name.copy()


struct UnmatchedSource(Source):
    """The build rows a join's probe side owes once it has all run — the
    unmatched ones, or SEMI's matched ones: one morsel."""

    var _build: ArcPointer[JoinBuild]

    def __init__(out self, var build: ArcPointer[JoinBuild]):
        self._build = build^

    def count(self) -> Int:
        return 1

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var rows = self._build[].unmatched(ctx)
        if len(rows) == 0:
            return []
        return [rows^]

    def describe(self) -> String:
        return "Unmatched"


@no_inline
def _in_file(path: String, e: Error) -> DynError:
    """`e`, its message prefixed by the file it came from."""
    var err = DynError(e)
    return DynError(err.kind, t"{path}: {err.message}")


@fieldwise_init
struct _RowGroup(Copyable, Movable):
    """One morsel of a `RowGroupSource`: a row group of one file, and the rows
    of it to decode, all of them when `selection` is `None`."""

    var file: Int
    var group: Int
    var selection: Optional[RowSelection]


struct _Opened(Movable):
    """One file of a `RowGroupSource`, its footer read: the row groups left
    to read once the pushed predicates had their say."""

    var file: ParquetFile[DynByteSource, LeafSet.all()]
    var groups: List[_RowGroup]
    var rows: Int

    def __init__(
        out self,
        path: String,
        schema: Schema,
        pushed: List[DynValue],
        bindings: Bindings,
        options: StorageOptions,
    ) raises:
        self.file = ParquetFile[DynByteSource, LeafSet.all()](
            DynByteSource.open(path, options), schema=schema.copy()
        )
        self.groups = List[_RowGroup]()
        self.rows = 0
        if len(pushed) == 0:
            # Nothing to prove, so nothing to decode. Statistics are per
            # `(row group x leaf)`, so building an index nobody queries is the
            # most expensive way to read every group.
            for rg in range(self.file.num_row_groups()):
                self.groups.append(_RowGroup(0, rg, None))
        else:
            var index = Index.from_parquet(self.file)
            var kept = index.read_plan(pushed, bindings)
            # Then again, one level down: a surviving group still decodes
            # every row unless its page index says otherwise.
            var found = page_selections(
                self.file, index, kept, pushed, bindings
            )
            for i in range(len(kept)):
                var selection = Optional[RowSelection](None)
                if i < len(found) and not found[i].selects_all():
                    selection = found[i].copy()
                self.groups.append(_RowGroup(0, kept[i], selection^))
        for ref g in self.groups:
            self.rows += self.file.metadata().row_groups[g.group].num_rows
        count["parquet.row_groups"](len(self.groups))
        count["parquet.row_groups_pruned"](
            self.file.num_row_groups() - len(self.groups)
        )

    def into_file(deinit self) -> ParquetFile[DynByteSource, LeafSet.all()]:
        return self.file^


struct RowGroupSource(Source):
    """Parquet files, a row group per morsel.

    **Pruning is decided here, once, before any morsel is read.** With
    predicates pushed into the scan, a row group whose statistics prove no
    row matches is not a morsel at all, and a group that survives decodes
    only the pages its page index could not rule out. A predicate may name a
    `param`, so the decision takes this execution's bindings. Speed only:
    the `Filter` above still applies the predicate to every row read.

    **The schema doubles as the projection.** Only its columns are read, and
    it decides each string column's layout, leaf by leaf: a column declared
    `string_view` decodes straight into views.

    Each file's footer is read when the source is built, which is what the
    morsel count needs; no row group is decoded until it is read. Morsels may
    be read from several threads at once.
    """

    var _paths: List[String]
    var _files: List[ParquetFile[DynByteSource, LeafSet.all()]]
    var _columns: List[String]
    var _morsels: List[_RowGroup]
    var _rows: Int
    var _groups: Int
    """Row groups in the files, read or pruned."""

    def __init__(
        out self,
        var paths: List[String],
        schema: Schema,
        pushed: List[DynValue],
        bindings: Bindings,
        ctx: ExecContext,
        options: StorageOptions = StorageOptions(),
    ) raises:
        """`paths`, each opened with `options` -- an object store's
        credentials, say."""
        self._columns = schema.names()
        # Footers are fetched one file per worker, since each may be a round
        # trip; the morsels are numbered in file order after.
        var opened = List[Optional[_Opened]](capacity=len(paths))
        for _ in range(len(paths)):
            opened.append(None)

        def open(w: Int, f: Int) raises {mut opened, imm}:
            # A scan over many files must say which of them is missing or
            # does not match the schema.
            try:
                opened[f] = _Opened(paths[f], schema, pushed, bindings, options)
            except e:
                raise _in_file(paths[f], e)

        ctx.fan_out(len(paths), open, len(paths))
        self._files = List[ParquetFile[DynByteSource, LeafSet.all()]]()
        self._morsels = List[_RowGroup]()
        self._rows = 0
        self._groups = 0
        for f in range(len(opened)):
            var one = opened[f].take()
            self._rows += one.rows
            self._groups += one.file.num_row_groups()
            for ref g in one.groups:
                self._morsels.append(_RowGroup(f, g.group, g.selection.copy()))
            self._files.append(one^.into_file())
        self._paths = paths^
        # Workers read row groups at once, and a codec library's first open
        # must not race: it is opened here, on the calling thread.
        for i in range(len(self._files)):
            if self._files[i].needs_libs():
                Codecs.preload()
                break

    def count(self) -> Int:
        return len(self._morsels)

    def rows(self) -> Int:
        return self._rows

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        ref morsel = self._morsels[i]
        var groups: List[Int] = [morsel.group]
        var picked = Optional[List[RowSelection]](None)
        if morsel.selection:
            var one: List[RowSelection] = [morsel.selection.value().copy()]
            picked = Optional(one^)
        var out = List[StructArray]()
        try:
            var table = self._files[morsel.file].read(
                columns=Optional(self._columns.copy()),
                row_groups=Optional(groups^),
                row_selections=picked^,
                ctx=ctx,
            )
            # A row group can decode to several chunks; each is pushed on its
            # own rather than concatenated back together.
            for ref b in table.to_batches():
                out.append(b.to_struct_array())
        except e:
            raise _in_file(self._paths[morsel.file], e)
        return out^

    def describe(self) -> String:
        var pruned = String(" of ", self._groups) if self._groups > len(
            self._morsels
        ) else String("")
        return String(
            "ParquetScan(",
            self._paths,
            ", ",
            len(self._morsels),
            pruned,
            " row groups)",
        )


struct IpcSource(Source):
    """Arrow IPC files, a record batch per morsel, with the columns of the
    scan's schema. Each file's footer is read when the source is built."""

    var _paths: List[String]
    var _readers: List[RecordBatchFileReader[DynByteSource]]
    var _names: List[String]
    var _morsels: List[Tuple[Int, Int]]
    """The file and the batch within it of each morsel."""

    def __init__(out self, var paths: List[String], schema: Schema) raises:
        self._names = schema.names()
        self._readers = List[RecordBatchFileReader[DynByteSource]]()
        self._morsels = List[Tuple[Int, Int]]()
        for f in range(len(paths)):
            try:
                var reader = RecordBatchFileReader[DynByteSource](
                    DynByteSource.open(paths[f])
                )
                for b in range(reader.num_record_batches()):
                    self._morsels.append((f, b))
                self._readers.append(reader^)
            except e:
                raise _in_file(paths[f], e)
        self._paths = paths^

    def count(self) -> Int:
        return len(self._morsels)

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var f = self._morsels[i][0]
        var batch = self._readers[f].read_batch(self._morsels[i][1])
        return [batch.select(self._names).to_struct_array()]

    def describe(self) -> String:
        return String(
            "IpcScan(", self._paths, ", ", len(self._morsels), " batches)"
        )
