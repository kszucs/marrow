# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a file scan reads its morsels from: the sources a scan lowers to.

A Parquet scan is a row group per morsel, an IPC scan a record batch per
morsel; each file's footer is read when the scan lowers, which is what a
morsel count needs. A JSON or Iceberg scan is a `Stream`, whose length only
reading tells. The Parquet source is also where pushed predicates prune:
they are decided once, against the footer's statistics and page index, before
any morsel is read.
"""

from ..arrays import StructArray
from ..engine.sources import Source, Stream
from ..errors import DynError
from ..execution import ExecContext
from ..formats import Format
from ..iceberg.catalog import IcebergTable
from ..iceberg.manifest import DataFile
from ..iceberg.planning import scan_tasks, snapshot_files
from ..iceberg.projection import NameMapping
from ..iceberg.scan import FileScanTask, IcebergFileReader
from ..io import DynSource, StorageOptions
from ..formats.parquet.reader import LeafSet, ParquetFile, RowSelection
from ..schema import Schema
from ..tabular import RecordBatch
from ..utils.stats import count
from .index import Index, page_selections
from .logical import Bindings, DynValue


@fieldwise_init
struct _RowGroup(Copyable, Movable):
    """One morsel of a `RowGroupSource`: a row group of one file, and the rows
    of it to decode, all of them when `selection` is `None`."""

    var file: Int
    var group: Int
    var selection: Optional[RowSelection]


struct _Opened(Movable):
    """One file of a `RowGroupSource`, the `position`-th, its footer read: the
    row groups left to read once the pushed predicates had their say."""

    var file: ParquetFile[DynSource, LeafSet.all()]
    var groups: List[_RowGroup]
    var rows: Int

    def __init__(
        out self,
        position: Int,
        path: String,
        schema: Schema,
        pushed: List[DynValue],
        bindings: Bindings,
        options: StorageOptions,
    ) raises:
        self.file = ParquetFile[DynSource, LeafSet.all()](
            DynSource.open(path, options), schema=schema.copy()
        )
        self.groups = List[_RowGroup]()
        self.rows = 0
        if len(pushed) == 0:
            # Nothing to prove, so nothing to decode. Statistics are per
            # `(row group x leaf)`, so building an index nobody queries is the
            # most expensive way to read every group.
            for rg in range(self.file.num_row_groups()):
                self.groups.append(_RowGroup(position, rg, None))
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
                self.groups.append(_RowGroup(position, kept[i], selection^))
        for ref g in self.groups:
            self.rows += self.file.metadata().row_groups[g.group].num_rows
        count["parquet.row_groups"](len(self.groups))
        count["parquet.row_groups_pruned"](
            self.file.num_row_groups() - len(self.groups)
        )


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
    var _files: List[_Opened]
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
                opened[f] = _Opened(
                    f, paths[f], schema, pushed, bindings, options
                )
            except e:
                raise DynError(e).prefixed(paths[f])

        ctx.fan_out(len(paths), open, len(paths))
        self._files = List[_Opened](capacity=len(paths))
        self._morsels = List[_RowGroup]()
        self._rows = 0
        self._groups = 0
        for f in range(len(opened)):
            var one = opened[f].take()
            self._rows += one.rows
            self._groups += one.file.num_row_groups()
            self._morsels.extend(one.groups.copy())
            self._files.append(one^)
        self._paths = paths^

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
            var table = self._files[morsel.file].file.read(
                columns=Optional(self._columns.copy()),
                row_groups=Optional(groups^),
                row_selections=picked,
                ctx=ctx,
            )
            # A row group can decode to several chunks; each is pushed on its
            # own rather than concatenated back together.
            for ref b in table.to_batches():
                out.append(b.to_struct_array())
        except e:
            raise DynError(e).prefixed(self._paths[morsel.file])
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


struct FormatSource[Fmt: Format](Source):
    """A `FormatScan[Fmt]`'s files, a chunk per morsel -- a record batch, a
    row group, a run of blocks, a range of lines -- with the columns of the
    scan's schema. Each file is opened when the source is built, reading only
    what tells where its chunks are."""

    var _paths: List[String]
    var _files: List[Self.Fmt.Chunks]
    var _names: List[String]
    var _morsels: List[Tuple[Int, Int]]
    """The file and the chunk within it of each morsel."""

    def __init__(
        out self,
        var paths: List[String],
        schema: Schema,
        options: StorageOptions,
        ctx: ExecContext,
    ) raises:
        """`paths`, each opened with `options` -- an object store's
        credentials, say -- one file per worker of `ctx`, since each may be a
        round trip."""
        self._names = schema.names()
        var opened = List[Optional[Self.Fmt.Chunks]](capacity=len(paths))
        for _ in range(len(paths)):
            opened.append(None)

        def open(w: Int, f: Int) raises {mut opened, imm}:
            try:
                opened[f] = Self.Fmt.chunks(
                    paths[f], Optional(schema.copy()), options
                )
            except e:
                raise DynError(e).prefixed(paths[f])

        ctx.fan_out(len(paths), open, len(paths))
        self._files = List[Self.Fmt.Chunks](capacity=len(paths))
        self._morsels = List[Tuple[Int, Int]]()
        for f in range(len(opened)):
            var file = opened[f].take()
            for c in range(file.num_chunks()):
                self._morsels.append((f, c))
            self._files.append(file^)
        self._paths = paths^

    def count(self) -> Int:
        return len(self._morsels)

    def rows(self) -> Int:
        return -1

    def read(self, i: Int, ctx: ExecContext) raises -> List[StructArray]:
        var f = self._morsels[i][0]
        var out = List[StructArray]()
        try:
            for ref batch in self._files[f].read_chunk(
                self._morsels[i][1], ctx
            ):
                out.append(batch.select(self._names).to_struct_array())
        except e:
            raise DynError(e).prefixed(self._paths[f])
        return out^

    def describe(self) -> String:
        return String(
            Self.Fmt.name(),
            "Scan(",
            self._paths,
            ", ",
            len(self._morsels),
            " chunks)",
        )


struct IcebergScanStream(Stream):
    """An `IcebergScan`'s row groups: one of one data file per `next`, in the
    scan's schema, over the data files `plan` chose."""

    var _tasks: List[FileScanTask]
    var _schema: Schema
    var _mapping: Optional[NameMapping]
    """The table's name mapping, for data files written without field ids;
    a table need not have one."""
    var _options: StorageOptions
    var _ctx: ExecContext
    var _next: Int
    var _reader: Optional[IcebergFileReader]
    """The data file being read; none between files."""

    def __init__(
        out self,
        var tasks: List[FileScanTask],
        var schema: Schema,
        var mapping: Optional[NameMapping],
        var options: StorageOptions,
        var ctx: ExecContext,
    ):
        self._tasks = tasks^
        self._schema = schema^
        self._mapping = mapping^
        self._options = options^
        self._ctx = ctx^
        self._next = 0
        self._reader = None

    @staticmethod
    def plan(
        table: IcebergTable,
        snapshot_id: Optional[Int],
        schema: Schema,
        pushed: List[DynValue],
        bindings: Bindings,
        ctx: ExecContext,
    ) raises -> List[FileScanTask]:
        """The data files of snapshot `snapshot_id` to read: every one but
        those the `pushed` predicates prove cannot match, from the column
        bounds and partition values its manifests record. The predicates are
        still evaluated by the `Filter` above, so skipping saves reads and
        never changes an answer."""
        if not snapshot_id:
            return []
        var files = snapshot_files(table, snapshot_id.value(), ctx)
        if len(pushed) == 0:
            return scan_tasks(table, files)
        # `Index.from_manifest` describes files of one partition spec, so the
        # files are pruned spec by spec and the verdicts put back in order.
        var keep = List[Bool](length=len(files.data), fill=False)
        var specs = List[Int]()
        for ref entry in files.data:
            if entry.spec_id not in specs:
                specs.append(entry.spec_id)
        for spec_id in specs:
            var positions = List[Int]()
            var group = List[DataFile]()
            for i in range(len(files.data)):
                if files.data[i].spec_id == spec_id:
                    positions.append(i)
                    group.append(files.data[i].data_file.copy())
            var index = Index.from_manifest(
                group, schema, table.metadata.spec(spec_id)
            )
            for chunk in index.read_plan(pushed, bindings):
                keep[positions[chunk]] = True
        return scan_tasks(table, files, keep^)

    def next(mut self) raises -> List[StructArray]:
        # A scan over many data files must say which of them failed.
        while True:
            if self._reader:
                var batch: Optional[RecordBatch]
                try:
                    batch = self._reader.value().next()
                except e:
                    raise DynError(e).prefixed(self._tasks[self._next - 1].path)
                if batch:
                    return [batch.value().to_struct_array()]
                self._reader = None
            if self._next >= len(self._tasks):
                return []
            try:
                self._reader = IcebergFileReader(
                    self._tasks[self._next],
                    self._schema,
                    self._mapping,
                    options=self._options,
                    ctx=self._ctx,
                )
            except e:
                raise DynError(e).prefixed(self._tasks[self._next].path)
            self._next += 1
