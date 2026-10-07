# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Reading one Iceberg data file as rows of the table.

Scan planning (the spec's "Scan Planning") turns a snapshot into
`FileScanTask`s: a live data file and the delete files that apply to it.
`IcebergFileReader` executes one task — it opens the data file, drops the
rows its position deletes name, and projects what is left onto the table
schema by field id (`SchemaProjection`), one row group at a time.

A table that was copied or moved still names files under its old
`location`. Planning relocates every path a task opens — `FileScanTask.path`
and its delete files' `file_path` — and keeps the data file's `file_path` as
recorded, because that is what a position delete file's `file_path` column
names it by.

[Reference](https://iceberg.apache.org/spec/#scan-planning)
"""

from std.builtin.sort import sort

from ..arrays import BoolArray, DynArray
from ..scalars import DynScalar
from ..builders import BoolBuilder
from ..dtypes import field_index
from ..errors import InvalidError, NotImplementedError
from ..execution import ExecContext
from ..io import DynSource
from ..io.uri import StorageOptions
from ..kernels.filter import filter
from ..parquet.reader import LeafSet, ParquetFile, RowSelection
from ..schema import Schema
from ..tabular import RecordBatch

from .manifest import DataFile
from .metadata import PartitionSpec
from .projection import NameMapping, SchemaProjection


comptime POSITION_DELETE_FILE_PATH_ID = 2147483546
"""The reserved field id of a position delete file's `file_path` column."""
comptime POSITION_DELETE_POS_ID = 2147483545
"""The reserved field id of a position delete file's `pos` column."""


@fieldwise_init
struct FileScanTask(Copyable, Movable):
    """One data file to read, with everything needed to read it correctly.

    `position_deletes` and `equality_deletes` are the delete files scan
    planning found to apply, with their `file_path` already relocated; the
    reader does not re-check that they apply. Equality deletes are not
    implemented yet: a task carrying any is refused.
    """

    var file: DataFile
    """The data file, its `file_path` as its manifest records it."""
    var path: String
    """Where the data file is opened from: `file.file_path`, relocated."""
    var spec: PartitionSpec
    """The spec the file was written with; `file.partition` follows it."""
    var position_deletes: List[DataFile]
    var equality_deletes: List[DataFile]


def read_position_deletes(
    files: List[DataFile],
    data_path: String,
    options: StorageOptions = StorageOptions(),
    ctx: ExecContext = ExecContext.auto(),
) raises -> List[Int]:
    """The row positions `files` delete from the data file `data_path`,
    ascending and without duplicates.

    Each of `files` is opened from its `file_path`. `data_path` is the data
    file's path as its manifest records it, which is how a delete file's
    `file_path` column names it; rows for other files are ignored. Columns
    are found by their reserved field ids, so the optional `row` column is
    never read.
    """
    var positions = List[Int]()
    for ref f in files:
        if f.content != DataFile.POSITION_DELETES:
            raise InvalidError(
                t"iceberg: {f.file_path} is not a position delete file"
            )
        if f.file_format != "PARQUET":
            raise NotImplementedError(
                t"iceberg: position delete file format '{f.file_format}'"
            )
        var file = ParquetFile[DynSource, LeafSet.all()](
            DynSource.open(f.file_path, options)
        )
        var schema = file.schema()
        var path_index = field_index(
            schema.fields, POSITION_DELETE_FILE_PATH_ID
        )
        var pos_index = field_index(schema.fields, POSITION_DELETE_POS_ID)
        if path_index < 0 or pos_index < 0:
            raise InvalidError(
                t"iceberg: {f.file_path} lacks the file_path and pos columns"
                t" of a position delete file"
            )
        if file.num_rows() == 0:
            continue
        var columns: List[String] = [
            schema.fields[path_index].name,
            schema.fields[pos_index].name,
        ]
        var batch = file.read(columns=columns^, ctx=ctx).combine_chunks()
        ref paths = batch.column(0).as_string()
        ref pos = batch.column(1).as_int64()
        for i in range(batch.num_rows()):
            if paths.is_null(i) or pos.is_null(i):
                continue
            if paths.unsafe_get(UInt(i)) == data_path:
                positions.append(Int(pos.unsafe_get(i)))
    sort(positions)
    var unique = List[Int](capacity=len(positions))
    for p in positions:
        if len(unique) == 0 or unique[len(unique) - 1] != p:
            unique.append(p)
    return unique^


def _lower_bound(values: List[Int], value: Int) -> Int:
    """The index of the first of the ascending `values` not below `value`."""
    var lo = 0
    var hi = len(values)
    while lo < hi:
        var mid = (lo + hi) // 2
        if values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


struct IcebergFileReader(Movable):
    """Reads one `FileScanTask` as projected `RecordBatch`es, one row group
    per `next()`.

    Position deletes are applied by filtering each decoded row group with a
    mask rather than by folding them into a Parquet `RowSelection`: the
    reader refuses a selection on a repeated column, and a table with a list
    or map column must still honour its deletes. A row group whose rows are
    all deleted is never decoded.

    `row_groups` and `row_selections` pass through to `ParquetFile.read`, so
    a caller that pruned row groups or pages reads only those; positions are
    still matched against the rows each selection keeps.
    """

    var _file: ParquetFile[DynSource, LeafSet.all()]
    var _projection: SchemaProjection
    var _columns: List[String]
    var _deleted: List[Int]
    """File row positions to drop, ascending and unique."""
    var _row_groups: List[Int]
    var _selections: Optional[List[RowSelection]]
    var _starts: List[Int]
    """The file position of each row group's first row, by row group, and
    the file's row count last: group `rg` holds `_starts[rg + 1] -
    _starts[rg]` rows."""
    var _next: Int
    var _ctx: ExecContext

    def __init__(
        out self,
        task: FileScanTask,
        table_schema: Schema,
        name_mapping: Optional[NameMapping] = None,
        row_groups: Optional[List[Int]] = None,
        var row_selections: Optional[List[RowSelection]] = None,
        options: StorageOptions = StorageOptions(),
        ctx: ExecContext = ExecContext.auto(),
    ) raises:
        """Open `task`'s data file and resolve it against `table_schema`.

        `table_schema` is what the caller reads — the snapshot's schema,
        possibly narrowed to the columns a query needs; every field carries
        its id. `name_mapping` gives ids to a file written without them.
        """
        if len(task.equality_deletes) > 0:
            raise NotImplementedError(
                "iceberg: equality deletes are not supported yet"
            )
        if task.file.content != DataFile.DATA:
            raise InvalidError(
                t"iceberg: {task.file.file_path} is not a data file"
            )
        if task.file.file_format != "PARQUET":
            raise NotImplementedError(
                t"iceberg: data file format '{task.file.file_format}'"
            )
        self._file = ParquetFile[DynSource, LeafSet.all()](
            DynSource.open(task.path, options)
        )
        var file_schema = self._file.schema()
        if name_mapping:
            file_schema = name_mapping.value().apply(file_schema)
        self._projection = SchemaProjection(
            table_schema,
            file_schema,
            # `SchemaProjection` fills a column from one of these only when
            # the file lacks it, at any nesting depth.
            _one_row_values(task.file.identity_values(task.spec)),
        )
        self._columns = self._projection.file_columns()
        if len(self._columns) == 0 and len(file_schema.fields) > 0:
            # Every table column is a constant or null, but a batch's length
            # comes from its columns: read one only to count the rows.
            self._columns.append(file_schema.fields[0].name)
        self._deleted = read_position_deletes(
            task.position_deletes, task.file.file_path, options, ctx
        )

        ref meta = self._file.metadata()
        self._starts = List[Int](capacity=len(meta.row_groups) + 1)
        var at = 0
        for ref rg in meta.row_groups:
            self._starts.append(at)
            at += rg.num_rows
        self._starts.append(at)
        if row_groups:
            self._row_groups = row_groups.value().copy()
        else:
            self._row_groups = List[Int](capacity=len(meta.row_groups))
            for rg in range(len(meta.row_groups)):
                self._row_groups.append(rg)
        if row_selections and len(row_selections.value()) != len(
            self._row_groups
        ):
            raise InvalidError(
                "iceberg: row_selections must match the selected row groups"
            )
        self._selections = row_selections^
        self._next = 0
        self._ctx = ctx.copy()

    def next(mut self) raises -> Optional[RecordBatch]:
        """The next row group's surviving rows as a batch of the table
        schema; `None` once every row group has been read. A row group with
        no surviving row is skipped, so a batch is never empty."""
        while self._next < len(self._row_groups):
            var slot = self._next
            self._next += 1
            var rg = self._row_groups[slot]
            var selection = Optional[RowSelection](None)
            if self._selections:
                selection = self._selections.value()[slot].copy()
                if not selection.value().selects_any():
                    continue
            var mask = Optional[BoolArray](None)
            var lo = _lower_bound(self._deleted, self._starts[rg])
            var hi = _lower_bound(self._deleted, self._starts[rg + 1])
            if lo < hi:
                var kept = 0
                var flags = self._mask(rg, lo, selection, kept)
                if kept == 0:
                    continue
                elif kept < len(flags):
                    # A selection may already have skipped every deleted row.
                    mask = flags^

            var picked = Optional[List[RowSelection]](None)
            if selection:
                var one: List[RowSelection] = [selection.value().copy()]
                picked = one^
            var groups: List[Int] = [rg]
            var batch = self._file.read(
                columns=self._columns.copy(),
                row_groups=groups^,
                row_selections=picked^,
                ctx=self._ctx,
            ).combine_chunks()
            if mask:
                var keep: DynArray = mask.value().copy()
                var columns = List[DynArray](capacity=batch.num_columns())
                for i in range(batch.num_columns()):
                    columns.append(filter(batch.column(i), keep, self._ctx))
                batch = RecordBatch(batch.schema, columns^)
            return self._projection.project(batch)
        return None

    def _mask(
        self,
        rg: Int,
        first: Int,
        selection: Optional[RowSelection],
        mut kept: Int,
    ) raises -> BoolArray:
        """One flag per row `rg` decodes — every row, or the rows
        `selection` keeps — False where the position is deleted. `first` is
        the index of the first deleted position at or after the group's
        start; `kept` counts the True flags."""
        var start = self._starts[rg]
        var size = self._starts[rg + 1] - start
        var runs = List[Tuple[Int, Int]]()
        if selection:
            runs = selection.value().runs_in(0, size)
        else:
            runs.append((0, size))
        var builder = BoolBuilder(size)
        var d = first
        kept = 0
        for ref run in runs:
            for row in range(run[0], run[1]):
                var position = start + row
                while d < len(self._deleted) and self._deleted[d] < position:
                    d += 1
                var live = (
                    d >= len(self._deleted) or self._deleted[d] != position
                )
                builder.append(live)
                if live:
                    kept += 1
        return builder.finish()


def _one_row_values(values: Dict[Int, DynScalar]) raises -> Dict[Int, DynArray]:
    """Each value as a one-row array, the form `SchemaProjection` repeats."""
    var out = Dict[Int, DynArray]()
    for entry in values.items():
        out[entry.key] = entry.value.to_array(1)
    return out^
