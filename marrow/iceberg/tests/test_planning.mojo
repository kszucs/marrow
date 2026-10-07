# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Whole-table reads of the fixtures, against pyiceberg's answers.

Every snapshot of every fixture table is planned, every task read, and the
rows compared with `expected/<snapshot-id>.parquet` — as text and sorted,
since row order across data files is not defined and pyiceberg reads strings
as `large_string`. Matching delete files to data files is checked on a plan
built by hand.
"""

from std.os import listdir
from std.testing import assert_equal, assert_true

from ..catalog import IcebergTable
from ...execution import ExecContext
from ..manifest import DataFile, ManifestEntry
from ..metadata import PartitionSpec, TableMetadata, TableSchema
from ..planning import SnapshotFiles, scan_tasks, snapshot_files
from ..scan import IcebergFileReader
from ..scan import POSITION_DELETE_FILE_PATH_ID
from ...parquet import ParquetFile
from ...schema import Schema
from ...tabular import RecordBatch
from ...io.uri import StorageOptions

comptime DATA = "marrow/iceberg/tests/data/"


def _rows(batch: RecordBatch) raises -> List[String]:
    var rows = List[String](capacity=batch.num_rows())
    for i in range(batch.num_rows()):
        var row = String()
        for c in range(batch.num_columns()):
            row += String(batch.column(c)[i]) + "|"
        rows.append(row^)
    return rows^


def _read_snapshot(
    table: IcebergTable, snapshot_id: Int
) raises -> Tuple[List[String], List[String]]:
    """Every row of the snapshot, and its column names."""
    ref m = table.metadata
    var schema = m.schema_for(m.snapshot(snapshot_id))
    var mapping = m.name_mapping()
    var rows = List[String]()
    for task in scan_tasks(table, snapshot_files(table, snapshot_id)):
        var reader = IcebergFileReader(task, schema, mapping)
        while True:
            var batch = reader.next()
            if not batch:
                break
            rows.extend(_rows(batch.value()))
    return (rows^, schema.names())


def _check_table(name: String) raises:
    var table = IcebergTable.open(DATA + name)
    var expected_dir = DATA + name + "/expected"
    var checked = 0
    for file in listdir(expected_dir):
        var snapshot_id = Int(String(file.removesuffix(".parquet")))
        var expected = ParquetFile(expected_dir + "/" + file).read()
        var want = _rows(expected.combine_chunks())
        var got = _read_snapshot(table, snapshot_id)
        assert_true(
            got[1] == expected.column_names(),
            String(t"{name}@{snapshot_id}: columns differ"),
        )
        sort(want)
        sort(got[0])
        assert_equal(len(got[0]), len(want), String(t"{name}@{snapshot_id}"))
        for i in range(len(want)):
            assert_equal(got[0][i], want[i], String(t"{name}@{snapshot_id}"))
        checked += 1
    assert_true(checked == len(table.metadata.snapshots))


def test_iceberg_planning_simple_v1() raises:
    _check_table("simple_v1")


def test_iceberg_planning_simple_v2() raises:
    _check_table("simple_v2")


def test_iceberg_planning_partitioned() raises:
    _check_table("partitioned")


def test_iceberg_planning_evolved() raises:
    _check_table("evolved")


def test_iceberg_planning_no_field_ids() raises:
    _check_table("no_field_ids")


def test_iceberg_planning_tasks() raises:
    var table = IcebergTable.open(DATA + "partitioned")
    var current = table.metadata.current_snapshot().value()
    var files = snapshot_files(table, current)
    var tasks = scan_tasks(table, files)
    assert_equal(len(tasks), len(files.data))
    var rows = 0
    for ref t in tasks:
        rows += t.file.record_count
        assert_equal(len(t.file.partition), 3)
        # Read from where the table is now; recorded where it was written.
        assert_true(t.path.startswith(DATA + "partitioned/data/"))
        assert_true(t.file.file_path.startswith(table.metadata.location))
    assert_equal(rows, 12)
    var keep = List[Bool](length=len(files.data), fill=False)
    keep[0] = True
    assert_equal(len(scan_tasks(table, files, keep^)), 1)


# ---------------------------------------------------------------------------
# Matching deletes, on a plan built by hand
# ---------------------------------------------------------------------------

comptime LOCATION = "s3://w/t"
comptime ROOT = "/r/t"


def _bytes(text: String) -> List[UInt8]:
    var out = List[UInt8]()
    out.extend(text.as_bytes())
    return out^


def _entry(
    content: Int,
    path: String,
    seq: Int,
    bounds: Optional[Tuple[String, String]] = None,
) -> ManifestEntry:
    """A live entry; `bounds` are a position delete file's `file_path`
    bounds."""
    var lower = Dict[Int, List[UInt8]]()
    var upper = Dict[Int, List[UInt8]]()
    if bounds:
        lower[POSITION_DELETE_FILE_PATH_ID] = _bytes(bounds.value()[0])
        upper[POSITION_DELETE_FILE_PATH_ID] = _bytes(bounds.value()[1])
    var file = DataFile(
        content=content,
        file_path=String(LOCATION) + path,
        file_format="PARQUET",
        partition=[],
        record_count=1,
        null_value_counts={},
        lower_bounds=lower^,
        upper_bounds=upper^,
        equality_ids=[],
    )
    return ManifestEntry(ManifestEntry.ADDED, 1, seq, seq, 0, file^)


def _names(files: List[DataFile]) -> List[String]:
    var out = List[String]()
    for ref f in files:
        out.append(f.file_path)
    return out^


def test_iceberg_planning_matches_deletes() raises:
    """A position delete applies to data at or below its sequence number
    whose recorded path its `file_path` bounds admit — truncated bounds
    included; an equality delete to data strictly below it. Every path a
    task opens is relocated, the data file's recorded one is not."""
    var metadata = TableMetadata(
        format_version=2,
        location=LOCATION,
        schemas=[TableSchema(0, Schema(fields=[]))],
        current_schema_id=0,
        partition_specs=[PartitionSpec(0, [])],
        default_spec_id=0,
        snapshots=[],
        current_snapshot_id=None,
        refs={},
        snapshot_log=[],
        properties={},
    )
    var table = IcebergTable(metadata^, ROOT, StorageOptions())
    var a = String(LOCATION) + "/data/a.parquet"
    var data: List[ManifestEntry] = [
        _entry(DataFile.DATA, "/data/a.parquet", 1),
        _entry(DataFile.DATA, "/data/m.parquet", 1),
        _entry(DataFile.DATA, "/data/z.parquet", 1),
    ]
    var deletes: List[ManifestEntry] = [
        # Exactly a.parquet.
        _entry(DataFile.POSITION_DELETES, "/d/a.parquet", 2, (a, a)),
        # No bounds: may name any file.
        _entry(DataFile.POSITION_DELETES, "/d/any.parquet", 2),
        # Truncated: "m" up to "n", which admits m.parquet only.
        _entry(
            DataFile.POSITION_DELETES,
            "/d/m.parquet",
            2,
            (String(LOCATION) + "/data/m", String(LOCATION) + "/data/n"),
        ),
        # Older than the data.
        _entry(DataFile.POSITION_DELETES, "/d/old.parquet", 0),
        _entry(DataFile.EQUALITY_DELETES, "/d/eq.parquet", 2),
        _entry(DataFile.EQUALITY_DELETES, "/d/eq-old.parquet", 1),
    ]
    var tasks = scan_tasks(table, SnapshotFiles(data^, deletes^))
    assert_equal(len(tasks), 3)
    var unbounded = String(ROOT) + "/d/any.parquet"
    var want: List[List[String]] = [
        [String(ROOT) + "/d/a.parquet", unbounded],
        [unbounded, String(ROOT) + "/d/m.parquet"],
        [unbounded],
    ]
    for i in range(3):
        ref t = tasks[i]
        assert_true(t.path.startswith(String(ROOT) + "/data/"))
        assert_true(t.file.file_path.startswith(String(LOCATION) + "/data/"))
        assert_true(_names(t.position_deletes) == want[i])
        assert_true(
            _names(t.equality_deletes) == [String(ROOT) + "/d/eq.parquet"]
        )


def _paths(entries: List[ManifestEntry]) -> List[String]:
    var out = List[String]()
    for ref e in entries:
        out.append(e.data_file.file_path)
    return out^


def test_iceberg_planning_parallel_manifests() raises:
    """Reading manifests concurrently yields what a serial read does, in the
    same order."""
    var table = IcebergTable.open(DATA + "simple_v2")
    var current = table.metadata.current_snapshot().value()
    var serial = snapshot_files(table, current, ExecContext.serial())
    var parallel = snapshot_files(table, current, ExecContext.parallel(4))
    assert_equal(len(serial.data), 2)
    assert_true(_paths(serial.data) == _paths(parallel.data))
