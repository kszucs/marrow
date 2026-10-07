# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for the table metadata and manifest models."""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..manifest import DataFile, ManifestEntry, ManifestFile
from ..metadata import (
    PartitionField,
    PartitionSpec,
    Snapshot,
    SnapshotLogEntry,
    SnapshotRef,
    TableMetadata,
    TableSchema,
    relocate,
)
from ..transforms import Transform
from ...dtypes import Field, FIELD_ID_KEY, int64, string
from ...schema import Schema


def _schema(id: Int, var names: List[String]) -> TableSchema:
    var fields = List[Field]()
    for i in range(len(names)):
        var f = Field(names[i], int64)
        f.metadata[FIELD_ID_KEY] = String(i + 1)
        fields.append(f^)
    return TableSchema(id, Schema(fields=fields^))


def _snapshot(id: Int, seq: Int, schema_id: Optional[Int]) -> Snapshot:
    return Snapshot(
        snapshot_id=id,
        sequence_number=seq,
        manifest_list=String(t"file:///w/t/metadata/snap-{id}.avro"),
        schema_id=schema_id,
    )


def _table() raises -> TableMetadata:
    var spec = PartitionSpec(
        0,
        [PartitionField(2, 1000, "b_bucket", Transform.parse("bucket[4]"))],
    )
    return TableMetadata(
        format_version=2,
        location="file:///w/t",
        schemas=[_schema(0, ["a", "b"]), _schema(1, ["a", "b", "c"])],
        current_schema_id=1,
        partition_specs=[PartitionSpec(1, []), spec^],
        default_spec_id=0,
        snapshots=[_snapshot(10, 1, 0), _snapshot(20, 2, 1)],
        current_snapshot_id=10,
        refs={"main": SnapshotRef(20), "v1": SnapshotRef(10)},
        snapshot_log=[SnapshotLogEntry(100, 10), SnapshotLogEntry(200, 20)],
        properties={},
    )


def test_iceberg_metadata_lookups() raises:
    var t = _table()
    assert_equal(len(t.current_schema().fields), 3)
    assert_equal(t.current_snapshot().value(), 20)  # refs["main"] wins
    assert_equal(t.snapshot_for_ref("v1"), 10)
    assert_equal(t.snapshot(20).sequence_number, 2)
    assert_false(t.spec(0).is_unpartitioned())
    assert_true(t.spec(1).is_unpartitioned())
    with assert_raises(contains="no snapshot with id 30"):
        _ = t.snapshot(30)
    with assert_raises(contains="no branch or tag named 'nope'"):
        _ = t.snapshot_for_ref("nope")


def test_iceberg_metadata_time_travel() raises:
    var t = _table()
    assert_equal(t.snapshot_as_of(150), 10)
    assert_equal(t.snapshot_as_of(200), 20)
    with assert_raises(contains="no snapshot was current"):
        _ = t.snapshot_as_of(50)
    # An old snapshot reads with its own schema.
    assert_equal(len(t.schema_for(t.snapshot(10)).fields), 2)
    assert_equal(len(t.schema_for(t.snapshot(20)).fields), 3)


def test_iceberg_metadata_relocate() raises:
    var at = String("file:///w/t")
    var path = String("file:///w/t/data/f.parquet")
    assert_equal(relocate(path, at, "/repo/t"), "/repo/t/data/f.parquet")
    assert_equal(relocate(path, at, at), path)
    assert_equal(
        relocate("s3://elsewhere/f", at, "/repo/t"), "s3://elsewhere/f"
    )


def _entry(status: Int, seq: Optional[Int]) -> ManifestEntry:
    var file = DataFile(
        content=DataFile.DATA,
        file_path="f.parquet",
        file_format="PARQUET",
        partition=[],
        record_count=1,
        null_value_counts={},
        lower_bounds={},
        upper_bounds={},
        equality_ids=[],
    )
    return ManifestEntry(status, None, seq, seq, 0, file^)


def _manifest(seq: Int) -> ManifestFile:
    return ManifestFile(
        path="m.avro",
        partition_spec_id=0,
        content=ManifestFile.DATA,
        sequence_number=seq,
        added_snapshot_id=7,
    )


def test_iceberg_manifest_inherit() raises:
    # An added file inherits the manifest's snapshot and sequence numbers.
    var added = _entry(ManifestEntry.ADDED, None)
    added.inherit(_manifest(5))
    assert_equal(added.snapshot_id.value(), 7)
    assert_equal(added.sequence_number.value(), 5)
    assert_equal(added.file_sequence_number.value(), 5)
    # An existing file keeps its own.
    var existing = _entry(ManifestEntry.EXISTING, 3)
    existing.inherit(_manifest(5))
    assert_equal(existing.sequence_number.value(), 3)
    # v1: everything is sequence number 0.
    var v1 = _entry(ManifestEntry.EXISTING, None)
    v1.inherit(_manifest(0))
    assert_equal(v1.sequence_number.value(), 0)
    assert_false(_entry(ManifestEntry.DELETED, 1).is_live())
