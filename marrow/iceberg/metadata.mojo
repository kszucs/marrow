# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""An Iceberg table's metadata: schemas, partition specs and snapshots.

This is the model a `vN.metadata.json` file describes, and the questions a
reader asks of it — which snapshot, read with which schema, partitioned by
which spec. `TableMetadata.parse` reads one from its JSON text; nothing here
touches I/O.

[Reference](https://iceberg.apache.org/spec/#table-metadata)
"""

from emberjson import Value as Json

from ..errors import InvalidError, KeyError, NotImplementedError
from ..schema import Schema

from .document import (
    as_string,
    expect_array,
    expect_member,
    expect_object,
    has,
    int_member,
    optional_int,
    parse_json,
    string_member,
)
from .projection import NameMapping
from .transforms import Transform
from .types import schema_from_json

comptime NAME_MAPPING_PROPERTY = "schema.name-mapping.default"


@fieldwise_init
struct TableSchema(Copyable, Movable):
    """One entry of `schemas`: an Arrow schema whose every field carries its
    Iceberg id under `FIELD_ID_KEY`."""

    var schema_id: Int
    var schema: Schema


@fieldwise_init
struct PartitionField(Copyable, Movable):
    """A partition column: `transform` applied to the column `source_id`.

    `field_id` identifies the partition value in a manifest's partition
    struct, which is not the source column's id."""

    var source_id: Int
    var field_id: Int
    var name: String
    var transform: Transform


@fieldwise_init
struct PartitionSpec(Copyable, Movable):
    var spec_id: Int
    var fields: List[PartitionField]

    def is_unpartitioned(self) -> Bool:
        """No field, or only `void` ones: every file is in the one partition."""
        for ref f in self.fields:
            if f.transform.kind != Transform.VOID:
                return False
        return True


@fieldwise_init
struct Snapshot(Copyable, Movable):
    """The table's state after one commit, listed by its manifest list.

    `sequence_number` is 0 in a v1 table, where none is written. `schema_id`
    is absent in metadata written before v2 introduced it; such a snapshot is
    read with the current schema."""

    var snapshot_id: Int
    var sequence_number: Int
    var manifest_list: String
    var schema_id: Optional[Int]


@fieldwise_init
struct SnapshotRef(Copyable, Movable):
    """A named branch or tag. `main` is the table's current state."""

    var snapshot_id: Int


@fieldwise_init
struct SnapshotLogEntry(Copyable, Movable):
    """When `snapshot_id` became current — what time travel by timestamp
    reads, since a rolled-back snapshot is not in the log."""

    var timestamp_ms: Int
    var snapshot_id: Int


@fieldwise_init
struct TableMetadata(Copyable, Movable):
    """One version of a table's metadata, v1 or v2."""

    var format_version: Int
    var location: String
    """Where the table was written. Every path inside the metadata, manifest
    lists and manifests begins with it — see `relocate`."""
    var schemas: List[TableSchema]
    var current_schema_id: Int
    var partition_specs: List[PartitionSpec]
    var default_spec_id: Int
    var snapshots: List[Snapshot]
    var current_snapshot_id: Optional[Int]
    var refs: Dict[String, SnapshotRef]
    var snapshot_log: List[SnapshotLogEntry]
    var properties: Dict[String, String]

    @staticmethod
    def parse(text: StringSlice) raises -> TableMetadata:
        """A `metadata.json` file's contents, v1 or v2.

        v1 metadata may carry a single `schema` and `partition-spec` where v2
        has lists; both are read. A v1 snapshot listing its `manifests`
        inline, without a manifest list, is refused."""
        var json = parse_json(text, "table metadata")
        expect_object(json, "table metadata")
        ref o = json.object()
        var version = int_member(o, "format-version")
        if version < 1 or version > 2:
            raise NotImplementedError(
                t"iceberg: format version {version} is not supported"
            )

        var schemas = List[TableSchema]()
        var current_schema_id: Int
        if has(o, "schemas"):
            ref schemas_json = o["schemas"]
            expect_array(schemas_json, "schemas")
            for ref js in schemas_json.array():
                schemas.append(_table_schema(js))
            current_schema_id = int_member(o, "current-schema-id")
        else:
            expect_member(o, "schema")
            schemas.append(_table_schema(o["schema"]))
            current_schema_id = schemas[0].schema_id

        var specs = List[PartitionSpec]()
        var default_spec_id: Int
        if has(o, "partition-specs"):
            ref specs_json = o["partition-specs"]
            expect_array(specs_json, "partition-specs")
            for ref js in specs_json.array():
                expect_object(js, "partition spec")
                ref spec = js.object()
                expect_member(spec, "fields")
                specs.append(
                    PartitionSpec(
                        int_member(spec, "spec-id"),
                        _partition_fields(spec["fields"]),
                    )
                )
            default_spec_id = int_member(o, "default-spec-id")
        else:
            expect_member(o, "partition-spec")
            specs.append(
                PartitionSpec(0, _partition_fields(o["partition-spec"]))
            )
            default_spec_id = 0

        var snapshots = List[Snapshot]()
        if has(o, "snapshots"):
            ref snapshots_json = o["snapshots"]
            expect_array(snapshots_json, "snapshots")
            for ref js in snapshots_json.array():
                snapshots.append(_snapshot(js))

        var current = optional_int(o, "current-snapshot-id")
        if current and current.value() == -1:
            current = None

        var refs = Dict[String, SnapshotRef]()
        if has(o, "refs"):
            ref refs_json = o["refs"]
            expect_object(refs_json, "refs")
            for entry in refs_json.object().items():
                expect_object(entry.value, "ref")
                ref r = entry.value.object()
                refs[entry.key] = SnapshotRef(int_member(r, "snapshot-id"))

        var log = List[SnapshotLogEntry]()
        if has(o, "snapshot-log"):
            ref log_json = o["snapshot-log"]
            expect_array(log_json, "snapshot-log")
            for ref js in log_json.array():
                expect_object(js, "snapshot log entry")
                ref e = js.object()
                log.append(
                    SnapshotLogEntry(
                        int_member(e, "timestamp-ms"),
                        int_member(e, "snapshot-id"),
                    )
                )

        var properties = Dict[String, String]()
        if has(o, "properties"):
            ref properties_json = o["properties"]
            expect_object(properties_json, "properties")
            for entry in properties_json.object().items():
                properties[entry.key] = as_string(entry.value, entry.key)

        return TableMetadata(
            format_version=version,
            location=string_member(o, "location"),
            schemas=schemas^,
            current_schema_id=current_schema_id,
            partition_specs=specs^,
            default_spec_id=default_spec_id,
            snapshots=snapshots^,
            current_snapshot_id=current,
            refs=refs^,
            snapshot_log=log^,
            properties=properties^,
        )

    def name_mapping(self) raises -> Optional[NameMapping]:
        """The table's default name mapping, which assigns ids to the columns
        of a data file written without them; `None` when the table has none.
        """
        var text = self.properties.get(NAME_MAPPING_PROPERTY)
        if not text:
            return None
        return NameMapping.parse(text.value())

    def schema(self, schema_id: Int) raises KeyError -> Schema:
        for i in range(len(self.schemas)):
            if self.schemas[i].schema_id == schema_id:
                return self.schemas[i].schema.copy()
        raise KeyError(t"iceberg: no schema with id {schema_id}")

    def current_schema(self) raises KeyError -> Schema:
        return self.schema(self.current_schema_id)

    def spec(self, spec_id: Int) raises KeyError -> PartitionSpec:
        for i in range(len(self.partition_specs)):
            if self.partition_specs[i].spec_id == spec_id:
                return self.partition_specs[i].copy()
        raise KeyError(t"iceberg: no partition spec with id {spec_id}")

    def snapshot(self, snapshot_id: Int) raises KeyError -> Snapshot:
        for i in range(len(self.snapshots)):
            if self.snapshots[i].snapshot_id == snapshot_id:
                return self.snapshots[i].copy()
        raise KeyError(t"iceberg: no snapshot with id {snapshot_id}")

    def current_snapshot(self) raises KeyError -> Optional[Int]:
        """The current snapshot's id, `None` for a table with no data yet.

        `refs["main"]` wins over `current-snapshot-id` when both are written,
        as the spec says the latter is kept only for older readers."""
        var main = self.refs.get("main")
        if main:
            return main.value().snapshot_id
        else:
            return self.current_snapshot_id

    def snapshot_for_ref(self, name: String) raises KeyError -> Int:
        var ref_ = self.refs.get(name)
        if not ref_:
            raise KeyError(t"iceberg: no branch or tag named '{name}'")
        return ref_.value().snapshot_id

    def snapshot_as_of(self, timestamp_ms: Int) raises KeyError -> Int:
        """The snapshot that was current at `timestamp_ms`: the last log entry
        at or before it."""
        var found: Optional[Int] = None
        for ref entry in self.snapshot_log:
            if entry.timestamp_ms <= timestamp_ms:
                found = entry.snapshot_id
        if not found:
            raise KeyError(
                t"iceberg: no snapshot was current at {timestamp_ms} ms"
            )
        return found.value()

    def schema_for(self, snapshot: Snapshot) raises KeyError -> Schema:
        """The schema a snapshot is read with: its own, falling back to the
        current one. Time travel shows the columns as they were then, as
        Spark and pyiceberg do."""
        if snapshot.schema_id:
            return self.schema(snapshot.schema_id.value())
        else:
            return self.current_schema()


def relocate(path: String, location: String, root: String) -> String:
    """`path` with the prefix `location` replaced by `root`, the directory
    the table was actually opened from.

    Iceberg records absolute paths, so a table that was copied or moved
    names files where it used to be. A path outside `location` is returned
    unchanged; with `root == location` this is the identity."""
    if root == location or not path.startswith(location):
        return path
    return root + String(path.removeprefix(location))


def _table_schema(json: Json) raises -> TableSchema:
    expect_object(json, "schema")
    return TableSchema(
        optional_int(json.object(), "schema-id").or_else(0),
        schema_from_json(json),
    )


def _partition_fields(json: Json) raises -> List[PartitionField]:
    """A spec's fields. v1 metadata may omit `field-id`; the spec assigns
    1000, 1001, ... in order, as Java's v1 reader does."""
    var fields = List[PartitionField]()
    var i = 0
    expect_array(json, "partition fields")
    for ref js in json.array():
        expect_object(js, "partition field")
        ref f = js.object()
        fields.append(
            PartitionField(
                int_member(f, "source-id"),
                optional_int(f, "field-id").or_else(1000 + i),
                string_member(f, "name"),
                Transform.parse(string_member(f, "transform")),
            )
        )
        i += 1
    return fields^


def _snapshot(json: Json) raises -> Snapshot:
    expect_object(json, "snapshot")
    ref o = json.object()
    var id = int_member(o, "snapshot-id")
    if not has(o, "manifest-list"):
        raise NotImplementedError(
            t"iceberg: snapshot {id} lists its manifests inline (v1)"
            t" rather than in a manifest list"
        )
    return Snapshot(
        snapshot_id=id,
        sequence_number=optional_int(o, "sequence-number").or_else(0),
        manifest_list=string_member(o, "manifest-list"),
        schema_id=optional_int(o, "schema-id"),
    )
