# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""What a snapshot's manifest list and manifests describe.

A snapshot lists manifests (`ManifestFile`), and each manifest lists files
(`ManifestEntry` around a `DataFile`) with their partition values and column
statistics. `read_manifest_list` and `read_manifest` decode them from Avro,
finding every field by its Iceberg id rather than by name or position, which
is how one decoder reads both the v1 and the v2 layout.

[Reference](https://iceberg.apache.org/spec/#manifests)
"""

from ..arrays import BinaryArray, DynArray, Int32Array, MapArray, StructArray
from ..dtypes import DynType, field_index, int32
from ..avro import AvroFile
from ..errors import CorruptError
from ..io import DynSource
from ..io.uri import StorageOptions
from ..scalars import DynScalar
from ..tabular import RecordBatch

from .metadata import PartitionSpec
from .transforms import Transform


@fieldwise_init
struct ManifestFile(Copyable, Movable):
    """One entry of a manifest list."""

    comptime DATA = 0
    comptime DELETES = 1

    var path: String
    var partition_spec_id: Int
    var content: Int
    """What the manifest's files hold: `DATA` or `DELETES`."""
    var sequence_number: Int
    """The sequence number of the commit that added this manifest; 0 in v1."""
    var added_snapshot_id: Int


@fieldwise_init
struct DataFile(Copyable, Movable):
    """A data or delete file, as a manifest describes it.

    Statistics are keyed by field id. Any of them may be missing for any
    column — a writer is free to omit them — and a missing one proves
    nothing."""

    comptime DATA = 0
    comptime POSITION_DELETES = 1
    comptime EQUALITY_DELETES = 2

    var content: Int
    """What the file holds: `DATA` rows, or rows to delete by position
    (`POSITION_DELETES`) or by value (`EQUALITY_DELETES`)."""
    var file_path: String
    var file_format: String
    """`PARQUET`, `AVRO` or `ORC`, upper-case as written."""
    var partition: List[DynScalar]
    """One value per field of the file's partition spec, in order; a null
    partition value is a null scalar."""
    var record_count: Int
    var null_value_counts: Dict[Int, Int]
    var lower_bounds: Dict[Int, List[UInt8]]
    var upper_bounds: Dict[Int, List[UInt8]]
    var equality_ids: List[Int]
    """The columns an equality delete file matches on; empty otherwise."""

    def identity_values(self, spec: PartitionSpec) -> Dict[Int, DynScalar]:
        """The file's value of every identity partition field, keyed by the
        field's source column id; `spec` is the one the file was written
        with. Every row of the file holds exactly that value, a null one
        included. Other transforms lose information and are left out."""
        var values = Dict[Int, DynScalar]()
        ref fields = spec.fields
        for i in range(min(len(fields), len(self.partition))):
            if fields[i].transform.kind == Transform.IDENTITY:
                values[fields[i].source_id] = self.partition[i].copy()
        return values^


@fieldwise_init
struct ManifestEntry(Copyable, Movable):
    """One file of a manifest, and whether the snapshot that wrote the
    manifest added it, carried it over, or removed it. A `DELETED` entry is
    history, not part of the table."""

    comptime EXISTING = 0
    comptime ADDED = 1
    comptime DELETED = 2

    var status: Int
    var snapshot_id: Optional[Int]
    var sequence_number: Optional[Int]
    """The data sequence number; `None` when it is inherited."""
    var file_sequence_number: Optional[Int]
    var spec_id: Int
    """The partition spec `data_file.partition` follows: its manifest's."""
    var data_file: DataFile

    def is_live(self) -> Bool:
        return self.status != Self.DELETED

    def inherit(mut self, manifest: ManifestFile):
        """Fill what the writer left to be inherited from the manifest.

        A v2 writer omits the snapshot id and sequence numbers of a file it
        is adding, because they are not known until the commit succeeds; they
        are the manifest's. Only an `ADDED` entry may inherit a sequence
        number — an existing one always carries its own. In v1, where there
        are no sequence numbers, the manifest's is 0 and so is every file's.
        """
        if not self.snapshot_id:
            self.snapshot_id = manifest.added_snapshot_id
        if self.status == Self.ADDED or manifest.sequence_number == 0:
            if not self.sequence_number:
                self.sequence_number = manifest.sequence_number
            if not self.file_sequence_number:
                self.file_sequence_number = manifest.sequence_number


# ---------------------------------------------------------------------------
# Decoding, by field id
# ---------------------------------------------------------------------------


def _read_avro(path: String, options: StorageOptions) raises -> RecordBatch:
    var file = AvroFile[DynSource](DynSource.open(path, options))
    return file.read().combine_chunks()


def _top(batch: RecordBatch, id: Int) raises -> Optional[DynArray]:
    """The column with field id `id`, or `None` when the writer's schema has
    no such field — a v1 file lacks every v2 addition."""
    var i = field_index(batch.schema.fields, id)
    if i < 0:
        return None
    return batch.column(i).copy()


def _child(parent: StructArray, id: Int) raises -> Optional[DynArray]:
    var i = field_index(parent.dtype.as_struct().fields, id)
    if i < 0:
        return None
    return parent.field(i)


def _required(
    column: Optional[DynArray], id: Int, what: String
) raises -> DynArray:
    if not column:
        raise CorruptError(t"iceberg: {what} has no field {id}")
    return column.value().copy()


def _int_value(column: DynArray, row: Int) raises -> Int:
    """A required `int` or `long` field's value."""
    if not column.is_valid(row):
        raise CorruptError(t"iceberg: a required field is null at row {row}")
    if column.dtype() == DynType(int32):
        return Int(column.as_int32().unsafe_get(row))
    return Int(column.as_int64().unsafe_get(row))


def _int_at(column: Optional[DynArray], row: Int) raises -> Optional[Int]:
    """An optional `int` or `long` field's value, `None` when absent or
    null."""
    if not column or not column.value().is_valid(row):
        return None
    return _int_value(column.value(), row)


def _string_at(column: DynArray, row: Int) raises -> String:
    return String(column.as_string().unsafe_get(UInt(row)))


@fieldwise_init
struct _StatsMap(Copyable, Movable):
    """A `map<int, V>` statistic column, keyed by column id, with its
    entries' keys and values resolved once rather than per row."""

    var map: MapArray
    var keys: Int32Array
    var values: DynArray

    @staticmethod
    def find(parent: StructArray, id: Int) raises -> Optional[Self]:
        var column = _child(parent, id)
        if not column:
            return None
        ref m = column.value().as_map()
        ref entries = m.values().as_struct()
        return Self(
            m.copy(), entries.field(0).as_int32().copy(), entries.field(1)
        )

    def entries(self, row: Int) -> Tuple[Int, Int]:
        """`row`'s `[start, end)` entries; empty when its map is null."""
        if not self.map.is_valid(row):
            return (0, 0)
        return self.map.child_range(row)


def _count_map(stats: Optional[_StatsMap], row: Int) raises -> Dict[Int, Int]:
    """A `map<int, long>` statistic — counts by column id."""
    var out = Dict[Int, Int]()
    if stats:
        ref m = stats.value()
        var span = m.entries(row)
        ref values = m.values.as_int64()
        for i in range(span[0], span[1]):
            out[Int(m.keys.unsafe_get(i))] = Int(values.unsafe_get(i))
    return out^


def _bound_map(
    stats: Optional[_StatsMap], row: Int
) raises -> Dict[Int, List[UInt8]]:
    """A `map<int, binary>` statistic — serialized bounds by column id."""
    var out = Dict[Int, List[UInt8]]()
    if stats:
        ref m = stats.value()
        var span = m.entries(row)
        ref values = m.values.as_binary()
        for i in range(span[0], span[1]):
            var bound = List[UInt8]()
            bound.extend(values.unsafe_get(UInt(i)).as_bytes())
            out[Int(m.keys.unsafe_get(i))] = bound^
    return out^


def _int_list(column: Optional[DynArray], row: Int) raises -> List[Int]:
    var out = List[Int]()
    if not column or not column.value().is_valid(row):
        return out^
    ref l = column.value().as_list()
    var span = l.child_range(row)
    ref items = l.values().as_int32()
    for i in range(span[0], span[1]):
        out.append(Int(items.unsafe_get(i)))
    return out^


def read_manifest_list(
    path: String, options: StorageOptions = StorageOptions()
) raises -> List[ManifestFile]:
    """A snapshot's manifest list, v1 or v2. v1 has no content type or
    sequence numbers: every manifest tracks data, at sequence number 0."""
    var batch = _read_avro(path, options)
    var what = String(t"manifest list {path}")
    var paths = _required(_top(batch, 500), 500, what)
    var specs = _top(batch, 502)
    var contents = _top(batch, 517)
    var sequences = _top(batch, 515)
    var snapshots = _top(batch, 503)
    var out = List[ManifestFile](capacity=batch.num_rows())
    for row in range(batch.num_rows()):
        out.append(
            ManifestFile(
                path=_string_at(paths, row),
                partition_spec_id=_int_at(specs, row).or_else(0),
                content=_int_at(contents, row).or_else(ManifestFile.DATA),
                sequence_number=_int_at(sequences, row).or_else(0),
                added_snapshot_id=_int_at(snapshots, row).or_else(0),
            )
        )
    return out^


def read_manifest(
    path: String,
    manifest: ManifestFile,
    options: StorageOptions = StorageOptions(),
) raises -> List[ManifestEntry]:
    """Every entry of one manifest, with what the writer left to inheritance
    filled from `manifest` — see `ManifestEntry.inherit`."""
    var batch = _read_avro(path, options)
    var what = String(t"manifest {path}")
    var statuses = _required(_top(batch, 0), 0, what)
    var snapshot_ids = _top(batch, 1)
    var sequences = _top(batch, 3)
    var file_sequences = _top(batch, 4)
    var files = _required(_top(batch, 2), 2, what)
    ref f = files.as_struct()
    var contents = _child(f, 134)
    var paths = _required(_child(f, 100), 100, what)
    var formats = _required(_child(f, 101), 101, what)
    var partition = _required(_child(f, 102), 102, what)
    var records = _required(_child(f, 103), 103, what)
    var null_counts = _StatsMap.find(f, 110)
    var lowers = _StatsMap.find(f, 125)
    var uppers = _StatsMap.find(f, 128)
    var equality_ids = _child(f, 135)
    ref p = partition.as_struct()
    var partition_fields = List[DynArray](
        capacity=len(p.dtype.as_struct().fields)
    )
    for i in range(len(p.dtype.as_struct().fields)):
        partition_fields.append(p.field(i))

    var out = List[ManifestEntry](capacity=batch.num_rows())
    for row in range(batch.num_rows()):
        var values = List[DynScalar](capacity=len(partition_fields))
        for ref field in partition_fields:
            values.append(field[row])
        var file = DataFile(
            content=_int_at(contents, row).or_else(DataFile.DATA),
            file_path=_string_at(paths, row),
            file_format=_string_at(formats, row).upper(),
            partition=values^,
            record_count=_int_value(records, row),
            null_value_counts=_count_map(null_counts, row),
            lower_bounds=_bound_map(lowers, row),
            upper_bounds=_bound_map(uppers, row),
            equality_ids=_int_list(equality_ids, row),
        )
        var entry = ManifestEntry(
            status=_int_value(statuses, row),
            snapshot_id=_int_at(snapshot_ids, row),
            sequence_number=_int_at(sequences, row),
            file_sequence_number=_int_at(file_sequences, row),
            spec_id=manifest.partition_spec_id,
            data_file=file^,
        )
        entry.inherit(manifest)
        out.append(entry^)
    return out^
