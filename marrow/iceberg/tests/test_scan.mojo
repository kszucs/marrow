# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Reading single data files through `FileScanTask`s: the pyiceberg fixtures
against pyiceberg's own answers, and position deletes and partition constants
against files written here."""

from std.os.path import join
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..manifest import DataFile
from ..metadata import PartitionField, PartitionSpec
from ..projection import MappedField, NameMapping
from ..scan import (
    FileScanTask,
    IcebergFileReader,
    POSITION_DELETE_FILE_PATH_ID,
    POSITION_DELETE_POS_ID,
    read_position_deletes,
)
from ..transforms import Transform
from ...arrays import DynArray
from ...builders import array
from ...dtypes import DynType, Field, FIELD_ID_KEY, float64, int32, int64
from ...dtypes import date32, string
from ...io import FileSink
from ...kernels.filter import take
from ...parquet import ParquetFile, RowSelection, write_table
from ...parquet.codecs import Compression
from ...parquet.writer import FileWriter
from ...scalars import Date32Scalar, DynScalar, Int32Scalar, NullScalar
from ...scalars import StringScalar
from ...schema import Schema
from ...tabular import RecordBatch, Table
from ...utils.testing import ScratchDir, assert_values_equal

comptime DATA = "marrow/iceberg/tests/data/"
comptime WAREHOUSE = "file:///tmp/marrow-iceberg-fixtures/db/"
"""Where `generate.py` wrote the tables: every recorded path starts here."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f(
    name: String, var dtype: DynType, id: Int, nullable: Bool = True
) -> Field:
    var f = Field(name, dtype^, nullable)
    f.metadata[FIELD_ID_KEY] = String(id)
    return f^


def _file(
    path: String,
    var partition: List[DynScalar] = [],
    content: Int = DataFile.DATA,
    format: String = "PARQUET",
) -> DataFile:
    return DataFile(
        content=content,
        file_path=path,
        file_format=format,
        partition=partition^,
        record_count=0,
        null_value_counts={},
        lower_bounds={},
        upper_bounds={},
        equality_ids=[],
    )


def _task(
    var file: DataFile,
    var spec: PartitionSpec = PartitionSpec(0, []),
    var deletes: List[DataFile] = [],
    path: Optional[String] = None,
) -> FileScanTask:
    """A task reading `file` from `path`, by default its recorded one."""
    var at = path.or_else(file.file_path)
    return FileScanTask(
        file=file^,
        path=at^,
        spec=spec^,
        position_deletes=deletes^,
        equality_deletes=[],
    )


def _read_all(mut reader: IcebergFileReader) raises -> List[RecordBatch]:
    var batches = List[RecordBatch]()
    while True:
        var batch = reader.next()
        if not batch:
            break
        batches.append(batch.value().copy())
    return batches^


def _one(mut reader: IcebergFileReader) raises -> RecordBatch:
    var batches = _read_all(reader)
    assert_equal(len(batches), 1)
    return batches[0].copy()


def _assert_same(got: DynArray, expected: DynArray, what: String) raises:
    """Value equality, recursing into structs field by field. A struct's
    field metadata is not compared: only `got`'s children carry ids."""
    if expected.dtype().is_struct():
        assert_true(got.dtype().is_struct(), what)
        ref got_fields = got.dtype().as_struct().fields
        ref expected_fields = expected.dtype().as_struct().fields
        assert_equal(len(got_fields), len(expected_fields), what)
        assert_equal(len(got), len(expected), what)
        for i in range(len(expected)):
            assert_equal(got.is_null(i), expected.is_null(i), what)
        ref g = got.as_struct()
        ref e = expected.as_struct()
        for k in range(len(expected_fields)):
            var name = what + "." + expected_fields[k].name
            assert_equal(got_fields[k].name, expected_fields[k].name, name)
            _assert_same(g.field(k), e.field(k), name)
    else:
        assert_values_equal(got, expected, what)


def _assert_rows(
    got: RecordBatch, expected: RecordBatch, rows: List[Int]
) raises:
    """`got` holds `expected`'s `rows`, in that order, column for column."""
    assert_true(got.column_names() == expected.column_names())
    var indices = List[Optional[Int]](capacity=len(rows))
    for r in rows:
        indices.append(r)
    var picked = array(indices, int32)
    for i in range(got.num_columns()):
        _assert_same(
            got.column(i),
            take(expected.column(i), picked),
            got.column_names()[i],
        )


def _ids(batches: List[RecordBatch]) raises -> List[Int]:
    var ids = List[Int]()
    for ref b in batches:
        ref col = b.column(0).as_int64()
        for i in range(len(col)):
            ids.append(Int(col.unsafe_get(i)))
    return ids^


# ---------------------------------------------------------------------------
# pyiceberg fixtures
# ---------------------------------------------------------------------------


def _expected(
    table: String, snapshot: String, mapping: NameMapping
) raises -> Tuple[Schema, RecordBatch]:
    """pyiceberg's answer for `snapshot`, and the table schema it was read
    with: the expected file's own, given the table's ids by name."""
    var file = ParquetFile(DATA + table + "/expected/" + snapshot + ".parquet")
    var schema = mapping.apply(file.schema())
    return (schema^, file.read().combine_chunks())


def test_iceberg_scan_evolved_files() raises:
    """Both data files of `evolved`, read as the final schema: the old one
    through rename, promotion, a dropped and an added column."""
    var mapping = NameMapping(
        [
            MappedField(1, ["a"]),
            MappedField(2, ["b"]),
            MappedField(3, ["c2"]),
            MappedField(
                5,
                ["s"],
                [
                    MappedField(6, ["x"]),
                    MappedField(7, ["y"]),
                    MappedField(9, ["z"]),
                ],
            ),
            MappedField(8, ["e"]),
        ]
    )
    var expected = _expected("evolved", "2992740543627362547", mapping)
    ref schema = expected[0]
    ref rows = expected[1]
    var location = String(WAREHOUSE) + "evolved"
    var root = String(DATA) + "evolved"
    # The snapshot's rows: the second append first, then the first's three.
    var files: List[String] = [
        "00000-0-a5b15f71-b069-4492-acd8-b0ac971b2e36.parquet",
        "00000-0-e48e059f-6bfc-4b7b-aa0e-d18736748494.parquet",
    ]
    var want: List[List[Int]] = [[0], [1, 2, 3]]
    for i in range(len(files)):
        var task = _task(
            _file(location + "/data/" + files[i]),
            path=root + "/data/" + files[i],
        )
        var reader = IcebergFileReader(task, schema)
        _assert_rows(_one(reader), rows, want[i])


def test_iceberg_scan_partitioned_file() raises:
    """A file of `partitioned` — identity, day and bucket partitions — holds
    the snapshot's rows with ids 4 and 10."""
    var mapping = NameMapping(
        [
            MappedField(1, ["id"]),
            MappedField(2, ["category"]),
            MappedField(3, ["ts"]),
            MappedField(4, ["value"]),
        ]
    )
    var expected = _expected("partitioned", "5279110126697455324", mapping)
    ref schema = expected[0]
    ref rows = expected[1]
    var location = String(WAREHOUSE) + "partitioned"
    var spec = PartitionSpec(
        0,
        [
            PartitionField(2, 1000, "category", Transform.parse("identity")),
            PartitionField(3, 1001, "ts_day", Transform.parse("day")),
            PartitionField(1, 1002, "id_bucket", Transform.parse("bucket[2]")),
        ],
    )
    var partition = List[DynScalar](capacity=3)
    partition.append(StringScalar("b"))
    partition.append(Date32Scalar(Int32(19783), date32()))  # 2024-03-01
    partition.append(Int32Scalar(0))
    var name = String(
        "/data/category=b/ts_day=2024-03-01/id_bucket=0/"
        "00000-4-325caa80-ef36-4b0d-8fc8-224b41edc1bf.parquet"
    )
    var task = _task(
        _file(location + name, partition^),
        spec^,
        path=String(DATA) + "partitioned" + name,
    )
    var reader = IcebergFileReader(task, schema)
    var got = _one(reader)

    var at = List[Int]()
    ref ids = rows.column(0).as_int64()
    for want in [4, 10]:
        for i in range(len(ids)):
            if Int(ids.unsafe_get(i)) == want:
                at.append(i)
    _assert_rows(got, rows, at)


# ---------------------------------------------------------------------------
# Files written here
# ---------------------------------------------------------------------------


def _write_data(path: String, n: Int, row_group_size: Int) raises:
    """`id` (1) and `value` (4) = `0 .. n-1`, in groups of `row_group_size`."""
    var ids = List[Optional[Int]](capacity=n)
    var values = List[Optional[Float64]](capacity=n)
    for i in range(n):
        ids.append(i)
        values.append(Float64(i))
    var schema = Schema(
        fields=[_f("id", int64, 1, nullable=False), _f("value", float64, 4)]
    )
    var columns: List[DynArray] = [array(ids, int64), array(values, float64)]
    var batch = RecordBatch(schema, columns^)
    var w = FileWriter(FileSink(path), Compression.SNAPPY)
    w.write(Table.from_batches(schema, [batch^]), row_group_size=row_group_size)


def _write_deletes(
    path: String, paths: List[String], positions: List[Int]
) raises:
    var file_paths = List[Optional[String]](capacity=len(paths))
    var pos = List[Optional[Int]](capacity=len(positions))
    for i in range(len(paths)):
        file_paths.append(paths[i])
        pos.append(positions[i])
    var schema = Schema(
        fields=[
            _f("file_path", string, POSITION_DELETE_FILE_PATH_ID, False),
            _f("pos", int64, POSITION_DELETE_POS_ID, False),
        ]
    )
    var columns: List[DynArray] = [array(file_paths), array(pos, int64)]
    var batch = RecordBatch(schema, columns^)
    write_table(Table.from_batches(schema, [batch^]), path)


def _schema() -> Schema:
    return Schema(
        fields=[
            _f("id", int64, 1, nullable=False),
            _f("category", string, 2),
            _f("value", float64, 4),
        ]
    )


def test_iceberg_scan_identity_constant() raises:
    """An identity partition column the file does not hold is filled from
    the partition value — null included; a bucket value fills nothing."""
    with ScratchDir() as dir:
        var path = join(dir, "data.parquet")
        _write_data(path, 3, 1024)
        var spec = PartitionSpec(
            0,
            [
                PartitionField(
                    2, 1000, "category", Transform.parse("identity")
                ),
                PartitionField(
                    1, 1001, "id_bucket", Transform.parse("bucket[4]")
                ),
            ],
        )
        var partition = List[DynScalar](capacity=2)
        partition.append(StringScalar("eu"))
        partition.append(Int32Scalar(3))
        var reader = IcebergFileReader(
            _task(_file(path, partition^), spec.copy()), _schema()
        )
        var got = _one(reader)
        assert_true(got.column_names() == ["id", "category", "value"])
        assert_true(got.column(0).as_int64() == array([0, 1, 2], int64))
        assert_true(got.column(1).as_string() == array(["eu", "eu", "eu"]))

        var nulls = List[DynScalar](capacity=2)
        nulls.append(NullScalar())
        nulls.append(Int32Scalar(3))
        var null_reader = IcebergFileReader(
            _task(_file(path, nulls^), spec^), _schema()
        )
        var null_got = _one(null_reader)
        assert_equal(null_got.column(1).null_count(), 3)
        assert_true(null_got.column(1).dtype() == string)


comptime LOCATION = "s3://warehouse/t"
"""Where the data files written below are recorded to live; their tasks
read them from the scratch directory."""


def test_iceberg_scan_position_deletes() raises:
    """Deletes from two files — duplicated, out of order, spanning row
    groups, and with rows for another data file — drop exactly their
    positions; a row group they empty is skipped."""
    with ScratchDir() as dir:
        _write_data(join(dir, "data.parquet"), 10, 3)
        var data = String(LOCATION) + "/data.parquet"
        var other = String(LOCATION) + "/other.parquet"
        _write_deletes(
            join(dir, "d1.parquet"),
            [other, other, data, data, data],
            [0, 1, 5, 2, 3],
        )
        _write_deletes(join(dir, "d2.parquet"), [data, data], [2, 9])
        var deletes: List[DataFile] = [
            _file(join(dir, "d1.parquet"), content=DataFile.POSITION_DELETES),
            _file(join(dir, "d2.parquet"), content=DataFile.POSITION_DELETES),
        ]
        assert_true(read_position_deletes(deletes, data) == [2, 3, 5, 9])
        assert_true(read_position_deletes(deletes, other) == [0, 1])

        var task = _task(
            _file(data),
            deletes=deletes.copy(),
            path=join(dir, "data.parquet"),
        )
        assert_equal(ParquetFile(join(dir, "data.parquet")).num_row_groups(), 4)
        var reader = IcebergFileReader(task, _schema())
        var batches = _read_all(reader)
        # Groups [0 1 2] [3 4 5] [6 7 8] [9]; the last is wholly deleted.
        assert_equal(len(batches), 3)
        assert_true(_ids(batches) == [0, 1, 4, 6, 7, 8])
        ref values = batches[0].column(2).as_float64()
        assert_true(values == array([0.0, 1.0], float64))
        assert_equal(batches[0].column(1).null_count(), 2)

        # Pruned row groups and row selections pass through, and positions
        # match the rows a selection keeps: group 1 keeps rows 3 and 5, both
        # deleted, so it yields nothing.
        var selections: List[RowSelection] = [
            RowSelection([True, False, True]),
            RowSelection([False, True, True]),
        ]
        var groups: List[Int] = [1, 2]
        var pruned = IcebergFileReader(
            task,
            _schema(),
            row_groups=groups^,
            row_selections=selections^,
        )
        var kept = _read_all(pruned)
        assert_equal(len(kept), 1)
        assert_true(_ids(kept) == [7, 8])


def test_iceberg_scan_all_rows_deleted() raises:
    with ScratchDir() as dir:
        _write_data(join(dir, "data.parquet"), 4, 2)
        var data = String(LOCATION) + "/data.parquet"
        _write_deletes(
            join(dir, "d.parquet"), [data, data, data, data], [3, 2, 1, 0]
        )
        var deletes: List[DataFile] = [
            _file(join(dir, "d.parquet"), content=DataFile.POSITION_DELETES)
        ]
        var reader = IcebergFileReader(
            _task(
                _file(data), deletes=deletes^, path=join(dir, "data.parquet")
            ),
            _schema(),
        )
        assert_false(Bool(reader.next()))


def test_iceberg_scan_refuses_unsupported() raises:
    var orc = _task(_file("t/data/f.orc", format="ORC"))
    with assert_raises(contains="NotImplementedError"):
        _ = IcebergFileReader(orc, _schema())

    var task = _task(_file("t/data/f.parquet"))
    task.equality_deletes.append(
        _file("t/data/eq.parquet", content=DataFile.EQUALITY_DELETES)
    )
    with assert_raises(contains="NotImplementedError"):
        _ = IcebergFileReader(task, _schema())
