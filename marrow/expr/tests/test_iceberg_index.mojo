# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Index.from_manifest`: an Iceberg manifest's data files as chunks.

Every case builds its `DataFile`s by hand, bounds through `encode_bound`, and
asks `read_plan` which files are left to read — the decision needs no Avro and
no Parquet. A wrong "keep" costs time and a wrong "drop" costs the answer, so
the cases where nothing may prune (absent bounds, a nested column) matter as
much as the ones that do.
"""

from std.testing import assert_equal, assert_true

from ...dtypes import (
    DynType,
    FIELD_ID_KEY,
    Field,
    int64,
    string,
    struct_,
)
from ...iceberg.bounds import encode_bound
from ...iceberg.manifest import DataFile
from ...iceberg.metadata import PartitionField, PartitionSpec
from ...iceberg.transforms import Transform
from ...scalars import DynScalar, Int64Scalar, NullScalar, StringScalar
from ...schema import Schema, schema
from ..builders import col, lit
from ..index import Index
from ..logical import DynValue
from ..runtime.values import column, gt, literal


def _field(name: String, var dtype: DynType, id: Int) -> Field:
    var f = Field(name, dtype^)
    f.metadata[FIELD_ID_KEY] = String(id)
    return f^


def _i64(v: Int) -> DynScalar:
    return Int64Scalar(Scalar[int64.native](v)).to_dyn()


def _str(v: String) -> DynScalar:
    return StringScalar(v).to_dyn()


struct _File:
    """A `DataFile` under construction: just the fields pruning reads."""

    var rows: Int
    var partition: List[DynScalar]
    var nulls: Dict[Int, Int]
    var lower: Dict[Int, List[UInt8]]
    var upper: Dict[Int, List[UInt8]]

    def __init__(out self, rows: Int, var partition: List[DynScalar] = []):
        self.rows = rows
        self.partition = partition^
        self.nulls = {}
        self.lower = {}
        self.upper = {}

    def bounds(mut self, id: Int, lo: DynScalar, hi: DynScalar) raises:
        self.lower[id] = encode_bound(lo)
        self.upper[id] = encode_bound(hi)

    def null_count(mut self, id: Int, n: Int):
        self.nulls[id] = n

    def build(self) -> DataFile:
        return DataFile(
            DataFile.DATA,
            String("f.parquet"),
            String("PARQUET"),
            self.partition.copy(),
            self.rows,
            self.nulls.copy(),
            self.lower.copy(),
            self.upper.copy(),
            [],
        )


def _unpartitioned() -> PartitionSpec:
    return PartitionSpec(0, [])


def _schema() -> Schema:
    return schema([_field("x", int64, 1), _field("s", string, 2)])


def _int_files() raises -> List[DataFile]:
    """Three files holding x in [0, 9], [10, 19], [20, 29]."""
    var out = List[DataFile]()
    for i in range(3):
        var f = _File(10)
        f.bounds(1, _i64(i * 10), _i64(i * 10 + 9))
        f.null_count(1, 0)
        out.append(f.build())
    return out^


def test_iceberg_index_one_chunk_per_file() raises:
    var idx = Index.from_manifest(_int_files(), _schema(), _unpartitioned())
    assert_equal(idx.chunks, 3)
    assert_equal(idx.rows, [10, 10, 10])
    assert_equal(idx.num_rows().value(), 30)
    assert_equal(idx.null_count("x").value(), 0)


def test_iceberg_index_int_range() raises:
    var idx = Index.from_manifest(_int_files(), _schema(), _unpartitioned())
    var gt15: List[DynValue] = [DynValue(col("x", int64) > lit(15, int64))]
    assert_equal(idx.read_plan(gt15), [1, 2])
    var band: List[DynValue] = [
        DynValue(col("x", int64) > lit(12, int64)),
        DynValue(col("x", int64) < lit(18, int64)),
    ]
    assert_equal(idx.read_plan(band), [1])
    var runtime: List[DynValue] = [DynValue(gt(column("x"), literal(_i64(25))))]
    assert_equal(idx.read_plan(runtime), [2])


def test_iceberg_index_string_bounds() raises:
    """String bounds decode as the column's dtype and are recorded — but no
    lane prunes a string comparison yet, Parquet statistics included, so
    every file is kept."""
    var files = List[DataFile]()
    var a = _File(5)
    a.bounds(2, _str("a"), _str("f"))
    files.append(a.build())
    var b = _File(5)
    b.bounds(2, _str("g"), _str("m"))
    files.append(b.build())
    var idx = Index.from_manifest(files, _schema(), _unpartitioned())
    assert_true(idx.dtype_of("s") == DynType(string))
    var eq: List[DynValue] = [
        DynValue(col("s", string) == lit(String("h"), string))
    ]
    assert_equal(idx.read_plan(eq), [0, 1])


def test_iceberg_index_absent_bounds_are_kept() raises:
    """A file without bounds or a null count proves nothing."""
    var files = _int_files()
    files.append(_File(10).build())
    var idx = Index.from_manifest(files, _schema(), _unpartitioned())
    var gt15: List[DynValue] = [DynValue(col("x", int64) > lit(15, int64))]
    assert_equal(idx.read_plan(gt15), [1, 2, 3])
    assert_equal(idx.zones.null_counts("x"), [0, 0, 0, -1])
    assert_true(not idx.null_count("x"))


def test_iceberg_index_all_null_column() raises:
    """A file whose column is all null holds no row for any comparison, and
    every row for `is_null`."""
    var files = _int_files()
    var empty = _File(7)
    empty.null_count(1, 7)
    files.append(empty.build())
    var idx = Index.from_manifest(files, _schema(), _unpartitioned())
    var defined = idx.defined("x")
    assert_true(defined[0].value() and not defined[3].value())
    var gt_neg: List[DynValue] = [DynValue(col("x", int64) > lit(-1, int64))]
    assert_equal(idx.read_plan(gt_neg), [0, 1, 2])
    var is_null: List[DynValue] = [DynValue(col("x", int64).is_null())]
    assert_equal(idx.read_plan(is_null), [0, 1, 2, 3])
    var is_valid: List[DynValue] = [DynValue(col("x", int64).is_valid())]
    assert_equal(idx.read_plan(is_valid), [0, 1, 2, 3])


def _identity_on_x() -> PartitionSpec:
    return PartitionSpec(
        1,
        [
            PartitionField(
                1, 1000, String("x"), Transform(Transform.IDENTITY, 0)
            )
        ],
    )


def test_iceberg_index_identity_partition_without_bounds() raises:
    """With no bounds, an identity partition value is both min and max."""
    var files = List[DataFile]()
    for v in [3, 7, 11]:
        files.append(_File(4, [_i64(v)]).build())
    var idx = Index.from_manifest(files, _schema(), _identity_on_x())
    var eq7: List[DynValue] = [DynValue(col("x", int64) == lit(7, int64))]
    assert_equal(idx.read_plan(eq7), [1])
    var gt5: List[DynValue] = [DynValue(col("x", int64) > lit(5, int64))]
    assert_equal(idx.read_plan(gt5), [1, 2])
    assert_equal(idx.zones.null_counts("x"), [0, 0, 0])


def test_iceberg_index_null_partition_value() raises:
    """A null identity partition value means the whole column is null."""
    var files = List[DataFile]()
    files.append(_File(4, [_i64(3)]).build())
    files.append(_File(6, [NullScalar().to_dyn()]).build())
    var idx = Index.from_manifest(files, _schema(), _identity_on_x())
    assert_equal(idx.zones.null_counts("x"), [0, 6])
    var gt0: List[DynValue] = [DynValue(col("x", int64) > lit(0, int64))]
    assert_equal(idx.read_plan(gt0), [0])
    var is_null: List[DynValue] = [DynValue(col("x", int64).is_null())]
    assert_equal(idx.read_plan(is_null), [0, 1])


def test_iceberg_index_non_identity_partition_prunes_nothing() raises:
    var spec = PartitionSpec(
        2,
        [
            PartitionField(
                1, 1000, String("x_b"), Transform(Transform.BUCKET, 4)
            )
        ],
    )
    var files = List[DataFile]()
    for v in [0, 1]:
        files.append(_File(4, [_i64(v)]).build())
    var idx = Index.from_manifest(files, _schema(), spec)
    var eq0: List[DynValue] = [DynValue(col("x", int64) == lit(0, int64))]
    assert_equal(idx.read_plan(eq0), [0, 1])


def test_iceberg_index_ignores_nested_columns() raises:
    """Bounds keyed by a struct column's id describe no leaf of it, so the
    column gets no zone map rather than somebody else's bounds."""
    var nested = _field("n", struct_([_field("x", int64, 4)]), 3)
    var sch = schema([_field("x", int64, 1), nested^])
    var f = _File(10)
    f.bounds(3, _i64(0), _i64(1))
    var files: List[DataFile] = [f.build()]
    var idx = Index.from_manifest(files, sch, _unpartitioned())
    assert_equal(idx.zones.num_columns(), 1)
    assert_true(idx.dtype_of("n").is_null())
    var gt5: List[DynValue] = [DynValue(gt(column("n"), literal(_i64(5))))]
    assert_equal(idx.read_plan(gt5), [0])
