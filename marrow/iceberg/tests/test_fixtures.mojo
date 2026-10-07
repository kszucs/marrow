# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Projection over data files pyiceberg wrote (`data/generate.py`).

The evolved schema is also spelled by hand, so parsing it out of the table's
`metadata.json` is checked against something independent.
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..catalog import IcebergTable
from ..metadata import TableMetadata
from ..projection import SchemaProjection
from ...arrays import StructArray
from ...builders import array
from ...dtypes import DynType, Field, FIELD_ID_KEY, float64, int32, int64
from ...dtypes import string, struct_
from ...parquet import ParquetFile
from ...schema import Schema

comptime EVOLVED_V1 = (
    "marrow/iceberg/tests/data/evolved/data/"
    "00000-0-e48e059f-6bfc-4b7b-aa0e-d18736748494.parquet"
)
"""The first append to `evolved`: `a int, b float, c, gone, s<x, y>`."""


def _f(name: String, var dtype: DynType, id: Int) -> Field:
    var f = Field(name, dtype^)
    f.metadata[FIELD_ID_KEY] = String(id)
    return f^


def _evolved_current() -> Schema:
    """`evolved` after its schema update: `c` renamed to `c2`, `a` and `b`
    promoted, `gone` dropped, `e` added, `s.z` added."""
    var s = struct_(
        [_f("x", int32, 6), _f("y", string, 7), _f("z", float64, 9)]
    )
    return Schema(
        fields=[
            _f("a", int64, 1),
            _f("b", float64, 2),
            _f("c2", string, 3),
            _f("s", s^, 5),
            _f("e", int64, 8),
        ]
    )


def test_iceberg_fixture_field_ids_read() raises:
    var file = ParquetFile(EVOLVED_V1)
    var ids = List[Int]()
    for ref f in file.schema().fields:
        ids.append(f.field_id().value())
    assert_true(ids == [1, 2, 3, 4, 5])


def test_iceberg_fixture_evolved_projection() raises:
    var file = ParquetFile(EVOLVED_V1)
    var projection = SchemaProjection(_evolved_current(), file.schema())
    assert_true(projection.file_columns() == ["a", "b", "c", "s"])
    var batch = file.read(columns=projection.file_columns()).combine_chunks()
    var out = projection.project(batch)

    assert_true(out.column_names() == ["a", "b", "c2", "s", "e"])
    assert_true(out.column(0).as_int64() == array([1, 2, None], int64))
    assert_true(out.column(1).as_float64() == array([1.5, None, 3.5], float64))
    assert_true(out.column(2).as_string() == array(["one", "two", None]))
    assert_equal(out.column(4).null_count(), 3)

    ref s = out.column(3).as_struct()
    assert_equal(s.null_count(), 1)
    assert_true(s.field(0).as_int32() == array([1, None, 3], int32))
    assert_equal(s.field(2).null_count(), 3)


comptime DATA = "marrow/iceberg/tests/data/"


def _ids(schema: Schema) raises -> List[Int]:
    var ids = List[Int]()
    for ref f in schema.fields:
        ids.append(f.field_id().value())
    return ids^


def test_iceberg_fixture_open_evolved() raises:
    var table = IcebergTable.open(DATA + "evolved")
    ref m = table.metadata
    assert_equal(m.format_version, 2)
    assert_equal(len(m.snapshots), 2)
    # The parsed current schema is the one spelled by hand above, nested ids
    # included.
    assert_true(m.current_schema() == _evolved_current())
    var first = m.snapshot(m.snapshot_log[0].snapshot_id)
    var old = m.schema_for(first)
    assert_true(old.names() == ["a", "b", "c", "gone", "s"])
    assert_true(_ids(old) == [1, 2, 3, 4, 5])
    assert_equal(m.current_snapshot().value(), m.snapshot_log[1].snapshot_id)
    assert_equal(
        table.relocate(m.location + "/data/x.parquet"),
        DATA + "evolved/data/x.parquet",
    )


def test_iceberg_fixture_open_v1() raises:
    var table = IcebergTable.open(DATA + "simple_v1")
    ref m = table.metadata
    assert_equal(m.format_version, 1)
    for ref s in m.snapshots:
        assert_equal(s.sequence_number, 0)
    assert_true(m.spec(m.default_spec_id).is_unpartitioned())
    assert_true(_ids(m.current_schema()) == [1, 2, 3, 4, 5, 6, 7, 8])


def test_iceberg_fixture_open_partitioned() raises:
    var m = IcebergTable.open(DATA + "partitioned").metadata.copy()
    var spec = m.spec(m.default_spec_id)
    assert_equal(len(spec.fields), 3)
    assert_equal(String(spec.fields[0].transform), "identity")
    assert_equal(String(spec.fields[1].transform), "day")
    assert_equal(String(spec.fields[2].transform), "bucket[2]")
    assert_equal(spec.fields[2].source_id, 1)
    assert_equal(spec.fields[2].field_id, 1002)


def test_iceberg_fixture_name_mapping() raises:
    var m = IcebergTable.open(DATA + "no_field_ids").metadata.copy()
    var mapping = m.name_mapping().value().copy()
    var file = ParquetFile(DATA + "no_field_ids/data/plain.parquet")
    assert_false(file.schema().fields[0].field_id())
    assert_true(_ids(mapping.apply(file.schema())) == [1, 2])


def test_iceberg_fixture_metadata_errors() raises:
    with assert_raises(contains="not valid JSON"):
        _ = TableMetadata.parse("{")
    with assert_raises(contains="missing 'schema'"):
        _ = TableMetadata.parse('{"format-version": 2}')
    with assert_raises(contains="format version 3"):
        _ = TableMetadata.parse('{"format-version": 3}')
    with assert_raises(contains="neither a metadata file"):
        _ = IcebergTable.open(DATA + "nope")
