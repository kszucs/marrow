# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for by-field-id schema projection and name mapping."""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..projection import MappedField, NameMapping, SchemaProjection
from ...arrays import DynArray, ListArray, MapArray, StructArray
from ...builders import array
from ...dtypes import (
    DynType,
    Field,
    LargeListType,
    ListType,
    MapType,
    decimal128,
    float32,
    float64,
    int32,
    int64,
    string,
    struct_,
)
from ...kernels.cast import cast
from ...schema import Schema
from ...tabular import RecordBatch


def _f(
    name: String, var dtype: DynType, id: Int, nullable: Bool = True
) -> Field:
    """A field carrying `id` as its field id."""
    return Field(name, dtype^, nullable).with_field_id(id)


def _schema(var fields: List[Field]) -> Schema:
    return Schema(fields=fields^)


def test_iceberg_projection_identity() raises:
    var s = _schema([_f("a", int64, 1), _f("b", string, 2)])
    var batch = RecordBatch(
        s, [array([1, 2, 3], int64), array(["x", "y", None])]
    )
    var p = SchemaProjection(s, s)
    assert_true(p.file_columns() == ["a", "b"])
    assert_true(p.project(batch) == batch)


def test_iceberg_projection_rename() raises:
    var file = _schema([_f("old", int64, 1)])
    var table = _schema([_f("new", int64, 1, nullable=False)])
    var col: DynArray = array([1, 2, 3], int64)
    var p = SchemaProjection(table, file)
    assert_true(p.file_columns() == ["old"])
    var result = p.project(RecordBatch(file, [col.copy()]))
    assert_true(result == RecordBatch(table, [col.copy()]))
    assert_false(result.schema.fields[0].nullable)


def test_iceberg_projection_reorder() raises:
    var file = _schema([_f("a", int64, 1), _f("b", string, 2)])
    var table = _schema([_f("b", string, 2), _f("a", int64, 1)])
    var a: DynArray = array([1, 2], int64)
    var b: DynArray = array(["x", "y"])
    var p = SchemaProjection(table, file)
    assert_true(p.file_columns() == ["b", "a"])
    var result = p.project(RecordBatch(file, [a.copy(), b.copy()]))
    assert_true(result == RecordBatch(table, [b.copy(), a.copy()]))


def test_iceberg_projection_added_optional_column() raises:
    var file = _schema([_f("a", int64, 1)])
    var table = _schema([_f("a", int64, 1), _f("c", string, 3)])
    var p = SchemaProjection(table, file)
    assert_true(p.file_columns() == ["a"])
    var result = p.project(RecordBatch(file, [array([1, 2, 3], int64)]))
    assert_true(result.schema == table)
    assert_equal(result.num_rows(), 3)
    ref c = result.column(1)
    assert_true(c.dtype() == string.to_dyn())
    assert_equal(c.null_count(), 3)


def test_iceberg_projection_dropped_column_ignored() raises:
    var file = _schema(
        [_f("a", int64, 1), _f("b", string, 2), _f("x", float64, 9)]
    )
    var table = _schema([_f("a", int64, 1)])
    var a: DynArray = array([1, 2], int64)
    var p = SchemaProjection(table, file)
    assert_true(p.file_columns() == ["a"])
    # A batch read with `file_columns()` holds only what the table needs.
    var result = p.project(
        RecordBatch(_schema([_f("a", int64, 1)]), [a.copy()])
    )
    assert_true(result == RecordBatch(table, [a.copy()]))


def test_iceberg_projection_required_missing_raises() raises:
    var file = _schema([_f("a", int64, 1)])
    var table = _schema([_f("a", int64, 1), _f("c", string, 3, False)])
    with assert_raises(contains="required field 'c' (id 3)"):
        _ = SchemaProjection(table, file)


def test_iceberg_projection_int_to_long() raises:
    var file = _schema([_f("a", int32, 1)])
    var table = _schema([_f("a", int64, 1)])
    var result = SchemaProjection(table, file).project(
        RecordBatch(file, [array([1, None, 3], int32)])
    )
    assert_true(result.schema == table)
    assert_true(result.column(0).as_int64() == array([1, None, 3], int64))


def test_iceberg_projection_float_to_double() raises:
    var file = _schema([_f("f", float32, 1)])
    var table = _schema([_f("f", float64, 1)])
    var result = SchemaProjection(table, file).project(
        RecordBatch(file, [array([1.5, None, -2.25], float32)])
    )
    assert_true(
        result.column(0).as_float64() == array([1.5, None, -2.25], float64)
    )


def test_iceberg_projection_decimal_widening() raises:
    var file = _schema([_f("d", decimal128(9, 2), 1)])
    var table = _schema([_f("d", decimal128(12, 2), 1)])
    var src = cast(array([1, None, -3], int64), decimal128(9, 2))
    var result = SchemaProjection(table, file).project(
        RecordBatch(file, [src.copy()])
    )
    ref d = result.column(0)
    assert_true(d.dtype() == decimal128(12, 2).to_dyn())
    assert_true(cast(d, int64).as_int64() == array([1, None, -3], int64))


def test_iceberg_projection_decimal_scale_change_raises() raises:
    var file = _schema([_f("d", decimal128(9, 2), 1)])
    var table = _schema([_f("d", decimal128(12, 3), 1)])
    with assert_raises(contains="does not promote"):
        _ = SchemaProjection(table, file)


def test_iceberg_projection_incompatible_type_raises() raises:
    var narrowing = _schema([_f("a", int32, 1)])
    var wide = _schema([_f("a", int64, 1)])
    with assert_raises(contains="field 'a' is int64 in the data file"):
        _ = SchemaProjection(narrowing, wide)
    var text = _schema([_f("a", string, 1)])
    with assert_raises(contains="does not promote"):
        _ = SchemaProjection(wide, text)


def test_iceberg_projection_nested_struct() raises:
    """Children resolve by id: renamed, reordered, added and promoted."""
    var x = _f("x", int32, 11)
    var y = _f("y", string, 12)
    var file = _schema([_f("s", struct_([x.copy(), y.copy()]), 10)])
    var table_struct = struct_(
        [_f("why", string, 12), _f("z", float64, 13), _f("x", int64, 11)]
    )
    var table = _schema([_f("renamed", table_struct.copy(), 10)])
    var xs: DynArray = array([1, 2, 3], int32)
    var ys: DynArray = array(["a", None, "c"])
    var mask = array([False, True, False])
    var s: DynArray = StructArray.from_arrays([xs^, ys^], [x^, y^], mask^)
    var result = SchemaProjection(table, file).project(RecordBatch(file, [s^]))
    assert_true(result.schema == table)
    ref out = result.column(0).as_struct()
    assert_true(out.type() == table_struct^.to_dyn())
    # The struct's own nulls survive the rebuild.
    assert_equal(out.null_count(), 1)
    assert_false(out.is_valid(1))
    assert_true(out.field(0).as_string() == array(["a", None, "c"]))
    assert_equal(out.field(1).null_count(), 3)
    assert_true(out.field(2).as_int64() == array([1, 2, 3], int64))


def test_iceberg_projection_list_of_evolved_struct() raises:
    var a = _f("a", int32, 22)
    var file_element = _f("element", struct_([a.copy()]), 21)
    var file = _schema([_f("l", ListType(file_element.copy()), 20)])
    var table_element = _f(
        "element", struct_([_f("a", int64, 22), _f("b", string, 23)]), 21
    )
    var table = _schema([_f("l", ListType(table_element^), 20)])

    var values: DynArray = StructArray.from_arrays(
        [array([1, 2, 3], int32)], [a^]
    )
    var offsets = array([0, 2, 2, 3], int32)
    var l: DynArray = ListArray(
        dtype=ListType(file_element^).to_dyn(),
        length=3,
        nulls=0,
        offset=0,
        bitmap=None,
        offsets=offsets.buffer,
        values=values^,
    )
    var result = SchemaProjection(table, file).project(RecordBatch(file, [l^]))
    assert_true(result.schema == table)
    ref out = result.column(0).as_list()
    assert_true(out.offsets == offsets.buffer)
    ref elements = out.values().as_struct()
    assert_true(elements.field(0).as_int64() == array([1, 2, 3], int64))
    assert_equal(elements.field(1).null_count(), 3)
    assert_equal(len(elements.field(1)), 3)


def test_iceberg_projection_map_of_evolved_struct() raises:
    var key = _f("key", string, 31, nullable=False)
    var v = _f("v", int32, 33)
    var value = _f("value", struct_([v.copy()]), 32)
    var file_map = MapType(
        Field("entries", struct_([key.copy(), value.copy()]), False)
    )
    var file = _schema([_f("m", file_map.copy(), 30)])
    var table_value = _f(
        "value", struct_([_f("w", int64, 33), _f("extra", string, 34)]), 32
    )
    var table_map = MapType(
        Field("entries", struct_([key.copy(), table_value^]), False)
    )
    var table = _schema([_f("m", table_map.copy(), 30)])

    var values: DynArray = StructArray.from_arrays(
        [array([7, 8, 9], int32)], [v^]
    )
    var entries: DynArray = StructArray.from_arrays(
        [array(["p", "q", "r"]), values^], [key^, value^]
    )
    var offsets = array([0, 1, 3], int32)
    var m: DynArray = MapArray(
        dtype=file_map.copy().to_dyn(),
        length=2,
        nulls=0,
        offset=0,
        bitmap=None,
        offsets=offsets.buffer,
        values=entries^,
    )
    var result = SchemaProjection(table, file).project(RecordBatch(file, [m^]))
    assert_true(result.schema == table)
    ref out = result.column(0).as_map()
    assert_true(out.type() == table_map^.to_dyn())
    assert_true(out.offsets == offsets.buffer)
    ref kv = out.values().as_struct()
    assert_true(kv.field(0).as_string() == array(["p", "q", "r"]))
    ref w = kv.field(1).as_struct()
    assert_true(w.field(0).as_int64() == array([7, 8, 9], int64))
    assert_equal(w.field(1).null_count(), 3)


def test_iceberg_projection_name_mapping() raises:
    var file = _schema(
        [
            Field("a", int64),
            Field("s", struct_([Field("x", int32)])),
            Field("l", ListType(Field("item", struct_([Field("p", int32)])))),
            Field("m", MapType(string, int32)),
            Field("unmapped", int64),
        ]
    )
    var mapping = NameMapping(
        [
            MappedField(1, ["a", "alias"]),
            MappedField(2, ["s"], [MappedField(3, ["x"])]),
            MappedField(
                4, ["l"], [MappedField(5, ["element"], [MappedField(6, ["p"])])]
            ),
            MappedField(
                7, ["m"], [MappedField(8, ["key"]), MappedField(9, ["value"])]
            ),
        ]
    )
    var mapped = mapping.apply(file)
    assert_equal(mapped.fields[0].field_id().value(), 1)
    assert_equal(mapped.fields[1].field_id().value(), 2)
    ref s = mapped.fields[1].dtype.as_struct()
    assert_equal(s.fields[0].field_id().value(), 3)
    ref element = mapped.fields[2].dtype.as_list().value_field()
    assert_equal(mapped.fields[2].field_id().value(), 4)
    assert_equal(element.field_id().value(), 5)
    assert_equal(element.dtype.as_struct().fields[0].field_id().value(), 6)
    ref m = mapped.fields[3].dtype.as_map()
    assert_equal(m.key_field().field_id().value(), 8)
    assert_equal(m.item_field().field_id().value(), 9)
    assert_false(Bool(mapped.fields[4].field_id()))

    # The mapped schema projects by id: renamed in the table, still read.
    var table = _schema([_f("renamed", int64, 1), _f("other", int64, 99)])
    var p = SchemaProjection(table, mapped)
    assert_true(p.file_columns() == ["a"])
    var a: DynArray = array([5, 6], int64)
    var result = p.project(
        RecordBatch(_schema([Field("a", int64)]), [a.copy()])
    )
    assert_true(result.column(0).as_int64() == array([5, 6], int64))
    assert_equal(result.column(1).null_count(), 2)


def test_iceberg_projection_name_mapping_keeps_existing_ids() raises:
    var file = _schema([_f("a", int64, 42)])
    var mapped = NameMapping([MappedField(1, ["a"])]).apply(file)
    assert_equal(mapped.fields[0].field_id().value(), 42)


def test_iceberg_projection_constant_for_absent_column() raises:
    var file = _schema([_f("a", int64, 1)])
    var table = _schema(
        [_f("a", int64, 1), _f("region", string, 2, nullable=False)]
    )
    var constants = Dict[Int, DynArray]()
    constants[2] = array(["eu"])
    var p = SchemaProjection(table, file, constants^)
    assert_true(p.file_columns() == ["a"])
    var result = p.project(RecordBatch(file, [array([1, 2, 3], int64)]))
    assert_true(result.schema == table)
    assert_true(result.column(1).as_string() == array(["eu", "eu", "eu"]))


def test_iceberg_projection_constant_ignored_when_present() raises:
    var file = _schema([_f("a", int64, 1), _f("region", string, 2)])
    var table = _schema([_f("region", string, 2)])
    var constants = Dict[Int, DynArray]()
    constants[2] = array(["eu"])
    var region: DynArray = array(["us", None])
    var result = SchemaProjection(table, file, constants^).project(
        RecordBatch(file, [array([1, 2], int64), region.copy()])
    )
    assert_true(result == RecordBatch(table, [region.copy()]))
