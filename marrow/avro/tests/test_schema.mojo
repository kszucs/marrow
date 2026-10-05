# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro schemas: JSON parsing and writing, and the mapping to and from Arrow."""

from std.testing import assert_equal, assert_raises, assert_true

from ...avro import AvroKind, AvroSchema, from_arrow, to_arrow
from ...dtypes import (
    DynType,
    Field,
    ListType,
    binary,
    bool_,
    date32,
    decimal128,
    decimal256,
    dictionary,
    field,
    fixed_size_binary_,
    float32,
    float64,
    int32,
    int64,
    map_,
    microsecond,
    millisecond,
    month_day_nano_interval,
    nanosecond,
    string,
    struct_,
    time32,
    time64,
    timestamp,
    uint64,
)
from ...schema import Schema


def _column(type_json: String) raises -> Field:
    """The Arrow field of a one-field record whose field has `type_json`."""
    var text = (
        '{"type": "record", "name": "r", "fields": [{"name": "c", "type": '
        + type_json
        + "}]}"
    )
    return to_arrow(AvroSchema.parse(text)).fields[0].copy()


def _type_of(type_json: String) raises -> DynType:
    return _column(type_json).dtype.copy()


def test_avro_primitive_types() raises:
    assert_true(_type_of('"boolean"') == bool_.to_dyn())
    assert_true(_type_of('"int"') == int32.to_dyn())
    assert_true(_type_of('"long"') == int64.to_dyn())
    assert_true(_type_of('"float"') == float32.to_dyn())
    assert_true(_type_of('"double"') == float64.to_dyn())
    assert_true(_type_of('"bytes"') == binary.to_dyn())
    assert_true(_type_of('"string"') == string.to_dyn())
    assert_true(_type_of('{"type": "string"}') == string.to_dyn())
    assert_true(_type_of('"null"').is_null())
    assert_true(not _column('"long"').nullable)


def test_avro_logical_types() raises:
    assert_true(
        _type_of('{"type": "int", "logicalType": "date"}') == date32().to_dyn()
    )
    assert_true(
        _type_of('{"type": "int", "logicalType": "time-millis"}')
        == time32(millisecond).to_dyn()
    )
    assert_true(
        _type_of('{"type": "long", "logicalType": "time-micros"}')
        == time64(microsecond).to_dyn()
    )
    assert_true(
        _type_of('{"type": "long", "logicalType": "timestamp-micros"}')
        == timestamp(microsecond, "UTC").to_dyn()
    )
    assert_true(
        _type_of('{"type": "long", "logicalType": "timestamp-nanos"}')
        == timestamp(nanosecond, "UTC").to_dyn()
    )
    assert_true(
        _type_of('{"type": "long", "logicalType": "local-timestamp-millis"}')
        == timestamp(millisecond).to_dyn()
    )
    assert_true(
        _type_of(
            '{"type": "bytes", "logicalType": "decimal", "precision": 10,'
            ' "scale": 2}'
        )
        == decimal128(10, 2).to_dyn()
    )
    assert_true(
        _type_of(
            '{"type": "fixed", "name": "d", "size": 32, "logicalType":'
            ' "decimal", "precision": 76, "scale": 10}'
        )
        == decimal256(76, 10).to_dyn()
    )
    assert_true(
        _type_of(
            '{"type": "fixed", "name": "d", "size": 12, "logicalType":'
            ' "duration"}'
        )
        == month_day_nano_interval().to_dyn()
    )
    var uuid = _column(
        '{"type": "fixed", "name": "u", "size": 16, "logicalType": "uuid"}'
    )
    assert_true(uuid.dtype == fixed_size_binary_(16).to_dyn())
    assert_equal(uuid.metadata["ARROW:extension:name"], "arrow.uuid")
    # An unknown logical type, or an invalid one, reads as its base type.
    assert_true(
        _type_of('{"type": "long", "logicalType": "bogus"}') == int64.to_dyn()
    )
    assert_true(
        _type_of(
            '{"type": "fixed", "name": "d", "size": 2, "logicalType":'
            ' "decimal", "precision": 9}'
        )
        == fixed_size_binary_(2).to_dyn()
    )


def test_avro_complex_types() raises:
    assert_true(
        _type_of('{"type": "array", "items": "long"}')
        == ListType(Field("item", int64.to_dyn(), False)).to_dyn()
    )
    var m = _type_of('{"type": "map", "values": ["null", "int"]}')
    assert_true(m == map_(string.to_dyn(), int32.to_dyn()).to_dyn())
    assert_true(
        _type_of('{"type": "fixed", "name": "f", "size": 5}')
        == fixed_size_binary_(5).to_dyn()
    )
    assert_true(
        _type_of('{"type": "enum", "name": "e", "symbols": ["A", "B"]}')
        == dictionary(int32.to_dyn(), string.to_dyn()).to_dyn()
    )
    assert_true(
        _type_of(
            '{"type": "record", "name": "s", "fields": [{"name": "x", "type":'
            ' "int"}, {"name": "y", "type": ["string", "null"]}]}'
        )
        == struct_(
            Field("x", int32.to_dyn(), False), Field("y", string.to_dyn())
        ).to_dyn()
    )


def test_avro_nullable_unions() raises:
    var a = _column('["null", "string"]')
    assert_true(a.nullable)
    assert_true(a.dtype == string.to_dyn())
    var b = _column('["string", "null"]')
    assert_true(b.nullable)
    assert_true(b.dtype == string.to_dyn())
    var c = _column('["long"]')
    assert_true(not c.nullable)
    with assert_raises(contains="several non-null types"):
        _ = _column('["null", "string", "long"]')
    with assert_raises(contains="may not contain a union"):
        _ = _column('["null", ["string"]]')


def test_avro_named_references_and_namespaces() raises:
    var text = """{
        "type": "record", "name": "outer", "namespace": "com.example",
        "fields": [
            {"name": "a", "type": {"type": "fixed", "name": "md5", "size": 16}},
            {"name": "b", "type": "md5"},
            {"name": "c", "type": "com.example.md5"},
            {"name": "d", "type": {"type": "enum", "name": "color",
                                   "namespace": "other", "symbols": ["RED"]}},
            {"name": "e", "type": "other.color"}
        ]
    }"""
    var s = AvroSchema.parse(text)
    assert_equal(s.name, "com.example.outer")
    assert_equal(s.children[0].name, "com.example.md5")
    assert_equal(s.children[1].size, 16)
    assert_equal(s.children[2].size, 16)
    assert_equal(s.children[4].name, "other.color")
    assert_equal(len(s.children[4].symbols), 1)
    # Written back, each named type is defined once and referenced after.
    var again = AvroSchema.parse(s.to_json())
    assert_true(to_arrow(again) == to_arrow(s))
    with assert_raises(contains="unknown type 'nope'"):
        _ = _column('"nope"')
    with assert_raises(contains="defined twice"):
        _ = AvroSchema.parse(
            '{"type": "record", "name": "r", "fields": [{"name": "a", "type":'
            ' {"type": "fixed", "name": "f", "size": 1}}, {"name": "b",'
            ' "type": {"type": "fixed", "name": "f", "size": 2}}]}'
        )


def test_avro_recursive_type_is_refused() raises:
    var text = """{"type": "record", "name": "node", "fields": [
        {"name": "next", "type": ["null", "node"]}]}"""
    with assert_raises(contains="recursive type 'node'"):
        _ = AvroSchema.parse(text)


def test_avro_invalid_json_is_refused() raises:
    with assert_raises(contains="not valid JSON"):
        _ = AvroSchema.parse("{")
    with assert_raises(contains="missing 'fields'"):
        _ = AvroSchema.parse('{"type": "record", "name": "r"}')


def test_avro_field_ids_land_in_metadata() raises:
    # The shape of an Iceberg manifest's schema.
    var text = """{"type": "record", "name": "manifest_entry", "fields": [
        {"name": "status", "type": "int", "field-id": 0},
        {"name": "tags", "type": {"type": "array", "items": "string",
                                  "element-id": 7}, "field-id": 6},
        {"name": "props", "type": {"type": "map", "values": "long",
                                   "key-id": 9, "value-id": 10}, "field-id": 8},
        {"name": "sizes", "type": ["null", {"type": "array",
            "logicalType": "map", "items": {"type": "record", "name": "k12_v13",
            "fields": [{"name": "key", "type": "int", "field-id": 12},
                       {"name": "value", "type": "long", "field-id": 13}]}}],
         "field-id": 11}
    ]}"""
    var schema = to_arrow(AvroSchema.parse(text))
    assert_equal(schema.fields[0].metadata["field_id"], "0")
    ref tags = schema.fields[1]
    assert_equal(tags.metadata["field_id"], "6")
    assert_equal(tags.dtype.as_list().value_field().metadata["field_id"], "7")
    ref props = schema.fields[2]
    assert_equal(props.metadata["field_id"], "8")
    assert_equal(props.dtype.as_map().key_field().metadata["field_id"], "9")
    assert_equal(props.dtype.as_map().item_field().metadata["field_id"], "10")
    ref sizes = schema.fields[3]
    assert_true(sizes.nullable)
    assert_true(sizes.dtype.is_map())
    assert_true(sizes.dtype.as_map().key_type() == int32.to_dyn())
    assert_true(sizes.dtype.as_map().item_type() == int64.to_dyn())
    assert_equal(sizes.dtype.as_map().key_field().metadata["field_id"], "12")
    assert_equal(sizes.dtype.as_map().item_field().metadata["field_id"], "13")


def test_avro_from_arrow_roundtrips() raises:
    var id_md = Dict[String, String]()
    id_md["field_id"] = "1"
    var item_md = Dict[String, String]()
    item_md["field_id"] = "2"
    var schema = Schema(
        fields=[
            Field("id", int64.to_dyn(), False, id_md^),
            field("name", string.to_dyn()),
            field("flag", bool_.to_dyn()),
            field("f", float32.to_dyn()),
            field("day", date32().to_dyn()),
            field("ts", timestamp(microsecond, "UTC").to_dyn()),
            field("local", timestamp(nanosecond).to_dyn()),
            field("t", time64(microsecond).to_dyn()),
            field("price", decimal128(12, 3).to_dyn()),
            field("big", decimal256(60, 0).to_dyn()),
            field("digest", fixed_size_binary_(4).to_dyn()),
            field("blob", binary.to_dyn()),
            field("span", month_day_nano_interval().to_dyn()),
            field(
                "tags",
                ListType(
                    Field("item", string.to_dyn(), True, item_md^)
                ).to_dyn(),
            ),
            field("attrs", map_(string.to_dyn(), int64.to_dyn()).to_dyn()),
            field("counts", map_(int32.to_dyn(), int64.to_dyn()).to_dyn()),
            field(
                "point",
                struct_(
                    Field("x", float64.to_dyn(), False),
                    field("y", float64.to_dyn()),
                ).to_dyn(),
            ),
        ]
    )
    var avro = from_arrow(schema)
    var parsed = AvroSchema.parse(avro.to_json())
    assert_true(to_arrow(parsed) == schema)
    # Iceberg writes a decimal as a fixed of the fewest bytes that hold it.
    assert_equal(avro.children[8].children[1].kind, AvroKind.FIXED)
    assert_equal(avro.children[8].children[1].size, 6)
    assert_equal(avro.field_ids[0].value(), 1)
    assert_equal(avro.children[13].children[1].element_id.value(), 2)
    assert_equal(avro.field_props[1]["default"], "null")


def test_avro_from_arrow_refusals() raises:
    with assert_raises(contains="has no Avro form"):
        _ = from_arrow(Schema(fields=[field("u", uint64.to_dyn())]))
    with assert_raises(contains="not a valid field name"):
        _ = from_arrow(Schema(fields=[field("a b", int64.to_dyn())]))


def test_avro_ids_must_be_integers() raises:
    with assert_raises(contains="'field-id' is not an integer"):
        _ = AvroSchema.parse(
            '{"type": "record", "name": "r", "fields": [{"name": "a", "type":'
            ' "int", "field-id": "1"}]}'
        )
    var md = Dict[String, String]()
    md["field_id"] = "one"
    with assert_raises(contains="non-integer field_id"):
        _ = from_arrow(Schema(fields=[Field("a", int64.to_dyn(), True, md^)]))
