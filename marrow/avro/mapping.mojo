# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro schemas mapped to and from Arrow.

`to_arrow` gives the Arrow schema a file reads as, `from_arrow` the Avro
schema a table writes as, and `writes_as` which columns a node accepts:

| Avro | Arrow |
|---|---|
| `null`, `boolean`, `int`, `long`, `float`, `double` | null, bool, int32, int64, float32, float64 |
| `bytes`, `string` | binary, string |
| `record` | struct |
| `enum` | dictionary<int32, string> |
| `array` | list |
| `map` | map<string, V> |
| `fixed` | fixed_size_binary |
| `["null", T]`, `[T, "null"]` | T, nullable |
| `date` | date32 |
| `time-millis`, `time-micros` | time32[ms], time64[us] |
| `timestamp-{millis,micros,nanos}` | timestamp[unit, UTC] |
| `local-timestamp-{millis,micros,nanos}` | timestamp[unit] |
| `decimal` on `bytes` or `fixed` | decimal128, or decimal256 past 38 digits |
| `duration` on `fixed(12)` | interval[month_day_nano] |
| `uuid` on `fixed(16)` or `string` | fixed_size_binary(16), extension `arrow.uuid` |
| `array` of `record{key, value}`, `logicalType: map` | map<K, V> |

The last row is how Iceberg writes a map whose keys are not strings.

A union of more than one non-null branch has no Arrow form here, since marrow
has no union layout, and neither does a recursive record.

**Field ids.** A record field's `field-id`, an array's `element-id` and a map's
`key-id` / `value-id` land in the `field_id` metadata of the Arrow field they
describe, and `from_arrow` writes them back from there. That is the whole of
what Iceberg needs from Avro's schema: it projects and evolves columns by those
ids itself, so this module maps the writer's schema and does no Avro schema
resolution.
"""

from ..dtypes import (
    DynType,
    Field,
    ListType,
    MapType,
    binary,
    bool_,
    date32,
    decimal128,
    decimal256,
    dictionary,
    fixed_size_binary_,
    float32,
    float64,
    int32,
    int64,
    microsecond,
    millisecond,
    month_day_nano_interval,
    nanosecond,
    null,
    second,
    string,
    struct_,
    time32,
    time64,
    timestamp,
)
from ..errors import InvalidError, NotImplementedError
from ..schema import Schema
from .schema import AvroKind, AvroSchema, check_name

comptime FIELD_ID = "field_id"
"""The Arrow field metadata key that carries an Avro field, element, key or
value id."""

comptime EXTENSION_NAME = "ARROW:extension:name"
comptime UUID_EXTENSION = "arrow.uuid"


# ---------------------------------------------------------------------------
# Avro -> Arrow
# ---------------------------------------------------------------------------


def to_arrow(avro: AvroSchema) raises -> Schema:
    """The Arrow schema a file of `avro` records reads as: a column per field
    of the record."""
    if avro.kind != AvroKind.RECORD:
        raise NotImplementedError(
            t"avro: only a record schema reads as a table, got {avro.kind}"
        )
    return Schema(fields=arrow_type(avro).as_struct().fields.copy())


def arrow_field(record: AvroSchema, i: Int) raises -> Field:
    """Field `i` of `record` as an Arrow field."""
    return _arrow_field(
        record.field_names[i],
        record.children[i],
        _id_metadata(record.field_ids[i]),
    )


def _id_metadata(id: Optional[Int]) -> Dict[String, String]:
    var md = Dict[String, String]()
    if id:
        md[FIELD_ID] = String(id.value())
    return md^


def _arrow_field(
    name: String, schema: AvroSchema, var metadata: Dict[String, String]
) raises -> Field:
    """`schema` as a field named `name`: nullable when it is a union with a
    `null` branch."""
    if schema.kind == AvroKind.UNION:
        var v = schema.value_index()
        if v < 0:
            return Field(name, null, True, metadata^)
        var nullable = schema.null_index() >= 0
        return _arrow_field_of(name, schema.children[v], nullable, metadata^)
    return _arrow_field_of(name, schema, False, metadata^)


def _arrow_field_of(
    name: String,
    schema: AvroSchema,
    nullable: Bool,
    var metadata: Dict[String, String],
) raises -> Field:
    if is_uuid(schema):
        metadata[EXTENSION_NAME] = UUID_EXTENSION
    return Field(name, arrow_type(schema), nullable, metadata^)


def is_uuid(schema: AvroSchema) -> Bool:
    """A `uuid`: 16 raw bytes, or its 36-character RFC 4122 text."""
    if schema.logical != "uuid":
        return False
    return schema.kind == AvroKind.STRING or (
        schema.kind == AvroKind.FIXED and schema.size == 16
    )


def is_logical_map(schema: AvroSchema) -> Bool:
    """An Iceberg map: an `array` of two-field `record{key, value}` items
    marked `logicalType: map`."""
    if schema.kind != AvroKind.ARRAY or schema.logical != "map":
        return False
    ref items = schema.children[0]
    return (
        items.kind == AvroKind.RECORD
        and len(items.field_names) == 2
        and items.field_names[0] == "key"
        and items.field_names[1] == "value"
    )


def decimal_type(schema: AvroSchema) raises -> Optional[DynType]:
    """The Arrow decimal a `decimal` logical type reads as, or `None` when the
    annotation is invalid and the spec says to ignore it."""
    if schema.logical != "decimal":
        return None
    var p = schema.precision
    var s = schema.scale
    if p < 1 or s < 0 or s > p:
        return None
    if schema.kind == AvroKind.FIXED and p > _max_precision(schema.size):
        return None
    if p <= 38:
        return decimal128(p, s).to_dyn()
    if p <= 76:
        return decimal256(p, s).to_dyn()
    raise NotImplementedError(
        t"avro: decimal precision {p} exceeds Arrow's 76 digits"
    )


def arrow_type(schema: AvroSchema) raises -> DynType:
    """The Arrow type of a non-union schema node."""
    ref k = schema.kind
    ref lt = schema.logical
    if k == AvroKind.NULL:
        return null.to_dyn()
    elif k == AvroKind.BOOLEAN:
        return bool_.to_dyn()
    elif k == AvroKind.INT:
        if lt == "date":
            return date32().to_dyn()
        elif lt == "time-millis":
            return time32(millisecond).to_dyn()
        return int32.to_dyn()
    elif k == AvroKind.LONG:
        if lt == "time-micros":
            return time64(microsecond).to_dyn()
        elif lt == "timestamp-millis":
            return timestamp(millisecond, "UTC").to_dyn()
        elif lt == "timestamp-micros":
            return timestamp(microsecond, "UTC").to_dyn()
        elif lt == "timestamp-nanos":
            return timestamp(nanosecond, "UTC").to_dyn()
        elif lt == "local-timestamp-millis":
            return timestamp(millisecond).to_dyn()
        elif lt == "local-timestamp-micros":
            return timestamp(microsecond).to_dyn()
        elif lt == "local-timestamp-nanos":
            return timestamp(nanosecond).to_dyn()
        return int64.to_dyn()
    elif k == AvroKind.FLOAT:
        return float32.to_dyn()
    elif k == AvroKind.DOUBLE:
        return float64.to_dyn()
    elif k == AvroKind.BYTES:
        var d = decimal_type(schema)
        if d:
            return d.value().copy()
        return binary.to_dyn()
    elif k == AvroKind.STRING:
        if is_uuid(schema):
            return fixed_size_binary_(16).to_dyn()
        return string.to_dyn()
    elif k == AvroKind.RECORD:
        var fields = List[Field]()
        for i in range(len(schema.children)):
            fields.append(arrow_field(schema, i))
        return struct_(fields^).to_dyn()
    elif k == AvroKind.ENUM:
        return dictionary(int32.to_dyn(), string.to_dyn()).to_dyn()
    elif k == AvroKind.ARRAY:
        if is_logical_map(schema):
            ref items = schema.children[0]
            var key = arrow_field(items, 0)
            if key.nullable:
                raise NotImplementedError(
                    "avro: a map with nullable keys has no Arrow form"
                )
            return _map_type(key^, arrow_field(items, 1))
        return ListType(
            _arrow_field(
                "item",
                schema.children[0],
                _id_metadata(schema.element_id),
            )
        ).to_dyn()
    elif k == AvroKind.MAP:
        var key = Field(
            "key", string.to_dyn(), False, _id_metadata(schema.key_id)
        )
        var value = _arrow_field(
            "value", schema.children[0], _id_metadata(schema.value_id)
        )
        return _map_type(key^, value^)
    elif k == AvroKind.FIXED:
        var d = decimal_type(schema)
        if d:
            return d.value().copy()
        if lt == "duration" and schema.size == 12:
            return month_day_nano_interval().to_dyn()
        return fixed_size_binary_(schema.size).to_dyn()
    raise NotImplementedError(
        t"avro: a union of several non-null types has no Arrow form: {schema}"
    )


def _map_type(var key: Field, var value: Field) -> DynType:
    return MapType(
        Field("entries", struct_([key^, value^]).to_dyn(), False)
    ).to_dyn()


def _precision_scale(dtype: DynType) -> Tuple[Int, Int]:
    """A decimal128 or decimal256 type's precision and scale."""
    if dtype.is_decimal128():
        ref d = dtype.as_decimal128()
        return (d.precision(), d.scale())
    ref d = dtype.as_decimal256()
    return (d.precision(), d.scale())


def _max_precision(size: Int) -> Int:
    """The most decimal digits a `size`-byte two's-complement integer holds:
    floor(log10(2^(8 size - 1) - 1))."""
    if size <= 0:
        return 0
    return Int(Float64(8 * size - 1) * 0.30102999566398120)


def _min_bytes(precision: Int) -> Int:
    """The fewest bytes whose two's complement holds every `precision`-digit
    integer -- the `fixed` size Iceberg requires of a decimal."""
    var n = 1
    while _max_precision(n) < precision:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Arrow -> Avro
# ---------------------------------------------------------------------------


def from_arrow(
    schema: Schema, record_name: String = "topLevelRecord"
) raises -> AvroSchema:
    """The Avro schema a table of `schema` writes as: a record named
    `record_name` with a field per column. Nested records and fixeds are named
    by their path below it, which keeps every name unique."""
    check_name(record_name, "record")
    return _record_of(schema.fields, record_name)


def _id_of(f: Field) raises -> Optional[Int]:
    """`f`'s `field_id` metadata, if it has one."""
    var id = f.metadata.get(FIELD_ID)
    if not id:
        return None
    try:
        return atol(id.value())
    except:
        raise InvalidError(
            t"avro: field '{f.name}' has a non-integer {FIELD_ID}:"
            t" '{id.value()}'"
        )


def _value_of(f: Field, namespace: String) raises -> AvroSchema:
    """An array item or map value: nullable becomes a union."""
    var t = _from_type(f, namespace)
    if f.nullable and t.kind != AvroKind.NULL:
        return AvroSchema.optional(t^)
    return t^


def _from_type(f: Field, namespace: String) raises -> AvroSchema:
    ref dt = f.dtype
    var path = namespace + "." + f.name
    if dt.is_null():
        return AvroSchema(AvroKind.NULL)
    elif dt.is_bool():
        return AvroSchema(AvroKind.BOOLEAN)
    elif dt.is_int32():
        return AvroSchema(AvroKind.INT)
    elif dt.is_int64():
        return AvroSchema(AvroKind.LONG)
    elif dt.is_float32():
        return AvroSchema(AvroKind.FLOAT)
    elif dt.is_float64():
        return AvroSchema(AvroKind.DOUBLE)
    elif dt.is_string() or dt.is_large_string():
        return AvroSchema(AvroKind.STRING)
    elif dt.is_binary() or dt.is_large_binary():
        return AvroSchema(AvroKind.BYTES)
    elif dt.is_dictionary() and dt.as_dictionary().value_type().is_string():
        return AvroSchema(AvroKind.STRING)
    elif dt.is_fixed_size_binary():
        var s = AvroSchema.fixed(path, dt.as_fixed_size_binary().byte_width)
        var ext = f.metadata.get(EXTENSION_NAME)
        if ext and ext.value() == UUID_EXTENSION and s.size == 16:
            s.logical = "uuid"
        return s^
    elif dt.is_date32():
        return AvroSchema(AvroKind.INT, logical="date")
    elif dt.is_time32() and dt.as_time32().unit == millisecond:
        return AvroSchema(AvroKind.INT, logical="time-millis")
    elif dt.is_time64() and dt.as_time64().unit == microsecond:
        return AvroSchema(AvroKind.LONG, logical="time-micros")
    elif dt.is_timestamp() and dt.as_timestamp().unit != second:
        ref ts = dt.as_timestamp()
        var unit = String("millis")
        if ts.unit == microsecond:
            unit = "micros"
        elif ts.unit == nanosecond:
            unit = "nanos"
        var local = ts.timezone == ""
        return AvroSchema(
            AvroKind.LONG,
            logical=("local-timestamp-" if local else "timestamp-") + unit,
        )
    elif dt.is_decimal128() or dt.is_decimal256():
        var p, sc = _precision_scale(dt)
        var s = AvroSchema.fixed(path, _min_bytes(p))
        s.logical = "decimal"
        s.precision = p
        s.scale = sc
        return s^
    elif dt.is_month_day_nano_interval():
        var s = AvroSchema.fixed(path, 12)
        s.logical = "duration"
        return s^
    elif dt.is_list() or dt.is_large_list():
        var item: Field
        if dt.is_list():
            item = dt.as_list().value_field().copy()
        else:
            item = dt.as_large_list().value_field().copy()
        var s = AvroSchema.array(_value_of(item, path))
        s.element_id = _id_of(item)
        return s^
    elif dt.is_map():
        ref m = dt.as_map()
        var key = m.key_field()
        var value = m.item_field()
        if key.dtype.is_string() or key.dtype.is_large_string():
            var s = AvroSchema.map(_value_of(value, path))
            s.key_id = _id_of(key)
            s.value_id = _id_of(value)
            return s^
        # Iceberg's form for a map whose keys are not strings.
        var entry = _record_of([key^, value^], path + ".entries")
        var s = AvroSchema.array(entry^)
        s.logical = "map"
        return s^
    elif dt.is_struct():
        return _record_of(dt.as_struct().fields, path)
    raise NotImplementedError(
        t"avro: column '{f.name}' of type {dt} has no Avro form"
    )


def _record_of(fields: List[Field], name: String) raises -> AvroSchema:
    var names = List[String]()
    var children = List[AvroSchema]()
    var ids = List[Optional[Int]]()
    var props = List[Dict[String, String]]()
    for ref f in fields:
        check_name(f.name, "field")
        var t = _value_of(f, name)
        # A nullable field defaults to null, the branch its union lists first.
        var p = Dict[String, String]()
        if t.kind == AvroKind.UNION:
            p["default"] = "null"
        names.append(f.name)
        children.append(t^)
        ids.append(_id_of(f))
        props.append(p^)
    return AvroSchema.record(name, names^, children^, ids^, props^)


# ---------------------------------------------------------------------------
# Writing: which Arrow columns a schema node accepts
# ---------------------------------------------------------------------------


def writes_as(dtype: DynType, avro: AvroSchema) raises -> Bool:
    """Whether a column of `dtype` holds what the leaf node `avro` stores: the
    type `avro` reads back as, or one whose values it stores unchanged -- the
    large layouts, an instant in any time zone as `timestamp-*`, a decimal of
    either width with the same precision and scale, and for an enum any string
    column. A dictionary column reaches here as its values."""
    var expected = (
        string.to_dyn() if avro.kind == AvroKind.ENUM else arrow_type(avro)
    )
    if dtype.is_timestamp() and expected.is_timestamp():
        ref t = dtype.as_timestamp()
        ref e = expected.as_timestamp()
        return t.unit == e.unit and (t.timezone == "") == (e.timezone == "")
    if dtype.is_large_binary():
        return expected.is_binary()
    if dtype.is_large_string():
        return expected.is_string()
    if (dtype.is_decimal128() or dtype.is_decimal256()) and (
        expected.is_decimal128() or expected.is_decimal256()
    ):
        return _precision_scale(dtype) == _precision_scale(expected)
    return dtype == expected


# ---------------------------------------------------------------------------
# Leaves: how a leaf column's values are laid out, for the decoder and encoder
# ---------------------------------------------------------------------------

comptime LEAF_BOOL = 1
comptime LEAF_INT32 = 2
comptime LEAF_INT64 = 3
comptime LEAF_FLOAT32 = 4
comptime LEAF_FLOAT64 = 5
comptime LEAF_STRING = 6
comptime LEAF_LARGE_STRING = 7
comptime LEAF_BINARY = 8
comptime LEAF_LARGE_BINARY = 9
comptime LEAF_FIXED = 10
comptime LEAF_UUID_TEXT = 11
comptime LEAF_DATE32 = 12
comptime LEAF_TIME32 = 13
comptime LEAF_TIME64 = 14
comptime LEAF_TIMESTAMP = 15
comptime LEAF_DECIMAL128 = 16
comptime LEAF_DECIMAL256 = 17
comptime LEAF_DURATION = 18


def leaf_of(dtype: DynType, kind: AvroKind) -> Int:
    """The `LEAF_*` layout of a leaf column of `dtype` stored as an Avro
    `kind`, or 0 when `dtype` is not a leaf this module maps. A 16-byte
    fixed-size binary stored as a `string` is a uuid's text."""
    if dtype.is_bool():
        return LEAF_BOOL
    elif dtype.is_int32():
        return LEAF_INT32
    elif dtype.is_int64():
        return LEAF_INT64
    elif dtype.is_float32():
        return LEAF_FLOAT32
    elif dtype.is_float64():
        return LEAF_FLOAT64
    elif dtype.is_string():
        return LEAF_STRING
    elif dtype.is_large_string():
        return LEAF_LARGE_STRING
    elif dtype.is_binary():
        return LEAF_BINARY
    elif dtype.is_large_binary():
        return LEAF_LARGE_BINARY
    elif dtype.is_fixed_size_binary():
        return LEAF_UUID_TEXT if kind == AvroKind.STRING else LEAF_FIXED
    elif dtype.is_date32():
        return LEAF_DATE32
    elif dtype.is_time32():
        return LEAF_TIME32
    elif dtype.is_time64():
        return LEAF_TIME64
    elif dtype.is_timestamp():
        return LEAF_TIMESTAMP
    elif dtype.is_decimal128():
        return LEAF_DECIMAL128
    elif dtype.is_decimal256():
        return LEAF_DECIMAL256
    elif dtype.is_month_day_nano_interval():
        return LEAF_DURATION
    return 0
