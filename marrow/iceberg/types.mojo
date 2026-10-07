# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Iceberg primitive types, as the type strings of a schema's JSON.

One mapping in each direction between an Iceberg primitive and the Arrow
dtype marrow reads it as — pyiceberg's and iceberg-rust's choice:

| Iceberg          | Arrow                        |
|------------------|------------------------------|
| `boolean`        | `bool`                       |
| `int` / `long`   | `int32` / `int64`            |
| `float`/`double` | `float32` / `float64`        |
| `decimal(P, S)`  | `decimal128(P, S)`, P <= 38  |
| `date`           | `date32`                     |
| `time`           | `time64[us]`                 |
| `timestamp`      | `timestamp[us]`              |
| `timestamptz`    | `timestamp[us, tz=UTC]`      |
| `timestamp_ns`   | `timestamp[ns]`              |
| `timestamptz_ns` | `timestamp[ns, tz=UTC]`      |
| `string`         | `string`                     |
| `uuid`           | `fixed_size_binary[16]`      |
| `fixed[L]`       | `fixed_size_binary[L]`       |
| `binary`         | `binary`                     |

The reverse direction also takes `large_string`, `large_binary`, the narrower
decimal widths and a timestamp zoned by any spelling of UTC, since a Parquet
file may carry those for an Iceberg column. A `fixed_size_binary[16]` names
`fixed[16]`: the dtype alone cannot tell it from a `uuid`.

Nested types come from a schema's JSON (`schema_from_json`): a struct's
fields, a list's element and a map's key and value each carry their Iceberg
id under `FIELD_ID_KEY`, and their names are the ones pyiceberg gives them —
`element`, `key`, `value` — so a schema read here equals pyiceberg's.
"""

from emberjson import Value as Json

from ..dtypes import (
    DecimalType,
    DynType,
    Field,
    FIELD_ID_KEY,
    ListType,
    MapType,
    StructType,
    binary,
    bool_,
    date32,
    decimal128,
    fixed_size_binary_,
    float32,
    float64,
    int32,
    int64,
    microsecond,
    nanosecond,
    string,
    time64,
    timestamp,
)
from ..errors import InvalidError, NotImplementedError, TypeError
from ..schema import Schema

from .document import (
    bool_member,
    expect_array,
    expect_member,
    expect_object,
    int_member,
    string_member,
)

comptime UUID_BYTES = 16
"""How many bytes an Iceberg `uuid` holds."""

comptime MAX_DECIMAL_PRECISION = 38
"""The most digits an Iceberg `decimal(P, S)` may declare."""


def _is_utc(zone: StringSlice) -> Bool:
    return zone == "UTC" or zone == "Etc/UTC" or zone == "Z" or zone == "+00:00"


def parse_int(text: StringSlice, what: StringSlice) raises InvalidError -> Int:
    """A non-negative decimal integer, whitespace around it allowed; `what`
    names it in the `InvalidError` raised for anything else."""
    var digits = text.strip()
    if digits == "":
        raise InvalidError(t"{what} is empty")
    var value = 0
    for b in digits.as_bytes():
        if Int(b) < ord("0") or Int(b) > ord("9"):
            raise InvalidError(t"{what} '{digits}' is not a number")
        value = value * 10 + (Int(b) - ord("0"))
    return value


def _parameters(
    name: StringSlice, open: StringSlice, close: StringSlice
) raises InvalidError -> String:
    """The text between `open` and a trailing `close` in `name`."""
    if not name.endswith(close):
        raise InvalidError(t"iceberg type: '{name}' lacks a closing '{close}'")
    var start = name.find(open)
    var end = name.byte_length() - close.byte_length()
    return String(name[byte = start + open.byte_length() : end])


def primitive_type(name: StringSlice) raises -> DynType:
    """The Arrow dtype of the Iceberg primitive type string `name`.

    Raises `NotImplementedError` for a legal type marrow does not read
    (`unknown`, `variant`, `geometry`, `geography`), and `InvalidError` for a
    string that is not an Iceberg type.
    """
    var text = name.strip()
    if text == "boolean":
        return bool_
    elif text == "int":
        return int32
    elif text == "long":
        return int64
    elif text == "float":
        return float32
    elif text == "double":
        return float64
    elif text == "date":
        return date32()
    elif text == "time":
        return time64(microsecond)
    elif text == "timestamp":
        return timestamp(microsecond)
    elif text == "timestamptz":
        return timestamp(microsecond, "UTC")
    elif text == "timestamp_ns":
        return timestamp(nanosecond)
    elif text == "timestamptz_ns":
        return timestamp(nanosecond, "UTC")
    elif text == "string":
        return string
    elif text == "uuid":
        return fixed_size_binary_(UUID_BYTES)
    elif text == "binary":
        return binary
    elif text.startswith("fixed"):
        var rest = text.removeprefix("fixed").lstrip()
        if not rest.startswith("["):
            raise InvalidError(t"iceberg type: malformed '{text}'")
        var length = parse_int(
            _parameters(rest, "[", "]"), "iceberg type: fixed length"
        )
        return fixed_size_binary_(length)
    elif text.startswith("decimal"):
        var rest = text.removeprefix("decimal").lstrip()
        if not rest.startswith("("):
            raise InvalidError(t"iceberg type: malformed '{text}'")
        var params = _parameters(rest, "(", ")")
        var parts = params.split(",")
        if len(parts) != 2:
            raise InvalidError(
                t"iceberg type: '{text}' needs a precision and a scale"
            )
        var precision = parse_int(parts[0], "iceberg type: decimal precision")
        var scale = parse_int(parts[1], "iceberg type: decimal scale")
        if precision < 1 or precision > MAX_DECIMAL_PRECISION:
            raise InvalidError(
                t"iceberg type: '{text}': precision must be 1 to"
                t" {MAX_DECIMAL_PRECISION}"
            )
        if scale > precision:
            raise InvalidError(
                t"iceberg type: '{text}': scale exceeds precision"
            )
        return decimal128(precision, scale)
    elif (
        text == "unknown"
        or text == "variant"
        or text.startswith("geometry")
        or text.startswith("geography")
    ):
        raise NotImplementedError(t"iceberg type: '{text}' is not supported")
    raise InvalidError(t"iceberg type: unknown primitive type '{text}'")


def primitive_name(dtype: DynType) raises -> String:
    """The Iceberg primitive type string for the Arrow dtype `dtype`.

    Raises `TypeError` for a dtype with no Iceberg counterpart.
    """
    if dtype.is_bool():
        return "boolean"
    elif dtype.is_int32():
        return "int"
    elif dtype.is_int64():
        return "long"
    elif dtype.is_float32():
        return "float"
    elif dtype.is_float64():
        return "double"
    elif dtype.is_date32():
        return "date"
    elif dtype.is_time64() and dtype.as_time64().unit == microsecond:
        return "time"
    elif dtype.is_timestamp():
        ref ts = dtype.as_timestamp()
        var zoned = ts.timezone != ""
        if zoned and not _is_utc(ts.timezone):
            raise TypeError(
                t"iceberg type: {dtype} is zoned to a time zone other than UTC"
            )
        if ts.unit == microsecond:
            return "timestamptz" if zoned else "timestamp"
        elif ts.unit == nanosecond:
            return "timestamptz_ns" if zoned else "timestamp_ns"
    elif dtype.is_string_like():
        return "string"
    elif dtype.is_binary() or dtype.is_large_binary():
        return "binary"
    elif dtype.is_fixed_size_binary():
        return String(t"fixed[{dtype.as_fixed_size_binary().byte_width}]")
    elif dtype.is_decimal():

        def decimal[T: DecimalType](d: T) raises {imm} -> String:
            if d.precision() > MAX_DECIMAL_PRECISION:
                raise TypeError(
                    t"iceberg type: decimal precision {d.precision()} exceeds"
                    t" {MAX_DECIMAL_PRECISION}"
                )
            return String(t"decimal({d.precision()}, {d.scale()})")

        return dtype.dispatch_decimal(decimal)
    raise TypeError(t"iceberg type: {dtype} has no Iceberg primitive type")


def _with_id(var field: Field, id: Int) -> Field:
    field.metadata[FIELD_ID_KEY] = String(id)
    return field^


def type_from_json(json: Json) raises -> DynType:
    """An Iceberg type: a primitive's name, or a struct, list or map object."""
    if json.is_string():
        return primitive_type(json.string())
    expect_object(json, "type")
    ref o = json.object()
    var kind = string_member(o, "type")
    if kind == "struct":
        expect_member(o, "fields")
        return StructType(_fields_from_json(o["fields"]))
    elif kind == "list":
        expect_member(o, "element")
        var element = Field(
            "element",
            type_from_json(o["element"]),
            nullable=not bool_member(o, "element-required"),
        )
        return ListType(_with_id(element^, int_member(o, "element-id")))
    elif kind == "map":
        expect_member(o, "key")
        expect_member(o, "value")
        var key = Field("key", type_from_json(o["key"]), nullable=False)
        var value = Field(
            "value",
            type_from_json(o["value"]),
            nullable=not bool_member(o, "value-required"),
        )
        var entries = Field(
            "key_value",
            StructType(
                [
                    _with_id(key^, int_member(o, "key-id")),
                    _with_id(value^, int_member(o, "value-id")),
                ]
            ),
            nullable=False,
        )
        return MapType(entries^)
    raise InvalidError(t"iceberg: unknown nested type '{kind}'")


def _fields_from_json(json: Json) raises -> List[Field]:
    expect_array(json, "fields")
    var fields = List[Field]()
    for ref f in json.array():
        expect_object(f, "field")
        ref o = f.object()
        expect_member(o, "type")
        var field = Field(
            string_member(o, "name"),
            type_from_json(o["type"]),
            nullable=not bool_member(o, "required"),
        )
        fields.append(_with_id(field^, int_member(o, "id")))
    return fields^


def schema_from_json(json: Json) raises -> Schema:
    """A schema object — `{"type": "struct", "fields": [...]}` — as an Arrow
    schema whose every field, nested ones included, carries its id."""
    expect_object(json, "schema")
    ref o = json.object()
    expect_member(o, "fields")
    return Schema(fields=_fields_from_json(o["fields"]))
