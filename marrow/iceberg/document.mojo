# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Reading Iceberg's JSON documents: table metadata, schemas, name mappings.

The checks every parser here shares, each failing as `InvalidError` with the
key and the document it was looked up in, so a malformed file names what is
wrong with it. Navigation stays at the call site (`json.object()`,
`o[key]`): a reference into a `Json` cannot be handed back through a helper.
"""

from emberjson import Object as JsonObject, Value as Json

from ..errors import InvalidError


def parse_json(text: StringSlice, what: String) raises InvalidError -> Json:
    try:
        return Json(parse_bytes=text.as_bytes())
    except e:
        raise InvalidError(t"iceberg: {what} is not valid JSON: {e}")


def expect_object(json: Json, what: String) raises InvalidError:
    if not json.is_object():
        raise InvalidError(t"iceberg: {what} is not an object: {json}")


def expect_array(json: Json, what: String) raises InvalidError:
    if not json.is_array():
        raise InvalidError(t"iceberg: {what} is not an array: {json}")


def expect_member(o: JsonObject, key: String) raises InvalidError:
    if key not in o:
        raise InvalidError(t"iceberg: missing '{key}' in {o}")


def has(o: JsonObject, key: String) -> Bool:
    """Whether `key` is present and not `null` — writers spell an absent
    optional either way."""
    try:
        return key in o and not o[key].is_null()
    except:
        return False


def as_int(json: Json, what: String) raises InvalidError -> Int:
    """An integer, written signed or unsigned; JSON does not say which."""
    if json.is_int() or json.is_uint():
        return Int(json.int())
    raise InvalidError(t"iceberg: {what} is not an integer: {json}")


def as_string(json: Json, what: String) raises InvalidError -> String:
    if not json.is_string():
        raise InvalidError(t"iceberg: {what} is not a string: {json}")
    return json.string()


def as_bool(json: Json, what: String) raises InvalidError -> Bool:
    if not json.is_bool():
        raise InvalidError(t"iceberg: {what} is not a boolean: {json}")
    return json.bool()


def int_member(o: JsonObject, key: String) raises -> Int:
    expect_member(o, key)
    ref value = o[key]
    return as_int(value, key)


def string_member(o: JsonObject, key: String) raises -> String:
    expect_member(o, key)
    ref value = o[key]
    return as_string(value, key)


def bool_member(o: JsonObject, key: String) raises -> Bool:
    expect_member(o, key)
    ref value = o[key]
    return as_bool(value, key)


def optional_int(o: JsonObject, key: String) raises -> Optional[Int]:
    if has(o, key):
        ref value = o[key]
        return as_int(value, key)
    return None
