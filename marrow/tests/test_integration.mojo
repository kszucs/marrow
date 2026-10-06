# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The integration JSON reader, one type family at a time.

The snippets are shaped like `archery`'s generated files (`datagen.py`): 64-bit
values as strings, binary as uppercase hex, booleans as JSON `true`/`false`.
The whole generated corpus runs in the archery suite, `pixi run -e integration
integration`.
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ..arrays import DynArray
from ..builders import array
from ..dtypes import (
    DynType,
    Field,
    FixedSizeListType,
    LargeListType,
    ListType,
    MapType,
    date32,
    decimal128,
    decimal256,
    dictionary,
    fixed_size_binary_,
    float64,
    int8,
    int16,
    int32,
    large_string,
    nanosecond,
    null,
    string,
    string_view,
    struct_,
    time64,
    timestamp,
    year_month_interval,
)
from ..tabular import RecordBatch

from ..integration import IntegrationJson


def _file(fields: String, columns: String, count: Int = 3) -> String:
    """A one-batch file: the given fields, and one batch of `count` rows."""
    return String(
        '{"schema": {"fields": [',
        fields,
        ']}, "batches": [{"count": ',
        count,
        ', "columns": [',
        columns,
        "]}]}",
    )


def _batch(text: String) raises -> RecordBatch:
    var parsed = IntegrationJson.parse(text)
    assert_equal(len(parsed.batches), 1)
    return parsed.batches[0].copy()


def _raises(text: String, contains: String) raises:
    with assert_raises(contains=contains):
        _ = IntegrationJson.parse(text)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_integration_json_schema_and_metadata() raises:
    var parsed = IntegrationJson.parse(
        '{"schema": {"fields": [{"name": "a", "nullable": false, "type":'
        ' {"name": "int", "isSigned": true, "bitWidth": 8}, "children": [],'
        ' "metadata": [{"key": "k", "value": "v"}]}], "metadata": [{"key":'
        ' "s", "value": "t"}]}, "batches": []}'
    )
    assert_equal(len(parsed.batches), 0)
    ref f = parsed.schema.fields[0]
    assert_equal(f.name, "a")
    assert_false(f.nullable)
    assert_true(f.dtype == int8)
    assert_equal(f.metadata["k"], "v")
    assert_equal(parsed.schema.metadata["s"], "t")


def test_integration_json_nested_types_keep_their_names() raises:
    var parsed = IntegrationJson.parse(
        '{"schema": {"fields": [{"name": "l", "nullable": true, "type":'
        ' {"name": "list"}, "children": [{"name": "element", "nullable":'
        ' false, "type": {"name": "int", "isSigned": true, "bitWidth": 32},'
        ' "children": []}]}, {"name": "m", "nullable": true, "type": {"name":'
        ' "map", "keysSorted": true}, "children": [{"name": "some_entries",'
        ' "nullable": false, "type": {"name": "struct"}, "children":'
        ' [{"name": "some_key", "nullable": false, "type": {"name": "utf8"},'
        ' "children": []}, {"name": "some_value", "nullable": true, "type":'
        ' {"name": "int", "isSigned": true, "bitWidth": 32}, "children":'
        ' []}]}]}]}, "batches": []}'
    )
    assert_true(
        parsed.schema.fields[0].dtype
        == ListType(Field("element", int32, nullable=False))
    )
    var entries = Field(
        "some_entries",
        struct_(
            Field("some_key", string, nullable=False),
            Field("some_value", int32),
        ),
        nullable=False,
    )
    assert_true(
        parsed.schema.fields[1].dtype == MapType(entries^, keys_sorted=True)
    )


def test_integration_json_temporal_types() raises:
    var parsed = IntegrationJson.parse(
        '{"schema": {"fields": [{"name": "d", "nullable": true, "type":'
        ' {"name": "date", "unit": "DAY"}, "children": []}, {"name": "t",'
        ' "nullable": true, "type": {"name": "time", "unit": "NANOSECOND",'
        ' "bitWidth": 64}, "children": []}, {"name": "ts", "nullable": true,'
        ' "type": {"name": "timestamp", "unit": "NANOSECOND", "timezone":'
        ' "US/Eastern"}, "children": []}, {"name": "i", "nullable": true,'
        ' "type": {"name": "interval", "unit": "YEAR_MONTH"}, "children":'
        " []}]}}"
    )
    ref fields = parsed.schema.fields
    assert_true(fields[0].dtype == date32())
    assert_true(fields[1].dtype == time64(nanosecond))
    assert_true(fields[2].dtype == timestamp(nanosecond, "US/Eastern"))
    assert_true(fields[3].dtype == year_month_interval())


def test_integration_json_time_width_must_match_its_unit() raises:
    _raises(
        (
            '{"schema": {"fields": [{"name": "t", "nullable": true, "type":'
            ' {"name": "time", "unit": "SECOND", "bitWidth": 64}, "children":'
            " []}]}}"
        ),
        "64-bit time",
    )


def test_integration_json_union_is_not_supported() raises:
    _raises(
        (
            '{"schema": {"fields": [{"name": "u", "nullable": true, "type":'
            ' {"name": "union", "mode": "SPARSE", "typeIds": []}, "children":'
            " []}]}}"
        ),
        "NotImplementedError",
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_integration_json_integers_as_numbers_and_strings() raises:
    var b = _batch(
        _file(
            (
                '{"name": "a", "nullable": true, "type": {"name": "int",'
                ' "isSigned": true, "bitWidth": 16}, "children": []}, {"name":'
                ' "b", "nullable": true, "type": {"name": "int", "isSigned":'
                ' true, "bitWidth": 64}, "children": []}, {"name": "c",'
                ' "nullable": false, "type": {"name": "int", "isSigned": false,'
                ' "bitWidth": 64}, "children": []}'
            ),
            (
                '{"name": "a", "count": 3, "VALIDITY": [1, 0, 1], "DATA": [-5,'
                ' 0, 7]}, {"name": "b", "count": 3, "VALIDITY": [1, 1, 1],'
                ' "DATA": ["-9223372036854775808", "0",'
                ' "9223372036854775807"]}, {"name": "c", "count": 3,'
                ' "VALIDITY": [1, 1, 1], "DATA": ["18446744073709551615", "1",'
                ' "0"]}'
            ),
        )
    )
    assert_true(b.columns[0].as_int16() == array([-5, None, 7], int16))
    assert_equal(b.columns[1].as_int64()[0].value(), Int64.MIN)
    assert_equal(b.columns[1].as_int64()[2].value(), Int64.MAX)
    assert_equal(b.columns[2].as_uint64()[0].value(), UInt64.MAX)
    assert_equal(b.columns[2].as_uint64()[1].value(), 1)


def test_integration_json_floats_accept_integral_literals() raises:
    var b = _batch(
        _file(
            (
                '{"name": "f", "nullable": true, "type": {"name":'
                ' "floatingpoint", "precision": "DOUBLE"}, "children": []}'
            ),
            (
                '{"name": "f", "count": 3, "VALIDITY": [1, 1, 0], "DATA": [1.5,'
                " 2, 0]}"
            ),
        )
    )
    assert_true(b.columns[0].as_float64() == array([1.5, 2.0, None], float64))


def test_integration_json_half_floats_are_bit_patterns() raises:
    # 0x3C00 is 1.0 and 0xC000 is -2.0, as Arrow C++ writes `HALF` values.
    var b = _batch(
        _file(
            (
                '{"name": "h", "nullable": true, "type": {"name":'
                ' "floatingpoint", "precision": "HALF"}, "children": []}'
            ),
            (
                '{"name": "h", "count": 2, "VALIDITY": [1, 1], "DATA": [15360,'
                " 49152]}"
            ),
            count=2,
        )
    )
    assert_equal(b.columns[0].as_float16()[0].value(), 1.0)
    assert_equal(b.columns[0].as_float16()[1].value(), -2.0)


def test_integration_json_booleans_are_json_booleans() raises:
    var b = _batch(
        _file(
            (
                '{"name": "b", "nullable": true, "type": {"name": "bool"},'
                ' "children": []}'
            ),
            (
                '{"name": "b", "count": 3, "VALIDITY": [1, 0, 1], "DATA":'
                " [true, false, false]}"
            ),
        )
    )
    assert_true(b.columns[0].as_bool() == array([True, None, False]))


def test_integration_json_strings_and_hex_binary() raises:
    var b = _batch(
        _file(
            (
                '{"name": "s", "nullable": true, "type": {"name": "utf8"},'
                ' "children": []}, {"name": "x", "nullable": true, "type":'
                ' {"name": "binary"}, "children": []}, {"name": "l",'
                ' "nullable": true, "type": {"name": "largeutf8"},'
                ' "children": []}'
            ),
            (
                '{"name": "s", "count": 3, "VALIDITY": [1, 0, 1], "OFFSET": [0,'
                ' 2, 2, 6], "DATA": ["ab", "", "cdé"]}, {"name": "x", "count":'
                ' 3, "VALIDITY": [1, 1, 1], "OFFSET": [0, 1, 1, 3], "DATA":'
                ' ["FF", "", "00a1"]}, {"name": "l", "count": 3, "VALIDITY":'
                ' [1, 1, 1], "OFFSET": ["0", "1", "2", "3"], "DATA": ["x", "y",'
                ' "z"]}'
            ),
        )
    )
    assert_true(b.columns[0].as_string() == array(["ab", None, "cdé"]))
    var x = b.columns[1].to_data()
    assert_equal(x.buffers[0].unsafe_get[DType.int32](3), 3)
    assert_equal(x.buffers[1].unsafe_get(0), 0xFF)
    assert_equal(x.buffers[1].unsafe_get(2), 0xA1)
    assert_true(b.columns[2].dtype() == large_string)
    assert_equal(String(b.columns[2]), "LargeStringArray([x, y, z])")


def test_integration_json_offsets_must_match_the_values() raises:
    _raises(
        _file(
            (
                '{"name": "s", "nullable": true, "type": {"name": "utf8"},'
                ' "children": []}'
            ),
            (
                '{"name": "s", "count": 1, "VALIDITY": [1], "OFFSET": [0, 3],'
                ' "DATA": ["ab"]}'
            ),
            count=1,
        ),
        "is 2 bytes, not 3",
    )


def test_integration_json_validity_length_must_match_the_count() raises:
    _raises(
        _file(
            (
                '{"name": "a", "nullable": true, "type": {"name": "int",'
                ' "isSigned": true, "bitWidth": 32}, "children": []}'
            ),
            '{"name": "a", "count": 2, "VALIDITY": [1], "DATA": [1, 2]}',
            count=2,
        ),
        "1 validity flags for 2 slots",
    )


def test_integration_json_fixed_size_binary() raises:
    var b = _batch(
        _file(
            (
                '{"name": "f", "nullable": true, "type": {"name":'
                ' "fixedsizebinary", "byteWidth": 2}, "children": []}'
            ),
            (
                '{"name": "f", "count": 2, "VALIDITY": [1, 1], "DATA": ["0102",'
                ' "FFFE"]}'
            ),
            count=2,
        )
    )
    assert_true(b.columns[0].dtype() == fixed_size_binary_(2))
    var data = b.columns[0].to_data()
    assert_equal(data.buffers[0].unsafe_get(1), 0x02)
    assert_equal(data.buffers[0].unsafe_get(2), 0xFF)


def test_integration_json_views_inline_and_out_of_line() raises:
    var b = _batch(
        _file(
            (
                '{"name": "v", "nullable": true, "type": {"name": "utf8view"},'
                ' "children": []}'
            ),
            (
                '{"name": "v", "count": 2, "VALIDITY": [1, 1], "VIEWS":'
                ' [{"SIZE": 3, "INLINED": "abc"}, {"SIZE": 14, "PREFIX_HEX":'
                ' "61626364", "BUFFER_INDEX": 0, "OFFSET": 0}],'
                ' "VARIADIC_DATA_BUFFERS": ["6162636465666768696A6B6C6D6E"]}'
            ),
            count=2,
        )
    )
    assert_true(b.columns[0].dtype() == string_view)
    assert_equal(String(b.columns[0]), "StringViewArray([abc, abcdefghijklmn])")


def test_integration_json_decimals_are_unscaled_strings() raises:
    var b = _batch(
        _file(
            (
                '{"name": "d", "nullable": true, "type": {"name": "decimal",'
                ' "precision": 10, "scale": 2, "bitWidth": 128}, "children":'
                ' []}, {"name": "w", "nullable": true, "type": {"name":'
                ' "decimal", "precision": 76, "scale": 0, "bitWidth": 256},'
                ' "children": []}'
            ),
            (
                '{"name": "d", "count": 2, "VALIDITY": [1, 1], "DATA":'
                ' ["-12345", "7"]}, {"name": "w", "count": 2, "VALIDITY": [1,'
                ' 1], "DATA": ["-1",'
                ' "1000000000000000000000000000000000000000"]}'
            ),
            count=2,
        )
    )
    assert_true(b.columns[0].dtype() == decimal128(10, 2))
    assert_equal(b.columns[0].as_decimal128()[0].value(), -12345)
    assert_equal(b.columns[0].as_decimal128()[1].value(), 7)
    assert_true(b.columns[1].dtype() == decimal256(76, 0))
    assert_equal(b.columns[1].as_decimal256()[0].value(), -1)
    assert_equal(
        b.columns[1].as_decimal256()[1].value(),
        Scalar[DType.int256](10) ** 39,
    )


def test_integration_json_intervals() raises:
    var b = _batch(
        _file(
            (
                '{"name": "ym", "nullable": true, "type": {"name": "interval",'
                ' "unit": "YEAR_MONTH"}, "children": []}, {"name": "dt",'
                ' "nullable": true, "type": {"name": "interval", "unit":'
                ' "DAY_TIME"}, "children": []}, {"name": "mdn", "nullable":'
                ' true, "type": {"name": "interval", "unit": "MONTH_DAY_NANO"},'
                ' "children": []}'
            ),
            (
                '{"name": "ym", "count": 2, "VALIDITY": [1, 0], "DATA": [-13,'
                ' 0]}, {"name": "dt", "count": 2, "VALIDITY": [1, 0], "DATA":'
                ' [{"days": -1, "milliseconds": 2}, {}]}, {"name": "mdn",'
                ' "count": 2, "VALIDITY": [1, 0], "DATA": [{"months": 1,'
                ' "days": -2, "nanoseconds": -9223372036854775808}, {}]}'
            ),
            count=2,
        )
    )
    assert_equal(b.columns[0].as_year_month_interval()[0].value(), -13)
    assert_false(b.columns[0].is_valid(1))
    # Little-endian Arrow layout: days in the low half, milliseconds above.
    assert_equal(
        b.columns[1].as_day_time_interval()[0].value(),
        (Int64(2) << 32) | Int64(0xFFFFFFFF),
    )
    assert_equal(
        b.columns[2].as_month_day_nano_interval()[0].value(),
        (Int64.MIN.cast[DType.int128]() << 64)
        | (Scalar[DType.int128](0xFFFFFFFE) << 32)
        | Scalar[DType.int128](1),
    )


def test_integration_json_nested_columns() raises:
    var b = _batch(
        _file(
            (
                '{"name": "l", "nullable": true, "type": {"name": "list"},'
                ' "children": [{"name": "element", "nullable": true, "type":'
                ' {"name": "int", "isSigned": true, "bitWidth": 32},'
                ' "children": []}]}, {"name": "s", "nullable": true, "type":'
                ' {"name": "struct"}, "children": [{"name": "x", "nullable":'
                ' true, "type": {"name": "int", "isSigned": true, "bitWidth":'
                ' 32}, "children": []}, {"name": "", "nullable": true, "type":'
                ' {"name": "utf8"}, "children": []}]}, {"name": "f",'
                ' "nullable": true, "type": {"name": "fixedsizelist",'
                ' "listSize": 2}, "children": [{"name": "item", "nullable":'
                ' true, "type": {"name": "int", "isSigned": true, "bitWidth":'
                ' 8}, "children": []}]}'
            ),
            (
                '{"name": "l", "count": 2, "VALIDITY": [1, 0], "OFFSET": [0, 2,'
                ' 2], "children": [{"name": "element", "count": 2, "VALIDITY":'
                ' [1, 1], "DATA": [1, 2]}]}, {"name": "s", "count": 2,'
                ' "VALIDITY": [1, 1], "children": [{"name": "x", "count": 2,'
                ' "VALIDITY": [1, 0], "DATA": [3, 0]}, {"name": "", "count": 2,'
                ' "VALIDITY": [1, 1], "OFFSET": [0, 1, 2], "DATA": ["p",'
                ' "q"]}]}, {"name": "f", "count": 2, "VALIDITY": [1, 1],'
                ' "children": [{"name": "item", "count": 4, "VALIDITY": [1, 1,'
                ' 1, 1], "DATA": [1, 2, 3, 4]}]}'
            ),
            count=2,
        )
    )
    assert_true(b.columns[0].dtype() == ListType(Field("element", int32)))
    assert_equal(
        String(b.columns[0]), "ListArray([PrimitiveArray[int32]([1, 2]), NULL])"
    )
    assert_equal(
        String(b.columns[1]),
        (
            "StructArray({'x': PrimitiveArray[int32]([3, NULL]), '':"
            " StringArray([p, q])})"
        ),
    )
    assert_true(
        b.columns[2].dtype() == FixedSizeListType(Field("item", int8), 2)
    )
    assert_equal(
        String(b.columns[2]),
        (
            "FixedSizeListArray([PrimitiveArray[int8]([1, 2]),"
            " PrimitiveArray[int8]([3, 4])])"
        ),
    )


def test_integration_json_large_list_offsets_are_strings() raises:
    var b = _batch(
        _file(
            (
                '{"name": "l", "nullable": true, "type": {"name": "largelist"},'
                ' "children": [{"name": "item", "nullable": true, "type":'
                ' {"name": "int", "isSigned": true, "bitWidth": 32},'
                ' "children": []}]}'
            ),
            (
                '{"name": "l", "count": 2, "VALIDITY": [1, 1], "OFFSET": ["0",'
                ' "1", "3"], "children": [{"name": "item", "count": 3,'
                ' "VALIDITY": [1, 1, 1], "DATA": [7, 8, 9]}]}'
            ),
            count=2,
        )
    )
    assert_true(b.columns[0].dtype() == LargeListType(Field("item", int32)))
    assert_equal(
        String(b.columns[0]),
        (
            "LargeListArray([PrimitiveArray[int32]([7]),"
            " PrimitiveArray[int32]([8, 9])])"
        ),
    )


def test_integration_json_nested_dictionaries_resolve_in_any_order() raises:
    # Dictionary 1 holds lists of codes into dictionary 0, and is listed first.
    var text = String(
        '{"schema": {"fields": [{"name": "d", "nullable": true, "type":'
        ' {"name": "list"}, "children": [{"name": "item", "nullable": true,'
        ' "type": {"name": "utf8"}, "children": [], "dictionary": {"id": 0,'
        ' "indexType": {"name": "int", "isSigned": true, "bitWidth": 8},'
        ' "isOrdered": false}}], "dictionary": {"id": 1, "indexType":'
        ' {"name": "int", "isSigned": true, "bitWidth": 16}, "isOrdered":'
        ' false}}]}, "dictionaries": [{"id": 1, "data": {"count": 2,'
        ' "columns": [{"name": "DICT1", "count": 2, "VALIDITY": [1, 1],'
        ' "OFFSET": [0, 2, 3], "children": [{"name": "item", "count": 3,'
        ' "VALIDITY": [1, 1, 1], "DATA": [1, 0, 1]}]}]}}, {"id": 0, "data":'
        ' {"count": 2, "columns": [{"name": "DICT0", "count": 2, "VALIDITY":'
        ' [1, 1], "OFFSET": [0, 1, 2], "DATA": ["a", "b"]}]}}], "batches":'
        ' [{"count": 3, "columns": [{"name": "d", "count": 3, "VALIDITY": [1,'
        ' 0, 1], "DATA": [1, 0, 0]}]}]}'
    )
    var parsed = IntegrationJson.parse(text)
    var inner = Field("item", dictionary(int8, string), nullable=True)
    var expected: DynType = dictionary(int16, ListType(inner^))
    assert_true(parsed.schema.fields[0].dtype == expected)
    ref column = parsed.batches[0].columns[0]
    assert_true(column.dtype() == expected)
    assert_equal(column.null_count(), 1)
    var values = column.as_dictionary().dictionary().to_data()
    assert_equal(values.buffers[0].unsafe_get[DType.int32](1), 2)
    ref codes = values.children[0]
    assert_equal(codes.buffers[0].unsafe_get[DType.int8](0), 1)
    assert_equal(codes.buffers[0].unsafe_get[DType.int8](2), 1)
    assert_equal(
        String(DynArray.from_data(codes.children[0])), "StringArray([a, b])"
    )


def test_integration_json_unknown_dictionary_id() raises:
    _raises(
        _file(
            (
                '{"name": "d", "nullable": true, "type": {"name": "utf8"},'
                ' "children": [], "dictionary": {"id": 3, "indexType": {"name":'
                ' "int", "isSigned": true, "bitWidth": 8}}}'
            ),
            '{"name": "d", "count": 1, "VALIDITY": [1], "DATA": [0]}',
            count=1,
        ),
        "no dictionary with id 3",
    )


def test_integration_json_null_column_has_no_buffers() raises:
    var b = _batch(
        _file(
            (
                '{"name": "n", "nullable": true, "type": {"name": "null"},'
                ' "children": []}'
            ),
            '{"name": "n", "count": 3}',
        )
    )
    assert_true(b.columns[0].dtype() == null)
    assert_equal(b.columns[0].null_count(), 3)


def test_integration_json_rejects_malformed_json() raises:
    _raises('{"schema": ', "InvalidError")
