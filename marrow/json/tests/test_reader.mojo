# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The NDJSON reader against Arrow C++'s behaviour.

Every expectation here was checked against `pyarrow.json.read_json` 23.0 —
types, values and error wording — and the cases follow the ones Arrow C++'s
`reader_test.cc` and pyarrow's `test_json.py` pin: inference and promotion,
nulls and absent keys, nesting, explicit schemas under each
`unexpected_field_behavior`, blocks, and the malformed inputs. Value-by-value
agreement on generated data is `python/marrow/tests/test_json.py`'s job.
"""

from std.memory import ArcPointer
from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import (
    ListType,
    bool_,
    field,
    float64,
    int32,
    int64,
    list_,
    millisecond,
    null,
    second,
    string,
    struct_,
    timestamp,
)
from ...execution import ExecContext
from ...io import BufferSource, ByteSource, Fetched
from ...schema import Schema
from ...tabular import RecordBatch, Table

from ..reader import (
    JsonReader,
    ParseOptions,
    ReadOptions,
    UnexpectedFieldBehavior,
    read_json,
)


def _read(
    text: String,
    read_options: ReadOptions = ReadOptions(),
    parse_options: ParseOptions = ParseOptions(),
) raises -> Table:
    return read_json(BufferSource(text.as_bytes()), read_options, parse_options)


def _strings(var values: List[Optional[String]]) raises -> DynArray:
    return array(values^)


def _bools(var values: List[Optional[Bool]]) raises -> DynArray:
    return array(values^)


def _batch(text: String) raises -> RecordBatch:
    return _read(text).combine_chunks()


def _raises(text: String, contains: String) raises:
    with assert_raises(contains=contains):
        _ = _read(text)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_json_scalars_infer_in_first_seen_order() raises:
    var b = _batch(
        '{"i": 1, "f": 1.5, "s": "x", "b": true}\n'
        + '{"i": 2, "f": -0.25, "s": "y", "b": false}\n'
    )
    assert_equal(
        String(b.schema),
        String(
            Schema(
                fields=[
                    field("i", int64),
                    field("f", float64),
                    field("s", string),
                    field("b", bool_),
                ]
            )
        ),
    )
    assert_true(b.column("i") == array([1, 2], int64).to_dyn())
    assert_true(b.column("f") == array([1.5, -0.25], float64).to_dyn())
    assert_true(b.column("s") == _strings([String("x"), String("y")]))
    assert_true(b.column("b") == _bools([True, False]))


def test_json_int_widens_to_double() raises:
    var b = _batch('{"a": 1}\n{"a": 2.5}\n')
    assert_true(b.column("a").dtype() == float64)
    assert_true(b.column("a") == array([1.0, 2.5], float64).to_dyn())


def test_json_int64_overflow_widens_to_double() raises:
    var b = _batch('{"a": 9223372036854775808}\n')
    assert_true(b.column("a").dtype() == float64)


def test_json_exponent_is_a_double() raises:
    var b = _batch('{"a": 1e2}\n')
    assert_true(b.column("a") == array([100.0], float64).to_dyn())


def test_json_strings_infer_timestamp_seconds() raises:
    var b = _batch('{"t": "2020-01-01"}\n{"t": "2020-01-01 10:00:00"}\n')
    assert_true(b.column("t").dtype() == timestamp(second))
    ref t = b.column("t").as_timestamp()
    assert_equal(Int(t[0].value()), 1577836800)
    assert_equal(Int(t[1].value()), 1577872800)


def test_json_timestamp_widens_to_string() raises:
    # A fraction does not fit timestamp[s], and neither does free text.
    assert_true(
        _batch('{"t": "2020-01-01 10:00:00.5"}\n').column("t").dtype() == string
    )
    var b = _batch('{"t": "2020-01-01"}\n{"t": "x"}\n')
    assert_true(b.column("t") == _strings([String("2020-01-01"), String("x")]))


def test_json_all_null_column_is_null_typed() raises:
    var b = _batch('{"a": null}\n{"a": null}\n')
    assert_true(b.column("a").dtype() == null)
    assert_equal(len(b.column("a")), 2)


def test_json_absent_keys_are_null() raises:
    var b = _batch('{"a": 1}\n{"b": 2}\n')
    assert_true(b.column("a") == array([1, None], int64).to_dyn())
    assert_true(b.column("b") == array([None, 2], int64).to_dyn())


def test_json_empty_objects_have_no_columns() raises:
    # pyarrow answers 2 rows and 0 columns. marrow's `Table` counts rows off
    # its columns, so without one it cannot hold a row count: 0 rows here.
    var t = _read("{}\n{}\n")
    assert_equal(t.num_columns(), 0)
    assert_equal(t.num_rows(), 0)


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_json_struct_fields_union() raises:
    var b = _batch('{"a": {"x": 1}}\n{"a": {"y": "s"}}\n{"a": null}\n')
    assert_true(
        b.column("a").dtype()
        == struct_([field("x", int64), field("y", string)])
    )
    ref a = b.column("a").as_struct()
    assert_equal(a.null_count(), 1)
    assert_true(a.children[0] == array([1, None, None], int64).to_dyn())
    assert_true(a.children[1] == _strings([None, String("s"), None]))


def test_json_list_of_struct() raises:
    var b = _batch('{"a": [{"x": 1}, {"y": 2.5}]}\n')
    assert_true(
        b.column("a").dtype()
        == list_(struct_([field("x", int64), field("y", float64)]))
    )
    assert_equal(len(b.column("a").as_list().values()), 2)


def test_json_empty_and_null_lists_are_list_of_null() raises:
    assert_true(
        _batch('{"a": []}\n{"a": []}\n').column("a").dtype() == list_(null)
    )
    assert_true(_batch('{"a": [null]}\n').column("a").dtype() == list_(null))


# ---------------------------------------------------------------------------
# Input shape
# ---------------------------------------------------------------------------


def test_json_objects_are_a_whitespace_separated_stream() raises:
    # Several per line, one across lines, blank lines, CRLF: all rows.
    var b = _batch('{"a": 1} {"a": 2}\n\n{"a":\n3}\r\n  {"a" : 4 }  ')
    assert_true(b.column("a") == array([1, 2, 3, 4], int64).to_dyn())


def test_json_only_newlines_is_an_empty_table() raises:
    var t = _read("\n\n\n")
    assert_equal(t.num_rows(), 0)
    assert_equal(t.num_columns(), 0)


def test_json_one_chunk_per_block() raises:
    var text = String()
    for _ in range(5):
        text += '{"a": 1}\n'
    text += '{"a": 2.5}\n'
    var t = _read(text, ReadOptions(block_size=16))
    assert_equal(t.num_rows(), 6)
    assert_true(t.schema.fields[0].dtype == float64)
    assert_true(len(t.column(0).chunks) > 1)


def test_json_row_longer_than_a_block_grows_the_read() raises:
    var t = _read(
        '{"a": "' + "x" * 100 + '"}\n{"a": "y"}\n', ReadOptions(block_size=8)
    )
    assert_equal(t.num_rows(), 2)


def test_json_unicode_escapes_decode() raises:
    var b = _batch('{"a": "\\u00e9\\ud83d\\ude00"}\n')
    assert_true(b.column("a") == _strings([String("é😀")]))


def test_json_escaped_keys_match_their_columns() raises:
    # `a` is `a`: an escaped key is decoded before it is matched, so
    # both rows land in one column, and a quote inside a value survives.
    var b = _batch('{"a": "x\\"y"}\n{"\\u0061": "z"}\n')
    assert_equal(b.num_columns(), 1)
    assert_true(b.column("a") == _strings([String('x"y'), String("z")]))


def test_json_escaped_timestamp_still_infers() raises:
    # Arrow decodes before it tries a timestamp; `-` is `-`.
    var b = _batch('{"t": "2024\\u002d01-02"}\n{"t": "2024-01-03"}\n')
    assert_true(b.schema.fields[0].dtype == timestamp(second))


def test_json_keys_in_another_order_per_row() raises:
    # The expected-key hint must not mismatch when rows reorder their keys.
    var b = _batch('{"a": 1, "b": "x"}\n{"b": "y", "a": 2}\n{"b": "z"}\n')
    assert_true(b.column("a") == array([1, 2, None], int64).to_dyn())
    assert_true(
        b.column("b") == _strings([String("x"), String("y"), String("z")])
    )


# ---------------------------------------------------------------------------
# Explicit schema
# ---------------------------------------------------------------------------


def _explicit(behavior: UnexpectedFieldBehavior) -> ParseOptions:
    return ParseOptions(
        Schema(fields=[field("a", int32), field("b", string)]), behavior
    )


def test_json_explicit_list_keeps_its_item_field() raises:
    # The item's name and nullability come from the explicit type, under
    # every behaviour: a null item in a non-nullable list is rejected.
    var item = field("element", int64, nullable=False)
    var schema = Schema(fields=[field("a", ListType(item.copy()))])
    for behavior in [
        UnexpectedFieldBehavior.INFER,
        UnexpectedFieldBehavior.ERROR,
    ]:
        var options = ParseOptions(schema, behavior)
        var t = _read('{"a": [1, 2]}\n', parse_options=options)
        assert_true(t.schema.fields[0].dtype == ListType(item.copy()))
        with assert_raises(contains="a required field was null"):
            _ = _read('{"a": [1, null]}\n', parse_options=options)


def test_json_block_size_must_be_positive() raises:
    with assert_raises(contains="block_size must be positive"):
        _ = _read('{"a": 1}\n', ReadOptions(block_size=0))


def test_json_explicit_schema_ignore() raises:
    var t = _read(
        '{"a": 1, "c": true}\n{"b": "x"}\n',
        parse_options=_explicit(UnexpectedFieldBehavior.IGNORE),
    )
    var b = t.combine_chunks()
    assert_equal(b.num_columns(), 2)
    assert_true(b.column("a") == array([1, None], int32).to_dyn())


def test_json_explicit_schema_error() raises:
    with assert_raises(contains="unexpected field"):
        _ = _read(
            '{"a": 1, "c": true}\n',
            parse_options=_explicit(UnexpectedFieldBehavior.ERROR),
        )


def test_json_explicit_schema_infer_appends_after() raises:
    var b = _read(
        '{"a": 1, "c": true}\n{"b": "x"}\n',
        parse_options=_explicit(UnexpectedFieldBehavior.INFER),
    ).combine_chunks()
    assert_equal(b.num_columns(), 3)
    assert_equal(b.schema.fields[2].name, "c")
    assert_true(b.schema.fields[0].dtype == int32)
    assert_true(b.column("c") == _bools([True, None]))


def test_json_explicit_timestamp_unit_parses_fractions() raises:
    var b = _read(
        '{"t": "2020-01-01 10:00:00.5"}\n',
        parse_options=ParseOptions(
            Schema(fields=[field("t", timestamp(millisecond))])
        ),
    ).combine_chunks()
    assert_equal(Int(b.column("t").as_timestamp()[0].value()), 1577872800500)


def test_json_explicit_float_accepts_integers() raises:
    var b = _read(
        '{"f": 1}\n',
        parse_options=ParseOptions(Schema(fields=[field("f", float64)])),
    ).combine_chunks()
    assert_true(b.column("f") == array([1.0], float64).to_dyn())


def test_json_explicit_int_rejects_what_it_cannot_hold() raises:
    var options = ParseOptions(Schema(fields=[field("a", int32)]))
    with assert_raises(contains="Failed to convert JSON to int32"):
        _ = _read('{"a": 3000000000}\n', parse_options=options)
    with assert_raises(contains="couldn't parse:1.5"):
        _ = _read('{"a": 1.5}\n', parse_options=options)


def test_json_explicit_non_nullable_absent() raises:
    var options = ParseOptions(
        Schema(fields=[field("a", int64, nullable=False)])
    )
    with assert_raises(contains="a required field was absent"):
        _ = _read('{"b": "x"}\n', parse_options=options)


# ---------------------------------------------------------------------------
# Errors, worded as Arrow's
# ---------------------------------------------------------------------------


def test_json_empty_file() raises:
    _raises("", "Empty JSON file")


def test_json_kind_change_is_an_error() raises:
    _raises(
        '{"a": 1}\n{"a": "x"}\n',
        "Column(/a) changed from number to string in row 1",
    )
    _raises(
        '{"a": true}\n{"a": 1}\n',
        "Column(/a) changed from boolean to number in row 1",
    )


def test_json_row_that_is_not_an_object() raises:
    _raises("1\n", "Column() changed from object to number in row 0")
    _raises("[1]\n", "Column() changed from object to array in row 0")


def test_json_duplicate_key() raises:
    _raises('{"a": 1, "a": 2}\n', "Column(/a) was specified twice in row 0")


def test_json_trailing_garbage() raises:
    _raises('{"a": 1} x\n', "JSON parse error")


def test_json_invalid_utf8() raises:
    var bytes = List[Byte]()
    for b in '{"a": "'.as_bytes():
        bytes.append(b)
    bytes.append(0xFF)
    for b in '"}\n'.as_bytes():
        bytes.append(b)
    with assert_raises(contains="invalid UTF-8"):
        _ = read_json(BufferSource(Span(bytes)))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_json_reader_infers_from_the_first_block_then_is_strict() raises:
    var text = String('{"a": 1}\n{"a": 2}\n{"a": 3, "b": 4}\n')
    var reader = JsonReader[BufferSource](
        BufferSource(text.as_bytes()), ReadOptions(block_size=18)
    )
    assert_equal(len(reader.schema.fields), 1)
    var first = reader.read_next_batch()
    assert_true(Bool(first))
    assert_equal(first.value().num_rows(), 2)
    with assert_raises(contains="unexpected field"):
        _ = reader.read_next_batch()


def test_json_nan_literal_is_rejected() raises:
    # Arrow accepts `NaN`, which is not JSON; EmberJson, rightly, does not.
    _raises('{"a": NaN}\n', "JSON parse error")


def test_json_list_items_widen_together() raises:
    var b = _batch('{"a": [1, 2.5]}\n{"a": [3]}\n')
    assert_true(b.column("a").dtype() == list_(float64))
    assert_true(
        b.column("a").as_list().values()
        == array([1.0, 2.5, 3.0], float64).to_dyn()
    )


def test_json_explicit_struct_ignores_unknown_inner_keys() raises:
    var b = _read(
        '{"s": {"x": 1, "y": "skip"}}\n{"s": {"x": 2}}\n',
        parse_options=ParseOptions(
            Schema(fields=[field("s", struct_([field("x", int64)]))]),
            UnexpectedFieldBehavior.IGNORE,
        ),
    ).combine_chunks()
    ref s = b.column("s").as_struct()
    assert_equal(len(s.children), 1)
    assert_true(s.children[0] == array([1, 2], int64).to_dyn())


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


struct _Recorder(ByteSource):
    """A `BufferSource` that records each fetch and counts `read_at` calls:
    on an object store `read_at` keeps every fetch alive with the source."""

    var _inner: BufferSource
    var _fetches: ArcPointer[List[Tuple[Int, Int]]]
    var _read_ats: ArcPointer[Int]

    def __init__(
        out self,
        var inner: BufferSource,
        fetches: ArcPointer[List[Tuple[Int, Int]]],
        read_ats: ArcPointer[Int],
    ):
        self._inner = inner^
        self._fetches = fetches
        self._read_ats = read_ats

    def size(self) -> Int:
        return self._inner.size()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        self._read_ats[] += 1
        return rebind[Span[UInt8, origin_of(self)]](
            self._inner.read_at(offset, length)
        )

    def read_ranges(
        ref self, ranges: List[Tuple[Int, Int]], ctx: ExecContext
    ) raises -> Fetched:
        for ref r in ranges:
            self._fetches[].append((r[0], r[1]))
        return self._inner.read_ranges(ranges, ctx)


def test_json_fetches_each_block_once_per_pass() raises:
    # Pass 1 and pass 2 each walk the blocks once, every block one fetch the
    # reader owns -- never `read_at`, which an object store would retain.
    var text = String()
    for i in range(40):
        text += '{"a": ' + String(i) + ', "b": "' + "x" * (i % 7) + '"}\n'
    var fetches = ArcPointer(List[Tuple[Int, Int]]())
    var read_ats = ArcPointer(0)
    var source = _Recorder(BufferSource(text.as_bytes()), fetches, read_ats)
    var t = read_json(source^, ReadOptions(block_size=64))
    assert_equal(t.num_rows(), 40)
    assert_equal(read_ats[], 0)
    var rewinds = 0
    for i in range(1, len(fetches[])):
        if fetches[][i][0] <= fetches[][i - 1][0]:
            rewinds += 1
    # Starts strictly increase within a pass; the one step back is pass 2.
    assert_equal(rewinds, 1)
