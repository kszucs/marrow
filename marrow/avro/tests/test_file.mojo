# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Avro object container files: reading the arrow-testing corpus and an
Iceberg manifest list, and writing tables back under every codec.

The expected values for the arrow-testing files are the ones Arrow Rust's
`arrow-avro/src/reader/mod.rs` asserts."""

from std.os.path import join
from std.python import Python, PythonObject
from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DynArray
from ...avro import (
    AvroCodec,
    AvroFile,
    AvroSchema,
    AvroWriter,
    from_arrow,
    read_avro,
    write_avro,
)
from ...builders import array
from ...c_data import CArrowArrayStream
from ...dtypes import (
    FIELD_ID_KEY,
    binary,
    float64,
    int32,
    int64,
    microsecond,
    string,
    timestamp,
)
from ...io import BufferSource, MemorySink
from ...schema import Schema
from ...tabular import RecordBatch, Table
from ...utils.testing import ScratchDir

comptime DATA = "marrow/avro/tests/data/"


def _read(name: String) raises -> RecordBatch:
    return read_avro(DATA + name).combine_chunks()


def _to_marrow(py_tbl: PythonObject) raises -> Table:
    var caps = py_tbl.__arrow_c_stream__(Python.none())
    return CArrowArrayStream.from_pycapsule(caps).to_table()


def _to_pyarrow(var t: Table) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    var schema = t.schema.copy()
    var batches = t.to_batches()
    var caps = CArrowArrayStream.from_batches(schema^, batches^).to_pycapsule()
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _bytes_at(col: DynArray, i: Int) raises -> String:
    return String(col.as_binary().unsafe_get(UInt(i)))


# ---------------------------------------------------------------------------
# Reading the corpus
# ---------------------------------------------------------------------------


def _check_alltypes(b: RecordBatch) raises:
    assert_equal(b.num_rows(), 8)
    assert_true(
        b.column("id").as_int32() == array([4, 5, 6, 7, 2, 3, 0, 1], int32)
    )
    ref flags = b.column("bool_col").as_bool()
    for i in range(8):
        assert_equal(flags.values().test(i), i % 2 == 0)
    assert_true(
        b.column("int_col").as_int32() == array([0, 1, 0, 1, 0, 1, 0, 1], int32)
    )
    assert_true(
        b.column("bigint_col").as_int64()
        == array([0, 10, 0, 10, 0, 10, 0, 10], int64)
    )
    ref doubles = b.column("double_col").as_float64()
    for i in range(8):
        assert_equal(doubles.unsafe_get(i), Float64(i % 2) * 10.1)
    ref floats = b.column("float_col").as_float32()
    for i in range(8):
        assert_equal(floats.unsafe_get(i), Float32(i % 2) * 1.1)
    var dates: List[String] = [
        "03/01/09",
        "03/01/09",
        "04/01/09",
        "04/01/09",
        "02/01/09",
        "02/01/09",
        "01/01/09",
        "01/01/09",
    ]
    for i in range(8):
        assert_equal(_bytes_at(b.column("date_string_col"), i), dates[i])
        assert_equal(_bytes_at(b.column("string_col"), i), String(i % 2))
    assert_true(
        b.schema.field(name="timestamp_col").dtype
        == timestamp(microsecond, "UTC").to_dyn()
    )
    var stamps: List[Int64] = [
        1235865600000000,
        1235865660000000,
        1238544000000000,
        1238544060000000,
        1233446400000000,
        1233446460000000,
        1230768000000000,
        1230768060000000,
    ]
    ref ts = b.column("timestamp_col").as_timestamp()
    for i in range(8):
        assert_equal(ts.unsafe_get(i), stamps[i])


def test_avro_read_alltypes_under_each_codec() raises:
    for name in [
        "alltypes_plain.avro",
        "alltypes_plain.snappy.avro",
        "alltypes_plain.zstandard.avro",
    ]:
        _check_alltypes(_read(name))


def test_avro_read_unsupported_codecs() raises:
    for name in ["alltypes_plain.bzip2.avro", "alltypes_plain.xz.avro"]:
        with assert_raises(contains="NotImplementedError"):
            _ = _read(name)


def test_avro_read_projection() raises:
    var f = AvroFile(DATA + "alltypes_plain.avro")
    var t = f.read(columns=List[String](["bigint_col", "id"]))
    var b = t.combine_chunks()
    assert_equal(len(b.columns), 2)
    assert_equal(b.schema.fields[0].name, "bigint_col")
    assert_true(
        b.column("id").as_int32() == array([4, 5, 6, 7, 2, 3, 0, 1], int32)
    )
    with assert_raises(contains="no column named 'nope'"):
        _ = f.read(columns=List[String](["nope"]))


def test_avro_read_small_batches() raises:
    var f = AvroFile(DATA + "alltypes_plain.avro")
    var t = f.read(batch_size=3)
    assert_equal(t.num_rows(), 8)
    _check_alltypes(t.combine_chunks())


def test_avro_read_nulls() raises:
    var b = _read("alltypes_nulls_plain.avro")
    assert_equal(b.num_rows(), 1)
    for ref c in b.columns:
        assert_equal(c.null_count(), 1)
    var nan = _read("single_nan.avro")
    assert_equal(nan.num_rows(), 1)
    assert_true(nan.schema.fields[0].dtype == float64.to_dyn())
    assert_equal(nan.columns[0].null_count(), 1)


def test_avro_read_enum() raises:
    var b = _read("simple_enum.avro")
    ref f1 = b.column("f1").as_dictionary()
    assert_true(f1.indices().as_int32() == array([0, 1, 2, 3], int32))
    assert_true(f1.dictionary().as_string() == array(["a", "b", "c", "d"]))
    ref f3 = b.column("f3").as_dictionary()
    assert_true(f3.indices().as_int32() == array([1, 2, None, 0], int32))
    assert_true(b.schema.fields[2].nullable)


def test_avro_read_fixed() raises:
    var b = _read("simple_fixed.avro")
    ref f1 = b.column("f1").as_fixed_size_binary()
    assert_equal(f1.byte_width, 5)
    var first = f1.buffer.slice(f1.offset * 5, 5).as_span()
    assert_equal(String(StringSlice(unsafe_from_utf8=first)), "abcde")
    assert_equal(b.column("f3").null_count(), 1)


def test_avro_read_decimals() raises:
    # Each holds 1..24, scaled.
    for name in [
        "fixed_length_decimal.avro",
        "int32_decimal.avro",
        "int128_decimal.avro",
    ]:
        var b = _read(name)
        ref dt = b.schema.fields[0].dtype
        assert_true(dt.is_decimal128())
        var pow10 = 1
        for _ in range(dt.as_decimal128().scale()):
            pow10 *= 10
        ref col = b.columns[0].as_decimal128()
        assert_equal(len(col), 24)
        for i in range(24):
            assert_equal(col.unsafe_get(i), Int128((i + 1) * pow10))
    var big = _read("fixed256_decimal.avro")
    ref dt = big.schema.fields[0].dtype
    assert_true(dt.is_decimal256())
    assert_equal(dt.as_decimal256().precision(), 76)
    var scale = Int256(1)
    for _ in range(dt.as_decimal256().scale()):
        scale *= 10
    ref col = big.columns[0].as_decimal256()
    for i in range(24):
        assert_equal(col.unsafe_get(i), Int256(i + 1) * scale)


def test_avro_read_duration_and_uuid() raises:
    var b = _read("duration_uuid.avro")
    ref d = b.column("duration_field").as_month_day_nano_interval()
    # (months, days, nanoseconds), packed little-endian into 128 bits.
    var expected: List[Tuple[Int, Int, Int]] = [
        (1, 15, 500_000_000),
        (0, 5, 2_500_000_000),
        (2, 0, 0),
        (12, 31, 999_000_000),
    ]
    for i in range(4):
        var bits = d.unsafe_get(i).cast[DType.uint128]()
        assert_equal(Int(bits & 0xFFFFFFFF), expected[i][0])
        assert_equal(Int((bits >> 32) & 0xFFFFFFFF), expected[i][1])
        assert_equal(Int(bits >> 64), expected[i][2])
    ref uuid = b.schema.field(name="uuid_field")
    assert_equal(uuid.metadata["ARROW:extension:name"], "arrow.uuid")
    ref u = b.column("uuid_field").as_fixed_size_binary()
    var head = u.buffer.slice(u.offset * 16, 16).as_span()
    var want: List[UInt8] = [
        0xFE,
        0x7B,
        0xC3,
        0x0B,
        0x4C,
        0xE8,
        0x4C,
        0x5E,
        0xB6,
        0x7C,
        0x22,
        0x34,
        0xA2,
        0xD3,
        0x8E,
        0x66,
    ]
    for i in range(16):
        assert_equal(head[i], want[i])
    # Written back under the same schema, the text survives byte for byte.
    var f = AvroFile(DATA + "duration_uuid.avro")
    var w = AvroWriter(MemorySink(), f.avro_schema(), AvroCodec.NULL)
    w.write(f.read())
    w.close()
    var written = List[UInt8](w.out.sink().bytes())
    var again = AvroFile(BufferSource(Span(written)))
    assert_true(again.read() == f.read())


def test_avro_read_zero_byte_values() raises:
    var b = _read("zero_byte.avro")
    assert_equal(b.num_rows(), 3)
    assert_true(b.schema.fields[0].dtype == binary.to_dyn())


def test_avro_read_nested() raises:
    var lists = _read("nested_lists.snappy.avro")
    assert_true(lists.num_rows() > 0)
    var records = _read("nested_records.avro")
    assert_equal(records.num_rows(), 2)
    var impala = _read("nullable.impala.avro")
    assert_equal(impala.num_rows(), 7)
    # Every nested shape these files hold survives a write and a re-read.
    for name in [
        "nested_lists.snappy.avro",
        "nested_records.avro",
        "nullable.impala.avro",
    ]:
        var original = read_avro(DATA + name)
        with ScratchDir() as tmp:
            var path = join(tmp, "out.avro")
            write_avro(original, path)
            assert_true(read_avro(path) == original)


# ---------------------------------------------------------------------------
# Iceberg
# ---------------------------------------------------------------------------


def test_avro_iceberg_manifest_list() raises:
    var f = AvroFile(DATA + "manifest-list-v2-1.avro")
    var schema = f.schema()
    assert_equal(schema.metadata["format-version"], "2")
    assert_true("snapshot-id" in schema.metadata)
    ref path = schema.field(name="manifest_path")
    assert_equal(path.metadata[FIELD_ID_KEY], "500")
    assert_true(not path.nullable)
    ref partitions = schema.field(name="partitions")
    assert_equal(partitions.metadata[FIELD_ID_KEY], "507")
    assert_equal(
        partitions.dtype.as_list().value_field().metadata[FIELD_ID_KEY], "508"
    )
    var table = f.read()
    assert_true(table.num_rows() > 0)
    # Written back with its header metadata, it reads back the same.
    with ScratchDir() as tmp:
        var out = join(tmp, "manifest-list.avro")
        var w = AvroWriter(out, table.schema)
        w.write(table)
        w.close()
        var again = AvroFile(out)
        assert_true(again.read() == table)
        assert_equal(again.schema().metadata["format-version"], "2")


def test_avro_writer_keeps_explicit_schema() raises:
    """A writer given an Avro schema writes it verbatim -- names, docs and
    ids -- which is what an Iceberg writer needs."""
    var f = AvroFile(DATA + "manifest-list-v2-1.avro")
    var table = f.read()
    var meta = Dict[String, String]()
    meta["format-version"] = "2"
    var w = AvroWriter(MemorySink(), f.avro_schema(), AvroCodec.SNAPPY, meta)
    w.write(table)
    w.close()
    var written = List[UInt8](w.out.sink().bytes())
    var again = AvroFile(BufferSource(Span(written)))
    assert_equal(again.avro_schema().to_json(), f.avro_schema().to_json())
    assert_true(again.read() == table)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


comptime ROUNDTRIP_TABLE = """
import datetime, decimal, pyarrow as pa

table = pa.table({
    "i": pa.array([1, None, -3, 2**31 - 1], pa.int32()),
    "l": pa.array([2**62, -1, None, 0], pa.int64()),
    "f": pa.array([1.5, None, float("inf"), -0.0], pa.float32()),
    "d": pa.array([0.1, 2.5, None, -1e300], pa.float64()),
    "b": pa.array([True, False, None, True]),
    "s": pa.array(["a", "héllo", None, ""]),
    "y": pa.array([b"\\x00\\x01", b"", None, b"z"], pa.binary()),
    "fx": pa.array([b"abcd", b"efgh", None, b"ijkl"], pa.binary(4)),
    "dt": pa.array([0, 19000, None, -1], pa.date32()),
    "tm": pa.array([0, 1000, None, 86399999], pa.time32("ms")),
    "tu": pa.array([0, 1, None, 86399999999], pa.time64("us")),
    "ts": pa.array([0, 1, None, -1], pa.timestamp("us", tz="UTC")),
    "tn": pa.array([0, 1, None, -1], pa.timestamp("ns")),
    "dec": pa.array([decimal.Decimal("1.23"), decimal.Decimal("-99999.99"),
                     None, decimal.Decimal("0.00")], pa.decimal128(7, 2)),
    "big": pa.array([decimal.Decimal("1" * 50), None, decimal.Decimal("-1"),
                     decimal.Decimal("0")], pa.decimal256(50, 0)),
    "iv": pa.array([pa.MonthDayNano([1, 2, 3_000_000]), None,
                    pa.MonthDayNano([0, 0, 0]),
                    pa.MonthDayNano([12, 31, 999_000_000])],
                   pa.month_day_nano_interval()),
    "li": pa.array([[1, None], [], None, [4]], pa.list_(pa.int64())),
    "ls": pa.array([["x"], None, [], ["y", "z"]], pa.list_(pa.string())),
    "m": pa.array([[("k", 1)], [], None, [("a", None), ("b", 2)]],
                  pa.map_(pa.string(), pa.int64())),
    "mi": pa.array([[(1, "one")], None, [], [(2, None)]],
                   pa.map_(pa.int32(), pa.string())),
    "st": pa.array([{"x": 1, "y": "a"}, None, {"x": None, "y": None},
                    {"x": 3, "y": "c"}],
                   pa.struct([("x", pa.int32()), ("y", pa.string())])),
    "nested": pa.array([[{"p": [1, 2]}], None, [{"p": None}], []],
                       pa.list_(pa.struct([("p", pa.list_(pa.int32()))]))),
})
"""


def _roundtrip_table() raises -> PythonObject:
    """The pyarrow table `ROUNDTRIP_TABLE` builds."""
    var scope = Python.dict()
    _ = Python.import_module("builtins").exec(ROUNDTRIP_TABLE, scope)
    return scope["table"]


def test_avro_write_roundtrip_under_each_codec() raises:
    var py = _roundtrip_table()
    for codec in [
        AvroCodec.NULL,
        AvroCodec.DEFLATE,
        AvroCodec.SNAPPY,
        AvroCodec.ZSTANDARD,
    ]:
        with ScratchDir() as tmp:
            var path = join(tmp, "t.avro")
            write_avro(_to_marrow(py), path, codec)
            var back = _to_pyarrow(read_avro(path))
            assert_true(
                Bool(back.equals(py)),
                String(codec, ": ", back.to_string(), " != ", py.to_string()),
            )


def test_avro_write_converts_dictionary_and_decimal_width() raises:
    """A dictionary column is written as its strings, and a decimal256 whose
    precision fits 128 bits reads back as the decimal128 Avro's precision
    implies."""
    var scope = Python.dict()
    _ = Python.import_module("builtins").exec(
        """
import decimal, pyarrow as pa
strings = pa.array(["a", "b", None, "a"])
decimals = [decimal.Decimal("1.25"), None, decimal.Decimal("-3.50"),
            decimal.Decimal("0.00")]
source = pa.table({
    "d": strings.dictionary_encode().cast(pa.dictionary(pa.int8(), pa.string())),
    "x": pa.array(decimals, pa.decimal256(10, 2)),
})
expected = pa.table({
    "d": strings,
    "x": pa.array(decimals, pa.decimal128(10, 2)),
})
""",
        scope,
    )
    with ScratchDir() as tmp:
        var path = join(tmp, "t.avro")
        write_avro(_to_marrow(scope["source"]), path)
        var back = _to_pyarrow(read_avro(path))
        assert_true(Bool(back.equals(scope["expected"])), String(back))


def test_avro_write_many_blocks() raises:
    """A sync interval smaller than a row puts every row in its own block."""
    var py = _roundtrip_table()
    var t = _to_marrow(py)
    var w = AvroWriter(
        MemorySink(),
        _avro_of(t.schema),
        AvroCodec.DEFLATE,
        sync_interval=1,
    )
    w.write(t)
    w.close()
    var written = List[UInt8](w.out.sink().bytes())
    var back = AvroFile(BufferSource(Span(written))).read(batch_size=2)
    assert_true(Bool(_to_pyarrow(back^).equals(py)))


def _avro_of(schema: Schema) raises -> AvroSchema:
    return from_arrow(schema)


def test_avro_corrupt_sync_marker() raises:
    var t = _to_marrow(_roundtrip_table())
    var w = AvroWriter(MemorySink(), _avro_of(t.schema), AvroCodec.NULL)
    w.write(t)
    w.close()
    var written = List[UInt8](w.out.sink().bytes())
    # The file ends with the sync marker after the only block.
    written[len(written) - 1] ^= 0xFF
    var f = AvroFile(BufferSource(Span(written)))
    with assert_raises(contains="sync marker mismatch"):
        _ = f.read()


def test_avro_write_refusals() raises:
    var t = _to_marrow(_roundtrip_table())
    with assert_raises(contains="NotImplementedError"):
        _ = AvroWriter(MemorySink(), _avro_of(t.schema), AvroCodec.BZIP2)
    with assert_raises(contains="is reserved"):
        var meta = Dict[String, String]()
        meta["avro.codec"] = "null"
        _ = AvroWriter(MemorySink(), _avro_of(t.schema), AvroCodec.NULL, meta)
    # A null in a column the schema says is not nullable.
    var nonnull = Schema(fields=[t.schema.fields[0].copy()])
    nonnull.fields[0].nullable = False
    var w = AvroWriter(MemorySink(), _avro_of(nonnull), AvroCodec.NULL)
    var one = RecordBatch(
        t.schema.copy(), [t.combine_chunks().columns[0].copy()]
    )
    with assert_raises(contains="non-nullable"):
        w.write(one)
