# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Iceberg partition transforms, against the spec's Appendix B test vectors."""

from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DynArray, Int32Array
from ...builders import (
    BinaryBuilder,
    FixedSizeBinaryBuilder,
    PrimitiveBuilder,
    array,
)
from ...dtypes import (
    Date32Type,
    Decimal128Type,
    Time64Type,
    TimestampType,
    binary,
    bool_,
    date32,
    decimal128,
    float64,
    int32,
    int64,
    list_,
    microsecond,
    nanosecond,
    string,
    time64,
    timestamp,
)
from ..transforms import Transform, murmur3_x86_32


# --- helpers ---------------------------------------------------------------


def _buckets(hashes: List[Optional[Int]], n: Int) raises -> Int32Array:
    """What `bucket[n]` gives for values of the given 32-bit hashes."""
    var out = List[Optional[Int]]()
    for h in hashes:
        if h:
            out.append((h.value() & 0x7FFFFFFF) % n)
        else:
            out.append(None)
    return array(out, int32)


def _dates(values: List[Optional[Int]]) raises -> DynArray:
    var b = PrimitiveBuilder[Date32Type](date32(), len(values))
    for v in values:
        if v:
            b.append(Int32(v.value()))
        else:
            b.append_null()
    return b.finish()


def _times(values: List[Optional[Int]]) raises -> DynArray:
    var b = PrimitiveBuilder[Time64Type](time64(microsecond), len(values))
    for v in values:
        if v:
            b.append(Int64(v.value()))
        else:
            b.append_null()
    return b.finish()


def _timestamps(
    values: List[Optional[Int]], dtype: TimestampType
) raises -> DynArray:
    var b = PrimitiveBuilder[TimestampType](dtype, len(values))
    for v in values:
        if v:
            b.append(Int64(v.value()))
        else:
            b.append_null()
    return b.finish()


def _decimals(
    values: List[Optional[Int]], precision: Int, scale: Int
) raises -> DynArray:
    var b = PrimitiveBuilder[Decimal128Type](
        decimal128(precision, scale), len(values)
    )
    for v in values:
        if v:
            b.append(Int128(v.value()))
        else:
            b.append_null()
    return b.finish()


def _binaries(values: List[List[UInt8]]) raises -> DynArray:
    """A binary array; an empty list stands for null."""
    var b = BinaryBuilder()
    for v in values:
        if len(v) == 0:
            b.append_null()
        else:
            b.append(StringSlice(unsafe_from_utf8=Span(v)))
    return b.finish()


def _apply(text: String, values: DynArray) raises -> DynArray:
    return Transform.parse(text).apply(values)


def _uuid() -> List[UInt8]:
    """f79c3e09-677c-4bbd-a479-3f349cb785e7, big-endian."""
    return [
        0xF7,
        0x9C,
        0x3E,
        0x09,
        0x67,
        0x7C,
        0x4B,
        0xBD,
        0xA4,
        0x79,
        0x3F,
        0x34,
        0x9C,
        0xB7,
        0x85,
        0xE7,
    ]


comptime _TS = 1_510_871_468_000_000
"""2017-11-16T22:31:08 in microseconds since the epoch."""


# --- parsing ---------------------------------------------------------------


def test_iceberg_transform_parse_round_trip() raises:
    for text in [
        "identity",
        "bucket[16]",
        "truncate[10]",
        "year",
        "month",
        "day",
        "hour",
        "void",
    ]:
        assert_equal(String(Transform.parse(text)), text)
    assert_true(
        Transform.parse("bucket[16]") == Transform(Transform.BUCKET, 16)
    )
    assert_true(
        Transform.parse(" truncate[ 4 ] ") == Transform(Transform.TRUNCATE, 4)
    )


def test_iceberg_transform_parse_errors() raises:
    for text in [
        "bucket[0]",
        "bucket[]",
        "bucket[x]",
        "bucket",
        "truncate[-1]",
        "truncate",
        "months",
        "",
    ]:
        with assert_raises(contains="InvalidError"):
            _ = Transform.parse(text)


# --- murmur3 and bucket: the spec's Appendix B vectors ----------------------


def test_iceberg_murmur3_vectors() raises:
    var iceberg = String("iceberg")
    assert_equal(murmur3_x86_32(iceberg.as_bytes()), 1210000089)
    var four: List[UInt8] = [0, 1, 2, 3]
    assert_equal(murmur3_x86_32(Span(four)), -188683207)
    var uuid = _uuid()
    assert_equal(murmur3_x86_32(Span(uuid)), 1488055340)
    var decimal: List[UInt8] = [0x05, 0x8C]
    assert_equal(murmur3_x86_32(Span(decimal)), -500754589)
    var empty = List[UInt8]()
    assert_equal(murmur3_x86_32(Span(empty)), 0)


def test_iceberg_bucket_int_and_long() raises:
    var ints: DynArray = array([34, None], int32)
    var longs: DynArray = array([34, None], int64)
    for n in [16, 1000, 2147483647]:
        var expected = _buckets([2017239379, None], n)
        var bucket = String(t"bucket[{n}]")
        assert_true(_apply(bucket, ints).as_int32() == expected)
        assert_true(_apply(bucket, longs).as_int32() == expected)


def test_iceberg_bucket_decimal() raises:
    # 14.20 in decimal(9, 2).
    var r = _apply("bucket[100]", _decimals([1420, None], 9, 2))
    assert_true(r.as_int32() == _buckets([-500754589, None], 100))


def test_iceberg_bucket_date_and_time() raises:
    var dates = _apply("bucket[100]", _dates([17486, None]))  # 2017-11-16
    assert_true(dates.as_int32() == _buckets([-653330422, None], 100))
    var times = _apply("bucket[100]", _times([81_068_000_000, None]))
    assert_true(times.as_int32() == _buckets([-662762989, None], 100))


def test_iceberg_bucket_timestamps() raises:
    var expected = _buckets([-2047944441, -1207196810, None], 100)
    var us: List[Optional[Int]] = [_TS, _TS + 1, None]
    var ns: List[Optional[Int]] = [_TS * 1000, _TS * 1000 + 1001, None]
    for dtype in [timestamp(microsecond), timestamp(microsecond, "UTC")]:
        var r = _apply("bucket[100]", _timestamps(us, dtype))
        assert_true(r.as_int32() == expected, String(dtype))
    for dtype in [timestamp(nanosecond), timestamp(nanosecond, "UTC")]:
        var r = _apply("bucket[100]", _timestamps(ns, dtype))
        assert_true(r.as_int32() == expected, String(dtype))


def test_iceberg_bucket_bytes() raises:
    var strings: DynArray = array(["iceberg", None])
    var r = _apply("bucket[100]", strings)
    assert_true(r.as_int32() == _buckets([1210000089, None], 100))

    var four: List[UInt8] = [0, 1, 2, 3]
    r = _apply("bucket[100]", _binaries([four.copy(), []]))
    assert_true(r.as_int32() == _buckets([-188683207, None], 100))

    var fixed = FixedSizeBinaryBuilder(4)
    fixed.append(Span(four))
    fixed.append_null()
    r = _apply("bucket[100]", fixed.finish())
    assert_true(r.as_int32() == _buckets([-188683207, None], 100))

    var uuid = FixedSizeBinaryBuilder(16)
    var uuid_bytes = _uuid()
    uuid.append(Span(uuid_bytes))
    uuid.append_null()
    r = _apply("bucket[100]", uuid.finish())
    assert_true(r.as_int32() == _buckets([1488055340, None], 100))


def test_iceberg_bucket_rejects() raises:
    var bools: DynArray = array([True])
    with assert_raises(contains="TypeError"):
        _ = _apply("bucket[4]", bools)
    var floats: DynArray = array([1.0], float64)
    with assert_raises(contains="TypeError"):
        _ = _apply("bucket[4]", floats)


# --- truncate --------------------------------------------------------------


def test_iceberg_truncate_integers() raises:
    var values: List[Optional[Int]] = [1, -1, 0, 10, -10, 15, -15, None]
    var expected: List[Optional[Int]] = [0, -10, 0, 10, -10, 10, -20, None]
    var ints: DynArray = array(values, int32)
    assert_true(
        _apply("truncate[10]", ints).as_int32() == array(expected, int32)
    )
    var longs: DynArray = array(values, int64)
    assert_true(
        _apply("truncate[10]", longs).as_int64() == array(expected, int64)
    )


def test_iceberg_truncate_decimal() raises:
    # W=50 at scale 2: 10.65 -> 10.50, -0.05 -> -0.50, 12.34 -> 12.00 (W=100).
    var r = _apply("truncate[50]", _decimals([1065, -5, 1050, None], 9, 2))
    assert_true(
        r.as_decimal128()
        == _decimals([1050, -50, 1050, None], 9, 2).as_decimal128()
    )
    assert_true(r.dtype() == decimal128(9, 2))


def test_iceberg_truncate_strings() raises:
    var strings: DynArray = array(
        ["iceberg", "ab", "héllo", "日本語テキスト", "🙂🙂🙂🙂", "", None]
    )
    var expected = array(["ice", "ab", "hél", "日本語", "🙂🙂🙂", "", None])
    assert_true(_apply("truncate[3]", strings).as_string() == expected)


def test_iceberg_truncate_binary() raises:
    var long_: List[UInt8] = [1, 2, 3, 4, 5]
    var short: List[UInt8] = [0xFF]
    var cut: List[UInt8] = [1, 2, 3]
    var r = _apply("truncate[3]", _binaries([long_.copy(), short.copy(), []]))
    assert_true(
        r.as_binary() == _binaries([cut.copy(), short.copy(), []]).as_binary()
    )


def test_iceberg_truncate_rejects() raises:
    with assert_raises(contains="TypeError"):
        _ = _apply("truncate[3]", _dates([1]))
    var floats: DynArray = array([1.0], float64)
    with assert_raises(contains="TypeError"):
        _ = _apply("truncate[3]", floats)


# --- year / month / day / hour ---------------------------------------------


def test_iceberg_calendar_dates() raises:
    # 2017-11-16, 1969-12-31, 1900-03-01, 1970-01-01.
    var dates = _dates([17486, -1, -25508, 0, None])
    var years = _apply("year", dates)
    assert_true(years.as_int32() == array([47, -1, -70, 0, None], int32))
    var months = _apply("month", dates)
    assert_true(months.as_int32() == array([574, -1, -838, 0, None], int32))
    var days = _apply("day", dates)
    assert_true(days.as_date32() == dates.as_date32())
    with assert_raises(contains="TypeError"):
        _ = _apply("hour", dates)


def test_iceberg_calendar_timestamps() raises:
    var us: List[Optional[Int]] = [_TS, -1, 0, None]
    var ns: List[Optional[Int]] = [_TS * 1000, -1, 0, None]
    var cases = [
        _timestamps(us, timestamp(microsecond)),
        _timestamps(us, timestamp(microsecond, "UTC")),
        _timestamps(ns, timestamp(nanosecond)),
        _timestamps(ns, timestamp(nanosecond, "UTC")),
    ]
    for ts in cases:
        var years = _apply("year", ts)
        assert_true(years.as_int32() == array([47, -1, 0, None], int32))
        var months = _apply("month", ts)
        assert_true(months.as_int32() == array([574, -1, 0, None], int32))
        var days = _apply("day", ts)
        assert_true(
            days.as_date32() == _dates([17486, -1, 0, None]).as_date32()
        )
        var hours = _apply("hour", ts)
        assert_true(hours.as_int32() == array([419686, -1, 0, None], int32))


def test_iceberg_calendar_rejects() raises:
    var strings: DynArray = array(["2017"])
    with assert_raises(contains="TypeError"):
        _ = _apply("year", strings)
    with assert_raises(contains="TypeError"):
        _ = _apply("day", _times([0]))


# --- identity / void / result types ----------------------------------------


def test_iceberg_identity_and_void() raises:
    var ints: DynArray = array([1, None, 3], int32)
    assert_true(_apply("identity", ints).as_int32() == ints.as_int32())
    var voided = _apply("void", ints)
    assert_true(voided.dtype() == int32)
    assert_equal(voided.length(), 3)
    assert_equal(voided.null_count(), 3)


def test_iceberg_result_types() raises:
    var ts = timestamp(microsecond)
    assert_true(Transform.parse("bucket[8]").result_type(string) == int32)
    assert_true(Transform.parse("year").result_type(ts) == int32)
    assert_true(Transform.parse("month").result_type(date32()) == int32)
    assert_true(Transform.parse("hour").result_type(ts) == int32)
    assert_true(Transform.parse("day").result_type(ts) == date32())
    assert_true(Transform.parse("truncate[4]").result_type(binary) == binary)
    assert_true(
        Transform.parse("identity").result_type(decimal128(9, 2))
        == decimal128(9, 2)
    )
    assert_true(Transform.parse("void").result_type(bool_) == bool_)
    with assert_raises(contains="TypeError"):
        _ = Transform.parse("identity").result_type(list_(int32))
