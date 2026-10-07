# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Iceberg's binary single-value serialization (spec Appendix D)."""

from std.testing import assert_equal, assert_raises, assert_true

from ...dtypes import (
    Date32Type,
    Decimal128Type,
    Time64Type,
    TimestampType,
    date32,
    decimal128,
    fixed_size_binary_,
    int16,
    microsecond,
    nanosecond,
    time64,
    timestamp,
)
from ...scalars import (
    BinaryScalar,
    BoolScalar,
    DynScalar,
    FixedSizeBinaryScalar,
    Float32Scalar,
    Float64Scalar,
    Int32Scalar,
    Int64Scalar,
    PrimitiveScalar,
    StringScalar,
)
from ..bounds import decode_bound, encode_bound


def _round_trip(value: DynScalar, expected: List[UInt8]) raises:
    """`value` encodes to `expected`, which decodes back to `value`."""
    var encoded = encode_bound(value)
    assert_true(encoded == expected, String(value))
    # Compared through its text and its re-encoding: `DynScalar.__eq__`
    # deadlocks the compiler when instantiated (see CLAUDE.md).
    var decoded = decode_bound(value.type(), Span(expected))
    assert_true(decoded.type() == value.type(), String(decoded.type()))
    assert_equal(String(decoded), String(value))
    assert_true(encode_bound(decoded) == expected)


def _decimal(unscaled: Int, precision: Int, scale: Int) -> DynScalar:
    return PrimitiveScalar[Decimal128Type](
        Int128(unscaled), decimal128(precision, scale)
    )


def test_iceberg_bound_fixed_width() raises:
    _round_trip(BoolScalar(True), [0x01])
    _round_trip(BoolScalar(False), [0x00])
    _round_trip(Int32Scalar(34), [0x22, 0, 0, 0])
    _round_trip(Int32Scalar(-1), [0xFF, 0xFF, 0xFF, 0xFF])
    _round_trip(Int64Scalar(34), [0x22, 0, 0, 0, 0, 0, 0, 0])
    _round_trip(Float32Scalar(1.0), [0, 0, 0x80, 0x3F])
    _round_trip(Float64Scalar(-2.0), [0, 0, 0, 0, 0, 0, 0, 0xC0])


def test_iceberg_bound_temporal() raises:
    # 2017-11-16 is day 17486 = 0x444E.
    _round_trip(
        PrimitiveScalar[Date32Type](Int32(17486), date32()), [0x4E, 0x44, 0, 0]
    )
    _round_trip(
        PrimitiveScalar[Date32Type](Int32(-1), date32()),
        [0xFF, 0xFF, 0xFF, 0xFF],
    )
    # 22:31:08 is 81,068,000,000 us = 0x12_E007_8300.
    _round_trip(
        PrimitiveScalar[Time64Type](Int64(81_068_000_000), time64(microsecond)),
        [0x00, 0x83, 0x07, 0xE0, 0x12, 0, 0, 0],
    )
    var micros = Int64(1_510_871_468_000_000)
    var expected = List[UInt8]()
    for i in range(8):
        expected.append(UInt8((micros >> Int64(8 * i)) & 0xFF))
    _round_trip(
        PrimitiveScalar[TimestampType](micros, timestamp(microsecond)),
        expected,
    )
    _round_trip(
        PrimitiveScalar[TimestampType](micros, timestamp(microsecond, "UTC")),
        expected,
    )
    _round_trip(
        PrimitiveScalar[TimestampType](micros, timestamp(nanosecond)),
        expected,
    )
    _round_trip(
        PrimitiveScalar[TimestampType](micros, timestamp(nanosecond, "UTC")),
        expected,
    )


def test_iceberg_bound_bytes() raises:
    _round_trip(
        StringScalar("iceberg"), [0x69, 0x63, 0x65, 0x62, 0x65, 0x72, 0x67]
    )
    _round_trip(StringScalar("é"), [0xC3, 0xA9])
    _round_trip(StringScalar(""), [])
    var raw: List[UInt8] = [0x00, 0x01, 0xFF, 0x80]
    _round_trip(
        BinaryScalar(String(StringSlice(unsafe_from_utf8=Span(raw)))), raw
    )
    _round_trip(FixedSizeBinaryScalar(raw.copy(), 4), raw)
    var uuid: List[UInt8] = [
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
    _round_trip(FixedSizeBinaryScalar(uuid.copy(), 16), uuid)


def test_iceberg_bound_decimal() raises:
    # 14.20 as decimal(9, 2): unscaled 1420 = 0x058C.
    _round_trip(_decimal(1420, 9, 2), [0x05, 0x8C])
    _round_trip(_decimal(-1420, 9, 2), [0xFA, 0x74])
    _round_trip(_decimal(0, 9, 2), [0x00])
    _round_trip(_decimal(-1, 9, 2), [0xFF])
    _round_trip(_decimal(127, 9, 0), [0x7F])
    _round_trip(_decimal(128, 9, 0), [0x00, 0x80])
    _round_trip(_decimal(-128, 9, 0), [0x80])
    _round_trip(_decimal(-129, 9, 0), [0xFF, 0x7F])


def test_iceberg_bound_decimal_wide() raises:
    # 10^38 - 1 needs all 16 bytes.
    var big = Int128(10) ** 38 - 1
    var value: DynScalar = PrimitiveScalar[Decimal128Type](
        big, decimal128(38, 0)
    )
    var encoded = encode_bound(value)
    assert_equal(len(encoded), 16)
    var decoded = decode_bound(value.type(), Span(encoded))
    assert_true(decoded.as_decimal128().value() == big)
    var negative: DynScalar = PrimitiveScalar[Decimal128Type](
        -big, decimal128(38, 0)
    )
    encoded = encode_bound(negative)
    assert_equal(len(encoded), 16)
    decoded = decode_bound(negative.type(), Span(encoded))
    assert_true(decoded.as_decimal128().value() == -big)


def test_iceberg_bound_wrong_length() raises:
    var three: List[UInt8] = [1, 2, 3]
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(Int32Scalar(0).type(), Span(three))
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(Int64Scalar(0).type(), Span(three))
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(date32(), Span(three))
    with assert_raises(contains="CorruptError: iceberg bound: timestamp[us]"):
        _ = decode_bound(timestamp(microsecond), Span(three))
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(BoolScalar(True).type(), Span(three))
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(fixed_size_binary_(16), Span(three))
    var empty = List[UInt8]()
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(decimal128(9, 2), Span(empty))
    var seventeen = List[UInt8](length=17, fill=0)
    with assert_raises(contains="CorruptError"):
        _ = decode_bound(decimal128(38, 2), Span(seventeen))


def test_iceberg_bound_rejects() raises:
    var one: List[UInt8] = [1, 0]
    with assert_raises(contains="TypeError"):
        _ = decode_bound(int16, Span(one))
    with assert_raises(contains="InvalidError"):
        _ = encode_bound(Int32Scalar(None))
