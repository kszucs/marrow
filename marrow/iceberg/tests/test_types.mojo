# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Iceberg primitive type strings to and from Arrow dtypes."""

from std.testing import assert_equal, assert_raises, assert_true

from ...dtypes import (
    DynType,
    binary,
    bool_,
    date32,
    decimal32,
    decimal128,
    fixed_size_binary_,
    float32,
    float64,
    int16,
    int32,
    int64,
    large_string,
    list_,
    microsecond,
    millisecond,
    nanosecond,
    string,
    time64,
    timestamp,
)
from ..types import primitive_name, primitive_type


def _check(name: String, dtype: DynType) raises:
    assert_true(primitive_type(name) == dtype, String(primitive_type(name)))
    assert_equal(primitive_name(dtype), name)


def test_iceberg_primitive_round_trip() raises:
    _check("boolean", bool_)
    _check("int", int32)
    _check("long", int64)
    _check("float", float32)
    _check("double", float64)
    _check("decimal(9, 2)", decimal128(9, 2))
    _check("decimal(38, 0)", decimal128(38, 0))
    _check("date", date32())
    _check("time", time64(microsecond))
    _check("timestamp", timestamp(microsecond))
    _check("timestamptz", timestamp(microsecond, "UTC"))
    _check("timestamp_ns", timestamp(nanosecond))
    _check("timestamptz_ns", timestamp(nanosecond, "UTC"))
    _check("string", string)
    _check("fixed[7]", fixed_size_binary_(7))
    _check("binary", binary)


def test_iceberg_primitive_uuid_and_spellings() raises:
    assert_true(primitive_type("uuid") == fixed_size_binary_(16))
    assert_true(primitive_type("decimal(9,2)") == decimal128(9, 2))
    assert_true(primitive_type(" decimal( 10 ,  3 ) ") == decimal128(10, 3))
    assert_true(primitive_type("fixed[ 4 ]") == fixed_size_binary_(4))


def test_iceberg_primitive_name_alternatives() raises:
    assert_equal(primitive_name(large_string), "string")
    assert_equal(primitive_name(decimal32(5, 1)), "decimal(5, 1)")
    assert_equal(
        primitive_name(timestamp(microsecond, "Etc/UTC")), "timestamptz"
    )
    assert_equal(
        primitive_name(timestamp(nanosecond, "+00:00")), "timestamptz_ns"
    )


def test_iceberg_primitive_name_rejects() raises:
    with assert_raises(contains="TypeError"):
        _ = primitive_name(int16)
    with assert_raises(contains="TypeError"):
        _ = primitive_name(timestamp(millisecond))
    with assert_raises(contains="TypeError"):
        _ = primitive_name(timestamp(microsecond, "America/New_York"))
    with assert_raises(contains="TypeError"):
        _ = primitive_name(time64(nanosecond))
    with assert_raises(contains="TypeError"):
        _ = primitive_name(list_(int32))


def test_iceberg_primitive_unsupported() raises:
    with assert_raises(contains="NotImplementedError"):
        _ = primitive_type("variant")
    with assert_raises(contains="NotImplementedError"):
        _ = primitive_type("unknown")
    with assert_raises(contains="NotImplementedError"):
        _ = primitive_type("geometry(srid:4326)")
    with assert_raises(contains="NotImplementedError"):
        _ = primitive_type("geography(srid:4326, spherical)")


def test_iceberg_primitive_malformed() raises:
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("integer")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("decimal(9)")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("decimal(39, 2)")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("decimal(4, 5)")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("decimal(a, 2)")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("fixed[16")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("fixed[]")
    with assert_raises(contains="InvalidError"):
        _ = primitive_type("fixed(16)")
