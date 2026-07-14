from std.testing import assert_equal, assert_true, assert_raises
from marrow.testing import TestSuite

from marrow.arrays import AnyArray, NullArray
from marrow.builders import array
from marrow.dtypes import (
    bool_,
    null,
    timestamp,
    date32,
    date64,
    second,
    millisecond,
    microsecond,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    string,
    Int16Type,
    Int32Type,
    Float16Type,
    Float32Type,
    Float64Type,
    Int8Type,
    UInt8Type,
    Int64Type,
)
from marrow.kernels.cast import cast


# ---------------------------------------------------------------------------
# Typed numeric casts
# ---------------------------------------------------------------------------


def test_int_to_float_widen() raises:
    var a = array([1, 2, 3], int32)
    var r = cast[Int32Type, Float64Type](a)
    assert_true(r == array([1.0, 2.0, 3.0], float64))


def test_float_to_int_truncates_toward_zero() raises:
    var a = array([1.9, -1.9, 2.5, -2.5], float64)
    var r = cast[Float64Type, Int32Type](a, safe=False)
    assert_true(r == array([1, -1, 2, -2], int32))


def test_negative_narrowing_wraps() raises:
    var a = array([-1, 300, 44], int32)
    var r = cast[Int32Type, UInt8Type](a, safe=False)
    assert_equal(Int(r.unsafe_get(0)), 255)
    assert_equal(Int(r.unsafe_get(1)), 44)  # 300 & 0xFF
    assert_equal(Int(r.unsafe_get(2)), 44)


def test_int16_wraps_to_int8() raises:
    var a = array([300], int16)
    var r = cast[Int16Type, Int8Type](a, safe=False)
    assert_equal(Int(r.unsafe_get(0)), 44)  # 300 & 0xFF = 44


def test_float16_roundtrip() raises:
    var a = array([1.0, 2.5, -3.25], float32)
    var half = cast[Float32Type, Float16Type](a)
    var back = cast[Float16Type, Float32Type](half)
    assert_true(back == a)


# ---------------------------------------------------------------------------
# Null preservation
# ---------------------------------------------------------------------------


def test_nulls_preserved() raises:
    var a = array([1, None, 3], int32)
    var r = cast[Int32Type, Float64Type](a)
    assert_equal(r.nulls, 1)
    assert_true(r.is_valid(0))
    assert_true(not r.is_valid(1))
    assert_true(r.is_valid(2))


# ---------------------------------------------------------------------------
# Safe mode
# ---------------------------------------------------------------------------


def test_safe_lossless_ok() raises:
    var a = array([1, 2, 3], int32)
    var r = cast[Int32Type, Int64Type](a, safe=True)
    assert_true(r == array([1, 2, 3], int64))


def test_safe_overflow_raises() raises:
    var a = array([300], int32)
    with assert_raises():
        _ = cast[Int32Type, Int8Type](a, safe=True)


def test_safe_float_truncation_raises() raises:
    var a = array([3.9], float64)
    with assert_raises():
        _ = cast[Float64Type, Int32Type](a, safe=True)


def test_safe_negative_to_unsigned_raises() raises:
    var a = array([-1], int32)
    with assert_raises():
        _ = cast[Int32Type, UInt8Type](a, safe=True)


def test_safe_skips_null_lanes() raises:
    # A null lane holds arbitrary data that need not be representable; safe mode
    # must not raise on it.
    var a = array([1, None, 2], int32)
    var r = cast[Int32Type, Int8Type](a, safe=True)
    assert_equal(r.nulls, 1)


def test_unsafe_overflow_ok() raises:
    var a = array([300], int32)
    var r = cast[Int32Type, Int8Type](a, safe=False)
    assert_equal(Int(r.unsafe_get(0)), 44)


# ---------------------------------------------------------------------------
# Runtime (AnyArray) dispatch
# ---------------------------------------------------------------------------


def test_anyarray_dispatch() raises:
    var a: AnyArray = array([1, 2, 3], int32)
    var r = cast(a, float64)
    assert_true(r.dtype() == float64)
    assert_true(r.as_float64() == array([1.0, 2.0, 3.0], float64))


def test_identity_zero_copy() raises:
    var a: AnyArray = array([1, 2, 3], int32)
    var r = cast(a, int32)
    assert_true(r.dtype() == int32)
    assert_true(r.as_int32() == array([1, 2, 3], int32))


def test_anyarray_every_target() raises:
    var src: AnyArray = array([1, 2, 3], int32)
    assert_true(cast(src, int8).dtype() == int8)
    assert_true(cast(src, uint16).dtype() == uint16)
    assert_true(cast(src, float32).dtype() == float32)
    assert_true(cast(src, uint64).dtype() == uint64)
    assert_true(cast(src, float16).dtype() == float16)


# ---------------------------------------------------------------------------
# Bool casts
# ---------------------------------------------------------------------------


def test_numeric_to_bool() raises:
    var a: AnyArray = array([0, 5, 0, -3], int32)
    var r = cast(a, bool_)
    assert_true(r.dtype() == bool_)
    assert_true(r.as_bool() == array([False, True, False, True]))


def test_bool_to_numeric() raises:
    var a: AnyArray = array([True, False, True])
    var r = cast(a, int8)
    assert_true(r.dtype() == int8)
    assert_true(r.as_int8() == array([1, 0, 1], int8))


def test_bool_to_float() raises:
    var a: AnyArray = array([True, False, True])
    var r = cast(a, float64)
    assert_true(r.as_float64() == array([1.0, 0.0, 1.0], float64))


def test_bool_cast_nulls_preserved() raises:
    var a: AnyArray = array([1, None, 0], int32)
    var r = cast(a, bool_)
    ref rb = r.as_bool()
    assert_equal(rb.nulls, 1)
    assert_true(not rb.is_valid(1))


def test_bool_identity() raises:
    var a: AnyArray = array([True, False, True])
    var r = cast(a, bool_)
    assert_true(r.dtype() == bool_)


# ---------------------------------------------------------------------------
# Temporal casts
# ---------------------------------------------------------------------------


def test_temporal_int_reinterpret() raises:
    var i: AnyArray = array([10, 20, 30], int64)
    var ts = cast(i, timestamp(microsecond))
    assert_true(ts.dtype() == timestamp(microsecond).to_any())
    var back = cast(ts, int64)
    assert_true(back.as_int64() == array([10, 20, 30], int64))


def test_timestamp_unit_upscale() raises:
    var i: AnyArray = array([1, 2, 3], int64)
    var ts_s = cast(i, timestamp(second))
    var ts_ms = cast(ts_s, timestamp(millisecond))  # * 1000
    assert_true(ts_ms.dtype() == timestamp(millisecond).to_any())
    assert_true(
        cast(ts_ms, int64).as_int64() == array([1000, 2000, 3000], int64)
    )


def test_timestamp_unit_downscale() raises:
    var i: AnyArray = array([1500, 2500], int64)
    var ts_ms = cast(i, timestamp(millisecond))
    var ts_s = cast(ts_ms, timestamp(second))  # // 1000 truncates
    assert_true(cast(ts_s, int64).as_int64() == array([1, 2], int64))


def test_date32_to_date64() raises:
    var i: AnyArray = array([1, 2], int32)
    var d32 = cast(i, date32())  # days, int32
    var d64 = cast(d32, date64())  # * 86_400_000, widen to int64
    assert_true(d64.dtype() == date64().to_any())
    assert_true(
        cast(d64, int64).as_int64() == array([86_400_000, 172_800_000], int64)
    )


def test_timestamp_tz_relabel() raises:
    var i: AnyArray = array([5], int64)
    var naive = cast(i, timestamp(second))
    var aware = cast(naive, timestamp(second, "UTC"))  # metadata only
    assert_true(aware.dtype() == timestamp(second, "UTC").to_any())
    assert_true(cast(aware, int64).as_int64() == array([5], int64))


def test_temporal_nulls_preserved() raises:
    var i: AnyArray = array([1, None, 3], int64)
    var ts = cast(i, timestamp(second))
    var ms = cast(ts, timestamp(millisecond))
    assert_equal(ms.null_count(), 1)
    assert_true(not ms.is_valid(1))


# ---------------------------------------------------------------------------
# String casts
# ---------------------------------------------------------------------------


def test_numeric_to_string() raises:
    var a: AnyArray = array([1, 2, 3], int32)
    var r = cast(a, string)
    assert_true(r.dtype() == string)
    assert_equal(String(r.as_string()[0]), "1")
    assert_equal(String(r.as_string()[2]), "3")


def test_string_to_int() raises:
    var a: AnyArray = array(["1", "22", "-3"])
    var r = cast(a, int32)
    assert_true(r.dtype() == int32)
    assert_true(r.as_int32() == array([1, 22, -3], int32))


def test_string_to_float() raises:
    var a: AnyArray = array(["1.5", "-2.25", "3.0"])
    var r = cast(a, float64)
    assert_true(r.as_float64() == array([1.5, -2.25, 3.0], float64))


def test_string_to_int_parse_error_safe_raises() raises:
    var a: AnyArray = array(["1", "oops", "3"])
    with assert_raises():
        _ = cast(a, int32, safe=True)


def test_string_to_int_parse_error_unsafe_nulls() raises:
    var a: AnyArray = array(["1", "oops", "3"])
    var r = cast(a, int32, safe=False)
    assert_equal(r.null_count(), 1)
    assert_true(not r.is_valid(1))
    assert_true(r.is_valid(0))


def test_string_roundtrip_nulls() raises:
    var a: AnyArray = array([1, None, 3], int32)
    var s = cast(a, string)
    assert_equal(s.null_count(), 1)
    var back = cast(s, int32)
    assert_true(back.as_int32() == array([1, None, 3], int32))


def test_bool_to_string() raises:
    var a: AnyArray = array([True, False, True])
    var r = cast(a, string)
    assert_equal(String(r.as_string()[0]), "true")
    assert_equal(String(r.as_string()[1]), "false")


def test_string_to_bool() raises:
    var a: AnyArray = array(["true", "False", "1", "0"])
    var r = cast(a, bool_)
    assert_true(r.as_bool() == array([True, False, True, False]))


# ---------------------------------------------------------------------------
# Null casts
# ---------------------------------------------------------------------------


def test_null_to_numeric() raises:
    var a: AnyArray = NullArray(length=3)
    var r = cast(a, int64)
    assert_true(r.dtype() == int64)
    assert_equal(len(r), 3)
    assert_equal(r.null_count(), 3)


def test_null_to_string() raises:
    var a: AnyArray = NullArray(length=2)
    var r = cast(a, string)
    assert_true(r.dtype() == string)
    assert_equal(r.null_count(), 2)


def main() raises:
    TestSuite.run[__functions_in_module()]()
