# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from std.utils.numerics import inf, nan

from ...arrays import (
    BinaryArray,
    DynArray,
    NullArray,
    DictionaryArray,
    MapArray,
)
from ...buffers import Bitmap
from ...builders import (
    array,
    FixedSizeBinaryBuilder,
    ListBuilder,
    MapBuilder,
    StructBuilder,
    Int32Builder,
)
from ...dtypes import (
    map_,
    DynType,
    bool_,
    null,
    timestamp,
    time32,
    time64,
    date32,
    date64,
    second,
    millisecond,
    microsecond,
    nanosecond,
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
    binary,
    large_binary,
    large_string,
    fixed_size_binary_,
    decimal32,
    decimal64,
    decimal128,
    decimal256,
    list_,
    struct_,
    dictionary,
    field,
    Int16Type,
    Int32Type,
    Float16Type,
    Float32Type,
    Float64Type,
    Int8Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Int64Type,
    NumericType,
    BinaryType,
    StringType,
)
from ...kernels.cast import (
    cast,
    BinaryLikeCastKernel,
    FixedSizeBinaryCastKernel,
    NumericCastKernel,
)


# ---------------------------------------------------------------------------
# Typed numeric casts
# ---------------------------------------------------------------------------


def test_int_to_float_widen() raises:
    var a = array([1, 2, 3], int32)
    var r = NumericCastKernel.apply[Int32Type, Float64Type](a)
    assert_true(r == array([1.0, 2.0, 3.0], float64))


def test_float_to_int_truncates_toward_zero() raises:
    var a = array([1.9, -1.9, 2.5, -2.5], float64)
    var r = NumericCastKernel.apply[Float64Type, Int32Type, False](a)
    assert_true(r == array([1, -1, 2, -2], int32))


def test_negative_narrowing_wraps() raises:
    var a = array([-1, 300, 44], int32)
    var r = NumericCastKernel.apply[Int32Type, UInt8Type, False](a)
    assert_equal(Int(r.unsafe_get(0)), 255)
    assert_equal(Int(r.unsafe_get(1)), 44)  # 300 & 0xFF
    assert_equal(Int(r.unsafe_get(2)), 44)


def test_int16_wraps_to_int8() raises:
    var a = array([300], int16)
    var r = NumericCastKernel.apply[Int16Type, Int8Type, False](a)
    assert_equal(Int(r.unsafe_get(0)), 44)  # 300 & 0xFF = 44


def test_float16_roundtrip() raises:
    var a = array([1.0, 2.5, -3.25], float32)
    var half = NumericCastKernel.apply[Float32Type, Float16Type](a)
    var back = NumericCastKernel.apply[Float16Type, Float32Type](half)
    assert_true(back == a)


# ---------------------------------------------------------------------------
# Null preservation
# ---------------------------------------------------------------------------


def test_nulls_preserved() raises:
    var a = array([1, None, 3], int32)
    var r = NumericCastKernel.apply[Int32Type, Float64Type](a)
    assert_equal(r.nulls, 1)
    assert_true(r.is_valid(0))
    assert_true(not r.is_valid(1))
    assert_true(r.is_valid(2))


# ---------------------------------------------------------------------------
# Safe mode
# ---------------------------------------------------------------------------


def test_safe_lossless_ok() raises:
    var a = array([1, 2, 3], int32)
    var r = NumericCastKernel.apply[Int32Type, Int64Type, True](a)
    assert_true(r == array([1, 2, 3], int64))


def test_safe_overflow_raises() raises:
    var a = array([300], int32)
    with assert_raises():
        _ = NumericCastKernel.apply[Int32Type, Int8Type, True](a)


def test_safe_float_truncation_raises() raises:
    var a = array([3.9], float64)
    with assert_raises():
        _ = NumericCastKernel.apply[Float64Type, Int32Type, True](a)


def test_safe_negative_to_unsigned_raises() raises:
    var a = array([-1], int32)
    with assert_raises():
        _ = NumericCastKernel.apply[Int32Type, UInt8Type, True](a)


def test_safe_sign_change_raises() raises:
    """A sign change survives a round trip (int8 -1 -> uint8 255 -> -1), so
    the check must compare against the target's bounds."""
    with assert_raises():
        _ = NumericCastKernel.apply[Int8Type, UInt8Type, True](
            array([-1], int8)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[Int8Type, UInt16Type, True](
            array([-1], int8)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[UInt8Type, Int8Type, True](
            array([200], uint8)
        )
    var r = NumericCastKernel.apply[UInt8Type, Int8Type, True](
        array([0, 127], uint8)
    )
    assert_true(r == array([0, 127], int8))


def _check_safe_cast[
    In: NumericType, Out: NumericType
](fits: List[Scalar[In.native]], overflows: List[Scalar[In.native]]) raises:
    """Under safe, every value in ``fits`` casts to ``Out`` and back unchanged,
    and each value in ``overflows`` raises."""
    var r = NumericCastKernel.apply[In, Out, True](array(fits, In()))
    for i in range(len(fits)):
        assert_equal(
            r[i].value().cast[In.native](),
            fits[i],
            msg=String(In(), " -> ", Out()),
        )
    for v in overflows:
        var raised = False
        try:
            _ = NumericCastKernel.apply[In, Out, True](array([v], In()))
        except:
            raised = True
        assert_true(raised, msg=String(In(), " ", v, " -> ", Out()))


def test_safe_int_cast_boundaries() raises:
    """Under safe an integer casts exactly when the target holds it: each
    bound is accepted and the value beyond it refused. A float holds an
    integer within ``±2^(mantissa + 1)``."""
    # signed -> unsigned, narrower
    _check_safe_cast[Int16Type, UInt8Type]([0, 255], [-1, 256, -32768, 32767])
    _check_safe_cast[Int64Type, UInt32Type]([0, 4294967295], [-1, 4294967296])
    # signed -> unsigned, as wide or wider: a sign change survives a round trip
    _check_safe_cast[Int8Type, UInt8Type]([0, 127], [-1, -128])
    _check_safe_cast[Int8Type, UInt16Type]([0, 127], [-1, -128])
    _check_safe_cast[Int64Type, UInt64Type](
        [0, 9223372036854775807], [-1, -9223372036854775808]
    )
    # signed -> signed, narrower
    _check_safe_cast[Int16Type, Int8Type](
        [-128, 127], [-129, 128, -32768, 32767]
    )
    _check_safe_cast[Int64Type, Int32Type](
        [-2147483648, 2147483647], [-2147483649, 2147483648]
    )
    # unsigned -> narrower
    _check_safe_cast[UInt8Type, Int8Type]([0, 127], [128, 255])
    _check_safe_cast[UInt16Type, UInt8Type]([0, 255], [256, 65535])
    _check_safe_cast[UInt32Type, Int16Type]([0, 32767], [32768, 4294967295])
    _check_safe_cast[UInt64Type, Int64Type](
        [0, 9223372036854775807],
        [9223372036854775808, 18446744073709551615],
    )
    # wider: nothing to refuse
    _check_safe_cast[Int8Type, Int64Type]([-128, 127], [])
    _check_safe_cast[UInt8Type, UInt16Type]([0, 255], [])
    _check_safe_cast[UInt32Type, Int64Type]([0, 4294967295], [])
    # integer -> float: within 2^(mantissa + 1)
    _check_safe_cast[Int8Type, Float16Type]([-128, 127], [])
    _check_safe_cast[Int16Type, Float16Type]([-2048, 2048], [-2049, 2049, 4096])
    _check_safe_cast[UInt16Type, Float16Type]([0, 2048], [2049, 65535])
    _check_safe_cast[Int32Type, Float32Type](
        [-16777216, 16777216], [-16777217, 16777217, 33554432]
    )
    _check_safe_cast[Int32Type, Float64Type]([-2147483648, 2147483647], [])
    _check_safe_cast[Int64Type, Float64Type](
        [-9007199254740992, 9007199254740992],
        [-9007199254740993, 9007199254740993, 18014398509481984],
    )
    _check_safe_cast[UInt64Type, Float64Type](
        [0, 9007199254740992], [9007199254740993, 18446744073709551615]
    )


def test_safe_int_to_float_beyond_mantissa_raises() raises:
    """Arrow accepts an integer into a float only within 2^(mantissa + 1),
    even where a larger one happens to be representable."""
    var ok = array([9007199254740992, -9007199254740992], int64)
    _ = NumericCastKernel.apply[Int64Type, Float64Type, True](ok)
    with assert_raises():
        _ = NumericCastKernel.apply[Int64Type, Float64Type, True](
            array([1152921504606846976], int64)
        )


def test_safe_float_to_int_checks_range() raises:
    """Out of range is tested on the float itself: the converted value of an
    out-of-range float is undefined, so a round trip cannot catch it."""
    with assert_raises():
        _ = NumericCastKernel.apply[Float64Type, Int8Type, True](
            array([128.0], float64)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[Float64Type, Int64Type, True](
            array([9223372036854775808.0], float64)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[Float32Type, Int16Type, True](
            array([-32769.0], float32)
        )
    with assert_raises():  # 2^31: the float32 nearest INT32_MAX
        _ = NumericCastKernel.apply[Float32Type, Int32Type, True](
            array([2147483648.0], float32)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[Float64Type, UInt64Type, True](
            array([18446744073709551616.0], float64)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[Float16Type, Int32Type, True](
            array([inf[DType.float64]()], float16)
        )
    with assert_raises():
        _ = NumericCastKernel.apply[Float16Type, Int16Type, True](
            array([65504.0], float16)
        )
    with assert_raises():  # the smallest subnormal is a fraction
        _ = NumericCastKernel.apply[Float64Type, Int32Type, True](
            array([5e-324], float64)
        )
    var r = NumericCastKernel.apply[Float64Type, Int8Type, True](
        array([-128.0, 127.0, -0.0], float64)
    )
    assert_true(r == array([-128, 127, 0], int8))
    var r64 = NumericCastKernel.apply[Float64Type, Int64Type, True](
        array([-9223372036854775808.0, 9223372036854774784.0], float64)
    )
    assert_true(r64.unsafe_get(0) == Int64.MIN)
    assert_true(r64.unsafe_get(1) == 9223372036854774784)
    var u64 = NumericCastKernel.apply[Float64Type, UInt64Type, True](
        array([-0.0, 18446744073709549568.0], float64)
    )
    assert_true(u64.unsafe_get(0) == 0)
    assert_true(u64.unsafe_get(1) == 18446744073709549568)
    var h = NumericCastKernel.apply[Float16Type, UInt16Type, True](
        array([65504.0], float16)
    )
    assert_true(h == array([65504], uint16))


def test_safe_float_to_int_boundaries() raises:
    """Under safe a float casts exactly when it is an integer in ``[MIN, MAX]``
    of the target: each bound, and the largest float below ``MAX + 1``, is
    accepted; the float on the far side of each, a fraction, NaN and the
    infinities are refused."""
    var nan64 = nan[DType.float64]()
    var inf64 = inf[DType.float64]()
    var nan32 = nan[DType.float32]()
    var inf32 = inf[DType.float32]()
    var nan16 = nan[DType.float16]()
    var inf16 = inf[DType.float16]()
    # the smallest subnormals, a fraction
    var tiny32 = bitcast[DType.float32, 1](UInt32(1))
    var tiny16 = bitcast[DType.float16, 1](UInt16(1))
    _check_safe_cast[Float64Type, Int8Type](
        [-128.0, 127.0, -0.0],
        [-129.0, 128.0, 127.5, 0.5, 5e-324, nan64, inf64, -inf64],
    )
    _check_safe_cast[Float64Type, Int32Type](
        [-2147483648.0, 2147483647.0],
        [-2147483649.0, 2147483648.0, 2147483647.5],
    )
    _check_safe_cast[Float64Type, UInt32Type](
        [0.0, 4294967295.0], [-1.0, 4294967296.0]
    )
    _check_safe_cast[Float64Type, Int64Type](
        [-9223372036854775808.0, 9223372036854774784.0],
        [-9223372036854777856.0, 9223372036854775808.0, nan64],
    )
    _check_safe_cast[Float64Type, UInt64Type](
        [-0.0, 18446744073709549568.0],
        [-1.0, -5e-324, 18446744073709551616.0, inf64],
    )
    _check_safe_cast[Float32Type, Int8Type](
        [-128.0, 127.0], [-129.0, 128.0, 127.5, tiny32]
    )
    _check_safe_cast[Float32Type, Int32Type](
        [-2147483648.0, 2147483520.0],
        [-2147483904.0, 2147483648.0, nan32, inf32],
    )
    _check_safe_cast[Float32Type, UInt32Type](
        [0.0, 4294967040.0], [-1.0, 4294967296.0]
    )
    _check_safe_cast[Float32Type, Int64Type](
        [-9223372036854775808.0, 9223371487098961920.0],
        [-9223373136366403584.0, 9223372036854775808.0],
    )
    _check_safe_cast[Float32Type, UInt64Type](
        [0.0, 18446742974197923840.0], [-1.0, 18446744073709551616.0]
    )
    # float16 compares in float32, where 2^31 is not infinity
    _check_safe_cast[Float16Type, Int8Type](
        [-128.0, 127.0], [-129.0, 128.0, 127.5, 0.5]
    )
    _check_safe_cast[Float16Type, Int16Type](
        [-32768.0, 32752.0], [-32800.0, 32768.0, nan16, inf16]
    )
    _check_safe_cast[Float16Type, UInt16Type](
        [0.0, 65504.0], [-1.0, nan16, inf16, -inf16]
    )
    _check_safe_cast[Float16Type, Int32Type](
        [-65504.0, 65504.0], [tiny16, nan16, inf16, -inf16]
    )
    _check_safe_cast[Float16Type, UInt64Type](
        [0.0, 65504.0], [-1.0, nan16, inf16]
    )


def test_safe_skips_null_lanes() raises:
    # A null lane holds arbitrary data that need not be representable; safe mode
    # must not raise on it.
    var a = array([1, None, 2], int32)
    var r = NumericCastKernel.apply[Int32Type, Int8Type, True](a)
    assert_equal(r.nulls, 1)


def test_unsafe_overflow_ok() raises:
    var a = array([300], int32)
    var r = NumericCastKernel.apply[Int32Type, Int8Type, False](a)
    assert_equal(Int(r.unsafe_get(0)), 44)


# ---------------------------------------------------------------------------
# Runtime (DynArray) dispatch
# ---------------------------------------------------------------------------


def test_anyarray_dispatch() raises:
    var a: DynArray = array([1, 2, 3], int32)
    var r = cast(a, float64)
    assert_true(r.dtype() == float64)
    assert_true(r.as_float64() == array([1.0, 2.0, 3.0], float64))


def test_identity_zero_copy() raises:
    var a: DynArray = array([1, 2, 3], int32)
    var r = cast(a, int32)
    assert_true(r.dtype() == int32)
    assert_true(r.as_int32() == array([1, 2, 3], int32))


def test_anyarray_every_target() raises:
    var src: DynArray = array([1, 2, 3], int32)
    assert_true(cast(src, int8).dtype() == int8)
    assert_true(cast(src, uint16).dtype() == uint16)
    assert_true(cast(src, float32).dtype() == float32)
    assert_true(cast(src, uint64).dtype() == uint64)
    assert_true(cast(src, float16).dtype() == float16)


# ---------------------------------------------------------------------------
# Bool casts
# ---------------------------------------------------------------------------


def test_numeric_to_bool() raises:
    var a: DynArray = array([0, 5, 0, -3], int32)
    var r = cast(a, bool_)
    assert_true(r.dtype() == bool_)
    assert_true(r.as_bool() == array([False, True, False, True]))


def test_bool_to_numeric() raises:
    var a: DynArray = array([True, False, True])
    var r = cast(a, int8)
    assert_true(r.dtype() == int8)
    assert_true(r.as_int8() == array([1, 0, 1], int8))


def test_bool_to_float() raises:
    var a: DynArray = array([True, False, True])
    var r = cast(a, float64)
    assert_true(r.as_float64() == array([1.0, 0.0, 1.0], float64))


def test_bool_cast_nulls_preserved() raises:
    var a: DynArray = array([1, None, 0], int32)
    var r = cast(a, bool_)
    ref rb = r.as_bool()
    assert_equal(rb.nulls, 1)
    assert_true(not rb.is_valid(1))


def test_bool_identity() raises:
    var a: DynArray = array([True, False, True])
    var r = cast(a, bool_)
    assert_true(r.dtype() == bool_)


# ---------------------------------------------------------------------------
# Temporal casts
# ---------------------------------------------------------------------------


def test_temporal_int_reinterpret() raises:
    var i: DynArray = array([10, 20, 30], int64)
    var ts = cast(i, timestamp(microsecond))
    assert_true(ts.dtype() == timestamp(microsecond).to_dyn())
    var back = cast(ts, int64)
    assert_true(back.as_int64() == array([10, 20, 30], int64))


def test_timestamp_unit_upscale() raises:
    var i: DynArray = array([1, 2, 3], int64)
    var ts_s = cast(i, timestamp(second))
    var ts_ms = cast(ts_s, timestamp(millisecond))  # * 1000
    assert_true(ts_ms.dtype() == timestamp(millisecond).to_dyn())
    assert_true(
        cast(ts_ms, int64).as_int64() == array([1000, 2000, 3000], int64)
    )


def test_timestamp_unit_downscale() raises:
    # `safe=False` is required: 1500 ms is not a whole number of seconds, and
    # the default `safe=True` now raises rather than discarding the remainder.
    # This case asserted the truncation under the default until S4 threaded
    # `safe` through to `TemporalCastKernel` — the suite encoded the defect.
    var i: DynArray = array([1500, 2500], int64)
    var ts_ms = cast(i, timestamp(millisecond))
    var ts_s = cast(ts_ms, timestamp(second), safe=False)  # // 1000 truncates
    assert_true(cast(ts_s, int64).as_int64() == array([1, 2], int64))


def test_timestamp_unit_downscale_truncates_toward_zero() raises:
    var ts = cast(array([-1, -1500, 1500], int64), timestamp(millisecond))
    var r = cast(ts, timestamp(second), safe=False)
    assert_true(cast(r, int64).as_int64() == array([0, -1, 1], int64))


def test_timestamp_to_date_floors_to_the_day() raises:
    """The day an instant falls in, floored: -1 s is the day before."""
    var ts = cast(array([1, -1, 86_400], int64), timestamp(second))
    var d64 = cast(ts, date64(), safe=True)
    assert_true(
        cast(d64, int64).as_int64()
        == array([0, -86_400_000, 86_400_000], int64)
    )
    var d32 = cast(ts, date32(), safe=True)
    assert_true(cast(d32, int32).as_int32() == array([0, -1, 1], int32))


def test_timestamp_to_time_takes_the_time_of_day() raises:
    var ts = cast(array([-1, 2_091_084], int64), timestamp(second))
    var t = cast(ts, time32(millisecond))
    assert_true(
        cast(t, int32).as_int32() == array([86_399_000, 17_484_000], int32)
    )
    var ms = cast(array([1500], int64), timestamp(millisecond))
    with assert_raises():
        _ = cast(ms, time32(second), safe=True)
    var lax = cast(ms, time32(second), safe=False)
    assert_true(cast(lax, int32).as_int32() == array([1], int32))


def test_timestamp_to_date_and_time_every_unit() raises:
    """Every unit, before and after the epoch: the floored day, and a time of
    day that is never negative."""
    var units = [second, millisecond, microsecond, nanosecond]
    var per_day = 86_400
    for unit in units:
        var ts = cast(
            array([-1, -per_day, -per_day - 1, per_day - 1, None], int64),
            timestamp(unit),
        )
        var days = cast(cast(ts, date32()), int32)
        assert_true(days.as_int32() == array([-1, -1, -2, 0, None], int32))
        var last = (per_day - 1) * (86_400_000_000_000 // per_day)
        var tod = cast(cast(ts, time64(nanosecond)), int64)
        assert_true(tod.as_int64() == array([last, 0, last, last, None], int64))
        per_day *= 1000
    var neg = cast(array([-1000], int64), timestamp(millisecond))
    var t = cast(neg, time32(second), safe=True)
    assert_true(cast(t, int32).as_int32() == array([86_399], int32))
    var lossy = cast(array([-1500], int64), timestamp(millisecond))
    with assert_raises():
        _ = cast(lossy, time32(second), safe=True)


def test_timestamp_to_date_and_time_reads_the_zone() raises:
    """A zoned timestamp is split in its own wall-clock time: the epoch is
    19:00 the day before in New York."""
    var naive = cast(array([0, -1, 18_000], int64), timestamp(second))
    var ny = cast(naive, timestamp(second, "America/New_York"))
    var days = cast(cast(ny, date32()), int32)
    assert_true(days.as_int32() == array([-1, -1, 0], int32))
    var tod = cast(cast(ny, time64(microsecond)), int64)
    assert_true(
        tod.as_int64() == array([68_400_000_000, 68_399_000_000, 0], int64)
    )


def test_date32_to_date64() raises:
    var i: DynArray = array([1, 2], int32)
    var d32 = cast(i, date32())  # days, int32
    var d64 = cast(d32, date64())  # * 86_400_000, widen to int64
    assert_true(d64.dtype() == date64().to_dyn())
    assert_true(
        cast(d64, int64).as_int64() == array([86_400_000, 172_800_000], int64)
    )


def test_timestamp_tz_relabel() raises:
    var i: DynArray = array([5], int64)
    var naive = cast(i, timestamp(second))
    var aware = cast(naive, timestamp(second, "UTC"))  # metadata only
    assert_true(aware.dtype() == timestamp(second, "UTC").to_dyn())
    assert_true(cast(aware, int64).as_int64() == array([5], int64))


def test_temporal_nulls_preserved() raises:
    var i: DynArray = array([1, None, 3], int64)
    var ts = cast(i, timestamp(second))
    var ms = cast(ts, timestamp(millisecond))
    assert_equal(ms.null_count(), 1)
    assert_true(not ms.is_valid(1))


# ---------------------------------------------------------------------------
# String casts
# ---------------------------------------------------------------------------


def test_numeric_to_string() raises:
    var a: DynArray = array([1, 2, 3], int32)
    var r = cast(a, string)
    assert_true(r.dtype() == string)
    assert_equal(String(r.as_string()[0]), "1")
    assert_equal(String(r.as_string()[2]), "3")


def test_string_to_int() raises:
    var a: DynArray = array(["1", "22", "-3"])
    var r = cast(a, int32)
    assert_true(r.dtype() == int32)
    assert_true(r.as_int32() == array([1, 22, -3], int32))


def test_string_to_float() raises:
    var a: DynArray = array(["1.5", "-2.25", "3.0"])
    var r = cast(a, float64)
    assert_true(r.as_float64() == array([1.5, -2.25, 3.0], float64))


def test_a_nan_is_not_zero_so_it_casts_to_true() raises:
    """`cast(nan, bool)` is True, as `pyarrow` 23.0.1 answers it.

    `NumToBoolKernel.core` was `a.ne(0)`, and `SIMD.ne` lowers to an *ordered*
    compare — False whenever either operand is a NaN — so a NaN cast to False
    while its own docstring called the test total. Negating `eq` is total.
    """
    var a: DynArray = array([nan[float64.native](), 0.0, 2.0], float64)
    var r = cast(a, bool_)
    assert_true(r.as_bool() == array([True, False, True]))


def test_a_nan_does_not_round_trip_so_a_safe_cast_to_int_raises() raises:
    """The same ordered-compare trap, one layer down in `core_checked`.

    `needs_check` is True for every float→int pair, so a NaN is exactly what
    arrives here; `out.cast[In]().ne(a)` answered False and waved it through,
    producing whatever `fptosi` gave. `pyarrow.compute.cast` raises.

    **`safe=False` does not null the row**, unlike the string parsers — it
    keeps the hardware's answer, which is 0 for a NaN on arm64. Asserted as
    "does not raise" rather than as that value, since it is a lowering detail
    and not a promise.
    """
    var a: DynArray = array([nan[float64.native](), 1.0], float64)
    with assert_raises():
        _ = cast(a, int64, safe=True)
    var lax = cast(a, int64, safe=False)
    assert_equal(lax.null_count(), 0)
    assert_equal(len(lax), 2)


def test_string_to_int_parse_error_safe_raises() raises:
    var a: DynArray = array(["1", "oops", "3"])
    with assert_raises():
        _ = cast(a, int32, safe=True)


def test_string_to_int_out_of_range_raises() raises:
    with assert_raises():
        _ = cast(array(["128"]), int8, safe=True)
    with assert_raises():
        _ = cast(array(["-1"]), uint8, safe=True)
    with assert_raises():
        _ = cast(array(["300"]), uint8, safe=True)


def test_string_to_int_range_edges() raises:
    """Each target's own bounds parse; one past them raises under safe."""
    var r = cast(array(["-128", "127", "-0", "007", None]), int8)
    assert_true(r.as_int8() == array([-128, 127, 0, 7, None], int8))
    var limits = cast(
        array(["-9223372036854775808", "9223372036854775807"]), int64
    )
    assert_true(limits.as_int64().unsafe_get(0) == Int64.MIN)
    assert_true(limits.as_int64().unsafe_get(1) == Int64.MAX)
    var u = cast(array(["0", "255"]), uint8)
    assert_true(u.as_uint8() == array([0, 255], uint8))
    var bad8: List[String] = ["128", "-129", "12a"]
    for text in bad8:
        with assert_raises():
            _ = cast(array([text]), int8, safe=True)
    with assert_raises():
        _ = cast(array(["99999999999999999999999"]), int64, safe=True)
    with assert_raises():
        _ = cast(array(["-1"]), uint64, safe=True)


def test_string_to_int_parse_error_unsafe_nulls() raises:
    var a: DynArray = array(["1", "oops", "3"])
    var r = cast(a, int32, safe=False)
    assert_equal(r.null_count(), 1)
    assert_true(not r.is_valid(1))
    assert_true(r.is_valid(0))


def test_string_roundtrip_nulls() raises:
    var a: DynArray = array([1, None, 3], int32)
    var s = cast(a, string)
    assert_equal(s.null_count(), 1)
    var back = cast(s, int32)
    assert_true(back.as_int32() == array([1, None, 3], int32))


def test_bool_to_string() raises:
    var a: DynArray = array([True, False, True])
    var r = cast(a, string)
    assert_equal(String(r.as_string()[0]), "true")
    assert_equal(String(r.as_string()[1]), "false")


def test_string_to_bool() raises:
    var a: DynArray = array(["true", "False", "1", "0"])
    var r = cast(a, bool_)
    assert_true(r.as_bool() == array([True, False, True, False]))


# ---------------------------------------------------------------------------
# Null casts
# ---------------------------------------------------------------------------


def test_null_to_numeric() raises:
    var a: DynArray = NullArray(length=3)
    var r = cast(a, int64)
    assert_true(r.dtype() == int64)
    assert_equal(len(r), 3)
    assert_equal(r.null_count(), 3)


def test_null_to_string() raises:
    var a: DynArray = NullArray(length=2)
    var r = cast(a, string)
    assert_true(r.dtype() == string)
    assert_equal(r.null_count(), 2)


# ---------------------------------------------------------------------------
# Binary-like family (utf8 / large_utf8 / binary / large_binary / fsb)
# ---------------------------------------------------------------------------


def test_string_to_binary_roundtrip() raises:
    var s: DynArray = array(["ab", "cd", "e"])
    var b = cast(s, binary)  # relabel, same 32-bit offsets
    assert_true(b.dtype() == binary)
    var back = cast(b, string)  # validates UTF-8, relabel
    assert_true(back.as_string() == array(["ab", "cd", "e"]))


def test_string_to_large_string_widen_narrow() raises:
    var s: DynArray = array(["ab", "cd", "e"])
    var ls = cast(s, large_string)  # 32 → 64-bit offsets
    assert_true(ls.dtype() == large_string)
    var back = cast(ls, string)  # 64 → 32-bit offsets
    assert_true(back.as_string() == array(["ab", "cd", "e"]))


def test_binary_to_large_binary() raises:
    var b = cast(array(["xy", "z"]), binary)
    var lb = cast(b, large_binary)
    assert_true(lb.dtype() == large_binary)
    assert_true(cast(lb, string).as_string() == array(["xy", "z"]))


def test_large_string_to_numeric() raises:
    var ls = cast(array(["1", "22", "-3"]), large_string)
    var r = cast(ls, int32)
    assert_true(r.as_int32() == array([1, 22, -3], int32))


def test_large_string_to_bool() raises:
    var ls = cast(array(["true", "0", "False"]), large_string)
    assert_true(cast(ls, bool_).as_bool() == array([True, False, False]))


def test_numeric_to_large_string() raises:
    var ls = cast(array([1, 2, 3], int32), large_string)
    assert_true(ls.dtype() == large_string)
    assert_true(cast(ls, int32).as_int32() == array([1, 2, 3], int32))


def test_bool_to_large_string() raises:
    var ls = cast(array([True, False]), large_string)
    assert_equal(String(ls.as_large_string()[0]), "true")
    assert_equal(String(ls.as_large_string()[1]), "false")


def test_fixed_size_binary_roundtrip() raises:
    var s: DynArray = array(["ab", "cd", "ef"])
    var fsb = cast(s, fixed_size_binary_(2))
    assert_true(fsb.dtype() == fixed_size_binary_(2).to_dyn())
    var back = cast(fsb, string)
    assert_true(back.as_string() == array(["ab", "cd", "ef"]))


def test_binary_to_fixed_size_binary_width_mismatch_raises() raises:
    var b = cast(array(["ab", "c"]), binary)  # "c" is 1 byte, target width 2
    with assert_raises():
        _ = cast(b, fixed_size_binary_(2))


def test_binary_to_string_invalid_utf8_raises() raises:
    # A raw 0xFF byte is not valid UTF-8; safe mode must reject it.
    var raw = List[UInt8]()
    raw.append(0xFF)
    var fb = FixedSizeBinaryBuilder(1)
    fb.append(Span(raw))
    var bad: DynArray = cast(fb.finish(), binary)
    with assert_raises():
        _ = cast(bad, string, safe=True)


# ---------------------------------------------------------------------------
# Decimal ↔ numeric / decimal (constructed by casting from integers)
# ---------------------------------------------------------------------------


def test_int_to_decimal_roundtrip() raises:
    var d = cast(array([1, 2, 3], int64), decimal128(10, 2))  # × 100
    assert_true(d.dtype() == decimal128(10, 2).to_dyn())
    assert_true(cast(d, int64).as_int64() == array([1, 2, 3], int64))


def test_decimal_to_float() raises:
    var d = cast(array([3], int64), decimal128(10, 2))  # 3 → 300 at scale 2
    assert_true(cast(d, float64).as_float64() == array([3.0], float64))


def test_float_to_decimal_roundtrip() raises:
    var f: DynArray = array([1.5, 2.25, -0.5], float64)
    var d = cast(f, decimal128(10, 2))  # round(×100): 150, 225, -50
    assert_true(
        cast(d, float64).as_float64() == array([1.5, 2.25, -0.5], float64)
    )


def test_float_to_decimal_is_exact() raises:
    """An integral double past 2^53 / 10^scale stays integral, and a tie
    rounds to even."""
    var d = cast(
        array([14411518807587.0, 0.125, -0.375], float64), decimal128(20, 4)
    )
    assert_true(
        cast(d, string).as_string()
        == array(["14411518807587.0000", "0.1250", "-0.3750"])
    )
    var tie = cast(array([0.125, 0.375], float64), decimal128(10, 2))
    assert_true(cast(tie, string).as_string() == array(["0.12", "0.38"]))
    with assert_raises():
        _ = cast(array([1000.0], float64), decimal128(5, 2), safe=True)


def test_float_to_decimal_edges() raises:
    """Ties to even on both signs, every digit of a double at a large scale,
    the precision limit, subnormals, and non-finite values."""
    var ties = cast(array([0.5, 1.5, 2.5, -0.5, -2.5], float64), decimal128(10))
    assert_true(
        cast(ties, string).as_string() == array(["0", "2", "2", "0", "-2"])
    )
    var wide = cast(array([0.1], float64), decimal256(76, 50))
    assert_true(
        cast(wide, string).as_string()
        == array(["0.10000000000000000555111512312578270211815834045410"])
    )
    var big = cast(array([1e38], float64), decimal128(38, 0))
    assert_true(
        cast(big, string).as_string()
        == array(["99999999999999997748809823456034029568"])
    )
    var tiny = cast(array([5e-324, -0.0], float64), decimal128(10, 4))
    assert_true(cast(tiny, string).as_string() == array(["0.0000", "0.0000"]))
    var edge = cast(array([99999.995, 0.001], float64), decimal128(7, 2))
    assert_true(cast(edge, string).as_string() == array(["99999.99", "0.00"]))
    var below_one = cast(array([0.001], float64), decimal128(5, 7))
    assert_true(cast(below_one, string).as_string() == array(["0.0010000"]))
    with assert_raises():
        _ = cast(array([0.05], float64), decimal128(5, 7), safe=True)
    var odd: List[Float64] = [nan[DType.float64](), inf[DType.float64]()]
    for x in odd:
        with assert_raises():
            _ = cast(array([x], float64), decimal128(10, 2), safe=True)
    var lax = cast(
        array([odd[0], odd[1], 1.0], float64), decimal128(10, 2), safe=False
    )
    assert_true(
        cast(lax, string).as_string() == array(["0.00", "0.00", "1.00"])
    )


def test_decimal_rescale_widen() raises:
    var d = cast(array([1, 2], int64), decimal64(10, 1))  # scale 1: 10, 20
    var d2 = cast(d, decimal128(20, 3))  # scale 1 → 3: × 100
    assert_true(d2.dtype() == decimal128(20, 3).to_dyn())
    assert_true(cast(d2, int64).as_int64() == array([1, 2], int64))


def test_decimal_nulls_preserved() raises:
    var d = cast(array([1, None, 3], int64), decimal128(10, 2))
    assert_equal(d.null_count(), 1)
    assert_true(not d.is_valid(1))


# ---------------------------------------------------------------------------
# Nested (list / struct) + dictionary decode
# ---------------------------------------------------------------------------


def test_list_to_list_cast() raises:
    var ib = Int32Builder()
    ib.append(1)
    ib.append(2)
    ib.append(3)
    var lb = ListBuilder(ib^)
    lb.append_valid()  # one list [1, 2, 3]
    var lst: DynArray = lb.finish()
    var casted = cast(lst, list_(int64))  # list<int32> → list<int64>
    assert_true(casted.dtype() == list_(int64).to_dyn())
    ref child = casted.as_list().values()
    assert_true(child.as_int64() == array([1, 2, 3], int64))


def test_struct_to_struct_cast() raises:
    var sb = StructBuilder([field("a", int32), field("b", int32)], capacity=2)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(1).as_int32().append(10)
    sb.field_builder(1).as_int32().append(20)
    sb.append_valid()
    sb.append_valid()
    var st: DynArray = sb.finish()
    var casted = cast(st, struct_([field("a", int64), field("b", float64)]))
    assert_true(casted.dtype().is_struct())
    assert_true(casted.as_struct().field(0).as_int64() == array([1, 2], int64))
    assert_true(
        casted.as_struct().field(1).as_float64() == array([10.0, 20.0], float64)
    )


def test_dictionary_decode() raises:
    var values: DynArray = array(["a", "b", "c"])
    var indices: DynArray = array([0, 2, 1, 0], int32)
    var d: DynArray = DictionaryArray(
        dtype=dictionary(int32, string).to_dyn(),
        length=4,
        nulls=0,
        offset=0,
        indices=indices^,
        values=values^,
    )
    assert_true(cast(d, string).as_string() == array(["a", "c", "b", "a"]))


def test_dictionary_decode_then_cast() raises:
    var values: DynArray = array([10, 20, 30], int32)
    var indices: DynArray = array([0, 1, 2, 1], int32)
    var d: DynArray = DictionaryArray(
        dtype=dictionary(int32, int32).to_dyn(),
        length=4,
        nulls=0,
        offset=0,
        indices=indices^,
        values=values^,
    )
    assert_true(cast(d, int64).as_int64() == array([10, 20, 30, 20], int64))


def test_cast_map_casts_the_entry_values() raises:
    """V0. `cast` had no map arm, so `map<string, int64> -> map<string, int32>`
    raised "unsupported cast".

    A map needs no kernel of its own: physically it is a list whose single
    child is the non-nullable `entries` struct, so `ListCastKernel` casts that struct
    and `StructCastKernel` casts the fields. Only the *target child type* had to be
    read differently — from `entries_field()` rather than `value_type()`.
    """
    var b = MapBuilder(map_(DynType(string), DynType(int64)))
    var entries_any = b.entries()
    ref entries = entries_any.as_struct()
    var keys_any = entries.field_builder(0)
    var values_any = entries.field_builder(1)
    ref keys = keys_any.as_string()
    ref values = values_any.as_int64()
    keys.append("a")
    values.append(Int64(7))
    entries.append_valid()
    b.append_valid()
    var m = b.finish()

    var target = map_(DynType(string), DynType(int32)).to_dyn()
    var out = cast(m^.to_dyn(), target)

    assert_true(out.dtype() == target)
    assert_equal(len(out), 1)
    ref got = out.as_type[MapArray]()
    var got_entries = got.values().copy()
    assert_equal(
        got_entries.as_struct().field(1).as_int32()[0].value(), Int32(7)
    )


# ---------------------------------------------------------------------------
# S4 — `safe` must reach every kernel `cast()` delegates to.
#
# `cast()`'s ladder called `DecimalCastKernel.dispatch(array, to)` and
# `TemporalCastKernel.dispatch(array, to, ctx)`, so the caller's `safe` flag was
# dropped on both arms and neither kernel could honour it. Arrow C++ raises in
# each case below (`CastOptions::Safe()` clears `allow_decimal_truncate`,
# `allow_time_truncate` and `allow_time_overflow`).
# ---------------------------------------------------------------------------


def test_cast_float_to_decimal_overflow_raises_under_safe() raises:
    """1e38 scaled by 10^2 is 1e40, far outside int128."""
    var f: DynArray = array([1.0e38], float64)
    with assert_raises():
        _ = cast(f, decimal128(38, 2), safe=True)


def test_cast_decimal_upscale_overflow_raises_under_safe() raises:
    """12345 at scale 6 is 12_345_000_000, past decimal32's int32 backing."""
    var i: DynArray = array([12345], int64)
    with assert_raises():
        _ = cast(i, decimal32(9, 6), safe=True)


def test_cast_decimal_downscale_truncation_raises_under_safe() raises:
    """1.234 at scale 3 cannot be held at scale 1 — the `34` is discarded."""
    var f: DynArray = array([1.234], float64)
    var d3 = cast(f, decimal64(18, 3), safe=False)
    with assert_raises():
        _ = cast(d3, decimal64(18, 1), safe=True)


def test_cast_decimal_downscale_exact_passes_under_safe() raises:
    """The truncation check must not fire when the rescale is lossless.

    Read back through `float64` rather than `int64` — casting a decimal to an
    integer rescales it to scale 0, so `12 @ scale 1` would read as `1`.
    """
    var f: DynArray = array([1.2], float64)
    var d3 = cast(f, decimal64(18, 3), safe=False)
    var d1 = cast(d3, decimal64(18, 1), safe=True)
    assert_true(
        cast(d1, float64, safe=False).as_float64() == array([1.2], float64)
    )


def test_cast_timestamp_downscale_raises_under_safe() raises:
    """1500 ms is not a whole number of seconds."""
    var i: DynArray = array([1500], int64)
    var ts_ms = cast(i, timestamp(millisecond))
    with assert_raises():
        _ = cast(ts_ms, timestamp(second), safe=True)


def test_cast_timestamp_upscale_overflow_raises_under_safe() raises:
    """1e10 seconds is 1e19 nanoseconds, past int64's 9.22e18."""
    var i: DynArray = array([10_000_000_000], int64)
    var ts_s = cast(i, timestamp(second))
    with assert_raises():
        _ = cast(ts_s, timestamp(nanosecond), safe=True)


def test_cast_timestamp_downscale_truncates_under_unsafe() raises:
    """`safe=False` keeps the old truncating behaviour."""
    var i: DynArray = array([1500, 2500], int64)
    var ts_ms = cast(i, timestamp(millisecond))
    var ts_s = cast(ts_ms, timestamp(second), safe=False)
    assert_true(
        cast(ts_s, int64, safe=False).as_int64() == array([1, 2], int64)
    )


def test_cast_timestamp_downscale_exact_passes_under_safe() raises:
    """The truncation check must not fire on a whole number of seconds."""
    var i: DynArray = array([1000, 2000], int64)
    var ts_ms = cast(i, timestamp(millisecond))
    var ts_s = cast(ts_ms, timestamp(second), safe=True)
    assert_true(cast(ts_s, int64).as_int64() == array([1, 2], int64))


# ---------------------------------------------------------------------------
# binary → string UTF-8 validation.
#
# `_check_utf8` puts two whole-buffer fast paths in front of the per-element
# loop (all-ASCII; valid-window-with-block-skipping + element starts on
# character boundaries). Both are sufficient conditions that fall through to
# the loop, so these tests pin the accept/reject decision rather than the route
# taken to it — a fast path that stops rejecting bad input is the failure mode
# they exist to catch.
#
# Raw byte payloads go through `FixedSizeBinaryBuilder` + `cast(..., binary)`,
# which is how the older invalid-UTF-8 test above builds them.
# ---------------------------------------------------------------------------


def _binary_of_width(
    var cells: List[List[UInt8]], width: Int
) raises -> DynArray:
    """A `binary` array whose elements are the given fixed-width byte cells."""
    var fb = FixedSizeBinaryBuilder(width)
    for cell in cells:
        fb.append(Span(cell))
    return cast(fb.finish(), binary)


def test_binary_to_string_invalid_utf8_raises_past_fast_path() raises:
    # A lone 0xFF behind enough ASCII to clear the SIMD block and chunk sizes:
    # neither the all-ASCII probe nor the block-skipping window check may let
    # it through.
    var cells = List[List[UInt8]]()
    for i in range(600):
        var ok = List[UInt8]()
        ok.append(UInt8(97 + (i % 26)))
        cells.append(ok^)
    var bad = List[UInt8]()
    bad.append(0xFF)
    cells.append(bad^)

    var arr = _binary_of_width(cells^, 1)
    with assert_raises():
        _ = cast(arr, string)  # safe=True is the default


def test_binary_to_string_split_character_raises() raises:
    """The exact way a whole-buffer validator goes wrong.

    "é" is 0xC3 0xA9. Split across two adjacent elements the *concatenation* is
    valid UTF-8 while each element on its own is not, so a check that only
    looks at the byte window would accept it. The element-start boundary scan
    is what makes this still raise."""
    var lead = List[UInt8]()
    lead.append(0xC3)
    var trail = List[UInt8]()
    trail.append(0xA9)
    var cells = List[List[UInt8]]()
    cells.append(lead^)
    cells.append(trail^)

    var arr = _binary_of_width(cells^, 1)
    with assert_raises():
        _ = cast(arr, string)


def test_binary_to_string_valid_multibyte_passes() raises:
    """Non-ASCII input misses the all-ASCII path and must still be accepted."""
    var src = cast(
        array(["Здравствуйте", "ünïcødé", "日本語", "ascii", ""]), binary
    )
    var out = cast(src, string)
    assert_true(out.dtype() == string)
    assert_equal(len(out), 5)
    assert_equal(String(out.as_string()[0]), "Здравствуйте")
    assert_equal(String(out.as_string()[2]), "日本語")


def test_binary_to_string_ascii_fast_path_matches_loop() raises:
    var src = cast(array(["a", "bc", "", "def"]), binary)
    var out = cast(src, string)
    assert_equal(len(out), 4)
    assert_equal(String(out.as_string()[3]), "def")


def test_binary_to_string_null_slot_bytes_are_not_validated() raises:
    """A null slot may hold arbitrary bytes. The whole-window check fails on
    them, and the fall-through to the per-element loop — which skips nulls — is
    what keeps that from becoming a false rejection."""
    var cells = List[List[UInt8]]()
    var a = List[UInt8]()
    a.append(0x61)  # "a"
    cells.append(a^)
    var junk = List[UInt8]()
    junk.append(0xFF)  # stays in `values`, but the slot is null
    cells.append(junk^)
    var c = List[UInt8]()
    c.append(0x62)  # "b"
    cells.append(c^)

    ref built = _binary_of_width(cells^, 1).as_binary()

    var bm = Bitmap.alloc_zeroed(3)
    bm.set(0)
    bm.set(2)
    var with_null = BinaryArray(
        length=3,
        nulls=1,
        offset=0,
        bitmap=bm^.to_immutable(length=3),
        offsets=built.offsets.copy(),
        values=built.values.copy(),
    )

    var out = BinaryLikeCastKernel.apply[BinaryType, StringType, True](
        with_null
    )
    assert_equal(len(out), 3)
    assert_true(out.is_null(1))
    assert_equal(String(out[0]), "a")


def test_binary_to_string_slice_validates_only_its_window() raises:
    """Validation must read the sliced window, not the whole values buffer:
    bad bytes outside the slice are none of this cast's business, and bad bytes
    inside it must still raise."""
    var cells = List[List[UInt8]]()
    var g1 = List[UInt8]()
    g1.append(0x61)
    cells.append(g1^)
    var g2 = List[UInt8]()
    g2.append(0x62)
    cells.append(g2^)
    var bad = List[UInt8]()
    bad.append(0xFF)
    cells.append(bad^)

    var src = _binary_of_width(cells^, 1)

    assert_true(cast(src.slice(0, 2), string).dtype() == string)
    with assert_raises():
        _ = cast(src.slice(1, 2), string)


# ---------------------------------------------------------------------------
# Casting a slice — the result is offset 0, so its validity must be rebased
# ---------------------------------------------------------------------------


def test_numeric_cast_of_a_slice_keeps_nulls_in_place() raises:
    var sl = array([1, None, 3, 4], int64).slice(1, 3)
    var out = cast(sl.copy(), int32)
    assert_equal(out.null_count(), 1)
    assert_true(not out.is_valid(0))
    assert_true(out.is_valid(1) and out.is_valid(2))


def test_decimal_cast_of_a_slice_keeps_nulls_in_place() raises:
    var sl = array([1, None, 3, 4], int64).slice(1, 3)
    var d = cast(sl.copy(), decimal128(20, 0))
    assert_equal(d.null_count(), 1)
    assert_true(not d.is_valid(0))
    assert_true(d.is_valid(1) and d.is_valid(2))


def test_temporal_cast_of_a_slice_keeps_nulls_in_place() raises:
    var ts = cast(array([1, None, 3, 4], int64), timestamp(second))
    var out = cast(ts.slice(1, 3), timestamp(millisecond))
    assert_equal(out.null_count(), 1)
    assert_true(not out.is_valid(0))
    assert_true(out.is_valid(1) and out.is_valid(2))


def test_num_to_bool_cast_of_a_slice_keeps_nulls_in_place() raises:
    var sl = array([1, None, 3, 4], int64).slice(1, 3)
    var out = cast(sl.copy(), bool_)
    assert_equal(out.null_count(), 1)
    assert_true(not out.is_valid(0))
    assert_true(out.is_valid(1) and out.is_valid(2))


def test_bool_to_num_cast_of_a_slice_keeps_nulls_in_place() raises:
    var b = cast(array([1, None, 3, 4], int64), bool_)
    var out = cast(b.slice(1, 3), int32)
    assert_equal(out.null_count(), 1)
    assert_true(not out.is_valid(0))
    assert_true(out.is_valid(1) and out.is_valid(2))


# ---------------------------------------------------------------------------
# Decimal rescale bounds: overflow, truncation direction, declared precision
# ---------------------------------------------------------------------------


def test_decimal_upscale_negative_boundary_raises() raises:
    # Int32.MIN // 1000 floors to -2147484, whose product is below Int32.MIN.
    # A floored lower bound lets it through and wraps.
    with assert_raises():
        _ = cast(array([-2147484], int64), decimal32(9, 3), True)


def test_decimal_upscale_positive_boundary_raises() raises:
    with assert_raises():
        _ = cast(array([2147484], int64), decimal32(9, 3), True)


def test_decimal_to_int_truncates_toward_zero() raises:
    # unscaled -15 at scale 1 (= -1.5); Arrow truncates, so -1 not -2.
    var d = cast(array([-1.5], float64), decimal128(10, 1))
    assert_true(cast(d, int64, False).as_int64() == array([-1], int64))


def test_decimal_to_int_positive_is_unaffected() raises:
    var d = cast(array([1.5], float64), decimal128(10, 1))
    assert_true(cast(d, int64, False).as_int64() == array([1], int64))


def test_decimal_downscale_truncates_toward_zero() raises:
    # -1.234 at scale 3 narrowed to scale 1 is -1.2, not -1.3.
    var d = cast(array([-1.234], float64), decimal64(18, 3))
    var narrowed = cast(d, decimal64(18, 1), False)
    assert_true(
        cast(narrowed, float64, False).as_float64() == array([-1.2], float64)
    )


def test_decimal_downscale_positive_is_unaffected() raises:
    var d = cast(array([1.234], float64), decimal64(18, 3))
    var narrowed = cast(d, decimal64(18, 1), False)
    assert_true(
        cast(narrowed, float64, False).as_float64() == array([1.2], float64)
    )


def test_decimal_cast_past_target_precision_raises() raises:
    # 7 digits into a declared precision of 5. delta == 0, so every
    # storage-width check passes and only the digit bound can catch it.
    var wide = cast(array([1_000_000], int64), decimal128(38, 0))
    with assert_raises():
        _ = cast(wide, decimal128(5, 0), True)


def test_decimal_cast_within_target_precision_passes() raises:
    var wide = cast(array([12_345], int64), decimal128(38, 0))
    assert_true(cast(wide, decimal128(5, 0), True).null_count() == 0)


def test_scale_zero_decimal_to_int_roundtrips() raises:
    var d = cast(array([7, -7, 0], int64), decimal128(10, 0))
    assert_true(cast(d, int64).as_int64() == array([7, -7, 0], int64))


def test_scale_zero_decimal_to_int_overflow_raises() raises:
    var d = cast(array([10_000_000_000], int64), decimal128(20, 0))
    with assert_raises():
        _ = cast(d, int32, True)


from ...arrays import BinaryViewArray, StringArray, StringViewArray
from ...builders import (
    StringViewBuilder,
    BinaryViewBuilder,
    StringBuilder,
    BinaryBuilder,
)
from std.testing import assert_false
from ...dtypes import string_view, binary_view


# ---------------------------------------------------------------------------
# string_view / binary_view casts
# ---------------------------------------------------------------------------


def test_cast_string_to_string_view_zero_copy() raises:
    var b = StringBuilder(4)
    b.append("skip")
    b.append("inline")
    b.append_null()
    b.append("a value longer than twelve")
    var src = b.finish().slice(1)
    var out = cast(src.copy().to_dyn(), string_view)
    ref v = out.as_string_view()
    assert_equal(len(v), 3)
    assert_equal(v.null_count(), 1)
    assert_equal(v[0].value(), "inline")
    assert_false(v.is_valid(1))
    assert_equal(v[2].value(), "a value longer than twelve")
    # The one data buffer is the source's values buffer, shared.
    assert_equal(len(v.buffers), 1)
    assert_true(v.buffers[0] == src.values)
    v.validate()


def test_cast_short_strings_to_string_view_hold_no_buffer() raises:
    """Every value fits in its view, so the source's values buffer is not
    taken: the result keeps nothing of it alive."""
    var src: StringArray = ["a", "twelve bytes", ""]
    var out = cast(src^.to_dyn(), string_view)
    ref v = out.as_string_view()
    assert_equal(len(v.buffers), 0)
    assert_equal(v[1].value(), "twelve bytes")
    assert_equal(v[2].value(), "")


def test_cast_string_view_to_string() raises:
    var src: StringViewArray = ["a", "a value longer than twelve", ""]
    var out = cast(src^.to_dyn(), string)
    ref s = out.as_string()
    assert_equal(len(s), 3)
    assert_equal(s[1].value(), "a value longer than twelve")
    assert_equal(s[2].value(), "")


def test_cast_string_view_roundtrip_large_string() raises:
    var src: StringViewArray = ["x", "a value longer than twelve"]
    var back = cast(cast(src.copy().to_dyn(), large_string), string_view)
    assert_true(back.as_string_view() == src)


def test_cast_binary_view_to_string_view_validates_utf8() raises:
    var b = BinaryViewBuilder()
    b.append("valid bytes")
    var ok = cast(b.finish().to_dyn(), string_view)
    assert_equal(ok.as_string_view()[0].value(), "valid bytes")
    var bad = BinaryBuilder(1)
    bad.append(StringSlice(unsafe_from_utf8=Span[Byte]([0xFF, 0xFE])))
    with assert_raises(contains="UTF-8"):
        _ = cast(bad.finish().to_dyn(), string_view)


def test_cast_string_view_numeric() raises:
    var src: StringViewArray = ["1", "22", "333"]
    var ints = cast(src^.to_dyn(), int64)
    assert_equal(ints.as_int64()[2].value(), 333)
    var back = cast(ints, string_view)
    assert_equal(back.as_string_view()[1].value(), "22")
