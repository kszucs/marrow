from std.testing import assert_equal, assert_true, assert_raises

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
    Int64Type,
    BinaryType,
    StringType,
)
from ...kernels.cast import (
    cast,
    BinaryLikeCast,
    FixedSizeBinaryCast,
    NumericCast,
)


# ---------------------------------------------------------------------------
# Typed numeric casts
# ---------------------------------------------------------------------------


def test_int_to_float_widen() raises:
    var a = array([1, 2, 3], int32)
    var r = NumericCast.apply[Int32Type, Float64Type](a)
    assert_true(r == array([1.0, 2.0, 3.0], float64))


def test_float_to_int_truncates_toward_zero() raises:
    var a = array([1.9, -1.9, 2.5, -2.5], float64)
    var r = NumericCast.apply[Float64Type, Int32Type, False](a)
    assert_true(r == array([1, -1, 2, -2], int32))


def test_negative_narrowing_wraps() raises:
    var a = array([-1, 300, 44], int32)
    var r = NumericCast.apply[Int32Type, UInt8Type, False](a)
    assert_equal(Int(r.unsafe_get(0)), 255)
    assert_equal(Int(r.unsafe_get(1)), 44)  # 300 & 0xFF
    assert_equal(Int(r.unsafe_get(2)), 44)


def test_int16_wraps_to_int8() raises:
    var a = array([300], int16)
    var r = NumericCast.apply[Int16Type, Int8Type, False](a)
    assert_equal(Int(r.unsafe_get(0)), 44)  # 300 & 0xFF = 44


def test_float16_roundtrip() raises:
    var a = array([1.0, 2.5, -3.25], float32)
    var half = NumericCast.apply[Float32Type, Float16Type](a)
    var back = NumericCast.apply[Float16Type, Float32Type](half)
    assert_true(back == a)


# ---------------------------------------------------------------------------
# Null preservation
# ---------------------------------------------------------------------------


def test_nulls_preserved() raises:
    var a = array([1, None, 3], int32)
    var r = NumericCast.apply[Int32Type, Float64Type](a)
    assert_equal(r.nulls, 1)
    assert_true(r.is_valid(0))
    assert_true(not r.is_valid(1))
    assert_true(r.is_valid(2))


# ---------------------------------------------------------------------------
# Safe mode
# ---------------------------------------------------------------------------


def test_safe_lossless_ok() raises:
    var a = array([1, 2, 3], int32)
    var r = NumericCast.apply[Int32Type, Int64Type, True](a)
    assert_true(r == array([1, 2, 3], int64))


def test_safe_overflow_raises() raises:
    var a = array([300], int32)
    with assert_raises():
        _ = NumericCast.apply[Int32Type, Int8Type, True](a)


def test_safe_float_truncation_raises() raises:
    var a = array([3.9], float64)
    with assert_raises():
        _ = NumericCast.apply[Float64Type, Int32Type, True](a)


def test_safe_negative_to_unsigned_raises() raises:
    var a = array([-1], int32)
    with assert_raises():
        _ = NumericCast.apply[Int32Type, UInt8Type, True](a)


def test_safe_skips_null_lanes() raises:
    # A null lane holds arbitrary data that need not be representable; safe mode
    # must not raise on it.
    var a = array([1, None, 2], int32)
    var r = NumericCast.apply[Int32Type, Int8Type, True](a)
    assert_equal(r.nulls, 1)


def test_unsafe_overflow_ok() raises:
    var a = array([300], int32)
    var r = NumericCast.apply[Int32Type, Int8Type, False](a)
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
    # `safe` through to `TemporalCast` — the suite encoded the defect.
    var i: DynArray = array([1500, 2500], int64)
    var ts_ms = cast(i, timestamp(millisecond))
    var ts_s = cast(ts_ms, timestamp(second), safe=False)  # // 1000 truncates
    assert_true(cast(ts_s, int64).as_int64() == array([1, 2], int64))


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


def test_string_to_int_parse_error_safe_raises() raises:
    var a: DynArray = array(["1", "oops", "3"])
    with assert_raises():
        _ = cast(a, int32, safe=True)


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
    child is the non-nullable `entries` struct, so `ListCast` casts that struct
    and `StructCast` casts the fields. Only the *target child type* had to be
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
# `cast()`'s ladder called `DecimalCast.dispatch(array, to)` and
# `TemporalCast.dispatch(array, to, ctx)`, so the caller's `safe` flag was
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

    var out = BinaryLikeCast.apply[BinaryType, StringType, True](with_null)
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
