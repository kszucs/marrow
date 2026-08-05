from std.testing import assert_equal, assert_true, assert_false
from .. import dtypes as dt
from ..dtypes import *


def test_bool_type() raises:
    assert_true(dt.bool_ == dt.bool_)
    assert_false(DynType(dt.bool_) == DynType(dt.int64))

    var t = DynType(dt.bool_)
    assert_true(t.is_bool())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    # marrow deliberately diverges from PyArrow and Arrow C++ here, both of
    # which call bool primitive. marrow has something neither has: a
    # `PrimitiveType` *trait* that generic code dispatches on, which `BoolType`
    # cannot conform to because bool is bit-packed. `is_primitive()` exists to
    # guard `variant_dispatch[PrimitiveType]`, so a True here is a trap — it
    # aborted `byte_width()`. Do not "fix" this back.
    assert_false(t.is_primitive())
    assert_false(t.is_string())
    assert_equal(String(t), "bool")


def test_is_integer() raises:
    assert_true(DynType(dt.int8).is_integer())
    assert_true(DynType(dt.int16).is_integer())
    assert_true(DynType(dt.int32).is_integer())
    assert_true(DynType(dt.int64).is_integer())
    assert_true(DynType(dt.uint8).is_integer())
    assert_true(DynType(dt.uint16).is_integer())
    assert_true(DynType(dt.uint32).is_integer())
    assert_true(DynType(dt.uint64).is_integer())
    assert_false(DynType(dt.bool_).is_integer())
    assert_false(DynType(dt.float32).is_integer())
    assert_false(DynType(dt.float64).is_integer())
    assert_false(DynType(dt.list_(dt.int64)).is_integer())


def test_is_signed_integer() raises:
    assert_true(DynType(dt.int8).is_signed_integer())
    assert_true(DynType(dt.int16).is_signed_integer())
    assert_true(DynType(dt.int32).is_signed_integer())
    assert_true(DynType(dt.int64).is_signed_integer())
    assert_false(DynType(dt.uint8).is_signed_integer())
    assert_false(DynType(dt.uint16).is_signed_integer())
    assert_false(DynType(dt.uint32).is_signed_integer())
    assert_false(DynType(dt.uint64).is_signed_integer())
    assert_false(DynType(dt.bool_).is_signed_integer())
    assert_false(DynType(dt.float32).is_signed_integer())
    assert_false(DynType(dt.float64).is_signed_integer())


def test_is_unsigned_integer() raises:
    assert_false(DynType(dt.int8).is_unsigned_integer())
    assert_false(DynType(dt.int16).is_unsigned_integer())
    assert_false(DynType(dt.int32).is_unsigned_integer())
    assert_false(DynType(dt.int64).is_unsigned_integer())
    assert_true(DynType(dt.uint8).is_unsigned_integer())
    assert_true(DynType(dt.uint16).is_unsigned_integer())
    assert_true(DynType(dt.uint32).is_unsigned_integer())
    assert_true(DynType(dt.uint64).is_unsigned_integer())
    assert_false(DynType(dt.bool_).is_unsigned_integer())
    assert_false(DynType(dt.float32).is_unsigned_integer())
    assert_false(DynType(dt.float64).is_unsigned_integer())


def test_is_floating_point() raises:
    assert_false(DynType(dt.int8).is_floating_point())
    assert_false(DynType(dt.int16).is_floating_point())
    assert_false(DynType(dt.int32).is_floating_point())
    assert_false(DynType(dt.int64).is_floating_point())
    assert_false(DynType(dt.uint8).is_floating_point())
    assert_false(DynType(dt.uint16).is_floating_point())
    assert_false(DynType(dt.uint32).is_floating_point())
    assert_false(DynType(dt.uint64).is_floating_point())
    assert_false(DynType(dt.bool_).is_floating_point())
    assert_true(DynType(dt.float32).is_floating_point())
    assert_true(DynType(dt.float64).is_floating_point())


def test_dtypes_bit_width() raises:
    assert_equal(dt.int8.bit_width(), 8)
    assert_equal(dt.int16.bit_width(), 16)
    assert_equal(dt.int32.bit_width(), 32)
    assert_equal(dt.int64.bit_width(), 64)
    assert_equal(dt.uint8.bit_width(), 8)
    assert_equal(dt.uint16.bit_width(), 16)
    assert_equal(dt.uint32.bit_width(), 32)
    assert_equal(dt.uint64.bit_width(), 64)
    assert_equal(dt.float32.bit_width(), 32)
    assert_equal(dt.float64.bit_width(), 64)


def test_null_type() raises:
    var t = DynType(NullType())
    assert_true(t.is_null())
    assert_false(t.is_bool())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_false(t.is_primitive())
    assert_false(t.is_string())
    assert_equal(String(t), "null")


def test_string_type() raises:
    var t = DynType(StringType())
    assert_true(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_bool())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_false(t.is_primitive())
    assert_equal(String(t), "string")


def test_byte_width() raises:
    assert_equal(DynType(Int8Type()).byte_width(), 1)
    assert_equal(DynType(Int16Type()).byte_width(), 2)
    assert_equal(DynType(Int32Type()).byte_width(), 4)
    assert_equal(DynType(Int64Type()).byte_width(), 8)
    assert_equal(DynType(UInt8Type()).byte_width(), 1)
    assert_equal(DynType(UInt16Type()).byte_width(), 2)
    assert_equal(DynType(UInt32Type()).byte_width(), 4)
    assert_equal(DynType(UInt64Type()).byte_width(), 8)
    assert_equal(DynType(Float16Type()).byte_width(), 2)
    assert_equal(DynType(Float32Type()).byte_width(), 4)
    assert_equal(DynType(Float64Type()).byte_width(), 8)


def test_byte_width_zero_for_non_fixed_width() raises:
    # 0, not an error: `byte_width` must be non-raising to *override*
    # `PrimitiveType.byte_width`, whose default would otherwise report
    # `size_of[DynType.native]` — the `bool` placeholder — as 1 for every dtype.
    # The cases above are what proves the override took; these pin the contract
    # the override costs, so a caller reading 0 as a real width is a test failure
    # and not a silent mis-sized buffer.
    assert_equal(DynType(StringType()).byte_width(), 0)
    assert_equal(DynType(BinaryType()).byte_width(), 0)
    assert_equal(DynType(NullType()).byte_width(), 0)


def test_bool_byte_width_is_zero_not_an_abort() raises:
    """Bool has no byte width — it is one bit.

    This used to abort the process: `is_primitive()` answered True, so
    `byte_width()` fell through to `variant_dispatch[PrimitiveType]`, which
    `BoolType` cannot enter. `c_data.mojo:1001` is one branch-ordering mistake
    away from reaching it on the C Data import path.
    """
    assert_equal(DynType(dt.bool_).byte_width(), 0)
    assert_false(DynType(dt.bool_).is_primitive())


def test_eq() raises:
    var a = DynType(UInt64Type())
    var b = DynType(UInt64Type())
    var c = DynType(Int32Type())
    assert_true(a == b)
    assert_false(a == c)
    assert_false(a != b)
    assert_true(a != c)
    assert_true(NullType() == NullType())
    assert_false(DynType(NullType()) == DynType(BoolType()))
    assert_true(Float32Type() == Float32Type())
    assert_false(DynType(Float32Type()) == DynType(Float64Type()))


def test_copy() raises:
    var original = DynType(Int64Type())
    var copied = DynType(copy=original)
    assert_true(original == copied)
    assert_equal(String(original), String(copied))


def test_native() raises:
    assert_equal(Int8Type.native, DType.int8)
    assert_equal(Int16Type.native, DType.int16)
    assert_equal(Int32Type.native, DType.int32)
    assert_equal(Int64Type.native, DType.int64)
    assert_equal(UInt8Type.native, DType.uint8)
    assert_equal(UInt16Type.native, DType.uint16)
    assert_equal(UInt32Type.native, DType.uint32)
    assert_equal(UInt64Type.native, DType.uint64)
    assert_equal(Float16Type.native, DType.float16)
    assert_equal(Float32Type.native, DType.float32)
    assert_equal(Float64Type.native, DType.float64)
    assert_equal(BoolType.native, DType.bool)


def test_singletons() raises:
    assert_equal(String(null), "null")
    assert_equal(String(bool_), "bool")
    assert_equal(String(int8), "int8")
    assert_equal(String(int16), "int16")
    assert_equal(String(int32), "int32")
    assert_equal(String(int64), "int64")
    assert_equal(String(uint8), "uint8")
    assert_equal(String(uint16), "uint16")
    assert_equal(String(uint32), "uint32")
    assert_equal(String(uint64), "uint64")
    assert_equal(String(float16), "float16")
    assert_equal(String(float32), "float32")
    assert_equal(String(float64), "float64")
    assert_equal(String(binary), "binary")
    assert_equal(String(string), "string")


def test_binary_type() raises:
    var t = DynType(BinaryType())
    assert_true(t.is_binary())
    assert_false(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_equal(String(t), "binary")


def test_list_type() raises:
    var t = list_(DynType(Int32Type()))
    var at: DynType = t.copy().to_dyn()
    assert_true(at.is_list())
    assert_false(at.is_fixed_size_list())
    assert_false(at.is_struct())
    assert_false(at.is_primitive())
    assert_equal(String(t), "list<int32>")

    var t2 = list_(DynType(Int32Type()))
    assert_true(t == t2)
    assert_false(t == list_(DynType(Float64Type())))

    var nested = list_(list_(DynType(Int64Type())))
    assert_equal(String(nested), "list<list<int64>>")

    var t3 = list_(int64)
    assert_equal(t3.value_type(), int64)


def test_fixed_size_list_type() raises:
    var t = fixed_size_list_(DynType(Float32Type()), 4)
    var at: DynType = t.copy().to_dyn()
    assert_true(at.is_fixed_size_list())
    assert_false(at.is_list())
    assert_false(at.is_struct())
    assert_equal(String(t), "fixed_size_list<item: float32>")

    var t2 = fixed_size_list_(DynType(Float32Type()), 4)
    var t3 = fixed_size_list_(DynType(Float32Type()), 8)
    assert_true(t == t2)
    assert_false(t == t3)


def test_struct_type() raises:
    var f1 = field("x", DynType(Int32Type()))
    var f2 = field("y", DynType(Float64Type()))
    var t = struct_(f1^, f2^)
    var at: DynType = t.copy().to_dyn()
    assert_true(at.is_struct())
    assert_false(at.is_list())
    assert_false(at.is_primitive())
    assert_equal(String(t), "struct<x: int32, y: float64>")

    var t2 = struct_(
        field("x", DynType(Int32Type())),
        field("y", DynType(Float64Type())),
    )
    assert_true(t == t2)

    var t3 = struct_(
        field("x", DynType(Int32Type())),
        field("y", DynType(Float64Type())),
        field("z", DynType(Int8Type())),
    )
    assert_false(t == t3)


def test_field() raises:
    var f = field("val", DynType(Int64Type()))
    assert_equal(f.name, "val")
    assert_equal(f.dtype, DynType(Int64Type()))
    assert_equal(f.nullable, True)
    assert_equal(String(f), "val: int64")

    var f2 = field("val", DynType(Int64Type()))
    assert_true(f == f2)
    assert_false(f == field("other", DynType(Int64Type())))
    assert_false(f == field("val", DynType(Float32Type())))

    var f3 = field("a", DynType(Int64Type()), nullable=False)
    assert_equal(String(f3), "a: int64")


def test_temporal_dtypes_predicates() raises:
    assert_true(DynType(date32()).is_date32())
    assert_true(DynType(date32()).is_temporal())
    assert_false(DynType(date32()).is_date64())
    assert_true(DynType(date32()).is_primitive())
    assert_false(DynType(date32()).is_integer())

    assert_true(DynType(date64()).is_date64())
    assert_true(DynType(date64()).is_temporal())
    assert_false(DynType(date64()).is_date32())

    assert_true(DynType(time32(second)).is_time32())
    assert_true(DynType(time32(second)).is_temporal())
    assert_false(DynType(time32(second)).is_time64())

    assert_true(DynType(time64(microsecond)).is_time64())
    assert_true(DynType(time64(microsecond)).is_temporal())
    assert_false(DynType(time64(microsecond)).is_time32())

    assert_true(DynType(timestamp(second)).is_timestamp())
    assert_true(DynType(timestamp(second)).is_temporal())
    assert_false(DynType(timestamp(second)).is_duration())

    assert_true(DynType(duration(second)).is_duration())
    assert_true(DynType(duration(second)).is_temporal())
    assert_false(DynType(duration(second)).is_timestamp())


def test_temporal_dtypes_string() raises:
    assert_equal(String(DynType(date32())), "date32")
    assert_equal(String(DynType(date64())), "date64")
    assert_equal(String(DynType(time32(second))), "time32[s]")
    assert_equal(String(DynType(time32(millisecond))), "time32[ms]")
    assert_equal(String(DynType(time64(microsecond))), "time64[us]")
    assert_equal(String(DynType(time64(nanosecond))), "time64[ns]")
    assert_equal(String(DynType(timestamp(second))), "timestamp[s]")
    assert_equal(String(DynType(timestamp(millisecond))), "timestamp[ms]")
    assert_equal(String(DynType(timestamp(microsecond))), "timestamp[us]")
    assert_equal(String(DynType(timestamp(nanosecond))), "timestamp[ns]")
    assert_equal(
        String(DynType(timestamp(second, "UTC"))), "timestamp[s][tz=UTC]"
    )
    assert_equal(String(DynType(duration(second))), "duration[s]")
    assert_equal(String(DynType(duration(nanosecond))), "duration[ns]")


def test_temporal_dtypes_equality() raises:
    assert_true(date32() == date32())
    assert_false(DynType(date32()) == DynType(date64()))
    assert_true(time32(second) == time32(second))
    assert_false(time32(second) == time32(millisecond))
    assert_true(timestamp(second) == timestamp(second))
    assert_false(timestamp(second) == timestamp(millisecond))
    assert_true(timestamp(second, "UTC") == timestamp(second, "UTC"))
    assert_false(timestamp(second, "UTC") == timestamp(second, "US/Pacific"))
    assert_false(timestamp(second, "UTC") == timestamp(second))
    assert_true(duration(nanosecond) == duration(nanosecond))
    assert_false(duration(second) == duration(nanosecond))
    assert_false(DynType(date32()) == DynType(int32))
    assert_false(date32() != date32())
    assert_true(DynType(date32()) != DynType(date64()))
    assert_true(time32(second) != time32(millisecond))
    assert_true(timestamp(second, "UTC") != timestamp(second, "US/Pacific"))


def test_interval_dtypes_predicates() raises:
    assert_true(DynType(year_month_interval()).is_year_month_interval())
    assert_true(DynType(year_month_interval()).is_interval())
    assert_true(DynType(year_month_interval()).is_primitive())
    assert_false(DynType(year_month_interval()).is_temporal())
    assert_false(DynType(year_month_interval()).is_day_time_interval())

    assert_true(DynType(day_time_interval()).is_day_time_interval())
    assert_true(DynType(day_time_interval()).is_interval())
    assert_true(DynType(day_time_interval()).is_primitive())
    assert_false(DynType(day_time_interval()).is_year_month_interval())

    assert_true(DynType(month_day_nano_interval()).is_month_day_nano_interval())
    assert_true(DynType(month_day_nano_interval()).is_interval())
    assert_true(DynType(month_day_nano_interval()).is_primitive())
    assert_false(DynType(month_day_nano_interval()).is_year_month_interval())


def test_interval_dtypes_string() raises:
    assert_equal(String(DynType(year_month_interval())), "month_interval")
    assert_equal(String(DynType(day_time_interval())), "day_time_interval")
    assert_equal(
        String(DynType(month_day_nano_interval())),
        "month_day_nano_interval",
    )


def test_interval_dtypes_equality() raises:
    assert_true(year_month_interval() == year_month_interval())
    assert_true(day_time_interval() == day_time_interval())
    assert_true(month_day_nano_interval() == month_day_nano_interval())
    assert_false(DynType(year_month_interval()) == DynType(day_time_interval()))
    assert_false(
        DynType(day_time_interval()) == DynType(month_day_nano_interval())
    )


def test_dictionary_dtype() raises:
    # Basic construction and predicates
    var dt_d = dictionary(DynType(int32), DynType(string))
    var at: DynType = dt_d.copy().to_dyn()
    assert_true(at.is_dictionary())
    assert_false(at.is_list())
    assert_false(at.is_primitive())
    assert_false(at.is_struct())

    # Field access via as_dictionary()
    ref dd = at.as_dictionary()
    assert_true(dd.index_type() == DynType(int32))
    assert_true(dd.value_type() == DynType(string))
    assert_false(dd.ordered)

    # Ordered variant
    var dt_ord = dictionary(DynType(int8), DynType(int32), ordered=True)
    var at_ord: DynType = dt_ord.copy().to_dyn()
    ref dd_ord = at_ord.as_dictionary()
    assert_true(dd_ord.ordered)
    assert_true(dd_ord.index_type() == DynType(int8))

    # Equality
    var d1 = dictionary(DynType(int32), DynType(string)).copy().to_dyn()
    var d2 = dictionary(DynType(int32), DynType(string)).copy().to_dyn()
    var d3 = dictionary(DynType(int64), DynType(string)).copy().to_dyn()
    var d4 = (
        dictionary(DynType(int32), DynType(string), ordered=True)
        .copy()
        .to_dyn()
    )
    assert_true(d1 == d2)
    assert_false(d1 == d3)  # different index type
    assert_false(d1 == d4)  # ordered differs

    # String representation
    assert_equal(
        String(dictionary(DynType(int32), DynType(string))),
        "dictionary<values=string, indices=int32, ordered=0>",
    )
    assert_equal(
        String(dictionary(DynType(int8), DynType(int32), ordered=True)),
        "dictionary<values=int32, indices=int8, ordered=1>",
    )

    # All valid integer index types
    assert_true(
        dictionary(DynType(int8), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )
    assert_true(
        dictionary(DynType(int16), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )
    assert_true(
        dictionary(DynType(int64), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )
    assert_true(
        dictionary(DynType(uint8), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )
    assert_true(
        dictionary(DynType(uint16), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )
    assert_true(
        dictionary(DynType(uint32), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )
    assert_true(
        dictionary(DynType(uint64), DynType(string))
        .copy()
        .to_dyn()
        .is_dictionary()
    )

    # Non-integer index type must raise
    var raised = False
    try:
        _ = dictionary(DynType(float32), DynType(string))
    except:
        raised = True
    assert_true(raised)

    # Nested value type
    var nested = dictionary(DynType(int32), DynType(list_(DynType(int64))))
    assert_equal(
        String(nested),
        "dictionary<values=list<int64>, indices=int32, ordered=0>",
    )


def test_time_unit_string() raises:
    assert_equal(String(second), "s")
    assert_equal(String(millisecond), "ms")
    assert_equal(String(microsecond), "us")
    assert_equal(String(nanosecond), "ns")


# ---------------------------------------------------------------------------
# num_buffers — the one structural fact the codecs kept re-deriving.
#
# Counts *data* buffers only. Unlike Arrow C++ and arrow-rs, where `buffers[0]`
# is the validity bitmap, marrow's `ArrayData` carries `bitmap` as its own field
# — so these numbers are one lower than the references' throughout, and there is
# no BITMAP buffer kind to model.
#
# This is deliberately just a count. Both references also carry a per-buffer
# `BufferSpec` (kind + byte width), but neither drives its IPC or C-ABI codec
# from it — see the A3 card. The count is what marrow's IPC buffer walk actually
# asks for.
# ---------------------------------------------------------------------------


def test_num_buffers_fixed_width_types() raises:
    """One values buffer, whatever the width — bool included, whose values are
    bit-packed rather than byte-addressed."""
    assert_equal(DynType(dt.bool_).num_buffers(), 1)
    assert_equal(DynType(Int8Type()).num_buffers(), 1)
    assert_equal(DynType(Int64Type()).num_buffers(), 1)
    assert_equal(DynType(Float64Type()).num_buffers(), 1)
    assert_equal(DynType(dt.date32()).num_buffers(), 1)
    assert_equal(DynType(dt.timestamp(dt.second)).num_buffers(), 1)
    assert_equal(DynType(dt.decimal128(10, 2)).num_buffers(), 1)
    assert_equal(DynType(dt.fixed_size_binary_(4)).num_buffers(), 1)


def test_num_buffers_variable_width_types() raises:
    """Offsets plus data, for all four binary-like types."""
    assert_equal(DynType(StringType()).num_buffers(), 2)
    assert_equal(DynType(dt.large_string).num_buffers(), 2)
    assert_equal(DynType(dt.binary).num_buffers(), 2)
    assert_equal(DynType(dt.large_binary).num_buffers(), 2)


def test_num_buffers_nested_types() raises:
    """A list owns its offsets; a struct and a fixed-size list own nothing —
    their data lives entirely in their children."""
    assert_equal(DynType(list_(DynType(Int32Type()))).num_buffers(), 1)
    assert_equal(DynType(dt.large_list_(DynType(Int32Type()))).num_buffers(), 1)
    assert_equal(
        DynType(fixed_size_list_(DynType(Int32Type()), 3)).num_buffers(), 0
    )
    assert_equal(
        DynType(struct_(Field("a", DynType(Int32Type())))).num_buffers(), 0
    )


def test_num_buffers_null_owns_nothing() raises:
    assert_equal(DynType(NullType()).num_buffers(), 0)


def test_num_buffers_dictionary_follows_its_index_type() raises:
    """A dictionary array stores indices; the values live in a separate
    dictionary. arrow-rs resolves this the same way — `layout(key_type)`."""
    var d = DynType(dt.dictionary(DynType(Int32Type()), DynType(StringType())))
    assert_equal(d.num_buffers(), 1)


def test_num_buffers_map_owns_its_offsets() raises:
    """`map` is a list of structs, so it owns an offsets buffer like any list.

    Worth pinning: `map` is absent from IPC's buffer-consuming ladder entirely
    (see V0), so it silently read as owning zero buffers there.
    """
    var m = DynType(
        map_(DynType(StringType()), DynType(Int32Type()))
    )
    assert_equal(m.num_buffers(), 1)
