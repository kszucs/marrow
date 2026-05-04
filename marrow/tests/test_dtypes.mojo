from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite
import marrow.dtypes as dt
from marrow.dtypes import *


def test_bool_type() raises:
    assert_true(dt.bool_ == dt.bool_)
    assert_false(AnyDataType(dt.bool_) == AnyDataType(dt.int64))

    var t = AnyDataType(dt.bool_)
    assert_true(t.is_bool())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_true(t.is_primitive())
    assert_false(t.is_string())
    assert_equal(String(t), "bool")


def test_is_integer() raises:
    assert_true(AnyDataType(dt.int8).is_integer())
    assert_true(AnyDataType(dt.int16).is_integer())
    assert_true(AnyDataType(dt.int32).is_integer())
    assert_true(AnyDataType(dt.int64).is_integer())
    assert_true(AnyDataType(dt.uint8).is_integer())
    assert_true(AnyDataType(dt.uint16).is_integer())
    assert_true(AnyDataType(dt.uint32).is_integer())
    assert_true(AnyDataType(dt.uint64).is_integer())
    assert_false(AnyDataType(dt.bool_).is_integer())
    assert_false(AnyDataType(dt.float32).is_integer())
    assert_false(AnyDataType(dt.float64).is_integer())
    assert_false(AnyDataType(dt.list_(dt.int64)).is_integer())


def test_is_signed_integer() raises:
    assert_true(AnyDataType(dt.int8).is_signed_integer())
    assert_true(AnyDataType(dt.int16).is_signed_integer())
    assert_true(AnyDataType(dt.int32).is_signed_integer())
    assert_true(AnyDataType(dt.int64).is_signed_integer())
    assert_false(AnyDataType(dt.uint8).is_signed_integer())
    assert_false(AnyDataType(dt.uint16).is_signed_integer())
    assert_false(AnyDataType(dt.uint32).is_signed_integer())
    assert_false(AnyDataType(dt.uint64).is_signed_integer())
    assert_false(AnyDataType(dt.bool_).is_signed_integer())
    assert_false(AnyDataType(dt.float32).is_signed_integer())
    assert_false(AnyDataType(dt.float64).is_signed_integer())


def test_is_unsigned_integer() raises:
    assert_false(AnyDataType(dt.int8).is_unsigned_integer())
    assert_false(AnyDataType(dt.int16).is_unsigned_integer())
    assert_false(AnyDataType(dt.int32).is_unsigned_integer())
    assert_false(AnyDataType(dt.int64).is_unsigned_integer())
    assert_true(AnyDataType(dt.uint8).is_unsigned_integer())
    assert_true(AnyDataType(dt.uint16).is_unsigned_integer())
    assert_true(AnyDataType(dt.uint32).is_unsigned_integer())
    assert_true(AnyDataType(dt.uint64).is_unsigned_integer())
    assert_false(AnyDataType(dt.bool_).is_unsigned_integer())
    assert_false(AnyDataType(dt.float32).is_unsigned_integer())
    assert_false(AnyDataType(dt.float64).is_unsigned_integer())


def test_is_floating_point() raises:
    assert_false(AnyDataType(dt.int8).is_floating_point())
    assert_false(AnyDataType(dt.int16).is_floating_point())
    assert_false(AnyDataType(dt.int32).is_floating_point())
    assert_false(AnyDataType(dt.int64).is_floating_point())
    assert_false(AnyDataType(dt.uint8).is_floating_point())
    assert_false(AnyDataType(dt.uint16).is_floating_point())
    assert_false(AnyDataType(dt.uint32).is_floating_point())
    assert_false(AnyDataType(dt.uint64).is_floating_point())
    assert_false(AnyDataType(dt.bool_).is_floating_point())
    assert_true(AnyDataType(dt.float32).is_floating_point())
    assert_true(AnyDataType(dt.float64).is_floating_point())


def test_bit_width() raises:
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
    var t = AnyDataType(NullType())
    assert_true(t.is_null())
    assert_false(t.is_bool())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_false(t.is_primitive())
    assert_false(t.is_string())
    assert_equal(String(t), "null")


def test_string_type() raises:
    var t = AnyDataType(StringType())
    assert_true(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_bool())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_false(t.is_primitive())
    assert_equal(String(t), "string")


def test_byte_width() raises:
    assert_equal(AnyDataType(Int8Type()).byte_width(), 1)
    assert_equal(AnyDataType(Int16Type()).byte_width(), 2)
    assert_equal(AnyDataType(Int32Type()).byte_width(), 4)
    assert_equal(AnyDataType(Int64Type()).byte_width(), 8)
    assert_equal(AnyDataType(UInt8Type()).byte_width(), 1)
    assert_equal(AnyDataType(UInt16Type()).byte_width(), 2)
    assert_equal(AnyDataType(UInt32Type()).byte_width(), 4)
    assert_equal(AnyDataType(UInt64Type()).byte_width(), 8)
    assert_equal(AnyDataType(Float16Type()).byte_width(), 2)
    assert_equal(AnyDataType(Float32Type()).byte_width(), 4)
    assert_equal(AnyDataType(Float64Type()).byte_width(), 8)


def test_eq() raises:
    var a = AnyDataType(UInt64Type())
    var b = AnyDataType(UInt64Type())
    var c = AnyDataType(Int32Type())
    assert_true(a == b)
    assert_false(a == c)
    assert_false(a != b)
    assert_true(a != c)
    assert_true(NullType() == NullType())
    assert_false(AnyDataType(NullType()) == AnyDataType(BoolType()))
    assert_true(Float32Type() == Float32Type())
    assert_false(AnyDataType(Float32Type()) == AnyDataType(Float64Type()))


def test_copy() raises:
    var original = AnyDataType(Int64Type())
    var copied = AnyDataType(copy=original)
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
    var t = AnyDataType(BinaryType())
    assert_true(t.is_binary())
    assert_false(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_equal(String(t), "binary")


def test_list_type() raises:
    var t = list_(AnyDataType(Int32Type()))
    var at: AnyDataType = t.copy().to_any()
    assert_true(at.is_list())
    assert_false(at.is_fixed_size_list())
    assert_false(at.is_struct())
    assert_false(at.is_primitive())
    assert_equal(String(t), "list<int32>")

    var t2 = list_(AnyDataType(Int32Type()))
    assert_true(t == t2)
    assert_false(t == list_(AnyDataType(Float64Type())))

    var nested = list_(list_(AnyDataType(Int64Type())))
    assert_equal(String(nested), "list<list<int64>>")

    var t3 = list_(int64)
    assert_equal(t3.value_type(), int64)


def test_fixed_size_list_type() raises:
    var t = fixed_size_list_(AnyDataType(Float32Type()), 4)
    var at: AnyDataType = t.copy().to_any()
    assert_true(at.is_fixed_size_list())
    assert_false(at.is_list())
    assert_false(at.is_struct())
    assert_equal(String(t), "fixed_size_list<item: float32>")

    var t2 = fixed_size_list_(AnyDataType(Float32Type()), 4)
    var t3 = fixed_size_list_(AnyDataType(Float32Type()), 8)
    assert_true(t == t2)
    assert_false(t == t3)


def test_struct_type() raises:
    var f1 = field("x", AnyDataType(Int32Type()))
    var f2 = field("y", AnyDataType(Float64Type()))
    var t = struct_(f1^, f2^)
    var at: AnyDataType = t.copy().to_any()
    assert_true(at.is_struct())
    assert_false(at.is_list())
    assert_false(at.is_primitive())
    assert_equal(String(t), "struct<x: int32, y: float64>")

    var t2 = struct_(
        field("x", AnyDataType(Int32Type())),
        field("y", AnyDataType(Float64Type())),
    )
    assert_true(t == t2)

    var t3 = struct_(
        field("x", AnyDataType(Int32Type())),
        field("y", AnyDataType(Float64Type())),
        field("z", AnyDataType(Int8Type())),
    )
    assert_false(t == t3)


def test_field() raises:
    var f = field("val", AnyDataType(Int64Type()))
    assert_equal(f.name, "val")
    assert_equal(f.dtype, AnyDataType(Int64Type()))
    assert_equal(f.nullable, True)
    assert_equal(String(f), "val: int64")

    var f2 = field("val", AnyDataType(Int64Type()))
    assert_true(f == f2)
    assert_false(f == field("other", AnyDataType(Int64Type())))
    assert_false(f == field("val", AnyDataType(Float32Type())))

    var f3 = field("a", AnyDataType(Int64Type()), nullable=False)
    assert_equal(String(f3), "a: int64")


def test_is_fixed_size() raises:
    assert_true(AnyDataType(Int32Type()).is_fixed_size())
    assert_true(AnyDataType(Float64Type()).is_fixed_size())
    assert_true(AnyDataType(BoolType()).is_fixed_size())
    assert_false(AnyDataType(NullType()).is_fixed_size())
    assert_false(AnyDataType(StringType()).is_fixed_size())
    assert_false(AnyDataType(list_(AnyDataType(Int32Type()))).is_fixed_size())


def test_temporal_dtypes_predicates() raises:
    assert_true(AnyDataType(date32()).is_date32())
    assert_true(AnyDataType(date32()).is_temporal())
    assert_false(AnyDataType(date32()).is_date64())
    assert_true(AnyDataType(date32()).is_primitive())
    assert_false(AnyDataType(date32()).is_integer())

    assert_true(AnyDataType(date64()).is_date64())
    assert_true(AnyDataType(date64()).is_temporal())
    assert_false(AnyDataType(date64()).is_date32())

    assert_true(AnyDataType(time32(second)).is_time32())
    assert_true(AnyDataType(time32(second)).is_temporal())
    assert_false(AnyDataType(time32(second)).is_time64())

    assert_true(AnyDataType(time64(microsecond)).is_time64())
    assert_true(AnyDataType(time64(microsecond)).is_temporal())
    assert_false(AnyDataType(time64(microsecond)).is_time32())

    assert_true(AnyDataType(timestamp(second)).is_timestamp())
    assert_true(AnyDataType(timestamp(second)).is_temporal())
    assert_false(AnyDataType(timestamp(second)).is_duration())

    assert_true(AnyDataType(duration(second)).is_duration())
    assert_true(AnyDataType(duration(second)).is_temporal())
    assert_false(AnyDataType(duration(second)).is_timestamp())


def test_temporal_dtypes_string() raises:
    assert_equal(String(AnyDataType(date32())), "date32")
    assert_equal(String(AnyDataType(date64())), "date64")
    assert_equal(String(AnyDataType(time32(second))), "time32[s]")
    assert_equal(String(AnyDataType(time32(millisecond))), "time32[ms]")
    assert_equal(String(AnyDataType(time64(microsecond))), "time64[us]")
    assert_equal(String(AnyDataType(time64(nanosecond))), "time64[ns]")
    assert_equal(String(AnyDataType(timestamp(second))), "timestamp[s]")
    assert_equal(String(AnyDataType(timestamp(millisecond))), "timestamp[ms]")
    assert_equal(String(AnyDataType(timestamp(microsecond))), "timestamp[us]")
    assert_equal(String(AnyDataType(timestamp(nanosecond))), "timestamp[ns]")
    assert_equal(
        String(AnyDataType(timestamp(second, "UTC"))), "timestamp[s][tz=UTC]"
    )
    assert_equal(String(AnyDataType(duration(second))), "duration[s]")
    assert_equal(String(AnyDataType(duration(nanosecond))), "duration[ns]")


def test_temporal_dtypes_equality() raises:
    assert_true(date32() == date32())
    assert_false(AnyDataType(date32()) == AnyDataType(date64()))
    assert_true(time32(second) == time32(second))
    assert_false(time32(second) == time32(millisecond))
    assert_true(timestamp(second) == timestamp(second))
    assert_false(timestamp(second) == timestamp(millisecond))
    assert_true(timestamp(second, "UTC") == timestamp(second, "UTC"))
    assert_false(timestamp(second, "UTC") == timestamp(second, "US/Pacific"))
    assert_false(timestamp(second, "UTC") == timestamp(second))
    assert_true(duration(nanosecond) == duration(nanosecond))
    assert_false(duration(second) == duration(nanosecond))
    assert_false(AnyDataType(date32()) == AnyDataType(int32))
    assert_false(date32() != date32())
    assert_true(AnyDataType(date32()) != AnyDataType(date64()))
    assert_true(time32(second) != time32(millisecond))
    assert_true(timestamp(second, "UTC") != timestamp(second, "US/Pacific"))


def test_dictionary_dtype() raises:
    # Basic construction and predicates
    var dt_d = dictionary(AnyDataType(int32), AnyDataType(string))
    var at: AnyDataType = dt_d.copy().to_any()
    assert_true(at.is_dictionary())
    assert_false(at.is_list())
    assert_false(at.is_primitive())
    assert_false(at.is_struct())

    # Field access via as_dictionary()
    ref dd = at.as_dictionary()
    assert_true(dd.index_type() == AnyDataType(int32))
    assert_true(dd.value_type() == AnyDataType(string))
    assert_false(dd.ordered)

    # Ordered variant
    var dt_ord = dictionary(AnyDataType(int8), AnyDataType(int32), ordered=True)
    var at_ord: AnyDataType = dt_ord.copy().to_any()
    ref dd_ord = at_ord.as_dictionary()
    assert_true(dd_ord.ordered)
    assert_true(dd_ord.index_type() == AnyDataType(int8))

    # Equality
    var d1 = dictionary(AnyDataType(int32), AnyDataType(string)).copy().to_any()
    var d2 = dictionary(AnyDataType(int32), AnyDataType(string)).copy().to_any()
    var d3 = dictionary(AnyDataType(int64), AnyDataType(string)).copy().to_any()
    var d4 = dictionary(AnyDataType(int32), AnyDataType(string), ordered=True).copy().to_any()
    assert_true(d1 == d2)
    assert_false(d1 == d3)  # different index type
    assert_false(d1 == d4)  # ordered differs

    # String representation
    assert_equal(
        String(dictionary(AnyDataType(int32), AnyDataType(string))),
        "dictionary<values=string, indices=int32, ordered=0>",
    )
    assert_equal(
        String(dictionary(AnyDataType(int8), AnyDataType(int32), ordered=True)),
        "dictionary<values=int32, indices=int8, ordered=1>",
    )

    # All valid integer index types
    assert_true(dictionary(AnyDataType(int8), AnyDataType(string)).copy().to_any().is_dictionary())
    assert_true(dictionary(AnyDataType(int16), AnyDataType(string)).copy().to_any().is_dictionary())
    assert_true(dictionary(AnyDataType(int64), AnyDataType(string)).copy().to_any().is_dictionary())
    assert_true(dictionary(AnyDataType(uint8), AnyDataType(string)).copy().to_any().is_dictionary())
    assert_true(dictionary(AnyDataType(uint16), AnyDataType(string)).copy().to_any().is_dictionary())
    assert_true(dictionary(AnyDataType(uint32), AnyDataType(string)).copy().to_any().is_dictionary())
    assert_true(dictionary(AnyDataType(uint64), AnyDataType(string)).copy().to_any().is_dictionary())

    # Non-integer index type must raise
    var raised = False
    try:
        _ = dictionary(AnyDataType(float32), AnyDataType(string))
    except:
        raised = True
    assert_true(raised)

    # Nested value type
    var nested = dictionary(AnyDataType(int32), AnyDataType(list_(AnyDataType(int64))))
    assert_equal(
        String(nested),
        "dictionary<values=list<int64>, indices=int32, ordered=0>",
    )


def test_time_unit_string() raises:
    assert_equal(String(second), "s")
    assert_equal(String(millisecond), "ms")
    assert_equal(String(microsecond), "us")
    assert_equal(String(nanosecond), "ns")


def main() raises:
    TestSuite.run[__functions_in_module()]()
