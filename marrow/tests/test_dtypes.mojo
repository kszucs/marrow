from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite
import marrow.dtypes as dt
from marrow.dtypes import (
    AnyDataType,
    string,
    Field,
    NullType,
    BoolType,
    StringType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    BinaryType,
    list_,
    fixed_size_list_,
    struct_,
    field,
    binary,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    bool_,
    null,
)


def test_bool_type() raises:
    assert_true(AnyDataType(dt.bool_) == AnyDataType(dt.bool_))
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
    assert_equal(dt.bool_.bit_width(), 1)
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
    assert_true(AnyDataType(NullType()) == AnyDataType(NullType()))
    assert_false(AnyDataType(NullType()) == AnyDataType(BoolType()))
    assert_true(AnyDataType(Float32Type()) == AnyDataType(Float32Type()))
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


def main() raises:
    TestSuite.run[__functions_in_module()]()
