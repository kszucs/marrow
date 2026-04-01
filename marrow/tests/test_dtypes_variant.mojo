from std.testing import assert_equal, assert_true, assert_false, TestSuite
import marrow.dtypes_variant as vdt
from marrow.dtypes_variant import (
    AnyType,
    Field,
    NullType,
    BoolType,
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
    StringType,
    ListType,
    FixedSizeListType,
    StructType,
    field,
    list_,
    fixed_size_list_,
    struct_,
)


def test_null_type() raises:
    var t = AnyType(NullType())
    assert_true(t.is_null())
    assert_false(t.is_bool())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_false(t.is_primitive())
    assert_false(t.is_string())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "null")


def test_bool_type() raises:
    var t = AnyType(BoolType())
    assert_true(t.is_bool())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_true(t.is_primitive())
    assert_false(t.is_string())
    assert_equal(t.bit_width(), UInt8(1))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "bool")


def test_string_type() raises:
    var t = AnyType(StringType())
    assert_true(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_bool())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_false(t.is_numeric())
    assert_false(t.is_primitive())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "string")


def test_is_integer() raises:
    assert_true(AnyType(Int8Type()).is_integer())
    assert_true(AnyType(Int16Type()).is_integer())
    assert_true(AnyType(Int32Type()).is_integer())
    assert_true(AnyType(Int64Type()).is_integer())
    assert_true(AnyType(UInt8Type()).is_integer())
    assert_true(AnyType(UInt16Type()).is_integer())
    assert_true(AnyType(UInt32Type()).is_integer())
    assert_true(AnyType(UInt64Type()).is_integer())
    assert_false(AnyType(BoolType()).is_integer())
    assert_false(AnyType(Float32Type()).is_integer())
    assert_false(AnyType(Float64Type()).is_integer())
    assert_false(AnyType(NullType()).is_integer())
    assert_false(AnyType(StringType()).is_integer())


def test_is_signed_integer() raises:
    assert_true(AnyType(Int8Type()).is_signed_integer())
    assert_true(AnyType(Int16Type()).is_signed_integer())
    assert_true(AnyType(Int32Type()).is_signed_integer())
    assert_true(AnyType(Int64Type()).is_signed_integer())
    assert_false(AnyType(UInt8Type()).is_signed_integer())
    assert_false(AnyType(UInt16Type()).is_signed_integer())
    assert_false(AnyType(UInt32Type()).is_signed_integer())
    assert_false(AnyType(UInt64Type()).is_signed_integer())
    assert_false(AnyType(BoolType()).is_signed_integer())
    assert_false(AnyType(Float32Type()).is_signed_integer())
    assert_false(AnyType(Float64Type()).is_signed_integer())


def test_is_unsigned_integer() raises:
    assert_false(AnyType(Int8Type()).is_unsigned_integer())
    assert_false(AnyType(Int16Type()).is_unsigned_integer())
    assert_false(AnyType(Int32Type()).is_unsigned_integer())
    assert_false(AnyType(Int64Type()).is_unsigned_integer())
    assert_true(AnyType(UInt8Type()).is_unsigned_integer())
    assert_true(AnyType(UInt16Type()).is_unsigned_integer())
    assert_true(AnyType(UInt32Type()).is_unsigned_integer())
    assert_true(AnyType(UInt64Type()).is_unsigned_integer())
    assert_false(AnyType(BoolType()).is_unsigned_integer())
    assert_false(AnyType(Float32Type()).is_unsigned_integer())
    assert_false(AnyType(Float64Type()).is_unsigned_integer())


def test_is_floating_point() raises:
    assert_false(AnyType(Int8Type()).is_floating_point())
    assert_false(AnyType(Int32Type()).is_floating_point())
    assert_false(AnyType(UInt64Type()).is_floating_point())
    assert_false(AnyType(BoolType()).is_floating_point())
    assert_true(AnyType(Float16Type()).is_floating_point())
    assert_true(AnyType(Float32Type()).is_floating_point())
    assert_true(AnyType(Float64Type()).is_floating_point())
    assert_false(AnyType(NullType()).is_floating_point())
    assert_false(AnyType(StringType()).is_floating_point())


def test_is_numeric() raises:
    assert_true(AnyType(Int8Type()).is_numeric())
    assert_true(AnyType(Int16Type()).is_numeric())
    assert_true(AnyType(Int32Type()).is_numeric())
    assert_true(AnyType(Int64Type()).is_numeric())
    assert_true(AnyType(UInt8Type()).is_numeric())
    assert_true(AnyType(UInt16Type()).is_numeric())
    assert_true(AnyType(UInt32Type()).is_numeric())
    assert_true(AnyType(UInt64Type()).is_numeric())
    assert_true(AnyType(Float32Type()).is_numeric())
    assert_true(AnyType(Float64Type()).is_numeric())
    assert_false(AnyType(BoolType()).is_numeric())
    assert_false(AnyType(NullType()).is_numeric())
    assert_false(AnyType(StringType()).is_numeric())


def test_is_primitive() raises:
    assert_true(AnyType(BoolType()).is_primitive())
    assert_true(AnyType(Int8Type()).is_primitive())
    assert_true(AnyType(Int16Type()).is_primitive())
    assert_true(AnyType(Int32Type()).is_primitive())
    assert_true(AnyType(Int64Type()).is_primitive())
    assert_true(AnyType(UInt8Type()).is_primitive())
    assert_true(AnyType(UInt16Type()).is_primitive())
    assert_true(AnyType(UInt32Type()).is_primitive())
    assert_true(AnyType(UInt64Type()).is_primitive())
    assert_true(AnyType(Float32Type()).is_primitive())
    assert_true(AnyType(Float64Type()).is_primitive())
    assert_false(AnyType(NullType()).is_primitive())
    assert_false(AnyType(StringType()).is_primitive())


def test_bit_width() raises:
    assert_equal(AnyType(NullType()).bit_width(), UInt8(0))
    assert_equal(AnyType(BoolType()).bit_width(), UInt8(1))
    assert_equal(AnyType(Int8Type()).bit_width(), UInt8(8))
    assert_equal(AnyType(Int16Type()).bit_width(), UInt8(16))
    assert_equal(AnyType(Int32Type()).bit_width(), UInt8(32))
    assert_equal(AnyType(Int64Type()).bit_width(), UInt8(64))
    assert_equal(AnyType(UInt8Type()).bit_width(), UInt8(8))
    assert_equal(AnyType(UInt16Type()).bit_width(), UInt8(16))
    assert_equal(AnyType(UInt32Type()).bit_width(), UInt8(32))
    assert_equal(AnyType(UInt64Type()).bit_width(), UInt8(64))
    assert_equal(AnyType(Float16Type()).bit_width(), UInt8(16))
    assert_equal(AnyType(Float32Type()).bit_width(), UInt8(32))
    assert_equal(AnyType(Float64Type()).bit_width(), UInt8(64))
    assert_equal(AnyType(BinaryType()).bit_width(), UInt8(0))
    assert_equal(AnyType(StringType()).bit_width(), UInt8(0))


def test_byte_width() raises:
    assert_equal(AnyType(Int8Type()).byte_width(), 1)
    assert_equal(AnyType(Int16Type()).byte_width(), 2)
    assert_equal(AnyType(Int32Type()).byte_width(), 4)
    assert_equal(AnyType(Int64Type()).byte_width(), 8)
    assert_equal(AnyType(UInt8Type()).byte_width(), 1)
    assert_equal(AnyType(UInt16Type()).byte_width(), 2)
    assert_equal(AnyType(UInt32Type()).byte_width(), 4)
    assert_equal(AnyType(UInt64Type()).byte_width(), 8)
    assert_equal(AnyType(Float16Type()).byte_width(), 2)
    assert_equal(AnyType(Float32Type()).byte_width(), 4)
    assert_equal(AnyType(Float64Type()).byte_width(), 8)
    assert_equal(AnyType(BoolType()).byte_width(), 0)
    assert_equal(AnyType(NullType()).byte_width(), 0)
    assert_equal(AnyType(BinaryType()).byte_width(), 0)
    assert_equal(AnyType(StringType()).byte_width(), 0)


def test_write_string() raises:
    assert_equal(String(AnyType(NullType())), "null")
    assert_equal(String(AnyType(BoolType())), "bool")
    assert_equal(String(AnyType(Int8Type())), "int8")
    assert_equal(String(AnyType(Int16Type())), "int16")
    assert_equal(String(AnyType(Int32Type())), "int32")
    assert_equal(String(AnyType(Int64Type())), "int64")
    assert_equal(String(AnyType(UInt8Type())), "uint8")
    assert_equal(String(AnyType(UInt16Type())), "uint16")
    assert_equal(String(AnyType(UInt32Type())), "uint32")
    assert_equal(String(AnyType(UInt64Type())), "uint64")
    assert_equal(String(AnyType(Float16Type())), "float16")
    assert_equal(String(AnyType(Float32Type())), "float32")
    assert_equal(String(AnyType(Float64Type())), "float64")
    assert_equal(String(AnyType(BinaryType())), "binary")
    assert_equal(String(AnyType(StringType())), "string")


def test_eq() raises:
    var a = AnyType(UInt64Type())
    var b = AnyType(UInt64Type())
    var c = AnyType(Int32Type())
    assert_true(a == b)
    assert_false(a == c)
    assert_false(a != b)
    assert_true(a != c)
    assert_true(AnyType(NullType()) == AnyType(NullType()))
    assert_false(AnyType(NullType()) == AnyType(BoolType()))
    assert_true(AnyType(Float32Type()) == AnyType(Float32Type()))
    assert_false(AnyType(Float32Type()) == AnyType(Float64Type()))


def test_copy() raises:
    var original = AnyType(Int64Type())
    var copied = AnyType(copy=original)
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
    assert_equal(String(vdt.null), "null")
    assert_equal(String(vdt.bool_), "bool")
    assert_equal(String(vdt.int8), "int8")
    assert_equal(String(vdt.int16), "int16")
    assert_equal(String(vdt.int32), "int32")
    assert_equal(String(vdt.int64), "int64")
    assert_equal(String(vdt.uint8), "uint8")
    assert_equal(String(vdt.uint16), "uint16")
    assert_equal(String(vdt.uint32), "uint32")
    assert_equal(String(vdt.uint64), "uint64")
    assert_equal(String(vdt.float16), "float16")
    assert_equal(String(vdt.float32), "float32")
    assert_equal(String(vdt.float64), "float64")
    assert_equal(String(vdt.binary), "binary")
    assert_equal(String(vdt.string), "string")


def test_binary_type() raises:
    var t = AnyType(BinaryType())
    assert_true(t.is_binary())
    assert_false(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "binary")


def test_list_type() raises:
    var t = list_(AnyType(Int32Type()))
    assert_true(t.is_list())
    assert_false(t.is_fixed_size_list())
    assert_false(t.is_struct())
    assert_false(t.is_primitive())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "list<int32>")

    var t2 = list_(AnyType(Int32Type()))
    assert_true(t == t2)
    assert_false(t == list_(AnyType(Float64Type())))


def test_fixed_size_list_type() raises:
    var t = fixed_size_list_(AnyType(Float32Type()), 4)
    assert_true(t.is_fixed_size_list())
    assert_false(t.is_list())
    assert_false(t.is_struct())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "fixed_size_list<item: float32>")

    var t2 = fixed_size_list_(AnyType(Float32Type()), 4)
    var t3 = fixed_size_list_(AnyType(Float32Type()), 8)
    assert_true(t == t2)
    assert_false(t == t3)


def test_struct_type() raises:
    var f1 = field("x", AnyType(Int32Type()))
    var f2 = field("y", AnyType(Float64Type()))
    var t = struct_(f1, f2)
    assert_true(t.is_struct())
    assert_false(t.is_list())
    assert_false(t.is_primitive())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "struct<x: int32, y: float64>")

    var t2 = struct_(field("x", AnyType(Int32Type())), field("y", AnyType(Float64Type())))
    assert_true(t == t2)


def test_field() raises:
    var f = field("val", AnyType(Int64Type()))
    assert_equal(f.name, "val")
    assert_equal(f.dtype[], AnyType(Int64Type()))
    assert_equal(f.nullable, True)
    assert_equal(String(f), "val: int64")

    var f2 = field("val", AnyType(Int64Type()))
    assert_true(f == f2)
    assert_false(f == field("other", AnyType(Int64Type())))
    assert_false(f == field("val", AnyType(Float32Type())))


def test_is_fixed_size() raises:
    assert_true(AnyType(Int32Type()).is_fixed_size())
    assert_true(AnyType(Float64Type()).is_fixed_size())
    assert_true(AnyType(BoolType()).is_fixed_size())
    assert_false(AnyType(NullType()).is_fixed_size())
    assert_false(AnyType(StringType()).is_fixed_size())
    assert_false(list_(AnyType(Int32Type())).is_fixed_size())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
