from std.testing import assert_equal, assert_true, assert_false, TestSuite
import marrow.dtypes as vdt
from marrow.dtypes import (
    ArrowType,
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
    var t = ArrowType(NullType())
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
    var t = ArrowType(BoolType())
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
    var t = ArrowType(StringType())
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
    assert_true(ArrowType(Int8Type()).is_integer())
    assert_true(ArrowType(Int16Type()).is_integer())
    assert_true(ArrowType(Int32Type()).is_integer())
    assert_true(ArrowType(Int64Type()).is_integer())
    assert_true(ArrowType(UInt8Type()).is_integer())
    assert_true(ArrowType(UInt16Type()).is_integer())
    assert_true(ArrowType(UInt32Type()).is_integer())
    assert_true(ArrowType(UInt64Type()).is_integer())
    assert_false(ArrowType(BoolType()).is_integer())
    assert_false(ArrowType(Float32Type()).is_integer())
    assert_false(ArrowType(Float64Type()).is_integer())
    assert_false(ArrowType(NullType()).is_integer())
    assert_false(ArrowType(StringType()).is_integer())


def test_is_signed_integer() raises:
    assert_true(ArrowType(Int8Type()).is_signed_integer())
    assert_true(ArrowType(Int16Type()).is_signed_integer())
    assert_true(ArrowType(Int32Type()).is_signed_integer())
    assert_true(ArrowType(Int64Type()).is_signed_integer())
    assert_false(ArrowType(UInt8Type()).is_signed_integer())
    assert_false(ArrowType(UInt16Type()).is_signed_integer())
    assert_false(ArrowType(UInt32Type()).is_signed_integer())
    assert_false(ArrowType(UInt64Type()).is_signed_integer())
    assert_false(ArrowType(BoolType()).is_signed_integer())
    assert_false(ArrowType(Float32Type()).is_signed_integer())
    assert_false(ArrowType(Float64Type()).is_signed_integer())


def test_is_unsigned_integer() raises:
    assert_false(ArrowType(Int8Type()).is_unsigned_integer())
    assert_false(ArrowType(Int16Type()).is_unsigned_integer())
    assert_false(ArrowType(Int32Type()).is_unsigned_integer())
    assert_false(ArrowType(Int64Type()).is_unsigned_integer())
    assert_true(ArrowType(UInt8Type()).is_unsigned_integer())
    assert_true(ArrowType(UInt16Type()).is_unsigned_integer())
    assert_true(ArrowType(UInt32Type()).is_unsigned_integer())
    assert_true(ArrowType(UInt64Type()).is_unsigned_integer())
    assert_false(ArrowType(BoolType()).is_unsigned_integer())
    assert_false(ArrowType(Float32Type()).is_unsigned_integer())
    assert_false(ArrowType(Float64Type()).is_unsigned_integer())


def test_is_floating_point() raises:
    assert_false(ArrowType(Int8Type()).is_floating_point())
    assert_false(ArrowType(Int32Type()).is_floating_point())
    assert_false(ArrowType(UInt64Type()).is_floating_point())
    assert_false(ArrowType(BoolType()).is_floating_point())
    assert_true(ArrowType(Float16Type()).is_floating_point())
    assert_true(ArrowType(Float32Type()).is_floating_point())
    assert_true(ArrowType(Float64Type()).is_floating_point())
    assert_false(ArrowType(NullType()).is_floating_point())
    assert_false(ArrowType(StringType()).is_floating_point())


def test_is_numeric() raises:
    assert_true(ArrowType(Int8Type()).is_numeric())
    assert_true(ArrowType(Int16Type()).is_numeric())
    assert_true(ArrowType(Int32Type()).is_numeric())
    assert_true(ArrowType(Int64Type()).is_numeric())
    assert_true(ArrowType(UInt8Type()).is_numeric())
    assert_true(ArrowType(UInt16Type()).is_numeric())
    assert_true(ArrowType(UInt32Type()).is_numeric())
    assert_true(ArrowType(UInt64Type()).is_numeric())
    assert_true(ArrowType(Float32Type()).is_numeric())
    assert_true(ArrowType(Float64Type()).is_numeric())
    assert_false(ArrowType(BoolType()).is_numeric())
    assert_false(ArrowType(NullType()).is_numeric())
    assert_false(ArrowType(StringType()).is_numeric())


def test_is_primitive() raises:
    assert_true(ArrowType(BoolType()).is_primitive())
    assert_true(ArrowType(Int8Type()).is_primitive())
    assert_true(ArrowType(Int16Type()).is_primitive())
    assert_true(ArrowType(Int32Type()).is_primitive())
    assert_true(ArrowType(Int64Type()).is_primitive())
    assert_true(ArrowType(UInt8Type()).is_primitive())
    assert_true(ArrowType(UInt16Type()).is_primitive())
    assert_true(ArrowType(UInt32Type()).is_primitive())
    assert_true(ArrowType(UInt64Type()).is_primitive())
    assert_true(ArrowType(Float32Type()).is_primitive())
    assert_true(ArrowType(Float64Type()).is_primitive())
    assert_false(ArrowType(NullType()).is_primitive())
    assert_false(ArrowType(StringType()).is_primitive())


def test_bit_width() raises:
    assert_equal(ArrowType(NullType()).bit_width(), UInt8(0))
    assert_equal(ArrowType(BoolType()).bit_width(), UInt8(1))
    assert_equal(ArrowType(Int8Type()).bit_width(), UInt8(8))
    assert_equal(ArrowType(Int16Type()).bit_width(), UInt8(16))
    assert_equal(ArrowType(Int32Type()).bit_width(), UInt8(32))
    assert_equal(ArrowType(Int64Type()).bit_width(), UInt8(64))
    assert_equal(ArrowType(UInt8Type()).bit_width(), UInt8(8))
    assert_equal(ArrowType(UInt16Type()).bit_width(), UInt8(16))
    assert_equal(ArrowType(UInt32Type()).bit_width(), UInt8(32))
    assert_equal(ArrowType(UInt64Type()).bit_width(), UInt8(64))
    assert_equal(ArrowType(Float16Type()).bit_width(), UInt8(16))
    assert_equal(ArrowType(Float32Type()).bit_width(), UInt8(32))
    assert_equal(ArrowType(Float64Type()).bit_width(), UInt8(64))
    assert_equal(ArrowType(BinaryType()).bit_width(), UInt8(0))
    assert_equal(ArrowType(StringType()).bit_width(), UInt8(0))


def test_byte_width() raises:
    assert_equal(ArrowType(Int8Type()).byte_width(), 1)
    assert_equal(ArrowType(Int16Type()).byte_width(), 2)
    assert_equal(ArrowType(Int32Type()).byte_width(), 4)
    assert_equal(ArrowType(Int64Type()).byte_width(), 8)
    assert_equal(ArrowType(UInt8Type()).byte_width(), 1)
    assert_equal(ArrowType(UInt16Type()).byte_width(), 2)
    assert_equal(ArrowType(UInt32Type()).byte_width(), 4)
    assert_equal(ArrowType(UInt64Type()).byte_width(), 8)
    assert_equal(ArrowType(Float16Type()).byte_width(), 2)
    assert_equal(ArrowType(Float32Type()).byte_width(), 4)
    assert_equal(ArrowType(Float64Type()).byte_width(), 8)
    assert_equal(ArrowType(BoolType()).byte_width(), 0)
    assert_equal(ArrowType(NullType()).byte_width(), 0)
    assert_equal(ArrowType(BinaryType()).byte_width(), 0)
    assert_equal(ArrowType(StringType()).byte_width(), 0)


def test_write_string() raises:
    assert_equal(String(ArrowType(NullType())), "null")
    assert_equal(String(ArrowType(BoolType())), "bool")
    assert_equal(String(ArrowType(Int8Type())), "int8")
    assert_equal(String(ArrowType(Int16Type())), "int16")
    assert_equal(String(ArrowType(Int32Type())), "int32")
    assert_equal(String(ArrowType(Int64Type())), "int64")
    assert_equal(String(ArrowType(UInt8Type())), "uint8")
    assert_equal(String(ArrowType(UInt16Type())), "uint16")
    assert_equal(String(ArrowType(UInt32Type())), "uint32")
    assert_equal(String(ArrowType(UInt64Type())), "uint64")
    assert_equal(String(ArrowType(Float16Type())), "float16")
    assert_equal(String(ArrowType(Float32Type())), "float32")
    assert_equal(String(ArrowType(Float64Type())), "float64")
    assert_equal(String(ArrowType(BinaryType())), "binary")
    assert_equal(String(ArrowType(StringType())), "string")


def test_eq() raises:
    var a = ArrowType(UInt64Type())
    var b = ArrowType(UInt64Type())
    var c = ArrowType(Int32Type())
    assert_true(a == b)
    assert_false(a == c)
    assert_false(a != b)
    assert_true(a != c)
    assert_true(ArrowType(NullType()) == ArrowType(NullType()))
    assert_false(ArrowType(NullType()) == ArrowType(BoolType()))
    assert_true(ArrowType(Float32Type()) == ArrowType(Float32Type()))
    assert_false(ArrowType(Float32Type()) == ArrowType(Float64Type()))


def test_copy() raises:
    var original = ArrowType(Int64Type())
    var copied = ArrowType(copy=original)
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
    var t = ArrowType(BinaryType())
    assert_true(t.is_binary())
    assert_false(t.is_string())
    assert_false(t.is_null())
    assert_false(t.is_integer())
    assert_false(t.is_floating_point())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "binary")


def test_list_type() raises:
    var t = list_(ArrowType(Int32Type()))
    assert_true(t.is_list())
    assert_false(t.is_fixed_size_list())
    assert_false(t.is_struct())
    assert_false(t.is_primitive())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "list<int32>")

    var t2 = list_(ArrowType(Int32Type()))
    assert_true(t == t2)
    assert_false(t == list_(ArrowType(Float64Type())))


def test_fixed_size_list_type() raises:
    var t = fixed_size_list_(ArrowType(Float32Type()), 4)
    assert_true(t.is_fixed_size_list())
    assert_false(t.is_list())
    assert_false(t.is_struct())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "fixed_size_list<item: float32>")

    var t2 = fixed_size_list_(ArrowType(Float32Type()), 4)
    var t3 = fixed_size_list_(ArrowType(Float32Type()), 8)
    assert_true(t == t2)
    assert_false(t == t3)


def test_struct_type() raises:
    var f1 = field("x", ArrowType(Int32Type()))
    var f2 = field("y", ArrowType(Float64Type()))
    var t = struct_(f1, f2)
    assert_true(t.is_struct())
    assert_false(t.is_list())
    assert_false(t.is_primitive())
    assert_equal(t.bit_width(), UInt8(0))
    assert_equal(t.byte_width(), 0)
    assert_equal(String(t), "struct<x: int32, y: float64>")

    var t2 = struct_(field("x", ArrowType(Int32Type())), field("y", ArrowType(Float64Type())))
    assert_true(t == t2)


def test_field() raises:
    var f = field("val", ArrowType(Int64Type()))
    assert_equal(f.name, "val")
    assert_equal(f.dtype[], ArrowType(Int64Type()))
    assert_equal(f.nullable, True)
    assert_equal(String(f), "val: int64")

    var f2 = field("val", ArrowType(Int64Type()))
    assert_true(f == f2)
    assert_false(f == field("other", ArrowType(Int64Type())))
    assert_false(f == field("val", ArrowType(Float32Type())))


def test_is_fixed_size() raises:
    assert_true(ArrowType(Int32Type()).is_fixed_size())
    assert_true(ArrowType(Float64Type()).is_fixed_size())
    assert_true(ArrowType(BoolType()).is_fixed_size())
    assert_false(ArrowType(NullType()).is_fixed_size())
    assert_false(ArrowType(StringType()).is_fixed_size())
    assert_false(list_(ArrowType(Int32Type())).is_fixed_size())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
