from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import (
    AnyArray,
    BoolArray,
    PrimitiveArray,
    StringArray,
    FixedSizeListArray,
    StructArray,
)
from marrow.builders import (
    array,
    AnyBuilder,
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    FixedSizeListBuilder,
    StructBuilder,
)
from marrow.dtypes import (
    int32,
    int64,
    float64,
    string,
    bool_,
    field,
    Int32Type,
    Int64Type,
    Float64Type,
)
from marrow.scalars import (
    AnyScalar,
    BoolScalar,
    PrimitiveScalar,
    StringScalar,
    ListScalar,
    StructScalar,
)


# ---------------------------------------------------------------------------
# PrimitiveScalar
# ---------------------------------------------------------------------------


def test_primitive_scalar_int32() raises:
    var s = PrimitiveScalar[Int32Type](Scalar[int32.native](42))
    assert_true(s.is_valid())
    assert_false(s.is_null())
    assert_equal(s.value(), 42)


def test_primitive_scalar_float64() raises:
    var s = PrimitiveScalar[Float64Type](Scalar[float64.native](3.14))
    assert_true(s.is_valid())
    assert_equal(s.value(), 3.14)


def test_primitive_scalar_null() raises:
    var s = PrimitiveScalar[Int32Type].null()
    assert_false(s.is_valid())
    assert_true(s.is_null())


def test_primitive_scalar_from_array() raises:
    """Construct via array __getitem__."""
    var arr = array[Int32Type]([10, 20, 30])
    var s = arr[1]
    assert_true(s.is_valid())
    assert_equal(s.value(), 20)


def test_primitive_scalar_to_array() raises:
    var s = PrimitiveScalar[Int32Type](Scalar[int32.native](7))
    var arr = s.to_array()
    assert_equal(len(arr), 1)
    assert_equal(arr[0], 7)


def test_primitive_scalar_write_to() raises:
    var s = PrimitiveScalar[Int32Type](Scalar[int32.native](42))
    assert_equal(String(s), "42")


def test_primitive_scalar_write_to_null() raises:
    var s = PrimitiveScalar[Int32Type].null()
    assert_equal(String(s), "null")


# ---------------------------------------------------------------------------
# StringScalar
# ---------------------------------------------------------------------------


def test_string_scalar() raises:
    var s = StringScalar("hello")
    assert_true(s.is_valid())
    assert_equal(s.to_string(), "hello")


def test_string_scalar_null() raises:
    var s = StringScalar.null()
    assert_false(s.is_valid())
    assert_true(s.is_null())


def test_string_scalar_from_array() raises:
    """Construct via array __getitem__."""
    var b = StringBuilder(2)
    b.append("foo")
    b.append("bar")
    var arr = b.finish()
    var s = arr[1]
    assert_true(s.is_valid())
    assert_equal(s.to_string(), "bar")


def test_string_scalar_write_to() raises:
    var s = StringScalar("hi")
    assert_equal(String(s), "hi")


# ---------------------------------------------------------------------------
# Scalar (type-erased)
# ---------------------------------------------------------------------------


def test_scalar_from_primitive() raises:
    var typed = PrimitiveScalar[Int32Type](Scalar[int32.native](99))
    var erased = AnyScalar(typed^)
    assert_true(erased.is_valid())
    assert_equal(erased.type(), int32)
    var back = erased.as_primitive[Int32Type]()
    assert_equal(back.value(), 99)


def test_scalar_from_string() raises:
    var typed = StringScalar("world")
    var erased = AnyScalar(typed^)
    assert_true(erased.is_valid())
    assert_true(erased.type().is_string())
    var back = erased.as_string()
    assert_equal(back.to_string(), "world")


def test_scalar_null() raises:
    var typed = PrimitiveScalar[Int32Type].null()
    var erased = AnyScalar(typed^)
    assert_true(erased.is_null())


# ---------------------------------------------------------------------------
# BoolScalar from BoolArray.__getitem__
# ---------------------------------------------------------------------------


def test_bool_scalar_from_array() raises:
    var b = BoolBuilder(3)
    b.append(True)
    b.append(False)
    b.append_null()
    var arr = b.finish()
    var s0 = arr[0]
    assert_true(s0.is_valid())
    assert_equal(s0.value(), True)
    var s1 = arr[1]
    assert_true(s1.is_valid())
    assert_equal(s1.value(), False)
    var s2 = arr[2]
    assert_false(s2.is_valid())


# ---------------------------------------------------------------------------
# ListScalar from FixedSizeListArray.__getitem__
# ---------------------------------------------------------------------------


def test_list_scalar_from_fixed_size_list_array() raises:
    var inner = PrimitiveBuilder[Int32Type]()
    var fsl = FixedSizeListBuilder(AnyBuilder(inner^), 2)
    fsl.values().as_primitive[Int32Type]().append(10)
    fsl.values().as_primitive[Int32Type]().append(20)
    fsl.append_valid()
    fsl.values().as_primitive[Int32Type]().append(0)
    fsl.values().as_primitive[Int32Type]().append(0)
    fsl.append_null()
    var arr = fsl.finish()
    var s0 = arr[0]
    assert_true(s0.is_valid())
    assert_equal(len(s0.value()), 2)
    assert_equal(s0.value().as_primitive[Int32Type]()[0].value(), 10)
    assert_equal(s0.value().as_primitive[Int32Type]()[1].value(), 20)
    var s1 = arr[1]
    assert_false(s1.is_valid())


# ---------------------------------------------------------------------------
# StructScalar from StructArray.__getitem__
# ---------------------------------------------------------------------------


def test_struct_scalar_from_array() raises:
    var sb = StructBuilder([field("x", int32), field("y", int64)], capacity=2)
    sb.field_builder(0).as_primitive[Int32Type]().append(1)
    sb.field_builder(0).as_primitive[Int32Type]().append(2)
    sb.field_builder(1).as_primitive[Int64Type]().append(10)
    sb.field_builder(1).as_primitive[Int64Type]().append(20)
    sb.append_valid()
    sb.append_valid()
    var arr = sb.finish()
    var s0 = arr[0]
    assert_true(s0.is_valid())
    assert_equal(s0.num_fields(), 2)
    assert_equal(s0.field(0).as_primitive[Int32Type]().value(), 1)
    assert_equal(s0.field(1).as_primitive[Int64Type]().value(), 10)


def test_struct_scalar_null_from_array() raises:
    var sb = StructBuilder([field("x", int32)], capacity=2)
    sb.field_builder(0).as_primitive[Int32Type]().append(5)
    sb.field_builder(0).as_primitive[Int32Type]().append(0)
    sb.append_valid()
    sb.append_null()
    var arr = sb.finish()
    assert_true(arr[0].is_valid())
    assert_false(arr[1].is_valid())


# ---------------------------------------------------------------------------
# AnyArray.__getitem__ -> AnyScalar
# ---------------------------------------------------------------------------


def test_any_array_getitem_primitive() raises:
    var arr: AnyArray = array[Int64Type]([10, 20, 30])
    var s = arr[1]
    assert_true(s.is_valid())
    assert_equal(s.type(), int64)
    assert_equal(s.as_primitive[Int64Type]().value(), 20)


def test_any_array_getitem_primitive_null() raises:
    var b = PrimitiveBuilder[Int32Type](3)
    b.append(1)
    b.append_null()
    b.append(3)
    var arr: AnyArray = b.finish()
    assert_true(arr[0].is_valid())
    assert_false(arr[1].is_valid())


def test_any_array_getitem_bool() raises:
    var b = BoolBuilder(2)
    b.append(True)
    b.append_null()
    var arr: AnyArray = b.finish()
    var s0 = arr[0]
    assert_true(s0.is_valid())
    assert_equal(s0.as_bool().value(), True)
    assert_false(arr[1].is_valid())


def test_any_array_getitem_string() raises:
    var b = StringBuilder(2)
    b.append("hello")
    b.append_null()
    var arr: AnyArray = b.finish()
    var s0 = arr[0]
    assert_true(s0.is_valid())
    assert_equal(s0.as_string().to_string(), "hello")
    assert_false(arr[1].is_valid())


def test_any_array_getitem_fixed_size_list() raises:
    var inner = PrimitiveBuilder[Int32Type]()
    var fsl = FixedSizeListBuilder(AnyBuilder(inner^), 2)
    fsl.values().as_primitive[Int32Type]().append(7)
    fsl.values().as_primitive[Int32Type]().append(8)
    fsl.append_valid()
    var arr: AnyArray = fsl.finish()
    var s = arr[0]
    assert_true(s.is_valid())
    assert_equal(len(s.as_list().value()), 2)


def test_any_array_getitem_struct() raises:
    var sb = StructBuilder([field("n", int32)], capacity=1)
    sb.field_builder(0).as_primitive[Int32Type]().append(42)
    sb.append_valid()
    var arr: AnyArray = sb.finish()
    var s = arr[0]
    assert_true(s.is_valid())
    assert_equal(s.as_struct().num_fields(), 1)
    assert_equal(s.as_struct().field(0).as_primitive[Int32Type]().value(), 42)


def test_any_array_getitem_out_of_bounds() raises:
    var arr: AnyArray = array[Int64Type]([1, 2, 3])
    var raised = False
    try:
        _ = arr[5]
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.run[__functions_in_module()]()
