from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.arrays import array, AnyArray, PrimitiveArray, StringArray
from marrow.builders import PrimitiveBuilder, StringBuilder
from marrow.dtypes import int32, int64, float64, bool_
from marrow.scalars import (
    AnyScalar,
    PrimitiveScalar,
    StringScalar,
)


# ---------------------------------------------------------------------------
# PrimitiveScalar
# ---------------------------------------------------------------------------


def test_primitive_scalar_int32() raises:
    var s = PrimitiveScalar[int32](Scalar[int32.native](42))
    assert_true(s.is_valid())
    assert_false(s.is_null())
    assert_equal(s.value(), 42)


def test_primitive_scalar_float64() raises:
    var s = PrimitiveScalar[float64](Scalar[float64.native](3.14))
    assert_true(s.is_valid())
    assert_equal(s.value(), 3.14)


def test_primitive_scalar_null() raises:
    var s = PrimitiveScalar[int32].null()
    assert_false(s.is_valid())
    assert_true(s.is_null())


def test_primitive_scalar_from_array() raises:
    """Construct from a length-1 array slice."""
    var arr = array[int32]([10, 20, 30])
    var s = PrimitiveScalar[int32](data=arr.slice(1, 1))
    assert_true(s.is_valid())
    assert_equal(s.value(), 20)


def test_primitive_scalar_as_array() raises:
    var s = PrimitiveScalar[int32](Scalar[int32.native](7))
    var arr = s.as_array()
    assert_equal(len(arr), 1)
    assert_equal(arr[0], 7)


def test_primitive_scalar_write_to() raises:
    var s = PrimitiveScalar[int32](Scalar[int32.native](42))
    assert_equal(String(s), "42")


def test_primitive_scalar_write_to_null() raises:
    var s = PrimitiveScalar[int32].null()
    assert_equal(String(s), "null")


# ---------------------------------------------------------------------------
# StringScalar
# ---------------------------------------------------------------------------


def test_string_scalar() raises:
    var s = StringScalar("hello")
    assert_true(s.is_valid())
    assert_equal(s.as_string(), "hello")


def test_string_scalar_null() raises:
    var s = StringScalar.null()
    assert_false(s.is_valid())
    assert_true(s.is_null())


def test_string_scalar_from_array() raises:
    var b = StringBuilder(2)
    b.append("foo")
    b.append("bar")
    var arr = b.finish_typed()
    var s = StringScalar(data=arr.slice(1, 1))
    assert_true(s.is_valid())
    assert_equal(s.as_string(), "bar")


def test_string_scalar_write_to() raises:
    var s = StringScalar("hi")
    assert_equal(String(s), '"hi"')


# ---------------------------------------------------------------------------
# Scalar (type-erased)
# ---------------------------------------------------------------------------


def test_scalar_from_primitive() raises:
    var typed = PrimitiveScalar[int32](Scalar[int32.native](99))
    var erased = AnyScalar(typed)
    assert_true(erased.is_valid())
    assert_equal(erased.dtype(), int32)
    var back = erased.as_primitive[int32]()
    assert_equal(back.value(), 99)


def test_scalar_from_string() raises:
    var typed = StringScalar("world")
    var erased = AnyScalar(typed)
    assert_true(erased.is_valid())
    assert_true(erased.dtype().is_string())
    var back = erased.as_string()
    assert_equal(back.as_string(), "world")


def test_scalar_null() raises:
    var typed = PrimitiveScalar[int32].null()
    var erased = AnyScalar(typed)
    assert_true(erased.is_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
