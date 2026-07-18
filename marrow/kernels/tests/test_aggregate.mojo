from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import array, nulls, PrimitiveBuilder, Int32Builder
from marrow.dtypes import int32, int64, float64, Int32Type, Int64Type
from marrow.kernels.aggregate import sum, mean


def test_sumtyped() raises:
    var a = array([1, 2, 3, 4, 5], int64)
    var result = sum[Int64Type](a)
    assert_equal(result.value(), 15)


def test_sumwith_nulls() raises:
    """Sum skips null values."""
    var a = Int32Builder(3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var result = sum[Int32Type](a.finish())
    assert_equal(result.value(), 30)


def test_sumall_nulls() raises:
    var a = nulls(5, int64)
    var result = sum[Int64Type](a)
    assert_equal(result.value(), 0)


def test_sumempty() raises:
    var a = array(int32)
    var result = sum[Int32Type](a)
    assert_equal(result.value(), 0)


def test_sumuntyped() raises:
    var a: AnyArray = array([1, 2, 3], int64)
    var result = sum(a)
    assert_equal(result.as_int64().value(), 6)


def test_mean_int() raises:
    """Mean of integers is a float64 scalar."""
    var a: AnyArray = array([1, 2, 3, 4], int32)
    var result = mean(a)
    assert_true(result.type() == float64)
    assert_equal(result.as_float64().value(), 2.5)


def test_mean_float() raises:
    var a: AnyArray = array([1.0, 2.0, 6.0], float64)
    var result = mean(a)
    assert_equal(result.as_float64().value(), 3.0)


def test_mean_skips_nulls() raises:
    """Nulls are excluded from both the sum and the divisor."""
    var a = Int32Builder(4)
    a.append(10)
    a.append(20)
    a.append_null()
    a.append(30)
    var result = mean(a.finish())
    assert_equal(result.as_float64().value(), 20.0)  # (10+20+30)/3


def test_mean_all_null_is_null() raises:
    var a: AnyArray = nulls(4, int64)
    var result = mean(a)
    assert_false(result.is_valid())


def test_mean_empty_is_null() raises:
    var a: AnyArray = array(int32)
    var result = mean(a)
    assert_false(result.is_valid())


def main() raises:
    TestSuite.run[__functions_in_module()]()
