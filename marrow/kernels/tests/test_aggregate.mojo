from std.testing import assert_equal
from marrow.testing import TestSuite

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import array, nulls, PrimitiveBuilder, Int32Builder
from marrow.dtypes import int32, int64, Int32Type, Int64Type
from marrow.kernels.aggregate import sum_


def test_sum_typed() raises:
    var a = array([1, 2, 3, 4, 5], int64)
    var result = sum_[Int64Type](a)
    assert_equal(result.value(), 15)


def test_sum_with_nulls() raises:
    """Sum skips null values."""
    var a = Int32Builder(3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var result = sum_[Int32Type](a.finish())
    assert_equal(result.value(), 30)


def test_sum_all_nulls() raises:
    var a = nulls(5, int64)
    var result = sum_[Int64Type](a)
    assert_equal(result.value(), 0)


def test_sum_empty() raises:
    var a = array(int32)
    var result = sum_[Int32Type](a)
    assert_equal(result.value(), 0)


def test_sum_untyped() raises:
    var a = AnyArray(array([1, 2, 3], int64))
    var result = sum_(a)
    assert_equal(result.as_int64().value(), 6)


def main() raises:
    TestSuite.run[__functions_in_module()]()
