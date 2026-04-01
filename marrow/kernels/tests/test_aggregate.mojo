from std.testing import assert_equal, TestSuite

from marrow.arrays import AnyArray, PrimitiveArray
from marrow.builders import array, nulls, PrimitiveBuilder
from marrow.dtypes import int32, int64
from marrow.kernels.aggregate import sum_


def test_sum_typed() raises:
    var a = array[Int64Type]([1, 2, 3, 4, 5])
    var result = sum_[Int64Type](a)
    assert_equal(result.value(), 15)


def test_sum_with_nulls() raises:
    """Sum skips null values."""
    var a = PrimitiveBuilder[Int32Type](3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var result = sum_[Int32Type](a.finish())
    assert_equal(result.value(), 30)


def test_sum_all_nulls() raises:
    var a = nulls[Int64Type](5)
    var result = sum_[Int64Type](a)
    assert_equal(result.value(), 0)


def test_sum_empty() raises:
    var a = array[Int32Type]()
    var result = sum_[Int32Type](a)
    assert_equal(result.value(), 0)


def test_sum_untyped() raises:
    var a = AnyArray(array[Int64Type]([1, 2, 3]))
    var result = sum_(a)
    assert_equal(result.as_primitive[Int64Type]().value(), 6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
