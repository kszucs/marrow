from std.testing import assert_equal, TestSuite

from marrow.arrays import array, AnyArray, PrimitiveArray, nulls
from marrow.builders import PrimitiveBuilder
from marrow.dtypes import int32, int64
from marrow.kernels.aggregate import sum_


def test_sum_typed() raises:
    var a = array[int64]([1, 2, 3, 4, 5])
    var result = sum_[int64](a)
    assert_equal(result.value(), 15)


def test_sum_with_nulls() raises:
    """Sum skips null values."""
    var a = PrimitiveBuilder[int32](3)
    a.append(10)
    a.append(20)
    a.append_null()  # index 2 is null
    var result = sum_[int32](a.finish_typed())
    assert_equal(result.value(), 30)


def test_sum_all_nulls() raises:
    var a = nulls[int64](5)
    var result = sum_[int64](a)
    assert_equal(result.value(), 0)


def test_sum_empty() raises:
    var a = array[int32]()
    var result = sum_[int32](a)
    assert_equal(result.value(), 0)


def test_sum_untyped() raises:
    var a = AnyArray(array[int64]([1, 2, 3]))
    var result = sum_(a)
    assert_equal(result.as_primitive[int64]().value(), 6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
