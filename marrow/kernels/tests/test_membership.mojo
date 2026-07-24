from std.testing import assert_true, assert_raises
from marrow.testing import TestSuite

from marrow.arrays import AnyArray
from marrow.builders import array
from marrow.dtypes import int32, int64, uint32, float64
from marrow.kernels.membership import is_in


def test_is_in_clickbench_int() raises:
    """The ClickBench ``x IN (-1, 6)`` shape."""
    var values = array([-1, 0, 6, 3, 6], int64)
    var value_set = array([-1, 6], int64)
    var expected = array([True, False, True, False, True])
    assert_true(is_in(values, value_set) == expected)


def test_is_in_int32() raises:
    var values = array([1, 2, 3, 4, 5], int32)
    var value_set = array([2, 4], int32)
    assert_true(
        is_in(values, value_set) == array([False, True, False, True, False])
    )


def test_is_in_uint32() raises:
    var values = array([10, 20, 30], uint32)
    var value_set = array([30, 10], uint32)
    assert_true(is_in(values, value_set) == array([True, False, True]))


def test_is_in_float64() raises:
    var values = array([1.5, 2.5, 3.5], float64)
    var value_set = array([2.5, 3.5], float64)
    assert_true(is_in(values, value_set) == array([False, True, True]))


def test_is_in_bool() raises:
    var values = array([True, False, True, False])
    var value_set = array([True])
    assert_true(is_in(values, value_set) == array([True, False, True, False]))


def test_is_in_string() raises:
    var values = array([String("apple"), "banana", "cherry", "apple", "date"])
    var value_set = array([String("apple"), "cherry"])
    assert_true(
        is_in(values, value_set) == array([True, False, True, True, False])
    )


def test_is_in_empty_value_set() raises:
    """Nothing is in the empty set."""
    var values = array([1, 2, 3], int64)
    var value_set = array(int64)
    assert_true(is_in(values, value_set) == array([False, False, False]))


def test_is_in_empty_values() raises:
    var values = array(int64)
    var value_set = array([1, 2], int64)
    assert_true(len(is_in(values, value_set)) == 0)


def test_is_in_duplicate_value_set() raises:
    """Duplicates in ``value_set`` do not change membership."""
    var values = array([1, 2, 3], int64)
    var value_set = array([2, 2, 2, 3, 3], int64)
    assert_true(is_in(values, value_set) == array([False, True, True]))


def test_is_in_null_value_set_without_null() raises:
    """PyArrow default: a null in ``values`` is false when the set has no null.

    Output is always valid (never null)."""
    var values = array([1, 2, None, 5], int64)
    var value_set = array([2, 5], int64)
    var result = is_in(values, value_set)
    assert_true(result == array([False, True, False, True]))
    assert_true(result.null_count() == 0)


def test_is_in_null_value_set_with_null() raises:
    """PyArrow default: a null in ``values`` is true when the set contains null.
    """
    var values = array([1, 2, None, 5], int64)
    var value_set = array([2, 5, None], int64)
    var result = is_in(values, value_set)
    assert_true(result == array([False, True, True, True]))
    assert_true(result.null_count() == 0)


def test_is_in_string_with_nulls() raises:
    var values = array([String("a"), "b", "c"])
    var value_set = array([String("b")])
    assert_true(is_in(values, value_set) == array([False, True, False]))


def test_is_in_type_mismatch_raises() raises:
    var values: AnyArray = array([1, 2, 3], int32)
    var value_set: AnyArray = array([1, 2], int64)
    with assert_raises():
        _ = is_in(values, value_set)


def main() raises:
    TestSuite.run[__functions_in_module()]()
