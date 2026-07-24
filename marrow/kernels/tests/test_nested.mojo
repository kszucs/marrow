"""Unit tests for the nested (list) compute kernels (marrow.kernels.nested).

Expected patterns match PyArrow's `pc.list_value_length`.
"""

from std.testing import assert_equal, assert_true, assert_false

from marrow.testing import TestSuite
from marrow.arrays import ListArray, Int32Array
from marrow.builders import ListBuilder, Int64Builder, array
from marrow.dtypes import int32

from marrow.kernels.nested import ArrayLengthKernel


def _lists_with_null() raises -> ListArray:
    """[[1, 2], [3], null, [4, 5, 6]]."""
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    list_b.append_valid()  # [1, 2]
    child.append(3)
    list_b.append_valid()  # [3]
    list_b.append_null()  # null
    child.append(4)
    child.append(5)
    child.append(6)
    list_b.append_valid()  # [4, 5, 6]
    return list_b.finish()


def _lists_no_null() raises -> ListArray:
    """[[10], [], [20, 30]]."""
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(10)
    list_b.append_valid()  # [10]
    list_b.append_valid()  # []
    child.append(20)
    child.append(30)
    list_b.append_valid()  # [20, 30]
    return list_b.finish()


def test_array_length_no_nulls() raises:
    var r = ArrayLengthKernel.apply(_lists_no_null())
    assert_equal(r.null_count(), 0)
    assert_true(r == array([1, 0, 2], int32))


def test_array_length_propagates_nulls() raises:
    # matches pc.list_value_length: null list -> null length (not 0)
    var r = ArrayLengthKernel.apply(_lists_with_null())
    assert_equal(r.null_count(), 1)
    assert_true(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_true(r.is_valid(3))
    assert_equal(r[0].value(), 2)
    assert_equal(r[1].value(), 1)
    assert_equal(r[3].value(), 3)


def test_array_length_sliced_with_nulls() raises:
    var full = _lists_with_null()  # [[1,2],[3],null,[4,5,6]]
    var a = full.slice(2, 2)  # [null, [4,5,6]]
    var r = ArrayLengthKernel.apply(a)
    assert_equal(r.null_count(), 1)
    assert_false(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_equal(r[1].value(), 3)


def main() raises:
    TestSuite.run[__functions_in_module()]()
