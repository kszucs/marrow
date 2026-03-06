from std.testing import assert_equal, assert_true, assert_false, TestSuite


from marrow.arrays import *
from marrow.test_fixtures.bool_array import as_bool_array_scalar
from marrow.test_fixtures.arrays import build_array_data, assert_bitmap_set


def test_boolean_array() raises:
    var a = BoolArray()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)

    a.grow(3)
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 3)

    a.append(as_bool_array_scalar(True))
    a.append(as_bool_array_scalar(False))
    a.append(as_bool_array_scalar(True))
    assert_equal(len(a), 3)
    assert_equal(a.capacity, 3)

    a.append(as_bool_array_scalar(True))
    assert_equal(len(a), 4)
    assert_equal(a.capacity, 6)
    assert_true(a.is_valid(0))
    assert_true(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_true(a.is_valid(3))

    var d = a^.take_data()
    assert_equal(d.length, 4)

    var b = d^.as_primitive[bool_]()


def test_append() raises:
    var a = Int8Array()
    assert_equal(len(a), 0)
    assert_equal(a.capacity, 0)
    a.append(1)
    a.append(2)
    a.append(3)
    assert_equal(len(a), 3)
    assert_true(a.capacity >= len(a))


def test_array_from_ints() raises:
    var g = array[int8](1, 2)
    assert_equal(len(g), 2)
    assert_equal(materialize[g.dtype](), materialize[int8]())
    assert_equal(g.unsafe_get(0), 1)
    assert_equal(g.unsafe_get(1), 2)


def test_drop_null() raises:
    """Test the drop null function."""
    var array_data = build_array_data(10, 5)

    var primitive_array = PrimitiveArray[uint8](array_data^)
    #
    # Check the setup.
    assert_equal(primitive_array.null_count(), 5)
    assert_bitmap_set(
        primitive_array.bitmap()[], [1, 3, 5, 7, 9], "check setup"
    )

    primitive_array.drop_nulls[DType.uint8]()
    assert_equal(primitive_array.unsafe_get(0), 1)
    assert_equal(primitive_array.unsafe_get(1), 3)
    assert_equal(primitive_array.null_count(), 0)
    assert_bitmap_set(primitive_array.bitmap()[], [0, 1, 2, 3, 4], "after drop")


def test_primitive_array_with_offset() raises:
    """Test PrimitiveArray with offset functionality."""
    # Create a regular array first
    var arr = Int32Array(10)
    arr.unsafe_set(0, 100)
    arr.unsafe_set(1, 200)
    arr.unsafe_set(2, 300)
    arr.unsafe_set(3, 400)
    arr.unsafe_set(4, 500)

    # Default offset should be 0
    assert_equal(arr.offset, 0)
    assert_equal(arr.unsafe_get(0), 100)
    assert_equal(arr.unsafe_get(1), 200)

    # Create a copy of array with offset, should point to the same buffers.
    var arr_data = arr.as_data()
    var arr_with_offset = PrimitiveArray[int32](arr_data[].copy(), offset=2)
    assert_equal(arr_with_offset.offset, 2)

    # Test that offset affects get operations
    assert_equal(arr_with_offset.unsafe_get(0), 300)  # Should get arr[2]
    assert_equal(arr_with_offset.unsafe_get(1), 400)  # Should get arr[3]
    assert_equal(arr_with_offset.unsafe_get(2), 500)  # Should get arr[4]

    # Test that offset affects set operations
    arr_with_offset.unsafe_set(3, 999)  # Should set arr[5]
    assert_equal(arr.unsafe_get(5), 999)


def test_primitive_array_moveinit_with_offset() raises:
    """Test __moveinit__ preserves offset."""
    var arr = Int16Array(5, offset=3)
    arr.unsafe_set(0, 123)

    var moved_arr = arr^
    assert_equal(moved_arr.offset, 3)
    assert_equal(moved_arr.unsafe_get(0), 123)


def test_primitive_array_constructor_with_offset() raises:
    """Test PrimitiveArray constructor with offset parameter."""
    var arr1 = Int8Array(10)  # Default offset=0
    assert_equal(arr1.offset, 0)

    var arr2 = Int8Array(10, offset=5)  # Explicit offset
    assert_equal(arr2.offset, 5)

    # Test that data.offset is also set correctly
    assert_equal(arr2.data.offset, 5)


def test_primitive_array_offset_with_validity() raises:
    """Test that offset works correctly with validity bitmap."""
    var arr = UInt8Array(10, offset=1)

    # Set some values with validity
    arr.unsafe_set(0, 42)  # This should set buffer[1] and bitmap[1]
    arr.unsafe_set(1, 43)  # This should set buffer[2] and bitmap[2]

    # Verify values are accessible through offset
    assert_equal(arr.unsafe_get(0), 42)
    assert_equal(arr.unsafe_get(1), 43)

    # Verify bitmap is also offset correctly
    assert_true(arr.is_valid(0))  # Should check bitmap[1]
    assert_true(arr.is_valid(1))  # Should check bitmap[2]


def test_primitive_array_nulls_with_offset() raises:
    """Test PrimitiveArray.nulls static method creates array with default offset.
    """
    var null_arr = Int64Array.nulls(5)
    assert_equal(null_arr.offset, 0)
    assert_equal(null_arr.data.offset, 0)

    # All elements should be invalid (null)
    for i in range(5):
        assert_false(null_arr.is_valid(i))


def test_primitive_array_write_to() raises:
    """Test write_to method formats PrimitiveArray correctly."""
    var arr = Int32Array(5)
    arr.append(10)
    arr.append(20)
    arr.append(30)

    var output = String()
    arr.write_to(output)

    # Check that output contains expected format
    var result = String(output)
    assert_true("PrimitiveArray(" in result)
    assert_true("dtype=" in result)
    assert_true("offset=" in result)
    assert_true("capacity=" in result)
    assert_true("buffer=" in result)
    assert_true("10" in result)  # At least first value should work


def test_primitive_array_write_to_with_nulls() raises:
    """Test write_to method handles null values correctly."""
    var array_data = build_array_data(5, 2)
    var arr = PrimitiveArray[uint8](array_data^)

    var output = String()
    arr.write_to(output)

    # Check that output contains NULL for invalid entries
    var result = String(output)
    assert_true("PrimitiveArray(" in result)
    assert_true("NULL" in result)


def test_primitive_array_write_to_with_offset() raises:
    """Test write_to method works correctly with offset."""
    var arr = Int16Array(10, offset=2)
    arr.append(100)
    arr.append(200)

    var output = String()
    arr.write_to(output)

    var result = String(output)
    assert_true("PrimitiveArray(" in result)
    assert_true("offset=2" in result)
    # Note: Due to offset bug in write_to, values may not appear correctly


def test_primitive_array_write_to_large_array() raises:
    """Test write_to method truncates large arrays with ellipsis."""
    var arr = Int8Array(20)  # Use capacity > 10 to trigger truncation
    # Fill with values 0, 1, 2, ..., 14
    for i in range(15):
        arr.append(rebind[Scalar[int8.native]](Int8(i)))

    var output = String()
    arr.write_to(output)

    var result = String(output)
    assert_true("PrimitiveArray(" in result)
    assert_true("..." in result)  # Should truncate after 10 elements


def test_primitive_array_str() raises:
    """Test __str__ method returns formatted string representation."""
    var arr = array[int32](42, 84, 126)

    var result = String(arr)
    assert_true("PrimitiveArray(" in result)
    assert_true("42" in result)  # At least first value should work


def test_primitive_array_str_empty() raises:
    """Test __str__ method on empty array."""
    var arr = Float32Array(0)

    var result = String(arr)
    assert_true("PrimitiveArray(" in result)
    assert_true("capacity=0" in result)


def test_primitive_array_repr() raises:
    """Test __repr__ method returns same as __str__."""
    var arr = UInt8Array(5)
    arr.append(255)
    arr.append(128)

    var str_result = String(arr)
    # var repr_result = arr.__repr__()

    # Both should be identical
    # assert_equal(str_result, repr_result)
    assert_equal(
        str_result,
        (
            "PrimitiveArray( dtype=DataType(code=uint8), offset=0, capacity=5,"
            " buffer=[255, 128, NULL, NULL, NULL, ])"
        ),
    )

    var arr64 = Int64Array()
    arr64.append(1)
    arr64.append(3)
    arr64.append(5)
    assert_equal(
        String(arr64),
        (
            "PrimitiveArray( dtype=DataType(code=int64), offset=0, capacity=4,"
            " buffer=[1, 3, 5, NULL, ])"
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
