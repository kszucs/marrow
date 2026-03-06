"""Test the base module."""
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from std.memory import ArcPointer

from marrow.arrays.base import ArrayData
from marrow.buffers import Buffer, Bitmap
from marrow.dtypes import DType, int8, uint8, int64
from marrow.test_fixtures.arrays import build_array_data, assert_bitmap_set


def test_array_data_with_offset() raises:
    """Test ArrayData with offset functionality."""
    # Create ArrayData with offset
    var bitmap = ArcPointer(Bitmap.alloc(10))
    var buffer = ArcPointer(Buffer.alloc[int8.native](10))

    # Set some data in the buffer
    buffer[].unsafe_set[int8.native](2, 100)
    buffer[].unsafe_set[int8.native](3, 200)
    buffer[].unsafe_set[int8.native](4, 300)

    # Set validity bits
    bitmap[].unsafe_set(2, True)
    bitmap[].unsafe_set(3, True)
    bitmap[].unsafe_set(4, True)

    # Create ArrayData with offset=2
    var array_data = ArrayData(
        dtype=materialize[int8](),
        length=3,
        bitmap=bitmap,
        buffers=[buffer],
        children=[],
        offset=2,
    )

    assert_equal(array_data.offset, 2)

    # Test is_valid with offset
    assert_true(array_data.is_valid(0))  # Should check bitmap[2]
    assert_true(array_data.is_valid(1))  # Should check bitmap[3]
    assert_true(array_data.is_valid(2))  # Should check bitmap[4]


def test_array_data_fieldwise_init() raises:
    """Test that @fieldwise_init decorator works with offset field."""
    var bitmap = ArcPointer(Bitmap.alloc(5))
    var buffer = ArcPointer(Buffer.alloc[int8.native](5))

    # Test creating ArrayData with all fields specified including offset
    var array_data = ArrayData(
        dtype=materialize[int8](),
        length=5,
        bitmap=bitmap,
        buffers=[buffer],
        children=[],
        offset=3,
    )

    assert_equal(array_data.dtype, materialize[int8]())
    assert_equal(array_data.length, 5)
    assert_equal(array_data.offset, 3)


def test_array_data_write_to_with_offset() raises:
    """Test ArrayData write_to method respects offset."""

    var bitmap = ArcPointer(Bitmap.alloc(10))
    var buffer = ArcPointer(Buffer.alloc[DType.uint8](10))

    comptime for dtype in [uint8, int64]:
        # Set up data with values at positions 1,2,3
        buffer[].unsafe_set[dtype.native](1, 10)
        buffer[].unsafe_set[dtype.native](2, 11)
        buffer[].unsafe_set[dtype.native](3, 12)

        # Set validity for positions 1,2,3
        bitmap[].unsafe_set(1, True)
        bitmap[].unsafe_set(2, True)
        bitmap[].unsafe_set(3, True)

        # Create ArrayData with offset=1, so logical indices 0,1,2 map to physical indices 1,2,3
        var array_data = ArrayData(
            dtype=materialize[dtype](),
            length=3,
            bitmap=bitmap,
            buffers=[buffer],
            children=[],
            offset=1,
        )

        var writer = String()
        writer.write(array_data)
        assert_equal(writer.strip(), "10 11 12")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
