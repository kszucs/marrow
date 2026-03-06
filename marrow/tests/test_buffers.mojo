from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.test_fixtures.arrays import assert_bitmap_set
from marrow.buffers import *


fn is_aligned[
    T: AnyType
](ptr: UnsafePointer[T, MutAnyOrigin], alignment: Int) -> Bool:
    return (Int(ptr) % alignment) == 0


def test_buffer_init() raises:
    var b = Buffer.alloc(10)
    assert_equal(b.size, 64)
    assert_true(is_aligned(b.ptr, 64))

    var b1 = Buffer.alloc[DType.bool](10)
    assert_equal(b1.size, 64)
    assert_true(is_aligned(b1.ptr, 64))

    var b2 = Buffer.alloc[DType.bool](64 * 8 + 1)
    assert_equal(b2.size, 128)
    assert_true(is_aligned(b2.ptr, 64))


def test_buffer_grow() raises:
    var b = Buffer.alloc(10)
    b.unsafe_set(0, 111)
    assert_equal(b.size, 64)
    b.grow(20)
    assert_equal(b.size, 64)
    assert_equal(b.unsafe_get(0), 111)
    b.grow(80)
    assert_equal(b.size, 128)
    assert_equal(b.unsafe_get(0), 111)


def test_buffer_set_get() raises:
    var buf = Buffer.alloc(10)
    assert_equal(buf.size, 64)

    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 43)
    buf.unsafe_set(2, 44)
    assert_equal(buf.unsafe_get(0), 42)
    assert_equal(buf.unsafe_get(1), 43)
    assert_equal(buf.unsafe_get(2), 44)

    assert_equal(buf.size, 64)
    assert_equal(
        buf.length[DType.uint16](), 32
    )  # 64 bytes / 2 bytes per element
    # reinterpreting the underlying bits as uint16
    assert_equal(buf.unsafe_get[DType.uint16](0), 42 + (43 << 8))
    assert_equal(buf.unsafe_get[DType.uint16](1), 44)


def test_buffer_from_values() raises:
    var buf = Buffer.from_values[DType.int64](-3, 9, 81)

    assert_equal(buf.unsafe_get[DType.int64](0), -3)
    assert_equal(buf.unsafe_get[DType.int64](1), 9)
    assert_equal(buf.unsafe_get[DType.int64](2), 81)


def test_buffer_swap() raises:
    var one = Buffer.alloc(10)
    one.unsafe_set(0, 111)
    var two = Buffer.alloc(10)
    two.unsafe_set(0, 222)

    one.swap(two)

    assert_equal(one.unsafe_get(0), 222)
    assert_equal(two.unsafe_get(0), 111)


def test_bitmap() raises:
    var b = Bitmap.alloc(10)
    assert_equal(b.size(), 64)
    assert_equal(b.length(), 64 * 8)
    assert_equal(b.bit_count(), 0)

    assert_false(b.unsafe_get(0))
    b.unsafe_set(0, True)
    assert_true(b.unsafe_get(0))
    assert_equal(b.bit_count(), 1)
    assert_false(b.unsafe_get(1))
    b.unsafe_set(1, True)
    assert_true(b.unsafe_get(1))
    assert_equal(b.bit_count(), 2)


def test_count_leading_zeros() raises:
    var b = Bitmap.alloc(10)
    var expected_bits = b.length()
    assert_equal(b.count_leading_zeros(), expected_bits)
    assert_equal(b.count_leading_zeros(10), expected_bits - 10)

    b.unsafe_set(0, True)
    assert_equal(b.count_leading_zeros(), 0)
    assert_equal(b.count_leading_zeros(1), expected_bits - 1)
    b.unsafe_set(0, False)

    var to_test = [0, 1, 7, 8, 10, 16, 31]
    for i in range(len(to_test)):
        bit_position = to_test[i]
        b.unsafe_set(bit_position, True)
        assert_equal(b.count_leading_zeros(), bit_position)
        if bit_position > 4:
            # Count with start position.
            assert_equal(b.count_leading_bits(4), bit_position - 4)
        b.unsafe_set(bit_position, False)


def test_count_leading_ones() raises:
    var b = Bitmap.alloc(10)
    assert_equal(b.count_leading_ones(), 0)
    b.unsafe_set(0, True)
    assert_equal(b.count_leading_ones(), 1)
    assert_equal(b.count_leading_ones(1), 0)

    b.unsafe_set(1, True)
    assert_equal(b.count_leading_ones(), 2)
    assert_equal(b.count_leading_ones(1), 1)


def _reset(mut bitmap: Bitmap) raises:
    bitmap.unsafe_range_set(0, bitmap.length(), False)
    assert_bitmap_set(bitmap, [], "after _reset")


def test_unsafe_range_set() raises:
    var bitmap = Bitmap.alloc(16)

    bitmap.unsafe_range_set(0, 10, True)
    assert_bitmap_set(bitmap, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "range 0-10")
    bitmap.unsafe_range_set(0, 10, False)
    assert_bitmap_set(bitmap, [], "reset")

    bitmap.unsafe_range_set(0, 0, True)
    assert_bitmap_set(bitmap, [], "range 0")

    var to_test = [0, 1, 7, 8, 15]
    for pos in range(len(to_test)):
        _reset(bitmap)
        var start_bit = to_test[pos]
        bitmap.unsafe_range_set(start_bit, 1, True)
        assert_bitmap_set(bitmap, [start_bit], "range  1")
        if to_test[pos] < bitmap.length() - 1:
            _reset(bitmap)
            bitmap.unsafe_range_set(start_bit, 2, True)
            assert_bitmap_set(bitmap, [start_bit, start_bit + 1], "range 2")


def test_partial_byte_set() raises:
    var bitmap = Bitmap.alloc(16)

    bitmap.unsafe_range_set(0, 0, True)
    assert_bitmap_set(bitmap, [], "range 0")

    # Set one bit to True.
    bitmap.partial_byte_set(0, 0, 1, True)
    assert_bitmap_set(bitmap, [0], "set bit 0")

    # Set one bit to False.
    bitmap.partial_byte_set(0, 0, 1, False)
    assert_bitmap_set(bitmap, [], "reset bit 0")

    # Set multiple bits to True.
    bitmap.partial_byte_set(1, 2, 5, True)
    assert_bitmap_set(bitmap, [10, 11, 12], "set multiple bits")

    # Set multiple bits to False.
    bitmap.partial_byte_set(1, 3, 5, False)
    assert_bitmap_set(bitmap, [10], "reset multiple bits")


def test_expand_bitmap() raises:
    var bitmap = Bitmap.alloc(6)
    bitmap.unsafe_set(0, True)
    bitmap.unsafe_set(5, True)
    assert_bitmap_set(bitmap, [0, 5], "initial setup")

    # Create a new bitmap with 2 bits
    var new_bitmap = Bitmap.alloc(2)
    new_bitmap.unsafe_set(0, True)

    # Expand the bitmap
    bitmap.extend(new_bitmap, 6, 2)
    assert_bitmap_set(bitmap, [0, 5, 6], "after expand")


def test_buffer_with_offset() raises:
    # Test Buffer with offset functionality
    var buf = Buffer.alloc(10)
    assert_equal(buf.offset, 0)  # Default offset should be 0

    # Set values in buffer without offset
    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 43)
    buf.unsafe_set(2, 44)

    # Create buffer with offset
    var buf_with_offset = Buffer(buf.ptr, buf.size, buf.owns, offset=2)
    assert_equal(buf_with_offset.offset, 2)

    # Test that offset affects get operations
    assert_equal(buf_with_offset.unsafe_get(0), 44)  # Should get buf[2]

    # Test that offset affects set operations
    buf_with_offset.unsafe_set(1, 99)  # Should set buf[3]
    assert_equal(buf.unsafe_get(3), 99)

    # Test offset with boolean data type - simplified test
    var buf_bool = Buffer.alloc[DType.bool](16)
    # Test basic functionality first
    buf_bool.unsafe_set[DType.bool](0, True)
    assert_true(buf_bool.unsafe_get[DType.bool](0))

    # Now test with offset - use a simple offset of 1 bit
    var buf_bool_offset = Buffer(buf_bool.ptr, buf_bool.size, False, offset=1)
    buf_bool_offset.unsafe_set[DType.bool](0, True)  # Should set buf[1]
    assert_true(buf_bool.unsafe_get[DType.bool](1))  # Check if buf[1] was set


def test_buffer_moveinit_with_offset() raises:
    # Test __moveinit__ preserves offset
    var buf = Buffer.alloc(5)
    buf.offset = 3
    buf.unsafe_set(0, 123)

    var moved_buf = buf^
    assert_equal(moved_buf.offset, 3)
    assert_equal(moved_buf.unsafe_get(0), 123)


def test_buffer_swap_with_offset() raises:
    # Test swap preserves offsets correctly
    var buf1 = Buffer.alloc(5)
    buf1.offset = 2
    buf1.unsafe_set(0, 111)

    var buf2 = Buffer.alloc(5)
    buf2.offset = 4
    buf2.unsafe_set(0, 222)

    buf1.swap(buf2)

    # After swap, buf1 should have buf2's original offset and data
    assert_equal(buf1.offset, 4)
    assert_equal(buf1.unsafe_get(0), 222)

    # And buf2 should have buf1's original offset and data
    assert_equal(buf2.offset, 2)
    assert_equal(buf2.unsafe_get(0), 111)


def test_bitmap_with_offset() raises:
    # Test Bitmap with offset functionality
    var buffer = Buffer.alloc[DType.bool](16)
    # Set some bits in the underlying buffer
    buffer.unsafe_set[DType.bool](3, True)
    buffer.unsafe_set[DType.bool](4, False)
    buffer.unsafe_set[DType.bool](5, True)
    buffer.unsafe_set[DType.bool](6, True)

    var bitmap = Bitmap(buffer^, offset=3)
    assert_equal(bitmap.offset, 3)

    # Test that offset affects get operations
    assert_true(bitmap.unsafe_get(0))  # Should get buffer[3]
    assert_false(bitmap.unsafe_get(1))  # Should get buffer[4]
    assert_true(bitmap.unsafe_get(2))  # Should get buffer[5]
    assert_true(bitmap.unsafe_get(3))  # Should get buffer[6]

    # Test that offset affects set operations
    bitmap.unsafe_set(4, True)  # Should set buffer[7]
    assert_true(bitmap.buffer.unsafe_get[DType.bool](7))


def test_bitmap_moveinit_with_offset() raises:
    # Test __moveinit__ preserves offset
    var buffer = Buffer.alloc[DType.bool](8)
    var bitmap = Bitmap(buffer^, offset=2)
    bitmap.unsafe_set(0, True)

    var moved_bitmap = bitmap^
    assert_equal(moved_bitmap.offset, 2)
    assert_true(moved_bitmap.unsafe_get(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
