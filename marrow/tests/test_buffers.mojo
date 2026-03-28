import std.math as math
from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.reflection import call_location
from std.memory import ArcPointer

from marrow.buffers import *


@always_inline
def assert_bitmap_set(
    ptr: UnsafePointer[UInt8, MutAnyOrigin],
    n_bits: Int,
    expected_true_pos: List[Int],
    message: StringLiteral,
) raises -> None:
    var list_pos = 0
    for i in range(n_bits):
        var expected_value = False
        if list_pos < len(expected_true_pos):
            if expected_true_pos[list_pos] == i:
                expected_value = True
                list_pos += 1
        var current_value = Bool((ptr[i // 8] >> UInt8(i % 8)) & 1)
        assert_equal(
            current_value,
            expected_value,
            String(
                t"{message}: Bitmap index {i} is {current_value}, expected"
                t" {expected_value} as per list position {list_pos}"
            ),
            location=call_location(),
        )


def is_aligned[
    T: AnyType
](ptr: UnsafePointer[T, MutAnyOrigin], alignment: Int) raises -> Bool:
    return (Int(ptr) % alignment) == 0


def test_buffer_init() raises:
    var b = Buffer.alloc_zeroed(10)
    assert_equal(b.size, 64)
    assert_true(is_aligned(b.ptr, 64))


def test_alloc_bits() raises:
    # DType.bool is 1 byte per element; 10 bytes → aligned to 64
    var b1 = Buffer.alloc_zeroed[DType.bool](10)
    assert_equal(b1.size, 64)

    # 513 bytes → aligned to 576
    var b2 = Buffer.alloc_zeroed[DType.bool](64 * 8 + 1)
    assert_equal(b2.size, 576)


def test_buffer_grow() raises:
    var b = Buffer.alloc_zeroed(10)
    b.unsafe_set(0, 111)
    assert_equal(b.size, 64)
    b.resize(20)
    assert_equal(b.size, 64)
    assert_equal(b.unsafe_get(0), 111)
    b.resize(80)
    assert_equal(b.size, 128)
    assert_equal(b.unsafe_get(0), 111)


def test_buffer_set_get() raises:
    var buf = Buffer.alloc_zeroed(10)
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


def test_buffer_swap() raises:
    var one = Buffer.alloc_zeroed(10)
    one.unsafe_set(0, 111)
    var two = Buffer.alloc_zeroed(10)
    two.unsafe_set(0, 222)

    swap(one, two)

    assert_equal(one.unsafe_get(0), 222)
    assert_equal(two.unsafe_get(0), 111)


def test_bitmap_get_set() raises:
    # DType.bool stores 1 byte per element (no bit-packing)
    var b = Buffer.alloc_zeroed[DType.bool](10)
    assert_equal(b.size, 64)

    assert_false(Bool(b.ptr[0]))
    b.unsafe_set[DType.bool](0, True)
    assert_true(Bool(b.ptr[0]))
    b.unsafe_set[DType.bool](1, True)
    assert_true(Bool(b.ptr[1]))

    var frozen = b.to_immutable()
    assert_true(frozen.unsafe_get[DType.bool](0))
    assert_true(frozen.unsafe_get[DType.bool](1))
    assert_false(frozen.unsafe_get[DType.bool](2))


def _reset(mut bitmap: Bitmap[mut=True], n_bits: Int) raises:
    bitmap.set_range(0, n_bits, False)
    assert_bitmap_set(bitmap.buffer.ptr, n_bits, [], "after _reset")


def test_bitmap_range_set() raises:
    var bitmap = Bitmap.alloc_zeroed(16)
    var n_bits = 16

    bitmap.set_range(0, 10, True)
    assert_bitmap_set(
        bitmap.buffer.ptr,
        n_bits,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "range 0-10",
    )
    bitmap.set_range(0, 10, False)
    assert_bitmap_set(bitmap.buffer.ptr, n_bits, [], "reset")

    bitmap.set_range(0, 0, True)
    assert_bitmap_set(bitmap.buffer.ptr, n_bits, [], "range 0")

    var to_test = [0, 1, 7, 8, 15]
    for pos in range(len(to_test)):
        _reset(bitmap, n_bits)
        var start_bit = to_test[pos]
        bitmap.set_range(start_bit, 1, True)
        assert_bitmap_set(bitmap.buffer.ptr, n_bits, [start_bit], "range  1")
        if to_test[pos] < n_bits - 1:
            _reset(bitmap, n_bits)
            bitmap.set_range(start_bit, 2, True)
            assert_bitmap_set(
                bitmap.buffer.ptr,
                n_bits,
                [start_bit, start_bit + 1],
                "range 2",
            )


def test_bitmap_extend() raises:
    var src_b = Bitmap.alloc_zeroed(6)
    src_b.set(0)
    src_b.set(5)
    var src = src_b.to_immutable()

    var dst = Bitmap.alloc_zeroed(8)
    dst.extend(src, 0, 6)
    assert_bitmap_set(dst.buffer.ptr, 8, [0, 5], "after extend")

    # extend into offset position
    var dst2 = Bitmap.alloc_zeroed(8)
    var src2_b = Bitmap.alloc_zeroed(2)
    src2_b.set(0)
    var src2 = src2_b.to_immutable()
    dst2.extend(src2, 6, 2)
    assert_bitmap_set(dst2.buffer.ptr, 8, [6], "extend at offset 6")


def test_buffer_finish() raises:
    var buf = Buffer.alloc_zeroed(10)
    buf.unsafe_set(0, 42)
    buf.unsafe_set(1, 99)

    var frozen = buf.to_immutable()
    # Reads still work on the frozen buffer.
    assert_equal(frozen.unsafe_get(0), 42)
    assert_equal(frozen.unsafe_get(1), 99)
    assert_equal(frozen.size, 64)
    assert_equal(frozen.length(), 64)


def test_buffer_cpu_kind() raises:
    """CPU buffers from Buffer[mut=True].to_immutable() have kind=CPU and are CPU-accessible.
    """
    var b = Buffer.alloc_zeroed(64)
    var frozen = b.to_immutable()
    assert_true(frozen.is_cpu())
    assert_false(frozen.is_device())
    assert_false(frozen.is_host())
    assert_equal(frozen.device_type(), Int32(1))
    assert_equal(frozen.device_id(), Int64(-1))


def test_buffer_foreign_kind() raises:
    """Foreign CPU buffers are CPU-accessible; release fires on last drop."""
    # n_released lives on the stack; its address is embedded in the buffer
    # data so the release callback can increment it before freeing the memory.
    var n_released: Int = 0
    var raw = alloc[UInt8](size_of[UnsafePointer[Int, MutAnyOrigin]]())
    raw.bitcast[UnsafePointer[Int, MutAnyOrigin]]()[0] = rebind[
        UnsafePointer[Int, MutAnyOrigin]
    ](UnsafePointer(to=n_released))

    def count_and_free(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
        var counter = ptr.bitcast[UnsafePointer[Int, MutAnyOrigin]]()[0]
        counter[0] += 1
        ptr.free()

    var mut_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](raw)
    var keeper = ArcPointer(Allocation.foreign(mut_ptr, count_and_free))
    var buf = Buffer.from_foreign(
        raw.bitcast[NoneType](),
        size_of[UnsafePointer[Int, MutAnyOrigin]](),
        keeper,
    )
    assert_true(buf.is_cpu())
    assert_false(buf.is_device())
    assert_false(buf.is_host())
    assert_equal(buf.device_type(), Int32(1))
    assert_equal(buf.device_id(), Int64(-1))
    assert_equal(n_released, 0)
    # Drop both ArcPointer copies; release fires exactly once on last drop.
    _ = buf^
    _ = keeper^
    assert_equal(n_released, 1)


def test_buffer_no_device() raises:
    # CPU-allocated buffers have no device buffer
    var buf = Buffer.alloc_zeroed(10)
    var frozen = buf.to_immutable()
    assert_false(frozen.is_device())


def test_buffer_builder_resize_noop_same_size() raises:
    # Resize to a length that maps to the same byte allocation is a no-op:
    # the pointer must not change.
    var buf = Buffer.alloc_zeroed[DType.int64](10)
    var ptr_before = buf.ptr
    buf.resize[DType.int64](10)
    assert_equal(buf.ptr, ptr_before)


def test_buffer_builder_resize_noop_same_aligned_size() raises:
    # Different element counts that map to the same aligned byte size are also
    # no-ops (alignment is 64 bytes, so 1..8 int64 elements all fit in 64 bytes).
    var buf = Buffer.alloc_zeroed[DType.int64](1)
    var ptr_before = buf.ptr
    buf.resize[DType.int64](8)  # still 64 bytes after alignment
    assert_equal(buf.ptr, ptr_before)


def test_buffer_builder_resize_reallocates_when_larger() raises:
    # A resize that requires more bytes must produce a new allocation.
    var buf = Buffer.alloc_zeroed[DType.int64](1)
    var ptr_before = buf.ptr
    buf.resize[DType.int64](9)  # 9 * 8 = 72 bytes → 128-byte aligned block
    assert_true(buf.ptr != ptr_before)


def test_bitmap_builder_resize_noop_same_capacity() raises:
    # BitmapBuilder.resize delegates to Buffer[mut=True]; same capacity is a no-op.
    var bm = Bitmap.alloc_zeroed(64)
    var ptr_before = bm.buffer.ptr
    bm.resize(64)
    assert_equal(bm.buffer.ptr, ptr_before)


def test_bitmap_builder_resize_noop_same_aligned_capacity() raises:
    # 1 and 511 bits both fit in a 64-byte block → no-op.
    var bm = Bitmap.alloc_zeroed(1)
    var ptr_before = bm.buffer.ptr
    bm.resize(511)
    assert_equal(bm.buffer.ptr, ptr_before)


def test_bitmap_builder_resize_reallocates_when_larger() raises:
    # 513 bits require a second 64-byte block → reallocation.
    var bm = Bitmap.alloc_zeroed(1)
    var ptr_before = bm.buffer.ptr
    bm.resize(513)
    assert_true(bm.buffer.ptr != ptr_before)


def test_buffer_eq_equal() raises:
    var b1 = Buffer.alloc_zeroed(10)
    var b2 = Buffer.alloc_zeroed(10)
    b1.unsafe_set(0, 42)
    b1.unsafe_set(1, 99)
    b2.unsafe_set(0, 42)
    b2.unsafe_set(1, 99)
    assert_true(b1.to_immutable() == b2.to_immutable())


def test_buffer_eq_unequal() raises:
    var b1 = Buffer.alloc_zeroed(10)
    var b2 = Buffer.alloc_zeroed(10)
    b1.unsafe_set(0, 42)
    b2.unsafe_set(0, 43)
    assert_false(b1.to_immutable() == b2.to_immutable())


def test_buffer_eq_different_size() raises:
    var b1 = Buffer.alloc_zeroed[DType.int32](10)  # 64 bytes
    var b2 = Buffer.alloc_zeroed[DType.int32](20)  # 128 bytes
    assert_false(b1.to_immutable() == b2.to_immutable())


def test_aligned_ptr_at_zero_offset() raises:
    """With offset=0, aligned_ptr_at equals ptr_at."""
    var bb = Buffer.alloc_zeroed[DType.int64](16)
    var buf = bb.to_immutable()
    var ptr = buf.ptr_at[DType.int64](0)
    var aligned = buf.aligned_ptr_at[DType.int64](0)
    assert_true(ptr == aligned)


def test_aligned_ptr_at_aligned_offset() raises:
    """Offset already on a 64-byte boundary stays unchanged."""
    # 64 bytes / 8 bytes per int64 = 8 elements per 64-byte block
    var bb = Buffer.alloc_zeroed[DType.int64](32)
    var buf = bb.to_immutable()
    var aligned = buf.aligned_ptr_at[DType.int64](8)
    var expected = buf.ptr_at[DType.int64](8)
    assert_true(aligned == expected)


def test_aligned_ptr_at_unaligned_offset() raises:
    """Offset in the middle of a 64-byte block rounds down."""
    # int64: 8 bytes each, 64/8 = 8 elements per block
    # offset=5 → byte offset=40, align_down(40,64)=0 → element 0
    var bb = Buffer.alloc_zeroed[DType.int64](32)
    var buf = bb.to_immutable()
    var aligned = buf.aligned_ptr_at[DType.int64](5)
    var expected = buf.ptr_at[DType.int64](0)
    assert_true(aligned == expected)


def test_aligned_ptr_at_second_block() raises:
    """Offset in the second 64-byte block rounds to start of that block."""
    # int32: 4 bytes each, 64/4 = 16 elements per block
    # offset=20 → byte offset=80, align_down(80,64)=64 → element 16
    var bb = Buffer.alloc_zeroed[DType.int32](64)
    var buf = bb.to_immutable()
    var aligned = buf.aligned_ptr_at[DType.int32](20)
    var expected = buf.ptr_at[DType.int32](16)
    assert_true(aligned == expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
