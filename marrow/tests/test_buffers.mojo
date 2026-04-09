from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite
from std.memory import ArcPointer

from marrow.buffers import *
from marrow.views import BufferView


# ---------------------------------------------------------------------------
# Buffer — allocation, mutation, equality, alignment
# ---------------------------------------------------------------------------


def test_buffer_init() raises:
    var b = Buffer.alloc_zeroed(10)
    assert_equal(len(b), 64)
    # alignment is a private implementation detail, tested indirectly via correct behavior


def test_alloc_bits() raises:
    var b1 = Buffer.alloc_zeroed[DType.bool](10)
    assert_equal(len(b1), 64)

    # 64*8+1 bits → needs two 64-byte blocks
    var b2 = Buffer.alloc_zeroed[DType.bool](64 * 8 + 1)
    assert_equal(len(b2), 576)


def test_buffer_grow() raises:
    var b = Buffer.alloc_zeroed(10)
    b[0] = 111
    b.resize(20)
    assert_equal(len(b), 64)  # still fits in the same 64-byte block
    assert_equal(b[0], 111)
    b.resize(80)
    assert_equal(len(b), 128)  # grew into a second block
    assert_equal(b[0], 111)  # data preserved


def test_buffer_set_get() raises:
    var buf = Buffer.alloc_zeroed(10)
    buf[0] = 42
    buf[1] = 43
    buf[2] = 44
    assert_equal(buf[0], 42)
    assert_equal(buf[1], 43)
    assert_equal(buf[2], 44)
    assert_equal(buf.length[DType.uint16](), 32)  # 64 bytes / 2 bytes each
    # reinterpreting as uint16: bytes 0+1 → 42 + (43<<8)
    assert_equal(buf.unsafe_get[DType.uint16](0), 42 + (43 << 8))
    assert_equal(buf.unsafe_get[DType.uint16](1), 44)


def test_buffer_swap() raises:
    var one = Buffer.alloc_zeroed(10)
    one[0] = 111
    var two = Buffer.alloc_zeroed(10)
    two[0] = 222
    swap(one, two)
    assert_equal(one[0], 222)
    assert_equal(two[0], 111)


def test_buffer_finish() raises:
    var buf = Buffer.alloc_zeroed(10)
    buf[0] = 42
    buf[1] = 99
    var frozen = buf^.to_immutable()
    assert_equal(frozen[0], 42)
    assert_equal(frozen[1], 99)
    assert_equal(len(frozen), 64)
    assert_equal(frozen.length(), 64)


def test_buffer_eq_equal() raises:
    var b1 = Buffer.alloc_zeroed(10)
    var b2 = Buffer.alloc_zeroed(10)
    b1[0] = 42
    b1[1] = 99
    b2[0] = 42
    b2[1] = 99
    assert_true(b1 == b2)


def test_buffer_eq_unequal() raises:
    var b1 = Buffer.alloc_zeroed(10)
    var b2 = Buffer.alloc_zeroed(10)
    b1[0] = 42
    b2[0] = 43
    assert_false(b1 == b2)


def test_buffer_eq_different_size() raises:
    var b1 = Buffer.alloc_zeroed[DType.int32](10)  # 64 bytes
    var b2 = Buffer.alloc_zeroed[DType.int32](20)  # 128 bytes
    assert_false(b1 == b2)


# ---------------------------------------------------------------------------
# Buffer — allocation kind and device presence
# ---------------------------------------------------------------------------


def test_buffer_cpu_kind() raises:
    var b = Buffer.alloc_zeroed(64)
    var frozen = b^.to_immutable()
    assert_true(frozen.is_cpu())
    assert_false(frozen.is_device())
    assert_false(frozen.is_host())
    assert_equal(frozen.device_type(), Int32(1))
    assert_equal(frozen.device_id(), Int64(-1))


def test_buffer_foreign_kind() raises:
    """Foreign CPU buffers are CPU-accessible; release fires on last drop."""
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
    assert_equal(n_released, 0)
    _ = buf^
    _ = keeper^
    assert_equal(n_released, 1)


def test_buffer_no_device() raises:
    var buf = Buffer.alloc_zeroed(10)
    assert_false(buf^.to_immutable().is_device())


# ---------------------------------------------------------------------------
# Bitmap — set / clear / test
# ---------------------------------------------------------------------------


def test_bitmap_set_clear_test() raises:
    var bm = Bitmap.alloc_zeroed(16)
    for i in range(16):
        assert_false(bm.test(i))

    bm.set(0)
    bm.set(7)
    bm.set(8)
    bm.set(15)
    for i in range(16):
        if i in (0, 7, 8, 15):
            assert_true(bm.test(i))
        else:
            assert_false(bm.test(i))

    bm.clear(7)
    assert_false(bm.test(7))
    assert_true(bm.test(0))


def test_bitmap_eq() raises:
    var bm1 = Bitmap.alloc_zeroed(8)
    var bm2 = Bitmap.alloc_zeroed(8)
    assert_true(bm1 == bm2)
    bm1.set(3)
    assert_false(bm1 == bm2)
    bm2.set(3)
    assert_true(bm1 == bm2)


# ---------------------------------------------------------------------------
# Bitmap — resize no-ops and reallocation
# ---------------------------------------------------------------------------


def test_bitmap_resize_noop_same_capacity() raises:
    var bm = Bitmap.alloc_zeroed(64)
    bm.set(0)
    bm.set(63)
    bm.resize(64)
    assert_true(bm[0])
    assert_true(bm[63])
    assert_false(bm[1])


def test_bitmap_resize_noop_same_aligned_capacity() raises:
    # 1 and 511 bits both fit in a single 64-byte block
    var bm = Bitmap.alloc_zeroed(1)
    bm.set(0)
    bm.resize(511)
    assert_true(bm[0])


def test_bitmap_resize_reallocates_when_larger() raises:
    var bm = Bitmap.alloc_zeroed(1)
    bm.set(0)
    bm.resize(513)  # needs a second 64-byte block
    assert_true(bm[0])
    assert_false(bm[512])


# ---------------------------------------------------------------------------
# Bitmap — set_range
# ---------------------------------------------------------------------------


def test_bitmap_set_range_full() raises:
    var bm = Bitmap.alloc_zeroed(16)
    bm.set_range(0, 10, True)
    for i in range(10):
        assert_true(bm.test(i))
    for i in range(10, 16):
        assert_false(bm.test(i))


def test_bitmap_set_range_clear() raises:
    var bm = Bitmap.alloc_zeroed(16)
    bm.set_range(0, 16, True)
    bm.set_range(3, 5, False)
    for i in range(16):
        if 3 <= i < 8:
            assert_false(bm.test(i))
        else:
            assert_true(bm.test(i))


def test_bitmap_set_range_empty() raises:
    var bm = Bitmap.alloc_zeroed(16)
    bm.set_range(0, 0, True)
    for i in range(16):
        assert_false(bm.test(i))


def test_bitmap_set_range_single_bits() raises:
    # Verify each boundary position: first bit, last in byte, first of next byte, last
    for start in [0, 1, 7, 8, 15]:
        var bm = Bitmap.alloc_zeroed(16)
        bm.set_range(start, 1, True)
        for i in range(16):
            if i == start:
                assert_true(bm.test(i))
            else:
                assert_false(bm.test(i))


def test_bitmap_set_range_cross_byte() raises:
    # Range starting mid-byte and ending mid-next-byte: bits 5..10
    var bm = Bitmap.alloc_zeroed(16)
    bm.set_range(5, 6, True)
    for i in range(16):
        if 5 <= i < 11:
            assert_true(bm.test(i))
        else:
            assert_false(bm.test(i))


# ---------------------------------------------------------------------------
# Bitmap — extend
# ---------------------------------------------------------------------------


def test_bitmap_extend_at_start() raises:
    var src = Bitmap.alloc_zeroed(6)
    src.set(0)
    src.set(5)

    var dst = Bitmap.alloc_zeroed(8)
    dst.extend(src.view(), 0, 6)
    assert_true(dst.test(0))
    assert_false(dst.test(1))
    assert_true(dst.test(5))
    assert_false(dst.test(6))


def test_bitmap_extend_at_offset() raises:
    var src = Bitmap.alloc_zeroed(2)
    src.set(0)

    var dst = Bitmap.alloc_zeroed(8)
    dst.extend(src.view(), 6, 2)
    for i in range(6):
        assert_false(dst.test(i))
    assert_true(dst.test(6))
    assert_false(dst.test(7))


# ---------------------------------------------------------------------------
# Bitmap — constructors
# ---------------------------------------------------------------------------


def test_bitmap_from_bool_list() raises:
    var bm = Bitmap([True, False, True, False, True, False, True, False])
    assert_equal(len(bm), 8)
    assert_true(bm[0])
    assert_false(bm[1])
    assert_true(bm[2])
    assert_false(bm[3])
    assert_true(bm[6])
    assert_false(bm[7])


def test_bitmap_from_length_and_indices() raises:
    # Construct a bitmap of length 8 with bits 1, 4, 6 set
    var bm = Bitmap(8, [1, 4, 6])
    assert_false(bm[0])
    assert_true(bm[1])
    assert_false(bm[2])
    assert_false(bm[3])
    assert_true(bm[4])
    assert_false(bm[5])
    assert_true(bm[6])
    assert_false(bm[7])


def test_bitmap_setitem() raises:
    var bm = Bitmap.alloc_zeroed(8)
    bm[0] = True
    bm[3] = True
    bm[7] = True
    assert_true(bm[0])
    assert_false(bm[1])
    assert_true(bm[3])
    assert_true(bm[7])
    bm[0] = False
    assert_false(bm[0])


def test_bitmap_resize_truncates_length() raises:
    var bm = Bitmap.alloc_zeroed(20)
    bm.set_range(0, 20, True)
    bm.resize(15)
    assert_equal(len(bm), 15)


# ---------------------------------------------------------------------------
# Bitmap — slice
# ---------------------------------------------------------------------------


def test_bitmap_slice_basic() raises:
    var bm = Bitmap.alloc_zeroed(16)
    bm[0] = True
    bm[3] = True
    bm[7] = True
    # slice starting at bit 4, length 8
    var s = bm.slice(4, 8)
    assert_equal(len(s), 8)
    assert_false(s[0])  # original bit 4
    assert_false(s[1])  # original bit 5
    assert_false(s[2])  # original bit 6
    assert_true(s[3])  # original bit 7


def test_bitmap_slice_single_bit() raises:
    var bm = Bitmap(8, [3])
    var s = bm.slice(3, 1)
    assert_equal(len(s), 1)
    assert_true(s[0])


def test_bitmap_slice_preserves_pattern() raises:
    # slice(2, 8) over a 10-bit bitmap with bits 2, 5, 9 set
    var bm = Bitmap(10, [2, 5, 9])
    var s = bm.slice(2, 8)
    assert_equal(len(s), 8)
    assert_true(s[0])  # original bit 2
    assert_false(s[1])  # original bit 3
    assert_false(s[2])  # original bit 4
    assert_true(s[3])  # original bit 5


# ---------------------------------------------------------------------------
# Bitmap — count_set_bits (SIMD popcount correctness)
# ---------------------------------------------------------------------------


def _count_naive(bm: BitmapView[_]) -> Int:
    """Bit-by-bit reference popcount."""
    var n = 0
    for i in range(len(bm)):
        if bm[i]:
            n += 1
    return n


def test_bitmap_count_set_bits_known_pattern() raises:
    # bits 0, 3, 7, 8 → 4 set bits
    var bm = Bitmap(16, [0, 3, 7, 8])
    assert_true(bm[0])
    assert_true(bm[3])
    assert_true(bm[7])
    assert_true(bm[8])
    assert_equal(bm.view().count_set_bits(), 4)


def test_bitmap_count_set_bits_small_slice_in_large_bitmap() raises:
    # Tiny slice deep inside a large all-ones bitmap
    var b = Bitmap.alloc_zeroed(2000)
    b.set_range(0, 2000, True)
    var full = b^.to_immutable()
    var sliced = full.slice(1003, 5)
    assert_equal(sliced.count_set_bits(), 5)


def test_bitmap_count_set_bits_vs_naive() raises:
    """Exhaustive check of count_set_bits vs naive popcount across sizes/offsets/patterns.
    """
    comptime sizes = (1, 7, 13, 63, 64, 65, 127, 512, 513, 1023, 1024, 4097)
    comptime offsets = (0, 3, 7, 32 << 3, 96 << 3, 128 << 3)

    comptime for si in range(len(sizes)):
        comptime size = sizes[si]
        comptime for oi in range(len(offsets)):
            comptime offset = offsets[oi]
            comptime total = size + offset

            # all-zeros
            var bz = Bitmap.alloc_zeroed(total)
            var sz = bz^.to_immutable().slice(offset, size)
            assert_equal(sz.count_set_bits(), _count_naive(sz))

            # all-ones
            var bo = Bitmap.alloc_zeroed(total)
            bo.set_range(0, total, True)
            var so = bo^.to_immutable().slice(offset, size)
            assert_equal(so.count_set_bits(), _count_naive(so))

            # alternating (even bits set)
            var ba = Bitmap.alloc_zeroed(total)
            var k = 0
            while k < total:
                ba.set(k)
                k += 2
            var sa = ba^.to_immutable().slice(offset, size)
            assert_equal(sa.count_set_bits(), _count_naive(sa))


def test_bitmap_count_set_bits_interior_slices() raises:
    """`count_set_bits` on slices ending before buffer end (trailing bytes with real data).
    """
    comptime sizes = (1, 7, 13, 63, 64, 65, 127, 512, 513)
    comptime offsets = (0, 3, 7, 32 << 3, 96 << 3, 128 << 3)
    comptime extra = 512

    comptime for si in range(len(sizes)):
        comptime size = sizes[si]
        comptime for oi in range(len(offsets)):
            comptime offset = offsets[oi]
            comptime total = size + offset + extra

            # all-zeros
            var bz = Bitmap.alloc_zeroed(total)
            var sz = bz^.to_immutable().slice(offset, size)
            assert_equal(sz.count_set_bits(), _count_naive(sz))

            # all-ones: trailing bytes contain 1s — must be masked away
            var bo = Bitmap.alloc_zeroed(total)
            bo.set_range(0, total, True)
            var so = bo^.to_immutable().slice(offset, size)
            assert_equal(so.count_set_bits(), _count_naive(so))

            # alternating
            var ba = Bitmap.alloc_zeroed(total)
            var k = 0
            while k < total:
                ba.set(k)
                k += 2
            var sa = ba^.to_immutable().slice(offset, size)
            assert_equal(sa.count_set_bits(), _count_naive(sa))


def test_bitmap_count_set_bits_exact_boundary() raises:
    # 512 bits = 64 bytes, no lead or trail correction needed
    var b = Bitmap.alloc_zeroed(512)
    b.set_range(0, 512, True)
    assert_equal(b^.to_immutable().view().count_set_bits(), 512)


def test_bitmap_count_set_bits_trail_bytes_only() raises:
    # 8-bit slice at offset 0 in a large all-ones bitmap (byte_end=1, aligned_end=64)
    var full = Bitmap.alloc_zeroed(1000)
    full.set_range(0, 1000, True)
    var s = full^.to_immutable().slice(0, 8)
    assert_equal(s.count_set_bits(), 8)


def test_bitmap_count_set_bits_lead_and_trail() raises:
    # Slice [520, 530) inside a 1000-bit all-ones bitmap
    var full = Bitmap.alloc_zeroed(1000)
    full.set_range(0, 1000, True)
    var s = full^.to_immutable().slice(520, 10)
    assert_equal(s.count_set_bits(), 10)
    assert_equal(s.count_set_bits(), _count_naive(s))


# ---------------------------------------------------------------------------
# BitmapView — boolean operations
# ---------------------------------------------------------------------------


def test_bitmapview_eq_equal() raises:
    var a = Bitmap(8, [0, 3, 7])
    var b = Bitmap(8, [0, 3, 7])
    assert_true(a.view() == b.view())


def test_bitmapview_eq_unequal() raises:
    var a = Bitmap(8, [0, 3, 7])
    var b = Bitmap(8, [0, 3, 6])
    assert_false(a.view() == b.view())


def test_bitmapview_eq_different_length() raises:
    var a = Bitmap(8, [0, 3])
    var b = Bitmap(9, [0, 3])
    assert_false(a.view() == b.view())


def test_bitmapview_eq_slices() raises:
    var base_a = Bitmap(10, [2, 5, 9])
    var base_b = Bitmap(10, [2, 5, 9])
    assert_true(base_a.slice(2, 8) == base_b.slice(2, 8))
    var base_c = Bitmap(10, [3, 5, 9])
    assert_false(base_a.slice(2, 8) == base_c.slice(2, 8))


def test_bitmapview_and() raises:
    var a = Bitmap(8, [0, 2, 4, 6])
    var b = Bitmap(8, [0, 1, 4, 5])
    var r = a.view() & b.view()
    assert_true(r[0])
    assert_false(r[1])
    assert_false(r[2])
    assert_false(r[3])
    assert_true(r[4])
    assert_false(r[5])
    assert_false(r[6])
    assert_false(r[7])


def test_bitmapview_or() raises:
    var a = Bitmap(8, [0, 2])
    var b = Bitmap(8, [1, 2])
    var r = a.view() | b.view()
    assert_true(r[0])
    assert_true(r[1])
    assert_true(r[2])
    assert_false(r[3])


def test_bitmapview_xor() raises:
    var a = Bitmap(4, [0, 2])
    var b = Bitmap(4, [0, 1])
    var r = a.view() ^ b.view()
    assert_false(r[0])
    assert_true(r[1])
    assert_true(r[2])
    assert_false(r[3])


def test_bitmapview_invert() raises:
    var bm = Bitmap(8, [])
    var inv = ~bm.view()
    for i in range(8):
        assert_true(inv[i])


def test_bitmapview_invert_pattern() raises:
    # bits 1, 3, 5 set → inverted: 0, 2, 4, 6, 7 set
    var bm = Bitmap(8, [1, 3, 5])
    var inv = ~bm.view()
    assert_true(inv[0])
    assert_false(inv[1])
    assert_true(inv[2])
    assert_false(inv[3])
    assert_true(inv[4])
    assert_false(inv[5])
    assert_true(inv[6])
    assert_true(inv[7])


def test_bitmapview_invert_no_bleed_past_length() raises:
    """~view must not set bits beyond _len (non-byte-aligned length)."""
    var bm = Bitmap(10, [])  # 10 bits, all clear
    var inv = ~bm.view()
    assert_equal(inv.view().count_set_bits(), 10)


def test_bitmapview_invert_with_offset() raises:
    """~view on a non-zero-offset slice inverts only the slice bits."""
    var bm = Bitmap(16, [4, 5, 6, 7])
    var s = bm.slice(4, 8)  # bits 4-11 → [1,1,1,1,0,0,0,0]
    var inv = ~s
    assert_false(inv[0])
    assert_false(inv[1])
    assert_false(inv[2])
    assert_false(inv[3])
    assert_true(inv[4])
    assert_true(inv[5])
    assert_true(inv[6])
    assert_true(inv[7])


# ---------------------------------------------------------------------------
# Bitmap — zero-length
# ---------------------------------------------------------------------------


def test_bitmap_empty() raises:
    var b = Bitmap.alloc_zeroed(0)
    assert_equal(len(b), 0)


# ---------------------------------------------------------------------------
# BitmapView — difference (AND-NOT)
# ---------------------------------------------------------------------------


def test_bitmapview_difference_basic() raises:
    # [1,0,1,0] & ~[1,1,0,0] = [0,0,1,0]
    var a = Bitmap(4, [0, 2])
    var b = Bitmap(4, [0, 1])
    var r = a.view().difference(b.view())
    assert_false(r[0])
    assert_false(r[1])
    assert_true(r[2])
    assert_false(r[3])


def test_bitmapview_difference_identity() raises:
    # a & ~(all-zeros) == a
    var a = Bitmap(8, [0, 3, 7])
    var zeros = Bitmap(8, [])
    var r = a.view().difference(zeros.view())
    for i in range(8):
        assert_equal(r[i], a[i])


def test_bitmapview_difference_annihilator() raises:
    # a & ~a == all-zeros
    var a = Bitmap(8, [0, 3, 7])
    var b = Bitmap(8, [0, 3, 7])
    var r = a.view().difference(b.view())
    for i in range(8):
        assert_false(r[i])


# ---------------------------------------------------------------------------
# BitmapView — binary ops on sliced (offset) views
# ---------------------------------------------------------------------------


def test_bitmapview_and_with_offset() raises:
    """AND of two sliced bitmaps sharing the same non-zero sub-byte offset."""
    var bm_a = Bitmap(16, [2, 3, 4, 6])
    var bm_b = Bitmap(16, [2, 3, 4, 6])
    var a = bm_a.slice(2, 6)  # logical [0..6) maps to original [2..8)
    var b = bm_b.slice(2, 6)  # same data, different allocation: a & a == a
    var r = a & b
    for i in range(6):
        assert_equal(r[i], a[i])


def test_bitmapview_or_with_offset() raises:
    """OR of two sliced bitmaps with a non-zero offset."""
    var fa = Bitmap(16, [3, 5])
    var fb = Bitmap(16, [3, 4])
    var a = fa.slice(3, 5)  # slice indices 0, 2 set
    var b = fb.slice(3, 5)  # slice indices 0, 1 set
    var r = a | b
    assert_true(r[0])  # in both
    assert_true(r[1])  # only b
    assert_true(r[2])  # only a
    assert_false(r[3])
    assert_false(r[4])


def test_bitmapview_xor_with_offset() raises:
    """XOR of two sliced bitmaps with the same non-zero offset."""
    var fa = Bitmap(16, [3, 5])
    var fb = Bitmap(16, [3, 4])
    var a = fa.slice(3, 5)  # slice indices 0, 2 set
    var b = fb.slice(3, 5)  # slice indices 0, 1 set
    var r = a ^ b
    assert_false(r[0])  # 1 ^ 1 = 0
    assert_true(r[1])  # 0 ^ 1 = 1
    assert_true(r[2])  # 1 ^ 0 = 1
    assert_false(r[3])
    assert_false(r[4])


# ---------------------------------------------------------------------------
# BitmapView — length-mismatch raises
# ---------------------------------------------------------------------------


def test_bitmapview_and_length_mismatch_raises() raises:
    var a = Bitmap(8, [0, 2])
    var b = Bitmap(4, [0, 2])
    var raised = False
    try:
        _ = a.view() & b.view()
    except:
        raised = True
    assert_true(raised)


def test_bitmapview_or_length_mismatch_raises() raises:
    var a = Bitmap(8, [0, 2])
    var b = Bitmap(4, [0])
    var raised = False
    try:
        _ = a.view() | b.view()
    except:
        raised = True
    assert_true(raised)


def test_bitmapview_xor_length_mismatch_raises() raises:
    var a = Bitmap(8, [0, 2])
    var b = Bitmap(4, [0])
    var raised = False
    try:
        _ = a.view() ^ b.view()
    except:
        raised = True
    assert_true(raised)


def test_bitmapview_difference_length_mismatch_raises() raises:
    var a = Bitmap(8, [0, 2])
    var b = Bitmap(4, [0])
    var raised = False
    try:
        _ = a.view().difference(b.view())
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.run[__functions_in_module()]()
