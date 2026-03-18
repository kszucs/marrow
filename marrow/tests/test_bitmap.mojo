"""Tests for Bitmap and BitmapBuilder."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.bitmap import Bitmap, BitmapBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(n_bits: Int, set_bits: List[Int]) -> Bitmap:
    """Build a Bitmap with exactly the specified bits set."""
    var b = BitmapBuilder.alloc(n_bits)
    for i in range(len(set_bits)):
        b.set_bit(set_bits[i], True)
    return b.finish(n_bits)


def _count_naive(bm: Bitmap) -> Int:
    """Reference popcount: walks every bit via is_valid."""
    var n = 0
    for i in range(len(bm)):
        if bm.is_valid(i):
            n += 1
    return n


# ---------------------------------------------------------------------------
# BitmapBuilder
# ---------------------------------------------------------------------------


def test_builder_alloc_zero_fills() raises:
    """Freshly-allocated builder has all bits cleared."""
    var b = BitmapBuilder.alloc(10)
    var bm = b.finish(10)
    for i in range(10):
        assert_false(bm.is_valid(i))


def test_builder_set_bit_true() raises:
    var b = BitmapBuilder.alloc(8)
    b.set_bit(0, True)
    b.set_bit(3, True)
    b.set_bit(7, True)
    var bm = b.finish(8)
    assert_true(bm.is_valid(0))
    assert_false(bm.is_valid(1))
    assert_false(bm.is_valid(2))
    assert_true(bm.is_valid(3))
    assert_false(bm.is_valid(4))
    assert_false(bm.is_valid(5))
    assert_false(bm.is_valid(6))
    assert_true(bm.is_valid(7))


def test_builder_set_bit_false_clears() raises:
    var b = BitmapBuilder.alloc(8)
    b.set_bit(0, True)
    b.set_bit(1, True)
    b.set_bit(1, False)
    var bm = b.finish(8)
    assert_true(bm.is_valid(0))
    assert_false(bm.is_valid(1))


def test_builder_set_range_all_true() raises:
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    var bm = b.finish(16)
    for i in range(16):
        assert_true(bm.is_valid(i))


def test_builder_set_range_partial() raises:
    # set_range sets exactly the requested range, leaving the rest unchanged
    var b = BitmapBuilder.alloc(16)
    b.set_range(4, 8, True)  # bits 4-11 set
    var bm = b.finish(16)
    for i in range(4):
        assert_false(bm.is_valid(i))
    for i in range(4, 12):
        assert_true(bm.is_valid(i))
    for i in range(12, 16):
        assert_false(bm.is_valid(i))


def test_builder_set_range_clear() raises:
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    b.set_range(3, 5, False)  # clear bits 3-7
    var bm = b.finish(16)
    for i in range(3):
        assert_true(bm.is_valid(i))
    for i in range(3, 8):
        assert_false(bm.is_valid(i))
    for i in range(8, 16):
        assert_true(bm.is_valid(i))


def test_builder_set_range_zero_length() raises:
    var b = BitmapBuilder.alloc(8)
    b.set_range(0, 0, True)
    var bm = b.finish(8)
    for i in range(8):
        assert_false(bm.is_valid(i))


def test_builder_extend() raises:
    # extend copies bits from a Bitmap into the builder at dst_start
    var src_b = BitmapBuilder.alloc(6)
    src_b.set_bit(0, True)
    src_b.set_bit(5, True)
    var src = src_b.finish(6)

    var dst = BitmapBuilder.alloc(8)
    dst.extend(src, 0, 6)
    var bm = dst.finish(8)
    assert_true(bm.is_valid(0))
    assert_false(bm.is_valid(1))
    assert_false(bm.is_valid(4))
    assert_true(bm.is_valid(5))
    assert_false(bm.is_valid(6))
    assert_false(bm.is_valid(7))


def test_builder_extend_with_offset() raises:
    # extend into a non-zero dst_start position
    var src_b = BitmapBuilder.alloc(2)
    src_b.set_bit(0, True)
    var src = src_b.finish(2)

    var dst = BitmapBuilder.alloc(8)
    dst.extend(src, 6, 2)  # copy 2 bits starting at dst bit 6
    var bm = dst.finish(8)
    for i in range(6):
        assert_false(bm.is_valid(i))
    assert_true(bm.is_valid(6))
    assert_false(bm.is_valid(7))


def test_builder_finish_length() raises:
    var b = BitmapBuilder.alloc(20)
    b.set_range(0, 20, True)
    var bm = b.finish(15)  # finish with fewer bits than allocated
    assert_equal(len(bm), 15)


# ---------------------------------------------------------------------------
# Bitmap — basic queries
# ---------------------------------------------------------------------------


def test_len() raises:
    var bm = _make(13, [])
    assert_equal(len(bm), 13)


def test_is_valid_and_is_null() raises:
    var bm = _make(8, [1, 4, 6])
    assert_false(bm.is_valid(0))
    assert_true(bm.is_valid(1))
    assert_false(bm.is_valid(2))
    assert_false(bm.is_valid(3))
    assert_true(bm.is_valid(4))
    assert_false(bm.is_valid(5))
    assert_true(bm.is_valid(6))
    assert_false(bm.is_valid(7))
    # is_null is the inverse
    assert_true(bm.is_null(0))
    assert_false(bm.is_null(1))


# ---------------------------------------------------------------------------
# count_set_bits
# ---------------------------------------------------------------------------


def test_count_set_bits_empty() raises:
    var b = BitmapBuilder.alloc(0)
    var bm = b.finish(0)
    assert_equal(bm.count_set_bits(), 0)


def test_count_set_bits_none_set() raises:
    var bm = _make(16, [])
    assert_equal(bm.count_set_bits(), 0)


def test_count_set_bits_all_set() raises:
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    var bm = b.finish(16)
    assert_equal(bm.count_set_bits(), 16)


def test_count_set_bits_known_pattern() raises:
    # bits 0, 3, 7, 8, 15 → 5 set bits
    var bm = _make(16, [0, 3, 7, 8, 15])
    assert_equal(bm.count_set_bits(), 5)


def test_count_set_bits_partial_last_byte() raises:
    # 10-bit bitmap: bits 0-7 all set, bits 8-9 both set
    var b = BitmapBuilder.alloc(10)
    b.set_range(0, 10, True)
    var bm = b.finish(10)
    assert_equal(bm.count_set_bits(), 10)


def test_count_set_bits_with_offset() raises:
    # count_set_bits on a sliced bitmap respects _offset
    var b = BitmapBuilder.alloc(16)
    b.set_range(0, 16, True)
    var full = b.finish(16)
    # slice starting at bit 4, length 8 → all 8 bits set
    var sliced = full.slice(4, 8)
    assert_equal(sliced.count_set_bits(), 8)


def test_count_set_bits_large() raises:
    # exercises the SIMD loop for multi-SIMD-width bitmaps
    var b = BitmapBuilder.alloc(1024)
    b.set_range(0, 512, True)  # first half set
    var bm = b.finish(1024)
    assert_equal(bm.count_set_bits(), 512)


def test_count_set_bits_large_offset_byte_aligned() raises:
    """Count bits on a slice with a large byte-aligned offset (> 64 bytes).

    The aligned range from _simd_offset_range will include extra leading
    bytes that must not be counted.
    """
    # 800 bits all set, then slice at bit 576 (byte 72) for 64 bits.
    var b = BitmapBuilder.alloc(800)
    b.set_range(0, 800, True)
    var full = b.finish(800)
    var sliced = full.slice(576, 64)
    assert_equal(sliced.count_set_bits(), 64)


def test_count_set_bits_large_offset_with_shift() raises:
    """Count bits on a slice with a large offset AND sub-byte shift.

    Offset 577 → byte 72, shift 1. The aligned range starts at byte 64,
    so there are 8 leading bytes of garbage to exclude.
    """
    var b = BitmapBuilder.alloc(800)
    b.set_range(0, 800, True)
    var full = b.finish(800)
    var sliced = full.slice(577, 48)
    assert_equal(sliced.count_set_bits(), 48)


def test_count_set_bits_large_offset_sparse() raises:
    """Count bits with large offset where surrounding bytes have set bits.

    Ensures the count only includes bits within the slice, not the
    extra bytes pulled in by 64-byte alignment.
    """
    # Set every bit in a 1000-bit bitmap, then slice a narrow window.
    var b = BitmapBuilder.alloc(1000)
    b.set_range(0, 1000, True)
    var full = b.finish(1000)
    # Slice at bit 700 (byte 87, shift 4), length 10.
    # Aligned range starts at byte 64, so ~23 leading bytes of all-ones
    # must be excluded from the count.
    var sliced = full.slice(700, 10)
    assert_equal(sliced.count_set_bits(), 10)


def test_count_set_bits_large_offset_none_set() raises:
    """Count on a slice where only surrounding bytes have bits set.

    Verifies zero count when the slice itself has no bits set but
    the aligned padding region does.
    """
    # Set bits around but not in the slice window.
    var b = BitmapBuilder.alloc(1000)
    b.set_range(0, 500, True)  # bits 0-499 set
    b.set_range(520, 480, True)  # bits 520-999 set
    var full = b.finish(1000)
    # Slice at bit 500, length 20 → all clear.
    var sliced = full.slice(500, 20)
    assert_equal(sliced.count_set_bits(), 0)


def test_count_set_bits_small_slice_in_large_bitmap() raises:
    """Tiny slice (< 1 SIMD width) deep inside a large all-ones bitmap.

    Both lead and trail corrections must fire, and the aligned range
    includes many full bytes on both sides.
    """
    var b = BitmapBuilder.alloc(2000)
    b.set_range(0, 2000, True)
    var full = b.finish(2000)
    # Slice at bit 1003 (byte 125, shift 3), length 5.
    var sliced = full.slice(1003, 5)
    assert_equal(sliced.count_set_bits(), 5)


def test_count_set_bits_vs_naive_all_patterns() raises:
    """Compare count_set_bits against _count_naive across sizes and offsets.

    Covers:
      - sizes that are sub-word, exactly one Tier-2 block, straddle a
        Tier-1 boundary, and large enough to exercise multiple Tier-1 blocks
      - offsets 0 (byte-aligned), 32 (64-byte-unaligned), 128 (64-byte-aligned)
      - three fill patterns: all-zeros, all-ones, alternating
    """
    comptime sizes = (
        1,
        7,
        13,
        63,
        64,
        65,
        127,
        512,
        513,
        1023,
        1024,
        4097,
        10000,
    )
    comptime offsets = (
        0,
        3,
        7,
        32 << 3,
        96 << 3,
        128 << 3,
        65 * 8 + 3,
        96 * 8 + 5,
    )

    comptime for si in range(len(sizes)):
        comptime size = sizes[si]
        comptime for oi in range(len(offsets)):
            comptime offset = offsets[oi]
            comptime total = size + offset

            # all-zeros
            var bz = BitmapBuilder.alloc(total)
            var fz = bz.finish(total)
            var sz = fz.slice(offset, size)
            assert_equal(sz.count_set_bits(), _count_naive(sz))

            # all-ones
            var bo = BitmapBuilder.alloc(total)
            bo.set_range(0, total, True)
            var fo = bo.finish(total)
            var so = fo.slice(offset, size)
            assert_equal(so.count_set_bits(), _count_naive(so))

            # alternating (even bits set)
            var ba = BitmapBuilder.alloc(total)
            var k = 0
            while k < total:
                ba.set_bit(k, True)
                k += 2
            var fa = ba.finish(total)
            var sa = fa.slice(offset, size)
            assert_equal(sa.count_set_bits(), _count_naive(sa))


def test_count_set_bits_interior_slices() raises:
    """Test count_set_bits on slices that end BEFORE the buffer end.

    The aligned range includes trailing bytes with real data from the larger
    buffer, exercising the trail_bytes correction in trail_bits.
    The leading bytes before the slice also contain real data, exercising
    the lead_bytes correction.
    """
    comptime sizes = (1, 7, 13, 63, 64, 65, 127, 512, 513, 1023, 1024)
    comptime offsets = (
        0,
        3,
        7,
        32 << 3,
        96 << 3,
        128 << 3,
        65 * 8 + 3,
        96 * 8 + 5,
    )
    # Extra bits after the slice window to ensure non-zero trailing bytes.
    comptime extra = 512

    comptime for si in range(len(sizes)):
        comptime size = sizes[si]
        comptime for oi in range(len(offsets)):
            comptime offset = offsets[oi]
            comptime total = size + offset + extra

            # all-zeros
            var bz = BitmapBuilder.alloc(total)
            var fz = bz.finish(total)
            var sz = fz.slice(offset, size)
            assert_equal(sz.count_set_bits(), _count_naive(sz))

            # all-ones: trailing bytes contain 1s, must be subtracted
            var bo = BitmapBuilder.alloc(total)
            bo.set_range(0, total, True)
            var fo = bo.finish(total)
            var so = fo.slice(offset, size)
            assert_equal(so.count_set_bits(), _count_naive(so))

            # alternating (even bits set)
            var ba = BitmapBuilder.alloc(total)
            var k = 0
            while k < total:
                ba.set_bit(k, True)
                k += 2
            var fa = ba.finish(total)
            var sa = fa.slice(offset, size)
            assert_equal(sa.count_set_bits(), _count_naive(sa))


def test_count_set_bits_trail_bits_exact_boundary() raises:
    # trail_bits == 0 when bit_end lands exactly on a 64-byte boundary
    # 512 bits = 64 bytes, exact fit: no lead or trail correction needed.
    var b = BitmapBuilder.alloc(512)
    b.set_range(0, 512, True)
    var bm = b.finish(512)
    assert_equal(bm.count_set_bits(), 512)
    # Slice ending exactly at byte 64 within a larger buffer.
    var large = BitmapBuilder.alloc(1024)
    large.set_range(0, 1024, True)
    var fl = large.finish(1024)
    var s = fl.slice(0, 512)
    assert_equal(s.count_set_bits(), 512)


def test_count_set_bits_trail_bytes_only() raises:
    # trail_bytes > 0, trail_sub_byte == 0: byte-aligned end, non-zero trailing bytes
    # Slice of 8 bits (1 byte) at offset 0 in a large all-ones bitmap.
    # byte_end=1, aligned_end=64, trail_bytes=63, trail_sub_byte=0.
    var full = BitmapBuilder.alloc(1000)
    full.set_range(0, 1000, True)
    var bm = full.finish(1000)
    var s = bm.slice(0, 8)
    assert_equal(s.count_set_bits(), 8)


def test_count_set_bits_lead_and_trail_bytes_nonzero() raises:
    """Both lead_bytes > 0 and trail_bytes > 0, with real data in both regions.
    """
    # Slice [520, 530) inside a 1000-bit all-ones bitmap.
    # offset=520: byte 65, bit 0 → aligned_start=64, lead_bytes=1, in_byte_bits=0.
    # bit_end=530: byte_end=67, aligned_end=128, trail_bytes=61, trail_sub_byte=2.
    var full = BitmapBuilder.alloc(1000)
    full.set_range(0, 1000, True)
    var bm = full.finish(1000)
    var s = bm.slice(520, 10)
    assert_equal(s.count_set_bits(), 10)
    assert_equal(s.count_set_bits(), _count_naive(s))


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------


def test_slice_shares_buffer() raises:
    # slice returns a zero-copy view with correct offset and length
    var bm = _make(16, [0, 5, 10, 15])
    var s = bm.slice(4, 8)  # bits 4-11 of original
    assert_equal(len(s), 8)
    # bit 5 of original = index 1 in sliced view
    assert_true(s.is_valid(1))
    # bit 10 of original = index 6 in sliced view
    assert_true(s.is_valid(6))
    # bit 4 of original = index 0, not set
    assert_false(s.is_valid(0))


def test_slice_single_bit() raises:
    var bm = _make(8, [3])
    var s = bm.slice(3, 1)
    assert_equal(len(s), 1)
    assert_true(s.is_valid(0))


def test_slice_count_set_bits() raises:
    var bm = _make(16, [2, 3, 4, 5, 6])
    var s = bm.slice(2, 5)  # bits 2-6 → all 5 set
    assert_equal(s.count_set_bits(), 5)


# ---------------------------------------------------------------------------
# __invert__
# ---------------------------------------------------------------------------


def test_invert_all_zeros() raises:
    var bm = _make(8, [])
    var inv = ~bm
    assert_equal(len(inv), 8)
    for i in range(8):
        assert_true(inv.is_valid(i))


def test_invert_all_ones() raises:
    var b = BitmapBuilder.alloc(8)
    b.set_range(0, 8, True)
    var bm = b.finish(8)
    var inv = ~bm
    for i in range(8):
        assert_false(inv.is_valid(i))


def test_invert_pattern() raises:
    # bits 1, 3, 5 set → inverted: 0, 2, 4, 6, 7 set
    var bm = _make(8, [1, 3, 5])
    var inv = ~bm
    assert_true(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_true(inv.is_valid(2))
    assert_false(inv.is_valid(3))
    assert_true(inv.is_valid(4))
    assert_false(inv.is_valid(5))
    assert_true(inv.is_valid(6))
    assert_true(inv.is_valid(7))


def test_invert_does_not_bleed_past_length() raises:
    """Bits beyond _length must be 0 in the result (no spurious set bits)."""
    var bm = _make(10, [])  # 10 bits, all clear
    var inv = ~bm
    # only bits 0-9 are inverted; bits 10-15 of last byte must stay 0
    assert_equal(inv.count_set_bits(), 10)


# ---------------------------------------------------------------------------
# __and__
# ---------------------------------------------------------------------------


def test_and_basic() raises:
    # [1,0,1,0,1,0,1,0] & [1,1,0,0,1,1,0,0] = [1,0,0,0,1,0,0,0]
    var a = _make(8, [0, 2, 4, 6])
    var b = _make(8, [0, 1, 4, 5])
    var r = a & b
    assert_equal(len(r), 8)
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_false(r.is_valid(3))
    assert_true(r.is_valid(4))
    assert_false(r.is_valid(5))
    assert_false(r.is_valid(6))
    assert_false(r.is_valid(7))


def test_and_identity() raises:
    # a & all-ones == a
    var a = _make(16, [1, 5, 9, 13])
    var ones_b = BitmapBuilder.alloc(16)
    ones_b.set_range(0, 16, True)
    var ones = ones_b.finish(16)
    var r = a & ones
    for i in range(16):
        assert_equal(r.is_valid(i), a.is_valid(i))


def test_and_annihilator() raises:
    # a & all-zeros == all-zeros
    var a = _make(16, [1, 5, 9, 13])
    var zeros = _make(16, [])
    var r = a & zeros
    for i in range(16):
        assert_false(r.is_valid(i))


def test_and_large() raises:
    # exercises the SIMD loop
    var b1 = BitmapBuilder.alloc(1024)
    b1.set_range(0, 512, True)
    var a = b1.finish(1024)

    var b2 = BitmapBuilder.alloc(1024)
    b2.set_range(256, 512, True)
    var b = b2.finish(1024)

    var r = a & b
    assert_equal(r.count_set_bits(), 256)  # overlap in bits 256-511


# ---------------------------------------------------------------------------
# __or__
# ---------------------------------------------------------------------------


def test_or_basic() raises:
    var a = _make(8, [0, 2])
    var b = _make(8, [1, 2])
    var r = a | b
    assert_true(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))


def test_or_idempotent() raises:
    # a | a == a
    var a = _make(16, [0, 3, 7, 10])
    var r = a | a
    for i in range(16):
        assert_equal(r.is_valid(i), a.is_valid(i))


# ---------------------------------------------------------------------------
# __xor__
# ---------------------------------------------------------------------------


def test_xor_basic() raises:
    # [1,0,1,0] ^ [1,1,0,0] = [0,1,1,0]
    var a = _make(4, [0, 2])
    var b = _make(4, [0, 1])
    var r = a ^ b
    assert_false(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))


def test_xor_self_is_zero() raises:
    # a ^ a == all-zeros
    var a = _make(16, [1, 3, 5, 7])
    var r = a ^ a
    for i in range(16):
        assert_false(r.is_valid(i))


# ---------------------------------------------------------------------------
# and_not
# ---------------------------------------------------------------------------


def test_and_not_basic() raises:
    # [1,0,1,0] & ~[1,1,0,0] = [1,0,1,0] & [0,0,1,1] = [0,0,1,0]
    var a = _make(4, [0, 2])
    var b = _make(4, [0, 1])
    var r = a.and_not(b)
    assert_false(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))


def test_and_not_with_none_mask() raises:
    # a.and_not(all-zeros) == a
    var a = _make(8, [0, 3, 7])
    var zeros = _make(8, [])
    var r = a.and_not(zeros)
    for i in range(8):
        assert_equal(r.is_valid(i), a.is_valid(i))


# ---------------------------------------------------------------------------
# Fallback path (non-zero offset)
# ---------------------------------------------------------------------------


def test_and_with_same_nonzero_offset() raises:
    """Binary ops on sliced bitmaps sharing the same non-zero sub-byte offset.
    """
    var full = _make(16, [2, 3, 4, 6, 10, 11, 12, 14])
    var a = full.slice(2, 8)  # bits 2-9 of original: [1,1,1,0,1,0,0,0]
    var b = full.slice(2, 8)  # same slice
    var r = a & b
    # a & a == a
    for i in range(8):
        assert_equal(r.is_valid(i), a.is_valid(i))


def test_and_same_shift_fast_path() raises:
    """Bitmaps with identical non-zero sub-byte offsets use the same-shift SIMD path.
    """
    # Build two 12-bit bitmaps with known patterns; slice both at offset=3
    # so both have sub-byte shift = 3 (same shift, non-zero).
    var fa = _make(16, [3, 5, 7, 9, 11])  # bits 3,5,7,9,11 set
    var fb = _make(16, [3, 4, 7, 8, 11])  # bits 3,4,7,8,11 set
    # slice(3, 9): 9 bits starting at offset 3 → indices 0-8 in sliced view
    # a slice indices: orig bits 3,5,7,9,11 → slice indices 0,2,4,6,8
    # b slice indices: orig bits 3,4,7,8,11 → slice indices 0,1,4,5,8
    var a = fa.slice(3, 9)
    var b = fb.slice(3, 9)
    var r = a & b
    assert_equal(len(r), 9)
    # AND: intersection at slice indices 0,4,8
    assert_true(r.is_valid(0))
    assert_false(r.is_valid(1))
    assert_false(r.is_valid(2))
    assert_false(r.is_valid(3))
    assert_true(r.is_valid(4))
    assert_false(r.is_valid(5))
    assert_false(r.is_valid(6))
    assert_false(r.is_valid(7))
    assert_true(r.is_valid(8))


def test_or_same_shift_fast_path() raises:
    """OR of two sliced bitmaps with same non-zero sub-byte offset."""
    var fa = _make(16, [3, 5])
    var fb = _make(16, [3, 4])
    var a = fa.slice(3, 5)  # slice indices 0,2 set
    var b = fb.slice(3, 5)  # slice indices 0,1 set
    var r = a | b
    assert_equal(len(r), 5)
    assert_true(r.is_valid(0))
    assert_true(r.is_valid(1))
    assert_true(r.is_valid(2))
    assert_false(r.is_valid(3))
    assert_false(r.is_valid(4))


def test_and_different_offsets() raises:
    """AND of bitmaps with different sub-byte offsets (shift-one path)."""
    # fa bits 3,5,7,9,11 set; sliced at offset 3 → shift_a=3, indices 0,2,4,6,8
    # fb bits 5,7,9,11,13 set; sliced at offset 5 → shift_b=5, indices 0,2,4,6,8
    var fa = _make(16, [3, 5, 7, 9, 11])
    var fb = _make(16, [5, 7, 9, 11, 13])
    var a = fa.slice(3, 9)  # shift_a = 3
    var b = fb.slice(5, 9)  # shift_b = 5
    var r = a & b
    assert_equal(len(r), 9)
    # AND: both have indices 0,2,4,6,8 set → intersection is 0,2,4,6,8
    for i in range(9):
        assert_equal(r.is_valid(i), i % 2 == 0)


def test_and_different_offsets_large_byte_delta() raises:
    """AND where byte-level offsets differ by more than 8 bytes."""
    # Build a large bitmap so we can slice at widely separated positions.
    # Set every even bit in the range we care about.
    var full = _make(
        600,
        [
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            114,
            500,
            502,
            504,
            506,
            508,
            510,
            512,
            514,
        ],
    )
    # a: slice at bit 100 (byte 12, shift 4), 16 bits → indices 0,2,4,6,8,10,12,14 set
    var a = full.slice(100, 16)
    # b: slice at bit 500 (byte 62, shift 4), 16 bits → indices 0,2,4,6,8,10,12,14 set
    var b = full.slice(500, 16)
    # Same sub-byte shift (4), but byte_delta = 50 — well beyond a single SIMD width.
    var r = a & b
    assert_equal(len(r), 16)
    for i in range(16):
        assert_equal(r.is_valid(i), i % 2 == 0)


def test_and_different_offsets_large_byte_delta_different_shift() raises:
    """AND where byte offsets AND sub-byte shifts both differ."""
    # a at bit 100 (byte 12, shift 4); b at bit 503 (byte 62, shift 7).
    # byte_delta = 50, bit shift delta = 3.
    var bits_a = List[Int](capacity=16)
    var bits_b = List[Int](capacity=16)
    for i in range(16):
        bits_a.append(100 + i)  # all 16 bits set starting at 100
        bits_b.append(503 + i)  # all 16 bits set starting at 503
    var full_a = _make(600, bits_a)
    var full_b = _make(600, bits_b)
    var a = full_a.slice(100, 16)
    var b = full_b.slice(503, 16)
    var r = a & b
    assert_equal(len(r), 16)
    # Both slices are all-ones, so AND should be all-ones.
    for i in range(16):
        assert_true(r.is_valid(i))


def test_or_different_offsets_large_byte_delta() raises:
    """OR where byte-level offsets differ significantly."""
    var full_a = _make(600, [100, 104, 108])
    var full_b = _make(600, [500, 501, 502])
    var a = full_a.slice(100, 12)  # indices 0,4,8 set
    var b = full_b.slice(500, 12)  # indices 0,1,2 set
    var r = a | b
    assert_equal(len(r), 12)
    assert_true(r.is_valid(0))  # set in both
    assert_true(r.is_valid(1))  # set in b
    assert_true(r.is_valid(2))  # set in b
    assert_false(r.is_valid(3))
    assert_true(r.is_valid(4))  # set in a
    assert_false(r.is_valid(5))
    assert_false(r.is_valid(6))
    assert_false(r.is_valid(7))
    assert_true(r.is_valid(8))  # set in a
    assert_false(r.is_valid(9))
    assert_false(r.is_valid(10))
    assert_false(r.is_valid(11))


def test_xor_different_offsets_large_byte_delta() raises:
    """XOR where byte-level offsets differ by > 64 bytes."""
    var bits_a = List[Int](capacity=16)
    var bits_b = List[Int](capacity=16)
    for i in range(16):
        bits_a.append(80 + i)  # all set
        bits_b.append(592 + i)  # all set
    var full_a = _make(700, bits_a)
    var full_b = _make(700, bits_b)
    var a = full_a.slice(80, 16)  # byte 10, shift 0
    var b = full_b.slice(592, 16)  # byte 74, shift 0
    var r = a ^ b
    assert_equal(len(r), 16)
    # Both all-ones → XOR should be all-zeros.
    for i in range(16):
        assert_false(r.is_valid(i))


def test_and_not_different_offsets_large_byte_delta() raises:
    """AND-NOT where byte-level offsets differ significantly."""
    var bits_a = List[Int](capacity=16)
    for i in range(16):
        bits_a.append(100 + i)  # all set
    var full_a = _make(600, bits_a)
    var full_b = _make(600, [500, 502, 504, 506])  # even indices set
    var a = full_a.slice(100, 16)  # all-ones
    var b = full_b.slice(500, 16)  # indices 0,2,4,6 set
    var r = a.and_not(b)
    assert_equal(len(r), 16)
    # a & ~b: clear bits where b is set → odd indices remain
    for i in range(16):
        if i < 8:
            assert_equal(r.is_valid(i), i % 2 != 0)
        else:
            assert_true(r.is_valid(i))


def test_invert_with_offset() raises:
    var full = _make(16, [4, 5, 6, 7])
    var s = full.slice(4, 8)  # bits 4-11 → [1,1,1,1,0,0,0,0]
    var inv = ~s
    assert_false(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_false(inv.is_valid(2))
    assert_false(inv.is_valid(3))
    assert_true(inv.is_valid(4))
    assert_true(inv.is_valid(5))
    assert_true(inv.is_valid(6))
    assert_true(inv.is_valid(7))


def test_invert_large_byte_offset() raises:
    # __invert__ with byte_offset > 63: exercises the lead_bytes > 0 code path
    # 600-bit source; slice at bit 576 → byte_offset=72, lead_bytes=8, shift=0.
    # Set bits 577 and 578 of the full bitmap (slice indices 1 and 2).
    var full = _make(600, [577, 578])
    var s = full.slice(576, 24)
    assert_false(s.is_valid(0))
    assert_true(s.is_valid(1))
    assert_true(s.is_valid(2))
    var inv = ~s
    assert_equal(len(inv), 24)
    assert_true(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_false(inv.is_valid(2))
    for i in range(3, 24):
        assert_true(inv.is_valid(i))
    assert_equal(inv.count_set_bits(), 22)


def test_invert_large_byte_offset_with_shift() raises:
    # __invert__ with large byte_offset AND non-zero sub-byte shift
    # Slice at bit 577 → byte_offset=72, shift=1, lead_bytes=8.
    # full bits 577, 578, 580 set → slice indices 0, 1, 3 set.
    var full = _make(600, [577, 578, 580])
    var s = full.slice(577, 8)
    assert_true(s.is_valid(0))
    assert_true(s.is_valid(1))
    assert_false(s.is_valid(2))
    assert_true(s.is_valid(3))
    var inv = ~s
    assert_equal(len(inv), 8)
    assert_false(inv.is_valid(0))
    assert_false(inv.is_valid(1))
    assert_true(inv.is_valid(2))
    assert_false(inv.is_valid(3))
    assert_true(inv.is_valid(4))
    assert_true(inv.is_valid(5))
    assert_true(inv.is_valid(6))
    assert_true(inv.is_valid(7))
    assert_equal(inv.count_set_bits(), 5)


def test_and_length_mismatch_raises() raises:
    var a = _make(8, [0, 2])
    var b = _make(4, [0, 2])
    try:
        _ = a & b
        assert_true(False, "should have raised")
    except:
        pass


def test_or_length_mismatch_raises() raises:
    var a = _make(8, [0, 2])
    var b = _make(4, [0])
    try:
        _ = a | b
        assert_true(False, "should have raised")
    except:
        pass


def test_xor_length_mismatch_raises() raises:
    var a = _make(8, [0, 2])
    var b = _make(4, [0])
    try:
        _ = a ^ b
        assert_true(False, "should have raised")
    except:
        pass


def test_and_not_length_mismatch_raises() raises:
    var a = _make(8, [0, 2])
    var b = _make(4, [0])
    try:
        _ = a.and_not(b)
        assert_true(False, "should have raised")
    except:
        pass


def test_bitmap_eq_equal_aligned() raises:
    var a = _make(8, [0, 3, 7])
    var b = _make(8, [0, 3, 7])
    assert_true(a == b)


def test_bitmap_eq_unequal() raises:
    var a = _make(8, [0, 3, 7])
    var b = _make(8, [0, 3, 6])
    assert_false(a == b)


def test_bitmap_eq_different_length() raises:
    var a = _make(8, [0, 3])
    var b = _make(9, [0, 3])
    assert_false(a == b)


def test_bitmap_eq_equal_offset() raises:
    # Two bitmaps with different non-zero offsets but the same logical bit pattern.
    # Build a 10-bit bitmap, then slice both starting at bit 2.
    var base_a = _make(10, [2, 5, 9])
    var base_b = _make(10, [2, 5, 9])
    var slice_a = base_a.slice(2, 8)  # logical bits [0..8) → original [2..10)
    var slice_b = base_b.slice(2, 8)
    assert_true(slice_a == slice_b)


def test_bitmap_eq_offset_mismatch() raises:
    # Slices that expose different bits are not equal.
    var base_a = _make(10, [2, 5, 9])
    var base_b = _make(10, [3, 5, 9])
    var slice_a = base_a.slice(2, 8)
    var slice_b = base_b.slice(2, 8)
    assert_false(slice_a == slice_b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
