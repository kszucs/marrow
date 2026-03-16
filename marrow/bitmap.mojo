"""Bitmap — immutable bit-packed validity buffer with SIMD operations.

Bitmap
------
`Bitmap` wraps an immutable `Buffer` plus `_offset: Int` and `_length: Int`.

  _offset
      Bit offset into `_buffer`.  Enables zero-copy slicing: `slice(n, m)`
      returns a new `Bitmap` that shares the same `Buffer` but starts at a
      later bit position.  Always 0 for freshly-built bitmaps.

  _length
      Number of valid bits.

Copying a `Bitmap` is O(1) — it bumps the `Buffer`'s `ArcPointer` ref-count.

BitmapBuilder
-------------
Mutable counterpart.  Wraps `BufferBuilder` for incremental construction.
Call `finish(length)` to freeze into an immutable `Bitmap`.
"""

from std.sys.info import simd_width_of
from std.bit import pop_count
import std.math as math
from std.memory import bitcast, memset

from .buffers import Buffer, BufferBuilder


# ---------------------------------------------------------------------------
# Bitmap — immutable, bit-packed
# ---------------------------------------------------------------------------


struct Bitmap(Equatable, ImplicitlyCopyable, Movable, Sized, Writable):
    """Immutable bit-packed validity bitmap.

    Wraps a `Buffer` with a `_offset` (bit offset) and `_length` (bit count).
    Copying is O(1); the backing `Buffer` uses `ArcPointer` shared ownership.
    """

    var _buffer: Buffer
    var _offset: Int
    var _length: Int

    fn __init__(out self, buffer: Buffer, offset: Int, length: Int):
        self._buffer = buffer
        self._offset = offset
        self._length = length

    @always_inline
    fn __len__(self) -> Int:
        return self._length

    @always_inline
    fn bit_offset(self) -> Int:
        """Return the bit offset into the backing buffer."""
        return self._offset

    @always_inline
    fn _aligned_byte_range(
        self,
    ) -> Tuple[UnsafePointer[UInt8, ImmutExternalOrigin], Int, Int, Int]:
        """Return 64-byte-aligned pointer and byte range with boundary bits.

        Returns (ptr, total_bytes, lead_bits, trail_bits):
          ptr:         buffer pointer backed up to the 64-byte boundary
          total_bytes: aligned_end - aligned_start (multiple of 64)
          lead_bits:   bit offset from aligned_start to first data bit
                       (lead_bytes*8 + sub-byte offset, range 0-511)
          trail_bits:  bit offset from last data bit to aligned_end
                       (trail_bytes*8 + sub-byte remainder, 0 = exact fit)

        Arrow buffers are 64-byte aligned and zero-padded, so reading the
        full [ptr, ptr + total_bytes) range is always safe.
        """
        var byte_start = self._offset >> 3
        var bit_end = self._offset + self._length
        var byte_end = (bit_end + 7) >> 3
        var aligned_start = math.align_down(byte_start, 64)
        var aligned_end = math.align_up(byte_end, 64)
        var lead_bits = self._offset - (aligned_start << 3)
        var trail_bits = (aligned_end - byte_end) * 8 + (bit_end & 7)
        return Tuple(
            self._buffer.unsafe_ptr() + aligned_start,
            aligned_end - aligned_start,
            lead_bits,
            trail_bits,
        )

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        """Return True if bit at (_offset + index) is set (value is valid)."""
        var bit_index = self.bit_offset() + index
        return Bool(
            (self._buffer.ptr[bit_index >> 3] >> UInt8(bit_index & 7)) & 1
        )

    @always_inline
    fn is_null(self, index: Int) -> Bool:
        """Return True if bit at (_offset + index) is unset (value is null)."""
        return not self.is_valid(index)

    fn count_set_bits(self) -> Int:
        """Count set bits in [_offset, _offset + _length) using SIMD popcount.

        Tier 1: 512-byte blocks with 2 interleaved uint8 accumulators.
        Tier 2: 64-byte blocks for the remainder.
        Lead/trail corrections subtract bits outside the bitmap's logical range.
        """
        comptime width = simd_width_of[DType.uint8]()
        comptime t1_iters = 512 // width // 2
        comptime t1_bytes = 512
        comptime t2_iters = 64 // width

        if self._length == 0:
            return 0

        ptr, total_bytes, lead_bits, trail_bits = self._aligned_byte_range()

        # Tier 1: 512-byte blocks, 2 interleaved uint8 accumulators.
        # NEON: 16 iters/acc, max 128/lane ≤ 255 ✓ (cast to uint16 before sum).
        var t1_end = (total_bytes // t1_bytes) * t1_bytes
        var count = 0
        for i in range(0, t1_end, t1_bytes):
            var acc0 = SIMD[DType.uint8, width](0)
            var acc1 = SIMD[DType.uint8, width](0)
            comptime for j in range(t1_iters):
                acc0 += pop_count(
                    (ptr + i + (j * 2) * width).load[width=width]()
                )
                acc1 += pop_count(
                    (ptr + i + (j * 2 + 1) * width).load[width=width]()
                )
            count += Int(
                (
                    acc0.cast[DType.uint16]() + acc1.cast[DType.uint16]()
                ).reduce_add()
            )

        # Tier 2: 64-byte blocks for the remainder.
        # NEON: 4 iters, max 32/lane ≤ 255 ✓.
        for i in range(t1_end, total_bytes, 64):
            var acc = SIMD[DType.uint8, width](0)
            comptime for j in range(t2_iters):
                acc += pop_count((ptr + i + j * width).load[width=width]())
            count += Int(acc.cast[DType.uint16]().reduce_add())

        # Subtract bits outside [_offset, _offset + _length).
        if lead_bits:
            var lead_bytes = lead_bits >> 3
            var lead_sub_byte = lead_bits & 7
            for i in range(lead_bytes):
                count -= Int(pop_count(ptr[i]))
            if lead_sub_byte:
                count -= Int(
                    pop_count(ptr[lead_bytes] & UInt8((1 << lead_sub_byte) - 1))
                )
        if trail_bits:
            var trail_bytes = trail_bits >> 3
            var trail_sub_byte = trail_bits & 7
            var first_trail = total_bytes - trail_bytes
            if trail_sub_byte:
                count -= Int(
                    pop_count(ptr[first_trail - 1] >> UInt8(trail_sub_byte))
                )
            for i in range(first_trail, total_bytes):
                count -= Int(pop_count(ptr[i]))

        return count

    fn slice(self, offset: Int, length: Int) -> Bitmap:
        """Return a zero-copy view of `length` bits starting at `offset`."""
        return Bitmap(self._buffer, self.bit_offset() + offset, length)

    fn __eq__(self, other: Bitmap) -> Bool:
        """Return True if both bitmaps have identical logical bit patterns.

        Uses XOR + count_set_bits to correctly handle arbitrary bit offsets
        on either side. Returns False if either bitmap's buffer is device-resident.
        """
        if self._length != other._length:
            return False
        if not self._buffer.is_cpu() or not other._buffer.is_cpu():
            return False
        try:
            var xor_result = self ^ other
            return xor_result.count_set_bits() == 0
        except:
            return False

    # --- Bulk SIMD operations ---
    #
    # Arrow buffers are 64-byte aligned and zero-padded to multiples of 64 B.
    # simd_width_of[DType.uint8]() ∈ {16,32,64} all divide 64, so SIMD loops over ranges
    # returned by `_simd_byte_offset_range` terminate exactly on a cache-line boundary
    # with no overshoot.  Bits outside [_offset, _offset+_length) in results
    # are arbitrary; all operations that consume bitmaps respect _offset and _length.

    fn __and__(self, other: Bitmap) raises -> Bitmap:
        """Return the bitwise AND of self and other."""
        return self._binop[_and](other)

    fn __or__(self, other: Bitmap) raises -> Bitmap:
        """Return the bitwise OR of self and other."""
        return self._binop[_or](other)

    fn __xor__(self, other: Bitmap) raises -> Bitmap:
        """Return the bitwise XOR of self and other."""
        return self._binop[_xor](other)

    fn __invert__(self) -> Bitmap:
        """Return the bitwise NOT of this bitmap."""
        comptime width = simd_width_of[DType.uint8]()
        comptime assert 64 % width == 0
        comptime unroll = 64 // width

        src, total_bytes, lead_bits, _ = self._aligned_byte_range()

        var builder = BufferBuilder.alloc_uninit(total_bytes)
        var dst = builder.unsafe_ptr()
        for i in range(0, total_bytes, 64):
            comptime for j in range(unroll):
                comptime k = j * width
                (dst + i + k).store(~(src + i + k).load[width=width]())

        return Bitmap(builder.finish(), lead_bits, self._length)

    fn and_not(self, other: Bitmap) raises -> Bitmap:
        """Return self & ~other  (A AND NOT B).

        Useful for null propagation: combine validity where *both* must be valid,
        and exclude elements that are null in `other`.
        """
        return self._binop[_and_not](other)

    @always_inline
    fn _binop[
        op: fn[W: Int](SIMD[DType.uint8, W], SIMD[DType.uint8, W]) -> SIMD[
            DType.uint8, W
        ]
    ](self, other: Bitmap) raises -> Bitmap:
        """Apply a byte-level binary operation across two bitmaps.

        Two code paths based on sub-byte alignment:
        - Same sub-byte offset: direct SIMD op, no bit shifting.
        - Different sub-byte offset: `other` is bit-shifted to align with `self`
          via overlapping loads: `(lo >> delta) | (hi << (8 - delta))`.

        src_a is backed up to its 64-byte boundary so src_a and dst are
        cache-line aligned.  src_b is shifted by the same lead byte offset to
        maintain byte correspondence.
        """
        if self._length != other._length:
            raise Error("Bitmap lengths must match")
        comptime width = simd_width_of[DType.uint8]()
        comptime assert 64 % width == 0
        comptime unroll = 64 // width

        src_a, total_bytes, lead_bits_a, _ = self._aligned_byte_range()
        ptr_b, _, lead_bits_b, _ = other._aligned_byte_range()

        var src_b = ptr_b + ((lead_bits_b >> 3) - (lead_bits_a >> 3))
        var builder = BufferBuilder.alloc_uninit(total_bytes)
        var dst = builder.unsafe_ptr()

        if lead_bits_a & 7 == lead_bits_b & 7:
            for i in range(0, total_bytes, 64):
                comptime for j in range(unroll):
                    comptime k = j * width
                    (dst + i + k).store(
                        op(
                            (src_a + i + k).load[width=width](),
                            (src_b + i + k).load[width=width](),
                        )
                    )
        else:
            var in_byte_shift = (lead_bits_b - lead_bits_a) & 7
            var rs = UInt8(in_byte_shift)
            var ls = UInt8(8 - in_byte_shift)
            for i in range(0, total_bytes, 64):
                comptime for j in range(unroll):
                    comptime k = j * width
                    var a = (src_a + i + k).load[width=width]()
                    var lo = (src_b + i + k).load[width=width]()
                    var hi = (src_b + i + k + 1).load[width=width]()
                    (dst + i + k).store(op(a, (lo >> rs) | (hi << ls)))

        return Bitmap(builder.finish(), lead_bits_a, self._length)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Bitmap(offset=", self.bit_offset(), ", length=", self._length, ")"
        )

    fn write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


@always_inline
fn _and[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a & b


@always_inline
fn _or[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a | b


@always_inline
fn _xor[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a ^ b


@always_inline
fn _and_not[
    W: Int
](a: SIMD[DType.uint8, W], b: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    return a & ~b


# ---------------------------------------------------------------------------
# BitmapBuilder — mutable counterpart to Bitmap
# ---------------------------------------------------------------------------


struct BitmapBuilder(Movable):
    """Mutable bit-packed bitmap builder.

    Wraps `BufferBuilder` with bit-level write operations.  Call `finish(length)`
    to freeze into an immutable `Bitmap`.

    Example:
        var bm = BitmapBuilder.alloc(10)
        bm.set_bit(0, True)
        bm.set_bit(5, True)
        var bitmap = bm.finish(10)   # Bitmap of 10 bits, 2 set
    """

    var _builder: BufferBuilder

    fn __init__(out self, var builder: BufferBuilder):
        self._builder = builder^

    @staticmethod
    fn alloc(length: Int) -> BitmapBuilder:
        """Allocate a zero-filled builder for `length` bits."""
        return BitmapBuilder(BufferBuilder.alloc[DType.bool](length))

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8, MutExternalOrigin]:
        """Return the raw mutable byte pointer (for low-level bit operations).
        """
        return self._builder.ptr

    # TODO: add safe apis
    @always_inline
    fn set_bit(mut self, index: Int, value: Bool):
        """Set or clear the bit at `index`."""
        self._builder.unsafe_set[DType.bool](index, value)

    fn set_range(mut self, start: Int, length: Int, value: Bool):
        """Set `length` bits starting at `start` to `value`.

        Handles byte-aligned bulk fills via `memset` for the middle bytes,
        with partial masks at the boundaries.
        """
        if length == 0:
            return
        var ptr = self._builder.ptr
        var end = start + length
        var start_byte = start >> 3
        var start_bit = start & 7
        var end_byte = end >> 3
        var end_bit = end & 7

        var fill = UInt8(255 if value else 0)

        if start_byte == end_byte:
            # All bits in one byte
            var mask = UInt8((1 << end_bit) - 1) & (
                UInt8(0xFF) << UInt8(start_bit)
            )
            if value:
                ptr[start_byte] = ptr[start_byte] | mask
            else:
                ptr[start_byte] = ptr[start_byte] & ~mask
            return

        # Leading partial byte
        if start_bit != 0:
            var mask = UInt8(0xFF) << UInt8(start_bit)
            if value:
                ptr[start_byte] = ptr[start_byte] | mask
            else:
                ptr[start_byte] = ptr[start_byte] & ~mask
            start_byte += 1

        # Trailing partial byte
        if end_bit != 0:
            var mask = UInt8((1 << end_bit) - 1)
            if value:
                ptr[end_byte] = ptr[end_byte] | mask
            else:
                ptr[end_byte] = ptr[end_byte] & ~mask

        # Full middle bytes
        if end_byte > start_byte:
            memset(ptr + start_byte, fill, end_byte - start_byte)

    fn extend(mut self, src: Bitmap, dst_start: Int, length: Int):
        """Copy `length` bits from `src` (from its `_offset`) into self at `dst_start`.

        Replaces the `bitmap_extend` free function.
        """
        var ptr = self._builder.ptr
        for i in range(length):
            var bit = src.is_valid(i)
            var pos = dst_start + i
            var byte_idx = pos >> 3
            var bit_mask = UInt8(1 << (pos & 7))
            if bit:
                ptr[byte_idx] = ptr[byte_idx] | bit_mask
            else:
                ptr[byte_idx] = ptr[byte_idx] & ~bit_mask

    fn resize(mut self, capacity: Int) raises:
        """Resize the underlying buffer to hold `capacity` bits."""
        self._builder.resize[DType.bool](capacity)

    fn finish(mut self, length: Int) -> Bitmap:
        """Freeze the builder into an immutable `Bitmap` of `length` bits.

        The builder is reset to empty and can be reused after this call.
        """
        return Bitmap(self._builder.finish(), 0, length)
