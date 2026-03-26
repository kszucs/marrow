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
Mutable counterpart.  Wraps `Buffer[mut=True]` for incremental construction.
Call `finish(length)` to freeze into an immutable `Bitmap`.
"""

from std.memory import memcpy, memset

from .buffers import Buffer
from .views import BitmapView


# ---------------------------------------------------------------------------
# Bitmap — immutable, bit-packed
# ---------------------------------------------------------------------------


struct Bitmap(ImplicitlyCopyable, Movable, Sized, Writable):
    """Immutable bit-packed validity bitmap.

    Wraps a `Buffer` with a `_offset` (bit offset) and `_length` (bit count).
    Copying is O(1); the backing `Buffer` uses `ArcPointer` shared ownership.
    """

    var _buffer: Buffer[]
    var _offset: Int
    var _length: Int

    def __init__(out self, buffer: Buffer[], offset: Int, length: Int):
        self._buffer = buffer
        self._offset = offset
        self._length = length

    @always_inline
    def __len__(self) -> Int:
        return self._length

    @always_inline
    def bit_offset(self) -> Int:
        """Return the bit offset into the backing buffer."""
        return self._offset

    @always_inline
    def view(ref self) -> BitmapView[ImmutOrigin(origin_of(self))]:
        """Return a non-owning BitmapView over this bitmap's bits."""
        return BitmapView[ImmutOrigin(origin_of(self))](
            ptr=rebind[UnsafePointer[UInt8, ImmutOrigin(origin_of(self))]](
                self._buffer.unsafe_ptr()
            ),
            offset=self._offset,
            length=self._length,
        )

    def slice(self, offset: Int, length: Int) -> Bitmap:
        """Return a zero-copy view of `length` bits starting at `offset`."""
        return Bitmap(self._buffer, self.bit_offset() + offset, length)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "Bitmap(offset=", self.bit_offset(), ", length=", self._length, ")"
        )

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)


# ---------------------------------------------------------------------------
# BitmapBuilder — mutable counterpart to Bitmap
# ---------------------------------------------------------------------------


struct BitmapBuilder(Movable):
    """Mutable bit-packed bitmap builder.

    Wraps `Buffer[mut=True]` with bit-level write operations.  Call `finish(length)`
    to freeze into an immutable `Bitmap`.

    Example:
        var bm = BitmapBuilder.alloc(10)
        bm.set_bit(0, True)
        bm.set_bit(5, True)
        var bitmap = bm.finish(10)   # Bitmap of 10 bits, 2 set
    """

    var _builder: Buffer[mut=True]

    def __init__(out self, var builder: Buffer[mut=True]):
        self._builder = builder^

    @staticmethod
    def alloc(length: Int) -> BitmapBuilder:
        """Allocate a zero-filled builder for `length` bits."""
        return BitmapBuilder(Buffer.alloc_zeroed[DType.bool](length))

    @always_inline
    def unsafe_ptr(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """Return the raw mutable byte pointer (for low-level bit operations).
        """
        return self._builder.ptr

    @always_inline
    def deposit_bits(mut self, bit_offset: Int, bits: UInt64, count: Int):
        """Deposit `count` LSBs from `bits` into the builder at `bit_offset`.

        The builder must be zero-filled (from `alloc`), as this uses OR to set
        bits. Handles arbitrary bit alignment — writes up to 9 bytes when the
        64-bit value straddles a byte boundary.
        """
        if count == 0:
            return
        var dst = self._builder.ptr
        var byte_idx = bit_offset >> 3
        var bit_off = bit_offset & 7
        var shifted = bits << UInt64(bit_off)
        # OR the low 8 bytes.
        var ptr64 = (dst + byte_idx).bitcast[UInt64]()
        ptr64.store[alignment=1](ptr64.load[alignment=1]() | shifted)
        # Handle overflow into the 9th byte when shifted bits spill past 64.
        if bit_off > 0 and bit_off + count > 64:
            dst[byte_idx + 8] = dst[byte_idx + 8] | UInt8(
                bits >> UInt64(64 - bit_off)
            )

    # TODO: add safe apis
    @always_inline
    def set_bit(mut self, index: Int, value: Bool):
        """Set or clear the bit at `index`."""
        self._builder.unsafe_set[DType.bool](index, value)

    def set_range(mut self, start: Int, length: Int, value: Bool):
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

    def copy_bits(
        mut self,
        src_ptr: UnsafePointer[UInt8, _],
        src_offset: Int,
        dst_offset: Int,
        length: Int,
    ):
        """Bulk-copy `length` bits from `src_ptr` at bit `src_offset` into
        self at bit `dst_offset`.

        Three code paths ordered by expected frequency:
        1. Same sub-byte alignment: memcpy for middle bytes, masks at edges.
        2. Different alignment: shift-and-merge byte-by-byte.
        3. Short runs (< 16 bits): bit-by-bit fallback to avoid setup overhead.
        """
        if length == 0:
            return
        var dst = self._builder.ptr

        # Short runs: bit-by-bit is faster than computing byte masks.
        if length < 16:
            for i in range(length):
                var s_byte = (src_offset + i) >> 3
                var s_bit = (src_offset + i) & 7
                var val = (src_ptr[s_byte] >> UInt8(s_bit)) & 1
                var d_byte = (dst_offset + i) >> 3
                var d_bit = (dst_offset + i) & 7
                var d_mask = UInt8(1 << d_bit)
                if val:
                    dst[d_byte] = dst[d_byte] | d_mask
                else:
                    dst[d_byte] = dst[d_byte] & ~d_mask
            return

        var src_bit = src_offset & 7
        var dst_bit = dst_offset & 7

        if src_bit == dst_bit:
            # Same sub-byte alignment: can use memcpy for the bulk.
            var src_byte = src_offset >> 3
            var dst_byte = dst_offset >> 3
            var end_bit = dst_offset + length
            var end_byte = end_bit >> 3
            var end_sub = end_bit & 7

            if dst_bit != 0:
                # Merge leading partial byte.
                var keep_mask = UInt8((1 << dst_bit) - 1)
                dst[dst_byte] = (dst[dst_byte] & keep_mask) | (
                    src_ptr[src_byte] & ~keep_mask
                )
                src_byte += 1
                dst_byte += 1

            # Full middle bytes via memcpy.
            if end_byte > dst_byte:
                memcpy(
                    dest=dst + dst_byte,
                    src=src_ptr + src_byte,
                    count=end_byte - dst_byte,
                )

            if end_sub != 0:
                # Merge trailing partial byte.
                var trail_byte_src = src_byte + (end_byte - dst_byte)
                var keep_mask = UInt8(0xFF) << UInt8(end_sub)
                dst[end_byte] = (dst[end_byte] & keep_mask) | (
                    src_ptr[trail_byte_src] & ~keep_mask
                )
        else:
            # Different sub-byte alignment: shift-and-merge byte-by-byte.
            var src_byte = src_offset >> 3
            var dst_byte_start = dst_offset >> 3
            var end_bit = dst_offset + length
            var end_byte = end_bit >> 3
            var end_sub = end_bit & 7
            var delta = src_bit - dst_bit

            if dst_bit != 0:
                # Leading partial byte.
                var keep_mask = UInt8((1 << dst_bit) - 1)
                var shifted: UInt8
                if delta > 0:
                    shifted = (src_ptr[src_byte] >> UInt8(delta)) | (
                        src_ptr[src_byte + 1] << UInt8(8 - delta)
                    )
                else:
                    shifted = src_ptr[src_byte] << UInt8(-delta)
                    if src_byte > 0:
                        shifted |= src_ptr[src_byte - 1] >> UInt8(8 + delta)
                dst[dst_byte_start] = (dst[dst_byte_start] & keep_mask) | (
                    shifted & ~keep_mask
                )
                dst_byte_start += 1

            # Full middle bytes.
            var src_bit_pos = src_offset + ((dst_byte_start << 3) - dst_offset)
            for j in range(dst_byte_start, end_byte):
                var sb = src_bit_pos >> 3
                var so = src_bit_pos & 7
                if so == 0:
                    dst[j] = src_ptr[sb]
                else:
                    dst[j] = (src_ptr[sb] >> UInt8(so)) | (
                        src_ptr[sb + 1] << UInt8(8 - so)
                    )
                src_bit_pos += 8

            if end_sub != 0:
                # Trailing partial byte.
                var sb = src_bit_pos >> 3
                var so = src_bit_pos & 7
                var shifted: UInt8
                if so == 0:
                    shifted = src_ptr[sb]
                else:
                    shifted = (src_ptr[sb] >> UInt8(so)) | (
                        src_ptr[sb + 1] << UInt8(8 - so)
                    )
                var keep_mask = UInt8(0xFF) << UInt8(end_sub)
                dst[end_byte] = (dst[end_byte] & keep_mask) | (
                    shifted & ~keep_mask
                )

    def extend(mut self, src: Bitmap, dst_start: Int, length: Int):
        """Copy `length` bits from `src` (from its `_offset`) into self at `dst_start`.

        Replaces the `bitmap_extend` free function.
        """
        self.copy_bits(
            src._buffer.unsafe_ptr(), src.bit_offset(), dst_start, length
        )

    def resize(mut self, capacity: Int) raises:
        """Resize the underlying buffer to hold `capacity` bits."""
        self._builder.resize[DType.bool](capacity)

    def finish(mut self, length: Int) -> Bitmap:
        """Freeze the builder into an immutable `Bitmap` of `length` bits.

        The builder is reset to empty and can be reused after this call.
        """
        return Bitmap(self._builder.finish(), 0, length)
