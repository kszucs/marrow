    def slice(
        ref self, offset: Int, length: Int
    ) -> BitmapView[origin_of(self)]:
        """Return a zero-copy view of `length` bits starting at `offset`."""
        return self.view(offset, length)

    def __eq__(self: Bitmap[mut=_], other: Bitmap[mut=_]) -> Bool:
        """Compare two bitmaps bit-by-bit over their valid ranges."""
        if self._length != other._length:
            return False
        for i in range(self._length):
            if self[i] != other[i]:
                return False
        return True

    def byte_count(self) -> Int:
        """Return the size of the backing buffer in bytes."""
        return len(self._buffer)

    @always_inline
    def _check_bounds(self, index: Int):
        debug_assert(
            0 <= index < self._length,
            "Bitmap index ",
            index,
            " out of bounds for length ",
            self._length,
        )

    def set(mut self: Bitmap[mut=True], index: Int):
        """Set the bit at `index` to 1."""
        self._check_bounds(index)
        self.unsafe_set(index)

    @always_inline
    def unsafe_set(mut self: Bitmap[mut=True], index: Int):
        """Set the bit at `index` to 1."""
        var byte_index = index // 8
        var bit_mask = UInt8(1 << (index % 8))
        self._buffer._ptr[byte_index] = self._buffer._ptr[byte_index] | bit_mask

    def clear(mut self: Bitmap[mut=True], index: Int):
        """Clear the bit at `index` to 0."""
        self._check_bounds(index)
        self.unsafe_clear(index)

    @always_inline
    def unsafe_clear(mut self: Bitmap[mut=True], index: Int):
        """Clear the bit at `index` to 0."""
        var byte_index = index // 8
        var bit_mask = UInt8(1 << (index % 8))
        self._buffer._ptr[byte_index] = (
            self._buffer._ptr[byte_index] & ~bit_mask
        )

    def test(self, raw_index: Int) -> Bool:
        """Return True if the bit at `raw_index` (not offset-adjusted) is set.
        """
        self._check_bounds(raw_index)
        return self.unsafe_test(raw_index)

    @always_inline
    def unsafe_test(self, raw_index: Int) -> Bool:
        """Return True if the bit at `raw_index` (not offset-adjusted) is set.
        """
        var byte_index = raw_index // 8
        var bit_mask = UInt8(1 << (raw_index % 8))
        return (self._buffer._ptr[byte_index] & bit_mask) != 0

    @always_inline
    def __getitem__(self, index: Int) -> Bool:
        """Return the bit at logical `index` (0-based within this bitmap's window).
        """
        var i = index if index >= 0 else index + self._length
        self._check_bounds(i)
        return self.unsafe_test(i)

    @always_inline
    def __getitem__(
        self: Bitmap[], slc: ContiguousSlice
    ) -> BitmapView[origin_of(self)]:
        """Return a zero-copy sub-bitmap view for the given slice."""
        var start, end = slc.indices(self._length)
        return self.slice(start, end - start)

    def __setitem__(mut self: Bitmap[mut=True], index: Int, value: Bool):
        """Set or clear the bit at `index`."""
        var i = index if index >= 0 else index + self._length
        self._check_bounds(i)
        if value:
            self.unsafe_set(i)
        else:
            self.unsafe_clear(i)

    def set_range(
        mut self: Bitmap[mut=True], start: Int, length: Int, value: Bool
    ):
        """Set `length` bits starting at `start` to `value`."""
        if length == 0:
            return
        var end = start + length
        var start_byte = start >> 3
        var start_bit = start & 7
        var end_byte = end >> 3
        var end_bit = end & 7
        var fill = UInt8(255 if value else 0)
        var ptr = self._buffer._ptr

        if start_byte == end_byte:
            var mask = UInt8((1 << end_bit) - 1) & (
                UInt8(0xFF) << UInt8(start_bit)
            )
            if value:
                ptr[start_byte] = ptr[start_byte] | mask
            else:
                ptr[start_byte] = ptr[start_byte] & ~mask
            return

        if start_bit != 0:
            var mask = UInt8(0xFF) << UInt8(start_bit)
            if value:
                ptr[start_byte] = ptr[start_byte] | mask
            else:
                ptr[start_byte] = ptr[start_byte] & ~mask
            start_byte += 1

        if end_bit != 0:
            var mask = UInt8((1 << end_bit) - 1)
            if value:
                ptr[end_byte] = ptr[end_byte] | mask
            else:
                ptr[end_byte] = ptr[end_byte] & ~mask

        if end_byte > start_byte:
            memset(ptr + start_byte, fill, end_byte - start_byte)

    def extend(
        mut self: Bitmap[mut=True],
        src: BitmapView[_],
        dst_start: Int,
        length: Int,
    ):
        """Copy `length` bits from `src` into self at `dst_start`.

        Three code paths:
        1. Same sub-byte alignment → memcpy for middle bytes.
        2. Different alignment → shift-and-merge byte-by-byte.
        3. Short runs (< 16 bits) → bit-by-bit fallback.
        """
        if length == 0:
            return
        var dst = self._buffer._ptr
        var dst_offset = dst_start
        var src_ptr = src._data
        var src_offset = src._offset

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
            var src_byte = src_offset >> 3
            var dst_byte = dst_offset >> 3
            var end_bit = dst_offset + length
            var end_byte = end_bit >> 3
            var end_sub = end_bit & 7

            if dst_bit != 0:
                var keep_mask = UInt8((1 << dst_bit) - 1)
                dst[dst_byte] = (dst[dst_byte] & keep_mask) | (
                    src_ptr[src_byte] & ~keep_mask
                )
                src_byte += 1
                dst_byte += 1

            if end_byte > dst_byte:
                memcpy(
                    dest=dst + dst_byte,
                    src=src_ptr + src_byte,
                    count=end_byte - dst_byte,
                )

            if end_sub != 0:
                var trail_byte_src = src_byte + (end_byte - dst_byte)
                var keep_mask = UInt8(0xFF) << UInt8(end_sub)
                dst[end_byte] = (dst[end_byte] & keep_mask) | (
                    src_ptr[trail_byte_src] & ~keep_mask
                )
        else:
            var src_byte = src_offset >> 3
            var dst_byte_start = dst_offset >> 3
            var end_bit = dst_offset + length
            var end_byte = end_bit >> 3
            var end_sub = end_bit & 7
            var delta = src_bit - dst_bit

            if dst_bit != 0:
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

    def extend(
        mut self: Bitmap[mut=True], src: Bitmap[], dst_start: Int, length: Int
    ):
        """Copy `length` bits from `src` into self at `dst_start`."""
        # TODO: do we need extend on view? if not move it here
        self.extend(src.view(0, length), dst_start, length)

    def resize(mut self: Bitmap[mut=True], capacity: Int) raises:
        """Resize the underlying buffer to hold `capacity` bits.

        When shrinking, the logical length is truncated to `capacity`.
        """
        self._buffer.resize(math.ceildiv(capacity, 8))
        if capacity < self._length:
            self._length = capacity

    def is_device(self) -> Bool:
        """Return True if the bitmap lives on a GPU device."""
        return self._buffer.is_device()

    def to_device(
        self: Bitmap[mut=False], ctx: DeviceContext
    ) raises -> Bitmap[mut=False]:
        """Upload bitmap to the GPU; returns a new device-resident Bitmap."""
        return Bitmap[mut=False](
            self._buffer.to_device(ctx), length=self._length
        )

    def to_cpu(
        self: Bitmap[mut=False], ctx: DeviceContext
    ) raises -> Bitmap[mut=False]:
        """Download bitmap from the GPU to owned CPU heap buffers."""
        return Bitmap[mut=False](self._buffer.to_cpu(ctx), length=self._length)

    def to_immutable(
        deinit self: Bitmap[mut=True], *, length: Int = -1
    ) -> Bitmap[mut=False]:
        """Consume and freeze the builder into an immutable `Bitmap[]`.

        Pass `length` to set the number of meaningful bits explicitly; otherwise
        the builder's current `_length` is used.
        """
        var n = length if length >= 0 else self._length
        return Bitmap[mut=False](self._buffer^.to_immutable(), length=n)
