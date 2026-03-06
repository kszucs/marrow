from std.bit import pop_count, count_trailing_zeros
from std.math import ceildiv, align_up
from std.memory import (
    memset_zero,
    memcpy,
    memset,
)
from std.sys import size_of, simd_byte_width

from marrow.dtypes import dynamic_size_of


fn _required_bytes(length: Int, T: DType) -> Int:
    var size: Int
    if T == DType.bool:
        size = ceildiv(length, 8)
    else:
        size = length * dynamic_size_of(T)
    return align_up(size, 64)


comptime simd_width = simd_byte_width()

comptime simd_widths = (simd_width, simd_width // 2, 1)


struct Buffer(Movable):
    var ptr: UnsafePointer[UInt8, MutAnyOrigin]
    var size: Int
    var owns: Bool
    var offset: Int

    fn __init__(
        out self,
        ptr: UnsafePointer[UInt8, MutAnyOrigin],
        size: Int,
        owns: Bool = True,
        offset: Int = 0,
    ):
        self.ptr = ptr
        self.size = size
        self.owns = owns
        self.offset = offset

    fn __moveinit__(out self, deinit take: Self):
        self.ptr = take.ptr
        self.size = take.size
        self.owns = take.owns
        self.offset = take.offset

    fn swap(mut self, mut other: Self):
        """Swap the content of this buffer with another buffer."""
        swap(self.ptr, other.ptr)
        swap(self.size, other.size)
        swap(self.owns, other.owns)
        swap(self.offset, other.offset)

    @staticmethod
    fn alloc[I: Intable, //, T: DType = DType.uint8](length: I) -> Buffer:
        var size = _required_bytes(Int(length), T)
        var ptr = alloc[UInt8](size, alignment=64)
        memset_zero(ptr, size)
        return Buffer(ptr, size)

    @staticmethod
    fn from_values[dtype: DType](*values: Scalar[dtype]) -> Buffer:
        """Build a buffer from a list of values."""
        var buffer = Self.alloc[dtype](len(values))

        for i in range(len(values)):
            buffer.unsafe_set[dtype](i, values[i])

        return buffer^

    @staticmethod
    fn view[
        I: Intable, //
    ](
        ptr: UnsafePointer[NoneType, MutAnyOrigin],
        length: I,
        dtype: DType = DType.uint8,
    ) raises -> Buffer:
        var size = _required_bytes(Int(length), dtype)
        return Buffer(ptr.bitcast[UInt8](), size, owns=False)

    @always_inline
    fn get_ptr_at(self, index: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return self.ptr + index

    fn grow[I: Intable, //, T: DType = DType.uint8](mut self, target_length: I):
        if self.length[T]() >= Int(target_length):
            return

        var new = Buffer.alloc[T](target_length)
        memcpy(
            dest=new.ptr,
            src=self.ptr,
            count=self.size,
        )
        self.swap(new)

    fn __del__(deinit self):
        if self.owns:
            self.ptr.free()

    @always_inline
    fn length[T: DType = DType.uint8](self) -> Int:
        comptime if T == DType.bool:
            return self.size * 8
        else:
            return self.size // size_of[T]()

    @always_inline
    fn unsafe_get[T: DType = DType.uint8](self, index: Int) -> Scalar[T]:
        comptime output = Scalar[T]

        comptime if T == DType.bool:
            var adjusted_index = index + self.offset
            var byte_index = adjusted_index // 8
            var bit_index = UInt8(adjusted_index % 8)
            var byte = self.ptr[byte_index]
            var wanted_bit = (byte >> bit_index) & 1
            return Scalar[T](wanted_bit)
        else:
            return self.ptr.bitcast[output]()[index + self.offset]

    @always_inline
    fn unsafe_set[
        T: DType = DType.uint8
    ](mut self, index: Int, value: Scalar[T]):
        comptime if T == DType.bool:
            var adjusted_index = index + self.offset
            var byte_index = adjusted_index // 8
            var bit_index = adjusted_index % 8
            var byte = self.ptr[byte_index]
            if value:
                self.ptr[byte_index] = byte | UInt8(1 << bit_index)
            else:
                self.ptr[byte_index] = byte & UInt8(~(1 << bit_index))
        else:
            comptime output = Scalar[T]
            self.ptr.bitcast[output]()[index + self.offset] = value

    fn bit_count(self) -> Int:
        """The number of bits with value 1 in the buffer."""
        var start = 0
        var count = 0
        while start < self.size:
            if self.size - start > simd_width:
                count += (
                    self.get_ptr_at(start)
                    .load[width=simd_width]()
                    .reduce_bit_count()
                )
                start += simd_width
            else:
                count += (
                    self.get_ptr_at(start).load[width=1]().reduce_bit_count()
                )
                start += 1
        return count


struct Bitmap(Movable, Writable):

    """Hold information about the null records in an array."""

    var buffer: Buffer
    var offset: Int

    @staticmethod
    fn alloc[I: Intable](length: I) -> Bitmap:
        return Bitmap(Buffer.alloc[DType.bool](length))

    fn __init__(out self, var buffer: Buffer, offset: Int = 0):
        self.buffer = buffer^
        self.offset = offset

    fn __moveinit__(out self, deinit take: Self):
        self.buffer = take.buffer^
        self.offset = take.offset

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this buffer to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        for i in range(self.length()):
            var value = self.unsafe_get(i)
            if value:
                writer.write("T")
            else:
                writer.write("f")
            if i > 16:
                writer.write("...")
                break

    fn __str__(self) -> String:
        var output = String()
        output.write(self)
        return output

    fn __repr__(self) -> String:
        var output = String()
        output.write(self)
        return output

    fn unsafe_get(self, index: Int) -> Bool:
        return self.buffer.unsafe_get[DType.bool](index + self.offset)

    fn unsafe_set(mut self, index: Int, value: Bool) -> None:
        self.buffer.unsafe_set[DType.bool](index + self.offset, value)

    @always_inline
    fn length(self) -> Int:
        return self.buffer.length[DType.bool]()

    @always_inline
    fn size(self) -> Int:
        return self.buffer.size

    fn grow[I: Intable](mut self, target_length: I):
        return self.buffer.grow[DType.bool](target_length)

    fn bit_count(self) -> Int:
        """The number of bits with value 1 in the Bitmap."""
        var start = 0
        var count = 0
        while start < self.buffer.size:
            if self.buffer.size - start > simd_width:
                count += (
                    self.buffer.get_ptr_at(start)
                    .load[width=simd_width]()
                    .reduce_bit_count()
                )
                start += simd_width
            else:
                count += (
                    self.buffer.get_ptr_at(start)
                    .load[width=1]()
                    .reduce_bit_count()
                )
                start += 1
        return count

    fn count_leading_bits(self, start: Int = 0, value: Bool = False) -> Int:
        """Count the number of leading bits with the given value in the bitmap, starting at a given posiion.

        Note that index 0 in the bitmap translates to right most bit in the first byte of the buffer.
        So when we are looking for leading zeros from a bitmap standpoing we need to look at
        trailing zeros in the bitmap's associated buffer.

        The SIMD API available looks at leading zeros only, we negate the input when needed.

        Args:
          start: The position where we should start counting.
          value: The value of the bits we want to count.

        Returns:
          The number of leadinging bits with the given value in the bitmap.
        """

        var count = 0
        var index = start // 8
        var bit_in_first_byte = start % 8

        if bit_in_first_byte != 0:
            # Process the partial first byte by applying a mask.
            var loaded = self.buffer.get_ptr_at(index).load[width=1]()
            if value:
                loaded = ~loaded
            var mask = (1 << bit_in_first_byte) - 1
            loaded &= UInt8(~mask)
            leading_zeros = Int(count_trailing_zeros(loaded))
            if leading_zeros == 0:
                return count
            count = leading_zeros - bit_in_first_byte
            if leading_zeros != 8:
                # The first byte has some bits of the other value, just return the count.
                return count

            index += 1

        # Process full bytes.
        while index < self.size():
            comptime for width_index in range(len(simd_widths)):
                comptime width = simd_widths[width_index]
                if self.size() - index >= width:
                    var loaded = self.buffer.get_ptr_at(index).load[
                        width=width
                    ]()
                    if value:
                        loaded = ~loaded
                    var leading_zeros = count_trailing_zeros(loaded)
                    for i in range(width):
                        count += Int(leading_zeros[i])
                        if leading_zeros[i] != 8:
                            return count
                    index += width
                    # break from the simd widths loop
                    break
        return count

    fn count_leading_zeros(self, start: Int = 0) -> Int:
        """Count the number of leading 0s in the given value in the bitmap, starting at a given posiion.

        Note that index 0 in the bitmap translates to right most bit in the first byte of the buffer.
        So when we are looking for leading zeros from a bitmap standpoing we need to look at
        trailing zeros in the bitmap's associated buffer.

        Args:
            start: The position where we should start counting.

        Returns:
             The number of leading zeros in the bitmap.
        """
        return self.count_leading_bits(start, value=False)

    fn count_leading_ones(self, start: Int = 0) -> Int:
        """Count the number of leading 1s in the given value in the bitmap, starting at a given posiion.

        Note that index 0 in the bitmap translates to right most bit in the first byte of the buffer.
        So when we are looking for leading zeros from a bitmap standpoing we need to look at
        trailing zeros in the bitmap's associated buffer.

        Args:
          start: The position where we should start counting.

        Returns:
          The number of leading ones in the bitmap.
        """
        return self.count_leading_bits(start, value=True)

    fn extend(
        mut self,
        other: Bitmap,
        start: Int,
        length: Int,
    ) -> None:
        """Extends the bitmap with the other's array's bitmap.

        Args:
            other: The bitmap to take content from.
            start: The starting index in the destination array.
            length: The number of elements to copy from the source array.
        """
        var desired_size = _required_bytes(start + length, DType.bool)
        self.buffer.grow[DType.bool](desired_size)

        for i in range(length):
            self.unsafe_set(i + start, other.unsafe_get(i))

    fn partial_byte_set(
        mut self,
        byte_index: Int,
        bit_pos_start: Int,
        bit_pos_end: Int,
        value: Bool,
    ) -> None:
        """Set a range of bits in one specific byte of the bitmap to the specified value.
        """

        debug_assert(
            bit_pos_start >= 0
            and bit_pos_end <= 8
            and bit_pos_start <= bit_pos_end,
            "Invalid range: ",
            bit_pos_start,
            " to ",
            bit_pos_end,
        )

        # Process the partial byte at the start, if appropriate.
        var mask = UInt8((1 << (bit_pos_end - bit_pos_start)) - 1)
        mask = UInt8(mask << UInt8(bit_pos_start))
        var initial_value = self.buffer.unsafe_get[DType.uint8](byte_index)
        var buffer_value = initial_value
        if value:
            buffer_value = buffer_value | mask
        else:
            buffer_value = buffer_value & ~mask
        self.buffer.unsafe_set[DType.uint8](byte_index, buffer_value)

    fn unsafe_range_set[
        T: Intable, U: Intable, //
    ](mut self, start: T, length: U, value: Bool) -> None:
        """Set a range of bits in the bitmap to the specified value.

        Args:
            start: The starting index in the bitmap.
            length: The number of bits to set.
            value: The value to set the bits to.
        """

        # Process the partial byte at the ends.
        var start_int = Int(start)
        var end_int = start_int + Int(length)
        var start_index = start_int // 8
        var bit_pos_start = start_int % 8
        var end_index = end_int // 8
        var bit_pos_end = end_int % 8

        if bit_pos_start != 0 or bit_pos_end != 0:
            if start_index == end_index:
                self.partial_byte_set(
                    start_index, bit_pos_start, bit_pos_end, value
                )
            else:
                if bit_pos_start != 0:
                    self.partial_byte_set(start_index, bit_pos_start, 8, value)
                    start_index += 1
                if bit_pos_end != 0:
                    self.partial_byte_set(end_index, 0, bit_pos_end, value)

        # Now take care of the full bytes.
        if end_index > start_index:
            var byte_value = UInt8(255 if value else 0)
            memset(
                self.buffer.get_ptr_at(start_index),
                value=byte_value,
                count=end_index - start_index,
            )
