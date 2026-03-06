from std.memory import ArcPointer
from std.sys import size_of

from ..buffers import Buffer, Bitmap
from ..dtypes import *


fn drop_nulls[
    T: DType
](
    mut buffer: ArcPointer[Buffer],
    mut bitmap: ArcPointer[Bitmap],
    buffer_start: Int,
    buffer_end: Int,
) -> None:
    """Drop nulls from a region in the buffer.

    Args:
        buffer: The buffer to drop nulls from.
        bitmap: The validity bitmap.
        buffer_start: The start individualx of the buffer.
        buffer_end: The end index of the buffer.
    """
    var start = buffer_start
    # Find the end of a run of valid bits.
    start = start + bitmap[].count_leading_bits(start, value=True)
    while start < buffer_end:
        # Find the end of the run of nulls, could be just one null.
        var leading = bitmap[].count_leading_bits(start, value=False)
        var end_nulls = start + leading
        end_nulls = min(end_nulls, buffer_end)

        # Find the end of the run of values after the end of nulls.
        var end_values = end_nulls + bitmap[].count_leading_bits(
            end_nulls, value=True
        )
        end_values = min(end_values, buffer_end)
        var values_len = end_values - end_nulls
        if values_len == 0:
            # No valid entries to move, just skip.
            start = end_nulls
            continue

        # Compact the data.
        memcpy(
            dest=buffer[].get_ptr_at(start),
            src=buffer[].get_ptr_at(end_nulls),
            count=values_len * size_of[T](),
        )
        # Adjust the bitmp.
        var new_values_start = start
        var new_values_end = start + values_len
        var new_nulls_end = end_values
        bitmap[].unsafe_range_set(
            new_values_start, new_values_end - new_values_start, True
        )
        bitmap[].unsafe_range_set(
            new_values_end, new_nulls_end - new_values_end, False
        )

        # Get ready for next iteration.
        start = new_values_end


struct PrimitiveArray[T: DataType](Array):
    """An Arrow array of primitive types."""

    comptime dtype = Self.T
    comptime scalar = Scalar[Self.T.native]
    var data: ArrayData
    var offset: Int
    var capacity: Int

    fn __init__(out self, var data: ArrayData, offset: Int = 0) raises:
        # TODO(kszucs): put a dtype constraint here
        if data.dtype != materialize[Self.T]():
            raise Error(
                "Unexpected dtype '{}' instead of '{}'.".format(
                    data.dtype, materialize[Self.T]()
                )
            )
        elif len(data.buffers) != 1:
            raise Error("PrimitiveArray requires exactly one buffer")

        self.offset = data.offset + offset
        self.capacity = data.length
        self.data = data^

    fn __init__(out self, capacity: Int = 0, offset: Int = 0):
        self.capacity = capacity
        self.offset = offset
        bitmap = ArcPointer(Bitmap.alloc(capacity))
        buffer = ArcPointer(Buffer.alloc[Self.T.native](capacity))
        self.data = ArrayData(
            dtype=materialize[Self.T](),
            length=0,
            bitmap=bitmap,
            buffers=[buffer],
            children=[],
            offset=self.offset,
        )

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^
        self.capacity = take.capacity
        self.offset = take.offset

    fn bitmap(self) -> ref[self.data.bitmap] ArcPointer[Bitmap]:
        return self.data.bitmap

    fn buffer(self) -> ref[self.data.buffers] ArcPointer[Buffer]:
        return self.data.buffers[0]

    fn take_data(deinit self) -> ArrayData:
        return self.data^

    fn as_data(self) -> UnsafePointer[ArrayData, ImmutAnyOrigin]:
        return UnsafePointer(to=self.data)

    fn grow(mut self, capacity: Int):
        self.bitmap()[].grow(capacity)
        self.buffer()[].grow[Self.T.native](capacity)
        self.capacity = capacity

    @always_inline
    fn __len__(self) -> Int:
        return self.data.length

    @always_inline
    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap()[].unsafe_get(index + self.offset)

    @always_inline
    fn unsafe_get(self, index: Int) -> Self.scalar:
        return self.buffer()[].unsafe_get[Self.T.native](index + self.offset)

    @always_inline
    fn unsafe_set(mut self, index: Int, value: Self.scalar):
        self.bitmap()[].unsafe_set(index + self.offset, True)
        self.buffer()[].unsafe_set[Self.T.native](index + self.offset, value)

    @always_inline
    fn unsafe_append(mut self, value: Self.scalar):
        self.unsafe_set(self.data.length, value)
        self.data.length += 1

    @staticmethod
    fn nulls(size: Int) raises -> PrimitiveArray[Self.T]:
        """Creates a new PrimitiveArray filled with null values."""
        var bitmap = Bitmap.alloc(size)
        bitmap.unsafe_range_set(0, size, False)
        var buffer = Buffer.alloc[Self.T.native](size)
        return PrimitiveArray[Self.T](
            data=ArrayData(
                dtype=materialize[Self.T](),
                length=size,
                bitmap=ArcPointer(bitmap^),
                buffers=[ArcPointer(buffer^)],
                children=[],
                offset=0,
            ),
        )

    fn append(mut self, value: Self.scalar):
        if self.data.length >= self.capacity:
            self.grow(max(self.capacity * 2, self.data.length + 1))
        self.unsafe_append(value)

    # fn append(mut self, value: Optional[Self.scalar]):

    fn extend(mut self, values: List[self.scalar]):
        if self.__len__() + len(values) >= self.capacity:
            self.grow(self.capacity + len(values))
        for value in values:
            self.unsafe_append(value)

    fn drop_nulls[dtype: DType](mut self) -> None:
        """Drops null values from the Array.

        Currently we drop nulls from individual buffers, we do not delete buffers.
        """
        drop_nulls[dtype](
            self.data.buffers[0], self.data.bitmap, 0, self.data.length
        )
        self.data.length = self.bitmap()[].buffer.bit_count()

    fn null_count(self) -> Int:
        """Returns the number of null values in the array."""
        var valid_count = self.bitmap()[].buffer.bit_count()
        return self.data.length - valid_count

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this PrimitiveArray to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("PrimitiveArray( dtype=")
        writer.write(materialize[Self.dtype]())
        writer.write(", offset=")
        writer.write(self.offset)
        writer.write(", capacity=")
        writer.write(self.capacity)
        writer.write(", buffer=[")
        for i in range(self.capacity):
            if self.is_valid(i):
                writer.write(
                    self.buffer()[].unsafe_get[Self.T.native](i + self.offset)
                )
            else:
                writer.write("NULL")
            writer.write(", ")
            if i > 10:
                writer.write("...")
                break
        writer.write("])")

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String.write(self)


comptime BoolArray = PrimitiveArray[bool_]
comptime Int8Array = PrimitiveArray[int8]
comptime Int16Array = PrimitiveArray[int16]
comptime Int32Array = PrimitiveArray[int32]
comptime Int64Array = PrimitiveArray[int64]
comptime UInt8Array = PrimitiveArray[uint8]
comptime UInt16Array = PrimitiveArray[uint16]
comptime UInt32Array = PrimitiveArray[uint32]
comptime UInt64Array = PrimitiveArray[uint64]
comptime Float32Array = PrimitiveArray[float32]
comptime Float64Array = PrimitiveArray[float64]
