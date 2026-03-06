from std.memory import ArcPointer
from std.testing import assert_equal
from std.reflection import call_location

from marrow.arrays import (
    BoolArray,
    ArrayData,
    ListArray,
    StructArray,
)
from marrow.buffers import Buffer, Bitmap
from marrow.dtypes import uint8, DataType, list_, int32, Field, struct_


fn as_bool_array_scalar(value: Bool) -> BoolArray.scalar:
    """Bool conversion function."""
    return BoolArray.scalar(Scalar[DType.bool](value))


fn bool_array(*values: Bool) -> BoolArray:
    var a = BoolArray(len(values))
    for value in values:
        a.unsafe_append(as_bool_array_scalar(value))
    return a^


def build_array_data(length: Int, nulls: Int) raises -> ArrayData:
    """Builds an ArrayData object with nulls.

    Args:
        length: The length of the array.
        nulls: The number of nulls to set.
    """
    var bitmap = Bitmap.alloc(length)
    var buffer = Buffer.alloc[DType.uint8](length)
    for i in range(length):
        buffer.unsafe_set(i, UInt8(i % 256))
        # Check to see if the current index should be valid or null.
        var is_valid = True
        if nulls > 0:
            if i % (length // nulls) == 0:
                is_valid = False
        bitmap.unsafe_set(i, is_valid)

    var buffers = [ArcPointer(buffer^)]
    return ArrayData(
        dtype=materialize[uint8](),
        length=length,
        bitmap=ArcPointer(bitmap^),
        buffers=buffers^,
        children=[],
        offset=0,
    )


@always_inline
def assert_bitmap_set(
    bitmap: Bitmap, expected_true_pos: List[Int], message: StringLiteral
) raises -> None:
    var list_pos = 0
    for i in range(bitmap.length()):
        var expected_value = False
        if list_pos < len(expected_true_pos):
            if expected_true_pos[list_pos] == i:
                expected_value = True
                list_pos += 1
        var current_value = bitmap.unsafe_get(i)
        assert_equal(
            current_value,
            expected_value,
            String(
                "{}: Bitmap index {} is {}, expected {} as per list position {}"
            ).format(message, i, current_value, expected_value, list_pos),
            location=call_location(),
        )


fn build_list_of_int[data_type: DataType]() raises -> ListArray:
    """Build a test ListArray that itself contains a ListArray of IntArrays."""
    # Define all the values.
    var bitmap = Bitmap.alloc(10)
    bitmap.unsafe_range_set(0, 10, True)
    var buffer = ArcPointer(Buffer.alloc[data_type.native](10))
    for i in range(10):
        buffer[].unsafe_set[data_type.native](
            i, Scalar[data_type.native](i + 1)
        )

    var value_data = ArrayData(
        dtype=materialize[data_type](),
        length=10,
        bitmap=ArcPointer(bitmap^),
        buffers=[buffer],
        children=[],
        offset=0,
    )

    # Define the PrimitiveArrays.
    var value_offset = ArcPointer(
        Buffer.from_values[DType.int32](0, 2, 4, 7, 7, 8, 10)
    )

    var list_bitmap = ArcPointer(Bitmap.alloc(6))
    list_bitmap[].unsafe_range_set(0, 6, True)
    list_bitmap[].unsafe_set(3, False)
    var list_data = ArrayData(
        dtype=list_(materialize[data_type]()),
        length=6,
        buffers=[value_offset],
        children=[ArcPointer(value_data^)],
        bitmap=list_bitmap,
        offset=0,
    )
    return ListArray(list_data^)


fn build_list_of_list[data_type: DataType]() raises -> ListArray:
    """Build a test ListArray that itself contains a ListArray of IntArrays.

    See: https://elferherrera.github.io/arrow_guide/arrays_nested.html
    """

    # Define all the values.
    var bitmap = ArcPointer(Bitmap.alloc(10))
    bitmap[].unsafe_range_set(0, 10, True)
    var buffer = ArcPointer(Buffer.alloc[data_type.native](10))
    for i in range(10):
        buffer[].unsafe_set[data_type.native](
            i, Scalar[data_type.native](i + 1)
        )

    var value_data = ArrayData(
        dtype=materialize[data_type](),
        length=10,
        bitmap=bitmap,
        buffers=[buffer],
        children=[],
        offset=0,
    )

    # Define the PrimitiveArrays.
    var value_offset = ArcPointer(
        Buffer.from_values[DType.int32](0, 2, 4, 7, 7, 8, 10)
    )

    var list_bitmap = ArcPointer(Bitmap.alloc(6))
    list_bitmap[].unsafe_range_set(0, 6, True)
    list_bitmap[].unsafe_set(3, False)
    var list_data = ArrayData(
        dtype=list_(materialize[data_type]()),
        length=6,
        buffers=[value_offset],
        children=[ArcPointer(value_data^)],
        bitmap=list_bitmap,
        offset=0,
    )

    # Now define the master array data.
    var top_offsets = Buffer.from_values[DType.int32](0, 2, 5, 6)
    var top_bitmap = ArcPointer(Bitmap.alloc(4))
    top_bitmap[].unsafe_range_set(0, 4, True)
    return ListArray(
        ArrayData(
            dtype=list_(list_(materialize[data_type]())),
            length=4,
            buffers=[ArcPointer(top_offsets^)],
            children=[ArcPointer(list_data^)],
            bitmap=top_bitmap,
            offset=0,
        )
    )


def build_struct() raises -> StructArray:
    var int_data_a = ArrayData.from_buffer[int32](
        Buffer.from_values[DType.int32](1, 2, 3, 4, 5), 5
    )
    var field_1 = Field("int_data_a", materialize[int32]())

    var int_data_b = ArrayData.from_buffer[int32](
        Buffer.from_values[DType.int32](10, 20, 30), 3
    )
    var field_2 = Field("int_data_b", materialize[int32]())
    bitmap = Bitmap.alloc(2)
    bitmap.unsafe_range_set(0, 2, True)
    var struct_array_data = ArrayData(
        dtype=struct_([field_1^, field_2^]),
        length=2,
        bitmap=ArcPointer(bitmap^),
        offset=0,
        buffers=[],
        children=[ArcPointer(int_data_a^), ArcPointer(int_data_b^)],
    )
    return StructArray(data=struct_array_data^)
