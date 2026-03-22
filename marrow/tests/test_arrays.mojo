from std.testing import assert_equal, assert_true, assert_false, TestSuite
from marrow.arrays import *
from marrow.builders import (
    AnyBuilder,
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
)
from marrow.dtypes import *
from marrow.buffers import Buffer, BufferBuilder
from marrow.bitmap import Bitmap, BitmapBuilder
from marrow.kernels.filter import drop_nulls
from std.reflection import call_location


# @always_inline
# def assert_bitmap_set(
#     ptr: UnsafePointer[UInt8, ImmutExternalOrigin],
#     n_bits: Int,
#     expected_true_pos: List[Int],
#     message: StringLiteral,
# ) -> None:
#     var list_pos = 0
#     for i in range(n_bits):
#         var expected_value = False
#         if list_pos < len(expected_true_pos):
#             if expected_true_pos[list_pos] == i:
#                 expected_value = True
#                 list_pos += 1
#         var current_value = Bool((ptr[i // 8] >> UInt8(i % 8)) & 1)
#         assert_equal(
#             current_value,
#             expected_value,
#             String(
#                 t"{message}: Bitmap index {i} is {current_value}, expected {expected_value} as per list position {list_pos}"
#             )
#             location=call_location(),
#         )


def test_array_data_with_offset() raises:
    """Test ArrayData with offset functionality."""
    # Create ArrayData with offset
    var bitmap = BitmapBuilder.alloc(10)
    var buffer = BufferBuilder.alloc[int8.native](10)

    # Set some data in the buffer
    buffer.unsafe_set[int8.native](2, 100)
    buffer.unsafe_set[int8.native](3, 200)
    buffer.unsafe_set[int8.native](4, 300)

    # Set validity bits (bits 2, 3, 4 are set; offset=2 maps index 0→bit2, etc.)
    bitmap.set_bit(2, True)
    bitmap.set_bit(3, True)
    bitmap.set_bit(4, True)

    # Create ArrayData with offset=2
    var array_data = AnyArray(
        dtype=int8,
        length=3,
        nulls=0,
        bitmap=bitmap.finish(10),
        buffers=[buffer.finish()],
        children=[],
        offset=2,
    )

    assert_equal(array_data.offset, 2)

    # Test is_valid with offset
    assert_true(array_data.is_valid(0))  # Should check bitmap[2]
    assert_true(array_data.is_valid(1))  # Should check bitmap[3]
    assert_true(array_data.is_valid(2))  # Should check bitmap[4]


def test_array_data_fieldwise_init() raises:
    """Test that @fieldwise_init decorator works with offset field."""
    var buffer_b = BufferBuilder.alloc[int8.native](5)
    var buffer = buffer_b.finish()

    # Test creating ArrayData with all fields specified including offset
    var array_data = AnyArray(
        dtype=int8,
        length=5,
        nulls=0,
        bitmap=None,
        buffers=[buffer],
        children=[],
        offset=3,
    )

    assert_equal(array_data.dtype, int8)
    assert_equal(array_data.length, 5)
    assert_equal(array_data.offset, 3)


def test_array_from_primitive() raises:
    var a = array[int32]([1, 2, 3])
    assert_equal(a.length, 3)


def test_array_from_string() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a: AnyArray = s.finish()
    assert_equal(a.length, 2)


def test_array_from_list() raises:
    var ints_b = PrimitiveBuilder[int64]()
    var l = ListBuilder(AnyBuilder(ints_b^))
    var a: AnyArray = l.finish()
    assert_true(a.dtype.is_list())


def test_array_from_struct() raises:
    var fields = [Field("x", int32)]
    var s = StructBuilder(fields^, [], capacity=5)
    var a: AnyArray = s.finish()
    assert_true(a.dtype.is_struct())


def test_array_copy() raises:
    var _sb = BufferBuilder.alloc[int8.native](3)
    var src = AnyArray(
        dtype=int8,
        length=3,
        nulls=0,
        bitmap=None,
        buffers=[_sb.finish()],
        children=[],
        offset=0,
    )
    var copy = src.copy()
    assert_equal(copy.length, src.length)
    assert_equal(copy.dtype, src.dtype)
    assert_equal(copy.offset, src.offset)
    # Mutating copy's length does not affect src
    copy.length = 99
    assert_equal(src.length, 3)


def test_array_move() raises:
    var _ab = BufferBuilder.alloc[int8.native](5)
    var a = AnyArray(
        dtype=int8,
        length=5,
        nulls=0,
        bitmap=None,
        buffers=[_ab.finish()],
        children=[],
        offset=0,
    )
    var b = a^
    assert_equal(b.length, 5)
    assert_equal(b.dtype, int8)


def test_boolean_array() raises:
    var a = BoolBuilder()
    assert_equal(len(a), 0)
    assert_equal(a._capacity, 0)

    a.reserve(3)
    assert_equal(len(a), 0)
    assert_equal(a._capacity, 3)

    a.append(True)
    a.append(False)
    a.append(True)
    assert_equal(len(a), 3)
    assert_equal(a._capacity, 3)

    a.append(True)
    assert_equal(len(a), 4)
    assert_equal(a._capacity, 6)

    var frozen = a.finish_typed()
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))

    assert_equal(frozen.length, 4)


def test_append() raises:
    var a = PrimitiveBuilder[int8]()
    assert_equal(len(a), 0)
    assert_equal(a._capacity, 0)
    a.append(1)
    a.append(2)
    a.append(3)
    assert_equal(len(a), 3)
    assert_true(a._capacity >= len(a))


def test_array_empty() raises:
    var a = array[int32]()
    assert_equal(len(a), 0)


def test_array_from_ints() raises:
    var g = array[int8]([1, 2])
    assert_equal(len(g), 2)
    assert_equal(g[0], 1)
    assert_equal(g[1], 2)

    var b = array([True, False, True])
    assert_equal(len(b), 3)
    assert_true(b[0])
    assert_false(b[1])
    assert_true(b[2])


def test_array_with_nulls() raises:
    var a = array[int32]([1, None, 3])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 1)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_equal(a[0], 1)
    assert_equal(a[2], 3)

    var b = array([True, None, False])
    assert_equal(b.length, 3)
    assert_true(b.is_valid(0))
    assert_false(b.is_valid(1))
    assert_true(b.is_valid(2))


def test_arange() raises:
    var a = arange[int32](1, 5)
    assert_equal(len(a), 4)
    assert_equal(a[0], 1)
    assert_equal(a[1], 2)
    assert_equal(a[2], 3)
    assert_equal(a[3], 4)

    var b = arange[uint8](0, 3)
    assert_equal(len(b), 3)
    assert_equal(b[0], 0)
    assert_equal(b[2], 2)


def test_arange_empty() raises:
    var a = arange[int32](5, 5)
    assert_equal(len(a), 0)


def test_arange_single() raises:
    var a = arange[int64](7, 8)
    assert_equal(len(a), 1)
    assert_equal(a[0], 7)


def test_arange_validity() raises:
    var a = arange[int16](0, 4)
    for i in range(4):
        assert_true(a.is_valid(i))


def test_arange_int8() raises:
    var a = arange[int8](10, 15)
    assert_equal(len(a), 5)
    assert_equal(a[0], 10)
    assert_equal(a[4], 14)


def test_arange_uint64() raises:
    var a = arange[uint64](100, 103)
    assert_equal(len(a), 3)
    assert_equal(a[0], 100)
    assert_equal(a[2], 102)


# TODO: move this to compute kernels
# def test_drop_null() -> None:
#     """Test the drop null function via the compute module."""
#     from marrow.kernels.filter import drop_nulls

#     var primitive_array = array[uint8](
#         [None, 1, None, 3, None, 5, None, 7, None, 9]
#     )
#     # Check the setup.
#     assert_equal(primitive_array.null_count(), 5)
#     assert_bitmap_set(
#         primitive_array.bitmap.unsafe_ptr(),
#         primitive_array.length,
#         [1, 3, 5, 7, 9],
#         "check setup",
#     )

#     var result = drop_nulls[uint8](primitive_array)
#     assert_equal(result.unsafe_get(0), 1)
#     assert_equal(result.unsafe_get(1), 3)
#     assert_equal(result.null_count(), 0)


def test_primitive_array_with_offset() raises:
    """Test PrimitiveArray with offset functionality."""
    var b = PrimitiveBuilder[int32](10)
    b.append(100)
    b.append(200)
    b.append(300)
    b.append(400)
    b.append(500)
    var arr = b.finish_typed()

    # Default offset should be 0
    assert_equal(arr.offset, 0)
    assert_equal(arr[0], 100)
    assert_equal(arr[1], 200)

    # Create a zero-copy slice, should point to the same buffers.
    var sliced = arr.slice(2)
    assert_equal(sliced.offset, 2)

    # Test that offset affects get operations
    assert_equal(sliced[0], 300)  # Should get arr[2]
    assert_equal(sliced[1], 400)  # Should get arr[3]
    assert_equal(sliced[2], 500)  # Should get arr[4]


def test_primitive_array_nulls_with_offset() raises:
    """Test nulls() creates an array with all null values and default offset."""
    var null_arr = nulls[int64](5)
    assert_equal(null_arr.offset, 0)

    # All elements should be invalid (null)
    for i in range(5):
        assert_false(null_arr.is_valid(i))


# TODO: expose capacity() on builders and test that as well
def test_string_builder() raises:
    var a = StringBuilder()
    assert_equal(len(a), 0)
    assert_equal(a._capacity, 0)

    a.reserve(2)
    assert_equal(len(a), 0)
    assert_equal(a._capacity, 2)

    a.append("hello")
    a.append("world")
    assert_equal(len(a), 2)
    assert_equal(a._capacity, 2)

    var frozen = a.finish_typed()
    assert_equal(frozen[0], "hello")
    assert_equal(frozen[1], "world")


def test_list_bool_array() raises:
    var bool_b = BoolBuilder(3)
    bool_b.append(True)
    bool_b.append_null()
    bool_b.append(True)
    var list_b = ListBuilder(AnyBuilder(bool_b^))
    list_b.append_valid()
    var lists = list_b.finish_typed()
    assert_equal(len(lists), 1)

    # TODO: fix listarray.unsafe_get
    var first_value = lists[0]
    var bool_array = first_value.as_bool()
    assert_true(bool_array[0])
    assert_false(bool_array[1])
    assert_true(bool_array[2])


def test_list_str() raises:
    var str_b = StringBuilder()
    str_b.append("hello")
    str_b.append("world")
    var list_b = ListBuilder(AnyBuilder(str_b^))
    list_b.append_valid()
    var lists = list_b.finish_typed()
    assert_equal(len(lists), 1)

    var first_value = StringArray(lists[0])
    assert_equal(first_value[0], "hello")
    assert_equal(first_value[1], "world")


def test_list_of_list() raises:
    var top_b = ListBuilder(
        AnyBuilder(
            ListBuilder(
                AnyBuilder(PrimitiveBuilder[int64](capacity=10)), capacity=6
            )
        ),
        capacity=3,
    )
    var middle = top_b.values().as_list()
    var child = middle[].values().as_primitive[int64]()
    child[].append(1)
    child[].append(2)
    middle[].append_valid()
    child[].append(3)
    child[].append(4)
    middle[].append_valid()
    top_b.append_valid()
    child[].append(5)
    child[].append(6)
    child[].append(7)
    middle[].append_valid()
    middle[].append_null()
    child[].append(8)
    middle[].append_valid()
    top_b.append_valid()
    child[].append(9)
    child[].append(10)
    middle[].append_valid()
    top_b.append_valid()
    list2 = top_b.finish_typed()

    top = ListArray(list2[0])
    middle_0 = top[0]
    bottom = middle_0.as_primitive[int64]()
    assert_equal(bottom[1], 2)
    assert_equal(bottom[0], 1)
    middle_1 = top[1]
    bottom = middle_1.as_primitive[int64]()
    assert_equal(bottom[0], 3)
    assert_equal(bottom[1], 4)


def test_fixed_size_list_int_array() raises:
    """Construct a FixedSizeListArray of int64 lists, size=3."""
    var ints_b = PrimitiveBuilder[int64](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=3)
    builder.append_valid()
    builder.append_valid()
    assert_equal(builder.dtype(), fixed_size_list_(int64, 3))
    assert_equal(len(builder), 2)
    var fsl = builder.finish_typed()
    assert_equal(len(fsl), 2)
    assert_equal(fsl.dtype.size, 3)

    # First list: [1, 2, 3]
    var first = fsl[0].as_int64()
    assert_equal(len(first), 3)
    assert_equal(first[0], 1)
    assert_equal(first[1], 2)
    assert_equal(first[2], 3)

    # Second list: [4, 5, 6]
    var second = fsl[1].as_int64()
    assert_equal(second[0], 4)
    assert_equal(second[1], 5)
    assert_equal(second[2], 6)


def test_fixed_size_list_roundtrip() raises:
    """FixedSizeListArray round-trip through builder."""
    var ints_b = PrimitiveBuilder[int32](4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish_typed()

    assert_true(fsl.dtype.is_fixed_size_list())
    assert_equal(fsl.dtype.size, 2)
    assert_equal(len(fsl), 2)

    var first = fsl[0].as_int32()
    assert_equal(first[0], 10)
    assert_equal(first[1], 20)


def test_fixed_size_list_with_nulls() raises:
    """FixedSizeListArray with null lists."""
    var ints_b = PrimitiveBuilder[int64](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(
        AnyBuilder(ints_b^), list_size=3, capacity=3
    )
    builder.append_valid()
    builder.append_valid()
    builder.append_null()
    assert_equal(len(builder), 3)

    var fsl = builder.finish_typed()
    assert_true(fsl.is_valid(0))
    assert_true(fsl.is_valid(1))
    assert_false(fsl.is_valid(2))

    # unsafe_get on valid entries returns correct values even when array has nulls
    var first = fsl[0].as_int64()
    assert_equal(first[0], 1)
    assert_equal(first[1], 2)
    assert_equal(first[2], 3)
    var second = fsl[1].as_int64()
    assert_equal(second[0], 4)
    assert_equal(second[1], 5)
    assert_equal(second[2], 6)


def test_fixed_size_list_unsafe_get_dtype() raises:
    # unsafe_get returns a slice with the child element dtype, not the list dtype.
    var ints_b = PrimitiveBuilder[int32](4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish_typed()

    var slice0 = fsl[0]
    assert_equal(slice0.dtype, int32)
    assert_equal(slice0.length, 2)
    assert_equal(slice0.offset, 0)

    var slice1 = fsl[1]
    assert_equal(slice1.dtype, int32)
    assert_equal(slice1.length, 2)
    assert_equal(slice1.offset, 2)


# # def test_fixed_size_list_pretty_print():
# #     """Pretty printing FixedSizeListArray."""
# #     var ints_b = PrimitiveBuilder[int64](4)
# #     ints_b.append(1)
# #     ints_b.append(2)
# #     ints_b.append(3)
# #     ints_b.append(4)
# #     var builder = FixedSizeListBuilder(ints_b, list_size=2)
# #     builder.append_valid()
# #     builder.append_valid()
# #     var fsl = builder.finish()
# #     var s = String(AnyArray(fsl^))
# #     assert_true("FixedSizeListArray" in s)


def test_struct_array() raises:
    var fields = [
        Field("id", int64),
        Field("name", string),
        Field("active", bool_),
    ]

    var struct_builder = StructBuilder(fields^, [], capacity=10)
    assert_equal(len(struct_builder), 0)
    assert_equal(struct_builder._capacity, 10)

    var data: AnyArray = struct_builder.finish()
    assert_equal(data.length, 0)
    assert_true(data.dtype.is_struct())
    assert_equal(len(data.dtype.fields), 3)
    assert_equal(data.dtype.fields[0].name, "id")
    assert_equal(data.dtype.fields[1].name, "name")
    assert_equal(data.dtype.fields[2].name, "active")


def test_struct_array_unsafe_get() raises:
    var a_b = PrimitiveBuilder[int32](5)
    a_b.append(1)
    a_b.append(2)
    a_b.append(3)
    a_b.append(4)
    a_b.append(5)
    var b_b = PrimitiveBuilder[int32](3)
    b_b.append(10)
    b_b.append(20)
    b_b.append(30)
    var fields: List[Field] = [
        Field("int_data_a", int32),
        Field("int_data_b", int32),
    ]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^), AnyBuilder(b_b^)]
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append_valid()
    sb.append_valid()
    var struct_array = sb.finish_typed()
    ref int_data_a = struct_array.unsafe_get("int_data_a")
    var int_a = int_data_a.as_primitive[int32]()
    assert_equal(int_a[0], 1)
    assert_equal(int_a[4], 5)
    ref int_data_b = struct_array.unsafe_get("int_data_b")
    var int_b = int_data_b.as_primitive[int32]()
    assert_equal(int_b[0], 10)
    assert_equal(int_b[2], 30)


def test_chunked_array() raises:
    var arrays: List[AnyArray] = [array[uint8]([0]), array[uint8]([0, 1])]

    var chunked_array = ChunkedArray(int8, arrays^)
    assert_equal(chunked_array.length, 3)

    assert_equal(chunked_array.chunk(0).length, 1)
    var second_chunk = chunked_array.chunk(1).copy().as_uint8()
    assert_equal(second_chunk.length, 2)
    assert_equal(second_chunk[0], 0)
    assert_equal(second_chunk[1], 1)


def test_combine_chunked_array() raises:
    var arrays: List[AnyArray] = [array[uint8]([0]), array[uint8]([0, 1])]

    var chunked_array = ChunkedArray(uint8, arrays^)
    assert_equal(chunked_array.length, 3)
    assert_equal(len(chunked_array.chunks), 2)
    assert_equal(chunked_array.chunk(1).copy().as_uint8()[1], 1)

    var combined_array = chunked_array^.combine_chunks()
    assert_equal(combined_array.length, 3)
    assert_equal(combined_array.dtype, uint8)
    # Single concatenated values buffer: [0, 0, 1]
    assert_equal(combined_array.buffers[0].unsafe_get(0), 0)
    assert_equal(combined_array.buffers[0].unsafe_get(2), 1)


def test_primitive_finish_shrinks() raises:
    """Freeze() on an over-allocated builder trims capacity to length."""
    var a = PrimitiveBuilder[int64](capacity=100)
    a.append(42)
    a.append(99)
    var frozen = a.finish_typed()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0], 42)
    assert_equal(frozen[1], 99)

    var values_buffer = frozen.buffer
    # 2 int64 values = 16 bytes, but buffer padded to 64 bytes for alignment
    assert_equal(values_buffer.size, 64)


def test_primitive_finish_via_append() raises:
    """Freeze() works on a builder built with append() (auto-grow capacity)."""
    var a = PrimitiveBuilder[int64]()
    a.append(1)
    a.append(2)
    a.append(3)
    var frozen = a.finish_typed()
    assert_equal(frozen.length, 3)
    assert_equal(frozen[0], 1)
    assert_equal(frozen[2], 3)


def test_primitive_finish_preserves_nulls() raises:
    """Freeze() preserves null validity information."""
    var a = PrimitiveBuilder[int64](capacity=3)
    a.append(1)
    a.append_null()
    a.append(3)
    var frozen = a.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_primitive_finish_converts_to_array() raises:
    """PrimitiveBuilder.finish_typed() returns a typed PrimitiveArray."""
    var a = PrimitiveBuilder[int64]()
    a.append(7)
    a.append(8)
    var frozen = a.finish_typed()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0], 7)
    assert_equal(frozen[1], 8)


def test_getitem_bounds_check() raises:
    """__getitem__ raises on out-of-bounds access."""
    var b = PrimitiveBuilder[int64]()
    b.append(1)
    b.append(2)
    var a = b.finish_typed()
    try:
        _ = a[5]
        assert_true(False, "should have raised")
    except:
        pass
    try:
        _ = a[-1]
        assert_true(False, "should have raised")
    except:
        pass
    assert_equal(a[0], 1)
    assert_equal(a[1], 2)


def test_setitem_bounds_check() raises:
    """PrimitiveArray __getitem__ returns correct values."""
    var a = PrimitiveBuilder[int64]()
    a.append(99)
    var frozen = a.finish_typed()
    assert_equal(frozen[0], 99)


def test_string_finish_zero_copy() raises:
    """Freeze() on an exact-size StringBuilder moves buffers."""
    var s = StringBuilder(capacity=2)
    s.append("hello")
    s.append("world")
    var frozen = s.finish_typed()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0], "hello")
    assert_equal(frozen[1], "world")


def test_string_finish_shrinks() raises:
    """Freeze() on an over-allocated StringBuilder trims to exact size."""
    var s = StringBuilder(capacity=100)
    s.append("hi")
    var frozen = s.finish_typed()
    assert_equal(frozen.length, 1)
    assert_equal(frozen[0], "hi")


def test_string_getitem_bounds_check() raises:
    """StringArray __getitem__ raises on out-of-bounds."""
    var s = StringBuilder()
    s.append("a")
    var frozen = s.finish_typed()
    assert_equal(frozen[0], "a")
    try:
        _ = frozen[1]
        assert_true(False, "should have raised")
    except:
        pass


# ---------------------------------------------------------------------------
# String representation tests (__str__ / write_to)
# ---------------------------------------------------------------------------


def test_str_primitive_array() raises:
    var a = array[int32]([1, 2, 3])
    var s = String(a)
    assert_true("PrimitiveArray" in s)
    assert_true("1" in s)
    assert_true("2" in s)
    assert_true("3" in s)


def test_str_primitive_array_with_nulls() raises:
    var a = array[int32]([1, None, 3])
    var s = String(a)
    assert_true("NULL" in s)
    assert_true("1" in s)
    assert_true("3" in s)


def test_str_bool_array() raises:
    var a = array([True, False, True])
    var s = String(a)
    assert_true("PrimitiveArray" in s)


def test_str_string_array() raises:
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("world")
    var a = sb.finish_typed()
    var s = String(a)
    assert_true("StringArray" in s)
    assert_true("hello" in s)
    assert_true("world" in s)


def test_str_string_array_with_nulls() raises:
    var sb = StringBuilder(3)
    sb.append("foo")
    sb.append_null()
    sb.append("bar")
    var a = sb.finish_typed()
    var s = String(a)
    assert_true("NULL" in s)
    assert_true("foo" in s)
    assert_true("bar" in s)


def test_str_list_array() raises:
    var ints_b = PrimitiveBuilder[int64]()
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    list_b.append_valid()
    var lists = list_b.finish_typed()
    var s = String(lists)
    assert_true("ListArray" in s)


def test_str_fixed_size_list_array() raises:
    var ints_b = PrimitiveBuilder[int64](4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish_typed()
    var s = String(fsl)
    assert_true("FixedSizeListArray" in s)


def test_str_struct_array() raises:
    var a_b = PrimitiveBuilder[int32](2)
    a_b.append(1)
    a_b.append(2)
    var fields: List[Field] = [Field("x", int32)]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^)]
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish_typed()
    var s = String(sa)
    assert_true("StructArray" in s)
    assert_true("x" in s)


# ---------------------------------------------------------------------------
# is_valid tests for all array types
# ---------------------------------------------------------------------------


def test_string_array_is_valid() raises:
    var sb = StringBuilder(4)
    sb.append("a")
    sb.append_null()
    sb.append("c")
    sb.append_null()
    var a = sb.finish_typed()
    assert_equal(a.null_count(), 2)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_false(a.is_valid(3))


def test_string_array_no_nulls() raises:
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("world")
    var a = sb.finish_typed()
    assert_equal(a.null_count(), 0)
    assert_true(a.is_valid(0))
    assert_true(a.is_valid(1))


def test_list_array_is_valid() raises:
    var ints_b = PrimitiveBuilder[int64]()
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    list_b.append_valid()
    list_b.append_null()
    list_b.append_valid()
    var lists = list_b.finish_typed()
    assert_equal(len(lists), 3)
    assert_equal(lists.null_count(), 1)
    assert_true(lists.is_valid(0))
    assert_false(lists.is_valid(1))
    assert_true(lists.is_valid(2))


def test_struct_array_is_valid() raises:
    var a_b = PrimitiveBuilder[int32](3)
    a_b.append(10)
    a_b.append(20)
    a_b.append(30)
    var fields: List[Field] = [Field("val", int32)]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^)]
    var sb = StructBuilder(fields^, children^, capacity=3)
    sb.append_valid()
    sb.append_null()
    sb.append_valid()
    var sa = sb.finish_typed()
    assert_equal(len(sa), 3)
    assert_equal(sa.null_count(), 1)
    assert_true(sa.is_valid(0))
    assert_false(sa.is_valid(1))
    assert_true(sa.is_valid(2))


# ---------------------------------------------------------------------------
# __getitem__ tests for all array types
# ---------------------------------------------------------------------------


def test_string_array_getitem() raises:
    var sb = StringBuilder()
    sb.append("alpha")
    sb.append("beta")
    sb.append("gamma")
    var a = sb.finish_typed()
    assert_equal(a[0], "alpha")
    assert_equal(a[1], "beta")
    assert_equal(a[2], "gamma")


def test_string_array_getitem_bounds() raises:
    var sb = StringBuilder()
    sb.append("only")
    var a = sb.finish_typed()
    try:
        _ = a[1]
        assert_true(False, "should have raised")
    except:
        pass
    try:
        _ = a[-1]
        assert_true(False, "should have raised")
    except:
        pass


def test_list_array_getitem() raises:
    var ints_b = PrimitiveBuilder[int64]()
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    var child = list_b.values().as_primitive[int64]()
    child[].append(10)
    child[].append(20)
    list_b.append_valid()  # [10, 20]
    child[].append(30)
    child[].append(40)
    child[].append(50)
    list_b.append_valid()  # [30, 40, 50]
    var lists = list_b.finish_typed()
    var first = lists[0]
    assert_equal(first.length, 2)
    var second = lists[1]
    assert_equal(second.length, 3)


def test_list_array_getitem_bounds() raises:
    var ints_b = PrimitiveBuilder[int64]()
    ints_b.append(1)
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    list_b.append_valid()
    var lists = list_b.finish_typed()
    try:
        _ = lists[1]
        assert_true(False, "should have raised")
    except:
        pass
    try:
        _ = lists[-1]
        assert_true(False, "should have raised")
    except:
        pass


def test_fixed_size_list_getitem() raises:
    var ints_b = PrimitiveBuilder[int32](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=3)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish_typed()
    var first = fsl[0].as_int32()
    assert_equal(first[0], 1)
    assert_equal(first[1], 2)
    assert_equal(first[2], 3)
    var second = fsl[1].as_int32()
    assert_equal(second[0], 4)
    assert_equal(second[1], 5)
    assert_equal(second[2], 6)


def test_fixed_size_list_getitem_bounds() raises:
    var ints_b = PrimitiveBuilder[int32](3)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=3)
    builder.append_valid()
    var fsl = builder.finish_typed()
    try:
        _ = fsl[1]
        assert_true(False, "should have raised")
    except:
        pass
    try:
        _ = fsl[-1]
        assert_true(False, "should have raised")
    except:
        pass


def test_struct_array_field_by_index() raises:
    var a_b = PrimitiveBuilder[int32](2)
    a_b.append(1)
    a_b.append(2)
    var b_b = StringBuilder()
    b_b.append("x")
    b_b.append("y")
    var fields: List[Field] = [Field("id", int32), Field("name", string)]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^), AnyBuilder(b_b^)]
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish_typed()

    var id_arr = sa.field(0).as_int32()
    assert_equal(id_arr[0], 1)
    assert_equal(id_arr[1], 2)

    var name_arr = sa.field(1).as_string()
    assert_equal(name_arr[0], "x")
    assert_equal(name_arr[1], "y")


def test_struct_array_field_by_name() raises:
    var a_b = PrimitiveBuilder[int32](2)
    a_b.append(10)
    a_b.append(20)
    var fields: List[Field] = [Field("val", int32)]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^)]
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish_typed()

    var val_arr = sa.field("val").as_int32()
    assert_equal(val_arr[0], 10)
    assert_equal(val_arr[1], 20)


def test_struct_array_field_bounds() raises:
    var a_b = PrimitiveBuilder[int32](1)
    a_b.append(1)
    var fields: List[Field] = [Field("x", int32)]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^)]
    var sb = StructBuilder(fields^, children^, capacity=1)
    sb.append_valid()
    var sa = sb.finish_typed()
    try:
        _ = sa.field(1)
        assert_true(False, "should have raised")
    except:
        pass
    try:
        _ = sa.field("nonexistent")
        assert_true(False, "should have raised")
    except:
        pass


# ---------------------------------------------------------------------------
# Property / offset tests for underrepresented types
# ---------------------------------------------------------------------------


def test_string_array_slice() raises:
    var sb = StringBuilder()
    sb.append("aa")
    sb.append("bb")
    sb.append("cc")
    sb.append("dd")
    var a = sb.finish_typed()
    var sliced = a.slice(2)
    assert_equal(len(sliced), 2)
    assert_equal(sliced.offset, 2)
    assert_equal(sliced[0], "cc")
    assert_equal(sliced[1], "dd")


def test_fixed_size_list_len_and_null_count() raises:
    var ints_b = PrimitiveBuilder[int64](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=2)
    builder.append_valid()
    builder.append_null()
    builder.append_valid()
    var fsl = builder.finish_typed()
    assert_equal(len(fsl), 3)
    assert_equal(fsl.null_count(), 1)


def test_list_array_null_count() raises:
    var ints_b = PrimitiveBuilder[int64]()
    ints_b.append(1)
    ints_b.append(2)
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    list_b.append_valid()
    list_b.append_null()
    var lists = list_b.finish_typed()
    assert_equal(lists.null_count(), 1)
    assert_equal(len(lists), 2)


def test_primitive_array_no_nulls_is_valid() raises:
    var a = array[int64]([10, 20, 30])
    assert_equal(a.null_count(), 0)
    for i in range(3):
        assert_true(a.is_valid(i))


# ---------------------------------------------------------------------------
# slice() tests
# ---------------------------------------------------------------------------


def test_primitive_array_slice_with_length() raises:
    var a = array[int32]([10, 20, 30, 40, 50])
    var s = a.slice(1, 3)
    assert_equal(len(s), 3)
    assert_equal(s[0], 20)
    assert_equal(s[1], 30)
    assert_equal(s[2], 40)


def test_string_array_slice_with_length() raises:
    var sb = StringBuilder()
    sb.append("aa")
    sb.append("bb")
    sb.append("cc")
    sb.append("dd")
    var a = sb.finish_typed()
    var s = a.slice(1, 2)
    assert_equal(len(s), 2)
    assert_equal(s[0], "bb")
    assert_equal(s[1], "cc")


def test_list_array_slice() raises:
    var ints_b = PrimitiveBuilder[int64]()
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    var child = list_b.values().as_primitive[int64]()
    child[].append(1)
    child[].append(2)
    list_b.append_valid()
    child[].append(3)
    list_b.append_valid()
    child[].append(4)
    child[].append(5)
    list_b.append_valid()
    var lists = list_b.finish_typed()
    var s = lists.slice(1)
    assert_equal(len(s), 2)


def test_fixed_size_list_slice() raises:
    var ints_b = PrimitiveBuilder[int32](6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=2)
    builder.append_valid()
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish_typed()
    var s = fsl.slice(1, 2)
    assert_equal(len(s), 2)
    var first = s[0].as_int32()
    assert_equal(first[0], 3)
    assert_equal(first[1], 4)


# ---------------------------------------------------------------------------
# BoolArray true_count / false_count
# ---------------------------------------------------------------------------


def test_bool_true_count() raises:
    var a = array([True, True, False, True, False])
    assert_equal(a.true_count(), 3)
    assert_equal(a.false_count(), 2)


def test_bool_true_count_with_nulls() raises:
    var a = array([True, None, False, True])
    assert_equal(a.true_count(), 2)
    assert_equal(a.false_count(), 1)
    assert_equal(a.null_count(), 1)


def test_bool_all_false() raises:
    var a = array([False, False, False])
    assert_equal(a.true_count(), 0)
    assert_equal(a.false_count(), 3)


def test_bool_all_true() raises:
    var a = array([True, True, True])
    assert_equal(a.true_count(), 3)
    assert_equal(a.false_count(), 0)


# ---------------------------------------------------------------------------
# flatten() and value_lengths() tests
# ---------------------------------------------------------------------------


def test_list_array_flatten() raises:
    var ints_b = PrimitiveBuilder[int64]()
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    var child = list_b.values().as_primitive[int64]()
    child[].append(1)
    child[].append(2)
    list_b.append_valid()
    child[].append(3)
    list_b.append_valid()
    var lists = list_b.finish_typed()
    var flat = lists.flatten()
    assert_equal(flat.length, 3)


def test_list_array_value_lengths() raises:
    var ints_b = PrimitiveBuilder[int64]()
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    var child = list_b.values().as_primitive[int64]()
    child[].append(1)
    child[].append(2)
    list_b.append_valid()  # length 2
    child[].append(3)
    list_b.append_valid()  # length 1
    child[].append(4)
    child[].append(5)
    child[].append(6)
    list_b.append_valid()  # length 3
    var lists = list_b.finish_typed()
    var lengths = lists.value_lengths()
    assert_equal(len(lengths), 3)
    assert_equal(lengths[0], 2)
    assert_equal(lengths[1], 1)
    assert_equal(lengths[2], 3)


def test_fixed_size_list_flatten() raises:
    var ints_b = PrimitiveBuilder[int32](4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(AnyBuilder(ints_b^), list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish_typed()
    var flat = fsl.flatten()
    assert_equal(flat.length, 4)


def test_struct_array_flatten() raises:
    var a_b = PrimitiveBuilder[int32](2)
    a_b.append(1)
    a_b.append(2)
    var b_b = StringBuilder()
    b_b.append("x")
    b_b.append("y")
    var fields: List[Field] = [Field("id", int32), Field("name", string)]
    var children: List[AnyBuilder] = [AnyBuilder(a_b^), AnyBuilder(b_b^)]
    var sb = StructBuilder(fields^, children^, capacity=2)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish_typed()
    var flat = sa.flatten()
    assert_equal(len(flat), 2)
    assert_equal(flat[0].length, 2)
    assert_equal(flat[1].length, 2)


# ---------------------------------------------------------------------------
# Equality tests
# ---------------------------------------------------------------------------


def test_primitive_array_eq() raises:
    # Fast path: no nulls, offset=0 — uses Buffer.__eq__
    var a = array[int32]([1, 2, 3])
    var b = array[int32]([1, 2, 3])
    assert_true(a == b)


def test_primitive_array_eq_unequal() raises:
    var a = array[int32]([1, 2, 3])
    var b = array[int32]([1, 2, 4])
    assert_false(a == b)


def test_primitive_array_eq_length_mismatch() raises:
    var a = array[int32]([1, 2, 3])
    var b = array[int32]([1, 2])
    assert_false(a == b)


def test_primitive_array_eq_sliced() raises:
    # Regression test: sliced arrays with non-zero offset must compare correctly.
    # Old _arrays_equal bug: compared raw buffer bytes ignoring offset.
    var a = array[int32]([10, 20, 30, 40, 50])
    var b = array[int32]([10, 20, 30, 40, 50])
    var sa = a.slice(1, 3)  # [20, 30, 40], offset=1
    var sb = b.slice(1, 3)  # [20, 30, 40], offset=1
    assert_true(sa == sb)


def test_primitive_array_eq_sliced_unequal() raises:
    var a = array[int32]([10, 20, 30, 40, 50])
    var b = array[int32]([10, 20, 99, 40, 50])
    var sa = a.slice(1, 3)  # [20, 30, 40]
    var sb = b.slice(1, 3)  # [20, 99, 40]
    assert_false(sa == sb)


def test_primitive_array_eq_nulls_equal() raises:
    var a = array[int32]([1, None, 3])
    var b = array[int32]([1, None, 3])
    assert_true(a == b)


def test_primitive_array_eq_nulls_mismatch_count() raises:
    var a = array[int32]([1, None, 3])
    var b = array[int32]([1, 2, 3])
    assert_false(a == b)


def test_primitive_array_eq_nulls_mismatch_pattern() raises:
    # Same null count but different null positions
    var a = array[int32]([None, 2, 3])
    var b = array[int32]([1, None, 3])
    assert_false(a == b)


def test_bool_array_eq() raises:
    var a = array([True, False, True])
    var b = array([True, False, True])
    assert_true(a == b)
    var c = array([True, True, True])
    assert_false(a == c)


def test_string_array_eq() raises:
    var sa = StringBuilder()
    sa.append("hello")
    sa.append("world")
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("world")
    assert_true(sa.finish_typed() == sb.finish_typed())


def test_string_array_eq_unequal() raises:
    var sa = StringBuilder()
    sa.append("hello")
    sa.append("world")
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("mars")
    assert_false(sa.finish_typed() == sb.finish_typed())


def test_string_array_eq_sliced() raises:
    # Sliced string arrays with matching logical values are equal.
    var sa = StringBuilder()
    sa.append("a")
    sa.append("b")
    sa.append("c")
    sa.append("d")
    var sb = StringBuilder()
    sb.append("a")
    sb.append("b")
    sb.append("c")
    sb.append("d")
    var a = sa.finish_typed()
    var b = sb.finish_typed()
    assert_true(a.slice(1, 2) == b.slice(1, 2))


def test_string_array_eq_nulls() raises:
    var sa = StringBuilder(3)
    sa.append("x")
    sa.append_null()
    sa.append("z")
    var sb = StringBuilder(3)
    sb.append("x")
    sb.append_null()
    sb.append("z")
    assert_true(sa.finish_typed() == sb.finish_typed())


def test_list_array_eq() raises:
    var ints_a = PrimitiveBuilder[int64]()
    var list_a = ListBuilder(AnyBuilder(ints_a^))
    var child_a = list_a.values().as_primitive[int64]()
    child_a[].append(1)
    child_a[].append(2)
    list_a.append_valid()
    child_a[].append(3)
    list_a.append_valid()
    var a = list_a.finish_typed()

    var ints_b = PrimitiveBuilder[int64]()
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    var child_b = list_b.values().as_primitive[int64]()
    child_b[].append(1)
    child_b[].append(2)
    list_b.append_valid()
    child_b[].append(3)
    list_b.append_valid()
    var b = list_b.finish_typed()

    assert_true(a == b)


def test_list_array_eq_unequal() raises:
    var ints_a = PrimitiveBuilder[int64]()
    var list_a = ListBuilder(AnyBuilder(ints_a^))
    var child_a = list_a.values().as_primitive[int64]()
    child_a[].append(1)
    child_a[].append(2)
    list_a.append_valid()
    var a = list_a.finish_typed()

    var ints_b = PrimitiveBuilder[int64]()
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    var child_b = list_b.values().as_primitive[int64]()
    child_b[].append(1)
    child_b[].append(99)
    list_b.append_valid()
    var b = list_b.finish_typed()

    assert_false(a == b)


def test_list_array_eq_nulls() raises:
    var ints_a = PrimitiveBuilder[int64]()
    ints_a.append(1)
    var list_a = ListBuilder(AnyBuilder(ints_a^))
    list_a.append_valid()
    list_a.append_null()
    var a = list_a.finish_typed()

    var ints_b = PrimitiveBuilder[int64]()
    ints_b.append(1)
    var list_b = ListBuilder(AnyBuilder(ints_b^))
    list_b.append_valid()
    list_b.append_null()
    var b = list_b.finish_typed()

    assert_true(a == b)


def test_fixed_size_list_array_eq() raises:
    var a_b = PrimitiveBuilder[int32](4)
    a_b.append(1)
    a_b.append(2)
    a_b.append(3)
    a_b.append(4)
    var builder_a = FixedSizeListBuilder(AnyBuilder(a_b^), list_size=2)
    builder_a.append_valid()
    builder_a.append_valid()

    var b_b = PrimitiveBuilder[int32](4)
    b_b.append(1)
    b_b.append(2)
    b_b.append(3)
    b_b.append(4)
    var builder_b = FixedSizeListBuilder(AnyBuilder(b_b^), list_size=2)
    builder_b.append_valid()
    builder_b.append_valid()

    assert_true(builder_a.finish_typed() == builder_b.finish_typed())


def test_fixed_size_list_array_eq_unequal() raises:
    var a_b = PrimitiveBuilder[int32](4)
    a_b.append(1)
    a_b.append(2)
    a_b.append(3)
    a_b.append(4)
    var builder_a = FixedSizeListBuilder(AnyBuilder(a_b^), list_size=2)
    builder_a.append_valid()
    builder_a.append_valid()

    var b_b = PrimitiveBuilder[int32](4)
    b_b.append(1)
    b_b.append(2)
    b_b.append(3)
    b_b.append(99)
    var builder_b = FixedSizeListBuilder(AnyBuilder(b_b^), list_size=2)
    builder_b.append_valid()
    builder_b.append_valid()

    assert_false(builder_a.finish_typed() == builder_b.finish_typed())


def test_struct_array_eq() raises:
    var a_b = PrimitiveBuilder[int32](2)
    a_b.append(1)
    a_b.append(2)
    var fields_a = List[Field]()
    fields_a.append(Field("x", int32))
    var children_a = List[AnyBuilder]()
    children_a.append(AnyBuilder(a_b^))
    var sa = StructBuilder(fields_a^, children_a^, capacity=2)
    sa.append_valid()
    sa.append_valid()

    var b_b = PrimitiveBuilder[int32](2)
    b_b.append(1)
    b_b.append(2)
    var fields_b = List[Field]()
    fields_b.append(Field("x", int32))
    var children_b = List[AnyBuilder]()
    children_b.append(AnyBuilder(b_b^))
    var sb = StructBuilder(fields_b^, children_b^, capacity=2)
    sb.append_valid()
    sb.append_valid()

    assert_true(sa.finish_typed() == sb.finish_typed())


def test_struct_array_eq_unequal() raises:
    var a_b = PrimitiveBuilder[int32](2)
    a_b.append(1)
    a_b.append(2)
    var fields_a = List[Field]()
    fields_a.append(Field("x", int32))
    var children_a = List[AnyBuilder]()
    children_a.append(AnyBuilder(a_b^))
    var sa = StructBuilder(fields_a^, children_a^, capacity=2)
    sa.append_valid()
    sa.append_valid()

    var b_b = PrimitiveBuilder[int32](2)
    b_b.append(1)
    b_b.append(99)
    var fields_b = List[Field]()
    fields_b.append(Field("x", int32))
    var children_b = List[AnyBuilder]()
    children_b.append(AnyBuilder(b_b^))
    var sb = StructBuilder(fields_b^, children_b^, capacity=2)
    sb.append_valid()
    sb.append_valid()

    assert_false(sa.finish_typed() == sb.finish_typed())


def test_struct_array_eq_dtype_mismatch() raises:
    var a_b = PrimitiveBuilder[int32](1)
    a_b.append(1)
    var fields_a = List[Field]()
    fields_a.append(Field("x", int32))
    var children_a = List[AnyBuilder]()
    children_a.append(AnyBuilder(a_b^))
    var sa = StructBuilder(fields_a^, children_a^, capacity=1)
    sa.append_valid()

    var b_b = PrimitiveBuilder[int32](1)
    b_b.append(1)
    var fields_b = List[Field]()
    fields_b.append(Field("y", int32))  # different field name
    var children_b = List[AnyBuilder]()
    children_b.append(AnyBuilder(b_b^))
    var sb = StructBuilder(fields_b^, children_b^, capacity=1)
    sb.append_valid()

    assert_false(sa.finish_typed() == sb.finish_typed())


def test_array_eq_dtype_mismatch() raises:
    # Type-erased AnyArray: int32 vs int64 → False
    var a: AnyArray = array[int32]([1, 2, 3])
    var b: AnyArray = array[int64]([1, 2, 3])
    assert_false(a == b)


def test_array_eq_via_dispatch() raises:
    # Equal arrays accessed as type-erased AnyArray verify dispatch works.
    var a: AnyArray = array[int32]([10, 20, 30])
    var b: AnyArray = array[int32]([10, 20, 30])
    assert_true(a == b)
    assert_true(a == b)


def test_primitive_array_list_literal() raises:
    var arr: PrimitiveArray[int64] = [1, 2, 3, 4, 5]
    assert_equal(len(arr), 5)
    assert_equal(arr[0], 1)
    assert_equal(arr[4], 5)
    assert_equal(arr.null_count(), 0)


def test_primitive_array_list_literal_float() raises:
    var arr: PrimitiveArray[float64] = [1.0, 2.5, 3.14]
    assert_equal(len(arr), 3)
    assert_equal(arr[0], 1.0)


def test_string_array_list_literal() raises:
    var arr: StringArray = ["hello", "world", "foo"]
    assert_equal(len(arr), 3)
    assert_equal(arr[0], "hello")
    assert_equal(arr[1], "world")
    assert_equal(arr[2], "foo")
    assert_equal(arr.null_count(), 0)


def test_primitive_array_list_literal_empty() raises:
    var arr: PrimitiveArray[int32] = []
    assert_equal(len(arr), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
