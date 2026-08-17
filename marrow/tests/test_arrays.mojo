from std.testing import assert_equal, assert_true, assert_false
from ..arrays import *
from ..builders import (
    array,
    arange,
    nulls,
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
    DictionaryBuilder,
    Int8Builder,
    Int32Builder,
    Int64Builder,
    Float64Builder,
)
from ..dtypes import *
from ..buffers import Buffer
from ..buffers import Bitmap
from ..kernels.filter import drop_null

from std.reflection import call_location


def test_array_data_with_offset() raises:
    """Test ArrayData with offset functionality."""
    # Create ArrayData with offset
    var bitmap = Bitmap.alloc_zeroed(10)
    var buffer = Buffer.alloc_zeroed[int8.native](10)

    # Set some data in the buffer
    buffer.unsafe_set[int8.native](2, 100)
    buffer.unsafe_set[int8.native](3, 200)
    buffer.unsafe_set[int8.native](4, 300)

    # Set validity bits (bits 2, 3, 4 are set; offset=2 maps index 0→bit2, etc.)
    bitmap.set(2)
    bitmap.set(3)
    bitmap.set(4)

    # Create ArrayData with offset=2
    var array_data = DynArray.from_data(
        ArrayData(
            dtype=int8,
            length=3,
            nulls=0,
            offset=2,
            bitmap=bitmap.to_immutable(),
            buffers=[buffer.to_immutable()],
            children=[],
        )
    )

    assert_equal(array_data.to_data().offset, 2)

    # Test is_valid with offset
    assert_true(array_data.is_valid(0))  # Should check bitmap[2]
    assert_true(array_data.is_valid(1))  # Should check bitmap[3]
    assert_true(array_data.is_valid(2))  # Should check bitmap[4]


def test_array_data_fieldwise_init() raises:
    """Test that @fieldwise_init decorator works with offset field."""
    var buffer_b = Buffer.alloc_zeroed[int8.native](5)
    var buffer = buffer_b.to_immutable()

    # Test creating ArrayData with all fields specified including offset
    var array_data = DynArray.from_data(
        ArrayData(
            dtype=int8,
            length=5,
            nulls=0,
            offset=3,
            bitmap=None,
            buffers=[buffer],
            children=[],
        )
    )

    assert_equal(array_data.dtype(), int8)
    assert_equal(array_data.length(), 5)
    assert_equal(array_data.to_data().offset, 3)


def test_array_from_primitive() raises:
    var a = array([1, 2, 3], int32)
    assert_equal(a.length, 3)


def test_array_from_string() raises:
    var s = StringBuilder()
    s.append("hello")
    s.append("world")
    var a: DynArray = s.finish()
    assert_equal(a.length(), 2)


def test_array_from_list() raises:
    var ints_b = Int64Builder()
    var l = ListBuilder(ints_b^)
    var a: DynArray = l.finish()
    assert_true(a.dtype().is_list())


def test_array_from_struct() raises:
    var s = StructBuilder([field("x", int32)], capacity=5)
    var a: DynArray = s.finish()
    assert_true(a.dtype().is_struct())


def test_array_copy() raises:
    var _sb = Buffer.alloc_zeroed[int8.native](3)
    var src = DynArray.from_data(
        ArrayData(
            dtype=int8,
            length=3,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[_sb.to_immutable()],
            children=[],
        )
    )
    var copy = src.copy()
    assert_equal(copy.length(), src.length())
    assert_equal(copy.dtype(), src.dtype())
    assert_equal(copy.to_data().offset, src.to_data().offset)


def test_array_move() raises:
    var _ab = Buffer.alloc_zeroed[int8.native](5)
    var a = DynArray.from_data(
        ArrayData(
            dtype=int8,
            length=5,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[_ab.to_immutable()],
            children=[],
        )
    )
    var b = a^
    assert_equal(b.length(), 5)
    assert_equal(b.dtype(), int8)


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

    var frozen = a.finish()
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))

    assert_equal(frozen.length, 4)


def test_append() raises:
    var a = Int8Builder()
    assert_equal(len(a), 0)
    assert_equal(a._capacity, 0)
    a.append(1)
    a.append(2)
    a.append(3)
    assert_equal(len(a), 3)
    assert_true(a._capacity >= len(a))


def test_array_empty() raises:
    var b = Int32Builder()
    var a = b.finish()
    assert_equal(len(a), 0)


def test_array_from_ints() raises:
    var g = array([1, 2], int8)
    assert_equal(len(g), 2)
    assert_equal(g[0].value(), 1)
    assert_equal(g[1].value(), 2)

    var b = array([True, False, True])
    assert_equal(len(b), 3)
    assert_true(b[0].value())
    assert_false(b[1].value())
    assert_true(b[2].value())


def test_array_with_nulls() raises:
    var a = array([1, None, 3], int32)
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 1)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_equal(a[0].value(), 1)
    assert_equal(a[2].value(), 3)

    var b = array([True, None, False])
    assert_equal(b.length, 3)
    assert_true(b.is_valid(0))
    assert_false(b.is_valid(1))
    assert_true(b.is_valid(2))


def test_arange() raises:
    var a = arange[Int32Type](1, 5)
    assert_equal(len(a), 4)
    assert_equal(a[0].value(), 1)
    assert_equal(a[1].value(), 2)
    assert_equal(a[2].value(), 3)
    assert_equal(a[3].value(), 4)

    var b = arange[UInt8Type](0, 3)
    assert_equal(len(b), 3)
    assert_equal(b[0].value(), 0)
    assert_equal(b[2].value(), 2)


def test_arange_empty() raises:
    var a = arange[Int32Type](5, 5)
    assert_equal(len(a), 0)


def test_arange_single() raises:
    var a = arange[Int64Type](7, 8)
    assert_equal(len(a), 1)
    assert_equal(a[0].value(), 7)


def test_arange_validity() raises:
    var a = arange[Int16Type](0, 4)
    for i in range(4):
        assert_true(a.is_valid(i))


def test_arange_int8() raises:
    var a = arange[Int8Type](10, 15)
    assert_equal(len(a), 5)
    assert_equal(a[0].value(), 10)
    assert_equal(a[4].value(), 14)


def test_arange_uint64() raises:
    var a = arange[UInt64Type](100, 103)
    assert_equal(len(a), 3)
    assert_equal(a[0].value(), 100)
    assert_equal(a[2].value(), 102)


def test_primitive_array_with_offset() raises:
    """Test PrimitiveArray with offset functionality."""
    var b = Int32Builder(10)
    b.append(100)
    b.append(200)
    b.append(300)
    b.append(400)
    b.append(500)
    var arr = b.finish()

    # Default offset should be 0
    assert_equal(arr.offset, 0)
    assert_equal(arr[0].value(), 100)
    assert_equal(arr[1].value(), 200)

    # Create a zero-copy slice, should point to the same buffers.
    var sliced = arr.slice(2)
    assert_equal(sliced.offset, 2)

    # Test that offset affects get operations
    assert_equal(sliced[0].value(), 300)  # Should get arr[2]
    assert_equal(sliced[1].value(), 400)  # Should get arr[3]
    assert_equal(sliced[2].value(), 500)  # Should get arr[4]


def test_primitive_array_nulls_with_offset() raises:
    """Test nulls() creates an array with all null values and default offset."""
    var null_arr = nulls(5, int64)
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

    var frozen = a.finish()
    assert_equal(frozen[0], "hello")
    assert_equal(frozen[1], "world")


def test_string_builder_amortized() raises:
    # Append many strings without pre-allocated bytes capacity.
    # Exercises amortized reserve_bytes growth (previously O(N²)).
    var a = StringBuilder()
    for i in range(100):
        a.append(String(i))
    var frozen = a.finish()
    assert_equal(len(frozen), 100)
    for i in range(100):
        assert_equal(frozen[i], String(i))


def test_list_bool_array() raises:
    var bool_b = BoolBuilder(3)
    bool_b.append(True)
    bool_b.append_null()
    bool_b.append(True)
    var list_b = ListBuilder(bool_b^)
    list_b.append_valid()
    var lists = list_b.finish()
    assert_equal(len(lists), 1)

    # TODO: fix listarray.unsafe_get
    var first_value = lists[0].value()
    ref bool_array = first_value.as_bool()
    assert_true(bool_array[0].value())
    assert_false(bool_array[1].value())
    assert_true(bool_array[2].value())


def test_list_str() raises:
    var str_b = StringBuilder()
    str_b.append("hello")
    str_b.append("world")
    var list_b = ListBuilder(str_b^)
    list_b.append_valid()
    var lists = list_b.finish()
    assert_equal(len(lists), 1)

    var first_val = lists[0].value()
    ref first_value = first_val.as_string()
    assert_equal(first_value[0], "hello")
    assert_equal(first_value[1], "world")


def test_arrays_list_of_list() raises:
    var top_b = ListBuilder(
        ListBuilder(Int64Builder(capacity=10), capacity=6),
        capacity=3,
    )
    var middle_any = top_b.values()
    ref middle = middle_any.as_list()
    var child_any = middle.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    middle.append_valid()
    child.append(3)
    child.append(4)
    middle.append_valid()
    top_b.append_valid()
    child.append(5)
    child.append(6)
    child.append(7)
    middle.append_valid()
    middle.append_null()
    child.append(8)
    middle.append_valid()
    top_b.append_valid()
    child.append(9)
    child.append(10)
    middle.append_valid()
    top_b.append_valid()
    var list2 = top_b.finish()

    var top_val = list2[0].value()
    ref top = top_val.as_list()
    var middle_0 = top[0].value()
    ref bottom_0 = middle_0.as_int64()
    assert_equal(bottom_0[1].value(), 2)
    assert_equal(bottom_0[0].value(), 1)
    var middle_1 = top[1].value()
    ref bottom_1 = middle_1.as_int64()
    assert_equal(bottom_1[0].value(), 3)
    assert_equal(bottom_1[1].value(), 4)


def test_fixed_size_list_int_array() raises:
    """Construct a FixedSizeListArray of int64 lists, size=3."""
    var ints_b = Int64Builder(6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b^, list_size=3)
    builder.append_valid()
    builder.append_valid()
    assert_equal(builder.dtype(), fixed_size_list_(int64, 3))
    assert_equal(len(builder), 2)
    var fsl = builder.finish()
    assert_equal(len(fsl), 2)
    assert_equal(fsl.dtype.as_fixed_size_list().size, 3)

    # First list: [1, 2, 3]
    ref first = fsl[0].value().as_int64()
    assert_equal(len(first), 3)
    assert_equal(first[0].value(), 1)
    assert_equal(first[1].value(), 2)
    assert_equal(first[2].value(), 3)

    # Second list: [4, 5, 6]
    ref second = fsl[1].value().as_int64()
    assert_equal(second[0].value(), 4)
    assert_equal(second[1].value(), 5)
    assert_equal(second[2].value(), 6)


def test_fixed_size_list_roundtrip() raises:
    """FixedSizeListArray round-trip through builder."""
    var ints_b = Int32Builder(4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(ints_b^, list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish()

    assert_true(fsl.dtype.is_fixed_size_list())
    assert_equal(fsl.dtype.as_fixed_size_list().size, 2)
    assert_equal(len(fsl), 2)

    ref first = fsl[0].value().as_int32()
    assert_equal(first[0].value(), 10)
    assert_equal(first[1].value(), 20)


def test_fixed_size_list_with_nulls() raises:
    """FixedSizeListArray with null lists."""
    var ints_b = Int64Builder(6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b^, list_size=3, capacity=3)
    builder.append_valid()
    builder.append_valid()
    builder.append_null()
    assert_equal(len(builder), 3)

    var fsl = builder.finish()
    assert_true(fsl.is_valid(0))
    assert_true(fsl.is_valid(1))
    assert_false(fsl.is_valid(2))

    # unsafe_get on valid entries returns correct values even when array has nulls
    ref first = fsl[0].value().as_int64()
    assert_equal(first[0].value(), 1)
    assert_equal(first[1].value(), 2)
    assert_equal(first[2].value(), 3)
    ref second = fsl[1].value().as_int64()
    assert_equal(second[0].value(), 4)
    assert_equal(second[1].value(), 5)
    assert_equal(second[2].value(), 6)


def test_fixed_size_list_unsafe_get_dtype() raises:
    # unsafe_get returns a slice with the child element dtype, not the list dtype.
    var ints_b = Int32Builder(4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(ints_b^, list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish()

    var slice0 = fsl[0].value()
    assert_equal(slice0.dtype(), int32)
    assert_equal(slice0.length(), 2)
    assert_equal(slice0.to_data().offset, 0)

    var slice1 = fsl[1].value()
    assert_equal(slice1.dtype(), int32)
    assert_equal(slice1.length(), 2)
    assert_equal(slice1.to_data().offset, 2)


# # def test_fixed_size_list_pretty_print():
# #     """Pretty printing FixedSizeListArray."""
# #     var ints_b = Int64Builder(4)
# #     ints_b.append(1)
# #     ints_b.append(2)
# #     ints_b.append(3)
# #     ints_b.append(4)
# #     var builder = FixedSizeListBuilder(ints_b, list_size=2)
# #     builder.append_valid()
# #     builder.append_valid()
# #     var fsl = builder.finish()
# #     var s = String(DynArray(fsl^))
# #     assert_true("FixedSizeListArray" in s)


def test_struct_array() raises:
    var struct_builder = StructBuilder(
        [field("id", int64), field("name", string), field("active", bool_)],
        capacity=10,
    )
    assert_equal(len(struct_builder), 0)
    assert_equal(struct_builder._capacity, 10)

    var data: DynArray = struct_builder.finish()
    assert_equal(data.length(), 0)
    assert_true(data.dtype().is_struct())
    assert_equal(len(data.dtype().as_struct().fields), 3)
    assert_equal(data.dtype().as_struct().fields[0].name, "id")
    assert_equal(data.dtype().as_struct().fields[1].name, "name")
    assert_equal(data.dtype().as_struct().fields[2].name, "active")


def test_struct_array_unsafe_get() raises:
    var sb = StructBuilder(
        [field("int_data_a", int32), field("int_data_b", int32)], capacity=2
    )
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(0).as_int32().append(3)
    sb.field_builder(0).as_int32().append(4)
    sb.field_builder(0).as_int32().append(5)
    sb.field_builder(1).as_int32().append(10)
    sb.field_builder(1).as_int32().append(20)
    sb.field_builder(1).as_int32().append(30)
    sb.append_valid()
    sb.append_valid()
    var struct_array = sb.finish()
    ref int_data_a = struct_array.unsafe_get("int_data_a")
    ref int_a = int_data_a.as_int32()
    assert_equal(int_a[0].value(), 1)
    assert_equal(int_a[4].value(), 5)
    ref int_data_b = struct_array.unsafe_get("int_data_b")
    ref int_b = int_data_b.as_int32()
    assert_equal(int_b[0].value(), 10)
    assert_equal(int_b[2].value(), 30)


def test_chunked_array() raises:
    var arrays: List[DynArray] = [
        array([0], uint8),
        array([0, 1], uint8),
    ]

    var chunked_array = ChunkedArray(int8, arrays^)
    assert_equal(chunked_array.length, 3)

    assert_equal(chunked_array.chunk(0).length(), 1)
    var second_chunk_any = chunked_array.chunk(1).copy()
    ref second_chunk = second_chunk_any.as_uint8()
    assert_equal(second_chunk.length, 2)
    assert_equal(second_chunk[0].value(), 0)
    assert_equal(second_chunk[1].value(), 1)


def test_combine_chunked_array() raises:
    var arrays: List[DynArray] = [
        array([0], uint8),
        array([0, 1], uint8),
    ]

    var chunked_array = ChunkedArray(uint8, arrays^)
    assert_equal(chunked_array.length, 3)
    assert_equal(len(chunked_array.chunks), 2)
    assert_equal(chunked_array.chunk(1).copy().as_uint8()[1].value(), 1)

    var combined_array = chunked_array^.combine_chunks()
    assert_equal(combined_array.length(), 3)
    assert_equal(combined_array.dtype(), uint8)
    # Single concatenated values buffer: [0, 0, 1]
    assert_equal(combined_array.to_data().buffers[0].unsafe_get(0), 0)
    assert_equal(combined_array.to_data().buffers[0].unsafe_get(2), 1)


def test_primitive_finish_shrinks() raises:
    """Freeze() on an over-allocated builder trims capacity to length."""
    var a = Int64Builder(capacity=100)
    a.append(42)
    a.append(99)
    var frozen = a.finish()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0].value(), 42)
    assert_equal(frozen[1].value(), 99)

    var values_buffer = frozen.buffer
    # 2 int64 values = 16 bytes, but buffer padded to 64 bytes for alignment
    assert_equal(len(values_buffer), 64)


def test_primitive_finish_via_append() raises:
    """Freeze() works on a builder built with append() (auto-grow capacity)."""
    var a = Int64Builder()
    a.append(1)
    a.append(2)
    a.append(3)
    var frozen = a.finish()
    assert_equal(frozen.length, 3)
    assert_equal(frozen[0].value(), 1)
    assert_equal(frozen[2].value(), 3)


def test_primitive_finish_preserves_nulls() raises:
    """Freeze() preserves null validity information."""
    var a = Int64Builder(capacity=3)
    a.append(1)
    a.append_null()
    a.append(3)
    var frozen = a.finish()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_primitive_finish_converts_to_array() raises:
    """PrimitiveBuilder.finish() returns a typed PrimitiveArray."""
    var a = Int64Builder()
    a.append(7)
    a.append(8)
    var frozen = a.finish()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0].value(), 7)
    assert_equal(frozen[1].value(), 8)


def test_getitem_bounds_check() raises:
    """__getitem__ raises on out-of-bounds access."""
    var b = Int64Builder()
    b.append(1)
    b.append(2)
    var a = b.finish()
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
    assert_equal(a[0].value(), 1)
    assert_equal(a[1].value(), 2)


def test_setitem_bounds_check() raises:
    """PrimitiveArray __getitem__ returns correct values."""
    var a = Int64Builder()
    a.append(99)
    var frozen = a.finish()
    assert_equal(frozen[0].value(), 99)


def test_string_finish_zero_copy() raises:
    """Freeze() on an exact-size StringBuilder moves buffers."""
    var s = StringBuilder(capacity=2)
    s.append("hello")
    s.append("world")
    var frozen = s.finish()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0], "hello")
    assert_equal(frozen[1], "world")


def test_string_finish_shrinks() raises:
    """Freeze() on an over-allocated StringBuilder trims to exact size."""
    var s = StringBuilder(capacity=100)
    s.append("hi")
    var frozen = s.finish()
    assert_equal(frozen.length, 1)
    assert_equal(frozen[0], "hi")


def test_string_getitem_bounds_check() raises:
    """StringArray __getitem__ raises on out-of-bounds."""
    var s = StringBuilder()
    s.append("a")
    var frozen = s.finish()
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
    var a = array([1, 2, 3], int32)
    var s = String(a)
    assert_true("PrimitiveArray" in s)
    assert_true("1" in s)
    assert_true("2" in s)
    assert_true("3" in s)


def test_str_primitive_array_with_nulls() raises:
    var a = array([1, None, 3], int32)
    var s = String(a)
    assert_true("NULL" in s)
    assert_true("1" in s)
    assert_true("3" in s)


def test_str_bool_array() raises:
    var a = array([True, False, True])
    var s = String(a)
    assert_true("BoolArray" in s)


def test_str_string_array() raises:
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("world")
    var a = sb.finish()
    var s = String(a)
    assert_true("StringArray" in s)
    assert_true("hello" in s)
    assert_true("world" in s)


def test_str_string_array_with_nulls() raises:
    var sb = StringBuilder(3)
    sb.append("foo")
    sb.append_null()
    sb.append("bar")
    var a = sb.finish()
    var s = String(a)
    assert_true("NULL" in s)
    assert_true("foo" in s)
    assert_true("bar" in s)


def test_str_list_array() raises:
    var ints_b = Int64Builder()
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    var list_b = ListBuilder(ints_b^)
    list_b.append_valid()
    var lists = list_b.finish()
    var s = String(lists)
    assert_true("ListArray" in s)


def test_str_fixed_size_list_array() raises:
    var ints_b = Int64Builder(4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(ints_b^, list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish()
    var s = String(fsl)
    assert_true("FixedSizeListArray" in s)


def test_str_struct_array() raises:
    var sb = StructBuilder([field("x", int32)], capacity=2)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish()
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
    var a = sb.finish()
    assert_equal(a.null_count(), 2)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_false(a.is_valid(3))


def test_string_array_no_nulls() raises:
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("world")
    var a = sb.finish()
    assert_equal(a.null_count(), 0)
    assert_true(a.is_valid(0))
    assert_true(a.is_valid(1))


def test_list_array_is_valid() raises:
    var ints_b = Int64Builder()
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    var list_b = ListBuilder(ints_b^)
    list_b.append_valid()
    list_b.append_null()
    list_b.append_valid()
    var lists = list_b.finish()
    assert_equal(len(lists), 3)
    assert_equal(lists.null_count(), 1)
    assert_true(lists.is_valid(0))
    assert_false(lists.is_valid(1))
    assert_true(lists.is_valid(2))


def test_struct_array_is_valid() raises:
    var sb = StructBuilder([field("val", int32)], capacity=3)
    sb.field_builder(0).as_int32().append(10)
    sb.field_builder(0).as_int32().append(20)
    sb.field_builder(0).as_int32().append(30)
    sb.append_valid()
    sb.append_null()
    sb.append_valid()
    var sa = sb.finish()
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
    var a = sb.finish()
    assert_equal(a[0], "alpha")
    assert_equal(a[1], "beta")
    assert_equal(a[2], "gamma")


def test_string_array_getitem_bounds() raises:
    var sb = StringBuilder()
    sb.append("only")
    var a = sb.finish()
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
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(10)
    child.append(20)
    list_b.append_valid()  # [10, 20]
    child.append(30)
    child.append(40)
    child.append(50)
    list_b.append_valid()  # [30, 40, 50]
    var lists = list_b.finish()
    var first = lists[0].value()
    assert_equal(first.length(), 2)
    var second = lists[1].value()
    assert_equal(second.length(), 3)


def test_list_array_getitem_bounds() raises:
    var ints_b = Int64Builder()
    ints_b.append(1)
    var list_b = ListBuilder(ints_b^)
    list_b.append_valid()
    var lists = list_b.finish()
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
    var ints_b = Int32Builder(6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b^, list_size=3)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish()
    ref first = fsl[0].value().as_int32()
    assert_equal(first[0].value(), 1)
    assert_equal(first[1].value(), 2)
    assert_equal(first[2].value(), 3)
    ref second = fsl[1].value().as_int32()
    assert_equal(second[0].value(), 4)
    assert_equal(second[1].value(), 5)
    assert_equal(second[2].value(), 6)


def test_fixed_size_list_getitem_bounds() raises:
    var ints_b = Int32Builder(3)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    var builder = FixedSizeListBuilder(ints_b^, list_size=3)
    builder.append_valid()
    var fsl = builder.finish()
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
    var sb = StructBuilder(
        [field("id", int32), field("name", string)], capacity=2
    )
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(1).as_string().append("x")
    sb.field_builder(1).as_string().append("y")
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish()

    var field_0 = sa.field(0)
    ref id_arr = field_0.as_int32()
    assert_equal(id_arr[0].value(), 1)
    assert_equal(id_arr[1].value(), 2)

    var field_1 = sa.field(1)
    ref name_arr = field_1.as_string()
    assert_equal(name_arr[0], "x")
    assert_equal(name_arr[1], "y")


def test_struct_array_field_by_name() raises:
    var sb = StructBuilder([field("val", int32)], capacity=2)
    sb.field_builder(0).as_int32().append(10)
    sb.field_builder(0).as_int32().append(20)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish()

    var field_val = sa.field("val")
    ref val_arr = field_val.as_int32()
    assert_equal(val_arr[0].value(), 10)
    assert_equal(val_arr[1].value(), 20)


def test_struct_array_field_bounds() raises:
    var sb = StructBuilder([field("x", int32)], capacity=1)
    sb.field_builder(0).as_int32().append(1)
    sb.append_valid()
    var sa = sb.finish()
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
    var a = sb.finish()
    var sliced = a.slice(2)
    assert_equal(len(sliced), 2)
    assert_equal(sliced.offset, 2)
    assert_equal(sliced[0], "cc")
    assert_equal(sliced[1], "dd")


def test_fixed_size_list_len_and_null_count() raises:
    var ints_b = Int64Builder(6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b^, list_size=2)
    builder.append_valid()
    builder.append_null()
    builder.append_valid()
    var fsl = builder.finish()
    assert_equal(len(fsl), 3)
    assert_equal(fsl.null_count(), 1)


def test_list_array_null_count() raises:
    var ints_b = Int64Builder()
    ints_b.append(1)
    ints_b.append(2)
    var list_b = ListBuilder(ints_b^)
    list_b.append_valid()
    list_b.append_null()
    var lists = list_b.finish()
    assert_equal(lists.null_count(), 1)
    assert_equal(len(lists), 2)


def test_primitive_array_no_nulls_is_valid() raises:
    var a = array([10, 20, 30], int64)
    assert_equal(a.null_count(), 0)
    for i in range(3):
        assert_true(a.is_valid(i))


# ---------------------------------------------------------------------------
# slice() tests
# ---------------------------------------------------------------------------


def test_primitive_array_slice_with_length() raises:
    var a = array([10, 20, 30, 40, 50], int32)
    var s = a.slice(1, 3)
    assert_equal(len(s), 3)
    assert_equal(s[0].value(), 20)
    assert_equal(s[1].value(), 30)
    assert_equal(s[2].value(), 40)


def test_string_array_slice_with_length() raises:
    var sb = StringBuilder()
    sb.append("aa")
    sb.append("bb")
    sb.append("cc")
    sb.append("dd")
    var a = sb.finish()
    var s = a.slice(1, 2)
    assert_equal(len(s), 2)
    assert_equal(s[0], "bb")
    assert_equal(s[1], "cc")


def test_list_array_slice() raises:
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    list_b.append_valid()
    child.append(3)
    list_b.append_valid()
    child.append(4)
    child.append(5)
    list_b.append_valid()
    var lists = list_b.finish()
    var s = lists.slice(1)
    assert_equal(len(s), 2)


def test_fixed_size_list_slice() raises:
    var ints_b = Int32Builder(6)
    ints_b.append(1)
    ints_b.append(2)
    ints_b.append(3)
    ints_b.append(4)
    ints_b.append(5)
    ints_b.append(6)
    var builder = FixedSizeListBuilder(ints_b^, list_size=2)
    builder.append_valid()
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish()
    var s = fsl.slice(1, 2)
    assert_equal(len(s), 2)
    ref first = s[0].value().as_int32()
    assert_equal(first[0].value(), 3)
    assert_equal(first[1].value(), 4)


# ---------------------------------------------------------------------------
# flatten() and value_lengths() tests
# ---------------------------------------------------------------------------


def test_list_array_flatten() raises:
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    list_b.append_valid()
    child.append(3)
    list_b.append_valid()
    var lists = list_b.finish()
    var flat = lists.flatten()
    assert_equal(flat.length(), 3)


def test_list_array_value_lengths() raises:
    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_any = list_b.values()
    ref child = child_any.as_int64()
    child.append(1)
    child.append(2)
    list_b.append_valid()  # length 2
    child.append(3)
    list_b.append_valid()  # length 1
    child.append(4)
    child.append(5)
    child.append(6)
    list_b.append_valid()  # length 3
    var lists = list_b.finish()
    var lengths = lists.value_lengths()
    assert_equal(len(lengths), 3)
    assert_equal(lengths[0].value(), 2)
    assert_equal(lengths[1].value(), 1)
    assert_equal(lengths[2].value(), 3)


def test_fixed_size_list_flatten() raises:
    var ints_b = Int32Builder(4)
    ints_b.append(10)
    ints_b.append(20)
    ints_b.append(30)
    ints_b.append(40)
    var builder = FixedSizeListBuilder(ints_b^, list_size=2)
    builder.append_valid()
    builder.append_valid()
    var fsl = builder.finish()
    var flat = fsl.flatten()
    assert_equal(flat.length(), 4)


def test_struct_array_flatten() raises:
    var sb = StructBuilder(
        [field("id", int32), field("name", string)], capacity=2
    )
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(1).as_string().append("x")
    sb.field_builder(1).as_string().append("y")
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish()
    var flat = sa.flatten()
    assert_equal(len(flat), 2)
    assert_equal(flat[0].length(), 2)
    assert_equal(flat[1].length(), 2)


# ---------------------------------------------------------------------------
# StructArray.select tests
# ---------------------------------------------------------------------------


def test_struct_array_select_basic() raises:
    """`select` returns a StructArray with only the requested fields."""
    var sb = StructBuilder(
        [field("a", int32), field("b", int32), field("c", int32)], capacity=2
    )
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(1).as_int32().append(10)
    sb.field_builder(1).as_int32().append(20)
    sb.field_builder(2).as_int32().append(100)
    sb.field_builder(2).as_int32().append(200)
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish()

    var indices = List[Int]()
    indices.append(0)
    indices.append(2)
    var result = sa.select(indices)

    assert_equal(len(result.children), 2)
    assert_equal(result.dtype.as_struct().fields[0].name, "a")
    assert_equal(result.dtype.as_struct().fields[1].name, "c")
    assert_equal(len(result), 2)


def test_struct_array_select_inherits_nulls_and_bitmap() raises:
    """`select` preserves nulls count, bitmap, and offset from the source."""
    var sb = StructBuilder([field("x", int32), field("y", int32)], capacity=3)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(0).as_int32().append(3)
    sb.field_builder(1).as_int32().append(10)
    sb.field_builder(1).as_int32().append(20)
    sb.field_builder(1).as_int32().append(30)
    sb.append_null()
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish()
    assert_equal(sa.null_count(), 1)

    var indices = List[Int]()
    indices.append(0)
    var result = sa.select(indices)

    assert_equal(result.null_count(), 1)
    assert_equal(result.offset, sa.offset)
    assert_true(result.bitmap.__bool__())  # bitmap is present


def test_struct_array_select_inherits_offset() raises:
    """`select` preserves the offset of the source array."""
    var sb = StructBuilder([field("a", int32), field("b", int32)], capacity=3)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.field_builder(0).as_int32().append(3)
    sb.field_builder(1).as_int32().append(10)
    sb.field_builder(1).as_int32().append(20)
    sb.field_builder(1).as_int32().append(30)
    sb.append_valid()
    sb.append_valid()
    sb.append_valid()
    var sa = sb.finish().slice(1)  # offset = 1, length = 2

    var indices = List[Int]()
    indices.append(0)
    var result = sa.select(indices)

    assert_equal(result.offset, 1)
    assert_equal(len(result), 2)


# ---------------------------------------------------------------------------
# Equality tests
# ---------------------------------------------------------------------------


def test_primitive_array_eq() raises:
    # Fast path: no nulls, offset=0 — uses Buffer.__eq__
    var a = array([1, 2, 3], int32)
    var b = array([1, 2, 3], int32)
    assert_true(a == b)


def test_primitive_array_eq_unequal() raises:
    var a = array([1, 2, 3], int32)
    var b = array([1, 2, 4], int32)
    assert_false(a == b)


def test_primitive_array_eq_length_mismatch() raises:
    var a = array([1, 2, 3], int32)
    var b = array([1, 2], int32)
    assert_false(a == b)


def test_primitive_array_eq_sliced() raises:
    # Regression test: sliced arrays with non-zero offset must compare correctly.
    # Old _arrays_equal bug: compared raw buffer bytes ignoring offset.
    var a = array([10, 20, 30, 40, 50], int32)
    var b = array([10, 20, 30, 40, 50], int32)
    var sa = a.slice(1, 3)  # [20, 30, 40], offset=1
    var sb = b.slice(1, 3)  # [20, 30, 40], offset=1
    assert_true(sa == sb)


def test_primitive_array_eq_sliced_unequal() raises:
    var a = array([10, 20, 30, 40, 50], int32)
    var b = array([10, 20, 99, 40, 50], int32)
    var sa = a.slice(1, 3)  # [20, 30, 40]
    var sb = b.slice(1, 3)  # [20, 99, 40]
    assert_false(sa == sb)


def test_primitive_array_eq_nulls_equal() raises:
    var a = array([1, None, 3], int32)
    var b = array([1, None, 3], int32)
    assert_true(a == b)


def test_primitive_array_eq_nulls_mismatch_count() raises:
    var a = array([1, None, 3], int32)
    var b = array([1, 2, 3], int32)
    assert_false(a == b)


def test_primitive_array_eq_nulls_mismatch_pattern() raises:
    # Same null count but different null positions
    var a = array([None, 2, 3], int32)
    var b = array([1, None, 3], int32)
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
    assert_true(sa.finish() == sb.finish())


def test_string_array_eq_unequal() raises:
    var sa = StringBuilder()
    sa.append("hello")
    sa.append("world")
    var sb = StringBuilder()
    sb.append("hello")
    sb.append("mars")
    assert_false(sa.finish() == sb.finish())


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
    var a = sa.finish()
    var b = sb.finish()
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
    assert_true(sa.finish() == sb.finish())


def test_list_array_eq() raises:
    var ints_a = Int64Builder()
    var list_a = ListBuilder(ints_a^)
    var child_a_any = list_a.values()
    ref child_a = child_a_any.as_int64()
    child_a.append(1)
    child_a.append(2)
    list_a.append_valid()
    child_a.append(3)
    list_a.append_valid()
    var a = list_a.finish()

    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_b_any = list_b.values()
    ref child_b = child_b_any.as_int64()
    child_b.append(1)
    child_b.append(2)
    list_b.append_valid()
    child_b.append(3)
    list_b.append_valid()
    var b = list_b.finish()

    assert_true(a == b)


def test_list_array_eq_unequal() raises:
    var ints_a = Int64Builder()
    var list_a = ListBuilder(ints_a^)
    var child_a_any = list_a.values()
    ref child_a = child_a_any.as_int64()
    child_a.append(1)
    child_a.append(2)
    list_a.append_valid()
    var a = list_a.finish()

    var ints_b = Int64Builder()
    var list_b = ListBuilder(ints_b^)
    var child_b_any = list_b.values()
    ref child_b = child_b_any.as_int64()
    child_b.append(1)
    child_b.append(99)
    list_b.append_valid()
    var b = list_b.finish()

    assert_false(a == b)


def test_list_array_eq_nulls() raises:
    var ints_a = Int64Builder()
    ints_a.append(1)
    var list_a = ListBuilder(ints_a^)
    list_a.append_valid()
    list_a.append_null()
    var a = list_a.finish()

    var ints_b = Int64Builder()
    ints_b.append(1)
    var list_b = ListBuilder(ints_b^)
    list_b.append_valid()
    list_b.append_null()
    var b = list_b.finish()

    assert_true(a == b)


def test_fixed_size_list_array_eq() raises:
    var a_b = Int32Builder(4)
    a_b.append(1)
    a_b.append(2)
    a_b.append(3)
    a_b.append(4)
    var builder_a = FixedSizeListBuilder(a_b^, list_size=2)
    builder_a.append_valid()
    builder_a.append_valid()

    var b_b = Int32Builder(4)
    b_b.append(1)
    b_b.append(2)
    b_b.append(3)
    b_b.append(4)
    var builder_b = FixedSizeListBuilder(b_b^, list_size=2)
    builder_b.append_valid()
    builder_b.append_valid()

    assert_true(builder_a.finish() == builder_b.finish())


def test_fixed_size_list_array_eq_unequal() raises:
    var a_b = Int32Builder(4)
    a_b.append(1)
    a_b.append(2)
    a_b.append(3)
    a_b.append(4)
    var builder_a = FixedSizeListBuilder(a_b^, list_size=2)
    builder_a.append_valid()
    builder_a.append_valid()

    var b_b = Int32Builder(4)
    b_b.append(1)
    b_b.append(2)
    b_b.append(3)
    b_b.append(99)
    var builder_b = FixedSizeListBuilder(b_b^, list_size=2)
    builder_b.append_valid()
    builder_b.append_valid()

    assert_false(builder_a.finish() == builder_b.finish())


def test_struct_array_eq() raises:
    var sa = StructBuilder([field("x", int32)], capacity=2)
    sa.field_builder(0).as_int32().append(1)
    sa.field_builder(0).as_int32().append(2)
    sa.append_valid()
    sa.append_valid()

    var sb = StructBuilder([field("x", int32)], capacity=2)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(2)
    sb.append_valid()
    sb.append_valid()

    assert_true(sa.finish() == sb.finish())


def test_struct_array_eq_unequal() raises:
    var sa = StructBuilder([field("x", int32)], capacity=2)
    sa.field_builder(0).as_int32().append(1)
    sa.field_builder(0).as_int32().append(2)
    sa.append_valid()
    sa.append_valid()

    var sb = StructBuilder([field("x", int32)], capacity=2)
    sb.field_builder(0).as_int32().append(1)
    sb.field_builder(0).as_int32().append(99)
    sb.append_valid()
    sb.append_valid()

    assert_false(sa.finish() == sb.finish())


def test_struct_array_eq_dtype_mismatch() raises:
    var sa = StructBuilder([field("x", int32)], capacity=1)
    sa.field_builder(0).as_int32().append(1)
    sa.append_valid()

    var sb = StructBuilder(
        [field("y", int32)], capacity=1
    )  # different field name
    sb.field_builder(0).as_int32().append(1)
    sb.append_valid()

    assert_false(sa.finish() == sb.finish())


def test_array_eq_dtype_mismatch() raises:
    # Type-erased DynArray: int32 vs int64 → False
    var a: DynArray = array([1, 2, 3], int32)
    var b: DynArray = array([1, 2, 3], int64)
    assert_false(a == b)


def test_array_eq_via_dispatch() raises:
    # Equal arrays accessed as type-erased DynArray verify dispatch works.
    var a: DynArray = array([10, 20, 30], int32)
    var b: DynArray = array([10, 20, 30], int32)
    assert_true(a == b)
    assert_true(a == b)


def test_primitive_array_list_literal() raises:
    var arr = array([1, 2, 3, 4, 5], int64)
    assert_equal(len(arr), 5)
    assert_equal(arr[0].value(), 1)
    assert_equal(arr[4].value(), 5)
    assert_equal(arr.null_count(), 0)


def test_primitive_array_list_literal_float() raises:
    var b = Float64Builder(3)
    b.append(1.0)
    b.append(2.5)
    b.append(3.14)
    var arr = b.finish()
    assert_equal(len(arr), 3)
    assert_equal(arr[0].value(), 1.0)


def test_string_array_list_literal() raises:
    var arr: StringArray = ["hello", "world", "foo"]
    assert_equal(len(arr), 3)
    assert_equal(arr[0], "hello")
    assert_equal(arr[1], "world")
    assert_equal(arr[2], "foo")
    assert_equal(arr.null_count(), 0)


def test_primitive_array_list_literal_empty() raises:
    var arr: Int32Array = []
    assert_equal(len(arr), 0)


def test_temporal_array_date32() raises:
    var arr = Date32Array(
        ArrayData(
            dtype=date32().to_dyn(),
            length=3,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[Buffer.alloc_zeroed[DType.int32](3).to_immutable()],
            children=[],
        )
    )
    assert_equal(len(arr), 3)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.type() == date32().to_dyn())
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_true(arr.is_valid(2))


def test_temporal_array_timestamp_values() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int64](4)
    buf.unsafe_set[DType.int64](0, 1_000_000)
    buf.unsafe_set[DType.int64](1, 2_000_000)
    buf.unsafe_set[DType.int64](2, 3_000_000)
    buf.unsafe_set[DType.int64](3, 4_000_000)
    var arr = TimestampArray(
        ArrayData(
            dtype=timestamp(second, "UTC").to_dyn(),
            length=4,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    assert_equal(len(arr), 4)
    assert_true(arr.type() == timestamp(second, "UTC").to_dyn())
    assert_equal(arr[0].value(), 1_000_000)
    assert_equal(arr[1].value(), 2_000_000)
    assert_equal(arr[2].value(), 3_000_000)
    assert_equal(arr[3].value(), 4_000_000)


def test_temporal_array_with_nulls() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int32](3)
    buf.unsafe_set[DType.int32](0, 10)
    buf.unsafe_set[DType.int32](1, 20)
    buf.unsafe_set[DType.int32](2, 30)
    var bm = Bitmap.alloc_zeroed(3)
    bm.set(0)
    bm.set(2)
    var arr = Date32Array(
        ArrayData(
            dtype=date32().to_dyn(),
            length=3,
            nulls=1,
            offset=0,
            bitmap=bm.to_immutable(),
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    assert_equal(len(arr), 3)
    assert_equal(arr.null_count(), 1)
    assert_true(arr.is_valid(0))
    assert_false(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_equal(arr[0].value(), 10)
    assert_true(arr[1].is_null())
    assert_equal(arr[2].value(), 30)


def test_temporal_array_slice() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int64](5)
    for i in range(5):
        buf.unsafe_set[DType.int64](i, Int64(i * 1000))
    var arr = DurationArray(
        ArrayData(
            dtype=duration(millisecond).to_dyn(),
            length=5,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    var sliced = arr.slice(1, 3)
    assert_equal(len(sliced), 3)
    assert_equal(sliced[0].value(), 1000)
    assert_equal(sliced[1].value(), 2000)
    assert_equal(sliced[2].value(), 3000)


def test_temporal_array_equality() raises:
    var buf1 = Buffer[mut=True].alloc_zeroed[DType.int32](2)
    buf1.unsafe_set[DType.int32](0, 100)
    buf1.unsafe_set[DType.int32](1, 200)
    var buf2 = Buffer[mut=True].alloc_zeroed[DType.int32](2)
    buf2.unsafe_set[DType.int32](0, 100)
    buf2.unsafe_set[DType.int32](1, 200)
    var arr1 = Date32Array(
        ArrayData(
            dtype=date32().to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf1.to_immutable()],
            children=[],
        )
    )
    var arr2 = Date32Array(
        ArrayData(
            dtype=date32().to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf2.to_immutable()],
            children=[],
        )
    )
    assert_true(arr1 == arr2)


def test_temporal_array_dtype_mismatch() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int32](2)
    buf.unsafe_set[DType.int32](0, 1)
    buf.unsafe_set[DType.int32](1, 2)
    var arr_date32 = Date32Array(
        ArrayData(
            dtype=date32().to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    var buf64 = Buffer[mut=True].alloc_uninit[DType.int64](2)
    buf64.unsafe_set[DType.int64](0, 1)
    buf64.unsafe_set[DType.int64](1, 2)
    var arr_date64 = Date64Array(
        ArrayData(
            dtype=date64().to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf64.to_immutable()],
            children=[],
        )
    )
    assert_false(arr_date32^.to_dyn() == arr_date64^.to_dyn())


def test_temporal_array_to_any_roundtrip() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int64](2)
    buf.unsafe_set[DType.int64](0, 999)
    buf.unsafe_set[DType.int64](1, 1999)
    var arr = TimestampArray(
        ArrayData(
            dtype=timestamp(nanosecond).to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    var any_arr = arr^.to_dyn()
    assert_true(any_arr.dtype() == timestamp(nanosecond).to_dyn())
    assert_equal(any_arr.length(), 2)
    ref ta = any_arr.as_timestamp()
    assert_equal(ta[0].value(), 999)
    assert_equal(ta[1].value(), 1999)


def test_temporal_array_index_out_of_bounds() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int32](1)
    buf.unsafe_set[DType.int32](0, 42)
    var arr = Date32Array(
        ArrayData(
            dtype=date32().to_dyn(),
            length=1,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    var raised = False
    try:
        _ = arr[5]
    except:
        raised = True
    assert_true(raised)


def test_year_month_interval_array() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int32](3)
    buf.unsafe_set[DType.int32](0, Int32(12))
    buf.unsafe_set[DType.int32](1, Int32(24))
    buf.unsafe_set[DType.int32](2, Int32(36))
    var arr = YearMonthIntervalArray(
        ArrayData(
            dtype=year_month_interval().to_dyn(),
            length=3,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    assert_equal(len(arr), 3)
    assert_true(arr.type() == year_month_interval().to_dyn())
    assert_equal(arr[0].value(), 12)
    assert_equal(arr[1].value(), 24)
    assert_equal(arr[2].value(), 36)


def test_day_time_interval_array() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int64](3)
    buf.unsafe_set[DType.int64](0, Int64(86400000))
    buf.unsafe_set[DType.int64](1, Int64(172800000))
    buf.unsafe_set[DType.int64](2, Int64(259200000))
    var arr = DayTimeIntervalArray(
        ArrayData(
            dtype=day_time_interval().to_dyn(),
            length=3,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    assert_equal(len(arr), 3)
    assert_true(arr.type() == day_time_interval().to_dyn())
    assert_equal(arr[0].value(), 86400000)
    assert_equal(arr[1].value(), 172800000)
    assert_equal(arr[2].value(), 259200000)


def test_month_day_nano_interval_array() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int128](2)
    buf.unsafe_set[DType.int128](0, 1)
    buf.unsafe_set[DType.int128](1, 2)
    var arr = MonthDayNanoIntervalArray(
        ArrayData(
            dtype=month_day_nano_interval().to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    assert_equal(len(arr), 2)
    assert_true(arr.type() == month_day_nano_interval().to_dyn())
    assert_equal(arr[0].value(), 1)
    assert_equal(arr[1].value(), 2)


def test_interval_array_to_any_roundtrip() raises:
    var buf = Buffer[mut=True].alloc_uninit[DType.int32](2)
    buf.unsafe_set[DType.int32](0, Int32(6))
    buf.unsafe_set[DType.int32](1, Int32(18))
    var arr = YearMonthIntervalArray(
        ArrayData(
            dtype=year_month_interval().to_dyn(),
            length=2,
            nulls=0,
            offset=0,
            bitmap=None,
            buffers=[buf.to_immutable()],
            children=[],
        )
    )
    var any_arr = arr^.to_dyn()
    assert_true(any_arr.dtype() == year_month_interval().to_dyn())
    assert_equal(any_arr.length(), 2)
    ref ia = any_arr.as_year_month_interval()
    assert_equal(ia[0].value(), 6)
    assert_equal(ia[1].value(), 18)


def test_dictionary_array() raises:
    # Build values: ["cat", "dog", "fish"]
    var vb = StringBuilder()
    vb.append("cat")
    vb.append("dog")
    vb.append("fish")
    var values: DynArray = vb.finish()

    # Build indices: [0, 1, 2, 0, 1]
    var ib = Int8Builder()
    ib.append(0)
    ib.append(1)
    ib.append(2)
    ib.append(0)
    ib.append(1)
    var indices: DynArray = ib.finish()

    var arr = DictionaryArray.from_arrays(indices^, values^)
    assert_equal(len(arr), 5)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.type().is_dictionary())

    # Decoded values match
    assert_equal(arr[0].value().as_string().to_string(), "cat")
    assert_equal(arr[1].value().as_string().to_string(), "dog")
    assert_equal(arr[2].value().as_string().to_string(), "fish")
    assert_equal(arr[3].value().as_string().to_string(), "cat")
    assert_equal(arr[4].value().as_string().to_string(), "dog")

    # All entries valid
    assert_true(arr[0].is_valid())
    assert_true(arr[4].is_valid())

    # indices() and dictionary() accessors
    assert_equal(arr.indices().length(), 5)
    assert_equal(arr.dictionary().length(), 3)


def test_dictionary_array_null() raises:
    var vb = StringBuilder()
    vb.append("cat")
    vb.append("dog")
    var values: DynArray = vb.finish()

    # indices: [0, null, 1]
    var ib = Int32Builder()
    ib.append(0)
    ib.append_null()
    ib.append(1)
    var indices: DynArray = ib.finish()

    var arr = DictionaryArray.from_arrays(indices^, values^)
    assert_equal(len(arr), 3)
    assert_equal(arr.null_count(), 1)

    assert_true(arr[0].is_valid())
    assert_equal(arr[0].value().as_string().to_string(), "cat")
    assert_false(arr[1].is_valid())
    assert_true(arr[2].is_valid())
    assert_equal(arr[2].value().as_string().to_string(), "dog")


def test_dictionary_array_slice() raises:
    var vb = StringBuilder()
    vb.append("a")
    vb.append("b")
    vb.append("c")
    var values: DynArray = vb.finish()

    var ib = Int32Builder()
    for i in range(3):
        ib.append(Int32(i))
    var indices: DynArray = ib.finish()
    var arr = DictionaryArray.from_arrays(indices^, values^)

    # Slice [1:3] -> ["b", "c"]
    var sliced = arr.slice(1, 2)
    assert_equal(len(sliced), 2)
    assert_equal(sliced[0].value().as_string().to_string(), "b")
    assert_equal(sliced[1].value().as_string().to_string(), "c")

    # Slice [0:1] -> ["a"]
    var head = arr.slice(0, 1)
    assert_equal(len(head), 1)
    assert_equal(head[0].value().as_string().to_string(), "a")


def test_dictionary_indices_offset() raises:
    # A sliced dictionary's .indices() must apply the logical offset, not just
    # return the raw index buffer.
    var vb = StringBuilder()
    vb.append("a")
    vb.append("b")
    vb.append("c")
    var ib = Int32Builder()
    for i in range(3):
        ib.append(Int32(i))
    var indices: DynArray = ib.finish()
    var values: DynArray = vb.finish()
    var arr = DictionaryArray.from_arrays(indices^, values^)

    var full_any = arr.indices()
    ref full = full_any.as_int32()
    assert_equal(len(full), 3)
    assert_equal(full[0].value(), 0)

    var sliced_any = arr.slice(1, 2).indices()
    ref idx = sliced_any.as_int32()
    assert_equal(len(idx), 2)
    assert_equal(idx[0].value(), 1)
    assert_equal(idx[1].value(), 2)


def test_validity_accessor() raises:
    # .validity() exposes the null bitmap on nested arrays (list / fsl / struct /
    # fixed-size-binary share one implementation) and is None when fully valid.
    var ib = Int64Builder()
    ib.append(1)
    var lb = ListBuilder(ib^)
    lb.append_valid()
    lb.append_null()
    var lst = lb.finish()
    var v = lst.validity()
    assert_true(Bool(v))
    assert_true(v.value().test(0))
    assert_false(v.value().test(1))

    var ib2 = Int64Builder()
    ib2.append(1)
    var lb2 = ListBuilder(ib2^)
    lb2.append_valid()
    var lst2 = lb2.finish()
    assert_false(Bool(lst2.validity()))


def test_empty_factory() raises:
    # empty() builds a zero-length, null-free array of the given type.
    assert_equal(len(BoolArray.empty()), 0)
    assert_equal(len(StringArray.empty()), 0)
    var prim = PrimitiveArray[Int32Type].empty(Int32Type())
    assert_equal(len(prim), 0)
    assert_equal(prim.null_count(), 0)


def test_dictionary_array_data_roundtrip() raises:
    var vb = StringBuilder()
    vb.append("x")
    vb.append("y")
    var values: DynArray = vb.finish()

    var ib = Int32Builder()
    ib.append(0)
    ib.append(1)
    ib.append(0)
    var indices: DynArray = ib.finish()
    var arr = DictionaryArray.from_arrays(indices^, values^)

    # to_data round-trip
    var data = arr.to_data()
    assert_true(data.dtype.is_dictionary())
    assert_equal(data.length, 3)
    assert_equal(len(data.children), 1)

    # Reconstruct from ArrayData
    var arr2 = DictionaryArray(data)
    assert_equal(len(arr2), 3)
    assert_equal(arr2[0].value().as_string().to_string(), "x")
    assert_equal(arr2[1].value().as_string().to_string(), "y")
    assert_equal(arr2[2].value().as_string().to_string(), "x")

    # DynArray.from_data dispatch
    var any2 = DynArray.from_data(data)
    assert_true(any2.dtype().is_dictionary())
    ref da = any2.as_dictionary()
    assert_equal(da[0].value().as_string().to_string(), "x")


def test_dictionary_builder() raises:
    var vb = StringBuilder()
    vb.append("red")
    vb.append("green")
    vb.append("blue")
    var values: DynArray = vb.finish()

    var builder = DictionaryBuilder(Int8Builder(), values^)
    builder.append(0)  # "red"
    builder.append(1)  # "green"
    builder.append(2)  # "blue"
    builder.append_null()
    builder.append(0)  # "red"

    var arr = builder.finish()
    assert_equal(len(arr), 5)
    assert_equal(arr.null_count(), 1)
    assert_equal(arr[0].value().as_string().to_string(), "red")
    assert_equal(arr[1].value().as_string().to_string(), "green")
    assert_equal(arr[2].value().as_string().to_string(), "blue")
    assert_false(arr[3].is_valid())
    assert_equal(arr[4].value().as_string().to_string(), "red")


def test_dictionary_out_of_bounds() raises:
    var vb = StringBuilder()
    vb.append("only")
    var values: DynArray = vb.finish()

    var ib = Int32Builder()
    ib.append(0)
    var indices: DynArray = ib.finish()
    var arr = DictionaryArray.from_arrays(indices^, values^)

    var raised = False
    try:
        _ = arr[5]
    except:
        raised = True
    assert_true(raised)


def test_empty_null() raises:
    var arr = array(null.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_null())


def test_empty_bool() raises:
    var arr = array(bool_.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_bool())


def test_empty_int32() raises:
    var arr = array(int32.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype() == int32)


def test_empty_float64() raises:
    var arr = array(float64.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype() == float64)


def test_empty_string() raises:
    var arr = array(string.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_string())


def test_empty_large_string() raises:
    var arr = array(large_string.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_large_string())


def test_empty_binary() raises:
    var arr = array(binary.to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_binary())


def test_empty_list() raises:
    var arr = array(list_(int32).to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_list())
    assert_equal(len(arr.as_list().values()), 0)


def test_empty_large_list() raises:
    var arr = array(large_list_(float64).to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_large_list())


def test_empty_fixed_size_list() raises:
    var arr = array(fixed_size_list_(int32, 4).to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_fixed_size_list())
    assert_equal(len(arr.as_fixed_size_list().values()), 0)


def test_empty_struct() raises:
    var arr = array(struct_(Field("x", int32), Field("y", float64)).to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_struct())
    ref sa = arr.as_struct()
    assert_equal(len(sa.children), 2)
    assert_equal(len(sa.children[0]), 0)
    assert_equal(len(sa.children[1]), 0)


def test_empty_dictionary() raises:
    var arr = array(dictionary(int32, string).to_dyn())
    assert_equal(len(arr), 0)
    assert_equal(arr.null_count(), 0)
    assert_true(arr.dtype().is_dictionary())


def test_empty_nested_list() raises:
    var arr = array(list_(list_(int32)).to_dyn())
    assert_equal(len(arr), 0)
    assert_true(arr.dtype().is_list())
    assert_equal(len(arr.as_list().values()), 0)
    assert_true(arr.as_list().values().dtype().is_list())


def test_dictionary_builder_preserves_ordered() raises:
    """`finish()` must not drop the `ordered` flag the builder was given.

    It stored `ordered` in `_dtype` and then built the result with
    `DictionaryArray.from_arrays(indices, values)`, whose `ordered` defaults to
    False — so the flag survived construction and was lost on the way out.
    """
    var vb = StringBuilder()
    vb.append("red")
    vb.append("green")
    var values: DynArray = vb.finish()

    var builder = DictionaryBuilder(Int8Builder(), values^, ordered=True)
    builder.append(0)
    builder.append(1)

    var arr = builder.finish()
    assert_true(arr.type().as_dictionary().ordered)


def test_dictionary_builder_defaults_to_unordered() raises:
    var vb = StringBuilder()
    vb.append("red")
    var values: DynArray = vb.finish()
    var builder = DictionaryBuilder(Int8Builder(), values^)
    builder.append(0)
    assert_true(not builder.finish().type().as_dictionary().ordered)


# ---------------------------------------------------------------------------
# Sliced BoolArray — `values()` is already offset-applied.
#
# `values()` returns `buffer.view(self.offset, self.length)` and
# `BitmapView.test` adds its own `_offset`, so `values().test(self.offset + i)`
# applied the offset twice. Every read of a sliced bool array was off by
# `offset` bits. Recorded as blocked by the layout freeze, which was wrong:
# dropping the redundant addition changes no field.
# ---------------------------------------------------------------------------


def _bools(values: List[Bool]) raises -> BoolArray:
    var b = BoolBuilder(capacity=len(values))
    for v in values:
        b.append(v)
    return b.finish()


def test_sliced_bool_array_getitem() raises:
    var a = _bools([True, False, True, True, False])
    var s = a.slice(1, 3)  # [False, True, True]
    assert_true(not s[0].value())
    assert_true(s[1].value())
    assert_true(s[2].value())


def test_sliced_bool_array_getitem_crosses_byte_boundary() raises:
    # offset 6, so the window straddles the first byte
    var vals = List[Bool]()
    for i in range(12):
        vals.append(i % 3 == 0)  # T F F T F F T F F T F F
    var a = _bools(vals)
    var s = a.slice(6, 4)  # indices 6..9 -> T F F T
    assert_true(s[0].value())
    assert_true(not s[1].value())
    assert_true(not s[2].value())
    assert_true(s[3].value())


def test_sliced_bool_array_write_to() raises:
    var a = _bools([True, False, True, True, False])
    var s = a.slice(1, 3)
    assert_true(String(s) == "BoolArray([False, True, True])")


# ---------------------------------------------------------------------------
# Null count of a slice.
#
# `slice()` copied the parent's count verbatim, so a slice reported the wrong
# number of nulls. That is not only a reporting bug: `PrimitiveBuilder.extend`
# branches on it and accumulates it, so appending a slice corrupted the
# builder's own count on the plain CPU path, and `to_data()` carried it across
# the C ABI where PyArrow trusts it.
# ---------------------------------------------------------------------------


def _nullable_five() raises -> Int64Array:
    # [10, null, 30, null, 50] — two nulls
    var b = Int64Builder(capacity=5)
    b.append(Int64(10))
    b.append_null()
    b.append(Int64(30))
    b.append_null()
    b.append(Int64(50))
    return b.finish()


def test_slice_reports_its_own_null_count() raises:
    var full = _nullable_five()
    assert_equal(full.null_count(), 2)
    # [30, null, 50] — one null, where the parent has two
    assert_equal(full.slice(2, 3).null_count(), 1)
    # [10, null] — one null
    assert_equal(full.slice(0, 2).null_count(), 1)
    # [30] — none at all
    assert_equal(full.slice(2, 1).null_count(), 0)


def test_slice_of_null_free_array_has_no_nulls() raises:
    var full = array([1, 2, 3, 4, 5], int64)
    assert_equal(full.slice(1, 3).null_count(), 0)


def test_slice_null_count_survives_to_data() raises:
    """`to_data` must resolve the count — it crosses the C ABI."""
    var s = _nullable_five().slice(2, 3)
    assert_equal(s.to_data().nulls, 1)


def test_extending_a_builder_with_a_slice_keeps_the_count_right() raises:
    var s = _nullable_five().slice(2, 3)  # [30, null, 50]
    var b = Int64Builder(capacity=3)
    b.extend(s)
    var out = b.finish()
    assert_equal(len(out), 3)
    assert_equal(out.null_count(), 1)
    assert_true(out.is_valid(0))
    assert_true(not out.is_valid(1))
    assert_true(out.is_valid(2))


def test_equal_slices_compare_equal() raises:
    var a = _nullable_five().slice(2, 3)
    var b = _nullable_five().slice(2, 3)
    assert_true(a == b)


def test_slice_of_all_null_array_is_all_null() raises:
    var b = Int64Builder(capacity=4)
    for _ in range(4):
        b.append_null()
    var full = b.finish()
    assert_equal(full.null_count(), 4)
    assert_equal(full.slice(1, 2).null_count(), 2)


# ---------------------------------------------------------------------------
# B26 — equality is a question about values and null *positions*, not about
# how the validity happens to be stored.
#
# `__eq__` compared the bitmaps themselves: presence against presence, then
# whole bitmap against whole bitmap. So an all-valid array carrying a bitmap was
# unequal to one carrying none, and two slices whose logical validity matched
# were unequal whenever their offsets differed. Six array types shared the shape.
#
# This matters more than it looks: CLAUDE.md tells you to write
# `assert_true(result == expected)` rather than an element loop, and every kernel
# that intersects validity emits an array with a bitmap while `array([...])`
# emits one without — so the recommended assertion was unreliable for exactly
# the values a kernel test wants to check.
# ---------------------------------------------------------------------------


def test_eq_ignores_a_redundant_all_valid_bitmap() raises:
    """`[1, 2, 3]` with an all-valid bitmap equals `[1, 2, 3]` with none."""
    # A builder given no nulls produces no bitmap at all, so the bitmap has to
    # come from a parent: slice past every null and the child keeps the parent's
    # bitmap while its own null count is 0.
    var b = Int32Builder(5)
    b.append_null()
    b.append_null()
    b.append(Scalar[int32.native](1))
    b.append(Scalar[int32.native](2))
    b.append(Scalar[int32.native](3))
    var with_bitmap = b.finish().slice(2, 3)
    assert_true(with_bitmap.bitmap.__bool__())

    var plain = array([1, 2, 3], int32)
    assert_false(plain.bitmap.__bool__())
    assert_equal(with_bitmap.null_count(), 0)
    assert_equal(plain.null_count(), 0)
    assert_true(with_bitmap == plain)
    assert_true(plain == with_bitmap)


def test_eq_compares_null_positions_not_bitmap_offsets() raises:
    """Two slices with the same logical validity are equal, whatever offset
    they were taken at."""
    var b1 = Int32Builder(5)
    b1.append_null()
    b1.append_null()
    b1.append(Scalar[int32.native](7))
    b1.append_null()
    b1.append(Scalar[int32.native](9))
    var left = b1.finish().slice(2, 3)  # [7, null, 9]

    var b2 = Int32Builder(3)
    b2.append(Scalar[int32.native](7))
    b2.append_null()
    b2.append(Scalar[int32.native](9))
    var right = b2.finish()  # [7, null, 9] at offset 0

    assert_equal(left.null_count(), right.null_count())
    assert_true(left == right)


def test_eq_still_separates_different_null_positions() raises:
    """The complement: same null *count*, different positions, still unequal."""
    var b1 = Int32Builder(3)
    b1.append_null()
    b1.append(Scalar[int32.native](1))
    b1.append(Scalar[int32.native](2))

    var b2 = Int32Builder(3)
    b2.append(Scalar[int32.native](1))
    b2.append_null()
    b2.append(Scalar[int32.native](2))

    assert_false(b1.finish() == b2.finish())


# ---------------------------------------------------------------------------
# S5 — validity equality, and dictionary slice/equality semantics
# ---------------------------------------------------------------------------


def test_validity_equal_all_valid_bitmap_vs_none() raises:
    """B26: a missing bitmap means all-valid, which is a value, not a
    representation. The slice excludes the only null but still carries the
    parent's bitmap at an offset; the plain array has no bitmap at all."""
    var sliced = array([None, 2, 3], int32).slice(1, 2)
    var plain = array([2, 3], int32)
    assert_equal(sliced.null_count(), 0)
    assert_true(sliced == plain)
    assert_true(plain == sliced)


def test_validity_equal_slices_at_different_offsets() raises:
    """Same logical validity reached from different offsets must compare equal —
    the views are offset-applied, so the bit patterns line up."""
    var a = array([1, None, 3, 4], int32).slice(1, 2)
    var b = array([9, 9, None, 3], int32).slice(2, 2)
    assert_true(a == b)


def test_validity_equal_same_count_different_positions_word_path() raises:
    """Equal null counts must still compare bit patterns. 100 elements puts a
    full 64-bit word plus a tail through `BitmapView.__eq__`, which is the
    word-level comparison that replaced the bit-by-bit loop."""
    var ab = Int32Builder()
    var bb = Int32Builder()
    for i in range(100):
        if i == 5:
            ab.append_null()
        else:
            ab.append(Int32(i))
        if i == 7:
            bb.append_null()
        else:
            bb.append(Int32(i))
    var a = ab.finish()
    var b = bb.finish()
    assert_equal(a.null_count(), b.null_count())
    assert_false(a == b)


def test_bool_array_eq_nulls_equal() raises:
    var a = array([True, None, False])
    var b = array([True, None, False])
    assert_true(a == b)


def test_bool_array_eq_nulls_mismatch_pattern() raises:
    var a = array([None, True, False])
    var b = array([True, None, False])
    assert_false(a == b)


def _dict_abc(var indices: DynArray) raises -> DictionaryArray:
    """A dictionary over ["a", "b", "c"] with the given indices."""
    var vb = StringBuilder()
    vb.append("a")
    vb.append("b")
    vb.append("c")
    var values: DynArray = vb.finish()
    return DictionaryArray.from_arrays(indices^, values^)


def test_dictionary_array_slice_recounts_nulls() raises:
    """A slice must report its own sub-range's null count, not the parent's."""
    var arr = _dict_abc(array([0, None, 1, None, 2], int8))
    assert_equal(arr.null_count(), 2)
    assert_equal(arr.slice(0, 2).null_count(), 1)
    assert_equal(arr.slice(2, 3).null_count(), 1)
    assert_equal(arr.slice(0, 1).null_count(), 0)
    assert_equal(arr.slice(1, 1).null_count(), 1)
    assert_equal(arr.slice(0, 5).null_count(), 2)


def test_dictionary_array_eq_differing_offset() raises:
    """Two different slices of one parent with equal length must not compare
    equal. The old `__eq__` compared `_indices`/`_values` whole and never looked
    at `_offset`, so this returned True."""
    var arr = _dict_abc(array([0, 1, 2], int8))
    assert_false(arr.slice(0, 1) == arr.slice(1, 1))
    assert_true(arr.slice(1, 2) == arr.slice(1, 2))


def test_dictionary_array_eq_permuted_dictionary() raises:
    """Logical equality (arrow-rs), not representation (Arrow C++): the same
    column encoded against differently ordered dictionaries is equal."""
    var vb1 = StringBuilder()
    vb1.append("x")
    vb1.append("y")
    var d1 = DictionaryArray.from_arrays(array([0, 1, 0], int8), vb1.finish())
    var vb2 = StringBuilder()
    vb2.append("y")
    vb2.append("x")
    var d2 = DictionaryArray.from_arrays(array([1, 0, 1], int8), vb2.finish())
    assert_true(d1 == d2)


def test_dictionary_array_eq_differing_values() raises:
    var arr = _dict_abc(array([0, 1], int8))
    var other = _dict_abc(array([0, 2], int8))
    assert_false(arr == other)


def test_dictionary_scalar_eq_permuted_dictionary() raises:
    """The scalar makes the same call as the array: `_index` is where the value
    was stored, not the value."""
    var vb1 = StringBuilder()
    vb1.append("x")
    vb1.append("y")
    var d1 = DictionaryArray.from_arrays(array([1], int8), vb1.finish())
    var vb2 = StringBuilder()
    vb2.append("y")
    vb2.append("x")
    var d2 = DictionaryArray.from_arrays(array([0], int8), vb2.finish())
    assert_true(d1[0] == d2[0])
    assert_equal(d1[0].index(), 1)
    assert_equal(d2[0].index(), 0)


def test_slice_default_length_null_count() raises:
    """`slice(offset)` must count nulls over the slice's own range, not to the
    end of the parent's bitmap. Every `slice` passed the raw `length` — `-1` on
    this call — where `Bitmap.view` reads -1 as "to the end of the bitmap", so a
    slice of a slice counted bits it does not own."""
    var a = array([1, None, 3, 4, None], int32)
    var b = a.slice(1, 3)  # [None, 3, 4]
    assert_equal(b.null_count(), 1)
    var c = b.slice(1)  # [3, 4] — bit 4 of the parent is null and not ours
    assert_equal(len(c), 2)
    assert_equal(c.null_count(), 0)
