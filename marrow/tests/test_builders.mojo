"""Tests for array builders (BoolBuilder, PrimitiveBuilder, StringBuilder,
ListBuilder, FixedSizeListBuilder, StructBuilder) and factory functions."""

from std.testing import assert_equal, assert_true, assert_false, TestSuite
from marrow.arrays import (
    AnyArray,
    BoolArray,
    PrimitiveArray,
    StringArray,
    ListArray,
    FixedSizeListArray,
    StructArray,
)
from marrow.builders import (
    AnyBuilder,
    BoolBuilder,
    PrimitiveBuilder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
    array,
    nulls,
    arange,
)
from marrow.dtypes import *


# ---------------------------------------------------------------------------
# BoolBuilder
# ---------------------------------------------------------------------------


def test_bool_builder_zero_length() raises:
    var b = BoolBuilder()
    assert_equal(len(b), 0)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 0)


def test_bool_builder_append_null() raises:
    var b = BoolBuilder()
    b.append(True)
    b.append_null()
    b.append(False)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_true(frozen[0])
    assert_false(frozen[2])


def test_bool_builder_null_count() raises:
    var b = BoolBuilder(4)
    b.append(True)
    b.append_null()
    b.append_null()
    b.append(False)
    var frozen = b.finish_typed()
    assert_equal(frozen.null_count(), 2)


def test_bool_builder_all_nulls() raises:
    var b = BoolBuilder(3)
    b.append_null()
    b.append_null()
    b.append_null()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.null_count(), 3)
    for i in range(3):
        assert_false(frozen.is_valid(i))


def test_bool_builder_all_false() raises:
    var b = BoolBuilder(3)
    b.append(False)
    b.append(False)
    b.append(False)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_equal(frozen.null_count(), 0)
    for i in range(3):
        assert_true(frozen.is_valid(i))
        assert_false(frozen[i])


def test_bool_builder_as_any_builder() raises:
    var b = BoolBuilder(2)
    b.append(True)
    b.append(False)
    var builder: AnyBuilder = b^
    assert_equal(builder.length(), 2)


# ---------------------------------------------------------------------------
# PrimitiveBuilder — type coverage
# ---------------------------------------------------------------------------


def test_primitive_builder_int16() raises:
    var b = PrimitiveBuilder[int16](2)
    b.append(32767)
    b.append(-32768)
    var frozen = b.finish_typed()
    assert_equal(frozen[0], 32767)
    assert_equal(frozen[1], -32768)


def test_primitive_builder_uint32() raises:
    var b = PrimitiveBuilder[uint32](2)
    b.append(0)
    b.append(42)
    var frozen = b.finish_typed()
    assert_equal(frozen[0], 0)
    assert_equal(frozen[1], 42)


def test_primitive_builder_float32() raises:
    var b = PrimitiveBuilder[float32](3)
    b.append(1.5)
    b.append(0.0)
    b.append(-3.14)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_true(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_primitive_builder_float64() raises:
    var b = PrimitiveBuilder[float64](2)
    b.append(1.0)
    b.append_null()
    var frozen = b.finish_typed()
    assert_equal(frozen.null_count(), 1)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))


def test_primitive_builder_capacity_doubling() raises:
    """Builder doubles capacity starting from zero capacity."""
    var b = PrimitiveBuilder[int32]()
    b.append(0)
    b.append(1)
    b.append(2)
    b.append(3)
    b.append(4)
    b.append(5)
    b.append(6)
    b.append(7)
    b.append(8)
    b.append(9)
    assert_equal(len(b), 10)
    var frozen = b.finish_typed()
    assert_equal(frozen[0], 0)
    assert_equal(frozen[4], 4)
    assert_equal(frozen[9], 9)


def test_primitive_builder_as_any_builder() raises:
    var b = PrimitiveBuilder[int64](3)
    b.append(1)
    b.append(2)
    b.append(3)
    var builder: AnyBuilder = b^
    assert_equal(builder.length(), 3)
    assert_equal(builder.dtype(), int64)


def test_primitive_builder_null_count() raises:
    var b = PrimitiveBuilder[int32](5)
    b.append(1)
    b.append_null()
    b.append_null()
    b.append(4)
    b.append_null()
    var frozen = b.finish_typed()
    assert_equal(frozen.null_count(), 3)


def test_primitive_builder_all_nulls() raises:
    var b = PrimitiveBuilder[int64](4)
    for _ in range(4):
        b.append_null()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 4)
    assert_equal(frozen.null_count(), 4)
    for i in range(4):
        assert_false(frozen.is_valid(i))


# ---------------------------------------------------------------------------
# StringBuilder
# ---------------------------------------------------------------------------


def test_string_builder_append_null() raises:
    var b = StringBuilder(3)
    b.append("hello")
    b.append_null()
    b.append("world")
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))
    assert_equal(frozen[0], "hello")
    assert_equal(frozen[2], "world")


def test_string_builder_null_count() raises:
    var b = StringBuilder()
    b.append("a")
    b.append_null()
    b.append_null()
    b.append("b")
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 4)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))


def test_string_builder_empty_string() raises:
    var b = StringBuilder(3)
    b.append("")
    b.append("x")
    b.append("")
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_equal(frozen[0], "")
    assert_equal(frozen[1], "x")
    assert_equal(frozen[2], "")


def test_string_builder_offsets_correct() raises:
    """Null entry must not advance the offset; valid entries advance by len."""
    var b = StringBuilder(4)
    b.append("ab")  # bytes [0..2)
    b.append("cde")  # bytes [2..5)
    b.append_null()  # bytes [5..5) — offset stays at 5
    b.append("f")  # bytes [5..6)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 4)
    assert_equal(frozen[0], "ab")
    assert_equal(frozen[1], "cde")
    assert_equal(frozen[3], "f")


def test_string_builder_all_nulls() raises:
    var b = StringBuilder(2)
    b.append_null()
    b.append_null()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    assert_false(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))


def test_string_builder_capacity_growth() raises:
    """Builder grows when appending beyond initial capacity."""
    var b = StringBuilder(1)
    b.append("first")
    b.append("second")
    b.append("third")
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_equal(frozen[0], "first")
    assert_equal(frozen[1], "second")
    assert_equal(frozen[2], "third")


def test_string_builder_append_string_slice() raises:
    """Append(StringSlice) stores the string correctly."""
    var b = StringBuilder(2)
    var s1 = StringSlice("hello")
    var s2 = StringSlice("world")
    b.append(s1)
    b.append(s2)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0], "hello")
    assert_equal(frozen[1], "world")


def test_string_builder_unsafe_append_string_slice() raises:
    """Unsafe_append(StringSlice) stores the string correctly when capacity is pre-reserved.
    """
    var b = StringBuilder(2, bytes_capacity=10)
    var s1 = StringSlice("hi")
    var s2 = StringSlice("bye")
    b.unsafe_append(s1)
    b.unsafe_append(s2)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    assert_equal(frozen[0], "hi")
    assert_equal(frozen[1], "bye")


def test_string_builder_as_any_builder() raises:
    var b = StringBuilder(1)
    b.append("test")
    var builder: AnyBuilder = b^
    assert_equal(builder.length(), 1)
    assert_equal(builder.dtype(), string)


# ---------------------------------------------------------------------------
# ListBuilder
# ---------------------------------------------------------------------------


def test_list_builder_empty() raises:
    var child = PrimitiveBuilder[int32]()
    var b = ListBuilder(child^)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 0)
    assert_true(frozen.dtype.is_list())


def test_list_builder_append_null() raises:
    var child = PrimitiveBuilder[int64]()
    var b = ListBuilder(child^)
    b.append_valid()
    b.append_null()
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_list_builder_null_count() raises:
    var child = PrimitiveBuilder[int32]()
    var b = ListBuilder(child^)
    b.append_valid()
    b.append_null()
    b.append_null()
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 4)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    assert_true(frozen.is_valid(3))


def test_list_builder_empty_list() raises:
    """A valid but empty list (zero child elements) is valid."""
    var child = PrimitiveBuilder[int64]()
    var b = ListBuilder(child^)
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 1)
    assert_true(frozen.is_valid(0))
    var inner_val = frozen[0].value()
    ref inner = inner_val.as_primitive[int64]()
    assert_equal(inner.length, 0)


def test_list_builder_dtype() raises:
    var child = PrimitiveBuilder[int64]()
    var b: AnyBuilder = ListBuilder(child^)
    assert_equal(b.dtype(), list_(int64))


def test_list_builder_child_accessor() raises:
    var child = PrimitiveBuilder[int32](4)
    child.append(10)
    var b = ListBuilder(child^)
    var child_view = b.values()
    assert_equal(child_view.length(), 1)


def test_list_builder_multiple_nulls_offsets() raises:
    """Multiple null entries must not advance child offsets."""
    var child = PrimitiveBuilder[int32]()
    child.append(1)
    child.append(2)
    var b = ListBuilder(child^)
    b.append_valid()  # list with child elements [1, 2]
    b.append_null()  # null — child length unchanged
    b.append_null()  # null — child length unchanged
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_false(frozen.is_valid(2))
    var first_val = frozen[0].value()
    ref first = first_val.as_primitive[int32]()
    assert_equal(first.length, 2)
    assert_equal(first[0], 1)
    assert_equal(first[1], 2)


def test_list_builder_string_child() raises:
    var str_b = StringBuilder()
    str_b.append("hello")
    str_b.append("world")
    var b = ListBuilder(str_b^)
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 1)
    var inner_val = frozen[0].value()
    ref inner = inner_val.as_string()
    assert_equal(inner[0], "hello")
    assert_equal(inner[1], "world")


# ---------------------------------------------------------------------------
# FixedSizeListBuilder
# ---------------------------------------------------------------------------


def test_fixed_size_list_builder_zero_length() raises:
    var child = PrimitiveBuilder[int32]()
    var b = FixedSizeListBuilder(child^, list_size=4)
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 0)
    assert_true(frozen.dtype.is_fixed_size_list())
    assert_equal(frozen.dtype.size, 4)


def test_fixed_size_list_builder_float32() raises:
    var child = PrimitiveBuilder[float32](4)
    child.append(1.0)
    child.append(2.0)
    child.append(3.0)
    child.append(4.0)
    var b = FixedSizeListBuilder(child^, list_size=2)
    b.append_valid()
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    ref first = frozen[0].as_float32()
    assert_equal(first.length, 2)
    ref second = frozen[1].as_float32()
    assert_equal(second.length, 2)


def test_fixed_size_list_builder_with_nulls() raises:
    var child = PrimitiveBuilder[int64](6)
    child.append(0)
    child.append(1)
    child.append(2)
    child.append(3)
    child.append(4)
    child.append(5)
    var b = FixedSizeListBuilder(child^, list_size=2, capacity=3)
    b.append_valid()
    b.append_null()
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    assert_true(frozen.is_valid(0))
    assert_false(frozen.is_valid(1))
    assert_true(frozen.is_valid(2))


def test_fixed_size_list_builder_dtype() raises:
    var child = PrimitiveBuilder[int32]()
    var b: AnyBuilder = FixedSizeListBuilder(child^, list_size=3)
    assert_equal(b.dtype(), fixed_size_list_(int32, 3))


def test_fixed_size_list_builder_child_accessor() raises:
    var child = PrimitiveBuilder[int64](2)
    child.append(100)
    child.append(200)
    var b = FixedSizeListBuilder(child^, list_size=2)
    var child_view = b.values()
    assert_equal(child_view.length(), 2)


def test_fixed_size_list_builder_size1() raises:
    """FixedSizeList of size 1 — each entry is a single-element list."""
    var child = PrimitiveBuilder[int32](3)
    child.append(7)
    child.append(8)
    child.append(9)
    var b = FixedSizeListBuilder(child^, list_size=1)
    b.append_valid()
    b.append_valid()
    b.append_valid()
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 3)
    ref first = frozen[0].as_int32()
    assert_equal(first.length, 1)
    assert_equal(first[0], 7)
    ref third = frozen[2].as_int32()
    assert_equal(third[0], 9)


# ---------------------------------------------------------------------------
# StructBuilder
# ---------------------------------------------------------------------------


def test_struct_builder_zero_length() raises:
    var sb = StructBuilder([])
    var frozen = sb.finish_typed()
    assert_equal(frozen.length, 0)
    assert_true(frozen.dtype.is_struct())


def test_struct_builder_append_valid() raises:
    """Struct validity tracks correct; child builders drive field values."""
    var sb = StructBuilder([field("id", int64), field("score", float64)], capacity=3)
    sb.field_builder(0).as_primitive[int64]().append(1)
    sb.field_builder(0).as_primitive[int64]().append(2)
    sb.field_builder(0).as_primitive[int64]().append(3)
    sb.field_builder(1).as_primitive[float64]().append(0.1)
    sb.field_builder(1).as_primitive[float64]().append(0.2)
    sb.field_builder(1).as_primitive[float64]().append(0.3)
    sb.append_valid()
    sb.append_valid()
    sb.append_valid()
    assert_equal(len(sb), 3)
    var frozen = sb.finish_typed()
    assert_equal(frozen.length, 3)
    assert_equal(len(frozen.dtype.fields), 2)
    assert_equal(frozen.dtype.fields[0].name, "id")
    assert_equal(frozen.dtype.fields[1].name, "score")
    # All entries valid — bitmap is omitted when null_count == 0
    assert_equal(frozen.nulls, 0)


def test_struct_builder_append_null() raises:
    """Null struct entries — validity bitmap reflects nulls."""
    var sb = StructBuilder([field("id", int32)], capacity=2)
    sb.field_builder(0).as_primitive[int32]().append(10)
    sb.field_builder(0).as_primitive[int32]().append(20)
    sb.append_valid()
    sb.append_null()
    var frozen = sb.finish_typed()
    assert_equal(frozen.length, 2)
    assert_true(frozen.bitmap.value().is_valid(0))
    assert_false(frozen.bitmap.value().is_valid(1))


def test_struct_builder_field_values_accessible() raises:
    """Child field values are accessible after finish."""
    var sb = StructBuilder([field("x", int32)], capacity=2)
    sb.field_builder(0).as_primitive[int32]().append(42)
    sb.field_builder(0).as_primitive[int32]().append(99)
    sb.append_valid()
    sb.append_valid()
    var frozen = sb.finish_typed()

    ref field_data = frozen.unsafe_get("x")
    ref x_arr = field_data.as_primitive[int32]()
    assert_equal(x_arr[0], 42)
    assert_equal(x_arr[1], 99)


def test_struct_builder_multi_type_fields() raises:
    """Struct with primitive, string, and bool fields."""
    var sb = StructBuilder(
        [field("id", int64), field("name", string), field("active", bool_)],
        capacity=2,
    )
    sb.field_builder(0).as_primitive[int64]().append(1)
    sb.field_builder(0).as_primitive[int64]().append(2)
    sb.field_builder(1).as_string().append("alice")
    sb.field_builder(1).as_string().append("bob")
    sb.field_builder(2).as_primitive[bool_]().append(True)
    sb.field_builder(2).as_primitive[bool_]().append_null()
    sb.append_valid()
    sb.append_valid()

    var frozen = sb.finish_typed()
    assert_equal(frozen.length, 2)
    assert_equal(len(frozen.dtype.fields), 3)
    assert_equal(frozen.dtype.fields[0].name, "id")
    assert_equal(frozen.dtype.fields[1].name, "name")
    assert_equal(frozen.dtype.fields[2].name, "active")


def test_struct_builder_field_builder() raises:
    var sb = StructBuilder([field("x", int32), field("y", int32)])

    sb.field_builder(0).as_primitive[int32]().append(7)
    sb.field_builder(1).as_primitive[int32]().append(8)
    sb.append_valid()

    assert_equal(sb.field_builder(0).length(), 1)
    assert_equal(sb.field_builder(1).length(), 1)


def test_struct_builder_capacity_growth() raises:
    var sb = StructBuilder([field("id", int32)])
    for i in range(5):
        sb.field_builder(0).as_primitive[int32]().append(i)
        sb.append_valid()
    var frozen = sb.finish_typed()
    assert_equal(frozen.length, 5)


def test_struct_builder_field_names_preserved() raises:
    """Field names survive builder → finish cycle."""
    var sb = StructBuilder([field("alpha", int8), field("beta", int8)])
    sb.field_builder(0).as_primitive[int8]().append(1)
    sb.field_builder(1).as_primitive[int8]().append(2)
    sb.append_valid()
    var frozen = sb.finish_typed()
    assert_equal(frozen.dtype.fields[0].name, "alpha")
    assert_equal(frozen.dtype.fields[1].name, "beta")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_factory_array_empty() raises:
    var a = array[int32]()
    assert_equal(len(a), 0)
    assert_equal(a.null_count(), 0)


def test_factory_array_bool_with_nulls() raises:
    var a = array([True, None, False, None, True])
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 2)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_true(a.is_valid(2))
    assert_false(a.is_valid(3))
    assert_true(a.is_valid(4))
    assert_true(a[0])
    assert_false(a[2])
    assert_true(a[4])


def test_factory_array_bool_all_nulls() raises:
    var a = array([None, None, None])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 3)


def test_factory_array_int_with_nulls() raises:
    var a = array[int32]([1, None, 3, None, 5])
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 2)
    assert_equal(a[0], 1)
    assert_equal(a[2], 3)
    assert_equal(a[4], 5)
    assert_true(a.is_valid(0))
    assert_false(a.is_valid(1))
    assert_false(a.is_valid(3))


def test_factory_array_int_all_valid() raises:
    var a = array[int64]([10, 20, 30])
    assert_equal(len(a), 3)
    assert_equal(a.null_count(), 0)
    assert_equal(a[0], 10)
    assert_equal(a[1], 20)
    assert_equal(a[2], 30)


def test_factory_nulls_all_invalid() raises:
    var a = nulls[int64](5)
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 5)
    for i in range(5):
        assert_false(a.is_valid(i))


def test_factory_nulls_zero() raises:
    var a = nulls[int32](0)
    assert_equal(len(a), 0)


def test_factory_nulls_one() raises:
    var a = nulls[int8](1)
    assert_equal(len(a), 1)
    assert_false(a.is_valid(0))


def test_factory_arange_validity() raises:
    var a = arange[int32](0, 5)
    assert_equal(len(a), 5)
    assert_equal(a.null_count(), 0)
    for i in range(5):
        assert_true(a.is_valid(i))
    assert_equal(a[0], 0)
    assert_equal(a[4], 4)


def test_factory_arange_non_zero_start() raises:
    var a = arange[int64](10, 15)
    assert_equal(len(a), 5)
    assert_equal(a[0], 10)
    assert_equal(a[4], 14)


def test_factory_arange_single() raises:
    var a = arange[int32](7, 8)
    assert_equal(len(a), 1)
    assert_equal(a[0], 7)


def test_factory_arange_empty() raises:
    var a = arange[int32](3, 3)
    assert_equal(len(a), 0)


# ---------------------------------------------------------------------------
# shrink_to_fit / finish buffer sizing
# ---------------------------------------------------------------------------
# Buffer sizes are always 64-byte aligned. With a large initial capacity the
# data buffer is much bigger than needed; finish() must shrink it down to the
# 64-aligned size for exactly `length` elements.
#
# Sentinel sizes used below (capacity=128):
#   int32 data:  128 * 4  = 512 bytes before shrink → 64 bytes after (2 elems)
#   uint32 offs: 129 * 4  = 516 bytes before shrink → 64 bytes after (3 offs)


def test_primitive_builder_finish_shrinks_data_buffer() raises:
    """Finish() shrinks the data buffer to fit the actual element count."""
    var b = PrimitiveBuilder[int32](128)
    b.append(1)
    b.append(2)
    # capacity allocated 512 bytes; only 2 elements written
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    # 2 int32s = 8 bytes → 64-byte aligned = 64 bytes
    assert_equal(frozen.buffer.size, 64)


def test_string_builder_finish_shrinks_offsets_buffer() raises:
    """Finish() shrinks the offsets buffer to (length+1) uint32 entries."""
    var b = StringBuilder(128)
    b.append("hello")
    b.append("world")
    # capacity allocated 129 * 4 = 516 bytes for offsets; only 3 needed
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    # 3 uint32 offsets = 12 bytes → 64-byte aligned = 64 bytes
    assert_equal(frozen.offsets.size, 64)


def test_list_builder_finish_shrinks_offsets_buffer() raises:
    """Finish() shrinks the offsets buffer to (length+1) uint32 entries."""
    var child = PrimitiveBuilder[int32]()
    child.append(10)
    child.append(20)
    child.append(30)
    var b = ListBuilder(child^, 128)
    b.append_valid()  # list [10, 20]
    b.append_valid()  # list [30]
    # capacity allocated 129 * 4 = 516 bytes for offsets; only 3 needed
    var frozen = b.finish_typed()
    assert_equal(frozen.length, 2)
    # 3 uint32 offsets = 12 bytes → 64-byte aligned = 64 bytes
    assert_equal(frozen.offsets.size, 64)


def test_primitive_builder_finish_with_nulls_shrinks_bitmap() raises:
    """Finish() with nulls resizes the bitmap to fit exactly `length` bits."""
    var b = PrimitiveBuilder[int32](128)
    b.append(1)
    b.append_null()
    b.append(3)
    var frozen = b.finish_typed()
    assert_equal(frozen.nulls, 1)
    # 3 bits → 1 byte → 64-byte aligned = 64 bytes
    assert_equal(frozen.bitmap.value()._buffer.size, 64)


def test_any_builder_finish_dispatch_primitive() raises:
    """AnyBuilder.finish() dispatches to PrimitiveBuilder.finish() and returns AnyArray.
    """
    var b = PrimitiveBuilder[int32](10)
    b.append(42)
    b.append(99)
    var builder: AnyBuilder = b^
    var arr = builder.finish()
    assert_equal(arr.length(), 2)
    assert_equal(arr.as_int32()[0], 42)
    assert_equal(arr.as_int32()[1], 99)
    # data buffer is shrunk: 2 int32s = 8 bytes → 64 bytes
    assert_equal(arr.as_data().buffers[0].size, 64)


def test_any_builder_finish_dispatch_string() raises:
    """AnyBuilder.finish() dispatches to StringBuilder.finish() and returns AnyArray.
    """
    var b = StringBuilder(10)
    b.append("foo")
    b.append("bar")
    var builder: AnyBuilder = b^
    var arr = builder.finish()
    assert_equal(arr.length(), 2)
    # offsets buffer shrunk: 3 uint32s = 12 bytes → 64 bytes
    assert_equal(arr.as_data().buffers[0].size, 64)


def test_any_builder_finish_dispatch_list() raises:
    """AnyBuilder.finish() dispatches to ListBuilder.finish() and returns AnyArray.
    """
    var child = PrimitiveBuilder[int32]()
    child.append(1)
    child.append(2)
    var b = ListBuilder(child^, 10)
    b.append_valid()
    b.append_valid()
    var builder: AnyBuilder = b^
    var arr = builder.finish()
    assert_equal(arr.length(), 2)
    # offsets buffer shrunk: 3 uint32s = 12 bytes → 64 bytes
    assert_equal(arr.as_data().buffers[0].size, 64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
