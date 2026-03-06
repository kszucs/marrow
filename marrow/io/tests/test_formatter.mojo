from std.testing import assert_equal, assert_true, assert_false, TestSuite

from marrow.arrays import *
from marrow.dtypes import *
from marrow.io.formatter import Formatter
from marrow.test_fixtures.arrays import (
    build_list_of_list,
    build_list_of_int,
    build_struct,
)


def test_primitive_array() raises:
    """Test the formatter for a primitive array."""
    var arr = array[int32](42, 84, 126)

    var output = String()
    var formatter = Formatter()
    formatter.format(output, arr)
    assert_equal(
        output,
        """PrimitiveArray[DataType(code=int32)]([42, 84, 126])""",
    )


def test_list_int_array() raises:
    var arr = build_list_of_int[int64]()
    var output = String()
    var formatter = Formatter()
    formatter.format(output, arr)
    assert_equal(
        output,
        (
            "ListArray([PrimitiveArray[DataType(code=int64)]([1, 2]),"
            " PrimitiveArray[DataType(code=int64)]([3, 4]),"
            " PrimitiveArray[DataType(code=int64)]([5, 6, 7]), ...])"
        ),
    )


def test_list_list_array() raises:
    var arr = build_list_of_list[int16]()
    var output = String()
    var formatter = Formatter()
    formatter.format(output, arr)
    assert_equal(
        output,
        (
            "ListArray([ListArray([PrimitiveArray[DataType(code=int16)]([1,"
            " 2]), PrimitiveArray[DataType(code=int16)]([3, 4])]),"
            " ListArray([PrimitiveArray[DataType(code=int16)]([5, 6, 7]),"
            " PrimitiveArray[DataType(code=int16)]([]),"
            " PrimitiveArray[DataType(code=int16)]([8])]),"
            " ListArray([PrimitiveArray[DataType(code=int16)]([9, 10])]),"
            " ...])"
        ),
    )


def test_empty_struct() raises:
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
        Field("active", materialize[bool_]()),
    ]

    var struct_arr = StructArray(fields^, capacity=10)

    var output = String()
    var formatter = Formatter()
    formatter.format(output, struct_arr)
    assert_equal(
        output,
        "StructArray({})",
    )


def test_struct() raises:
    var struct_arr = build_struct()

    var output = String()
    var formatter = Formatter()
    formatter.format(output, struct_arr)
    assert_equal(
        output,
        (
            "StructArray({'int_data_a': "
            "PrimitiveArray[DataType(code=int32)]([1, 2, 3, ...]), "
            "'int_data_b': PrimitiveArray[DataType(code=int32)]([10, 20, "
            "30])})"
        ),
    )


def test_formatter_with_different_limits() raises:
    """Test formatter with various limit values."""
    var arr = array[int32](1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    # Test limit=0
    var output0 = String()
    var formatter0 = Formatter(limit=0)
    formatter0.format(output0, arr)
    assert_equal(output0, "PrimitiveArray[DataType(code=int32)]([...])")

    # Test limit=1
    var output1 = String()
    var formatter1 = Formatter(limit=1)
    formatter1.format(output1, arr)
    assert_equal(output1, "PrimitiveArray[DataType(code=int32)]([1, ...])")

    # Test limit=10 (should show all elements)
    var output10 = String()
    var formatter10 = Formatter(limit=10)
    formatter10.format(output10, arr)
    assert_equal(
        output10,
        "PrimitiveArray[DataType(code=int32)]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])",
    )


def test_empty_array() raises:
    """Test formatter with an empty array."""
    var arr = Int32Array(0)

    var output = String()
    var formatter = Formatter()
    formatter.format(output, arr)
    assert_equal(output, "PrimitiveArray[DataType(code=int32)]([])")


def test_all_null_array() raises:
    """Test formatter with an array of all NULL values."""
    var arr = Int32Array(3)
    arr.data.length = 3
    arr.data.bitmap[].unsafe_range_set(0, 3, False)

    var output = String()
    var formatter = Formatter()
    formatter.format(output, arr)
    assert_equal(
        output, "PrimitiveArray[DataType(code=int32)]([NULL, NULL, NULL])"
    )


def test_array_with_nulls() raises:
    """Test formatter with an array containing some NULL values."""
    var arr = Int32Array(5)
    arr.append(1)
    arr.append(2)
    arr.data.bitmap[].unsafe_set(2, False)  # Make third element NULL
    arr.data.length = 3
    arr.append(4)

    var output = String()
    var formatter = Formatter()
    formatter.format(output, arr)
    assert_equal(
        output, "PrimitiveArray[DataType(code=int32)]([1, 2, NULL, ...])"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
