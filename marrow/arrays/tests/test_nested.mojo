from std.testing import assert_equal, assert_true, assert_false, TestSuite


from marrow.arrays import *
from marrow.dtypes import *
from marrow.test_fixtures.bool_array import as_bool_array_scalar
from marrow.test_fixtures.arrays import build_list_of_list, build_struct


def test_list_int_array() raises:
    var ints = Int64Array(
        ArrayData.from_buffer[int64](
            Buffer.from_values[DType.int64](1, 2, 3), 3
        )
    )
    var lists = ListArray(ints^)
    assert_equal(lists.data.dtype, list_(materialize[int64]()))

    var first_value = lists.unsafe_get(0)
    assert_equal(String(first_value).strip(), "1 2 3")

    assert_equal(len(lists), 1)

    var data = lists^.take_data()
    assert_equal(data.length, 1)

    var arr = data^.as_list()
    assert_equal(len(arr), 1)


def test_list_bool_array() raises:
    var bools = BoolArray()

    bools.append(as_bool_array_scalar(True))
    bools.append(as_bool_array_scalar(False))
    bools.append(as_bool_array_scalar(True))

    var lists = ListArray(bools^)
    assert_equal(len(lists), 1)
    var first_value = lists.unsafe_get(0)
    var buffer = first_value.buffers[0]

    fn get(index: Int) -> Bool:
        return buffer[].unsafe_get[DType.bool](index)

    assert_equal(get(0), True)
    assert_equal(get(1), False)
    assert_equal(get(2), True)


def test_list_str() raises:
    var strings = StringArray()
    strings.unsafe_append("hello")
    strings.unsafe_append("world")

    var lists = ListArray(strings^)
    var first_value = StringArray(lists.unsafe_get(0))

    assert_equal(first_value.unsafe_get(0), "hello")
    assert_equal(first_value.unsafe_get(1), "world")


def test_list_of_list() raises:
    list2 = build_list_of_list[int64]()
    top = ListArray(list2.unsafe_get(0))
    middle_0 = top.unsafe_get(0)
    bottom = Int64Array(middle_0^)
    assert_equal(bottom.unsafe_get(1), 2)
    assert_equal(bottom.unsafe_get(0), 1)
    middle_1 = top.unsafe_get(1)
    bottom = Int64Array(middle_1^)
    assert_equal(bottom.unsafe_get(0), 3)
    assert_equal(bottom.unsafe_get(1), 4)


def test_struct_array() raises:
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
        Field("active", materialize[bool_]()),
    ]

    var struct_arr = StructArray(fields^, capacity=10)
    assert_equal(len(struct_arr), 0)
    assert_equal(struct_arr.capacity, 10)

    var data = struct_arr^.take_data()
    assert_equal(data.length, 0)
    assert_true(data.dtype.is_struct())
    assert_equal(len(data.dtype.fields), 3)
    assert_equal(data.dtype.fields[0].name, "id")
    assert_equal(data.dtype.fields[1].name, "name")
    assert_equal(data.dtype.fields[2].name, "active")


def test_list_array_str_repr() raises:
    var ints = Int64Array()
    var lists = ListArray(ints^)

    var str_repr = String(lists)
    # var repr_repr = lists.__repr__()

    assert_equal(str_repr, "ListArray(length=1)")
    # assert_equal(repr_repr, "ListArray(length=1)")
    # assert_equal(str_repr, repr_repr)


def test_struct_array_str_repr() raises:
    var fields = [
        Field("id", materialize[int64]()),
        Field("name", materialize[string]()),
    ]

    var struct_arr = StructArray(fields^, capacity=5)

    var str_repr = String(struct_arr)
    # var repr_repr = struct_arr.__repr__()

    assert_equal(str_repr, "StructArray(length=0)")
    # assert_equal(repr_repr, "StructArray(length=0)")
    # assert_equal(str_repr, repr_repr)


def test_struct_array_unsafe_get() raises:
    var struct_array = build_struct()
    ref int_data_a = struct_array.unsafe_get("int_data_a")
    var int_a = Int32Array(int_data_a.copy())
    assert_equal(int_a.unsafe_get(0), 1)
    assert_equal(int_a.unsafe_get(4), 5)
    ref int_data_b = struct_array.unsafe_get("int_data_b")
    var int_b = Int32Array(int_data_b.copy())
    assert_equal(int_b.unsafe_get(0), 10)
    assert_equal(int_b.unsafe_get(2), 30)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
