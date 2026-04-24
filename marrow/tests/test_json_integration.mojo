"""Integration tests: validate Marrow arrays against Arrow integration JSON files.

Each test reads one of the JSON files from testing/integration/, parses it
with `marrow.integration.read_json_file`, and asserts that the resulting
arrays contain the expected values and validity bits.
"""

from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite
from marrow.arrays import Array
from marrow.dtypes import *
from marrow.integration import read_json_file


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_primitives_schema() raises:
    var result = read_json_file("testing/integration/primitives.json")
    assert_equal(len(result.schema), 10)
    assert_equal(result.schema.fields[0].name, "i8")
    assert_equal(result.schema.fields[0].dtype, int8)
    assert_equal(result.schema.fields[3].name, "i64")
    assert_equal(result.schema.fields[3].dtype, int64)
    assert_equal(result.schema.fields[8].name, "f32")
    assert_equal(result.schema.fields[8].dtype, float32)
    assert_equal(result.schema.fields[9].name, "f64")
    assert_equal(result.schema.fields[9].dtype, float64)


def test_primitives_int8() raises:
    var result = read_json_file("testing/integration/primitives.json")
    assert_equal(len(result.batches), 1)
    ref arr = result.batches[0].columns[0].as_int8()
    assert_equal(arr.length, 5)
    assert_equal(arr.null_count(), 1)
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_false(arr.is_valid(2))
    assert_true(arr.is_valid(3))
    assert_true(arr.is_valid(4))
    assert_equal(arr[0], -128)
    assert_equal(arr[1], -1)
    assert_equal(arr[3], 1)
    assert_equal(arr[4], 127)


def test_primitives_int32() raises:
    var result = read_json_file("testing/integration/primitives.json")
    ref arr = result.batches[0].columns[2].as_int32()
    assert_equal(arr.length, 5)
    assert_equal(arr.null_count(), 1)
    assert_false(arr.is_valid(2))
    assert_equal(arr[0], -2147483648)
    assert_equal(arr[1], -1)
    assert_equal(arr[3], 1)
    assert_equal(arr[4], 2147483647)


def test_primitives_int64() raises:
    var result = read_json_file("testing/integration/primitives.json")
    ref arr = result.batches[0].columns[3].as_int64()
    assert_equal(arr.length, 5)
    assert_equal(arr.null_count(), 1)
    assert_false(arr.is_valid(2))
    assert_equal(arr[0], -100)
    assert_equal(arr[1], -1)
    assert_equal(arr[3], 1)
    assert_equal(arr[4], 100)


def test_primitives_uint8() raises:
    var result = read_json_file("testing/integration/primitives.json")
    ref arr = result.batches[0].columns[4].as_uint8()
    assert_equal(arr.length, 5)
    assert_equal(arr[0], 0)
    assert_equal(arr[1], 1)
    assert_equal(arr[3], 127)
    assert_equal(arr[4], 255)


def test_primitives_uint32() raises:
    var result = read_json_file("testing/integration/primitives.json")
    ref arr = result.batches[0].columns[6].as_uint32()
    assert_equal(arr[4], 4294967295)


def test_primitives_float32() raises:
    var result = read_json_file("testing/integration/primitives.json")
    ref arr = result.batches[0].columns[8].as_float32()
    assert_equal(arr.length, 5)
    assert_equal(arr.null_count(), 1)
    assert_false(arr.is_valid(2))
    assert_equal(arr[0], -1.5)
    assert_equal(arr[3], 1.5)


def test_primitives_float64() raises:
    var result = read_json_file("testing/integration/primitives.json")
    ref arr = result.batches[0].columns[9].as_float64()
    assert_equal(arr[0], -1.5)
    assert_equal(arr[3], 1.5)
    assert_equal(arr[4], 3.141592653589793)


# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------


def test_booleans() raises:
    var result = read_json_file("testing/integration/booleans.json")
    assert_equal(len(result.schema), 1)
    assert_equal(result.schema.fields[0].dtype, bool_)
    ref arr = result.batches[0].columns[0].as_bool()
    assert_equal(arr.length, 6)
    assert_equal(arr.null_count(), 2)
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_false(arr.is_valid(2))
    assert_true(arr.is_valid(3))
    assert_true(arr.is_valid(4))
    assert_false(arr.is_valid(5))
    assert_true(arr[0].value())
    assert_false(arr[1].value())
    assert_false(arr[3].value())
    assert_true(arr[4].value())


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def test_strings() raises:
    var result = read_json_file("testing/integration/strings.json")
    assert_equal(result.schema.fields[0].dtype, string)
    ref arr = result.batches[0].columns[0].as_string()
    assert_equal(arr.length, 5)
    assert_equal(arr.null_count(), 1)
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_false(arr.is_valid(2))
    assert_true(arr.is_valid(3))
    assert_true(arr.is_valid(4))
    assert_equal(arr[0], "hello")
    assert_equal(arr[1], "world")
    assert_equal(arr[3], "marrow")
    assert_equal(arr[4], "arrow")


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_lists_schema() raises:
    var result = read_json_file("testing/integration/lists.json")
    assert_equal(len(result.schema), 1)
    assert_true(result.schema.fields[0].dtype.is_list())
    var lt = result.schema.fields[0].dtype.as_list_type()
    assert_equal(lt.value_type(), int32)


def test_lists_validity() raises:
    var result = read_json_file("testing/integration/lists.json")
    ref arr = result.batches[0].columns[0].as_list()
    assert_equal(arr.length, 4)
    assert_equal(arr.null_count(), 1)
    assert_true(arr.is_valid(0))
    assert_false(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_true(arr.is_valid(3))


def test_lists_child_values() raises:
    var result = read_json_file("testing/integration/lists.json")
    ref arr = result.batches[0].columns[0].as_list()
    var first_any = arr[0].value()
    ref first = first_any.as_int32()
    assert_equal(first.length, 3)
    assert_equal(first[0], 1)
    assert_equal(first[1], 2)
    assert_equal(first[2], 3)
    var third_any = arr[2].value()
    ref third = third_any.as_int32()
    assert_equal(third.length, 0)
    var fourth_any = arr[3].value()
    ref fourth = fourth_any.as_int32()
    assert_equal(fourth.length, 3)
    assert_equal(fourth[0], 4)
    assert_equal(fourth[1], 5)
    assert_equal(fourth[2], 6)


# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


def test_structs_schema() raises:
    var result = read_json_file("testing/integration/structs.json")
    assert_equal(len(result.schema), 1)
    ref rec_dt = result.schema.fields[0].dtype
    assert_true(rec_dt.is_struct())
    var st = rec_dt.as_struct_type()
    assert_equal(len(st.fields), 3)
    assert_equal(st.fields[0].name, "x")
    assert_equal(st.fields[0].dtype, int32)
    assert_equal(st.fields[1].name, "y")
    assert_equal(st.fields[1].dtype, float64)
    assert_equal(st.fields[2].name, "s")
    assert_equal(st.fields[2].dtype, string)


def test_structs_validity() raises:
    var result = read_json_file("testing/integration/structs.json")
    ref arr = result.batches[0].columns[0].as_struct()
    assert_equal(arr.length, 3)
    assert_equal(arr.null_count(), 1)
    assert_true(arr.is_valid(0))
    assert_false(arr.is_valid(1))
    assert_true(arr.is_valid(2))


def test_structs_children() raises:
    var result = read_json_file("testing/integration/structs.json")
    ref arr = result.batches[0].columns[0].as_struct()
    ref x_arr = arr.children[0].as_int32()
    assert_equal(x_arr[0], 10)
    assert_equal(x_arr[2], 30)
    ref y_arr = arr.children[1].as_float64()
    assert_equal(y_arr[0], 1.5)
    ref s_arr = arr.children[2].as_string()
    assert_equal(s_arr[0], "foo")
    assert_equal(s_arr[2], "marrow")


def main() raises:
    TestSuite.run[__functions_in_module()]()
