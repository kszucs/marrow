from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.python import Python, PythonObject
from std.memory import alloc
from marrow.c_data import *


fn c_array_from_pyobj(pyobj: PythonObject) raises -> CArrowArray:
    """Import a CArrowArray from any PyArrow object supporting _export_to_c."""
    var ptr = alloc[CArrowArray](1)
    pyobj._export_to_c(Int(ptr))
    var c_array = ptr.take_pointee()
    ptr.free()
    return c_array^


fn c_schema_from_pyobj(pyobj: PythonObject) raises -> CArrowSchema:
    """Import a CArrowSchema from any PyArrow object supporting _export_to_c."""
    var ptr = alloc[CArrowSchema](1)
    pyobj._export_to_c(Int(ptr))
    var c_schema = ptr.take_pointee()
    ptr.free()
    return c_schema^


def test_schema_from_pyarrow() raises:
    var pa = Python.import_module("pyarrow")
    var pyint = pa.field("int_field", pa.int32())
    var pystring = pa.field("string_field", pa.string())
    var pyschema = pa.schema(Python.list())
    pyschema = pyschema.append(pyint)
    pyschema = pyschema.append(pystring)

    var c_schema = c_schema_from_pyobj(pyschema)
    var schema = c_schema.to_dtype()

    assert_equal(schema.fields[0].name, "int_field")
    assert_equal(schema.fields[0].dtype, int32)
    assert_equal(schema.fields[1].name, "string_field")
    assert_equal(schema.fields[1].dtype, string)


def test_primitive_array_from_pyarrow() raises:
    var pa = Python.import_module("pyarrow")
    var pyarr = pa.array(
        Python.list(1, 2, 3, 4, 5),
        mask=Python.list(False, False, False, False, True),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(dtype, int64)
    assert_equal(c_array.length, 5)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.offset, 0)
    assert_equal(c_array.n_buffers, 2)
    assert_equal(c_array.n_children, 0)

    var data = c_array^.to_array(dtype)
    var array = data^.as_int64()
    assert_equal(array.bitmap.value()._buffer.size, 64)
    assert_equal(array.is_valid(0), True)
    assert_equal(array.is_valid(1), True)
    assert_equal(array.is_valid(2), True)
    assert_equal(array.is_valid(3), True)
    assert_equal(array.is_valid(4), False)
    assert_equal(array.unsafe_get(0), 1)
    assert_equal(array.unsafe_get(1), 2)
    assert_equal(array.unsafe_get(2), 3)
    assert_equal(array.unsafe_get(3), 4)
    assert_equal(array.unsafe_get(4), 0)


def test_binary_array_from_pyarrow() raises:
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(
        Python.list("foo", "bar", "baz"),
        mask=Python.list(False, False, True),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(dtype, string)

    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.offset, 0)
    assert_equal(c_array.n_buffers, 3)
    assert_equal(c_array.n_children, 0)

    var data = c_array^.to_array(dtype)
    var array = data^.as_string()

    assert_equal(array.bitmap.value()._buffer.size, 64)
    assert_equal(array.is_valid(0), True)
    assert_equal(array.is_valid(1), True)
    assert_equal(array.is_valid(2), False)

    assert_equal(array.unsafe_get(0), "foo")
    assert_equal(array.unsafe_get(1), "bar")
    assert_equal(array.unsafe_get(2), "")


def test_list_array_from_pyarrow() raises:
    var pa = Python.import_module("pyarrow")

    var pylist1 = Python.list(1, 2, 3)
    var pylist2 = Python.list(4, 5)
    var pylist3 = Python.list(6, 7)
    var pyarr = pa.array(
        Python.list(pylist1, pylist2, pylist3),
        mask=Python.list(False, True, False),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(dtype, list_(int64))

    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.offset, 0)
    assert_equal(c_array.n_buffers, 2)
    assert_equal(c_array.n_children, 1)

    var data = c_array^.to_array(dtype)
    var array = data^.as_list()

    assert_equal(array.bitmap.value()._buffer.size, 64)
    assert_equal(array.is_valid(0), True)
    assert_equal(array.is_valid(1), False)
    assert_equal(array.is_valid(2), True)

    # TODO: reenable once ListArray.unsafe_get properly works
    # var values = array.unsafe_get(0).as_int64()
    # assert_equal(values.unsafe_get(0), 1)
    # assert_equal(values.unsafe_get(1), 2)


def test_schema_from_dtype() raises:
    var c_schema = CArrowSchema.from_dtype(int32)
    var dtype = c_schema[].to_dtype()
    assert_equal(dtype, int32)

    var c_schema_str = CArrowSchema.from_dtype(string)
    var dtype_str = c_schema_str[].to_dtype()
    assert_equal(dtype_str, string)

    var c_schema_bool = CArrowSchema.from_dtype(bool_)
    var dtype_bool = c_schema_bool[].to_dtype()
    assert_equal(dtype_bool, bool_)

    var c_schema_float64 = CArrowSchema.from_dtype(float64)
    var dtype_float64 = c_schema_float64[].to_dtype()
    assert_equal(dtype_float64, float64)


def test_schema_to_field() raises:
    var pa = Python.import_module("pyarrow")
    var pyfield = pa.field(
        "test_field", pa.int32(), nullable=PythonObject(True)
    )
    var c_schema = c_schema_from_pyobj(pyfield)
    var field = c_schema.to_field()
    assert_equal(field.name, "test_field")
    assert_equal(field.dtype, int32)
    assert_equal(field.nullable, True)

    var pyfield_str = pa.field(
        "string_field", pa.string(), nullable=PythonObject(False)
    )
    var c_schema_str = c_schema_from_pyobj(pyfield_str)
    var field_str = c_schema_str.to_field()
    assert_equal(field_str.name, "string_field")
    assert_equal(field_str.dtype, string)
    assert_equal(field_str.nullable, False)


def test_arrow_array_stream() raises:
    var pa = Python.import_module("pyarrow")
    var python = Python()
    ref cpython = python.cpython()

    var data = Python.dict(
        col1=Python.list(1.0, 2.0, 3.0, 4.0, 5.0),
        col2=Python.list("a", "b", "c", "d", "e"),
    )
    var pyschema = pa.schema(
        python.list(
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
        )
    )
    var table = pa.table(data, schema=pyschema)

    var array_stream = ArrowArrayStream.from_pyarrow(table, cpython)

    var c_schema = array_stream.c_schema()
    var schema = c_schema.to_dtype()
    assert_equal(len(schema.fields), 2)
    assert_equal(schema.fields[0].name, "col1")
    assert_equal(schema.fields[0].dtype, int64)
    assert_equal(schema.fields[1].name, "col2")
    assert_equal(schema.fields[1].dtype, string)

    var c_array = array_stream.c_next()
    assert_equal(c_array.length, 5)
    assert_equal(c_array.null_count, 0)

    var array_data = c_array^.to_array(schema)
    assert_equal(array_data.length, 5)
    assert_equal(len(array_data.children), 2)

    var col1_array = array_data.children[0].copy().as_int64()
    assert_equal(col1_array.unsafe_get(0), 1)
    assert_equal(col1_array.unsafe_get(4), 5)

    var col2_array = array_data.children[1].copy().as_string()
    assert_equal(col2_array.unsafe_get(0), "a")
    assert_equal(col2_array.unsafe_get(4), "e")


def test_struct_dtype_conversion() raises:
    var pa = Python.import_module("pyarrow")

    var struct_fields = Python.list(
        Python.tuple("x", pa.int32()), Python.tuple("y", pa.float64())
    )
    var struct_type = pa.`struct`(struct_fields)
    var c_schema = c_schema_from_pyobj(struct_type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_struct())
    assert_equal(len(dtype.fields), 2)
    assert_equal(dtype.fields[0].name, "x")
    assert_equal(dtype.fields[0].dtype, int32)
    assert_equal(dtype.fields[1].name, "y")
    assert_equal(dtype.fields[1].dtype, float64)


def test_list_dtype_conversion() raises:
    var pa = Python.import_module("pyarrow")

    var list_type = pa.list_(pa.int32())
    var c_schema = c_schema_from_pyobj(list_type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_list())
    assert_equal(dtype.fields[0].dtype, int32)


def test_fixed_size_list_dtype_conversion() raises:
    """Format string +w:3 roundtrip through CArrowSchema."""
    var pa = Python.import_module("pyarrow")

    var fsl_type = pa.list_(pa.float32(), 3)
    var c_schema = c_schema_from_pyobj(fsl_type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_fixed_size_list())
    assert_equal(dtype.size, 3)
    assert_equal(dtype.fields[0].dtype, float32)


def test_fixed_size_list_from_pyarrow() raises:
    """Import a FixedSizeList array from PyArrow."""
    var pa = Python.import_module("pyarrow")

    # Create [[1,2,3], [4,5,6], [7,8,9]] as fixed_size_list(int32, 3)
    var pyarr = pa.FixedSizeListArray.from_arrays(
        pa.array(
            Python.list(1, 2, 3, 4, 5, 6, 7, 8, 9),
            type=pa.int32(),
        ),
        3,
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_true(dtype.is_fixed_size_list())
    assert_equal(dtype.size, 3)

    assert_equal(c_array.length, 3)
    assert_equal(c_array.n_buffers, 1)
    assert_equal(c_array.n_children, 1)

    var data = c_array^.to_array(dtype)
    var fsl = data^.as_fixed_size_list()
    assert_equal(len(fsl), 3)

    # First list: [1, 2, 3]
    var first = fsl.unsafe_get(0).as_int32()
    assert_equal(first.unsafe_get(0), 1)
    assert_equal(first.unsafe_get(1), 2)
    assert_equal(first.unsafe_get(2), 3)

    # Second list: [4, 5, 6]
    var second = fsl.unsafe_get(1).as_int32()
    assert_equal(second.unsafe_get(0), 4)
    assert_equal(second.unsafe_get(1), 5)
    assert_equal(second.unsafe_get(2), 6)


def test_numeric_dtypes() raises:
    var pa = Python.import_module("pyarrow")

    var types_to_test = [
        (pa.int8(), int8),
        (pa.uint8(), uint8),
        (pa.int16(), int16),
        (pa.uint16(), uint16),
        (pa.int32(), int32),
        (pa.uint32(), uint32),
        (pa.int64(), int64),
        (pa.uint64(), uint64),
        (pa.float32(), float32),
        (pa.float64(), float64),
    ]

    for i in range(len(types_to_test)):
        var type_pair = types_to_test[i]
        var py_type = type_pair[0]
        ref expected_mojo_type = type_pair[1]

        var c_schema = c_schema_from_pyobj(py_type)
        var dtype = c_schema.to_dtype()
        assert_equal(dtype, expected_mojo_type)


# def test_schema_to_pyarrow():
#     var pa = Python.import_module("pyarrow")

#     var struct_type = struct_(
#         Field("int_field", int32),
#         Field("string_field", string),
#     )

#     try:
#         # mojo->python direction is not working yet
#         var c_schema = CArrowSchema.from_dtype(int32)
#     except Error:
#         pass


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
