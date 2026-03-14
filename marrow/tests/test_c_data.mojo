from std.testing import assert_equal, assert_true, assert_false, TestSuite
from std.python import Python, PythonObject
from std.memory import alloc
from marrow.c_data import *
from marrow.arrays import Array, BoolArray, PrimitiveArray, StringArray
from marrow.builders import PrimitiveBuilder, StringBuilder, BoolBuilder
from marrow.dtypes import *


fn c_array_from_pyobj(pyobj: PythonObject) raises -> CArrowArray:
    """Import a CArrowArray from any Arrow-compatible Python object via PyCapsule."""
    var py = Python()
    ref cpy = py.cpython()
    var capsule_tuple = pyobj.__arrow_c_array__()
    var raw = cpy.PyCapsule_GetPointer(capsule_tuple[1]._obj_ptr, "arrow_array")
    var src = raw.bitcast[CArrowArray]()
    var copy = src[]
    UnsafePointer(to=src[].release).bitcast[UInt64]()[0] = 0
    return copy


fn c_schema_from_pyobj(pyobj: PythonObject) raises -> CArrowSchema:
    """Import a CArrowSchema from any Arrow-compatible Python object via PyCapsule."""
    var py = Python()
    ref cpy = py.cpython()
    var schema_capsule = pyobj.__arrow_c_schema__()
    var raw = cpy.PyCapsule_GetPointer(schema_capsule._obj_ptr, "arrow_schema")
    var src = raw.bitcast[CArrowSchema]()
    var copy = src[]
    UnsafePointer(to=src[].release).bitcast[UInt64]()[0] = 0
    return copy


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

    from python.c_data import arrow_array_stream_from_python
    var array_stream = arrow_array_stream_from_python(table, cpython)

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


def test_bool_array_from_pyarrow() raises:
    """Boolean arrays are bit-packed; both the values and validity bitmaps."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(
        Python.list(True, False, True, False),
        mask=Python.list(False, False, False, True),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(dtype, bool_)
    assert_equal(c_array.length, 4)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.n_buffers, 2)
    assert_equal(c_array.n_children, 0)

    var data = c_array^.to_array(dtype)
    var arr = data^.as_bool()

    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_false(arr.is_valid(3))

    assert_true(arr.unsafe_get(0))
    assert_false(arr.unsafe_get(1))
    assert_true(arr.unsafe_get(2))


def test_primitive_array_no_nulls() raises:
    """Array with no nulls: buffers[0] (validity bitmap) pointer is null."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(Python.list(10, 20, 30))

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(c_array.null_count, 0)

    var data = c_array^.to_array(dtype)
    var arr = data^.as_int64()

    assert_equal(arr.nulls, 0)  # no null bitmap → all valid
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_equal(arr.unsafe_get(0), 10)
    assert_equal(arr.unsafe_get(1), 20)
    assert_equal(arr.unsafe_get(2), 30)


def test_primitive_array_with_offset() raises:
    """Sliced primitive array: offset field is non-zero."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(Python.list(10, 20, 30, 40)).slice(1)

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    assert_equal(c_array.offset, 1)
    assert_equal(c_array.length, 3)

    var dtype = c_schema.to_dtype()
    var data = c_array^.to_array(dtype)
    var arr = data^.as_int64()

    assert_equal(arr.length, 3)
    assert_equal(arr.offset, 1)
    # Values at logical positions 0..2 correspond to physical positions 1..3
    assert_equal(arr.unsafe_get(0), 20)
    assert_equal(arr.unsafe_get(1), 30)
    assert_equal(arr.unsafe_get(2), 40)


def test_string_array_with_offset() raises:
    """Sliced string array: offset is propagated into StringArray."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(Python.list("foo", "bar", "baz")).slice(1)

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    assert_equal(c_array.offset, 1)
    assert_equal(c_array.length, 2)

    var dtype = c_schema.to_dtype()
    var data = c_array^.to_array(dtype)
    var arr = data^.as_string()

    assert_equal(arr.length, 2)
    assert_equal(arr.offset, 1)
    assert_equal(String(arr.unsafe_get(0)), "bar")
    assert_equal(String(arr.unsafe_get(1)), "baz")


def test_empty_array_from_pyarrow() raises:
    """Empty array (length=0): buffers may be null without error."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(Python.list(), type=pa.int32())

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    assert_equal(c_array.length, 0)
    assert_equal(c_array.null_count, 0)

    var dtype = c_schema.to_dtype()
    assert_equal(dtype, int32)

    var data = c_array^.to_array(dtype)
    assert_equal(data.length, 0)


def test_binary_dtype_array_from_pyarrow() raises:
    """Binary (bytes) array uses the same 3-buffer layout as strings."""
    var pa = Python.import_module("pyarrow")

    var pydata = Python.evaluate("[b'hello', b'world', b'']")
    var pyarr = pa.array(
        pydata,
        type=pa.binary(),
        mask=Python.list(False, False, True),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(dtype, binary)
    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.n_buffers, 3)
    assert_equal(c_array.n_children, 0)

    var data = c_array^.to_array(dtype)
    assert_equal(data.length, 3)
    assert_true(data.is_valid(0))
    assert_true(data.is_valid(1))
    assert_false(data.is_valid(2))


def test_struct_array_values_from_pyarrow() raises:
    """Struct array: verify child column values are accessible."""
    var pa = Python.import_module("pyarrow")

    var col_x = pa.array(Python.list(1, 2, 3), type=pa.int32())
    var col_y = pa.array(Python.list("a", "b", "c"))
    var pyarr = pa.StructArray.from_arrays(
        Python.list(col_x, col_y),
        names=Python.list("x", "y"),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_true(dtype.is_struct())
    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 0)
    assert_equal(c_array.n_buffers, 1)
    assert_equal(c_array.n_children, 2)

    var data = c_array^.to_array(dtype)
    assert_equal(data.length, 3)
    assert_equal(len(data.children), 2)

    var xs = data.children[0].copy().as_int32()
    assert_equal(xs.unsafe_get(0), 1)
    assert_equal(xs.unsafe_get(1), 2)
    assert_equal(xs.unsafe_get(2), 3)

    var ys = data.children[1].copy().as_string()
    assert_equal(String(ys.unsafe_get(0)), "a")
    assert_equal(String(ys.unsafe_get(1)), "b")
    assert_equal(String(ys.unsafe_get(2)), "c")


def test_fixed_size_list_with_nulls() raises:
    """FixedSizeList array with a null row: bitmap must reflect validity."""
    var pa = Python.import_module("pyarrow")

    var flat = pa.array(Python.list(1, 2, 3, 0, 0, 0), type=pa.int32())
    var pyarr = pa.FixedSizeListArray.from_arrays(
        flat,
        3,
        mask=pa.array(Python.list(False, True)),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_true(dtype.is_fixed_size_list())
    assert_equal(c_array.length, 2)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.n_buffers, 1)
    assert_equal(c_array.n_children, 1)

    var data = c_array^.to_array(dtype)
    var fsl = data^.as_fixed_size_list()

    assert_true(fsl.is_valid(0))
    assert_false(fsl.is_valid(1))

    var first = fsl.unsafe_get(0).as_int32()
    assert_equal(first.unsafe_get(0), 1)
    assert_equal(first.unsafe_get(1), 2)
    assert_equal(first.unsafe_get(2), 3)


def test_export_primitive_array_roundtrip() raises:
    """PrimitiveArray built in Mojo round-trips through PyArrow and back."""
    var pa = Python.import_module("pyarrow")
    var b = PrimitiveBuilder[int32](3)
    b.append(7)
    b.append(42)
    b.append(-1)
    var arr: Array = b.finish_typed()

    # Export via C Data Interface, re-import into PyArrow, then back into Mojo.
    var c_schema = CArrowSchema.from_dtype(arr.dtype)
    var c_array = CArrowArray.from_array(arr)
    var pyarr = pa.Array._import_from_c(Int(c_array), Int(c_schema))
    c_array.free()
    c_schema.free()
    var reimported = c_array_from_pyobj(pyarr)^.to_array(int32)
    var result = reimported^.as_int32()

    assert_equal(result.length, 3)
    assert_equal(result.unsafe_get(0), 7)
    assert_equal(result.unsafe_get(1), 42)
    assert_equal(result.unsafe_get(2), -1)


def test_export_string_array_roundtrip() raises:
    """StringArray built in Mojo round-trips through PyArrow and back."""
    var pa = Python.import_module("pyarrow")
    var b = StringBuilder(3)
    b.append("hello")
    b.append("world")
    b.append("mojo")
    var arr: Array = b.finish_typed()

    var c_schema = CArrowSchema.from_dtype(arr.dtype)
    var c_array = CArrowArray.from_array(arr)
    var pyarr = pa.Array._import_from_c(Int(c_array), Int(c_schema))
    c_array.free()
    c_schema.free()
    var reimported = c_array_from_pyobj(pyarr)^.to_array(string)
    var result = reimported^.as_string()

    assert_equal(result.length, 3)
    assert_equal(String(result.unsafe_get(0)), "hello")
    assert_equal(String(result.unsafe_get(1)), "world")
    assert_equal(String(result.unsafe_get(2)), "mojo")


def test_export_array_with_nulls_roundtrip() raises:
    """Null bitmap survives Mojo → PyArrow → Mojo roundtrip."""
    var pa = Python.import_module("pyarrow")
    var b = PrimitiveBuilder[int64](4)
    b.append(1)
    b.append_null()
    b.append(3)
    b.append_null()
    var arr: Array = b.finish_typed()

    var c_schema = CArrowSchema.from_dtype(arr.dtype)
    var c_array = CArrowArray.from_array(arr)
    var pyarr = pa.Array._import_from_c(Int(c_array), Int(c_schema))
    c_array.free()
    c_schema.free()
    var reimported = c_array_from_pyobj(pyarr)^.to_array(int64)
    var result = reimported^.as_int64()

    assert_equal(result.length, 4)
    assert_true(result.is_valid(0))
    assert_false(result.is_valid(1))
    assert_true(result.is_valid(2))
    assert_false(result.is_valid(3))
    assert_equal(result.unsafe_get(0), 1)
    assert_equal(result.unsafe_get(2), 3)


def test_schema_from_dtype_all_types() raises:
    """All supported dtypes survive a from_dtype → to_dtype roundtrip."""
    var types = [
        int8, uint8, int16, uint16, int32, uint32, int64, uint64,
        float16, float32, float64, bool_, binary, string,
    ]
    for i in range(len(types)):
        var dt = types[i]
        var c_schema = CArrowSchema.from_dtype(dt)
        var roundtripped = c_schema[].to_dtype()
        assert_equal(roundtripped, dt)

    # Nested types
    var list_dt = list_(int64)
    var c_list = CArrowSchema.from_dtype(list_dt)
    var rt_list = c_list[].to_dtype()
    assert_true(rt_list.is_list())
    assert_equal(rt_list.fields[0].dtype, int64)

    var fsl_dt = fixed_size_list_(float32, 4)
    var c_fsl = CArrowSchema.from_dtype(fsl_dt)
    var rt_fsl = c_fsl[].to_dtype()
    assert_true(rt_fsl.is_fixed_size_list())
    assert_equal(rt_fsl.size, 4)
    assert_equal(rt_fsl.fields[0].dtype, float32)

    var struct_fields = List[Field]()
    struct_fields.append(Field("a", int32, True))
    var struct_dt = struct_(struct_fields)
    var c_struct = CArrowSchema.from_dtype(struct_dt)
    var rt_struct = c_struct[].to_dtype()
    assert_true(rt_struct.is_struct())
    assert_equal(len(rt_struct.fields), 1)
    assert_equal(rt_struct.fields[0].name, "a")
    assert_equal(rt_struct.fields[0].dtype, int32)


def test_schema_field_nullable_flags() raises:
    """ARROW_FLAG_NULLABLE is set iff field.nullable == True."""
    var c_nullable = CArrowSchema.from_field(Field("x", int32, True))
    var f_nullable = c_nullable[].to_field()
    assert_true(f_nullable.nullable)

    var c_required = CArrowSchema.from_field(Field("y", int64, False))
    var f_required = c_required[].to_field()
    assert_false(f_required.nullable)


def test_all_numeric_array_imports() raises:
    """Each numeric type can be imported and values accessed via as_*()."""
    var pa = Python.import_module("pyarrow")

    # int8
    var arr_i8 = c_array_from_pyobj(
        pa.array(Python.list(1, 2, 3), type=pa.int8())
    )
    var data_i8 = arr_i8^.to_array(int8)
    assert_equal(data_i8^.as_int8().unsafe_get(0), 1)

    # uint8
    var arr_u8 = c_array_from_pyobj(
        pa.array(Python.list(10, 20, 30), type=pa.uint8())
    )
    var data_u8 = arr_u8^.to_array(uint8)
    assert_equal(data_u8^.as_uint8().unsafe_get(1), 20)

    # int16
    var arr_i16 = c_array_from_pyobj(
        pa.array(Python.list(100, 200), type=pa.int16())
    )
    var data_i16 = arr_i16^.to_array(int16)
    assert_equal(data_i16^.as_int16().unsafe_get(0), 100)

    # uint16
    var arr_u16 = c_array_from_pyobj(
        pa.array(Python.list(300, 400), type=pa.uint16())
    )
    var data_u16 = arr_u16^.to_array(uint16)
    assert_equal(data_u16^.as_uint16().unsafe_get(1), 400)

    # int32
    var arr_i32 = c_array_from_pyobj(
        pa.array(Python.list(-1, 0, 1), type=pa.int32())
    )
    var data_i32 = arr_i32^.to_array(int32)
    assert_equal(data_i32^.as_int32().unsafe_get(0), -1)

    # uint32
    var arr_u32 = c_array_from_pyobj(
        pa.array(Python.list(0, 4294967295), type=pa.uint32())
    )
    var data_u32 = arr_u32^.to_array(uint32)
    assert_equal(data_u32^.as_uint32().unsafe_get(1), 4294967295)

    # int64 (already covered by test_primitive_array_from_pyarrow, include for completeness)
    var arr_i64 = c_array_from_pyobj(
        pa.array(Python.list(9999999999), type=pa.int64())
    )
    var data_i64 = arr_i64^.to_array(int64)
    assert_equal(data_i64^.as_int64().unsafe_get(0), 9999999999)

    # uint64
    var arr_u64 = c_array_from_pyobj(
        pa.array(Python.list(0, 1), type=pa.uint64())
    )
    var data_u64 = arr_u64^.to_array(uint64)
    assert_equal(data_u64^.as_uint64().unsafe_get(0), 0)

    # float32
    var arr_f32 = c_array_from_pyobj(
        pa.array(Python.list(1.5, 2.5), type=pa.float32())
    )
    var data_f32 = arr_f32^.to_array(float32)
    assert_equal(data_f32^.as_float32().unsafe_get(0), 1.5)

    # float64
    var arr_f64 = c_array_from_pyobj(
        pa.array(Python.list(3.14, 2.71), type=pa.float64())
    )
    var data_f64 = arr_f64^.to_array(float64)
    assert_equal(data_f64^.as_float64().unsafe_get(1), 2.71)


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
