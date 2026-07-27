from std.testing import assert_equal, assert_true, assert_false
from std.python import Python, PythonObject
from std.memory import alloc
from ..c_data import *
from ..tabular import Table
from ..arrays import AnyArray, BoolArray, PrimitiveArray, StringArray
from ..builders import PrimitiveBuilder, StringBuilder, BoolBuilder
from ..dtypes import *


def c_array_from_pyobj(pyobj: PythonObject) raises -> CArrowArray:
    """Import a CArrowArray from any Arrow-compatible Python object via PyCapsule.
    """
    var capsule_tuple = pyobj.__arrow_c_array__()
    return CArrowArray.from_pycapsule(capsule_tuple[1])


def c_schema_from_pyobj(pyobj: PythonObject) raises -> CArrowSchema:
    """Import a CArrowSchema from any Arrow-compatible Python object via PyCapsule.
    """
    return CArrowSchema.from_pycapsule(pyobj.__arrow_c_schema__())


def test_schema_from_pyarrow() raises:
    var pa = Python.import_module("pyarrow")
    var pyint = pa.field("int_field", pa.int32())
    var pystring = pa.field("string_field", pa.string())
    var pyschema = pa.schema(Python.list())
    pyschema = pyschema.append(pyint)
    pyschema = pyschema.append(pystring)

    var c_schema = c_schema_from_pyobj(pyschema)
    var schema = c_schema.to_dtype()

    var sf = schema.as_struct().fields.copy()
    assert_equal(sf[0].name, "int_field")
    assert_equal(sf[0].dtype, int32)
    assert_equal(sf[1].name, "string_field")
    assert_equal(sf[1].dtype, string)


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
    ref array = data.as_int64()
    assert_equal(array.is_valid(0), True)
    assert_equal(array.is_valid(1), True)
    assert_equal(array.is_valid(2), True)
    assert_equal(array.is_valid(3), True)
    assert_equal(array.is_valid(4), False)
    assert_equal(array[0].value(), 1)
    assert_equal(array[1].value(), 2)
    assert_equal(array[2].value(), 3)
    assert_equal(array[3].value(), 4)


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
    ref array = data.as_string()

    assert_equal(array.is_valid(0), True)
    assert_equal(array.is_valid(1), True)
    assert_equal(array.is_valid(2), False)

    assert_equal(array[0].to_string(), "foo")
    assert_equal(array[1].to_string(), "bar")


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
    ref array = data.as_list()

    assert_equal(array.is_valid(0), True)
    assert_equal(array.is_valid(1), False)
    assert_equal(array.is_valid(2), True)

    # TODO: reenable once ListArray.unsafe_get properly works
    # var values = array.unsafe_get(0).as_int64()
    # assert_equal(values.unsafe_get(0), 1)
    # assert_equal(values.unsafe_get(1), 2)


def test_schema_from_dtype() raises:
    var c_schema = CArrowSchema.from_dtype(int32)
    var dtype = c_schema.to_dtype()
    assert_equal(dtype, int32)

    var c_schema_str = CArrowSchema.from_dtype(string)
    var dtype_str = c_schema_str.to_dtype()
    assert_equal(dtype_str, string)

    var c_schema_bool = CArrowSchema.from_dtype(bool_)
    var dtype_bool = c_schema_bool.to_dtype()
    assert_equal(dtype_bool, bool_)

    var c_schema_float64 = CArrowSchema.from_dtype(float64)
    var dtype_float64 = c_schema_float64.to_dtype()
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

    var data = Python.dict(
        col1=Python.list(1.0, 2.0, 3.0, 4.0, 5.0),
        col2=Python.list("a", "b", "c", "d", "e"),
    )
    var pyschema = pa.schema(
        Python.list(
            pa.field("col1", pa.int64()),
            pa.field("col2", pa.string()),
        )
    )
    var py_table = pa.table(data, schema=pyschema)

    var capsule = py_table.__arrow_c_stream__(Python.none())
    var stream = CArrowArrayStream.from_pycapsule(capsule)
    var table = stream.to_table()

    assert_equal(table.num_columns(), 2)
    assert_equal(table.num_rows(), 5)
    assert_equal(table.schema.fields[0].name, "col1")
    assert_equal(table.schema.fields[0].dtype, int64)
    assert_equal(table.schema.fields[1].name, "col2")
    assert_equal(table.schema.fields[1].dtype, string)

    var batches = table.to_batches()
    assert_true(len(batches) >= 1)

    var batch = batches[0].copy()
    ref col1_array = batch.columns[0].as_int64()
    assert_equal(col1_array[0].value(), 1)
    assert_equal(col1_array[4].value(), 5)

    ref col2_array = batch.columns[1].as_string()
    assert_equal(col2_array[0].to_string(), "a")
    assert_equal(col2_array[4].to_string(), "e")


def test_struct_dtype_conversion() raises:
    var pa = Python.import_module("pyarrow")

    var struct_fields = Python.list(
        Python.tuple("x", pa.int32()), Python.tuple("y", pa.float64())
    )
    var struct_type = pa.`struct`(struct_fields)
    var c_schema = c_schema_from_pyobj(struct_type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_struct())
    var df = dtype.as_struct().fields.copy()
    assert_equal(len(df), 2)
    assert_equal(df[0].name, "x")
    assert_equal(df[0].dtype, int32)
    assert_equal(df[1].name, "y")
    assert_equal(df[1].dtype, float64)


def test_list_dtype_conversion() raises:
    var pa = Python.import_module("pyarrow")

    var list_type = pa.list_(pa.int32())
    var c_schema = c_schema_from_pyobj(list_type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_list())
    assert_equal(dtype.as_list().value_type(), int32)


def test_fixed_size_list_dtype_conversion() raises:
    """Format string +w:3 roundtrip through CArrowSchema."""
    var pa = Python.import_module("pyarrow")

    var fsl_type = pa.list_(pa.float32(), 3)
    var c_schema = c_schema_from_pyobj(fsl_type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_fixed_size_list())
    ref fsl = dtype.as_fixed_size_list()
    assert_equal(fsl.size, 3)
    assert_equal(fsl.value_type(), float32)


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
    assert_equal(dtype.as_fixed_size_list().size, 3)

    assert_equal(c_array.length, 3)
    assert_equal(c_array.n_buffers, 1)
    assert_equal(c_array.n_children, 1)

    var data = c_array^.to_array(dtype)
    ref fsl = data.as_fixed_size_list()
    assert_equal(len(fsl), 3)

    # First list: [1, 2, 3]
    ref first = fsl[0].value().as_int32()
    assert_equal(first[0].value(), 1)
    assert_equal(first[1].value(), 2)
    assert_equal(first[2].value(), 3)

    # Second list: [4, 5, 6]
    ref second = fsl[1].value().as_int32()
    assert_equal(second[0].value(), 4)
    assert_equal(second[1].value(), 5)
    assert_equal(second[2].value(), 6)


def test_numeric_dtypes() raises:
    var pa = Python.import_module("pyarrow")

    var pa_types = List[PythonObject]()
    pa_types.append(pa.int8())
    pa_types.append(pa.uint8())
    pa_types.append(pa.int16())
    pa_types.append(pa.uint16())
    pa_types.append(pa.int32())
    pa_types.append(pa.uint32())
    pa_types.append(pa.int64())
    pa_types.append(pa.uint64())
    pa_types.append(pa.float32())
    pa_types.append(pa.float64())
    var arrow_types = List[AnyDataType]()
    arrow_types.append(int8)
    arrow_types.append(uint8)
    arrow_types.append(int16)
    arrow_types.append(uint16)
    arrow_types.append(int32)
    arrow_types.append(uint32)
    arrow_types.append(int64)
    arrow_types.append(uint64)
    arrow_types.append(float32)
    arrow_types.append(float64)

    for i in range(len(pa_types)):
        var c_schema = c_schema_from_pyobj(pa_types[i])
        var dtype = c_schema.to_dtype()
        assert_equal(dtype, arrow_types[i])


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
    ref arr = data.as_bool()

    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_false(arr.is_valid(3))

    assert_true(arr[0].value())
    assert_false(arr[1].value())
    assert_true(arr[2].value())


def test_primitive_array_no_nulls() raises:
    """AnyArray with no nulls: buffers[0] (validity bitmap) pointer is null."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(Python.list(10, 20, 30))

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    var dtype = c_schema.to_dtype()
    assert_equal(c_array.null_count, 0)

    var data = c_array^.to_array(dtype)
    ref arr = data.as_int64()

    assert_equal(arr.nulls, 0)  # no null bitmap → all valid
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_equal(arr[0].value(), 10)
    assert_equal(arr[1].value(), 20)
    assert_equal(arr[2].value(), 30)


def test_c_data_primitive_array_with_offset() raises:
    """Sliced primitive array: offset field is non-zero."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(Python.list(10, 20, 30, 40)).slice(1)

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)

    assert_equal(c_array.offset, 1)
    assert_equal(c_array.length, 3)

    var dtype = c_schema.to_dtype()
    var data = c_array^.to_array(dtype)
    ref arr = data.as_int64()

    assert_equal(arr.length, 3)
    assert_equal(arr.offset, 1)
    # Values at logical positions 0..2 correspond to physical positions 1..3
    assert_equal(arr[0].value(), 20)
    assert_equal(arr[1].value(), 30)
    assert_equal(arr[2].value(), 40)


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
    ref arr = data.as_string()

    assert_equal(arr.length, 2)
    assert_equal(arr.offset, 1)
    assert_equal(String(arr[0]), "bar")
    assert_equal(String(arr[1]), "baz")


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
    assert_equal(data.length(), 0)


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
    assert_equal(data.length(), 3)
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
    assert_equal(data.length(), 3)
    ref data_struct = data.as_struct()
    assert_equal(len(data_struct.children), 2)

    ref xs = data_struct.children[0].as_int32()
    assert_equal(xs[0].value(), 1)
    assert_equal(xs[1].value(), 2)
    assert_equal(xs[2].value(), 3)

    ref ys = data_struct.children[1].as_string()
    assert_equal(String(ys[0]), "a")
    assert_equal(String(ys[1]), "b")
    assert_equal(String(ys[2]), "c")


def test_c_data_fixed_size_list_with_nulls() raises:
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
    ref fsl = data.as_fixed_size_list()

    assert_true(fsl.is_valid(0))
    assert_false(fsl.is_valid(1))

    ref first = fsl[0].value().as_int32()
    assert_equal(first[0].value(), 1)
    assert_equal(first[1].value(), 2)
    assert_equal(first[2].value(), 3)


def test_schema_from_dtype_all_types() raises:
    """All supported dtypes survive a from_dtype → to_dtype roundtrip."""
    var types = List[AnyDataType]()
    types.append(int8)
    types.append(uint8)
    types.append(int16)
    types.append(uint16)
    types.append(int32)
    types.append(uint32)
    types.append(int64)
    types.append(uint64)
    types.append(float16)
    types.append(float32)
    types.append(float64)
    types.append(bool_)
    types.append(binary)
    types.append(string)

    for i in range(len(types)):
        var t = types[i].copy()
        var c_schema = CArrowSchema.from_dtype(t)
        var roundtripped = c_schema.to_dtype()
        assert_equal(roundtripped, t)

    # Nested types
    var list_dt = list_(int64)
    var c_list = CArrowSchema.from_dtype(list_dt.copy().to_any())
    var rt_list = c_list.to_dtype()
    assert_true(rt_list.is_list())
    assert_equal(rt_list.as_list().value_type(), int64)

    var fsl_dt = fixed_size_list_(float32, 4)
    var c_fsl = CArrowSchema.from_dtype(fsl_dt.copy().to_any())
    var rt_fsl = c_fsl.to_dtype()
    assert_true(rt_fsl.is_fixed_size_list())
    ref rt_fsl_t = rt_fsl.as_fixed_size_list()
    assert_equal(rt_fsl_t.size, 4)
    assert_equal(rt_fsl_t.value_type(), float32)

    var struct_fields = List[Field]()
    struct_fields.append(Field("a", int32, True))
    var struct_dt = struct_(struct_fields^)
    var c_struct = CArrowSchema.from_dtype(struct_dt.copy().to_any())
    var rt_struct = c_struct.to_dtype()
    assert_true(rt_struct.is_struct())
    var rt_sf = rt_struct.as_struct().fields.copy()
    assert_equal(len(rt_sf), 1)
    assert_equal(rt_sf[0].name, "a")
    assert_equal(rt_sf[0].dtype, int32)


def test_schema_field_nullable_flags() raises:
    """ARROW_FLAG_NULLABLE is set iff field.nullable == True."""
    var c_nullable = CArrowSchema.from_field(Field("x", int32, True))
    var f_nullable = c_nullable.to_field()
    assert_true(f_nullable.nullable)

    var c_required = CArrowSchema.from_field(Field("y", int64, False))
    var f_required = c_required.to_field()
    assert_false(f_required.nullable)


def test_all_numeric_array_imports() raises:
    """Each numeric type can be imported and values accessed via as_*()."""
    var pa = Python.import_module("pyarrow")

    # int8
    var arr_i8 = c_array_from_pyobj(
        pa.array(Python.list(1, 2, 3), type=pa.int8())
    )
    var data_i8 = arr_i8^.to_array(int8)
    assert_equal(data_i8^.as_int8()[0].value(), 1)

    # uint8
    var arr_u8 = c_array_from_pyobj(
        pa.array(Python.list(10, 20, 30), type=pa.uint8())
    )
    var data_u8 = arr_u8^.to_array(uint8)
    assert_equal(data_u8^.as_uint8()[1].value(), 20)

    # int16
    var arr_i16 = c_array_from_pyobj(
        pa.array(Python.list(100, 200), type=pa.int16())
    )
    var data_i16 = arr_i16^.to_array(int16)
    assert_equal(data_i16^.as_int16()[0].value(), 100)

    # uint16
    var arr_u16 = c_array_from_pyobj(
        pa.array(Python.list(300, 400), type=pa.uint16())
    )
    var data_u16 = arr_u16^.to_array(uint16)
    assert_equal(data_u16^.as_uint16()[1].value(), 400)

    # int32
    var arr_i32 = c_array_from_pyobj(
        pa.array(Python.list(-1, 0, 1), type=pa.int32())
    )
    var data_i32 = arr_i32^.to_array(int32)
    assert_equal(data_i32^.as_int32()[0].value(), -1)

    # uint32
    var arr_u32 = c_array_from_pyobj(
        pa.array(Python.list(0, 4294967295), type=pa.uint32())
    )
    var data_u32 = arr_u32^.to_array(uint32)
    assert_equal(data_u32^.as_uint32()[1].value(), 4294967295)

    # int64 (already covered by test_primitive_array_from_pyarrow, include for completeness)
    var arr_i64 = c_array_from_pyobj(
        pa.array(Python.list(9999999999), type=pa.int64())
    )
    var data_i64 = arr_i64^.to_array(int64)
    assert_equal(data_i64^.as_int64()[0].value(), 9999999999)

    # uint64
    var arr_u64 = c_array_from_pyobj(
        pa.array(Python.list(0, 1), type=pa.uint64())
    )
    var data_u64 = arr_u64^.to_array(uint64)
    assert_equal(data_u64^.as_uint64()[0].value(), 0)

    # float32
    var arr_f32 = c_array_from_pyobj(
        pa.array(Python.list(1.5, 2.5), type=pa.float32())
    )
    var data_f32 = arr_f32^.to_array(float32)
    assert_equal(data_f32^.as_float32()[0].value(), 1.5)

    # float64
    var arr_f64 = c_array_from_pyobj(
        pa.array(Python.list(3.14, 2.71), type=pa.float64())
    )
    var data_f64 = arr_f64^.to_array(float64)
    assert_equal(data_f64^.as_float64()[1].value(), 2.71)


def test_temporal_dtype_schema_roundtrip() raises:
    """All temporal dtypes survive a CArrowSchema.from_dtype → to_dtype roundtrip.

    Regression coverage for the tts/ttS and tss:/tsS: format-string bugs where
    time32[s] and timestamp[s, tz] failed to parse back from PyArrow capsules.
    """
    # date
    var c = CArrowSchema.from_dtype(date32().to_any())
    assert_equal(c.to_dtype(), date32().to_any())
    c = CArrowSchema.from_dtype(date64().to_any())
    assert_equal(c.to_dtype(), date64().to_any())
    # time32 — seconds unit was the bug (tts, not ttS)
    c = CArrowSchema.from_dtype(time32(second).to_any())
    assert_equal(c.to_dtype(), time32(second).to_any())
    c = CArrowSchema.from_dtype(time32(millisecond).to_any())
    assert_equal(c.to_dtype(), time32(millisecond).to_any())
    # time64
    c = CArrowSchema.from_dtype(time64(microsecond).to_any())
    assert_equal(c.to_dtype(), time64(microsecond).to_any())
    c = CArrowSchema.from_dtype(time64(nanosecond).to_any())
    assert_equal(c.to_dtype(), time64(nanosecond).to_any())
    # timestamp — seconds unit was the bug (tss:, not tsS:)
    c = CArrowSchema.from_dtype(timestamp(second).to_any())
    assert_equal(c.to_dtype(), timestamp(second).to_any())
    c = CArrowSchema.from_dtype(timestamp(millisecond).to_any())
    assert_equal(c.to_dtype(), timestamp(millisecond).to_any())
    c = CArrowSchema.from_dtype(timestamp(microsecond).to_any())
    assert_equal(c.to_dtype(), timestamp(microsecond).to_any())
    c = CArrowSchema.from_dtype(timestamp(nanosecond).to_any())
    assert_equal(c.to_dtype(), timestamp(nanosecond).to_any())
    # timestamp with timezone — seconds+tz was the bug (tss:UTC, not tsS:UTC)
    c = CArrowSchema.from_dtype(timestamp(second, "UTC").to_any())
    assert_equal(c.to_dtype(), timestamp(second, "UTC").to_any())
    c = CArrowSchema.from_dtype(timestamp(millisecond, "US/Eastern").to_any())
    assert_equal(c.to_dtype(), timestamp(millisecond, "US/Eastern").to_any())
    c = CArrowSchema.from_dtype(timestamp(microsecond, "Europe/Paris").to_any())
    assert_equal(c.to_dtype(), timestamp(microsecond, "Europe/Paris").to_any())
    c = CArrowSchema.from_dtype(timestamp(nanosecond, "US/Pacific").to_any())
    assert_equal(c.to_dtype(), timestamp(nanosecond, "US/Pacific").to_any())
    # duration
    c = CArrowSchema.from_dtype(duration(second).to_any())
    assert_equal(c.to_dtype(), duration(second).to_any())
    c = CArrowSchema.from_dtype(duration(millisecond).to_any())
    assert_equal(c.to_dtype(), duration(millisecond).to_any())
    c = CArrowSchema.from_dtype(duration(microsecond).to_any())
    assert_equal(c.to_dtype(), duration(microsecond).to_any())
    c = CArrowSchema.from_dtype(duration(nanosecond).to_any())
    assert_equal(c.to_dtype(), duration(nanosecond).to_any())


def test_temporal_schema_from_pyarrow() raises:
    """PyArrow temporal type schemas are correctly parsed into Mojo temporal dtypes.

    Regression coverage: time32('s') exports format 'tts' (not 'ttS'), and
    timestamp('s') / timestamp('s', tz='UTC') export 'tss:' (not 'tsS:').
    """
    var pa = Python.import_module("pyarrow")

    var c = c_schema_from_pyobj(pa.date32())
    assert_equal(c.to_dtype(), date32().to_any())
    c = c_schema_from_pyobj(pa.date64())
    assert_equal(c.to_dtype(), date64().to_any())
    # time32[s] — was broken (ttS), now tts
    c = c_schema_from_pyobj(pa.time32("s"))
    assert_equal(c.to_dtype(), time32(second).to_any())
    c = c_schema_from_pyobj(pa.time32("ms"))
    assert_equal(c.to_dtype(), time32(millisecond).to_any())
    c = c_schema_from_pyobj(pa.time64("us"))
    assert_equal(c.to_dtype(), time64(microsecond).to_any())
    c = c_schema_from_pyobj(pa.time64("ns"))
    assert_equal(c.to_dtype(), time64(nanosecond).to_any())
    # timestamp[s] — was broken (tsS:), now tss:
    c = c_schema_from_pyobj(pa.timestamp("s"))
    assert_equal(c.to_dtype(), timestamp(second).to_any())
    c = c_schema_from_pyobj(pa.timestamp("ms"))
    assert_equal(c.to_dtype(), timestamp(millisecond).to_any())
    c = c_schema_from_pyobj(pa.timestamp("us"))
    assert_equal(c.to_dtype(), timestamp(microsecond).to_any())
    c = c_schema_from_pyobj(pa.timestamp("ns"))
    assert_equal(c.to_dtype(), timestamp(nanosecond).to_any())
    # timestamp with timezone — was broken for seconds unit
    c = c_schema_from_pyobj(pa.timestamp("s", tz="UTC"))
    assert_equal(c.to_dtype(), timestamp(second, "UTC").to_any())
    c = c_schema_from_pyobj(pa.timestamp("ms", tz="US/Eastern"))
    assert_equal(c.to_dtype(), timestamp(millisecond, "US/Eastern").to_any())
    c = c_schema_from_pyobj(pa.timestamp("us", tz="Europe/Paris"))
    assert_equal(c.to_dtype(), timestamp(microsecond, "Europe/Paris").to_any())
    c = c_schema_from_pyobj(pa.timestamp("ns", tz="US/Pacific"))
    assert_equal(c.to_dtype(), timestamp(nanosecond, "US/Pacific").to_any())
    c = c_schema_from_pyobj(pa.duration("s"))
    assert_equal(c.to_dtype(), duration(second).to_any())
    c = c_schema_from_pyobj(pa.duration("ms"))
    assert_equal(c.to_dtype(), duration(millisecond).to_any())
    c = c_schema_from_pyobj(pa.duration("us"))
    assert_equal(c.to_dtype(), duration(microsecond).to_any())
    c = c_schema_from_pyobj(pa.duration("ns"))
    assert_equal(c.to_dtype(), duration(nanosecond).to_any())


def test_date32_array_from_pyarrow() raises:
    """Date32 array import: length, null bitmap, and values."""
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(
        Python.list(10, 20, 30),
        type=pa.date32(),
        mask=Python.list(False, True, False),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_date32())
    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.n_buffers, 2)

    var data = c_array^.to_array(dtype)
    ref arr = data.as_date32()

    assert_equal(len(arr), 3)
    assert_true(arr.is_valid(0))
    assert_false(arr.is_valid(1))
    assert_true(arr.is_valid(2))
    assert_equal(arr[0].value(), 10)
    assert_equal(arr[2].value(), 30)


def test_timestamp_array_from_pyarrow() raises:
    """Timestamp[s, tz=UTC] array import — regression for the tss: format bug.
    """
    var pa = Python.import_module("pyarrow")

    var pyarr = pa.array(
        Python.list(1000, 2000, 3000),
        type=pa.timestamp("s", tz="UTC"),
        mask=Python.list(False, False, True),
    )

    var c_array = c_array_from_pyobj(pyarr)
    var c_schema = c_schema_from_pyobj(pyarr.type)
    var dtype = c_schema.to_dtype()

    assert_true(dtype.is_timestamp())
    ref ts_type = dtype.as_timestamp()
    assert_equal(ts_type.unit, second)
    assert_equal(ts_type.timezone, "UTC")
    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 1)

    var data = c_array^.to_array(dtype)
    ref arr = data.as_timestamp()

    assert_equal(len(arr), 3)
    assert_true(arr.is_valid(0))
    assert_true(arr.is_valid(1))
    assert_false(arr.is_valid(2))
    assert_equal(arr[0].value(), 1000)
    assert_equal(arr[1].value(), 2000)


def test_all_temporal_array_types_from_pyarrow() raises:
    """One representative value import per temporal base type via C Data Interface.
    """
    var pa = Python.import_module("pyarrow")

    # date32
    var ca_d32 = c_array_from_pyobj(
        pa.array(Python.list(100), type=pa.date32())
    )
    var data_d32 = ca_d32^.to_array(date32().to_any())
    ref arr_d32 = data_d32.as_date32()
    assert_equal(arr_d32[0].value(), 100)

    # date64
    var ca_d64 = c_array_from_pyobj(
        pa.array(Python.list(86400000), type=pa.date64())
    )
    var data_d64 = ca_d64^.to_array(date64().to_any())
    ref arr_d64 = data_d64.as_date64()
    assert_equal(arr_d64[0].value(), 86400000)

    # time32[s]
    var ca_t32s = c_array_from_pyobj(
        pa.array(Python.list(3600), type=pa.time32("s"))
    )
    var data_t32s = ca_t32s^.to_array(time32(second).to_any())
    ref arr_t32s = data_t32s.as_time32()
    assert_equal(arr_t32s[0].value(), 3600)

    # time32[ms]
    var ca_t32m = c_array_from_pyobj(
        pa.array(Python.list(3600000), type=pa.time32("ms"))
    )
    var data_t32m = ca_t32m^.to_array(time32(millisecond).to_any())
    ref arr_t32m = data_t32m.as_time32()
    assert_equal(arr_t32m[0].value(), 3600000)

    # time64[us]
    var ca_t64u = c_array_from_pyobj(
        pa.array(Python.list(3600000000), type=pa.time64("us"))
    )
    var data_t64u = ca_t64u^.to_array(time64(microsecond).to_any())
    ref arr_t64u = data_t64u.as_time64()
    assert_equal(arr_t64u[0].value(), 3600000000)

    # time64[ns]
    var ca_t64n = c_array_from_pyobj(
        pa.array(Python.list(3600000000000), type=pa.time64("ns"))
    )
    var data_t64n = ca_t64n^.to_array(time64(nanosecond).to_any())
    ref arr_t64n = data_t64n.as_time64()
    assert_equal(arr_t64n[0].value(), 3600000000000)

    # timestamp[s]
    var ca_ts_s = c_array_from_pyobj(
        pa.array(Python.list(1000), type=pa.timestamp("s"))
    )
    var data_ts_s = ca_ts_s^.to_array(timestamp(second).to_any())
    ref arr_ts_s = data_ts_s.as_timestamp()
    assert_equal(arr_ts_s[0].value(), 1000)

    # timestamp[ns, tz=UTC]
    var ca_ts_ntz = c_array_from_pyobj(
        pa.array(Python.list(1000000000000), type=pa.timestamp("ns", tz="UTC"))
    )
    var data_ts_ntz = ca_ts_ntz^.to_array(timestamp(nanosecond, "UTC").to_any())
    ref arr_ts_ntz = data_ts_ntz.as_timestamp()
    assert_equal(arr_ts_ntz[0].value(), 1000000000000)

    # duration[ms]
    var ca_dur = c_array_from_pyobj(
        pa.array(Python.list(5000), type=pa.duration("ms"))
    )
    var data_dur = ca_dur^.to_array(duration(millisecond).to_any())
    ref arr_dur = data_dur.as_duration()
    assert_equal(arr_dur[0].value(), 5000)


def test_dictionary_dtype_schema_roundtrip() raises:
    """CArrowSchema round-trip for dictionary(int32, string)."""
    var dt = dictionary(AnyDataType(int32), AnyDataType(string)).to_any()
    var c_schema = CArrowSchema.from_dtype(dt)
    var rt = c_schema.to_dtype()
    assert_true(rt.is_dictionary())
    ref dd = rt.as_dictionary()
    assert_true(dd.index_type() == AnyDataType(int32))
    assert_true(dd.value_type() == AnyDataType(string))
    assert_false(dd.ordered)


def test_dictionary_ordered_roundtrip() raises:
    """ARROW_FLAG_DICT_ORDERED is preserved in the schema round-trip."""
    var dt = dictionary(
        AnyDataType(int8), AnyDataType(int32), ordered=True
    ).to_any()
    var c_schema = CArrowSchema.from_dtype(dt)
    var rt = c_schema.to_dtype()
    assert_true(rt.is_dictionary())
    assert_true(rt.as_dictionary().ordered)


def test_dictionary_from_pyarrow() raises:
    """Import a PyArrow dictionary array via C Data Interface."""
    var pa = Python.import_module("pyarrow")

    var pa_vals = pa.array(Python.list("cat", "dog", "fish"))
    var pa_idx = pa.array(Python.list(0, 1, 2, 0, 1), type=pa.int8())
    var pa_dict = pa.DictionaryArray.from_arrays(pa_idx, pa_vals)

    var c_schema = c_schema_from_pyobj(pa_dict.type)
    var dtype = c_schema.to_dtype()
    assert_true(dtype.is_dictionary())

    var c_array = c_array_from_pyobj(pa_dict)
    var data = c_array^.to_array(dtype)
    ref da = data.as_dictionary()
    assert_equal(len(da), 5)
    assert_equal(da[0].value().as_string().to_string(), "cat")
    assert_equal(da[1].value().as_string().to_string(), "dog")
    assert_equal(da[2].value().as_string().to_string(), "fish")
    assert_equal(da[3].value().as_string().to_string(), "cat")
    assert_equal(da[4].value().as_string().to_string(), "dog")


def test_dictionary_to_pyarrow() raises:
    """Export a Mojo dictionary array to PyArrow via C Data Interface."""
    from ..arrays import DictionaryArray
    from ..builders import Int32Builder, StringBuilder

    var pa = Python.import_module("pyarrow")

    var vb = StringBuilder()
    vb.append("red")
    vb.append("green")
    var values: AnyArray = vb.finish()

    var ib = Int32Builder()
    ib.append(0)
    ib.append(1)
    ib.append(0)
    var indices: AnyArray = ib.finish()
    var arr: AnyArray = DictionaryArray.from_arrays(indices^, values^)

    # Verify CArrowSchema structure: format = "i" (int32 index), dictionary != null
    var c_schema = CArrowSchema.from_dtype(arr.dtype())
    var fmt = String(StringSlice(unsafe_from_utf8_ptr=c_schema.format))
    assert_equal(fmt, "i")  # int32 index type format
    # dictionary schema pointer must be non-null
    assert_true(UnsafePointer(to=c_schema.dictionary).bitcast[UInt64]()[0] != 0)

    # Round-trip the array back through CArrow and check values
    var c_array = CArrowArray.from_array(arr)
    var data = c_array^.to_array(arr.dtype())
    ref da = data.as_dictionary()
    assert_equal(len(da), 3)
    assert_equal(da[0].value().as_string().to_string(), "red")
    assert_equal(da[1].value().as_string().to_string(), "green")
    assert_equal(da[2].value().as_string().to_string(), "red")


def test_map_dtype_from_pyarrow() raises:
    """Import a PyArrow map type ('+m') via CArrowSchema."""
    var pa = Python.import_module("pyarrow")
    var map_type = pa.map_(pa.string(), pa.int32())
    var c_schema = c_schema_from_pyobj(map_type)
    var dtype = c_schema.to_dtype()
    assert_true(dtype.is_map())
    ref mt = dtype.as_map()
    assert_equal(mt.key_type(), string)
    assert_equal(mt.item_type(), int32)
    assert_false(mt.keys_sorted)


def test_map_dtype_schema_roundtrip() raises:
    """CArrowSchema round-trip for map(string, int64) incl. keys_sorted flag."""
    var dt = map_(
        AnyDataType(string), AnyDataType(int64), keys_sorted=True
    ).to_any()
    var c_schema = CArrowSchema.from_dtype(dt)
    var fmt = String(StringSlice(unsafe_from_utf8_ptr=c_schema.format))
    assert_equal(fmt, "+m")
    var rt = c_schema.to_dtype()
    assert_true(rt.is_map())
    ref mt = rt.as_map()
    assert_equal(mt.key_type(), string)
    assert_equal(mt.item_type(), int64)
    assert_true(mt.keys_sorted)


def test_map_array_from_pyarrow() raises:
    """Import a PyArrow map array: {"a":1,"b":2}, None, {"c":3}."""
    var pa = Python.import_module("pyarrow")
    var e0 = Python.list(Python.tuple("a", 1), Python.tuple("b", 2))
    var e2 = Python.list(Python.tuple("c", 3))
    var pyarr = pa.array(
        Python.list(e0, Python.none(), e2),
        type=pa.map_(pa.string(), pa.int64()),
    )

    var c_schema = c_schema_from_pyobj(pyarr.type)
    var dtype = c_schema.to_dtype()
    assert_true(dtype.is_map())

    var c_array = c_array_from_pyobj(pyarr)
    assert_equal(c_array.length, 3)
    assert_equal(c_array.null_count, 1)
    assert_equal(c_array.n_buffers, 2)  # validity + offsets
    assert_equal(c_array.n_children, 1)  # entries struct

    var data = c_array^.to_array(dtype)
    ref m = data.as_map()
    assert_equal(len(m), 3)
    assert_true(m.is_valid(0))
    assert_false(m.is_valid(1))
    assert_true(m.is_valid(2))

    var lens = m.value_lengths()
    assert_equal(lens[0].value(), 2)
    assert_equal(lens[1].value(), 0)
    assert_equal(lens[2].value(), 1)

    ref entries = m.values().as_struct()
    assert_equal(len(entries), 3)
    var keys = entries.children[0].as_string().copy()
    var vals = entries.children[1].as_int64().copy()
    assert_equal(keys[0].to_string(), "a")
    assert_equal(keys[1].to_string(), "b")
    assert_equal(keys[2].to_string(), "c")
    assert_equal(vals[0].value(), 1)
    assert_equal(vals[1].value(), 2)
    assert_equal(vals[2].value(), 3)


def test_map_array_roundtrip() raises:
    """Build a Mojo MapArray, export it, and re-import through CArrowArray."""
    from ..arrays import MapArray, Int32Array
    from ..builders import Int32Builder, StringBuilder

    # map: [ {"x":10}, {"y":20,"z":30} ]  ->  offsets [0,1,3]
    var ob = Int32Builder()
    ob.append(0)
    ob.append(1)
    ob.append(3)
    var kb = StringBuilder()
    kb.append("x")
    kb.append("y")
    kb.append("z")
    var vb = Int32Builder()
    vb.append(10)
    vb.append(20)
    vb.append(30)

    var arr: AnyArray = MapArray.from_arrays(
        ob.finish(), kb.finish(), vb.finish()
    )
    assert_true(arr.dtype().is_map())

    var c_array = CArrowArray.from_array(arr)
    var data = c_array^.to_array(arr.dtype())
    ref m = data.as_map()
    assert_equal(len(m), 2)
    var lens = m.value_lengths()
    assert_equal(lens[0].value(), 1)
    assert_equal(lens[1].value(), 2)

    ref entries = m.values().as_struct()
    var keys = entries.children[0].as_string().copy()
    var vals = entries.children[1].as_int32().copy()
    assert_equal(keys[0].to_string(), "x")
    assert_equal(keys[2].to_string(), "z")
    assert_equal(vals[0].value(), 10)
    assert_equal(vals[2].value(), 30)
