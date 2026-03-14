"""Tests for RecordBatch and Table abstractions."""
from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python, PythonObject
from marrow.tabular import RecordBatch, Table
from marrow.arrays import array, Array, PrimitiveArray, StringArray
from marrow.schema import Schema
from marrow.dtypes import int32, int64, float64, string, Field
from marrow.builders import PrimitiveBuilder
from marrow.c_data import CArrowArray, CArrowSchema


fn batch_from_arrow(pa_batch: PythonObject) raises -> RecordBatch:
    """Import a RecordBatch from any Arrow-compatible Python object."""
    var py = Python()
    ref cpy = py.cpython()
    var capsule_tuple = pa_batch.__arrow_c_array__()
    var schema_raw = cpy.PyCapsule_GetPointer(
        capsule_tuple[0]._obj_ptr, "arrow_schema"
    )
    var array_raw = cpy.PyCapsule_GetPointer(
        capsule_tuple[1]._obj_ptr, "arrow_array"
    )
    var c_schema_src = schema_raw.bitcast[CArrowSchema]()
    var c_array_src = array_raw.bitcast[CArrowArray]()
    var c_schema = c_schema_src[]
    var c_array = c_array_src[]
    UnsafePointer(to=c_schema_src[].release).bitcast[UInt64]()[0] = 0
    UnsafePointer(to=c_array_src[].release).bitcast[UInt64]()[0] = 0
    var schema = c_schema.to_schema()
    var n_cols = Int(c_array.n_children)
    var columns = List[Array]()
    for i in range(n_cols):
        var child_c_schema = c_schema.children[i][]
        var child_c_array = c_array.children[i][]
        UnsafePointer(to=c_schema.children[i][].release).bitcast[UInt64]()[0] = 0
        UnsafePointer(to=c_array.children[i][].release).bitcast[UInt64]()[0] = 0
        var dtype = child_c_schema.to_dtype()
        columns.append(child_c_array^.to_array(dtype))
    return RecordBatch(schema=schema, columns=columns^)


fn batch_to_arrow(batch: RecordBatch) raises -> PythonObject:
    """Export a RecordBatch to PyArrow via the C Data Interface."""
    var pa = Python.import_module("pyarrow")
    var arrays = Python.list()
    var names = Python.list()
    for i in range(batch.num_columns()):
        var col = batch.column(i)
        var c_schema = CArrowSchema.from_dtype(col.dtype)
        var c_array = CArrowArray.from_array(col)
        arrays.append(pa.Array._import_from_c(
            Int(UnsafePointer(to=c_array)), Int(UnsafePointer(to=c_schema))
        ))
        names.append(String(batch.schema.fields[i].name))
    return pa.RecordBatch.from_arrays(arrays, names=names)


def test_record_batch_construction() raises:
    """Test basic RecordBatch construction and property accessors."""
    var schema = Schema(fields=[Field("x", int32), Field("y", float64)])
    var col_x: Array = array[int32]([1, 2, 3])
    var by = PrimitiveBuilder[float64](3)
    by.append(1.0)
    by.append(2.0)
    by.append(3.0)
    var col_y: Array = by.finish()
    var columns = List[Array]()
    columns.append(col_x^)
    columns.append(col_y^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    assert_equal(batch.num_rows(), 3)
    assert_equal(batch.num_columns(), 2)
    assert_equal(batch.column_names(), List[String](["x", "y"]))


def test_record_batch_column_access_by_index() raises:
    """Test column access by index."""
    var schema = Schema(fields=[Field("a", int32), Field("b", int64)])
    var col_a: Array = array[int32]([10, 20, 30])
    var col_b: Array = array[int64]([100, 200, 300])
    var columns = List[Array]()
    columns.append(col_a^)
    columns.append(col_b^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    assert_equal(batch.column(0).length, 3)
    assert_equal(batch.column(1).length, 3)
    assert_equal(batch.column(0).as_int32().unsafe_get(0), 10)
    assert_equal(batch.column(1).as_int64().unsafe_get(2), 300)


def test_record_batch_column_access_by_name() raises:
    """Test column access by name."""
    var schema = Schema(fields=[Field("a", int32), Field("b", int64)])
    var col_a: Array = array[int32]([7, 8, 9])
    var col_b: Array = array[int64]([70, 80, 90])
    var columns = List[Array]()
    columns.append(col_a^)
    columns.append(col_b^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    assert_equal(batch.column("a").as_int32().unsafe_get(0), 7)
    assert_equal(batch.column("b").as_int64().unsafe_get(1), 80)


def test_record_batch_slice() raises:
    """Test zero-copy RecordBatch slice."""
    var schema = Schema(fields=[Field("v", int32)])
    var col: Array = array[int32]([10, 20, 30, 40, 50])
    var columns = List[Array]()
    columns.append(col^)
    var batch = RecordBatch(schema=schema, columns=columns^)

    var sliced = batch.slice(1, 3)
    assert_equal(sliced.num_rows(), 3)
    assert_equal(sliced.column(0).as_int32().unsafe_get(0), 20)
    assert_equal(sliced.column(0).as_int32().unsafe_get(2), 40)

    var tail = batch.slice(3)
    assert_equal(tail.num_rows(), 2)
    assert_equal(tail.column(0).as_int32().unsafe_get(0), 40)


def test_record_batch_from_pyarrow() raises:
    """Test importing a RecordBatch from PyArrow."""
    var pa = Python.import_module("pyarrow")
    var pa_batch = pa.RecordBatch.from_pydict(
        Python.dict(
            x=pa.array(Python.list(1, 2, 3), type=pa.int32()),
            y=pa.array(Python.list(4.0, 5.0, 6.0), type=pa.float64()),
        )
    )
    var batch = batch_from_arrow(pa_batch)
    assert_equal(batch.num_rows(), 3)
    assert_equal(batch.num_columns(), 2)
    assert_equal(batch.column("x").as_int32().unsafe_get(0), 1)
    assert_equal(batch.column("x").as_int32().unsafe_get(2), 3)


def test_record_batch_to_pyarrow() raises:
    """Test exporting a RecordBatch to PyArrow."""
    var pa = Python.import_module("pyarrow")
    var schema = Schema(fields=[Field("a", int32), Field("b", float64)])
    var col_a: Array = array[int32]([1, 2, 3])
    var bb = PrimitiveBuilder[float64](3)
    bb.append(1.5)
    bb.append(2.5)
    bb.append(3.5)
    var col_b: Array = bb.finish()
    var columns = List[Array]()
    columns.append(col_a^)
    columns.append(col_b^)
    var batch = RecordBatch(schema=schema, columns=columns^)
    var pa_batch = batch_to_arrow(batch)
    assert_equal(Int(py=pa_batch.num_rows), 3)
    assert_equal(Int(py=pa_batch.num_columns), 2)
    assert_equal(Int(py=pa_batch.column("a")[0].as_py()), 1)
    assert_equal(Int(py=pa_batch.column("a")[2].as_py()), 3)


def test_record_batch_roundtrip() raises:
    """Test RecordBatch PyArrow roundtrip."""
    var pa = Python.import_module("pyarrow")
    var pa_batch = pa.RecordBatch.from_pydict(
        Python.dict(
            x=pa.array(Python.list(10, 20, 30), type=pa.int32()),
        )
    )
    var batch = batch_from_arrow(pa_batch)
    var pa_batch2 = batch_to_arrow(batch)
    assert_equal(Bool(pa_batch.equals(pa_batch2)), True)


def test_table_from_batches() raises:
    """Test Table.from_batches with multiple record batches."""
    var schema = Schema(fields=[Field("v", int32)])
    var cols1 = List[Array]()
    cols1.append(array[int32]([1, 2, 3]))
    var b1 = RecordBatch(schema=schema, columns=cols1^)
    var cols2 = List[Array]()
    cols2.append(array[int32]([4, 5]))
    var b2 = RecordBatch(schema=schema, columns=cols2^)
    var batches = List[RecordBatch]()
    batches.append(b1^)
    batches.append(b2^)
    var table = Table.from_batches(schema, batches^)
    assert_equal(table.num_rows(), 5)
    assert_equal(table.num_columns(), 1)
    assert_equal(len(table.column(0).chunks), 2)
    assert_equal(table.column_names(), List[String](["v"]))


def test_table_column_access() raises:
    """Test Table column access by index and name."""
    var schema = Schema(fields=[Field("a", int32), Field("b", float64)])
    var bf = PrimitiveBuilder[float64](2)
    bf.append(3.0)
    bf.append(4.0)
    var columns = List[Array]()
    columns.append(array[int32]([1, 2]))
    columns.append(bf.finish())
    var b = RecordBatch(schema=schema, columns=columns^)
    var batches = List[RecordBatch]()
    batches.append(b^)
    var table = Table.from_batches(schema, batches^)
    assert_equal(table.column(0).length, 2)
    assert_equal(table.column("b").length, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
