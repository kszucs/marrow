"""Tests for Arrow IPC file and stream I/O.

All tests use only the public top-level functions and reader/writer classes.
PyArrow is used as the reference implementation to validate wire format
correctness in both directions.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python, PythonObject
from ..dtypes import *
from ..arrays import DynArray, DictionaryArray
from ..builders import (
    array,
    BoolBuilder,
    Int8Builder,
    Int16Builder,
    Int32Builder,
    Int64Builder,
    UInt8Builder,
    UInt16Builder,
    UInt32Builder,
    UInt64Builder,
    Float32Builder,
    Float64Builder,
    StringBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
)
from ..schema import Schema
from ..tabular import RecordBatch
from ..ipc import (
    read_ipc_file,
    read_ipc_stream,
    read_ipc_file_schema,
    read_ipc_stream_schema,
    write_ipc_file,
    write_ipc_stream,
    RecordBatchFileReader,
    RecordBatchStreamReader,
    RecordBatchFileWriter,
    RecordBatchStreamWriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_path(suffix: String = ".arrow") raises -> String:
    var tempfile = Python.import_module("tempfile")
    var tmp = tempfile.mkstemp(suffix=suffix)
    var _ = Python.import_module("os").close(tmp[0])
    return String(tmp[1])


def _mk_batch() raises -> RecordBatch:
    """Two-column batch: int32 + float64."""
    var a: DynArray = array([1, 2, 3, 4, 5], int32)
    var b: DynArray = array([1.1, 2.2, 3.3, 4.4, 5.5], float64)
    var fields = List[Field]()
    fields.append(field("a", int32))
    fields.append(field("b", float64))
    var cols = List[DynArray]()
    cols.append(a^)
    cols.append(b^)
    return RecordBatch(schema=Schema(fields=fields^), columns=cols^)


def _single_col_batch(arr: DynArray, f: Field) raises -> RecordBatch:
    var fields = List[Field]()
    fields.append(f.copy())
    var cols = List[DynArray]()
    cols.append(arr.copy())
    return RecordBatch(schema=Schema(fields=fields^), columns=cols^)


def _roundtrip_file(batch: RecordBatch) raises -> RecordBatch:
    var path = _tmp_path()
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_file(path, batches_in)
    var batches_out = read_ipc_file(path)
    assert_equal(len(batches_out), 1)
    return batches_out[0].copy()


def _roundtrip_stream(batch: RecordBatch) raises -> RecordBatch:
    var path = _tmp_path(suffix=".arrows")
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_stream(path, batches_in)
    var batches_out = read_ipc_stream(path)
    assert_equal(len(batches_out), 1)
    return batches_out[0].copy()


# ---------------------------------------------------------------------------
# File format: all supported array types
# ---------------------------------------------------------------------------


def test_primitives_file() raises:
    """All integer and float primitive types round-trip through the file format.
    """
    var i8: DynArray = array([-128, 0, 127], int8)
    var i16: DynArray = array([-32768, 0, 32767], int16)
    var i32: DynArray = array([-1, 0, 1], int32)
    var i64: DynArray = array([-9999999999, 0, 9999999999], int64)
    var u8: DynArray = array([0, 128, 255], uint8)
    var u16: DynArray = array([0, 1000, 65535], uint16)
    var u32: DynArray = array([0, 1, 4294967295], uint32)
    var u64: DynArray = array([0, 1, 18446744073709551615], uint64)
    var f32: DynArray = array([-1.5, 0.0, 1.5], float32)
    var f64: DynArray = array([-1.5, 0.0, 1.5], float64)

    var fields = List[Field]()
    fields.append(field("i8", int8))
    fields.append(field("i16", int16))
    fields.append(field("i32", int32))
    fields.append(field("i64", int64))
    fields.append(field("u8", uint8))
    fields.append(field("u16", uint16))
    fields.append(field("u32", uint32))
    fields.append(field("u64", uint64))
    fields.append(field("f32", float32))
    fields.append(field("f64", float64))

    var cols = List[DynArray]()
    cols.append(i8^)
    cols.append(i16^)
    cols.append(i32^)
    cols.append(i64^)
    cols.append(u8^)
    cols.append(u16^)
    cols.append(u32^)
    cols.append(u64^)
    cols.append(f32^)
    cols.append(f64^)

    var batch = RecordBatch(schema=Schema(fields=fields^), columns=cols^)
    var result = _roundtrip_file(batch)
    assert_true(batch == result)


def test_bool_file() raises:
    var b = BoolBuilder(5)
    b.append(True)
    b.append(False)
    b.append(True)
    b.append(True)
    b.append(False)
    var arr: DynArray = b.finish()
    var batch = _single_col_batch(arr^, field("flags", bool_))
    var result = _roundtrip_file(batch)
    assert_true(batch == result)


def test_string_file() raises:
    var b = StringBuilder(3)
    b.append("hello")
    b.append("world")
    b.append("!")
    var arr: DynArray = b.finish()
    var batch = _single_col_batch(arr^, field("s", string))
    var result = _roundtrip_file(batch)
    assert_true(batch == result)


def test_list_file() raises:
    """List(int32) column round-trips through the file format."""
    var ints_b = Int32Builder()
    var lb = ListBuilder(ints_b^)
    var child_any = lb.values()
    ref child = child_any.as_int32()
    child.append(Int32(1))
    child.append(Int32(2))
    lb.append_valid()
    child.append(Int32(3))
    lb.append_valid()
    child.append(Int32(4))
    child.append(Int32(5))
    child.append(Int32(6))
    lb.append_valid()
    var arr: DynArray = lb.finish()
    var batch = _single_col_batch(arr^, field("items", list_(int32)))
    var result = _roundtrip_file(batch)
    assert_true(batch == result)


def test_fixed_size_list_file() raises:
    """FixedSizeList(float32, 3) column round-trips through the file format."""
    var vals_b = Float32Builder()
    var fslb = FixedSizeListBuilder(vals_b^, 3)
    var child_any = fslb.values()
    ref child = child_any.as_float32()
    child.append(Float32(1.0))
    child.append(Float32(2.0))
    child.append(Float32(3.0))
    fslb.append_valid()
    child.append(Float32(4.0))
    child.append(Float32(5.0))
    child.append(Float32(6.0))
    fslb.append_valid()
    var arr: DynArray = fslb.finish()
    var batch = _single_col_batch(
        arr^, field("vecs", fixed_size_list_(float32, 3))
    )
    var result = _roundtrip_file(batch)
    assert_true(batch == result)


def test_struct_file() raises:
    """Struct(x: float64, y: float64) column round-trips through the file format.
    """
    var child_flds = List[Field]()
    child_flds.append(field("x", float64))
    child_flds.append(field("y", float64))
    var sb = StructBuilder(child_flds.copy(), capacity=3)
    # Re-fetch the field builders per append rather than holding refs across
    # `sb.append_valid()` (which mutates `sb` and invalidates interior refs).
    sb.field_builder(0).as_float64().append(Float64(1.0))
    sb.field_builder(1).as_float64().append(Float64(2.0))
    sb.append_valid()
    sb.field_builder(0).as_float64().append(Float64(3.0))
    sb.field_builder(1).as_float64().append(Float64(4.0))
    sb.append_valid()
    sb.field_builder(0).as_float64().append(Float64(5.0))
    sb.field_builder(1).as_float64().append(Float64(6.0))
    sb.append_valid()
    var arr: DynArray = sb.finish()
    var point_field = field("point", struct_(child_flds^))
    var batch = _single_col_batch(arr^, point_field^)
    var result = _roundtrip_file(batch)
    assert_true(batch == result)


def test_nullable_file() raises:
    """Nullable int32 column with null values round-trips correctly."""
    var path = _tmp_path()
    var b = Int32Builder(4)
    b.append(Int32(10))
    b.append_null()
    b.append(Int32(30))
    b.append_null()
    var arr: DynArray = b.finish()
    var fields = List[Field]()
    fields.append(field("x", int32, nullable=True))
    var cols = List[DynArray]()
    cols.append(arr^)
    var batch = RecordBatch(schema=Schema(fields=fields^), columns=cols^)
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_file(path, batches_in)
    var batches_out = read_ipc_file(path)
    assert_equal(len(batches_out), 1)
    assert_true(batch == batches_out[0])
    assert_equal(Int(batches_out[0].columns[0].to_data().nulls), 2)


def test_multi_batch_file() raises:
    """Multiple batches are stored and recovered in order."""
    var path = _tmp_path()
    var b1 = _mk_batch()
    var b2 = _mk_batch()
    var batches_in = List[RecordBatch]()
    batches_in.append(b1.copy())
    batches_in.append(b2.copy())
    write_ipc_file(path, batches_in)
    var batches_out = read_ipc_file(path)
    assert_equal(len(batches_out), 2)
    assert_true(b1 == batches_out[0])
    assert_true(b2 == batches_out[1])


def test_schema_only_file() raises:
    """Schema-only file (0 batches) round-trips via the schema overload."""
    var path = _tmp_path()
    var fields = List[Field]()
    fields.append(field("a", int32))
    fields.append(field("b", float64))
    var empty_batches = List[RecordBatch]()
    write_ipc_file(path, Schema(fields=fields^), empty_batches)
    var result = read_ipc_file_schema(path)
    assert_equal(result.num_rows(), 0)
    assert_equal(len(result.schema.fields), 2)
    assert_true(result.schema.fields[0].name == "a")
    assert_true(result.schema.fields[1].name == "b")


# ---------------------------------------------------------------------------
# File format: reader class with random access
# ---------------------------------------------------------------------------


def test_file_reader_random_access() raises:
    """RecordBatchFileReader.read_batch(i) returns each batch by index."""
    var path = _tmp_path()
    var b1 = _mk_batch()
    var b2 = _mk_batch()
    var batches_in = List[RecordBatch]()
    batches_in.append(b1.copy())
    batches_in.append(b2.copy())
    write_ipc_file(path, batches_in)

    var reader = RecordBatchFileReader(path)
    assert_equal(reader.num_record_batches(), 2)
    assert_true(b2 == reader.read_batch(1))
    assert_true(b1 == reader.read_batch(0))


def test_file_writer_class() raises:
    """RecordBatchFileWriter writes batches incrementally."""
    var path = _tmp_path()
    var b1 = _mk_batch()
    var b2 = _mk_batch()

    var writer = RecordBatchFileWriter(path, b1.schema)
    writer.write_batch(b1)
    writer.write_batch(b2)
    writer.close()

    var batches_out = read_ipc_file(path)
    assert_equal(len(batches_out), 2)
    assert_true(b1 == batches_out[0])
    assert_true(b2 == batches_out[1])


# ---------------------------------------------------------------------------
# Stream format round-trips
# ---------------------------------------------------------------------------


def test_primitives_stream() raises:
    """Primitive types round-trip through the stream format."""
    var batch = _mk_batch()
    var result = _roundtrip_stream(batch)
    assert_true(batch == result)


def test_bool_stream() raises:
    var b = BoolBuilder(3)
    b.append(True)
    b.append(False)
    b.append(True)
    var arr: DynArray = b.finish()
    var batch = _single_col_batch(arr^, field("flags", bool_))
    var result = _roundtrip_stream(batch)
    assert_true(batch == result)


def test_multi_batch_stream() raises:
    """Multiple batches round-trip through the stream format in order."""
    var path = _tmp_path(suffix=".arrows")
    var b1 = _mk_batch()
    var b2 = _mk_batch()
    var batches_in = List[RecordBatch]()
    batches_in.append(b1.copy())
    batches_in.append(b2.copy())
    write_ipc_stream(path, batches_in)
    var batches_out = read_ipc_stream(path)
    assert_equal(len(batches_out), 2)
    assert_true(b1 == batches_out[0])
    assert_true(b2 == batches_out[1])


def test_schema_only_stream() raises:
    """Schema-only stream (0 batches) round-trips via the schema overload."""
    var path = _tmp_path(suffix=".arrows")
    var fields = List[Field]()
    fields.append(field("x", float32))
    var empty_batches = List[RecordBatch]()
    write_ipc_stream(path, Schema(fields=fields^), empty_batches)
    var result = read_ipc_stream_schema(path)
    assert_equal(result.num_rows(), 0)
    assert_equal(len(result.schema.fields), 1)
    assert_true(result.schema.fields[0].name == "x")


def test_stream_writer_reader() raises:
    """RecordBatchStreamWriter/Reader classes work end-to-end."""
    var path = _tmp_path(suffix=".arrows")
    var b1 = _mk_batch()
    var b2 = _mk_batch()

    var writer = RecordBatchStreamWriter(path, b1.schema)
    writer.write_batch(b1)
    writer.write_batch(b2)
    writer.close()

    var reader = RecordBatchStreamReader(path)
    var all_batches = reader.read_all()
    assert_equal(len(all_batches), 2)
    assert_true(b1 == all_batches[0])
    assert_true(b2 == all_batches[1])


# ---------------------------------------------------------------------------
# PyArrow interop: marrow writes, PyArrow reads
# ---------------------------------------------------------------------------


def test_pyarrow_reads_file() raises:
    """A file written by marrow is correctly read by PyArrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()
    var batch = _mk_batch()
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_file(path, batches_in)

    var reader = pa.ipc.open_file(path)
    assert_equal(Int(py=reader.num_record_batches), 1)
    var pa_batch = reader.get_batch(0)
    assert_equal(Int(py=pa_batch.num_rows), 5)
    assert_equal(Int(py=pa_batch.num_columns), 2)
    assert_true(String(py=pa_batch.schema.field("a").type) == "int32")
    assert_true(String(py=pa_batch.schema.field("b").type) == "double")


def test_pyarrow_reads_stream() raises:
    """A stream written by marrow is correctly read by PyArrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path(suffix=".arrows")
    var batch = _mk_batch()
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_stream(path, batches_in)

    var reader = pa.ipc.open_stream(path)
    var pa_batch = reader.read_next_batch()
    assert_equal(Int(py=pa_batch.num_rows), 5)
    assert_equal(Int(py=pa_batch.num_columns), 2)


def test_pyarrow_reads_bool_and_string() raises:
    """PyArrow correctly reads bool and string columns written by marrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var bb = BoolBuilder(3)
    bb.append(True)
    bb.append(False)
    bb.append(True)
    var bools: DynArray = bb.finish()
    var sb = StringBuilder(3)
    sb.append("a")
    sb.append("b")
    sb.append("c")
    var strs: DynArray = sb.finish()

    var fields = List[Field]()
    fields.append(field("b", bool_))
    fields.append(field("s", string))
    var cols = List[DynArray]()
    cols.append(bools^)
    cols.append(strs^)
    var batch = RecordBatch(schema=Schema(fields=fields^), columns=cols^)
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_file(path, batches_in)

    var reader = pa.ipc.open_file(path)
    var pa_batch = reader.get_batch(0)
    assert_equal(Int(py=pa_batch.num_rows), 3)
    assert_true(Bool(py=pa_batch.column(0)[0].as_py()))
    assert_true(String(py=pa_batch.column(1)[0].as_py()) == "a")


def test_pyarrow_reads_nullable() raises:
    """PyArrow correctly reads a nullable column written by marrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var b = Int32Builder(4)
    b.append(Int32(10))
    b.append_null()
    b.append(Int32(30))
    b.append_null()
    var arr: DynArray = b.finish()
    var fields = List[Field]()
    fields.append(field("x", int32, nullable=True))
    var cols = List[DynArray]()
    cols.append(arr^)
    var batch = RecordBatch(schema=Schema(fields=fields^), columns=cols^)
    var batches_in = List[RecordBatch]()
    batches_in.append(batch.copy())
    write_ipc_file(path, batches_in)

    var reader = pa.ipc.open_file(path)
    var pa_batch = reader.get_batch(0)
    assert_equal(Int(py=pa_batch.column(0).null_count), 2)
    assert_equal(Int(py=pa_batch.column(0)[0].as_py()), 10)


# ---------------------------------------------------------------------------
# PyArrow interop: PyArrow writes, marrow reads
# ---------------------------------------------------------------------------


def test_marrow_reads_pyarrow_file() raises:
    """A file written by PyArrow is correctly read by marrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var fx = pa.field("x", pa.int32())
    var fy = pa.field("y", pa.float64())
    var schema = pa.schema(Python.list(fx, fy))
    var col_x = pa.array(Python.list(10, 20, 30), type=pa.int32())
    var col_y = pa.array(Python.list(1.1, 2.2, 3.3), type=pa.float64())
    var pa_batch = pa.RecordBatch.from_arrays(
        Python.list(col_x, col_y), schema=schema
    )
    var writer = pa.ipc.new_file(path, schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_file(path)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 3)
    assert_equal(len(batches[0].schema.fields), 2)
    assert_true(batches[0].schema.fields[0].name == "x")
    assert_true(batches[0].schema.fields[0].dtype == int32)
    assert_true(batches[0].schema.fields[1].dtype == float64)


def test_marrow_reads_pyarrow_stream() raises:
    """A stream written by PyArrow is correctly read by marrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path(suffix=".arrows")

    var fv = pa.field("v", pa.float32())
    var schema = pa.schema(Python.list(fv))
    var col_v = pa.array(Python.list(1.0, 2.0, 3.0), type=pa.float32())
    var pa_batch = pa.RecordBatch.from_arrays(Python.list(col_v), schema=schema)
    var writer = pa.ipc.new_stream(path, schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_stream(path)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 3)
    assert_true(batches[0].schema.fields[0].dtype == float32)


def test_marrow_reads_pyarrow_all_types() raises:
    """Marrow correctly reads all Arrow primitive types written by PyArrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var schema = pa.schema(
        Python.list(
            pa.field("i8", pa.int8()),
            pa.field("i16", pa.int16()),
            pa.field("i32", pa.int32()),
            pa.field("i64", pa.int64()),
            pa.field("u8", pa.uint8()),
            pa.field("u16", pa.uint16()),
            pa.field("u32", pa.uint32()),
            pa.field("u64", pa.uint64()),
            pa.field("f32", pa.float32()),
            pa.field("f64", pa.float64()),
            pa.field("b", pa.bool_()),
            pa.field("s", pa.utf8()),
        )
    )
    var cols = Python.list(
        pa.array(Python.list(1), type=pa.int8()),
        pa.array(Python.list(2), type=pa.int16()),
        pa.array(Python.list(3), type=pa.int32()),
        pa.array(Python.list(4), type=pa.int64()),
        pa.array(Python.list(5), type=pa.uint8()),
        pa.array(Python.list(6), type=pa.uint16()),
        pa.array(Python.list(7), type=pa.uint32()),
        pa.array(Python.list(8), type=pa.uint64()),
        pa.array(Python.list(9.0), type=pa.float32()),
        pa.array(Python.list(10.0), type=pa.float64()),
        pa.array(Python.list(True), type=pa.bool_()),
        pa.array(Python.list("hello"), type=pa.utf8()),
    )
    var pa_batch = pa.RecordBatch.from_arrays(cols, schema=schema)
    var writer = pa.ipc.new_file(path, schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_file(path)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 1)
    assert_equal(len(batches[0].schema.fields), 12)
    assert_true(batches[0].schema.fields[0].dtype == int8)
    assert_true(batches[0].schema.fields[2].dtype == int32)
    assert_true(batches[0].schema.fields[8].dtype == float32)
    assert_true(batches[0].schema.fields[10].dtype == bool_)
    assert_true(batches[0].schema.fields[11].dtype == string)


def test_marrow_reads_pyarrow_list() raises:
    """Marrow correctly reads a List(int32) column written by PyArrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var list_type = pa.list_(pa.int32())
    var fitems = pa.field("items", list_type)
    var schema = pa.schema(Python.list(fitems))
    var col = pa.array(
        Python.list(
            Python.list(1, 2),
            Python.list(3),
            Python.list(4, 5, 6),
        ),
        type=list_type,
    )
    var pa_batch = pa.RecordBatch.from_arrays(Python.list(col), schema=schema)
    var writer = pa.ipc.new_file(path, schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_file(path)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 3)
    assert_true(batches[0].schema.fields[0].dtype.is_list())


def test_marrow_reads_pyarrow_nullable() raises:
    """Marrow correctly reads null values in a column written by PyArrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var null = Python.evaluate("None")
    var fnull = pa.field("x", pa.int32())
    var schema = pa.schema(Python.list(fnull))
    var col = pa.array(
        Python.list(PythonObject(10), null, PythonObject(30), null),
        type=pa.int32(),
    )
    var pa_batch = pa.RecordBatch.from_arrays(Python.list(col), schema=schema)
    var writer = pa.ipc.new_file(path, schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_file(path)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 4)
    assert_equal(Int(batches[0].columns[0].to_data().nulls), 2)


def _mk_dict_batch() raises -> RecordBatch:
    """Single-column batch: dictionary<int32, string> with 4 elements."""
    var indices: DynArray = array([0, 1, 0, 2], int32)
    var sb = StringBuilder(3)
    sb.append("cat")
    sb.append("dog")
    sb.append("fish")
    var values: DynArray = sb.finish()
    var dict_arr: DynArray = DictionaryArray.from_arrays(indices^, values^)
    var fields = List[Field]()
    fields.append(field("d", dictionary(int32, string)))
    var cols = List[DynArray]()
    cols.append(dict_arr^)
    return RecordBatch(schema=Schema(fields=fields^), columns=cols^)


def test_file_dictionary_roundtrip() raises:
    """IPC file round-trip preserves a dictionary<int32, string> column."""
    var path = _tmp_path()
    var batch = _mk_dict_batch()
    var expected = DictionaryArray(batch.columns[0].to_data())
    var batches_in = List[RecordBatch]()
    batches_in.append(batch^)
    write_ipc_file(path, batches_in)
    var read_back = read_ipc_file(path)
    assert_equal(len(read_back), 1)
    assert_equal(read_back[0].num_rows(), 4)
    assert_true(read_back[0].schema.fields[0].dtype.is_dictionary())
    var got = DictionaryArray(read_back[0].columns[0].to_data())
    assert_true(got == expected)


def test_stream_dictionary_roundtrip() raises:
    """IPC stream round-trip preserves a dictionary<int32, string> column."""
    var path = _tmp_path(".arrows")
    var batch = _mk_dict_batch()
    var expected = DictionaryArray(batch.columns[0].to_data())
    var batches_in = List[RecordBatch]()
    batches_in.append(batch^)
    write_ipc_stream(path, batches_in)
    var read_back = read_ipc_stream(path)
    assert_equal(len(read_back), 1)
    assert_equal(read_back[0].num_rows(), 4)
    assert_true(read_back[0].schema.fields[0].dtype.is_dictionary())
    var got = DictionaryArray(read_back[0].columns[0].to_data())
    assert_true(got == expected)


def test_marrow_reads_pyarrow_dictionary() raises:
    """Marrow correctly reads a dictionary column written by PyArrow."""
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()

    var pa_arr = pa.array(
        Python.list(
            PythonObject("cat"),
            PythonObject("dog"),
            PythonObject("cat"),
            PythonObject("fish"),
        )
    ).dictionary_encode()
    var pa_schema = pa.schema(Python.list(pa.field("d", pa_arr.type)))
    var pa_batch = pa.RecordBatch.from_arrays(
        Python.list(pa_arr), schema=pa_schema
    )
    var writer = pa.ipc.new_file(path, pa_schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_file(path)
    assert_equal(len(batches), 1)
    assert_equal(batches[0].num_rows(), 4)
    assert_true(batches[0].schema.fields[0].dtype.is_dictionary())
