# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arrow IPC file and stream I/O.

All tests use only the public top-level functions and reader/writer classes.
PyArrow is used as the reference implementation to validate wire format
correctness in both directions.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.utils.numerics import nan
from std.python import Python, PythonObject
from ..utils.testing import ScratchDir
from ..dtypes import *
from ..arrays import DynArray, DictionaryArray
from ..builders import (
    DynBuilder,
    MapBuilder,
    arange,
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
    StringViewBuilder,
    BinaryViewBuilder,
    ListBuilder,
    FixedSizeListBuilder,
    StructBuilder,
    FixedSizeBinaryBuilder,
)
from std.memory import ArcPointer
from std.os.path import join

from ..buffers import Buffer
from ..execution import ExecContext
from ..io import ByteSource, Fetched, BufferSource
from ..schema import Schema
from ..tabular import RecordBatch, record_batch
from ..c_data import CArrowArrayStream
from ..kernels.hashing import KeyCompare
from ..errors import (
    ArrowError,
    CorruptError,
    DynError,
    InvalidError,
    NotImplementedError,
)
from ..ipc import (
    BodyCompression,
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


def _dict_batch(var values: DynArray) raises -> RecordBatch:
    """One dictionary column indexing each entry of `values` once."""
    var indices = arange[Int32Type](0, len(values))
    var d: DynArray = DictionaryArray.from_arrays(indices^, values^)
    return record_batch([d^], names=["d"])


def _write_file(batches: List[RecordBatch]) raises -> List[RecordBatch]:
    """Write `batches` to an IPC file and read them back."""
    var path = _tmp_path()
    write_ipc_file(path, batches)
    return read_ipc_file(path)


def _dictionaries() raises -> List[DynArray]:
    """Strings, floats holding a NaN and a null, and fixed-size binary."""
    var fsb = FixedSizeBinaryBuilder(2)
    fsb.append("ab".as_bytes())
    fsb.append_null()
    fsb.append("cd".as_bytes())
    var out = List[DynArray]()
    out.append(array(["cat", "dog"]))
    out.append(array([1.5, nan[DType.float64](), None], float64))
    out.append(fsb.finish())
    return out^


def test_file_dictionary_equal_across_batches() raises:
    """A later batch whose dictionary is a separately built copy of the one
    already written is accepted, and both batches read back as written."""
    var firsts = _dictionaries()
    var seconds = _dictionaries()
    for k in range(len(firsts)):
        var written: List[RecordBatch] = [
            _dict_batch(firsts[k].copy()),
            _dict_batch(seconds[k].copy()),
        ]
        var read = _write_file(written)
        assert_equal(len(read), 2)
        for i in range(2):
            assert_true(
                KeyCompare.equals(read[i].columns[0], written[i].columns[0])
            )


def test_file_dictionary_replacement_refused() raises:
    """The IPC file format holds one dictionary per field, so a batch whose
    dictionary differs from the written one -- other values, one extending
    it, a value where it had a null -- is an `InvalidError` rather than a file
    decoding against the wrong values."""
    var q = nan[DType.float64]()
    var firsts: List[DynArray] = [
        array(["cat", "dog"]),
        array(["cat", "dog"]),
        array([1.5, q, None], float64),
    ]
    var seconds: List[DynArray] = [
        array(["dog", "fish"]),
        array(["cat", "dog", "fish"]),
        array([1.5, q, 2.0], float64),
    ]
    for k in range(len(firsts)):
        var refused = False
        try:
            _ = _write_file(
                [_dict_batch(firsts[k].copy()), _dict_batch(seconds[k].copy())]
            )
        except e:
            refused = DynError(e).isa[InvalidError]()
        assert_true(refused, String("dictionary replacement ", k, " accepted"))


def test_stream_dictionary_replacement_roundtrip() raises:
    """The stream format allows a dictionary to be replaced between batches,
    and the stream writer resends each batch's dictionary, so both batches
    decode against their own values."""
    var path = _tmp_path(".arrows")
    var written: List[RecordBatch] = [
        _dict_batch(array(["cat", "dog"])),
        _dict_batch(array(["fish", "owl", "yak"])),
    ]
    write_ipc_stream(path, written)
    var read = read_ipc_stream(path)
    assert_equal(len(read), 2)
    for i in range(2):
        assert_true(
            KeyCompare.equals(read[i].columns[0], written[i].columns[0])
        )


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


# ---------------------------------------------------------------------------
# Sliced columns.
#
# Arrow IPC bodies are written dense from index 0 — there is no offset field on
# the wire. The encoder wrote the *whole* value buffer while declaring the
# slice's length, and the decoder hardcodes `offset=0`, so a sliced column came
# back as the array's first `length` elements instead of its own.
# ---------------------------------------------------------------------------


def test_file_roundtrip_sliced_column() raises:
    var full = array([10, 20, 30, 40, 50], int64)
    var batch = record_batch([full.slice(2, 3)], names=["a"])
    var got = _roundtrip_file(batch^)
    assert_equal(got.num_rows(), 3)
    assert_true(got.columns[0] == array([30, 40, 50], int64).to_dyn())


def test_file_roundtrip_sliced_column_with_nulls() raises:
    var b = Int64Builder(capacity=5)
    b.append(Int64(10))
    b.append_null()
    b.append(Int64(30))
    b.append_null()
    b.append(Int64(50))
    var full: DynArray = b.finish().to_dyn()
    var batch = record_batch([full.slice(1, 3)], names=["a"])
    var got = _roundtrip_file(batch^)
    assert_equal(got.num_rows(), 3)
    assert_equal(got.columns[0].null_count(), 2)
    assert_true(got.columns[0].is_null(0))
    assert_true(got.columns[0].is_valid(1))
    assert_true(got.columns[0].is_null(2))


def test_file_roundtrip_sliced_string_column() raises:
    var full = array(["a", "b", "c", "d"])
    var batch = record_batch([full.slice(1, 2)], names=["s"])
    var got = _roundtrip_file(batch^)
    assert_equal(got.num_rows(), 2)
    assert_true(got.columns[0] == array(["b", "c"]).to_dyn())


def _pyarrow_of(var batches: List[RecordBatch]) raises -> PythonObject:
    """`batches` as a pyarrow table, through the C stream interface."""
    var pa = Python.import_module("pyarrow")
    var schema = batches[0].schema.copy()
    var caps = CArrowArrayStream.from_batches(schema^, batches^).to_pycapsule()
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _compressed_table() raises -> PythonObject:
    """Nulls, nested and dictionary columns, and one string column large
    enough that its buffers span several 64 KiB LZ4 frame blocks."""
    var pa = Python.import_module("pyarrow")
    var n = 6000
    var ints = Python.list()
    var words = Python.list()
    var lists = Python.list()
    var structs = Python.list()
    var maps = Python.list()
    var dicts = Python.list()
    var flags = Python.list()
    for i in range(n):
        ints.append(Python.none() if i % 7 == 0 else PythonObject(i * 31))
        words.append(String("value-", i % 97, "-", i))
        lists.append(Python.list(i, i + 1) if i % 5 else Python.list())
        structs.append(Python.dict(a=i, b=String("s", i % 13)))
        var m = Python.list()
        m.append(Python.tuple(String("k", i % 3), i))
        maps.append(m)
        dicts.append(String("cat", i % 4))
        flags.append(i % 3 == 0)
    return pa.table(
        Python.dict(
            ints=pa.array(ints, type=pa.int64()),
            words=pa.array(words),
            lists=pa.array(lists, type=pa.list_(pa.int64())),
            structs=pa.array(
                structs,
                type=pa.struct(
                    Python.list(
                        pa.field("a", pa.int64()), pa.field("b", pa.string())
                    )
                ),
            ),
            maps=pa.array(maps, type=pa.map_(pa.string(), pa.int64())),
            dicts=pa.array(dicts).dictionary_encode(),
            flags=pa.array(flags, type=pa.bool_()),
        )
    )


def test_ipc_reads_compressed_bodies() raises:
    """Files and streams pyarrow writes with LZ4 frame and ZSTD bodies,
    dictionaries included, in several batches -- decoded in Mojo and through
    liblz4 and libzstd."""
    var pa = Python.import_module("pyarrow")
    var want = _compressed_table()
    for codec in ["lz4", "zstd"]:
        for stream in [False, True]:
            var path = _tmp_path()
            var opts = pa.ipc.IpcWriteOptions(compression=codec)
            var sink = pa.OSFile(path, "wb")
            var writer = pa.ipc.new_stream(
                sink, want.schema, options=opts
            ) if stream else pa.ipc.new_file(sink, want.schema, options=opts)
            _ = writer.write_table(want, max_chunksize=2500)
            _ = writer.close()
            _ = sink.close()
            for native in [True, False]:
                var batches = read_ipc_stream(
                    path, native_codecs=native
                ) if stream else read_ipc_file(path, native_codecs=native)
                assert_equal(len(batches), 3)
                var got = _pyarrow_of(batches^)
                assert_true(
                    Bool(got.equals(want)),
                    String(
                        codec,
                        " stream" if stream else " file",
                        " native" if native else " library",
                    ),
                )


def test_ipc_writes_compressed_bodies() raises:
    """Pyarrow reads the files and streams marrow writes with LZ4 frame and
    ZSTD bodies, dictionaries included, and so does marrow -- compressed in
    Mojo or through liblz4 and libzstd, and decoded either way."""
    var pa = Python.import_module("pyarrow")
    var want = _compressed_table()
    var caps = want.__arrow_c_stream__(Python.none())
    var batches = CArrowArrayStream.from_pycapsule(caps).to_table().to_batches()
    for codec in [BodyCompression.LZ4_FRAME, BodyCompression.ZSTD]:
        for written in [True, False]:
            var path = _tmp_path()
            write_ipc_file(
                path, batches, compression=codec, native_codecs=written
            )
            assert_true(
                Bool(pa.ipc.open_file(path).read_all().equals(want)), "pa file"
            )
            write_ipc_stream(
                path + "s", batches, compression=codec, native_codecs=written
            )
            assert_true(
                Bool(pa.ipc.open_stream(path + "s").read_all().equals(want)),
                "pa stream",
            )
            for read_native in [True, False]:
                var file = read_ipc_file(path, native_codecs=read_native)
                assert_true(Bool(_pyarrow_of(file^).equals(want)), "file")
                var stream = read_ipc_stream(
                    path + "s", native_codecs=read_native
                )
                assert_true(Bool(_pyarrow_of(stream^).equals(want)), "stream")


def test_ipc_reads_arrow_testing_compression() raises:
    """The 2.0.0-compression integration files from apache/arrow-testing,
    the uncompressible ones storing their buffers raw behind a -1 length."""
    var pa = Python.import_module("pyarrow")
    var dir = "marrow/tests/data/ipc/"
    for name in [
        "generated_lz4",
        "generated_zstd",
        "generated_uncompressible_lz4",
        "generated_uncompressible_zstd",
    ]:
        var file = dir + name + ".arrow_file"
        var stream = dir + name + ".stream"
        var want_file = pa.ipc.open_file(file).read_all()
        var want_stream = pa.ipc.open_stream(stream).read_all()
        for native in [True, False]:
            var got = read_ipc_file(file, native_codecs=native)
            assert_true(Bool(_pyarrow_of(got^).equals(want_file)), file)
            got = read_ipc_stream(stream, native_codecs=native)
            assert_true(Bool(_pyarrow_of(got^).equals(want_stream)), stream)


def test_ipc_body_compression_buffers() raises:
    """A buffer is its uncompressed length as an i64, then its frame -- or,
    behind -1, its bytes as they are; anything else is corrupt."""
    var raw: List[UInt8] = [
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        7,
        8,
        9,
    ]
    var buf = BodyCompression.ZSTD.decompress(Span(raw))
    assert_equal(len(buf), 64)  # 3 bytes, in a 64-byte allocation
    assert_equal(Int(buf.unsafe_get(0)), 7)
    assert_equal(Int(buf.unsafe_get(2)), 9)
    # Nothing at all, as `compress` writes an empty buffer, and a length of
    # 0 with nothing after it, as Arrow Java writes one: both empty, without
    # asking the codec.
    for codec in [BodyCompression.LZ4_FRAME, BodyCompression.ZSTD]:
        assert_equal(len(codec.decompress(Span(List[UInt8]()))), 0)
        var zero = List[UInt8](length=8, fill=0)
        assert_equal(len(codec.decompress(Span(zero))), 0)
    var cases = List[List[UInt8]]()
    cases.append([3, 0, 0, 0, 0, 0, 0])  # a short length
    cases.append([0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1])  # -2
    cases.append([5, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3])  # not a frame
    # More than any 3-byte frame holds: refused before it is allocated.
    cases.append([0xF0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 1, 2, 3])
    for bad in cases:
        for codec in [BodyCompression.LZ4_FRAME, BodyCompression.ZSTD]:
            for native in [True, False]:
                var refused = False
                try:
                    _ = codec.decompress(Span(bad), native)
                except e:
                    refused = e.isa[CorruptError]()
                assert_true(refused, String(t"{len(bad)}-byte buffer accepted"))


def test_ipc_body_compression_names() raises:
    """The codecs by pyarrow's names for them, in any case."""
    for name in ["lz4", "LZ4", "lz4_frame", "LZ4_FRAME"]:
        assert_true(
            BodyCompression.from_name(name) == BodyCompression.LZ4_FRAME
        )
    for name in ["zstd", "ZSTD", "Zstd"]:
        assert_true(BodyCompression.from_name(name) == BodyCompression.ZSTD)
    var refused = False
    try:
        _ = BodyCompression.from_name("snappy")
    except:
        refused = True  # an `InvalidError`, as `from_name` declares
    assert_true(refused, "snappy accepted")


def test_delta_dictionary_batch_appends_not_replaces() raises:
    """A delta DictionaryBatch extends the dictionary; it does not replace it.

    `isDelta` is DictionaryBatch slot 2 and was never read, and the decoder
    overwrote the stored dictionary unconditionally — so after a delta carrying
    only the new values, every index in the following batch resolved against a
    truncated dictionary.
    """
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()
    var opts = pa.ipc.IpcWriteOptions(emit_dictionary_deltas=True)
    var dtype = pa.dictionary(pa.int32(), pa.string())
    var schema = pa.schema([pa.field("d", dtype)])
    var b1 = pa.record_batch(
        [pa.array(["a", "b"]).dictionary_encode().cast(dtype)], schema=schema
    )
    var b2 = pa.record_batch(
        [pa.array(["a", "b", "c"]).dictionary_encode().cast(dtype)],
        schema=schema,
    )
    var sink = pa.OSFile(path, "wb")
    var writer = pa.ipc.new_stream(sink, schema, options=opts)
    _ = writer.write_batch(b1)
    _ = writer.write_batch(b2)
    _ = writer.close()
    _ = sink.close()

    var batches = read_ipc_stream(path)
    assert_equal(len(batches), 2)
    var d2 = DictionaryArray(batches[1].columns[0].to_data())
    # the second batch spells "a", "b", "c" — resolvable only if the delta was
    # appended to the dictionary the first batch established
    assert_equal(len(d2), 3)
    assert_equal(len(d2.dictionary().as_string()), 3)


# ---------------------------------------------------------------------------
# V0 — map through IPC.
#
# `map` was implemented in dtypes, arrays, builders, the C Data Interface and
# Parquet, and absent from IPC in both directions: type code 17 simply was not
# in the writer's ladder or the reader's. A map written by marrow came back as
# something else, or not at all.
#
# The buffer walk needed no work: a map owns one offsets buffer like any list,
# which `DynType.layout()` already answers.
# ---------------------------------------------------------------------------


def _map_batch() raises -> RecordBatch:
    """One map column: [{"a": 1}, {}, {"b": 2, "c": 3}]."""
    var b = MapBuilder(map_(DynType(string), DynType(int64)))
    var entries_any = b.entries()
    ref entries = entries_any.as_struct()
    var keys_any = entries.field_builder(0)
    var values_any = entries.field_builder(1)
    ref keys = keys_any.as_string()
    ref values = values_any.as_int64()

    keys.append("a")
    values.append(1)
    entries.append_valid()
    b.append_valid()

    b.append_valid()  # {}

    keys.append("b")
    values.append(2)
    entries.append_valid()
    keys.append("c")
    values.append(3)
    entries.append_valid()
    b.append_valid()

    return record_batch([b.finish().to_dyn()], names=["m"])


def test_ipc_file_round_trips_a_map() raises:
    var batch = _map_batch()
    var back = _roundtrip_file(batch)
    assert_true(back.schema.fields[0].dtype.is_map())
    assert_equal(back.num_rows(), 3)
    assert_true(back.columns[0] == batch.columns[0])


def test_ipc_stream_round_trips_a_map() raises:
    var batch = _map_batch()
    var back = _roundtrip_stream(batch)
    assert_true(back.schema.fields[0].dtype.is_map())
    assert_equal(back.num_rows(), 3)
    assert_true(back.columns[0] == batch.columns[0])


def test_ipc_map_keeps_keys_sorted_flag() raises:
    """`keysSorted` is part of the Map type in the IPC schema, not decoration —
    a reader that drops it reports an unsorted map as sorted."""
    var mt = map_(DynType(string), DynType(int64), keys_sorted=True)
    var b = MapBuilder(mt)
    b.append_valid()  # one empty map is enough to carry the type
    var batch = record_batch([b.finish().to_dyn()], names=["m"])

    var back = _roundtrip_file(batch)
    assert_true(back.schema.fields[0].dtype.as_map().keys_sorted)


def test_pyarrow_reads_a_marrow_written_map() raises:
    """PyArrow must agree, not just marrow with itself.

    A self-round-trip proves the writer and reader share a convention; it does
    not prove the convention is Arrow's. Writing type code 12 (List) for a map
    would pass every test above. This is what pins the format.
    """
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()
    var batches_in = List[RecordBatch]()
    batches_in.append(_map_batch())
    write_ipc_file(path, batches_in)

    var reader = pa.ipc.open_file(path)
    var pa_batch = reader.get_batch(0)
    assert_equal(Int(py=pa_batch.num_rows), 3)
    # PyArrow renders the type as `map<string, int64>`; a list would render as
    # `list<...>`, which is the failure this catches.
    assert_true(String(py=pa_batch.schema.field("m").type).startswith("map<"))

    var got = pa_batch.column(0).to_pylist()
    assert_equal(Int(py=got.__len__()), 3)
    # PyArrow surfaces a map as a list of (key, value) tuples.
    assert_equal(Int(py=got[0].__len__()), 1)
    assert_equal(Int(py=got[1].__len__()), 0)
    assert_equal(Int(py=got[2].__len__()), 2)


def _file_bytes(path: String) raises -> List[UInt8]:
    var buf = Buffer.mmap_file(path)
    var n = buf.mapped_size()
    return List[UInt8](buf.view[DType.uint8](0, n).as_span())


def test_ipc_file_reads_from_a_memory_source() raises:
    """The file reader over bytes that were never a file.

    This is what the `ByteSource` seam buys IPC: the same reader serves a
    memory map, a heap buffer, and -- once the backend lands -- an object
    store, because it only ever asks for byte ranges.
    """
    with ScratchDir() as dir:
        var path = join(dir, "marrow_ipc_memory_source.arrow")
        var i1: DynArray = array([1, 2, 3, 4], int64)
        var s1: DynArray = array(["a", "b", "c", "d"])
        var b1 = record_batch([i1^, s1^], names=["i", "s"])
        var i2: DynArray = array([5, 6], int64)
        var s2: DynArray = array(["e", "f"])
        var b2 = record_batch([i2^, s2^], names=["i", "s"])
        write_ipc_file(path, [b1.copy(), b2.copy()])

        var r = RecordBatchFileReader(BufferSource(Span(_file_bytes(path))))
        assert_equal(r.num_record_batches(), 2)
        var got = r.read_all()
        assert_equal(len(got), 2)
        assert_true(got[0] == b1)
        assert_true(got[1] == b2)


def test_ipc_stream_reads_from_a_memory_source() raises:
    """Same for the stream reader, whose framing is sequential rather than
    indexed."""
    with ScratchDir() as dir:
        var path = join(dir, "marrow_ipc_memory_source_stream.arrow")
        var i3: DynArray = array([7, 8, 9], int64)
        var s3: DynArray = array(["g", "h", "i"])
        var b = record_batch([i3^, s3^], names=["i", "s"])
        write_ipc_stream(path, [b.copy()])

        var r = RecordBatchStreamReader(BufferSource(Span(_file_bytes(path))))
        var got = r.read_all()
        assert_equal(len(got), 1)
        assert_true(got[0] == b)


struct _CountingSource(ByteSource):
    """A `BufferSource` that remembers how many bytes were read through it.

    The count goes through an `ArcPointer` because `ByteSource.read_at` takes
    `ref self`, not `mut self` — the same reason
    `marrow/parquet/tests/test_page_io.mojo` does it that way.
    """

    var _inner: BufferSource
    var _read: ArcPointer[Int]

    def __init__(out self, data: Span[UInt8, _], var counter: ArcPointer[Int]):
        self._inner = BufferSource(data)
        self._read = counter^

    def size(self) -> Int:
        return self._inner.size()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        self._read[] += length
        return rebind[Span[UInt8, origin_of(self)]](
            self._inner.read_at(offset, length)
        )

    def read_ranges(
        ref self, ranges: List[Tuple[Int, Int]], ctx: ExecContext
    ) raises -> Fetched:
        for ref r in ranges:
            self._read[] += r[1]
        return self._inner.read_ranges(ranges, ctx)


def test_ipc_file_reader_reads_only_its_tail() raises:
    """Opening a file must not fetch the whole thing.

    The footer sits at the end, so the reader asks for a bounded tail and finds
    it there. Before the `ByteSource` seam this was a whole-file `mmap`, which
    costs nothing locally and would be a full download from an object store —
    the same reason `ParquetFile` opens by its tail. Counting the bytes read is
    the only way to tell "found the footer" from "read the file to find it".
    """
    with ScratchDir() as dir:
        var path = join(dir, "marrow_ipc_tail.arrow")
        var bld = Int64Builder()
        for i in range(50000):
            bld.append(Int64(i))
        var col: DynArray = bld.finish()
        var b = record_batch([col^], names=["i"])
        write_ipc_file(path, [b.copy()])

        var whole = _file_bytes(path)
        assert_true(
            len(whole) > 256 * 1024,
            "the fixture must be much bigger than one tail read",
        )

        var counter = ArcPointer[Int](0)
        var r = RecordBatchFileReader(_CountingSource(Span(whole), counter))
        assert_equal(r.num_record_batches(), 1)

        # Opening the file reads the leading magic, one bounded tail, and nothing
        # else -- not the 400 KB of column data sitting between them.
        assert_true(
            counter[] < len(whole) // 4,
            String("opening read ", counter[], " of ", len(whole), " bytes"),
        )

        # And the batch is still readable, so the cheap open did not skip anything
        # it needed.
        var got = r.read_batch(0)
        assert_equal(got.num_rows(), 50000)
        _ = whole^


# ---------------------------------------------------------------------------
# string_view / binary_view: variadicBufferCounts
# ---------------------------------------------------------------------------


def _view_batch() raises -> RecordBatch:
    """A string_view column over several data buffers, a binary_view column
    with none, and an int32 column after both -- so a wrong variadic count
    shifts the int32 column's buffers and shows up there."""
    var sb = StringViewBuilder()
    for i in range(400):
        if i % 9 == 0:
            sb.append_null()
        else:
            sb.append(String(i) + String("-") * (i % 50))
    var sv = sb.finish()
    assert_true(len(sv.buffers) > 1)
    var bb = BinaryViewBuilder()
    for i in range(400):
        bb.append(String(i))
    var ib = Int32Builder(400)
    for i in range(400):
        ib.append(Int32(i))
    var fields = List[Field]()
    fields.append(field("s", string_view, nullable=True))
    fields.append(field("b", binary_view))
    fields.append(field("i", int32))
    var cols = List[DynArray]()
    cols.append(sv^.to_dyn())
    cols.append(bb.finish().to_dyn())
    cols.append(ib.finish().to_dyn())
    return RecordBatch(schema=Schema(fields=fields^), columns=cols^)


def test_view_file_roundtrip() raises:
    var batch = _view_batch()
    var result = _roundtrip_file(batch)
    assert_true(result.schema.fields[0].dtype == string_view)
    assert_true(result.schema.fields[1].dtype == binary_view)
    assert_true(batch == result)


def test_view_stream_roundtrip() raises:
    var batch = _view_batch()
    assert_true(batch == _roundtrip_stream(batch))


def test_pyarrow_reads_view_file() raises:
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()
    var batches_in = List[RecordBatch]()
    batches_in.append(_view_batch())
    write_ipc_file(path, batches_in)
    var pa_batch = pa.ipc.open_file(path).get_batch(0)
    pa_batch.validate(full=True)
    assert_true(pa_batch.schema.field(0).type == pa.string_view())
    assert_true(pa_batch.schema.field(1).type == pa.binary_view())
    var s = pa_batch.column(0)
    assert_equal(Int(py=s.null_count), 45)
    assert_true(s[0].as_py() is None)
    assert_equal(String(py=s[1].as_py()), "1-")
    assert_equal(String(py=s[49].as_py()), "49" + String("-") * 49)
    assert_equal(Int(py=pa_batch.column(2)[399].as_py()), 399)


def test_marrow_reads_pyarrow_view_file() raises:
    var pa = Python.import_module("pyarrow")
    var path = _tmp_path()
    var sv = pa.array(
        Python.list("a", None, "a value longer than twelve bytes"),
        type=pa.string_view(),
    )
    var bv = pa.array(
        Python.evaluate("[b'x', b'y', b'bytes beyond the inline limit']"),
        type=pa.binary_view(),
    )
    var iv = pa.array(Python.list(1, 2, 3), type=pa.int32())
    var pa_batch = pa.RecordBatch.from_arrays(
        Python.list(sv, bv, iv), names=Python.list("s", "b", "i")
    )
    var writer = pa.ipc.new_file(path, pa_batch.schema)
    writer.write(pa_batch)
    writer.close()

    var batches = read_ipc_file(path)
    ref cols = batches[0].columns
    assert_true(cols[0].dtype() == string_view)
    assert_true(cols[1].dtype() == binary_view)
    ref s = cols[0].as_string_view()
    assert_equal(s.null_count(), 1)
    assert_equal(s[2].value(), "a value longer than twelve bytes")
    assert_equal(
        cols[1].as_binary_view()[2].value(),
        "bytes beyond the inline limit",
    )
    assert_equal(cols[2].as_int32()[2].value(), 3)


def test_view_sparse_column_is_compacted_on_write() raises:
    """A view column reaching a small part of its data buffers -- two rows of
    a hundred -- is written with only the bytes it reaches, so the body does
    not carry the rest of its source."""
    var sb = StringViewBuilder()
    for i in range(100):
        sb.append(String(i) + " is a value longer than twelve")
    var sparse = sb.finish().slice(40, 2)
    assert_true(sparse.is_sparse())
    var batch = _single_col_batch(
        sparse.copy().to_dyn(), field("s", string_view)
    )
    var result = _roundtrip_file(batch)
    ref got = result.columns[0].as_string_view()
    assert_false(got.is_sparse())
    assert_equal(got[0].value(), "40 is a value longer than twelve")
    assert_equal(got[1].value(), "41 is a value longer than twelve")


def test_view_sliced_roundtrip() raises:
    """A sliced view column is written dense: its views are gathered to start
    at 0, and the data buffers go with them."""
    var sb = StringViewBuilder()
    for i in range(20):
        sb.append(String(i) + " is a value longer than twelve")
    var sliced = sb.finish().slice(5, 10)
    var expected = sliced.copy()
    var batch = _single_col_batch(sliced^.to_dyn(), field("s", string_view))
    var result = _roundtrip_file(batch)
    ref got = result.columns[0].as_string_view()
    assert_equal(len(got), 10)
    for i in range(10):
        assert_equal(
            got[i].value(),
            expected[i].value(),
        )


# ---------------------------------------------------------------------------
# Malformed streams: a valid stream with one field corrupted
# ---------------------------------------------------------------------------
#
# Each test writes a stream, walks its flatbuffer metadata to the one field the
# reader must not trust, and overwrites it. Positions are offsets into the
# whole stream; a flatbuffer offset is relative to where it is stored, so
# following one needs no base.


def _stream_bytes(batch: RecordBatch) raises -> List[UInt8]:
    var out = List[UInt8]()
    with ScratchDir() as dir:
        var path = join(dir, "malformed.arrows")
        write_ipc_stream(path, [batch.copy()])
        out = _file_bytes(path)
    return out^


def _u16(b: List[UInt8], pos: Int) -> Int:
    return Int(b[pos]) | Int(b[pos + 1]) << 8


def _u32(b: List[UInt8], pos: Int) -> Int:
    return _u16(b, pos) | _u16(b, pos + 2) << 16


def _put(mut b: List[UInt8], pos: Int, value: Int, width: Int = 4):
    """Store `value` little-endian in `width` bytes at `pos`."""
    for i in range(width):
        b[pos + i] = UInt8((value >> (8 * i)) & 0xFF)


def _follow(b: List[UInt8], pos: Int) -> Int:
    """Where the offset stored at `pos` points: a table, vector or string."""
    return pos + _u32(b, pos)


def _vtable(b: List[UInt8], table: Int) -> Int:
    """Where the vtable of the table at `table` is: a signed offset back."""
    var soffset = _u32(b, table)
    if soffset >= 1 << 31:
        soffset -= 1 << 32
    return table - soffset


def _slot(b: List[UInt8], table: Int, slot: Int) raises -> Int:
    """The position of field `slot` of the table at `table`."""
    var vtable = _vtable(b, table)
    var at = 4 + 2 * slot
    var voffset = _u16(b, vtable + at) if at < _u16(b, vtable) else 0
    assert_true(voffset != 0, String(t"slot {slot} is absent"))
    return table + voffset


def _entry(b: List[UInt8], vector: Int, i: Int) -> Int:
    """The table at entry `i` of the vector of offsets at `vector`."""
    return _follow(b, vector + 4 + 4 * i)


def _message(b: List[UInt8], index: Int) raises -> Int:
    """Where the stream's `index`-th message starts: its continuation marker,
    metadata length, metadata, then body."""
    var pos = 0
    for _ in range(index):
        var message = _follow(b, pos + 8)
        pos += 8 + _u32(b, pos + 4) + _u32(b, _slot(b, message, 3))
    return pos


def _header(b: List[UInt8], index: Int) raises -> Int:
    """The header table -- Schema, DictionaryBatch or RecordBatch -- of the
    stream's `index`-th message."""
    return _follow(b, _slot(b, _follow(b, _message(b, index) + 8), 2))


def _body(b: List[UInt8], index: Int) raises -> Int:
    """Where the body of the stream's `index`-th message starts."""
    var pos = _message(b, index)
    return pos + 8 + _u32(b, pos + 4)


def _buffer(b: List[UInt8], index: Int, i: Int) raises -> Int:
    """The `i`-th Buffer struct -- offset into the body, then length -- of
    the stream's `index`-th message, a record or dictionary batch."""
    return _follow(b, _slot(b, _header(b, index), 2)) + 4 + 16 * i


def _schema_field(b: List[UInt8], i: Int) raises -> Int:
    """The Field table of the schema's `i`-th top-level field."""
    return _entry(b, _follow(b, _slot(b, _header(b, 0), 1)), i)


def _child_field(b: List[UInt8], field: Int, i: Int) raises -> Int:
    """The Field table of a Field's `i`-th child."""
    return _entry(b, _follow(b, _slot(b, field, 5)), i)


def _read_every_value(b: List[UInt8]) raises:
    """Read the stream and format every column, as the fuzz harness does."""
    var reader = RecordBatchStreamReader(BufferSource(Span(b)))
    for batch in reader.read_all():
        for i in range(batch.num_columns()):
            _ = String(batch.column(i))


def _refused[E: ArrowError](b: List[UInt8]) -> Bool:
    """Whether reading the stream raises an `E`."""
    try:
        _read_every_value(b)
    except e:
        return DynError(e).isa[E]()
    return False


def test_ipc_refuses_batch_with_too_few_field_nodes() raises:
    """A batch carrying fewer field nodes than its schema has fields is
    refused rather than indexed past its node list."""
    var b = _stream_bytes(_mk_batch())
    _read_every_value(b)
    var nodes = _follow(b, _slot(b, _header(b, 1), 1))
    assert_equal(_u32(b, nodes), 2)
    _put(b, nodes, 1)
    assert_true(_refused[CorruptError](b))


def test_ipc_refuses_batch_with_too_few_buffers() raises:
    """Likewise a batch carrying fewer buffers than its fields own."""
    var b = _stream_bytes(_mk_batch())
    var buffers = _follow(b, _slot(b, _header(b, 1), 2))
    assert_equal(_u32(b, buffers), 4)  # validity and values, per column
    _put(b, buffers, 3)
    assert_true(_refused[CorruptError](b))


def _metadata_batch() raises -> RecordBatch:
    """A column `point: struct<x: int32>`, with metadata on the schema and on
    the field."""
    var children: List[Field] = [field("x", int32)]
    var sb = StructBuilder(children.copy(), capacity=1)
    sb.field_builder(0).as_int32().append(Int32(1))
    sb.append_valid()
    var point = field("point", struct_(children^))
    point.metadata["unit"] = "cm"
    var columns: List[DynArray] = [sb.finish()]
    var schema = Schema(fields=[point^], metadata={"origin": "test"})
    return RecordBatch(schema=schema^, columns=columns^)


def test_ipc_refuses_strings_not_utf8() raises:
    """Every flatbuffer string is checked as UTF-8 -- a field's name, a
    child's, and metadata on the schema and on a field -- and each one that
    is not is refused, rather than aborting or being dropped."""
    var valid = _stream_bytes(_metadata_batch())
    _read_every_value(valid)
    var point = _schema_field(valid, 0)
    var schema_kv = _entry(
        valid, _follow(valid, _slot(valid, _header(valid, 0), 2)), 0
    )
    var field_kv = _entry(valid, _follow(valid, _slot(valid, point, 6)), 0)
    var names: List[String] = [
        "field name",
        "child name",
        "schema metadata value",
        "field metadata key",
    ]
    var strings: List[Int] = [
        _follow(valid, _slot(valid, point, 0)),
        _follow(valid, _slot(valid, _child_field(valid, point, 0), 0)),
        _follow(valid, _slot(valid, schema_kv, 1)),
        _follow(valid, _slot(valid, field_kv, 0)),
    ]
    var accepted = List[String]()
    for i in range(len(strings)):
        var b = valid.copy()
        b[strings[i] + 4] = 0xFF  # never valid in UTF-8
        if not _refused[CorruptError](b):
            accepted.append(names[i])
    assert_true(len(accepted) == 0, String(", ").join(accepted))


def _empty_batch(dtype: DynType) raises -> RecordBatch:
    """A zero-row batch with one column `a` of `dtype`."""
    var builder = DynBuilder(dtype)
    return _single_col_batch(builder.finish(), field("a", dtype.copy()))


def _nested_lists(levels: Int) -> DynType:
    """A type `levels` fields deep: lists down to an int32."""
    var dtype: DynType = int32
    for _ in range(levels - 1):
        dtype = list_(dtype^).to_dyn()
    return dtype^


def test_ipc_nesting_limit() raises:
    """Fields nest 64 deep, as Arrow C++ allows, and no deeper."""
    _read_every_value(_stream_bytes(_empty_batch(_nested_lists(64))))
    var b = _stream_bytes(_empty_batch(_nested_lists(65)))
    assert_true(_refused[CorruptError](b))


def test_ipc_refuses_fields_that_contain_themselves() raises:
    """A child offset pointing back at its own field makes a cycle, not a
    tree. It is refused at the nesting limit, for a list and for a struct,
    rather than recursing until the stack overflows."""
    var types: List[DynType] = [
        list_(int32).to_dyn(),
        struct_([field("x", int32)]).to_dyn(),
    ]
    for dtype in types:
        var b = _stream_bytes(_empty_batch(dtype))
        _read_every_value(b)
        var a = _schema_field(b, 0)
        var entry = _follow(b, _slot(b, a, 5)) + 4
        _put(b, entry, (a - entry) & 0xFFFFFFFF)  # offsets wrap at 32 bits
        assert_true(_refused[CorruptError](b), String(dtype))


def _dictionary_batch() raises -> RecordBatch:
    """Columns `a: dictionary<int32, string>` and `b: dictionary<int32,
    int64>`, written as dictionaries 0 and 1."""
    var words: DynArray = array(["x", "y"])
    var numbers: DynArray = array([7, 8], int64)
    var a: DynArray = DictionaryArray.from_arrays(array([1, 0], int32), words^)
    var b: DynArray = DictionaryArray.from_arrays(
        array([0, 1], int32), numbers^
    )
    return record_batch([a^, b^], names=["a", "b"])


def _dictionary_id(b: List[UInt8], field: Int) raises -> Int:
    """The position of a Field's dictionary id."""
    return _slot(b, _follow(b, _slot(b, field, 4)), 0)


def test_ipc_refuses_fields_sharing_a_dictionary_of_another_type() raises:
    """Two fields naming one dictionary id must agree on its value type; a
    column is never built over values of a type its field does not declare.
    """
    var b = _stream_bytes(_dictionary_batch())
    _read_every_value(b)
    _put(b, _dictionary_id(b, _schema_field(b, 1)), 0, width=8)
    assert_true(_refused[CorruptError](b))


def test_ipc_dictionary_batch_naming_no_dictionary() raises:
    """A dictionary batch whose id is -1 -- what every field that is not a
    dictionary carries internally -- matches no field. The column whose
    dictionary it replaced then has none, which is refused."""
    var b = _stream_bytes(_dictionary_batch())
    for message in [1, 2]:
        _put(b, _slot(b, _header(b, message), 0), -1, width=8)
    assert_true(_refused[CorruptError](b))


def test_ipc_refuses_negative_dictionary_ids() raises:
    """The readers keep dictionaries by id, so a negative one is refused
    when the schema is read rather than used as an index."""
    var b = _stream_bytes(_dictionary_batch())
    _put(b, _dictionary_id(b, _schema_field(b, 0)), -2, width=8)
    _put(b, _dictionary_id(b, _schema_field(b, 1)), -3, width=8)
    for message in [1, 2]:
        var id = _slot(b, _header(b, message), 0)
        _put(b, id, -2 - _u32(b, id), width=8)  # 0 -> -2, 1 -> -3
    assert_true(_refused[NotImplementedError](b))


def test_ipc_dictionary_without_index_type_reads_int32() raises:
    """A DictionaryEncoding may omit its index type, which then means signed
    32-bit indices; the field is still a dictionary."""
    var b = _stream_bytes(_dictionary_batch())
    var encoding = _follow(b, _slot(b, _schema_field(b, 0), 4))
    _put(b, _vtable(b, encoding) + 6, 0, width=2)  # slot 1, indexType: absent
    var reader = RecordBatchStreamReader(BufferSource(Span(b)))
    var batches = reader.read_all()
    assert_true(
        batches[0].schema.fields[0].dtype == dictionary(int32, string).to_dyn()
    )


def test_ipc_refuses_arrays_their_buffers_cannot_hold() raises:
    """Every array the reader builds is validated against its buffers before
    anything reads it. Each corruption below is refused; none is read past.
    """
    var accepted = List[String]()

    var ints = Int32Builder()
    for i in range(100):
        if i % 10 == 0:
            ints.append_null()
        else:
            ints.append(Int32(i))
    var valid = _stream_bytes(
        _single_col_batch(ints.finish(), field("a", int32))
    )
    _read_every_value(valid)
    var b = valid.copy()
    _put(b, _buffer(b, 1, 1) + 8, 4, width=8)
    if not _refused[CorruptError](b):
        accepted.append("values buffer of 4 bytes for 100 int32s")
    b = valid.copy()
    _put(b, _buffer(b, 1, 0) + 8, 0, width=8)
    if not _refused[CorruptError](b):
        accepted.append("nulls without a validity bitmap")
    b = valid.copy()
    _put(b, _slot(b, _header(b, 1), 0), 101, width=8)
    if not _refused[CorruptError](b):
        accepted.append("a column shorter than its batch")

    var strings: DynArray = array(["ab", "c", "d"])
    valid = _stream_bytes(_single_col_batch(strings, field("s", string)))
    _read_every_value(valid)
    var offsets = _body(valid, 1) + _u32(valid, _buffer(valid, 1, 1))
    assert_equal(_u32(valid, offsets + 12), 4)  # offsets 0, 2, 3, 4
    b = valid.copy()
    _put(b, offsets + 12, 1000)
    if not _refused[CorruptError](b):
        accepted.append("an offset past the string data")
    b = valid.copy()
    _put(b, offsets + 8, 1)
    if not _refused[CorruptError](b):
        accepted.append("decreasing offsets")

    valid = _stream_bytes(_dictionary_batch())
    _read_every_value(valid)
    b = valid.copy()
    _put(b, _body(b, 3) + _u32(b, _buffer(b, 3, 1)), 2)
    if not _refused[CorruptError](b):
        accepted.append("a dictionary index past its two values")

    assert_true(len(accepted) == 0, String(", ").join(accepted))


def test_ipc_reads_pyarrow_slices_and_empty_batches() raises:
    """Validation refuses nothing pyarrow writes: a batch sliced from the
    middle of a table and an empty one, with large and view columns beside
    nested, dictionary and null-bearing ones, compressed or not."""
    var pa = Python.import_module("pyarrow")
    var pc = Python.import_module("pyarrow.compute")
    var table = _compressed_table()
    var words = table.column("words")
    table = table.append_column("large", pc.cast(words, pa.large_string()))
    table = table.append_column("view", pc.cast(words, pa.string_view()))
    table = table.append_column(
        "large_lists", pc.cast(table.column("lists"), pa.large_list(pa.int64()))
    )
    var whole = table.combine_chunks().to_batches()[0]
    for codec in [Python.none(), PythonObject("zstd")]:
        for part in [whole.slice(1001, 2500), whole.slice(17, 0)]:
            var path = _tmp_path(suffix=".arrows")
            var opts = pa.ipc.IpcWriteOptions(compression=codec)
            var writer = pa.ipc.new_stream(path, part.schema, options=opts)
            writer.write_batch(part)
            writer.close()
            var got = read_ipc_stream(path)
            assert_equal(len(got), 1)
            assert_equal(got[0].num_rows(), Int(py=part.num_rows))
            var want = pa.Table.from_batches(Python.list(part))
            assert_true(Bool(_pyarrow_of(got^).equals(want)), String(codec))


def test_ipc_refuses_map_entries_that_are_not_key_value_structs() raises:
    """A map's one child must be a struct of a key and a value; a Field that
    says otherwise is refused when the schema is read."""
    var valid = _stream_bytes(_empty_batch(map_(string, int32).to_dyn()))
    _read_every_value(valid)
    var entries = _child_field(valid, _schema_field(valid, 0), 0)
    var accepted = List[String]()
    var b = valid.copy()
    assert_equal(Int(b[_slot(b, entries, 2)]), 13)  # Type.Struct_
    b[_slot(b, entries, 2)] = 2  # Type.Int
    if not _refused[CorruptError](b):
        accepted.append("int entries")
    b = valid.copy()
    var children = _follow(b, _slot(b, entries, 5))
    assert_equal(_u32(b, children), 2)
    _put(b, children, 1)
    if not _refused[CorruptError](b):
        accepted.append("entries without a value")
    assert_true(len(accepted) == 0, String(", ").join(accepted))
