"""Nested type reconstruction: (nullable) structs and arbitrarily nested lists.

Each case gives a plain Python data literal plus the Arrow type built with the
`pyarrow` module object — Marrow's read is compared to PyArrow's own read of the
same file.
"""

from std.testing import assert_true
from std.python import Python, PythonObject
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _to_pa(var t: Table) raises -> PythonObject:
    var caps = CArrowArrayStream.from_batches(
        t.schema.copy(), t.to_batches()
    ).to_pycapsule()
    var pa = Python.import_module("pyarrow")
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _check(
    data: String,
    dtype: PythonObject,
    compression: String,
    encoding: String = "",
) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = pa.table(
        Python.dict(v=pa.array(Python.evaluate(data), type=dtype))
    )
    var path = String("/tmp/marrow_nested.parquet")
    if encoding != "":
        pq.write_table(
            want,
            path,
            compression=compression,
            use_dictionary=False,
            column_encoding=encoding,
        )
    else:
        pq.write_table(want, path, compression=compression)
    # oracle is PyArrow's own read of the same file
    var got = _to_pa(read_table(path))
    assert_true(
        Bool(
            got.column(0).to_pylist()
            == pq.read_table(path).column(0).to_pylist()
        ),
        "value mismatch",
    )
    remove(path)


def _struct(*fields: PythonObject) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    var fs = Python.list()
    for f in fields:
        fs.append(f)
    return pa.struct(fs)


def test_list_of_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        (
            "[[{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], [], None, [{'a': 3,"
            " 'b': 'z'}]]"
        ),
        pa.list_(
            _struct(pa.field("a", pa.int64()), pa.field("b", pa.string()))
        ),
        "none",
    )


def test_list_of_struct_snappy() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        "[[{'x': i, 'y': i * 2}] * (i % 3) for i in range(50)]",
        pa.list_(_struct(pa.field("x", pa.int32()), pa.field("y", pa.int64()))),
        "snappy",
    )


def test_nullable_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        (
            "[{'a': 1, 'b': 'x'}, None, {'a': None, 'b': 'z'}, {'a': 4, 'b':"
            " None}, None]"
        ),
        _struct(pa.field("a", pa.int64()), pa.field("b", pa.string())),
        "none",
    )


def test_nullable_struct_of_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        "[{'p': {'x': 1}}, None, {'p': None}, {'p': {'x': 4}}]",
        _struct(pa.field("p", _struct(pa.field("x", pa.int64())))),
        "snappy",
    )


def test_list_of_list() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        "[[[1, 2], [3]], [[4]], None, []]",
        pa.list_(pa.list_(pa.int64())),
        "none",
    )


def test_list_of_list_of_list() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        "[[[[1], [2, 3]]], [[[4]]], []]",
        pa.list_(pa.list_(pa.list_(pa.int64()))),
        "snappy",
    )


def test_list_of_list_of_struct() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        "[[[{'a': 1}], [{'a': 2}, {'a': 3}]], []]",
        pa.list_(pa.list_(_struct(pa.field("a", pa.int64())))),
        "none",
    )


def test_struct_with_list_child() raises:
    # nullable struct whose field is a list; includes a null struct row
    var pa = Python.import_module("pyarrow")
    _check(
        "[{'xs': [1, 2]}, {'xs': []}, None]",
        _struct(pa.field("xs", pa.list_(pa.int64()))),
        "none",
    )


def test_list_of_nullable_struct() raises:
    # struct nulls *inside* a list
    var pa = Python.import_module("pyarrow")
    _check(
        "[[{'a': 1}, None, {'a': 3}], [], None]",
        pa.list_(_struct(pa.field("a", pa.int64()))),
        "snappy",
    )


def test_list_of_bool() raises:
    # booleans inside a list — the nested path used to lack a bool decoder
    var pa = Python.import_module("pyarrow")
    _check(
        "[[True, False], [None, True], None, []]",
        pa.list_(pa.bool_()),
        "none",
    )


def test_list_of_delta_int() raises:
    # a DELTA_BINARY_PACKED-encoded list element — the nested path used to read
    # it as PLAIN and crash; it now shares the flat path's decoders
    var pa = Python.import_module("pyarrow")
    _check(
        "[[i, i * 2, i - 5] for i in range(40)]",
        pa.list_(pa.int64()),
        "snappy",
        "DELTA_BINARY_PACKED",
    )


def test_list_of_byte_stream_split_float() raises:
    var pa = Python.import_module("pyarrow")
    _check(
        "[[1.5, 2.5], [3.5], None]",
        pa.list_(pa.float64()),
        "none",
        "BYTE_STREAM_SPLIT",
    )


def main() raises:
    TestSuite.run[__functions_in_module()]()
