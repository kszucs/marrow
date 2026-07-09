"""Nested type reconstruction beyond single-level lists: list<struct>, etc."""

from std.testing import assert_true
from std.python import Python, PythonObject
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _to_pa(var t: Table) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    var caps = CArrowArrayStream.from_batches(
        t.schema.copy(), t.to_batches()
    ).to_pycapsule()
    return pa.RecordBatchReader._import_from_c_capsule(caps).read_all()


def _assert_reads(expr: String, compression: String) raises:
    var pq = Python.import_module("pyarrow.parquet")
    var want = Python.evaluate(expr)
    var path = String("/tmp/marrow_nested.parquet")
    pq.write_table(want, path, compression=compression)
    var got = _to_pa(read_table(path))
    # oracle is PyArrow's own read of the same file
    var ref_ = pq.read_table(path)
    for i in range(Int(py=want.num_columns)):
        assert_true(
            Bool(got.column(i).to_pylist() == ref_.column(i).to_pylist()),
            "column mismatch",
        )
    remove(path)


def test_list_of_struct() raises:
    # includes an empty list and a null list alongside populated ones
    _assert_reads(
        (
            "__import__('pyarrow').table({'ls': __import__('pyarrow').array("
            "[[{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], [], None,"
            " [{'a': 3, 'b': 'z'}]],"
            " type=__import__('pyarrow').list_(__import__('pyarrow').struct("
            "[__import__('pyarrow').field('a', __import__('pyarrow').int64()),"
            " __import__('pyarrow').field('b',"
            " __import__('pyarrow').string())])))})"
        ),
        "none",
    )


def test_list_of_struct_snappy() raises:
    _assert_reads(
        (
            "__import__('pyarrow').table({'ls': __import__('pyarrow').array("
            "[[{'x': i, 'y': i * 2}] * (i % 3) for i in range(50)],"
            " type=__import__('pyarrow').list_(__import__('pyarrow').struct("
            "[__import__('pyarrow').field('x', __import__('pyarrow').int32()),"
            " __import__('pyarrow').field('y',"
            " __import__('pyarrow').int64())])))})"
        ),
        "snappy",
    )


def main() raises:
    TestSuite.run[__functions_in_module()]()
