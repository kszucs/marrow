from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python
from std.os import remove
from marrow.parquet import read_table, write_table
from marrow.tabular import Table, RecordBatch
from marrow.arrays import AnyArray, PrimitiveArray, StringArray
from marrow.schema import Schema
from marrow.dtypes import Field, int32, int64, float64, string


def _make_table() raises -> Table:
    """Build a simple 3-column, 3-row Table via PyArrow roundtrip."""
    var pa = Python.import_module("pyarrow")
    var py_table = pa.table(
        Python.dict(
            x=pa.array(Python.list(1, 2, 3), type=pa.int64()),
            y=pa.array(Python.list(4.0, 5.0, 6.0), type=pa.float64()),
            z=pa.array(Python.list("a", "b", "c")),
        )
    )
    from marrow.c_data import CArrowArrayStream

    var capsule = py_table.__arrow_c_stream__(Python.none())
    return CArrowArrayStream.from_pycapsule(capsule).to_table()


def test_write_read_roundtrip() raises:
    var t = _make_table()
    var path = "/tmp/marrow_test_roundtrip.parquet"
    write_table(t, path)

    var t2 = read_table(path)
    assert_equal(t2.num_rows(), 3)
    assert_equal(t2.num_columns(), 3)

    var names = t2.column_names()
    assert_equal(names[0], "x")
    assert_equal(names[1], "y")
    assert_equal(names[2], "z")

    var batches = t2.to_batches()
    assert_true(len(batches) >= 1)
    var batch = batches[0].copy()

    var tmp_col_x = batch.columns[0].copy()
    ref col_x = tmp_col_x.as_int64()
    assert_equal(col_x[0], 1)
    assert_equal(col_x[2], 3)

    var tmp_col_z = batch.columns[2].copy()
    ref col_z = tmp_col_z.as_string()
    assert_equal(String(col_z[0]), "a")
    assert_equal(String(col_z[2]), "c")

    remove(path)


def test_read_pyarrow_written() raises:
    """Write with PyArrow, read with marrow."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")

    var pa_table = pa.table(
        Python.dict(
            a=Python.list(10, 20),
            b=Python.list("hello", "world"),
        )
    )
    var path = "/tmp/marrow_test_pa_written.parquet"
    pq.write_table(pa_table, path)

    var t = read_table(path)
    assert_equal(t.num_rows(), 2)
    assert_equal(t.num_columns(), 2)

    var names = t.column_names()
    assert_equal(names[0], "a")
    assert_equal(names[1], "b")

    var batches = t.to_batches()
    var batch = batches[0].copy()

    var tmp_col_a = batch.columns[0].copy()
    ref col_a = tmp_col_a.as_int64()
    assert_equal(col_a[0], 10)
    assert_equal(col_a[1], 20)

    var tmp_col_b = batch.columns[1].copy()
    ref col_b = tmp_col_b.as_string()
    assert_equal(String(col_b[0]), "hello")
    assert_equal(String(col_b[1]), "world")

    remove(path)


def test_write_readable_by_pyarrow() raises:
    """Write with marrow, read with PyArrow."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")

    var t = _make_table()
    var path = "/tmp/marrow_test_write_pa.parquet"
    write_table(t, path)

    var pa_table = pq.read_table(path)
    assert_equal(Int(py=pa_table.num_rows), 3)
    assert_equal(Int(py=pa_table.num_columns), 3)
    assert_equal(Int(py=pa_table.column("x").to_pylist()[0]), 1)
    assert_equal(String(py=pa_table.column("z").to_pylist()[0]), "a")

    remove(path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
