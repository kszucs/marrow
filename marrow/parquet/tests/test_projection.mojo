"""Column projection on read: `read_table(path, columns=[...])`."""

from std.testing import assert_equal, assert_true, assert_raises
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


def _write(path: String, code: String) raises:
    var pq = Python.import_module("pyarrow.parquet")
    pq.write_table(Python.evaluate(code), path, compression="snappy")


def test_project_subset_and_reorder() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            a=pa.array(Python.list(1, 2, 3), type=pa.int64()),
            b=pa.array(Python.list(1.5, 2.5, 3.5), type=pa.float64()),
            c=pa.array(Python.list("x", "y", "z")),
        )
    )
    var path = String("/tmp/marrow_proj.parquet")
    pq.write_table(tbl, path, compression="snappy")

    # projected subset, reordered
    var cols: List[String] = ["c", "a"]
    var t = read_table(path, columns=cols^)
    assert_equal(t.num_columns(), 2)
    var names = t.column_names()
    assert_equal(names[0], "c")
    assert_equal(names[1], "a")

    var b = t.to_batches()[0].copy()
    assert_equal(String(b.columns[0].copy().as_string()[0]), "x")
    assert_equal(String(b.columns[0].copy().as_string()[2]), "z")
    assert_equal(b.columns[1].copy().as_int64()[0].value(), 1)
    assert_equal(b.columns[1].copy().as_int64()[2].value(), 3)
    remove(path)


def test_project_single_column() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            a=pa.array(Python.list(10, 20, 30, 40), type=pa.int64()),
            b=pa.array(Python.list(1, 2, 3, 4), type=pa.int32()),
        )
    )
    var path = String("/tmp/marrow_proj1.parquet")
    pq.write_table(tbl, path, compression="none")

    var cols: List[String] = ["b"]
    var t = read_table(path, columns=cols^)
    assert_equal(t.num_columns(), 1)
    assert_equal(t.num_rows(), 4)
    assert_equal(t.column_names()[0], "b")
    var b = t.to_batches()[0].copy()
    assert_equal(b.columns[0].copy().as_int32()[3].value(), 4)
    remove(path)


def test_project_struct_column() raises:
    # a struct column (multi-leaf node) projected alongside a flat one
    var t_src = Python.evaluate(
        "__import__('pyarrow').table({"
        "'s': __import__('pyarrow').array("
        "[{'x': 1, 'y': 'a'}, {'x': 2, 'y': 'b'}],"
        " type=__import__('pyarrow').struct("
        "[__import__('pyarrow').field('x', __import__('pyarrow').int64()),"
        " __import__('pyarrow').field('y', __import__('pyarrow').string())])),"
        "'n': __import__('pyarrow').array([7, 8],"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_proj_struct.parquet")
    var pq = Python.import_module("pyarrow.parquet")
    pq.write_table(t_src, path, compression="snappy")

    # project just the struct
    var cols: List[String] = ["s"]
    var t = read_table(path, columns=cols^)
    assert_equal(t.num_columns(), 1)
    assert_true(t.schema.field(index=0).dtype.is_struct())
    var b = t.to_batches()[0].copy()
    ref sa = b.columns[0].copy().as_struct()
    assert_equal(sa.children[0].as_int64()[1].value(), 2)
    remove(path)


def test_project_missing_column() raises:
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = Python.evaluate(
        "__import__('pyarrow').table({'a':"
        " __import__('pyarrow').array([1, 2],"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_proj_miss.parquet")
    pq.write_table(tbl, path)
    var cols: List[String] = ["nope"]
    with assert_raises():
        _ = read_table(path, columns=cols^)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
