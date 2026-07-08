from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


def test_read_list_int() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(
                Python.list(
                    Python.list(1, 2, 3),
                    Python.list(),
                    Python.list(4, 5),
                    Python.list(6),
                ),
                type=pa.list_(pa.int64()),
            ),
        )
    )
    var path = String("/tmp/marrow_list_int.parquet")
    pq.write_table(tbl, path, compression="snappy")

    var t = read_table(path)
    assert_equal(t.num_rows(), 4)
    var b = t.to_batches()[0].copy()
    var col = b.columns[0].copy()
    ref lst = col.as_list()
    # offsets: [0,3,3,5,6]
    assert_equal(lst[0].value().length(), 3)
    assert_equal(lst[1].value().length(), 0)  # empty list
    assert_equal(lst[2].value().length(), 2)
    var elem0 = lst[0].value()
    assert_equal(elem0.as_int64()[0].value(), 1)
    assert_equal(elem0.as_int64()[2].value(), 3)
    remove(path)


def test_read_list_string_with_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(
                Python.list(
                    Python.list("a", "bb"),
                    Python.none(),  # null list
                    Python.list("ccc"),
                ),
                type=pa.list_(pa.string()),
            ),
        )
    )
    var path = String("/tmp/marrow_list_str.parquet")
    pq.write_table(tbl, path, compression="none")

    var t = read_table(path)
    assert_equal(t.num_rows(), 3)
    var b = t.to_batches()[0].copy()
    var col = b.columns[0].copy()
    ref lst = col.as_list()
    assert_true(lst.is_valid(0))
    assert_false(lst.is_valid(1))  # null list
    assert_true(lst.is_valid(2))
    var e0 = lst[0].value()
    assert_equal(String(e0.as_string()[0]), "a")
    assert_equal(String(e0.as_string()[1]), "bb")
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
