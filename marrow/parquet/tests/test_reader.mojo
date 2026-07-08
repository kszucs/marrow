from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet.reader import read_table


def _write(path: String, code: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = Python.evaluate(code)
    pq.write_table(tbl, path, compression="snappy")


def test_read_flat_no_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, 2, 3, 4, 5), type=pa.int64()),
            f=pa.array(Python.list(1.5, 2.5, 3.5, 4.5, 5.5), type=pa.float64()),
            s=pa.array(
                Python.list("apple", "banana", "cherry", "date", "elder")
            ),
        )
    )
    var path = String("/tmp/marrow_rd_flat.parquet")
    pq.write_table(tbl, path, compression="snappy")

    var t = read_table(path)
    assert_equal(t.num_rows(), 5)
    assert_equal(t.num_columns(), 3)
    var b = t.to_batches()[0].copy()

    var ci = b.columns[0].copy()
    ref col_i = ci.as_int64()
    assert_equal(col_i[0].value(), 1)
    assert_equal(col_i[4].value(), 5)

    var cf = b.columns[1].copy()
    ref col_f = cf.as_float64()
    assert_true(col_f[0].value() == 1.5)
    assert_true(col_f[4].value() == 5.5)

    var cs = b.columns[2].copy()
    ref col_s = cs.as_string()
    assert_equal(String(col_s[0]), "apple")
    assert_equal(String(col_s[2]), "cherry")
    remove(path)


def test_read_with_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(
                Python.list(10, Python.none(), 30, Python.none(), 50),
                type=pa.int64(),
            ),
            s=pa.array(
                Python.list("x", Python.none(), "z", "w", Python.none())
            ),
        )
    )
    var path = String("/tmp/marrow_rd_nulls.parquet")
    pq.write_table(tbl, path, compression="none")

    var t = read_table(path)
    assert_equal(t.num_rows(), 5)
    var b = t.to_batches()[0].copy()

    var ci = b.columns[0].copy()
    ref col_i = ci.as_int64()
    assert_equal(col_i.null_count(), 2)
    assert_true(col_i.is_valid(0))
    assert_false(col_i.is_valid(1))
    assert_equal(col_i[0].value(), 10)
    assert_equal(col_i[2].value(), 30)

    var cs = b.columns[1].copy()
    ref col_s = cs.as_string()
    assert_equal(col_s.null_count(), 2)
    assert_true(col_s.is_valid(0))
    assert_false(col_s.is_valid(1))
    assert_equal(String(col_s[0]), "x")
    assert_equal(String(col_s[3]), "w")
    remove(path)


def test_read_zstd() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, 2, 3), type=pa.int32()),
            b=pa.array(Python.list(True, False, True), type=pa.bool_()),
        )
    )
    var path = String("/tmp/marrow_rd_zstd.parquet")
    pq.write_table(tbl, path, compression="zstd")

    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var ci = b.columns[0].copy()
    ref col_i = ci.as_int32()
    assert_equal(col_i[0].value(), 1)
    assert_equal(col_i[2].value(), 3)
    var cb = b.columns[1].copy()
    ref col_b = cb.as_bool()
    assert_true(col_b[0].value())
    assert_false(col_b[1].value())
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
