from std.testing import assert_equal, assert_true
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


def test_read_narrow_ints() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i8=pa.array(Python.list(-1, 2, -3), type=pa.int8()),
            i16=pa.array(Python.list(1000, -2000, 3000), type=pa.int16()),
            u8=pa.array(Python.list(1, 200, 3), type=pa.uint8()),
            u16=pa.array(Python.list(1, 2, 60000), type=pa.uint16()),
        )
    )
    var path = String("/tmp/marrow_types_int.parquet")
    pq.write_table(tbl, path, compression="snappy")

    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var c0 = b.columns[0].copy()
    assert_equal(c0.as_int8()[0].value(), -1)
    assert_equal(c0.as_int8()[2].value(), -3)
    var c1 = b.columns[1].copy()
    assert_equal(c1.as_int16()[1].value(), -2000)
    var c3 = b.columns[3].copy()
    assert_equal(c3.as_uint16()[2].value(), 60000)
    remove(path)


def test_read_temporal() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var dt = Python.import_module("datetime")
    var tbl = pa.table(
        Python.dict(
            d=pa.array(
                Python.list(dt.date(2020, 1, 1), dt.date(2021, 6, 15)),
                type=pa.date32(),
            ),
            ts_us=pa.array(Python.list(1, 2), type=pa.timestamp("us")),
            ts_ns=pa.array(Python.list(10, 20), type=pa.timestamp("ns")),
            tm=pa.array(Python.list(5, 6), type=pa.time64("us")),
        )
    )
    var path = String("/tmp/marrow_types_time.parquet")
    pq.write_table(tbl, path, compression="none")

    var t = read_table(path)
    var names = t.column_names()
    assert_equal(names[0], "d")
    # dtypes should be temporal, not raw int
    assert_true(t.schema.field(index=1).dtype.is_timestamp())
    assert_true(t.schema.field(index=2).dtype.is_timestamp())
    assert_true(t.schema.field(index=3).dtype.is_time64())

    var b = t.to_batches()[0].copy()
    var cts = b.columns[2].copy()  # ts_ns stored as int64
    assert_equal(cts.as_timestamp()[0].value(), 10)
    assert_equal(cts.as_timestamp()[1].value(), 20)
    remove(path)


def test_read_binary() raises:
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = Python.evaluate(
        "__import__('pyarrow').table({'b':"
        " __import__('pyarrow').array([b'\\x00\\x01', b'\\xff\\xfe', b'ab'],"
        " type=__import__('pyarrow').binary())})"
    )
    var path = String("/tmp/marrow_types_bin.parquet")
    pq.write_table(tbl, path, compression="snappy")

    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var col = b.columns[0].copy()
    assert_true(t.schema.field(index=0).dtype.is_binary())
    assert_equal(col.length(), 3)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
