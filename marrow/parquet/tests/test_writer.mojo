from std.testing import assert_equal, assert_true
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table, write_table
from marrow.parquet.writer import FileWriter
from marrow.parquet.codecs import Compression
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _pa_table(code: String) raises -> Table:
    var caps = Python.evaluate(code).__arrow_c_stream__(Python.none())
    return CArrowArrayStream.from_pycapsule(caps).to_table()


def _v2_roundtrip(codec: Compression) raises:
    var pq = Python.import_module("pyarrow.parquet")
    var t = _pa_table(
        "__import__('pyarrow').table({'i':"
        " __import__('pyarrow').array([1, None, 3, None, 5],"
        " type=__import__('pyarrow').int64()), 's':"
        " __import__('pyarrow').array(['a', 'b', None, 'd', 'e'])})"
    )
    var path = String("/tmp/marrow_v2.parquet")
    var w = FileWriter(codec, version=2)
    w.write(t, path)

    # the file declares format version 2
    var pf = pq.ParquetFile(path)
    assert_equal(Int(py=pf.metadata.format_version[0]), 2)

    # PyArrow reads marrow's v2 pages (validates the DataPageV2 layout)
    var back = pq.read_table(path)
    assert_true(
        Bool(
            back.column(0).to_pylist()
            == Python.evaluate("[1, None, 3, None, 5]")
        )
    )
    assert_true(
        Bool(
            back.column(1).to_pylist()
            == Python.evaluate("['a', 'b', None, 'd', 'e']")
        )
    )

    # marrow round-trips (exercises the PAGE_DATA_V2 read branch)
    var mback = read_table(path)
    var b = mback.to_batches()[0].copy()
    assert_equal(b.columns[0].copy().as_int64().null_count(), 2)
    assert_equal(b.columns[0].copy().as_int64()[4].value(), 5)
    assert_equal(String(b.columns[1].copy().as_string()[0]), "a")
    remove(path)


def test_write_v2_snappy() raises:
    _v2_roundtrip(Compression.SNAPPY)


def test_write_v2_uncompressed() raises:
    _v2_roundtrip(Compression.UNCOMPRESSED)


def test_multiple_row_groups() raises:
    # 2500 rows, row_group_size 1000 -> 3 row groups
    var t = _pa_table(
        "__import__('pyarrow').table({'i':"
        " __import__('pyarrow').array(list(range(2500)),"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_rg.parquet")
    var w = FileWriter(Compression.SNAPPY)
    w.write(t, path, row_group_size=1000)

    # pyarrow sees 3 row groups
    var pq = Python.import_module("pyarrow.parquet")
    var pf = pq.ParquetFile(path)
    assert_equal(Int(py=pf.metadata.num_row_groups), 3)
    assert_equal(Int(py=pf.metadata.num_rows), 2500)

    # marrow reads all rows back
    var back = read_table(path)
    assert_equal(back.num_rows(), 2500)
    var b = back.to_batches()
    var total = 0
    for ref bat in b:
        total += bat.num_rows()
    assert_equal(total, 2500)
    remove(path)


def test_null_count_statistic() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var t = _pa_table(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([1, None, 3, None, 5],"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_stats.parquet")
    write_table(t, path)
    var pf = pq.ParquetFile(path)
    var stats = pf.metadata.row_group(0).column(0).statistics
    assert_equal(Int(py=stats.null_count), 2)
    remove(path)


def test_narrow_int_roundtrip() raises:
    var t = _pa_table(
        "__import__('pyarrow').table({'a': __import__('pyarrow').array([-1, 2,"
        " -3], type=__import__('pyarrow').int8()), 'b':"
        " __import__('pyarrow').array([10, 20, 30],"
        " type=__import__('pyarrow').uint16())})"
    )
    var path = String("/tmp/marrow_narrow.parquet")
    write_table(t, path)
    var back = read_table(path)
    var bat = back.to_batches()[0].copy()
    var ca = bat.columns[0].copy()
    assert_equal(ca.as_int8()[0].value(), -1)
    assert_equal(ca.as_int8()[2].value(), -3)
    var cb = bat.columns[1].copy()
    assert_equal(cb.as_uint16()[2].value(), 30)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
