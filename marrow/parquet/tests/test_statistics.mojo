"""Reading column statistics: `read_metadata` (raw footer) and `read_statistics`
(decoded typed min/max). Marrow reads the bounds PyArrow writes, and round-trips
its own — the write side is covered against PyArrow in test_writer.mojo."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_metadata, read_statistics, write_table
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _write_pa(code: String, use_dictionary: Bool = True) raises -> String:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_readstats.parquet")
    pq.write_table(
        Python.evaluate(code),
        path,
        compression="none",
        use_dictionary=use_dictionary,
    )
    return path


def test_read_metadata_shape() raises:
    var path = _write_pa(
        "__import__('pyarrow').table({'a':"
        " __import__('pyarrow').array(list(range(2500)),"
        " type=__import__('pyarrow').int64())})"
    )
    var pq = Python.import_module("pyarrow.parquet")
    var meta = read_metadata(path)
    # matches pyarrow's own view of num_rows and column count
    assert_equal(meta.num_rows, 2500)
    assert_equal(len(meta.row_groups[0].columns), 1)
    # per-column-chunk null_count is surfaced without decoding data
    ref cm = meta.row_groups[0].columns[0].meta_data
    assert_equal(cm.null_count, 0)
    assert_true(cm.has_min_max)
    remove(path)


def test_read_int_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([5, -1, None, 9, -3],"
            " type=__import__('pyarrow').int64())}, )"
        ),
        use_dictionary=False,
    )
    var st = read_statistics(path)
    assert_equal(len(st), 1)  # one row group
    ref cs = st[0][0]
    assert_true(cs.has_min_max)
    assert_equal(cs.null_count, 1)
    assert_equal(cs.min.as_int64().value(), -3)
    assert_equal(cs.max.as_int64().value(), 9)
    remove(path)


def test_read_uint_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([1, 3000000000, 2],"
            " type=__import__('pyarrow').uint32())})"
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_equal(cs.min.as_uint32().value(), UInt32(1))
    assert_equal(
        cs.max.as_uint32().value(), UInt32(3000000000)
    )  # unsigned order
    remove(path)


def test_read_float_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'x':"
            " __import__('pyarrow').array([1.5, -2.5, 3.25, None],"
            " type=__import__('pyarrow').float64())})"
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(cs.min.as_float64().value() == -2.5)
    assert_true(cs.max.as_float64().value() == 3.25)
    remove(path)


def test_read_string_minmax() raises:
    var path = _write_pa(
        (
            "__import__('pyarrow').table({'s':"
            " __import__('pyarrow').array(['banana', 'apple', 'cherry',"
            " None])})"
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(cs.has_min_max)
    assert_equal(cs.min.as_string().to_string(), "apple")
    assert_equal(cs.max.as_string().to_string(), "cherry")
    remove(path)


def test_read_stats_multiple_row_groups() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.evaluate("list(range(2500))"), type=pa.int64())
        )
    )
    var path = String("/tmp/marrow_readstats_rg.parquet")
    pq.write_table(
        tbl, path, row_group_size=1000, use_dictionary=False, compression="none"
    )
    var st = read_statistics(path)
    assert_equal(len(st), 3)  # 3 row groups
    # first row group covers rows [0, 1000)
    assert_equal(st[0][0].min.as_int64().value(), 0)
    assert_equal(st[0][0].max.as_int64().value(), 999)
    # last row group covers [2000, 2500)
    assert_equal(st[2][0].min.as_int64().value(), 2000)
    assert_equal(st[2][0].max.as_int64().value(), 2499)
    remove(path)


def test_roundtrip_own_stats() raises:
    # marrow writes -> marrow reads its own min/max back (closes the loop with
    # the PyArrow-oracle write tests in test_writer.mojo)
    var caps = Python.evaluate(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([7, 2, None, 11],"
        " type=__import__('pyarrow').int64()), 's':"
        " __import__('pyarrow').array(['m', 'a', 'z', None])})"
    ).__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_stats_rt.parquet")
    write_table(t, path)

    var st = read_statistics(path)
    assert_equal(st[0][0].min.as_int64().value(), 2)
    assert_equal(st[0][0].max.as_int64().value(), 11)
    assert_equal(st[0][0].null_count, 1)
    assert_equal(st[0][1].min.as_string().to_string(), "a")
    assert_equal(st[0][1].max.as_string().to_string(), "z")
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
