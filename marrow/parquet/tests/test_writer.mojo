from std.testing import assert_equal, assert_true, assert_false
from std.math import isnan, isinf
from std.python import Python, PythonObject
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


# ---------------------------------------------------------------------------
# min/max statistics — PyArrow reads the bounds marrow writes. Covers the
# logical orderings that matter: signed vs unsigned ints, IEEE floats (with
# negatives), and byte-wise string ordering. The writer also emits column_orders
# (TypeDefinedOrder), so PyArrow trusts the unsigned/byte-array bounds.
# ---------------------------------------------------------------------------


def _col_stats(path: String, col: Int) raises -> PythonObject:
    var pq = Python.import_module("pyarrow.parquet")
    return pq.ParquetFile(path).metadata.row_group(0).column(col).statistics


def test_minmax_signed_int() raises:
    var t = _pa_table(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([5, -1, None, 9, -3],"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_mm_int.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(Int(py=s.min), -3)
    assert_equal(Int(py=s.max), 9)
    assert_equal(Int(py=s.null_count), 1)
    remove(path)


def test_minmax_unsigned_int() raises:
    # unsigned ordering: 3e9 > 1 even though it is negative as signed int32
    var t = _pa_table(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([1, 3000000000, 2],"
        " type=__import__('pyarrow').uint32())})"
    )
    var path = String("/tmp/marrow_mm_uint.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(Int(py=s.min), 1)
    assert_equal(Int(py=s.max), 3000000000)
    remove(path)


def test_minmax_float_with_negatives() raises:
    var t = _pa_table(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([1.5, -2.5, 3.25, None],"
        " type=__import__('pyarrow').float64())})"
    )
    var path = String("/tmp/marrow_mm_float.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_true(Float64(py=s.min) == -2.5)
    assert_true(Float64(py=s.max) == 3.25)
    remove(path)


def test_minmax_string_lexicographic() raises:
    var t = _pa_table(
        "__import__('pyarrow').table({'s':"
        " __import__('pyarrow').array(['banana', 'apple', 'cherry', None])})"
    )
    var path = String("/tmp/marrow_mm_str.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_equal(String(py=s.min), "apple")
    assert_equal(String(py=s.max), "cherry")
    assert_equal(Int(py=s.null_count), 1)
    remove(path)


def test_minmax_absent_for_all_null() raises:
    # an all-null column carries null_count but no min/max bound
    var t = _pa_table(
        "__import__('pyarrow').table({'x':"
        " __import__('pyarrow').array([None, None, None],"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_mm_allnull.parquet")
    write_table(t, path)
    var s = _col_stats(path, 0)
    assert_false(Bool(py=s.has_min_max))
    assert_equal(Int(py=s.null_count), 3)
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


def test_write_empty_table() raises:
    # a zero-row table must write a valid file and read back empty
    var t = _pa_table(
        "__import__('pyarrow').table({'i': __import__('pyarrow').array([],"
        " type=__import__('pyarrow').int64()), 's':"
        " __import__('pyarrow').array([],"
        " type=__import__('pyarrow').string())})"
    )
    var path = String("/tmp/marrow_empty.parquet")
    write_table(t, path)

    # pyarrow reads it: 0 rows, schema preserved
    var pq = Python.import_module("pyarrow.parquet")
    var pf = pq.ParquetFile(path)
    assert_equal(Int(py=pf.metadata.num_rows), 0)

    var back = read_table(path)
    assert_equal(back.num_rows(), 0)
    assert_equal(back.num_columns(), 2)
    remove(path)


def test_write_all_null_roundtrip() raises:
    var t = _pa_table(
        "__import__('pyarrow').table({'x': __import__('pyarrow').array("
        "[None, None, None, None, None],"
        " type=__import__('pyarrow').int64())})"
    )
    var path = String("/tmp/marrow_allnull.parquet")
    write_table(t, path)
    var back = read_table(path)
    var c = back.to_batches()[0].copy().columns[0].copy()
    assert_equal(c.as_int64().null_count(), 5)

    # pyarrow agrees the column is entirely null
    var pq = Python.import_module("pyarrow.parquet")
    assert_equal(
        Int(py=pq.read_table(path).column(0).null_count),
        5,
    )
    remove(path)


def test_write_float_special_roundtrip() raises:
    # NaN / +Inf / -Inf must survive marrow write -> read byte-exact
    var t = _pa_table(
        "__import__('pyarrow').table({'f': __import__('pyarrow').array("
        "[float('nan'), float('inf'), float('-inf'), -0.0, 3.25],"
        " type=__import__('pyarrow').float64())})"
    )
    var path = String("/tmp/marrow_fspecial.parquet")
    write_table(t, path)
    var back = read_table(path)
    var c = back.to_batches()[0].copy().columns[0].copy()
    ref f = c.as_float64()
    assert_true(isnan(f[0].value()))
    assert_true(isinf(f[1].value()) and f[1].value() > 0)
    assert_true(isinf(f[2].value()) and f[2].value() < 0)
    assert_true(f[4].value() == 3.25)
    remove(path)


# ---------------------------------------------------------------------------
# Nested write — lists and maps (Dremel shredding). Oracle: PyArrow reads back
# marrow's written file and must match the original.
# ---------------------------------------------------------------------------


def _check_write(want: PythonObject, compression: Compression) raises:
    var pq = Python.import_module("pyarrow.parquet")
    var caps = want.__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_nested_write.parquet")
    write_table(t, path, compression)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "PyArrow read of marrow's write mismatched the original",
    )
    remove(path)


def _list_table(data: String, dtype: PythonObject) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    return pa.table(
        Python.dict(v=pa.array(Python.evaluate(data), type=dtype))
    )


def test_write_list_int() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table("[[1, 2, 3], [], None, [4, 5]]", pa.list_(pa.int64())),
        Compression.UNCOMPRESSED,
    )


def test_write_list_int_snappy() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            "[list(range(i % 6)) if i % 7 else None for i in range(80)]",
            pa.list_(pa.int32()),
        ),
        Compression.SNAPPY,
    )


def test_write_list_string() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            "[['a', 'bb'], [], None, ['ccc', None, 'd']]",
            pa.list_(pa.string()),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_list_of_list() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            "[[[1, 2], [3]], [], None, [[4], []]]",
            pa.list_(pa.list_(pa.int64())),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_map_string_int() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            "[{'a': 1, 'b': 2}, {}, None, {'c': 3}]",
            pa.map_(pa.string(), pa.int64()),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_map_nullable_values() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            "[{'a': 1, 'b': None}, {'c': 3}, None]",
            pa.map_(pa.string(), pa.int64()),
        ),
        Compression.SNAPPY,
    )


def test_write_map_int_key() raises:
    var pa = Python.import_module("pyarrow")
    _check_write(
        _list_table(
            "[{1: 'x', 2: 'y'}, None, {3: 'z'}]",
            pa.map_(pa.int32(), pa.string()),
        ),
        Compression.UNCOMPRESSED,
    )


def test_write_map_multichunk() raises:
    # A map Table with several chunks exercises combine_chunks -> concat ->
    # MapBuilder before the write.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var mt = pa.map_(pa.string(), pa.int64())
    var b1 = pa.RecordBatch.from_arrays(
        Python.list(pa.array(Python.evaluate("[{'a': 1}, {}]"), type=mt)),
        names=Python.list("v"),
    )
    var b2 = pa.RecordBatch.from_arrays(
        Python.list(pa.array(Python.evaluate("[None, {'b': 2, 'c': 3}]"), type=mt)),
        names=Python.list("v"),
    )
    var want = pa.Table.from_batches(Python.list(b1, b2))
    assert_true(Bool(want.column(0).num_chunks > 1))
    var t = CArrowArrayStream.from_pycapsule(
        want.__arrow_c_stream__(Python.none())
    ).to_table()
    var path = String("/tmp/marrow_map_multichunk.parquet")
    write_table(t, path, Compression.UNCOMPRESSED)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "multi-chunk map write mismatch",
    )
    remove(path)


def test_write_list_v2() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var want = _list_table("[[1, 2], [], None, [3]]", pa.list_(pa.int64()))
    var t = CArrowArrayStream.from_pycapsule(
        want.__arrow_c_stream__(Python.none())
    ).to_table()
    var path = String("/tmp/marrow_nested_write_v2.parquet")
    var w = FileWriter(Compression.UNCOMPRESSED, version=2)
    w.write(t, path)
    var back = pq.read_table(path)
    assert_true(
        Bool(back.column(0).to_pylist() == want.column(0).to_pylist()),
        "v2 nested write mismatch",
    )
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
