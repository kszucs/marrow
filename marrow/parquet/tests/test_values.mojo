"""Value and boundary edge cases (read), modelled on the pyarrow parquet test
suite: all-null columns, integer/float extremes, NaN/Inf, empty/unicode/long
strings, booleans with nulls, dictionary-encoded reads, wide tables, nulls
spanning many row groups, and pre-epoch dates. PyArrow writes; marrow reads."""

from std.testing import assert_equal, assert_true, assert_false
from std.math import isnan, isinf
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table
from marrow.tabular import Table


def _read(code: String, compression: String = "snappy") raises -> Table:
    """Write a one-line pyarrow table expression to a file and read it back."""
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_values.parquet")
    pq.write_table(Python.evaluate(code), path, compression=compression)
    var t = read_table(path)
    remove(path)
    return t^


# ---------------------------------------------------------------------------
# All-null and single-row columns
# ---------------------------------------------------------------------------


def test_all_null_int64() raises:
    var t = _read(
        "__import__('pyarrow').table({'x': __import__('pyarrow').array("
        "[None, None, None, None], type=__import__('pyarrow').int64())})"
    )
    assert_equal(t.num_rows(), 4)
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref a = c.as_int64()
    assert_equal(a.null_count(), 4)
    assert_false(a.is_valid(0))


def test_all_null_string() raises:
    var t = _read(
        "__import__('pyarrow').table({'s': __import__('pyarrow').array("
        "[None, None, None], type=__import__('pyarrow').string())})"
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    assert_equal(c.as_string().null_count(), 3)


def test_single_row() raises:
    var t = _read(
        "__import__('pyarrow').table({'i': __import__('pyarrow').array([42],"
        " type=__import__('pyarrow').int64()), 's':"
        " __import__('pyarrow').array(['solo'])})"
    )
    assert_equal(t.num_rows(), 1)
    var b = t.to_batches()[0].copy()
    var ci = b.columns[0].copy()
    var cs = b.columns[1].copy()
    assert_equal(ci.as_int64()[0].value(), 42)
    assert_equal(String(cs.as_string()[0]), "solo")


# ---------------------------------------------------------------------------
# Integer extremes
# ---------------------------------------------------------------------------


def test_int64_extremes() raises:
    var t = _read(
        "__import__('pyarrow').table({'x': __import__('pyarrow').array("
        "[-9223372036854775808, 9223372036854775807, 0],"
        " type=__import__('pyarrow').int64())})"
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref a = c.as_int64()
    assert_equal(a[0].value(), Int64(-9223372036854775808))
    assert_equal(a[1].value(), Int64(9223372036854775807))
    assert_equal(a[2].value(), Int64(0))


def test_uint64_max() raises:
    var t = _read(
        "__import__('pyarrow').table({'x': __import__('pyarrow').array("
        "[0, 18446744073709551615], type=__import__('pyarrow').uint64())})"
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref a = c.as_uint64()
    assert_equal(a[0].value(), UInt64(0))
    assert_equal(a[1].value(), UInt64(18446744073709551615))


def test_int32_extremes() raises:
    var t = _read(
        "__import__('pyarrow').table({'x': __import__('pyarrow').array("
        "[-2147483648, 2147483647], type=__import__('pyarrow').int32())})"
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref a = c.as_int32()
    assert_equal(a[0].value(), Int32(-2147483648))
    assert_equal(a[1].value(), Int32(2147483647))


# ---------------------------------------------------------------------------
# Special float values — NaN / Inf / -Inf must survive byte-exact
# ---------------------------------------------------------------------------


def test_float64_special() raises:
    var t = _read(
        (
            "__import__('pyarrow').table({'f': __import__('pyarrow').array("
            "[float('nan'), float('inf'), float('-inf'), 0.0, 1.5],"
            " type=__import__('pyarrow').float64())})"
        ),
        "none",
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref f = c.as_float64()
    assert_true(isnan(f[0].value()))
    assert_true(isinf(f[1].value()) and f[1].value() > 0)
    assert_true(isinf(f[2].value()) and f[2].value() < 0)
    assert_true(f[3].value() == 0.0)
    assert_true(f[4].value() == 1.5)


def test_float32_special() raises:
    var t = _read(
        (
            "__import__('pyarrow').table({'f': __import__('pyarrow').array("
            "[float('nan'), float('inf'), float('-inf'), -2.5],"
            " type=__import__('pyarrow').float32())})"
        ),
        "zstd",
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref f = c.as_float32()
    assert_true(isnan(f[0].value()))
    assert_true(isinf(f[1].value()) and f[1].value() > 0)
    assert_true(isinf(f[2].value()) and f[2].value() < 0)
    assert_true(f[3].value() == -2.5)


# ---------------------------------------------------------------------------
# Strings — empty, multibyte unicode, long
# ---------------------------------------------------------------------------


def test_string_edge_cases() raises:
    # empty, ascii, multibyte utf-8, a long value (> a page-ish size), null
    var t = _read(
        "__import__('pyarrow').table({'s': __import__('pyarrow').array("
        "['', 'a', 'h\\u00e9llo \\u4e16\\u754c', 'x' * 5000, None])})"
    )
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref s = c.as_string()
    assert_equal(s.null_count(), 1)
    assert_equal(String(s[0]), "")
    assert_equal(String(s[1]), "a")
    assert_equal(String(s[2]), "héllo 世界")
    assert_equal(String(s[3]).byte_length(), 5000)
    assert_false(s.is_valid(4))


# ---------------------------------------------------------------------------
# Boolean with nulls
# ---------------------------------------------------------------------------


def test_bool_with_nulls() raises:
    var t = _read(
        (
            "__import__('pyarrow').table({'b': __import__('pyarrow').array("
            "[True, None, False, None, True, False],"
            " type=__import__('pyarrow').bool_())})"
        ),
        "none",
    )
    var bat = t.to_batches()[0].copy()
    var c = bat.columns[0].copy()
    ref b = c.as_bool()
    assert_equal(b.null_count(), 2)
    assert_true(b[0].value())
    assert_false(b.is_valid(1))
    assert_false(b[2].value())
    assert_true(b[4].value())
    assert_false(b[5].value())


# ---------------------------------------------------------------------------
# Dictionary-encoded read (low cardinality; pyarrow's default)
# ---------------------------------------------------------------------------


def test_dictionary_encoded_read() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # 400 rows over 3 distinct strings -> pyarrow dictionary-encodes by default
    var t_src = pa.table(
        Python.dict(
            s=pa.array(
                Python.evaluate(
                    "[['red', 'green', 'blue'][i % 3] for i in range(400)]"
                )
            )
        )
    )
    var path = String("/tmp/marrow_dict.parquet")
    pq.write_table(t_src, path, use_dictionary=True, compression="snappy")
    var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
    assert_true(Bool(Python.evaluate("'RLE_DICTIONARY'") in enc))

    var t = read_table(path)
    assert_equal(t.num_rows(), 400)
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref s = c.as_string()
    assert_equal(String(s[0]), "red")  # 0 % 3 == 0
    assert_equal(String(s[1]), "green")
    assert_equal(String(s[2]), "blue")
    assert_equal(String(s[399]), "red")  # 399 % 3 == 0
    remove(path)


# ---------------------------------------------------------------------------
# Wide table (many columns)
# ---------------------------------------------------------------------------


def test_wide_table() raises:
    var pq = Python.import_module("pyarrow.parquet")
    # 40 int64 columns c0..c39; value in column j at row i is i*100 + j
    var t_src = Python.evaluate(
        "__import__('pyarrow').table({'c%d' % j:"
        " __import__('pyarrow').array([i * 100 + j for i in range(5)],"
        " type=__import__('pyarrow').int64()) for j in range(40)})"
    )
    var path = String("/tmp/marrow_wide.parquet")
    pq.write_table(t_src, path, compression="snappy")
    var t = read_table(path)
    assert_equal(t.num_columns(), 40)
    assert_equal(t.num_rows(), 5)
    var b = t.to_batches()[0].copy()
    var c0 = b.columns[0].copy()
    var c39 = b.columns[39].copy()
    assert_equal(c0.as_int64()[3].value(), 300)  # i=3, j=0
    assert_equal(c39.as_int64()[4].value(), 439)  # i=4, j=39
    remove(path)


# ---------------------------------------------------------------------------
# Nulls spanning many row groups
# ---------------------------------------------------------------------------


def test_nulls_across_row_groups() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # 5000 rows, every 3rd null, 500-row row groups -> 10 row groups
    var t_src = pa.table(
        Python.dict(
            x=pa.array(
                Python.evaluate(
                    "[None if i % 3 == 0 else i for i in range(5000)]"
                ),
                type=pa.int64(),
            )
        )
    )
    var path = String("/tmp/marrow_manyrg.parquet")
    pq.write_table(t_src, path, row_group_size=500, compression="snappy")
    assert_equal(Int(py=pq.ParquetFile(path).metadata.num_row_groups), 10)

    var t = read_table(path)
    assert_equal(t.num_rows(), 5000)
    var total_nulls = 0
    var checked = 0
    for ref bat in t.to_batches():
        var c = bat.columns[0].copy()
        total_nulls += c.as_int64().null_count()
        checked += c.length()
    assert_equal(checked, 5000)
    assert_equal(total_nulls, 1667)  # count of i in [0,5000) with i%3==0

    var b0 = t.to_batches()[0].copy()
    var c0 = b0.columns[0].copy()
    ref first = c0.as_int64()
    assert_false(first.is_valid(0))  # i=0 -> null
    assert_true(first.is_valid(1))
    assert_equal(first[1].value(), 1)
    remove(path)


# ---------------------------------------------------------------------------
# Pre-epoch (negative) date32
# ---------------------------------------------------------------------------


def test_date32_negative() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var d = Python.import_module("datetime")
    var t_src = pa.table(
        Python.dict(
            d=pa.array(
                Python.list(
                    d.date(1960, 1, 1),  # before the 1970 epoch -> negative
                    d.date(1970, 1, 1),  # epoch -> 0
                    d.date(2020, 1, 1),
                ),
                type=pa.date32(),
            )
        )
    )
    var path = String("/tmp/marrow_date.parquet")
    pq.write_table(t_src, path, compression="none")
    var t = read_table(path)
    assert_true(t.schema.field(index=0).dtype.is_date32())
    var b = t.to_batches()[0].copy()
    var c = b.columns[0].copy()
    ref days = c.as_date32()
    assert_true(days[0].value() < 0)  # pre-epoch
    assert_equal(days[1].value(), Int32(0))  # epoch
    assert_true(days[2].value() > 0)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
