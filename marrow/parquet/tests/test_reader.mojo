"""Reading PyArrow-written files: flat columns (nulls, codecs), the value types
marrow maps (narrow ints, temporal, binary), column projection, and multi-page
chunks. Also covers value/boundary edge cases modelled on the pyarrow parquet
test suite: all-null columns, integer/float extremes, NaN/Inf, empty/unicode/long
strings, booleans with nulls, dictionary-encoded reads, wide tables, nulls
spanning many row groups, and pre-epoch dates."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import isnan, isinf
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table, ParquetFile
from marrow.tabular import Table


# ---------------------------------------------------------------------------
# Flat columns
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


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


def test_read_temporal_ms_units() raises:
    # time32(ms) and timestamp(ms) — the millisecond leaf paths (only us/ns and
    # time64 were previously covered).
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            tm=pa.array(Python.list(0, 1000, 2000), type=pa.time32("ms")),
            ts=pa.array(Python.list(1, 2, 3), type=pa.timestamp("ms")),
        )
    )
    var path = String("/tmp/marrow_types_ms.parquet")
    pq.write_table(tbl, path, compression="none")

    var t = read_table(path)
    assert_true(t.schema.field(index=0).dtype.is_time32())
    assert_true(t.schema.field(index=1).dtype.is_timestamp())
    var b = t.to_batches()[0].copy()
    assert_equal(b.columns[0].copy().as_time32()[1].value(), 1000)
    assert_equal(b.columns[1].copy().as_timestamp()[2].value(), 3)
    remove(path)


def test_read_timestamp_with_timezone() raises:
    # An isAdjustedToUTC (tz-aware) timestamp: Parquet stores only the UTC flag,
    # so the instant values must survive intact and the type stays a timestamp.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            ts=pa.array(
                Python.list(1000, 2000, 3000),
                type=pa.timestamp("us", tz="America/New_York"),
            )
        )
    )
    var path = String("/tmp/marrow_types_tstz.parquet")
    pq.write_table(tbl, path, compression="none")

    var t = read_table(path)
    assert_true(t.schema.field(index=0).dtype.is_timestamp())
    var b = t.to_batches()[0].copy()
    ref ts = b.columns[0].copy().as_timestamp()
    assert_equal(ts[0].value(), 1000)
    assert_equal(ts[2].value(), 3000)
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


def test_read_large_binary() raises:
    # large_binary has no distinct Parquet physical type -> reads back as binary.
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = Python.evaluate(
        "__import__('pyarrow').table({'b':"
        " __import__('pyarrow').array([b'ab', None, b'cdef'],"
        " type=__import__('pyarrow').large_binary())})"
    )
    var path = String("/tmp/marrow_types_lbin.parquet")
    pq.write_table(tbl, path, compression="snappy")

    var t = read_table(path)
    assert_true(t.schema.field(index=0).dtype.is_binary())
    var b = t.to_batches()[0].copy()
    var col = b.columns[0].copy()
    assert_equal(col.length(), 3)
    assert_equal(col.null_count(), 1)
    remove(path)


# ---------------------------------------------------------------------------
# Column projection — read_table(path, columns=[...])
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Multiple data pages within a single column chunk (small data_page_size).
# The reader must loop over every page and concatenate their values; a single
# row group here holds many pages, distinct from the multi-row-group cases.
# ---------------------------------------------------------------------------


def test_many_pages_plain_int() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.evaluate("list(range(10000))"), type=pa.int64())
        )
    )
    var path = String("/tmp/marrow_pages_plain.parquet")
    # tiny page size forces many PLAIN pages inside one chunk
    pq.write_table(
        tbl, path, data_page_size=256, use_dictionary=False, compression="none"
    )
    # sanity: a single row group holding many pages
    assert_equal(Int(py=pq.ParquetFile(path).metadata.num_row_groups), 1)

    var t = read_table(path)
    assert_equal(t.num_rows(), 10000)
    var b = t.to_batches()[0].copy()
    ref a = b.columns[0].copy().as_int64()
    assert_equal(a[0].value(), 0)
    assert_equal(a[5000].value(), 5000)
    assert_equal(a[9999].value(), 9999)
    remove(path)


def test_many_pages_dict_string() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # low cardinality -> dictionary-encoded data pages, split across many pages
    var tbl = pa.table(
        Python.dict(
            s=pa.array(
                Python.evaluate(
                    "[['red', 'green', 'blue'][i % 3] for i in range(6000)]"
                )
            )
        )
    )
    var path = String("/tmp/marrow_pages_dict.parquet")
    pq.write_table(
        tbl, path, data_page_size=128, use_dictionary=True, compression="snappy"
    )
    var t = read_table(path)
    assert_equal(t.num_rows(), 6000)
    var b = t.to_batches()[0].copy()
    ref s = b.columns[0].copy().as_string()
    assert_equal(String(s[0]), "red")
    assert_equal(String(s[3001]), "green")  # 3001 % 3 == 1
    assert_equal(String(s[5999]), "blue")  # 5999 % 3 == 2
    remove(path)


def test_many_pages_string_with_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    # variable-length strings with every 4th null, spanning many pages
    var tbl = pa.table(
        Python.dict(
            s=pa.array(
                Python.evaluate(
                    "[None if i % 4 == 0 else 'v%d' % i for i in range(4000)]"
                ),
                type=pa.string(),
            )
        )
    )
    var path = String("/tmp/marrow_pages_strnull.parquet")
    pq.write_table(
        tbl, path, data_page_size=200, use_dictionary=False, compression="none"
    )
    var t = read_table(path)
    assert_equal(t.num_rows(), 4000)
    var total_nulls = 0
    for ref bat in t.to_batches():
        total_nulls += bat.columns[0].copy().as_string().null_count()
    assert_equal(total_nulls, 1000)  # i%4==0 over [0,4000)

    var b = t.to_batches()[0].copy()
    ref s = b.columns[0].copy().as_string()
    assert_false(s.is_valid(0))
    assert_equal(String(s[1]), "v1")
    assert_equal(String(s[3999]), "v3999")
    remove(path)


# ---------------------------------------------------------------------------
# Value and boundary edge cases (modelled on the pyarrow parquet test suite)
# ---------------------------------------------------------------------------


def _read(code: String, compression: String = "snappy") raises -> Table:
    """Write a one-line pyarrow table expression to a file and read it back."""
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_values.parquet")
    pq.write_table(Python.evaluate(code), path, compression=compression)
    var t = read_table(path)
    remove(path)
    return t^


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


def test_parquet_file() raises:
    # ParquetFile exposes the metadata/schema without decoding, and read()
    # (which read_table wraps) decodes; both share one opened file.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            a=pa.array(Python.list(1, 2, 3, 4), type=pa.int64()),
            b=pa.array(Python.list("w", "x", "y", "z")),
        )
    )
    var path = String("/tmp/marrow_rd_pqfile.parquet")
    pq.write_table(tbl, path, row_group_size=2)  # -> 2 row groups

    var f = ParquetFile(path)
    assert_equal(f.num_rows(), 4)
    assert_equal(f.num_row_groups(), 2)
    assert_equal(f.schema().num_fields(), 2)

    # metadata + statistics come off the same opened file (no re-mmap)
    assert_equal(f.metadata().num_rows, 4)
    var stats = f.statistics()
    assert_equal(len(stats), 2)  # 2 row groups
    assert_equal(len(stats[0]), 2)  # 2 leaf columns

    # full read
    var t = f.read()
    assert_equal(t.num_rows(), 4)
    assert_equal(t.num_columns(), 2)

    # column projection through the same object
    var cols: List[String] = [String("b")]
    var proj = f.read(columns=cols^)
    assert_equal(proj.num_columns(), 1)
    assert_equal(proj.num_rows(), 4)

    # single row group
    var rgs: List[Int] = [0]
    var rg0 = f.read(row_groups=rgs^)
    assert_equal(rg0.num_rows(), 2)
    remove(path)


def test_read_rle_bool_v2() raises:
    # PyArrow's DataPage v2 encodes boolean *values* as RLE (not PLAIN)
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var vals = Python.list()
    for i in range(500):
        if i % 9 == 0:
            vals.append(Python.evaluate("None"))
        else:
            vals.append(Python.evaluate("True" if i % 2 == 0 else "False"))
    var tbl = pa.table(Python.dict(b=pa.array(vals, type=pa.bool_())))
    var path = String("/tmp/marrow_rle_bool.parquet")
    pq.write_table(tbl, path, data_page_version="2.0", use_dictionary=False)

    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var cb = b.columns[0].copy()
    ref col = cb.as_bool()
    assert_equal(col.null_count(), 500 // 9 + 1)
    assert_true(col[2].value())  # i=2 -> True
    assert_false(col[1].value())  # i=1 -> False
    assert_false(col.is_valid(0))  # i=0 -> null
    remove(path)


def test_read_int96_timestamp() raises:
    # legacy INT96 timestamps (Impala/Spark/Hive) -> timestamp(ns)
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var ts = Python.list()
    for i in range(200):
        if i % 13 == 0:
            ts.append(Python.evaluate("None"))
        else:
            ts.append(i * 1000000000 + 500)
    var tbl = pa.table(Python.dict(t=pa.array(ts, type=pa.timestamp("ns"))))
    var path = String("/tmp/marrow_int96.parquet")
    pq.write_table(tbl, path, use_deprecated_int96_timestamps=True)
    # confirm it is really INT96 on disk
    assert_equal(
        String(
            py=pq.ParquetFile(path)
            .metadata.row_group(0)
            .column(0)
            .physical_type
        ),
        "INT96",
    )

    var t = read_table(path)
    var b = t.to_batches()[0].copy()
    var cts = b.columns[0].copy()
    ref col = cts.as_timestamp()
    assert_false(col.is_valid(0))  # i=0 -> null (0 % 13 == 0)
    assert_equal(col[1].value(), 1000000000 + 500)
    assert_equal(col[2].value(), 2000000000 + 500)
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
