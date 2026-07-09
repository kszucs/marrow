"""Reading PyArrow-written files: flat columns (nulls, codecs), the value types
marrow maps (narrow ints, temporal, binary), and column projection."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.python import Python
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import read_table


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


def main() raises:
    TestSuite.run[__functions_in_module()]()
