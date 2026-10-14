# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Reading PyArrow-written files: flat columns (nulls, codecs), the value types
marrow maps (narrow ints, temporal, binary), column projection, and multi-page
chunks. Also covers value/boundary edge cases modelled on the pyarrow parquet
test suite: all-null columns, integer/float extremes, NaN/Inf, empty/unicode/long
strings, booleans with nulls, dictionary-encoded reads, wide tables, nulls
spanning many row groups, and pre-epoch dates."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises
from std.math import isnan, isinf
from std.python import Python, PythonObject
from std.memory import ArcPointer
from std.os.path import join
from ....utils.testing import ScratchDir
from ...parquet import read_table, ParquetFile
from ..reader import RowSelection
from ..compression import Compression
from ..writer import FileWriter
from ....io import BufferSource, ByteSource, Fetched, FileSink
from ....tabular import Table, record_batch
from ....builders import StringBuilder, array
from ....dtypes import binary_view, int64
from ....execution import ExecContext


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_rd_flat.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_rd_nulls.parquet")
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


def test_read_zstd() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, 2, 3), type=pa.int32()),
            b=pa.array(Python.list(True, False, True), type=pa.bool_()),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_rd_zstd.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_types_int.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_types_time.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_types_ms.parquet")
        pq.write_table(tbl, path, compression="none")

        var t = read_table(path)
        assert_true(t.schema.field(index=0).dtype.is_time32())
        assert_true(t.schema.field(index=1).dtype.is_timestamp())
        var b = t.to_batches()[0].copy()
        assert_equal(b.columns[0].copy().as_time32()[1].value(), 1000)
        assert_equal(b.columns[1].copy().as_timestamp()[2].value(), 3)


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_types_tstz.parquet")
        pq.write_table(tbl, path, compression="none")

        var t = read_table(path)
        assert_true(t.schema.field(index=0).dtype.is_timestamp())
        var b = t.to_batches()[0].copy()
        ref ts = b.columns[0].copy().as_timestamp()
        assert_equal(ts[0].value(), 1000)
        assert_equal(ts[2].value(), 3000)


def test_read_binary() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var bi = Python.import_module("builtins")
    var tbl = pa.table(
        Python.dict(
            b=pa.array(
                Python.list(
                    bi.bytes(Python.list(0, 1)),
                    bi.bytes(Python.list(255, 254)),
                    Python.str("ab").encode(),
                ),
                type=pa.binary(),
            )
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_types_bin.parquet")
        pq.write_table(tbl, path, compression="snappy")

        var t = read_table(path)
        var b = t.to_batches()[0].copy()
        var col = b.columns[0].copy()
        assert_true(t.schema.field(index=0).dtype.is_binary())
        assert_equal(col.length(), 3)


def test_read_large_binary() raises:
    # large_binary has no distinct Parquet physical type -> reads back as binary.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            b=pa.array(
                Python.list(
                    Python.str("ab").encode(),
                    Python.none(),
                    Python.str("cdef").encode(),
                ),
                type=pa.large_binary(),
            )
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_types_lbin.parquet")
        pq.write_table(tbl, path, compression="snappy")

        var t = read_table(path)
        assert_true(t.schema.field(index=0).dtype.is_binary())
        var b = t.to_batches()[0].copy()
        var col = b.columns[0].copy()
        assert_equal(col.length(), 3)
        assert_equal(col.null_count(), 1)


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_proj.parquet")
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


def test_project_single_column() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            a=pa.array(Python.list(10, 20, 30, 40), type=pa.int64()),
            b=pa.array(Python.list(1, 2, 3, 4), type=pa.int32()),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_proj1.parquet")
        pq.write_table(tbl, path, compression="none")

        var cols: List[String] = ["b"]
        var t = read_table(path, columns=cols^)
        assert_equal(t.num_columns(), 1)
        assert_equal(t.num_rows(), 4)
        assert_equal(t.column_names()[0], "b")
        var b = t.to_batches()[0].copy()
        assert_equal(b.columns[0].copy().as_int32()[3].value(), 4)


def test_project_struct_column() raises:
    # a struct column (multi-leaf node) projected alongside a flat one
    var pa = Python.import_module("pyarrow")
    var t_src = pa.table(
        Python.dict(
            s=pa.array(
                Python.list(Python.dict(x=1, y="a"), Python.dict(x=2, y="b")),
                type=pa.struct(
                    Python.list(
                        pa.field("x", pa.int64()),
                        pa.field("y", pa.string()),
                    )
                ),
            ),
            n=pa.array(Python.list(7, 8), type=pa.int64()),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_proj_struct.parquet")
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


def test_project_missing_column() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(a=pa.array(Python.list(1, 2), type=pa.int64()))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_proj_miss.parquet")
        pq.write_table(tbl, path)
        var cols: List[String] = ["nope"]
        with assert_raises():
            _ = read_table(path, columns=cols^)


# ---------------------------------------------------------------------------
# Multiple data pages within a single column chunk (small data_page_size).
# The reader must loop over every page and concatenate their values; a single
# row group here holds many pages, distinct from the multi-row-group cases.
# ---------------------------------------------------------------------------


def test_many_pages_plain_int() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var tbl = pa.table(
        Python.dict(x=pa.array(np.arange(10000), type=pa.int64()))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_pages_plain.parquet")
        # tiny page size forces many PLAIN pages inside one chunk
        pq.write_table(
            tbl,
            path,
            data_page_size=256,
            use_dictionary=False,
            compression="none",
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


def test_many_pages_dict_string() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # low cardinality -> dictionary-encoded data pages, split across many pages
    var colors = np.array(Python.list("red", "green", "blue"))
    var tbl = pa.table(Python.dict(s=pa.array(colors[np.arange(6000) % 3])))
    with ScratchDir() as dir:
        var path = join(dir, "marrow_pages_dict.parquet")
        pq.write_table(
            tbl,
            path,
            data_page_size=128,
            use_dictionary=True,
            compression="snappy",
        )
        var t = read_table(path)
        assert_equal(t.num_rows(), 6000)
        var b = t.to_batches()[0].copy()
        ref s = b.columns[0].copy().as_string()
        assert_equal(String(s[0]), "red")
        assert_equal(String(s[3001]), "green")  # 3001 % 3 == 1
        assert_equal(String(s[5999]), "blue")  # 5999 % 3 == 2


def test_many_pages_string_with_nulls() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # variable-length strings with every 4th null, spanning many pages
    var idx = np.arange(4000)
    var vals = np.char.add("v", idx.astype("U"))
    var tbl = pa.table(
        Python.dict(s=pa.array(vals, mask=(idx % 4 == 0), type=pa.string()))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_pages_strnull.parquet")
        pq.write_table(
            tbl,
            path,
            data_page_size=200,
            use_dictionary=False,
            compression="none",
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


# ---------------------------------------------------------------------------
# Value and boundary edge cases (modelled on the pyarrow parquet test suite)
# ---------------------------------------------------------------------------


def _read(
    dir: String, want: PythonObject, compression: String = "snappy"
) raises -> Table:
    """Write a PyArrow table into `dir` and read it back through marrow."""
    var pq = Python.import_module("pyarrow.parquet")
    var path = join(dir, "marrow_values.parquet")
    pq.write_table(want, path, compression=compression)
    var t = read_table(path)
    return t^


def test_all_null_int64() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir, pa.table(Python.dict(x=pa.nulls(4, type=pa.int64())))
        )
        assert_equal(t.num_rows(), 4)
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        ref a = c.as_int64()
        assert_equal(a.null_count(), 4)
        assert_false(a.is_valid(0))


def test_all_null_string() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir, pa.table(Python.dict(s=pa.nulls(3, type=pa.string())))
        )
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        assert_equal(c.as_string().null_count(), 3)


def test_single_row() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    i=pa.array(Python.list(42), type=pa.int64()),
                    s=pa.array(Python.list("solo")),
                )
            ),
        )
        assert_equal(t.num_rows(), 1)
        var b = t.to_batches()[0].copy()
        var ci = b.columns[0].copy()
        var cs = b.columns[1].copy()
        assert_equal(ci.as_int64()[0].value(), 42)
        assert_equal(String(cs.as_string()[0]), "solo")


def test_int64_extremes() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    x=pa.array(
                        Python.list(
                            Int64(-9223372036854775808),
                            Int64(9223372036854775807),
                            Int64(0),
                        ),
                        type=pa.int64(),
                    )
                )
            ),
        )
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        ref a = c.as_int64()
        assert_equal(a[0].value(), Int64(-9223372036854775808))
        assert_equal(a[1].value(), Int64(9223372036854775807))
        assert_equal(a[2].value(), Int64(0))


def test_uint64_max() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    x=pa.array(
                        Python.list(UInt64(0), UInt64(18446744073709551615)),
                        type=pa.uint64(),
                    )
                )
            ),
        )
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        ref a = c.as_uint64()
        assert_equal(a[0].value(), UInt64(0))
        assert_equal(a[1].value(), UInt64(18446744073709551615))


def test_int32_extremes() raises:
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    x=pa.array(
                        Python.list(Int32(-2147483648), Int32(2147483647)),
                        type=pa.int32(),
                    )
                )
            ),
        )
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        ref a = c.as_int32()
        assert_equal(a[0].value(), Int32(-2147483648))
        assert_equal(a[1].value(), Int32(2147483647))


def test_float64_special() raises:
    var pa = Python.import_module("pyarrow")
    var m = Python.import_module("math")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    f=pa.array(
                        Python.list(m.nan, m.inf, -m.inf, 0.0, 1.5),
                        type=pa.float64(),
                    )
                )
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
    var pa = Python.import_module("pyarrow")
    var m = Python.import_module("math")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    f=pa.array(
                        Python.list(m.nan, m.inf, -m.inf, -2.5),
                        type=pa.float32(),
                    )
                )
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
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    s=pa.array(
                        Python.list(
                            "",
                            "a",
                            "héllo 世界",
                            Python.str("x") * 5000,
                            Python.none(),
                        )
                    )
                )
            ),
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
    var pa = Python.import_module("pyarrow")
    with ScratchDir() as dir:
        var t = _read(
            dir,
            pa.table(
                Python.dict(
                    b=pa.array(
                        Python.list(
                            True,
                            Python.none(),
                            False,
                            Python.none(),
                            True,
                            False,
                        ),
                        type=pa.bool_(),
                    )
                )
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
    var np = Python.import_module("numpy")
    # 400 rows over 3 distinct strings -> pyarrow dictionary-encodes by default
    var colors = np.array(Python.list("red", "green", "blue"))
    var t_src = pa.table(Python.dict(s=pa.array(colors[np.arange(400) % 3])))
    with ScratchDir() as dir:
        var path = join(dir, "marrow_dict.parquet")
        pq.write_table(t_src, path, use_dictionary=True, compression="snappy")
        var enc = pq.ParquetFile(path).metadata.row_group(0).column(0).encodings
        assert_true(Bool(Python.str("RLE_DICTIONARY") in enc))

        var t = read_table(path)
        assert_equal(t.num_rows(), 400)
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        ref s = c.as_string()
        assert_equal(String(s[0]), "red")  # 0 % 3 == 0
        assert_equal(String(s[1]), "green")
        assert_equal(String(s[2]), "blue")
        assert_equal(String(s[399]), "red")  # 399 % 3 == 0


def test_wide_table() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # 40 int64 columns c0..c39; value in column j at row i is i*100 + j
    var cols = Python.dict()
    var rows = np.arange(5) * 100
    for j in range(40):
        cols[Python.str("c") + String(j)] = pa.array(rows + j, type=pa.int64())
    var t_src = pa.table(cols)
    with ScratchDir() as dir:
        var path = join(dir, "marrow_wide.parquet")
        pq.write_table(t_src, path, compression="snappy")
        var t = read_table(path)
        assert_equal(t.num_columns(), 40)
        assert_equal(t.num_rows(), 5)
        var b = t.to_batches()[0].copy()
        var c0 = b.columns[0].copy()
        var c39 = b.columns[39].copy()
        assert_equal(c0.as_int64()[3].value(), 300)  # i=3, j=0
        assert_equal(c39.as_int64()[4].value(), 439)  # i=4, j=39


def test_nulls_across_row_groups() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    # 5000 rows, every 3rd null, 500-row row groups -> 10 row groups
    var idx = np.arange(5000)
    var t_src = pa.table(
        Python.dict(x=pa.array(idx, mask=(idx % 3 == 0), type=pa.int64()))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_manyrg.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_date.parquet")
        pq.write_table(t_src, path, compression="none")
        var t = read_table(path)
        assert_true(t.schema.field(index=0).dtype.is_date32())
        var b = t.to_batches()[0].copy()
        var c = b.columns[0].copy()
        ref days = c.as_date32()
        assert_true(days[0].value() < 0)  # pre-epoch
        assert_equal(days[1].value(), Int32(0))  # epoch
        assert_true(days[2].value() > 0)


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_rd_pqfile.parquet")
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


def test_read_rle_bool_v2() raises:
    # PyArrow's DataPage v2 encodes boolean *values* as RLE (not PLAIN)
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var vals = Python.list()
    for i in range(500):
        if i % 9 == 0:
            vals.append(Python.none())
        else:
            vals.append(i % 2 == 0)
    var tbl = pa.table(Python.dict(b=pa.array(vals, type=pa.bool_())))
    with ScratchDir() as dir:
        var path = join(dir, "marrow_rle_bool.parquet")
        pq.write_table(tbl, path, data_page_version="2.0", use_dictionary=False)

        var t = read_table(path)
        var b = t.to_batches()[0].copy()
        var cb = b.columns[0].copy()
        ref col = cb.as_bool()
        assert_equal(col.null_count(), 500 // 9 + 1)
        assert_true(col[2].value())  # i=2 -> True
        assert_false(col[1].value())  # i=1 -> False
        assert_false(col.is_valid(0))  # i=0 -> null


def test_read_int96_timestamp() raises:
    # legacy INT96 timestamps (Impala/Spark/Hive) -> timestamp(ns)
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var ts = Python.list()
    for i in range(200):
        if i % 13 == 0:
            ts.append(Python.none())
        else:
            ts.append(i * 1000000000 + 500)
    var tbl = pa.table(Python.dict(t=pa.array(ts, type=pa.timestamp("ns"))))
    with ScratchDir() as dir:
        var path = join(dir, "marrow_int96.parquet")
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


def _int96_overcounted(path: String, use_dictionary: Bool) raises:
    """Write 100 required INT96 timestamps to `path`, then raise the count in
    the page header holding their bytes (the dictionary page, or the data page
    without one) from 100 to 1000, leaving the body as it was."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var bi = Python.import_module("builtins")
    var ts = Python.list()
    for i in range(100):
        ts.append(i * 1000000000)
    var field = pa.field("t", pa.timestamp("ns"), nullable=False)
    var tbl = pa.table(
        Python.list(pa.array(ts, type=pa.timestamp("ns"))),
        schema=pa.schema(Python.list(field)),
    )
    pq.write_table(
        tbl,
        path,
        compression="none",
        use_dictionary=use_dictionary,
        write_statistics=False,
        use_deprecated_int96_timestamps=True,
    )
    var f = bi.open(path, "rb")
    var raw = f.read()
    f.close()
    # The thrift compact header: the nested page header struct (field 5 for a
    # data page, field 7 for a dictionary page), then its i32 field 1,
    # num_values, as a zigzag varint -- 100 is 0xC8 0x01, 1000 is 0xD0 0x0F.
    var tag = 0x4C if use_dictionary else 0x2C
    var count = bi.bytes(Python.list(tag, 0x15, 0xC8, 0x01))
    assert_equal(Int(py=raw.count(count)), 1)
    raw = raw.replace(count, bi.bytes(Python.list(tag, 0x15, 0xD0, 0x0F)))
    f = bi.open(path, "wb")
    _ = f.write(raw)
    f.close()


def test_read_int96_refuses_an_overcounted_page() raises:
    """An INT96 data page or dictionary page whose header counts more values
    than its body holds is refused as corrupt rather than read past."""
    with ScratchDir() as dir:
        var plain = join(dir, "marrow_int96_plain_overcount.parquet")
        _int96_overcounted(plain, use_dictionary=False)
        with assert_raises(contains="CorruptError"):
            _ = read_table(plain)
        var dict = join(dir, "marrow_int96_dict_overcount.parquet")
        _int96_overcounted(dict, use_dictionary=True)
        with assert_raises(contains="CorruptError"):
            _ = read_table(dict)


comptime _DICT_SIZE = 96
"""Distinct values written by `_past_the_dictionary`: 7-bit indices, one
bit-packed run of twelve groups of eight."""


def _one_per_list(values: PythonObject) raises -> PythonObject:
    """`values` as a list array holding one value per row."""
    var pa = Python.import_module("pyarrow")
    var offsets = Python.list()
    for i in range(_DICT_SIZE + 1):
        offsets.append(i)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), values)


def _past_the_dictionary(
    path: String, column: PythonObject, int96: Bool = False
) raises:
    """Write `column` -- `_DICT_SIZE` distinct values -- dictionary-encoded to
    `path`, then set every bit of the data page's index run, so each index
    reads 127, past the dictionary."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var bi = Python.import_module("builtins")
    pq.write_table(
        pa.table(Python.dict(c=column)),
        path,
        compression="none",
        write_statistics=False,
        use_deprecated_int96_timestamps=int96,
    )
    var f = bi.open(path, "rb")
    var raw = bi.bytearray(f.read())
    f.close()
    # The index stream: its bit width, 7, then the run header -- twelve
    # bit-packed groups, (12 << 1) | 1 -- and their 12 * 7 bytes.
    var head = bi.bytes(Python.list(0x07, 0x19))
    assert_equal(Int(py=raw.count(head)), 1)
    var at = Int(py=raw.find(head)) + 2
    for i in range(12 * 7):
        raw[at + i] = 0xFF
    f = bi.open(path, "wb")
    _ = f.write(raw)
    f.close()


def _refuses_past_the_dictionary(
    dir: String,
    name: String,
    values: PythonObject,
    int96: Bool = False,
    views: Bool = False,
) raises:
    """A flat column of `values` and a list of them, each with every index
    past the dictionary, are refused as corrupt -- with `views`, the flat
    column read as views too."""
    var flat = join(dir, name + "_flat.parquet")
    _past_the_dictionary(flat, values, int96)
    with assert_raises(contains="dictionary index out of range"):
        _ = read_table(flat)
    if views:
        with assert_raises(contains="dictionary index out of range"):
            _ = read_table(flat, binary_type=binary_view)
    var nested = join(dir, name + "_nested.parquet")
    _past_the_dictionary(nested, _one_per_list(values), int96)
    with assert_raises(contains="dictionary index out of range"):
        _ = read_table(nested)


def test_read_int96_refuses_a_dictionary_index_past_the_dictionary() raises:
    var pa = Python.import_module("pyarrow")
    var ts = Python.list()
    for i in range(_DICT_SIZE):
        ts.append(i * 1000000000)
    with ScratchDir() as dir:
        _refuses_past_the_dictionary(
            dir,
            "marrow_int96_dict_index",
            pa.array(ts, type=pa.timestamp("ns")),
            int96=True,
        )


def test_read_decimal_refuses_a_dictionary_index_past_the_dictionary() raises:
    var pa = Python.import_module("pyarrow")
    var ints = Python.list()
    for i in range(_DICT_SIZE):
        ints.append(i)
    with ScratchDir() as dir:
        _refuses_past_the_dictionary(
            dir,
            "marrow_decimal_dict_index",
            pa.array(ints, type=pa.int64()).cast(pa.decimal128(21, 2)),
        )


def test_read_fixed_size_binary_refuses_a_dictionary_index_past_the_dictionary() raises:
    var pa = Python.import_module("pyarrow")
    var bi = Python.import_module("builtins")
    var values = Python.list()
    for i in range(_DICT_SIZE):
        values.append(bi.bytes(Python.list(0, 0, 0, i)))
    with ScratchDir() as dir:
        _refuses_past_the_dictionary(
            dir,
            "marrow_fsb_dict_index",
            pa.array(values, type=pa.binary(4)),
        )


def test_read_strings_refuse_a_dictionary_index_past_the_dictionary() raises:
    """Copied into a string array, and pointed at by views."""
    var pa = Python.import_module("pyarrow")
    var values = Python.list()
    for i in range(_DICT_SIZE):
        values.append("value-" + String(i))
    with ScratchDir() as dir:
        _refuses_past_the_dictionary(
            dir, "marrow_string_dict_index", pa.array(values), views=True
        )


def _overcount_the_dictionary(path: String, column: PythonObject) raises:
    """Write `column` -- `_DICT_SIZE` distinct values -- dictionary-encoded to
    `path`, then raise the dictionary page's count from 96 to 1000, leaving
    its body as it was."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var bi = Python.import_module("builtins")
    pq.write_table(
        pa.table(Python.dict(c=column)),
        path,
        compression="none",
        write_statistics=False,
    )
    var f = bi.open(path, "rb")
    var raw = f.read()
    f.close()
    # The dictionary page header (field 7), then its num_values as a zigzag
    # varint -- 96 is 0xC0 0x01, 1000 is 0xD0 0x0F.
    var count = bi.bytes(Python.list(0x4C, 0x15, 0xC0, 0x01))
    assert_equal(Int(py=raw.count(count)), 1)
    raw = raw.replace(count, bi.bytes(Python.list(0x4C, 0x15, 0xD0, 0x0F)))
    f = bi.open(path, "wb")
    _ = f.write(raw)
    f.close()


def test_read_fixed_width_refuses_an_overcounted_dictionary() raises:
    """A decimal or fixed-size binary dictionary page counting more values
    than its body holds is refused, flat or in a list."""
    var pa = Python.import_module("pyarrow")
    var bi = Python.import_module("builtins")
    var ints = Python.list()
    var bytes = Python.list()
    for i in range(_DICT_SIZE):
        ints.append(i)
        bytes.append(bi.bytes(Python.list(0, 0, 0, i)))
    var columns = Python.list(
        pa.array(ints, type=pa.int64()).cast(pa.decimal128(21, 2)),
        pa.array(bytes, type=pa.binary(4)),
    )
    with ScratchDir() as dir:
        for i in range(2):
            var flat = join(
                dir, "marrow_dict_overcount_" + String(i) + "_flat.parquet"
            )
            _overcount_the_dictionary(flat, columns[i])
            with assert_raises(contains="fewer values than it says"):
                _ = read_table(flat)
            var nested = join(
                dir, "marrow_dict_overcount_" + String(i) + "_nested.parquet"
            )
            _overcount_the_dictionary(nested, _one_per_list(columns[i]))
            with assert_raises(contains="fewer values than it says"):
                _ = read_table(nested)


def _v2_null_count(path: String, nullable: Bool, zigzag: Int) raises:
    """Write 20 int64 values, none null, as one v2 data page to `path`, then
    set the page header's `num_nulls` to the zigzag varint byte `zigzag`."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var bi = Python.import_module("builtins")
    var ints = Python.list()
    for i in range(20):
        ints.append(i)
    var field = pa.field("i", pa.int64(), nullable=nullable)
    pq.write_table(
        pa.table(
            Python.list(pa.array(ints, type=pa.int64())),
            schema=pa.schema(Python.list(field)),
        ),
        path,
        compression="none",
        use_dictionary=False,
        write_statistics=False,
        data_page_version="2.0",
    )
    var f = bi.open(path, "rb")
    var raw = f.read()
    f.close()
    # The thrift compact header: the v2 page header struct (field 8), its
    # num_values (20, zigzag 0x28), then its num_nulls (0).
    var header = bi.bytes(Python.list(0x5C, 0x15, 0x28, 0x15, 0x00))
    assert_equal(Int(py=raw.count(header)), 1)
    raw = raw.replace(
        header, bi.bytes(Python.list(0x5C, 0x15, 0x28, 0x15, zigzag))
    )
    f = bi.open(path, "wb")
    _ = f.write(raw)
    f.close()


def test_read_v2_refuses_nulls_in_a_required_column() raises:
    with ScratchDir() as dir:
        var path = join(dir, "marrow_v2_required_nulls.parquet")
        _v2_null_count(path, nullable=False, zigzag=0x0A)  # 5
        with assert_raises(contains="no definition levels"):
            _ = read_table(path)


def test_read_v2_refuses_a_negative_null_count() raises:
    """Refused where the page header is read."""
    with ScratchDir() as dir:
        var path = join(dir, "marrow_v2_negative_nulls.parquet")
        _v2_null_count(path, nullable=False, zigzag=0x01)  # -1
        with assert_raises(contains="nulls in 20 values"):
            _ = read_table(path)


def test_read_v2_refuses_more_nulls_than_values() raises:
    """Refused where the page header is read, with levels or without."""
    with ScratchDir() as dir:
        var required = join(dir, "marrow_v2_required_overcount.parquet")
        _v2_null_count(required, nullable=False, zigzag=0x3C)  # 30
        with assert_raises(contains="nulls in 20 values"):
            _ = read_table(required)
        var nullable = join(dir, "marrow_v2_nullable_overcount.parquet")
        _v2_null_count(nullable, nullable=True, zigzag=0x3C)
        with assert_raises(contains="nulls in 20 values"):
            _ = read_table(nullable)


def test_read_float16() raises:
    # FLOAT16 = FIXED_LEN_BYTE_ARRAY(2) + FLOAT16 logical -> Arrow halffloat.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var vals = np.array(
        Python.list(1.5, -2.0, 3.25, 0.0, 100.0), dtype=np.float16
    )
    var mask = np.array(Python.list(False, False, True, False, False))
    var tbl = pa.table(
        Python.dict(h=pa.array(vals, mask=mask, type=pa.float16()))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_read_f16.parquet")
        pq.write_table(tbl, path, use_dictionary=False)
        assert_equal(
            String(
                py=pq.ParquetFile(path)
                .metadata.row_group(0)
                .column(0)
                .physical_type
            ),
            "FIXED_LEN_BYTE_ARRAY",
        )

        var t = read_table(path)
        var b = t.to_batches()[0].copy()
        var ch = b.columns[0].copy()
        ref col = ch.as_float16()
        assert_true(col[0].value() == Float16(1.5))
        assert_true(col[1].value() == Float16(-2.0))
        assert_false(col.is_valid(2))  # masked
        assert_true(col[4].value() == Float16(100.0))


def test_read_from_byte_source() raises:
    """`ParquetFile` reads through the `ByteSource` seam: constructing it from a
    `BufferSource` (a `ByteSource`) directly decodes identically to the
    path-convenience constructor."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            i=pa.array(Python.list(1, 2, 3, 4, 5), type=pa.int64()),
            s=pa.array(
                Python.list("apple", "banana", "cherry", "date", "elder")
            ),
        )
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_rd_byte_source.parquet")
        pq.write_table(tbl, path, compression="snappy")

        # Build the file from an explicit ByteSource instead of a path.
        var pf = ParquetFile(BufferSource(path))
        assert_equal(pf.num_rows(), 5)
        assert_equal(pf.num_row_groups(), 1)

        var t = pf.read()
        assert_equal(t.num_rows(), 5)
        assert_equal(t.num_columns(), 2)
        var b = t.to_batches()[0].copy()

        var ci = b.columns[0].copy()
        ref col_i = ci.as_int64()
        assert_equal(col_i[0].value(), 1)
        assert_equal(col_i[4].value(), 5)

        var cs = b.columns[1].copy()
        ref col_s = cs.as_string()
        assert_equal(String(col_s[0]), "apple")
        assert_equal(String(col_s[4]), "elder")


def test_read_plain_byte_array_refuses_an_overrunning_length() raises:
    """A PLAIN byte-array value whose length runs past its page is refused,
    by the copying reader and the view reader alike."""
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(s=pa.array(Python.list("abc", "de", "f"), type=pa.binary()))
    )
    with ScratchDir() as dir:
        var path = join(dir, "marrow_overrun.parquet")
        pq.write_table(tbl, path, compression="none", use_dictionary=False)
        var bi = Python.import_module("builtins")
        var f = bi.open(path, "rb")
        var raw = f.read()
        f.close()
        var value = Python.str("abc").encode()
        var prefix = bi.bytes(Python.list(3, 0, 0, 0))
        assert_true(Int(py=raw.count(prefix + value)) == 1)
        var huge = bi.bytes(Python.list(0xF0, 0xFF, 0xFF, 0x7F))
        raw = raw.replace(prefix + value, huge + value)
        f = bi.open(path, "wb")
        _ = f.write(raw)
        f.close()
        with assert_raises(contains="runs past"):
            _ = read_table(path)
        with assert_raises(contains="runs past"):
            _ = read_table(path, binary_type=binary_view)


struct _ArenaSource(ByteSource):
    """A local file that, like an object store, keeps every `read_at` answer
    in an arena it appends to without a lock -- what `ByteSource.read_at`
    allows, and why it is one thread's call."""

    var _inner: BufferSource
    var _arena: ArcPointer[List[Tuple[Int, Int]]]

    def __init__(out self, path: String) raises:
        self._inner = BufferSource(path)
        self._arena = ArcPointer(List[Tuple[Int, Int]]())

    def size(self) -> Int:
        return self._inner.size()

    def read_at(
        ref self, offset: Int, length: Int
    ) raises -> Span[UInt8, origin_of(self)]:
        self._arena[].append((offset, length))
        return rebind[Span[UInt8, origin_of(self)]](
            self._inner.read_at(offset, length)
        )

    def read_ranges(
        self, ranges: List[Tuple[Int, Int]], ctx: ExecContext
    ) raises -> Fetched:
        return self._inner.read_ranges(ranges, ctx)


def _read_group(f: ParquetFile, rg: Int, selected: Bool) raises -> String:
    """Row group `rg`, every row or every third, on a serial context. A
    selection makes the read fetch the group's offset index."""
    var groups: List[Int] = [rg]
    if not selected:
        return String(f.read(row_groups=groups^, ctx=ExecContext.serial()))
    var keep = List[Bool](capacity=500)
    for i in range(500):
        keep.append(i % 3 == 0)
    var selections: List[RowSelection] = [RowSelection(keep^)]
    return String(
        f.read(
            row_groups=groups^,
            row_selections=selections^,
            ctx=ExecContext.serial(),
        )
    )


def test_parquet_file_says_whether_it_needs_codec_libraries() raises:
    """Snappy pages decode in Mojo, so a native read opens no library and a
    read through the C codecs does."""
    var batch = record_batch([array([1, 2, 3], int64)], names=["v"])
    var table = Table.from_batches(batch.schema.copy(), [batch^])
    with ScratchDir() as dir:
        var path = join(dir, "marrow_needs_libs.parquet")
        var w = FileWriter(FileSink(path), table.schema, Compression.SNAPPY)
        w.write(table)
        w.close()
        assert_false(ParquetFile(path).needs_libs())
        assert_true(ParquetFile(path, native_codecs=False).needs_libs())


def test_parquet_read_one_file_from_many_threads() raises:
    """Threads reading row groups of one open file at once, each on a serial
    context, each get what a lone read gets. Snappy pages, decoded by every
    thread at once."""
    comptime GROUPS = 8
    var b = StringBuilder(GROUPS * 500)
    for i in range(GROUPS * 500):
        b.append(String("value-", i % 37, "-", i))
    var batch = record_batch([b.finish()], names=["s"])
    var schema = batch.schema.copy()
    var table = Table.from_batches(schema, [batch^])
    with ScratchDir() as dir:
        var path = join(dir, "marrow_concurrent_read.parquet")
        var w = FileWriter(
            FileSink(path), schema, Compression.SNAPPY, row_group_size=500
        )
        w.write(table)
        w.close()
        # The threads read first, so the file's first reads are concurrent,
        # through a source whose `read_at` must stay on one thread.
        var f = ParquetFile(_ArenaSource(path))
        assert_equal(f.num_row_groups(), GROUPS)
        # Item `i` reads group `i % GROUPS`, the second half with a selection.
        var got = List[String](length=2 * GROUPS, fill="")

        def read(wid: Int, i: Int) raises {mut got, imm f}:
            got[i] = _read_group(f, i % GROUPS, i >= GROUPS)

        ExecContext.parallel(GROUPS).fan_out(2 * GROUPS, read, GROUPS)
        var alone = ParquetFile(path)
        for i in range(2 * GROUPS):
            assert_equal(got[i], _read_group(alone, i % GROUPS, i >= GROUPS))
