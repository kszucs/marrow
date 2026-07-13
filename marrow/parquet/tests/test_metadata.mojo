"""Reading file metadata. Column statistics: `read_metadata` (raw footer) and
`read_statistics` (decoded typed min/max) — marrow reads the bounds PyArrow
writes, and round-trips its own (the write side is covered against PyArrow in
test_writer.mojo). Page index: `read_page_index` parses the OffsetIndex +
ColumnIndex PyArrow writes for a multi-page column, and the per-page bounds must
match the (sorted) data; marrow's own writer emits no page index yet, so those
chunks read back absent."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python, PythonObject
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet import (
    read_metadata,
    read_statistics,
    read_page_index,
    read_page_bounds,
    write_table,
)
from marrow.tabular import Table
from marrow.c_data import CArrowArrayStream


def _col(arr: PythonObject) raises -> PythonObject:
    """A single-column ("c") PyArrow table around `arr`."""
    return Python.import_module("pyarrow").table(Python.dict(c=arr))


def _write_pa(tbl: PythonObject, use_dictionary: Bool = True) raises -> String:
    var pq = Python.import_module("pyarrow.parquet")
    var path = String("/tmp/marrow_readstats.parquet")
    pq.write_table(tbl, path, compression="none", use_dictionary=use_dictionary)
    return path


def test_read_metadata_shape() raises:
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var path = _write_pa(_col(pa.array(np.arange(2500), type=pa.int64())))
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
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(
            pa.array(Python.list(5, -1, Python.none(), 9, -3), type=pa.int64())
        ),
        use_dictionary=False,
    )
    var st = read_statistics(path)
    assert_equal(len(st), 1)  # one row group
    ref cs = st[0][0]
    assert_true(Bool(cs.min))
    assert_equal(cs.null_count, 1)
    assert_equal(cs.min.value().as_int64().value(), -3)
    assert_equal(cs.max.value().as_int64().value(), 9)
    remove(path)


def test_read_uint_minmax() raises:
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(pa.array(Python.list(1, 3000000000, 2), type=pa.uint32())),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_equal(cs.min.value().as_uint32().value(), UInt32(1))
    assert_equal(
        cs.max.value().as_uint32().value(), UInt32(3000000000)
    )  # unsigned order
    remove(path)


def test_read_float_minmax() raises:
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(
            pa.array(
                Python.list(1.5, -2.5, 3.25, Python.none()),
                type=pa.float64(),
            )
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(cs.min.value().as_float64().value() == -2.5)
    assert_true(cs.max.value().as_float64().value() == 3.25)
    remove(path)


def test_read_string_minmax() raises:
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(pa.array(Python.list("banana", "apple", "cherry", Python.none()))),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(Bool(cs.min))
    assert_equal(cs.min.value().as_string().to_string(), "apple")
    assert_equal(cs.max.value().as_string().to_string(), "cherry")
    remove(path)


def test_read_stats_multiple_row_groups() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var tbl = pa.table(
        Python.dict(x=pa.array(np.arange(2500), type=pa.int64()))
    )
    var path = String("/tmp/marrow_readstats_rg.parquet")
    pq.write_table(
        tbl, path, row_group_size=1000, use_dictionary=False, compression="none"
    )
    var st = read_statistics(path)
    assert_equal(len(st), 3)  # 3 row groups
    # first row group covers rows [0, 1000)
    assert_equal(st[0][0].min.value().as_int64().value(), 0)
    assert_equal(st[0][0].max.value().as_int64().value(), 999)
    # last row group covers [2000, 2500)
    assert_equal(st[2][0].min.value().as_int64().value(), 2000)
    assert_equal(st[2][0].max.value().as_int64().value(), 2499)
    remove(path)


def test_roundtrip_own_stats() raises:
    # marrow writes -> marrow reads its own min/max back (closes the loop with
    # the PyArrow-oracle write tests in test_writer.mojo)
    var pa = Python.import_module("pyarrow")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.list(7, 2, Python.none(), 11), type=pa.int64()),
            s=pa.array(Python.list("m", "a", "z", Python.none())),
        )
    )
    var t = CArrowArrayStream.from_pycapsule(
        tbl.__arrow_c_stream__(Python.none())
    ).to_table()
    var path = String("/tmp/marrow_stats_rt.parquet")
    write_table(t, path)

    var st = read_statistics(path)
    assert_equal(st[0][0].min.value().as_int64().value(), 2)
    assert_equal(st[0][0].max.value().as_int64().value(), 11)
    assert_equal(st[0][0].null_count, 1)
    assert_equal(st[0][1].min.value().as_string().to_string(), "a")
    assert_equal(st[0][1].max.value().as_string().to_string(), "z")
    remove(path)


# ---------------------------------------------------------------------------
# Page index (OffsetIndex + ColumnIndex)
# ---------------------------------------------------------------------------


def _le_i64(b: List[UInt8]) -> Int:
    var v = 0
    for i in range(len(b)):
        v |= Int(b[i]) << (i * 8)
    return v


def test_page_index_sorted_int() raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var np = Python.import_module("numpy")
    var tbl = pa.table(
        Python.dict(x=pa.array(np.arange(10000), type=pa.int64()))
    )
    var path = String("/tmp/marrow_pageidx.parquet")
    # tiny pages -> many pages in one chunk; page index on
    pq.write_table(
        tbl,
        path,
        data_page_size=256,
        use_dictionary=False,
        compression="none",
        write_page_index=True,
    )

    var pi = read_page_index(path)
    assert_equal(len(pi), 1)  # one row group
    ref col = pi[0][0]
    assert_true(Bool(col.offset_index))
    assert_true(Bool(col.column_index))
    ref oi = col.offset_index.value()
    ref cix = col.column_index.value()

    var npages = len(oi.page_locations)
    assert_true(npages > 1)
    assert_equal(len(cix.min_values), npages)
    assert_equal(len(cix.max_values), npages)
    assert_equal(len(cix.null_pages), npages)

    # first_row_index starts at 0 and strictly increases
    assert_equal(oi.page_locations[0].first_row_index, 0)
    for i in range(1, npages):
        assert_true(
            oi.page_locations[i].first_row_index
            > oi.page_locations[i - 1].first_row_index
        )

    # ascending column -> BoundaryOrder.ASCENDING (1); no null pages
    assert_equal(cix.boundary_order, 1)
    # global bounds
    assert_equal(_le_i64(cix.min_values[0]), 0)
    assert_equal(_le_i64(cix.max_values[npages - 1]), 9999)
    # value == row index here, so each page's min == its first_row_index
    for i in range(npages):
        assert_false(cix.null_pages[i])
        assert_equal(
            _le_i64(cix.min_values[i]), oi.page_locations[i].first_row_index
        )
    remove(path)


def test_marrow_file_has_page_index() raises:
    # marrow's writer now emits an OffsetIndex + ColumnIndex per column chunk
    var pa = Python.import_module("pyarrow")
    var tbl = _col(
        pa.array(Python.list(5, 1, Python.none(), 9, 3), type=pa.int64())
    )
    var t = CArrowArrayStream.from_pycapsule(
        tbl.__arrow_c_stream__(Python.none())
    ).to_table()
    var path = String("/tmp/marrow_pageidx.parquet")
    write_table(t, path)
    var pi = read_page_index(path)
    assert_equal(len(pi), 1)
    assert_true(Bool(pi[0][0].offset_index))
    assert_true(Bool(pi[0][0].column_index))
    # one data page covering all rows, with the correct bounds and null count
    var pb = read_page_bounds(path)
    assert_equal(len(pb[0][0]), 1)
    var pg = pb[0][0][0].copy()
    assert_equal(pg.num_rows, 5)
    assert_equal(pg.min.value().as_int64().value(), 1)
    assert_equal(pg.max.value().as_int64().value(), 9)
    remove(path)


def test_marrow_page_index_pyarrow_reads() raises:
    # a marrow-written page index is spec-valid: PyArrow prunes with it and
    # returns exactly the matching rows.
    var pa = Python.import_module("pyarrow")
    var ds = Python.import_module("pyarrow.dataset")
    var ints = Python.list()
    for i in range(300):
        ints.append(i)
    var caps = pa.table(
        Python.dict(i=pa.array(ints, type=pa.int32()))
    ).__arrow_c_stream__(Python.none())
    var t = CArrowArrayStream.from_pycapsule(caps).to_table()
    var path = String("/tmp/marrow_pageidx_pa.parquet")
    write_table(t, path, use_dictionary=False)
    var d = ds.dataset(path, format="parquet")
    var got = d.to_table(filter=(ds.field("i") >= 295))
    assert_equal(Int(py=got.num_rows), 5)
    assert_equal(Int(py=got.column(0)[0]), 295)
    remove(path)


def test_read_temporal_minmax() raises:
    # timestamp (INT64) + date32 (INT32) bounds decode to typed scalars
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(
            pa.array(
                Python.list(10, 3, 7, Python.none(), 1),
                type=pa.timestamp("us"),
            )
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(Bool(cs.min))
    assert_equal(cs.min.value().as_timestamp().value(), Int64(1))
    assert_equal(cs.max.value().as_timestamp().value(), Int64(10))
    remove(path)

    var dpath = _write_pa(
        _col(pa.array(Python.list(10, 3, 7), type=pa.date32())),
        use_dictionary=False,
    )
    ref ds = read_statistics(dpath)[0][0]
    assert_equal(ds.min.value().as_date32().value(), Int32(3))
    assert_equal(ds.max.value().as_date32().value(), Int32(10))
    remove(dpath)


def test_read_decimal_minmax() raises:
    # decimal128 (big-endian FIXED_LEN_BYTE_ARRAY) bounds decode to the unscaled
    # integer (-3.25 -> -325, 2.00 -> 200 at scale 2)
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(
            pa.array(Python.list("1.50", "-3.25", "2.00")).cast(
                pa.decimal128(5, 2)
            )
        ),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(Bool(cs.min))
    assert_equal(Int(cs.min.value().as_decimal128().value()), -325)
    assert_equal(Int(cs.max.value().as_decimal128().value()), 200)
    remove(path)


def test_read_fixed_size_binary_minmax() raises:
    # fixed_size_binary bounds decode to a FixedSizeBinaryScalar of raw bytes
    var pa = Python.import_module("pyarrow")
    var np = Python.import_module("numpy")
    var vals = np.array(Python.list("yy", "aa", "mm")).astype("S2")
    var path = _write_pa(
        _col(pa.array(vals, type=pa.binary(2))),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(Bool(cs.min))
    assert_true(
        cs.min.value().as_fixed_size_binary().value() == [UInt8(97), UInt8(97)]
    )
    assert_true(
        cs.max.value().as_fixed_size_binary().value()
        == [UInt8(121), UInt8(121)]
    )
    remove(path)


def test_read_float16_minmax() raises:
    # float16 (FIXED_LEN_BYTE_ARRAY(2)) bounds decode to a Float16 scalar
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(pa.array(Python.list(1.5, 2.5, 3.0, 4.5), type=pa.float16())),
        use_dictionary=False,
    )
    ref cs = read_statistics(path)[0][0]
    assert_true(Bool(cs.min))
    assert_true(cs.min.value().as_float16().value() == Float16(1.5))
    assert_true(cs.max.value().as_float16().value() == Float16(4.5))
    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
