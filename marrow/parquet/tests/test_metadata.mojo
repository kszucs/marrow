"""Reading file metadata. Column statistics: `read_metadata` (raw footer) and
`ParquetFile.statistics()` (decoded typed min/max) — marrow reads the bounds PyArrow
writes, and round-trips its own (the write side is covered against PyArrow in
test_writer.mojo). Page index: `read_page_index` parses the OffsetIndex +
ColumnIndex PyArrow writes for a multi-page column, and the per-page bounds must
match the (sorted) data; marrow's own writer emits no page index yet, so those
chunks read back absent."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python, PythonObject
from std.os import remove
from std.pathlib import Path
from ... import dtypes as dt
from ...parquet import (
    read_metadata,
    ParquetFile,
    read_page_index,
    write_table,
    ColumnStatistics,
)
from ...parquet.format import ColumnMetaData, PhysicalType
from ...parquet.schema import LeafColumn
from ...parquet.statistics import Statistics
from ...tabular import Table
from ...c_data import CArrowArrayStream


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
    var st = ParquetFile(path).statistics()
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
    ref cs = ParquetFile(path).statistics()[0][0]
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
    ref cs = ParquetFile(path).statistics()[0][0]
    assert_true(cs.min.value().as_float64().value() == -2.5)
    assert_true(cs.max.value().as_float64().value() == 3.25)
    remove(path)


def test_read_string_minmax() raises:
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(pa.array(Python.list("banana", "apple", "cherry", Python.none()))),
        use_dictionary=False,
    )
    ref cs = ParquetFile(path).statistics()[0][0]
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
    var st = ParquetFile(path).statistics()
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

    var st = ParquetFile(path).statistics()
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
    var pb = ParquetFile(path).page_bounds()
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
    ref cs = ParquetFile(path).statistics()[0][0]
    assert_true(Bool(cs.min))
    assert_equal(cs.min.value().as_timestamp().value(), Int64(1))
    assert_equal(cs.max.value().as_timestamp().value(), Int64(10))
    remove(path)

    var dpath = _write_pa(
        _col(pa.array(Python.list(10, 3, 7), type=pa.date32())),
        use_dictionary=False,
    )
    ref ds = ParquetFile(dpath).statistics()[0][0]
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
    ref cs = ParquetFile(path).statistics()[0][0]
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
    ref cs = ParquetFile(path).statistics()[0][0]
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
    ref cs = ParquetFile(path).statistics()[0][0]
    assert_true(Bool(cs.min))
    assert_true(cs.min.value().as_float16().value() == Float16(1.5))
    assert_true(cs.max.value().as_float16().value() == Float16(4.5))
    remove(path)


# ---------------------------------------------------------------------------
# Statistics a reader must refuse. Every case below yields an *absent* bound,
# which prunes nothing — the only safe answer when the stored bytes do not mean
# what the column's Arrow type says they mean.
# ---------------------------------------------------------------------------


def test_stats_decode_rejects_wrong_width() raises:
    # `LittleEndian.fixed` is not bounds-checked, so a bound whose byte length
    # does not match the physical width would reinterpret adjacent heap bytes as
    # the bound. An empty one is what a null page stores.
    var leaf = LeafColumn(String("c"), dt.int64, PhysicalType.INT64, 1, 0)
    var short_bound: List[UInt8] = [1, 2, 3, 4]
    assert_false(Bool(Statistics.decode(leaf, short_bound)))
    assert_false(Bool(Statistics.decode(leaf, List[UInt8]())))
    var long_bound: List[UInt8] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert_false(Bool(Statistics.decode(leaf, long_bound)))
    # the exact width still decodes
    var exact: List[UInt8] = [7, 0, 0, 0, 0, 0, 0, 0]
    assert_equal(Statistics.decode(leaf, exact).value().as_int64().value(), 7)

    # BOOLEAN is one byte, FLBA is its declared width
    var bl = LeafColumn(String("b"), dt.bool_, PhysicalType.BOOLEAN, 1, 0)
    assert_false(Bool(Statistics.decode(bl, List[UInt8]())))
    var two: List[UInt8] = [1, 0]
    assert_false(Bool(Statistics.decode(bl, two)))
    var fl = LeafColumn(
        String("f"),
        dt.fixed_size_binary_(2),
        PhysicalType.FIXED_LEN_BYTE_ARRAY,
        1,
        0,
        type_length=2,
    )
    var one: List[UInt8] = [1]
    assert_false(Bool(Statistics.decode(fl, one)))


def test_stats_decode_rejects_int96() raises:
    # INT96's ColumnOrder is UNDEFINED per parquet-format, and its 12-byte value
    # is (nanoseconds-in-day, Julian day) — the low 8 bytes are ~4 orders of
    # magnitude below the epoch nanoseconds the Arrow dtype claims. The 8-byte
    # case passes the width check, so only the INT96 rule can reject it.
    var leaf = LeafColumn(
        String("ts"),
        dt.timestamp(dt.nanosecond, String("")),
        PhysicalType.INT96,
        1,
        0,
    )
    var eight: List[UInt8] = [0, 0, 0, 0, 0, 0, 0, 0]
    assert_false(Bool(Statistics.decode(leaf, eight)))
    var twelve: List[UInt8] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert_false(Bool(Statistics.decode(leaf, twelve)))


def test_stats_decode_rejects_nan_bound() raises:
    # a NaN orders nothing, so a NaN bound is not a bound — and it taints its
    # partner, since a writer that emitted one ordered the column wrongly
    var leaf = LeafColumn(String("f"), dt.float64, PhysicalType.DOUBLE, 1, 0)
    var qnan: List[UInt8] = [0, 0, 0, 0, 0, 0, 0xF8, 0x7F]
    var one_and_a_half: List[UInt8] = [0, 0, 0, 0, 0, 0, 0xF8, 0x3F]
    assert_false(Bool(Statistics.decode(leaf, qnan)))
    assert_true(
        Statistics.decode(leaf, one_and_a_half).value().as_float64().value()
        == 1.5
    )

    var cm = ColumnMetaData()
    cm.has_min_max = True
    cm.min_value = qnan.copy()
    cm.max_value = one_and_a_half.copy()
    var cs = ColumnStatistics.from_metadata(leaf, cm)
    assert_false(Bool(cs.min))
    assert_false(Bool(cs.max))


def test_stats_absent_for_nested_leaves() raises:
    # A leaf under a list or a struct is named by its bare Parquet element name
    # ("element", "x"), and under a repeated field its null_count and bounds
    # count *elements* while num_rows counts rows. Neither bounds anything a
    # row-level predicate names, so the whole statistic must read as absent.
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            l=pa.array(
                Python.evaluate("[[None, 5, 7], [1, 2, 3]]"),
                type=pa.list_(pa.int64()),
            ),
            s=pa.array(
                Python.evaluate("[{'x': 1}, {'x': 9}]"),
                type=pa.struct(Python.list(pa.field("x", pa.int64()))),
            ),
        )
    )
    var path = String("/tmp/marrow_nested_stats.parquet")
    pq.write_table(
        tbl,
        path,
        compression="none",
        use_dictionary=False,
        write_page_index=True,
    )
    var pf = ParquetFile(path)
    var st = pf.statistics()
    assert_equal(len(st[0]), 2)  # l.list.element and s.x
    var pb = pf.page_bounds()
    for ci in range(2):
        assert_false(Bool(st[0][ci].min))
        assert_false(Bool(st[0][ci].max))
        assert_equal(st[0][ci].null_count, -1)
        for p in range(len(pb[0][ci])):
            assert_false(Bool(pb[0][ci][p].copy().min))
            assert_false(Bool(pb[0][ci][p].copy().max))
    # a flat column in the same file keeps its statistics
    var flat = _write_pa(
        _col(pa.array(Python.list(5, 1, 9), type=pa.int64())),
        use_dictionary=False,
    )
    assert_true(Bool(ParquetFile(flat).statistics()[0][0].min))
    remove(path)
    remove(flat)


def test_stats_absent_without_type_defined_order() raises:
    # "Without column_orders, the meaning of the min_value and max_value fields
    # is undefined" (parquet-format). The footer's field 7 is a list of
    # ColumnOrder unions whose only defined member is field 1, TypeDefinedOrder.
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(pa.array(Python.list(5, 1, 9), type=pa.int64())),
        use_dictionary=False,
    )
    var meta = ParquetFile(path).metadata()
    assert_equal(len(meta.column_orders), 1)
    assert_true(meta.column_orders[0])
    assert_true(Bool(ParquetFile(path).statistics()[0][0].min))

    # The footer's last six bytes are the field-7 header, a one-element list
    # header, the union's `TypeDefinedOrder` field header, and three struct
    # stops. Retagging that field header from id 1 to an unknown id 2 leaves the
    # ordering undefined without changing any length.
    var raw = Path(path).read_bytes()
    assert_equal(raw[len(raw) - 12], UInt8(0x1C))
    var patched = String("/tmp/marrow_no_colorder.parquet")
    _ = Python.evaluate(
        "(lambda s, d: __import__('pathlib').Path(d).write_bytes("
        "(lambda b: b[:-12] + bytes([0x2C]) + b[-11:])("
        "__import__('pathlib').Path(s).read_bytes())))('"
        + path
        + "', '"
        + patched
        + "')"
    )
    var pf = ParquetFile(patched)
    assert_equal(len(pf.metadata().column_orders), 1)
    assert_false(pf.metadata().column_orders[0])
    var st = pf.statistics()
    assert_false(Bool(st[0][0].min))
    assert_false(Bool(st[0][0].max))
    remove(path)
    remove(patched)


def test_string_bound_high_byte_ordering() raises:
    # Parquet BYTE_ARRAY bounds are *unsigned* byte-wise lexicographic. "é" is
    # UTF-8 0xC3 0xA9 and "z" is 0x7A, so unsigned puts "z" < "é" while a signed
    # byte comparison inverts it. Records which order marrow's own String
    # comparison uses, since that is what a decoded bound is compared with.
    var pa = Python.import_module("pyarrow")
    var path = _write_pa(
        _col(pa.array(Python.list("a", "z", "é"))), use_dictionary=False
    )
    ref cs = ParquetFile(path).statistics()[0][0]
    assert_equal(cs.min.value().as_string().to_string(), "a")
    assert_equal(cs.max.value().as_string().to_string(), "é")
    assert_true(String("z") < String("é"))
    assert_true(StringSlice("z") < StringSlice("é"))
    remove(path)
