"""The Parquet format layer: the Thrift Compact Protocol codec and the file
footer / metadata parsing built on it."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.pathlib import Path
from std.os import remove
from marrow.testing import TestSuite
from marrow.parquet.format import (
    FileMetaData,
    PhysicalType,
    Repetition,
    CompactReader,
    CompactWriter,
    Zigzag,
    TC_I32,
    TC_I64,
    TC_BINARY,
    TC_LIST,
    TC_STOP,
)


# ---------------------------------------------------------------------------
# Thrift Compact Protocol codec
# ---------------------------------------------------------------------------


def test_zigzag_roundtrip() raises:
    for v in [Int64(0), 1, -1, 2, -2, 63, -64, 2147483647, -2147483648]:
        assert_equal(Zigzag.decode(Zigzag.encode(v)), v)


def test_varint_roundtrip() raises:
    var w = CompactWriter()
    for v in [UInt64(0), 1, 127, 128, 300, 16384, 1_000_000_000]:
        w.write_varint(v)
    var r = CompactReader(Span(w.buf))
    for v in [UInt64(0), 1, 127, 128, 300, 16384, 1_000_000_000]:
        assert_equal(r.read_varint(), v)


def test_int_roundtrip() raises:
    var w = CompactWriter()
    w.write_i32(-12345)
    w.write_i64(9_876_543_210)
    w.write_double(3.14159)
    var r = CompactReader(Span(w.buf))
    assert_equal(r.read_i32(), Int32(-12345))
    assert_equal(r.read_i64(), Int64(9_876_543_210))
    assert_true(r.read_double() == 3.14159)


def test_string_roundtrip() raises:
    var w = CompactWriter()
    w.write_string("hello")
    w.write_string("")
    w.write_string("parquet")
    var r = CompactReader(Span(w.buf))
    assert_equal(r.read_string(), "hello")
    assert_equal(r.read_string(), "")
    assert_equal(r.read_string(), "parquet")


def test_field_header_delta() raises:
    var w = CompactWriter()
    var last = 0
    last = w.write_field_begin(TC_I32, 1, last)
    w.write_i32(10)
    last = w.write_field_begin(TC_I32, 3, last)
    w.write_i32(20)
    # large jump forces a non-delta (full field id) encoding
    last = w.write_field_begin(TC_I64, 100, last)
    w.write_i64(30)
    w.write_field_stop()

    var r = CompactReader(Span(w.buf))
    var rlast = 0
    var ftype: UInt8
    var fid: Int

    ftype, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 1)
    assert_equal(r.read_i32(), Int32(10))

    ftype, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 3)
    assert_equal(r.read_i32(), Int32(20))

    ftype, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 100)
    assert_equal(r.read_i64(), Int64(30))

    ftype, fid = r.read_field_header(rlast)
    assert_equal(ftype, TC_STOP)


def test_list_and_skip() raises:
    # Build a struct-like stream: field 1 = list<i32>[3], field 2 = binary,
    # then STOP. Then reparse skipping field 1.
    var w = CompactWriter()
    var last = 0
    last = w.write_field_begin(TC_LIST, 1, last)
    w.write_list_begin(TC_I32, 3)
    w.write_i32(7)
    w.write_i32(8)
    w.write_i32(9)
    last = w.write_field_begin(TC_BINARY, 2, last)
    w.write_string("tail")
    w.write_field_stop()

    var r = CompactReader(Span(w.buf))
    var rlast = 0
    var ftype: UInt8
    var fid: Int

    ftype, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(ftype, TC_LIST)
    r.skip(ftype)  # skip the whole list

    ftype, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 2)
    assert_equal(r.read_string(), "tail")

    ftype, fid = r.read_field_header(rlast)
    assert_equal(ftype, TC_STOP)


# ---------------------------------------------------------------------------
# File footer / metadata
# ---------------------------------------------------------------------------


def _write_pyarrow(path: String, compression: String) raises:
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var tbl = pa.table(
        Python.dict(
            x=pa.array(Python.list(1, 2, 3, 4), type=pa.int64()),
            y=pa.array(Python.list(1.5, 2.5, 3.5, 4.5), type=pa.float64()),
            z=pa.array(Python.list("a", "b", "c", "d")),
        )
    )
    pq.write_table(tbl, path, compression=compression)


def test_read_footer_metadata() raises:
    var path = String("/tmp/marrow_test_format.parquet")
    _write_pyarrow(path, "snappy")
    var data = Path(path).read_bytes()
    var meta = FileMetaData.read_footer(Span(data))

    assert_equal(meta.num_rows, 4)
    assert_equal(len(meta.row_groups), 1)
    # schema[0] is the root group; then one leaf per column
    assert_equal(len(meta.schema), 4)
    assert_equal(meta.schema[0].num_children, 3)
    assert_equal(meta.schema[1].name, "x")
    assert_true(meta.schema[1].type == PhysicalType.INT64)
    assert_equal(meta.schema[2].name, "y")
    assert_true(meta.schema[2].type == PhysicalType.DOUBLE)
    assert_equal(meta.schema[3].name, "z")
    assert_true(meta.schema[3].type == PhysicalType.BYTE_ARRAY)
    # pyarrow marks value columns optional (nullable)
    assert_true(meta.schema[1].repetition_type == Repetition.OPTIONAL)

    ref rg = meta.row_groups[0]
    assert_equal(len(rg.columns), 3)
    assert_equal(rg.num_rows, 4)
    assert_equal(rg.columns[0].meta_data.path_in_schema[0], "x")
    assert_equal(rg.columns[0].meta_data.num_values, 4)
    assert_true(rg.columns[0].meta_data.data_page_offset >= 4)

    remove(path)


def main() raises:
    TestSuite.run[__functions_in_module()]()
