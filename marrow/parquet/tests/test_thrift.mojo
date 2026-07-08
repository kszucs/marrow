from std.testing import assert_equal, assert_true, assert_false
from marrow.testing import TestSuite
from marrow.parquet.thrift import (
    CompactReader,
    CompactWriter,
    zigzag_encode,
    zigzag_decode,
    TC_I32,
    TC_I64,
    TC_BINARY,
    TC_LIST,
    TC_STOP,
)


def test_zigzag_roundtrip() raises:
    for v in [Int64(0), 1, -1, 2, -2, 63, -64, 2147483647, -2147483648]:
        assert_equal(zigzag_decode(zigzag_encode(v)), v)


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


def main() raises:
    TestSuite.run[__functions_in_module()]()
