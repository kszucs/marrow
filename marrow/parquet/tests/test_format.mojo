"""The Parquet format layer: the Thrift Compact Protocol codec and the file
footer / metadata parsing built on it."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.pathlib import Path
from std.os import remove
from ...parquet.reader import ParquetFile, read_page_index
from ...parquet.source import MappedFile
from ...parquet.format import (
    Encoding,
    FileMetaData,
    PageHeader,
    PhysicalType,
    Repetition,
    ThriftCompactReader,
    ThriftCompactWriter,
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
    var w = ThriftCompactWriter()
    for v in [UInt64(0), 1, 127, 128, 300, 16384, 1_000_000_000]:
        w.write_varint(v)
    var r = ThriftCompactReader(Span(w.buf))
    for v in [UInt64(0), 1, 127, 128, 300, 16384, 1_000_000_000]:
        assert_equal(r.read_varint(), v)


def test_int_roundtrip() raises:
    var w = ThriftCompactWriter()
    w.write_i32(-12345)
    w.write_i64(9_876_543_210)
    w.write_double(3.14159)
    var r = ThriftCompactReader(Span(w.buf))
    assert_equal(r.read_i32(), Int32(-12345))
    assert_equal(r.read_i64(), Int64(9_876_543_210))
    assert_true(r.read_double() == 3.14159)


def test_string_roundtrip() raises:
    var w = ThriftCompactWriter()
    w.write_string("hello")
    w.write_string("")
    w.write_string("parquet")
    var r = ThriftCompactReader(Span(w.buf))
    assert_equal(r.read_string(), "hello")
    assert_equal(r.read_string(), "")
    assert_equal(r.read_string(), "parquet")


def test_field_header_delta() raises:
    var w = ThriftCompactWriter()
    var last = 0
    last = w.write_field_begin(TC_I32, 1, last)
    w.write_i32(10)
    last = w.write_field_begin(TC_I32, 3, last)
    w.write_i32(20)
    # large jump forces a non-delta (full field id) encoding
    _ = w.write_field_begin(TC_I64, 100, last)
    w.write_i64(30)
    w.write_field_stop()

    var r = ThriftCompactReader(Span(w.buf))
    var rlast = 0
    var ftype: UInt8
    var fid: Int

    _, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 1)
    assert_equal(r.read_i32(), Int32(10))

    _, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 3)
    assert_equal(r.read_i32(), Int32(20))

    _, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 100)
    assert_equal(r.read_i64(), Int64(30))

    ftype, _ = r.read_field_header(rlast)
    assert_equal(ftype, TC_STOP)


def test_list_and_skip() raises:
    # Build a struct-like stream: field 1 = list<i32>[3], field 2 = binary,
    # then STOP. Then reparse skipping field 1.
    var w = ThriftCompactWriter()
    var last = 0
    last = w.write_field_begin(TC_LIST, 1, last)
    w.write_list_begin(TC_I32, 3)
    w.write_i32(7)
    w.write_i32(8)
    w.write_i32(9)
    _ = w.write_field_begin(TC_BINARY, 2, last)
    w.write_string("tail")
    w.write_field_stop()

    var r = ThriftCompactReader(Span(w.buf))
    var rlast = 0
    var ftype: UInt8
    var fid: Int

    ftype, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(ftype, TC_LIST)
    r.skip(ftype)  # skip the whole list

    _, fid = r.read_field_header(rlast)
    rlast = fid
    assert_equal(fid, 2)
    assert_equal(r.read_string(), "tail")

    ftype, _ = r.read_field_header(rlast)
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


# ---------------------------------------------------------------------------
# A page is its header plus its body, and the two `compressed_page_size` fields
# do not mean the same thing
# ---------------------------------------------------------------------------
def test_page_header_reports_its_own_length() raises:
    """`read_at` answers how long the header was, and it is not folded into
    anything else.

    It used to advance a `mut pos` instead. That is invisible at a call site,
    and the caller then had to remember that `compressed_page_size` counts the
    *body* only — so `pos = pos + compressed_page_size` from an already
    advanced `pos` skipped a header's worth of bytes too few and landed inside
    the next page. The symptom was "unexpected page type" from somewhere else
    entirely.
    """
    var out = List[UInt8]()
    for _ in range(7):  # a non-zero start offset, so `pos` is not incidental
        out.append(0)
    var w = ThriftCompactWriter()
    PageHeader.data_page(
        uncompressed_size=400, compressed_size=128, num_values=50
    ).write(w)
    out.extend(Span(w.buf))
    var body = List[UInt8](length=128, fill=UInt8(9))
    out.extend(Span(body))

    var read = PageHeader.read_at(Span(out), 7)
    ref ph = read[0]
    var header_len = read[1]

    assert_equal(header_len, len(w.buf), "the header's own byte length")
    assert_equal(ph.compressed_page_size, 128, "the *body*, not the page")
    assert_equal(ph.data_page_header.value().num_values, 50)
    # The page occupies header + body, and that is what a reader must step by.
    assert_equal(7 + header_len + ph.compressed_page_size, len(out))


def test_page_location_size_covers_the_header_too() raises:
    """The `OffsetIndex`'s `compressed_page_size` includes the page header,
    where the `PageHeader`'s field of the same name does not.

    A seek over the page index steps by the first; a walk through the pages
    steps by the header's length plus the second. Getting the two confused is
    silent — both are plausible byte counts — so the relationship is pinned
    here against a file marrow wrote and read back.
    """
    var pa = Python.import_module("pyarrow")
    var pq = Python.import_module("pyarrow.parquet")
    var vals = Python.list()
    for i in range(300):
        vals.append(i)
    var path = String("/tmp/marrow_pageheader_sizes.parquet")
    pq.write_table(
        pa.table(Python.dict(a=pa.array(vals, type=pa.int64()))),
        path,
        row_group_size=300,
        data_page_size=1,
        write_batch_size=100,
        write_page_index=True,
        use_dictionary=False,
        compression="none",
    )

    var pf = ParquetFile(path)
    var pi = read_page_index(path)
    ref oi = pi[0][0].offset_index.value()
    assert_true(len(oi.page_locations) > 1, "the fixture needs several pages")

    var start = pf.metadata().row_groups[0].columns[0].meta_data.byte_range()[0]
    var chunk = MappedFile(path)
    for k in range(len(oi.page_locations)):
        ref loc = oi.page_locations[k]
        var read = PageHeader.read_at(
            chunk.read_at(loc.offset, loc.compressed_page_size), 0
        )
        assert_equal(
            read[1] + read[0].compressed_page_size,
            loc.compressed_page_size,
            "page " + String(k) + ": header + body must be the recorded size",
        )
    _ = start
    remove(path)
