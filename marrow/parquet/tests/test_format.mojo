# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The Parquet format layer: the Thrift Compact Protocol codec and the file
footer / metadata parsing built on it."""

from std.testing import assert_equal, assert_true, assert_false
from std.python import Python
from std.pathlib import Path
from std.os.path import join
from ...utils.testing import ScratchDir
from ...parquet.reader import ParquetFile, read_page_index
from ...io import BufferSource
from ...errors import ArrowError, CorruptError, DynError, NotImplementedError
from ...parquet.schema import MAX_SCHEMA_DEPTH, SchemaMapping
from ...parquet.format import (
    MAX_SKIP_DEPTH,
    Encoding,
    FileMetaData,
    PageHeader,
    PhysicalType,
    Repetition,
    SchemaElement,
    ConvertedType,
    ThriftCompactReader,
    ThriftCompactWriter,
    TC_BYTE,
    TC_I32,
    TC_I64,
    TC_BINARY,
    TC_LIST,
    TC_MAP,
    TC_STOP,
    TC_BOOL_TRUE,
)
from ...codecs import Zigzag


# ---------------------------------------------------------------------------
# Thrift Compact Protocol codec
# ---------------------------------------------------------------------------


def test_zigzag_roundtrip() raises:
    for v in [Int64(0), 1, -1, 2, -2, 63, -64, 2147483647, -2147483648]:
        assert_equal(
            Zigzag.decode_value[DType.int64](Zigzag.encode_value(v)), v
        )


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


def test_skip_list_of_bools() raises:
    """A boolean list element is a byte of its own, unlike a boolean field,
    whose value rides in the field header -- so skipping the list has to
    consume one byte per element to land on the next field."""
    var w = ThriftCompactWriter()
    var last = 0
    last = w.write_field_begin(TC_LIST, 1, last)
    w.write_list_begin(TC_BOOL_TRUE, 3)
    w.buf.append(1)
    w.buf.append(2)
    w.buf.append(1)
    _ = w.write_field_begin(TC_BINARY, 2, last)
    w.write_string("tail")
    w.write_field_stop()

    var r = ThriftCompactReader(Span(w.buf))
    var ftype, fid = r.read_field_header(0)
    r.skip(ftype)
    _, fid = r.read_field_header(fid)
    assert_equal(fid, 2)
    assert_equal(r.read_string(), "tail")


def test_thrift_byte_string_with_negative_length_refused() raises:
    """A length varint past `Int`'s range reads as negative, which a bounds
    check of `pos + n` alone lets through."""
    var w = ThriftCompactWriter()
    w.write_varint(UInt64(1) << 63)
    w.write_string("tail")
    var r = ThriftCompactReader(Span(w.buf))
    var refused = False
    try:
        _ = r.read_bytes()
    except e:
        refused = True
        assert_true("byte string exceeds input" in String(e), String(e))
    assert_true(refused, "a negative byte string length was accepted")


def test_thrift_list_larger_than_its_input_refused() raises:
    """Every element takes at least a byte, so a list or map claiming more
    elements than bytes left is refused before its loop starts."""
    var fits = ThriftCompactWriter()
    fits.write_list_begin(TC_BYTE, 20)
    for i in range(20):
        fits.buf.append(UInt8(i))
    var r = ThriftCompactReader(Span(fits.buf))
    r.skip(TC_LIST)
    assert_equal(r.pos, len(fits.buf))

    var big_list = ThriftCompactWriter()
    big_list.write_list_begin(TC_BYTE, 21)
    for i in range(20):
        big_list.buf.append(UInt8(i))
    var refused = False
    try:
        var lr = ThriftCompactReader(Span(big_list.buf))
        _ = lr.read_list_header()
    except e:
        refused = True
        assert_true("list of 21 elements" in String(e), String(e))
    assert_true(refused, "an oversized list was accepted")

    var big_map = ThriftCompactWriter()
    big_map.write_varint(1_000_000)
    big_map.buf.append((TC_BYTE << 4) | TC_BYTE)
    refused = False
    try:
        var mr = ThriftCompactReader(Span(big_map.buf))
        mr.skip(TC_MAP)
    except e:
        refused = True
        assert_true("map of 1000000 entries" in String(e), String(e))
    assert_true(refused, "an oversized map was accepted")


def _nested_lists(depth: Int) -> List[UInt8]:
    """`depth` lists, each the single element of the one around it."""
    var w = ThriftCompactWriter()
    for _ in range(depth - 1):
        w.write_list_begin(TC_LIST, 1)
    w.write_list_begin(TC_LIST, 0)
    return w.buf.copy()


def test_thrift_skip_depth_bounded() raises:
    """Skipping recurses once per nesting level, so a forged value nesting
    thousands deep would overflow the stack; past `MAX_SKIP_DEPTH` levels it
    is refused instead."""
    var fits = _nested_lists(MAX_SKIP_DEPTH)
    var r = ThriftCompactReader(Span(fits))
    r.skip(TC_LIST)
    assert_equal(r.pos, len(fits))

    var deep = _nested_lists(MAX_SKIP_DEPTH + 1)
    var refused = False
    try:
        var dr = ThriftCompactReader(Span(deep))
        dr.skip(TC_LIST)
    except e:
        refused = True
        assert_true("nest too deeply" in String(e), String(e))
    assert_true(refused, "a value nesting too deeply was skipped")


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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_test_format.parquet")
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
    with ScratchDir() as dir:
        var path = join(dir, "marrow_pageheader_sizes.parquet")
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
        assert_true(
            len(oi.page_locations) > 1, "the fixture needs several pages"
        )

        var start = (
            pf.metadata().row_groups[0].columns[0].meta_data.byte_range()[0]
        )
        var chunk = BufferSource(path)
        for k in range(len(oi.page_locations)):
            ref loc = oi.page_locations[k]
            var read = PageHeader.read_at(
                chunk.read_at(loc.offset, loc.compressed_page_size), 0
            )
            assert_equal(
                read[1] + read[0].compressed_page_size,
                loc.compressed_page_size,
                "page "
                + String(k)
                + ": header + body must be the recorded size",
            )
        _ = start


# ---------------------------------------------------------------------------
# A corrupt footer or page is refused as a `CorruptError`: each case starts
# from a valid file or header and breaks one field
# ---------------------------------------------------------------------------


def _pyarrow_file() raises -> List[UInt8]:
    """A valid four-row file of three columns, written by PyArrow with no
    compression."""
    var data = List[UInt8]()
    with ScratchDir() as dir:
        var path = join(dir, "marrow_corrupt_base.parquet")
        _write_pyarrow(path, "none")
        data = Path(path).read_bytes()
    return data^


def _with_footer(data: List[UInt8], meta: FileMetaData) raises -> List[UInt8]:
    """The file `data` with its footer replaced by `meta`."""
    var end = len(data) - 8 - FileMetaData.footer_length(Span(data))
    var out = List[UInt8](Span(data)[:end])
    meta.write_footer(out)
    return out^


def _num_rows(data: List[UInt8]) raises -> Int:
    return ParquetFile(BufferSource(Span(data))).read().num_rows()


def _read_error[
    E: ArrowError = CorruptError
](data: List[UInt8]) raises -> String:
    """The message reading the file `data` fails with, which must be an
    `E`."""
    try:
        _ = ParquetFile(BufferSource(Span(data))).read()
    except e:
        assert_true(DynError(e).isa[E](), String(e))
        return String(e)
    raise Error("the file was read")


def test_schema_with_too_few_elements_refused() raises:
    """A group declaring more children than the schema has elements left
    would read past the element list."""
    var data = _pyarrow_file()
    var meta = FileMetaData.read_footer(Span(data))
    assert_equal(_num_rows(_with_footer(data, meta)), 4)
    meta.schema[0].num_children = 4
    var msg = _read_error(_with_footer(data, meta))
    assert_true("its groups declare more" in msg, msg)


def test_schema_with_elements_left_over_refused() raises:
    var data = _pyarrow_file()
    var meta = FileMetaData.read_footer(Span(data))
    meta.schema[0].num_children = 2
    var msg = _read_error(_with_footer(data, meta))
    assert_true("schema has 4 elements, its groups declare 3" in msg, msg)


def test_schema_with_negative_children_refused() raises:
    # Checked on the footer: `SchemaElement.write` never writes a count < 1.
    var data = _pyarrow_file()
    var meta = FileMetaData.read_footer(Span(data))
    meta.schema[0].num_children = -1
    var refused = False
    try:
        _ = SchemaMapping.from_parquet(meta)
    except e:
        refused = True
        assert_true(DynError(e).isa[CorruptError](), String(e))
        assert_true("negative number of children" in String(e), String(e))
    assert_true(refused, "the schema was accepted")


def _element(
    name: String, repetition: Repetition, num_children: Int = 0
) -> SchemaElement:
    """A group of `num_children`, or an `int32` leaf when it has none."""
    var el = SchemaElement()
    el.name = name
    el.repetition_type = repetition
    el.num_children = num_children
    if num_children == 0:
        el.type = PhysicalType.INT32
    return el^


def _with_schema(
    data: List[UInt8], var elements: List[SchemaElement]
) raises -> List[UInt8]:
    """The file `data` re-footered as zero rows of the schema whose root has
    `elements` below it, the first one its only child."""
    var meta = FileMetaData.read_footer(Span(data))
    var root = meta.schema[0].copy()
    root.num_children = 1
    meta.schema = [root^]
    meta.schema.extend(elements^)
    meta.num_rows = 0
    meta.row_groups.clear()
    meta.key_value_metadata.clear()  # its ARROW:schema names other columns
    return _with_footer(data, meta)


def _nested_groups(data: List[UInt8], groups: Int) raises -> List[UInt8]:
    """One `int32` leaf under `groups` nested required groups."""
    var elements = List[SchemaElement]()
    for _ in range(groups):
        elements.append(_element("g", Repetition.REQUIRED, num_children=1))
    elements.append(_element("x", Repetition.REQUIRED))
    return _with_schema(data, elements^)


def test_schema_depth_limit() raises:
    """Depth counts as Arrow C++ counts it, the root at 1: 98 groups under
    the root put the leaf at `MAX_SCHEMA_DEPTH`, and one more is refused."""
    var data = _pyarrow_file()
    assert_equal(_num_rows(_nested_groups(data, MAX_SCHEMA_DEPTH - 2)), 0)
    var msg = _read_error(_nested_groups(data, MAX_SCHEMA_DEPTH - 1))
    assert_true("nests deeper than" in msg, msg)


def test_two_level_list_not_supported() raises:
    """A list whose repeated field is the element itself is the legacy
    two-level form: valid, and not something to blame on the file."""
    var data = _pyarrow_file()
    var three_level: List[SchemaElement] = [
        _element("a", Repetition.OPTIONAL, num_children=1),
        _element("list", Repetition.REPEATED, num_children=1),
        _element("element", Repetition.OPTIONAL),
    ]
    three_level[0].converted_type = ConvertedType.LIST
    assert_equal(_num_rows(_with_schema(data, three_level^)), 0)

    var two_level: List[SchemaElement] = [
        _element("a", Repetition.OPTIONAL, num_children=1),
        _element("array", Repetition.REPEATED),
    ]
    two_level[0].converted_type = ConvertedType.LIST
    var msg = _read_error[NotImplementedError](_with_schema(data, two_level^))
    assert_true("two-level list encoding (column 'a')" in msg, msg)


def test_row_group_missing_a_column_chunk_refused() raises:
    var data = _pyarrow_file()
    var meta = FileMetaData.read_footer(Span(data))
    _ = meta.row_groups[0].columns.pop()
    var msg = _read_error(_with_footer(data, meta))
    assert_true("2 column chunks, the schema 3 leaf columns" in msg, msg)


def _page(ph: PageHeader, body_len: Int) -> List[UInt8]:
    """`ph` serialized, then a body of `body_len` zero bytes."""
    var w = ThriftCompactWriter()
    ph.write(w)
    var out = w.buf.copy()
    for _ in range(body_len):
        out.append(0)
    return out^


def _header_error(ph: PageHeader, body_len: Int) raises -> String:
    """The message `PageHeader.read_at` refuses `ph` with, followed by
    `body_len` bytes of body."""
    var data = _page(ph, body_len)
    try:
        _ = PageHeader.read_at(Span(data), 0)
    except e:
        return String(e)
    raise Error("the page header was accepted")


def test_page_header_without_its_sub_header_refused() raises:
    var data = PageHeader.data_page(
        uncompressed_size=8, compressed_size=8, num_values=1
    )
    _ = PageHeader.read_at(Span(_page(data, 8)), 0)
    data.data_page_header = None
    assert_true("data page without" in _header_error(data, 8))

    var v2 = PageHeader.data_page_v2(
        uncompressed_size=8,
        compressed_size=8,
        num_values=1,
        num_nulls=0,
        num_rows=1,
        def_levels_byte_length=2,
        is_compressed=False,
    )
    _ = PageHeader.read_at(Span(_page(v2, 8)), 0)
    v2.data_page_header_v2 = None
    assert_true("data page v2 without" in _header_error(v2, 8))

    var dictionary = PageHeader.dictionary_page(
        uncompressed_size=8, compressed_size=8, num_values=1
    )
    _ = PageHeader.read_at(Span(_page(dictionary, 8)), 0)
    dictionary.dictionary_page_header = None
    assert_true("dictionary page without" in _header_error(dictionary, 8))


def test_page_header_sizes_checked_against_the_chunk() raises:
    """`read_at` is handed the column chunk, so a header whose body runs past
    it, or whose sizes and counts are negative, is refused there."""
    var page = PageHeader.data_page(
        uncompressed_size=8, compressed_size=8, num_values=1
    )
    assert_equal(
        PageHeader.read_at(Span(_page(page, 8)), 0)[0].compressed_page_size, 8
    )
    assert_true("runs past its column chunk" in _header_error(page, 7))
    page.compressed_page_size = -1
    assert_true("negative size" in _header_error(page, 8))
    page.compressed_page_size = 8
    page.uncompressed_page_size = -1
    assert_true("negative size" in _header_error(page, 8))
    page.uncompressed_page_size = 8
    page.data_page_header.value().num_values = -1
    assert_true("negative value count" in _header_error(page, 8))


def test_page_header_v2_levels_checked_against_the_body() raises:
    var page = PageHeader.data_page_v2(
        uncompressed_size=8,
        compressed_size=8,
        num_values=4,
        num_nulls=1,
        num_rows=4,
        def_levels_byte_length=6,
        is_compressed=False,
        rep_levels_byte_length=2,
    )
    _ = PageHeader.read_at(Span(_page(page, 8)), 0)
    page.data_page_header_v2.value().definition_levels_byte_length = 7
    assert_true("levels run past the page body" in _header_error(page, 8))
    page.data_page_header_v2.value().definition_levels_byte_length = -1
    assert_true("negative level length" in _header_error(page, 8))
    page.data_page_header_v2.value().definition_levels_byte_length = 6
    page.data_page_header_v2.value().num_nulls = 5
    assert_true("5 nulls in 4 values" in _header_error(page, 8))
    page.data_page_header_v2.value().num_nulls = 1
    page.data_page_header_v2.value().num_rows = -1
    assert_true("negative row count" in _header_error(page, 8))


def test_v1_level_length_past_the_page_refused() raises:
    """A v1 data page opens with its definition levels' 4-byte length; one
    longer than the page would slice past the body."""
    var data = _pyarrow_file()
    var meta = FileMetaData.read_footer(Span(data))
    var page = meta.row_groups[0].columns[0].meta_data.data_page_offset
    var levels = page + PageHeader.read_at(Span(data), page)[1]
    for i in range(4):
        data[levels + i] = 0x7F
    var msg = _read_error(data)
    assert_true("run past the page body" in msg, msg)


def test_row_group_with_negative_rows_refused() raises:
    var data = _pyarrow_file()
    var meta = FileMetaData.read_footer(Span(data))
    meta.row_groups[0].num_rows = -1
    var msg = _read_error(_with_footer(data, meta))
    assert_true("row group 0 has -1 rows" in msg, msg)
