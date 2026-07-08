"""Native Arrow → Parquet writer.

Milestone-1 writes flat columns with PLAIN value encoding and RLE definition
levels, one v1 data page per column, a single row group, and optional page
compression. The output is a spec-compliant file that PyArrow reads back.
"""

from std.sys import size_of
from std.pathlib import Path

from ..arrays import AnyArray, PrimitiveArray, StringArray, BoolArray
from ..dtypes import (
    NumericType,
    int32,
    int64,
    uint32,
    uint64,
    float32,
    float64,
    bool_,
)
from ..tabular import Table
from ..schema import Schema

from .thrift import CompactWriter, TC_I32, TC_STRUCT
from .encoding import rle_encode
from .compression import Codecs, CODEC_SNAPPY
from .schema import arrow_to_parquet, LeafColumn, SchemaNode
from .format import (
    DataPageHeader,
    ColumnMetaData,
    ColumnChunk,
    RowGroup,
    FileMetaData,
    PAGE_DATA,
    ENC_PLAIN,
    ENC_RLE,
)


def _append_u32le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


# ---------------------------------------------------------------------------
# Value (PLAIN) encoders — emit only the valid (non-null) values
# ---------------------------------------------------------------------------


def _plain_primitive[
    T: NumericType
](arr: PrimitiveArray[T], mut out: List[UInt8]) raises:
    comptime W = size_of[Scalar[T.native]]()
    for i in range(arr.length):
        if arr.is_valid(i):
            var bytes = arr[i].value().as_bytes[big_endian=False]()
            for b in range(W):
                out.append(bytes[b])


def _plain_bool(arr: BoolArray, mut out: List[UInt8]) raises:
    var acc: UInt8 = 0
    var nbits = 0
    for i in range(arr.length):
        if arr.is_valid(i):
            if arr[i].value():
                acc |= UInt8(1) << UInt8(nbits)
            nbits += 1
            if nbits == 8:
                out.append(acc)
                acc = 0
                nbits = 0
    if nbits > 0:
        out.append(acc)


def _plain_string(arr: StringArray, mut out: List[UInt8]) raises:
    for i in range(arr.length):
        if arr.is_valid(i):
            var s = String(arr[i])
            var b = s.as_bytes()
            _append_u32le(out, len(b))
            out.extend(b)


def _encode_column(col: AnyArray, leaf: LeafColumn) raises -> List[UInt8]:
    """Encode a v1 data page body: RLE definition levels (if nullable) followed
    by the PLAIN-encoded present values."""
    var n = col.length()
    var body = List[UInt8]()

    if leaf.max_def >= 1:
        var defs = List[Int32]()
        for i in range(n):
            defs.append(Int32(1) if col.is_valid(i) else Int32(0))
        var enc = rle_encode(defs, 1)
        _append_u32le(body, len(enc))
        body.extend(Span(enc))

    ref dt = leaf.dtype
    if dt == int32:
        _plain_primitive(col.as_int32(), body)
    elif dt == int64:
        _plain_primitive(col.as_int64(), body)
    elif dt == uint32:
        _plain_primitive(col.as_uint32(), body)
    elif dt == uint64:
        _plain_primitive(col.as_uint64(), body)
    elif dt == float32:
        _plain_primitive(col.as_float32(), body)
    elif dt == float64:
        _plain_primitive(col.as_float64(), body)
    elif dt == bool_:
        _plain_bool(col.as_bool(), body)
    elif dt.is_string():
        _plain_string(col.as_string(), body)
    else:
        raise Error("parquet: cannot write column type " + String(dt))
    return body^


def _write_data_page_header(
    mut w: CompactWriter, uncompressed: Int, compressed: Int, num_values: Int
):
    var last = 0
    last = w.write_field_begin(TC_I32, 1, last)
    w.write_i32(Int32(PAGE_DATA))
    last = w.write_field_begin(TC_I32, 2, last)
    w.write_i32(Int32(uncompressed))
    last = w.write_field_begin(TC_I32, 3, last)
    w.write_i32(Int32(compressed))
    _ = w.write_field_begin(TC_STRUCT, 5, last)
    var dph = DataPageHeader()
    dph.num_values = num_values
    dph.encoding = ENC_PLAIN
    dph.definition_level_encoding = ENC_RLE
    dph.repetition_level_encoding = ENC_RLE
    dph.write(w)
    w.write_field_stop()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def write_table(
    table: Table, path: String, compression: Int = CODEC_SNAPPY
) raises:
    """Write a Marrow `Table` to a Parquet file."""
    var batch = table.combine_chunks()
    var n = batch.num_rows()

    var leaves = List[LeafColumn]()
    var nodes = List[SchemaNode]()
    var elems = arrow_to_parquet(table.schema, leaves, nodes)

    var leaf_arrays = List[AnyArray]()
    for ci in range(len(nodes)):
        nodes[ci].flatten(batch.columns[ci], leaf_arrays)

    var out = List[UInt8]()
    out.extend(String("PAR1").as_bytes())  # header magic

    var codecs = Codecs()
    var columns = List[ColumnChunk]()
    var total_byte_size = 0

    for ci in range(len(leaves)):
        var body = _encode_column(leaf_arrays[ci], leaves[ci])
        var uncompressed_size = len(body)
        var compressed = codecs.compress(compression, Span(body))
        var compressed_size = len(compressed)

        var page_offset = len(out)
        var hw = CompactWriter()
        _write_data_page_header(hw, uncompressed_size, compressed_size, n)
        var header_len = len(hw.buf)
        out.extend(Span(hw.buf))
        out.extend(Span(compressed))

        var meta = ColumnMetaData()
        meta.type = leaves[ci].physical
        meta.path_in_schema.append(leaves[ci].name)
        meta.codec = compression
        meta.num_values = n
        meta.total_uncompressed_size = uncompressed_size + header_len
        meta.total_compressed_size = compressed_size + header_len
        meta.data_page_offset = page_offset
        total_byte_size += meta.total_uncompressed_size

        var cc = ColumnChunk()
        cc.file_offset = page_offset
        cc.meta_data = meta^
        columns.append(cc^)

    var rg = RowGroup()
    rg.total_byte_size = total_byte_size
    rg.num_rows = n
    rg.columns = columns^

    var row_group_encodings = List[List[Int]]()
    var enc0 = List[Int]()
    for _ in range(len(leaves)):
        enc0.append(ENC_PLAIN)
    row_group_encodings.append(enc0^)

    var fmeta = FileMetaData()
    fmeta.version = 1
    fmeta.schema = elems^
    fmeta.num_rows = n
    fmeta.row_groups.append(rg^)
    fmeta.created_by = "marrow"

    var mw = CompactWriter()
    fmeta.write(mw, row_group_encodings)
    var meta_len = len(mw.buf)
    out.extend(Span(mw.buf))

    _append_u32le(out, meta_len)
    out.extend(String("PAR1").as_bytes())

    Path(path).write_bytes(Span(out))
