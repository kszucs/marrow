"""Native Arrow → Parquet writer.

`FileWriter` orchestrates the file: header magic, one `RowGroup` per slice of the
table (bounded by `row_group_size`), then the footer. `ColumnChunkWriter` owns
one leaf column chunk — encoding a v1 data page (RLE definition levels + PLAIN
values), compressing it, and producing the `ColumnMetaData` (with a null-count
statistic). Value encoding is dispatched by Arrow type to typed methods, so the
writer mirrors the reader's structure.

Milestone: flat columns + struct; primitives (incl. int8/16 widened to INT32),
string/binary; PLAIN values; UNCOMPRESSED/SNAPPY/ZSTD. Dictionary encoding, page
splitting, temporal/list writing, and min/max statistics are follow-ups.
"""

from std.sys import size_of
from std.pathlib import Path

from ..arrays import AnyArray, PrimitiveArray, StringArray, BoolArray
from ..dtypes import (
    NumericType,
    Int8Type,
    Int16Type,
    UInt8Type,
    UInt16Type,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    bool_,
)
from ..tabular import Table, RecordBatch
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

comptime DEFAULT_ROW_GROUP_SIZE: Int = 1 << 20


def _append_u32le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


# ---------------------------------------------------------------------------
# ColumnChunkWriter — encode one leaf column chunk into the output buffer
# ---------------------------------------------------------------------------


struct ColumnChunkWriter(Movable):
    var leaf: LeafColumn
    var compression: Int

    def __init__(out self, var leaf: LeafColumn, compression: Int):
        self.leaf = leaf^
        self.compression = compression

    def _plain[
        store: NumericType, phys: DType
    ](self, arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        comptime W = size_of[Scalar[phys]]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var bytes = (
                    arr[i].value().cast[phys]().as_bytes[big_endian=False]()
                )
                for b in range(W):
                    out.append(bytes[b])

    def _plain_bool(self, arr: BoolArray, mut out: List[UInt8]) raises:
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

    def _plain_bytes(self, arr: StringArray, mut out: List[UInt8]) raises:
        for i in range(arr.length):
            if arr.is_valid(i):
                var b = String(arr[i]).as_bytes()
                _append_u32le(out, len(b))
                out.extend(b)

    def _encode_values(self, col: AnyArray, mut body: List[UInt8]) raises:
        ref dt = self.leaf.dtype
        if dt == int32:
            self._plain[phys=DType.int32](col.as_int32(), body)
        elif dt == int64:
            self._plain[phys=DType.int64](col.as_int64(), body)
        elif dt == uint32:
            self._plain[phys=DType.uint32](col.as_uint32(), body)
        elif dt == uint64:
            self._plain[phys=DType.uint64](col.as_uint64(), body)
        elif dt == float32:
            self._plain[phys=DType.float32](col.as_float32(), body)
        elif dt == float64:
            self._plain[phys=DType.float64](col.as_float64(), body)
        elif dt == int8:
            self._plain[phys=DType.int32](col.as_int8(), body)
        elif dt == int16:
            self._plain[phys=DType.int32](col.as_int16(), body)
        elif dt == uint8:
            self._plain[phys=DType.int32](col.as_uint8(), body)
        elif dt == uint16:
            self._plain[phys=DType.int32](col.as_uint16(), body)
        elif dt == bool_:
            self._plain_bool(col.as_bool(), body)
        elif dt.is_string():
            self._plain_bytes(col.as_string(), body)
        else:
            raise Error("parquet: cannot write column type " + String(dt))

    def _encode_body(
        self, col: AnyArray, mut null_count: Int
    ) raises -> List[UInt8]:
        """v1 data page body: RLE definition levels (if nullable) then PLAIN
        values. Also reports the null count."""
        var n = col.length()
        var body = List[UInt8]()
        null_count = 0
        if self.leaf.max_def >= 1:
            var defs = List[Int32]()
            for i in range(n):
                if col.is_valid(i):
                    defs.append(Int32(1))
                else:
                    defs.append(Int32(0))
                    null_count += 1
            var enc = rle_encode(defs, 1)
            _append_u32le(body, len(enc))
            body.extend(Span(enc))
        self._encode_values(col, body)
        return body^

    def write(
        self, col: AnyArray, mut out: List[UInt8], mut codecs: Codecs
    ) raises -> ColumnChunk:
        var n = col.length()
        var null_count = 0
        var body = self._encode_body(col, null_count)
        var uncompressed_size = len(body)
        var compressed = codecs.compress(self.compression, Span(body))
        var compressed_size = len(compressed)

        var page_offset = len(out)
        var hw = CompactWriter()
        _write_data_page_header(hw, uncompressed_size, compressed_size, n)
        var header_len = len(hw.buf)
        out.extend(Span(hw.buf))
        out.extend(Span(compressed))

        var meta = ColumnMetaData()
        meta.type = self.leaf.physical
        meta.path_in_schema.append(self.leaf.name)
        meta.codec = self.compression
        meta.num_values = n
        meta.total_uncompressed_size = uncompressed_size + header_len
        meta.total_compressed_size = compressed_size + header_len
        meta.data_page_offset = page_offset
        if self.leaf.max_def >= 1:
            meta.null_count = null_count

        var cc = ColumnChunk()
        cc.file_offset = page_offset
        cc.meta_data = meta^
        return cc^


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
# FileWriter — header, row groups, footer
# ---------------------------------------------------------------------------


struct FileWriter(Movable):
    var out: List[UInt8]
    var codecs: Codecs
    var compression: Int
    var leaves: List[LeafColumn]

    def __init__(out self, compression: Int):
        self.out = List[UInt8]()
        self.out.extend(String("PAR1").as_bytes())  # header magic
        self.codecs = Codecs()
        self.compression = compression
        self.leaves = List[LeafColumn]()

    def _write_row_group(
        mut self, batch: RecordBatch, nodes: List[SchemaNode]
    ) raises -> RowGroup:
        var leaf_arrays = List[AnyArray]()
        for ci in range(len(nodes)):
            nodes[ci].flatten(batch.columns[ci], leaf_arrays)

        var columns = List[ColumnChunk]()
        var total = 0
        for ci in range(len(self.leaves)):
            var ccw = ColumnChunkWriter(
                self.leaves[ci].copy(), self.compression
            )
            var cc = ccw.write(leaf_arrays[ci], self.out, self.codecs)
            total += cc.meta_data.total_uncompressed_size
            columns.append(cc^)

        var rg = RowGroup()
        rg.total_byte_size = total
        rg.num_rows = batch.num_rows()
        rg.columns = columns^
        return rg^

    def write(
        mut self,
        table: Table,
        path: String,
        row_group_size: Int = DEFAULT_ROW_GROUP_SIZE,
    ) raises:
        var batch = table.combine_chunks()
        var n = batch.num_rows()

        var nodes = List[SchemaNode]()
        var elems = arrow_to_parquet(table.schema, self.leaves, nodes)

        var fmeta = FileMetaData()
        fmeta.version = 1
        fmeta.schema = elems^
        fmeta.num_rows = n
        fmeta.created_by = "marrow"

        var start = 0
        while start < n or (n == 0 and start == 0):
            var length = min(row_group_size, n - start)
            var slice = batch.slice(start, length)
            fmeta.row_groups.append(self._write_row_group(slice, nodes))
            start += length
            if length == 0:
                break

        # PLAIN encoding for every column in every row group
        var encodings = List[Int]()
        for _ in range(len(self.leaves)):
            encodings.append(ENC_PLAIN)
        var rg_encodings = List[List[Int]]()
        for _ in range(len(fmeta.row_groups)):
            rg_encodings.append(encodings.copy())

        var mw = CompactWriter()
        fmeta.write(mw, rg_encodings)
        var meta_len = len(mw.buf)
        self.out.extend(Span(mw.buf))
        _append_u32le(self.out, meta_len)
        self.out.extend(String("PAR1").as_bytes())

        Path(path).write_bytes(Span(self.out))


def write_table(
    table: Table, path: String, compression: Int = CODEC_SNAPPY
) raises:
    """Write a Marrow `Table` to a Parquet file."""
    var writer = FileWriter(compression)
    writer.write(table, path)
