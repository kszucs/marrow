"""Native Arrow → Parquet writer.

`FileWriter` orchestrates the file: header magic, one `RowGroup` per slice of the
table (bounded by `row_group_size`), then the footer. `ColumnChunkWriter` owns
one leaf column chunk — encoding a data page (RLE definition levels + PLAIN
values), compressing it, and producing the `ColumnMetaData` (with a null-count
statistic). `version` selects the page format: v1 (levels + values compressed
together) or v2 (levels stored uncompressed ahead of the compressed values).
Value encoding is dispatched by Arrow type to typed methods, so the writer
mirrors the reader's structure.

Milestone: flat columns + struct; primitives (incl. int8/16 widened to INT32),
string/binary; PLAIN values; UNCOMPRESSED/SNAPPY/ZSTD/LZ4_RAW; v1 and v2 data
pages. Dictionary encoding, page splitting, temporal/list writing, and min/max
statistics are follow-ups.
"""

from std.sys import size_of
from std.pathlib import Path

from ..arrays import AnyArray, PrimitiveArray, StringArray, BoolArray
from .. import dtypes as dt
from ..tabular import Table, RecordBatch
from ..schema import Schema

from .encoding import Encoding, rle_encode
from .compression import Compression, CompressionLibs
from .schema import ParquetSchema, LeafColumn, SchemaNode
from .format import (
    PageHeader,
    ColumnMetaData,
    ColumnChunk,
    RowGroup,
    FileMetaData,
)

comptime DEFAULT_ROW_GROUP_SIZE: Int = 1 << 20


# ---------------------------------------------------------------------------
# ColumnChunkWriter — encode one leaf column chunk into the output buffer
# ---------------------------------------------------------------------------


struct ColumnChunkWriter(Movable):
    var leaf: LeafColumn
    var compression: Compression
    var version: Int  # data-page version: 1 or 2

    def __init__(
        out self,
        var leaf: LeafColumn,
        compression: Compression,
        version: Int = 1,
    ):
        self.leaf = leaf^
        self.compression = compression
        self.version = version

    @staticmethod
    def _u32le(mut out: List[UInt8], v: Int):
        out.append(UInt8(v & 0xFF))
        out.append(UInt8((v >> 8) & 0xFF))
        out.append(UInt8((v >> 16) & 0xFF))
        out.append(UInt8((v >> 24) & 0xFF))

    def _plain[
        store: dt.NumericType, phys: DType
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
                Self._u32le(out, len(b))
                out.extend(b)

    def _encode_values(self, col: AnyArray, mut body: List[UInt8]) raises:
        ref vt = self.leaf.dtype
        if vt == dt.int32:
            self._plain[phys=DType.int32](col.as_int32(), body)
        elif vt == dt.int64:
            self._plain[phys=DType.int64](col.as_int64(), body)
        elif vt == dt.uint32:
            self._plain[phys=DType.uint32](col.as_uint32(), body)
        elif vt == dt.uint64:
            self._plain[phys=DType.uint64](col.as_uint64(), body)
        elif vt == dt.float32:
            self._plain[phys=DType.float32](col.as_float32(), body)
        elif vt == dt.float64:
            self._plain[phys=DType.float64](col.as_float64(), body)
        elif vt == dt.int8:
            self._plain[phys=DType.int32](col.as_int8(), body)
        elif vt == dt.int16:
            self._plain[phys=DType.int32](col.as_int16(), body)
        elif vt == dt.uint8:
            self._plain[phys=DType.int32](col.as_uint8(), body)
        elif vt == dt.uint16:
            self._plain[phys=DType.int32](col.as_uint16(), body)
        elif vt == dt.bool_:
            self._plain_bool(col.as_bool(), body)
        elif vt.is_string():
            self._plain_bytes(col.as_string(), body)
        else:
            raise Error("parquet: cannot write column type " + String(vt))

    def _def_levels(
        self, col: AnyArray, mut null_count: Int
    ) raises -> List[UInt8]:
        """RLE definition levels (bit width 1), no length prefix — the shared
        level encoding for both page versions. Reports the null count; returns
        empty for a non-nullable column (`max_def == 0`)."""
        var out = List[UInt8]()
        null_count = 0
        if self.leaf.max_def >= 1:
            var defs = List[Int32]()
            for i in range(col.length()):
                if col.is_valid(i):
                    defs.append(Int32(1))
                else:
                    defs.append(Int32(0))
                    null_count += 1
            out = rle_encode(defs, 1)
        return out^

    def write(
        self, col: AnyArray, mut out: List[UInt8], mut codecs: CompressionLibs
    ) raises -> ColumnChunk:
        var n = col.length()
        var null_count = 0
        var page_offset = len(out)
        var uncompressed_size: Int
        var compressed_size: Int
        var header_len: Int
        if self.version == 2:
            # v2: uncompressed levels, then separately-compressed values.
            var def_bytes = self._def_levels(col, null_count)
            var values = List[UInt8]()
            self._encode_values(col, values)
            var is_comp = self.compression != Compression.UNCOMPRESSED
            var comp_values = self.compression.compress(codecs, Span(values))
            uncompressed_size = len(def_bytes) + len(values)
            compressed_size = len(def_bytes) + len(comp_values)
            header_len = PageHeader.data_page_v2(
                uncompressed_size,
                compressed_size,
                n,
                null_count,
                n,  # flat columns: num_rows == num_values
                len(def_bytes),
                is_comp,
            ).append_to(out)
            out.extend(Span(def_bytes))
            out.extend(Span(comp_values))
        else:
            # v1: length-prefixed RLE levels then PLAIN values, compressed
            # together.
            var body = List[UInt8]()
            if self.leaf.max_def >= 1:
                var def_bytes = self._def_levels(col, null_count)
                Self._u32le(body, len(def_bytes))
                body.extend(Span(def_bytes))
            self._encode_values(col, body)
            uncompressed_size = len(body)
            var compressed = self.compression.compress(codecs, Span(body))
            compressed_size = len(compressed)
            header_len = PageHeader.data_page(
                uncompressed_size, compressed_size, n
            ).append_to(out)
            out.extend(Span(compressed))

        var meta = ColumnMetaData()
        meta.type = self.leaf.physical.code
        meta.path_in_schema.append(self.leaf.name)
        meta.codec = self.compression.code
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


# ---------------------------------------------------------------------------
# FileWriter — header, row groups, footer
# ---------------------------------------------------------------------------


struct FileWriter(Movable):
    var out: List[UInt8]
    var codecs: CompressionLibs
    var compression: Compression
    var version: Int  # data-page version: 1 (default) or 2
    var leaves: List[LeafColumn]

    def __init__(out self, compression: Compression, version: Int = 1):
        self.out = List[UInt8]()
        FileMetaData.write_magic(self.out)  # file header magic
        self.codecs = CompressionLibs()
        self.compression = compression
        self.version = version
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
                self.leaves[ci].copy(), self.compression, self.version
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

        var ps = ParquetSchema.from_arrow(table.schema)
        self.leaves = ps.leaves.copy()

        var fmeta = FileMetaData()
        fmeta.version = self.version
        fmeta.schema = ps.elements.copy()
        fmeta.num_rows = n
        fmeta.created_by = "marrow"

        var start = 0
        while start < n or (n == 0 and start == 0):
            var length = min(row_group_size, n - start)
            var slice = batch.slice(start, length)
            fmeta.row_groups.append(self._write_row_group(slice, ps.nodes))
            start += length
            if length == 0:
                break

        # PLAIN encoding for every column in every row group
        var encodings = List[Int]()
        for _ in range(len(self.leaves)):
            encodings.append(Encoding.PLAIN.code)
        var rg_encodings = List[List[Int]]()
        for _ in range(len(fmeta.row_groups)):
            rg_encodings.append(encodings.copy())

        fmeta.write_footer(self.out, rg_encodings)
        Path(path).write_bytes(Span(self.out))


def write_table(
    table: Table,
    path: String,
    compression: Compression = Compression.SNAPPY,
    version: Int = 1,
) raises:
    """Write a Marrow `Table` to a Parquet file. `version` selects the data-page
    format: 1 (default) or 2 (levels stored uncompressed ahead of the values).
    """
    var writer = FileWriter(compression, version)
    writer.write(table, path)
