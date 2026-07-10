"""Native Arrow → Parquet writer.

`FileWriter` orchestrates the file: header magic, one `RowGroup` per slice of the
table (bounded by `row_group_size`), then the footer. `ColumnWriter` owns
one leaf column chunk — encoding a data page (RLE definition levels + PLAIN
values), compressing it, and producing the `ColumnMetaData` (with null-count and
min/max statistics). `version` selects the page format: v1 (levels + values
compressed together) or v2 (levels stored uncompressed ahead of the compressed
values). Value encoding is dispatched by Arrow type to typed methods, so the
writer mirrors the reader's structure.

Milestone: flat columns + struct; primitives (incl. int8/16 widened to INT32),
string/binary; PLAIN values; UNCOMPRESSED/SNAPPY/ZSTD/LZ4_RAW; v1 and v2 data
pages; per-column null-count and min/max statistics (with column_orders).
Dictionary encoding, page splitting, and temporal/list writing are follow-ups.
"""

from std.pathlib import Path
from std.math import isnan

from ..arrays import AnyArray, PrimitiveArray
from ..dtypes import PrimitiveType
from .. import dtypes as dt
from ..tabular import Table, RecordBatch
from ..schema import Schema

from .codecs import Rle, Plain, LittleEndian, Compression
from .utils import CompressionLibs
from .schema import SchemaMapping, LeafColumn, SchemaNode
from .format import (
    PageHeader,
    ColumnMetaData,
    ColumnChunk,
    RowGroup,
    FileMetaData,
)

comptime DEFAULT_ROW_GROUP_SIZE: Int = 1 << 20


# ---------------------------------------------------------------------------
# ColumnWriter — encode one leaf column chunk into the output buffer
# ---------------------------------------------------------------------------


struct ColumnWriter(Movable):
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

    def _encode_values(self, col: AnyArray, mut body: List[UInt8]) raises:
        """Dispatch on the leaf's Arrow type to the right `Plain` encoder (the
        writer's mirror of the reader's decode dispatch)."""
        ref vt = self.leaf.dtype
        if vt == dt.int32:
            Plain.encode_primitive[phys=DType.int32](col.as_int32(), body)
        elif vt == dt.int64:
            Plain.encode_primitive[phys=DType.int64](col.as_int64(), body)
        elif vt == dt.uint32:
            Plain.encode_primitive[phys=DType.uint32](col.as_uint32(), body)
        elif vt == dt.uint64:
            Plain.encode_primitive[phys=DType.uint64](col.as_uint64(), body)
        elif vt == dt.float32:
            Plain.encode_primitive[phys=DType.float32](col.as_float32(), body)
        elif vt == dt.float64:
            Plain.encode_primitive[phys=DType.float64](col.as_float64(), body)
        elif vt == dt.int8:
            Plain.encode_primitive[phys=DType.int32](col.as_int8(), body)
        elif vt == dt.int16:
            Plain.encode_primitive[phys=DType.int32](col.as_int16(), body)
        elif vt == dt.uint8:
            Plain.encode_primitive[phys=DType.int32](col.as_uint8(), body)
        elif vt == dt.uint16:
            Plain.encode_primitive[phys=DType.int32](col.as_uint16(), body)
        elif vt == dt.bool_:
            Plain.encode_bool(col.as_bool(), body)
        elif vt.is_string():
            Plain.encode_bytes(col.as_string(), body)
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
            out = Rle.encode(defs, 1)
        return out^

    # -----------------------------------------------------------------------
    # min/max statistics — PLAIN-encoded bounds over the non-null values, in
    # the column's logical (type-defined) ordering (see ColumnOrder emission).
    # -----------------------------------------------------------------------

    @staticmethod
    def _mm[
        T: PrimitiveType, skip_nan: Bool = False
    ](arr: PrimitiveArray[T]) raises -> Tuple[
        Scalar[T.native], Scalar[T.native], Bool
    ]:
        """min/max over the valid values in T's native (signed/unsigned) order.
        `skip_nan` excludes NaN — floats only, per the spec (NaN bounds nothing);
        the check is comptime-elided for integer columns."""
        var seen = False
        var mn = Scalar[T.native](0)
        var mx = Scalar[T.native](0)
        for i in range(len(arr)):
            if arr.is_valid(i):
                var v = arr[i].value()

                @parameter
                if skip_nan:
                    if isnan(v):
                        continue
                if not seen:
                    mn = v
                    mx = v
                    seen = True
                else:
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
        return (mn, mx, seen)

    @staticmethod
    def _int_stats[
        T: PrimitiveType, width: Int
    ](
        arr: PrimitiveArray[T],
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """Signed/unsigned integer bounds, widened to `width` little-endian bytes
        (the physical INT32 / INT64 width). False when there are no values."""
        var r = Self._mm(arr)
        if not r[2]:
            return False
        LittleEndian.put_le(min_out, r[0].cast[DType.uint64](), width)
        LittleEndian.put_le(max_out, r[1].cast[DType.uint64](), width)
        return True

    @staticmethod
    def _float_stats[
        T: PrimitiveType, width: Int
    ](
        arr: PrimitiveArray[T],
        mut min_out: List[UInt8],
        mut max_out: List[UInt8],
    ) raises -> Bool:
        """IEEE float bounds (NaN skipped), stored as their bit pattern with the
        signed zero normalised so the bound brackets both +0.0 and -0.0."""
        var r = Self._mm[skip_nan=True](arr)
        if not r[2]:
            return False
        var zero = Scalar[T.native](0)
        var mn = -zero if r[0] == zero else r[0]
        var mx = zero if r[1] == zero else r[1]
        LittleEndian.put_le(min_out, UInt64(mn.to_bits()), width)
        LittleEndian.put_le(max_out, UInt64(mx.to_bits()), width)
        return True

    def _stats(
        self, col: AnyArray, mut min_out: List[UInt8], mut max_out: List[UInt8]
    ) raises -> Bool:
        """Fill `min_out`/`max_out` with the PLAIN-encoded bounds; return False
        when there is nothing to summarise (all-null / empty) or the type carries
        no statistics. Byte arrays store raw bytes with no length prefix; the
        numeric widen/normalise rules live in `_int_stats` / `_float_stats`."""
        ref vt = self.leaf.dtype
        if vt == dt.int8:
            return Self._int_stats[width=4](col.as_int8(), min_out, max_out)
        elif vt == dt.int16:
            return Self._int_stats[width=4](col.as_int16(), min_out, max_out)
        elif vt == dt.int32:
            return Self._int_stats[width=4](col.as_int32(), min_out, max_out)
        elif vt == dt.uint8:
            return Self._int_stats[width=4](col.as_uint8(), min_out, max_out)
        elif vt == dt.uint16:
            return Self._int_stats[width=4](col.as_uint16(), min_out, max_out)
        elif vt == dt.uint32:
            return Self._int_stats[width=4](col.as_uint32(), min_out, max_out)
        elif vt == dt.int64:
            return Self._int_stats[width=8](col.as_int64(), min_out, max_out)
        elif vt == dt.uint64:
            return Self._int_stats[width=8](col.as_uint64(), min_out, max_out)
        elif vt == dt.float32:
            return Self._float_stats[width=4](
                col.as_float32(), min_out, max_out
            )
        elif vt == dt.float64:
            return Self._float_stats[width=8](
                col.as_float64(), min_out, max_out
            )
        elif vt == dt.bool_:
            ref b = col.as_bool()
            var seen = False
            var any_true = False
            var any_false = False
            for i in range(len(b)):
                if b.is_valid(i):
                    seen = True
                    if b[i].value():
                        any_true = True
                    else:
                        any_false = True
            if not seen:
                return False
            min_out.append(UInt8(0) if any_false else UInt8(1))
            max_out.append(UInt8(1) if any_true else UInt8(0))
            return True
        elif vt.is_string():
            ref s = col.as_string()
            var seen = False
            var lo = String()
            var hi = String()
            for i in range(len(s)):
                if s.is_valid(i):
                    var v = String(s[i])
                    if not seen:
                        lo = v.copy()
                        hi = v.copy()
                        seen = True
                    else:
                        if LittleEndian.bytes_less(v.as_bytes(), lo.as_bytes()):
                            lo = v.copy()
                        if LittleEndian.bytes_less(hi.as_bytes(), v.as_bytes()):
                            hi = v.copy()
            if not seen:
                return False
            # keep the footer small: skip stats for very long byte-array bounds
            # rather than truncate (a missing bound is always valid).
            if lo.byte_length() > 4096 or hi.byte_length() > 4096:
                return False
            min_out = List[UInt8](lo.as_bytes())
            max_out = List[UInt8](hi.as_bytes())
            return True
        else:
            return False

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
                LittleEndian.put_u32(body, len(def_bytes))
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
        var min_v = List[UInt8]()
        var max_v = List[UInt8]()
        if self._stats(col, min_v, max_v):
            meta.has_min_max = True
            meta.min_value = min_v^
            meta.max_value = max_v^

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
            nodes[ci].collect_leaf_arrays(batch.columns[ci], leaf_arrays)

        var columns = List[ColumnChunk]()
        var total = 0
        for ci in range(len(self.leaves)):
            var ccw = ColumnWriter(
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

        var ps = SchemaMapping.from_arrow(table.schema)
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

        fmeta.write_footer(self.out)
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
