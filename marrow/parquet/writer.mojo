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
from std.sys import size_of

from ..arrays import AnyArray, PrimitiveArray, StringArray
from ..dtypes import PrimitiveType, NumericType
from .. import dtypes as dt
from ..tabular import Table, RecordBatch
from ..schema import Schema

from .codecs import (
    Rle,
    Plain,
    LittleEndian,
    Compression,
    Encoding,
    DeltaBinaryPacked,
    DeltaByteArray,
    DeltaLengthByteArray,
)
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
    var encoding: Encoding  # value encoding: PLAIN, RLE_DICTIONARY, DELTA_*

    def __init__(
        out self,
        var leaf: LeafColumn,
        compression: Compression,
        version: Int = 1,
        encoding: Encoding = Encoding.PLAIN,
    ):
        self.leaf = leaf^
        self.compression = compression
        self.version = version
        self.encoding = encoding

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

    # -----------------------------------------------------------------------
    # Dictionary encoding — build the distinct-value dictionary (PLAIN page) and
    # the per-present-value indices (RLE_DICTIONARY data page). Indices cover the
    # non-null values only, exactly like the PLAIN encoders; the definition
    # levels place the nulls, so this composes with the flat and leveled paths.
    # -----------------------------------------------------------------------

    @staticmethod
    def _dict_prim[
        store: NumericType, phys: DType
    ](
        arr: PrimitiveArray[store],
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a primitive column: PLAIN-encode each distinct value
        (widened to `phys`) into `dict_body`, collect the per-value index."""
        comptime W = size_of[Scalar[phys]]()
        var seen = Dict[Scalar[store.native], Int]()
        var num_dict = 0
        for i in range(arr.length):
            if arr.is_valid(i):
                var v = arr[i].value()
                if v in seen:
                    indices.append(Int32(seen[v]))
                else:
                    seen[v] = num_dict
                    indices.append(Int32(num_dict))
                    num_dict += 1
                    var bytes = v.cast[phys]().as_bytes[big_endian=False]()
                    for b in range(W):
                        dict_body.append(bytes[b])
        return num_dict

    @staticmethod
    def _dict_string(
        arr: StringArray,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a string column: length-prefixed distinct values in
        the dictionary page, one index per present value."""
        var seen = Dict[String, Int]()
        var num_dict = 0
        for i in range(arr.length):
            if arr.is_valid(i):
                var v = String(arr[i])
                if v in seen:
                    indices.append(Int32(seen[v]))
                else:
                    seen[v] = num_dict
                    indices.append(Int32(num_dict))
                    num_dict += 1
                    var b = v.as_bytes()
                    LittleEndian.put_u32(dict_body, len(b))
                    dict_body.extend(b)
        return num_dict

    def _encode_dictionary(
        self,
        col: AnyArray,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Build the dictionary page bytes + per-value indices for `col`; returns
        the dictionary size. Mirrors `_encode_values`' type dispatch (bool is
        never dictionary-encoded)."""
        ref vt = self.leaf.dtype
        if vt == dt.int32:
            return Self._dict_prim[dt.Int32Type, DType.int32](
                col.as_int32(), dict_body, indices
            )
        elif vt == dt.int64:
            return Self._dict_prim[dt.Int64Type, DType.int64](
                col.as_int64(), dict_body, indices
            )
        elif vt == dt.uint32:
            return Self._dict_prim[dt.UInt32Type, DType.uint32](
                col.as_uint32(), dict_body, indices
            )
        elif vt == dt.uint64:
            return Self._dict_prim[dt.UInt64Type, DType.uint64](
                col.as_uint64(), dict_body, indices
            )
        elif vt == dt.float32:
            return Self._dict_prim[dt.Float32Type, DType.float32](
                col.as_float32(), dict_body, indices
            )
        elif vt == dt.float64:
            return Self._dict_prim[dt.Float64Type, DType.float64](
                col.as_float64(), dict_body, indices
            )
        elif vt == dt.int8:
            return Self._dict_prim[dt.Int8Type, DType.int32](
                col.as_int8(), dict_body, indices
            )
        elif vt == dt.int16:
            return Self._dict_prim[dt.Int16Type, DType.int32](
                col.as_int16(), dict_body, indices
            )
        elif vt == dt.uint8:
            return Self._dict_prim[dt.UInt8Type, DType.int32](
                col.as_uint8(), dict_body, indices
            )
        elif vt == dt.uint16:
            return Self._dict_prim[dt.UInt16Type, DType.int32](
                col.as_uint16(), dict_body, indices
            )
        elif vt.is_string():
            return Self._dict_string(col.as_string(), dict_body, indices)
        else:
            raise Error(
                "parquet: cannot dictionary-encode column type " + String(vt)
            )

    @staticmethod
    def can_dictionary(dtype: dt.AnyDataType) -> Bool:
        """Whether a column of `dtype` is dictionary-encoded when requested —
        numeric and string columns (bool and unsupported types stay PLAIN)."""
        return (
            dtype.is_integer() or dtype.is_floating_point() or dtype.is_string()
        )

    # -----------------------------------------------------------------------
    # Delta encoding — DELTA_BINARY_PACKED (signed ints) and DELTA_BYTE_ARRAY /
    # DELTA_LENGTH_BYTE_ARRAY (strings), over the present values only.
    # -----------------------------------------------------------------------

    @staticmethod
    def _ints[
        store: NumericType
    ](arr: PrimitiveArray[store], mut out: List[Int64]) raises:
        for i in range(arr.length):
            if arr.is_valid(i):
                out.append(Int64(arr[i].value()))

    @staticmethod
    def can_delta(dtype: dt.AnyDataType, encoding: Encoding) -> Bool:
        """Whether `encoding` (a DELTA_* variant) applies to `dtype`."""
        if encoding == Encoding.DELTA_BINARY_PACKED:
            return dtype.is_signed_integer()
        elif (
            encoding == Encoding.DELTA_BYTE_ARRAY
            or encoding == Encoding.DELTA_LENGTH_BYTE_ARRAY
        ):
            return dtype.is_string()
        return False

    def _encode_delta(self, col: AnyArray, mut out: List[UInt8]) raises:
        """Delta-encode the present values per `self.encoding`."""
        var ints = List[Int64]()
        ref vt = self.leaf.dtype
        if self.encoding == Encoding.DELTA_BINARY_PACKED:
            if vt == dt.int32:
                Self._ints(col.as_int32(), ints)
            elif vt == dt.int64:
                Self._ints(col.as_int64(), ints)
            elif vt == dt.int8:
                Self._ints(col.as_int8(), ints)
            elif vt == dt.int16:
                Self._ints(col.as_int16(), ints)
            else:
                raise Error(
                    "parquet: cannot DELTA_BINARY_PACKED type " + String(vt)
                )
            out.extend(Span(DeltaBinaryPacked.encode(ints)))
        elif self.encoding == Encoding.DELTA_LENGTH_BYTE_ARRAY:
            DeltaLengthByteArray.encode(col.as_string(), out)
        else:  # DELTA_BYTE_ARRAY
            DeltaByteArray.encode(col.as_string(), out)

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

    def _write_dict_page(
        self,
        values: AnyArray,
        mut index_bytes: List[UInt8],
        mut out: List[UInt8],
        mut codecs: CompressionLibs,
    ) raises -> Tuple[Int, Int, Int]:
        """Write the dictionary page (PLAIN distinct values, compressed) ahead of
        the data page and fill `index_bytes` with the data page's value bytes:
        a `bit_width` byte then the bit-packed per-value indices. Returns
        `(offset, uncompressed_size, compressed_size)` — the page's byte offset
        (the chunk start) and its total sizes (header included), which the chunk
        metadata must add to the data page's."""
        var offset = len(out)
        var dict_body = List[UInt8]()
        var indices = List[Int32]()
        var num_dict = self._encode_dictionary(values, dict_body, indices)
        var comp = self.compression.compress(codecs, Span(dict_body))
        var header_len = PageHeader.dictionary_page(
            len(dict_body), len(comp), num_dict
        ).append_to(out)
        out.extend(Span(comp))

        var width = Rle.bit_width(num_dict - 1) if num_dict > 0 else 0
        index_bytes.append(UInt8(width))
        index_bytes.extend(Span(Rle.encode_bitpacked(indices, width)))
        return (offset, header_len + len(dict_body), header_len + len(comp))

    def _emit_page(
        self,
        values: AnyArray,
        var value_bytes: List[UInt8],
        var rep_bytes: List[UInt8],
        var def_bytes: List[UInt8],
        num_values: Int,
        num_rows: Int,
        null_count: Int,
        dict_page_offset: Int,
        dict_uncompressed_size: Int,
        dict_compressed_size: Int,
        path: List[String],
        mut out: List[UInt8],
        mut codecs: CompressionLibs,
    ) raises -> ColumnChunk:
        """Serialize one data page (v1 or v2) from the already-encoded value bytes
        (PLAIN, dictionary indices, or DELTA) plus the RLE rep/def level bytes,
        and build its `ColumnChunk` metadata (null count, min/max stats, the
        encodings actually used, and — for a dictionary chunk — the dictionary
        page offset). v1 length-prefixes each level stream ahead of the values
        (all compressed together); v2 keeps rep+def uncompressed ahead of the
        compressed values. A dictionary page, if any, was already written to
        `out` at `dict_page_offset`, so the chunk starts there."""
        var page_offset = len(out)

        var uncompressed_size: Int
        var compressed_size: Int
        var header_len: Int
        if self.version == 2:
            var is_comp = self.compression != Compression.UNCOMPRESSED
            var comp_values = self.compression.compress(
                codecs, Span(value_bytes)
            )
            var level_len = len(rep_bytes) + len(def_bytes)
            uncompressed_size = level_len + len(value_bytes)
            compressed_size = level_len + len(comp_values)
            header_len = PageHeader.data_page_v2(
                uncompressed_size,
                compressed_size,
                num_values,
                null_count,
                num_rows,
                len(def_bytes),
                is_comp,
                rep_levels_byte_length=len(rep_bytes),
                encoding=self.encoding,
            ).append_to(out)
            out.extend(Span(rep_bytes))
            out.extend(Span(def_bytes))
            out.extend(Span(comp_values))
        else:
            var body = List[UInt8]()
            if self.leaf.max_rep >= 1:
                LittleEndian.put_u32(body, len(rep_bytes))
                body.extend(Span(rep_bytes))
            if self.leaf.max_def >= 1:
                LittleEndian.put_u32(body, len(def_bytes))
                body.extend(Span(def_bytes))
            body.extend(Span(value_bytes))
            uncompressed_size = len(body)
            var compressed = self.compression.compress(codecs, Span(body))
            compressed_size = len(compressed)
            header_len = PageHeader.data_page(
                uncompressed_size, compressed_size, num_values, self.encoding
            ).append_to(out)
            out.extend(Span(compressed))

        var meta = ColumnMetaData()
        meta.type = self.leaf.physical.code
        for p in path:
            meta.path_in_schema.append(p)
        meta.codec = self.compression.code
        meta.num_values = num_values
        meta.total_uncompressed_size = (
            uncompressed_size + header_len + dict_uncompressed_size
        )
        meta.total_compressed_size = (
            compressed_size + header_len + dict_compressed_size
        )
        meta.data_page_offset = page_offset
        # encodings: RLE levels, then the value encoding (a dictionary chunk also
        # advertises the PLAIN dictionary page).
        var encs = [Encoding.RLE.code]
        if dict_page_offset >= 0:
            meta.dictionary_page_offset = dict_page_offset
            encs.append(Encoding.PLAIN.code)
        encs.append(self.encoding.code)
        meta.encodings = encs^
        if self.leaf.max_def >= 1:
            meta.null_count = null_count
        var min_v = List[UInt8]()
        var max_v = List[UInt8]()
        if self._stats(values, min_v, max_v):
            meta.has_min_max = True
            meta.min_value = min_v^
            meta.max_value = max_v^

        var cc = ColumnChunk()
        cc.file_offset = dict_page_offset if dict_page_offset >= 0 else page_offset
        cc.meta_data = meta^
        return cc^

    def write(
        self,
        values: AnyArray,
        def_levels: List[Int32],
        rep_levels: List[Int32],
        path: List[String],
        mut out: List[UInt8],
        mut codecs: CompressionLibs,
    ) raises -> ColumnChunk:
        """Write one leaf column chunk. A flat column passes empty `def_levels`/
        `rep_levels` and its 0/1 definition levels are derived here (the fast path
        that never materialises rep levels); a leveled (list/map/nested) column
        passes the rep/def levels pre-shredded by `SchemaNode.shred_levels`. The
        value encoding is `self.encoding` (PLAIN, dictionary, or DELTA); nulls are
        skipped and placed by the definition levels."""
        var max_def = self.leaf.max_def
        var max_rep = self.leaf.max_rep

        # ---- levels ----
        var num_values: Int
        var null_count: Int
        var num_rows: Int
        var rep_bytes = List[UInt8]()
        var def_bytes = List[UInt8]()
        if max_rep == 0 and len(def_levels) == 0:
            # Flat: derive 0/1 definition levels from the array's own validity.
            num_values = values.length()
            null_count = 0
            def_bytes = self._def_levels(values, null_count)
            num_rows = num_values
        else:
            # Leveled: present values = definition level == max_def.
            num_values = len(def_levels)
            var num_present = 0
            for d in def_levels:
                if Int(d) == max_def:
                    num_present += 1
            null_count = num_values - num_present
            num_rows = num_values
            if max_rep >= 1:
                num_rows = 0
                for rl in rep_levels:
                    if Int(rl) == 0:
                        num_rows += 1
                rep_bytes = Rle.encode(rep_levels, Rle.bit_width(max_rep))
            if max_def >= 1:
                def_bytes = Rle.encode(def_levels, Rle.bit_width(max_def))

        # ---- values ----
        var value_bytes = List[UInt8]()
        var dict_page_offset = -1
        var dict_uncompressed = 0
        var dict_compressed = 0
        if self.encoding == Encoding.RLE_DICTIONARY:
            var d = self._write_dict_page(values, value_bytes, out, codecs)
            dict_page_offset = d[0]
            dict_uncompressed = d[1]
            dict_compressed = d[2]
        elif (
            self.encoding == Encoding.DELTA_BINARY_PACKED
            or self.encoding == Encoding.DELTA_BYTE_ARRAY
            or self.encoding == Encoding.DELTA_LENGTH_BYTE_ARRAY
        ):
            self._encode_delta(values, value_bytes)
        else:
            self._encode_values(values, value_bytes)

        return self._emit_page(
            values,
            value_bytes^,
            rep_bytes^,
            def_bytes^,
            num_values,
            num_rows,
            null_count,
            dict_page_offset,
            dict_uncompressed,
            dict_compressed,
            path,
            out,
            codecs,
        )


# ---------------------------------------------------------------------------
# FileWriter — header, row groups, footer
# ---------------------------------------------------------------------------


struct FileWriter(Movable):
    var out: List[UInt8]
    var codecs: CompressionLibs
    var compression: Compression
    var version: Int  # data-page version: 1 (default) or 2
    var use_dictionary: Bool  # dictionary-encode eligible columns (else PLAIN)
    var encoding: Optional[Encoding]  # forced DELTA_* for compatible columns
    var leaves: List[LeafColumn]

    def __init__(
        out self,
        compression: Compression,
        version: Int = 1,
        use_dictionary: Bool = True,
        encoding: Optional[Encoding] = None,
    ):
        self.out = List[UInt8]()
        FileMetaData.write_magic(self.out)  # file header magic
        self.codecs = CompressionLibs()
        self.compression = compression
        self.version = version
        self.use_dictionary = use_dictionary
        self.encoding = encoding
        self.leaves = List[LeafColumn]()

    def _encoding_for(self, leaf: LeafColumn) -> Encoding:
        """The value encoding for a leaf: a forced DELTA_* encoding where it
        applies, else RLE_DICTIONARY when dictionaries are enabled and the type
        is eligible, else PLAIN."""
        if self.encoding and ColumnWriter.can_delta(
            leaf.dtype, self.encoding.value()
        ):
            return self.encoding.value()
        if self.use_dictionary and ColumnWriter.can_dictionary(leaf.dtype):
            return Encoding.RLE_DICTIONARY
        return Encoding.PLAIN

    def _write_row_group(
        mut self, batch: RecordBatch, nodes: List[SchemaNode]
    ) raises -> RowGroup:
        var columns = List[ColumnChunk]()
        var total = 0
        for ci in range(len(nodes)):
            var col_leaves = List[Int]()
            nodes[ci].collect_leaf_indices(col_leaves)
            var leaf_values = List[AnyArray]()
            nodes[ci].collect_leaf_arrays(batch.columns[ci], leaf_values)

            # Per-leaf rep/def levels: empty for a flat column (each leaf derives
            # its own 0/1 def levels), Dremel-shredded when the column contains a
            # repeated (list/map) group.
            var defs = List[List[Int32]]()
            var reps = List[List[Int32]]()
            for _ in range(len(self.leaves)):
                defs.append(List[Int32]())
                reps.append(List[Int32]())
            if nodes[ci].contains_repeated():
                nodes[ci].shred_levels(
                    batch.columns[ci], self.leaves, defs, reps
                )

            for k in range(len(col_leaves)):
                var gi = col_leaves[k]
                var ccw = ColumnWriter(
                    self.leaves[gi].copy(),
                    self.compression,
                    self.version,
                    self._encoding_for(self.leaves[gi]),
                )
                var cc = ccw.write(
                    leaf_values[k],
                    defs[gi],
                    reps[gi],
                    [self.leaves[gi].name],
                    self.out,
                    self.codecs,
                )
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
    use_dictionary: Bool = True,
    encoding: Optional[Encoding] = None,
) raises:
    """Write a Marrow `Table` to a Parquet file. `version` selects the data-page
    format: 1 (default) or 2 (levels stored uncompressed ahead of the values).
    `use_dictionary` (default True, like PyArrow) dictionary-encodes numeric and
    string columns; set False to force PLAIN. `encoding` forces a `DELTA_*`
    variant on compatible columns (DELTA_BINARY_PACKED for signed ints,
    DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY for strings), overriding the
    dictionary for those columns.
    """
    var writer = FileWriter(compression, version, use_dictionary, encoding)
    writer.write(table, path)
