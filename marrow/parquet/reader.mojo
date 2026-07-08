"""Native Parquet reader → Arrow.

Footer → row groups → column chunks → pages → decode → Arrow arrays. Milestone-1
handles flat columns (primitives + string/binary), PLAIN and dictionary
(RLE_DICTIONARY / PLAIN_DICTIONARY) encodings, definition-level nullability, and
both v1 and v2 data pages. Values are materialized through the typed builders in
`marrow.builders`, so validity bitmaps and offsets come for free.
"""

from std.sys import size_of
from std.pathlib import Path

from ..arrays import AnyArray, StructArray
from ..builders import PrimitiveBuilder, StringBuilder, BoolBuilder
from ..dtypes import (
    Field,
    bool_,
    int32,
    int64,
    uint32,
    uint64,
    float32,
    float64,
    Int32Type,
    Int64Type,
    UInt32Type,
    UInt64Type,
    Float32Type,
    Float64Type,
    NumericType,
)
from ..schema import Schema
from ..tabular import Table, RecordBatch

from .thrift import CompactReader
from .encoding import rle_decode, bit_width
from .compression import Codecs
from .schema import (
    parquet_to_arrow,
    LeafColumn,
    SchemaNode,
    NODE_LEAF,
    NODE_STRUCT,
)
from .format import (
    read_footer,
    ColumnMetaData,
    PageHeader,
    PAGE_DICTIONARY,
    PAGE_DATA,
    PAGE_DATA_V2,
    ENC_PLAIN,
)


# ---------------------------------------------------------------------------
# Low-level byte helpers
# ---------------------------------------------------------------------------


def _read_u32le(body: Span[UInt8, _], off: Int) -> Int:
    return (
        Int(body[off])
        | (Int(body[off + 1]) << 8)
        | (Int(body[off + 2]) << 16)
        | (Int(body[off + 3]) << 24)
    )


def _read_fixed_le[dt: DType](body: Span[UInt8, _], off: Int) -> Scalar[dt]:
    comptime W = size_of[Scalar[dt]]()
    var arr = InlineArray[UInt8, W](fill=0)
    for i in range(W):
        arr[i] = body[off + i]
    return SIMD[dt, 1].from_bytes[big_endian=False](arr)


# ---------------------------------------------------------------------------
# Level decoding — returns (def_levels, value_byte_offset, num_present)
# ---------------------------------------------------------------------------


def _levels_v1(
    body: Span[UInt8, _],
    num_values: Int,
    max_def: Int,
    max_rep: Int,
    mut def_levels: List[Int32],
) raises -> Tuple[Int, Int]:
    """Decode v1 rep/def levels into `def_levels`; return (value_offset, num_present).
    """
    var cursor = 0
    if max_rep >= 1:
        var l = _read_u32le(body, cursor)
        cursor += 4 + l  # repetition levels not needed for flat columns
    if max_def >= 1:
        var bw = bit_width(max_def)
        var l = _read_u32le(body, cursor)
        cursor += 4
        def_levels = rle_decode(body[cursor : cursor + l], bw, num_values)
        cursor += l
    var num_present = num_values
    if max_def >= 1:
        num_present = 0
        for d in def_levels:
            if Int(d) == max_def:
                num_present += 1
    return (cursor, num_present)


# ---------------------------------------------------------------------------
# Typed column readers
# ---------------------------------------------------------------------------


def _read_primitive[
    T: NumericType
](
    data: Span[UInt8, _],
    meta: ColumnMetaData,
    leaf: LeafColumn,
    num_rows: Int,
    mut codecs: Codecs,
) raises -> AnyArray:
    comptime dt = T.native
    comptime W = size_of[Scalar[dt]]()
    var builder = PrimitiveBuilder[T](num_rows)
    var dict = List[Scalar[dt]]()

    var pos = meta.data_page_offset
    if meta.dictionary_page_offset != -1:
        pos = meta.dictionary_page_offset
    var produced = 0
    while produced < meta.num_values:
        var pr = CompactReader(data, pos)
        var ph = PageHeader.read(pr)
        var body_start = pr.pos
        var comp = data[body_start : body_start + ph.compressed_page_size]
        pos = body_start + ph.compressed_page_size

        if ph.type == PAGE_DICTIONARY:
            var nvals = ph.dictionary_page_header.value().num_values
            var body = codecs.decompress(
                meta.codec, comp, ph.uncompressed_page_size
            )
            var span = Span(body)
            var off = 0
            for _ in range(nvals):
                dict.append(_read_fixed_le[dt](span, off))
                off += W
            continue

        var body: List[UInt8]
        var num_values: Int
        var encoding: Int
        var def_levels: List[Int32]
        var voff: Int
        var num_present: Int

        if ph.type == PAGE_DATA:
            ref dph = ph.data_page_header.value()
            num_values = dph.num_values
            encoding = dph.encoding
            body = codecs.decompress(
                meta.codec, comp, ph.uncompressed_page_size
            )
            def_levels = List[Int32]()
            voff, num_present = _levels_v1(
                Span(body), num_values, leaf.max_def, leaf.max_rep, def_levels
            )
        elif ph.type == PAGE_DATA_V2:
            ref dph2 = ph.data_page_header_v2.value()
            num_values = dph2.num_values
            encoding = dph2.encoding
            var lvl_len = (
                dph2.repetition_levels_byte_length
                + dph2.definition_levels_byte_length
            )
            body = List[UInt8]()
            body.extend(comp[0:lvl_len])
            if dph2.is_compressed:
                body.extend(
                    codecs.decompress(
                        meta.codec,
                        comp[lvl_len:],
                        ph.uncompressed_page_size - lvl_len,
                    )
                )
            else:
                body.extend(comp[lvl_len:])
            def_levels = List[Int32]()
            var cursor = dph2.repetition_levels_byte_length
            if leaf.max_def >= 1 and dph2.definition_levels_byte_length > 0:
                def_levels = rle_decode(
                    Span(body)[
                        cursor : cursor + dph2.definition_levels_byte_length
                    ],
                    bit_width(leaf.max_def),
                    num_values,
                )
            voff = lvl_len
            num_present = num_values - dph2.num_nulls
        else:
            raise Error("parquet: unexpected page type")

        var vspan = Span(body)[voff:]
        var use_dict = encoding != ENC_PLAIN
        var indices = List[Int32]()
        if use_dict:
            var bw = Int(vspan[0])
            indices = rle_decode(vspan[1:], bw, num_present)

        var vi = 0  # byte cursor into PLAIN values
        var di = 0  # cursor into indices
        for row in range(num_values):
            var present = leaf.max_def == 0 or Int(def_levels[row]) == (
                leaf.max_def
            )
            if present:
                var v: Scalar[dt]
                if use_dict:
                    v = dict[Int(indices[di])]
                    di += 1
                else:
                    v = _read_fixed_le[dt](vspan, vi)
                    vi += W
                builder.append(v)
            else:
                builder.append_null()
        produced += num_values

    var out: AnyArray = builder.finish()
    return out^


def _read_string(
    data: Span[UInt8, _],
    meta: ColumnMetaData,
    leaf: LeafColumn,
    num_rows: Int,
    mut codecs: Codecs,
) raises -> AnyArray:
    var builder = StringBuilder(num_rows)
    var dict = List[String]()

    var pos = meta.data_page_offset
    if meta.dictionary_page_offset != -1:
        pos = meta.dictionary_page_offset
    var produced = 0
    while produced < meta.num_values:
        var pr = CompactReader(data, pos)
        var ph = PageHeader.read(pr)
        var body_start = pr.pos
        var comp = data[body_start : body_start + ph.compressed_page_size]
        pos = body_start + ph.compressed_page_size

        if ph.type == PAGE_DICTIONARY:
            var nvals = ph.dictionary_page_header.value().num_values
            var body = codecs.decompress(
                meta.codec, comp, ph.uncompressed_page_size
            )
            var span = Span(body)
            var off = 0
            for _ in range(nvals):
                var n = _read_u32le(span, off)
                off += 4
                dict.append(String(from_utf8=span[off : off + n]))
                off += n
            continue

        var body: List[UInt8]
        var num_values: Int
        var encoding: Int
        var def_levels: List[Int32]
        var voff: Int
        var num_present: Int

        if ph.type == PAGE_DATA:
            ref dph = ph.data_page_header.value()
            num_values = dph.num_values
            encoding = dph.encoding
            body = codecs.decompress(
                meta.codec, comp, ph.uncompressed_page_size
            )
            def_levels = List[Int32]()
            voff, num_present = _levels_v1(
                Span(body), num_values, leaf.max_def, leaf.max_rep, def_levels
            )
        elif ph.type == PAGE_DATA_V2:
            ref dph2 = ph.data_page_header_v2.value()
            num_values = dph2.num_values
            encoding = dph2.encoding
            var lvl_len = (
                dph2.repetition_levels_byte_length
                + dph2.definition_levels_byte_length
            )
            body = List[UInt8]()
            body.extend(comp[0:lvl_len])
            if dph2.is_compressed:
                body.extend(
                    codecs.decompress(
                        meta.codec,
                        comp[lvl_len:],
                        ph.uncompressed_page_size - lvl_len,
                    )
                )
            else:
                body.extend(comp[lvl_len:])
            def_levels = List[Int32]()
            var cursor = dph2.repetition_levels_byte_length
            if leaf.max_def >= 1 and dph2.definition_levels_byte_length > 0:
                def_levels = rle_decode(
                    Span(body)[
                        cursor : cursor + dph2.definition_levels_byte_length
                    ],
                    bit_width(leaf.max_def),
                    num_values,
                )
            voff = lvl_len
            num_present = num_values - dph2.num_nulls
        else:
            raise Error("parquet: unexpected page type")

        var vspan = Span(body)[voff:]
        var use_dict = encoding != ENC_PLAIN
        var indices = List[Int32]()
        if use_dict:
            var bw = Int(vspan[0])
            indices = rle_decode(vspan[1:], bw, num_present)

        var vi = 0
        var di = 0
        for row in range(num_values):
            var present = leaf.max_def == 0 or Int(def_levels[row]) == (
                leaf.max_def
            )
            if present:
                if use_dict:
                    builder.append(dict[Int(indices[di])])
                    di += 1
                else:
                    var n = _read_u32le(vspan, vi)
                    vi += 4
                    builder.append(String(from_utf8=vspan[vi : vi + n]))
                    vi += n
            else:
                builder.append_null()
        produced += num_values

    var out: AnyArray = builder.finish()
    return out^


def _read_bool(
    data: Span[UInt8, _],
    meta: ColumnMetaData,
    leaf: LeafColumn,
    num_rows: Int,
    mut codecs: Codecs,
) raises -> AnyArray:
    var builder = BoolBuilder(num_rows)
    var pos = meta.data_page_offset
    if meta.dictionary_page_offset != -1:
        pos = meta.dictionary_page_offset
    var produced = 0
    while produced < meta.num_values:
        var pr = CompactReader(data, pos)
        var ph = PageHeader.read(pr)
        var body_start = pr.pos
        var comp = data[body_start : body_start + ph.compressed_page_size]
        pos = body_start + ph.compressed_page_size

        if ph.type == PAGE_DICTIONARY:
            raise Error("parquet: dictionary-encoded bool not supported")

        var body: List[UInt8]
        var num_values: Int
        var def_levels: List[Int32]
        var voff: Int

        if ph.type == PAGE_DATA:
            ref dph = ph.data_page_header.value()
            num_values = dph.num_values
            if dph.encoding != ENC_PLAIN:
                raise Error("parquet: non-plain bool encoding not supported")
            body = codecs.decompress(
                meta.codec, comp, ph.uncompressed_page_size
            )
            def_levels = List[Int32]()
            var np: Int
            voff, np = _levels_v1(
                Span(body), num_values, leaf.max_def, leaf.max_rep, def_levels
            )
        else:
            raise Error("parquet: bool v2 pages not supported")

        var vspan = Span(body)[voff:]
        var bitpos = 0
        for row in range(num_values):
            var present = leaf.max_def == 0 or Int(def_levels[row]) == (
                leaf.max_def
            )
            if present:
                var byte = vspan[bitpos >> 3]
                var b = (byte >> UInt8(bitpos & 7)) & 1
                bitpos += 1
                builder.append(b == 1)
            else:
                builder.append_null()
        produced += num_values

    var out: AnyArray = builder.finish()
    return out^


def _read_column(
    data: Span[UInt8, _],
    meta: ColumnMetaData,
    leaf: LeafColumn,
    num_rows: Int,
    mut codecs: Codecs,
) raises -> AnyArray:
    ref dt = leaf.dtype
    if dt == int32:
        return _read_primitive[Int32Type](data, meta, leaf, num_rows, codecs)
    elif dt == int64:
        return _read_primitive[Int64Type](data, meta, leaf, num_rows, codecs)
    elif dt == uint32:
        return _read_primitive[UInt32Type](data, meta, leaf, num_rows, codecs)
    elif dt == uint64:
        return _read_primitive[UInt64Type](data, meta, leaf, num_rows, codecs)
    elif dt == float32:
        return _read_primitive[Float32Type](data, meta, leaf, num_rows, codecs)
    elif dt == float64:
        return _read_primitive[Float64Type](data, meta, leaf, num_rows, codecs)
    elif dt == bool_:
        return _read_bool(data, meta, leaf, num_rows, codecs)
    elif dt.is_string():
        return _read_string(data, meta, leaf, num_rows, codecs)
    else:
        raise Error("parquet: unsupported column type " + String(dt))


# ---------------------------------------------------------------------------
# Nested assembly — rebuild the Arrow tree from decoded leaf arrays
# ---------------------------------------------------------------------------


def _assemble(
    node: SchemaNode, ref leaf_arrays: List[AnyArray]
) raises -> AnyArray:
    if node.kind == NODE_LEAF:
        return leaf_arrays[node.leaf_index].copy()
    elif node.kind == NODE_STRUCT:
        var children = List[AnyArray]()
        var fields = List[Field]()
        for ref c in node.children:
            children.append(_assemble(c, leaf_arrays))
            fields.append(c.field.copy())
        var out: AnyArray = StructArray.from_arrays(children^, fields, None)
        return out^
    else:
        raise Error("parquet: unsupported schema node kind")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def read_table(path: String) raises -> Table:
    """Read a Parquet file into a Marrow `Table`."""
    var raw = Path(path).read_bytes()
    var data = Span(raw)
    var meta = read_footer(data)
    var parsed = parquet_to_arrow(meta)

    var codecs = Codecs()
    var batches = List[RecordBatch]()
    for ref rg in meta.row_groups:
        var leaf_arrays = List[AnyArray]()
        for ci in range(len(parsed.leaves)):
            leaf_arrays.append(
                _read_column(
                    data,
                    rg.columns[ci].meta_data,
                    parsed.leaves[ci],
                    rg.num_rows,
                    codecs,
                )
            )
        var cols = List[AnyArray]()
        for ref node in parsed.nodes:
            cols.append(_assemble(node, leaf_arrays))
        batches.append(
            RecordBatch(schema=Schema(copy=parsed.schema), columns=cols^)
        )

    if len(batches) == 0:
        batches.append(RecordBatch.empty(parsed.schema))
    return Table.from_batches(parsed.schema, batches)
