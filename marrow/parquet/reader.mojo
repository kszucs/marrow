"""Native Parquet reader → Arrow.

Footer → row groups → column chunks → pages → decode → Arrow arrays. Milestone-1
handles flat columns (primitives + string/binary), PLAIN and dictionary
(RLE_DICTIONARY / PLAIN_DICTIONARY) encodings, definition-level nullability, and
both v1 and v2 data pages, plus struct nesting.

Page navigation, decompression, and level decoding are centralized in
`_next_page`; the typed readers only decode values. Fixed-width primitive
columns take a memcpy fast path (contiguous PLAIN pages copy straight into the
output buffer); nullable/dictionary pages scatter element-by-element.
"""

from std.sys import size_of
from std.ffi import external_call
from std.io.file import FileHandle
from std.memory import memcpy

from ..arrays import AnyArray, StructArray, ArrayData
from ..buffers import Buffer, Bitmap
from ..builders import StringBuilder, BoolBuilder
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
from .encoding import rle_decode, rle_count_matches, bit_width
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
# Memory-mapped file — read the file zero-copy instead of copying it in
# ---------------------------------------------------------------------------


struct MappedFile(Movable):
    """A read-only mmap of the whole file, freed when the value drops. Decoded
    values are copied into owned Arrow buffers, so the map only needs to outlive
    the decode."""

    var ptr: UnsafePointer[UInt8, ImmutUntrackedOrigin]
    var size: Int

    def __init__(out self, path: String) raises:
        # Use Mojo's file open (the libc variadic `open` cannot be external_call'd
        # in an archive build); mmap/lseek are plain syscalls and are fine.
        var f = FileHandle(path, "r")
        var size = Int(
            external_call["lseek", Int64](
                f.handle, Int64(0), Int(2)
            )  # SEEK_END
        )
        # PROT_READ=1, MAP_PRIVATE=2; the mapping outlives the fd.
        var ptr = external_call[
            "mmap", UnsafePointer[UInt8, ImmutUntrackedOrigin]
        ](UInt(0), size, Int32(1), Int32(2), Int32(f.handle), Int64(0))
        _ = f^  # close the fd; mmap stays valid
        if Int(ptr) == 0 or Int(ptr) == -1:
            raise Error("parquet: mmap failed for " + path)
        self.ptr = ptr
        self.size = size

    def span(self) -> Span[UInt8, ImmutUntrackedOrigin]:
        return Span[UInt8, ImmutUntrackedOrigin](ptr=self.ptr, length=self.size)

    def __del__(deinit self):
        _ = external_call["munmap", Int32](self.ptr, self.size)


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
# Page reading — one decoded page (levels + value bytes), v1/v2 unified
# ---------------------------------------------------------------------------

comptime PAGEKIND_DICT: Int = 0
comptime PAGEKIND_DATA: Int = 1


struct PageData(Movable):
    """A decoded page: the (decompressed) body plus, for data pages, the
    definition levels, the byte offset where values start, and counts."""

    var kind: Int
    var body: List[UInt8]
    var def_levels: List[Int32]
    var value_offset: Int
    var num_present: Int
    var encoding: Int
    var num_values: Int

    def __init__(out self):
        self.kind = PAGEKIND_DATA
        self.body = List[UInt8]()
        self.def_levels = List[Int32]()
        self.value_offset = 0
        self.num_present = 0
        self.encoding = ENC_PLAIN
        self.num_values = 0


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
    var num_present = num_values
    if max_def >= 1:
        var bw = bit_width(max_def)
        var l = _read_u32le(body, cursor)
        cursor += 4
        var levels = body[cursor : cursor + l]
        # Count present values first (O(1) for the all-present run); only
        # materialize the full level list when there are actually nulls.
        num_present = rle_count_matches(levels, bw, num_values, Int32(max_def))
        if num_present != num_values:
            def_levels = rle_decode(levels, bw, num_values)
        cursor += l
    return (cursor, num_present)


def _next_page(
    data: Span[UInt8, _],
    mut pos: Int,
    meta: ColumnMetaData,
    leaf: LeafColumn,
    mut codecs: Codecs,
) raises -> PageData:
    """Read the page at `pos`, decompress it, decode its levels, and advance
    `pos` past it."""
    var pr = CompactReader(data, pos)
    var ph = PageHeader.read(pr)
    var body_start = pr.pos
    var comp = data[body_start : body_start + ph.compressed_page_size]
    pos = body_start + ph.compressed_page_size

    var pg = PageData()
    if ph.type == PAGE_DICTIONARY:
        pg.kind = PAGEKIND_DICT
        pg.num_values = ph.dictionary_page_header.value().num_values
        pg.body = codecs.decompress(meta.codec, comp, ph.uncompressed_page_size)
        return pg^

    if ph.type == PAGE_DATA:
        ref dph = ph.data_page_header.value()
        pg.num_values = dph.num_values
        pg.encoding = dph.encoding
        pg.body = codecs.decompress(meta.codec, comp, ph.uncompressed_page_size)
        var defs = List[Int32]()
        var voff, npresent = _levels_v1(
            Span(pg.body), pg.num_values, leaf.max_def, leaf.max_rep, defs
        )
        pg.def_levels = defs^
        pg.value_offset = voff
        pg.num_present = npresent
    elif ph.type == PAGE_DATA_V2:
        ref dph2 = ph.data_page_header_v2.value()
        pg.num_values = dph2.num_values
        pg.encoding = dph2.encoding
        var lvl_len = (
            dph2.repetition_levels_byte_length
            + dph2.definition_levels_byte_length
        )
        var body = List[UInt8]()
        body.extend(comp[0:lvl_len])  # levels are never compressed in v2
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
        var cursor = dph2.repetition_levels_byte_length
        var defs = List[Int32]()
        if leaf.max_def >= 1 and dph2.definition_levels_byte_length > 0:
            defs = rle_decode(
                Span(body)[
                    cursor : cursor + dph2.definition_levels_byte_length
                ],
                bit_width(leaf.max_def),
                pg.num_values,
            )
        pg.def_levels = defs^
        pg.value_offset = lvl_len
        pg.num_present = pg.num_values - dph2.num_nulls
        pg.body = body^
    else:
        raise Error("parquet: unexpected page type")
    return pg^


def _page_start(meta: ColumnMetaData) -> Int:
    if meta.dictionary_page_offset != -1:
        return meta.dictionary_page_offset
    return meta.data_page_offset


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

    var values = Buffer.alloc_uninit[dt](num_rows)
    var vptr = values.view[dt]().unsafe_ptr()
    var has_validity = leaf.max_def >= 1
    var bitmap = Bitmap[mut=True].alloc_zeroed(num_rows if has_validity else 0)
    var dict = List[Scalar[dt]]()
    var wpos = 0
    var null_count = 0

    var pos = _page_start(meta)
    var produced = 0
    while produced < meta.num_values:
        var pg = _next_page(data, pos, meta, leaf, codecs)
        if pg.kind == PAGEKIND_DICT:
            # dictionary values are a contiguous PLAIN block — bulk copy
            dict.resize(pg.num_values, 0)
            memcpy(
                dest=dict.unsafe_ptr(),
                src=Span(pg.body).unsafe_ptr().bitcast[Scalar[dt]](),
                count=pg.num_values,
            )
            continue

        var vspan = Span(pg.body)[pg.value_offset :]
        if pg.encoding == ENC_PLAIN and pg.num_present == pg.num_values:
            # fast path — a whole page of present values is a contiguous copy
            var src = vspan.unsafe_ptr().bitcast[Scalar[dt]]()
            memcpy(dest=vptr + wpos, src=src, count=pg.num_values)
            if has_validity:
                bitmap.set_range(wpos, pg.num_values, True)
            wpos += pg.num_values
        elif pg.encoding == ENC_PLAIN:
            var vi = 0
            for row in range(pg.num_values):
                if Int(pg.def_levels[row]) == leaf.max_def:
                    vptr[wpos] = _read_fixed_le[dt](vspan, vi)
                    vi += W
                    bitmap.set(wpos)
                else:
                    vptr[wpos] = 0
                    null_count += 1
                wpos += 1
        else:
            var indices = rle_decode(vspan[1:], Int(vspan[0]), pg.num_present)
            var all_present = pg.num_present == pg.num_values
            var di = 0
            for row in range(pg.num_values):
                var present = all_present or (
                    Int(pg.def_levels[row]) == leaf.max_def
                )
                if present:
                    vptr[wpos] = dict[Int(indices[di])]
                    di += 1
                    if has_validity:
                        bitmap.set(wpos)
                else:
                    vptr[wpos] = 0
                    null_count += 1
                wpos += 1
        produced += pg.num_values

    var buffers = List[Buffer[mut=False]]()
    buffers.append(values^.to_immutable())
    var bm: Optional[Bitmap[mut=False]] = None
    if null_count > 0:
        bm = bitmap^.to_immutable(length=num_rows)
    return AnyArray.from_data(
        ArrayData(
            dtype=leaf.dtype.copy(),
            length=num_rows,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffers=buffers^,
            children=List[ArrayData](),
        )
    )


def _read_string(
    data: Span[UInt8, _],
    meta: ColumnMetaData,
    leaf: LeafColumn,
    num_rows: Int,
    mut codecs: Codecs,
) raises -> AnyArray:
    var builder = StringBuilder(num_rows)
    var dict = List[String]()

    var pos = _page_start(meta)
    var produced = 0
    while produced < meta.num_values:
        var pg = _next_page(data, pos, meta, leaf, codecs)
        if pg.kind == PAGEKIND_DICT:
            var span = Span(pg.body)
            var off = 0
            for _ in range(pg.num_values):
                var n = _read_u32le(span, off)
                off += 4
                dict.append(String(from_utf8=span[off : off + n]))
                off += n
            continue

        var vspan = Span(pg.body)[pg.value_offset :]
        var all_present = pg.num_present == pg.num_values
        if pg.encoding == ENC_PLAIN:
            var vi = 0
            for row in range(pg.num_values):
                if all_present or Int(pg.def_levels[row]) == leaf.max_def:
                    var n = _read_u32le(vspan, vi)
                    vi += 4
                    builder.append(String(from_utf8=vspan[vi : vi + n]))
                    vi += n
                else:
                    builder.append_null()
        else:
            var indices = rle_decode(vspan[1:], Int(vspan[0]), pg.num_present)
            var di = 0
            for row in range(pg.num_values):
                if all_present or Int(pg.def_levels[row]) == leaf.max_def:
                    builder.append(dict[Int(indices[di])])
                    di += 1
                else:
                    builder.append_null()
        produced += pg.num_values

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
    var pos = _page_start(meta)
    var produced = 0
    while produced < meta.num_values:
        var pg = _next_page(data, pos, meta, leaf, codecs)
        if pg.kind == PAGEKIND_DICT:
            raise Error("parquet: dictionary-encoded bool not supported")
        if pg.encoding != ENC_PLAIN:
            raise Error("parquet: non-plain bool encoding not supported")

        var vspan = Span(pg.body)[pg.value_offset :]
        var all_present = pg.num_present == pg.num_values
        var bitpos = 0
        for row in range(pg.num_values):
            if all_present or Int(pg.def_levels[row]) == leaf.max_def:
                var byte = vspan[bitpos >> 3]
                var b = (byte >> UInt8(bitpos & 7)) & 1
                bitpos += 1
                builder.append(b == 1)
            else:
                builder.append_null()
        produced += pg.num_values

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
    var mapped = MappedFile(path)
    var data = mapped.span()
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
    var result = Table.from_batches(parsed.schema, batches)
    # `data` is an untracked view into `mapped`; keep the map alive until every
    # value has been copied into owned Arrow buffers above, then unmap.
    _ = mapped^
    return result^
