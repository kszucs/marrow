"""Column-chunk decoding: turn a column chunk's byte range into an Arrow leaf
array. The whole read pipeline for one column lives here.

`PageReader` walks the chunk's pages (locating, decompressing, and decoding the
Dremel levels of each); `LeafBuilder` is the abstraction each physical layout
implements (fixed-width primitive, byte array, boolean); and `ColumnReader.decode`
drives a `PageReader` through the right builder, picking the flat vs leveled
path from the leaf's max repetition. Adding a type means adding a builder (or
reusing one) and a dispatch arm — no new free-standing decode loops.
"""

from std.sys import size_of
from std.memory import memcpy

from ..arrays import AnyArray, ArrayData
from ..buffers import Buffer, Bitmap
from ..builders import BinaryLikeBuilder, BoolBuilder, PrimitiveBuilder
from .. import dtypes as dt

from .codecs import Encoding, Rle, LittleEndian, Dictionary, Compression
from .utils import CompressionLibs
from .schema import LeafColumn, DecodedLeaf
from .format import ColumnMetaData, PageHeader, PageType
from .reader import RowSelection


# ---------------------------------------------------------------------------
# Page — one decoded page
# ---------------------------------------------------------------------------


struct Page(Movable):
    """A decoded page. `body` is a *view* — into the mmap for uncompressed pages
    (zero copy) or into the `PageReader`'s reused scratch for compressed pages —
    not an owned buffer, so no per-page allocation. `dictionary` marks the
    column chunk's dictionary page (vs a data page)."""

    var dictionary: Bool
    var body: Span[UInt8, ImmutUntrackedOrigin]
    var def_levels: List[Int32]
    var rep_levels: List[Int32]  # only populated for repeated (list) leaves
    var value_offset: Int
    var num_present: Int
    var encoding: Encoding
    var num_values: Int

    def __init__(out self, body: Span[UInt8, ImmutUntrackedOrigin]):
        self.dictionary = False
        self.body = body
        self.def_levels = List[Int32]()
        self.rep_levels = List[Int32]()
        self.value_offset = 0
        self.num_present = 0
        self.encoding = Encoding.PLAIN
        self.num_values = 0

    def all_present(self) -> Bool:
        return self.num_present == self.num_values

    def is_plain(self) -> Bool:
        return self.encoding.is_plain()

    def is_dictionary(self) -> Bool:
        return self.encoding.is_dictionary()

    def values(self) -> Span[UInt8, ImmutUntrackedOrigin]:
        """The value bytes (after any level prefix)."""
        return self.body[self.value_offset :]


# ---------------------------------------------------------------------------
# PageReader — iterate the pages of one column chunk: locate each page, read its
# Thrift header, decompress the body, and decode the definition/repetition
# levels, so the LeafBuilders below only ever see already-decoded pages.
# ---------------------------------------------------------------------------


struct PageReader[o: Origin[mut=False]](Movable):
    var data: Span[UInt8, Self.o]
    var meta: ColumnMetaData
    var leaf: LeafColumn
    var pos: Int
    var produced: Int  # data-page values yielded so far
    var scratch: List[UInt8]  # reused decompression buffer for compressed pages

    def __init__(
        out self,
        data: Span[UInt8, Self.o],
        var meta: ColumnMetaData,
        var leaf: LeafColumn,
    ):
        self.data = data
        if meta.dictionary_page_offset != -1:
            self.pos = meta.dictionary_page_offset
        else:
            self.pos = meta.data_page_offset
        self.meta = meta^
        self.leaf = leaf^
        self.produced = 0
        self.scratch = List[UInt8]()

    @staticmethod
    def _untracked(s: Span[UInt8, _]) -> Span[UInt8, ImmutUntrackedOrigin]:
        """Drop origin tracking on a byte span. The backing storage (the mmap,
        or this reader's decompression scratch) is kept alive for as long as each
        page is consumed, so the untracked view is always valid in use."""
        return rebind[Span[UInt8, ImmutUntrackedOrigin]](s)

    def _body(
        mut self,
        comp: Span[UInt8, _],
        uncompressed_size: Int,
        mut codecs: CompressionLibs,
    ) raises -> Span[UInt8, ImmutUntrackedOrigin]:
        """A view of the page body: the mmap bytes directly for uncompressed
        pages, or the reused scratch after decompressing."""
        var codec = Compression(self.meta.codec)
        if codec == Compression.UNCOMPRESSED:
            return Self._untracked(comp)
        codec.decompress_into(codecs, comp, uncompressed_size, self.scratch)
        return Self._untracked(Span(self.scratch))

    def has_next(self) -> Bool:
        return self.produced < self.meta.num_values

    def _decode_levels_v1(mut self, mut page: Page) raises:
        var body = page.body
        var cursor = 0
        var leveled = self.leaf.max_rep >= 1
        if leveled:
            var l = LittleEndian.u32(body, cursor)
            cursor += 4
            page.rep_levels = Rle.decode(
                body[cursor : cursor + l],
                Rle.bit_width(self.leaf.max_rep),
                page.num_values,
            )
            cursor += l
        page.num_present = page.num_values
        if self.leaf.max_def >= 1:
            var bw = Rle.bit_width(self.leaf.max_def)
            var l = LittleEndian.u32(body, cursor)
            cursor += 4
            var levels = body[cursor : cursor + l]
            if leveled:
                # nested columns need the full def levels to rebuild offsets
                page.def_levels = Rle.decode(levels, bw, page.num_values)
                var present = 0
                for d in page.def_levels:
                    if Int(d) == self.leaf.max_def:
                        present += 1
                page.num_present = present
            else:
                # flat: count present in O(1) for the all-present run; only
                # materialize the level list when there are actually nulls.
                page.num_present = Rle.count_matches(
                    levels, bw, page.num_values, Int32(self.leaf.max_def)
                )
                if page.num_present != page.num_values:
                    page.def_levels = Rle.decode(levels, bw, page.num_values)
            cursor += l
        page.value_offset = cursor

    def next(mut self, mut codecs: CompressionLibs) raises -> Page:
        var body_start = self.pos
        # advances body_start to the page body
        var ph = PageHeader.read_at(self.data, body_start)
        var comp = self.data[body_start : body_start + ph.compressed_page_size]
        self.pos = body_start + ph.compressed_page_size

        if ph.type == PageType.DICTIONARY:
            var page = Page(self._body(comp, ph.uncompressed_page_size, codecs))
            page.dictionary = True
            page.num_values = ph.dictionary_page_header.value().num_values
            return page^

        if ph.type == PageType.DATA:
            var page = Page(self._body(comp, ph.uncompressed_page_size, codecs))
            ref dph = ph.data_page_header.value()
            page.num_values = dph.num_values
            page.encoding = dph.encoding
            self._decode_levels_v1(page)
            self.produced += page.num_values
            return page^
        elif ph.type == PageType.DATA_V2:
            # v2 keeps the (uncompressed) levels ahead of the (maybe compressed)
            # values; assemble both into scratch, then view it.
            ref dph2 = ph.data_page_header_v2.value()
            var lvl_len = (
                dph2.repetition_levels_byte_length
                + dph2.definition_levels_byte_length
            )
            self.scratch.clear()
            self.scratch.extend(comp[0:lvl_len])
            if dph2.is_compressed:
                self.scratch.extend(
                    Span(
                        Compression(self.meta.codec).decompress(
                            codecs,
                            comp[lvl_len:],
                            ph.uncompressed_page_size - lvl_len,
                        )
                    )
                )
            else:
                self.scratch.extend(comp[lvl_len:])

            var page = Page(Self._untracked(Span(self.scratch)))
            page.num_values = dph2.num_values
            page.encoding = dph2.encoding
            if (
                self.leaf.max_rep >= 1
                and dph2.repetition_levels_byte_length > 0
            ):
                page.rep_levels = Rle.decode(
                    page.body[0 : dph2.repetition_levels_byte_length],
                    Rle.bit_width(self.leaf.max_rep),
                    page.num_values,
                )
            var cursor = dph2.repetition_levels_byte_length
            if (
                self.leaf.max_def >= 1
                and dph2.definition_levels_byte_length > 0
            ):
                page.def_levels = Rle.decode(
                    page.body[
                        cursor : cursor + dph2.definition_levels_byte_length
                    ],
                    Rle.bit_width(self.leaf.max_def),
                    page.num_values,
                )
            page.value_offset = lvl_len
            page.num_present = page.num_values - dph2.num_nulls
            self.produced += page.num_values
            return page^
        else:
            raise Error("parquet: unexpected page type")

    def peek(self) raises -> Tuple[Bool, Int]:
        """Inspect the next page without consuming it: `(is_dictionary,
        num_values)`. Lets the caller decide, from the page's row count, whether
        to decode, skip, or partially select it before paying the decode."""
        var p = self.pos
        var ph = PageHeader.read_at(self.data, p)
        if ph.type == PageType.DICTIONARY:
            return (True, ph.dictionary_page_header.value().num_values)
        elif ph.type == PageType.DATA:
            return (False, ph.data_page_header.value().num_values)
        elif ph.type == PageType.DATA_V2:
            return (False, ph.data_page_header_v2.value().num_values)
        else:
            raise Error("parquet: unexpected page type")

    def skip_next(mut self) raises -> Int:
        """Advance past the next data page without decompressing or decoding it
        (the row-skip fast path); return the number of rows skipped."""
        var body_start = self.pos
        var ph = PageHeader.read_at(self.data, body_start)
        self.pos = body_start + ph.compressed_page_size
        var nv: Int
        if ph.type == PageType.DATA:
            nv = ph.data_page_header.value().num_values
        elif ph.type == PageType.DATA_V2:
            nv = ph.data_page_header_v2.value().num_values
        else:
            raise Error("parquet: skip_next on a non-data page")
        self.produced += nv
        return nv


# ---------------------------------------------------------------------------
# LeafBuilder — accumulates decoded pages into one Arrow array
# ---------------------------------------------------------------------------


trait LeafBuilder(ImplicitlyDeletable, Movable):
    """Accumulates the values of a column chunk, page by page, into an array."""

    def consume(mut self, var page: Page) raises:
        ...

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        """Append only the rows of `page` where `mask[row]` is set (page-relative
        row index) — used for a data page partially covered by a row selection.
        """
        ...

    def finish(deinit self) raises -> AnyArray:
        ...


struct PrimitiveLeafBuilder[store_dt: DType, phys_dt: DType = store_dt](
    LeafBuilder
):
    """Fixed-width values decoded straight into a contiguous buffer.

    `phys_dt` is the Parquet physical width read from the file, `store_dt` the
    Arrow storage width. When they match (the common case, incl. temporal types)
    a whole all-present PLAIN page is one memcpy; when they differ (dt.int8/16 stored
    as physical INT32) each value is read wide and narrowed.
    """

    comptime SAME = Self.store_dt == Self.phys_dt

    var dtype: dt.AnyDataType
    var max_def: Int
    var num_rows: Int
    var values: Buffer[mut=True]
    var bitmap: Bitmap[mut=True]
    var has_bitmap: Bool  # materialized only once a null actually appears
    var dict: List[Scalar[Self.store_dt]]
    var wpos: Int
    var null_count: Int

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.dtype = leaf.dtype.copy()
        self.max_def = leaf.max_def
        self.num_rows = num_rows
        self.values = Buffer.alloc_uninit[Self.store_dt](num_rows)
        self.bitmap = Bitmap[mut=True].alloc_zeroed(0)
        self.has_bitmap = False
        self.dict = List[Scalar[Self.store_dt]]()
        self.wpos = 0
        self.null_count = 0

    def _read(self, span: Span[UInt8, _], off: Int) -> Scalar[Self.store_dt]:
        return LittleEndian.fixed[Self.phys_dt](span, off).cast[Self.store_dt]()

    def _ensure_bitmap(mut self):
        """Allocate the validity bitmap on first null, backfilling all values
        written so far as valid (they were present). No-op if already built."""
        if not self.has_bitmap:
            self.bitmap = Bitmap[mut=True].alloc_zeroed(self.num_rows)
            self.bitmap.set_range(0, self.wpos, True)
            self.has_bitmap = True

    def _scatter(
        mut self,
        page: Page,
        present: UnsafePointer[Scalar[Self.store_dt], _],
    ) raises:
        """Place `page.num_present` contiguous decoded values into the output
        buffer, honoring definition levels — one memcpy when the page is
        all-present, else a per-row scatter that materializes the validity
        bitmap. Every encoding funnels its decoded present values through here.
        """
        var vptr = self.values.view[Self.store_dt]().unsafe_ptr()
        if page.all_present():
            memcpy(dest=vptr + self.wpos, src=present, count=page.num_present)
            self.wpos += page.num_present
            if self.has_bitmap:
                self.bitmap.set_range(
                    self.wpos - page.num_present, page.num_present, True
                )
        else:
            self._ensure_bitmap()
            var vi = 0
            for row in range(page.num_values):
                if Int(page.def_levels[row]) == self.max_def:
                    vptr[self.wpos] = present[vi]
                    vi += 1
                    self.bitmap.set(self.wpos)
                else:
                    vptr[self.wpos] = 0
                    self.null_count += 1
                self.wpos += 1

    def consume(mut self, var page: Page) raises:
        comptime PW = size_of[Scalar[Self.phys_dt]]()
        if page.dictionary:
            comptime if Self.SAME:
                self.dict.resize(unsafe_uninit_length=page.num_values)
                memcpy(
                    dest=self.dict.unsafe_ptr(),
                    src=page.body.unsafe_ptr().bitcast[Scalar[Self.store_dt]](),
                    count=page.num_values,
                )
            else:
                var span = page.body
                for i in range(page.num_values):
                    self.dict.append(self._read(span, i * PW))
            return

        var vspan = page.values()
        if page.is_plain() and Self.SAME:
            # fast path: PLAIN stores only present values, contiguous and already
            # the store width — scatter straight from the page (no copy).
            self._scatter(
                page, vspan.unsafe_ptr().bitcast[Scalar[Self.store_dt]]()
            )
        elif page.is_dictionary() and page.all_present():
            # fast path: fused index-decode + gather straight to the output.
            Rle.gather[Self.store_dt](
                vspan[1:],
                Int(vspan[0]),
                page.num_values,
                self.dict.unsafe_ptr(),
                self.values.view[Self.store_dt]().unsafe_ptr() + self.wpos,
            )
            self.wpos += page.num_values
        else:
            # every other encoding (narrow PLAIN, nullable dict, DELTA,
            # BYTE_STREAM_SPLIT) shares one decoder with the nested path.
            var present = List[Scalar[Self.store_dt]](capacity=page.num_present)
            page.encoding.decode_primitive[Self.store_dt, Self.phys_dt](
                page.values(), page.num_present, self.dict, present
            )
            self._scatter(page, present.unsafe_ptr())

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        # Decode every present value through the general per-encoding decoder
        # (PLAIN / dictionary / DELTA / BYTE_STREAM_SPLIT), then place only the
        # rows `mask` selects. Partial pages are the page-boundary minority, so
        # skipping the fast paths here costs little.
        var present = List[Scalar[Self.store_dt]](capacity=page.num_present)
        page.encoding.decode_primitive[Self.store_dt, Self.phys_dt](
            page.values(), page.num_present, self.dict, present
        )
        var vptr = self.values.view[Self.store_dt]().unsafe_ptr()
        self._ensure_bitmap()  # selected present rows are set valid explicitly
        var vi = 0
        for row in range(page.num_values):
            var present_here = page.all_present() or (
                Int(page.def_levels[row]) == self.max_def
            )
            if mask[row]:
                if present_here:
                    vptr[self.wpos] = present[vi]
                    self.bitmap.set(self.wpos)
                else:
                    vptr[self.wpos] = 0
                    self.null_count += 1
                self.wpos += 1
            if present_here:
                vi += 1

    def finish(deinit self) raises -> AnyArray:
        var buffers = List[Buffer[mut=False]]()
        buffers.append(self.values^.to_immutable())
        var bm: Optional[Bitmap[mut=False]] = None
        if self.null_count > 0:
            bm = self.bitmap^.to_immutable(length=self.wpos)
        return AnyArray.from_data(
            ArrayData(
                dtype=self.dtype^,
                length=self.wpos,
                nulls=self.null_count,
                offset=0,
                bitmap=bm^,
                buffers=buffers^,
                children=List[ArrayData](),
            )
        )


struct ByteArrayLeafBuilder[BT: dt.BinaryLikeType](LeafBuilder):
    """Variable-length byte values (string/binary, 32- or 64-bit offsets). Bytes
    are appended verbatim, so the same builder serves UTF-8 and dt.binary."""

    var builder: BinaryLikeBuilder[Self.BT]
    var max_def: Int
    var dict_body: List[UInt8]
    var dict_off: List[Int]
    var dict_len: List[Int]

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.builder = BinaryLikeBuilder[Self.BT](num_rows)
        self.max_def = leaf.max_def
        self.dict_body = List[UInt8]()
        self.dict_off = List[Int]()
        self.dict_len = List[Int]()

    def _append(mut self, span: Span[UInt8, _]) raises:
        self.builder.append(StringSlice(unsafe_from_utf8=span))

    def _scatter_values(mut self, page: Page, values: List[List[UInt8]]) raises:
        """Append the `page.num_present` decoded present values into the builder,
        honoring definition levels — the shared placement path for the encodings
        that materialize their values (dictionary, DELTA_*)."""
        var vi = 0
        for row in range(page.num_values):
            if page.all_present() or Int(page.def_levels[row]) == self.max_def:
                self._append(Span(values[vi]))
                vi += 1
            else:
                self.builder.append_null()

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            self.dict_body = List[UInt8]()
            self.dict_body.extend(page.body)
            var span = Span(self.dict_body)
            var off = 0
            for _ in range(page.num_values):
                var n = LittleEndian.u32(span, off)
                off += 4
                self.dict_off.append(off)
                self.dict_len.append(n)
                off += n
            return

        var vspan = page.values()
        if page.is_plain():
            # PLAIN is decoded in place (no per-value copy): walk the length-
            # prefixed present values, appending or emitting nulls by def level.
            var vi = 0
            for row in range(page.num_values):
                if page.all_present() or Int(page.def_levels[row]) == (
                    self.max_def
                ):
                    var n = LittleEndian.u32(vspan, vi)
                    vi += 4
                    self._append(vspan[vi : vi + n])
                    vi += n
                else:
                    self.builder.append_null()
        else:
            # dictionary and DELTA_* share one decoder with the nested path.
            var values = page.encoding.decode_bytes(
                page.values(),
                page.num_present,
                self.dict_body,
                self.dict_off,
                self.dict_len,
            )
            self._scatter_values(page, values)

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        var vspan = page.values()
        if page.is_plain():
            var vi = 0
            for row in range(page.num_values):
                if page.all_present() or Int(page.def_levels[row]) == (
                    self.max_def
                ):
                    var n = LittleEndian.u32(vspan, vi)
                    vi += 4
                    if mask[row]:
                        self._append(vspan[vi : vi + n])
                    vi += n
                else:
                    if mask[row]:
                        self.builder.append_null()
        else:
            var values = page.encoding.decode_bytes(
                page.values(),
                page.num_present,
                self.dict_body,
                self.dict_off,
                self.dict_len,
            )
            var vi = 0
            for row in range(page.num_values):
                if page.all_present() or Int(page.def_levels[row]) == (
                    self.max_def
                ):
                    if mask[row]:
                        self._append(Span(values[vi]))
                    vi += 1
                else:
                    if mask[row]:
                        self.builder.append_null()

    def finish(deinit self) raises -> AnyArray:
        var b = self.builder^
        var out: AnyArray = b.finish()
        return out^


struct BoolLeafBuilder(LeafBuilder):
    """Bit-packed PLAIN booleans."""

    var builder: BoolBuilder
    var max_def: Int

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.builder = BoolBuilder(num_rows)
        self.max_def = leaf.max_def

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            raise Error("parquet: dictionary-encoded bool not supported")
        if not page.is_plain():
            raise Error("parquet: non-plain bool encoding not supported")
        var vspan = page.values()
        var all_present = page.all_present()
        var bitpos = 0
        for row in range(page.num_values):
            if all_present or Int(page.def_levels[row]) == self.max_def:
                var byte = vspan[bitpos >> 3]
                var b = (byte >> UInt8(bitpos & 7)) & 1
                bitpos += 1
                self.builder.append(b == 1)
            else:
                self.builder.append_null()

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        if not page.is_plain():
            raise Error("parquet: non-plain bool encoding not supported")
        var vspan = page.values()
        var all_present = page.all_present()
        var bitpos = 0
        for row in range(page.num_values):
            if all_present or Int(page.def_levels[row]) == self.max_def:
                var byte = vspan[bitpos >> 3]
                var b = (byte >> UInt8(bitpos & 7)) & 1
                bitpos += 1
                if mask[row]:
                    self.builder.append(b == 1)
            else:
                if mask[row]:
                    self.builder.append_null()

    def finish(deinit self) raises -> AnyArray:
        var b = self.builder^
        var out: AnyArray = b.finish()
        return out^


# ---------------------------------------------------------------------------
# ColumnReader — decode one column chunk into a DecodedLeaf
# ---------------------------------------------------------------------------


struct ColumnReader[o: Origin[mut=False]](Movable):
    """Decode one column chunk. `decode` picks the path from the leaf's max
    repetition: a flat leaf (`max_rep == 0`) fills a fixed-size `LeafBuilder`
    with the PLAIN-memcpy / fused-gather fast paths; a repeated (list-element)
    leaf grows a builder while accumulating the Dremel rep/def levels the
    assembler folds into list offsets. Both share the per-encoding value
    decoders on `Encoding`, so every encoding works on either path."""

    var pages: PageReader[Self.o]
    var num_rows: Int
    var leveled: Bool
    var def_out: List[Int32]  # per-row def levels, kept only when carry_def
    var selection: Optional[RowSelection]  # None = decode every row

    def __init__(
        out self,
        data: Span[UInt8, Self.o],
        var meta: ColumnMetaData,
        var leaf: LeafColumn,
        num_rows: Int,
        var selection: Optional[RowSelection] = None,
    ):
        var leveled = leaf.max_rep >= 1
        self.pages = PageReader(data, meta^, leaf^)
        self.num_rows = num_rows
        self.leveled = leveled
        self.def_out = List[Int32]()
        self.selection = selection^

    def decode(mut self, mut codecs: CompressionLibs) raises -> DecodedLeaf:
        if self.leveled:
            # nested columns decode in full (page skipping is a flat-path
            # optimisation); a Filter above the scan still applies the predicate.
            return self._decode_leveled(codecs)
        else:
            return self._decode_flat(codecs)

    # -----------------------------------------------------------------------
    # Flat path — fixed-size LeafBuilder, one memcpy/gather per all-present page
    # -----------------------------------------------------------------------

    def _run[
        B: LeafBuilder
    ](mut self, mut builder: B, mut codecs: CompressionLibs) raises:
        if self.selection:
            return self._run_selected(builder, codecs)
        var carry = self.pages.leaf.carry_def
        var md = self.pages.leaf.max_def
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if carry and not pg.dictionary:
                # a page materializes def levels only when it has nulls; an
                # all-present page implies every slot is at max_def.
                if len(pg.def_levels) == pg.num_values:
                    self.def_out.extend(Span(pg.def_levels))
                else:
                    for _ in range(pg.num_values):
                        self.def_out.append(Int32(md))
            builder.consume(pg^)

    def _run_selected[
        B: LeafBuilder
    ](mut self, mut builder: B, mut codecs: CompressionLibs) raises:
        """Flat decode honoring a row selection: a data page whose rows are all
        deselected is skipped without decoding; a fully selected page decodes
        normally; a partially selected page keeps only its chosen rows. For flat
        columns `pages.produced` is the row-group-relative index of the next data
        page's first row, so the page's rows are `[produced, produced + nv)`."""
        ref sel = self.selection.value()
        while self.pages.has_next():
            var is_dict, nv = self.pages.peek()
            if is_dict:
                builder.consume(self.pages.next(codecs))  # build the dictionary
            else:
                var start = self.pages.produced
                var kept = sel.selected_in(start, nv)
                if kept == 0:
                    _ = self.pages.skip_next()
                elif kept == nv:
                    builder.consume(self.pages.next(codecs))
                else:
                    var m = sel.mask(start, nv)
                    builder.consume_selected(self.pages.next(codecs), m)

    def _build[
        B: LeafBuilder
    ](mut self, var builder: B, mut codecs: CompressionLibs) raises -> AnyArray:
        self._run(builder, codecs)
        return builder^.finish()

    def _decode_flat(
        mut self, mut codecs: CompressionLibs
    ) raises -> DecodedLeaf:
        var arr = self._flat_array(codecs)
        var defs = self.def_out^
        self.def_out = List[Int32]()
        return DecodedLeaf(False, arr^, List[Int32](), defs^)

    def _flat_array(mut self, mut codecs: CompressionLibs) raises -> AnyArray:
        ref leaf = self.pages.leaf
        ref vt = leaf.dtype
        if vt == dt.int32:
            return self._build(
                PrimitiveLeafBuilder[DType.int32](self.num_rows, leaf), codecs
            )
        elif vt == dt.int64:
            return self._build(
                PrimitiveLeafBuilder[DType.int64](self.num_rows, leaf), codecs
            )
        elif vt == dt.uint32:
            return self._build(
                PrimitiveLeafBuilder[DType.uint32](self.num_rows, leaf), codecs
            )
        elif vt == dt.uint64:
            return self._build(
                PrimitiveLeafBuilder[DType.uint64](self.num_rows, leaf), codecs
            )
        elif vt == dt.float32:
            return self._build(
                PrimitiveLeafBuilder[DType.float32](self.num_rows, leaf), codecs
            )
        elif vt == dt.float64:
            return self._build(
                PrimitiveLeafBuilder[DType.float64](self.num_rows, leaf), codecs
            )
        elif vt == dt.int8:
            return self._build(
                PrimitiveLeafBuilder[DType.int8, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif vt == dt.int16:
            return self._build(
                PrimitiveLeafBuilder[DType.int16, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif vt == dt.uint8:
            return self._build(
                PrimitiveLeafBuilder[DType.uint8, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif vt == dt.uint16:
            return self._build(
                PrimitiveLeafBuilder[DType.uint16, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif vt.is_date32() or vt.is_time32():
            return self._build(
                PrimitiveLeafBuilder[DType.int32](self.num_rows, leaf), codecs
            )
        elif (
            vt.is_timestamp()
            or vt.is_time64()
            or vt.is_date64()
            or vt.is_duration()
        ):
            return self._build(
                PrimitiveLeafBuilder[DType.int64](self.num_rows, leaf), codecs
            )
        elif vt == dt.bool_:
            return self._build(BoolLeafBuilder(self.num_rows, leaf), codecs)
        elif vt.is_string():
            return self._build(
                ByteArrayLeafBuilder[dt.StringType](self.num_rows, leaf), codecs
            )
        elif vt.is_large_string():
            return self._build(
                ByteArrayLeafBuilder[dt.LargeStringType](self.num_rows, leaf),
                codecs,
            )
        elif vt.is_binary():
            return self._build(
                ByteArrayLeafBuilder[dt.BinaryType](self.num_rows, leaf), codecs
            )
        elif vt.is_large_binary():
            return self._build(
                ByteArrayLeafBuilder[dt.LargeBinaryType](self.num_rows, leaf),
                codecs,
            )
        else:
            raise Error("parquet: unsupported column type " + String(vt))

    # -----------------------------------------------------------------------
    # Leveled path — grow the element array while accumulating rep/def levels.
    # A slot holds an element when its def reaches `floor`; the element is
    # present at `max_def`, else null. Present values come from the shared
    # `Encoding` decoders, so every encoding works here just as on the flat path.
    # -----------------------------------------------------------------------

    def _drive_primitive[
        T: dt.NumericType, phys: DType
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        var builder = PrimitiveBuilder[T](self.num_rows)
        var dict = List[Scalar[T.native]]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if pg.dictionary:
                Dictionary.decode_page_primitive[T.native, phys](
                    pg.body, pg.num_values, dict
                )
                continue
            rep_out.extend(Span(pg.rep_levels))
            def_out.extend(Span(pg.def_levels))
            var present = List[Scalar[T.native]]()
            pg.encoding.decode_primitive[T.native, phys](
                pg.values(), pg.num_present, dict, present
            )
            var vi = 0
            for k in range(pg.num_values):
                var d = Int(pg.def_levels[k])
                if d < floor:
                    continue
                if d == max_def:
                    builder.append(present[vi])
                    vi += 1
                else:
                    builder.append_null()
        var arr = builder.finish()
        return DecodedLeaf(True, arr^, rep_out^, def_out^)

    def _drive_bytes[
        BT: dt.BinaryLikeType
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        var builder = BinaryLikeBuilder[BT](self.num_rows)
        var dict_body = List[UInt8]()
        var dict_off = List[Int]()
        var dict_len = List[Int]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if pg.dictionary:
                Dictionary.decode_page_bytes(
                    pg.body, pg.num_values, dict_body, dict_off, dict_len
                )
                continue
            rep_out.extend(Span(pg.rep_levels))
            def_out.extend(Span(pg.def_levels))
            var values = pg.encoding.decode_bytes(
                pg.values(), pg.num_present, dict_body, dict_off, dict_len
            )
            var vi = 0
            for k in range(pg.num_values):
                var d = Int(pg.def_levels[k])
                if d < floor:
                    continue
                if d == max_def:
                    builder.append(
                        StringSlice(unsafe_from_utf8=Span(values[vi]))
                    )
                    vi += 1
                else:
                    builder.append_null()
        var arr = builder.finish()
        return DecodedLeaf(True, arr^, rep_out^, def_out^)

    def _drive_bool(
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        var builder = BoolBuilder(self.num_rows)
        var rep_out = List[Int32]()
        var def_out = List[Int32]()
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if pg.dictionary:
                raise Error("parquet: dictionary-encoded bool not supported")
            rep_out.extend(Span(pg.rep_levels))
            def_out.extend(Span(pg.def_levels))
            var present = pg.encoding.decode_bool(pg.values(), pg.num_present)
            var vi = 0
            for k in range(pg.num_values):
                var d = Int(pg.def_levels[k])
                if d < floor:
                    continue
                if d == max_def:
                    builder.append(present[vi])
                    vi += 1
                else:
                    builder.append_null()
        var arr = builder.finish()
        return DecodedLeaf(True, arr^, rep_out^, def_out^)

    def _decode_leveled(
        mut self, mut codecs: CompressionLibs
    ) raises -> DecodedLeaf:
        ref leaf = self.pages.leaf
        ref vt = leaf.dtype
        # a value slot exists at/above the leaf's repetition floor (the innermost
        # enclosing list's element level); the value is present at max_def.
        var f = leaf.rep_floor
        var md = leaf.max_def
        if vt == dt.int32:
            return self._drive_primitive[dt.Int32Type, DType.int32](
                codecs, f, md
            )
        elif vt == dt.int64:
            return self._drive_primitive[dt.Int64Type, DType.int64](
                codecs, f, md
            )
        elif vt == dt.uint32:
            return self._drive_primitive[dt.UInt32Type, DType.uint32](
                codecs, f, md
            )
        elif vt == dt.uint64:
            return self._drive_primitive[dt.UInt64Type, DType.uint64](
                codecs, f, md
            )
        elif vt == dt.float32:
            return self._drive_primitive[dt.Float32Type, DType.float32](
                codecs, f, md
            )
        elif vt == dt.float64:
            return self._drive_primitive[dt.Float64Type, DType.float64](
                codecs, f, md
            )
        elif vt == dt.int8:
            return self._drive_primitive[dt.Int8Type, DType.int32](
                codecs, f, md
            )
        elif vt == dt.int16:
            return self._drive_primitive[dt.Int16Type, DType.int32](
                codecs, f, md
            )
        elif vt == dt.uint8:
            return self._drive_primitive[dt.UInt8Type, DType.int32](
                codecs, f, md
            )
        elif vt == dt.uint16:
            return self._drive_primitive[dt.UInt16Type, DType.int32](
                codecs, f, md
            )
        elif vt == dt.bool_:
            return self._drive_bool(codecs, f, md)
        elif vt.is_string():
            return self._drive_bytes[dt.StringType](codecs, f, md)
        elif vt.is_large_string():
            return self._drive_bytes[dt.LargeStringType](codecs, f, md)
        elif vt.is_binary():
            return self._drive_bytes[dt.BinaryType](codecs, f, md)
        elif vt.is_large_binary():
            return self._drive_bytes[dt.LargeBinaryType](codecs, f, md)
        else:
            raise Error("parquet: unsupported list element type " + String(vt))
