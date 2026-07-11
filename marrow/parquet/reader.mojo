"""Native Parquet deserialization → Arrow: the full read path.

`read_table` memory-maps the file, reads the footer metadata, and decodes each
selected column chunk's pages into an Arrow leaf array via a `ColumnReader`,
folding the results back into the Arrow type tree with `SchemaNode.assemble`.
Predicate pushdown skips whole row groups (`row_groups`) and individual pages
(`row_selections`). The column-decode engine — `Page`, `PageReader`,
`LeafBuilder` and its concrete builders, and `ColumnReader` — lives here too, so
this module is the entire deserialization layer; the metadata / statistics /
page-index readers reuse the same footer decode without touching column data.
Milestone: flat columns + struct nesting; primitives, string/binary; PLAIN and
dictionary encodings; v1/v2 pages.
"""

from std.ffi import external_call
from std.io.file import FileHandle
from std.algorithm.functional import sync_parallelize
from std.sys import size_of
from std.sys.info import num_physical_cores
from std.memory import memcpy

from ..arrays import AnyArray, ArrayData
from ..buffers import Buffer, Bitmap
from ..builders import (
    BinaryLikeBuilder,
    BoolBuilder,
    FixedSizeBinaryBuilder,
    PrimitiveBuilder,
)
from ..schema import Schema
from ..tabular import Table, RecordBatch
from ..scalars import (
    AnyScalar,
    BoolScalar,
    StringScalar,
    Int8Scalar,
    Int16Scalar,
    Int32Scalar,
    Int64Scalar,
    UInt8Scalar,
    UInt16Scalar,
    UInt32Scalar,
    UInt64Scalar,
    Float32Scalar,
    Float64Scalar,
)
from .. import dtypes as dt

from .utils import CompressionLibs
from .codecs import Encoding, Rle, LittleEndian, Dictionary, Compression
from .schema import SchemaMapping, Projection, DecodedLeaf, LeafColumn
from .format import (
    FileMetaData,
    CompactReader,
    ColumnIndex,
    OffsetIndex,
    ColumnMetaData,
    PageHeader,
    PageType,
)


# ---------------------------------------------------------------------------
# Memory-mapped file — read zero-copy instead of copying the file in
# ---------------------------------------------------------------------------


struct MappedFile(Movable):
    """A read-only mmap of the whole file, unmapped when the value drops.
    Decoded values are copied into owned Arrow buffers, so the map only needs to
    outlive the decode."""

    var ptr: UnsafePointer[UInt8, ImmutUntrackedOrigin]
    var size: Int

    def __init__(out self, path: String) raises:
        # Mojo's file open (the libc variadic `open` cannot be external_call'd in
        # an archive build); mmap/lseek are plain syscalls and are fine.
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

    def __init__(
        out self,
        *,
        body: Span[UInt8, ImmutUntrackedOrigin],
        num_values: Int,
        num_present: Int,
        value_offset: Int = 0,
        encoding: Encoding = Encoding.PLAIN,
        var rep_levels: List[Int32] = [],
        var def_levels: List[Int32] = [],
        dictionary: Bool = False,
    ):
        self.dictionary = dictionary
        self.body = body
        self.def_levels = def_levels^
        self.rep_levels = rep_levels^
        self.value_offset = value_offset
        self.num_present = num_present
        self.encoding = encoding
        self.num_values = num_values

    @staticmethod
    def dictionary_page(
        body: Span[UInt8, ImmutUntrackedOrigin], num_values: Int
    ) -> Page:
        """The column chunk's dictionary page — carries only its distinct values;
        no levels, no present/null distinction."""
        return Page(
            body=body, num_values=num_values, num_present=num_values,
            dictionary=True,
        )

    def all_present(self) -> Bool:
        return self.num_present == self.num_values

    def present_at(self, row: Int, max_def: Int) -> Bool:
        """Whether output `row` holds a present value: the whole page is present,
        or its definition level reaches the leaf's max (a value slot, not a
        null). The def-level scatter every `LeafBuilder` shares pivots on this."""
        return self.all_present() or Int(self.def_levels[row]) == max_def

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

    def _data_page_v1(
        self,
        var body: Span[UInt8, ImmutUntrackedOrigin],
        num_values: Int,
        encoding: Encoding,
    ) raises -> Page:
        """Decode a v1 data page's leading rep/def level streams and build the
        `Page` — the single construction site for v1 pages (no field-by-field
        mutation)."""
        var reps = List[Int32]()
        var defs = List[Int32]()
        var num_present = num_values
        var cursor = 0
        var leveled = self.leaf.max_rep >= 1
        if leveled:
            var l = LittleEndian.u32(body, cursor)
            cursor += 4
            reps = Rle.decode(
                body[cursor : cursor + l],
                Rle.bit_width(self.leaf.max_rep),
                num_values,
            )
            cursor += l
        if self.leaf.max_def >= 1:
            var bw = Rle.bit_width(self.leaf.max_def)
            var l = LittleEndian.u32(body, cursor)
            cursor += 4
            var levels = body[cursor : cursor + l]
            if leveled:
                # nested columns need the full def levels to rebuild offsets
                defs = Rle.decode(levels, bw, num_values)
                var present = 0
                for d in defs:
                    if Int(d) == self.leaf.max_def:
                        present += 1
                num_present = present
            else:
                # flat: count present in O(1) for the all-present run; only
                # materialize the level list when there are actually nulls.
                num_present = Rle.count_matches(
                    levels, bw, num_values, Int32(self.leaf.max_def)
                )
                if num_present != num_values:
                    defs = Rle.decode(levels, bw, num_values)
            cursor += l
        return Page(
            body=body,
            num_values=num_values,
            num_present=num_present,
            value_offset=cursor,
            encoding=encoding,
            rep_levels=reps^,
            def_levels=defs^,
        )

    def next(mut self, mut codecs: CompressionLibs) raises -> Page:
        var body_start = self.pos
        # advances body_start to the page body
        var ph = PageHeader.read_at(self.data, body_start)
        var comp = self.data[body_start : body_start + ph.compressed_page_size]
        self.pos = body_start + ph.compressed_page_size

        if ph.type == PageType.DICTIONARY:
            return Page.dictionary_page(
                self._body(comp, ph.uncompressed_page_size, codecs),
                ph.dictionary_page_header.value().num_values,
            )

        if ph.type == PageType.DATA:
            ref dph = ph.data_page_header.value()
            self.produced += dph.num_values
            return self._data_page_v1(
                self._body(comp, ph.uncompressed_page_size, codecs),
                dph.num_values,
                dph.encoding,
            )
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

            var body = Self._untracked(Span(self.scratch))
            var reps = List[Int32]()
            var defs = List[Int32]()
            if (
                self.leaf.max_rep >= 1
                and dph2.repetition_levels_byte_length > 0
            ):
                reps = Rle.decode(
                    body[0 : dph2.repetition_levels_byte_length],
                    Rle.bit_width(self.leaf.max_rep),
                    dph2.num_values,
                )
            var cursor = dph2.repetition_levels_byte_length
            if (
                self.leaf.max_def >= 1
                and dph2.definition_levels_byte_length > 0
            ):
                defs = Rle.decode(
                    body[cursor : cursor + dph2.definition_levels_byte_length],
                    Rle.bit_width(self.leaf.max_def),
                    dph2.num_values,
                )
            self.produced += dph2.num_values
            return Page(
                body=body,
                num_values=dph2.num_values,
                num_present=dph2.num_values - dph2.num_nulls,
                value_offset=lvl_len,
                encoding=dph2.encoding,
                rep_levels=reps^,
                def_levels=defs^,
            )
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


def _walk_slots[
    body: def (Bool, Bool, Int) raises capturing [_] -> None,
](page: Page, max_def: Int, mask: Optional[List[Bool]]) raises:
    """Walk a data page's `num_values` output slots — the flat scatter skeleton
    every `LeafBuilder` shares. For each row `body(present, selected, vi)` fires
    with whether the slot holds a present value (`page.present_at`), whether the
    selection mask keeps it, and the running index into the page's decoded
    present values (advanced on every present slot, regardless of the mask). Only
    the per-slot placement differs across leaves; it lives entirely in `body`."""
    var vi = 0
    for row in range(page.num_values):
        var present_here = page.present_at(row, max_def)
        var selected = not mask or mask.value()[row]
        body(present_here, selected, vi)
        if present_here:
            vi += 1


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
        mask: Optional[List[Bool]] = None,
    ) raises:
        """Place `page.num_present` contiguous decoded values into the output
        buffer, honoring definition levels — one memcpy when the page is
        all-present and fully selected, else a per-row scatter that materializes
        the validity bitmap. With `mask`, only the rows it selects are placed
        (the page-boundary partial-page path). Every encoding funnels its decoded
        present values through here."""
        var vptr = self.values.view[Self.store_dt]().unsafe_ptr()
        if not mask and page.all_present():
            memcpy(dest=vptr + self.wpos, src=present, count=page.num_present)
            if self.has_bitmap:
                self.bitmap.set_range(self.wpos, page.num_present, True)
            self.wpos += page.num_present
            return
        self._ensure_bitmap()

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    vptr[self.wpos] = present[vi]
                    self.bitmap.set(self.wpos)
                else:
                    vptr[self.wpos] = 0
                    self.null_count += 1
                self.wpos += 1

        _walk_slots[place](page, self.max_def, mask)

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
                Span(self.dict),
                self.values.view[Self.store_dt]().as_span(),
                self.wpos,
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
        # skipping `consume`'s fast paths here costs little.
        var present = List[Scalar[Self.store_dt]](capacity=page.num_present)
        page.encoding.decode_primitive[Self.store_dt, Self.phys_dt](
            page.values(), page.num_present, self.dict, present
        )
        self._scatter(page, present.unsafe_ptr(), mask.copy())

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

    def _scatter_values(
        mut self,
        page: Page,
        values: List[List[UInt8]],
        mask: Optional[List[Bool]] = None,
    ) raises:
        """Append the decoded present values honoring definition levels — the
        shared placement path for the materializing encodings (dictionary,
        DELTA_*). With `mask`, only the selected rows are appended."""

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    self._append(Span(values[vi]))
                else:
                    self.builder.append_null()

        _walk_slots[place](page, self.max_def, mask)

    def _place_plain(
        mut self,
        page: Page,
        vspan: Span[UInt8, _],
        mask: Optional[List[Bool]] = None,
    ) raises:
        """PLAIN in place: walk the length-prefixed present values, appending or
        emitting nulls by def level. With `mask`, only selected rows are kept.
        The byte cursor advances past every present value (even masked-out ones)
        since PLAIN stores no offsets."""
        var bpos = 0

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if present_here:
                var n = LittleEndian.u32(vspan, bpos)
                bpos += 4
                if selected:
                    self._append(vspan[bpos : bpos + n])
                bpos += n
            elif selected:
                self.builder.append_null()

        _walk_slots[place](page, self.max_def, mask)

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
            self._place_plain(page, vspan)
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
            self._place_plain(page, vspan, mask.copy())
        else:
            var values = page.encoding.decode_bytes(
                page.values(),
                page.num_present,
                self.dict_body,
                self.dict_off,
                self.dict_len,
            )
            self._scatter_values(page, values, mask.copy())

    def finish(deinit self) raises -> AnyArray:
        var b = self.builder^
        var out: AnyArray = b.finish()
        return out^


struct DecimalLeafBuilder[native: DType](LeafBuilder):
    """FIXED_LEN_BYTE_ARRAY decimals (decimal128 -> int128, decimal256 -> int256).

    Each value is big-endian two's-complement of the schema's `type_length`
    bytes — which PyArrow minimises per precision, so it may be narrower than the
    16/32-byte storage — sign-extended into the native width. PLAIN and
    RLE_DICTIONARY only, the encodings PyArrow emits for decimals."""

    comptime FULL = size_of[Scalar[Self.native]]()

    var dtype: dt.AnyDataType
    var max_def: Int
    var num_rows: Int
    var width: Int
    var values: Buffer[mut=True]
    var bitmap: Bitmap[mut=True]
    var has_bitmap: Bool
    var wpos: Int
    var null_count: Int
    var dict: List[Scalar[Self.native]]

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.dtype = leaf.dtype.copy()
        self.max_def = leaf.max_def
        self.num_rows = num_rows
        self.width = leaf.type_length
        self.values = Buffer.alloc_uninit[Self.native](num_rows)
        self.bitmap = Bitmap[mut=True].alloc_zeroed(0)
        self.has_bitmap = False
        self.wpos = 0
        self.null_count = 0
        self.dict = List[Scalar[Self.native]]()

    def _decode_be(self, span: Span[UInt8, _], off: Int) -> Scalar[Self.native]:
        """Big-endian, sign-extended from `width` bytes to the native width."""
        var arr = InlineArray[UInt8, Self.FULL](fill=0)
        if (span[off] & 0x80) != 0:  # negative -> sign-extend with 0xFF
            for i in range(Self.FULL):
                arr[i] = 0xFF
        for i in range(self.width):
            arr[Self.FULL - self.width + i] = span[off + i]
        return SIMD[Self.native, 1].from_bytes[big_endian=True](arr)

    def _ensure_bitmap(mut self):
        if not self.has_bitmap:
            self.bitmap = Bitmap[mut=True].alloc_zeroed(self.num_rows)
            self.bitmap.set_range(0, self.wpos, True)
            self.has_bitmap = True

    def _append_present(mut self, v: Scalar[Self.native]):
        self.values.unsafe_set[Self.native](self.wpos, v)
        if self.has_bitmap:
            self.bitmap.set(self.wpos)
        self.wpos += 1

    def _append_null(mut self):
        self._ensure_bitmap()
        self.values.unsafe_set[Self.native](self.wpos, Scalar[Self.native](0))
        self.null_count += 1
        self.wpos += 1

    def _place(mut self, page: Page, mask: Optional[List[Bool]]) raises:
        var vspan = page.values()
        var idx = List[Int32]()
        var is_dict = page.is_dictionary()
        if is_dict:
            idx = Rle.decode(vspan[1:], Int(vspan[0]), page.num_present)
        elif not page.is_plain():
            raise Error("parquet: unsupported FIXED_LEN_BYTE_ARRAY encoding")

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    if is_dict:
                        self._append_present(self.dict[Int(idx[vi])])
                    else:
                        self._append_present(
                            self._decode_be(vspan, vi * self.width)
                        )
                else:
                    self._append_null()

        _walk_slots[place](page, self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            var span = page.body
            for i in range(page.num_values):
                self.dict.append(self._decode_be(span, i * self.width))
        else:
            self._place(page, None)

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        self._place(page, mask.copy())

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


struct FixedSizeBinaryLeafBuilder(LeafBuilder):
    """FIXED_LEN_BYTE_ARRAY raw bytes -> FixedSizeBinaryArray. PLAIN and
    RLE_DICTIONARY, the encodings PyArrow emits."""

    var builder: FixedSizeBinaryBuilder
    var max_def: Int
    var width: Int
    var dict_body: List[UInt8]  # concatenated dict values, `width` bytes each

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.width = leaf.type_length
        self.builder = FixedSizeBinaryBuilder(self.width, num_rows)
        self.max_def = leaf.max_def
        self.dict_body = List[UInt8]()

    def _place(mut self, page: Page, mask: Optional[List[Bool]]) raises:
        var vspan = page.values()
        var idx = List[Int32]()
        var is_dict = page.is_dictionary()
        if is_dict:
            idx = Rle.decode(vspan[1:], Int(vspan[0]), page.num_present)
        elif not page.is_plain():
            raise Error("parquet: unsupported FIXED_LEN_BYTE_ARRAY encoding")

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    if is_dict:
                        var o = Int(idx[vi]) * self.width
                        self.builder.append(
                            Span(self.dict_body)[o : o + self.width]
                        )
                    else:
                        var o = vi * self.width
                        self.builder.append(vspan[o : o + self.width])
                else:
                    self.builder.append_null()

        _walk_slots[place](page, self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            self.dict_body = List[UInt8]()
            self.dict_body.extend(page.body)
        else:
            self._place(page, None)

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        self._place(page, mask.copy())

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

    def _place(
        mut self,
        page: Page,
        vspan: Span[UInt8, _],
        mask: Optional[List[Bool]] = None,
    ) raises:
        """Unpack the bit-packed present booleans honoring def levels; with
        `mask`, only selected rows are appended. `vi` is the present-value index,
        which for a bit-packed PLAIN page is exactly the bit offset."""

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if present_here:
                var byte = vspan[vi >> 3]
                var b = (byte >> UInt8(vi & 7)) & 1
                if selected:
                    self.builder.append(b == 1)
            elif selected:
                self.builder.append_null()

        _walk_slots[place](page, self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            raise Error("parquet: dictionary-encoded bool not supported")
        if not page.is_plain():
            raise Error("parquet: non-plain bool encoding not supported")
        self._place(page, page.values())

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        if not page.is_plain():
            raise Error("parquet: non-plain bool encoding not supported")
        self._place(page, page.values(), mask.copy())

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
            return self._dispatch[True](codecs)
        else:
            return self._dispatch[False](codecs)

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

    def _flat_leaf(mut self, var arr: AnyArray) raises -> DecodedLeaf:
        """Wrap a flat-path array with the carry-def levels accumulated during
        the build (empty unless the leaf is under a nullable struct)."""
        var defs = self.def_out^
        self.def_out = List[Int32]()
        return DecodedLeaf(arr^, List[Int32](), defs^)

    # -----------------------------------------------------------------------
    # Leveled path — grow the element array while accumulating rep/def levels.
    # A slot holds an element when its def reaches `floor`; the element is
    # present at `max_def`, else null. Present values come from the shared
    # `Encoding` decoders, so every encoding works here just as on the flat path.
    # -----------------------------------------------------------------------

    def _drive_leveled[
        handle_dict: def(Page) raises capturing[_] -> None,
        decode_present: def(Page) raises capturing[_] -> None,
        place_present: def(Int) raises capturing[_] -> None,
        place_null: def() raises capturing[_] -> None,
    ](
        mut self,
        mut codecs: CompressionLibs,
        floor: Int,
        max_def: Int,
        mut rep_out: List[Int32],
        mut def_out: List[Int32],
    ) raises:
        """The shared page-loop for every leveled (list-element) leaf. For each
        page it folds a dictionary page into the builder's dict (`handle_dict`)
        and skips it, else accumulates the page's rep/def levels, decodes its
        present values (`decode_present`), and walks `num_values` slots placing a
        present value (`place_present(vi)`) or a null (`place_null`) per
        definition level — a slot below `floor` holds no element, a slot at
        `max_def` is present, else null. Only the value type and how it is
        appended differ across leaves; those live entirely in the closures."""
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if pg.dictionary:
                handle_dict(pg)
                continue
            rep_out.extend(Span(pg.rep_levels))
            def_out.extend(Span(pg.def_levels))
            decode_present(pg)
            var vi = 0
            for k in range(pg.num_values):
                var d = Int(pg.def_levels[k])
                if d < floor:
                    continue
                if d == max_def:
                    place_present(vi)
                    vi += 1
                else:
                    place_null()

    def _drive_primitive[
        T: dt.NumericType, phys: DType
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        var builder = PrimitiveBuilder[T](self.num_rows)
        var dict = List[Scalar[T.native]]()
        var present = List[Scalar[T.native]]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()

        @parameter
        def handle_dict(pg: Page) raises:
            Dictionary.decode_page_primitive[T.native, phys](
                pg.body, pg.num_values, dict
            )

        @parameter
        def decode_present(pg: Page) raises:
            present.clear()
            pg.encoding.decode_primitive[T.native, phys](
                pg.values(), pg.num_present, dict, present
            )

        @parameter
        def place_present(vi: Int) raises:
            builder.append(present[vi])

        @parameter
        def place_null() raises:
            builder.append_null()

        self._drive_leveled[
            handle_dict, decode_present, place_present, place_null
        ](codecs, floor, max_def, rep_out, def_out)
        var arr = builder.finish()
        return DecodedLeaf(arr^, rep_out^, def_out^)

    def _drive_bytes[
        BT: dt.BinaryLikeType
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        var builder = BinaryLikeBuilder[BT](self.num_rows)
        var dict_body = List[UInt8]()
        var dict_off = List[Int]()
        var dict_len = List[Int]()
        var values = List[List[UInt8]]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()

        @parameter
        def handle_dict(pg: Page) raises:
            Dictionary.decode_page_bytes(
                pg.body, pg.num_values, dict_body, dict_off, dict_len
            )

        @parameter
        def decode_present(pg: Page) raises:
            values.clear()
            values.extend(
                pg.encoding.decode_bytes(
                    pg.values(), pg.num_present, dict_body, dict_off, dict_len
                )
            )

        @parameter
        def place_present(vi: Int) raises:
            builder.append(StringSlice(unsafe_from_utf8=Span(values[vi])))

        @parameter
        def place_null() raises:
            builder.append_null()

        self._drive_leveled[
            handle_dict, decode_present, place_present, place_null
        ](codecs, floor, max_def, rep_out, def_out)
        var arr = builder.finish()
        return DecodedLeaf(arr^, rep_out^, def_out^)

    def _drive_bool(
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        var builder = BoolBuilder(self.num_rows)
        var present = List[Bool]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()

        @parameter
        def handle_dict(pg: Page) raises:
            raise Error("parquet: dictionary-encoded bool not supported")

        @parameter
        def decode_present(pg: Page) raises:
            present.clear()
            present.extend(pg.encoding.decode_bool(pg.values(), pg.num_present))

        @parameter
        def place_present(vi: Int) raises:
            builder.append(present[vi])

        @parameter
        def place_null() raises:
            builder.append_null()

        self._drive_leveled[
            handle_dict, decode_present, place_present, place_null
        ](codecs, floor, max_def, rep_out, def_out)
        var arr = builder.finish()
        return DecodedLeaf(arr^, rep_out^, def_out^)

    # -----------------------------------------------------------------------
    # Unified dispatch — one arrow-dtype -> (store, phys) decision table shared by
    # the flat and leveled paths. `leveled` is a compile-time flag: the emit
    # helpers pick the flat build (fixed-size `LeafBuilder`) or the leveled drive
    # (`_drive_*`), and each specialization DCEs the other branch, so the closed
    # per-dtype dispatch is preserved. Temporal types are flat-only (they map to
    # the same int32/int64 storage), guarded behind `comptime if not leveled`.
    # -----------------------------------------------------------------------

    def _emit_numeric[
        T: dt.NumericType, phys: DType, leveled: Bool
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        comptime if leveled:
            return self._drive_primitive[T, phys](codecs, floor, max_def)
        else:
            var arr = self._build(
                PrimitiveLeafBuilder[T.native, phys](
                    self.num_rows, self.pages.leaf
                ),
                codecs,
            )
            return self._flat_leaf(arr^)

    def _emit_bytes[
        BT: dt.BinaryLikeType, leveled: Bool
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        comptime if leveled:
            return self._drive_bytes[BT](codecs, floor, max_def)
        else:
            var arr = self._build(
                ByteArrayLeafBuilder[BT](self.num_rows, self.pages.leaf), codecs
            )
            return self._flat_leaf(arr^)

    def _emit_bool[
        leveled: Bool
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        comptime if leveled:
            return self._drive_bool(codecs, floor, max_def)
        else:
            var arr = self._build(
                BoolLeafBuilder(self.num_rows, self.pages.leaf), codecs
            )
            return self._flat_leaf(arr^)

    def _emit_decimal[
        native: DType
    ](mut self, mut codecs: CompressionLibs) raises -> DecodedLeaf:
        var arr = self._build(
            DecimalLeafBuilder[native](self.num_rows, self.pages.leaf), codecs
        )
        return self._flat_leaf(arr^)

    def _emit_fsb(
        mut self, mut codecs: CompressionLibs
    ) raises -> DecodedLeaf:
        var arr = self._build(
            FixedSizeBinaryLeafBuilder(self.num_rows, self.pages.leaf), codecs
        )
        return self._flat_leaf(arr^)

    def _dispatch[
        leveled: Bool
    ](mut self, mut codecs: CompressionLibs) raises -> DecodedLeaf:
        ref leaf = self.pages.leaf
        ref vt = leaf.dtype
        # leveled: a value slot exists at/above the leaf's repetition floor (the
        # innermost enclosing list's element level), present at max_def. flat
        # ignores these (the emit helper's flat branch drops them).
        var f = leaf.slot_def
        var md = leaf.max_def
        if vt == dt.int32:
            return self._emit_numeric[dt.Int32Type, DType.int32, leveled](
                codecs, f, md
            )
        elif vt == dt.int64:
            return self._emit_numeric[dt.Int64Type, DType.int64, leveled](
                codecs, f, md
            )
        elif vt == dt.uint32:
            return self._emit_numeric[dt.UInt32Type, DType.uint32, leveled](
                codecs, f, md
            )
        elif vt == dt.uint64:
            return self._emit_numeric[dt.UInt64Type, DType.uint64, leveled](
                codecs, f, md
            )
        elif vt == dt.float32:
            return self._emit_numeric[dt.Float32Type, DType.float32, leveled](
                codecs, f, md
            )
        elif vt == dt.float64:
            return self._emit_numeric[dt.Float64Type, DType.float64, leveled](
                codecs, f, md
            )
        elif vt == dt.int8:
            return self._emit_numeric[dt.Int8Type, DType.int32, leveled](
                codecs, f, md
            )
        elif vt == dt.int16:
            return self._emit_numeric[dt.Int16Type, DType.int32, leveled](
                codecs, f, md
            )
        elif vt == dt.uint8:
            return self._emit_numeric[dt.UInt8Type, DType.int32, leveled](
                codecs, f, md
            )
        elif vt == dt.uint16:
            return self._emit_numeric[dt.UInt16Type, DType.int32, leveled](
                codecs, f, md
            )
        elif vt == dt.bool_:
            return self._emit_bool[leveled](codecs, f, md)
        elif vt.is_string():
            return self._emit_bytes[dt.StringType, leveled](codecs, f, md)
        elif vt.is_large_string():
            return self._emit_bytes[dt.LargeStringType, leveled](codecs, f, md)
        elif vt.is_binary():
            return self._emit_bytes[dt.BinaryType, leveled](codecs, f, md)
        elif vt.is_large_binary():
            return self._emit_bytes[dt.LargeBinaryType, leveled](codecs, f, md)

        comptime if not leveled:
            # temporal types are physically int32/int64 and appear only on the
            # flat path (page skipping is a flat-path optimisation).
            if vt.is_date32() or vt.is_time32():
                return self._emit_numeric[dt.Int32Type, DType.int32, leveled](
                    codecs, f, md
                )
            elif (
                vt.is_timestamp()
                or vt.is_time64()
                or vt.is_date64()
                or vt.is_duration()
            ):
                return self._emit_numeric[dt.Int64Type, DType.int64, leveled](
                    codecs, f, md
                )
            elif vt.is_decimal128():
                return self._emit_decimal[DType.int128](codecs)
            elif vt.is_decimal256():
                return self._emit_decimal[DType.int256](codecs)
            elif vt.is_fixed_size_binary():
                return self._emit_fsb(codecs)
            else:
                raise Error("parquet: unsupported column type " + String(vt))
        else:
            raise Error("parquet: unsupported list element type " + String(vt))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Below this many rows a file is decoded on one thread — the thread-dispatch
# overhead is not worth it for small files.
comptime _PARALLEL_MIN_ROWS = 4096


def read_table(
    path: String,
    columns: Optional[List[String]] = None,
    row_groups: Optional[List[Int]] = None,
    row_selections: Optional[List[RowSelection]] = None,
) raises -> Table:
    """Read a Parquet file into a Marrow `Table`.

    `columns` projects the read to the named top-level columns (in the given
    order); `None` reads all. Only the selected columns' chunks are touched.

    `row_groups` restricts the read to the given row-group indices (in that
    order); `None` reads all. Unselected row groups are never decoded — this is
    what predicate pushdown uses to skip groups that cannot match a filter.

    Every (row group, selected leaf) pair decodes independently — each reads a
    disjoint byte range of the shared read-only mmap and writes its own result
    slot — so the whole grid is decoded across `num_physical_cores()` workers.
    Each worker owns a `CompressionLibs` (the lazy `dlopen` handles and reused size cell
    are not shareable across threads); the mmap and metadata are read-only.
    """
    var mapped = MappedFile(path)
    var data = mapped.span()
    var meta = FileMetaData.read_footer(data)
    var mapping = SchemaMapping.from_parquet(meta)

    # the read plan: which column chunks to decode and how to reassemble them
    var plan = mapping.project(columns.value()) if columns else mapping.full()

    # which row groups to decode (all, or the given subset in order)
    var rg_list = List[Int]()
    if row_groups:
        for rg in row_groups.value():
            if rg < 0 or rg >= len(meta.row_groups):
                raise Error("parquet: row group index out of range")
            rg_list.append(rg)
    else:
        for rg in range(len(meta.row_groups)):
            rg_list.append(rg)

    if row_selections and len(row_selections.value()) != len(rg_list):
        raise Error(
            "parquet: row_selections must match the selected row groups"
        )

    var num_leaves = len(plan.decode_order)
    var num_rg = len(rg_list)
    var total = num_rg * num_leaves

    # rows across the selected groups — governs the parallel-dispatch threshold
    var selected_rows = 0
    for rg in rg_list:
        selected_rows += meta.row_groups[rg].num_rows

    var nt = 1
    if total >= 2 and selected_rows >= _PARALLEL_MIN_ROWS:
        nt = min(num_physical_cores(), total)

    # One result slot per (row group, selected leaf), pre-sized so workers assign
    # by index without racing on list growth. (DecodedLeaf is move-only, so the
    # slots are filled by appending rather than the copy-based `fill=`.)
    var grid = List[Optional[DecodedLeaf]](capacity=total)
    for _ in range(total):
        grid.append(None)

    @parameter
    def worker(w: Int) raises:
        var codecs = (
            CompressionLibs()
        )  # per-worker: lazy dlopen handles are not shared
        var t = w
        while t < total:
            var slot = t // num_leaves
            var rg_idx = rg_list[slot]
            # original column-chunk index for this compact slot
            var orig = plan.decode_order[t % num_leaves]
            ref rg = meta.row_groups[rg_idx]
            # the row selection for this group (shared by all its leaf columns),
            # None when nothing is pushed down or the group is fully selected.
            var sel: Optional[RowSelection] = None
            if row_selections:
                sel = row_selections.value()[slot].copy()
            # ColumnReader.decode picks the flat vs leveled path from the leaf's
            # max repetition, so the same call serves every column shape.
            var reader = ColumnReader(
                data,
                rg.columns[orig].meta_data.copy(),
                mapping.leaves[orig].copy(),
                rg.num_rows,
                sel^,
            )
            grid[t] = reader.decode(codecs)
            t += nt

    sync_parallelize[worker](nt)

    # Fold each selected row group's decoded leaves back into the Arrow tree.
    var batches = List[RecordBatch]()
    for i in range(num_rg):
        var decoded = List[DecodedLeaf]()
        for ci in range(num_leaves):
            decoded.append(grid[i * num_leaves + ci].take())
        var cols = List[AnyArray]()
        for ref node in plan.nodes:
            cols.append(node.assemble(decoded))
        batches.append(
            RecordBatch(schema=Schema(copy=plan.schema), columns=cols^)
        )

    if len(batches) == 0:
        batches.append(RecordBatch.empty(Schema(copy=plan.schema)))
    var result = Table.from_batches(Schema(copy=plan.schema), batches)
    # `data` is an untracked view into `mapped`; keep the map alive until every
    # value has been copied into owned Arrow buffers above, then unmap.
    _ = mapped^
    return result^


# ---------------------------------------------------------------------------
# Metadata / statistics — read the footer without decoding any column data.
# Mirrors PyArrow's `read_metadata` / `ColumnChunkMetaData.statistics`.
# ---------------------------------------------------------------------------


struct RowSelection(Copyable, Movable):
    """Which rows of a row group to decode when a pushed-down predicate lets the
    reader skip pages. Row-group-relative, one flag per row: a data page whose
    rows are all deselected is skipped without decoding; a partially selected
    page keeps only its chosen rows, so every column yields the same rows and
    stays aligned. Built from per-page keep flags, combined with `intersect`, and
    queried by the decoder over each page's row range. (A run-length form is a
    possible future optimisation; it would not change this interface.)"""

    var _selected: List[Bool]

    def __init__(out self, var selected: List[Bool]):
        self._selected = selected^

    @staticmethod
    def all(n: Int) -> Self:
        """Select every one of `n` rows."""
        var s = List[Bool](capacity=n)
        for _ in range(n):
            s.append(True)
        return Self(s^)

    @staticmethod
    def from_pages(keep: List[Bool], page_rows: List[Int]) -> Self:
        """Expand per-page keep flags into per-row flags. `keep[i]` decides all
        `page_rows[i]` rows of page `i` (pages begin on row boundaries)."""
        var s = List[Bool]()
        for p in range(len(keep)):
            for _ in range(page_rows[p]):
                s.append(keep[p])
        return Self(s^)

    def total_rows(self) -> Int:
        return len(self._selected)

    def selected(self, row: Int) -> Bool:
        return self._selected[row]

    def num_selected(self) -> Int:
        var n = 0
        for i in range(len(self._selected)):
            if self._selected[i]:
                n += 1
        return n

    def selects_any(self) -> Bool:
        for i in range(len(self._selected)):
            if self._selected[i]:
                return True
        return False

    def selects_all(self) -> Bool:
        for i in range(len(self._selected)):
            if not self._selected[i]:
                return False
        return True

    def intersect(self, other: Self) raises -> Self:
        """AND two selections over the same row group (a row survives only if
        both keep it)."""
        if self.total_rows() != other.total_rows():
            raise Error("parquet: RowSelection size mismatch")
        var s = List[Bool](capacity=self.total_rows())
        for i in range(self.total_rows()):
            s.append(self._selected[i] and other._selected[i])
        return Self(s^)

    def selected_in(self, start: Int, length: Int) -> Int:
        """How many rows in the half-open range `[start, start+length)` are
        selected — lets the decoder decide skip / keep-all / mask for a page."""
        var n = 0
        for i in range(start, start + length):
            if self._selected[i]:
                n += 1
        return n

    def mask(self, start: Int, length: Int) -> List[Bool]:
        """The per-row keep flags for the page rows `[start, start+length)`."""
        var m = List[Bool](capacity=length)
        for i in range(start, start + length):
            m.append(self._selected[i])
        return m^


def read_metadata(path: String) raises -> FileMetaData:
    """Read only the file footer: schema, row groups, and per-column-chunk
    metadata (offsets, sizes, codec, null_count, and the raw min/max statistic
    bytes). No column data is decoded. Mirrors `pyarrow.parquet.read_metadata`.
    """
    var mapped = MappedFile(path)
    var meta = FileMetaData.read_footer(mapped.span())
    _ = mapped^  # read_footer copies every field into owned storage
    return meta^


struct PageIndex(Copyable, Movable):
    """One column chunk's page index: `offset_index` locates each data page,
    `column_index` holds its per-page min/max/null stats. Either may be absent
    (a file without a page index)."""

    var offset_index: Optional[OffsetIndex]
    var column_index: Optional[ColumnIndex]

    def __init__(out self):
        self.offset_index = None
        self.column_index = None


def read_page_index(path: String) raises -> List[List[PageIndex]]:
    """Read the page index (OffsetIndex + ColumnIndex) for every (row group,
    leaf column), indexed `result[row_group][leaf]`. Follows the offsets stored
    in each `ColumnChunk`; a chunk without a page index yields empty optionals.
    """
    var mapped = MappedFile(path)
    var data = mapped.span()
    var meta = FileMetaData.read_footer(data)
    var out = List[List[PageIndex]]()
    for ref rg in meta.row_groups:
        var row = List[PageIndex]()
        for ci in range(len(rg.columns)):
            ref cc = rg.columns[ci]
            var pi = PageIndex()
            if cc.offset_index_offset >= 0:
                var r = CompactReader(data, cc.offset_index_offset)
                pi.offset_index = OffsetIndex.read(r)
            if cc.column_index_offset >= 0:
                var r = CompactReader(data, cc.column_index_offset)
                pi.column_index = ColumnIndex.read(r)
            row.append(pi^)
        out.append(row^)
    _ = mapped^  # OffsetIndex/ColumnIndex copy their bytes into owned storage
    return out^


struct PageBounds(Copyable, Movable):
    """One data page's decoded bounds: its row count and the typed `min`/`max`
    (each `None` when the page is all-null or carried no bound)."""

    var num_rows: Int
    var min: Optional[AnyScalar]
    var max: Optional[AnyScalar]

    def __init__(
        out self,
        num_rows: Int,
        var min: Optional[AnyScalar],
        var max: Optional[AnyScalar],
    ):
        self.num_rows = num_rows
        self.min = min^
        self.max = max^


def read_page_bounds(path: String) raises -> List[List[List[PageBounds]]]:
    """Per (row group, leaf column, data page) decoded bounds, from the page
    index — indexed `result[rg][leaf][page]`. A column with no page index yields
    an empty page list. Predicate pushdown prunes individual pages with these.
    """
    var mapped = MappedFile(path)
    var data = mapped.span()
    var meta = FileMetaData.read_footer(data)
    var mapping = SchemaMapping.from_parquet(meta)
    var out = List[List[List[PageBounds]]]()
    for ref rg in meta.row_groups:
        var per_col = List[List[PageBounds]]()
        for ci in range(len(rg.columns)):
            ref cc = rg.columns[ci]
            var pages = List[PageBounds]()
            if cc.offset_index_offset >= 0 and cc.column_index_offset >= 0:
                var ro = CompactReader(data, cc.offset_index_offset)
                var oi = OffsetIndex.read(ro)
                var rc = CompactReader(data, cc.column_index_offset)
                var cix = ColumnIndex.read(rc)
                ref dtype = mapping.leaves[ci].dtype
                var np = len(oi.page_locations)
                for p in range(np):
                    var first = oi.page_locations[p].first_row_index
                    var nxt = (
                        oi.page_locations[p + 1].first_row_index if p + 1
                        < np else rg.num_rows
                    )
                    var mn: Optional[AnyScalar] = None
                    var mx: Optional[AnyScalar] = None
                    # an all-null page (or a missing bound) prunes nothing
                    if not (p < len(cix.null_pages) and cix.null_pages[p]):
                        if p < len(cix.min_values):
                            mn = _decode_stat(dtype, cix.min_values[p])
                        if p < len(cix.max_values):
                            mx = _decode_stat(dtype, cix.max_values[p])
                    pages.append(PageBounds(nxt - first, mn^, mx^))
            per_col.append(pages^)
        out.append(per_col^)
    _ = mapped^
    return out^


struct ColumnStatistics(Copyable, Movable):
    """Decoded per-column-chunk statistics: `null_count` (-1 if absent) and the
    `min`/`max` bounds as typed scalars (each `None` when the file stored no
    usable bound, so absence is in the type rather than a separate flag)."""

    var null_count: Int
    var min: Optional[AnyScalar]
    var max: Optional[AnyScalar]

    def __init__(out self):
        self.null_count = -1
        self.min = None
        self.max = None


def _decode_stat(
    dtype: dt.AnyDataType, b: List[UInt8]
) raises -> Optional[AnyScalar]:
    """Decode one PLAIN-encoded min/max value to a typed scalar, mirroring the
    writer's encoding (`LittleEndian.fixed` reads `size_of[dt]` LE bytes and
    reinterprets — the inverse of the writer's byte emission). Returns None for
    types this reader does not yet decode (raw bytes stay in `read_metadata`).
    """
    var s = Span(b)
    if dtype == dt.int8:
        return AnyScalar(
            Int8Scalar(LittleEndian.fixed[DType.int32](s, 0).cast[DType.int8]())
        )
    elif dtype == dt.int16:
        return AnyScalar(
            Int16Scalar(
                LittleEndian.fixed[DType.int32](s, 0).cast[DType.int16]()
            )
        )
    elif dtype == dt.int32:
        return AnyScalar(Int32Scalar(LittleEndian.fixed[DType.int32](s, 0)))
    elif dtype == dt.uint8:
        return AnyScalar(
            UInt8Scalar(
                LittleEndian.fixed[DType.uint32](s, 0).cast[DType.uint8]()
            )
        )
    elif dtype == dt.uint16:
        return AnyScalar(
            UInt16Scalar(
                LittleEndian.fixed[DType.uint32](s, 0).cast[DType.uint16]()
            )
        )
    elif dtype == dt.uint32:
        return AnyScalar(UInt32Scalar(LittleEndian.fixed[DType.uint32](s, 0)))
    elif dtype == dt.int64:
        return AnyScalar(Int64Scalar(LittleEndian.fixed[DType.int64](s, 0)))
    elif dtype == dt.uint64:
        return AnyScalar(UInt64Scalar(LittleEndian.fixed[DType.uint64](s, 0)))
    elif dtype == dt.float32:
        return AnyScalar(Float32Scalar(LittleEndian.fixed[DType.float32](s, 0)))
    elif dtype == dt.float64:
        return AnyScalar(Float64Scalar(LittleEndian.fixed[DType.float64](s, 0)))
    elif dtype == dt.bool_:
        return AnyScalar(BoolScalar(len(b) > 0 and b[0] != 0))
    elif dtype.is_string():
        return AnyScalar(
            StringScalar(String(StringSlice(unsafe_from_utf8=Span(b))))
        )
    else:
        return None


def read_statistics(path: String) raises -> List[List[ColumnStatistics]]:
    """Per-(row group, leaf column) decoded statistics, indexed
    `result[row_group][leaf]`. `min`/`max` are typed scalars matching each
    column's Arrow type; `has_min_max` is false when the file stored no bounds
    or the type's bounds are not yet decoded."""
    var meta = read_metadata(path)
    var mapping = SchemaMapping.from_parquet(meta)
    var out = List[List[ColumnStatistics]]()
    for ref rg in meta.row_groups:
        var row = List[ColumnStatistics]()
        for ci in range(len(rg.columns)):
            ref cm = rg.columns[ci].meta_data
            var cs = ColumnStatistics()
            cs.null_count = cm.null_count
            if cm.has_min_max:
                var mn = _decode_stat(mapping.leaves[ci].dtype, cm.min_value)
                var mx = _decode_stat(mapping.leaves[ci].dtype, cm.max_value)
                if Bool(mn) and Bool(mx):
                    cs.min = mn^
                    cs.max = mx^
            row.append(cs^)
        out.append(row^)
    return out^
