"""Native Parquet deserialization → Arrow: the full read path.

`read_table` memory-maps the file, reads the footer metadata, and decodes each
selected column chunk's pages into an Arrow leaf array via a `ColumnReader`,
folding the results back into the Arrow type tree with `SchemaNode.assemble`.
Predicate pushdown skips whole row groups (`row_groups`) and individual pages
(`row_selections`). The column-decode engine — `Page`, `PageReader`,
`LeafBuilder` and its concrete builders, and `ColumnReader` — lives here too, so
this module is the entire deserialization layer; the metadata / statistics /
page-index readers reuse the same footer decode without touching column data.
"""

from max.algorithm.functional import sync_parallelize
from std.memory import ArcPointer
from std.builtin.rebind import downcast
from std.sys import size_of
from std.sys.info import num_physical_cores
from std.memory import unsafe_memcpy

from ..arrays import DynArray, ArrayData
from ..buffers import Buffer, Bitmap
from ..builders import (
    BinaryLikeBuilder,
    BoolBuilder,
    FixedSizeBinaryBuilder,
    PrimitiveBuilder,
)
from ..schema import Schema
from ..tabular import Table, RecordBatch
from ..scalars import DynScalar
from .. import dtypes as dt
from ..dtypes import (
    BinaryLikeType,
    DataType,
    DecimalType,
    NumericType,
    PrimitiveType,
    StringLikeType,
    TemporalType,
)

from .utils import CompressionLibs
from .codecs import Encoding, Rle, Plain, Dictionary, Compression
from ..utils import LittleEndian, Crc32
from .bloom import SplitBlockBloomFilter, BloomFilterHeader
from .source import ByteSource, MappedFile
from .schema import SchemaMapping, DecodedLeaf, LeafColumn
from .statistics import Statistics
from .format import (
    FileMetaData,
    ThriftCompactReader,
    ColumnChunk,
    ColumnIndex,
    OffsetIndex,
    ColumnMetaData,
    PageHeader,
    PageType,
    PhysicalType,
)


# ---------------------------------------------------------------------------
# Page — one decoded page
# ---------------------------------------------------------------------------


struct Page[o: Origin[mut=False]](Movable):
    """A decoded page. `body` is a *view* — into the mmap for uncompressed pages
    (zero copy) or into the `PageReader`'s reused scratch for compressed pages —
    not an owned buffer, so no per-page allocation. `dictionary` marks the
    column chunk's dictionary page (vs a data page)."""

    var dictionary: Bool
    var body: Span[UInt8, Self.o]
    var def_levels: List[Int32]
    var rep_levels: List[Int32]  # only populated for repeated (list) leaves
    var value_offset: Int
    var num_present: Int
    var encoding: Encoding
    var num_values: Int

    def __init__(
        out self,
        *,
        body: Span[UInt8, Self.o],
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
    def dictionary_page[
        po: Origin[mut=False]
    ](body: Span[UInt8, po], num_values: Int) -> Page[po]:
        """The column chunk's dictionary page — carries only its distinct values;
        no levels, no present/null distinction."""
        return Page(
            body=body,
            num_values=num_values,
            num_present=num_values,
            dictionary=True,
        )

    def all_present(self) -> Bool:
        return self.num_present == self.num_values

    def scatter[
        body: def(Bool, Bool, Int) raises capturing[_] -> None,
    ](self, max_def: Int, mask: Optional[List[Bool]]) raises:
        """Walk this page's `num_values` output slots — the flat skeleton every
        `LeafBuilder` shares.

        For each row `body(present, selected, vi)` fires with whether the slot
        holds a present value (`present_at`), whether the selection mask keeps
        it, and the running index into the page's decoded present values
        (advanced on every present slot, regardless of the mask). Only the
        per-slot placement differs across leaves, and that lives in `body`.

        A free `_walk_slots(page, ...)` before this; walking a page's slots is
        the page's own business, and it already needed nothing but `self`.
        """
        var vi = 0
        for row in range(self.num_values):
            var present_here = self.present_at(row, max_def)
            var selected = not mask or mask.value()[row]
            body(present_here, selected, vi)
            if present_here:
                vi += 1

    def present_at(self, row: Int, max_def: Int) -> Bool:
        """Whether output `row` holds a present value: the whole page is present,
        or its definition level reaches the leaf's max (a value slot, not a
        null). The def-level scatter every `LeafBuilder` shares pivots on this.
        """
        return self.all_present() or Int(self.def_levels[row]) == max_def

    def is_plain(self) -> Bool:
        return self.encoding.is_plain()

    def is_dictionary(self) -> Bool:
        return self.encoding.is_dictionary()

    def values(self) -> Span[UInt8, Self.o]:
        """The value bytes (after any level prefix)."""
        return self.body[self.value_offset :]


# ---------------------------------------------------------------------------
# PageReader — iterate the pages of one column chunk: locate each page, read its
# Thrift header, decompress the body, and decode the definition/repetition
# levels, so the LeafBuilders below only ever see already-decoded pages.
# ---------------------------------------------------------------------------


struct PageReader[o: Origin[mut=False]](Movable):
    """Iterates the pages of **one column chunk**.

    `data` is the chunk's own bytes — exactly the range `meta.byte_range()`
    names — and `pos` is relative to it, so the reader never addresses the file
    as a whole. That is what lets a `ByteSource` fetch a chunk at a time instead
    of having to hand out a whole-file span."""

    var data: Span[UInt8, Self.o]
    var meta: ColumnMetaData
    var leaf: LeafColumn
    var pos: Int  # chunk-relative offset of the next page header
    var produced: Int  # data-page values yielded so far
    var scratch: List[UInt8]  # reused decompression buffer for compressed pages

    def __init__(
        out self,
        data: Span[UInt8, Self.o],
        var meta: ColumnMetaData,
        var leaf: LeafColumn,
    ):
        self.data = data
        # The chunk starts at its first page, so the first header is at 0.
        self.pos = 0
        self.meta = meta^
        self.leaf = leaf^
        self.produced = 0
        self.scratch = List[UInt8]()

    def _body(
        mut self,
        comp: Span[UInt8, Self.o],
        uncompressed_size: Int,
        mut codecs: CompressionLibs,
    ) raises -> Span[UInt8, origin_of(Self.o, origin_of(self.scratch))]:
        """A view of the page body: the source bytes directly for uncompressed
        pages, or the reused scratch after decompressing.

        The two arms have genuinely different origins — the byte source's, and
        this reader's own scratch — so the result is their *union*. That union
        is why this used to launder both through an `ImmUntrackedOrigin`: the
        return needed one type, and dropping the tracking was the only way to
        get one. It is not: `origin_of(a, b)` expresses exactly this, and now
        the compiler knows a page body borrows from whichever of the two it
        actually came from."""
        var codec = Compression(self.meta.codec)
        if codec == Compression.UNCOMPRESSED:
            return rebind[
                Span[UInt8, origin_of(Self.o, origin_of(self.scratch))]
            ](comp)
        codec.decompress_into(codecs, comp, uncompressed_size, self.scratch)
        return rebind[Span[UInt8, origin_of(Self.o, origin_of(self.scratch))]](
            Span(self.scratch)
        )

    def has_next(self) -> Bool:
        return self.produced < self.meta.num_values

    def _data_page_v1[
        bo: Origin[mut=False]
    ](
        self,
        var body: Span[UInt8, bo],
        num_values: Int,
        encoding: Encoding,
    ) raises -> Page[bo]:
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

    def next(
        mut self, mut codecs: CompressionLibs
    ) raises -> Page[origin_of(Self.o, origin_of(self.scratch))]:
        var body_start = self.pos
        # advances body_start to the page body
        var ph = PageHeader.read_at(self.data, body_start)
        var comp = self.data[body_start : body_start + ph.compressed_page_size]
        self.pos = body_start + ph.compressed_page_size

        # Verify the optional page checksum: for both v1 and v2 the CRC covers
        # exactly the on-disk body (v1's compressed blob; v2's uncompressed
        # levels + compressed values), which is `comp`.
        if ph.crc >= 0 and Int(Crc32.compute(comp)) != ph.crc:
            raise Error("parquet: page CRC-32 mismatch (corrupt data)")

        if ph.type == PageType.DICTIONARY:
            return Page[
                origin_of(Self.o, origin_of(self.scratch))
            ].dictionary_page(
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

            # 8 trailing bytes of slack so the bit-unpackers can do unaligned
            # 64-bit loads past the last value without overrunning the buffer
            # (decompress_into pads the same way; the manual v2 assembly above
            # otherwise ends flush against the last value byte).
            self.scratch.resize(unsafe_uninit_length=len(self.scratch) + 8)

            var body = rebind[
                Span[UInt8, origin_of(Self.o, origin_of(self.scratch))]
            ](Span(self.scratch))
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


def _finish_primitive(
    var values: Buffer[mut=True],
    var bitmap: Bitmap[mut=True],
    wpos: Int,
    null_count: Int,
    var dtype: dt.DynType,
) raises -> DynArray:
    """Freeze a fixed-width builder's storage into an `DynArray` — the shared
    tail of every primitive-backed leaf (numeric, temporal, decimal): one values
    buffer, a validity bitmap only when a null occurred, length `wpos`."""
    var buffers = List[Buffer[mut=False]]()
    buffers.append(values^.to_immutable())
    var bm: Optional[Bitmap[mut=False]] = None
    if null_count > 0:
        bm = bitmap^.to_immutable(length=wpos)
    return DynArray.from_data(
        ArrayData(
            dtype=dtype^,
            length=wpos,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffers=buffers^,
            children=List[ArrayData](),
        )
    )


comptime _JULIAN_DAY_OF_EPOCH = 2440588  # Julian day number of 1970-01-01
comptime _NANOS_PER_DAY = 86_400_000_000_000


def _int96_nanos(span: Span[UInt8, _], off: Int) -> Int64:
    """Decode a 12-byte INT96 timestamp to nanoseconds since the Unix epoch: the
    first 8 bytes are the little-endian nanoseconds within the day, the last 4
    the little-endian Julian day number. The shared flat/leveled INT96 decode.
    """
    var nanos_of_day = LittleEndian.fixed[DType.int64](span, off)
    var julian_day = LittleEndian.fixed[DType.int32](span, off + 8)
    return (
        nanos_of_day
        + Int64(Int(julian_day) - _JULIAN_DAY_OF_EPOCH) * _NANOS_PER_DAY
    )


trait LeafBuilder(ImplicitlyDeletable, Movable):
    """Accumulates the values of a column chunk, page by page, into an array."""

    def consume(mut self, var page: Page) raises:
        ...

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        """Append only the rows of `page` where `mask[row]` is set (page-relative
        row index) — used for a data page partially covered by a row selection.
        """
        ...

    def finish(deinit self) raises -> DynArray:
        ...


struct PrimitiveLeafBuilder[store_dt: DType, phys_dt: DType = store_dt](
    LeafBuilder
):
    """Fixed-width values decoded straight into a contiguous buffer.

    `phys_dt` is the Parquet physical width read from the file, `store_dt` the
    Arrow storage width. When they match (the common case, incl. temporal types)
    a whole all-present PLAIN page is one unsafe_memcpy; when they differ (dt.int8/16 stored
    as physical INT32) each value is read wide and narrowed.
    """

    comptime SAME = Self.store_dt == Self.phys_dt

    var dtype: dt.DynType
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
        buffer, honoring definition levels — one unsafe_memcpy when the page is
        all-present and fully selected, else a per-row scatter that materializes
        the validity bitmap. With `mask`, only the rows it selects are placed
        (the page-boundary partial-page path). Every encoding funnels its decoded
        present values through here."""
        var vptr = self.values.view[Self.store_dt]().unsafe_ptr()
        if not mask and page.all_present():
            unsafe_memcpy(
                dest=vptr + self.wpos, src=present, count=page.num_present
            )
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

        page.scatter[place](self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        comptime PW = size_of[Scalar[Self.phys_dt]]()
        if page.dictionary:
            comptime if Self.SAME:
                self.dict.resize(unsafe_uninit_length=page.num_values)
                unsafe_memcpy(
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

    def finish(deinit self) raises -> DynArray:
        return _finish_primitive(
            self.values^,
            self.bitmap^,
            self.wpos,
            self.null_count,
            self.dtype^,
        )


struct ByteArrayLeafBuilder[BT: BinaryLikeType](LeafBuilder):
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

        page.scatter[place](self.max_def, mask)

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

        page.scatter[place](self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            Dictionary.decode_page_bytes(
                page.body,
                page.num_values,
                self.dict_body,
                self.dict_off,
                self.dict_len,
            )
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

    def finish(deinit self) raises -> DynArray:
        var b = self.builder^
        var out: DynArray = b.finish()
        return out^


struct _FixedWidthAcc[native: DType](Movable):
    """Shared output bookkeeping for the per-value fixed-width leaf builders
    (decimal, INT96): a `native`-typed values buffer, a validity bitmap
    materialized lazily on the first null (backfilling prior values as valid), a
    write cursor, and the null count. The builders own the per-page decode and
    funnel each decoded value through `append_present` / `append_null`."""

    var num_rows: Int
    var values: Buffer[mut=True]
    var bitmap: Bitmap[mut=True]
    var has_bitmap: Bool
    var wpos: Int
    var null_count: Int

    def __init__(out self, num_rows: Int):
        self.num_rows = num_rows
        self.values = Buffer.alloc_uninit[Self.native](num_rows)
        self.bitmap = Bitmap[mut=True].alloc_zeroed(0)
        self.has_bitmap = False
        self.wpos = 0
        self.null_count = 0

    def ensure_bitmap(mut self):
        if not self.has_bitmap:
            self.bitmap = Bitmap[mut=True].alloc_zeroed(self.num_rows)
            self.bitmap.set_range(0, self.wpos, True)
            self.has_bitmap = True

    def append_present(mut self, v: Scalar[Self.native]):
        self.values.unsafe_set[Self.native](self.wpos, v)
        if self.has_bitmap:
            self.bitmap.set(self.wpos)
        self.wpos += 1

    def append_null(mut self):
        self.ensure_bitmap()
        self.values.unsafe_set[Self.native](self.wpos, Scalar[Self.native](0))
        self.null_count += 1
        self.wpos += 1

    def finish(deinit self, var dtype: dt.DynType) raises -> DynArray:
        return _finish_primitive(
            self.values^, self.bitmap^, self.wpos, self.null_count, dtype^
        )


struct DecimalLeafBuilder[native: DType](LeafBuilder):
    """FIXED_LEN_BYTE_ARRAY decimals (decimal128 -> int128, decimal256 -> int256).

    Each value is big-endian two's-complement of the schema's `type_length`
    bytes — which PyArrow minimises per precision, so it may be narrower than the
    16/32-byte storage — sign-extended into the native width. PLAIN and
    RLE_DICTIONARY only, the encodings PyArrow emits for decimals."""

    comptime FULL = size_of[Scalar[Self.native]]()

    var dtype: dt.DynType
    var max_def: Int
    var width: Int
    var acc: _FixedWidthAcc[Self.native]
    var dict: List[Scalar[Self.native]]

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.dtype = leaf.dtype.copy()
        self.max_def = leaf.max_def
        self.width = leaf.type_length
        self.acc = _FixedWidthAcc[Self.native](num_rows)
        self.dict = List[Scalar[Self.native]]()

    def _decode_be(self, span: Span[UInt8, _], off: Int) -> Scalar[Self.native]:
        """Big-endian, sign-extended from `width` bytes to the native width."""
        return Plain.decode_be_flba[Self.native](span, off, self.width)

    def _place(mut self, page: Page, mask: Optional[List[Bool]]) raises:
        var vspan = page.values()
        var idx = List[Int32]()
        var is_dict = page.is_dictionary()
        # DELTA_BYTE_ARRAY / BYTE_STREAM_SPLIT are decoded up-front into a
        # contiguous width-byte buffer; PLAIN reads `vspan` in place (no copy).
        var decoded = List[UInt8]()
        var use_decoded = False
        if is_dict:
            idx = Rle.decode(vspan[1:], Int(vspan[0]), page.num_present)
        elif not page.is_plain():
            decoded = page.encoding.decode_flba(
                vspan, page.num_present, self.width
            )
            use_decoded = True

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    if is_dict:
                        self.acc.append_present(self.dict[Int(idx[vi])])
                    elif use_decoded:
                        self.acc.append_present(
                            self._decode_be(Span(decoded), vi * self.width)
                        )
                    else:
                        self.acc.append_present(
                            self._decode_be(vspan, vi * self.width)
                        )
                else:
                    self.acc.append_null()

        page.scatter[place](self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            var span = page.body
            for i in range(page.num_values):
                self.dict.append(self._decode_be(span, i * self.width))
        else:
            self._place(page, None)

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        self._place(page, mask.copy())

    def finish(deinit self) raises -> DynArray:
        return self.acc^.finish(self.dtype^)


struct Int96LeafBuilder(LeafBuilder):
    """Legacy INT96 timestamps (12 bytes each) -> `int64` nanoseconds. PLAIN and
    RLE_DICTIONARY, the encodings INT96 ever uses. Mirrors `DecimalLeafBuilder`,
    decoding each fixed-width value with `_int96_nanos` instead of big-endian.
    """

    var dtype: dt.DynType  # timestamp(ns)
    var max_def: Int
    var acc: _FixedWidthAcc[DType.int64]
    var dict: List[Int64]

    def __init__(out self, num_rows: Int, leaf: LeafColumn):
        self.dtype = leaf.dtype.copy()
        self.max_def = leaf.max_def
        self.acc = _FixedWidthAcc[DType.int64](num_rows)
        self.dict = List[Int64]()

    def _place(mut self, page: Page, mask: Optional[List[Bool]]) raises:
        var vspan = page.values()
        var idx = List[Int32]()
        var is_dict = page.is_dictionary()
        if is_dict:
            idx = Rle.decode(vspan[1:], Int(vspan[0]), page.num_present)
        elif not page.is_plain():
            raise Error("parquet: unsupported INT96 encoding")

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    if is_dict:
                        self.acc.append_present(self.dict[Int(idx[vi])])
                    else:
                        self.acc.append_present(_int96_nanos(vspan, vi * 12))
                else:
                    self.acc.append_null()

        page.scatter[place](self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            var span = page.body
            for i in range(page.num_values):
                self.dict.append(_int96_nanos(span, i * 12))
        else:
            self._place(page, None)

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        self._place(page, mask.copy())

    def finish(deinit self) raises -> DynArray:
        return self.acc^.finish(self.dtype^)


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
        # DELTA_BYTE_ARRAY / BYTE_STREAM_SPLIT are decoded up-front into a
        # contiguous width-byte buffer; PLAIN reads `vspan` in place (no copy).
        var decoded = List[UInt8]()
        var use_decoded = False
        if is_dict:
            idx = Rle.decode(vspan[1:], Int(vspan[0]), page.num_present)
        elif not page.is_plain():
            decoded = page.encoding.decode_flba(
                vspan, page.num_present, self.width
            )
            use_decoded = True

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    if is_dict:
                        var o = Int(idx[vi]) * self.width
                        self.builder.append(
                            Span(self.dict_body)[o : o + self.width]
                        )
                    elif use_decoded:
                        var o = vi * self.width
                        self.builder.append(Span(decoded)[o : o + self.width])
                    else:
                        var o = vi * self.width
                        self.builder.append(vspan[o : o + self.width])
                else:
                    self.builder.append_null()

        page.scatter[place](self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            self.dict_body = List[UInt8]()
            self.dict_body.extend(page.body)
        else:
            self._place(page, None)

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        self._place(page, mask.copy())

    def finish(deinit self) raises -> DynArray:
        var b = self.builder^
        var out: DynArray = b.finish()
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

        page.scatter[place](self.max_def, mask)

    def _place_values(
        mut self,
        page: Page,
        values: List[Bool],
        mask: Optional[List[Bool]] = None,
    ) raises:
        """Scatter already-decoded present booleans (the RLE path) honoring def
        levels; with `mask`, only selected rows are appended."""

        @parameter
        def place(present_here: Bool, selected: Bool, vi: Int) raises:
            if selected:
                if present_here:
                    self.builder.append(values[vi])
                else:
                    self.builder.append_null()

        page.scatter[place](self.max_def, mask)

    def consume(mut self, var page: Page) raises:
        if page.dictionary:
            raise Error("parquet: dictionary-encoded bool not supported")
        if page.is_plain():
            self._place(page, page.values())
        else:
            # RLE booleans (arrow's DataPage v2) share the nested decoder.
            self._place_values(
                page, page.encoding.decode_bool(page.values(), page.num_present)
            )

    def consume_selected(mut self, var page: Page, mask: List[Bool]) raises:
        if page.is_plain():
            self._place(page, page.values(), mask.copy())
        else:
            self._place_values(
                page,
                page.encoding.decode_bool(page.values(), page.num_present),
                mask.copy(),
            )

    def finish(deinit self) raises -> DynArray:
        var b = self.builder^
        var out: DynArray = b.finish()
        return out^


# ---------------------------------------------------------------------------
# LeafSet — which leaf kinds a decode path is compiled for
# ---------------------------------------------------------------------------


struct LeafSet(ImplicitlyCopyable, Movable):
    """The set of Arrow leaf kinds a `ColumnReader` is compiled to decode.

    `ColumnReader._dispatch` is a ladder of ~20 arms, each instantiating its own
    `LeafBuilder` and decode loop, and it is instantiated twice (flat and
    leveled). Because the leaf's dtype is a *runtime* value, every arm is
    reachable — so a scan that only ever reads `int64` and `string` still links
    the builders for bool, binary, every other integer width, both float widths,
    the temporal types and all four decimals. Measured on the `query_scan` gate:
    24 `PrimitiveLeafBuilder` instantiations for a two-type schema, and a
    Parquet scan roughly doubling a minimal AOT binary (Q4.6).

    An AOT query knows its schema when it is compiled. `LeafSet` is how it says
    so: each bit enables one arm, and `comptime if` drops the rest before they
    are ever instantiated. `all()` is the default and compiles the full ladder,
    so the dynamic lane is unchanged.

    A file containing a leaf the set excludes raises at decode time rather than
    decoding it wrongly — the error names the missing kind and the parameter.
    """

    var mask: UInt32

    def __init__(out self, mask: UInt32):
        self.mask = mask

    def __or__(self, other: Self) -> Self:
        return Self(self.mask | other.mask)

    def has(self, other: Self) -> Bool:
        """Whether every kind in `other` is enabled here."""
        return (self.mask & other.mask) == other.mask

    @staticmethod
    def all() -> Self:
        return Self(0xFFFFFFFF)


# One bit per `_dispatch` arm. Narrow ints share INT32 storage in Parquet but
# each is its own arm (and its own `PrimitiveLeafBuilder`), so each gets a bit.
comptime LEAF_INT8 = LeafSet(1 << 0)
comptime LEAF_INT16 = LeafSet(1 << 1)
comptime LEAF_INT32 = LeafSet(1 << 2)
comptime LEAF_INT64 = LeafSet(1 << 3)
comptime LEAF_UINT8 = LeafSet(1 << 4)
comptime LEAF_UINT16 = LeafSet(1 << 5)
comptime LEAF_UINT32 = LeafSet(1 << 6)
comptime LEAF_UINT64 = LeafSet(1 << 7)
comptime LEAF_FLOAT16 = LeafSet(1 << 8)
comptime LEAF_FLOAT32 = LeafSet(1 << 9)
comptime LEAF_FLOAT64 = LeafSet(1 << 10)
comptime LEAF_BOOL = LeafSet(1 << 11)
comptime LEAF_STRING = LeafSet(1 << 12)
comptime LEAF_LARGE_STRING = LeafSet(1 << 13)
comptime LEAF_BINARY = LeafSet(1 << 14)
comptime LEAF_LARGE_BINARY = LeafSet(1 << 15)
comptime LEAF_TEMPORAL32 = LeafSet(1 << 16)
comptime LEAF_TEMPORAL64 = LeafSet(1 << 17)
comptime LEAF_DECIMAL32 = LeafSet(1 << 18)
comptime LEAF_DECIMAL64 = LeafSet(1 << 19)
comptime LEAF_DECIMAL128 = LeafSet(1 << 20)
comptime LEAF_DECIMAL256 = LeafSet(1 << 21)
comptime LEAF_FIXED_SIZE_BINARY = LeafSet(1 << 22)
comptime LEAF_INT96 = LeafSet(1 << 23)


def leaf_of[T: DataType]() -> LeafSet:
    """The bit for `T` — say which **types** a read will see, not which bits.

    `LeafSet.all()` is the runtime-dispatching default; this is how a caller
    that knows its schema at compile time narrows it without hand-picking
    constants:

        read_table[leaf_of[Int64Type]() | leaf_of[StringType]()](path)

    The dtype is already a comptime parameter throughout the fused lane —
    `col("a", int64)` is a `NumericColumn[Int64Type]` — so this is the move the
    kernels already make: resolve the type at elaboration and let the branches
    that cannot be reached fold away.

    One generic, not one overload per family. `downcast[T, Trait]` is a comptime
    **type alias**, so `downcast[T, Trait].native` reads the member with no
    bound on `Trait` at all; only `downcast[T, Trait]()`, which *constructs* a
    value, needs `Defaultable`. `TemporalType`/`DecimalType` are not
    `Defaultable`, which is why an earlier version of this split into four
    overloads — that reason was wrong.

    Order matters: temporal and decimal are `PrimitiveType` but not
    `NumericType`, and `StringLikeType` refines `BinaryLikeType`, so the
    narrower family has to be tested first in each pair.
    """
    comptime if conforms_to(T, TemporalType):
        comptime if downcast[T, TemporalType].native == DType.int32:
            return LEAF_TEMPORAL32
        else:
            return LEAF_TEMPORAL64
    elif conforms_to(T, DecimalType):
        comptime native = downcast[T, DecimalType].native
        comptime if native == DType.int32:
            return LEAF_DECIMAL32
        elif native == DType.int64:
            return LEAF_DECIMAL64
        elif native == DType.int128:
            return LEAF_DECIMAL128
        else:
            return LEAF_DECIMAL256
    elif conforms_to(T, StringLikeType):
        comptime if downcast[T, StringLikeType].offset == DType.int32:
            return LEAF_STRING
        else:
            return LEAF_LARGE_STRING
    elif conforms_to(T, BinaryLikeType):
        comptime if downcast[T, BinaryLikeType].offset == DType.int32:
            return LEAF_BINARY
        else:
            return LEAF_LARGE_BINARY
    elif conforms_to(T, NumericType):
        comptime native = downcast[T, NumericType].native
        comptime if native == DType.bool:
            return LEAF_BOOL
        elif native == DType.int8:
            return LEAF_INT8
        elif native == DType.int16:
            return LEAF_INT16
        elif native == DType.int32:
            return LEAF_INT32
        elif native == DType.int64:
            return LEAF_INT64
        elif native == DType.uint8:
            return LEAF_UINT8
        elif native == DType.uint16:
            return LEAF_UINT16
        elif native == DType.uint32:
            return LEAF_UINT32
        elif native == DType.uint64:
            return LEAF_UINT64
        elif native == DType.float16:
            return LEAF_FLOAT16
        elif native == DType.float32:
            return LEAF_FLOAT32
        else:
            return LEAF_FLOAT64
    elif conforms_to(T, PrimitiveType):
        # bool and fixed-size binary: primitives outside the numeric family
        comptime if downcast[T, PrimitiveType].native == DType.bool:
            return LEAF_BOOL
        else:
            return LEAF_FIXED_SIZE_BINARY
    else:
        return LEAF_FIXED_SIZE_BINARY


# ---------------------------------------------------------------------------
# ColumnReader — decode one column chunk into a DecodedLeaf
# ---------------------------------------------------------------------------


struct ColumnReader[o: Origin[mut=False], leaves: LeafSet = LeafSet.all()](
    Movable
):
    """Decode one column chunk. `decode` picks the path from the leaf's max
    repetition: a flat leaf (`max_rep == 0`) fills a fixed-size `LeafBuilder`
    with the PLAIN-unsafe_memcpy / fused-gather fast paths; a repeated (list-element)
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
        """`data` is this column chunk's bytes — `meta.byte_range()`, not the
        whole file."""
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
    # Flat path — fixed-size LeafBuilder, one unsafe_memcpy/gather per all-present page
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
    ](mut self, var builder: B, mut codecs: CompressionLibs) raises -> DynArray:
        self._run(builder, codecs)
        return builder^.finish()

    def _flat_leaf(mut self, var arr: DynArray) raises -> DecodedLeaf:
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
        T: NumericType, phys: DType
    ](
        mut self,
        mut codecs: CompressionLibs,
        floor: Int,
        max_def: Int,
        var retag_to: Optional[dt.DynType] = None,
    ) raises -> DecodedLeaf:
        """Leveled primitive decode. `retag_to`, when set, reinterprets the
        int32/int64 storage array under a temporal Arrow type (the leaf's dtype).
        """
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
        var arr: DynArray = builder.finish()
        if retag_to:
            arr = arr^.view(retag_to.take())
        return DecodedLeaf(arr^, rep_out^, def_out^)

    def _drive_bytes[
        BT: BinaryLikeType
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

    def _drive_decimal[
        T: PrimitiveType
    ](
        mut self,
        dtype_inst: T,
        mut codecs: CompressionLibs,
        floor: Int,
        max_def: Int,
    ) raises -> DecodedLeaf:
        """Decode leveled FIXED_LEN_BYTE_ARRAY decimals (list/map elements).
        Big-endian two's-complement values, PLAIN or RLE_DICTIONARY;
        `dtype_inst` carries the leaf's precision/scale so the builder yields the
        right decimal type."""
        comptime native = T.native
        var width = self.pages.leaf.type_length
        var builder = PrimitiveBuilder[T](dtype_inst.copy(), self.num_rows)
        var dict = List[Scalar[native]]()
        var present = List[Scalar[native]]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()

        @parameter
        def handle_dict(pg: Page) raises:
            for i in range(pg.num_values):
                dict.append(
                    Plain.decode_be_flba[native](pg.body, i * width, width)
                )

        @parameter
        def decode_present(pg: Page) raises:
            present.clear()
            var vspan = pg.values()
            if pg.is_dictionary():
                var idx = Rle.decode(vspan[1:], Int(vspan[0]), pg.num_present)
                for i in range(pg.num_present):
                    present.append(dict[Int(idx[i])])
            elif pg.is_plain():
                for i in range(pg.num_present):
                    present.append(
                        Plain.decode_be_flba[native](vspan, i * width, width)
                    )
            else:
                var bytes = pg.encoding.decode_flba(
                    vspan, pg.num_present, width
                )
                for i in range(pg.num_present):
                    present.append(
                        Plain.decode_be_flba[native](
                            Span(bytes), i * width, width
                        )
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

    def _drive_fsb(
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        """Leveled FIXED_LEN_BYTE_ARRAY raw bytes (list/map elements) ->
        FixedSizeBinaryArray, PLAIN or RLE_DICTIONARY."""
        var width = self.pages.leaf.type_length
        var builder = FixedSizeBinaryBuilder(width, self.num_rows)
        var dict_body = List[UInt8]()
        var present = List[List[UInt8]]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()

        @parameter
        def handle_dict(pg: Page) raises:
            dict_body.clear()
            dict_body.extend(pg.body)

        @parameter
        def decode_present(pg: Page) raises:
            present.clear()
            var vspan = pg.values()
            if pg.is_dictionary():
                var idx = Rle.decode(vspan[1:], Int(vspan[0]), pg.num_present)
                for i in range(pg.num_present):
                    var o = Int(idx[i]) * width
                    present.append(List[UInt8](Span(dict_body)[o : o + width]))
            elif pg.is_plain():
                for i in range(pg.num_present):
                    var o = i * width
                    present.append(List[UInt8](vspan[o : o + width]))
            else:
                var bytes = pg.encoding.decode_flba(
                    vspan, pg.num_present, width
                )
                for i in range(pg.num_present):
                    var o = i * width
                    present.append(List[UInt8](Span(bytes)[o : o + width]))

        @parameter
        def place_present(vi: Int) raises:
            builder.append(Span(present[vi]))

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
    # per-dtype dispatch is preserved. Temporal, decimal, and fixed-size-binary
    # work on both paths — the leveled drive grows a builder and, for temporal
    # (int32/int64 storage retagged) and decimal (precision/scale-carrying
    # builder), yields the leaf's Arrow type.
    # -----------------------------------------------------------------------

    def _emit_numeric[
        T: NumericType, phys: DType, leveled: Bool
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
        BT: BinaryLikeType, leveled: Bool
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

    def _emit_temporal[
        T: NumericType, phys: DType, leveled: Bool
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        """Temporal and small-decimal (decimal32/decimal64) columns are
        physically int32/int64; the flat builder already carries the leaf's Arrow
        dtype, while the leveled drive produces the int32/int64 storage and is
        retagged to the leaf's type."""
        comptime if leveled:
            return self._drive_primitive[T, phys](
                codecs, floor, max_def, self.pages.leaf.dtype.copy()
            )
        else:
            var arr = self._build(
                PrimitiveLeafBuilder[T.native, phys](
                    self.num_rows, self.pages.leaf
                ),
                codecs,
            )
            return self._flat_leaf(arr^)

    def _emit_decimal[
        T: PrimitiveType, leveled: Bool
    ](
        mut self,
        dtype_inst: T,
        mut codecs: CompressionLibs,
        floor: Int,
        max_def: Int,
    ) raises -> DecodedLeaf:
        comptime if leveled:
            return self._drive_decimal[T](dtype_inst, codecs, floor, max_def)
        else:
            var arr = self._build(
                DecimalLeafBuilder[T.native](self.num_rows, self.pages.leaf),
                codecs,
            )
            return self._flat_leaf(arr^)

    def _emit_fsb[
        leveled: Bool
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        comptime if leveled:
            return self._drive_fsb(codecs, floor, max_def)
        else:
            var arr = self._build(
                FixedSizeBinaryLeafBuilder(self.num_rows, self.pages.leaf),
                codecs,
            )
            return self._flat_leaf(arr^)

    def _drive_int96(
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        """Leveled INT96 timestamps (list/map elements): 12-byte values decoded to
        int64 nanoseconds, retagged to the leaf's `timestamp(ns)` type."""
        var builder = PrimitiveBuilder[dt.Int64Type](self.num_rows)
        var dict = List[Int64]()
        var present = List[Int64]()
        var rep_out = List[Int32]()
        var def_out = List[Int32]()

        @parameter
        def handle_dict(pg: Page) raises:
            for i in range(pg.num_values):
                dict.append(_int96_nanos(pg.body, i * 12))

        @parameter
        def decode_present(pg: Page) raises:
            present.clear()
            var vspan = pg.values()
            if pg.is_dictionary():
                var idx = Rle.decode(vspan[1:], Int(vspan[0]), pg.num_present)
                for i in range(pg.num_present):
                    present.append(dict[Int(idx[i])])
            elif pg.is_plain():
                for i in range(pg.num_present):
                    present.append(_int96_nanos(vspan, i * 12))
            else:
                raise Error("parquet: unsupported INT96 encoding")

        @parameter
        def place_present(vi: Int) raises:
            builder.append(present[vi])

        @parameter
        def place_null() raises:
            builder.append_null()

        self._drive_leveled[
            handle_dict, decode_present, place_present, place_null
        ](codecs, floor, max_def, rep_out, def_out)
        var arr: DynArray = builder.finish()
        return DecodedLeaf(
            arr^.view(self.pages.leaf.dtype.copy()), rep_out^, def_out^
        )

    def _emit_int96[
        leveled: Bool
    ](
        mut self, mut codecs: CompressionLibs, floor: Int, max_def: Int
    ) raises -> DecodedLeaf:
        comptime if leveled:
            return self._drive_int96(codecs, floor, max_def)
        else:
            var arr = self._build(
                Int96LeafBuilder(self.num_rows, self.pages.leaf), codecs
            )
            return self._flat_leaf(arr^)

    def _dispatch[
        leveled: Bool
    ](mut self, mut codecs: CompressionLibs) raises -> DecodedLeaf:
        """Resolve the leaf's runtime dtype to its decode arm.

        Each arm sits behind `comptime if Self.leaves.has(...)`, so a reader
        compiled for a narrow `LeafSet` never instantiates the builders for the
        kinds it excludes — that gating is the whole point of the parameter
        (Q4.6). With the default `LeafSet.all()` every arm compiles and this is
        the ladder it has always been.

        Written as guarded blocks with early returns rather than one `elif`
        chain because an `elif` clause cannot be `comptime if`-gated.
        """
        ref leaf = self.pages.leaf
        ref vt = leaf.dtype
        # leveled: a value slot exists at/above the leaf's repetition floor (the
        # innermost enclosing list's element level), present at max_def. flat
        # ignores these (the emit helper's flat branch drops them).
        var f = leaf.slot_def
        var md = leaf.max_def

        # INT96 is a 12-byte timestamp whose Arrow dtype is timestamp(ns), so
        # only the *physical* type distinguishes it from an INT64 timestamp.
        # This test is unconditional on purpose: gating it would let an INT96
        # column fall through to the temporal arm and be decoded as INT64.
        if leaf.physical == PhysicalType.INT96:
            comptime if Self.leaves.has(LEAF_INT96):
                return self._emit_int96[leveled](codecs, f, md)
            else:
                raise Error(
                    "parquet: this reader was not compiled to decode INT96 --"
                    " add LEAF_INT96 to its LeafSet"
                )

        comptime if Self.leaves.has(LEAF_INT32):
            if vt == dt.int32:
                return self._emit_numeric[dt.Int32Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_INT64):
            if vt == dt.int64:
                return self._emit_numeric[dt.Int64Type, DType.int64, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_UINT32):
            if vt == dt.uint32:
                return self._emit_numeric[dt.UInt32Type, DType.uint32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_UINT64):
            if vt == dt.uint64:
                return self._emit_numeric[dt.UInt64Type, DType.uint64, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_FLOAT32):
            if vt == dt.float32:
                return self._emit_numeric[
                    dt.Float32Type, DType.float32, leveled
                ](codecs, f, md)

        comptime if Self.leaves.has(LEAF_FLOAT64):
            if vt == dt.float64:
                return self._emit_numeric[
                    dt.Float64Type, DType.float64, leveled
                ](codecs, f, md)

        comptime if Self.leaves.has(LEAF_FLOAT16):
            if vt == dt.float16:
                return self._emit_numeric[
                    dt.Float16Type, DType.float16, leveled
                ](codecs, f, md)

        comptime if Self.leaves.has(LEAF_INT8):
            if vt == dt.int8:
                return self._emit_numeric[dt.Int8Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_INT16):
            if vt == dt.int16:
                return self._emit_numeric[dt.Int16Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_UINT8):
            if vt == dt.uint8:
                return self._emit_numeric[dt.UInt8Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_UINT16):
            if vt == dt.uint16:
                return self._emit_numeric[dt.UInt16Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_BOOL):
            if vt == dt.bool_:
                return self._emit_bool[leveled](codecs, f, md)

        comptime if Self.leaves.has(LEAF_STRING):
            if vt.is_string():
                return self._emit_bytes[dt.StringType, leveled](codecs, f, md)

        comptime if Self.leaves.has(LEAF_LARGE_STRING):
            if vt.is_large_string():
                return self._emit_bytes[dt.LargeStringType, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_BINARY):
            if vt.is_binary():
                return self._emit_bytes[dt.BinaryType, leveled](codecs, f, md)

        comptime if Self.leaves.has(LEAF_LARGE_BINARY):
            if vt.is_large_binary():
                return self._emit_bytes[dt.LargeBinaryType, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_TEMPORAL32):
            if vt.is_date32() or vt.is_time32():
                return self._emit_temporal[dt.Int32Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_TEMPORAL64):
            if vt.is_timestamp() or vt.is_time64():
                return self._emit_temporal[dt.Int64Type, DType.int64, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_DECIMAL32):
            if vt.is_decimal32():
                return self._emit_temporal[dt.Int32Type, DType.int32, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_DECIMAL64):
            if vt.is_decimal64():
                return self._emit_temporal[dt.Int64Type, DType.int64, leveled](
                    codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_DECIMAL128):
            if vt.is_decimal128():
                return self._emit_decimal[dt.Decimal128Type, leveled](
                    vt.as_decimal128().copy(), codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_DECIMAL256):
            if vt.is_decimal256():
                return self._emit_decimal[dt.Decimal256Type, leveled](
                    vt.as_decimal256().copy(), codecs, f, md
                )

        comptime if Self.leaves.has(LEAF_FIXED_SIZE_BINARY):
            if vt.is_fixed_size_binary():
                return self._emit_fsb[leveled](codecs, f, md)

        raise Error(
            "parquet: this reader was not compiled to decode ",
            String(vt),
            " -- widen the LeafSet parameter, or use the default LeafSet.all()",
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Below this many rows a file is decoded on one thread — the thread-dispatch
# overhead is not worth it for small files.
comptime _PARALLEL_MIN_ROWS = 4096


struct ParquetFile[S: ByteSource = MappedFile, leaves: LeafSet = LeafSet.all()](
    Movable
):
    """A Parquet file opened for reading — mirrors PyArrow's `ParquetFile`.

    Owns a `ByteSource` (the file's bytes), the footer metadata, and the
    Parquet->Arrow schema mapping, all read once at construction. `read()`
    decodes columns into a `Table`; the source is released when the
    `ParquetFile` drops. Decoded values are copied into owned Arrow buffers, so
    a returned `Table` outlives the file.

    The source defaults to a local `MappedFile` — `ParquetFile(path)` opens a
    memory map — but any `ByteSource` (streaming, remote object store, …) can be
    passed instead; the whole decode path reads only through `S.read_at`.

    `leaves` narrows which leaf kinds the decoder is compiled for. The default
    compiles all of them and is what the dynamic lane uses; an AOT query that
    knows its schema can pass a narrower set and not link the builders for
    types it will never see (see `LeafSet`)."""

    var _source: Self.S
    var _meta: FileMetaData
    var _mapping: SchemaMapping
    var _codecs: ArcPointer[List[CompressionLibs]]
    """Reusable per-worker codec handles — see the pool comment in `read`."""

    def __init__(
        out self: ParquetFile[MappedFile, Self.leaves], path: String
    ) raises:
        # Convenience: open the local file as a memory map (S == MappedFile).
        self._source = MappedFile(path)
        self._meta = FileMetaData.read_footer(
            self._source.read_at(0, self._source.size())
        )
        self._mapping = SchemaMapping.from_parquet(self._meta)
        self._codecs = ArcPointer(List[CompressionLibs]())

    def __init__(out self, var source: Self.S) raises:
        # Read from any byte source; everything downstream goes through it.
        self._source = source^
        self._meta = FileMetaData.read_footer(
            self._source.read_at(0, self._source.size())
        )
        self._mapping = SchemaMapping.from_parquet(self._meta)
        self._codecs = ArcPointer(List[CompressionLibs]())

    def _read_at(
        ref self, offset: Int, length: Int
    ) -> Span[UInt8, origin_of(self)]:
        """One region of the file. Every read goes through here, so the source
        is only ever asked for the bytes actually needed — a column chunk, a
        page index, a bloom filter — never the whole file.

        `_source` is a field of `self`; widening its origin to the file's is
        sound and is what every caller relies on."""
        return rebind[Span[UInt8, origin_of(self)]](
            self._source.read_at(offset, length)
        )

    def _metadata_at(
        ref self, offset: Int, length: Int
    ) -> Span[UInt8, origin_of(self)]:
        """A metadata region (page index, bloom filter) whose recorded `length`
        may be absent — writers may omit it. Then the region has to be bounded
        by the end of the file, and only the Thrift reader knows where it really
        stops."""
        return self._read_at(
            offset, length if length > 0 else self._source.size() - offset
        )

    def metadata(self) -> FileMetaData:
        """The file's footer metadata (row groups, column chunks, statistics).
        """
        return self._meta.copy()

    def schema(self) -> Schema:
        """The file's Arrow schema."""
        return self._mapping.schema.copy()

    def num_row_groups(self) -> Int:
        return len(self._meta.row_groups)

    def num_rows(self) -> Int:
        return self._meta.num_rows

    def read(
        self,
        columns: Optional[List[String]] = None,
        row_groups: Optional[List[Int]] = None,
        row_selections: Optional[List[RowSelection]] = None,
    ) raises -> Table:
        """Decode columns into a Marrow `Table`.

        `columns` projects the read to the named top-level columns (in the given
        order); `None` reads all. Only the selected columns' chunks are touched.

        `row_groups` restricts the read to the given row-group indices (in that
        order); `None` reads all. Unselected row groups are never decoded — this
        is what predicate pushdown uses to skip groups that cannot match a
        filter.

        Every (row group, selected leaf) pair decodes independently — each reads
        a disjoint byte range of the shared read-only mmap and writes its own
        result slot — so the whole grid is decoded across `num_physical_cores()`
        workers. Each worker owns a `CompressionLibs` (the lazy `dlopen` handles
        and reused size cell are not shareable across threads); the mmap and
        metadata are read-only."""
        # the read plan: which column chunks to decode and how to reassemble them
        var plan = self._mapping.project(
            columns.value()
        ) if columns else self._mapping.full()

        # which row groups to decode (all, or the given subset in order)
        var rg_list = List[Int]()
        if row_groups:
            for rg in row_groups.value():
                if rg < 0 or rg >= len(self._meta.row_groups):
                    raise Error("parquet: row group index out of range")
                rg_list.append(rg)
        else:
            for rg in range(len(self._meta.row_groups)):
                rg_list.append(rg)

        if row_selections and len(row_selections.value()) != len(rg_list):
            raise Error(
                "parquet: row_selections must match the selected row groups"
            )

        var num_leaves = len(plan.decode_order)
        var num_rg = len(rg_list)
        var total = num_rg * num_leaves

        # rows across selected groups — governs the parallel-dispatch threshold
        var selected_rows = 0
        for rg in rg_list:
            selected_rows += self._meta.row_groups[rg].num_rows

        var nt = 1
        if total >= 2 and selected_rows >= _PARALLEL_MIN_ROWS:
            nt = min(num_physical_cores(), total)

        # One result slot per (row group, selected leaf), pre-sized so workers
        # assign by index without racing on list growth. (DecodedLeaf is
        # move-only, so slots are filled by appending, not the copy-based
        # `fill=`.)
        var grid = List[Optional[DecodedLeaf]](capacity=total)
        for _ in range(total):
            grid.append(None)

        # Per-worker codec handles, reused across calls. Constructing a
        # `CompressionLibs` is cheap, but the first decompress for a codec
        # `dlopen`s its library — and a streaming scan calls `read` once per
        # row-group window, so building them fresh on every call re-paid that open on
        # every window (measured: 4.7x on a Snappy 16-row-group scan). Workers
        # touch disjoint slots and the list is grown before dispatch, so this
        # keeps the "one `CompressionLibs` per worker" rule the handles need.
        # Shared ownership rather than a `mut self` field on purpose: `read`
        # must stay a borrow, since `_read_at` hands out spans whose origin is
        # `self` and `ColumnReader` requires an immutable one.
        var codecs = self._codecs
        while len(codecs[]) < nt:
            codecs[].append(CompressionLibs())

        @parameter
        def worker(w: Int) raises:
            ref codecs_w = codecs[][w]
            var t = w
            while t < total:
                var slot = t // num_leaves
                var rg_idx = rg_list[slot]
                # original column-chunk index for this compact slot
                var orig = plan.decode_order[t % num_leaves]
                ref rg = self._meta.row_groups[rg_idx]
                # the row selection for this group (shared by all its leaf
                # columns), None when nothing is pushed down or the group is
                # fully selected.
                var sel: Optional[RowSelection] = None
                if row_selections:
                    sel = row_selections.value()[slot].copy()
                # ColumnReader.decode picks the flat vs leveled path from the
                # leaf's max repetition, so one call serves every column shape.
                # Each worker fetches only its own chunk's bytes.
                var start, length = rg.columns[orig].meta_data.byte_range()
                var reader = ColumnReader[leaves=Self.leaves](
                    self._read_at(start, length),
                    rg.columns[orig].meta_data.copy(),
                    self._mapping.leaves[orig].copy(),
                    rg.num_rows,
                    sel^,
                )
                grid[t] = reader.decode(codecs_w)
                t += nt

        sync_parallelize[worker](nt)

        # Fold each selected row group's decoded leaves back into the Arrow tree.
        var batches = List[RecordBatch]()
        for i in range(num_rg):
            var decoded = List[DecodedLeaf]()
            for ci in range(num_leaves):
                decoded.append(grid[i * num_leaves + ci].take())
            var cols = List[DynArray]()
            for ref node in plan.nodes:
                cols.append(node.assemble(decoded))
            batches.append(
                RecordBatch(schema=Schema(copy=plan.schema), columns=cols^)
            )

        if len(batches) == 0:
            batches.append(RecordBatch.empty(Schema(copy=plan.schema)))
        return Table.from_batches(Schema(copy=plan.schema), batches)

    def statistics(self) raises -> List[List[ColumnStatistics]]:
        """Per-(row group, leaf column) decoded column-chunk statistics, indexed
        `result[row_group][leaf]`. `min`/`max` are typed scalars matching each
        column's Arrow type (each `None` when the file stored no bound or the
        type's bounds are not yet decoded); `null_count` is -1 if absent."""
        var out = List[List[ColumnStatistics]]()
        for ref rg in self._meta.row_groups:
            var row = List[ColumnStatistics]()
            for ci in range(len(rg.columns)):
                row.append(
                    ColumnStatistics.from_metadata(
                        self._mapping.leaves[ci].dtype,
                        rg.columns[ci].meta_data,
                    )
                )
            out.append(row^)
        return out^

    def _chunk_page_index(ref self, cc: ColumnChunk) raises -> PageIndex:
        """One column chunk's page index, each half absent when the writer
        stored none. The two readers below share this so the offsets are
        resolved in exactly one place."""
        var pi = PageIndex()
        if cc.offset_index_offset >= 0:
            var r = ThriftCompactReader(
                self._metadata_at(
                    cc.offset_index_offset, cc.offset_index_length
                ),
                0,
            )
            pi.offset_index = OffsetIndex.read(r)
        if cc.column_index_offset >= 0:
            var r = ThriftCompactReader(
                self._metadata_at(
                    cc.column_index_offset, cc.column_index_length
                ),
                0,
            )
            pi.column_index = ColumnIndex.read(r)
        return pi^

    def page_index(self) raises -> List[List[PageIndex]]:
        """The raw page index (OffsetIndex + ColumnIndex) for every (row group,
        leaf column), indexed `result[row_group][leaf]`. A chunk without a page
        index yields empty optionals."""
        var out = List[List[PageIndex]]()
        for ref rg in self._meta.row_groups:
            var row = List[PageIndex]()
            for ci in range(len(rg.columns)):
                row.append(self._chunk_page_index(rg.columns[ci]))
            out.append(row^)
        return out^

    def bloom_filter(
        self, row_group: Int, column: Int
    ) raises -> Optional[SplitBlockBloomFilter]:
        """The split-block bloom filter for one `(row_group, leaf column)`, or
        `None` when the chunk has none. Use `.might_contain(bytes)` to test a
        value's XXH64-hashed bytes — a `False` proves absence (no false negatives).
        """
        ref cc = self._meta.row_groups[row_group].columns[column]
        if cc.meta_data.bloom_filter_offset < 0:
            return None
        var data = self._metadata_at(
            cc.meta_data.bloom_filter_offset, cc.meta_data.bloom_filter_length
        )
        var r = ThriftCompactReader(data, 0)
        var hdr = BloomFilterHeader.read(r)
        return SplitBlockBloomFilter.from_bytes(
            data[r.pos : r.pos + hdr.num_bytes]
        )

    def page_bounds(self) raises -> List[List[List[PageBounds]]]:
        """Per (row group, leaf column, data page) decoded bounds, from the page
        index — indexed `result[rg][leaf][page]`. A column with no page index
        yields an empty page list. Predicate pushdown prunes pages with these.
        """
        var out = List[List[List[PageBounds]]]()
        for ref rg in self._meta.row_groups:
            var per_col = List[List[PageBounds]]()
            for ci in range(len(rg.columns)):
                var pi = self._chunk_page_index(rg.columns[ci])
                var pages = List[PageBounds]()
                if pi.offset_index and pi.column_index:
                    ref oi = pi.offset_index.value()
                    ref cix = pi.column_index.value()
                    ref dtype = self._mapping.leaves[ci].dtype
                    var np = len(oi.page_locations)
                    for p in range(np):
                        var first = oi.page_locations[p].first_row_index
                        var nxt = (
                            oi.page_locations[p + 1].first_row_index if p + 1
                            < np else rg.num_rows
                        )
                        var mn: Optional[DynScalar] = None
                        var mx: Optional[DynScalar] = None
                        # an all-null page (or a missing bound) prunes nothing
                        if not (p < len(cix.null_pages) and cix.null_pages[p]):
                            if p < len(cix.min_values):
                                mn = Statistics.decode(dtype, cix.min_values[p])
                            if p < len(cix.max_values):
                                mx = Statistics.decode(dtype, cix.max_values[p])
                        pages.append(PageBounds(nxt - first, mn^, mx^))
                per_col.append(pages^)
            out.append(per_col^)
        return out^


def read_table[
    leaves: LeafSet = LeafSet.all()
](
    path: String,
    columns: Optional[List[String]] = None,
    row_groups: Optional[List[Int]] = None,
    row_selections: Optional[List[RowSelection]] = None,
) raises -> Table:
    """Read a Parquet file into a Marrow `Table` — a convenience wrapper over
    `ParquetFile(path).read(...)` (mirrors `pyarrow.parquet.read_table`).

    `leaves` narrows which leaf kinds the decoder is compiled for; the default
    compiles all of them. An AOT program that knows its schema can cut the
    decode ladder it links — see `LeafSet`."""
    var pf = ParquetFile[MappedFile, leaves](path)
    return pf.read(columns, row_groups, row_selections)


# ---------------------------------------------------------------------------
# Metadata / statistics — read the footer without decoding any column data.
# Mirrors PyArrow's `read_metadata` / `ColumnChunkMetaData.statistics`.
# ---------------------------------------------------------------------------


struct RowSelection(Copyable, Movable):
    """Which rows of a row group to decode when a pushed-down predicate lets the
    reader skip pages.

    Row-group-relative, one flag per row: a data page whose
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
    bytes). No column data is decoded. Mirrors `pyarrow.parquet.read_metadata` —
    a convenience wrapper over `ParquetFile(path).metadata()`."""
    return ParquetFile(path).metadata()


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
    leaf column), indexed `result[row_group][leaf]` — a convenience wrapper over
    `ParquetFile(path).page_index()`."""
    return ParquetFile(path).page_index()


struct PageBounds(Copyable, Movable):
    """One data page's decoded bounds: its row count and the typed `min`/`max`
    (each `None` when the page is all-null or carried no bound)."""

    var num_rows: Int
    var min: Optional[DynScalar]
    var max: Optional[DynScalar]

    def __init__(
        out self,
        num_rows: Int,
        var min: Optional[DynScalar],
        var max: Optional[DynScalar],
    ):
        self.num_rows = num_rows
        self.min = min^
        self.max = max^


struct ColumnStatistics(Copyable, Movable):
    """Decoded per-column-chunk statistics: `null_count` (-1 if absent) and the
    `min`/`max` bounds as typed scalars (each `None` when the file stored no
    usable bound, so absence is in the type rather than a separate flag)."""

    var null_count: Int
    var min: Optional[DynScalar]
    var max: Optional[DynScalar]

    def __init__(out self):
        self.null_count = -1
        self.min = None
        self.max = None

    @staticmethod
    def from_metadata(dtype: dt.DynType, cm: ColumnMetaData) raises -> Self:
        """Decode a column chunk's `ColumnMetaData` statistics: `null_count` plus
        the min/max bounds as typed scalars (kept only when both decode)."""
        var cs = Self()
        cs.null_count = cm.null_count
        if cm.has_min_max:
            var mn = Statistics.decode(dtype, cm.min_value)
            var mx = Statistics.decode(dtype, cm.max_value)
            if Bool(mn) and Bool(mx):
                cs.min = mn^
                cs.max = mx^
        return cs^
