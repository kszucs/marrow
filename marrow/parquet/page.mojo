"""Page-level reading: turn a column chunk's byte range into a stream of decoded
`Page` values.

`PageReader` owns the walk over a column chunk — locating each page, reading its
Thrift header, decompressing the body, and decoding definition levels — so the
column decoders in `column.mojo` only deal with already-decoded pages.
"""

from std.sys import size_of

from .encoding import rle_decode, rle_count_matches, bit_width
from .compression import Codecs, CODEC_UNCOMPRESSED
from .schema import LeafColumn
from .format import (
    read_page_header,
    ColumnMetaData,
    PageHeader,
    PAGE_DICTIONARY,
    PAGE_DATA,
    PAGE_DATA_V2,
    ENC_PLAIN,
    ENC_PLAIN_DICTIONARY,
    ENC_RLE_DICTIONARY,
)


# ---------------------------------------------------------------------------
# Little-endian byte primitives
# ---------------------------------------------------------------------------


def read_u32le(body: Span[UInt8, _], off: Int) -> Int:
    return (
        Int(body[off])
        | (Int(body[off + 1]) << 8)
        | (Int(body[off + 2]) << 16)
        | (Int(body[off + 3]) << 24)
    )


def read_fixed_le[dt: DType](body: Span[UInt8, _], off: Int) -> Scalar[dt]:
    comptime W = size_of[Scalar[dt]]()
    var arr = InlineArray[UInt8, W](fill=0)
    for i in range(W):
        arr[i] = body[off + i]
    return SIMD[dt, 1].from_bytes[big_endian=False](arr)


def _count_equal(levels: List[Int32], target: Int) -> Int:
    var c = 0
    for d in levels:
        if Int(d) == target:
            c += 1
    return c


# ---------------------------------------------------------------------------
# Page — one decoded page
# ---------------------------------------------------------------------------

comptime PAGEKIND_DICT: Int = 0
comptime PAGEKIND_DATA: Int = 1


def _erase(s: Span[UInt8, _]) -> Span[UInt8, ImmutUntrackedOrigin]:
    """Drop origin tracking on a byte span. The reader keeps the backing storage
    (the mmap, or the PageReader's decompression scratch) alive for as long as
    each page is consumed, so the untracked view is always valid in use."""
    return rebind[Span[UInt8, ImmutUntrackedOrigin]](s)


struct Page(Movable):
    """A decoded page. `body` is a *view* — into the mmap for uncompressed pages
    (zero copy) or into the `PageReader`'s reused scratch for compressed pages —
    not an owned buffer, so no per-page allocation."""

    var kind: Int
    var body: Span[UInt8, ImmutUntrackedOrigin]
    var def_levels: List[Int32]
    var rep_levels: List[Int32]  # only populated for repeated (list) leaves
    var value_offset: Int
    var num_present: Int
    var encoding: Int
    var num_values: Int

    def __init__(out self, body: Span[UInt8, ImmutUntrackedOrigin]):
        self.kind = PAGEKIND_DATA
        self.body = body
        self.def_levels = List[Int32]()
        self.rep_levels = List[Int32]()
        self.value_offset = 0
        self.num_present = 0
        self.encoding = ENC_PLAIN
        self.num_values = 0

    def all_present(self) -> Bool:
        return self.num_present == self.num_values

    def is_plain(self) -> Bool:
        return self.encoding == ENC_PLAIN

    def is_dictionary(self) -> Bool:
        return (
            self.encoding == ENC_RLE_DICTIONARY
            or self.encoding == ENC_PLAIN_DICTIONARY
        )

    def values(self) -> Span[UInt8, ImmutUntrackedOrigin]:
        """The value bytes (after any level prefix)."""
        return self.body[self.value_offset :]


# ---------------------------------------------------------------------------
# PageReader — iterate the pages of one column chunk
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

    def _body(
        mut self,
        comp: Span[UInt8, _],
        uncompressed_size: Int,
        mut codecs: Codecs,
    ) raises -> Span[UInt8, ImmutUntrackedOrigin]:
        """A view of the page body: the mmap bytes directly for uncompressed
        pages, or the reused scratch after decompressing."""
        if self.meta.codec == CODEC_UNCOMPRESSED:
            return _erase(comp)
        codecs.decompress_into(
            self.meta.codec, comp, uncompressed_size, self.scratch
        )
        return _erase(Span(self.scratch))

    def has_next(self) -> Bool:
        return self.produced < self.meta.num_values

    def _decode_levels_v1(mut self, mut page: Page) raises:
        var body = page.body
        var cursor = 0
        var leveled = self.leaf.max_rep >= 1
        if leveled:
            var l = read_u32le(body, cursor)
            cursor += 4
            page.rep_levels = rle_decode(
                body[cursor : cursor + l],
                bit_width(self.leaf.max_rep),
                page.num_values,
            )
            cursor += l
        page.num_present = page.num_values
        if self.leaf.max_def >= 1:
            var bw = bit_width(self.leaf.max_def)
            var l = read_u32le(body, cursor)
            cursor += 4
            var levels = body[cursor : cursor + l]
            if leveled:
                # nested columns need the full def levels to rebuild offsets
                page.def_levels = rle_decode(levels, bw, page.num_values)
                page.num_present = _count_equal(
                    page.def_levels, self.leaf.max_def
                )
            else:
                # flat: count present in O(1) for the all-present run; only
                # materialize the level list when there are actually nulls.
                page.num_present = rle_count_matches(
                    levels, bw, page.num_values, Int32(self.leaf.max_def)
                )
                if page.num_present != page.num_values:
                    page.def_levels = rle_decode(levels, bw, page.num_values)
            cursor += l
        page.value_offset = cursor

    def next(mut self, mut codecs: Codecs) raises -> Page:
        var body_start = self.pos
        var ph = read_page_header(self.data, body_start)  # advances body_start
        var comp = self.data[body_start : body_start + ph.compressed_page_size]
        self.pos = body_start + ph.compressed_page_size

        if ph.type == PAGE_DICTIONARY:
            var page = Page(self._body(comp, ph.uncompressed_page_size, codecs))
            page.kind = PAGEKIND_DICT
            page.num_values = ph.dictionary_page_header.value().num_values
            return page^

        if ph.type == PAGE_DATA:
            var page = Page(self._body(comp, ph.uncompressed_page_size, codecs))
            ref dph = ph.data_page_header.value()
            page.num_values = dph.num_values
            page.encoding = dph.encoding
            self._decode_levels_v1(page)
            self.produced += page.num_values
            return page^
        elif ph.type == PAGE_DATA_V2:
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
                        codecs.decompress(
                            self.meta.codec,
                            comp[lvl_len:],
                            ph.uncompressed_page_size - lvl_len,
                        )
                    )
                )
            else:
                self.scratch.extend(comp[lvl_len:])

            var page = Page(_erase(Span(self.scratch)))
            page.num_values = dph2.num_values
            page.encoding = dph2.encoding
            if (
                self.leaf.max_rep >= 1
                and dph2.repetition_levels_byte_length > 0
            ):
                page.rep_levels = rle_decode(
                    page.body[0 : dph2.repetition_levels_byte_length],
                    bit_width(self.leaf.max_rep),
                    page.num_values,
                )
            var cursor = dph2.repetition_levels_byte_length
            if (
                self.leaf.max_def >= 1
                and dph2.definition_levels_byte_length > 0
            ):
                page.def_levels = rle_decode(
                    page.body[
                        cursor : cursor + dph2.definition_levels_byte_length
                    ],
                    bit_width(self.leaf.max_def),
                    page.num_values,
                )
            page.value_offset = lvl_len
            page.num_present = page.num_values - dph2.num_nulls
            self.produced += page.num_values
            return page^
        else:
            raise Error("parquet: unexpected page type")
