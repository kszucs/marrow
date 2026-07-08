"""Page-level reading: turn a column chunk's byte range into a stream of decoded
`Page` values.

`PageReader` owns the walk over a column chunk — locating each page, reading its
Thrift header, decompressing the body, and decoding definition levels — so the
column decoders in `column.mojo` only deal with already-decoded pages.
"""

from std.sys import size_of

from .encoding import rle_decode, rle_count_matches, bit_width
from .compression import Codecs
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


struct Page(Movable):
    """A decoded page: the (decompressed) body plus, for data pages, the
    definition levels, the byte offset where values start, and value counts."""

    var kind: Int
    var body: List[UInt8]
    var def_levels: List[Int32]
    var rep_levels: List[Int32]  # only populated for repeated (list) leaves
    var value_offset: Int
    var num_present: Int
    var encoding: Int
    var num_values: Int

    def __init__(out self):
        self.kind = PAGEKIND_DATA
        self.body = List[UInt8]()
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

    def values(ref self) -> Span[UInt8, origin_of(self.body)]:
        """The value bytes (after any level prefix)."""
        return Span(self.body)[self.value_offset :]


# ---------------------------------------------------------------------------
# PageReader — iterate the pages of one column chunk
# ---------------------------------------------------------------------------


struct PageReader[o: Origin[mut=False]](Movable):
    var data: Span[UInt8, Self.o]
    var meta: ColumnMetaData
    var leaf: LeafColumn
    var pos: Int
    var produced: Int  # data-page values yielded so far

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

    def has_next(self) -> Bool:
        return self.produced < self.meta.num_values

    def _decode_levels_v1(mut self, mut page: Page) raises:
        var body = Span(page.body)
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

        var page = Page()
        if ph.type == PAGE_DICTIONARY:
            page.kind = PAGEKIND_DICT
            page.num_values = ph.dictionary_page_header.value().num_values
            page.body = codecs.decompress(
                self.meta.codec, comp, ph.uncompressed_page_size
            )
            return page^

        if ph.type == PAGE_DATA:
            ref dph = ph.data_page_header.value()
            page.num_values = dph.num_values
            page.encoding = dph.encoding
            page.body = codecs.decompress(
                self.meta.codec, comp, ph.uncompressed_page_size
            )
            self._decode_levels_v1(page)
        elif ph.type == PAGE_DATA_V2:
            ref dph2 = ph.data_page_header_v2.value()
            page.num_values = dph2.num_values
            page.encoding = dph2.encoding
            var lvl_len = (
                dph2.repetition_levels_byte_length
                + dph2.definition_levels_byte_length
            )
            var body = List[UInt8]()
            body.extend(comp[0:lvl_len])  # levels are never compressed in v2
            if dph2.is_compressed:
                body.extend(
                    codecs.decompress(
                        self.meta.codec,
                        comp[lvl_len:],
                        ph.uncompressed_page_size - lvl_len,
                    )
                )
            else:
                body.extend(comp[lvl_len:])
            if (
                self.leaf.max_rep >= 1
                and dph2.repetition_levels_byte_length > 0
            ):
                page.rep_levels = rle_decode(
                    Span(body)[0 : dph2.repetition_levels_byte_length],
                    bit_width(self.leaf.max_rep),
                    page.num_values,
                )
            var cursor = dph2.repetition_levels_byte_length
            if (
                self.leaf.max_def >= 1
                and dph2.definition_levels_byte_length > 0
            ):
                page.def_levels = rle_decode(
                    Span(body)[
                        cursor : cursor + dph2.definition_levels_byte_length
                    ],
                    bit_width(self.leaf.max_def),
                    page.num_values,
                )
            page.value_offset = lvl_len
            page.num_present = page.num_values - dph2.num_nulls
            page.body = body^
        else:
            raise Error("parquet: unexpected page type")

        self.produced += page.num_values
        return page^
