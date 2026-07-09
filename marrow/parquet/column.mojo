"""Column-chunk decoding: turn a stream of `Page`s into an Arrow leaf array.

`LeafBuilder` is the abstraction each physical layout implements (fixed-width
primitive, byte array, boolean); `ColumnReader` drives a `PageReader` through the
right builder based on the leaf's Arrow type. Adding a type means adding a
builder (or reusing one) and a dispatch arm — no new free-standing decode loops.
"""

from std.sys import size_of
from std.memory import memcpy

from ..arrays import AnyArray, ArrayData
from ..buffers import Buffer, Bitmap
from ..builders import BinaryLikeBuilder, BoolBuilder
from ..dtypes import (
    AnyDataType,
    BinaryLikeType,
    StringType,
    LargeStringType,
    BinaryType,
    LargeBinaryType,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)

from .page import Page, PageReader, PAGEKIND_DICT, read_u32le, read_fixed_le
from .encoding import rle_gather
from .compression import Codecs
from .schema import LeafColumn
from .nested import DecodedLeaf
from .values import decode_primitive_present, decode_bytes_present
from .format import ColumnMetaData


# ---------------------------------------------------------------------------
# LeafBuilder — accumulates decoded pages into one Arrow array
# ---------------------------------------------------------------------------


trait LeafBuilder(ImplicitlyDeletable, Movable):
    """Accumulates the values of a column chunk, page by page, into an array."""

    def consume(mut self, var page: Page) raises:
        ...

    def finish(deinit self) raises -> AnyArray:
        ...


struct PrimitiveLeafBuilder[store_dt: DType, phys_dt: DType = store_dt](
    LeafBuilder
):
    """Fixed-width values decoded straight into a contiguous buffer.

    `phys_dt` is the Parquet physical width read from the file, `store_dt` the
    Arrow storage width. When they match (the common case, incl. temporal types)
    a whole all-present PLAIN page is one memcpy; when they differ (int8/16 stored
    as physical INT32) each value is read wide and narrowed.
    """

    comptime SAME = Self.store_dt == Self.phys_dt

    var dtype: AnyDataType
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
        return read_fixed_le[Self.phys_dt](span, off).cast[Self.store_dt]()

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
        if page.kind == PAGEKIND_DICT:
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
            rle_gather[Self.store_dt](
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
            var present = List[Scalar[Self.store_dt]](
                capacity=page.num_present
            )
            decode_primitive_present[Self.store_dt, Self.phys_dt](
                page, self.dict, present
            )
            self._scatter(page, present.unsafe_ptr())

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


struct ByteArrayLeafBuilder[BT: BinaryLikeType](LeafBuilder):
    """Variable-length byte values (string/binary, 32- or 64-bit offsets). Bytes
    are appended verbatim, so the same builder serves UTF-8 and binary."""

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
        if page.kind == PAGEKIND_DICT:
            self.dict_body = List[UInt8]()
            self.dict_body.extend(page.body)
            var span = Span(self.dict_body)
            var off = 0
            for _ in range(page.num_values):
                var n = read_u32le(span, off)
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
                    var n = read_u32le(vspan, vi)
                    vi += 4
                    self._append(vspan[vi : vi + n])
                    vi += n
                else:
                    self.builder.append_null()
        else:
            # dictionary and DELTA_* share one decoder with the nested path.
            var values = decode_bytes_present(
                page, self.dict_body, self.dict_off, self.dict_len
            )
            self._scatter_values(page, values)

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
        if page.kind == PAGEKIND_DICT:
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

    def finish(deinit self) raises -> AnyArray:
        var b = self.builder^
        var out: AnyArray = b.finish()
        return out^


# ---------------------------------------------------------------------------
# ColumnReader — drive a PageReader through the right LeafBuilder
# ---------------------------------------------------------------------------


struct ColumnReader[o: Origin[mut=False]](Movable):
    var pages: PageReader[Self.o]
    var num_rows: Int
    var def_out: List[Int32]  # per-row def levels, kept only when carry_def

    def __init__(
        out self,
        data: Span[UInt8, Self.o],
        var meta: ColumnMetaData,
        var leaf: LeafColumn,
        num_rows: Int,
    ):
        self.pages = PageReader(data, meta^, leaf^)
        self.num_rows = num_rows
        self.def_out = List[Int32]()

    def _run[
        B: LeafBuilder
    ](mut self, mut builder: B, mut codecs: Codecs) raises:
        var carry = self.pages.leaf.carry_def
        var md = self.pages.leaf.max_def
        while self.pages.has_next():
            var pg = self.pages.next(codecs)
            if carry and pg.kind != PAGEKIND_DICT:
                # a page materializes def levels only when it has nulls; an
                # all-present page implies every slot is at max_def.
                if len(pg.def_levels) == pg.num_values:
                    self.def_out.extend(Span(pg.def_levels))
                else:
                    for _ in range(pg.num_values):
                        self.def_out.append(Int32(md))
            builder.consume(pg^)

    def _build[
        B: LeafBuilder
    ](mut self, var builder: B, mut codecs: Codecs) raises -> AnyArray:
        self._run(builder, codecs)
        return builder^.finish()

    def read(mut self, mut codecs: Codecs) raises -> DecodedLeaf:
        var arr = self._read_array(codecs)
        var defs = self.def_out^
        self.def_out = List[Int32]()
        return DecodedLeaf(False, arr^, List[Int32](), defs^)

    def _read_array(mut self, mut codecs: Codecs) raises -> AnyArray:
        ref leaf = self.pages.leaf
        ref dt = leaf.dtype
        if dt == int32:
            return self._build(
                PrimitiveLeafBuilder[DType.int32](self.num_rows, leaf), codecs
            )
        elif dt == int64:
            return self._build(
                PrimitiveLeafBuilder[DType.int64](self.num_rows, leaf), codecs
            )
        elif dt == uint32:
            return self._build(
                PrimitiveLeafBuilder[DType.uint32](self.num_rows, leaf), codecs
            )
        elif dt == uint64:
            return self._build(
                PrimitiveLeafBuilder[DType.uint64](self.num_rows, leaf), codecs
            )
        elif dt == float32:
            return self._build(
                PrimitiveLeafBuilder[DType.float32](self.num_rows, leaf), codecs
            )
        elif dt == float64:
            return self._build(
                PrimitiveLeafBuilder[DType.float64](self.num_rows, leaf), codecs
            )
        elif dt == int8:
            return self._build(
                PrimitiveLeafBuilder[DType.int8, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif dt == int16:
            return self._build(
                PrimitiveLeafBuilder[DType.int16, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif dt == uint8:
            return self._build(
                PrimitiveLeafBuilder[DType.uint8, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif dt == uint16:
            return self._build(
                PrimitiveLeafBuilder[DType.uint16, DType.int32](
                    self.num_rows, leaf
                ),
                codecs,
            )
        elif dt.is_date32() or dt.is_time32():
            return self._build(
                PrimitiveLeafBuilder[DType.int32](self.num_rows, leaf), codecs
            )
        elif (
            dt.is_timestamp()
            or dt.is_time64()
            or dt.is_date64()
            or dt.is_duration()
        ):
            return self._build(
                PrimitiveLeafBuilder[DType.int64](self.num_rows, leaf), codecs
            )
        elif dt == bool_:
            return self._build(BoolLeafBuilder(self.num_rows, leaf), codecs)
        elif dt.is_string():
            return self._build(
                ByteArrayLeafBuilder[StringType](self.num_rows, leaf), codecs
            )
        elif dt.is_large_string():
            return self._build(
                ByteArrayLeafBuilder[LargeStringType](self.num_rows, leaf),
                codecs,
            )
        elif dt.is_binary():
            return self._build(
                ByteArrayLeafBuilder[BinaryType](self.num_rows, leaf), codecs
            )
        elif dt.is_large_binary():
            return self._build(
                ByteArrayLeafBuilder[LargeBinaryType](self.num_rows, leaf),
                codecs,
            )
        else:
            raise Error("parquet: unsupported column type " + String(dt))
