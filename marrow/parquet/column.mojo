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
from .encoding import rle_decode, rle_gather, delta_binary_packed_decode
from .compression import Codecs
from .schema import LeafColumn
from .format import (
    ColumnMetaData,
    ENC_DELTA_BINARY_PACKED,
    ENC_BYTE_STREAM_SPLIT,
)


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
        var np = page.num_present
        if page.is_plain():
            comptime if Self.SAME:
                # PLAIN stores only present values, contiguous — the store type
                # matches the physical width, so scatter straight from the page.
                self._scatter(
                    page, vspan.unsafe_ptr().bitcast[Scalar[Self.store_dt]]()
                )
            else:
                # narrow physical (e.g. int8 stored as INT32): widen each value.
                var vals = List[Scalar[Self.store_dt]](capacity=np)
                for i in range(np):
                    vals.append(self._read(vspan, i * PW))
                self._scatter(page, vals.unsafe_ptr())
        elif page.is_dictionary():
            if page.all_present():
                # fused index-decode + gather straight into the output buffer
                rle_gather[Self.store_dt](
                    vspan[1:],
                    Int(vspan[0]),
                    page.num_values,
                    self.dict.unsafe_ptr(),
                    self.values.view[Self.store_dt]().unsafe_ptr() + self.wpos,
                )
                self.wpos += page.num_values
            else:
                var vals = List[Scalar[Self.store_dt]](unsafe_uninit_length=np)
                rle_gather[Self.store_dt](
                    vspan[1:],
                    Int(vspan[0]),
                    np,
                    self.dict.unsafe_ptr(),
                    vals.unsafe_ptr(),
                )
                self._scatter(page, vals.unsafe_ptr())
        elif page.encoding == ENC_DELTA_BINARY_PACKED:
            var decoded = delta_binary_packed_decode(vspan, np)
            var vals = List[Scalar[Self.store_dt]](capacity=np)
            for i in range(np):
                vals.append(decoded[i].cast[Self.store_dt]())
            self._scatter(page, vals.unsafe_ptr())
        elif page.encoding == ENC_BYTE_STREAM_SPLIT:
            # each value's bytes are split across `np`-strided planes:
            # byte k of value i lives at vspan[k*np + i].
            var vals = List[Scalar[Self.store_dt]](capacity=np)
            for i in range(np):
                var raw = InlineArray[UInt8, PW](fill=0)

                comptime for k in range(PW):
                    raw[k] = vspan[k * np + i]
                vals.append(
                    SIMD[Self.phys_dt, 1]
                    .from_bytes[big_endian=False](raw)
                    .cast[Self.store_dt]()
                )
            self._scatter(page, vals.unsafe_ptr())
        else:
            raise Error(
                "parquet: unsupported data page encoding "
                + String(page.encoding)
            )

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
        var all_present = page.all_present()
        if page.is_plain():
            var vi = 0
            for row in range(page.num_values):
                if all_present or Int(page.def_levels[row]) == self.max_def:
                    var n = read_u32le(vspan, vi)
                    vi += 4
                    self._append(vspan[vi : vi + n])
                    vi += n
                else:
                    self.builder.append_null()
        elif page.is_dictionary():
            var indices = rle_decode(vspan[1:], Int(vspan[0]), page.num_present)
            var di = 0
            for row in range(page.num_values):
                if all_present or Int(page.def_levels[row]) == self.max_def:
                    var idx = Int(indices[di])
                    di += 1
                    var start = self.dict_off[idx]
                    # copy to a local so the append doesn't borrow self twice
                    var entry = List[UInt8]()
                    entry.extend(
                        Span(self.dict_body)[start : start + self.dict_len[idx]]
                    )
                    self._append(Span(entry))
                else:
                    self.builder.append_null()
        else:
            raise Error(
                "parquet: unsupported byte-array encoding "
                + String(page.encoding)
            )

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

    def __init__(
        out self,
        data: Span[UInt8, Self.o],
        var meta: ColumnMetaData,
        var leaf: LeafColumn,
        num_rows: Int,
    ):
        self.pages = PageReader(data, meta^, leaf^)
        self.num_rows = num_rows

    def _run[
        B: LeafBuilder
    ](mut self, mut builder: B, mut codecs: Codecs) raises:
        while self.pages.has_next():
            builder.consume(self.pages.next(codecs))

    def _build[
        B: LeafBuilder
    ](mut self, var builder: B, mut codecs: Codecs) raises -> AnyArray:
        self._run(builder, codecs)
        return builder^.finish()

    def read(mut self, mut codecs: Codecs) raises -> AnyArray:
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
