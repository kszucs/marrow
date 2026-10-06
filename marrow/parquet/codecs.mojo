# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Parquet's page encodings, over the codecs in `marrow.codecs`.

`Encoding` is the enum that dispatches a data page's *present* values to the
right one, so the flat and nested reader paths share one decoder per layout.
The value encodings themselves are codecs -- `Hybrid` (RLE / bit-packed, for
levels, dictionary indices and booleans), `DeltaBinaryPacked`, `BitPack`,
`ByteStreamSplitValues` -- and what is here is how Parquet frames them over Arrow
arrays:

- `PlainValues` · `DictionaryValues` · `ByteStreamSplitValues` -- the data-page value
  encodings `Encoding` dispatches to, over the codecs (`PlainBinary`,
  `DeltaLengthByteArray`, `DeltaByteArray` for byte arrays):
  present values taken out of an array, widened or narrowed to the physical
  type, and dictionary pages assembled.
- `Compression` -- the page compression codec (dispatches onto `Lz4`, `Zstd`
  and `CompressionLibs` in `marrow.utils.compression`).
"""

from std.memory import bitcast
from std.sys import size_of

from ..views import BufferView
from ..errors import CorruptError, NotImplementedError
from ..buffers import Buffer
from ..arrays import (
    DynArray,
    PrimitiveArray,
    BinaryLikeArray,
    BytesArray,
    BoolArray,
    FixedSizeBinaryArray,
)
from .. import dtypes as dt
from ..codecs import (
    BitPack,
    Bits,
    ByteStreamSplit,
    DeltaBinaryPacked,
    DeltaByteArray,
    DeltaLengthByteArray,
    Dictionary,
    Hybrid,
    PlainBinary,
    LittleEndian,
    Leb128,
)
from ..utils import CompressionLibs, Lz4, Zstd


def present_bytes[
    A: BytesArray
](arr: A, mut offsets: List[Int32], mut data: List[UInt8]) raises:
    """The present values of a byte-array column, nulls left out, as the
    offsets and data a binary codec takes."""
    offsets.append(Int32(len(data)))
    for i in range(len(arr)):
        if arr.is_valid(i):
            data.extend(arr.unsafe_get(UInt(i)).as_bytes())
            offsets.append(Int32(len(data)))


@no_inline
def _out_of_range() raises:
    """One raise for every dictionary-gather check: the check is copied into
    each value type's gather, the error once."""
    raise CorruptError(
        "parquet: dictionary page indices run past the page or the dictionary"
    )


struct PlainValues:
    """The PLAIN codec — values laid out in order: fixed-width little-endian for
    primitives, bit-packed for booleans, 4-byte-length-prefixed for byte arrays.
    Encode takes a present-value Arrow array; decode returns the present values.
    """

    @staticmethod
    def encode_primitive[
        store: dt.PrimitiveType, phys: DType, big_endian: Bool = False
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """PLAIN fixed-width encode of the present values, `phys`-wide. `store`
        may be numeric, temporal, decimal, or interval; `big_endian` (for DECIMAL
        FIXED_LEN_BYTE_ARRAY) writes the two's-complement value most-significant
        byte first."""
        comptime W = size_of[Scalar[phys]]()
        for i in range(arr.length):
            if arr.is_valid(i):
                var bytes = (
                    arr[i]
                    .value()
                    .cast[phys]()
                    .as_bytes[big_endian=big_endian]()
                )
                for b in range(W):
                    out.append(bytes[b])

    @staticmethod
    def encode_bool(arr: BoolArray, mut out: List[UInt8]) raises:
        var bits = arr.values()
        var present = List[UInt8](capacity=arr.length - arr.null_count())
        for i in range(arr.length):
            if arr.is_valid(i):
                present.append(UInt8(bits.test(i)))
        BitPack.pack(Span(present), 1, out)

    @staticmethod
    def encode_bytes[A: BytesArray](arr: A, mut out: List[UInt8]) raises:
        """PLAIN byte arrays: each present value's 4-byte LE length then its raw
        bytes. Serves string/binary and their large_ variants alike."""
        for i in range(len(arr)):
            if arr.is_valid(i):
                PlainBinary.put(out, arr.unsafe_get(UInt(i)).as_bytes())

    @staticmethod
    def encode_fixed_size_binary(
        arr: FixedSizeBinaryArray, mut out: List[UInt8]
    ) raises:
        """FIXED_LEN_BYTE_ARRAY: the present values' `byte_width` bytes, no
        length prefix (the width is in the schema)."""
        for i in range(arr.length):
            if arr.is_valid(i):
                out.extend(Span(arr[i].value()))

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], np: Int, mut out: List[Scalar[store]]) raises:
        comptime PW = size_of[Scalar[phys]]()
        for i in range(np):
            out.append(LittleEndian.fixed[phys](values, i * PW).cast[store]())

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _],
        np: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        _ = PlainBinary.decode(values, 0, np, offsets, data)

    @staticmethod
    def decode_bool(values: Span[UInt8, _], np: Int) raises -> List[Bool]:
        var bits = List[Int32](capacity=np)
        BitPack.unpack(values, 0, 1, np, bits)
        var out = List[Bool](capacity=np)
        for b in bits:
            out.append(b == 1)
        return out^


struct DictionaryValues:
    """The dictionary codec — a dictionary page of distinct values, then data
    pages of RLE/bit-packed indices into it. `encode` builds both from a column;
    `decode_page_*` reads the dictionary page and `decode_*` reads a data page's
    indices and gathers the values."""

    # --- encode: column -> dictionary-page bytes + per-present-value indices ---

    @staticmethod
    def gather[
        dt: DType, do: Origin[mut=True]
    ](
        data: Span[UInt8, _],
        width: Int,
        count: Int,
        dict: Span[Scalar[dt], _],
        dest: Span[Scalar[dt], do],
        dest_offset: Int = 0,
    ) raises:
        """Decode `count` RLE/bit-packed dictionary indices and write `dict[idx]`
        straight into `dest` at `dest_offset` -- fuses `Hybrid.decode_runs` and
        the gather, with no index buffer; an index past the dictionary raises.
        Callers pass `Span`s (a `List` or a `BufferView.as_span()`), so no
        pointer crosses the boundary."""
        if count == 0:
            return
        # Raw pointers rather than `Span` subscripts: indexing both spans
        # measured 7% slower on a dictionary-encoded read, every index having
        # been bounds-checked above.
        var dp = dest.unsafe_ptr().unsafe_offset(dest_offset)
        var kp = dict.unsafe_ptr()
        if width == 0:
            if len(dict) == 0:
                _out_of_range()
            for i in range(count):
                dp[unsafe_offset=i] = kp[unsafe_offset=0]
            return
        var byte_width = (width + 7) // 8
        var pos = 0
        var produced = 0
        while produced < count:
            var header: UInt64
            header, pos = Leb128.read(data, pos)
            if (header & 1) == 1:
                # SIMD-unpack 8 indices at a time (width <= 32 for dict indices),
                # then gather dict values — the gather stays scalar (the
                # dictionary is small and cache-resident, so it is ~free).
                var num_groups = Int(header >> 1)
                var num_vals = num_groups * 8
                if num_groups * width > len(data) - pos:
                    _out_of_range()
                var g = 0
                while g < num_vals:
                    var idxv = BitPack.unpack8(data, pos, g * width, width)
                    if Int(idxv.reduce_max()) >= len(dict):
                        _out_of_range()
                    if produced + 8 <= count:
                        # comptime-unrolled extraction: lane index must be a
                        # compile-time constant or the SIMD lane-select is
                        # runtime.
                        comptime for j in range(8):
                            dp[unsafe_offset=produced + j] = kp[
                                unsafe_offset=Int(idxv[j])
                            ]
                        produced += 8
                    else:
                        var take = count - produced
                        for j in range(take):
                            dp[unsafe_offset=produced] = kp[
                                unsafe_offset=Int(idxv[j])
                            ]
                            produced += 1
                    g += 8
                pos += num_groups * width
            else:
                var run_len = Int(header >> 1)
                if pos + byte_width > len(data):
                    _out_of_range()
                var idx = Int(BitPack.get(data, pos * 8, 8 * byte_width))
                pos += byte_width
                if idx >= len(dict):
                    _out_of_range()
                var take = min(run_len, count - produced)
                for _ in range(take):
                    dp[unsafe_offset=produced] = kp[unsafe_offset=idx]
                    produced += 1

    @staticmethod
    def _encode_prim[
        store: dt.NumericType, phys: DType
    ](
        arr: PrimitiveArray[store],
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a primitive column: PLAIN-encode each distinct value
        (widened to `phys`) into `dict_body`, collect the per-value index.
        Values are keyed by their bits, so `0.0` and `-0.0` stay distinct."""
        var present = List[Scalar[store.native]](
            capacity=arr.length - arr.null_count()
        )
        var values = arr.values()
        for i in range(arr.length):
            if arr.is_valid(i):
                present.append(values[i])
        var distinct = List[Scalar[store.native]]()
        Dictionary.build(Span(present), distinct, indices)
        for v in distinct:
            comptime U = Bits.unsigned[phys]
            LittleEndian.append[U](dict_body, bitcast[U](v.cast[phys]()))
        return len(distinct)

    @staticmethod
    def _encode_bytes[
        A: BytesArray
    ](
        arr: A,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Dictionary-encode a byte-array column (string/binary, their large_
        and view variants): length-prefixed distinct values in the dictionary page, one
        index per present value."""
        var index = Dict[StringSlice[origin_of(arr)], Int]()
        for i in range(len(arr)):
            if arr.is_valid(i):
                var v = arr.unsafe_get(UInt(i))
                var code = index.get(v)
                if code:
                    indices.append(Int32(code.value()))
                else:
                    indices.append(Int32(len(index)))
                    index[v] = len(index)
                    PlainBinary.put(dict_body, v.as_bytes())
        return len(index)

    @staticmethod
    def encode(
        dtype: dt.DynType,
        col: DynArray,
        mut dict_body: List[UInt8],
        mut indices: List[Int32],
    ) raises -> Int:
        """Build the dictionary page bytes + per-value indices for `col`; returns
        the dictionary size. Dispatches on the Arrow type like the PLAIN encoders
        (bool is never dictionary-encoded)."""
        if dtype == dt.int32:
            return Self._encode_prim[dt.Int32Type, DType.int32](
                col.as_int32(), dict_body, indices
            )
        elif dtype == dt.int64:
            return Self._encode_prim[dt.Int64Type, DType.int64](
                col.as_int64(), dict_body, indices
            )
        elif dtype == dt.uint32:
            return Self._encode_prim[dt.UInt32Type, DType.uint32](
                col.as_uint32(), dict_body, indices
            )
        elif dtype == dt.uint64:
            return Self._encode_prim[dt.UInt64Type, DType.uint64](
                col.as_uint64(), dict_body, indices
            )
        elif dtype == dt.float32:
            return Self._encode_prim[dt.Float32Type, DType.float32](
                col.as_float32(), dict_body, indices
            )
        elif dtype == dt.float64:
            return Self._encode_prim[dt.Float64Type, DType.float64](
                col.as_float64(), dict_body, indices
            )
        elif dtype == dt.float16:
            return Self._encode_prim[dt.Float16Type, DType.float16](
                col.as_float16(), dict_body, indices
            )
        elif dtype == dt.int8:
            return Self._encode_prim[dt.Int8Type, DType.int32](
                col.as_int8(), dict_body, indices
            )
        elif dtype == dt.int16:
            return Self._encode_prim[dt.Int16Type, DType.int32](
                col.as_int16(), dict_body, indices
            )
        elif dtype == dt.uint8:
            return Self._encode_prim[dt.UInt8Type, DType.int32](
                col.as_uint8(), dict_body, indices
            )
        elif dtype == dt.uint16:
            return Self._encode_prim[dt.UInt16Type, DType.int32](
                col.as_uint16(), dict_body, indices
            )
        elif (
            dtype.is_binary_like()
            or dtype.is_string_view()
            or dtype.is_binary_view()
        ):

            def encode_bytes[
                A: BytesArray
            ](arr: A) raises {mut dict_body, mut indices, imm} -> Int:
                return Self._encode_bytes(arr, dict_body, indices)

            return col.dispatch_bytes(encode_bytes)
        else:
            raise NotImplementedError(
                t"parquet: cannot dictionary-encode column type {dtype}"
            )

    # --- decode ---

    @staticmethod
    def decode_page_primitive[
        store: DType, phys: DType
    ](
        body: Span[UInt8, _], num_values: Int, mut dict: List[Scalar[store]]
    ) raises:
        """Read a primitive dictionary page (PLAIN fixed-width) into `dict`."""
        comptime PW = size_of[Scalar[phys]]()
        for i in range(num_values):
            dict.append(LittleEndian.fixed[phys](body, i * PW).cast[store]())

    @staticmethod
    def decode_page_bytes(
        body: Span[UInt8, _],
        num_values: Int,
        mut dict_body: List[UInt8],
        mut dict_off: List[Int],
        mut dict_len: List[Int],
    ) raises:
        """Read a byte-array dictionary page: its bytes into `dict_body`, and
        where each value sits in them (see `byte_offsets`)."""
        dict_body.clear()
        dict_body.extend(body)
        Self.byte_offsets(body, num_values, dict_off, dict_len)

    @staticmethod
    def byte_offsets(
        body: Span[UInt8, _],
        num_values: Int,
        mut dict_off: List[Int],
        mut dict_len: List[Int],
    ) raises:
        """Where each value of a byte-array dictionary page starts and how
        long it is: the page is length-prefixed values, so each offset points
        past its 4-byte prefix. Replaces what the lists held -- the indices of
        a data page address one chunk's dictionary."""
        dict_off.clear()
        dict_len.clear()
        var off = 0
        for i in range(num_values):
            if off + 4 > len(body):
                raise CorruptError(
                    t"parquet: a dictionary page of {num_values} values ends"
                    t" at value {i}"
                )
            var n = Int(LittleEndian.fixed[DType.uint32](body, off))
            off += 4
            if n > len(body) - off:
                raise CorruptError(
                    t"parquet: dictionary value {i} of {n} bytes runs past the"
                    t" page"
                )
            dict_off.append(off)
            dict_len.append(n)
            off += n

    @staticmethod
    def decode_primitive[
        store: DType
    ](
        values: Span[UInt8, _],
        np: Int,
        dict: List[Scalar[store]],
        mut out: List[Scalar[store]],
    ) raises:
        var base = len(out)
        # `gather` writes every slot.
        out.resize(unsafe_uninit_length=base + np)
        Self.gather[store](
            values[1:],
            Int(values[0]),
            np,
            Span(dict),
            Span(out),
            base,
        )

    @staticmethod
    def decode_bytes(
        values: Span[UInt8, _],
        np: Int,
        dict_body: List[UInt8],
        dict_off: List[Int],
        dict_len: List[Int],
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        var indices = List[Int32]()
        _ = Hybrid.decode_runs(values[1:], Int(values[0]), np, indices)
        for i in range(np):
            var idx = Int(indices[i])
            if idx >= len(dict_off):
                raise CorruptError("parquet: dictionary index out of range")
            var start = dict_off[idx]
            data.extend(Span(dict_body)[start : start + dict_len[idx]])
            offsets.append(Int32(len(data)))


struct ByteStreamSplitValues:
    """The BYTE_STREAM_SPLIT codec — each value's bytes are transposed into
    per-byte planes (byte `k` of value `i` at `values[k*np + i]`)."""

    @staticmethod
    def decode_primitive[
        store: DType, phys: DType
    ](values: Span[UInt8, _], np: Int, mut out: List[Scalar[store]]) raises:
        for i in range(np):
            out.append(
                ByteStreamSplit.merged[phys](values, np, i).cast[store]()
            )

    @staticmethod
    def encode[
        store: dt.NumericType, phys: DType
    ](arr: PrimitiveArray[store], mut out: List[UInt8]) raises:
        """Transpose the present values into per-byte planes: byte `k` of value
        `i` at `out[base + k*np + i]` (inverse of `decode_primitive`)."""
        var present = List[Scalar[phys]](capacity=arr.length)
        var values = arr.values()
        for i in range(arr.length):
            if arr.is_valid(i):
                present.append(values[i].cast[phys]())
        ByteStreamSplit.split[phys](present, out)


@fieldwise_init
struct Encoding(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `Encoding` enum value. The decode methods dispatch a data page's
    *present* values to the per-encoding codec above; each appends `num_present`
    values from a value byte-span (nulls are placed later by the caller from the
    definition levels), so the same logic serves the flat and nested reader
    paths."""

    var code: Int

    comptime PLAIN = Self(0)
    comptime PLAIN_DICTIONARY = Self(2)
    comptime RLE = Self(3)
    comptime BIT_PACKED = Self(4)
    comptime DELTA_BINARY_PACKED = Self(5)
    comptime DELTA_LENGTH_BYTE_ARRAY = Self(6)
    comptime DELTA_BYTE_ARRAY = Self(7)
    comptime RLE_DICTIONARY = Self(8)
    comptime BYTE_STREAM_SPLIT = Self(9)

    def is_plain(self) -> Bool:
        return self == Self.PLAIN

    def is_dictionary(self) -> Bool:
        return self == Self.RLE_DICTIONARY or self == Self.PLAIN_DICTIONARY

    def decode_primitive[
        store: DType, phys: DType
    ](
        self,
        values: Span[UInt8, _],
        num_present: Int,
        dict: List[Scalar[store]],
        mut out: List[Scalar[store]],
    ) raises:
        """Append the present fixed-width values (widened `phys` -> `store`)."""
        if self.is_plain():
            PlainValues.decode_primitive[store, phys](values, num_present, out)
        elif self.is_dictionary():
            DictionaryValues.decode_primitive[store](
                values, num_present, dict, out
            )
        elif self == Self.DELTA_BINARY_PACKED:
            comptime if phys.is_integral() and phys.is_signed():
                if num_present == 0:
                    return
                var decoded = List[Scalar[phys]](capacity=num_present)
                _ = DeltaBinaryPacked.decode_blocks(
                    values, 0, num_present, decoded
                )
                for v in decoded:
                    out.append(v.cast[store]())
            else:
                raise CorruptError(
                    t"parquet: DELTA_BINARY_PACKED over a {phys} column"
                )
        elif self == Self.BYTE_STREAM_SPLIT:
            ByteStreamSplitValues.decode_primitive[store, phys](
                values, num_present, out
            )
        else:
            raise NotImplementedError(
                t"parquet: unsupported data page encoding {self.code}"
            )

    def decode_bytes(
        self,
        values: Span[UInt8, _],
        num_present: Int,
        dict_body: List[UInt8],
        dict_off: List[Int],
        dict_len: List[Int],
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises:
        """Append the present variable-length byte values to `data`, and
        where each ends to `offsets`."""
        if self.is_plain():
            PlainValues.decode_bytes(values, num_present, offsets, data)
        elif self.is_dictionary():
            DictionaryValues.decode_bytes(
                values,
                num_present,
                dict_body,
                dict_off,
                dict_len,
                offsets,
                data,
            )
        elif num_present == 0:
            pass
        elif self == Self.DELTA_LENGTH_BYTE_ARRAY:
            _ = DeltaLengthByteArray.decode(
                values, 0, num_present, offsets, data
            )
        elif self == Self.DELTA_BYTE_ARRAY:
            _ = DeltaByteArray.decode(values, 0, num_present, offsets, data)
        else:
            raise NotImplementedError(
                t"parquet: unsupported byte-array encoding {self.code}"
            )

    def decode_flba(
        self, values: Span[UInt8, _], num_present: Int, width: Int
    ) raises -> List[UInt8]:
        """Decode the present FIXED_LEN_BYTE_ARRAY values (each `width` bytes) of
        a non-PLAIN/dict page into a contiguous `num_present * width` buffer — the
        DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT encodings PyArrow emits for decimal
        / fixed_size_binary. PLAIN and dictionary pages are read in place by the
        leaf builders, so they never reach here."""
        if self == Self.DELTA_BYTE_ARRAY:
            var offsets: List[Int32] = [0]
            var data = List[UInt8](capacity=num_present * width)
            if num_present > 0:
                _ = DeltaByteArray.decode(values, 0, num_present, offsets, data)
            if len(data) != num_present * width:
                raise CorruptError(
                    t"parquet: DELTA_BYTE_ARRAY values are not {width} bytes"
                )
            return data^
        elif self == Self.BYTE_STREAM_SPLIT:
            return ByteStreamSplit.merge_bytes(values, num_present, width)
        else:
            raise NotImplementedError(
                t"parquet: unsupported FIXED_LEN_BYTE_ARRAY encoding "
                t"{self.code}"
            )

    def decode_bool(
        self, values: Span[UInt8, _], num_present: Int
    ) raises -> List[Bool]:
        """Return the present booleans — PLAIN bit-packed, or RLE (the encoding
        arrow/PyArrow use for boolean values in DataPage v2). An RLE boolean
        stream is a 4-byte little-endian length then a width-1 RLE/bit-packed
        hybrid run, exactly like a level stream."""
        if self.is_plain():
            return PlainValues.decode_bool(values, num_present)
        elif self == Self.RLE:
            var length = Int(LittleEndian.fixed[DType.uint32](values, 0))
            var decoded = List[Int32](capacity=num_present)
            _ = Hybrid.decode_runs(
                values[4 : 4 + length], 1, num_present, decoded
            )
            var out = List[Bool](capacity=num_present)
            for b in decoded:
                out.append(b == 1)
            return out^
        else:
            raise NotImplementedError(
                "parquet: non-plain bool encoding not supported"
            )


@fieldwise_init
struct Compression(Equatable, ImplicitlyCopyable, Movable):
    """A Parquet `CompressionCodec` value: the codec identity plus the
    `compress` / `decompress` operations, dispatched onto `Lz4` and `Zstd` and
    onto a `CompressionLibs` handle pool (the `dlopen` bindings in
    `utils/compression.mojo`) for the rest.

    Enum values:
        0 UNCOMPRESSED  1 SNAPPY  2 GZIP  4 BROTLI  5 LZ4  6 ZSTD  7 LZ4_RAW
    """

    var code: Int

    comptime UNCOMPRESSED = Self(0)
    comptime SNAPPY = Self(1)
    comptime GZIP = Self(2)
    comptime BROTLI = Self(4)
    comptime LZ4 = Self(5)
    comptime ZSTD = Self(6)
    comptime LZ4_RAW = Self(7)

    def needs_libs(self, native: Bool) -> Bool:
        """Whether decompressing this codec may call into a `dlopen`ed
        library: all but UNCOMPRESSED and -- when `native`, as
        `CompressionLibs.native` says -- the codecs in Mojo, LZ4, LZ4_RAW and
        ZSTD. A codec not named here answers True, so one wired up later
        opens the libraries first rather than racing to."""
        if self == Self.UNCOMPRESSED:
            return False
        elif self == Self.LZ4 or self == Self.LZ4_RAW or self == Self.ZSTD:
            return not native
        else:
            return True

    def max_decompressed_length(self, n: Int) -> Int:
        """The most `n` bytes of this codec decode to, where it is known:
        `n` stored as is, and LZ4's and ZSTD's own bounds. The library
        codecs answer `Int.MAX`; their decoders check the length they are
        given against their input themselves."""
        if self == Self.UNCOMPRESSED:
            return n
        elif self == Self.ZSTD:
            return Zstd.max_decompressed_length(n)
        elif self == Self.LZ4 or self == Self.LZ4_RAW:
            return Lz4.max_decompressed_length(n)
        else:
            return Int.MAX

    def decompress_into(
        self,
        mut libs: CompressionLibs,
        src: Span[UInt8, _],
        out_size: Int,
        mut dst: List[UInt8],
    ) raises:
        """Append `src` decompressed, `out_size` bytes, to `dst` -- a page's
        reused scratch, after its levels for a v2 page."""
        self._check_size(src, out_size)
        var at = len(dst)
        dst.resize(unsafe_uninit_length=at + out_size)
        self._decompress_to(libs, src, Span(dst)[at:])

    def decompress_owned(
        self,
        mut libs: CompressionLibs,
        src: Span[UInt8, _],
        out_size: Int,
    ) raises -> Buffer[mut=False]:
        """Decompress `src` into a buffer of its own, for a decoder that keeps
        pointing into the page after the reader has moved on."""
        self._check_size(src, out_size)
        var out = Buffer.alloc_uninit[DType.uint8](out_size)
        self._decompress_to(
            libs, src, out.view[DType.uint8]().as_span()[:out_size]
        )
        return out^.to_immutable()

    def _check_size(self, src: Span[UInt8, _], out_size: Int) raises:
        """`out_size` comes from the page header: refuse one `src` cannot
        decode to before anything is allocated for it."""
        if out_size < 0 or out_size > self.max_decompressed_length(len(src)):
            raise CorruptError(
                t"parquet: {len(src)} compressed bytes cannot hold {out_size}"
            )

    def _decompress_to[
        o: Origin[mut=True]
    ](
        self,
        mut libs: CompressionLibs,
        src: Span[UInt8, _],
        dst: Span[UInt8, o],
    ) raises:
        """Decompress `src` into exactly `dst`."""
        var out_size = len(dst)
        var ptr = dst.unsafe_ptr()
        if self == Self.UNCOMPRESSED:
            BufferView(dst).copy_from(src[:out_size])
        elif self == Self.ZSTD:
            if libs.native:
                Zstd.decompress_into(src, dst)
            else:
                CompressionLibs.zstd_decompress(src, dst)
        elif self == Self.SNAPPY:
            libs.snappy_decompress(src, ptr, out_size)
        elif self == Self.LZ4_RAW:
            if libs.native:
                Lz4.decompress_block_into(src, dst)
            else:
                CompressionLibs.lz4_decompress_block(src, dst)
        elif self == Self.LZ4:
            if libs.native:
                Lz4.decompress_hadoop_into(src, dst)
            else:
                Lz4.decompress_hadoop_into[native=False](src, dst)
        elif self == Self.GZIP:
            libs.gzip_decompress(src, ptr, out_size)
        elif self == Self.BROTLI:
            libs.brotli_decompress(src, ptr, out_size)
        else:
            raise NotImplementedError(
                t"parquet: unsupported compression codec {self.code}"
            )

    def compress(
        self, mut libs: CompressionLibs, src: Span[UInt8, _]
    ) raises -> List[UInt8]:
        """Compress `src`, returning the codec's output bytes. Writers emit
        UNCOMPRESSED, SNAPPY, ZSTD, GZIP, BROTLI, LZ4, or LZ4_RAW."""
        var out = List[UInt8]()
        if self == Self.UNCOMPRESSED:
            out.extend(src)
        elif self == Self.ZSTD:
            if libs.native:
                Zstd.compress(src, out)
            else:
                CompressionLibs.zstd_compress(src, out)
        elif self == Self.SNAPPY:
            out = libs.snappy_compress(src)
        elif self == Self.LZ4_RAW:
            if libs.native:
                Lz4.compress_block(src, out)
            else:
                CompressionLibs.lz4_compress_block(src, out)
        elif self == Self.LZ4:
            if libs.native:
                Lz4.compress_hadoop(src, out)
            else:
                Lz4.compress_hadoop[native=False](src, out)
        elif self == Self.GZIP:
            out = libs.gzip_compress(src)
        elif self == Self.BROTLI:
            out = libs.brotli_compress(src)
        else:
            raise NotImplementedError(
                t"parquet: unsupported compression codec {self.code}"
            )
        return out^
