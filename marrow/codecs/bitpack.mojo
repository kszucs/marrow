# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Bit-packing: each value's distance from the column's smallest, at the
fewest bits that hold the largest distance, least significant bit first."""

from std.memory import bitcast
from std.sys import bit_width_of

from ..errors import CorruptError
from .bits import Bits
from .byteorder import Leb128, LittleEndian
from .core import Decoder, Emitter, Codec, Source


@fieldwise_init
struct BitPack(Codec):
    """The smallest value's bits as a ULEB128, a width byte, then each
    value's key less the smallest key, `width` bits each --
    `ceil(count * width / 8)` bytes. A key is a value's bits, its sign flipped for a signed integer, so
    a column clustered anywhere -- far from zero, or negative -- packs into
    few bits.

    The static `pack`, `unpack`, `unpack8` and `get` are the kernels over
    bytes that formats with their own framing -- Parquet's -- pack with."""

    @staticmethod
    @always_inline
    def _flip[T: DType]() -> UInt64:
        """What makes a value's bits order as the value does: the sign bit,
        flipped, for a signed integer; nothing for an unsigned one or a
        float, whose bits are ordered as they are."""
        comptime if T.is_integral() and T.is_signed():
            return UInt64(1) << UInt64(bit_width_of[T]() - 1)
        else:
            return 0

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        comptime U = Bits.unsigned[T]
        comptime flip = Self._flip[T]()
        var n = len(src)
        # One pass for the range of the keys, eight at a time ...
        src.rewind()
        var low8 = SIMD[DType.uint64, 8](UInt64.MAX)
        var high8 = SIMD[DType.uint64, 8](0)
        var i = 0
        while i + 8 <= n:
            var k = bitcast[U, 8](src.next8[T]()).cast[DType.uint64]() ^ flip
            low8 = min(low8, k)
            high8 = max(high8, k)
            i += 8
        var low = low8.reduce_min()
        var high = high8.reduce_max()
        while i < n:
            var k = Bits.of(src.next[T]()) ^ flip
            low = min(low, k)
            high = max(high, k)
            i += 1
        if n == 0:
            low = 0
            high = 0
        var width = Bits.width(high - low)
        Leb128.write(out, low ^ flip)
        out.append(UInt8(width))
        if width == 0:
            return
        # ... and one to pack each key's distance from the smallest.
        src.rewind()
        var acc: UInt64 = 0
        var acc_bits = 0
        for _ in range(n):
            var k = Bits.of(src.next[T]()) ^ flip
            Self._put(out, acc, acc_bits, k - low, width)
        Self._flush(out, acc, acc_bits)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        comptime U = Bits.unsigned[T]
        comptime flip = Self._flip[T]()
        var low = src.varint() ^ flip
        var w = Int(src.byte())
        if w > bit_width_of[T]():
            raise CorruptError(t"codecs: BitPack width {w} exceeds {T}")
        var nbytes = (count * w + 7) // 8
        src.need(nbytes)
        var at = src.pos
        src.pos += nbytes
        out.reserve(count)
        var i = 0
        while i + 8 <= count:
            var v = SIMD[DType.uint64, 8](0)
            if w > 0 and w <= 32:
                v = Self.unpack8(src.data, at, i * w, w)
            elif w > 32:
                comptime for lane in range(8):
                    v[lane] = Self.get(src.data, at * 8 + (i + lane) * w, w)
            out.emit8(bitcast[T, 8](((v + low) ^ flip).cast[U]()))
            i += 8
        while i < count:
            var off = Self.get(src.data, at * 8 + i * w, w)
            out.emit(Bits.to[T]((off + low) ^ flip))
            i += 1
        return out^

    # --- kernels ---------------------------------------------------------

    @staticmethod
    @always_inline
    def _put(
        mut out: List[UInt8],
        mut acc: UInt64,
        mut acc_bits: Int,
        v: UInt64,
        width: Int,
    ):
        """Append `v`, already under `1 << width`, above the `acc_bits < 64`
        bits pending in `acc`; each full word goes to `out` whole."""
        acc |= v << UInt64(acc_bits)
        acc_bits += width
        if acc_bits >= 64:
            LittleEndian.append[DType.uint64](out, acc)
            acc_bits -= 64
            acc = v >> UInt64(width - acc_bits) if acc_bits > 0 else 0

    @staticmethod
    def _flush(mut out: List[UInt8], acc: UInt64, acc_bits: Int):
        """Append the whole bytes covering the `acc_bits` pending."""
        for b in range((acc_bits + 7) // 8):
            out.append(UInt8((acc >> UInt64(8 * b)) & 0xFF))

    @staticmethod
    def pack[
        T: DType
    ](values: Span[Scalar[T], _], width: Int, mut out: List[UInt8]):
        """Append each value's low `width` bits (0 to 64), least significant
        first, then zero bits up to a whole byte."""
        if width == 0:
            return
        var mask = UInt64.MAX >> UInt64(64 - width)
        var acc: UInt64 = 0
        var acc_bits = 0
        for v in values:
            Self._put(out, acc, acc_bits, UInt64(v) & mask, width)
        Self._flush(out, acc, acc_bits)

    @staticmethod
    @always_inline
    def unpack8(
        data: Span[UInt8, _], byte_base: Int, bit_offset: Int, width: Int
    ) -> SIMD[DType.uint64, 8]:
        """The 8 values of `width` bits (1 to 32) starting `bit_offset` bits
        past `data[byte_base]`: one unaligned 64-bit load per lane, then a
        vector shift and mask. Bytes past the end of `data` read as zero."""
        debug_assert(
            width > 0 and width <= 32, "BitPack.unpack8 takes widths 1 to 32"
        )
        var first = byte_base + (bit_offset >> 3)
        if width <= 7 and first + 8 <= len(data):
            # All 8 values and the sub-byte offset fit one 64-bit word.
            var word = LittleEndian.fixed[DType.uint64](data, first)
            return Bits.unpack_lanes[DType.uint64, 8](
                word >> UInt64(bit_offset & 7), width
            )
        var mask = SIMD[DType.uint64, 8]((UInt64(1) << UInt64(width)) - 1)
        var words = SIMD[DType.uint64, 8](0)
        var shifts = SIMD[DType.uint64, 8](0)
        comptime for j in range(8):
            shifts[j] = UInt64((bit_offset + j * width) & 7)
        # A span is not padded -- a page can end on its last packed byte -- so
        # a group whose last word would run past the end reads `partial`ly.
        if byte_base + ((bit_offset + 7 * width) >> 3) + 8 <= len(data):
            comptime for j in range(8):
                words[j] = LittleEndian.fixed[DType.uint64](
                    data, byte_base + ((bit_offset + j * width) >> 3)
                )
        else:
            comptime for j in range(8):
                words[j] = LittleEndian.partial[DType.uint64](
                    data, byte_base + ((bit_offset + j * width) >> 3)
                )
        return (words >> shifts) & mask

    @staticmethod
    @always_inline
    def get(data: Span[UInt8, _], bit: Int, width: Int) -> UInt64:
        """The `width` bits (0 to 64) at bit `bit` of `data`, least
        significant first. Bytes past the end of `data` read as zero."""
        if width == 0:
            return 0
        var byte = bit >> 3
        var shift = bit & 7
        var v: UInt64
        if byte + 8 <= len(data):
            v = LittleEndian.fixed[DType.uint64](data, byte) >> UInt64(shift)
        else:
            v = LittleEndian.partial[DType.uint64](data, byte) >> UInt64(shift)
        if shift + width > 64:
            var spill = UInt64(data[byte + 8]) if byte + 8 < len(data) else 0
            v |= spill << UInt64(64 - shift)
        if width == 64:
            return v
        return v & ((UInt64(1) << UInt64(width)) - 1)

    @staticmethod
    def unpack[
        T: DType
    ](
        data: Span[UInt8, _],
        byte_base: Int,
        width: Int,
        count: Int,
        mut out: List[Scalar[T]],
    ):
        """Append the `count` values of `width` bits (0 to 64) packed from
        `data[byte_base]`. Bytes past the end of `data` read as zero."""
        var i = 0
        if width > 0 and width <= 32:
            while i + 8 <= count:
                var v = Self.unpack8(data, byte_base, i * width, width)
                comptime for j in range(8):
                    out.append(Scalar[T](v[j]))
                i += 8
        var base_bit = byte_base * 8
        while i < count:
            out.append(Scalar[T](Self.get(data, base_bit + i * width, width)))
            i += 1
