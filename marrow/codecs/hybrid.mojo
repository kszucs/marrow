# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The RLE / bit-packed hybrid: small unsigned integers as runs, each either
one repeated value or groups of eight bit-packed values -- Parquet's RLE
encoding, which its levels, dictionary indices and booleans use."""

from std.sys import bit_width_of

from ..errors import CorruptError
from .bitpack import BitPack
from .bits import Bits
from .byteorder import Leb128, LittleEndian
from .core import Decoder, Emitter, Codec, Source


@fieldwise_init
struct Hybrid(Codec):
    """A width byte, then runs: a ULEB128 header whose low bit says which
    kind follows -- clear, `header >> 1` copies of one value in `ceil(width
    / 8)` little-endian bytes; set, `header >> 1` groups of eight values
    bit-packed at `width` bits, the last group padded with zeros.

    The payload after the width byte is Parquet's RLE_DICTIONARY data page
    body. Its kernels take the width from the caller, for Parquet's levels,
    whose width the schema gives."""

    @staticmethod
    def takes[T: DType]() -> Bool:
        return T.is_unsigned()

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var block = src.collect[T]()
        var values = Span(block)
        var top = Scalar[T](0)
        var runs = 0
        for i in range(len(values)):
            top = max(top, values[i])
            if i == 0 or values[i] != values[i - 1]:
                runs += 1
        var width = Bits.width(UInt64(top))
        out.append(UInt8(width))
        # Whichever kind of run takes fewer bytes, counting a byte a header.
        var as_runs = runs * (1 + (width + 7) // 8)
        var as_groups = (len(values) + 7) // 8 * width + 1
        if as_runs <= as_groups:
            Self.encode_runs(values, width, out)
        else:
            Self.encode_packed(values, width, out)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var width = Int(src.byte())
        if width > bit_width_of[T]():
            raise CorruptError(t"codecs: Hybrid width {width} exceeds {T}")
        var values = List[Scalar[T]](capacity=count)
        src.pos += Self.decode_runs(src.data[src.pos :], width, count, values)
        out.reserve(count)
        for v in values:
            out.emit(v)
        return out^

    # --- kernels ---------------------------------------------------------

    @staticmethod
    def decode_runs[
        T: DType
    ](
        data: Span[UInt8, _], width: Int, count: Int, mut out: List[Scalar[T]]
    ) raises -> Int:
        """Append the first `count` values of the runs in `data`, `width`
        bits each; return the bytes the runs read took. A run past `count`
        is read whole, as a bit-packed run's padding is."""
        if width == 0:
            for _ in range(count):
                out.append(0)
            return 0
        var byte_width = (width + 7) // 8
        var pos = 0
        var produced = 0
        while produced < count:
            var header: UInt64
            header, pos = Leb128.read(data, pos)
            var n = header >> 1
            if (header & 1) == 1:
                if n > UInt64((len(data) - pos) // width):
                    raise CorruptError("codecs: truncated bit-packed run")
                var take = min(Int(n) * 8, count - produced)
                BitPack.unpack(data, pos, width, take, out)
                produced += take
                pos += Int(n) * width
            else:
                if pos + byte_width > len(data):
                    raise CorruptError("codecs: truncated RLE run")
                var v = Scalar[T](BitPack.get(data, pos * 8, 8 * byte_width))
                pos += byte_width
                var take = Int(min(n, UInt64(count - produced)))
                for _ in range(take):
                    out.append(v)
                produced += take
        return pos

    @staticmethod
    def count_matches(
        data: Span[UInt8, _], width: Int, count: Int, target: UInt64
    ) raises -> Int:
        """How many of the first `count` values equal `target`, without
        materializing them -- one step for an RLE run, so a stream that is
        a single run, as a column with no nulls has, costs O(1)."""
        if width == 0:
            return count if target == 0 else 0
        var byte_width = (width + 7) // 8
        var pos = 0
        var produced = 0
        var matches = 0
        while produced < count:
            var header: UInt64
            header, pos = Leb128.read(data, pos)
            var n = header >> 1
            if (header & 1) == 1:
                if n > UInt64((len(data) - pos) // width):
                    raise CorruptError("codecs: truncated bit-packed run")
                var take = min(Int(n) * 8, count - produced)
                for k in range(take):
                    if BitPack.get(data, pos * 8 + k * width, width) == target:
                        matches += 1
                produced += take
                pos += Int(n) * width
            else:
                if pos + byte_width > len(data):
                    raise CorruptError("codecs: truncated RLE run")
                var v = BitPack.get(data, pos * 8, 8 * byte_width)
                pos += byte_width
                var take = Int(min(n, UInt64(count - produced)))
                if v == target:
                    matches += take
                produced += take
        return matches

    @staticmethod
    def encode_runs[
        T: DType
    ](values: Span[Scalar[T], _], width: Int, mut out: List[UInt8]):
        """Append `values` as RLE runs only, one per stretch of equal
        values -- what levels, which are mostly long runs, want. A `width`
        of 0 writes nothing."""
        if width == 0:
            return
        var byte_width = (width + 7) // 8
        var i = 0
        var n = len(values)
        while i < n:
            var j = i + 1
            while j < n and values[j] == values[i]:
                j += 1
            Leb128.write(out, UInt64(j - i) << 1)
            LittleEndian.put_le(out, UInt64(values[i]), byte_width)
            i = j

    @staticmethod
    def encode_packed[
        T: DType
    ](values: Span[Scalar[T], _], width: Int, mut out: List[UInt8]):
        """Append `values` as one bit-packed run, zeros after the last value
        up to a whole group of eight -- what dictionary indices, which seldom
        repeat, want. A `width` of 0 writes nothing."""
        var n = len(values)
        if width == 0 or n == 0:
            return
        var groups = (n + 7) // 8
        Leb128.write(out, (UInt64(groups) << 1) | 1)
        var end = len(out) + groups * width
        BitPack.pack(values, width, out)
        while len(out) < end:
            out.append(0)
