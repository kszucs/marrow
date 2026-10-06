# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Dictionary: a column as its distinct values and a code per row."""

from ..errors import CorruptError
from .bitpack import BitPack
from .bits import Bits
from .byteorder import Leb128
from .core import Append, Decoder, Emitter, Codec, Source, Values
from .plain import Plain


@fieldwise_init
struct Dictionary(Codec):
    """The distinct-value count, the distinct values little-endian in
    first-seen order, then each row's code into them, bit-packed at the width
    of the largest code. Values compare by their bits."""

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var block = src.collect[T]()
        var values = Span(block)
        var distinct = List[Scalar[T]]()
        var codes = List[UInt32](capacity=len(values))
        Self.build(values, distinct, codes)
        Leb128.write(out, UInt64(len(distinct)))
        Plain.encode[T](Values(Span(distinct)), out)
        var width = Bits.width(UInt64(max(len(distinct) - 1, 0)))
        out.append(UInt8(width))
        BitPack.pack(Span(codes), width, out)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var k = src.length()
        if k > count:
            raise CorruptError(
                t"codecs: {k} dictionary values for {count} rows"
            )
        var read = Append[T](List[Scalar[T]](capacity=k))
        read = Plain.decode[T](src, k, read^)
        var distinct = read^.take()
        var width = Int(src.byte())
        if width > 32:
            raise CorruptError(t"codecs: dictionary code width {width}")
        var nbytes = (count * width + 7) // 8
        src.need(nbytes)
        var at = src.pos
        src.pos += nbytes
        out.reserve(count)
        # Eight codes unpacked in registers, checked at once, gathered and
        # emitted together -- no list of codes in between.
        var values = Span(distinct)
        var i = 0
        if width > 0:
            while i + 8 <= count:
                var codes = BitPack.unpack8(src.data, at, i * width, width)
                if codes.reduce_max() >= UInt64(k):
                    raise CorruptError(
                        t"codecs: dictionary code {codes.reduce_max()} out of"
                        t" {k}"
                    )
                var v = SIMD[T, 8](0)
                comptime for lane in range(8):
                    v[lane] = values.unsafe_get(Int(codes[lane]))
                out.emit8(v)
                i += 8
        while i < count:
            var code = Int(BitPack.get(src.data, at * 8 + i * width, width))
            if code >= k:
                raise CorruptError(t"codecs: dictionary code {code} out of {k}")
            out.emit(values[code])
            i += 1
        return out^

    @staticmethod
    def build[
        T: DType, C: DType
    ](
        values: Span[Scalar[T], _],
        mut distinct: List[Scalar[T]],
        mut codes: List[Scalar[C]],
    ):
        """Append each value's code to `codes`, and each value not seen
        before to `distinct` -- codes in first-seen order. Values compare by
        their bits, so `-0.0` is not `0.0` and every `NaN` with the same
        bits is one value."""
        var index = Dict[UInt64, Int]()
        for v in values:
            var key = Bits.of(v)
            var code = index.get(key)
            if code:
                codes.append(Scalar[C](code.value()))
            else:
                index[key] = len(distinct)
                codes.append(Scalar[C](len(distinct)))
                distinct.append(v)
