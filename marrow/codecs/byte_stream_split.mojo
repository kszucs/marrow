# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Byte stream split: each value's bytes transposed into per-byte planes."""

from std.memory import bitcast
from std.sys import size_of

from .bits import Bits
from .core import Decoder, Emitter, Codec, Source


@fieldwise_init
struct ByteStreamSplit(Codec):
    """Byte `k` of value `i` at `planes[k * count + i]`. Floating-point bytes
    of similar values line up -- signs and exponents in one plane, mantissa
    noise in another -- for a compressor to find. Parquet's
    BYTE_STREAM_SPLIT."""

    @staticmethod
    def split[T: DType](values: Span[Scalar[T], _], mut out: List[UInt8]):
        """Append `values`' byte planes to `out`."""
        comptime U = Bits.unsigned[T]
        comptime W = size_of[Scalar[T]]()
        comptime for k in range(W):
            for v in values:
                out.append(UInt8((bitcast[U](v) >> Scalar[U](8 * k)) & 0xFF))

    @staticmethod
    def merged[
        T: DType
    ](planes: Span[UInt8, _], count: Int, i: Int) -> Scalar[T]:
        """Value `i` of the `count` values whose planes are `planes`."""
        comptime U = Bits.unsigned[T]
        var v = Scalar[U](0)
        comptime for k in range(size_of[Scalar[T]]()):
            v |= Scalar[U](planes[k * count + i]) << Scalar[U](8 * k)
        return bitcast[T](v)

    @staticmethod
    def merge_bytes(
        planes: Span[UInt8, _], count: Int, width: Int
    ) -> List[UInt8]:
        """The `count` values of `width` bytes whose planes are `planes`,
        value after value -- `merged` for a width known only at run time."""
        var out = List[UInt8](length=count * width, fill=0)
        for i in range(count):
            for k in range(width):
                out[i * width + k] = planes[k * count + i]
        return out^

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var block = src.collect[T]()
        var values = Span(block)
        Self.split[T](values, out)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var n = count * size_of[Scalar[T]]()
        src.need(n)
        var planes = src.data[src.pos : src.pos + n]
        out.reserve(count)
        for i in range(count):
            out.emit(Self.merged[T](planes, count, i))
        src.pos += n
        return out^
