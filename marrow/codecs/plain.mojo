# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Plain: each value's bytes, little-endian."""

from std.memory import bitcast
from std.sys import size_of

from .bits import Bits
from .byteorder import LittleEndian
from .core import Decoder, Emitter, Codec, Source


@fieldwise_init
struct Plain(Codec):
    """Each value's little-endian bytes, at its own width -- what a chain
    stores its last stream as, and how other codecs store the values they
    keep."""

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        comptime U = Bits.unsigned[T]
        comptime n = size_of[Scalar[T]]()
        var at = len(out)
        out.resize(at + len(src) * n, 0)
        var dst = Span(out)
        src.rewind()
        for i in range(len(src)):
            LittleEndian.store[U](dst, at + i * n, bitcast[U](src.next[T]()))

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        comptime U = Bits.unsigned[T]
        comptime n = size_of[Scalar[T]]()
        src.need(count * n)
        out.reserve(count)
        for i in range(count):
            out.emit(
                bitcast[T](LittleEndian.fixed[U](src.data, src.pos + i * n))
            )
        src.pos += count * n
        return out^
