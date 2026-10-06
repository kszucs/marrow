# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Varint: each value's bits as a ULEB128, seven bits a byte."""

from ..errors import CorruptError
from .bits import Bits
from .byteorder import Leb128
from .core import Decoder, Emitter, Codec, Source


@fieldwise_init
struct Varint(Codec):
    """Each value's bits as a ULEB128: small values in few bytes, each at
    its own width rather than one `BitPack` sets for all."""

    @staticmethod
    def takes[T: DType]() -> Bool:
        return T.is_integral()

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        src.rewind()
        for _ in range(len(src)):
            Leb128.write(out, Bits.of(src.next[T]()))

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        out.reserve(count)
        for _ in range(count):
            var v = src.varint()
            if v > UInt64(Scalar[Bits.unsigned[T]].MAX):
                raise CorruptError(t"codecs: varint {v} overflows {T}")
            out.emit(Bits.to[T](v))
        return out^
