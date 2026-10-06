# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Constant: a column of one repeated value, as that value."""

from std.memory import bitcast

from ..errors import InvalidError
from .bits import Bits
from .byteorder import LittleEndian
from .core import Decoder, Emitter, Codec, Source


@fieldwise_init
struct Constant(Codec):
    """The value, once. Declared for a column that is not constant --
    comparing bits, so `-0.0` is not `0.0` -- it refuses to encode."""

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        if len(src) == 0:
            return
        src.rewind()
        var first = src.next[T]()
        for _ in range(1, len(src)):
            var v = src.next[T]()
            if Bits.of(v) != Bits.of(first):
                raise InvalidError(
                    t"codecs: Constant got a column holding {first} and {v}"
                )
        comptime U = Bits.unsigned[T]
        LittleEndian.append[U](out, bitcast[U](first))

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        if count == 0:
            return out^
        var v = src.fixed[T]()
        out.reserve(count)
        for _ in range(count):
            out.emit(v)
        return out^
