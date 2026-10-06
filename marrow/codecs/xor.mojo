# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Xor: each value's bits XOR the previous value's -- Gorilla's transform."""

from .bits import Bits
from .core import Decoder, Elementwise, Emitter, Mapped, Source, Unmapped
from .plain import Plain


@fieldwise_init
struct Xor(Elementwise):
    """Each value's bits XOR the previous value's. Neighbouring floats share
    sign, exponent and leading mantissa bits, which XOR to zeros."""

    comptime Out[T: DType]: DType = T

    @staticmethod
    @always_inline
    def forward[
        T: DType
    ](mut state: UInt64, v: Scalar[T]) -> Scalar[Self.Out[T]]:
        var b = Bits.of(v)
        var x = b ^ state
        state = b
        return Bits.to[Self.Out[T]](x)

    @staticmethod
    @always_inline
    def inverse[
        T: DType
    ](mut state: UInt64, v: Scalar[Self.Out[T]]) -> Scalar[T]:
        var b = Bits.of(v) ^ state
        state = b
        return Bits.to[T](b)

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var mapped = Mapped[Self, T](src^, out)
        Plain.encode[Self.Out[T]](mapped^, out)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var undo = Unmapped[Self, T](out^, src)
        return Plain.decode[Self.Out[T]](src, count, undo^).take()
