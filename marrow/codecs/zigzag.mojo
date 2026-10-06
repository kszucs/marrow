# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Zigzag: signed integers mapped to unsigned ones of the same width, so a
value of small magnitude stays small whichever its sign."""

from std.memory import bitcast
from std.sys import bit_width_of

from .bits import Bits
from .core import Decoder, Elementwise, Emitter, Mapped, Source, Unmapped
from .plain import Plain


@fieldwise_init
struct Zigzag(Elementwise):
    """`0, -1, 1, -2, ...` to `0, 1, 2, 3, ...`: a signed integer to the
    unsigned one of its width."""

    comptime Out[T: DType]: DType = Bits.unsigned[T]

    @staticmethod
    def takes[T: DType]() -> Bool:
        return T.is_integral() and T.is_signed()

    @staticmethod
    @always_inline
    def forward[
        T: DType
    ](mut state: UInt64, v: Scalar[T]) -> Scalar[Self.Out[T]]:
        return Bits.to[Self.Out[T]](Self.encode_value(v))

    @staticmethod
    @always_inline
    def inverse[
        T: DType
    ](mut state: UInt64, v: Scalar[Self.Out[T]]) -> Scalar[T]:
        return Self.decode_value[T](Bits.of(v))

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

    # --- one value, for protocols that zigzag a single integer ---------------

    @staticmethod
    @always_inline
    def encode_value[T: DType](v: Scalar[T]) -> UInt64:
        """`v` mapped to the unsigned integer of its width, zero-extended."""
        comptime top = Scalar[T](bit_width_of[T]() - 1)
        return Bits.of((v << 1) ^ (v >> top))

    @staticmethod
    @always_inline
    def decode_value[T: DType](u: UInt64) -> Scalar[T]:
        """The signed `T` that `encode_value` mapped to `u`."""
        var w = u.cast[Bits.unsigned[T]]()
        return bitcast[T](w >> 1) ^ -bitcast[T](w & 1)
