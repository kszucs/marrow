# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""OrderPreserving: numbers mapped to unsigned integers of the same width
whose order is the numbers' order, as a radix sort or a packed (key, row)
sort needs."""

from std.memory import bitcast
from std.sys import bit_width_of

from .bits import Bits
from .core import Decoder, Elementwise, Emitter, Mapped, Source, Unmapped
from .plain import Plain


@fieldwise_init
struct OrderPreserving(Elementwise):
    """A number to the unsigned integer of its width that sorts as it does:
    a signed integer flips its sign bit, a float flips every bit when
    negative and its sign bit otherwise. `-0.0` sorts just below `0.0`, and a
    NaN with its sign clear above `+inf`."""

    comptime Out[T: DType]: DType = Bits.unsigned[T]

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

    # --- one value, for a sort key ---------------------------------------

    @staticmethod
    @always_inline
    def encode_value[T: DType](v: Scalar[T]) -> UInt64:
        """`v`'s key, zero-extended: ascending, so a descending order
        complements it."""
        comptime U = Bits.unsigned[T]
        comptime top = Scalar[U](bit_width_of[T]() - 1)
        comptime sign = Scalar[U](1) << top
        var b = bitcast[U](v)
        comptime if T.is_floating_point():
            # All ones when negative, else just the sign.
            b ^= (Scalar[U](0) - (b >> top)) | sign
        elif T.is_signed():
            b ^= sign
        return UInt64(b)

    @staticmethod
    @always_inline
    def decode_value[T: DType](u: UInt64) -> Scalar[T]:
        """The `T` that `encode_value` mapped to `u`."""
        comptime U = Bits.unsigned[T]
        comptime top = Scalar[U](bit_width_of[T]() - 1)
        comptime sign = Scalar[U](1) << top
        var b = u.cast[U]()
        comptime if T.is_floating_point():
            # A clear top bit is a negative value: every bit was flipped.
            b ^= (Scalar[U](0) - ((b >> top) ^ 1)) | sign
        elif T.is_signed():
            b ^= sign
        return bitcast[T](b)
