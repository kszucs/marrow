# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Delta: each integer as its difference from the one before."""

from .bits import Bits
from .core import Decoder, Elementwise, Emitter, Mapped, Source, Unmapped
from .plain import Plain


@fieldwise_init
struct Delta(Elementwise):
    """Each value minus the one before it, in the same type and wrapping on
    overflow. The first value's predecessor is taken to be as far behind it as
    the second is ahead, so the first difference repeats the second and a
    second `Delta` over evenly spaced values sees only equal ones. Sorted or
    slowly changing columns become small differences -- through `Zigzag` when
    they can fall."""

    comptime Out[T: DType]: DType = T

    @staticmethod
    def takes[T: DType]() -> Bool:
        return T.is_integral()

    @staticmethod
    def init[T: DType](first: Scalar[T], second: Scalar[T]) -> UInt64:
        return Bits.of(first - (second - first))

    @staticmethod
    @always_inline
    def forward[
        T: DType
    ](mut state: UInt64, v: Scalar[T]) -> Scalar[Self.Out[T]]:
        var d = v - Bits.to[T](state)
        state = Bits.of(v)
        return rebind[Scalar[Self.Out[T]]](d)

    @staticmethod
    @always_inline
    def inverse[
        T: DType
    ](mut state: UInt64, v: Scalar[Self.Out[T]]) -> Scalar[T]:
        var x = Bits.to[T](state) + rebind[Scalar[T]](v)
        state = Bits.of(x)
        return x

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
