# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Bitwise kernels over integer columns: `bit_wise_and`, `bit_wise_or`,
`bit_wise_xor`, `bit_wise_not`, `shift_left` and `shift_right`.

Both operands of a binary kernel share one integer dtype, and the answer keeps
it. A shift moves the bits by the right operand's value, wrapping as the
type's arithmetic does; an amount below zero, or not below the type's
precision (its bit width, less the sign bit for a signed type), leaves the
value as it is. A right shift of a signed value is arithmetic.

`BitAndKernel`, `BitOrKernel` and `BitXorKernel` are associative and
commutative and have an identity, so they are folds too: `aggregate.mojo`'s
`BitAndFold`, `BitOrFold` and `BitXorFold`.
"""

from std.sys import bit_width_of

from ..dtypes import IntegerType
from .core import Associative, Commutative, Identity, family
from .numeric import NumericBinaryKernel, NumericUnaryKernel


struct BitAndKernel(Associative, Commutative, Identity, NumericBinaryKernel):
    comptime name = "bit_wise_and"
    comptime accepts = family[IntegerType]

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        """Every bit set: `~0 & x` is `x`."""
        return ~Scalar[T](0)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_integral(), "bit_wise_and: integers only"
        return a & b


struct BitOrKernel(Associative, Commutative, Identity, NumericBinaryKernel):
    comptime name = "bit_wise_or"
    comptime accepts = family[IntegerType]

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        """No bit set: `0 | x` is `x`."""
        return Scalar[T](0)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_integral(), "bit_wise_or: integers only"
        return a | b


struct BitXorKernel(Associative, Commutative, Identity, NumericBinaryKernel):
    comptime name = "bit_wise_xor"
    comptime accepts = family[IntegerType]

    @staticmethod
    def identity[T: DType]() -> Scalar[T]:
        """No bit set: `0 ^ x` is `x`."""
        return Scalar[T](0)

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_integral(), "bit_wise_xor: integers only"
        return a ^ b


struct BitNotKernel(NumericUnaryKernel):
    comptime name = "bit_wise_not"
    comptime accepts = family[IntegerType]

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        comptime assert T.is_integral(), "bit_wise_not: integers only"
        return ~a


@always_inline
def _shifts[T: DType, W: Int](amount: SIMD[T, W]) -> SIMD[DType.bool, W]:
    """The lanes whose shift amount is in `[0, precision)`: the bit width,
    less the sign bit for a signed type."""
    comptime precision = bit_width_of[T]() - Int(T.is_signed())
    return amount.ge(0) & amount.lt(SIMD[T, W](precision))


struct ShiftLeftKernel(NumericBinaryKernel):
    comptime name = "shift_left"
    comptime accepts = family[IntegerType]

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """`a << b`, wrapping, and `a` where `b` is out of range."""
        comptime assert T.is_integral(), "shift_left: integers only"
        var shifts = _shifts(b)
        # The amount is zeroed where it is out of range, so no lane shifts
        # by its bit width or more.
        return shifts.select(a << shifts.select(b, SIMD[T, W](0)), a)


struct ShiftRightKernel(NumericBinaryKernel):
    comptime name = "shift_right"
    comptime accepts = family[IntegerType]

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]:
        """`a >> b`, arithmetic for a signed type, and `a` where `b` is out of
        range."""
        comptime assert T.is_integral(), "shift_right: integers only"
        var shifts = _shifts(b)
        return shifts.select(a >> shifts.select(b, SIMD[T, W](0)), a)
