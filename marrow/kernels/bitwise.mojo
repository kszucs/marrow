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

The codec stages turn a value's bits into other bits of the same width, a
register at a time, each with its inverse: `ZigzagEncode` / `ZigzagDecode`,
signed integers as unsigned ones with small magnitudes kept small;
`OrderedEncode` / `OrderedDecode[To]`, numbers as the unsigned integers that
order as they do; and `ByteSwap`, the bytes in the other order, its own
inverse. `Compose` (`compose.mojo`) chains them before a packer and after an
unpacker.
"""

from std.bit import byte_swap
from std.sys import bit_width_of

from ..dtypes import IntegerType
from ..utils.integers import (
    Signed,
    Unsigned,
    ordered,
    unordered,
    unzigzag,
    zigzag,
)
from .core import Associative, Commutative, Identity, UnaryKernel, family
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


# --- codec stages ----------------------------------------------------------


@fieldwise_init
struct ZigzagEncode(Deinitable, Movable, UnaryKernel):
    """Signed integers as the unsigned integers of their width, small
    magnitudes staying small.

        [0, -1, 1, -2]  ->  [0, 1, 2, 3]
    """

    comptime name = "zigzag_encode"
    comptime Out[T: DType]: DType = Unsigned[T]

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        return zigzag(a)


@fieldwise_init
struct ZigzagDecode(Deinitable, Movable, UnaryKernel):
    """The signed integers `ZigzagEncode` made unsigned ones of.

        [0, 1, 2, 3]  ->  [0, -1, 1, -2]
    """

    comptime name = "zigzag_decode"
    comptime Out[T: DType]: DType = Signed[T]

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        comptime assert T.is_unsigned(), "zigzag_decode takes unsigned integers"
        return unzigzag[Signed[T]](a)


@fieldwise_init
struct OrderedEncode(Deinitable, Movable, UnaryKernel):
    """Numbers as the unsigned integers of their width that order as they
    do: a signed integer's sign bit flipped, a negative float's every bit.

        [-1, 0, 1] as int8  ->  [0x7f, 0x80, 0x81]
    """

    comptime name = "ordered_encode"
    comptime Out[T: DType]: DType = Unsigned[T]

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        return ordered(a)


@fieldwise_init
struct OrderedDecode[To: DType](Deinitable, Movable, UnaryKernel):
    """The numbers of dtype `To` that `OrderedEncode` made unsigned integers
    of: the bits alone do not say whether they were signed or a float.

        [0x7f, 0x80, 0x81], OrderedDecode[DType.int8]  ->  [-1, 0, 1]
    """

    comptime name = "ordered_decode"
    comptime Out[T: DType]: DType = Self.To

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[Self.Out[T], W]:
        comptime assert (
            T == Unsigned[Self.To]
        ), "ordered_decode takes the unsigned integers of its dtype's width"
        return unordered[Self.To](a)


@fieldwise_init
struct ByteSwap(Deinitable, Movable, UnaryKernel):
    """Each integer's bytes in the other order: big-endian values from
    little-endian ones, and back.

        [0x0102] as uint16  ->  [0x0201]
    """

    comptime name = "byte_swap"

    @always_inline
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W]) -> SIMD[T, W]:
        return byte_swap(a)
