# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Numbers as integers of their width, a register at a time: `Unsigned[T]`
and `Signed[T]`, the integer types as wide as `T`; `zigzag`, signed integers
with small magnitudes kept small, and `ordered`, numbers ordering as they do
-- each with its inverse. The varints, the codecs and the sort keys share
them."""

from std.memory import bitcast
from std.sys import bit_width_of


def _none_as_wide[T: DType]() -> DType:
    """No integer type is as wide as `T`: refused when compiled."""
    comptime assert bit_width_of[T]() < 0, "no integer type is as wide"
    return T


# Comparisons, which the parser folds: a concrete call site sees
# `DType.uint32`, not an expression it must rebind. Every dtype maps by its
# width; one no integer is as wide as -- a 4-bit float -- is refused.
comptime Unsigned[T: DType]: DType = DType.uint8 if (
    T == DType.int8
    or T == DType.uint8
    or T == DType.bool
    or T == DType.float8_e3m4
    or T == DType.float8_e4m3fn
    or T == DType.float8_e4m3fnuz
    or T == DType.float8_e5m2
    or T == DType.float8_e5m2fnuz
    or T == DType.float8_e8m0fnu
) else (
    DType.uint16 if (
        T == DType.int16
        or T == DType.uint16
        or T == DType.float16
        or T == DType.bfloat16
    ) else (
        DType.uint32 if (
            T == DType.int32 or T == DType.uint32 or T == DType.float32
        ) else (
            DType.uint64 if (
                T == DType.int64
                or T == DType.uint64
                or T == DType.float64
                or T == DType.int
                or T == DType.uint
            ) else (
                DType.uint128 if (
                    T == DType.int128 or T == DType.uint128
                ) else (
                    DType.uint256 if (
                        T == DType.int256 or T == DType.uint256
                    ) else _none_as_wide[T]()
                )
            )
        )
    )
)
"""The unsigned integer type as wide as `T`: what a value is compared,
packed and transformed as, so a float or a signed integer goes through the
same code as its bits."""

comptime Signed[T: DType]: DType = DType.int8 if (
    Unsigned[T] == DType.uint8
) else (
    DType.int16 if (Unsigned[T] == DType.uint16) else (
        DType.int32 if (Unsigned[T] == DType.uint32) else (
            DType.int64 if (Unsigned[T] == DType.uint64) else (
                DType.int128 if (Unsigned[T] == DType.uint128) else (
                    DType.int256
                )
            )
        )
    )
)
"""The signed integer type as wide as `T`: what an unsigned integer is read
back as once `zigzag` is undone."""


@always_inline
def zigzag[T: DType](v: SIMD[T, _]) -> SIMD[Unsigned[T], type_of(v).length]:
    """Signed integers as unsigned ones, small magnitudes staying small:
    `[0, -1, 1, -2]` to `[0, 1, 2, 3]`."""
    comptime assert (
        T.is_signed() and T.is_integral()
    ), "zigzag takes signed integers"
    comptime top = Scalar[T](bit_width_of[T]() - 1)
    return bitcast[Unsigned[T], type_of(v).length]((v << 1) ^ (v >> top))


@always_inline
def unzigzag[T: DType](v: SIMD) -> SIMD[T, type_of(v).length]:
    """The signed integers `zigzag` made `v` of."""
    comptime assert type_of(v).dtype == Unsigned[T], "unzigzag takes zigzag's"
    comptime W = type_of(v).length
    return bitcast[T, W](v >> 1) ^ -bitcast[T, W](v & 1)


@always_inline
def ordered[T: DType](v: SIMD[T, _]) -> SIMD[Unsigned[T], type_of(v).length]:
    """Numbers as the unsigned integers of their width that order as they
    do: a signed integer's sign bit flipped, a negative float's every bit.
    `[-1, 0, 1]` as `int8` to `[0x7f, 0x80, 0x81]`."""
    comptime U = Unsigned[T]
    comptime W = type_of(v).length
    comptime msb = Scalar[U](bit_width_of[T]() - 1)
    comptime top = Scalar[U](1) << msb
    var bits = bitcast[U, W](v)
    comptime if T.is_floating_point():
        bits ^= (SIMD[U, W](0) - (bits >> msb)) | top
    elif T.is_signed():
        bits ^= top
    return bits


@always_inline
def unordered[T: DType](v: SIMD) -> SIMD[T, type_of(v).length]:
    """The numbers `ordered` made `v` of."""
    comptime U = Unsigned[T]
    comptime assert type_of(v).dtype == U, "unordered takes ordered's"
    comptime W = type_of(v).length
    comptime msb = Scalar[U](bit_width_of[T]() - 1)
    comptime top = Scalar[U](1) << msb
    var key = rebind[SIMD[U, W]](v)
    comptime if T.is_floating_point():
        key ^= ((key >> msb) - 1) | top
    elif T.is_signed():
        key ^= top
    return bitcast[T, W](key)
