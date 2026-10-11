# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Numbers as unsigned integers of their width, a register at a time:
`zigzag`, signed integers with small magnitudes kept small, and `ordered`,
numbers ordering as they do -- each with its inverse. The varints, the
codecs and the sort keys share them."""

from std.memory import bitcast
from std.sys import bit_width_of, size_of


comptime Unsigned[T: DType]: DType = DType.uint8 if size_of[
    Scalar[T]
]() == 1 else (
    DType.uint16 if size_of[Scalar[T]]()
    == 2 else (DType.uint32 if size_of[Scalar[T]]() == 4 else DType.uint64)
)
"""The unsigned integer type as wide as `T`: what a value is compared,
packed and transformed as, so a float or a signed integer goes through the
same code as its bits."""


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
