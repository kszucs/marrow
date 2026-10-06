# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Bit packing in registers: no memory touched, no allocation, nothing
raised -- so a GPU kernel can call it as well as the CPU."""

from std.math import iota
from std.memory import bitcast
from std.sys import bit_width_of, size_of


struct Bits:
    """The register-only bit-packing kernels Arrow bitmaps are read and
    written with, and `BitPack` builds on -- and a value's bits, which every
    codec compares and transforms values by."""

    comptime unsigned[T: DType] = DType.uint8 if size_of[
        Scalar[T]
    ]() == 1 else (
        DType.uint16 if size_of[Scalar[T]]()
        == 2 else (DType.uint32 if size_of[Scalar[T]]() == 4 else DType.uint64)
    )
    """The unsigned integer type as wide as `T`."""

    @staticmethod
    @always_inline
    def of[T: DType](v: Scalar[T]) -> UInt64:
        """`v`'s bits, zero-extended."""
        return UInt64(bitcast[Self.unsigned[T]](v))

    @staticmethod
    @always_inline
    def to[T: DType](b: UInt64) -> Scalar[T]:
        """The `T` whose bits are the low bits of `b`."""
        return bitcast[T](b.cast[Self.unsigned[T]]())

    @staticmethod
    def _packed[W: Int]() -> DType:
        """The unsigned integer type of `W` bits, for `W` of 8, 16, 32 or
        64."""
        comptime assert (
            W == 8 or W == 16 or W == 32 or W == 64
        ), "W must be 8, 16, 32 or 64"
        if W == 8:
            return DType.uint8
        elif W == 16:
            return DType.uint16
        elif W == 32:
            return DType.uint32
        return DType.uint64

    @staticmethod
    def width(max_value: UInt64) -> Int:
        """The bits needed to represent every value in `[0, max_value]`."""
        var w = 0
        while w < 64 and (max_value >> UInt64(w)) > 0:
            w += 1
        return w

    @staticmethod
    @always_inline
    def pack_bools[
        W: Int
    ](mask: SIMD[DType.bool, W]) -> SIMD[DType.uint8, W // 8]:
        """`W` bools as the `W / 8` bytes holding them, lane `i` at bit `i`
        -- width-1 packing in registers, with no memory touched.

        Each lane is cast to a `W`-bit integer, shifted by its index and the
        lanes OR-reduced. `std.memory.pack_bits` -- one bitcast from
        `<W x i1>`, which x86 lowers to `pmovmskb` -- is slower on ARM, which
        has no mask-move instruction: it measured +12-15% on every bitmap-pack
        benchmark there, with none faster. Re-measure on the target before
        swapping it in."""
        comptime T = Self._packed[W]()
        var word = (mask.cast[T]() << iota[T, W]()).reduce_or()
        return bitcast[DType.uint8, W // 8](word)

    @staticmethod
    @always_inline
    def unpack_lanes[
        D: DType, W: Int
    ](word: Scalar[D], width: Int) -> SIMD[D, W]:
        """The `W` values of `width` bits packed from bit 0 of `word`, lane
        `j` from bit `j * width` -- unpacking in registers, with no memory
        touched. `W * width` must not exceed `word`'s bits."""
        var mask = (Scalar[D](1) << Scalar[D](width)) - 1
        return (SIMD[D, W](word) >> (iota[D, W]() * Scalar[D](width))) & mask

    @staticmethod
    @always_inline
    def unpack_lanes[
        D: DType, W: Int, width: Int
    ](word: Scalar[D]) -> SIMD[D, W]:
        """`unpack_lanes` for a `width` known at compile time, which folds
        the shifts and the mask into constants."""
        comptime assert W * width <= bit_width_of[D](), "the lanes overflow D"
        var shifts = iota[D, W]()
        comptime if width != 1:
            shifts *= Scalar[D](width)
        return (SIMD[D, W](word) >> shifts) & Scalar[D]((1 << width) - 1)
