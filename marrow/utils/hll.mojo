# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`HyperLogLog` — approximate distinct counts over 64-bit hashes."""

import std.math as math
from std.bit import count_leading_zeros

from ..errors import InvalidError


struct HyperLogLog(Movable, Sized):
    """A row of HyperLogLog sketches, `2**precision` one-byte registers each,
    stored back to back so one allocation serves every group.

    A hash's top `precision` bits pick a register, and the leading-zero run of
    the bits below them is kept as that register's maximum. The estimate is
    the harmonic mean with the standard bias constant, falling back to linear
    counting while many registers are still empty; 64-bit hashes make the
    large-range correction unnecessary.
    """

    var _p: Int
    var _registers: List[UInt8]

    def __init__(out self, precision: Int, sketches: Int = 0):
        self._p = precision
        self._registers = List[UInt8](length=sketches << precision, fill=0)

    def registers(self) -> Int:
        """Registers per sketch: one sketch's width in bytes."""
        return 1 << self._p

    def __len__(self) -> Int:
        """The number of sketches."""
        return len(self._registers) >> self._p

    def grow(mut self, sketches: Int):
        """Make room for at least `sketches` sketches; new ones are empty."""
        if sketches > len(self):
            self._registers.resize(sketches << self._p, 0)

    @always_inline
    def add(mut self, sketch: Int, h: UInt64):
        """Fold hash `h` into sketch `sketch`."""
        var p = self._p
        var idx = (sketch << p) + Int(h >> UInt64(64 - p))
        # 1 + the leading-zero run below the top `p` bits; the sentinel bit
        # caps it at `64 - p + 1`.
        var w = (h << UInt64(p)) | (UInt64(1) << UInt64(p - 1))
        var rho = UInt8(count_leading_zeros(w) + 1)
        if rho > self._registers[idx]:
            self._registers[idx] = rho

    def sketch(self, sketch: Int) -> Span[UInt8, origin_of(self._registers)]:
        """Sketch `sketch`'s registers."""
        return Span(self._registers)[
            sketch << self._p : (sketch + 1) << self._p
        ]

    def merge(mut self, sketch: Int, registers: Span[UInt8, _]) raises:
        """Fold `registers`, a sketch of the same precision, into sketch
        `sketch`: each register keeps the larger, which counts the union."""
        var m = self.registers()
        if len(registers) != m:
            raise InvalidError(
                t"HyperLogLog: a sketch of {len(registers)} registers merged"
                t" into one of {m}, another precision"
            )
        var base = sketch * m
        for r in range(m):
            if registers[r] > self._registers[base + r]:
                self._registers[base + r] = registers[r]

    def estimate(self, sketch: Int) -> Int64:
        """The cardinality estimate of sketch `sketch`."""
        var m = self.registers()
        var base = sketch * m
        var inv_sum = Float64(0)
        var zeros = 0
        for j in range(m):
            var r = Int(self._registers[base + j])
            inv_sum += Float64(1) / Float64(UInt64(1) << UInt64(r))
            if r == 0:
                zeros += 1
        var alpha = 0.7213 / (1 + 1.079 / Float64(m))
        var estimate = alpha * Float64(m) * Float64(m) / inv_sum
        if estimate <= 2.5 * Float64(m) and zeros > 0:
            estimate = Float64(m) * math.log(Float64(m) / Float64(zeros))
        return Int64(Int(estimate + 0.5))
