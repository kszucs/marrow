# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Reflected CRC-32: the ISO-3309 / zlib / gzip checksum and CRC-32C."""


struct Crc[polynomial: UInt32](Copyable, Movable):
    """A reflected CRC-32 over `polynomial`, with all-ones on entry and exit --
    `Crc32` and `Crc32c` differ only in it. Incremental: `update` each byte span
    in order (Parquet v2 pages checksum the levels then the compressed values),
    then read `value`."""

    var _state: UInt32

    def __init__(out self):
        self._state = UInt32(0xFFFFFFFF)

    def update(mut self, data: Span[UInt8, _]):
        var crc = self._state
        for i in range(len(data)):
            crc = Self._shift[8](crc ^ UInt32(data[i]))
        self._state = crc

    def value(self) -> UInt32:
        return self._state ^ UInt32(0xFFFFFFFF)

    @staticmethod
    def compute(data: Span[UInt8, _]) -> UInt32:
        """The CRC of a single contiguous span."""
        var c = Self()
        c.update(data)
        return c.value()

    @staticmethod
    def step(crc: UInt32, data: UInt32) -> UInt32:
        """The CRC register after shifting in the 4 bytes of `data`, low byte
        first, with none of a whole-message CRC's inversions -- for CRC-32C,
        what ARM's `__crc32cw(crc, data)` and x86's `_mm_crc32_u32(crc, data)`
        compute. It depends only on `crc ^ data`.

        Portable, one bit at a time, rather than either instruction: marrow
        prefers code with no architecture-specific symbols, and nothing hot
        calls this."""
        return Self._shift[32](crc ^ data)

    @staticmethod
    @always_inline
    def _shift[bits: Int](var crc: UInt32) -> UInt32:
        """`bits` bits of the register shifted out, the polynomial folded in
        under each set one."""
        for _ in range(bits):
            crc = (crc >> 1) ^ (Self.polynomial & (UInt32(0) - (crc & 1)))
        return crc


comptime Crc32 = Crc[0xEDB88320]
"""The ISO-3309 / zlib / gzip checksum Parquet uses for its optional per-page
checksum."""

comptime Crc32c = Crc[0x82F63B78]
"""CRC-32C (Castagnoli), whose `step` Snappy's compressor can hash with."""
