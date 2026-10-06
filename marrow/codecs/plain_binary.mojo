# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Plain byte strings: each value's length, then its bytes."""

from ..errors import CorruptError
from .binary import BinaryCodec
from .byteorder import LittleEndian


@fieldwise_init
struct PlainBinary(BinaryCodec):
    """Each value as a 4-byte little-endian length, then its bytes --
    Parquet's PLAIN for BYTE_ARRAY, its dictionary pages' too."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        for i in range(len(offsets) - 1):
            Self.put(out, data[Int(offsets[i]) : Int(offsets[i + 1])])

    @staticmethod
    @always_inline
    def put(mut out: List[UInt8], value: Span[UInt8, _]):
        """Append one value."""
        LittleEndian.append[DType.uint32](out, UInt32(len(value)))
        out.extend(value)

    @staticmethod
    def decode(
        src: Span[UInt8, _],
        pos: Int,
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises -> Int:
        var p = pos
        for _ in range(count):
            if p + 4 > len(src):
                raise CorruptError("codecs: truncated byte string length")
            var n = Int(LittleEndian.fixed[DType.uint32](src, p))
            p += 4
            if n > len(src) - p:
                raise CorruptError(
                    t"codecs: byte string of {n} bytes truncated"
                )
            data.extend(src[p : p + n])
            offsets.append(Int32(len(data)))
            p += n
        return p
