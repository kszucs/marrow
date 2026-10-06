# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Front-coded byte strings: each value as the length of the prefix it
shares with the one before, and the rest."""

from ..errors import CorruptError
from .binary import BinaryCodec
from .delta_binary_packed import DeltaBinaryPacked
from .delta_length_byte_array import DeltaLengthByteArray


@fieldwise_init
struct DeltaByteArray(BinaryCodec):
    """Every shared-prefix length as one `DeltaBinaryPacked` stream, then the
    suffixes as `DeltaLengthByteArray` -- Parquet's DELTA_BYTE_ARRAY. Sorted
    strings share long prefixes and store little else."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        var n = len(offsets) - 1
        var prefixes = List[Int32](capacity=n)
        var suffix_offsets: List[Int32] = [0]
        var suffixes = List[UInt8]()
        for i in range(n):
            var v = data[Int(offsets[i]) : Int(offsets[i + 1])]
            var p = 0
            if i > 0:
                var prev = data[Int(offsets[i - 1]) : Int(offsets[i])]
                var m = min(len(prev), len(v))
                while p < m and prev[p] == v[p]:
                    p += 1
            prefixes.append(Int32(p))
            suffixes.extend(v[p:])
            suffix_offsets.append(Int32(len(suffixes)))
        DeltaBinaryPacked.encode_blocks(Span(prefixes), out)
        DeltaLengthByteArray.encode(Span(suffix_offsets), Span(suffixes), out)

    @staticmethod
    def decode(
        src: Span[UInt8, _],
        pos: Int,
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises -> Int:
        var prefixes = List[Int32](capacity=count)
        var p = DeltaBinaryPacked.decode_blocks(src, pos, count, prefixes)
        var suffix_offsets: List[Int32] = [0]
        var suffixes = List[UInt8]()
        p = DeltaLengthByteArray.decode(src, p, count, suffix_offsets, suffixes)
        var prev_start = len(data)
        var prev_len = 0
        for i in range(count):
            var n = Int(prefixes[i])
            if n < 0 or n > prev_len:
                raise CorruptError(
                    t"codecs: prefix of {n} bytes from a value of {prev_len}"
                )
            var start = len(data)
            for k in range(n):
                data.append(data[prev_start + k])
            data.extend(
                Span(suffixes)[
                    Int(suffix_offsets[i]) : Int(suffix_offsets[i + 1])
                ]
            )
            offsets.append(Int32(len(data)))
            prev_start = start
            prev_len = len(data) - start
        return p
