# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Delta-length byte strings: the lengths, delta-packed, then the bytes."""

from ..errors import CorruptError
from .binary import BinaryCodec
from .delta_binary_packed import DeltaBinaryPacked


@fieldwise_init
struct DeltaLengthByteArray(BinaryCodec):
    """Every length as one `DeltaBinaryPacked` stream, then every value's
    bytes end to end -- Parquet's DELTA_LENGTH_BYTE_ARRAY. Lengths of similar
    values become small differences, and the bytes are left whole for a
    compressor."""

    @staticmethod
    def encode(
        offsets: Span[Int32, _], data: Span[UInt8, _], mut out: List[UInt8]
    ) raises:
        var lengths = List[Int32](capacity=len(offsets))
        for i in range(len(offsets) - 1):
            lengths.append(offsets[i + 1] - offsets[i])
        DeltaBinaryPacked.encode_blocks(Span(lengths), out)
        out.extend(data[Int(offsets[0]) : Int(offsets[len(offsets) - 1])])

    @staticmethod
    def decode(
        src: Span[UInt8, _],
        pos: Int,
        count: Int,
        mut offsets: List[Int32],
        mut data: List[UInt8],
    ) raises -> Int:
        var lengths = List[Int32](capacity=count)
        var p = DeltaBinaryPacked.decode_blocks(src, pos, count, lengths)
        for n in lengths:
            if n < 0 or Int(n) > len(src) - p:
                raise CorruptError(
                    t"codecs: byte string of {n} bytes truncated"
                )
            data.extend(src[p : p + Int(n)])
            offsets.append(Int32(len(data)))
            p += Int(n)
        return p
