# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""DELTA_BINARY_PACKED: signed integers as blocks of bit-packed differences
from each block's smallest difference -- Parquet's integer delta encoding."""

from std.memory import bitcast

from ..errors import CorruptError
from .bitpack import BitPack
from .bits import Bits
from .byteorder import Leb128
from .core import Decoder, Emitter, Codec, Source
from .zigzag import Zigzag


@fieldwise_init
struct DeltaBinaryPacked(Codec):
    """A header -- the block size, the miniblocks per block, the value count
    and the first value -- then blocks, each its smallest difference, a
    width byte per miniblock, and the miniblocks: each difference less the
    smallest, bit-packed. Differences wrap in the column's type.

    Unlike `Delta(BitPack)`, a width per 32 values rather than one for the
    column, so one outlier widens 32 values and not all of them. The header
    and the smallest differences are zigzag ULEB128s."""

    @staticmethod
    def takes[T: DType]() -> Bool:
        return T.is_integral() and T.is_signed()

    @staticmethod
    def encode[T: DType, S: Source](var src: S, mut out: List[UInt8]) raises:
        var block = src.collect[T]()
        var values = Span(block)
        Self.encode_blocks(values, out)

    @staticmethod
    def decode[
        T: DType, E: Emitter
    ](mut src: Decoder[_], count: Int, var out: E) raises -> E:
        var values = List[Scalar[T]](capacity=count)
        src.pos = Self.decode_blocks(src.data, src.pos, count, values)
        out.reserve(count)
        for v in values:
            out.emit(v)
        return out^

    # --- kernels ---------------------------------------------------------

    @staticmethod
    def encode_blocks[
        T: DType
    ](values: Span[Scalar[T], _], mut out: List[UInt8]):
        """Append `values` in blocks of 128 and miniblocks of 32. A short
        last block's unused miniblocks get width 0 and no bytes."""
        comptime BLOCK = 128
        comptime MINIBLOCKS = 4
        comptime PER = BLOCK // MINIBLOCKS
        comptime U = Bits.unsigned[T]
        var n = len(values)
        Leb128.write(out, UInt64(BLOCK))
        Leb128.write(out, UInt64(MINIBLOCKS))
        Leb128.write(out, UInt64(n))
        Leb128.write(
            out, Zigzag.encode_value(values[0] if n > 0 else Scalar[T](0))
        )
        var rel = List[Scalar[U]](length=BLOCK, fill=0)
        var i = 1
        while i < n:
            var end = min(i + BLOCK, n)
            var smallest = values[i] - values[i - 1]
            for k in range(i + 1, end):
                smallest = min(smallest, values[k] - values[k - 1])
            Leb128.write(out, Zigzag.encode_value(smallest))
            for k in range(BLOCK):
                rel[k] = 0
            for k in range(i, end):
                rel[k - i] = bitcast[U](values[k] - values[k - 1] - smallest)
            var widths = Array[Int, MINIBLOCKS](fill=0)
            for m in range(MINIBLOCKS):
                var top = Scalar[U](0)
                for k in range(m * PER, (m + 1) * PER):
                    top = max(top, rel[k])
                widths[m] = Bits.width(UInt64(top))
                out.append(UInt8(widths[m]))
            for m in range(MINIBLOCKS):
                # PER * width is a multiple of 8, so a miniblock ends on a byte.
                BitPack.pack(Span(rel)[m * PER : (m + 1) * PER], widths[m], out)
            i = end

    @staticmethod
    def decode_blocks[
        T: DType
    ](
        data: Span[UInt8, _], start: Int, count: Int, mut out: List[Scalar[T]]
    ) raises -> Int:
        """Append the first `count` values of the blocks at `data[start]`;
        return the position after the last miniblock read, where whatever
        follows the stream begins."""
        var pos = start
        var block_size: UInt64
        block_size, pos = Leb128.read(data, pos)
        var miniblocks: UInt64
        miniblocks, pos = Leb128.read(data, pos)
        var total: UInt64
        total, pos = Leb128.read(data, pos)
        var first: UInt64
        first, pos = Leb128.read(data, pos)
        if UInt64(count) > total:
            raise CorruptError(
                t"codecs: {count} values from a stream of {total}"
            )
        if (
            miniblocks == 0
            or miniblocks > block_size
            or block_size > UInt64(1 << 31)
        ):
            raise CorruptError(
                t"codecs: {miniblocks} miniblocks in a block of {block_size}"
            )
        var num_miniblocks = Int(miniblocks)
        var per = Int(block_size) // num_miniblocks
        if count == 0:
            return pos
        var value = Zigzag.decode_value[T](first)
        out.append(value)
        var produced = 1
        while produced < count:
            var smallest_z: UInt64
            smallest_z, pos = Leb128.read(data, pos)
            var smallest = Zigzag.decode_value[T](smallest_z)
            if pos + num_miniblocks > len(data):
                raise CorruptError("codecs: truncated miniblock widths")
            var widths_at = pos
            pos += num_miniblocks
            for m in range(num_miniblocks):
                var w = Int(data[widths_at + m])
                if w > 64:
                    raise CorruptError(t"codecs: miniblock width {w}")
                var take = min(per, count - produced)
                if pos + (take * w + 7) // 8 > len(data):
                    raise CorruptError("codecs: truncated miniblock")
                var first_new = len(out)
                BitPack.unpack(data, pos, w, take, out)
                # A miniblock is stored whole, its padding included.
                pos += per * w // 8
                for k in range(first_new, len(out)):
                    value += smallest + out[k]
                    out[k] = value
                produced += take
                if produced == count:
                    return pos
        return pos
