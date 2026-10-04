# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Zstandard frames, and `Zstd` -- the codec Parquet's ZSTD pages and Arrow
IPC's ZSTD buffers are compressed with.

A frame is a magic number, a header -- the window or a single-segment flag,
an optional dictionary id and content size, an optional checksum flag --
then blocks of at most 128 KiB, each raw, a repeated byte, or compressed,
and the low 32 bits of the content's XXH64 if the flag is set. Frames may be
concatenated, with skippable frames among them.
"""

from ...errors import CorruptError, DynError
from ..byteorder import LittleEndian
from ..hashing import XxHash64
from .block import BlockDecoder
from .encoder import BlockEncoder
from .headers import BLOCK_MAX, BlockHeader, FrameHeader


struct Zstd:
    """Zstandard compression, in Mojo."""

    comptime MAGIC = UInt32(0xFD2FB528)
    comptime _SKIPPABLE = UInt32(0x184D2A50)
    """The first of the 16 skippable frames' magic numbers."""

    @staticmethod
    def max_compressed_length(n: Int) -> Int:
        """The most `compress` can produce for `n` bytes: the magic number,
        the longest header -- a descriptor, a window and an 8-byte size --
        and each block stored raw behind its 3-byte header."""
        var blocks = max(1, (n + BLOCK_MAX - 1) // BLOCK_MAX)
        return 4 + 1 + 1 + 8 + 3 * blocks + n

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """The most `n` bytes of frames decode to: the densest block, a
        header and one byte to repeat, holds `BLOCK_MAX` bytes -- what a
        reader checks a length it was given against, before allocating
        it."""
        return n * (BLOCK_MAX // (BlockHeader.SIZE + 1))

    @staticmethod
    def compress(src: Span[mut=False, UInt8, _], mut dst: List[UInt8]) raises:
        """Append one frame holding `src` to `dst`, with the content size
        and without a checksum, as libzstd's one-shot `ZSTD_compress` writes
        it at level 1 -- the same bytes, with libzstd 1.5.7."""
        var n = len(src)
        # At least doubling, so frames appended one after another to one
        # list -- an IPC body's buffers -- do not copy it once per frame;
        # and room for the scratch a block's sections write before they are
        # cut to size, up to 3 bytes an input byte where the frame keeps 1:
        # the sequences bitstream is sized for 12 bytes a 4-byte match.
        var need = (
            len(dst)
            + Self.max_compressed_length(n)
            + 2 * min(n, BLOCK_MAX)
            + 512
        )
        if dst.capacity() < need:
            dst.reserve(max(need, 2 * dst.capacity()))
        LittleEndian.append[DType.uint32](dst, Self.MAGIC)
        var blocks = BlockEncoder(n)
        var header = FrameHeader.describing(n, blocks.window())
        header.write(dst)
        var block_max = header.block_max()
        var pos = 0
        # What the blocks so far have saved, headers included: a full block
        # may be split only once that is 3 bytes, so splitting cannot grow
        # incompressible input -- `ZSTD_optimalBlockSize`.
        var savings = 0
        # At least one block, so an empty frame holds an empty raw one.
        while True:
            var size = min(block_max, n - pos)
            if size == BLOCK_MAX and savings >= 3:
                size = BlockEncoder.split(src[pos : pos + size])
            var at = len(dst)
            blocks.encode(src, pos, pos + size, pos + size == n, dst)
            savings += size - (len(dst) - at)
            pos += size
            if pos == n:
                break

    @staticmethod
    def decompress_into[
        o: MutOrigin
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        """Decompress the frames in `src` into exactly `dst`, whose length must
        be the decompressed one. Nothing past `len(dst)` is written; on corrupt
        input, what was decoded before the corruption is left in `dst`.
        Frames that need a dictionary raise `NotImplementedError`. No input
        is no frames, and decodes to nothing, as libzstd's `ZSTD_decompress`
        decodes it."""
        # The length check at the end catches this too, but without the
        # early exit the sequence decoding inlined below is laid out 13-25%
        # slower on text and integers (`codec_ab`): a layout effect, as the
        # Huffman loops' `@no_inline` is.
        if len(src) == 0 and len(dst) > 0:
            raise CorruptError(t"zstd: no frames for {len(dst)} bytes")
        var blocks = BlockDecoder(len(dst))
        var ip = 0
        var op = 0
        while ip < len(src):
            if len(src) - ip < 4:
                raise CorruptError(t"zstd: {len(src) - ip} bytes after a frame")
            var magic = LittleEndian.fixed[DType.uint32](src, ip)
            if magic & 0xFFFFFFF0 == Self._SKIPPABLE:
                if len(src) - ip < 8:
                    raise CorruptError("zstd: truncated skippable frame")
                var size = Int(LittleEndian.fixed[DType.uint32](src, ip + 4))
                if size > len(src) - ip - 8:
                    raise CorruptError("zstd: truncated skippable frame")
                ip += 8 + size
            elif magic == Self.MAGIC:
                ip, op = Self._frame(blocks, src, ip + 4, dst, op)
            else:
                raise CorruptError(t"zstd: bad frame magic {hex(magic)}")
        if op != len(dst):
            raise CorruptError(
                t"zstd: frames decode to {op} bytes, not the expected"
                t" {len(dst)}"
            )

    @staticmethod
    def _frame[
        o: MutOrigin
    ](
        mut blocks: BlockDecoder,
        src: Span[UInt8, _],
        var ip: Int,
        dst: Span[UInt8, o],
        var op: Int,
    ) raises DynError -> Tuple[Int, Int]:
        """Decode the frame whose header is at `ip` into `dst` at `op`;
        return where the input and output stand after it."""
        var header = FrameHeader.read(src, ip)
        if header.content > len(dst) - op:
            raise CorruptError(
                t"zstd: a frame of {header.content} bytes overflows the output"
            )
        var block_max = header.block_max()
        var start = op
        blocks.reset()
        while True:
            var block = BlockHeader.read(src, ip)
            ip += BlockHeader.SIZE
            if block.size > block_max:
                raise CorruptError(
                    t"zstd: a {block.size}-byte block in a frame of {block_max}"
                )
            var payload = block.payload()
            if payload > len(src) - ip:
                raise CorruptError("zstd: truncated block")
            var before = op
            op = blocks.decode(block, src[ip : ip + payload], dst, op, start)
            if op - before > block_max:
                raise CorruptError("zstd: a block decodes past its maximum")
            ip += payload
            if block.last:
                break
        if header.content >= 0 and op - start != header.content:
            raise CorruptError(
                t"zstd: a frame of {header.content} bytes decodes to"
                t" {op - start}"
            )
        if header.checksum:
            if len(src) - ip < 4:
                raise CorruptError("zstd: truncated frame checksum")
            var want = LittleEndian.fixed[DType.uint32](src, ip)
            if UInt32(XxHash64.hash(dst[start:op]) & 0xFFFFFFFF) != want:
                raise CorruptError("zstd: frame checksum mismatch")
            ip += 4
        return (ip, op)
