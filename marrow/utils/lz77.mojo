# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# The copies are ported from google/snappy 1.2.2 (`snappy.cc`), Copyright
# 2005-2011 Google Inc. (BSD 3-Clause), and `match_length` from libzstd
# 1.5.7's `ZSTD_count`, Copyright (c) Meta Platforms, Inc. (BSD 3-Clause);
# the licences are reproduced in NOTICE.txt.

"""What the Snappy, LZ4 and Zstandard codecs share: a decoder's copies, and
counting how far a match runs.

`LzCopy` is the decoder side: whole 32-byte blocks for a copy of at most 64
bytes, a shuffle-built pattern for an overlapping copy, and an exact byte
loop for the end of the output. The fast ones write up to 63 bytes past the
copy -- a caller keeps `LzCopy.SLOP` free -- and read a source that may
overlap the destination, which neither `unsafe_memcpy` nor `memmove` allows.
`match_length` is the compressors' match extension, LZ4's and Zstandard's.
Each codec's match finder is its own library's, in its own module.

The names here are public so the codecs can call them, but the module is not
re-exported from `marrow.utils`: it is the codecs' shared implementation, not
an API.
"""

from std.bit import count_trailing_zeros
from std.builtin.globals import global_constant

from ..views import BufferView
from .byteorder import LittleEndian


struct LzCopy:
    """A decoder's copies: fast ones that write up to 63 bytes past the copy
    and may read a source overlapping it, and an exact one for the end of the
    output."""

    comptime SLOP = 64
    """The room a fast copy needs past its end, which it may write: a
    decoder takes one only that far from the end of its output."""

    comptime PATTERN_MASKS = Self._pattern_masks(0)
    comptime RESHUFFLE_MASKS = Self._pattern_masks(16)

    @staticmethod
    def _pattern_masks(start: Int) -> Array[SIMD[DType.uint8, 16], 16]:
        """Row `p - 1`, lane `i`: `(start + i) % p`. Shuffling the first `p`
        output bytes by row `p - 1` of the `start = 0` table repeats them
        across 16 lanes; shuffling *that* by the `start = 16` table rotates it
        to the next 16 bytes of the same repetition."""
        var t = Array[SIMD[DType.uint8, 16], 16](fill=0)
        for p in range(1, 17):
            for i in range(16):
                t[p - 1][i] = UInt8((start + i) % p)
        return t^

    @staticmethod
    @always_inline
    def blocks64(
        src: BufferView[DType.uint8, _],
        dst: BufferView[mut=True, DType.uint8, _],
        length: Int,
    ):
        """`length <= 64` bytes from the start of `src` to the start of `dst`,
        as one or two whole 32-byte blocks: up to 63 bytes past `length` are
        written, and each block is loaded before it is stored, so a source that
        overlaps the destination past `length` reads what was there. Neither
        is `unsafe_memcpy`'s contract."""
        dst.store[32](0, src.load[32](0))
        if length > 32:
            dst.store[32](32, src.load[32](32))

    @staticmethod
    @always_inline
    def match_long(
        buf: BufferView[mut=True, DType.uint8, _],
        op: Int,
        offset: Int,
        length: Int,
    ):
        """`length` bytes from `offset` back to `op`, repeating where they
        overlap, in whole blocks: up to 63 bytes past `op + length` are
        written. Below 16 the pattern is rebuilt every 64 bytes from what the
        last round wrote; from 16 every 16- or 32-byte source block lies
        wholly behind the block it fills."""
        var i = 0
        if offset < 16:
            while i < length:
                Self.pattern64(buf, op + i, offset)
                i += 64
        else:
            var from_ = buf.slice(op - offset)
            var to = buf.slice(op)
            if offset < 32:
                while i < length:
                    to.store[16](i, from_.load[16](i))
                    i += 16
            else:
                while i < length:
                    to.store[32](i, from_.load[32](i))
                    i += 32

    @staticmethod
    @always_inline
    def pattern64(
        dst: BufferView[mut=True, DType.uint8, _], op: Int, offset: Int
    ):
        """64 bytes at `op` repeating the `offset` (1..64) bytes before it --
        libsnappy's `Copy64BytesWithPatternExtension`. Up to 16, the pattern
        is built in a register with one shuffle and rotated with one more per
        16 bytes, so no store feeds a later load."""
        # Two slices, not one view indexed at `op - offset + 16 * i`: the
        # single view measured 5-9% slower decoding, the slices as fast as raw
        # pointers.
        var from_ = dst.slice(op - offset)
        var to = dst.slice(op)
        if offset <= 16:
            var gen = global_constant[Self.PATTERN_MASKS]().unsafe_get(
                offset - 1
            )
            var rot = global_constant[Self.RESHUFFLE_MASKS]().unsafe_get(
                offset - 1
            )
            # `_dynamic_shuffle` is private to the stdlib, and used here on
            # purpose: it is one NEON `tbl` or SSSE3 `pshufb`, measured equal
            # to writing the intrinsics out.
            var pattern = from_.load[16](0)._dynamic_shuffle(gen)
            comptime for i in range(4):
                to.store[16](16 * i, pattern)
                pattern = pattern._dynamic_shuffle(rot)
        else:
            comptime for i in range(4):
                to.store[16](16 * i, from_.load[16](16 * i))

    @staticmethod
    @always_inline
    def copy_exact(
        dst: Span[mut=True, UInt8, _], op: Int, offset: Int, length: Int
    ):
        """`length` bytes from `offset` back to `op`, writing nothing past
        the end of `dst`: as `match_long` while that stays `SLOP` clear of
        the end, then byte by byte forward, so where the regions overlap the
        bytes just written are read again and the pattern repeats -- not
        `memmove`'s semantics. Bytes past `op + length` may be written, with
        what the match would continue as. Only the last 64 go byte by byte:
        a match can run to the end of the output -- a constant page is one
        -- and would otherwise copy at a byte a store-forwarding stall.

        Inlined, with no call inside: it sits in the zstd sequence loop's
        rarely taken branch, and a call there, however rare, keeps values in
        callee-saved registers across it -- the loop then spilled more and
        decoded text 10-20% slower."""
        var bulk = max(0, min(length, len(dst) - Self.SLOP - op))
        if bulk > 0:
            Self.match_long(BufferView(dst), op, offset, bulk)
        for i in range(op + bulk, op + length):
            dst[i] = dst[i - offset]


@always_inline
def match_length(src: Span[UInt8, _], var a: Int, var b: Int, end: Int) -> Int:
    """How many bytes from `a` equal those from `b`, `a` stopping at `end`
    -- `ZSTD_count`, and `LZ4_count`."""
    var start = a
    while a + 8 <= end:
        var diff = LittleEndian.fixed[DType.uint64](
            src, a
        ) ^ LittleEndian.fixed[DType.uint64](src, b)
        if diff != 0:
            return a - start + Int(count_trailing_zeros(diff)) // 8
        a += 8
        b += 8
    while a < end and src.unsafe_get(a) == src.unsafe_get(b):
        a += 1
        b += 1
    return a - start
