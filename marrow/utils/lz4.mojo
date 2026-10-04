# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause

# `Lz4.decompress_hadoop_into` is ported from Arrow C++'s `Lz4HadoopCodec`
# (`cpp/src/arrow/util/compression_lz4.cc`), Apache-2.0, and the compressor
# from liblz4 1.10.0's `lz4.c` (`LZ4_compress_generic`) and `lz4frame.c`,
# Copyright (c) 2011-2023 Yann Collet (BSD 2-Clause); see NOTICE.txt.

"""LZ4, in Mojo: the block format Parquet's LZ4_RAW pages hold, the Hadoop
framing of it Parquet's deprecated LZ4 pages hold, and the frame format Arrow
IPC's LZ4_FRAME buffers hold.

A block is a run of sequences, each a token -- the literal length in its high
four bits, the match length less 4 in its low four, either extended by bytes
that add 255 until one does not -- then the literals, then a 2-byte offset
and the match length's extension. The last sequence is literals only, the
last 5 bytes are always literals, and no match starts within 12 bytes of the
end (`lz4_Block_format.md`).

**The compressor is liblz4's `LZ4_compress_default`**, and writes its
bytes: one table of positions over the whole input -- 4096 four-byte slots
hashed from 5 bytes, or under 64 KiB 8192 two-byte ones hashed from 4 -- a
64 KiB window, a search that steps further the longer it misses, matches
extended backwards, and after each match the table filled 2 bytes back and
the next position tried at once.

**The decoder is Snappy's two tiers.** A fast loop takes every sequence,
long ones included, while the input and output have room for whole-block
copies: 16-byte blocks for a short literal or match, 32-byte ones for a long
one, a shuffle-built pattern for an overlapping match. A checked path takes
the end of the block and every error.

**A frame** (`lz4_Frame_format.md`) is a 4-byte magic, a descriptor and its
XXH32-derived checksum byte, then blocks each prefixed by a 4-byte size --
its high bit marking one stored uncompressed -- an end mark, and optional
block and content checksums. `compress_frame` writes what Arrow C++ writes,
`LZ4F_compressFrame` with default preferences, byte for byte: no checksums,
one independent block up to 64 KiB, and above that 64 KiB blocks linked
through one table, so a match reaches into the block before; a block that
would not come out shorter than its input is stored as is.
`decompress_frame_into` reads what any writer does: block sizes 64 KiB..4 MiB,
independent or linked blocks -- one contiguous output makes a link to an earlier
block an ordinary offset -- both checksums and the content size. Like Arrow, it
takes exactly one frame; dictionaries and the legacy format are refused.

**A Hadoop frame** is a big-endian u32 decompressed size, a big-endian u32
compressed size and one block. `compress_hadoop` writes one per page, as Arrow
C++ does; `decompress_hadoop_into` reads a run of them, falling back to a
plain block as Arrow C++ does for pages from before Parquet C++ framed them.
"""

from std.bit import byte_swap

from ..errors import CorruptError, DynError, InvalidError, NotImplementedError
from ..views import BufferView
from .byteorder import LittleEndian
from .hashing import XxHash32
from .lz77 import LzCopy, match_length


# ---------------------------------------------------------------------------
# compressor
# ---------------------------------------------------------------------------


comptime _MF_LIMIT = 12
"""No match starts within this many bytes of a block's end."""
comptime _LAST_LITERALS = 5
"""The last bytes of a block, always literals."""
comptime _SMALL = (1 << 16) + _MF_LIMIT - 1
"""Inputs below this take two-byte table slots -- `LZ4_64Klimit`."""
comptime _HASH_LOG = 12
"""`LZ4_HASHLOG` at liblz4's default `LZ4_MEMORY_USAGE` of 14: a 16 KiB
table."""
comptime _SKIP_TRIGGER = 6
"""Misses after which the search step grows by one -- `LZ4_skipTrigger`."""
comptime _WINDOW = 65535
"""The farthest back a match reaches -- `LZ4_DISTANCE_MAX`."""


struct _Matcher[small: Bool](Movable):
    """liblz4's `LZ4_compress_generic` without a dictionary: the table of
    positions an input's blocks share, and the search over one block.
    `small` is liblz4's `byU16`, for an input under `_SMALL`: twice the
    slots, half as wide, hashed from 4 bytes rather than 5."""

    comptime Slot = DType.uint16 if Self.small else DType.uint32
    var _table: List[Scalar[Self.Slot]]

    def __init__(out self):
        self._table = List[Scalar[Self.Slot]](
            length=1 << (_HASH_LOG + Int(Self.small)), fill=0
        )

    @staticmethod
    @always_inline
    def _hash(src: Span[UInt8, _], pos: Int) -> Int:
        """`LZ4_hashPosition`: Knuth's multiply over 4 bytes, or 5 read as 8
        and shifted up -- `LZ4_hash4` and `LZ4_hash5`."""
        comptime if Self.small:
            var v = LittleEndian.fixed[DType.uint32](src, pos)
            return Int((v * 2654435761) >> UInt32(32 - (_HASH_LOG + 1)))
        else:
            var v = LittleEndian.fixed[DType.uint64](src, pos)
            return Int(((v << 24) * 889523592379) >> UInt64(64 - _HASH_LOG))

    @staticmethod
    @always_inline
    def _extend(
        buf: Span[mut=True, UInt8, _], var op: Int, var rest: Int
    ) -> Int:
        """A length past the token's 15, as 255s and a final byte below 255;
        return where it ends."""
        while rest >= 255:
            buf.unsafe_get(op) = 255
            op += 1
            rest -= 255
        buf.unsafe_get(op) = UInt8(rest)
        return op + 1

    def block[
        limited: Bool
    ](
        mut self,
        src: Span[mut=False, UInt8, _],
        start: Int,
        end: Int,
        mut dst: List[UInt8],
    ) -> Bool:
        """Append `src[start:end]` as one block, its matches reaching back
        into `src[:start]`, which earlier calls compressed. With `limited`,
        give up -- `False`, `dst` as it was -- at the first of liblz4's
        checks that finds `end - start - 1` bytes too few, the capacity
        `LZ4F` gives a block; what the table learnt stays."""
        var n = end - start
        var at = len(dst)
        # The worst case, and the literal copy's last 8-byte store.
        dst.resize(unsafe_uninit_length=at + Lz4.max_block_length(n) + 8)
        var out = Span(dst)
        var table = Span(self._table)
        var olimit = at + n - 1
        var op = at
        var anchor = start
        var mflimit = end - _MF_LIMIT + 1
        var match_limit = end - _LAST_LITERALS

        @always_inline
        def read32(pos: Int) {src} -> UInt32:
            return LittleEndian.fixed[DType.uint32](src, pos)

        if n >= _MF_LIMIT + 1:
            var ip = start
            table.unsafe_get(Self._hash(src, ip)) = Scalar[Self.Slot](ip)
            ip += 1
            var forward_h = Self._hash(src, ip)
            var tail = False
            while not tail:
                # Search forwards, the step growing every 64 misses.
                var forward_ip = ip
                var step = 1
                var misses = 1 << _SKIP_TRIGGER
                var prior = 0
                while True:
                    var h = forward_h
                    var current = forward_ip
                    var candidate = Int(table.unsafe_get(h))
                    ip = forward_ip
                    forward_ip += step
                    step = misses >> _SKIP_TRIGGER
                    misses += 1
                    if forward_ip > mflimit:
                        tail = True
                        break
                    prior = candidate
                    forward_h = Self._hash(src, forward_ip)
                    table.unsafe_get(h) = Scalar[Self.Slot](current)
                    comptime if not Self.small:
                        if candidate + _WINDOW < current:
                            continue
                    if read32(prior) == read32(ip):
                        break
                if tail:
                    break
                # Extend the match backwards over the pending literals.
                if prior > 0 and src.unsafe_get(ip - 1) == src.unsafe_get(
                    prior - 1
                ):
                    ip -= 1
                    prior -= 1
                    while (
                        ip > anchor
                        and prior > 0
                        and src.unsafe_get(ip - 1) == src.unsafe_get(prior - 1)
                    ):
                        ip -= 1
                        prior -= 1
                var lit = ip - anchor
                var token_at = op
                op += 1
                comptime if limited:
                    if (
                        op + lit + (2 + 1 + _LAST_LITERALS) + lit // 255
                        > olimit
                    ):
                        dst.shrink(at)
                        return False
                var token: Int
                if lit >= 15:
                    token = 15 << 4
                    op = Self._extend(out, op, lit - 15)
                else:
                    token = lit << 4
                # 8 bytes at a time: a match starts at least 12 bytes before
                # the block's end, so the reads stay inside it.
                var i = 0
                while i < lit:
                    LittleEndian.store[DType.uint64](
                        out,
                        op + i,
                        LittleEndian.fixed[DType.uint64](src, anchor + i),
                    )
                    i += 8
                op += lit
                # The match, and every one the next position starts at once.
                while True:
                    LittleEndian.store[DType.uint16](
                        out, op, UInt16(ip - prior)
                    )
                    op += 2
                    var extra = match_length(
                        src, ip + 4, prior + 4, match_limit
                    )
                    ip += extra + 4
                    comptime if limited:
                        if (
                            op + (1 + _LAST_LITERALS) + (extra + 240) // 255
                            > olimit
                        ):
                            dst.shrink(at)
                            return False
                    if extra >= 15:
                        token += 15
                        op = Self._extend(out, op, extra - 15)
                    else:
                        token += extra
                    out.unsafe_get(token_at) = UInt8(token)
                    anchor = ip
                    if ip >= mflimit:
                        tail = True
                        break
                    table.unsafe_get(Self._hash(src, ip - 2)) = Scalar[
                        Self.Slot
                    ](ip - 2)
                    var h = Self._hash(src, ip)
                    var candidate = Int(table.unsafe_get(h))
                    table.unsafe_get(h) = Scalar[Self.Slot](ip)
                    var near = True
                    comptime if not Self.small:
                        near = candidate + _WINDOW >= ip
                    if near and read32(candidate) == read32(ip):
                        prior = candidate
                        token_at = op
                        op += 1
                        token = 0
                        continue
                    ip += 1
                    forward_h = Self._hash(src, ip)
                    break
        var last = end - anchor
        comptime if limited:
            if op + last + 1 + (last + 255 - 15) // 255 > olimit:
                dst.shrink(at)
                return False
        if last >= 15:
            out.unsafe_get(op) = 15 << 4
            op = Self._extend(out, op + 1, last - 15)
        else:
            out.unsafe_get(op) = UInt8(last << 4)
            op += 1
        BufferView(out).slice(op).copy_from(BufferView(src).slice(anchor), last)
        dst.shrink(op + last)
        return True


# ---------------------------------------------------------------------------
# decompressor
# ---------------------------------------------------------------------------


struct _BlockDecoder[sm: Bool, //, so: Origin[mut=sm], do: MutOrigin](
    TrivialRegisterPassable
):
    """One block being decoded into `dst` from position `op`. A match may
    reach back to `floor` and no further: the block's own start, or an
    earlier block's when a frame links them."""

    comptime SLOP = 128
    """The room the fast loop keeps at the end of the output: a short
    sequence writes a 16-byte literal block and a 64-byte pattern at most,
    and a long literal or match is taken only when it leaves this much after
    it for its last whole block."""

    comptime IN_MARGIN = 48
    """The room the fast loop keeps at the end of the input: a short sequence
    reads the token, a 16-byte literal block and the offset, and a long
    literal is taken only when it leaves this much after it for its last
    32-byte block, the offset and the match length's extension bytes."""

    var _src: Span[UInt8, Self.so]
    var ip: Int
    var _dst: Span[UInt8, Self.do]
    var op: Int
    var _floor: Int

    def __init__(
        out self,
        src: Span[UInt8, Self.so],
        dst: Span[UInt8, Self.do],
        op: Int,
        floor: Int,
    ):
        self._src = src
        self.ip = 0
        self._dst = dst
        self.op = op
        self._floor = floor

    @always_inline
    def _fits(self) -> Bool:
        return (
            self.ip < len(self._src) - Self.IN_MARGIN
            and self.op <= len(self._dst) - Self.SLOP
        )

    @always_inline
    def _fast_sequence(mut self) -> Bool:
        """One sequence while the input and output have room for whole-block
        copies, unchecked in a release build -- long literals and matches
        included, as liblz4's fast loop takes them; `False`, with nothing
        consumed, for any other, which the checked path then takes."""
        var src = BufferView(self._src)
        var out = BufferView(self._dst)
        var in_limit = len(self._src) - Self.IN_MARGIN
        var out_limit = len(self._dst) - Self.SLOP
        var ip = self.ip
        var op = self.op
        var token = Int(src.unsafe_get(ip))
        ip += 1
        var lit = token >> 4
        if lit == 15:
            while True:
                if ip >= in_limit:
                    return False
                var b = Int(src.unsafe_get(ip))
                ip += 1
                lit += b
                if b != 255:
                    break
            if lit > in_limit - ip or lit > out_limit - op:
                return False
            var i = 0
            while i < lit:
                out.store[32](op + i, src.load[32](ip + i))
                i += 32
        else:
            # At most 14: one 16-byte block.
            out.store[16](op, src.load[16](ip))
        ip += lit
        op += lit
        var offset = Int(LittleEndian.fixed[DType.uint16](self._src, ip))
        ip += 2
        if offset == 0 or offset > op - self._floor:
            return False
        var ml = token & 15
        if ml == 15:
            while True:
                if ip >= in_limit:
                    return False
                var b = Int(src.unsafe_get(ip))
                ip += 1
                ml += b
                if b != 255:
                    break
            ml += 4
            if ml > out_limit - op:
                return False
            LzCopy.match_long(out, op, offset, ml)
        else:
            ml += 4
            if offset < ml:
                LzCopy.pattern64(out, op, offset)
            else:
                # 16-byte blocks, only as many as the match needs: a wider
                # store than that leaves the next sequence's match loading
                # across two stores, which cannot be forwarded -- 22% on
                # 8-byte integers, whose matches are 5 bytes at offset 8.
                out.store[16](op, out.load[16](op - offset))
                if ml > 16:
                    out.store[16](op + 16, out.load[16](op - offset + 16))
        self.ip = ip
        self.op = op + ml
        return True

    @staticmethod
    @always_inline
    def _extension(
        src: Span[UInt8, _], mut ip: Int, what: StaticString
    ) raises CorruptError -> Int:
        """The bytes extending a `what` length past the token's 15, at `ip`,
        which is moved past them: 255s, then one below 255, summed. Inlined:
        a long literal run is thousands of them, and called, the loop made
        decoding floats 10% slower."""
        var sum = 0
        while True:
            if ip >= len(src):
                raise CorruptError(t"lz4: truncated {what} length")
            var b = Int(src[ip])
            ip += 1
            sum += b
            if b != 255:
                break
        return sum

    def _checked_sequence(mut self) raises CorruptError -> Bool:
        """Decode the sequence at `ip`, checking every bound and writing
        nothing past the output; `True` if it was the block's last."""
        var src = self._src
        var n_in = len(src)
        var n_out = len(self._dst)
        var ip = self.ip
        var op = self.op
        if ip >= n_in:
            raise CorruptError("lz4: block ends without its final literals")
        var token = Int(src[ip])
        ip += 1
        var lit = token >> 4
        if lit == 15:
            lit += Self._extension(src, ip, "literal")
        if lit > n_in - ip:
            raise CorruptError(
                t"lz4: {lit}-byte literal at input {ip} runs past the"
                t" {n_in}-byte block"
            )
        if lit > n_out - op:
            raise CorruptError(
                t"lz4: {lit}-byte literal at output {op} overflows the"
                t" declared {n_out} bytes"
            )
        BufferView(self._dst).slice(op).copy_from(
            BufferView(src).slice(ip), lit
        )
        ip += lit
        op += lit
        if ip == n_in:
            self.ip = ip
            self.op = op
            return True
        if n_in - ip < 2:
            raise CorruptError("lz4: truncated match offset")
        var offset = Int(LittleEndian.fixed[DType.uint16](src, ip))
        ip += 2
        if offset == 0 or offset > op - self._floor:
            raise CorruptError(
                t"lz4: match offset {offset} at output {op} is outside the"
                t" output"
            )
        var ml = token & 15
        if ml == 15:
            ml += Self._extension(src, ip, "match")
        ml += 4
        if ml > n_out - op:
            raise CorruptError(
                t"lz4: {ml}-byte match at output {op} overflows the"
                t" declared {n_out} bytes"
            )
        if op + ml + LzCopy.SLOP <= n_out:
            LzCopy.match_long(BufferView(self._dst), op, offset, ml)
        else:
            LzCopy.copy_exact(self._dst, op, offset, ml)
        self.ip = ip
        self.op = op + ml
        return False

    def run(mut self) raises CorruptError:
        """Decode the whole block: the fast loop, and the checked path for
        each sequence it hands back, until the literals-only last one."""
        while True:
            var s = self  # in registers: `self` is a reference stores may alias
            while s._fits() and s._fast_sequence():
                pass
            self = s
            if self._checked_sequence():
                break


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


struct Lz4:
    """LZ4 compression: the block and frame formats."""

    comptime _FRAME_MAGIC = UInt32(0x184D2204)
    comptime _LEGACY_MAGIC = UInt32(0x184C2102)
    comptime _FRAME_BLOCK = 1 << 16
    """The block size `compress_frame` writes: LZ4F's default, 64 KiB."""

    comptime _MAX_INPUT = 0x7E000000
    """The largest block liblz4 compresses, its `LZ4_MAX_INPUT_SIZE`."""

    @staticmethod
    def max_block_length(n: Int) -> Int:
        """The most bytes `compress_block` can produce for `n` input bytes --
        liblz4's `LZ4_compressBound`: stored as one run of literals, with its
        length's extension bytes."""
        return n + n // 255 + 16

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """The most `n` bytes of a block, a frame or Hadoop frames decode
        to: under 256 times as many, as a length grows by at most 255 for
        each byte that extends it -- what a reader checks a length it was
        given against, before allocating it."""
        return 256 * n

    @staticmethod
    def compress_block(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises InvalidError:
        """Append the LZ4 block for `src` to `dst` -- `LZ4_compress_default`'s
        bytes."""
        var n = len(src)
        if n > Self._MAX_INPUT:
            raise InvalidError(
                t"lz4: {n} bytes is over the block format's {Self._MAX_INPUT}"
            )
        if n < _SMALL:
            var matcher = _Matcher[small=True]()
            _ = matcher.block[limited=False](src, 0, n, dst)
        else:
            var matcher = _Matcher[small=False]()
            _ = matcher.block[limited=False](src, 0, n, dst)

    @staticmethod
    def decompress_block_into[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises CorruptError:
        """Decompress the block `src` into exactly `dst`, whose length must be
        the decompressed one -- liblz4's `LZ4_decompress_safe` with an exact
        capacity. Nothing past `len(dst)` is written; on a corrupt block, what
        was decoded before the corruption is left in `dst`."""
        if len(src) == 0:
            raise CorruptError("lz4: empty block")
        var decoder = _BlockDecoder(src, dst, 0, 0)
        decoder.run()
        if decoder.op != len(dst):
            raise CorruptError(
                t"lz4: block decodes to {decoder.op} bytes, not the expected"
                t" {len(dst)}"
            )

    @staticmethod
    def compress_hadoop(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises InvalidError:
        """Append `src` to `dst` as one Hadoop frame: a big-endian u32
        decompressed size, a big-endian u32 compressed size, then the block --
        what Arrow C++ writes for Parquet's `LZ4` codec."""
        var at = len(dst)
        LittleEndian.append[DType.uint64](dst, 0)  # both sizes, below
        Self.compress_block(src, dst)
        var size = len(dst) - at - 8
        LittleEndian.write[DType.uint32](dst, at, byte_swap(UInt32(len(src))))
        LittleEndian.write[DType.uint32](dst, at + 4, byte_swap(UInt32(size)))

    @staticmethod
    def decompress_hadoop_into[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises CorruptError:
        """Decompress `src` into exactly `dst` as Arrow C++ reads Parquet's
        `LZ4` codec: a run of Hadoop frames, or, when `src` does not parse as
        one, a plain block -- what Parquet C++ wrote before it adopted the
        Hadoop framing."""
        if not Self._hadoop_frames_into(src, dst):
            Self.decompress_block_into(src, dst)

    @staticmethod
    def _hadoop_frames_into[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) -> Bool:
        """Whether `src` is a run of Hadoop frames that decodes to exactly
        `dst` -- Arrow C++'s `TryDecompressHadoop`. A frame here is one block;
        Hadoop's own codec may split one across several, which no Parquet
        writer does."""
        var ip = 0
        var op = 0
        while len(src) - ip >= 8:
            var raw = Int(byte_swap(LittleEndian.fixed[DType.uint32](src, ip)))
            var size = Int(
                byte_swap(LittleEndian.fixed[DType.uint32](src, ip + 4))
            )
            ip += 8
            if size > len(src) - ip or raw > len(dst) - op:
                return False
            try:
                Self.decompress_block_into(
                    src[ip : ip + size], dst[op : op + raw]
                )
            except:
                return False
            ip += size
            op += raw
        return ip == len(src) and op == len(dst)

    @staticmethod
    def max_frame_length(n: Int) -> Int:
        """The most bytes `compress_frame` can produce for `n` input bytes:
        the 7-byte header, the end mark, and each 64 KiB block at most its own
        length (a block that does not shrink is stored) plus its size field."""
        var blocks = (n + Self._FRAME_BLOCK - 1) // Self._FRAME_BLOCK
        return 7 + 4 + n + 4 * blocks

    @staticmethod
    def compress_frame(src: Span[mut=False, UInt8, _], mut dst: List[UInt8]):
        """Append an LZ4 frame holding `src` to `dst` -- `LZ4F_compressFrame`
        with the default preferences Arrow C++ passes, byte for byte. Its
        blocks are 64 KiB, so unlike `compress_block` it takes any size:
        past 4 GiB the table's 32-bit positions wrap, and a wrapped one is
        out of the window and never matched, so the frame stays correct."""
        var n = len(src)
        var start = len(dst)
        # At least doubling, so frames appended one after another to one
        # list -- an IPC body's buffers -- do not copy it once per frame.
        var need = start + Self.max_frame_length(n)
        if dst.capacity() < need:
            dst.reserve(max(need, 2 * dst.capacity()))
        LittleEndian.append[DType.uint32](dst, Self._FRAME_MAGIC)
        # Version 01; one block is independent, more are linked.
        var linked = n > Self._FRAME_BLOCK
        dst.append(UInt8(0x40 if linked else 0x60))
        dst.append(0x40)  # 64 KiB maximum block size
        var hc = Self._header_checksum(Span(dst)[start + 4 : start + 6])
        dst.append(hc)
        if linked:
            var matcher = _Matcher[small=False]()
            var pos = 0
            while pos < n:
                var size = min(n - pos, Self._FRAME_BLOCK)
                Self._frame_block(matcher, src, pos, pos + size, dst)
                pos += size
        elif n > 0:
            var matcher = _Matcher[small=True]()
            Self._frame_block(matcher, src, 0, n, dst)
        LittleEndian.append[DType.uint32](dst, 0)  # end mark

    @staticmethod
    def _header_checksum(descriptor: Span[UInt8, _]) -> UInt8:
        """A frame descriptor's checksum: the second byte of its XXH32."""
        return UInt8((XxHash32.hash(descriptor) >> 8) & 0xFF)

    @staticmethod
    def _frame_block[
        small: Bool
    ](
        mut matcher: _Matcher[small],
        src: Span[mut=False, UInt8, _],
        start: Int,
        end: Int,
        mut dst: List[UInt8],
    ):
        """One frame block behind its size, stored as is when compressing it
        would not save a byte -- `LZ4F_makeBlock`."""
        var at = len(dst)
        LittleEndian.append[DType.uint32](dst, 0)
        if matcher.block[limited=True](src, start, end, dst):
            LittleEndian.write[DType.uint32](dst, at, UInt32(len(dst) - at - 4))
        else:
            dst.extend(src[start:end])
            LittleEndian.write[DType.uint32](
                dst, at, UInt32(end - start) | 0x80000000
            )

    @staticmethod
    def decompress_frame_into[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        """Decompress the one LZ4 frame `src` into exactly `dst`, whose length
        must be the decompressed one. Nothing past `len(dst)` is written; on
        a corrupt frame, what was decoded before the corruption is left in
        `dst`."""
        var n_in = len(src)
        if n_in < 7:
            raise CorruptError(t"lz4: a {n_in}-byte frame is too short")
        var magic = LittleEndian.fixed[DType.uint32](src, 0)
        if magic == Self._LEGACY_MAGIC:
            raise NotImplementedError("lz4: the legacy frame format")
        if magic != Self._FRAME_MAGIC:
            raise CorruptError(t"lz4: bad frame magic {hex(magic)}")
        var flg = Int(src[4])
        var bd = Int(src[5])
        if flg >> 6 != 1:
            raise CorruptError(t"lz4: frame version {flg >> 6}, not 1")
        if flg & 0x02 != 0 or bd & 0x8F != 0:
            raise CorruptError("lz4: reserved frame descriptor bits set")
        var block_id = (bd >> 4) & 7
        if block_id < 4:
            raise CorruptError(t"lz4: block size id {block_id} is reserved")
        var max_block = 1 << (2 * block_id + 8)
        var independent = flg & 0x20 != 0
        var block_checksum = flg & 0x10 != 0
        var has_size = flg & 0x08 != 0
        var content_checksum = flg & 0x04 != 0
        var has_dict = flg & 0x01 != 0
        var ip = 6 + (8 if has_size else 0) + (4 if has_dict else 0)
        if n_in < ip + 1:
            raise CorruptError("lz4: truncated frame header")
        if src[ip] != Self._header_checksum(src[4:ip]):
            raise CorruptError("lz4: frame header checksum mismatch")
        if has_dict:
            raise NotImplementedError("lz4: frames that need a dictionary")
        var content_size = -1
        if has_size:
            content_size = Int(LittleEndian.fixed[DType.uint64](src, 6))
        ip += 1
        if content_size >= 0 and content_size != len(dst):
            raise CorruptError(
                t"lz4: frame declares {content_size} bytes, destination"
                t" holds {len(dst)}"
            )
        var op = 0
        while True:
            if n_in - ip < 4:
                raise CorruptError("lz4: frame ends without its end mark")
            var word = LittleEndian.fixed[DType.uint32](src, ip)
            ip += 4
            if word == 0:
                break
            var size = Int(word & 0x7FFFFFFF)
            if size > max_block:
                raise CorruptError(
                    t"lz4: a {size}-byte block in a frame of {max_block}-byte"
                    t" blocks"
                )
            var tail = 4 if block_checksum else 0
            if n_in - ip < size + tail:
                raise CorruptError("lz4: truncated frame block")
            var block = src[ip : ip + size]
            if block_checksum:
                var want = LittleEndian.fixed[DType.uint32](src, ip + size)
                if XxHash32.hash(block) != want:
                    raise CorruptError("lz4: frame block checksum mismatch")
            if word & 0x80000000 != 0:
                if size > len(dst) - op:
                    raise CorruptError(
                        t"lz4: a stored block at output {op} overflows the"
                        t" declared {len(dst)} bytes"
                    )
                BufferView(dst).slice(op).copy_from(BufferView(block), size)
                op += size
            else:
                if size == 0:
                    raise CorruptError("lz4: empty compressed frame block")
                var decoder = _BlockDecoder(
                    block, dst, op, op if independent else 0
                )
                decoder.run()
                if decoder.op - op > max_block:
                    raise CorruptError(
                        t"lz4: a frame block decodes past its {max_block} bytes"
                    )
                op = decoder.op
            ip += size + tail
        if content_checksum:
            if n_in - ip < 4:
                raise CorruptError("lz4: truncated frame content checksum")
            var want = LittleEndian.fixed[DType.uint32](src, ip)
            if XxHash32.hash(dst[:op]) != want:
                raise CorruptError("lz4: frame content checksum mismatch")
            ip += 4
        if ip != n_in:
            raise CorruptError(
                t"lz4: {n_in - ip} bytes after the frame; one frame expected"
            )
        if op != len(dst):
            raise CorruptError(
                t"lz4: frame decodes to {op} bytes, not the expected {len(dst)}"
            )
