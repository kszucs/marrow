# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause

# The compressor is ported from liblz4 1.10.0's `lz4.c` (`LZ4_compress_generic`) and `lz4frame.c`,
# Copyright (c) 2011-2023 Yann Collet (BSD 2-Clause); see NOTICE.txt.

"""LZ4, in Mojo: `Lz4`, the block format Parquet's LZ4_RAW pages hold, and
`Lz4Frame`, the frame format Arrow IPC's LZ4_FRAME buffers hold.

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
block and content checksums. `Lz4Frame.compress` writes what Arrow C++ writes,
`LZ4F_compressFrame` with default preferences, byte for byte: no checksums,
one independent block up to 64 KiB, and above that 64 KiB blocks linked
through one table, so a match reaches into the block before; a block that
would not come out shorter than its input is stored as is.
`Lz4Frame.decompress` reads what any writer does: block sizes 64 KiB..4 MiB,
independent or linked blocks -- one contiguous output makes a link to an earlier
block an ordinary offset -- both checksums and the content size. Like Arrow, it
takes exactly one frame; dictionaries and the legacy format are refused.

`LibLz4` and `LibLz4Frame` are the same two formats through the system's
liblz4.
"""


from std.memory.alloc import unsafe_alloc

from ..errors import (
    CorruptError,
    DynError,
    InternalError,
    InvalidError,
    NotImplementedError,
)
from ..views import BufferView
from ..utils.hashing import XxHash32
from .lz77 import LzCopy, hash_bytes, match_length
from .core import Compressor, append_uninit, library, reserve_more
from ..utils.byteorder import LittleEndian


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
            return hash_bytes[5](src, pos, _HASH_LOG)

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
        dst.resize(unsafe_uninit_length=at + Lz4.max_compressed_length(n) + 8)
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
            if not Self._extend(src, ip, in_limit, lit):
                return False
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
            if not Self._extend(src, ip, in_limit, ml):
                return False
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
    def _extend(
        src: BufferView[DType.uint8, _],
        mut ip: Int,
        limit: Int,
        mut length: Int,
    ) -> Bool:
        """Add the bytes extending `length` past the token's 15, at `ip`,
        which is moved past them: 255s, then one below 255. `False` if they
        run to `limit`."""
        # Inlined: a long literal run is thousands of them, and called, the
        # loop made decoding floats 10% slower.
        while True:
            if ip >= limit:
                return False
            var b = Int(src.unsafe_get(ip))
            ip += 1
            length += b
            if b != 255:
                return True

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
        if lit == 15 and not Self._extend(BufferView(src), ip, n_in, lit):
            raise CorruptError("lz4: truncated literal length")
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
        if ml == 15 and not Self._extend(BufferView(src), ip, n_in, ml):
            raise CorruptError("lz4: truncated match length")
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


struct Lz4(Compressor):
    """LZ4's block format: what Parquet's LZ4_RAW pages hold."""

    comptime MAX_INPUT = 0x7E000000
    """The largest block liblz4 compresses, its `LZ4MAX_INPUT_SIZE`."""

    @staticmethod
    def max_compressed_length(n: Int) -> Int:
        """The most bytes `compress` can produce for `n` input bytes --
        liblz4's `LZ4_compressBound`: stored as one run of literals, with its
        length's extension bytes."""
        return n + n // 255 + 16

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """Under 256 times as many: a length grows by at most 255 for each
        byte that extends it."""
        return 256 * n

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append the LZ4 block for `src` to `dst` -- `LZ4_compress_default`'s
        bytes."""
        var n = len(src)
        if n > Self.MAX_INPUT:
            raise InvalidError(
                t"lz4: {n} bytes is over the block format's {Self.MAX_INPUT}"
            )
        if n < _SMALL:
            var matcher = _Matcher[small=True]()
            _ = matcher.block[limited=False](src, 0, n, dst)
        else:
            var matcher = _Matcher[small=False]()
            _ = matcher.block[limited=False](src, 0, n, dst)

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
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


struct Lz4Frame(Compressor):
    """LZ4's frame format: what Arrow IPC's LZ4_FRAME buffers hold."""

    comptime _MAGIC = UInt32(0x184D2204)
    comptime _LEGACY_MAGIC = UInt32(0x184C2102)
    comptime _BLOCK = 1 << 16
    """The block size `compress` writes: LZ4F's default, 64 KiB."""

    # The descriptor's FLG byte: the version in the top two bits, then flags.
    comptime _VERSION_1 = 0x40
    comptime _INDEPENDENT = 0x20
    comptime _BLOCK_CHECKSUM = 0x10
    comptime _CONTENT_SIZE = 0x08
    comptime _CONTENT_CHECKSUM = 0x04
    comptime _DICTIONARY_ID = 0x01
    comptime _STORED = UInt32(0x80000000)
    """A block size's high bit: the block is stored as is."""

    @staticmethod
    def max_compressed_length(n: Int) -> Int:
        """The most bytes `compress` can produce for `n` input bytes:
        the 7-byte header, the end mark, and each 64 KiB block at most its own
        length (a block that does not shrink is stored) plus its size field."""
        var blocks = (n + Self._BLOCK - 1) // Self._BLOCK
        return 7 + 4 + n + 4 * blocks

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """A block's bound: the frame's own fields only take room."""
        return Lz4.max_decompressed_length(n)

    @staticmethod
    def content_length(src: Span[UInt8, _]) raises DynError -> Optional[Int]:
        """The content size the frame's descriptor declares, or None when
        it leaves it out -- as `compress` does, and Arrow C++."""
        if len(src) < 7:
            raise CorruptError(t"lz4: a {len(src)}-byte frame is too short")
        var magic = LittleEndian.fixed[DType.uint32](src, 0)
        if magic == Self._LEGACY_MAGIC:
            raise NotImplementedError("lz4: the legacy frame format")
        if magic != Self._MAGIC:
            raise CorruptError(t"lz4: bad frame magic {hex(magic)}")
        if Int(src[4]) & Self._CONTENT_SIZE == 0:
            return None
        if len(src) < 14:
            raise CorruptError("lz4: truncated frame header")
        var n = Int(LittleEndian.fixed[DType.uint64](src, 6))
        if n < 0 or n > Self.max_decompressed_length(len(src)):
            raise CorruptError(t"lz4: a {len(src)}-byte frame cannot hold {n}")
        return n

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append an LZ4 frame holding `src` to `dst` -- `LZ4F_compressFrame`
        with the default preferences Arrow C++ passes, byte for byte. Its
        blocks are 64 KiB, so unlike `Lz4.compress` it takes any size:
        past 4 GiB the table's 32-bit positions wrap, and a wrapped one is
        out of the window and never matched, so the frame stays correct."""
        var n = len(src)
        var start = len(dst)
        reserve_more(dst, Self.max_compressed_length(n))
        LittleEndian.append[DType.uint32](dst, Self._MAGIC)
        # One block is independent, more are linked.
        var linked = n > Self._BLOCK
        dst.append(
            UInt8(Self._VERSION_1 | (0 if linked else Self._INDEPENDENT))
        )
        dst.append(4 << 4)  # block size id 4: 64 KiB
        var hc = Self._header_checksum(Span(dst)[start + 4 : start + 6])
        dst.append(hc)
        if linked:
            var matcher = _Matcher[small=False]()
            var pos = 0
            while pos < n:
                var size = min(n - pos, Self._BLOCK)
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
                dst, at, UInt32(end - start) | Self._STORED
            )

    @staticmethod
    def decompress[
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
        if magic != Self._MAGIC:
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
        var independent = flg & Self._INDEPENDENT != 0
        var block_checksum = flg & Self._BLOCK_CHECKSUM != 0
        var has_size = flg & Self._CONTENT_SIZE != 0
        var content_checksum = flg & Self._CONTENT_CHECKSUM != 0
        var has_dict = flg & Self._DICTIONARY_ID != 0
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
            var size = Int(word & ~Self._STORED)
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
            if word & Self._STORED != 0:
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


# ---------------------------------------------------------------------------
# The same formats through the system's liblz4.
# ---------------------------------------------------------------------------


struct LibLz4(Compressor):
    """LZ4 blocks through liblz4: `LZ4_compress_default` and
    `LZ4_decompress_safe`."""

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append the block for `src` -- what `Lz4.compress` writes, with
        liblz4 1.10.0 byte for byte."""
        if len(src) > Lz4.MAX_INPUT:
            raise InvalidError(
                t"lz4: {len(src)} bytes is over the block format's"
                t" {Lz4.MAX_INPUT}"
            )
        var l = library["lz4"]()
        var bound = Int(l.call["LZ4_compressBound", Int32](Int32(len(src))))
        var at = append_uninit(dst, bound)
        var n = l.call["LZ4_compress_default", Int32](
            src.unsafe_ptr(),
            dst.unsafe_ptr().unsafe_offset(at),
            Int32(len(src)),
            Int32(bound),
        )
        if n <= 0:
            dst.shrink(at)
            raise InternalError("lz4: liblz4 could not compress")
        dst.shrink(at + Int(n))

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        if len(src) > Lz4.MAX_INPUT or len(dst) > Int(Int32.MAX):
            raise InvalidError("lz4: liblz4 decodes blocks of under 2 GiB")
        var n = library["lz4"]().call["LZ4_decompress_safe", Int32](
            src.unsafe_ptr(), dst.unsafe_ptr(), Int32(len(src)), Int32(len(dst))
        )
        if Int(n) != len(dst):
            raise CorruptError("lz4: liblz4 could not decode the expected size")


struct LibLz4Frame(Compressor):
    """LZ4 frames through liblz4: `LZ4F_compressFrame` with the default
    preferences Arrow C++ passes, and `LZ4F_decompress`."""

    comptime _VERSION = UInt32(100)
    """`LZ4F_VERSION`, which a decompression context is made for."""

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append the frame for `src` -- what `Lz4Frame.compress` writes,
        with liblz4 1.10.0 byte for byte."""
        var l = library["lz4"]()
        var bound = l.call["LZ4F_compressFrameBound", Int](len(src), 0)
        var at = append_uninit(dst, bound)
        var n = l.call["LZ4F_compressFrame", Int](
            dst.unsafe_ptr().unsafe_offset(at),
            bound,
            src.unsafe_ptr(),
            len(src),
            0,
        )
        if l.call["LZ4F_isError", UInt32](n) != 0:
            dst.shrink(at)
            raise InternalError("lz4: liblz4 could not compress a frame")
        dst.shrink(at + n)

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        """The one frame `src` into exactly `dst`, `LZ4F_decompress` called
        until it reports the frame's end."""
        var l = library["lz4"]()
        var ctx = unsafe_alloc[Int](1)
        var sizes = unsafe_alloc[UInt](2)  # out capacity, in available
        var rc = l.call["LZ4F_createDecompressionContext", Int](
            ctx, Self._VERSION
        )
        if l.call["LZ4F_isError", UInt32](rc) != 0:
            ctx.unsafe_free()
            sizes.unsafe_free()
            raise InternalError("lz4: liblz4 could not make a frame decoder")
        var dctx = ctx[unsafe_offset=0]
        var ip = 0
        var op = 0
        var done = False
        var failed = False
        while not done and not failed:
            sizes[unsafe_offset=0] = UInt(len(dst) - op)
            sizes[unsafe_offset=1] = UInt(len(src) - ip)
            var hint = l.call["LZ4F_decompress", Int](
                dctx,
                dst.unsafe_ptr().unsafe_offset(op),
                sizes,
                src.unsafe_ptr().unsafe_offset(ip),
                sizes.unsafe_offset(1),
                0,
            )
            var wrote = Int(sizes[unsafe_offset=0])
            var read = Int(sizes[unsafe_offset=1])
            op += wrote
            ip += read
            # An error, or no progress short of the end: the input ran out.
            failed = l.call["LZ4F_isError", UInt32](hint) != 0 or (
                hint != 0 and wrote == 0 and read == 0
            )
            done = hint == 0
        _ = l.call["LZ4F_freeDecompressionContext", Int](dctx)
        ctx.unsafe_free()
        sizes.unsafe_free()
        if failed or ip != len(src) or op != len(dst):
            raise CorruptError(
                "lz4: liblz4 could not decode one frame of the expected size"
            )
