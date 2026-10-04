# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# `FastMatcher` is ported from libzstd 1.5.7's `zstd_fast.c`
# (`ZSTD_compressBlock_fast_noDict_generic`), Copyright (c) Meta Platforms,
# Inc. (BSD 3-Clause); see NOTICE.txt.

"""Finding a zstd block's matches: libzstd's fast strategy, which level 1
compresses with.

`FastMatcher` keeps one `uint32` table over the whole input, so a match
reaches back across blocks; it hashes 5 to 7 bytes, probes two positions at
a time, tries the most recent offset before the table and the one before it
right after each match. It reports into a `SeqStore`, with each offset
already as it is sent: the match finder knows which matches repeat an
offset, because it looked for them, so nothing after it has to find out
again.
"""

from std.bit import bit_width

from ...views import BufferView
from ..byteorder import LittleEndian
from ..lz77 import LzCopy, match_length
from .block import RepeatOffsets


struct SeqStore[lo: MutOrigin, so: MutOrigin](TrivialRegisterPassable):
    """Where `FastMatcher` reports a block -- libzstd's `seqStore_t`: the
    literals, copied out, and for each match its literal count, offset code
    and length. The offset code is as Zstandard sends it: 1 for the most
    recent offset -- the one before it when no literals precede the match --
    or the offset plus 3."""

    var _lits: BufferView[DType.uint8, Self.lo]
    var _seqs: BufferView[DType.uint32, Self.so]
    var n_lits: Int
    var n_seqs: Int
    var _pending: Int

    def __init__(
        out self, lits: Span[UInt8, Self.lo], seqs: Span[UInt32, Self.so]
    ):
        self._lits = BufferView(lits)
        self._seqs = BufferView(seqs)
        self.n_lits = 0
        self.n_seqs = 0
        self._pending = 0

    @always_inline
    def literal(mut self, src: Span[UInt8, _], start: Int, end: Int):
        """The next input bytes, `src[start:end]`, unmatched. A run of at most
        64 is copied as one or two 32-byte blocks, whatever its length, while
        the input has 64 bytes to read -- `ZSTD_storeSeq` copies 16 and
        wildcopies the rest; a fixed copy leaves one branch, where a loop's
        count mispredicted on every other run. The literals buffer has the
        slack those blocks write past the run. A longer run, a block of
        incompressible values, goes to `memcpy` as libzstd's last literals
        do."""
        var n = end - start
        if n <= LzCopy.SLOP and start + LzCopy.SLOP <= len(src):
            LzCopy.blocks64(
                BufferView(src).slice(start), self._lits.slice(self.n_lits), n
            )
        else:
            self._lits.slice(self.n_lits).copy_from(
                BufferView(src).slice(start), n
            )
        self.n_lits += n
        self._pending += n

    @always_inline
    def match(mut self, offset_code: Int, length: Int):
        """The next `length` input bytes, at the offset `offset_code` names."""
        var at = 3 * self.n_seqs
        self._seqs.unsafe_set(at, UInt32(self._pending))
        self._seqs.unsafe_set(at + 1, UInt32(offset_code))
        self._seqs.unsafe_set(at + 2, UInt32(length))
        self.n_seqs += 1
        self._pending = 0


struct FastMatcher(Movable):
    """The fast strategy of libzstd, at level 1's parameters for the input's
    size -- `ZSTD_compressBlock_fast_noDict_generic`. It keeps its table and
    its two recent offsets from one block to the next.

    The two offsets are the decoder's two most recent, exactly, so a match
    can be sent as a repeat of either: every match updates them as the
    decoder will. A block stored raw or as one repeated byte leaves the
    decoder's as they were, so a block's take effect only when the caller
    `commit`s it. A new offset is always sent as one, even when it equals a
    recent one -- as libzstd's fast strategy does -- which is what keeps the
    sending free of a chain from each match to the next."""

    comptime _STEP_EVERY = 1 << 7
    """Unmatched bytes after which the search step grows by one."""

    var _table: List[UInt32]
    var _log: Int
    var _width: Int
    """The bytes hashed: the minimum match, 5 to 7."""
    var _window: Int
    """How far back from a block's end a match may start."""
    var _rep1: Int
    """The decoder's most recent offset."""
    var _rep2: Int
    """The one before it."""
    var _next1: Int
    var _next2: Int
    """What the two become once the block last matched is committed."""

    def __init__(out self, n: Int):
        """For an input of `n` bytes: level 1's row for that size, its window
        and hash table shrunk to fit a small input -- `ZSTD_getCParams` and
        `ZSTD_adjustCParams`."""
        var window: Int
        var log: Int
        if n <= 1 << 14:
            window, log, self._width = 14, 15, 5
        elif n <= 1 << 17:
            window, log, self._width = 17, 13, 6
        elif n <= 1 << 18:
            window, log, self._width = 18, 14, 6
        else:
            window, log, self._width = 19, 14, 7
        var src_log = max(6, bit_width(max(n, 1) - 1))
        self._log = min(log, min(window, src_log) + 1)
        self._window = 1 << window
        self._table = List[UInt32](length=1 << self._log, fill=0)
        self._rep1 = RepeatOffsets.INITIAL.offset1
        self._rep2 = RepeatOffsets.INITIAL.offset2
        self._next1 = self._rep1
        self._next2 = self._rep2

    def window(self) -> Int:
        """The farthest a match reaches back, in bytes."""
        return self._window

    def commit(mut self):
        """The block last matched is sent compressed: the decoder's recent
        offsets are now the ones its matches left."""
        self._rep1 = self._next1
        self._rep2 = self._next2

    @always_inline
    def _hash[width: Int](self, src: Span[UInt8, _], pos: Int) -> Int:
        """The table slot for the `width` bytes at `pos`, read as 8 --
        `ZSTD_hash5Ptr` to `ZSTD_hash7Ptr`."""
        comptime prime = UInt64(889523592379) if width == 5 else (
            UInt64(227718039650203) if width == 6 else UInt64(58295818150454627)
        )
        var u = LittleEndian.fixed[DType.uint64](src, pos)
        var h = (u << UInt64(64 - 8 * width)) * prime
        return Int(h >> UInt64(64 - self._log))

    def compress(
        mut self,
        src: Span[mut=False, UInt8, _],
        start: Int,
        end: Int,
        mut out: SeqStore[_, _],
    ):
        """Report `src[start:end]` to `out`; a match may reach back to
        `src[0]`, and stops at `end`. The hashed width is a parameter, as in
        libzstd, which instantiates the loop once per width."""
        if self._width == 5:
            self._compress[5](src, start, end, out)
        elif self._width == 6:
            self._compress[6](src, start, end, out)
        else:
            self._compress[7](src, start, end, out)

    def _compress[
        width: Int
    ](
        mut self,
        src: Span[mut=False, UInt8, _],
        start: Int,
        end: Int,
        mut out: SeqStore[_, _],
    ):
        var table = BufferView(Span(self._table))

        @always_inline
        def read32(pos: Int) {src} -> UInt32:
            return LittleEndian.fixed[DType.uint32](src, pos)

        var anchor = start
        var ip0 = start + Int(start == 0)
        # Nothing before the window, measured from the block's end, is a
        # candidate; nor is an empty slot, which reads as position 0 -- a
        # position never stored, since the search starts at 1.
        var low = max(0, end - self._window)
        var lowest = max(1, low)
        var rep1 = self._rep1
        var rep2 = self._rep2
        var saved1 = 0
        var saved2 = 0
        # An offset reaching before the input, or the window, is no candidate.
        var max_rep = min(ip0, self._window)
        if rep2 > max_rep:
            saved2 = rep2
            rep2 = 0
        if rep1 > max_rep:
            saved1 = rep1
            rep1 = 0
        var limit = end - 8
        while True:
            var step = 2
            var next_step = ip0 + Self._STEP_EVERY
            var ip1 = ip0 + 1
            var ip2 = ip0 + step
            var ip3 = ip2 + 1
            if ip3 >= limit:
                break
            var hash0 = self._hash[width](src, ip0)
            var hash1 = self._hash[width](src, ip1)
            var candidate = Int(table.unsafe_get(hash0))
            var current: Int
            var match0 = 0
            var length = 0
            # 0: nothing before the limit; 1: the recent offset at ip2;
            # 2: the table's candidate for ip0.
            var found = 0
            while True:
                var repeat = read32(ip2 - rep1)
                current = ip0
                table.unsafe_set(hash0, UInt32(current))
                if read32(ip2) == repeat and rep1 > 0:
                    ip0 = ip2
                    match0 = ip0 - rep1
                    length = Int(
                        src.unsafe_get(ip0 - 1) == src.unsafe_get(match0 - 1)
                    )
                    ip0 -= length
                    match0 -= length
                    length += 4
                    table.unsafe_set(hash1, UInt32(ip1))
                    found = 1
                    break
                if read32(ip0) == read32(candidate) and candidate >= lowest:
                    table.unsafe_set(hash1, UInt32(ip1))
                    found = 2
                    break
                candidate = Int(table.unsafe_get(hash1))
                hash0 = hash1
                hash1 = self._hash[width](src, ip2)
                ip0 = ip1
                ip1 = ip2
                ip2 = ip3
                current = ip0
                table.unsafe_set(hash0, UInt32(current))
                if read32(ip0) == read32(candidate) and candidate >= lowest:
                    if step <= 4:
                        table.unsafe_set(hash1, UInt32(ip1))
                    found = 2
                    break
                candidate = Int(table.unsafe_get(hash1))
                hash0 = hash1
                hash1 = self._hash[width](src, ip2)
                ip0 = ip1
                ip1 = ip2
                ip2 = ip0 + step
                ip3 = ip1 + step
                if ip2 >= next_step:
                    step += 1
                    next_step += Self._STEP_EVERY
                if ip3 >= limit:
                    break
            if found == 0:
                break
            if found == 2:
                match0 = candidate
                rep2 = rep1
                rep1 = ip0 - match0
                length = 4
                while (
                    ip0 > anchor
                    and match0 > low
                    and src.unsafe_get(ip0 - 1) == src.unsafe_get(match0 - 1)
                ):
                    ip0 -= 1
                    match0 -= 1
                    length += 1
            length += match_length(src, ip0 + length, match0 + length, end)
            # A repeat follows literals: it starts at least 2 bytes past the
            # anchor, and backs up by 1 at most.
            debug_assert(found == 2 or ip0 > anchor, "repeat without literals")
            out.literal(src, anchor, ip0)
            out.match(1 if found == 1 else ip0 - match0 + 3, length)
            ip0 += length
            anchor = ip0
            if ip0 <= limit:
                table.unsafe_set(
                    self._hash[width](src, current + 2), UInt32(current + 2)
                )
                table.unsafe_set(
                    self._hash[width](src, ip0 - 2), UInt32(ip0 - 2)
                )
                # The offset before the last one, straight after it.
                while (
                    rep2 > 0
                    and ip0 <= limit
                    and read32(ip0) == read32(ip0 - rep2)
                ):
                    var run = (
                        match_length(src, ip0 + 4, ip0 + 4 - rep2, end) + 4
                    )
                    rep1, rep2 = rep2, rep1
                    table.unsafe_set(self._hash[width](src, ip0), UInt32(ip0))
                    # Without literals, a first repeat is the one before.
                    out.match(1, run)
                    ip0 += run
                    anchor = ip0
        if saved1 != 0 and rep1 != 0:
            saved2 = saved1
        self._next1 = rep1 if rep1 != 0 else saved1
        self._next2 = rep2 if rep2 != 0 else saved2
        out.literal(src, anchor, end)
