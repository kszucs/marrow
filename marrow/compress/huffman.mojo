# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# The encoding half is ported from libzstd 1.5.7's `huf_compress.c`, the
# literals section's choice of coding from its `zstd_compress_literals.c`,
# and the choice between one and two symbols per lookup
# (`HuffmanTable._TIMES`) from its `huf_decompress.c`, Copyright (c) Meta
# Platforms, Inc. (BSD 3-Clause); see NOTICE.txt.

"""Huffman coding, and the literals section Zstandard holds it in -- a
block's literals stored, one byte repeated, or Huffman-coded with a tree of
their own or the block before's. `Huffman` is sections back to back.

A tree is sent as a weight per symbol: weight `w > 0` is a code of
`log + 1 - w` bits, the last symbol's weight is implied by completing the sum
of `2 ** (w - 1)` to a power of two, and codes are assigned in ascending order
from the lowest weight, symbols of one weight in their natural order. The
weights come as 4-bit fields, or FSE-compressed with two interleaved states.

| Type | What it is |
|---|---|
| `HuffmanTable` | a tree read from a block, as the tables that decode with it: one symbol a lookup, or two |
| `HuffmanEncoder` | a code built from symbol counts, written as a tree and coded with |
| `_Container` | the bits a stream holds between stores, as the encoder packs them |
| `LiteralsHeader` | a section's type, streams, literal count and size |
| `LiteralsEncoder` | a section's coding, and the code the decoder holds |

A one-symbol table is indexed by the next `log` bits of a stream: each
symbol fills the `2 ** (w - 1)` entries its code prefixes, so one lookup
decodes one symbol and says how many bits it took.
"""

from std.bit import bit_width, log2_floor
from std.builtin.globals import global_constant
from std.memory import bitcast

from ..errors import CorruptError, DynError
from ..views import BufferView
from .core import Compressor
from ..utils.bits import BitReader, BitWriter
from .fse import (
    MAX_WEIGHT,
    Alphabet,
    Distribution,
    FseEncoder,
    FseTable,
    Histogram,
)
from ..utils.byteorder import LittleEndian


comptime BLOCK_MAX = 1 << 17
"""The most a Zstandard block holds, compressed or not, and so the most
literals a section holds: 128 KiB."""


struct HuffmanTable(Movable):
    """A literals table, kept across blocks for the "treeless" ones."""

    var _entries: List[UInt16]
    """`symbol | bits << 8`, indexed by the next `log` bits."""
    var _pairs: List[UInt32]
    """Up to two symbols per lookup, indexed by the next `_PAIR_LOG` bits:
    `first | second << 8 | bits << 16 | count << 24` -- libzstd's X2
    table. Built from `_entries` when a block wants it."""
    var _templates: List[UInt32]
    """Scratch for `_build_pairs`: each code length's second codes."""
    var _pairs_ready: Bool
    var _weights: List[UInt8]
    var _fse: FseTable
    var log: Int
    var ready: Bool
    """Whether a tree has been read in this frame."""

    # fmt: off
    comptime _TIMES: Array[Int, 64] = [
        0, 0, 1, 1,
        0, 0, 1, 1,
        150, 216, 381, 119,
        170, 205, 514, 112,
        177, 199, 539, 110,
        197, 194, 644, 107,
        221, 192, 735, 107,
        256, 189, 881, 106,
        359, 188, 1167, 109,
        582, 187, 1570, 114,
        688, 187, 1712, 122,
        825, 186, 1965, 136,
        976, 185, 2131, 150,
        1180, 186, 2070, 175,
        1377, 185, 1731, 202,
        1412, 185, 1695, 202,
    ]
    """`algoTime` from libzstd's `huf_decompress.c`: per compressed-size
    sixteenth, the table-build and per-256-bytes costs of decoding one and
    two symbols per lookup."""
    # fmt: on

    @staticmethod
    def _pairs_pay(n: Int, size: Int) -> Bool:
        """Whether two symbols per lookup decode `n` literals compressed to
        `size` bytes faster, table build included -- `HUF_selectDecoder`."""
        var q = 15 if size >= n else size * 16 // n
        var d256 = n >> 8
        ref t = global_constant[Self._TIMES]()
        var one = t.unsafe_get(4 * q) + t.unsafe_get(4 * q + 1) * d256
        var two = t.unsafe_get(4 * q + 2) + t.unsafe_get(4 * q + 3) * d256
        two += two >> 5
        return two < one

    @staticmethod
    @always_inline
    def _symbol(
        table: BufferView[DType.uint16, _], mut r: BitReader, log: Int
    ) -> UInt8:
        """One lookup in the one-symbol table -- libzstd's X1, `symbol |
        bits << 8` for each value of the next `log` bits: the symbol the
        next bits of `r` start with, taking its bits."""
        var e = Int(table.unsafe_get(r.peek_nonzero(log)))
        r.skip(e >> 8)
        return UInt8(e & 0xFF)

    @staticmethod
    @always_inline
    def _pair(
        pairs: BufferView[DType.uint32, _],
        mut r: BitReader,
        dst: BufferView[mut=True, DType.uint8, _],
        o: Int,
    ) -> Int:
        """One lookup in the two-symbol table -- libzstd's X2, `first |
        second << 8 | bits << 16 | count << 24` for each value of the next
        `_PAIR_LOG` bits: the one or two symbols the next bits of `r` start
        with, written at `o` -- both bytes always, the caller leaving room;
        the new `o`."""
        var e = pairs.unsafe_get(r.peek_nonzero(_PAIR_LOG))
        # Both bytes in one store, the entry's low half.
        dst.store[2](o, bitcast[DType.uint8, 2](UInt16(e & 0xFFFF)))
        r.skip(Int((e >> 16) & 0xFF))
        return o + Int(e >> 24)

    def __init__(out self):
        self._entries = List[UInt16](capacity=1 << MAX_WEIGHT)
        self._pairs = List[UInt32]()
        self._templates = List[UInt32]()
        self._pairs_ready = False
        self._weights = List[UInt8](capacity=256)
        self._fse = FseTable()
        self.log = 0
        self.ready = False

    def decode_section(
        mut self,
        header: LiteralsHeader,
        block: Span[UInt8, _],
        into: BufferView[mut=True, DType.uint8, _],
        room: Int,
    ) raises CorruptError:
        """Decode the literals section `header` heads, at the start of
        `block`, into `into`, writing nothing past its literals; more than
        `room` of them is corrupt. A section with a tree replaces the
        table's; a treeless one decodes with it."""
        var n = header.n
        if n > BLOCK_MAX or n > room:
            raise CorruptError(
                t"zstd: {n} literals overflow the block or the output"
            )
        if header.kind == LiteralsHeader.RAW:
            into.copy_from(BufferView(block).slice(header.length), n)
        elif header.kind == LiteralsHeader.RLE:
            into.as_span()[:n].fill(block[header.length])
        else:
            var payload = block[header.length : header.end()]
            if header.kind == LiteralsHeader.COMPRESSED:
                payload = payload[self.read(payload) :]
            elif not self.ready:
                raise CorruptError("zstd: treeless literals without a tree")
            self.decode(payload, into, n, header.streams, header.size)

    def read(mut self, src: Span[UInt8, _]) raises CorruptError -> Int:
        """Build the table from the tree description at the start of `src`;
        return the bytes it took."""
        if len(src) == 0:
            raise CorruptError("zstd: missing Huffman tree description")
        var header = Int(src[0])
        self._weights.clear()
        var used: Int
        if header >= 128:
            var n = header - 127
            used = 1 + (n + 1) // 2
            if used > len(src):
                raise CorruptError("zstd: truncated Huffman weights")
            for i in range(n):
                var b = src[1 + i // 2]
                self._weights.append(b >> 4 if i % 2 == 0 else b & 15)
        else:
            used = 1 + header
            if header == 0 or used > len(src):
                raise CorruptError("zstd: truncated Huffman weights")
            self._fse_weights(src[1:used])
        self._build()
        return used

    def _fse_weights(mut self, src: Span[UInt8, _]) raises CorruptError:
        """Weights coded by one FSE table and two states taking turns, until
        a state update would read past the stream's start."""
        var used = self._fse.read(src, Alphabet.WEIGHTS)
        var log = self._fse.log
        ref table = self._fse.entries
        var r = BitReader(src[used:])
        var s1 = r.read(log)
        r.refill()
        var s2 = r.read(log)
        if r.remaining() < 0:
            raise CorruptError("zstd: truncated Huffman weight states")
        while True:
            if len(self._weights) > 253:
                raise CorruptError("zstd: too many Huffman weights")
            var e = table[s1]
            self._weights.append(UInt8(e.base))
            r.refill()
            s1 = Int(e.next) + r.read(Int(e.bits))
            if r.remaining() < 0:
                self._weights.append(UInt8(table[s2].base))
                break
            e = table[s2]
            self._weights.append(UInt8(e.base))
            r.refill()
            s2 = Int(e.next) + r.read(Int(e.bits))
            if r.remaining() < 0:
                self._weights.append(UInt8(table[s1].base))
                break

    def _build(mut self) raises CorruptError:
        """The table for `_weights`, completing the last one."""
        if len(self._weights) > 255:
            raise CorruptError("zstd: too many Huffman weights")
        var rank = Array[Int, MAX_WEIGHT + 2](fill=0)
        var total = 0
        for w in self._weights:
            if w > MAX_WEIGHT:
                raise CorruptError(t"zstd: Huffman weight {w} is over 11")
            if w > 0:
                total += 1 << Int(w - 1)
                rank[Int(w)] += 1
        if total == 0:
            raise CorruptError("zstd: a Huffman tree without symbols")
        var log = bit_width(total)
        if log > MAX_WEIGHT:
            raise CorruptError(t"zstd: Huffman tree depth {log} is over 11")
        var rest = (1 << log) - total
        if rest & (rest - 1) != 0:
            raise CorruptError("zstd: Huffman weights do not complete a tree")
        var last = bit_width(rest)
        self._weights.append(UInt8(last))
        rank[last] += 1
        if rank[1] < 2 or rank[1] & 1 != 0:
            raise CorruptError("zstd: Huffman weights do not complete a tree")
        # Where each weight's codes start: the longest, weight 1, at 0.
        var start = Array[Int, MAX_WEIGHT + 2](fill=0)
        for w in range(1, MAX_WEIGHT + 1):
            start[w + 1] = start[w] + (rank[w] << (w - 1))
        self._entries.resize(1 << log, 0)
        var entries = Span(self._entries)
        var weights = Span(self._weights)
        for s in range(len(weights)):
            var w = Int(weights.unsafe_get(s))
            if w > 0:
                var at = start.unsafe_get(w)
                var span = 1 << (w - 1)
                var entry = UInt16(s | (log + 1 - w) << 8)
                for i in range(at, at + span):
                    entries.unsafe_get(i) = entry
                start.unsafe_get(w) = at + span
        self.log = log
        self.ready = True
        self._pairs_ready = False

    def decode(
        mut self,
        src: Span[UInt8, _],
        dst: BufferView[mut=True, DType.uint8, _],
        n: Int,
        streams: Int,
        size: Int,
    ) raises CorruptError:
        """Decode `n` literals from 1 or 4 streams into `dst`; `size` is the
        section's compressed size, which decides between one symbol and two
        per lookup."""
        if streams == 1:
            var r = BitReader(src)
            self._finish(r, dst, 0, n)
        else:
            if len(src) < 10:
                raise CorruptError("zstd: truncated 4-stream literals")
            var a = 6 + Int(LittleEndian.fixed[DType.uint16](src, 0))
            var b = a + Int(LittleEndian.fixed[DType.uint16](src, 2))
            var c = b + Int(LittleEndian.fixed[DType.uint16](src, 4))
            if c >= len(src):
                raise CorruptError("zstd: literal stream sizes overrun")
            if n < 3 * ((n + 3) // 4):
                raise CorruptError(t"zstd: {n} literals in 4 streams")
            if self.log <= _PAIR_LOG and Self._pairs_pay(n, size):
                if not self._pairs_ready:
                    self._build_pairs()
                self._four_pairs(src, a, b, c, dst, n)
            else:
                self._four_singles(src, a, b, c, dst, n)

    @no_inline
    def _four_pairs(
        self,
        src: Span[UInt8, _],
        a: Int,
        b: Int,
        c: Int,
        dst: BufferView[mut=True, DType.uint8, _],
        n: Int,
    ) raises CorruptError:
        """The four streams that start at 6, `a`, `b` and `c` of `src`, two
        symbols a lookup, in lockstep so their dependency chains overlap.

        Five lookups of at most 11 bits per refill, which readies 57 bits or
        all the stream has left, and at most two symbols each: 10 bytes a
        round -- `HUF_decompress4X2_usingDTable_internal_fast_c_loop`. The
        last stream, the shortest, advances at least 5 a round, so one limit
        on it bounds the rounds every stream has room for, instead of eight
        checks a round; the bytes each stream has left above its start bound
        them too, so a round refills without checking where a stream stands,
        and what is left goes to `_finish_pairs`. A valid stream has the bits
        for every symbol it has room for; a corrupt one reads zeros past its
        start there and fails `finished`.

        Never inlined, nor is `_four_singles`: four streams hold more state
        than there are registers, and which values spill depends on the code
        around the loop."""
        # One 16-bit store a lookup took 64 KiB of int64s from 1.14x
        # libzstd's time to 1.0x. Inlined into the block decoder, one loop or
        # the other reloaded its table every round, 5-11% slower.
        var seg = (n + 3) // 4
        var last = n - 3 * seg
        var r1 = BitReader(src[6:a])
        var r2 = BitReader(src[a:b])
        var r3 = BitReader(src[b:c])
        var r4 = BitReader(src[c:])
        var d1 = dst
        var d2 = dst.slice(seg)
        var d3 = dst.slice(2 * seg)
        var d4 = dst.slice(3 * seg)
        var pairs = BufferView(Span(self._pairs))
        var o1 = 0
        var o2 = 0
        var o3 = 0
        var o4 = 0
        while True:
            var room = min(min(seg - o1, seg - o2), min(seg - o3, last - o4))
            # As in `_four_singles`, the rounds every stream has the bytes to
            # refill unchecked for.
            var bytes = min(
                min(r1.headroom(), r2.headroom()),
                min(r3.headroom(), r4.headroom()),
            )
            var rounds = min(room // 10, bytes // 7 + 1)
            if bytes < 0 or rounds <= 0:
                break
            var limit = o4 + 5 * rounds
            while o4 < limit:
                r1.refill_fast()
                r2.refill_fast()
                r3.refill_fast()
                r4.refill_fast()
                comptime for _ in range(5):
                    o1 = Self._pair(pairs, r1, d1, o1)
                    o2 = Self._pair(pairs, r2, d2, o2)
                    o3 = Self._pair(pairs, r3, d3, o3)
                    o4 = Self._pair(pairs, r4, d4, o4)
        self._finish_pairs(r1, d1, o1, seg)
        self._finish_pairs(r2, d2, o2, seg)
        self._finish_pairs(r3, d3, o3, seg)
        self._finish_pairs(r4, d4, o4, last)

    @no_inline
    def _four_singles(
        self,
        src: Span[UInt8, _],
        a: Int,
        b: Int,
        c: Int,
        dst: BufferView[mut=True, DType.uint8, _],
        n: Int,
    ) raises CorruptError:
        """The four streams that start at 6, `a`, `b` and `c` of `src`, one
        symbol a lookup, in lockstep. Five symbols of at most 11 bits are 55
        of the 57 bits a refill readies, or of all the stream has left --
        libzstd's fast loop takes as many, and bounds it, as `_four_pairs`
        does, by the output and by the bytes each stream has left above its
        start, so a round refills without checking where a stream stands."""
        var seg = (n + 3) // 4
        var last = n - 3 * seg
        var r1 = BitReader(src[6:a])
        var r2 = BitReader(src[a:b])
        var r3 = BitReader(src[b:c])
        var r4 = BitReader(src[c:])
        var d1 = dst
        var d2 = dst.slice(seg)
        var d3 = dst.slice(2 * seg)
        var d4 = dst.slice(3 * seg)
        var table = BufferView(Span(self._entries))
        var log = self.log
        var i = 0
        # A round takes at most 55 bits, 7 bytes, from each stream: as many
        # rounds as every stream has room for refill unchecked, as libzstd's
        # fast loop does, and the rest of each stream goes to `_finish`.
        while True:
            var room = min(
                min(r1.headroom(), r2.headroom()),
                min(r3.headroom(), r4.headroom()),
            )
            var rounds = min((last - i) // 5, room // 7 + 1)
            if room < 0 or rounds <= 0:
                break
            for _ in range(rounds):
                r1.refill_fast()
                r2.refill_fast()
                r3.refill_fast()
                r4.refill_fast()
                comptime for k in range(5):
                    d1.unsafe_set(i + k, Self._symbol(table, r1, log))
                    d2.unsafe_set(i + k, Self._symbol(table, r2, log))
                    d3.unsafe_set(i + k, Self._symbol(table, r3, log))
                    d4.unsafe_set(i + k, Self._symbol(table, r4, log))
                i += 5
        self._finish(r1, d1, i, seg)
        self._finish(r2, d2, i, seg)
        self._finish(r3, d3, i, seg)
        self._finish(r4, d4, i, last)

    def _finish_pairs(
        self,
        mut r: BitReader,
        dst: BufferView[mut=True, DType.uint8, _],
        var o: Int,
        n: Int,
    ) raises CorruptError:
        """The rest of one stream from `o` to `n`, two symbols per lookup
        while two are left -- `HUF_decodeStreamX2` -- and the last, if odd,
        alone; the stream must then be consumed exactly."""
        var pairs = BufferView(Span(self._pairs))
        while o + 8 <= n and r.full():
            r.refill()
            comptime for _ in range(4):
                o = Self._pair(pairs, r, dst, o)
        while o + 2 <= n:
            r.refill()
            o = Self._pair(pairs, r, dst, o)
        self._finish(r, dst, o, n)

    def _build_pairs(mut self):
        """The two-symbol table, from the one-symbol one: an index's first
        code, then whatever code its remaining bits start with, if it ends
        within them. Those remaining bits run through the same second codes
        after every first code of one length, so each length's are laid out
        once, and a first code's range is them plus the first code -- runs
        of writes, as libzstd's `HUF_fillDTableX2` makes, rather than two
        dependent lookups for every index."""
        comptime T = _PAIR_LOG
        var log = self.log
        var single = BufferView(Span(self._entries))
        self._pairs.resize(1 << T, 0)
        # The lengths' templates take 2^(T - length) each, under 2^T in all.
        self._templates.resize(1 << T, 0)
        var pairs = BufferView(Span(self._pairs))
        var templates = BufferView(Span(self._templates))
        var start = Array[Int, T + 1](fill=-1)
        var used = 0
        var i = 0
        while i < 1 << T:
            var e1 = Int(single.unsafe_get(i >> (T - log)))
            var n1 = e1 >> 8
            var run = 1 << (T - n1)
            if start.unsafe_get(n1) < 0:
                start.unsafe_get(n1) = used
                for r in range(run):
                    var e2 = Int(single.unsafe_get((r << n1) >> (T - log)))
                    var n2 = e2 >> 8
                    # A second code that does not fit adds nothing.
                    var fits = Int(n1 + n2 <= T)
                    templates.unsafe_set(
                        used + r,
                        UInt32(((e2 & 0xFF) << 8 | n2 << 16 | 1 << 24) * fits),
                    )
                used += run
            var first = UInt32((e1 & 0xFF) | n1 << 16 | 1 << 24)
            var t = start.unsafe_get(n1)
            for r in range(run):
                pairs.unsafe_set(i + r, first + templates.unsafe_get(t + r))
            i += run
        self._pairs_ready = True

    def _finish(
        self,
        mut r: BitReader,
        dst: BufferView[mut=True, DType.uint8, _],
        var i: Int,
        n: Int,
    ) raises CorruptError:
        """Symbols `i` to `n` of one stream, which must then be consumed
        exactly."""
        var table = BufferView(Span(self._entries))
        var log = self.log
        # Four symbols take at most 44 of the 57 bits a refill readies.
        while i + 4 <= n and r.full():
            r.refill()
            comptime for k in range(4):
                dst.unsafe_set(i + k, Self._symbol(table, r, log))
            i += 4
        while i < n:
            r.refill()
            dst.unsafe_set(i, Self._symbol(table, r, log))
            i += 1
        if not r.finished():
            raise CorruptError("zstd: a literal stream is not consumed exactly")


comptime _PAIR_LOG = 11
"""The bits a two-symbol lookup reads: libzstd's X2 table log for a tree of
at most 11 bits, `HUF_DECODER_FAST_TABLELOG`."""


# ---------------------------------------------------------------------------
# encoding -- ported from libzstd's `huf_compress.c`
# ---------------------------------------------------------------------------


struct HuffmanEncoder(Movable):
    """A literals code: a length and a code per symbol, canonical so the
    decoder rebuilds it from the lengths alone -- `HUF_CElt`."""

    comptime _NODES = 512
    """Leaves sorted by count in `[0, 256)`, internal nodes from `_FIRST`."""
    comptime _FIRST = 256

    var _lengths: List[Int]
    var _codes: List[Int]
    var log: Int
    var top: Int
    """The largest symbol present; its weight goes unsent."""
    var _count: List[Int]
    var _parent: List[Int]
    var _depth: List[Int]
    var _symbol: List[Int]

    def __init__(out self):
        """An encoder that allocates on its first `build`: a frame of one
        block builds one code, and the encoder keeps two."""
        self._lengths = List[Int]()
        self._codes = List[Int]()
        self.log = 0
        self.top = 0
        self._count = List[Int]()
        self._parent = List[Int]()
        self._depth = List[Int]()
        self._symbol = List[Int]()

    def build(mut self, counts: Span[Int, _]):
        """The code for `counts`, the last a symbol present and at least two
        present, no code longer than their optimal depth --
        `HUF_optimalTableLog`, then `HUF_buildCTable`. Node `i` of the tree
        lives at index `i + 1` of the node arrays, so index 0 can hold the
        barrier the build compares against when the leaves run out. Every
        entry the build reads it writes first."""
        var top = len(counts) - 1
        var total = 0
        for c in counts:
            total += c
        var max_bits = Distribution.optimal_log(MAX_WEIGHT, total, top, 1)
        if len(self._lengths) == 0:
            self._lengths.resize(256, 0)
            self._codes.resize(256, 0)
            self._count.resize(unsafe_uninit_length=Self._NODES + 1)
            self._parent.resize(unsafe_uninit_length=Self._NODES + 1)
            self._depth.resize(unsafe_uninit_length=Self._NODES + 1)
            self._symbol.resize(unsafe_uninit_length=Self._NODES + 1)
        self.top = top
        self._sort(counts, top)
        var last = self._tree(top)
        var log = self._limit(last, max_bits)
        self._assign(last, log)

    def _sort(mut self, counts: Span[Int, _], top: Int):
        """The leaves by descending count -- `HUF_sort`: a bucket per count
        up to a cutoff, in symbol order, and a bucket per power of two above
        it, each of those then sorted in place as libzstd does, which decides
        how the tree breaks ties."""
        comptime BUCKETS = 192
        comptime LOG_FROM = BUCKETS - 1 - 32 - 1
        comptime CUTOFF = LOG_FROM + 7  # LOG_FROM + log2_floor(LOG_FROM)

        @always_inline
        def bucket(c: Int) -> Int:
            return c if c < CUTOFF else log2_floor(c) + LOG_FROM

        var base = Array[Int, BUCKETS](fill=0)
        var next = Array[Int, BUCKETS](fill=0)
        for s in range(top + 1):
            base.unsafe_get(bucket(counts.unsafe_get(s))) += 1
        for r in range(BUCKETS - 1, 0, -1):
            base.unsafe_get(r - 1) += base.unsafe_get(r)
            next.unsafe_get(r - 1) = base.unsafe_get(r - 1)
        var count = Span(self._count)
        var symbol = Span(self._symbol)
        for s in range(top + 1):
            var c = counts.unsafe_get(s)
            var r = bucket(c) + 1
            var pos = next.unsafe_get(r)
            next.unsafe_get(r) += 1
            count.unsafe_get(pos + 1) = c
            symbol.unsafe_get(pos + 1) = s
        for r in range(CUTOFF, BUCKETS - 1):
            if next.unsafe_get(r) - base.unsafe_get(r) > 1:
                Self._quick_sort(
                    count[1:],
                    symbol[1:],
                    base.unsafe_get(r),
                    next.unsafe_get(r) - 1,
                )

    def _tree(mut self, top: Int) -> Int:
        """Build an unlimited Huffman tree over the sorted leaves; return the
        last leaf with a count -- `HUF_buildTree`."""
        comptime B = 1  # node i is at index i + B
        var count = Span(self._count)
        var parent = Span(self._parent)
        var depth = Span(self._depth)
        var last = top
        while count.unsafe_get(last + B) == 0:
            last -= 1
        var low_leaf = last
        var low_node = Self._FIRST
        var node = Self._FIRST
        var root = Self._FIRST + last - 1
        count.unsafe_get(node + B) = count.unsafe_get(
            low_leaf + B
        ) + count.unsafe_get(low_leaf - 1 + B)
        parent.unsafe_get(low_leaf + B) = node
        parent.unsafe_get(low_leaf - 1 + B) = node
        node += 1
        low_leaf -= 2
        for i in range(node, root + 1):
            count.unsafe_get(i + B) = 1 << 30
        count.unsafe_get(0) = 1 << 31  # the barrier, node -1
        while node <= root:
            var a: Int
            if count.unsafe_get(low_leaf + B) < count.unsafe_get(low_node + B):
                a = low_leaf
                low_leaf -= 1
            else:
                a = low_node
                low_node += 1
            var b: Int
            if count.unsafe_get(low_leaf + B) < count.unsafe_get(low_node + B):
                b = low_leaf
                low_leaf -= 1
            else:
                b = low_node
                low_node += 1
            count.unsafe_get(node + B) = count.unsafe_get(
                a + B
            ) + count.unsafe_get(b + B)
            parent.unsafe_get(a + B) = node
            parent.unsafe_get(b + B) = node
            node += 1
        depth.unsafe_get(root + B) = 0
        for i in range(root - 1, Self._FIRST - 1, -1):
            depth.unsafe_get(i + B) = (
                depth.unsafe_get(parent.unsafe_get(i + B) + B) + 1
            )
        for i in range(last + 1):
            depth.unsafe_get(i + B) = (
                depth.unsafe_get(parent.unsafe_get(i + B) + B) + 1
            )
        return last

    def _limit(mut self, last: Int, target: Int) -> Int:
        """Cap every code at `target` bits and lengthen the cheapest shorter
        ones until the code is complete again -- `HUF_setMaxHeight`."""
        comptime B = 1
        var depth = Span(self._depth)
        var count = Span(self._count)
        var largest = depth.unsafe_get(last + B)
        if largest <= target:
            return largest
        var cost = 0
        var base_cost = 1 << (largest - target)
        var n = last
        while depth.unsafe_get(n + B) > target:
            cost += base_cost - (1 << (largest - depth.unsafe_get(n + B)))
            depth.unsafe_get(n + B) = target
            n -= 1
        while depth.unsafe_get(n + B) == target:
            n -= 1
        cost >>= largest - target
        comptime NONE = -1
        var rank_last = Array[Int, 14](fill=NONE)
        var current = target
        for pos in range(n, -1, -1):
            if depth.unsafe_get(pos + B) < current:
                current = depth.unsafe_get(pos + B)
                rank_last[target - current] = pos
        while cost > 0:
            var decrease = bit_width(cost)
            while decrease > 1:
                var high_pos = rank_last[decrease]
                var low_pos = rank_last[decrease - 1]
                if high_pos == NONE:
                    decrease -= 1
                    continue
                if low_pos == NONE:
                    break
                if count.unsafe_get(high_pos + B) <= 2 * count.unsafe_get(
                    low_pos + B
                ):
                    break
                decrease -= 1
            while decrease <= 12 and rank_last[decrease] == NONE:
                decrease += 1
            cost -= 1 << (decrease - 1)
            depth.unsafe_get(rank_last[decrease] + B) += 1
            if rank_last[decrease - 1] == NONE:
                rank_last[decrease - 1] = rank_last[decrease]
            if rank_last[decrease] == 0:
                rank_last[decrease] = NONE
            else:
                rank_last[decrease] -= 1
                if (
                    depth.unsafe_get(rank_last[decrease] + B)
                    != target - decrease
                ):
                    rank_last[decrease] = NONE
        while cost < 0:
            if rank_last[1] == NONE:
                while depth.unsafe_get(n + B) == target:
                    n -= 1
                depth.unsafe_get(n + 1 + B) -= 1
                rank_last[1] = n + 1
                cost += 1
                continue
            depth.unsafe_get(rank_last[1] + 1 + B) -= 1
            rank_last[1] += 1
            cost += 1
        return target

    def _assign(mut self, last: Int, log: Int):
        """Canonical codes: each length's codes in symbol order, the longest
        from 0 -- the order the decoder assigns them in."""
        comptime B = 1
        var lengths = Span(self._lengths)
        var codes = Span(self._codes)
        var symbol = Span(self._symbol)
        var depth = Span(self._depth)
        for s in range(256):
            lengths.unsafe_get(s) = 0
        for i in range(last + 1):
            lengths.unsafe_get(symbol.unsafe_get(i + B)) = depth.unsafe_get(
                i + B
            )
        var per_rank = Array[Int, 14](fill=0)
        for s in range(self.top + 1):
            per_rank.unsafe_get(lengths.unsafe_get(s)) += 1
        var start = Array[Int, 14](fill=0)
        var acc = 0
        for bits in range(log, 0, -1):
            start.unsafe_get(bits) = acc
            acc = (acc + per_rank.unsafe_get(bits)) >> 1
        for s in range(self.top + 1):
            var bits = lengths.unsafe_get(s)
            if bits > 0:
                codes.unsafe_get(s) = start.unsafe_get(bits)
                start.unsafe_get(bits) += 1
        self.log = log

    def cost(self, counts: Span[Int, _]) -> Int:
        """The bytes the literals would take with this code, which must
        cover them -- `HUF_estimateCompressedSize`."""
        var bits = 0
        for s in range(len(counts)):
            bits += self._lengths[s] * counts[s]
        return bits >> 3

    def covers(self, counts: Span[Int, _]) -> Bool:
        """Whether every symbol `counts` has has a code here --
        `HUF_validateCTable`."""
        if len(counts) - 1 > self.top:
            return False
        for s in range(len(counts)):
            if counts[s] != 0 and self._lengths[s] == 0:
                return False
        return True

    def write_tree(self, mut out: List[UInt8]) raises CorruptError -> Bool:
        """Append the tree description: the weights of every symbol but the
        last, FSE-compressed when that is less than half the 4-bit form --
        `HUF_writeCTable`; or append nothing and answer False when they fit
        neither form."""
        var weights = List[Int](capacity=self.top)
        for s in range(self.top):
            var bits = self._lengths[s]
            weights.append(self.log + 1 - bits if bits > 0 else 0)
        var at = len(out)
        out.append(0)
        var size = Self._compressed_weights(Span(weights), out)
        if size > 1 and size < self.top // 2:
            out[at] = UInt8(size)
            return True
        out.shrink(at)
        if self.top > 128:
            return False
        out.append(UInt8(127 + self.top))
        for i in range(0, self.top, 2):
            var second = weights[i + 1] if i + 1 < self.top else 0
            out.append(UInt8(weights[i] << 4 | second))
        return True

    def encode(self, src: Span[UInt8, _], streams: Int, mut out: List[UInt8]):
        """Append `src` coded as 1 stream, or as 4 behind a jump table --
        `HUF_compress1X_usingCTable` and `HUF_compress4X_usingCTable`. Each
        stream is written in place, so `out` first grows by a bound on all
        of them, and the 8 bytes every flush stores."""
        var table = Array[UInt64, 256](fill=0)
        for s in range(self.top + 1):
            var bits = self._lengths[s]
            if bits > 0:
                table[s] = UInt64(self._codes[s]) << UInt64(64 - bits) | UInt64(
                    bits
                )
        var n = len(src)
        var at = len(out)
        out.resize(unsafe_uninit_length=at + 6 + n * self.log // 8 + 64)
        var dst = Span(out)
        var end: Int
        if streams == 1:
            end = Self._stream(table, self.log, src, dst, at)
        else:
            var seg = (n + 3) // 4
            var op = at + 6
            for k in range(3):
                var next = Self._stream(
                    table, self.log, src[k * seg : (k + 1) * seg], dst, op
                )
                LittleEndian.store[DType.uint16](
                    dst, at + 2 * k, UInt16(next - op)
                )
                op = next
            end = Self._stream(table, self.log, src[3 * seg :], dst, op)
        out.shrink(end)

    @staticmethod
    def _quick_sort(
        count: Span[mut=True, Int, _],
        symbol: Span[mut=True, Int, _],
        var low: Int,
        var high: Int,
    ):
        """Leaves `low` to `high` by descending count -- `HUF_simpleQuickSort`:
        the rightmost as pivot, and an insertion sort below 9 leaves. It is not
        stable, and the order it leaves equal counts in is libzstd's."""
        if high - low < 8:
            for i in range(low + 1, high + 1):
                var c = count.unsafe_get(i)
                var sym = symbol.unsafe_get(i)
                var j = i - 1
                while j >= low and count.unsafe_get(j) < c:
                    count.unsafe_get(j + 1) = count.unsafe_get(j)
                    symbol.unsafe_get(j + 1) = symbol.unsafe_get(j)
                    j -= 1
                count.unsafe_get(j + 1) = c
                symbol.unsafe_get(j + 1) = sym
            return
        while low < high:
            var pivot = count.unsafe_get(high)
            var i = low - 1
            for j in range(low, high):
                if count.unsafe_get(j) > pivot:
                    i += 1
                    Self._swap(count, symbol, i, j)
            Self._swap(count, symbol, i + 1, high)
            var at = i + 1
            if at - low < high - at:
                Self._quick_sort(count, symbol, low, at - 1)
                low = at + 1
            else:
                Self._quick_sort(count, symbol, at + 1, high)
                high = at - 1

    @staticmethod
    @always_inline
    def _swap(
        count: Span[mut=True, Int, _],
        symbol: Span[mut=True, Int, _],
        a: Int,
        b: Int,
    ):
        var c = count.unsafe_get(a)
        count.unsafe_get(a) = count.unsafe_get(b)
        count.unsafe_get(b) = c
        var sym = symbol.unsafe_get(a)
        symbol.unsafe_get(a) = symbol.unsafe_get(b)
        symbol.unsafe_get(b) = sym

    @staticmethod
    def _stream(
        table: Array[UInt64, 256],
        log: Int,
        src: Span[UInt8, _],
        dst: Span[mut=True, UInt8, _],
        at: Int,
    ) -> Int:
        """Write `src` as one stream at `dst[at:]`; return where it ends. The
        symbols a flush takes, by the longest code -- libzstd's table."""
        if log == 11:
            return Self._stream_unrolled[5, False](table, src, dst, at)
        elif log == 10:
            return Self._stream_unrolled[5, True](table, src, dst, at)
        elif log == 9:
            return Self._stream_unrolled[6, False](table, src, dst, at)
        elif log == 8:
            return Self._stream_unrolled[7, False](table, src, dst, at)
        elif log == 7:
            return Self._stream_unrolled[8, False](table, src, dst, at)
        else:
            return Self._stream_unrolled[9, True](table, src, dst, at)

    @staticmethod
    def _stream_unrolled[
        unroll: Int, last_fast: Bool
    ](
        table: Array[UInt64, 256],
        src: Span[UInt8, _],
        dst: Span[mut=True, UInt8, _],
        at: Int,
    ) -> Int:
        """`HUF_compress1X_usingCTable_internal_body_loop`: the symbols from
        the last, `unroll` into one container and the next `unroll` into a
        second, independent of the first, then merged into it -- two chains of
        shifts the processor overlaps."""
        var c = _Container(0, 0)
        var op = at
        var n = len(src)

        @always_inline
        def code(i: Int) {src, table} -> UInt64:
            return table.unsafe_get(Int(src.unsafe_get(i)))

        var rem = n % unroll
        if rem > 0:
            for _ in range(rem):
                n -= 1
                c.add[False](code(n))
            c.flush(dst, op)
        if n % (2 * unroll) != 0:
            comptime for u in range(1, unroll):
                c.add[True](code(n - u))
            c.add[last_fast](code(n - unroll))
            c.flush(dst, op)
            n -= unroll
        while n > 0:
            comptime for u in range(1, unroll):
                c.add[True](code(n - u))
            c.add[last_fast](code(n - unroll))
            c.flush(dst, op)
            var c1 = _Container(0, 0)
            comptime for u in range(1, unroll):
                c1.add[True](code(n - unroll - u))
            c1.add[last_fast](code(n - 2 * unroll))
            c.merge(c1)
            c.flush(dst, op)
            n -= 2 * unroll
        # The end mark: a single 1 bit.
        c.add[False](UInt64(1) << 63 | 1)
        c.flush(dst, op)
        return op + Int(c.used & 0xFF > 0)

    @staticmethod
    def _compressed_weights(
        weights: Span[Int, _], mut out: List[UInt8]
    ) raises CorruptError -> Int:
        """Append `weights` FSE-compressed with two interleaved states and
        return the bytes taken, or 0, appending nothing, when that cannot pay --
        `HUF_compressWeights`."""
        var n = len(weights)
        if n <= 2:
            return 0
        var counts = List[Int](length=MAX_WEIGHT + 1, fill=0)
        var top = 0
        var most = 0
        for w in weights:
            counts[w] += 1
            top = max(top, w)
        for c in counts:
            most = max(most, c)
        if most == n or most == 1:
            return 0
        counts.shrink(top + 1)
        var dist = Distribution()
        dist.normalize(
            Span(counts),
            n,
            Distribution.optimal_log(Alphabet.WEIGHTS.max_log(), n, top),
            False,
        )
        var start = len(out)
        dist.write(out)
        var table = FseEncoder()
        table.build(dist)
        var at = len(out)
        out.resize(unsafe_uninit_length=at + n + 24)
        var w = BitWriter(Span(out), at)
        # Two states take turns from the last weight, the first of them taking
        # the last when there are an odd number.
        var odd = n & 1
        var s1 = table.start(weights[n - 2 + odd])
        var s2 = table.start(weights[n - 1 - odd])
        var i = n - 2
        if odd:
            i -= 1
            s1.encode(weights[i], w)
            w.flush()
        while i > 0:
            i -= 1
            s2.encode(weights[i], w)
            i -= 1
            s1.encode(weights[i], w)
            w.flush()
        s2.flush(w)
        s1.flush(w)
        out.shrink(w.close())
        return len(out) - start


@fieldwise_init
struct _Container(TrivialRegisterPassable):
    """The bits of a stream not yet stored, shifted in from the top -- one
    of libzstd's `HUF_CStream_t` containers. A code is its value above its
    length; only `used`'s low byte counts, so a fast `add` ORs the length
    in with the code, leaving dirty low bits a later shift moves out."""

    var bits: UInt64
    var used: UInt64

    @always_inline
    def add[fast: Bool](mut self, code: UInt64):
        """Shift `code` in -- `HUF_addBits`."""
        self.bits >>= code & 63
        self.bits |= code if fast else code & ~UInt64(0xFF)
        self.used += code

    @always_inline
    def merge(mut self, other: Self):
        """Shift in what `other` holds -- `HUF_mergeIndex1`: a second
        container filled independently, so the two chains of shifts
        overlap."""
        self.bits >>= other.used & 0xFF
        self.bits |= other.bits
        self.used += other.used

    @always_inline
    def flush(mut self, dst: Span[mut=True, UInt8, _], mut op: Int):
        """Store the whole bytes held at `op`, low first -- `HUF_flushBits`:
        8 bytes are written, and the bits past the whole bytes are stored
        again by the next flush."""
        var n = self.used & 0xFF
        LittleEndian.store[DType.uint64](dst, op, self.bits >> ((64 - n) & 63))
        op += Int(n >> 3)
        self.used &= 7


# ---------------------------------------------------------------------------
# A literals section: its header, and choosing how to code it
# ---------------------------------------------------------------------------


@fieldwise_init
struct LiteralsHeader(ImplicitlyCopyable, Movable):
    """A literals section's header: its type, how many Huffman streams, how
    many literals, the header's own bytes, and the bytes after it -- the
    literals themselves, their one repeated byte, or their streams."""

    var kind: Int
    var streams: Int
    var n: Int
    var length: Int
    """The header's own bytes."""
    var size: Int
    """The bytes after it."""

    comptime RAW = 0
    comptime RLE = 1
    comptime COMPRESSED = 2
    """Huffman-coded, with the tree."""
    comptime TREELESS = 3
    """Huffman-coded with the previous block's tree."""

    @staticmethod
    def raw(n: Int) -> Self:
        """The header of `n` literals stored as they are."""
        return Self(Self.RAW, 1, n, Self._plain_length(n), n)

    @staticmethod
    def rle(n: Int) -> Self:
        """The header of `n` copies of one byte."""
        return Self(Self.RLE, 1, n, Self._plain_length(n), 1)

    @staticmethod
    def coded(n: Int, streams: Int) -> Self:
        """The header of `n` Huffman-coded literals in `streams` streams,
        with a tree; its type and size are set once the code is."""
        var header = 3 + Int(n >= 1024) + Int(n >= 16384)
        return Self(Self.COMPRESSED, streams, n, header, 0)

    @staticmethod
    def _plain_length(n: Int) -> Int:
        return 1 if n < 32 else (2 if n < 4096 else 3)

    @staticmethod
    def read(block: Span[UInt8, _]) raises CorruptError -> Self:
        """The header the block starts with; the section must fit in it."""
        var n_in = len(block)
        if n_in == 0:
            raise CorruptError("zstd: an empty compressed block")
        var b0 = Int(block[0])
        var kind = b0 & 3
        var size_format = (b0 >> 2) & 3
        if kind <= Self.RLE:
            var n: Int
            var header: Int
            if size_format & 1 == 0:
                n = b0 >> 3
                header = 1
            elif size_format == 1:
                if n_in < 2:
                    raise CorruptError("zstd: truncated literals header")
                n = (b0 >> 4) + (Int(block[1]) << 4)
                header = 2
            else:
                if n_in < 3:
                    raise CorruptError("zstd: truncated literals header")
                n = (b0 >> 4) + (Int(block[1]) << 4) + (Int(block[2]) << 12)
                header = 3
            var size = n if kind == Self.RAW else 1
            if size > n_in - header:
                raise CorruptError(
                    "zstd: truncated raw literals" if kind
                    == Self.RAW else "zstd: truncated RLE literals"
                )
            return Self(kind, 1, n, header, size)
        var header = 3 if size_format <= 1 else size_format + 2
        if n_in < header:
            raise CorruptError("zstd: truncated literals header")
        var v = Int(LittleEndian.partial[DType.uint64](block[:header], 0))
        var width = Self._width(header)
        var n = (v >> 4) & ((1 << width) - 1)
        var size = (v >> (4 + width)) & ((1 << width) - 1)
        if size > n_in - header:
            raise CorruptError("zstd: truncated Huffman literals")
        var streams = 1 if size_format == 0 else 4
        if streams == 4 and n < 6:
            raise CorruptError(t"zstd: {n} literals in 4 streams")
        return Self(kind, streams, n, header, size)

    def end(self) -> Int:
        """Where the section ends."""
        return self.length + self.size

    @staticmethod
    def _width(header: Int) -> Int:
        """The bits a Huffman-coded section's header gives its literal count
        and its size each: 10, 14 or 18."""
        return 4 * header - 2

    def _value(self) -> Int:
        """The header as `read` parses it: the type, the size format -- for
        stored or repeated literals 0, 1 or 3 by the header's length, for
        coded ones the streams or the length -- and the counts after it."""
        if self.kind <= Self.RLE:
            var format = self.length - 1 + Int(self.length == 3)
            return (
                self.kind | format << 2 | self.n << (3 + Int(self.length > 1))
            )
        else:
            var format = (
                Int(self.streams == 4) if self.length == 3 else self.length - 2
            )
            return (
                self.kind
                | format << 2
                | self.n << 4
                | self.size << (4 + Self._width(self.length))
            )

    def write(self, mut out: List[UInt8]):
        """Append the header."""
        LittleEndian.append(out, UInt64(self._value()), self.length)

    def write_at(self, mut out: List[UInt8], at: Int):
        """Write the header over the bytes reserved for it at `at`."""
        LittleEndian.write(out, at, UInt64(self._value()), self.length)


@always_inline
def min_gain(n: Int) -> Int:
    """What coding `n` bytes must save to be worth it -- `ZSTD_minGain` at
    the fast strategy; the one rule the block and its literals share."""
    return (n >> 6) + 2


struct LiteralsEncoder(Movable):
    """A block's literals section -- `ZSTD_compressLiterals` at the fast
    strategy: stored, one repeated byte, or Huffman-coded, with a code of
    its own or the one the decoder holds from an earlier block."""

    var _huffman: List[HuffmanEncoder]
    """Two codes: the decoder's, at `_held`, and one to build into."""
    var _held: Int
    """Which of `_huffman` the decoder holds from the last block that sent
    one, -1 before any did."""
    var _hist: Histogram[]

    def __init__(out self):
        self._huffman = List[HuffmanEncoder](capacity=2)
        self._huffman.append(HuffmanEncoder())
        self._huffman.append(HuffmanEncoder())
        self._held = -1
        self._hist = Histogram()

    def adopt(mut self):
        """The block whose literals sent a new code went out compressed: the
        decoder now holds that code."""
        self._held = Int(self._held == 0)

    def encode(
        mut self, lits: Span[UInt8, _], suspect: Bool, mut out: List[UInt8]
    ) raises CorruptError -> Bool:
        """Append the section for `lits`; return whether it sends a new
        code, which `adopt` makes the decoder's once the block is sent.
        Under 64 literals, or when coding them does not pay, they are
        stored; `suspect` ones are sampled for that first."""
        var n = len(lits)
        var at = len(out)
        var coded = -1
        if n >= 64:
            coded = self._coded(lits, suspect, out)
        if coded < 0:
            out.shrink(at)
            LiteralsHeader.raw(n).write(out)
            out.extend(lits)
        return coded == 1

    @staticmethod
    def _largest(sample: Span[UInt8, _], limit: Int) -> Int:
        """The largest count of a byte in `sample`, or -1 as soon as one
        passes `limit` -- which compressible literals do within a few
        hundred bytes, so most samples are not counted through, where
        libzstd's `HIST_count_simple` counts all of them."""
        var counts = Array[Int32, 256](fill=0)
        for i in range(len(sample)):
            var b = Int(sample.unsafe_get(i))
            var c = counts.unsafe_get(b) + 1
            if Int(c) > limit:
                return -1
            counts.unsafe_get(b) = c
        var most = 0
        for s in range(256):
            most = max(most, Int(counts.unsafe_get(s)))
        return most

    def _coded(
        mut self, lits: Span[UInt8, _], suspect: Bool, mut out: List[UInt8]
    ) raises CorruptError -> Int:
        """Append `lits` as one repeated byte or Huffman-coded --
        `HUF_compress_internal` at the fast strategy, which repeats the
        decoder's code outright for at most 1 KiB of literals. Return 1 if
        that sends a new code, 0 if not, and -1 when they must be stored,
        what was appended then left for the caller to drop."""
        var n = len(lits)
        # Suspect literals of 40 KiB or more are stored, the rest uncounted,
        # when the counts of the most frequent byte in their first 4 KiB and
        # in their last add up to at most 68: 1/128th of the 8 KiB, plus 4.
        # A 4 KiB sample has some byte 16 times, so neither may pass 52.
        comptime SAMPLE = 4096
        if suspect and n >= 10 * SAMPLE:
            var first = Self._largest(lits[:SAMPLE], 52)
            if (
                first >= 0
                and Self._largest(lits[n - SAMPLE :], 68 - first) >= 0
            ):
                return -1
        self._hist.count(lits)
        var most = self._hist.most
        if most == n:
            LiteralsHeader.rle(n).write(out)
            out.append(lits[0])
            return 0
        if most <= (n >> 7) + 4:
            return -1
        var counts = self._hist.present()
        var held = self._held
        var reuse = held >= 0 and self._huffman[held].covers(counts)
        var header = LiteralsHeader.coded(n, 1 if n < 256 else 4)
        var at = len(out)
        out.resize(at + header.length, 0)
        var code = held
        if not reuse or n > 1024:
            var fresh = Int(held == 0)
            self._huffman[fresh].build(counts)
            if not self._huffman[fresh].write_tree(out):
                return -1
            var tree = len(out) - at - header.length
            if reuse and (
                self._huffman[held].cost(counts)
                <= tree + self._huffman[fresh].cost(counts)
                or tree + 12 >= n
            ):
                out.shrink(at + header.length)
            elif tree + 12 >= n:
                return -1
            else:
                code = fresh
        self._huffman[code].encode(lits, header.streams, out)
        header.size = len(out) - at - header.length
        if header.size >= n - min_gain(n):
            return -1
        var sent = code != held
        header.kind = (
            LiteralsHeader.COMPRESSED if sent else LiteralsHeader.TREELESS
        )
        header.write_at(out, at)
        return Int(sent)


struct Huffman(Compressor):
    """Bytes Huffman-coded as Zstandard codes a block's literals, in
    sections of up to 128 KiB, each with a tree of its own -- or stored, or
    one byte repeated, where that is smaller."""

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """2^18 times as many: the densest section, one byte repeated behind
        a 3-byte header, makes up to 2^20 bytes of 4."""
        return n << 18

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append `src` to `dst` as sections of up to `BLOCK_MAX` bytes."""
        var lo = 0
        while lo < len(src):
            var n = min(BLOCK_MAX, len(src) - lo)
            var coder = LiteralsEncoder()
            _ = coder.encode(src[lo : lo + n], False, dst)
            lo += n

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        """Decode the sections of `src` into exactly `dst`."""
        var into = BufferView(dst)
        var table = HuffmanTable()
        var pos = 0
        var done = 0
        while done < len(dst):
            var section = LiteralsHeader.read(src[pos:])
            if section.n == 0:
                raise CorruptError("zstd: an empty Huffman section")
            table.decode_section(
                section, src[pos:], into[done:], len(dst) - done
            )
            done += section.n
            pos += section.end()
        if pos != len(src):
            raise CorruptError(
                t"zstd: {len(src)} Huffman bytes, {pos} of them read"
            )
