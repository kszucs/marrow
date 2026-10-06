# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# The section encoders follow libzstd's `zstd_compress.c`,
# `zstd_compress_literals.c` and `zstd_compress_sequences.c` at its fast
# strategy, `Histogram` its `hist.c`, and `BlockEncoder.split` is
# `zstd_preSplit.c`'s `ZSTD_splitBlock_fromBorders`, Copyright (c) Meta
# Platforms, Inc. (BSD 3-Clause); see NOTICE.txt.

"""Compressing a zstd block.

| Type | What it is |
|---|---|
| `BlockEncoder` | what a frame's blocks share on the way in: matching, and the block around the sections |
| `LiteralsEncoder` | the literals section, and the Huffman code the decoder holds |
| `SequencesEncoder` | the sequences section: each field's table, then the bitstream |
| `Histogram` | each symbol's count over some input |

The matches are `matcher`'s, reported into a `SeqStore` that keeps the
literals and the sequences, each offset already as it is sent. The literals
are Huffman-coded unless that does not pay, with the code the decoder holds
from an earlier block where that is no dearer than sending a new one; each
sequence field gets the predefined table, a one-symbol table or one built
for the block, by libzstd's fast-strategy rules. A block that does not
shrink is stored raw, and one of a single repeated byte as an RLE block --
unless it is the frame's first, or codes to 25 bytes or more. Where a full
block's two ends look unlike each other, `BlockEncoder.split` ends it early,
so each part gets tables of its own.
"""

from ...views import BufferView
from ..lz77 import LzCopy, match_length
from .bits import BitWriter
from .fse import MAX_WEIGHT, Alphabet, Distribution, FseEncoder
from .headers import BLOCK_MAX, BlockHeader, LiteralsHeader, SequencesHeader
from .huffman import HuffmanEncoder
from .matcher import FastMatcher, SeqStore
from ...codecs.byteorder import LittleEndian


@always_inline
def _min_gain(n: Int) -> Int:
    """What coding `n` bytes must save to be worth it -- `ZSTD_minGain` at
    the fast strategy; the one rule the block and its literals share."""
    return (n >> 6) + 2


struct BlockEncoder(Movable):
    """What a frame's blocks share on the way in: the match finder and its
    table, the scratch it reports into, and the two sections' encoders."""

    comptime _SEGMENT = 512
    """The bytes `split` samples at each end of a block, and in its
    middle."""

    var _matcher: FastMatcher
    var _lits: List[UInt8]
    var _seqs: List[UInt32]
    """Each sequence's literal count, offset code and match length."""
    var _literals: LiteralsEncoder
    var _sequences: SequencesEncoder

    def __init__(out self, n: Int):
        """For an input of `n` bytes."""
        var size = min(n, BLOCK_MAX)
        # A sequence takes at least 4 bytes, its match's.
        var max_seqs = size // 4 + 1
        self._matcher = FastMatcher(n)
        # Scratch the store writes before anything reads it, and the `SLOP`
        # bytes its literal copies may write past the literals.
        self._lits = List[UInt8](unsafe_uninit_length=size + LzCopy.SLOP)
        self._seqs = List[UInt32](unsafe_uninit_length=3 * max_seqs)
        self._literals = LiteralsEncoder()
        self._sequences = SequencesEncoder(max_seqs)

    def window(self) -> Int:
        """The farthest a match reaches back, in bytes."""
        return self._matcher.window()

    @staticmethod
    def split(block: Span[UInt8, _]) -> Int:
        """How much of a full block to encode as one: all of it, unless its
        first and last 512 bytes differ enough, and then 32, 64 or 96 KiB, by
        which end its middle 512 bytes are nearer -- libzstd's
        `ZSTD_splitBlock_fromBorders`, which its fast strategy splits by."""
        comptime S = Self._SEGMENT
        var n = len(block)
        var first = Histogram()
        var last = Histogram()
        first.count(block[:S])
        last.count(block[n - S :])
        # libzstd weighs each count by the other sample's size, and its
        # thresholds by both sizes; the samples are all `S` bytes, so here
        # neither is. Two count as different from a distance of 14/16 of
        # `S` -- `compareFingerprints` without a penalty.
        if first.distance(last) < S * 14 // 16:
            return n
        var middle = Histogram()
        var mid = n // 2 - S // 2
        middle.count(block[mid : mid + S])
        var from_first = first.distance(middle)
        var from_last = last.distance(middle)
        if 3 * abs(from_first - from_last) < S:
            return 64 << 10
        return (32 << 10) if from_first > from_last else (96 << 10)

    def encode(
        mut self,
        input: Span[mut=False, UInt8, _],
        start: Int,
        end: Int,
        last: Bool,
        mut out: List[UInt8],
    ) raises:
        """Append the block holding `input[start:end]`, at most `BLOCK_MAX`
        bytes; its matches may reach back to the start of `input`."""
        var src = input[start:end]
        var n = len(src)
        var at = len(out)
        var size = 0
        var sent = False
        # Under 7 bytes nothing is tried, as in libzstd.
        if n >= 7:
            out.resize(at + BlockHeader.SIZE, 0)
            var store = SeqStore(Span(self._lits), Span(self._seqs))
            self._matcher.compress(input, start, end, store)
            size, sent = self._sections(store.n_lits, store.n_seqs, out)
            if size >= n - _min_gain(n):
                size = 0
        # One repeated byte that coded to under 25 bytes, or not at all, is
        # an RLE block -- after matching, whose table keeps the block's
        # positions. Never the first: the zstd tool before 1.4.4 rejected
        # one.
        # Each byte equal to the one before, compared 8 at a time --
        # `ZSTD_isRLE`.
        var rle = (
            n >= 7
            and size < 25
            and start > 0
            and match_length(src, 1, 0, n) == n - 1
        )
        if size == 0 or rle:
            # The decoder keeps its recent offsets and its Huffman code
            # across a raw or RLE block.
            out.shrink(at)
            if rle:
                BlockHeader(last, BlockHeader.RLE, n).write(out)
                out.append(src[0])
            else:
                BlockHeader(last, BlockHeader.RAW, n).write(out)
                out.extend(src)
        else:
            self._matcher.commit()
            if sent:
                self._literals.adopt()
            BlockHeader(last, BlockHeader.COMPRESSED, size).write_at(out, at)

    def _sections(
        mut self, n_lits: Int, n_seqs: Int, mut out: List[UInt8]
    ) raises -> Tuple[Int, Bool]:
        """Append the literals and sequences sections; return their size,
        or 0 when the block must be stored raw, and whether the literals
        send a new Huffman code."""
        var start = len(out)
        # Far more literals than matches suggests noise -- libzstd's
        # `suspectUncompressible`.
        var suspect = n_seqs == 0 or n_lits // n_seqs >= 20
        var sent = self._literals.encode(
            Span(self._lits)[:n_lits], suspect, out
        )
        if not self._sequences.encode(Span(self._seqs), n_seqs, out):
            out.shrink(start)
            return (0, False)
        return (len(out) - start, sent)


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
    ) raises -> Bool:
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
    ) raises -> Int:
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
        var top = self._hist.top
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
        out.resize(at + header.header, 0)
        var code = held
        if not reuse or n > 1024:
            var fresh = Int(held == 0)
            self._huffman[fresh].build(
                counts, top, Distribution.optimal_log(MAX_WEIGHT, n, top, 1)
            )
            try:
                self._huffman[fresh].write_tree(out)
            except:
                return -1
            var tree = len(out) - at - header.header
            if reuse and (
                self._huffman[held].cost(counts)
                <= tree + self._huffman[fresh].cost(counts)
                or tree + 12 >= n
            ):
                out.shrink(at + header.header)
            elif tree + 12 >= n:
                return -1
            else:
                code = fresh
        self._huffman[code].encode(lits, header.streams, out)
        header.size = len(out) - at - header.header
        if header.size >= n - _min_gain(n):
            return -1
        var sent = code != held
        header.kind = (
            LiteralsHeader.COMPRESSED if sent else LiteralsHeader.TREELESS
        )
        header.write_at(out, at)
        return Int(sent)


struct SequencesEncoder(Movable):
    """A block's sequences section -- `ZSTD_compressSequences` at the fast
    strategy: each field's code per sequence, a table chosen for each
    field, then the bitstream."""

    var _code: List[UInt8]
    """Each sequence's literal length codes, then its offset codes, then its
    match length codes."""
    var _hist: Histogram[64]
    var _dist: Distribution
    var _tables: List[FseEncoder]
    """Each field's table for the block in `0..2`, the predefined ones,
    built once when first used, in `3..5`."""

    def __init__(out self, max_seqs: Int):
        """For blocks of at most `max_seqs` sequences."""
        self._code = List[UInt8](capacity=3 * max_seqs)
        self._hist = Histogram[64]()
        self._dist = Distribution()
        self._tables = List[FseEncoder](capacity=6)
        for _ in range(6):
            self._tables.append(FseEncoder())

    def encode(
        mut self, seqs: Span[UInt32, _], n: Int, mut out: List[UInt8]
    ) raises -> Bool:
        """Append the section for the first `n` sequences of `seqs`; `False`
        when it trips the old decoders' small-table bug and the block must
        be stored."""
        var header = SequencesHeader.of(n)
        var at = len(out)
        out.resize(at + header.length, 0)
        if n == 0:
            header.write_at(out, at)
            return True
        self._codes(seqs, n)
        var last_table = 0
        # Which of `_tables` each field uses: its own, or the predefined.
        var use = Array[Int, 3](fill=0)
        comptime for k in range(3):
            comptime field = Alphabet(k)
            var mode, size = self._table(field, n, out)
            header.set_mode(field, mode)
            use[k] = 3 + k if mode == SequencesHeader.PREDEFINED else k
            if mode == SequencesHeader.COMPRESSED:
                last_table = size
        header.write_at(out, at)
        var stream = self._bitstream(seqs, n, use, out)
        # libzstd before 1.3.5 rejected a block whose last table and
        # bitstream together came to under 4 bytes.
        return not (last_table > 0 and last_table + stream < 4)

    def _codes(mut self, seqs: Span[UInt32, _], n: Int):
        """Each sequence's three codes -- `ZSTD_seqToCodes`."""
        self._code.resize(3 * n, 0)
        var s = BufferView(seqs)
        var code = BufferView(Span(self._code))
        for i in range(n):
            var ll = Int(s.unsafe_get(3 * i))
            var off = Int(s.unsafe_get(3 * i + 1))
            var ml = Int(s.unsafe_get(3 * i + 2))
            code.unsafe_set(i, UInt8(Alphabet.LITERAL_LENGTHS.code(ll)))
            code.unsafe_set(n + i, UInt8(Alphabet.OFFSETS.code(off)))
            code.unsafe_set(2 * n + i, UInt8(Alphabet.MATCH_LENGTHS.code(ml)))

    def _table(
        mut self, field: Alphabet, n: Int, mut out: List[UInt8]
    ) raises -> Tuple[Int, Int]:
        """Choose and prepare `field`'s table for `n` sequences; return its
        mode and the size of the description it appended --
        `ZSTD_selectEncodingType` and `ZSTD_buildCTable` at the fast
        strategy."""
        var k = field.index
        var codes = Span(self._code)[k * n : (k + 1) * n]
        ref table = self._tables[k]
        self._hist.count(codes)
        var top = self._hist.top
        var most = self._hist.most
        var log0 = field.default_log()
        # The predefined offsets distribution stops at code 28.
        var default_ok = field != Alphabet.OFFSETS or top <= 28
        if most == n and not (default_ok and n <= 2):
            out.append(UInt8(top))
            table.rle(top)
            return (SequencesHeader.RLE, 1)
        # The fast strategy's thresholds: few sequences, or no dominant
        # code, and the predefined table is close enough.
        var few = ((1 << log0) * 9) >> 3
        if default_ok and (most == n or n < few or most < (n >> (log0 - 1))):
            if self._tables[3 + k].log == 0:
                self._tables[3 + k].build(Distribution.predefined(field))
            return (SequencesHeader.PREDEFINED, 0)
        var log = Distribution.optimal_log(field.max_log(), n, top)
        var total = n
        var last = Int(codes[n - 1])
        # The last sequence's code starts the stream and costs no bits.
        if self._hist.counts[last] > 1:
            self._hist.counts[last] -= 1
            total -= 1
        self._dist.normalize(self._hist.present(), total, log, total >= 2048)
        var at = len(out)
        self._dist.write(out)
        table.build(self._dist)
        return (SequencesHeader.COMPRESSED, len(out) - at)

    def _bitstream(
        self,
        seqs: Span[UInt32, _],
        n: Int,
        use: Array[Int, 3],
        mut out: List[UInt8],
    ) -> Int:
        """Append the `n` sequences, last first, so the decoder reads the
        first first -- `ZSTD_encodeSequences`; return the stream's length.
        Everything is read through local spans and states: through `self`,
        each flush's store would make the compiler reload every table."""
        var s = BufferView(seqs)
        var code = BufferView(Span(self._code))
        # A sequence is at most 26 bits of states and 63 of extras; and the
        # last flush stores 8 bytes.
        var at = len(out)
        out.resize(unsafe_uninit_length=at + 12 * n + 24)
        var w = BitWriter(Span(out), at)
        var k = n - 1
        var ml = self._tables[use[2]].start(Int(code.unsafe_get(2 * n + k)))
        var of = self._tables[use[1]].start(Int(code.unsafe_get(n + k)))
        var ll = self._tables[use[0]].start(Int(code.unsafe_get(k)))
        Self._extras(s, code, k, n, w, True)
        while k > 0:
            k -= 1
            of.encode(Int(code.unsafe_get(n + k)), w)
            ml.encode(Int(code.unsafe_get(2 * n + k)), w)
            ll.encode(Int(code.unsafe_get(k)), w)
            Self._extras(s, code, k, n, w, False)
        ml.flush(w)
        of.flush(w)
        ll.flush(w)
        var end = w.close()
        out.shrink(end)
        return end - at

    @staticmethod
    @always_inline
    def _extras(
        seqs: BufferView[DType.uint32, _],
        code: BufferView[DType.uint8, _],
        k: Int,
        n: Int,
        mut w: BitWriter[_],
        first: Bool,
    ):
        """Sequence `k`'s extra bits: literal length, match length, offset,
        flushed so the three state transitions before them still fit."""
        comptime LL = Alphabet.LITERAL_LENGTHS
        comptime ML = Alphabet.MATCH_LENGTHS
        var ll_code = Int(code.unsafe_get(k))
        var ml_code = Int(code.unsafe_get(2 * n + k))
        var of_bits = Int(code.unsafe_get(n + k))
        var ll_bits = LL.extra(ll_code)
        var ml_bits = ML.extra(ml_code)
        var total = ll_bits + ml_bits + of_bits
        if not first and total >= 31:
            w.flush()
        w.add_fitting(
            Int(seqs.unsafe_get(3 * k)) - Int(LL.base(ll_code)), ll_bits
        )
        w.add_fitting(
            Int(seqs.unsafe_get(3 * k + 2)) - Int(ML.base(ml_code)), ml_bits
        )
        if total > 56:
            w.flush()
        w.add_fitting(Int(seqs.unsafe_get(3 * k + 1)) - (1 << of_bits), of_bits)
        w.flush()


struct Histogram[bins: Int = 256](Movable):
    """Each symbol's count over some input, the largest symbol present and
    the largest count -- `HIST_count`. Symbols are bytes; a histogram of
    sequence codes keeps the first 64."""

    var counts: List[Int]
    var top: Int
    var most: Int

    def __init__(out self):
        self.counts = List[Int](length=Self.bins, fill=0)
        self.top = 0
        self.most = 0

    def count(mut self, src: Span[UInt8, _]):
        """Count `src` -- `HIST_countFast`, which counts under 1500 bytes
        one at a time and more into four tables in turn, so a byte that
        recurs is not one chain of dependent increments. This takes 16
        bytes a step into eight: a byte recurring every 8, the top of an
        int64 or a float64, would chain in four. Eight arrays rather than
        one of eight rows, so each has a base register of its own; that
        takes an add off every count, and brings strings from 1.15x
        libzstd's time to 0.95x."""
        var n = len(src)
        var counts = Span(self.counts)
        if n < 1500:
            for s in range(Self.bins):
                counts.unsafe_get(s) = 0
            for i in range(n):
                counts.unsafe_get(Int(src.unsafe_get(i))) += 1
        else:
            var t0 = Array[UInt32, Self.bins](fill=0)
            var t1 = Array[UInt32, Self.bins](fill=0)
            var t2 = Array[UInt32, Self.bins](fill=0)
            var t3 = Array[UInt32, Self.bins](fill=0)
            var t4 = Array[UInt32, Self.bins](fill=0)
            var t5 = Array[UInt32, Self.bins](fill=0)
            var t6 = Array[UInt32, Self.bins](fill=0)
            var t7 = Array[UInt32, Self.bins](fill=0)
            var i = 0
            while i + 16 <= n:
                var a = LittleEndian.fixed[DType.uint64](src, i)
                var b = LittleEndian.fixed[DType.uint64](src, i + 8)
                t0.unsafe_get(Int(a & 0xFF)) += 1
                t1.unsafe_get(Int((a >> 8) & 0xFF)) += 1
                t2.unsafe_get(Int((a >> 16) & 0xFF)) += 1
                t3.unsafe_get(Int((a >> 24) & 0xFF)) += 1
                t0.unsafe_get(Int((a >> 32) & 0xFF)) += 1
                t1.unsafe_get(Int((a >> 40) & 0xFF)) += 1
                t2.unsafe_get(Int((a >> 48) & 0xFF)) += 1
                t3.unsafe_get(Int(a >> 56)) += 1
                t4.unsafe_get(Int(b & 0xFF)) += 1
                t5.unsafe_get(Int((b >> 8) & 0xFF)) += 1
                t6.unsafe_get(Int((b >> 16) & 0xFF)) += 1
                t7.unsafe_get(Int((b >> 24) & 0xFF)) += 1
                t4.unsafe_get(Int((b >> 32) & 0xFF)) += 1
                t5.unsafe_get(Int((b >> 40) & 0xFF)) += 1
                t6.unsafe_get(Int((b >> 48) & 0xFF)) += 1
                t7.unsafe_get(Int(b >> 56)) += 1
                i += 16
            while i < n:
                t0.unsafe_get(Int(src.unsafe_get(i))) += 1
                i += 1
            for s in range(Self.bins):
                counts.unsafe_get(s) = Int(
                    t0.unsafe_get(s)
                    + t1.unsafe_get(s)
                    + t2.unsafe_get(s)
                    + t3.unsafe_get(s)
                    + t4.unsafe_get(s)
                    + t5.unsafe_get(s)
                    + t6.unsafe_get(s)
                    + t7.unsafe_get(s)
                )
        var top = 0
        var most = 0
        for s in range(Self.bins):
            var c = counts.unsafe_get(s)
            if c > 0:
                top = s
            most = max(most, c)
        self.top = top
        self.most = most

    def present(self) -> Span[Int, origin_of(self.counts)]:
        """The counts up to the largest symbol present."""
        return Span(self.counts)[: self.top + 1]

    def distance(self, other: Self) -> Int:
        """How unlike two histograms of equally many symbols are: the sum of
        their counts' differences -- `fpDistance`."""
        var d = 0
        for s in range(Self.bins):
            d += abs(self.counts[s] - other.counts[s])
        return d
