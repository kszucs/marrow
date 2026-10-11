# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# `FastMatcher` is ported from libzstd 1.5.7's `zstd_fast.c`
# (`ZSTD_compressBlock_fast_noDict_generic`); the block and sequences
# encoders follow libzstd's `zstd_compress.c` and
# `zstd_compress_sequences.c` at its fast strategy, and
# `BlockEncoder.split` is `zstd_preSplit.c`'s `ZSTD_splitBlock_fromBorders`,
# Copyright (c) Meta Platforms, Inc. (BSD 3-Clause); see NOTICE.txt.

"""Zstandard, in Mojo: what Parquet's ZSTD pages and Arrow IPC's ZSTD buffers
hold (`zstd_compression_format.md`, RFC 8878).

The entropy coders it is built from -- `fse` and `huffman` -- sit beside it
in `marrow.compress`, and its bit streams are `marrow.utils.bits`'. The
sections below go from the headers, to decoding a block, finding a block's
matches and encoding it, to the frames `Zstd` reads and writes, and last
`LibZstd`, the same format through the system's libzstd.

Dictionaries are refused: neither format uses them. `Zstd.compress` writes
what libzstd 1.5.7's one-shot `ZSTD_compress` writes at level 1, byte for
byte: its fast match finder and window, its block splitting, its rules for
choosing each section's encoding and for repeating a Huffman code, and its
tie-breaking in building one. Parquet's writer takes no level, so level 1 is
the whole of what is needed; `test_zstd.mojo` holds the frames identical.

On literal-heavy input two things carry the decoder: decoding the four
literal streams in lockstep, so their dependency chains overlap, and keeping
the bit reader's bits left-aligned, so a lookup is one shift.
"""

# Against libzstd 1.5.7 on the codec corpora (`benchmarks/codecs/
# codec_ab.mojo`), decoding takes 0.94-1.07x its time and compressing
# 0.93-1.09x, text the slowest both ways. The lockstep literal streams are
# worth 3.7x on the 1 MiB floats, the left-aligned bits a further 1.23x.

from std.bit import bit_width, log2_floor

from std.memory.alloc import unsafe_alloc

from ..errors import CorruptError, DynError, InternalError, NotImplementedError
from ..utils.hashing import XxHash64
from ..views import BufferView
from ..utils.bits import BitReader, BitWriter
from ..utils.byteorder import LittleEndian
from .core import Compressor, append_uninit, library, reserve_more
from .fse import (
    MAX_WEIGHT,
    Alphabet,
    Distribution,
    FseEncoder,
    FseTable,
    Histogram,
)
from .huffman import (
    BLOCK_MAX,
    HuffmanEncoder,
    HuffmanTable,
    LiteralsEncoder,
    LiteralsHeader,
    min_gain,
)
from .lz77 import LzCopy, match_length


# ---------------------------------------------------------------------------
# The headers of a zstd frame, of its blocks, and of a compressed block's
# two sections (`zstd_compression_format.md`, RFC 8878).
#
# | Type | What it is |
# |---|---|
# | `FrameHeader` | a frame's descriptor, window, dictionary id and content size |
# | `BlockHeader` | a block's last flag, type and size |
# | `SequencesHeader` | a sequences section's count and each field's table mode |
#
# Each type both reads its header and writes it, so the decoder and the
# encoder share one definition of every format. The literals section --
# `LiteralsHeader`, `LiteralsEncoder` -- is `huffman`'s.
# ---------------------------------------------------------------------------


@fieldwise_init
struct FrameHeader(ImplicitlyCopyable, Movable):
    """What a frame's header says, after its magic number: whether the
    frame is one segment -- its window the whole content -- whether a
    checksum follows it, its window, and its content size, -1 when the
    header leaves it out."""

    var single: Bool
    var checksum: Bool
    var window: Int
    var content: Int

    @staticmethod
    def describing(content: Int, window: Int) -> Self:
        """The header of a frame of `content` bytes compressed within
        `window`: one segment when the window covers the content, and no
        checksum."""
        return Self(content <= window, False, window, content)

    @staticmethod
    def read(src: Span[UInt8, _], mut ip: Int) raises DynError -> Self:
        """The header at `ip`, which is moved past it. A frame that needs a
        dictionary raises `NotImplementedError`."""
        var n_in = len(src)
        if ip >= n_in:
            raise CorruptError("zstd: truncated frame header")
        var fhd = Int(src[ip])
        ip += 1
        var single = fhd & 0x20 != 0
        var checksum = fhd & 0x04 != 0
        if fhd & 0x08 != 0:
            raise CorruptError("zstd: reserved frame header bit set")
        var dict_size = (1 << (fhd & 3)) >> 1  # 0, 1, 2 or 4
        var fcs_flag = fhd >> 6
        var fcs_size = 1 << fcs_flag if fcs_flag > 0 else Int(single)
        var need = (0 if single else 1) + dict_size + fcs_size
        if n_in - ip < need:
            raise CorruptError("zstd: truncated frame header")
        var window = 0
        if not single:
            var wd = Int(src[ip])
            ip += 1
            var base = 1 << (10 + (wd >> 3))
            window = base + (base >> 3) * (wd & 7)
        if dict_size > 0:
            var id = Int(
                LittleEndian.partial[DType.uint32](src[ip : ip + dict_size], 0)
            )
            ip += dict_size
            if id != 0:
                raise NotImplementedError(
                    t"zstd: frames that need dictionary {id}"
                )
        var content = -1
        if fcs_size > 0:
            content = Int(
                LittleEndian.partial[DType.uint64](src[ip : ip + fcs_size], 0)
            )
            if fcs_size == 2:
                content += 256
            ip += fcs_size
        if single:
            window = content
        return Self(single, checksum, window, content)

    def write(self, mut dst: List[UInt8]):
        """Append the header: no dictionary id, the window as a power of two
        from 1 KiB unless the frame is one segment, and the content size in
        1 (one segment only), 2 (less 256), 4 or 8 bytes."""
        var n = self.content
        var code = Int(n >= 256) + Int(n >= 65536 + 256) + Int(n >= 0xFFFFFFFF)
        dst.append(
            UInt8(code << 6 | Int(self.single) << 5 | Int(self.checksum) << 2)
        )
        if not self.single:
            dst.append(UInt8((log2_floor(self.window) - 10) << 3))
        if code > 0 or self.single:
            LittleEndian.append(
                dst, UInt64(n - 256 if code == 1 else n), 1 << code
            )

    def block_max(self) -> Int:
        """The most a block of this frame may hold."""
        return min(self.window, BLOCK_MAX)


@fieldwise_init
struct BlockHeader(ImplicitlyCopyable, Movable):
    """A block's 3-byte header: whether it is the frame's last, its type,
    and its size -- the bytes it holds, or for an RLE block the bytes it
    repeats its one byte to."""

    var last: Bool
    var kind: Int
    var size: Int

    comptime RAW = 0
    comptime RLE = 1
    comptime COMPRESSED = 2

    comptime SIZE = 3
    """The header's own bytes."""

    @staticmethod
    def read(src: Span[UInt8, _], ip: Int) raises CorruptError -> Self:
        """The header at `ip`."""
        if len(src) - ip < Self.SIZE:
            raise CorruptError("zstd: truncated block header")
        var v = Int(LittleEndian.partial[DType.uint32](src[ip : ip + 3], 0))
        var kind = (v >> 1) & 3
        if kind == 3:
            raise CorruptError("zstd: reserved block type")
        return Self(v & 1 != 0, kind, v >> 3)

    def payload(self) -> Int:
        """The bytes after the header that belong to the block."""
        return 1 if self.kind == Self.RLE else self.size

    def _value(self) -> Int:
        return Int(self.last) | self.kind << 1 | self.size << 3

    def write(self, mut out: List[UInt8]):
        """Append the header."""
        LittleEndian.append(out, UInt64(self._value()), Self.SIZE)

    def write_at(self, mut out: List[UInt8], at: Int):
        """Write the header over the `SIZE` bytes reserved for it at `at`."""
        LittleEndian.write(out, at, UInt64(self._value()), Self.SIZE)


@fieldwise_init
struct SequencesHeader(ImplicitlyCopyable, Movable):
    """A sequences section's header: how many sequences, and -- when there
    are any -- the table mode of each field, two bits apiece; and the
    header's own bytes, which a reader takes as written: a count may be
    longer than it needs to be."""

    var count: Int
    var modes: Int
    var length: Int

    comptime PREDEFINED = 0
    comptime RLE = 1
    comptime COMPRESSED = 2

    @staticmethod
    def of(count: Int) -> Self:
        """The header of `count` sequences, in as few bytes as hold it; the
        modes are set as the tables are chosen."""
        var count_bytes = 1 if count < 128 else (2 if count < 0x7F00 else 3)
        return Self(count, 0, count_bytes + Int(count > 0))

    @staticmethod
    def read(src: Span[UInt8, _]) raises CorruptError -> Self:
        """The header the section starts with."""
        var n_in = len(src)
        if n_in == 0:
            raise CorruptError("zstd: missing sequences section")
        var b0 = Int(src[0])
        var count: Int
        var ip: Int
        if b0 < 128:
            count = b0
            ip = 1
        elif b0 < 255:
            if n_in < 2:
                raise CorruptError("zstd: truncated sequence count")
            count = ((b0 - 128) << 8) + Int(src[1])
            ip = 2
        else:
            if n_in < 3:
                raise CorruptError("zstd: truncated sequence count")
            count = Int(src[1]) + (Int(src[2]) << 8) + 0x7F00
            ip = 3
        var modes = 0
        if count > 0:
            if ip >= n_in:
                raise CorruptError("zstd: truncated sequence modes")
            modes = Int(src[ip])
            if modes & 3 != 0:
                raise CorruptError("zstd: reserved sequence mode bits set")
            ip += 1
        return Self(count, modes, ip)

    def mode(self, field: Alphabet) -> Int:
        """`field`'s table mode."""
        return (self.modes >> (6 - 2 * field.index)) & 3

    def set_mode(mut self, field: Alphabet, mode: Int):
        self.modes |= mode << (6 - 2 * field.index)

    def write_at(self, mut out: List[UInt8], at: Int):
        """Write the header over the `length` bytes reserved for it at
        `at`."""
        var n = self.count
        var pos = at
        if n < 128:
            out[pos] = UInt8(n)
            pos += 1
        elif n < 0x7F00:
            out[pos] = UInt8((n >> 8) + 0x80)
            out[pos + 1] = UInt8(n & 0xFF)
            pos += 2
        else:
            out[pos] = 0xFF
            out[pos + 1] = UInt8((n - 0x7F00) & 0xFF)
            out[pos + 2] = UInt8((n - 0x7F00) >> 8)
            pos += 3
        if n > 0:
            out[pos] = UInt8(self.modes)


# ---------------------------------------------------------------------------
# Decoding a zstd block: raw, one repeated byte, or compressed -- its
# literals section, then its sequences.
#
# | Type | What it is |
# |---|---|
# | `BlockDecoder` | what a frame's blocks share as they are decoded, and the decoding |
# | `RepeatOffsets` | the three most recent offsets, and the rules a sequence names them by |
#
# A compressed block's literals are decoded whole into a buffer first. The
# sequences are then decoded and executed in one pass, each a literal run
# copied from that buffer followed by a match copied from the output -- the
# copies are `lz77`'s, the whole-block ones while the output has room for
# their overshoot and exact ones near its end, as the LZ4 and Snappy decoders
# do.
# ---------------------------------------------------------------------------


@fieldwise_init
struct RepeatOffsets(TrivialRegisterPassable):
    """The three most recent offsets, which a sequence may name instead of
    giving its own -- the spec's "Repeat offsets". A decoding loop keeps
    them in registers."""

    var offset1: Int
    var offset2: Int
    var offset3: Int

    comptime INITIAL = Self(1, 4, 8)
    """What a frame starts with."""

    @always_inline
    def resolve(mut self, value: Int, ll: Int) raises CorruptError -> Int:
        """The offset a sequence's offset value stands for, given its
        literal length `ll`, the recent ones moved as it says: a value over
        3 is a new offset plus 3, and 1 to 3 name the recent ones -- shifted
        by one when no literals precede the match, the last then meaning the
        most recent less 1."""
        var offset: Int
        if value > 3:
            offset = value - 3
            self.offset3 = self.offset2
            self.offset2 = self.offset1
            self.offset1 = offset
        else:
            var which = value - 1 + Int(ll == 0)
            if which == 0:
                offset = self.offset1
            elif which == 1:
                offset = self.offset2
                self.offset2 = self.offset1
                self.offset1 = offset
            elif which == 2:
                offset = self.offset3
                self.offset3 = self.offset2
                self.offset2 = self.offset1
                self.offset1 = offset
            else:
                offset = self.offset1 - 1
                if offset == 0:
                    raise CorruptError("zstd: a repeat offset of 0")
                self.offset3 = self.offset2
                self.offset2 = self.offset1
                self.offset1 = offset
        return offset


struct BlockDecoder(Movable):
    """What a frame's blocks share as they are decoded: the literals buffer,
    the Huffman and FSE tables a later block may repeat, and the recent
    offsets."""

    var _literals: List[UInt8]
    var _huffman: HuffmanTable
    var _ll: FseTable
    var _of: FseTable
    var _ml: FseTable
    var _offsets: RepeatOffsets

    def __init__(out self, out_size: Int):
        """For an output of `out_size` bytes, which no block's literals can
        exceed."""
        # Written before it is read, but for the whole-block copies reading
        # up to `LzCopy.SLOP` past the literals: what they read there is
        # overwritten in the output before anything reads it.
        self._literals = List[UInt8](
            unsafe_uninit_length=min(out_size, BLOCK_MAX) + LzCopy.SLOP
        )
        self._huffman = HuffmanTable()
        self._ll = FseTable()
        self._of = FseTable()
        self._ml = FseTable()
        self._offsets = RepeatOffsets.INITIAL

    def reset(mut self):
        """Forget the previous frame: blocks may not repeat across frames."""
        self._huffman.ready = False
        self._ll.ready = False
        self._of.ready = False
        self._ml.ready = False
        self._offsets = RepeatOffsets.INITIAL

    def decode[
        o: MutOrigin
    ](
        mut self,
        header: BlockHeader,
        payload: Span[UInt8, _],
        dst: Span[UInt8, o],
        op: Int,
        floor: Int,
    ) raises CorruptError -> Int:
        """Decode the block `header` heads, whose `payload` follows it, into
        `dst` at `op`; return the new `op`. Matches may reach back to
        `floor`, the frame's start."""
        var room = len(dst) - op
        if header.kind == BlockHeader.RAW:
            if header.size > room:
                raise CorruptError("zstd: a raw block overruns")
            BufferView(dst).slice(op).copy_from(
                BufferView(payload), header.size
            )
            return op + header.size
        elif header.kind == BlockHeader.RLE:
            if header.size > room:
                raise CorruptError("zstd: an RLE block overruns")
            dst[op : op + header.size].fill(payload[0])
            return op + header.size
        else:
            return self._compressed(payload, dst, op, floor)

    def _compressed[
        o: MutOrigin
    ](
        mut self,
        block: Span[UInt8, _],
        dst: Span[UInt8, o],
        op: Int,
        floor: Int,
    ) raises CorruptError -> Int:
        """A compressed block. One without sequences is all literals, and
        they are decoded straight into the output, as libzstd does, rather
        than through `_literals` and a second pass over the whole block."""
        var room = len(dst) - op
        var lits = LiteralsHeader.read(block)
        var at = lits.end()
        var seqs = SequencesHeader.read(block[at:])
        at += seqs.length
        if seqs.count == 0:
            if at != len(block):
                raise CorruptError(
                    "zstd: bytes after an empty sequences section"
                )
            self._huffman.decode_section(
                lits, block, BufferView(dst).slice(op), room
            )
            return op + lits.n
        self._huffman.decode_section(
            lits, block, BufferView(Span(self._literals)), room
        )
        at += Self._prepare(
            self._ll, Alphabet.LITERAL_LENGTHS, seqs, block[at:]
        )
        at += Self._prepare(self._of, Alphabet.OFFSETS, seqs, block[at:])
        at += Self._prepare(self._ml, Alphabet.MATCH_LENGTHS, seqs, block[at:])
        var end = self._execute(block[at:], dst, op, floor, seqs.count, lits.n)
        return self._tail(dst, end[0], end[1], lits.n)

    @staticmethod
    def _prepare(
        mut table: FseTable,
        field: Alphabet,
        header: SequencesHeader,
        src: Span[UInt8, _],
    ) raises CorruptError -> Int:
        """Make `table` what `header` says for `field`; return the bytes its
        description took from `src`."""
        var mode = header.mode(field)
        if mode == SequencesHeader.PREDEFINED:
            table.predefined(field)
            return 0
        elif mode == SequencesHeader.RLE:
            if len(src) < 1:
                raise CorruptError("zstd: truncated RLE sequence mode")
            table.rle(Int(src[0]), field)
            return 1
        elif mode == SequencesHeader.COMPRESSED:
            return table.read(src, field)
        # Mode 3, "Repeat": the table the last block used.
        if not table.ready:
            raise CorruptError("zstd: a repeated sequence table without one")
        return 0

    def _execute[
        o: MutOrigin
    ](
        mut self,
        src: Span[UInt8, _],
        dst: Span[UInt8, o],
        var op: Int,
        floor: Int,
        count: Int,
        n_lits: Int,
    ) raises CorruptError -> Tuple[Int, Int]:
        """Decode `count` sequences from the bitstream `src` and run each
        against the output as it is decoded; return where the output and
        the literals stand after them."""
        var r = BitReader(src)
        var ll_table = Span(self._ll.entries)
        var of_table = Span(self._of.entries)
        var ml_table = Span(self._ml.entries)
        var ll_state = r.read(self._ll.log)
        var of_state = r.read(self._of.log)
        var ml_state = r.read(self._ml.log)
        r.refill()
        var offsets = self._offsets
        var lits = BufferView(Span(self._literals))
        var out = BufferView(dst)
        var n_out = len(dst)
        var lit = 0
        for i in range(count):
            # A state is always inside its table, whatever the input.
            var ll_e = ll_table.unsafe_get(ll_state)
            var of_e = of_table.unsafe_get(of_state)
            var ml_e = ml_table.unsafe_get(ml_state)
            # Offset, match length and literal length take at most 31, 16
            # and 16 extra bits, and the three state updates 26; a refill
            # readies 57. So one refill per sequence, and a second after the
            # match length only when the extra bits come to 31 or more --
            # libzstd's arrangement.
            var of_bits = Int(of_e.extra)
            var ml_bits = Int(ml_e.extra)
            var ll_bits = Int(ll_e.extra)
            var value = Int(of_e.base) + r.read(of_bits)
            var ml = Int(ml_e.base) + r.read(ml_bits)
            if of_bits + ml_bits + ll_bits >= 31:
                r.refill()
            var ll = Int(ll_e.base) + r.read(ll_bits)
            var offset = offsets.resolve(value, ll)
            if i + 1 < count:
                ll_state = Int(ll_e.next) + r.read(Int(ll_e.bits))
                ml_state = Int(ml_e.next) + r.read(Int(ml_e.bits))
                of_state = Int(of_e.next) + r.read(Int(of_e.bits))
            r.refill()
            if ll > n_lits - lit or ll + ml > n_out - op:
                raise CorruptError(
                    t"zstd: a sequence at output {op} overruns the literals"
                    t" or the output"
                )
            var at = op + ll
            if offset > at - floor:
                raise CorruptError(
                    t"zstd: offset {offset} at output {at} is before the frame"
                )
            if at + ml + LzCopy.SLOP <= n_out:
                if ll <= LzCopy.SLOP:
                    LzCopy.blocks64(lits.slice(lit), out.slice(op), ll)
                else:
                    out.slice(op).copy_from(lits.slice(lit), ll)
                LzCopy.match_long(out, at, offset, ml)
            else:
                out.slice(op).copy_from(lits.slice(lit), ll)
                LzCopy.copy_exact(dst, at, offset, ml)
            op = at + ml
            lit += ll
        if not r.finished():
            raise CorruptError("zstd: a sequences bitstream is not consumed")
        self._offsets = offsets
        return (op, lit)

    def _tail[
        o: MutOrigin
    ](
        self, dst: Span[UInt8, o], op: Int, lit: Int, n_lits: Int
    ) raises CorruptError -> Int:
        """Copy the literals left after the last sequence, from `lit` of
        `n_lits`."""
        var rest = n_lits - lit
        if rest > len(dst) - op:
            raise CorruptError("zstd: literals overflow the output")
        BufferView(dst).slice(op).copy_from(
            BufferView(Span(self._literals)).slice(lit), rest
        )
        return op + rest


# ---------------------------------------------------------------------------
# Finding a zstd block's matches: libzstd's fast strategy, which level 1
# compresses with.
#
# `FastMatcher` keeps one `uint32` table over the whole input, so a match
# reaches back across blocks; it hashes 5 to 7 bytes, probes two positions at
# a time, tries the most recent offset before the table and the one before it
# right after each match. It reports into a `SeqStore`, with each offset
# already as it is sent: the match finder knows which matches repeat an
# offset, because it looked for them, so nothing after it has to find out
# again.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Compressing a zstd block.
#
# | Type | What it is |
# |---|---|
# | `BlockEncoder` | what a frame's blocks share on the way in: matching, and the block around the sections |
# | `SequencesEncoder` | the sequences section: each field's table, then the bitstream |
#
# The matches are `FastMatcher`'s, reported into a `SeqStore` that keeps the
# literals and the sequences, each offset already as it is sent. The literals
# are Huffman-coded unless that does not pay, with the code the decoder holds
# from an earlier block where that is no dearer than sending a new one; each
# sequence field gets the predefined table, a one-symbol table or one built
# for the block, by libzstd's fast-strategy rules. A block that does not
# shrink is stored raw, and one of a single repeated byte as an RLE block --
# unless it is the frame's first, or codes to 25 bytes or more. Where a full
# block's two ends look unlike each other, `BlockEncoder.split` ends it early,
# so each part gets tables of its own.
# ---------------------------------------------------------------------------


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
    ) raises CorruptError:
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
            if size >= n - min_gain(n):
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
    ) raises CorruptError -> Tuple[Int, Bool]:
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
    ) raises CorruptError -> Bool:
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
    ) raises CorruptError -> Tuple[Int, Int]:
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


struct Zstd(Compressor):
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
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append one frame holding `src` to `dst`, with the content size
        and without a checksum, as libzstd's one-shot `ZSTD_compress` writes
        it at level 1 -- the same bytes, with libzstd 1.5.7."""
        var n = len(src)
        # Room for the scratch a block's sections write before they are cut
        # to size too, up to 3 bytes an input byte where the frame keeps 1:
        # the sequences bitstream is sized for 12 bytes a 4-byte match.
        reserve_more(
            dst, Self.max_compressed_length(n) + 2 * min(n, BLOCK_MAX) + 512
        )
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
    def decompress[
        o: Origin[mut=True]
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


# ---------------------------------------------------------------------------
# The same format through the system's libzstd.
# ---------------------------------------------------------------------------


comptime _Opaque = OpaquePointer[MutUntrackedOrigin]


struct LibZstd(Compressor):
    """Zstandard through libzstd: `ZSTD_compress` at level 1 and
    `ZSTD_decompress`."""

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        return Zstd.max_decompressed_length(n)

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append the level-1 frame for `src` -- what `Zstd.compress`
        writes, with libzstd 1.5.7 byte for byte."""
        var z = library["zstd"]()
        var bound = z.call["ZSTD_compressBound", Int](len(src))
        var at = append_uninit(dst, bound)
        var n = z.call["ZSTD_compress", Int](
            dst.unsafe_ptr().unsafe_offset(at),
            bound,
            src.unsafe_ptr(),
            len(src),
            Int32(1),
        )
        if z.call["ZSTD_isError", UInt32](n) != 0:
            dst.shrink(at)
            raise InternalError("zstd: libzstd could not compress")
        dst.shrink(at + n)

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        var n = library["zstd"]().call["ZSTD_decompress", Int](
            dst.unsafe_ptr(), len(dst), src.unsafe_ptr(), len(src)
        )
        if n != len(dst):
            raise CorruptError(
                "zstd: libzstd could not decode the expected size"
            )

    @staticmethod
    def decompress_unsized(src: Span[UInt8, _]) raises DynError -> List[UInt8]:
        """Decompress frames whose content size may be absent -- Avro's
        `zstandard` blocks record none, and a streaming writer may leave it
        out -- through the streaming API, growing the output. A frame that
        does declare its size sizes the first allocation, within reason:
        the declaration is the input's word, not a promise."""
        var z = library["zstd"]()
        var declared = z.call["ZSTD_getFrameContentSize", UInt64](
            src.unsafe_ptr(), len(src)
        )
        var hint = 4 * len(src)
        if declared < UInt64(0) - 2:  # neither UNKNOWN (-1) nor ERROR (-2)
            hint = min(Int(declared), 64 * len(src))
        var out = List[UInt8](unsafe_uninit_length=max(64, hint))
        # ZSTD_inBuffer {src, size, pos} then ZSTD_outBuffer {dst, size, pos}.
        var bufs = unsafe_alloc[UInt64](6)
        bufs[unsafe_offset=0] = UInt64(Int(src.unsafe_ptr()))
        bufs[unsafe_offset=1] = UInt64(len(src))
        bufs[unsafe_offset=2] = 0
        bufs[unsafe_offset=5] = 0
        var dctx = z.call["ZSTD_createDCtx", _Opaque]()
        var failure = String()
        while True:
            bufs[unsafe_offset=3] = UInt64(Int(out.unsafe_ptr()))
            bufs[unsafe_offset=4] = UInt64(len(out))
            var rc = z.call["ZSTD_decompressStream", UInt](
                dctx, bufs.unsafe_offset(3), bufs
            )
            if z.call["ZSTD_isError", UInt32](rc) != 0:
                failure = "zstd: decompress failed"
                break
            var consumed = Int(bufs[unsafe_offset=2]) == len(src)
            if consumed and rc == 0:
                break
            if Int(bufs[unsafe_offset=5]) == len(out):
                out.resize(unsafe_uninit_length=2 * len(out))
            elif consumed:
                failure = "zstd: truncated frame"
                break
        var produced = Int(bufs[unsafe_offset=5])
        _ = z.call["ZSTD_freeDCtx", UInt](dctx)
        bufs.unsafe_free()
        if failure:
            raise CorruptError(failure)
        out.resize(unsafe_uninit_length=produced)
        return out^
