# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Decoding a zstd block: raw, one repeated byte, or compressed -- its
literals section, then its sequences.

| Type | What it is |
|---|---|
| `BlockDecoder` | what a frame's blocks share as they are decoded, and the decoding |
| `RepeatOffsets` | the three most recent offsets, and the rules a sequence names them by |

A compressed block's literals are decoded whole into a buffer first. The
sequences are then decoded and executed in one pass, each a literal run
copied from that buffer followed by a match copied from the output -- the
copies are `lz77`'s, the whole-block ones while the output has room for
their overshoot and exact ones near its end, as the LZ4 and Snappy decoders
do.
"""

from ...errors import CorruptError
from ...views import BufferView
from ..lz77 import LzCopy
from .bits import BitReader
from .fse import Alphabet, FseTable
from .headers import BLOCK_MAX, BlockHeader, LiteralsHeader, SequencesHeader
from .huffman import HuffmanTable


comptime _SLOP = LzCopy.SLOP
"""How far past a copy the whole-block copies may write -- and how far past
the literals they may read, so the literal buffer carries that much more."""


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
        # up to `_SLOP` past the literals: what they read there is
        # overwritten in the output before anything reads it.
        self._literals = List[UInt8](
            unsafe_uninit_length=min(out_size, BLOCK_MAX) + _SLOP
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
            Self._decode_literals(
                lits, block, BufferView(dst).slice(op), room, self._huffman
            )
            return op + lits.n
        Self._decode_literals(
            lits,
            block,
            BufferView(Span(self._literals)),
            room,
            self._huffman,
        )
        at += Self._prepare(
            self._ll, Alphabet.LITERAL_LENGTHS, seqs, block[at:]
        )
        at += Self._prepare(self._of, Alphabet.OFFSETS, seqs, block[at:])
        at += Self._prepare(self._ml, Alphabet.MATCH_LENGTHS, seqs, block[at:])
        var end = self._execute(block[at:], dst, op, floor, seqs.count, lits.n)
        return self._tail(dst, end[0], end[1], lits.n)

    @staticmethod
    def _decode_literals(
        header: LiteralsHeader,
        block: Span[UInt8, _],
        into: BufferView[mut=True, DType.uint8, _],
        room: Int,
        mut huffman: HuffmanTable,
    ) raises CorruptError:
        """Decode the literals `header` heads into `into`, writing nothing
        past them. They all end up in the output, so more than `room` is
        corrupt."""
        var n = header.n
        if n > BLOCK_MAX or n > room:
            raise CorruptError(
                t"zstd: {n} literals overflow the block or the output"
            )
        if header.kind == LiteralsHeader.RAW:
            into.copy_from(BufferView(block).slice(header.header), n)
        elif header.kind == LiteralsHeader.RLE:
            into.as_span()[:n].fill(block[header.header])
        else:
            var payload = block[header.header : header.end()]
            if header.kind == LiteralsHeader.COMPRESSED:
                payload = payload[huffman.read(payload) :]
            elif not huffman.ready:
                raise CorruptError("zstd: treeless literals without a tree")
            huffman.decode(payload, into, n, header.streams, header.size)

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
            if at + ml + _SLOP <= n_out:
                if ll <= _SLOP:
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
