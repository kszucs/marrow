# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Ported from google/brotli 1.2.0, Copyright 2013 Google Inc. (MIT): the
# decoder from `c/dec/decode.c`, `huffman.c` and `prefix.c`, the dictionary
# transforms from `c/common/transform.c`, and the encoder from
# `c/enc/compress_fragment_two_pass.c`, `entropy_encode.c`,
# `brotli_bit_stream.c`, `bit_cost.c` and `encode.c`; see NOTICE.txt.

"""Brotli, in Mojo: what Parquet's BROTLI pages hold (RFC 7932).

The constants of the format -- the static dictionary, its transforms and the
literal context table -- sit beside it in `brotli_dictionary`. The sections
below go from the bit stream and its prefix codes, to a compressed
meta-block's header and its commands, to compressing, to the stream
`Brotli` reads and writes.

A stream is decoded whole into an output of the size the caller knows, so
the window is the output itself: a copy reads the bytes it repeats straight
from there, and no ring buffer is kept. Every valid stream decodes, whatever
its window or the quality that wrote it; bytes after the stream's end are
ignored, as libbrotli's `BrotliDecoderDecompress` ignores them.

`Brotli.compress` writes what libbrotli's one-shot `BrotliEncoderCompress`
writes at quality 1 with a 22-bit window, byte for byte: its two-pass
fragment compressor, its Huffman code construction and its rules for storing
a block that would not shrink. `test_brotli.mojo` holds the streams
identical. `LibBrotli` is the same format through the system's libbrotli,
which writes at quality 11.
"""

from std.bit import bit_reverse, bit_width
from std.builtin.globals import global_constant
from std.math import log2
from std.memory import bitcast

from ..errors import CorruptError, DynError, InternalError
from ..views import BufferView
from .brotli_dictionary import (
    CONTEXT_LOOKUP,
    IDENTITY,
    OFFSETS_BY_LENGTH,
    OMIT_FIRST_1,
    OMIT_FIRST_9,
    OMIT_LAST_9,
    PREFIX_SUFFIX,
    PREFIX_SUFFIX_MAP,
    SIZE_BITS_BY_LENGTH,
    TRANSFORMS,
    TRANSFORMS_COUNT,
    UPPERCASE_ALL,
    UPPERCASE_FIRST,
    copy_word,
)
from ..utils.byteorder import LittleEndian
from .core import Compressor, append_uninit, library
from .lz77 import LzCopy, match_length


# ---------------------------------------------------------------------------
# The bit stream: fields packed from the least significant bit of each byte
# up, read forwards.
# ---------------------------------------------------------------------------


struct _BitReader[m: Bool, //, o: Origin[mut=m]](TrivialRegisterPassable):
    """A Brotli bit stream, read from its first byte on.

    The unread bits sit at the bottom of `_val`, `_n` of them. A `refill`
    tops them up to at least 56 with one unaligned 8-byte load; within 8
    bytes of the end the missing bytes read as zero. `_pos` counts the bytes
    loaded, and may run past the end on zeros: `overrun` says whether more
    bits were read than the stream holds, which is how a truncated stream is
    told apart -- checked where a loop could otherwise run on zeros, and
    after each meta-block.
    """

    var _src: Span[UInt8, Self.o]
    var _val: UInt64
    var _n: Int
    var _pos: Int

    def __init__(out self, src: Span[UInt8, Self.o]):
        self._src = src
        self._val = 0
        self._n = 0
        self._pos = 0

    @always_inline
    def refill(mut self):
        """At least 56 bits ready."""
        var word: UInt64
        if self._pos + 8 <= len(self._src):
            word = LittleEndian.fixed[DType.uint64](self._src, self._pos)
        else:
            word = Self._tail(self._src, self._pos)
        # The bits above `_n` may already hold the same bytes' bits from the
        # last load, so OR-ing them in again changes nothing.
        self._val |= word << UInt64(self._n)
        self._pos += (63 - self._n) >> 3
        self._n |= 56

    @staticmethod
    @no_inline
    def _tail(src: Span[UInt8, _], pos: Int) -> UInt64:
        if pos >= len(src):
            return 0
        return LittleEndian.partial[DType.uint64](src, pos)

    @always_inline
    def available(self) -> Int:
        return self._n

    @always_inline
    def peek(self, n: Int) -> Int:
        """The next `n` (at most 56) bits, not taken."""
        return Int(self._val & ((UInt64(1) << UInt64(n)) - 1))

    @always_inline
    def skip(mut self, n: Int):
        self._val >>= UInt64(n)
        self._n -= n

    @always_inline
    def read(mut self, n: Int) -> Int:
        """Take the next `n` bits; that many must be ready."""
        var v = self.peek(n)
        self.skip(n)
        return v

    @always_inline
    def fetch(mut self, n: Int) -> Int:
        """Take the next `n` (at most 56) bits, refilling first if need
        be."""
        if self._n < n:
            self.refill()
        return self.read(n)

    @always_inline
    def symbol(mut self, table: Span[UInt32, _], base: Int) -> Int:
        """The next symbol of the prefix code whose table starts at `base`;
        15 bits must be ready. An entry is `value << 16 | bits`: a symbol and
        its code's length, or in the 256-entry root, for a code longer than
        8 bits, its second-level table's offset and width plus 8."""
        var e = table.unsafe_get(base + Int(self._val & 0xFF))
        var bits = Int(e & 0xFF)
        if bits > 8:
            var low = Int(self._val >> 8) & ((1 << (bits - 8)) - 1)
            e = table.unsafe_get(base + Int(e >> 16) + low)
            self.skip(8)
            bits = Int(e & 0xFF)
        self.skip(bits)
        return Int(e >> 16)

    def length(self) -> Int:
        """The stream's bytes."""
        return len(self._src)

    def overrun(self) -> Bool:
        """Whether more bits were taken than the stream holds."""
        return 8 * self._pos - self._n > 8 * len(self._src)

    def align(mut self) -> Int:
        """Skip to the next byte boundary; return the bits skipped, which a
        valid stream leaves zero."""
        return self.read(self._n & 7)

    def at_byte(self) -> Int:
        """The next byte to read, at a byte boundary."""
        return self._pos - (self._n >> 3)

    def jump(mut self, to: Int):
        """Continue from byte `to`."""
        self._pos = to
        self._val = 0
        self._n = 0


def _var_len_uint8(mut br: _BitReader[_]) -> Int:
    """0..255, in 1 to 11 bits."""
    if br.fetch(1) == 0:
        return 0
    var n = br.fetch(3)
    if n == 0:
        return 1
    return (1 << n) + br.fetch(n)


# ---------------------------------------------------------------------------
# Prefix codes: read from the stream as code lengths, or for at most four
# symbols as the symbols themselves, and built into lookup tables.
#
# A table has a 256-entry root indexed by the next 8 bits; a code longer
# than that continues in a second-level table the root entry points to.
# ---------------------------------------------------------------------------


comptime _ROOT_BITS = 8
comptime _MAX_LENGTH = 15
comptime _TABLE_SLACK = 376
"""What a table may need beyond one entry per symbol: libbrotli's bound."""

# fmt: off
comptime _CODE_LENGTH_ORDER: Array[UInt8, 18] = [
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10,
    11, 12, 13, 14, 15,
]
comptime _CODE_LENGTH_PREFIX_BITS: Array[UInt8, 16] = [
    2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2,
    2, 2, 4,
]
comptime _CODE_LENGTH_PREFIX_VALUE: Array[UInt8, 16] = [
    0, 4, 3, 2, 0, 4, 3, 1, 0, 4, 3, 2, 0,
    4, 3, 5,
]
# fmt: on
# The last two: the fixed code the code-length code's lengths are read with,
# by the next 4 bits.


@always_inline
def _entry(value: Int, bits: Int) -> UInt32:
    return UInt32(value << 16 | bits)


@always_inline
def _reversed(code: Int, n: Int) -> Int:
    """The low `n` bits of `code`, last first: a canonical code as the
    stream sends it."""
    return Int(bit_reverse(UInt16(code)) >> UInt16(16 - n))


def _build(
    lengths: Span[UInt8, _],
    mut table: List[UInt32],
    base: Int,
    root_bits: Int,
    mut sorted: List[UInt16],
) -> Int:
    """Fill `table` from `base` with the canonical code of `lengths`, which
    must be complete; return the entries used. Shorter codes come first,
    and symbols of one length in their order."""
    var count = Array[Int, 16](fill=0)
    for l in lengths:
        count[Int(l)] += 1
    var start = Array[Int, 16](fill=0)
    for l in range(1, _MAX_LENGTH):
        start[l + 1] = start[l] + count[l]
    var max_length = 0
    for l in range(1, _MAX_LENGTH + 1):
        if count[l] > 0:
            max_length = l
    if len(sorted) < len(lengths):
        sorted.resize(len(lengths), 0)
    var at = start.copy()
    for s in range(len(lengths)):
        var l = Int(lengths[s])
        if l > 0:
            sorted[at[l]] = UInt16(s)
            at[l] += 1

    var root_size = 1 << root_bits
    var table_bits = min(root_bits, max_length)
    var size = 1 << table_bits
    var code = 0
    var i = 0
    for l in range(1, table_bits + 1):
        for _ in range(count[l]):
            var e = _entry(Int(sorted[i]), l)
            i += 1
            var k = _reversed(code, l)
            while k < size:
                table[base + k] = e
                k += 1 << l
            code += 1
        code <<= 1
    while size < root_size:
        for k in range(size):
            table[base + size + k] = table[base + k]
        size <<= 1

    # Codes longer than the root: one second-level table per 8-bit prefix,
    # as wide as the codes sharing it need.
    var total = root_size
    var prefix = -1
    var sub = 0
    var sub_size = 0
    for l in range(root_bits + 1, max_length + 1):
        for _ in range(count[l]):
            var top = code >> (l - root_bits)
            if top != prefix:
                prefix = top
                var bits = l
                var left = 1 << (l - root_bits)
                while bits < _MAX_LENGTH:
                    left -= count[bits]
                    if left <= 0:
                        break
                    bits += 1
                    left <<= 1
                var width = bits - root_bits
                sub = base + total
                sub_size = 1 << width
                table[base + _reversed(top, root_bits)] = _entry(
                    total, width + root_bits
                )
                total += sub_size
            var e = _entry(Int(sorted[i]), l - root_bits)
            i += 1
            var k = _reversed(
                code & ((1 << (l - root_bits)) - 1), l - root_bits
            )
            while k < sub_size:
                table[sub + k] = e
                k += 1 << (l - root_bits)
            code += 1
            count[l] -= 1
        code <<= 1
    return total


struct _CodeReader(Movable):
    """What reading a prefix code needs between codes: the code lengths and
    the code-length code's table."""

    var _lengths: List[UInt8]
    var _sorted: List[UInt16]
    var _cl: List[UInt32]
    """The code-length code, 32 entries by the next 5 bits."""

    def __init__(out self):
        self._lengths = List[UInt8](length=704, fill=0)
        self._sorted = List[UInt16](length=704, fill=0)
        self._cl = List[UInt32](length=32, fill=0)

    def read(
        mut self,
        mut br: _BitReader[_],
        mut table: List[UInt32],
        base: Int,
        alphabet: Int,
    ) raises CorruptError -> Int:
        """Read a prefix code over `alphabet` symbols into `table` from
        `base`, which has room for `alphabet + _TABLE_SLACK` entries; return
        the entries used."""
        var hskip = br.fetch(2)
        if hskip == 1:
            return self._simple(br, table, base, alphabet)
        var cl = Array[UInt8, 18](fill=0)
        var space = 32
        var codes = 0
        ref order = global_constant[_CODE_LENGTH_ORDER]()
        ref prefix_bits = global_constant[_CODE_LENGTH_PREFIX_BITS]()
        ref prefix_value = global_constant[_CODE_LENGTH_PREFIX_VALUE]()
        for i in range(hskip, 18):
            if br.available() < 4:
                br.refill()
            var ix = br.peek(4)
            var v = Int(prefix_value[ix])
            br.skip(Int(prefix_bits[ix]))
            cl[Int(order[i])] = UInt8(v)
            if v != 0:
                space -= 32 >> v
                codes += 1
                if space <= 0:
                    break
        if not (codes == 1 or space == 0):
            raise CorruptError("brotli: an incomplete code-length code")
        if codes == 1:
            var only = 0
            for s in range(18):
                if cl[s] != 0:
                    only = s
            for k in range(32):
                self._cl[k] = _entry(only, 0)
        else:
            _ = _build(Span(cl), self._cl, 0, 5, self._sorted)
        self._symbol_lengths(br, alphabet)
        return _build(
            Span(self._lengths)[:alphabet],
            table,
            base,
            _ROOT_BITS,
            self._sorted,
        )

    def _symbol_lengths(
        mut self, mut br: _BitReader[_], alphabet: Int
    ) raises CorruptError:
        """The code length of each of `alphabet` symbols, through the
        code-length code: 0 to 15 as they are, 16 repeating the last nonzero
        length and 17 a zero 3 to 6 or 3 to 10 times -- and a run of either
        extending the run before it."""
        var cl = Span(self._cl)
        var lengths = Span(self._lengths)
        var symbol = 0
        var prev = 8
        var repeat = 0
        var repeat_length = 0
        var space = 32768
        while symbol < alphabet and space > 0:
            if br.available() < 8:
                br.refill()
            var e = cl[br.peek(5)]
            br.skip(Int(e & 0xFF))
            var code = Int(e >> 16)
            if code < 16:
                repeat = 0
                lengths[symbol] = UInt8(code)
                if code != 0:
                    prev = code
                    space -= 32768 >> code
                symbol += 1
            else:
                var extra = 2 if code == 16 else 3
                var length = prev if code == 16 else 0
                if repeat_length != length:
                    repeat = 0
                    repeat_length = length
                var old = repeat
                if repeat > 0:
                    repeat = (repeat - 2) << extra
                repeat += br.read(extra) + 3
                var delta = repeat - old
                if symbol + delta > alphabet:
                    raise CorruptError(
                        "brotli: code lengths overrun the alphabet"
                    )
                for k in range(symbol, symbol + delta):
                    lengths[k] = UInt8(length)
                if length != 0:
                    space -= delta << (15 - length)
                symbol += delta
        if space != 0:
            raise CorruptError("brotli: an incomplete prefix code")
        for k in range(symbol, alphabet):
            lengths[k] = 0

    def _simple(
        mut self,
        mut br: _BitReader[_],
        mut table: List[UInt32],
        base: Int,
        alphabet: Int,
    ) raises CorruptError -> Int:
        """A code of one to four symbols, given as the symbols. Their count
        -- and for four, one more bit -- fixes their lengths in the order
        given, and the code is the canonical one of those lengths."""
        var n = br.fetch(2) + 1
        var width = bit_width(alphabet - 1)
        var v = Array[Int, 4](fill=0)
        for i in range(n):
            v[i] = br.fetch(width)
            if v[i] >= alphabet:
                raise CorruptError("brotli: a simple code's symbol is too big")
        for i in range(n):
            for k in range(i + 1, n):
                if v[i] == v[k]:
                    raise CorruptError("brotli: a simple code repeats a symbol")
        if n == 1:
            for k in range(1 << _ROOT_BITS):
                table[base + k] = _entry(v[0], 0)
            return 1 << _ROOT_BITS
        var given: Array[UInt8, 4] = [1, 1, 0, 0]
        if n == 3:
            given = [1, 2, 2, 0]
        elif n == 4:
            given = [1, 2, 3, 3] if br.fetch(1) == 1 else [2, 2, 2, 2]
        var lengths = Span(self._lengths)[:alphabet]
        lengths.fill(0)
        for i in range(n):
            lengths[v[i]] = given[i]
        return _build(lengths, table, base, _ROOT_BITS, self._sorted)

    def context_map(
        mut self,
        mut br: _BitReader[_],
        size: Int,
        mut map: List[UInt8],
        mut code_table: List[UInt32],
    ) raises CorruptError -> Int:
        """Read a context map of `size` entries into `map`; return how many
        prefix codes it names. Runs of zeros may be coded as runs, and the
        whole map moved to front."""
        var trees = _var_len_uint8(br) + 1
        if len(map) < size:
            map.resize(size, 0)
        if trees == 1:
            for i in range(size):
                map[i] = 0
            return 1
        var rle_max = 0
        if br.fetch(1) == 1:
            rle_max = br.fetch(4) + 1
        if len(code_table) < 272 + _TABLE_SLACK:
            code_table.resize(unsafe_uninit_length=272 + _TABLE_SLACK)
        _ = self.read(br, code_table, 0, trees + rle_max)
        var codes = Span(code_table)
        var i = 0
        while i < size:
            br.refill()
            var code = br.symbol(codes, 0)
            if code == 0:
                map[i] = 0
                i += 1
            elif code > rle_max:
                map[i] = UInt8(code - rle_max)
                i += 1
            else:
                var reps = (1 << code) + br.read(code)
                if reps > size - i:
                    raise CorruptError("brotli: a context map run overruns")
                for k in range(i, i + reps):
                    map[k] = 0
                i += reps
        if br.fetch(1) == 1:
            _inverse_move_to_front(Span(map)[:size])
        return trees


def _inverse_move_to_front(values: Span[mut=True, UInt8, _]):
    var mtf = Array[UInt8, 256](fill=0)
    for k in range(256):
        mtf[k] = UInt8(k)
    for i in range(len(values)):
        var index = Int(values[i])
        var value = mtf[index]
        values[i] = value
        var k = index
        while k > 0:
            mtf[k] = mtf[k - 1]
            k -= 1
        mtf[0] = value


struct _Group(Movable):
    """The prefix codes of one kind in a meta-block -- literals, commands or
    distances -- in one table, each from its start."""

    var table: List[UInt32]
    var starts: List[Int32]

    def __init__(out self):
        self.table = List[UInt32]()
        self.starts = List[Int32]()

    def read(
        mut self,
        mut codes: _CodeReader,
        mut br: _BitReader[_],
        trees: Int,
        alphabet: Int,
    ) raises CorruptError:
        var bound = trees * (alphabet + _TABLE_SLACK)
        # Every entry a lookup can reach is written before it is read.
        if len(self.table) < bound:
            self.table.resize(unsafe_uninit_length=bound)
        self.starts.clear()
        var at = 0
        for _ in range(trees):
            self.starts.append(Int32(at))
            at += codes.read(br, self.table, at, alphabet)


# ---------------------------------------------------------------------------
# A compressed meta-block: a header of block switches, context maps and
# prefix codes, then commands -- each a run of literals, then a copy from the
# output or a word from the static dictionary.
#
# | Type | What it is |
# |---|---|
# | `_BlockSwitch` | one kind's block types: the current one, the one before, and what is left of it |
# | `_Recent` | the last four distances, which a distance code may name |
# | `_Decoder` | the tables a stream's meta-blocks are decoded with, reused from one to the next |
# ---------------------------------------------------------------------------


# fmt: off
comptime _BLOCK_LENGTH_BASE: Array[UInt16, 26] = [
    1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113,
    145, 177, 209, 241, 305, 369, 497, 753, 1265, 2289, 4337, 8433, 16625,
]
comptime _BLOCK_LENGTH_EXTRA: Array[UInt8, 26] = [
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5,
    5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24,
]
# fmt: on

comptime _TYPE_TREE_SIZE = 258 + _TABLE_SLACK
comptime _LENGTH_TREE_SIZE = 26 + _TABLE_SLACK


def _commands_table() -> Array[UInt64, 704]:
    """Per insert-and-copy symbol: the insert length's extra bits (bits 0-7),
    the copy length's (8-15), whether the distance is the last one (16), the
    distance context (17-18), and the insert and copy lengths' bases (32-47,
    48-63) -- libbrotli's `kCmdLut`."""
    # fmt: off
    var insert_extra: Array[Int, 24] = [
        0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4,
        4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24,
    ]
    # fmt: on
    # fmt: off
    var copy_extra: Array[Int, 24] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3,
        3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
    ]
    # fmt: on
    var cell: Array[Int, 11] = [0, 1, 0, 1, 8, 9, 2, 16, 10, 17, 18]
    var insert_base = Array[Int, 24](fill=0)
    var copy_base = Array[Int, 24](fill=2)
    for i in range(23):
        insert_base[i + 1] = insert_base[i] + (1 << insert_extra[i])
        copy_base[i + 1] = copy_base[i] + (1 << copy_extra[i])
    var out = Array[UInt64, 704](fill=0)
    for s in range(704):
        var c = cell[s >> 6]
        var copy_code = ((c << 3) & 0x18) + (s & 7)
        var insert_code = (c & 0x18) + ((s >> 3) & 7)
        var copy = copy_base[copy_code]
        var context = 3 if copy > 4 else copy - 2
        var implicit = 1 if (s >> 6) < 2 else 0
        out[s] = (
            UInt64(insert_extra[insert_code])
            | UInt64(copy_extra[copy_code]) << 8
            | UInt64(implicit) << 16
            | UInt64(context) << 17
            | UInt64(insert_base[insert_code]) << 32
            | UInt64(copy) << 48
        )
    return out^


comptime _COMMANDS = _commands_table()


@fieldwise_init
struct _BlockSwitch(TrivialRegisterPassable):
    """One kind's block types in a meta-block: how many there are, where
    their two codes sit in the switch table, the current type and the one
    before it, and the symbols left in the current block."""

    var n: Int
    var types: Int
    var lengths: Int
    var last: Int
    var prev: Int
    var count: Int

    @always_inline
    def switch(mut self, table: Span[UInt32, _], mut br: _BitReader[_]) -> Int:
        """Read the next block's type and length; return its type."""
        br.refill()
        var t = br.symbol(table, self.types)
        var code = br.symbol(table, self.lengths)
        self.count = Int(
            global_constant[_BLOCK_LENGTH_BASE]().unsafe_get(code)
        ) + br.read(
            Int(global_constant[_BLOCK_LENGTH_EXTRA]().unsafe_get(code))
        )
        if t == 1:
            t = self.last + 1
        elif t == 0:
            t = self.prev
        else:
            t -= 2
        if t >= self.n:
            t -= self.n
        self.prev = self.last
        self.last = t
        return t


@fieldwise_init
struct _Recent(TrivialRegisterPassable):
    """The last four distances, the last first."""

    var d0: Int
    var d1: Int
    var d2: Int
    var d3: Int

    comptime INITIAL = Self(4, 11, 15, 16)

    @always_inline
    def push(mut self, d: Int):
        self.d3 = self.d2
        self.d2 = self.d1
        self.d1 = self.d0
        self.d0 = d

    @always_inline
    def short(self, code: Int) -> Int:
        """The distance short code `code` (0 to 15) names: one of the four,
        or the last or the one before it give or take 1 to 3. May be zero or
        negative, which no copy can use."""
        if code < 4:
            if code == 0:
                return self.d0
            elif code == 1:
                return self.d1
            elif code == 2:
                return self.d2
            return self.d3
        var base = code - 4 if code < 10 else code - 10
        var delta = ((0x605142 >> (4 * base)) & 0xF) - 3
        return (self.d0 if code < 10 else self.d1) + delta


comptime _SLOP = LzCopy.SLOP


struct _Decoder(Movable):
    """The tables a stream's meta-blocks are decoded with, reused from one
    meta-block to the next, and the last distances, which carry over."""

    var codes: _CodeReader
    var switches: List[UInt32]
    var map_code: List[UInt32]
    """A context map's own prefix code."""
    var literals: _Group
    var commands: _Group
    var distances: _Group
    var literal_map: List[UInt8]
    var distance_map: List[UInt8]
    var modes: List[UInt8]
    var literal_bases: List[Int32]
    """Per literal block type and context, where its code's table starts."""
    var distance_bases: List[Int32]
    var trivial: List[Bool]
    """Per literal block type, whether every context has the same code."""
    var distance_extra: List[UInt8]
    var distance_offset: List[Int32]
    var recent: _Recent

    def __init__(out self):
        self.codes = _CodeReader()
        self.map_code = List[UInt32]()
        self.switches = List[UInt32](
            unsafe_uninit_length=3 * (_TYPE_TREE_SIZE + _LENGTH_TREE_SIZE)
        )
        self.literals = _Group()
        self.commands = _Group()
        self.distances = _Group()
        self.literal_map = List[UInt8]()
        self.distance_map = List[UInt8]()
        self.modes = List[UInt8]()
        self.literal_bases = List[Int32]()
        self.distance_bases = List[Int32]()
        self.trivial = List[Bool]()
        self.distance_extra = List[UInt8](length=520, fill=0)
        self.distance_offset = List[Int32](length=520, fill=0)
        self.recent = _Recent.INITIAL

    def stream[
        o: MutOrigin
    ](mut self, src: Span[UInt8, _], dst: Span[UInt8, o]) raises CorruptError:
        """Decode the stream `src` into exactly `dst`."""
        var br = _BitReader(src)
        br.refill()
        var wbits: Int
        if br.read(1) == 0:
            wbits = 16
        else:
            var n = br.read(3)
            if n != 0:
                wbits = 17 + n
            else:
                n = br.read(3)
                if n == 1:
                    raise CorruptError(
                        "brotli: a large-window stream, which RFC 7932 does"
                        " not allow"
                    )
                wbits = 8 + n if n != 0 else 17
        var max_backward = (1 << wbits) - 16
        var pos = 0
        while True:
            br.refill()
            var last = br.read(1) == 1
            if last and br.read(1) == 1:
                break
            var nibbles = br.read(2)
            if nibbles == 3:
                if br.read(1) != 0:
                    raise CorruptError("brotli: a reserved bit is set")
                var nbytes = br.read(2)
                var size = 0
                for i in range(nbytes):
                    var b = br.fetch(8)
                    if i + 1 == nbytes and nbytes > 1 and b == 0:
                        raise CorruptError(
                            "brotli: a metadata length's last byte is zero"
                        )
                    size |= b << (8 * i)
                if nbytes > 0:
                    size += 1
                _ = Self._skip(br, size)
            else:
                nibbles += 4
                var mlen = 0
                for i in range(nibbles):
                    var v = br.fetch(4)
                    if i + 1 == nibbles and nibbles > 4 and v == 0:
                        raise CorruptError(
                            "brotli: a meta-block length's last nibble is zero"
                        )
                    mlen |= v << (4 * i)
                mlen += 1
                var uncompressed = False
                if not last:
                    uncompressed = br.fetch(1) == 1
                if mlen > len(dst) - pos:
                    raise CorruptError(
                        t"brotli: a meta-block of {mlen} bytes at {pos}"
                        t" overflows the {len(dst)}-byte output"
                    )
                if uncompressed:
                    var at = Self._skip(br, mlen)
                    BufferView(dst).slice(pos).copy_from(
                        BufferView(src).slice(at), mlen
                    )
                    pos += mlen
                else:
                    pos = self._compressed(br, dst, pos, mlen, max_backward)
            if br.overrun():
                raise CorruptError("brotli: truncated stream")
            if last:
                break
        if br.align() != 0:
            raise CorruptError("brotli: nonzero padding after the stream")
        if br.overrun():
            raise CorruptError("brotli: truncated stream")
        if pos != len(dst):
            raise CorruptError(
                t"brotli: the stream decodes to {pos} bytes, not the expected"
                t" {len(dst)}"
            )

    @staticmethod
    def _skip(mut br: _BitReader[_], size: Int) raises CorruptError -> Int:
        """Skip to the byte boundary and then `size` bytes; return where
        they start."""
        if br.align() != 0:
            raise CorruptError("brotli: nonzero padding before a byte boundary")
        if br.overrun():
            raise CorruptError("brotli: truncated stream")
        var at = br.at_byte()
        if size > br.length() - at:
            raise CorruptError("brotli: truncated stream")
        br.jump(at + size)
        return at

    def _block_switch(
        mut self, mut br: _BitReader[_], kind: Int
    ) raises CorruptError -> _BlockSwitch:
        """One kind's block types, from the meta-block header."""
        var n = _var_len_uint8(br) + 1
        var types = kind * _TYPE_TREE_SIZE
        var lengths = 3 * _TYPE_TREE_SIZE + kind * _LENGTH_TREE_SIZE
        var count = Int.MAX
        if n >= 2:
            _ = self.codes.read(br, self.switches, types, n + 2)
            _ = self.codes.read(br, self.switches, lengths, 26)
            br.refill()
            var code = br.symbol(Span(self.switches), lengths)
            count = Int(
                global_constant[_BLOCK_LENGTH_BASE]().unsafe_get(code)
            ) + br.read(
                Int(global_constant[_BLOCK_LENGTH_EXTRA]().unsafe_get(code))
            )
        return _BlockSwitch(n, types, lengths, 0, 1, count)

    def _compressed[
        o: MutOrigin
    ](
        mut self,
        mut br: _BitReader[_],
        dst: Span[UInt8, o],
        pos: Int,
        mlen: Int,
        max_backward: Int,
    ) raises CorruptError -> Int:
        """Read a compressed meta-block's header, then decode its `mlen`
        bytes into `dst` at `pos`; return the new `pos`."""
        var lits = self._block_switch(br, 0)
        var cmds = self._block_switch(br, 1)
        var dists = self._block_switch(br, 2)
        var npostfix = br.fetch(2)
        var ndirect = br.fetch(4) << npostfix
        self.modes.resize(lits.n, 0)
        for i in range(lits.n):
            self.modes[i] = UInt8(br.fetch(2))
        var literal_trees = self.codes.context_map(
            br, lits.n * 64, self.literal_map, self.map_code
        )
        var distance_trees = self.codes.context_map(
            br, dists.n * 4, self.distance_map, self.map_code
        )
        var alphabet = 16 + ndirect + (48 << npostfix)
        self.literals.read(self.codes, br, literal_trees, 256)
        self.commands.read(self.codes, br, cmds.n, 704)
        self.distances.read(self.codes, br, distance_trees, alphabet)
        if br.overrun():
            raise CorruptError("brotli: truncated meta-block header")

        self.literal_bases.resize(lits.n * 64, 0)
        self.trivial.resize(lits.n, False)
        for t in range(lits.n):
            var same = True
            var first = self.literal_map[t * 64]
            for c in range(64):
                var tree = self.literal_map[t * 64 + c]
                same = same and tree == first
                self.literal_bases[t * 64 + c] = self.literals.starts[Int(tree)]
            self.trivial[t] = same
        self.distance_bases.resize(dists.n * 4, 0)
        for i in range(dists.n * 4):
            self.distance_bases[i] = self.distances.starts[
                Int(self.distance_map[i])
            ]
        for j in range(ndirect):
            self.distance_extra[16 + j] = 0
            self.distance_offset[16 + j] = Int32(j + 1)
        var i = 16 + ndirect
        var bits = 1
        var half = 0
        while i < alphabet:
            var base = ndirect + ((((2 + half) << bits) - 4) << npostfix) + 1
            for j in range(1 << npostfix):
                self.distance_extra[i] = UInt8(bits)
                self.distance_offset[i] = Int32(base + j)
                i += 1
            bits += half
            half ^= 1
        return self._commands(
            br, dst, pos, pos + mlen, max_backward, lits, cmds, dists, npostfix
        )

    def _commands[
        o: MutOrigin
    ](
        mut self,
        mut br: _BitReader[_],
        dst: Span[UInt8, o],
        var pos: Int,
        end: Int,
        max_backward: Int,
        var lits: _BlockSwitch,
        var cmds: _BlockSwitch,
        var dists: _BlockSwitch,
        npostfix: Int,
    ) raises CorruptError -> Int:
        """Run a meta-block's commands until its `end` in the output."""
        var out = BufferView(dst)
        var n_out = len(dst)
        var switches = Span(self.switches)
        var literal_table = Span(self.literals.table)
        var command_table = Span(self.commands.table)
        var distance_table = Span(self.distances.table)
        var literal_bases = Span(self.literal_bases)
        var distance_bases = Span(self.distance_bases)
        var command_starts = Span(self.commands.starts)
        var extra = Span(self.distance_extra)
        var offsets = Span(self.distance_offset)
        ref lookup = global_constant[CONTEXT_LOOKUP]()
        ref command_lut = global_constant[_COMMANDS]()
        var recent = self.recent

        var slice = 0
        var mode = 512 * Int(self.modes[0] & 3)
        var trivial = self.trivial[0]
        var trivial_base = Int(literal_bases[0])
        var command_base = Int(command_starts[0])
        var distance_slice = 0
        while True:
            if cmds.count == 0:
                command_base = Int(
                    command_starts.unsafe_get(cmds.switch(switches, br))
                )
            cmds.count -= 1
            br.refill()
            var c = command_lut.unsafe_get(
                br.symbol(command_table, command_base)
            )
            var insert = Int((c >> 32) & 0xFFFF) + br.read(Int(c & 0xFF))
            br.refill()
            var copy = Int(c >> 48) + br.read(Int((c >> 8) & 0xFF))
            if insert > end - pos:
                raise CorruptError(
                    t"brotli: {insert} literals at {pos} overrun the meta-block"
                )
            var todo = insert
            while todo > 0:
                if lits.count == 0:
                    var t = lits.switch(switches, br)
                    slice = 64 * t
                    mode = 512 * Int(self.modes[t] & 3)
                    trivial = self.trivial[t]
                    trivial_base = Int(literal_bases.unsafe_get(slice))
                var run = min(todo, lits.count)
                lits.count -= run
                todo -= run
                var stop = pos + run
                if trivial:
                    # A literal takes at most 15 bits, so a refill readies
                    # three.
                    while stop - pos >= 3:
                        br.refill()
                        var a = br.symbol(literal_table, trivial_base)
                        var b = br.symbol(literal_table, trivial_base)
                        var c3 = br.symbol(literal_table, trivial_base)
                        out.unsafe_set(pos, UInt8(a))
                        out.unsafe_set(pos + 1, UInt8(b))
                        out.unsafe_set(pos + 2, UInt8(c3))
                        pos += 3
                    while pos < stop:
                        if br.available() < 15:
                            br.refill()
                        out.unsafe_set(
                            pos, UInt8(br.symbol(literal_table, trivial_base))
                        )
                        pos += 1
                else:
                    var p1 = Int(out.unsafe_get(pos - 1)) if pos >= 1 else 0
                    var p2 = Int(out.unsafe_get(pos - 2)) if pos >= 2 else 0
                    while pos < stop:
                        var context = Int(
                            lookup.unsafe_get(mode + p1)
                            | lookup.unsafe_get(mode + 256 + p2)
                        )
                        var base = Int(
                            literal_bases.unsafe_get(slice + context)
                        )
                        if br.available() < 15:
                            br.refill()
                        var b = br.symbol(literal_table, base)
                        out.unsafe_set(pos, UInt8(b))
                        p2 = p1
                        p1 = b
                        pos += 1
            if pos == end:
                break

            var distance: Int
            var push: Bool
            if c & (1 << 16) != 0:
                distance = recent.d0
                push = False
            else:
                if dists.count == 0:
                    distance_slice = 4 * dists.switch(switches, br)
                dists.count -= 1
                br.refill()
                var base = Int(
                    distance_bases.unsafe_get(
                        distance_slice + Int((c >> 17) & 3)
                    )
                )
                var code = br.symbol(distance_table, base)
                if code < 16:
                    distance = recent.short(code)
                    push = code != 0
                else:
                    distance = Int(offsets.unsafe_get(code)) + (
                        br.read(Int(extra.unsafe_get(code))) << npostfix
                    )
                    push = True
            var max_distance = min(pos, max_backward)
            if distance > max_distance or distance <= 0:
                pos = Self._word(dst, pos, end, distance, max_distance, copy)
                if br.overrun():
                    raise CorruptError("brotli: truncated stream")
            else:
                if copy > end - pos:
                    raise CorruptError(
                        t"brotli: a {copy}-byte copy at {pos} overruns the"
                        t" meta-block"
                    )
                if push:
                    recent.push(distance)
                if pos + copy + _SLOP <= n_out:
                    if copy <= 16 and distance >= copy:
                        # One block, its source wholly behind the copy; what
                        # it writes past the copy is overwritten later.
                        out.store[16](pos, out.load[16](pos - distance))
                    else:
                        LzCopy.match_long(out, pos, distance, copy)
                else:
                    LzCopy.copy_exact(dst, pos, distance, copy)
                pos += copy
            if pos == end:
                break
        self.recent = recent
        return pos

    @staticmethod
    @no_inline
    def _word[
        o: MutOrigin
    ](
        dst: Span[UInt8, o],
        pos: Int,
        end: Int,
        distance: Int,
        max_distance: Int,
        length: Int,
    ) raises CorruptError -> Int:
        """A distance past the output so far: word `length` of the static
        dictionary, transformed. Return the new `pos`."""
        if distance <= 0 or distance > 0x7FFFFFFC:
            raise CorruptError(t"brotli: distance {distance} at {pos}")
        if length < 4 or length > 24:
            raise CorruptError(
                t"brotli: a {length}-byte copy from the dictionary at {pos}"
            )
        var address = distance - max_distance - 1
        var shift = Int(
            global_constant[SIZE_BITS_BY_LENGTH]().unsafe_get(length)
        )
        var index = address & ((1 << shift) - 1)
        var transform = address >> shift
        if transform >= TRANSFORMS_COUNT:
            raise CorruptError(
                t"brotli: dictionary transform {transform} at {pos}"
            )
        var word = Int(
            global_constant[OFFSETS_BY_LENGTH]().unsafe_get(length)
        ) + (index * length)
        var buf = Array[UInt8, 64](fill=0)
        var n = _transform(word, length, transform, buf)
        if n == 0 and distance <= 120:
            raise CorruptError(t"brotli: an empty dictionary word at {pos}")
        if n > end - pos:
            raise CorruptError(
                t"brotli: a dictionary word at {pos} overruns the meta-block"
            )
        for k in range(n):
            dst[pos + k] = buf[k]
        return pos + n


def _transform(
    word: Int,
    var length: Int,
    transform: Int,
    mut out: Array[UInt8, 64],
) -> Int:
    """Write dictionary word `word` of `length` bytes into `out` under
    transform `transform`: its prefix, the word cut or uppercased, its
    suffix. Return the bytes written."""
    ref transforms = global_constant[TRANSFORMS]()
    ref map = global_constant[PREFIX_SUFFIX_MAP]()
    ref affixes = global_constant[PREFIX_SUFFIX]()
    var n = 0
    var prefix = Int(map[Int(transforms[3 * transform])])
    for k in range(Int(affixes[prefix])):
        out[n] = affixes[prefix + 1 + k]
        n += 1
    var kind = Int(transforms[3 * transform + 1])
    var start = word
    if kind <= OMIT_LAST_9:
        length -= kind
    elif kind >= OMIT_FIRST_1 and kind <= OMIT_FIRST_9:
        var skip = kind - (OMIT_FIRST_1 - 1)
        start += skip
        length -= skip
    length = max(length, 0)  # a 4-byte word may omit up to 9
    var first = n
    copy_word(start, length, out, n)
    n += length
    if kind == UPPERCASE_FIRST:
        _ = _uppercase(out, first)
    elif kind == UPPERCASE_ALL:
        var at = first
        var left = length
        while left > 0:
            var step = _uppercase(out, at)
            at += step
            left -= step
    var suffix = Int(map[Int(transforms[3 * transform + 2])])
    for k in range(Int(affixes[suffix])):
        out[n] = affixes[suffix + 1 + k]
        n += 1
    return n


def _uppercase(mut buf: Array[UInt8, 64], i: Int) -> Int:
    """Uppercase the character at `i` as Brotli does -- ASCII letters, and
    for a multi-byte character a fixed bit of its second or third byte;
    return its bytes."""
    var c = buf[i]
    if c < 0xC0:
        if c >= 97 and c <= 122:
            buf[i] = c ^ 32
        return 1
    if c < 0xE0:
        buf[i + 1] ^= 32
        return 2
    buf[i + 2] ^= 5
    return 3


# ---------------------------------------------------------------------------
# Compressing: libbrotli's quality 1, its "two-pass" fragment compressor.
#
# | Type | What it is |
# |---|---|
# | `_Node` | a Huffman tree node, as libbrotli builds and sorts them |
# | `_BitStream` | the stream being written, as libbrotli's `BrotliWriteBits` writes it |
# | `_TwoPass` | the compressor: its hash table, command and literal buffers, codes |
#
# Input is compressed in blocks of 128 KiB. A first pass finds each block's
# matches -- hashing 4 or 6 bytes, skipping ahead faster the longer it finds
# none -- into a list of commands and literals; a second builds Huffman codes
# from their counts and writes the block as one meta-block: no block
# switches, no context modeling, a code for literals, one for commands and
# one for distances. A block that would not shrink is stored. Commands use a
# 128-symbol subset of the command alphabet: an insert alone, a copy alone,
# or a distance.
# ---------------------------------------------------------------------------


comptime _BLOCK_SIZE = 1 << 17
comptime _HASH_MUL = UInt64(0x1E35A7BD)
comptime _MAX_DISTANCE = (1 << 18) - 16
comptime _WINDOW_BITS = 22
"""The window written: what libbrotli's one-shot compression writes at the
Parquet writer's `lgwin`. Matches stay within 256 KiB of their copy."""


@fieldwise_init
struct _Node(TrivialRegisterPassable):
    """A Huffman tree node: its count, and either two children or, with no
    left child, a symbol in `right`."""

    var count: UInt32
    var left: Int16
    var right: Int16

    comptime SENTINEL = Self(UInt32.MAX, -1, -1)

    @always_inline
    def before[counts_only: Bool](self, other: Self) -> Bool:
        """libbrotli's `SortHuffmanTree`: the rarer first, and of equal counts
        the higher symbol -- or, as its fast tree builder has it, the rarer
        first and nothing more."""
        comptime if counts_only:
            return self.count < other.count
        else:
            if self.count != other.count:
                return self.count < other.count
            return self.right > other.right


def _sort_nodes[counts_only: Bool](mut items: List[_Node], n: Int):
    """libbrotli's `SortHuffmanTreeItems`: an insertion sort below 13 items,
    a shell sort above -- not stable, so its order must be its own."""
    if n < 13:
        for i in range(1, n):
            var tmp = items[i]
            var k = i
            var j = i - 1
            while tmp.before[counts_only](items[j]):
                items[k] = items[j]
                k = j
                if j == 0:
                    break
                j -= 1
            items[k] = tmp
    else:
        var gaps: Array[Int, 6] = [132, 57, 23, 10, 4, 1]
        var g = 2 if n < 57 else 0
        while g < 6:
            var gap = gaps[g]
            for i in range(gap, n):
                var j = i
                var tmp = items[i]
                while j >= gap and tmp.before[counts_only](items[j - gap]):
                    items[j] = items[j - gap]
                    j -= gap
                items[j] = tmp
            g += 1


def _set_depth(
    p0: Int, pool: List[_Node], depth: Span[mut=True, UInt8, _], max_depth: Int
) -> Bool:
    """Each leaf's depth below `p0`; False if one is deeper than
    `max_depth`."""
    var stack = Array[Int, 16](fill=-1)
    var level = 0
    var p = p0
    while True:
        if pool[p].left >= 0:
            level += 1
            if level > max_depth:
                return False
            stack[level] = Int(pool[p].right)
            p = Int(pool[p].left)
            continue
        else:
            depth[Int(pool[p].right)] = UInt8(level)
        while level >= 0 and stack[level] == -1:
            level -= 1
        if level < 0:
            return True
        p = stack[level]
        stack[level] = -1


def _huffman_tree[
    counts_only: Bool = False
](
    counts: Span[UInt32, _],
    tree_limit: Int,
    mut tree: List[_Node],
    depth: Span[mut=True, UInt8, _],
):
    """libbrotli's `BrotliCreateHuffmanTree`: code lengths for `counts`, at
    most `tree_limit` long, counts below a floor raised to it -- doubling
    the floor until the code fits. With `counts_only`, nodes of equal count
    are left in the order its fast tree builder sorts them into."""
    var length = len(counts)
    var count_limit = UInt32(1)
    while True:
        var n = 0
        var i = length
        while i != 0:
            i -= 1
            if counts[i] != 0:
                tree[n] = _Node(max(counts[i], count_limit), -1, Int16(i))
                n += 1
        if n == 1:
            depth[Int(tree[0].right)] = 1
            break
        _sort_nodes[counts_only](tree, n)
        tree[n] = _Node.SENTINEL
        tree[n + 1] = _Node.SENTINEL
        i = 0
        var j = n + 1
        var k = n - 1
        while k != 0:
            var left: Int
            var right: Int
            if tree[i].count <= tree[j].count:
                left = i
                i += 1
            else:
                left = j
                j += 1
            if tree[i].count <= tree[j].count:
                right = i
                i += 1
            else:
                right = j
                j += 1
            var end = 2 * n - k
            tree[end] = _Node(
                tree[left].count + tree[right].count, Int16(left), Int16(right)
            )
            tree[end + 1] = _Node.SENTINEL
            k -= 1
        if _set_depth(2 * n - 1, tree, depth, tree_limit):
            break
        count_limit *= 2


def _depths_to_codes(depth: Span[UInt8, _], bits: Span[mut=True, UInt16, _]):
    """The canonical code of each symbol with a length, bit-reversed as the
    stream sends it."""
    var count = Array[Int, 16](fill=0)
    for d in depth:
        count[Int(d)] += 1
    count[0] = 0
    var next = Array[Int, 16](fill=0)
    var code = 0
    for i in range(1, 16):
        code = (code + count[i - 1]) << 1
        next[i] = code
    for i in range(len(depth)):
        var d = Int(depth[i])
        if d != 0:
            bits[i] = UInt16(_reversed(next[d], d))
            next[d] += 1


# fmt: off
comptime _CODE_LENGTH_BITS: Array[UInt8, 18] = [
    0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 15, 31, 0, 11, 7,
]
comptime _CODE_LENGTH_DEPTH: Array[UInt8, 18] = [
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 0, 4, 4,
]
# fmt: on
"""The static code-length code the fast tree writer uses."""


def _runs_table(zeros: Bool) -> Array[UInt64, 704]:
    """libbrotli's `kZeroRepsBits`/`kZeroRepsDepth` and
    `kNonZeroRepsBits`/`kNonZeroRepsDepth` in one: a run of `r` zero code
    lengths -- or of `r + 3` repeats of the last nonzero one -- under the
    static code-length code, its bits from bit 8 and their count in the low
    8."""
    var out = Array[UInt64, 704](fill=0)
    var code_bits = materialize[_CODE_LENGTH_BITS]()
    for r in range(704):
        var codes = List[UInt8]()
        var extras = List[UInt8]()
        if zeros:
            _write_tree_zeros(r, codes, extras)
        else:
            _write_run(16, r + 3, codes, extras)
        var bits = UInt64(0)
        var n = 0
        for i in range(len(codes)):
            var code = Int(codes[i])
            bits |= UInt64(code_bits[code]) << UInt64(n)
            n += 4
            bits |= UInt64(extras[i]) << UInt64(n)
            n += _run_extra_bits(code)
        out[r] = bits << 8 | UInt64(n)
    return out^


comptime _ZERO_RUNS = _runs_table(True)
comptime _REPEAT_RUNS = _runs_table(False)


struct _BitStream(Movable):
    """The stream being written, its bits from the least significant of each
    byte up. A write ORs into the byte it starts in and stores eight, so the
    buffer keeps eight bytes past the last bit, and what lies above the last
    bit is always zero."""

    var buf: List[UInt8]
    var ix: Int
    """The bits written."""

    def __init__(out self):
        self.buf = List[UInt8]()
        self.ix = 0

    def reserve(mut self, n: Int):
        """Room for `n` more bytes."""
        var need = (self.ix >> 3) + n + 16
        if len(self.buf) < need:
            var had = len(self.buf)
            self.buf.resize(unsafe_uninit_length=need)
            # A write ORs into the byte it starts in; every later byte it
            # stores whole.
            if had <= self.ix >> 3:
                self.buf[self.ix >> 3] = 0

    @always_inline
    def write(mut self, n: Int, bits: UInt64):
        """Append the `n` (at most 56) bits of `bits`, which has no others
        set."""
        var p = self.ix >> 3
        var span = Span(self.buf)
        var v = UInt64(span.unsafe_get(p)) | (bits << UInt64(self.ix & 7))
        LittleEndian.store[DType.uint64](span, p, v)
        self.ix += n

    def align(mut self):
        self.ix = (self.ix + 7) & ~7

    def rewind(mut self, ix: Int):
        """Back to bit `ix`, clearing what was written after it."""
        self.buf[ix >> 3] &= UInt8((1 << (ix & 7)) - 1)
        self.ix = ix

    def bytes(mut self, src: Span[UInt8, _]):
        """Append `src` at the byte boundary the stream is at."""
        var p = self.ix >> 3
        for i in range(len(src)):
            self.buf[p + i] = src[i]
        self.ix += 8 * len(src)
        self.buf[self.ix >> 3] = 0


struct _BitCursor[o: MutOrigin](TrivialRegisterPassable):
    """A run of writes into a `_BitStream`'s bytes from bit `ix` on, the
    pending bits held in a register and stored eight bytes at a time --
    what a loop writing many short codes keeps its stream in."""

    var _out: BufferView[DType.uint8, Self.o]
    var _pos: Int
    var _acc: UInt64
    var _n: Int

    def __init__(out self, dst: BufferView[DType.uint8, Self.o], ix: Int):
        self._out = dst
        self._pos = ix >> 3
        self._n = ix & 7
        self._acc = UInt64(dst.unsafe_get(self._pos)) & (
            (UInt64(1) << UInt64(self._n)) - 1
        )

    @always_inline
    def put(mut self, n: Int, bits: UInt64):
        """Append the `n` (at most 24) bits of `bits`, which has no others
        set."""
        self._acc |= bits << UInt64(self._n)
        self._n += n
        if self._n >= 40:
            self._store()
            var k = self._n >> 3
            self._pos += k
            self._acc >>= UInt64(8 * k)
            self._n &= 7

    @always_inline
    def _store(self):
        self._out.store_unaligned[8](
            self._pos, bitcast[DType.uint8, 8](self._acc)
        )

    def finish(mut self) -> Int:
        """Store what is pending; return the bits the stream now holds."""
        self._store()
        return 8 * self._pos + self._n


def _store_meta_block_header(mut w: _BitStream, n: Int, uncompressed: Bool):
    """A meta-block of `n` bytes, not the last."""
    var nibbles = 6
    if n <= 1 << 16:
        nibbles = 4
    elif n <= 1 << 20:
        nibbles = 5
    w.write(1, 0)
    w.write(2, UInt64(nibbles - 4))
    w.write(4 * nibbles, UInt64(n - 1))
    w.write(1, UInt64(Int(uncompressed)))


def _store_uncompressed(mut w: _BitStream, src: Span[UInt8, _]):
    _store_meta_block_header(w, len(src), True)
    w.align()
    w.bytes(src)


def _write_tree_repetitions(
    previous: UInt8,
    value: UInt8,
    var reps: Int,
    mut tree: List[UInt8],
    mut extra: List[UInt8],
):
    """libbrotli's `BrotliWriteHuffmanTreeRepetitions`: `reps` code lengths
    of `value`, after `previous`."""
    if previous != value:
        tree.append(value)
        extra.append(0)
        reps -= 1
    if reps == 7:
        tree.append(value)
        extra.append(0)
        reps -= 1
    if reps < 3:
        for _ in range(reps):
            tree.append(value)
            extra.append(0)
    else:
        _write_run(16, reps, tree, extra)


def _write_tree_zeros(
    var reps: Int, mut tree: List[UInt8], mut extra: List[UInt8]
):
    """libbrotli's `BrotliWriteHuffmanTreeRepetitionsZeros`."""
    if reps == 11:
        tree.append(0)
        extra.append(0)
        reps -= 1
    if reps < 3:
        for _ in range(reps):
            tree.append(0)
            extra.append(0)
    else:
        _write_run(17, reps, tree, extra)


@always_inline
def _run_extra_bits(code: Int) -> Int:
    """The extra bits after a code-length code: 2 for 16, a repeat of the
    last nonzero length, 3 for 17, a run of zeros, and none for a length."""
    return 2 if code == 16 else (3 if code == 17 else 0)


def _write_run(
    code: UInt8, var reps: Int, mut tree: List[UInt8], mut extra: List[UInt8]
):
    """A run of `reps`, at least 3, as repeat codes `code` -- 16 or 17 --
    each with its extra bits, the most significant first."""
    var bits = _run_extra_bits(Int(code))
    var start = len(tree)
    reps -= 3
    while True:
        tree.append(code)
        extra.append(UInt8(reps & ((1 << bits) - 1)))
        reps >>= bits
        if reps == 0:
            break
        reps -= 1
    _reverse(tree, start)
    _reverse(extra, start)


def _reverse(mut v: List[UInt8], start: Int):
    var a = start
    var b = len(v) - 1
    while a < b:
        var t = v[a]
        v[a] = v[b]
        v[b] = t
        a += 1
        b -= 1


def _write_huffman_tree(
    depth: Span[UInt8, _], mut tree: List[UInt8], mut extra: List[UInt8]
):
    """libbrotli's `BrotliWriteHuffmanTree`: code lengths as code-length
    symbols, runs of zeros or of one length coded as runs where that pays."""
    var length = len(depth)
    var n = length
    for i in range(length):
        if depth[length - i - 1] == 0:
            n -= 1
        else:
            break
    var rle_nonzero = False
    var rle_zero = False
    if length > 50:
        var total_zero = 0
        var total_nonzero = 0
        var count_zero = 1
        var count_nonzero = 1
        var i = 0
        while i < n:
            var value = depth[i]
            var reps = 1
            var k = i + 1
            while k < n and depth[k] == value:
                reps += 1
                k += 1
            if reps >= 3 and value == 0:
                total_zero += reps
                count_zero += 1
            if reps >= 4 and value != 0:
                total_nonzero += reps
                count_nonzero += 1
            i += reps
        rle_nonzero = total_nonzero > count_nonzero * 2
        rle_zero = total_zero > count_zero * 2
    var previous = UInt8(8)
    var i = 0
    while i < n:
        var value = depth[i]
        var reps = 1
        if (value != 0 and rle_nonzero) or (value == 0 and rle_zero):
            var k = i + 1
            while k < n and depth[k] == value:
                reps += 1
                k += 1
        if value == 0:
            _write_tree_zeros(reps, tree, extra)
        else:
            _write_tree_repetitions(previous, value, reps, tree, extra)
            previous = value
        i += reps


def _store_huffman_tree(
    depth: Span[UInt8, _], mut nodes: List[_Node], mut w: _BitStream
):
    """libbrotli's `BrotliStoreHuffmanTree`: the code with lengths `depth`,
    as a complex prefix code."""
    var tree = List[UInt8](capacity=704)
    var extra = List[UInt8](capacity=704)
    _write_huffman_tree(depth, tree, extra)
    var histogram = Array[UInt32, 18](fill=0)
    for t in tree:
        histogram[Int(t)] += 1
    var num_codes = 0
    var code = 0
    for i in range(18):
        if histogram[i] != 0:
            if num_codes == 0:
                code = i
                num_codes = 1
            elif num_codes == 1:
                num_codes = 2
                break
    var cl_depth = Array[UInt8, 18](fill=0)
    var cl_bits = Array[UInt16, 18](fill=0)
    _huffman_tree(Span(histogram), 5, nodes, Span(cl_depth))
    _depths_to_codes(Span(cl_depth), Span(cl_bits))

    # The code-length code's own lengths, in their order, trailing zeros
    # cut, under a fixed code.
    ref order = global_constant[_CODE_LENGTH_ORDER]()
    var symbols: Array[Int, 6] = [0, 7, 3, 2, 1, 15]
    var lengths: Array[Int, 6] = [2, 4, 3, 2, 2, 4]
    var to_store = 18
    if num_codes > 1:
        while to_store > 0:
            if cl_depth[Int(order[to_store - 1])] != 0:
                break
            to_store -= 1
    var skip = 0
    if cl_depth[Int(order[0])] == 0 and cl_depth[Int(order[1])] == 0:
        skip = 2
        if cl_depth[Int(order[2])] == 0:
            skip = 3
    w.write(2, UInt64(skip))
    for i in range(skip, to_store):
        var l = Int(cl_depth[Int(order[i])])
        w.write(lengths[l], UInt64(symbols[l]))
    if num_codes == 1:
        cl_depth[code] = 0
    for i in range(len(tree)):
        var ix = Int(tree[i])
        w.write(Int(cl_depth[ix]), UInt64(cl_bits[ix]))
        if ix >= 16:
            w.write(_run_extra_bits(ix), UInt64(extra[i]))


def _store_tree_fast(
    counts: Span[UInt32, _],
    total: Int,
    max_bits: Int,
    mut nodes: List[_Node],
    depth: Span[mut=True, UInt8, _],
    bits: Span[mut=True, UInt16, _],
    mut w: _BitStream,
):
    """libbrotli's `BrotliBuildAndStoreHuffmanTreeFast`: a code for `counts`,
    which sum to `total`, at most 14 bits long, stored as a simple code for
    up to four symbols and otherwise under the static code-length code."""
    var count = 0
    var symbols = Array[Int, 4](fill=0)
    var length = 0
    var left = total
    while left != 0:
        if counts[length] != 0:
            if count < 4:
                symbols[count] = length
            count += 1
            left -= Int(counts[length])
        length += 1
    if count <= 1:
        w.write(4, 1)
        w.write(max_bits, UInt64(symbols[0]))
        depth[symbols[0]] = 0
        bits[symbols[0]] = 0
        return
    for i in range(length):
        depth[i] = 0
    _huffman_tree[counts_only=True](counts[:length], 14, nodes, depth)
    _depths_to_codes(depth[:length], bits)
    if count <= 4:
        w.write(2, 1)
        w.write(2, UInt64(count - 1))
        for i in range(count):
            for j in range(i + 1, count):
                if depth[symbols[j]] < depth[symbols[i]]:
                    var t = symbols[j]
                    symbols[j] = symbols[i]
                    symbols[i] = t
        for i in range(count):
            w.write(max_bits, UInt64(symbols[i]))
        if count == 4:
            w.write(1, UInt64(Int(depth[symbols[0]] == 1)))
    else:
        # The static code-length code: 18 lengths in 40 bits.
        w.write(40, 0x0000FF55555554)
        ref zero_runs = global_constant[_ZERO_RUNS]()
        ref repeat_runs = global_constant[_REPEAT_RUNS]()
        ref cl_bits = global_constant[_CODE_LENGTH_BITS]()
        ref cl_depth = global_constant[_CODE_LENGTH_DEPTH]()
        var previous = UInt8(8)
        var i = 0
        while i < length:
            var value = depth[i]
            var reps = 1
            var k = i + 1
            while k < length and depth[k] == value:
                reps += 1
                k += 1
            i += reps
            if value == 0:
                var r = zero_runs[reps]
                w.write(Int(r & 0xFF), r >> 8)
            else:
                var v = Int(value)
                if previous != value:
                    w.write(Int(cl_depth[v]), UInt64(cl_bits[v]))
                    reps -= 1
                if reps < 3:
                    while reps != 0:
                        reps -= 1
                        w.write(Int(cl_depth[v]), UInt64(cl_bits[v]))
                else:
                    var r = repeat_runs[reps - 3]
                    w.write(Int(r & 0xFF), r >> 8)
                previous = value


@always_inline
def _log2_floor(n: Int) -> Int:
    # Not `std.bit.log2_floor`: in the command emitters it measured 3% slower
    # compressing ints and floats.
    return bit_width(n) - 1


def _fast_log2(v: Int) -> Float64:
    """libbrotli's `FastLog2`, whose table holds single-precision values."""
    if v == 0:
        return 0
    if v < 256:
        return Float64(Float32(log2(Float64(v))))
    return log2(Float64(v))


@always_inline
def _load64(src: BufferView[DType.uint8, _], p: Int) -> UInt64:
    return bitcast[DType.uint64, 1](src.load_unaligned[8](p))


@always_inline
def _load32(src: BufferView[DType.uint8, _], p: Int) -> UInt32:
    return bitcast[DType.uint32, 1](src.load_unaligned[4](p))


@always_inline
def _hash_of(v: UInt64, shift: Int, min_match: Int) -> Int:
    """The table slot of the `min_match` bytes at the bottom of `v`."""
    return Int(
        ((v << UInt64(8 * (8 - min_match))) * _HASH_MUL) >> UInt64(shift)
    )


@always_inline
def _is_match(
    src: BufferView[DType.uint8, _], a: Int, b: Int, min_match: Int
) -> Bool:
    if _load32(src, a) != _load32(src, b):
        return False
    if min_match == 4:
        return True
    return src.unsafe_get(a + 4) == src.unsafe_get(b + 4) and src.unsafe_get(
        a + 5
    ) == src.unsafe_get(b + 5)


@fieldwise_init
struct _Commands[co: MutOrigin, lo: MutOrigin](TrivialRegisterPassable):
    """A block's commands and literals as they are found: each command a
    symbol of the 128 in its low byte and its extra bits above -- 0-23 an
    insert length, 24-63 a copy length, 64-127 a distance."""

    var commands: Span[UInt32, Self.co]
    var literals: Span[UInt8, Self.lo]
    var nc: Int
    var nl: Int

    @always_inline
    def push(mut self, cmd: Int):
        self.commands.unsafe_get(self.nc) = UInt32(cmd)
        self.nc += 1

    @always_inline
    def insert(mut self, src: BufferView[DType.uint8, _], start: Int, end: Int):
        """The literals from `start` to `end`, and the insert length that
        precedes them."""
        var n = end - start
        if n < 6:
            self.push(n)
        elif n < 130:
            var tail = n - 2
            var nbits = _log2_floor(tail) - 1
            var prefix = tail >> nbits
            self.push(
                ((nbits << 1) + prefix + 2) | (tail - (prefix << nbits)) << 8
            )
        elif n < 2114:
            var tail = n - 66
            var nbits = _log2_floor(tail)
            self.push((nbits + 10) | (tail - (1 << nbits)) << 8)
        elif n < 6210:
            self.push(21 | (n - 2114) << 8)
        elif n < 22594:
            self.push(22 | (n - 6210) << 8)
        else:
            self.push(23 | (n - 22594) << 8)
        BufferView(self.literals).slice(self.nl).copy_from(src.slice(start), n)
        self.nl += n

    @always_inline
    def copy(mut self, n: Int):
        if n < 10:
            self.push(n + 38)
        elif n < 134:
            var tail = n - 6
            var nbits = _log2_floor(tail) - 1
            var prefix = tail >> nbits
            self.push(
                ((nbits << 1) + prefix + 44) | (tail - (prefix << nbits)) << 8
            )
        elif n < 2118:
            var tail = n - 70
            var nbits = _log2_floor(tail)
            self.push((nbits + 52) | (tail - (1 << nbits)) << 8)
        else:
            self.push(63 | (n - 2118) << 8)

    @always_inline
    def copy_last_distance(mut self, n: Int):
        if n < 12:
            self.push(n + 20)
        elif n < 72:
            var tail = n - 8
            var nbits = _log2_floor(tail) - 1
            var prefix = tail >> nbits
            self.push(
                ((nbits << 1) + prefix + 28) | (tail - (prefix << nbits)) << 8
            )
        else:
            if n < 136:
                var tail = n - 8
                self.push(((tail >> 5) + 54) | (tail & 31) << 8)
            elif n < 2120:
                var tail = n - 72
                var nbits = _log2_floor(tail)
                self.push((nbits + 52) | (tail - (1 << nbits)) << 8)
            else:
                self.push(63 | (n - 2120) << 8)
            self.push(64)

    @always_inline
    def distance(mut self, distance: Int):
        var d = distance + 3
        var nbits = _log2_floor(d) - 1
        var prefix = (d >> nbits) & 1
        var offset = (2 + prefix) << nbits
        self.push((2 * (nbits - 1) + prefix + 80) | (d - offset) << 8)


@always_inline
def _rehash(
    src: BufferView[DType.uint8, _],
    table: Span[mut=True, Int32, _],
    ip: Int,
    shift: Int,
    min_match: Int,
    exact: Bool,
) -> Int:
    """Hash some positions within the copy just emitted into the table;
    return the candidate for `ip`, which takes its slot. libbrotli's first
    such update after literals hashes one position twice (`exact` False),
    which its output depends on."""
    var cur: Int
    if min_match == 4:
        var v = _load64(src, ip - 3)
        cur = _hash_of(v >> 24, shift, 4)
        table.unsafe_get(_hash_of(v, shift, 4)) = Int32(ip - 3)
        table.unsafe_get(_hash_of(v >> 8, shift, 4)) = Int32(ip - 2)
        table.unsafe_get(_hash_of(v >> 16 if exact else v, shift, 4)) = Int32(
            ip - 1
        )
    else:
        var v = _load64(src, ip - 5)
        table.unsafe_get(_hash_of(v, shift, 6)) = Int32(ip - 5)
        table.unsafe_get(_hash_of(v >> 8, shift, 6)) = Int32(ip - 4)
        table.unsafe_get(_hash_of(v >> 16, shift, 6)) = Int32(ip - 3)
        v = _load64(src, ip - 2)
        cur = _hash_of(v >> 16, shift, 6)
        table.unsafe_get(_hash_of(v, shift, 6)) = Int32(ip - 2)
        table.unsafe_get(_hash_of(v >> 8, shift, 6)) = Int32(ip - 1)
    var candidate = Int(table.unsafe_get(cur))
    table.unsafe_get(cur) = Int32(ip)
    return candidate


def _create_commands(
    span: Span[UInt8, _],
    start: Int,
    block_size: Int,
    input_size: Int,
    table_bits: Int,
    min_match: Int,
    table: Span[mut=True, Int32, _],
    mut out: _Commands,
):
    """libbrotli's `CreateCommands`: the commands and literals of the block
    of `block_size` bytes at `start` of the fragment `span`, `input_size`
    bytes being left in it."""
    var src = BufferView(span)
    var ip = start
    var shift = 64 - table_bits
    var ip_end = start + block_size
    var next_emit = start
    var last_distance = -1
    if block_size >= 16:
        var len_limit = min(block_size - min_match, input_size - 16)
        var ip_limit = start + len_limit
        ip += 1
        var next_hash = _hash_of(_load64(src, ip), shift, min_match)
        while True:
            var skip = 32
            var next_ip = ip
            var candidate = 0
            var found = False
            # Scan forward for a match, looking further apart the longer
            # none turns up.
            while True:
                var hash = next_hash
                var step = skip >> 5
                skip += 1
                ip = next_ip
                next_ip = ip + step
                if next_ip > ip_limit:
                    break
                next_hash = _hash_of(_load64(src, next_ip), shift, min_match)
                candidate = ip - last_distance
                if _is_match(src, ip, candidate, min_match) and candidate < ip:
                    table.unsafe_get(hash) = Int32(ip)
                else:
                    candidate = Int(table.unsafe_get(hash))
                    table.unsafe_get(hash) = Int32(ip)
                    if not _is_match(src, ip, candidate, min_match):
                        continue
                if ip - candidate > _MAX_DISTANCE:
                    continue
                found = True
                break
            if not found:
                break

            # The match, after the literals before it, then as many matches
            # as follow straight after.
            var matched = min_match + match_length(
                span, ip + min_match, candidate + min_match, ip_end
            )
            var distance = ip - candidate
            out.insert(src, next_emit, ip)
            if distance == last_distance:
                out.push(64)
            else:
                out.distance(distance)
                last_distance = distance
            out.copy_last_distance(matched)
            ip += matched
            next_emit = ip
            if ip >= ip_limit:
                break
            candidate = _rehash(src, table, ip, shift, min_match, False)
            var more = True
            while ip - candidate <= _MAX_DISTANCE and _is_match(
                src, ip, candidate, min_match
            ):
                var copied = min_match + match_length(
                    span, ip + min_match, candidate + min_match, ip_end
                )
                last_distance = ip - candidate
                ip += copied
                out.copy(copied)
                out.distance(last_distance)
                next_emit = ip
                if ip >= ip_limit:
                    more = False
                    break
                candidate = _rehash(src, table, ip, shift, min_match, True)
            if not more:
                break
            ip += 1
            next_hash = _hash_of(_load64(src, ip), shift, min_match)
    if next_emit < ip_end:
        out.insert(src, next_emit, ip_end)


# fmt: off
comptime _EXTRA_BITS: Array[UInt8, 128] = [
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
    6, 7, 8, 9, 10, 12, 14, 24, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
    17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
]
"""Per command symbol, its extra bits."""
comptime _INSERT_OFFSET: Array[UInt32, 24] = [
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26,
    34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210, 22594,
]
# fmt: on
"""Per insert symbol, its length's base."""


struct _TwoPass(Movable):
    """libbrotli's quality-1 compressor: `BrotliCompressFragmentTwoPass` and
    what it keeps between blocks."""

    var table: List[Int32]
    var commands: List[UInt32]
    """Per command, a symbol of the 128 in its low byte and its extra bits
    above: 0-23 an insert length, 24-63 a copy length, 64-127 a distance."""
    var literals: List[UInt8]
    var lit_histo: List[UInt32]
    var cmd_histo: List[UInt32]
    var lit_depth: List[UInt8]
    var lit_bits: List[UInt16]
    var cmd_depth: List[UInt8]
    var cmd_bits: List[UInt16]
    var nodes: List[_Node]

    def __init__(out self):
        self.table = List[Int32]()
        self.commands = List[UInt32]()
        self.literals = List[UInt8]()
        self.lit_histo = List[UInt32](length=256, fill=0)
        self.cmd_histo = List[UInt32](length=128, fill=0)
        self.lit_depth = List[UInt8](length=256, fill=0)
        self.lit_bits = List[UInt16](length=256, fill=0)
        self.cmd_depth = List[UInt8](length=128, fill=0)
        self.cmd_bits = List[UInt16](length=128, fill=0)
        self.nodes = List[_Node](length=2 * 704 + 1, fill=_Node.SENTINEL)

    def fragment(
        mut self, src: Span[UInt8, _], is_last: Bool, mut w: _BitStream
    ):
        """Compress `src`, matches reaching back no further than its start,
        as meta-blocks of at most `_BLOCK_SIZE` bytes; end the stream after
        them if `is_last`."""
        var n = len(src)
        var table_size = 256
        while table_size < (1 << 17) and table_size < n:
            table_size <<= 1
        var table_bits = _log2_floor(table_size)
        var min_match = 4 if table_bits <= 15 else 6
        self.table.resize(unsafe_uninit_length=table_size)
        Span(self.table).fill(0)
        if len(self.commands) < _BLOCK_SIZE:
            self.commands.resize(unsafe_uninit_length=_BLOCK_SIZE)
            self.literals.resize(unsafe_uninit_length=_BLOCK_SIZE)
        w.reserve(2 * n + 503)
        var initial = w.ix
        var at = 0
        while at < n:
            var size = min(n - at, _BLOCK_SIZE)
            var found = _Commands(
                Span(self.commands), Span(self.literals), 0, 0
            )
            _create_commands(
                src,
                at,
                size,
                n - at,
                table_bits,
                min_match,
                Span(self.table),
                found,
            )
            var num_commands = found.nc
            var num_literals = found.nl
            if self._should_compress(src[at : at + size], num_literals):
                _store_meta_block_header(w, size, False)
                # One block type of each kind, no distance parameters, a
                # literal context mode of 0 and one tree of each kind.
                w.write(13, 0)
                self._store_commands(num_literals, num_commands, w)
            else:
                _store_uncompressed(w, src[at : at + size])
            at += size
        if w.ix - initial > 31 + 8 * n:
            w.rewind(initial)
            _store_uncompressed(w, src)
        if is_last:
            w.write(1, 1)
            w.write(1, 1)
            w.align()

    def _should_compress(
        mut self, block: Span[UInt8, _], num_literals: Int
    ) -> Bool:
        """Whether the block is worth coding: fewer literals than 98% of it,
        or literals cheaper than 98% of 8 bits by a 1-in-43 sample."""
        var size = Float64(len(block))
        if Float64(num_literals) < 0.98 * size:
            return True
        var max_cost = size * 8 * 0.98 / 43
        var histo = Span(self.lit_histo)
        histo.fill(0)
        var i = 0
        while i < len(block):
            histo.unsafe_get(Int(block.unsafe_get(i))) += 1
            i += 43
        var sum = 0
        var cost = Float64(0)
        for k in range(256):
            var p = Int(self.lit_histo[k])
            sum += p
            cost -= Float64(p) * _fast_log2(p)
        if sum != 0:
            cost += Float64(sum) * _fast_log2(sum)
        if cost < Float64(sum):
            cost = Float64(sum)
        return cost < max_cost

    def _store_commands(
        mut self, num_literals: Int, num_commands: Int, mut w: _BitStream
    ):
        """libbrotli's `StoreCommands`: the literal, command and distance
        codes, then each command with its literals."""
        var lit_histo = Span(self.lit_histo)
        var cmd_histo = Span(self.cmd_histo)
        lit_histo.fill(0)
        cmd_histo.fill(0)
        Span(self.cmd_depth).fill(0)
        Span(self.cmd_bits).fill(0)
        var literals = Span(self.literals)
        for i in range(num_literals):
            lit_histo.unsafe_get(Int(literals.unsafe_get(i))) += 1
        _store_tree_fast(
            Span(self.lit_histo),
            num_literals,
            8,
            self.nodes,
            Span(self.lit_depth),
            Span(self.lit_bits),
            w,
        )
        var commands = Span(self.commands)
        for i in range(num_commands):
            cmd_histo.unsafe_get(Int(commands.unsafe_get(i) & 0xFF)) += 1
        self.cmd_histo[1] += 1
        self.cmd_histo[2] += 1
        self.cmd_histo[64] += 1
        self.cmd_histo[84] += 1
        self._store_command_codes(w)
        ref extras = global_constant[_EXTRA_BITS]()
        ref offsets = global_constant[_INSERT_OFFSET]()
        var lit_depth = Span(self.lit_depth)
        var lit_bits = Span(self.lit_bits)
        var cmd_depth = Span(self.cmd_depth)
        var cmd_bits = Span(self.cmd_bits)
        var bits = _BitCursor(BufferView(Span(w.buf)), w.ix)
        var lit = 0
        for i in range(num_commands):
            var cmd = commands.unsafe_get(i)
            var code = Int(cmd & 0xFF)
            var extra = cmd >> 8
            bits.put(
                Int(cmd_depth.unsafe_get(code)),
                UInt64(cmd_bits.unsafe_get(code)),
            )
            bits.put(Int(extras.unsafe_get(code)), UInt64(extra))
            if code < 24:
                var insert = Int(offsets.unsafe_get(code) + extra)
                for _ in range(insert):
                    var b = Int(literals.unsafe_get(lit))
                    bits.put(
                        Int(lit_depth.unsafe_get(b)),
                        UInt64(lit_bits.unsafe_get(b)),
                    )
                    lit += 1
        w.ix = bits.finish()

    def _store_command_codes(mut self, mut w: _BitStream):
        """libbrotli's `BuildAndStoreCommandPrefixCode`: codes for the 64
        command and 64 distance symbols, the commands' stored as the full
        704-symbol alphabet they are a subset of."""
        var depth = Span(self.cmd_depth)
        _huffman_tree(Span(self.cmd_histo)[:64], 15, self.nodes, depth[:64])
        _huffman_tree(Span(self.cmd_histo)[64:], 14, self.nodes, depth[64:])
        # Canonical codes are assigned in the full alphabet's order, which
        # puts the copies first, then the inserts by their copy-length cell.
        var tmp_depth = List[UInt8](length=704, fill=0)
        var tmp_bits = List[UInt16](length=64, fill=0)
        for k in range(24):
            tmp_depth[k] = depth[24 + k]
        for k in range(8):
            tmp_depth[24 + k] = depth[k]
            tmp_depth[32 + k] = depth[48 + k]
            tmp_depth[40 + k] = depth[8 + k]
            tmp_depth[48 + k] = depth[56 + k]
            tmp_depth[56 + k] = depth[16 + k]
        _depths_to_codes(Span(tmp_depth)[:64], Span(tmp_bits))
        for k in range(8):
            self.cmd_bits[k] = tmp_bits[24 + k]
            self.cmd_bits[8 + k] = tmp_bits[40 + k]
            self.cmd_bits[16 + k] = tmp_bits[56 + k]
            self.cmd_bits[48 + k] = tmp_bits[32 + k]
            self.cmd_bits[56 + k] = tmp_bits[48 + k]
        for k in range(24):
            self.cmd_bits[24 + k] = tmp_bits[k]
        _depths_to_codes(depth[64:], Span(self.cmd_bits)[64:])
        for k in range(64):
            tmp_depth[k] = 0
        for k in range(8):
            tmp_depth[k] = depth[24 + k]
            tmp_depth[64 + k] = depth[32 + k]
            tmp_depth[128 + k] = depth[40 + k]
            tmp_depth[192 + k] = depth[48 + k]
            tmp_depth[384 + k] = depth[56 + k]
        for k in range(8):
            tmp_depth[128 + 8 * k] = depth[k]
            tmp_depth[256 + 8 * k] = depth[8 + k]
            tmp_depth[448 + 8 * k] = depth[16 + k]
        _store_huffman_tree(Span(tmp_depth), self.nodes, w)
        _store_huffman_tree(depth[64:], self.nodes, w)


# ---------------------------------------------------------------------------
# Brotli streams, and `Brotli` -- the codec Parquet's BROTLI pages are
# compressed with.
#
# A stream is the window size, then meta-blocks, the last one marked: each
# compressed, stored, or metadata the decoder skips.
# ---------------------------------------------------------------------------


struct Brotli(Compressor):
    """Brotli compression, in Mojo."""

    @staticmethod
    def max_compressed_length(n: Int) -> Int:
        """The most `compress` can produce for `n` bytes: the bytes stored,
        in meta-blocks of 16 KiB with a header each -- libbrotli's
        `BrotliEncoderMaxCompressedSize`."""
        if n == 0:
            return 2
        return n + 2 + 4 * (n >> 14) + 3 + 1

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        """The most `n` bytes of stream decode to: stored bytes as they are,
        and at most 16 MiB a compressed meta-block, whose header alone takes
        more than 8 bytes -- what a reader checks a length it was given
        against, before allocating it."""
        return n + ((n // 8) << 24)

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        """Append one stream holding `src` to `dst`, as libbrotli's one-shot
        `BrotliEncoderCompress` writes it at quality 1 with a 4 MiB window
        -- the same bytes, with libbrotli 1.2.0."""
        var n = len(src)
        if n == 0:
            dst.append(6)
            return
        var w = _BitStream()
        w.reserve(16)
        w.write(4, UInt64(((_WINDOW_BITS - 17) << 1) | 1))
        var encoder = _TwoPass()
        var at = 0
        while at < n:
            var size = min(n - at, 1 << _WINDOW_BITS)
            encoder.fragment(src[at : at + size], at + size == n, w)
            at += size
        var size = w.ix >> 3
        if size <= Self.max_compressed_length(n):
            dst.extend(Span(w.buf)[:size])
        else:
            Self._stored(src, dst)

    @staticmethod
    def _stored(src: Span[UInt8, _], mut dst: List[UInt8]):
        """The stream libbrotli falls back to when compressing would grow
        the input: a 1 KiB window, an empty metadata block, and the bytes in
        stored meta-blocks of up to 16 MiB."""
        dst.append(0x21)
        dst.append(0x03)
        var at = 0
        while at < len(src):
            var size = min(len(src) - at, 1 << 24)
            var nibbles = 0
            if size > 1 << 16:
                nibbles = 2 if size > 1 << 20 else 1
            var bits = (
                (nibbles << 1) | ((size - 1) << 3) | (1 << (19 + 4 * nibbles))
            )
            LittleEndian.append(dst, UInt64(bits), 4 if nibbles == 2 else 3)
            dst.extend(src[at : at + size])
            at += size
        dst.append(3)

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        """Decompress the stream `src` into exactly `dst`, whose length must
        be the decompressed one. Nothing past `len(dst)` is written; on
        corrupt input, what was decoded before the corruption is left in
        `dst`."""
        if len(src) == 0:
            raise CorruptError("brotli: no stream")
        var decoder = _Decoder()
        decoder.stream(src, dst)


# ---------------------------------------------------------------------------
# The same format through the system's libbrotli.
# ---------------------------------------------------------------------------


struct LibBrotli(Compressor):
    """Brotli through libbrotli: `BrotliEncoderCompress` at quality 11,
    window 22, and `BrotliDecoderDecompress`."""

    @staticmethod
    def max_decompressed_length(n: Int) -> Int:
        return Brotli.max_decompressed_length(n)

    @staticmethod
    def compress(
        src: Span[mut=False, UInt8, _], mut dst: List[UInt8]
    ) raises DynError:
        var e = library["brotlienc"]()
        var bound = Int(
            e.call["BrotliEncoderMaxCompressedSize", UInt](UInt(len(src)))
        )
        if bound == 0:
            bound = len(src) + len(src) // 2 + 512
        var at = append_uninit(dst, bound)
        var size = UInt(bound)
        # BROTLI_BOOL BrotliEncoderCompress(int quality, int lgwin,
        #   BrotliEncoderMode mode, size_t input_size, const uint8_t* input,
        #   size_t* encoded_size, uint8_t* encoded); mode 0 is GENERIC.
        var rc = e.call["BrotliEncoderCompress", Int32](
            Int32(11),
            Int32(22),
            Int32(0),
            UInt(len(src)),
            src.unsafe_ptr(),
            Pointer(to=size),
            dst.unsafe_ptr().unsafe_offset(at),
        )
        if Int(rc) != 1:
            dst.shrink(at)
            raise InternalError("brotli: libbrotli could not compress")
        dst.shrink(at + Int(size))

    @staticmethod
    def decompress[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises DynError:
        var size = UInt(len(dst))
        # BrotliDecoderResult BrotliDecoderDecompress(size_t encoded_size,
        #   const uint8_t* encoded, size_t* decoded_size, uint8_t* decoded);
        # returns BROTLI_DECODER_RESULT_SUCCESS == 1.
        var rc = library["brotlidec"]().call["BrotliDecoderDecompress", Int32](
            len(src), src.unsafe_ptr(), Pointer(to=size), dst.unsafe_ptr()
        )
        if Int(rc) != 1 or Int(size) != len(dst):
            raise CorruptError("brotli: libbrotli could not decompress")
