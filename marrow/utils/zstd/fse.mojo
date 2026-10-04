# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# The encoding half is ported from libzstd 1.5.7's `fse_compress.c`,
# Copyright (c) Meta Platforms, Inc. (BSD 3-Clause); see NOTICE.txt.

"""Finite State Entropy -- the coder of zstd's sequences and Huffman weights.

| Type | What it is |
|---|---|
| `Alphabet` | what a table's symbols are, and what each one stands for |
| `Distribution` | a count per symbol summing to `1 << log`: read, written, normalized |
| `FseEntry` | one state of a decoding table |
| `FseTable` | a decoding table, spread from a distribution |
| `FseEncoder` | an encoding table, spread the same way |
| `FseState` | one encoding state as a stream is written |

A distribution's count of -1 means "less than 1". `FseTable` and `FseEncoder`
both lay a distribution over the states with `Distribution.spread`, so both ends
agree on a table without it being sent. A decoding table's entries carry what
their symbol *means* -- a literal or match length's baseline and extra bits, or
an offset code's `1 << code` and `code` extra bits -- rather than the bare
symbol, as libzstd's do, so decoding a field is one lookup.
"""

from std.bit import log2_floor
from std.builtin.globals import global_constant

from ...errors import CorruptError
from ..byteorder import LittleEndian
from .bits import BitWriter


comptime MAX_WEIGHT = 11
"""The largest Huffman weight, and so the deepest Huffman code."""

comptime ENCODER_SYMBOLS = 64
"""Symbols an encoding table has room for: 53 match length codes at most."""


@fieldwise_init
struct Alphabet(Equatable, ImplicitlyCopyable, Movable):
    """What an FSE table's symbols are: one of a sequence's three fields --
    literal length, offset or match length codes -- or the weights of a
    Huffman tree. It knows how many symbols there are, how accurate a table
    a block may give them, and what a code stands for, both ways."""

    var index: Int
    """The field's place in a sequence and in the modes byte: 0, 1 or 2; 3
    for the weights."""

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index

    comptime LITERAL_LENGTHS = Self(0)
    comptime OFFSETS = Self(1)
    comptime MATCH_LENGTHS = Self(2)
    comptime WEIGHTS = Self(3)

    comptime _LITERAL_LENGTH_CODES = Self._codes[36](
        16,
        0,
        [
            16,
            18,
            20,
            22,
            24,
            28,
            32,
            40,
            48,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
        [1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )
    """The spec's "Literals length codes": `extra << 24 | baseline`."""

    comptime _MATCH_LENGTH_CODES = Self._codes[53](
        32,
        3,
        [
            35,
            37,
            39,
            41,
            43,
            47,
            51,
            59,
            67,
            83,
            99,
            131,
            259,
            515,
            1027,
            2051,
            4099,
            8195,
            16387,
            32771,
            65539,
        ],
        [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )
    """The spec's "Match length codes": `extra << 24 | baseline`."""

    comptime _LITERAL_LENGTH_CODE_OF = Self._code_of[64](0)
    comptime _MATCH_LENGTH_CODE_OF = Self._code_of[128](2)
    """The codes of the short lengths, the match lengths less 3; beyond them
    a code follows the length's top bit -- `ZSTD_LLcode` and `ZSTD_MLcode`."""

    @staticmethod
    def _codes[
        n: Int
    ](direct: Int, add: Int, bases: List[Int], extras: List[Int]) -> Array[
        UInt32, n
    ]:
        """A length code table: codes below `direct` stand for themselves
        plus `add`, with no extra bits; the rest for `bases` and `extras` in
        order."""
        var t = Array[UInt32, n](fill=0)
        for c in range(n):
            if c < direct:
                t[c] = UInt32(c + add)
            else:
                t[c] = UInt32(extras[c - direct] << 24 | bases[c - direct])
        return t^

    @staticmethod
    def _code_of[n: Int](index: Int) -> Array[UInt8, n]:
        """Each value below `n`'s code: the last whose baseline it reaches,
        for the literal lengths (`index` 0) or the match lengths less 3."""
        var alphabet = Self(index)
        var t = Array[UInt8, n](fill=0)
        for v in range(n):
            var value = v if index == 0 else v + 3
            var code = 0
            while (
                code < alphabet.max_symbol()
                and Int(alphabet.base(code + 1)) <= value
            ):
                code += 1
            t[v] = UInt8(code)
        return t^

    @always_inline
    def max_symbol(self) -> Int:
        """The largest symbol: the limits libzstd sets on a 64-bit target,
        offsets up to `2**32 - 4`."""
        if self == Self.LITERAL_LENGTHS:
            return 35
        elif self == Self.MATCH_LENGTHS:
            return 52
        elif self == Self.OFFSETS:
            return 31
        else:
            return MAX_WEIGHT

    def max_log(self) -> Int:
        """The largest accuracy a block may give a table of these symbols."""
        if self == Self.OFFSETS:
            return 8
        elif self == Self.WEIGHTS:
            return 6
        else:
            return 9

    def default_log(self) -> Int:
        """The accuracy of a sequence field's predefined distribution."""
        return 5 if self == Self.OFFSETS else 6

    @always_inline
    def _length(self, code: Int) -> UInt32:
        """A literal or match length code's `extra << 24 | baseline`."""
        if self == Self.LITERAL_LENGTHS:
            return global_constant[Self._LITERAL_LENGTH_CODES]().unsafe_get(
                code
            )
        else:
            return global_constant[Self._MATCH_LENGTH_CODES]().unsafe_get(code)

    @always_inline
    def base(self, code: Int) -> UInt32:
        """The smallest value `code` stands for: a length's baseline, an
        offset code's `1 << code`, a weight itself."""
        if self == Self.OFFSETS:
            return UInt32(1) << UInt32(code)
        elif self == Self.WEIGHTS:
            return UInt32(code)
        else:
            return self._length(code) & ((1 << 24) - 1)

    @always_inline
    def extra(self, code: Int) -> Int:
        """The bits `code` adds to its `base`."""
        if self == Self.OFFSETS:
            return code
        elif self == Self.WEIGHTS:
            return 0
        else:
            return Int(self._length(code) >> 24)

    @always_inline
    def code(self, value: Int) -> Int:
        """The code standing for `value`, which `base` and `extra` take back
        to it: a literal length, a match length, or an offset as sent --
        `ZSTD_LLcode`, `ZSTD_MLcode` and the offset's top bit."""
        if self == Self.LITERAL_LENGTHS:
            if value < 64:
                return Int(
                    global_constant[Self._LITERAL_LENGTH_CODE_OF]().unsafe_get(
                        value
                    )
                )
            return log2_floor(value) + 19
        elif self == Self.MATCH_LENGTHS:
            var v = value - 3
            if v < 128:
                return Int(
                    global_constant[Self._MATCH_LENGTH_CODE_OF]().unsafe_get(v)
                )
            return log2_floor(v) + 36
        else:
            return log2_floor(value)


struct Distribution(Movable):
    """A normalized distribution: a count per symbol, summing to
    `1 << log`, a count of -1 meaning "less than 1" -- libzstd's normalized
    counter. A block carries one in a compact form `read` parses and
    `write` produces; an encoder makes one from symbol counts with
    `normalize`."""

    comptime _LITERAL_LENGTH_DEFAULT: List[Int16] = [
        4,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
    ]
    comptime _MATCH_LENGTH_DEFAULT: List[Int16] = [
        1,
        4,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    comptime _OFFSET_DEFAULT: List[Int16] = [
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    """The spec's "Default Distributions", at logs 6, 6 and 5."""

    var counts: List[Int16]
    var log: Int

    def __init__(out self):
        """An empty distribution, to read or normalize into."""
        self.counts = List[Int16](capacity=64)
        self.log = 0

    def __init__(out self, var counts: List[Int16], log: Int):
        self.counts = counts^
        self.log = log

    @staticmethod
    def predefined(field: Alphabet) -> Self:
        """A sequence field's predefined distribution -- the spec's "Default
        Distributions"."""
        var counts: List[Int16]
        if field == Alphabet.LITERAL_LENGTHS:
            counts = materialize[Self._LITERAL_LENGTH_DEFAULT]()
        elif field == Alphabet.MATCH_LENGTHS:
            counts = materialize[Self._MATCH_LENGTH_DEFAULT]()
        else:
            counts = materialize[Self._OFFSET_DEFAULT]()
        return Self(counts^, field.default_log())

    def read(
        mut self, src: Span[UInt8, _], alphabet: Alphabet
    ) raises CorruptError -> Int:
        """Parse the distribution at the start of `src`; return the bytes it
        took.

        A forward little-endian bitstream: the log less 5 in 4 bits, then
        each symbol's count plus one in as few bits as the counts still
        unassigned allow -- one fewer for the smallest values -- and after a
        zero count, a run of further zeros in 2-bit groups."""
        ref counts = self.counts
        counts.clear()
        if len(src) == 0:
            raise CorruptError("zstd: an empty FSE table description")
        var log = 5 + Int(src[0] & 15)
        if log > alphabet.max_log():
            raise CorruptError(t"zstd: FSE table log {log} is over the maximum")
        var top = alphabet.max_symbol()
        var remaining = 1 << log
        var pos = 4

        while remaining > 0:
            if len(counts) > top:
                raise CorruptError("zstd: an FSE table with too many symbols")
            var bits = log2_floor(remaining + 1) + 1
            var value = Self._take(src, pos, bits)
            var low = (1 << (bits - 1)) - 1
            var threshold = (1 << bits) - 1 - (remaining + 1)
            if value & low < threshold:
                pos -= 1
                value &= low
            elif value > low:
                value -= threshold
            var count = value - 1
            remaining -= -count if count < 0 else count
            counts.append(Int16(count))
            if count == 0:
                while True:
                    var run = Self._take(src, pos, 2)
                    for _ in range(run):
                        if len(counts) > top:
                            raise CorruptError(
                                "zstd: an FSE table with too many symbols"
                            )
                        counts.append(0)
                    if run != 3:
                        break
        if remaining != 0:
            raise CorruptError("zstd: FSE table counts do not sum to its size")
        var used = (pos + 7) >> 3
        if used > len(src):
            raise CorruptError("zstd: truncated FSE table description")
        self.log = log
        return used

    @staticmethod
    @always_inline
    def _take(src: Span[UInt8, _], mut pos: Int, n: Int) -> Int:
        """`n` bits at bit `pos` of a forward stream; past its end they read
        as zero, and the reader checks `pos` once at the end."""
        var word = LittleEndian.partial[DType.uint32](src, pos >> 3)
        var v = Int(word >> UInt32(pos & 7)) & ((1 << n) - 1)
        pos += n
        return v

    def spread(self, mut symbols: Array[UInt8, 1 << 12]):
        """Lay the symbols over the `1 << log` states of a table, one per
        state, as the spec prescribes ("From normalized distribution to
        decoding tables"): each "less than 1" symbol takes one state from
        the top down, and the rest go in symbol order by a step coprime
        with the size, skipping those. With no "less than 1" symbol no
        state is skipped, and the symbols are laid out in order and then
        scattered two at a time -- `FSE_buildCTable_wksp`'s fast path, the
        same table without a branch per state. The counts must sum to the
        size, as `read` checks and `normalize` makes them."""
        var counts = Span(self.counts)
        var size = 1 << self.log
        var mask = size - 1
        var n = len(counts)
        var step = (size >> 1) + (size >> 3) + 3
        var high = size - 1
        for s in range(n):
            if counts.unsafe_get(s) == -1:
                symbols.unsafe_get(high) = UInt8(s)
                high -= 1
        var pos = 0
        if high == size - 1:
            var spread = Array[UInt8, 1 << 12](uninitialized=True)
            var at = 0
            for s in range(n):
                for _ in range(Int(counts.unsafe_get(s))):
                    spread.unsafe_get(at) = UInt8(s)
                    at += 1
            for u in range(0, size, 2):
                symbols.unsafe_get(pos) = spread.unsafe_get(u)
                symbols.unsafe_get((pos + step) & mask) = spread.unsafe_get(
                    u + 1
                )
                pos = (pos + 2 * step) & mask
        else:
            for s in range(n):
                for _ in range(Int(counts.unsafe_get(s))):
                    symbols.unsafe_get(pos) = UInt8(s)
                    pos = (pos + step) & mask
                    while pos > high:
                        pos = (pos + step) & mask
        debug_assert(pos == 0, "FSE counts that do not sum to the size")

    def write(self, mut out: List[UInt8]) raises CorruptError:
        """Append the compact form `read` parses -- `FSE_writeNCount`."""
        var norm = Span(self.counts)
        var log = self.log
        var n = len(norm)
        var size = 1 << log
        var bits = log - 5
        var count_bits = 4
        var remaining = size + 1
        var threshold = size
        var width = log + 1
        var symbol = 0
        var after_zero = False

        @always_inline
        def emit16(mut out: List[UInt8], mut bits: Int):
            out.append(UInt8(bits & 0xFF))
            out.append(UInt8((bits >> 8) & 0xFF))
            bits >>= 16

        while symbol < n and remaining > 1:
            if after_zero:
                var start = symbol
                while symbol < n and norm[symbol] == 0:
                    symbol += 1
                if symbol == n:
                    break
                while symbol >= start + 24:
                    start += 24
                    bits += 0xFFFF << count_bits
                    emit16(out, bits)
                while symbol >= start + 3:
                    start += 3
                    bits += 3 << count_bits
                    count_bits += 2
                bits += (symbol - start) << count_bits
                count_bits += 2
                if count_bits > 16:
                    emit16(out, bits)
                    count_bits -= 16
            var count = Int(norm[symbol])
            symbol += 1
            var top = (2 * threshold - 1) - remaining
            remaining -= -count if count < 0 else count
            count += 1
            if count >= threshold:
                count += top
            bits += count << count_bits
            count_bits += width - Int(count < top)
            after_zero = count == 1
            if remaining < 1:
                raise CorruptError("zstd: an FSE distribution over its size")
            while remaining < threshold:
                width -= 1
                threshold >>= 1
            if count_bits > 16:
                emit16(out, bits)
                count_bits -= 16
        if remaining != 1:
            raise CorruptError("zstd: an FSE distribution short of its size")
        out.append(UInt8(bits & 0xFF))
        if count_bits > 8:
            out.append(UInt8((bits >> 8) & 0xFF))

    @staticmethod
    def optimal_log(max_log: Int, total: Int, top: Int, minus: Int = 2) -> Int:
        """The log for `total` symbols of which `top` is the largest:
        `max_log` at most, less when there are few symbols to describe,
        never too small to give each symbol a state --
        `FSE_optimalTableLog_internal`. `total` is at least 2."""
        var log = max_log
        # In libzstd this is unsigned, so a negative never lowers the log.
        var from_total = log2_floor(total - 1) - minus
        if from_total >= 0 and from_total < log:
            log = from_total
        var floor = min(log2_floor(total) + 1, log2_floor(max(top, 1)) + 2)
        if floor > log:
            log = floor
        return max(5, min(log, 12))

    def normalize(
        mut self, counts: Span[Int, _], total: Int, log: Int, low: Bool
    ) raises CorruptError:
        """Scale `counts`, summing to `total`, to sum to `1 << log` with
        every present symbol keeping a state -- `FSE_normalizeCount`. With
        `low`, a rare symbol gets the "less than 1" count -1 rather than
        1."""
        self.log = log
        ref norm = self.counts
        comptime rest_to_beat: List[Int] = [
            0,
            473195,
            504333,
            520860,
            550000,
            700000,
            750000,
            830000,
        ]
        var n = len(counts)
        norm.resize(n, 0)
        var low_count = Int16(-1) if low else Int16(1)
        var scale = 62 - log
        var step = (1 << 62) // total
        var v_step = 1 << (scale - 20)
        var still = 1 << log
        var largest = 0
        var largest_count = 0
        var low_threshold = total >> log
        for s in range(n):
            var c = counts[s]
            if c == 0:
                norm[s] = 0
            elif c <= low_threshold:
                norm[s] = low_count
                still -= 1
            else:
                var proba = (c * step) >> scale
                if proba < 8:
                    var beat = v_step * materialize[rest_to_beat]()[proba]
                    proba += Int((c * step) - (proba << scale) > beat)
                if proba > largest_count:
                    largest_count = proba
                    largest = s
                norm[s] = Int16(proba)
                still -= proba
        if -still >= Int(norm[largest] >> 1):
            self._spread(counts, total, low_count)
        else:
            norm[largest] += Int16(still)

    def _spread(
        mut self, counts: Span[Int, _], var total: Int, low_count: Int16
    ) raises CorruptError:
        """The fallback when rounding left the largest symbol short:
        `FSE_normalizeM2`."""
        var log = self.log
        ref norm = self.counts
        comptime UNSET = Int16(-2)
        var n = len(counts)
        var distributed = 0
        var low_threshold = total >> log
        var low_one = (total * 3) >> (log + 1)
        for s in range(n):
            var c = counts[s]
            if c == 0:
                norm[s] = 0
            elif c <= low_threshold:
                norm[s] = low_count
                distributed += 1
                total -= c
            elif c <= low_one:
                norm[s] = 1
                distributed += 1
                total -= c
            else:
                norm[s] = UNSET
        var to_distribute = (1 << log) - distributed
        if to_distribute == 0:
            return
        if total // to_distribute > low_one:
            low_one = (total * 3) // (to_distribute * 2)
            for s in range(n):
                if norm[s] == UNSET and counts[s] <= low_one:
                    norm[s] = 1
                    distributed += 1
                    total -= counts[s]
            to_distribute = (1 << log) - distributed
        if distributed == n:
            var top = 0
            for s in range(n):
                if counts[s] > counts[top]:
                    top = s
            norm[top] += Int16(to_distribute)
            return
        if total == 0:
            var s = 0
            while to_distribute > 0:
                if norm[s] > 0:
                    to_distribute -= 1
                    norm[s] += 1
                s = (s + 1) % n
            return
        var v_step_log = 62 - log
        var mid = (1 << (v_step_log - 1)) - 1
        var r_step = ((1 << v_step_log) * to_distribute + mid) // total
        var acc = mid
        for s in range(n):
            if norm[s] == UNSET:
                var end = acc + counts[s] * r_step
                var weight = (end >> v_step_log) - (acc >> v_step_log)
                if weight < 1:
                    raise CorruptError("zstd: FSE normalization failed")
                norm[s] = Int16(weight)
                acc = end


@fieldwise_init
struct FseEntry(TrivialRegisterPassable):
    """One state: what it decodes to, and how to reach the next."""

    var base: UInt32
    """The symbol, or the baseline of the value it codes."""
    var next: UInt16
    """The next state, less the bits read for it."""
    var bits: UInt8
    """The bits read for the next state."""
    var extra: UInt8
    """The extra bits added to `base`."""


struct FseTable(Movable):
    """A decoding table, kept across blocks for the "repeat" mode."""

    var entries: List[FseEntry]
    var log: Int
    var ready: Bool
    """Whether a table has been built in this frame."""
    var _predefined: Bool
    """Whether it is the predefined one, which a table built once keeps:
    a table serves one alphabet."""

    def __init__(out self):
        self.entries = List[FseEntry](capacity=1 << 9)
        self.log = 0
        self.ready = False
        self._predefined = False

    def build(mut self, dist: Distribution, alphabet: Alphabet):
        """The table for `dist`, whose counts `Distribution.read` or the spec
        guarantees sum to its size."""
        var counts = Span(dist.counts)
        var log = dist.log
        var size = 1 << log
        var n = len(counts)
        var symbols = Array[UInt8, 1 << 12](uninitialized=True)
        dist.spread(symbols)
        # What each symbol decodes to, resolved once rather than per state,
        # and how many states it has.
        var next = Array[Int, 64](fill=0)
        var bases = Array[UInt32, 64](fill=0)
        var extras = Array[UInt8, 64](fill=0)
        for s in range(n):
            next[s] = abs(Int(counts.unsafe_get(s)))
            bases[s] = alphabet.base(s)
            extras[s] = UInt8(alphabet.extra(s))
        # A symbol's states, in order, take ranges of the next state that
        # tile the table: the first ones a bit wider than the rest.
        self.entries.resize(size, FseEntry(0, 0, 0, 0))
        var table = Span(self.entries)
        for i in range(size):
            var s = Int(symbols.unsafe_get(i))
            var d = next.unsafe_get(s)
            next.unsafe_get(s) = d + 1
            var bits = log - log2_floor(d)
            table.unsafe_get(i) = FseEntry(
                bases.unsafe_get(s),
                UInt16((d << bits) - size),
                UInt8(bits),
                extras.unsafe_get(s),
            )
        self.log = log
        self.ready = True
        self._predefined = False

    def rle(mut self, symbol: Int, alphabet: Alphabet) raises CorruptError:
        """A one-state table that decodes to `symbol` every time."""
        if symbol > alphabet.max_symbol():
            raise CorruptError(t"zstd: RLE symbol {symbol} is out of range")
        self.entries.resize(1, FseEntry(0, 0, 0, 0))
        self.entries[0] = FseEntry(
            alphabet.base(symbol), 0, 0, UInt8(alphabet.extra(symbol))
        )
        self.log = 0
        self.ready = True
        self._predefined = False

    def predefined(mut self, alphabet: Alphabet):
        """The table of the spec's default distribution for `alphabet`."""
        if not self._predefined:
            self.build(Distribution.predefined(alphabet), alphabet)
            self._predefined = True
        self.ready = True

    def read(
        mut self, src: Span[UInt8, _], alphabet: Alphabet
    ) raises CorruptError -> Int:
        """Build the table described at the start of `src`; return the bytes
        the description took."""
        var dist = Distribution()
        var used = dist.read(src, alphabet)
        self.build(dist, alphabet)
        return used


# ---------------------------------------------------------------------------
# encoding -- ported from libzstd's `fse_compress.c`
# ---------------------------------------------------------------------------


struct FseEncoder(Movable):
    """An encoding table -- `FSE_CTable`. For symbol `s`, entry `2s` offsets
    the bit count and `2s + 1` is where its states start among the next
    states, which follow from `2 * ENCODER_SYMBOLS` grouped by symbol. One
    flat list, so a hot loop holds it as one span."""

    var _data: List[Int32]
    var log: Int

    def __init__(out self):
        """A table that allocates when it is first built."""
        self._data = List[Int32]()
        self.log = 0

    def build(mut self, dist: Distribution):
        """The table for `dist`, spread as `FseTable.build` spreads it."""
        var norm = Span(dist.counts)
        var log = dist.log
        var size = 1 << log
        var n = len(norm)
        var symbols = Array[UInt8, 1 << 12](uninitialized=True)
        dist.spread(symbols)
        # Where each symbol's states start among the next states.
        var cumul = Array[Int, ENCODER_SYMBOLS + 1](fill=0)
        for s in range(n):
            cumul.unsafe_get(s + 1) = cumul.unsafe_get(s) + abs(
                Int(norm.unsafe_get(s))
            )
        self._data.resize(2 * ENCODER_SYMBOLS + size, 0)
        var data = Span(self._data)
        comptime NEXT = 2 * ENCODER_SYMBOLS
        for u in range(size):
            var s = Int(symbols.unsafe_get(u))
            data.unsafe_get(NEXT + cumul.unsafe_get(s)) = Int32(size + u)
            cumul.unsafe_get(s) += 1
        var total = 0
        for s in range(n):
            var c = Int(norm.unsafe_get(s))
            var bits: Int
            var find = 0
            if c == 0:
                bits = ((log + 1) << 16) - size
            elif c == -1 or c == 1:
                bits = (log << 16) - size
                find = total - 1
                total += 1
            else:
                var max_out = log - log2_floor(c - 1)
                bits = (max_out << 16) - (c << max_out)
                find = total - c
                total += c
            data.unsafe_get(2 * s) = Int32(bits)
            data.unsafe_get(2 * s + 1) = Int32(find)
        self.log = log

    def rle(mut self, symbol: Int):
        """A table that encodes `symbol`, and nothing else, in no bits."""
        self._data.resize(2 * ENCODER_SYMBOLS + 1, 0)
        self._data[2 * symbol] = 0
        self._data[2 * symbol + 1] = 0
        self._data[2 * ENCODER_SYMBOLS] = 0
        self.log = 0

    def start(self, symbol: Int) -> FseState[origin_of(self._data)]:
        """The state a stream starts from, carrying its last symbol --
        `FSE_initCState2`: that symbol costs no bits."""
        var bits = Int(self._data[2 * symbol])
        var out = (bits + (1 << 15)) >> 16
        var value = (out << 16) - bits
        var find = Int(self._data[2 * symbol + 1])
        return FseState(
            Span(self._data),
            self.log,
            Int(self._data[2 * ENCODER_SYMBOLS + (value >> out) + find]),
        )


@fieldwise_init
struct FseState[m: Bool, //, o: Origin[mut=m]](TrivialRegisterPassable):
    """One encoding state as a stream is written, the symbols last first --
    `FSE_CState_t`. It holds its table as a span, so a loop keeps the whole
    of it in registers; written last, it is what the decoder reads first."""

    var _table: Span[Int32, Self.o]
    var _log: Int
    var _value: Int

    @always_inline
    def encode(mut self, symbol: Int, mut w: BitWriter[_]):
        """Move to a state that also carries `symbol`, writing the bits that
        undo the move -- `FSE_encodeSymbol`."""
        var out = (self._value + Int(self._table.unsafe_get(2 * symbol))) >> 16
        w.add(self._value, out)
        self._value = Int(
            self._table.unsafe_get(
                2 * ENCODER_SYMBOLS
                + (self._value >> out)
                + Int(self._table.unsafe_get(2 * symbol + 1))
            )
        )

    def flush(self, mut w: BitWriter[_]):
        """Write the state itself -- `FSE_flushCState`."""
        w.add(self._value, self._log)
