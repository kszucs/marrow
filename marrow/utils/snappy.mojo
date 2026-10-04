# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

# Ported from google/snappy 1.2.2 (`snappy.cc`, `snappy-internal.h`),
# Copyright 2005-2011 Google Inc. (BSD 3-Clause); the licence is reproduced in
# NOTICE.txt.

"""Snappy raw-block compression, in Mojo.

The format is google/snappy's `format_description.txt`: a little-endian varint
of the uncompressed length, then tags -- a literal (`00`), or a copy of 4..11
bytes at an 11-bit offset (`01`), 1..64 bytes at a 16-bit offset (`10`) or
1..64 bytes at a 32-bit offset (`11`). A copy may overlap its own output
(offset < length), which is how a run is spelled.

**The compressor is libsnappy's level 1, byte for byte**: 64 KiB fragments,
a `uint16` hash table of 256..32768 entries, the same hash, the 16-position
probe and the skip heuristic. libsnappy hashes with CRC32C when it is built
for a target with the instruction -- conda-forge's macOS arm64 build is -- and
with a `0x1e35a7bd` multiply otherwise, so `compress` takes that choice as a
parameter. It defaults to the multiply: portable, and measured as fast as the
instruction, where a portable CRC32C is 7-10x slower to compress with. Same
output means the tests compare bytes with libsnappy rather than only
round-tripping.

`Snappy` is the whole public surface. The decoder's copy primitives are shared
with the LZ4 and Zstandard codecs in `lz77.mojo`; the rest is Snappy's own: the
`_Compressor` and its `_Fragment` match finder, the `_Emitter` that turns
matches into tags, and the `_Decoder` that reads them -- one stream, or two that
`interleave`. They hold their bytes as `Span`s, indexed checked, except where a
check was measured to cost (below): the decoder's fast loop, the hash table and
the wide loads. Those go through `BufferView` and `LittleEndian`, whose bounds
are `debug_assert`s -- checked in a test build (`-D ASSERT=all`), free in a
release one -- so the tests catch an access a hot loop's argued bound got wrong.
The one raw address left picks a copy's source in `_fast_tag`. All of these are
small values the hot loops copy into locals: their methods take `self` by
reference, and a store through an output pointer could otherwise force every
position back to memory.

**The decoder is libsnappy's two-tier loop.** The fast loop takes two tags
per bounds check with unconditional 32-byte copies, NEON `tbl` pattern
extension for overlapping copies, and the tag table in rodata
(`global_constant`); the checked path takes what it hands back -- a long
literal, a copy-4, an error, the last ~130 input bytes -- with every bound
checked. Neither reads past `len(src)` nor writes past the destination: a page
buffer and a memory-mapped file carry no padding.

**Measured** on an M4 Max against the conda-forge libsnappy 1.2.2 marrow
`dlopen`s, both `-O3`, its entry points resolved once, thread CPU time, best of
100-400 calls alternated with libsnappy's, on `bench_snappy.mojo`'s
Parquet-shaped corpora (libsnappy time / ours, so above 1 is faster):

| | strings | ints | floats |
|---|---|---|---|
| `decompress_into` | 1.00-1.08 | 1.12-1.15 | 1.00 |
| `decompress_pair_into` (vs two calls) | 1.44-1.53 | 1.64-1.84 | -- |
| `compress` | 0.90-0.98 | 0.90-0.94 | 0.99-1.13 |

`floats` decodes as one long literal, a `memcpy` either way. The weak spot is a
synthetic stream of nothing but 3-byte literals and 5-byte copies -- tag
throughput with no work per tag -- which decodes at 0.88-1.02.

**`decompress_pair_into` is the one that beats libsnappy outright.** A stream
decodes serially, each tag's position depending on the previous tag's length,
so one stream leaves most of the core idle; two interleaved tag for tag run two
such chains side by side. Three and four streams gained less than two on
1 MiB pages.

Measured and reverted, each slower here:

- libsnappy's deferral of a tag's copy into the next tag's decode, 0.67-0.91;
- choosing literal or copy *data* rather than *address*, 0.27-0.65;
- unconditional 64-byte copies, 0.76 on offset-8 copies;
- libc `memset` for the hash table, no change;
- compressing two fragments interleaved, 0.46-0.71 against 0.79-0.99 -- each
  probe ends in a hard-to-predict branch, so a second stream adds
  mispredictions rather than filling idle slots (inferred, not verified);
- a 16-byte SIMD match length, 0.86 against 0.98 on `strings`: matches are
  short, and moving the vector mask out costs more than an 8-byte XOR.
- libsnappy's prefetch 64 bytes ahead in its match finder, no change -- a
  64 KiB fragment and its table fit in a 128 KiB L1 (inferred, not verified).

**What the unchecked code is worth**, each against its stdlib or checked
replacement in one process, order rotated per call (replacement time / this
code; an identical copy of the module reads 0.96-1.03):

- `LzCopy.blocks64`'s whole 32-byte blocks against an exact-length
  `unsafe_memcpy`: 1.85-2.35 decoding `strings`, 3.0-4.1 decoding pairs;
- selecting the copy's source *address* against branching on literal or copy:
  1.08-1.12 decoding `strings`, 1.18-1.37 decoding pairs;
- the fast loop's `BufferView` accesses, asserted in a test build only,
  against checked `Span` indexing: 1.16-1.30 decoding pairs;
- `bulk_copy` against `unsafe_memcpy`, whose long copies are a 32-byte loop
  ending byte by byte: 1.05-1.11 on incompressible `floats`;
- writing the `tbl`/`pshufb` intrinsics out against the stdlib's private
  `SIMD._dynamic_shuffle`: equal, so the stdlib's is used.

Replaced by the plainer form: the compressor's 16-byte block copies for
literals, a 16-byte loop in the checked path's overlapping copy,
`unsafe_memset_zero` for the table, and unchecked indexing anywhere outside the
fast loop. Each measured equal on its own; together they decode pairs of
`strings` 1.5-3.3% slower than before and are otherwise at parity. The checks
run a few times per page, so that is the hot loop compiling differently, not
their cost (inferred, not verified).
"""

from std.bit import count_trailing_zeros, log2_floor, next_power_of_two
from std.builtin.globals import global_constant
from std.sys.intrinsics import unlikely

from ..errors import CorruptError, InvalidError
from ..views import BufferView
from .byteorder import LittleEndian
from .checksum import Crc32c
from .lz77 import LzCopy


# ---------------------------------------------------------------------------
# compressor
# ---------------------------------------------------------------------------


struct _Compressor(Movable):
    """libsnappy's level-1 compressor: the input in `BLOCK_SIZE` fragments,
    each with a freshly cleared hash table, so no copy reaches back past 64 KiB
    and the table fits `uint16` positions. Owns the table's storage."""

    comptime BLOCK_SIZE = 1 << 16
    comptime MIN_TABLE = 1 << 8
    comptime MAX_TABLE_BITS = 15
    comptime MAX_TABLE = 1 << Self.MAX_TABLE_BITS

    var _slots: List[UInt16]

    def __init__(out self, n: Int):
        """Storage for the largest table an `n`-byte input needs."""
        self._slots = List[UInt16](unsafe_uninit_length=Self.table_size(n))

    @staticmethod
    def table_size(n: Int) -> Int:
        """Entries for an `n`-byte input: the next power of two, within
        `[MIN_TABLE, MAX_TABLE]` -- libsnappy's `CalculateTableSize`."""
        return min(max(next_power_of_two(n), Self.MIN_TABLE), Self.MAX_TABLE)

    def compress[
        crc32c: Bool
    ](mut self, src: Span[UInt8, _], mut out: _Emitter[_]):
        """Every fragment of `src`, in order, through `out`."""
        var pos = 0
        while pos < len(src):
            var n = min(len(src) - pos, Self.BLOCK_SIZE)
            var slots = Span(self._slots)[: Self.table_size(n)]
            slots.fill(0)
            _Fragment(src[pos : pos + n]).compress(
                out, _Table[crc32c=crc32c](slots)
            )
            pos += n


struct _Table[o: MutOrigin, crc32c: Bool](TrivialRegisterPassable):
    """One fragment's view of the hash table: the last position seen for each
    hash of 4 input bytes -- libsnappy's `TableEntry`."""

    var _slots: BufferView[DType.uint16, Self.o]
    var _mask: UInt32
    """`2 * (len(_slots) - 1)`: a hash is a *byte* offset, as in libsnappy.
    The table is a power of two long, so a masked hash is always in bounds."""

    def __init__(out self, slots: Span[UInt16, Self.o]):
        self._slots = BufferView(slots)
        self._mask = UInt32(2 * (len(slots) - 1))

    @always_inline
    def _index(self, bytes: UInt32) -> Int:
        """libsnappy's two hashes. The CRC takes the mask as its second
        operand only because the mask is in a register anyway."""
        var offset: UInt32
        comptime if Self.crc32c:
            offset = Crc32c.step(bytes, self._mask)
        else:
            offset = (bytes * 0x1E35A7BD) >> UInt32(
                31 - _Compressor.MAX_TABLE_BITS
            )
        return Int(offset & self._mask) >> 1

    @always_inline
    def exchange(self, bytes: UInt32, pos: Int) -> Int:
        """The position last recorded for `bytes`, now replaced by `pos`."""
        var i = self._index(bytes)
        var previous = Int(self._slots.unsafe_get(i))
        self._slots.unsafe_set(i, UInt16(pos))
        return previous

    @always_inline
    def insert(self, bytes: UInt32, pos: Int):
        self._slots.unsafe_set(self._index(bytes), UInt16(pos))


struct _Emitter[o: MutOrigin](TrivialRegisterPassable):
    """The compressed output and the position written up to. A tag is
    written as one 4-byte store, so a write may run up to 3 bytes past that
    position -- inside the slack `Snappy.max_compressed_length` leaves."""

    var _dst: Span[UInt8, Self.o]
    var pos: Int

    def __init__(out self, dst: Span[UInt8, Self.o], pos: Int):
        self._dst = dst
        self.pos = pos

    @always_inline
    def literal(mut self, src: Span[UInt8, _]):
        """A literal tag and its bytes."""
        var n = len(src) - 1
        if n < 60:
            self._dst[self.pos] = UInt8(n << 2)
            self.pos += 1
        else:
            var count = (log2_floor(n) >> 3) + 1
            self._dst[self.pos] = UInt8((59 + count) << 2)
            LittleEndian.store(self._dst, self.pos + 1, UInt32(n))
            self.pos += 1 + count
        BufferView(self._dst).slice(self.pos).copy_from(
            BufferView(src), len(src)
        )
        self.pos += len(src)

    @always_inline
    def copy(mut self, offset: Int, length: Int, under_12: Bool):
        """A copy of `length` bytes from `offset` back; `under_12` is what the
        match finder already knows. Longer than 64 is split into 64s, keeping
        the last piece at least 4 long so it can still use copy-1."""
        if under_12:
            self._copy_at_most_64[True](offset, length)
        else:
            var left = length
            while left >= 68:
                self._copy_at_most_64[False](offset, 64)
                left -= 64
            if left > 64:
                self._copy_at_most_64[False](offset, 60)
                left -= 60
            if left < 12:
                self._copy_at_most_64[True](offset, left)
            else:
                self._copy_at_most_64[False](offset, left)

    @always_inline
    def _copy_at_most_64[
        len_less_than_12: Bool
    ](mut self, offset: Int, length: Int):
        """One copy tag, written as a 4-byte store. Branch-free on
        `offset < 2048`, which libsnappy measured as its top branch-miss
        source."""
        comptime if len_less_than_12:
            var u = UInt32((length << 2) + (offset << 8))
            var copy1 = (
                UInt32(1) - UInt32(4 << 2) + UInt32((offset >> 3) & 0xE0)
            )
            var copy2 = UInt32(2) - UInt32(1 << 2)
            var short = offset < 2048
            u += copy1 if short else copy2
            LittleEndian.store(self._dst, self.pos, u)
            self.pos += 2 if short else 3
        else:
            var u = UInt32(2 + ((length - 1) << 2) + (offset << 8))
            LittleEndian.store(self._dst, self.pos, u)
            self.pos += 3


struct _Fragment[m: Bool, //, o: Origin[mut=m]](TrivialRegisterPassable):
    """At most `_Compressor.BLOCK_SIZE` input bytes, compressed on their own
    -- libsnappy's `CompressFragment`."""

    comptime INPUT_MARGIN = 15
    """The main loop stops this far from the end, so the 8- and 16-byte loads
    it makes ahead of `ip` stay inside the fragment."""

    var _src: Span[UInt8, Self.o]

    def __init__(out self, src: Span[UInt8, Self.o]):
        self._src = src

    @always_inline
    def _load32(self, i: Int) -> UInt32:
        return LittleEndian.fixed[DType.uint32](self._src, i)

    @always_inline
    def _load64(self, i: Int) -> UInt64:
        return LittleEndian.fixed[DType.uint64](self._src, i)

    @always_inline
    def _mismatch(
        self, s2: Int, a1: UInt64, a2_in: UInt64, mut data: UInt64
    ) -> Int:
        """The matching byte count of two differing 8-byte words loaded at
        `s2`, and `data` set to the bytes from the first mismatch on --
        chosen between `a2` and a load at `s2 + 4` without waiting for the
        count."""
        var x = a1 ^ a2_in
        var shift = count_trailing_zeros(x)
        var a3 = self._load64(s2 + 4)
        var a2 = a3 if UInt32(x & 0xFFFFFFFF) == 0 else a2_in
        data = a2 >> (shift & 24)
        return Int(shift >> 3)

    @always_inline
    def _match_length(
        self, s1: Int, s2_start: Int, mut data: UInt64
    ) -> Tuple[Int, Bool]:
        """How far `[s1:]` matches `[s2_start:]`, and whether that is under 8
        bytes. On a mismatch inside the 8-byte compare, `data` gets the bytes
        from the new `ip` without a load that depends on the match length --
        libsnappy's trick for keeping the candidate load off the critical
        path."""
        var limit = len(self._src)
        var matched = 0
        var s2 = s2_start
        if s2 <= limit - 16:
            var a1 = self._load64(s1)
            var a2 = self._load64(s2)
            if a1 != a2:
                return (self._mismatch(s2, a1, a2, data), True)
            matched = 8
            s2 += 8
        while s2 <= limit - 16:
            var a1 = self._load64(s1 + matched)
            var a2 = self._load64(s2)
            if a1 == a2:
                s2 += 8
                matched += 8
            else:
                return (matched + self._mismatch(s2, a1, a2, data), False)
        while s2 < limit:
            if self._src[s1 + matched] == self._src[s2]:
                s2 += 1
                matched += 1
            else:
                if s2 <= limit - 8:
                    data = self._load64(s2)
                break
        return (matched, matched < 8)

    @always_inline
    def _probe16(
        self,
        ip: Int,
        preload: UInt32,
        mut data: UInt64,
        table: _Table[_, _],
        mut candidate: Int,
    ) -> Int:
        """Hash the next 16 positions from four 8-byte loads; return the first
        whose 4 bytes match their table candidate, or -1."""
        comptime for j in range(4):
            comptime for k in range(4):
                comptime i = 4 * j + k
                var dword: UInt32
                comptime if i == 0:
                    dword = preload
                else:
                    dword = UInt32(data & 0xFFFFFFFF)
                candidate = table.exchange(dword, ip + i)
                if self._load32(candidate) == dword:
                    return i
                data >>= 8
            data = self._load64(ip + 4 * j + 4)
        return -1

    def compress(self, mut out: _Emitter[_], table: _Table[_, _]):
        """Compress the fragment into `out` through `table`, cleared."""
        var e = out  # in registers: `out` is a reference the stores may alias
        var n = len(self._src)
        var ip = 0
        if n >= Self.INPUT_MARGIN:
            var ip_limit = n - Self.INPUT_MARGIN
            var preload = self._load32(1)
            while True:
                # Bytes in [next_emit, ip) become a literal.
                var next_emit = ip
                ip += 1
                var data = self._load64(ip)
                var skip = UInt32(32)
                var candidate = 0
                var hit = -1
                if ip_limit - ip >= 16:
                    hit = self._probe16(ip, preload, data, table, candidate)
                    if hit >= 0:
                        e.literal(self._src[next_emit : next_emit + hit + 1])
                        ip += hit
                    else:
                        ip += 16
                        skip += 16
                if hit < 0:
                    # Heuristic match skipping: after 32 misses look at every
                    # other byte, after 32 more every third, and so on.
                    var exhausted = False
                    while True:
                        var step = skip >> 5
                        skip += step
                        var next_ip = ip + Int(step)
                        if next_ip > ip_limit:
                            exhausted = True
                            break
                        var dword = UInt32(data & 0xFFFFFFFF)
                        candidate = table.exchange(dword, ip)
                        if dword == self._load32(candidate):
                            break
                        data = UInt64(self._load32(next_ip))
                        ip = next_ip
                    if exhausted:
                        ip = next_emit
                        break
                    e.literal(self._src[next_emit:ip])
                # Emit copies while the bytes right after each one match again.
                var remainder = False
                while True:
                    var start = ip
                    var m = self._match_length(candidate + 4, ip + 4, data)
                    ip += 4 + m[0]
                    e.copy(start - candidate, 4 + m[0], m[1])
                    if ip >= ip_limit:
                        remainder = True
                        break
                    table.insert(self._load32(ip - 1), ip - 1)
                    var dword = UInt32(data & 0xFFFFFFFF)
                    candidate = table.exchange(dword, ip)
                    if dword != self._load32(candidate):
                        break
                if remainder:
                    break
                preload = UInt32((data >> 8) & 0xFFFFFFFF)
        if ip < n:
            e.literal(self._src[ip:])
        out = e


# ---------------------------------------------------------------------------
# decompressor
# ---------------------------------------------------------------------------


struct _Decoder[sm: Bool, //, so: Origin[mut=sm], do: MutOrigin](
    TrivialRegisterPassable
):
    """One stream being decoded: the tags from input position `ip`, into the
    output up to position `op`."""

    comptime SLOP = 2 * LzCopy.SLOP
    """The most a fast round of two tags writes past `op`: each tag is at most
    64 bytes and is written as whole 32- or 16-byte blocks, so 64 per tag. The
    fast loop stops this far from the end of the output."""

    comptime TAGS = Self._tag_table()

    var _src: Span[UInt8, Self.so]
    var ip: Int
    var _dst: Span[UInt8, Self.do]
    var op: Int
    var _tag: Int
    """In the fast loop, the tag whose bytes follow `ip - 1`."""

    def __init__(
        out self, src: Span[UInt8, Self.so], ip: Int, dst: Span[UInt8, Self.do]
    ):
        self._src = src
        self.ip = ip
        self._dst = dst
        self.op = 0
        self._tag = 0

    @staticmethod
    def _tag_table() -> Array[Int16, 256]:
        """libsnappy's `kLengthMinusOffset`: per tag byte, `length -
        (offset_hi << 8)`. Subtracting the low offset byte(s) then gives
        `length - offset` in one step, whose sign is the overlap test.
        Literals get a spurious offset of 256 so they never test as
        overlapping, and the two tags the fast loop hands back -- a long
        literal and a copy-4 -- get 0xFF, which always does."""
        var t = Array[Int16, 256](fill=0)
        for tag in range(256):
            var data = tag >> 2
            var kind = tag & 3
            var v: Int
            if kind == 3:
                v = 0xFF
            elif kind == 2:
                v = data + 1
            elif kind == 1:
                v = (data & 7) + 4 - ((data >> 3) << 8)
            elif data < 60:
                v = data + 1 - (1 << 8)
            else:
                v = 0xFF
            t[tag] = Int16(v)
        return t^

    # --- the fast loop -----------------------------------------------------

    @always_inline
    def _fits(self, at: Int) -> Bool:
        """Whether a round of two fast tags fits from tag position `at`: a tag
        reads at most 65 bytes from its own position and the next starts at
        most 61 after it, and each writes at most `SLOP` / 2 bytes."""
        return (
            at < len(self._src) - 130 and self.op <= len(self._dst) - Self.SLOP
        )

    @always_inline
    def _fast_tag(mut self) -> Bool:
        """Decode `_tag`, unchecked in a release build; `False` (and nothing
        written) for a tag the fast loop will not take -- a long literal, a
        copy-4 or an invalid offset -- with `ip` just past it.

        **The input position never waits on the table.** The next tag is
        located from the tag byte alone -- `(tag >> 2) + 2` for a literal,
        `kind + 1` for a copy -- and loaded before this tag's copy is issued,
        so the loop-carried chain is one byte load and one add, not a load, a
        table load and an add."""
        var tag = self._tag
        var at = self.ip
        var entry = Int(global_constant[Self.TAGS]().unsafe_get(tag))
        var kind = tag & 3
        var step = (tag >> 2) + 1 if kind == 0 else kind
        self._tag = Int(BufferView(self._src).unsafe_get(at + step))
        self.ip = at + step + 1
        var next = Int(LittleEndian.fixed[DType.uint16](self._src, at))
        var length = entry & 0xFF
        var extracted = next & ((0x0000FFFF00FF0000 >> (kind * 16)) & 0xFFFF)
        var offset = extracted - entry + length
        if unlikely(entry > extracted):
            # A copy that overlaps its output -- or the 0xFF exceptions.
            if unlikely(length & 0x80 != 0 or offset == 0 or offset > self.op):
                self.ip = at
                return False
            LzCopy.pattern64(BufferView(self._dst), self.op, offset)
        else:
            # Literal or copy, one path: pick the source *address* -- the
            # input after the tag, or `offset` back in the output -- so an
            # irregular literal/copy sequence costs no branch. `offset > op`
            # is rarely true and so tested first: a literal's spurious offset
            # of 256 trips it only within 256 bytes of the start, and
            # branching on `kind` here would bring back the very
            # misprediction the select removes.
            if unlikely(offset > self.op):
                if kind != 0:
                    self.ip = at
                    return False
            # The one raw address in the module. A view of `dst` passed beside
            # `dst` itself is an aliasing error to Mojo's checker, even
            # rebound to the union of the two origins, so the source is picked
            # as an address and the bytes left after it; `_src`'s type is
            # only the load handle.
            var out = BufferView(self._dst)
            var literal = kind == 0
            var in_at = Int(self._src.unsafe_ptr()) + at
            var out_at = Int(self._dst.unsafe_ptr()) + self.op - offset
            var in_left = len(self._src) - at
            var out_left = len(self._dst) - self.op + offset
            var source = BufferView[DType.uint8, Self.so](
                ptr=Pointer[UInt8, Self.so](
                    unsafe_from_address=in_at if literal else out_at
                ),
                length=in_left if literal else out_left,
            )
            LzCopy.blocks64(source, out.slice(self.op), length)
        self.op += length
        return True

    @always_inline
    def _begin(mut self):
        """Into the fast loop: `ip` moves past the tag, which `_tag` holds."""
        self._tag = Int(BufferView(self._src).unsafe_get(self.ip))
        self.ip += 1

    @always_inline
    def _end(mut self):
        """Out of the fast loop: `ip` back on the tag to decode next."""
        self.ip -= 1

    @always_inline
    def _run_fast(mut self):
        """Decode tags two at a time with no per-tag bounds check --
        libsnappy's `DecompressBranchless` -- while they fit, stopping at the
        first tag it will not take."""
        if self._fits(self.ip):
            var s = self  # in registers: `self` is a reference stores may alias
            s._begin()
            while True:
                BufferView(s._src).prefetch_at(s.ip + 128)
                if not s._fast_tag():
                    break
                if not s._fast_tag():
                    break
                if not s._fits(s.ip - 1):
                    break
            s._end()
            self = s

    # --- the checked path --------------------------------------------------

    def _checked_tag(mut self) raises CorruptError:
        """Decode the one tag at `ip`, checking every bound and writing
        nothing past `n` -- what the fast loop hands back: a long literal, a
        copy-4, an error, or a tag too close to either end."""
        var src = self._src
        var src_len = len(src)
        var n = len(self._dst)
        var ip = self.ip
        var op = self.op
        var c = Int(src[ip])
        ip += 1
        var kind = c & 3
        if kind == 0:
            var length = (c >> 2) + 1
            if length > 60:
                var nb = length - 60
                if nb > src_len - ip:
                    raise CorruptError("snappy: truncated literal length")
                var field = src[ip : ip + nb]
                length = Int(LittleEndian.partial[DType.uint32](field, 0)) + 1
                ip += nb
            if length > src_len - ip:
                raise CorruptError(
                    t"snappy: {length}-byte literal at input {ip} runs past"
                    t" the {src_len}-byte input"
                )
            if length > n - op:
                raise CorruptError(
                    t"snappy: {length}-byte literal at output {op} overflows"
                    t" the declared {n} bytes"
                )
            BufferView(self._dst).slice(op).copy_from(
                BufferView(src).slice(ip), length
            )
            self.ip = ip + length
            self.op = op + length
        else:
            var trailer = 1 << (kind - 1)  # offset bytes: 1, 2 or 4
            if src_len - ip < trailer:
                raise CorruptError("snappy: truncated copy tag")
            var length: Int
            var offset: Int
            if kind == 1:
                length = ((c >> 2) & 7) + 4
                offset = ((c >> 5) << 8) | Int(src[ip])
            elif kind == 2:
                length = (c >> 2) + 1
                offset = Int(LittleEndian.fixed[DType.uint16](src, ip))
            else:
                length = (c >> 2) + 1
                offset = Int(LittleEndian.fixed[DType.uint32](src, ip))
            if offset == 0 or offset > op:
                raise CorruptError(
                    t"snappy: copy offset {offset} at output {op} is outside"
                    t" the output"
                )
            if length > n - op:
                raise CorruptError(
                    t"snappy: {length}-byte copy at output {op} overflows the"
                    t" declared {n} bytes"
                )
            LzCopy.copy_exact(self._dst, op, offset, length)
            self.ip = ip + trailer
            self.op = op + length

    # --- whole streams -----------------------------------------------------

    def run(mut self) raises CorruptError:
        """Decode the rest of the stream: the fast loop, the checked path for
        each tag it hands back, and a check that the output is complete."""
        while True:
            self._run_fast()
            if self.ip >= len(self._src):
                break
            self._checked_tag()
        if self.op != len(self._dst):
            raise CorruptError(
                t"snappy: stream ends after {self.op} of {len(self._dst)} bytes"
            )

    def interleave[
        sm2: Bool, //, so2: Origin[mut=sm2], do2: MutOrigin
    ](mut self, mut other: _Decoder[so2, do2]) raises CorruptError:
        """Decode this stream and `other` with their fast loops interleaved,
        tag for tag: each stream's tag chain is serial, so a second chain
        beside it fills the issue slots the first leaves idle. Each finishes
        alone once either leaves the shared loop for good."""
        while True:
            var stop_a = False
            var stop_b = False
            if self._fits(self.ip) and other._fits(other.ip):
                var a = self  # both in registers, as in `_run_fast`
                var b = other
                a._begin()
                b._begin()
                while True:
                    BufferView(a._src).prefetch_at(a.ip + 128)
                    BufferView(b._src).prefetch_at(b.ip + 128)
                    var ok_a = a._fast_tag()
                    var ok_b = b._fast_tag()
                    if ok_a and ok_b:
                        ok_a = a._fast_tag()
                        ok_b = b._fast_tag()
                    if not (ok_a and ok_b):
                        stop_a = not ok_a
                        stop_b = not ok_b
                        break
                    if not (a._fits(a.ip - 1) and b._fits(b.ip - 1)):
                        break
                a._end()
                b._end()
                self = a
                other = b
            if stop_a:
                self._checked_tag()
            if stop_b:
                other._checked_tag()
            if not (stop_a or stop_b):
                break
        self.run()
        other.run()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


struct Snappy:
    """Snappy raw-block compression (no framing format)."""

    @staticmethod
    def max_compressed_length(n: Int) -> Int:
        """The most bytes `compress` can produce for `n` input bytes: a 1-byte
        literal followed by a 5-byte copy turns 6 bytes into 7, plus 32 bytes
        the emitters may overwrite."""
        return 32 + n + n // 6

    @staticmethod
    def uncompressed_length(src: Span[UInt8, _]) raises CorruptError -> Int:
        """The length a stream declares, from its varint preamble."""
        return Self._preamble(src)[0]

    @staticmethod
    def _preamble(src: Span[UInt8, _]) raises CorruptError -> Tuple[Int, Int]:
        """`(n, tag_start)`: the declared length, a LEB128 varint held to
        libsnappy's varint32 -- at most 5 bytes and under 2^32."""
        var v: UInt64
        var end: Int
        try:
            v, end = LittleEndian.varint(src, 0)
        except e:
            raise CorruptError(t"snappy: length preamble: {e.message()}")
        if end > 5 or v > 0xFFFFFFFF:
            raise CorruptError("snappy: length preamble overflows 32 bits")
        return (Int(v), end)

    @staticmethod
    def compress[
        crc32c: Bool = False
    ](src: Span[UInt8, _], mut dst: List[UInt8]) raises InvalidError:
        """Append the compressed form of `src` to `dst`, hashing with
        libsnappy's multiply, or with CRC32C to match a libsnappy built with
        the instruction byte for byte."""
        var n = len(src)
        if n > 0xFFFFFFFF:
            raise InvalidError(
                t"snappy: {n} bytes is over the format's 4 GiB limit"
            )
        var start = len(dst)
        var end = start + Self.max_compressed_length(n)
        dst.reserve(end)
        LittleEndian.put_varint(dst, UInt64(n))
        var header = len(dst) - start
        dst.resize(unsafe_uninit_length=end)
        var out = _Emitter(Span(dst)[start:], header)
        if n > 0:
            var compressor = _Compressor(n)
            compressor.compress[crc32c](src, out)
        var used = out.pos
        dst.shrink(start + used)

    @staticmethod
    def decompress(
        src: Span[UInt8, _], mut dst: List[UInt8]
    ) raises CorruptError:
        """Append the decompressed form of `src` to `dst`."""
        var header = Self._preamble(src)
        var n = header[0]
        # A copy-2 tag expands 3 bytes to 64, the most any tag does; a length
        # past that is corrupt, and refusing it here saves allocating it.
        if n > 22 * (len(src) - header[1]):
            raise CorruptError(
                t"snappy: {len(src)} bytes cannot hold the declared {n}"
            )
        var start = len(dst)
        dst.resize(unsafe_uninit_length=start + n)
        var decoder = _Decoder(src, header[1], Span(dst)[start:])
        try:
            decoder.run()
        except e:
            dst.shrink(start)
            raise e^

    @staticmethod
    def decompress_into[
        o: Origin[mut=True]
    ](src: Span[UInt8, _], dst: Span[UInt8, o]) raises CorruptError:
        """Decompress `src` into exactly `dst`, whose length must be the
        declared one -- libsnappy's `snappy_uncompress` contract. Nothing past
        `len(dst)` is written; on a corrupt stream, what was decoded before
        the corruption is left in `dst`."""
        var header = Self._preamble(src)
        if header[0] != len(dst):
            raise CorruptError(
                t"snappy: stream declares {header[0]} bytes, destination"
                t" holds {len(dst)}"
            )
        var decoder = _Decoder(src, header[1], dst)
        decoder.run()

    @staticmethod
    def decompress_pair_into[
        oa: Origin[mut=True], ob: Origin[mut=True]
    ](
        src_a: Span[mut=False, UInt8, _],
        dst_a: Span[UInt8, oa],
        src_b: Span[mut=False, UInt8, _],
        dst_b: Span[UInt8, ob],
    ) raises CorruptError:
        """`decompress_into` for two independent streams at once -- two pages
        of a column chunk, say -- their tag loops interleaved so each fills
        the issue slots the other's serial tag chain leaves idle. The sources
        are immutable, so both may be slices of one buffer. On a corrupt
        stream, what either decoder wrote before the raise is left in its
        destination."""
        var ha = Self._preamble(src_a)
        var hb = Self._preamble(src_b)
        if ha[0] != len(dst_a) or hb[0] != len(dst_b):
            raise CorruptError(
                t"snappy: streams declare {ha[0]} and {hb[0]} bytes,"
                t" destinations hold {len(dst_a)} and {len(dst_b)}"
            )
        var a = _Decoder(src_a, ha[1], dst_a)
        var b = _Decoder(src_b, hb[1], dst_b)
        a.interleave(b)
