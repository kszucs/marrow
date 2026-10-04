# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The bitstreams of Zstandard's entropy coders.

Every Huffman and FSE stream in a zstd block is written forwards and read
backwards: the writer puts each field above the last, low bits first, then a
single `1` bit, and pads the byte with zeros; the reader starts below that end
mark and takes fields from the top down. A stream is valid only if the reader
ends exactly at its first bit.
"""

from std.bit import count_leading_zeros

from ...errors import CorruptError
from ..byteorder import LittleEndian


struct BitReader[m: Bool, //, o: Origin[mut=m]](TrivialRegisterPassable):
    """A zstd bitstream, read from its end mark back to its first bit.

    It holds the 8 bytes at `_ptr` and counts in `_used` the bits already
    taken from their top -- libzstd's `BIT_DStream_t` -- and keeps the rest
    left-aligned in `_bits`, so a peek is one shift and taking bits another.
    `refill` steps `_ptr` back by the whole bytes used, so after it at least
    57 bits are ready until the stream's start is within 8 bytes; from then
    on fewer are, and reading past the start reads zeros and leaves `_used`
    above 64, which `finished` reports as a stream not consumed exactly.
    """

    var _src: Span[UInt8, Self.o]
    var _ptr: Int
    var _bits: UInt64
    var _used: Int

    def __init__(out self, src: Span[UInt8, Self.o]) raises CorruptError:
        var n = len(src)
        if n == 0:
            raise CorruptError("zstd: an empty bitstream")
        var last = src[n - 1]
        if last == 0:
            raise CorruptError("zstd: a bitstream without its end mark")
        # The padding zeros and the end mark itself.
        var mark = Int(count_leading_zeros(last)) + 1
        self._src = src
        if n >= 8:
            self._ptr = n - 8
            self._used = mark
            self._bits = LittleEndian.fixed[DType.uint64](src, n - 8) << UInt64(
                mark
            )
        else:
            # The missing high bytes read as zero and count as used.
            self._ptr = 0
            self._used = mark + 8 * (8 - n)
            var word = LittleEndian.partial[DType.uint64](src, 0)
            self._bits = word << UInt64(self._used) if self._used < 64 else 0

    @always_inline
    def peek(self, n: Int) -> Int:
        """The next `n` bits (0..57 after a `refill`), not taken. Bits before
        the stream's start read as zero."""
        return Int((self._bits >> 1) >> UInt64(63 - n))

    @always_inline
    def peek_nonzero(self, n: Int) -> Int:
        """`peek` for `n >= 1`, one shift shorter."""
        return Int(self._bits >> UInt64(64 - n))

    @always_inline
    def skip(mut self, n: Int):
        """Take `n` (at most 57) bits."""
        self._bits <<= UInt64(n)
        self._used += n

    @always_inline
    def read(mut self, n: Int) -> Int:
        var v = self.peek(n)
        self.skip(n)
        return v

    @always_inline
    def refill(mut self):
        """Step back over the bytes used, so 57 bits are ready again -- or,
        within 8 bytes of the start, as many as are left."""
        if self._ptr > 0:
            var step = min(self._used >> 3, self._ptr)
            self._ptr -= step
            self._used -= 8 * step
            var word = LittleEndian.fixed[DType.uint64](self._src, self._ptr)
            self._bits = word << UInt64(self._used) if self._used < 64 else 0

    @always_inline
    def full(self) -> Bool:
        """Whether a `refill` now leaves at least 57 bits ready."""
        return self._ptr >= 8

    def remaining(self) -> Int:
        """The bits not yet read; negative once reads ran past the start."""
        return 8 * self._ptr + 64 - self._used

    def finished(self) -> Bool:
        """Whether exactly every bit of the stream has been read."""
        return self.remaining() == 0


struct BitWriter[o: MutOrigin](Movable):
    """A zstd bitstream written in place: each field above the last, low
    bits first, and on `close` the end mark `BitReader` starts below --
    libzstd's `BIT_CStream_t`. At most 63 bits may be pending between
    `flush`es.

    The destination is the caller's, sized up front to its bound on the
    stream plus the 8 bytes a `flush` stores: a `flush` that could grow it
    would call out of the loop, and that alone keeps the compiler from
    holding a writer in registers."""

    var _dst: Span[UInt8, Self.o]
    var _pos: Int
    """Where the next whole byte goes."""
    var _bits: UInt64
    var _n: Int
    """The bits pending in `_bits`."""

    def __init__(out self, dst: Span[UInt8, Self.o], at: Int):
        """For a stream starting at `dst[at]`."""
        self._dst = dst
        self._pos = at
        self._bits = 0
        self._n = 0

    @always_inline
    def add(mut self, value: Int, n: Int):
        """Append the low `n` bits of `value`."""
        var mask = (UInt64(1) << UInt64(n)) - 1
        self._bits |= (UInt64(value) & mask) << UInt64(self._n)
        self._n += n

    @always_inline
    def add_fitting(mut self, value: Int, n: Int):
        """`add` for a `value` already under `1 << n`."""
        self._bits |= UInt64(value) << UInt64(self._n)
        self._n += n

    @always_inline
    def flush(mut self):
        """Write out the whole bytes pending."""
        debug_assert(
            self._pos + 8 <= len(self._dst), "BitWriter: over its capacity"
        )
        LittleEndian.store[DType.uint64](self._dst, self._pos, self._bits)
        var whole = self._n >> 3
        self._pos += whole
        self._bits >>= UInt64(8 * whole)
        self._n &= 7

    def close(mut self) -> Int:
        """Add the end mark; return where the stream ends."""
        self.add(1, 1)
        self.flush()
        return self._pos + Int(self._n > 0)
