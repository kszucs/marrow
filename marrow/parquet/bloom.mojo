"""Parquet split-block bloom filter (SBBF) with the XXH64 value hash.

The layout follows the `parquet.thrift` `BloomFilterHeader` / SBBF spec and the
reference implementations in arrow-rs (`bloom_filter.rs`) and Arrow C++:

- `xxh64` — the 64-bit xxHash (seed 0) the filter hashes every value with.
- `SplitBlockBloomFilter` — a sequence of 256-bit blocks; each value touches one
  block (chosen from the hash's high 32 bits) and sets one bit in each of the
  block's eight 32-bit words (chosen by the eight SALT multipliers). `insert` /
  `might_contain` never yield false negatives, so a present value always tests
  positive — the invariant the round-trip and duckdb-oracle tests rely on.
- `BloomFilterHeader` — the small Thrift struct (numBytes + the SPLIT_BLOCK /
  XXHASH / UNCOMPRESSED union members) written just before the bitset and pointed
  to by `ColumnMetaData.bloom_filter_offset`.
"""

from std.math import log

from ..utils import LittleEndian
from .format import (
    ThriftCompactWriter,
    ThriftCompactReader,
    FieldHeader,
    ThriftWritable,
    TC_I32,
    TC_STRUCT,
)


# ---------------------------------------------------------------------------
# XXH64 — the 64-bit xxHash (seed 0), the hash the Parquet bloom filter uses.
# ---------------------------------------------------------------------------

comptime _P1: UInt64 = 0x9E3779B185EBCA87
comptime _P2: UInt64 = 0xC2B2AE3D27D4EB4F
comptime _P3: UInt64 = 0x165667B19E3779F9
comptime _P4: UInt64 = 0x85EBCA77C2B2AE63
comptime _P5: UInt64 = 0x27D4EB2F165667C5


@always_inline
def _rotl(x: UInt64, r: Int) -> UInt64:
    return (x << UInt64(r)) | (x >> UInt64(64 - r))


@always_inline
def _round(acc: UInt64, input: UInt64) -> UInt64:
    var a = acc + input * _P2
    a = _rotl(a, 31)
    return a * _P1


@always_inline
def _merge_round(acc: UInt64, val: UInt64) -> UInt64:
    var v = _round(0, val)
    var a = acc ^ v
    return a * _P1 + _P4


def xxh64(data: Span[UInt8, _], seed: UInt64 = 0) -> UInt64:
    """The 64-bit xxHash of `data` (wrapping arithmetic throughout), matching the
    canonical XXH64 — the value hash for the Parquet split-block bloom filter.
    """
    var n = len(data)
    var h: UInt64
    var p = 0
    if n >= 32:
        var v1 = seed + _P1 + _P2
        var v2 = seed + _P2
        var v3 = seed + 0
        var v4 = seed - _P1
        var limit = n - 32
        while p <= limit:
            v1 = _round(v1, LittleEndian.fixed[DType.uint64](data, p))
            p += 8
            v2 = _round(v2, LittleEndian.fixed[DType.uint64](data, p))
            p += 8
            v3 = _round(v3, LittleEndian.fixed[DType.uint64](data, p))
            p += 8
            v4 = _round(v4, LittleEndian.fixed[DType.uint64](data, p))
            p += 8
        h = _rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)
        h = _merge_round(h, v1)
        h = _merge_round(h, v2)
        h = _merge_round(h, v3)
        h = _merge_round(h, v4)
    else:
        h = seed + _P5

    h += UInt64(n)

    while p + 8 <= n:
        var k1 = _round(0, LittleEndian.fixed[DType.uint64](data, p))
        h ^= k1
        h = _rotl(h, 27) * _P1 + _P4
        p += 8
    if p + 4 <= n:
        h ^= UInt64(LittleEndian.fixed[DType.uint32](data, p)) * _P1
        h = _rotl(h, 23) * _P2 + _P3
        p += 4
    while p < n:
        h ^= UInt64(data[p]) * _P5
        h = _rotl(h, 11) * _P1
        p += 1

    h ^= h >> 33
    h *= _P2
    h ^= h >> 29
    h *= _P3
    h ^= h >> 32
    return h


# ---------------------------------------------------------------------------
# Split-block bloom filter
# ---------------------------------------------------------------------------

comptime _SALT = InlineArray[UInt32, 8](
    UInt32(0x47B6137B),
    UInt32(0x44974D91),
    UInt32(0x8824AD5B),
    UInt32(0xA2B7289D),
    UInt32(0x705495C7),
    UInt32(0x2DF1424B),
    UInt32(0x9EFC4947),
    UInt32(0x5C6BFB31),
    __list_literal__=None,
)
comptime _BITS_PER_BLOCK = 256
comptime _BYTES_PER_BLOCK = 32
comptime _WORDS_PER_BLOCK = 8


@always_inline
def _block_mask(x: UInt32, mut out: InlineArray[UInt32, 8]):
    """The eight per-word bit masks a value sets/tests within one block."""
    comptime for i in range(8):
        var y = x * _SALT[i]  # wrapping 32-bit multiply
        out[i] = UInt32(1) << (y >> 27)  # top 5 bits -> a bit in [0, 32)


struct SplitBlockBloomFilter(Movable):
    """A split-block bloom filter: `num_blocks` 256-bit blocks stored as a flat
    `UInt32` word array (8 words per block). Hashes are XXH64; the block is
    chosen from the hash's high 32 bits and the in-block bits from the low 32.
    """

    var words: List[UInt32]  # num_blocks * 8 words
    var num_blocks: Int

    def __init__(out self, num_blocks: Int):
        self.num_blocks = num_blocks
        self.words = List[UInt32](length=num_blocks * _WORDS_PER_BLOCK, fill=0)

    @staticmethod
    def _num_bits(ndv: Int, fpp: Float64) -> Int:
        """The parquet-spec optimal bit count for `ndv` distinct values at false-
        positive probability `fpp` (arrow-rs `num_of_bits_from_ndv_fpp`)."""
        var m = -8.0 * Float64(ndv) / log(1.0 - fpp ** (1.0 / 8.0))
        return Int(m)

    @staticmethod
    def with_ndv(ndv: Int, fpp: Float64 = 0.01) -> Self:
        """Size a filter for `ndv` distinct values: round the optimal bit count up
        to a power-of-two number of 256-bit blocks (clamped to at least one)."""
        var bits = Self._num_bits(max(ndv, 1), fpp)
        var blocks = (bits + _BITS_PER_BLOCK - 1) // _BITS_PER_BLOCK
        # round up to a power of two (the conventional SBBF byte size)
        var nb = 1
        while nb < blocks:
            nb <<= 1
        return Self(nb)

    @staticmethod
    def from_bytes(data: Span[UInt8, _]) raises -> Self:
        """Load a filter from its serialized bitset (`num_bytes` = a multiple of
        32); each block's eight words are little-endian `UInt32`s."""
        var nbytes = len(data)
        if nbytes % _BYTES_PER_BLOCK != 0:
            raise Error("parquet: bloom filter size not a block multiple")
        var nb = nbytes // _BYTES_PER_BLOCK
        var out = Self(nb)
        for w in range(nb * _WORDS_PER_BLOCK):
            out.words[w] = LittleEndian.fixed[DType.uint32](data, w * 4)
        return out^

    def to_bytes(self) -> List[UInt8]:
        """Serialize the bitset: each word little-endian, blocks contiguous."""
        var out = List[UInt8](capacity=len(self.words) * 4)
        for w in self.words:
            LittleEndian.put_le(out, UInt64(w), 4)
        return out^

    @always_inline
    def _block_of(self, hash: UInt64) -> Int:
        # the block index from the hash's high 32 bits (multiply-shift)
        return Int(((hash >> 32) * UInt64(self.num_blocks)) >> 32)

    def insert_hash(mut self, hash: UInt64):
        var bi = self._block_of(hash) * _WORDS_PER_BLOCK
        var m = InlineArray[UInt32, 8](fill=0)
        _block_mask(UInt32(hash & 0xFFFFFFFF), m)

        comptime for i in range(8):
            self.words[bi + i] |= m[i]

    def contains_hash(self, hash: UInt64) -> Bool:
        var bi = self._block_of(hash) * _WORDS_PER_BLOCK
        var m = InlineArray[UInt32, 8](fill=0)
        _block_mask(UInt32(hash & 0xFFFFFFFF), m)

        comptime for i in range(8):
            if (self.words[bi + i] & m[i]) == 0:
                return False
        return True

    def insert(mut self, value: Span[UInt8, _]):
        self.insert_hash(xxh64(value))

    def might_contain(self, value: Span[UInt8, _]) -> Bool:
        """Whether `value` may be present — no false negatives, so a False result
        proves absence (the pushdown guarantee)."""
        return self.contains_hash(xxh64(value))


# ---------------------------------------------------------------------------
# BloomFilterHeader — the Thrift struct preceding the bitset in the file.
# ---------------------------------------------------------------------------


struct BloomFilterHeader(Copyable, Movable, ThriftWritable):
    """The `BloomFilterHeader`: `num_bytes` plus the algorithm/hash/compression
    union members. Marrow only writes and expects SPLIT_BLOCK + XXHASH +
    UNCOMPRESSED, so the unions are fixed on write and skipped on read."""

    var num_bytes: Int

    def __init__(out self):
        self.num_bytes = 0

    @staticmethod
    def read[
        o: Origin[mut=False]
    ](mut r: ThriftCompactReader[o]) raises -> Self:
        var out = Self()
        var f = FieldHeader()
        while r.next_field(f):
            if f.id == 1:
                out.num_bytes = Int(r.read_i32())
            else:
                r.skip(f.type)
        return out^

    def write(self, mut w: ThriftCompactWriter):
        var last = 0
        last = w.write_field_begin(TC_I32, 1, last)
        w.write_i32(Int32(self.num_bytes))
        # algorithm (2): BlockBasedAlgorithm (union member 1, empty struct)
        last = w.write_field_begin(TC_STRUCT, 2, last)
        _ = w.write_field_begin(TC_STRUCT, 1, 0)
        w.write_field_stop()  # empty BlockBasedAlgorithm
        w.write_field_stop()  # close the algorithm union
        # hash (3): XxHash (union member 1, empty struct)
        last = w.write_field_begin(TC_STRUCT, 3, last)
        _ = w.write_field_begin(TC_STRUCT, 1, 0)
        w.write_field_stop()  # empty XxHash
        w.write_field_stop()  # close the hash union
        # compression (4): Uncompressed (union member 1, empty struct)
        last = w.write_field_begin(TC_STRUCT, 4, last)
        _ = w.write_field_begin(TC_STRUCT, 1, 0)
        w.write_field_stop()  # empty Uncompressed
        w.write_field_stop()  # close the compression union
        w.write_field_stop()  # close the header
