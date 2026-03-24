"""Shared hash table infrastructure for join and groupby kernels.

Provides:
  - ``HashTable`` trait — core interface for hash-based row indexing
  - ``SwissHashTable`` — flat open-addressing hash table (primary, fast)
  - ``SwissHashTable`` — SIMD group matching with pipelined probing
  - ``Partition`` / ``Partitioner`` / ``NoPartition`` — partitioning layer

Architecture:
  Hash Function  →  Partitioner  →  HashTable  →  Operator (join / groupby)
  Each layer is independently swappable.
"""

from std.bit import count_trailing_zeros, next_power_of_two
from std.memory import alloc, memset, pack_bits
from std.sys.intrinsics import prefetch, PrefetchOptions

from ..arrays import PrimitiveArray, StructArray
from ..builders import PrimitiveBuilder
from ..buffers import BufferBuilder
from ..dtypes import int32, uint64
from .hashing import rapidhash



# ---------------------------------------------------------------------------
# Partitioner — splits rows into partitions by hash
# ---------------------------------------------------------------------------


struct Partition(Movable):
    """A subset of rows with pre-computed hashes.

    ``row_indices = None`` means all rows in order (NoPartition fast-path,
    avoids allocating an identity index array).
    """

    var row_indices: Optional[PrimitiveArray[int32]]
    var hashes: PrimitiveArray[uint64]

    def __init__(
        out self,
        var hashes: PrimitiveArray[uint64],
        var row_indices: Optional[PrimitiveArray[int32]] = None,
    ):
        self.hashes = hashes^
        self.row_indices = row_indices^

    def num_rows(self) -> Int:
        return len(self.hashes)

    def original_row(self, i: Int) -> Int:
        """Map partition-local index → original row index."""
        if self.row_indices:
            return Int(self.row_indices.value().unsafe_get(i))
        return i


trait Partitioner(Movable):
    """Splits rows into partitions by hash prefix."""

    def num_partitions(self) -> Int:
        ...

    def partition(
        self, hashes: PrimitiveArray[uint64]
    ) raises -> List[Partition]:
        ...


struct NoPartition(Partitioner):
    """Single partition containing all rows (default, current behavior)."""

    def __init__(out self):
        pass

    def num_partitions(self) -> Int:
        return 1

    def partition(
        self, hashes: PrimitiveArray[uint64]
    ) raises -> List[Partition]:
        var result = List[Partition]()
        result.append(Partition(hashes^))
        return result^


# Future:
# struct RadixPartitioner(Partitioner):
#     """Partition by hash prefix bits. Enables partition-parallel joins
#     and better cache locality for large build sides.
#     Not yet implemented."""
#     var num_bits: Int


# ---------------------------------------------------------------------------
# HashTable trait — core interface for hash-based row indexing
# ---------------------------------------------------------------------------


trait HashTable(Movable):
    """Maps UInt64 hashes → sequential bucket IDs with row index storage.

    Shared by join and groupby:
      - Join uses ``build()`` + ``probe_pairs()``
      - GroupBy uses ``find_or_insert()`` + ``num_buckets()``
    """

    def hash_keys(
        self, keys: StructArray
    ) raises -> PrimitiveArray[uint64]:
        """Batch hash key columns using the table's hash function."""
        ...

    def build(mut self, hashes: PrimitiveArray[uint64]) raises:
        """Batch insert all rows from a pre-computed hash array."""
        ...

    def probe_pairs(
        self,
        hashes: PrimitiveArray[uint64],
        mut left_out: PrimitiveBuilder[int32],
        mut right_out: PrimitiveBuilder[int32],
        single_match: Bool = False,
    ) raises:
        """Find matching (build_row, probe_row) pairs with pipelined lookups.
        If single_match, emit at most one match per probe row (JOIN_ANY)."""
        ...

    def find(self, h: UInt64) -> Int:
        """Find bucket_id for hash. Return -1 if not found."""
        ...

    def find_or_insert(mut self, h: UInt64) -> Int:
        """Find or create bucket for hash. Return bucket_id (groupby)."""
        ...

    def num_buckets(self) -> Int:
        """Number of unique keys (buckets created so far)."""
        ...


# ---------------------------------------------------------------------------
# SwissHashTable — flat open-addressing hash table
# ---------------------------------------------------------------------------


comptime _GROUP_WIDTH: Int = 16
"""Number of control bytes per group (matches Mojo Dict / abseil)."""

comptime _CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""


@always_inline
def _h2(h: UInt64) -> UInt8:
    """Extract 7-bit H2 fingerprint from hash (top 7 bits)."""
    return UInt8(h >> 57)


struct _Group:
    """SIMD group of 16 control bytes for parallel matching.

    Loads 16 control bytes at once and uses SIMD comparison to produce
    bitmasks of matching slots. Same pattern as Mojo Dict / abseil / hashbrown.
    """

    var ctrl: SIMD[DType.uint8, _GROUP_WIDTH]

    @always_inline
    def __init__(out self, ptr: UnsafePointer[UInt8, _]):
        self.ctrl = ptr.load[width=_GROUP_WIDTH]()

    @always_inline
    def match_h2(self, h2: UInt8) -> UInt16:
        """Bitmask of slots matching the H2 fingerprint."""
        return pack_bits(
            self.ctrl.eq(SIMD[DType.uint8, _GROUP_WIDTH](h2))
        )

    @always_inline
    def match_empty(self) -> UInt16:
        """Bitmask of empty slots."""
        return pack_bits(
            self.ctrl.eq(SIMD[DType.uint8, _GROUP_WIDTH](_CTRL_EMPTY))
        )

# True pipelining would need to inline the find logic and interleave the individual memory loads across 4 probes — a much deeper rewrite.
struct SwissHashTable[
    hash_fn: def (StructArray) raises -> PrimitiveArray[uint64] = rapidhash
](HashTable):
    """Swiss Table hash table with SIMD group matching.

    Adopts Mojo Dict's design (16-slot groups, 7-bit H2 fingerprints,
    SIMD parallel matching via ``pack_bits``). Control bytes use Dict
    convention: 0xFF = EMPTY, 0x00-0x7F = H2 fingerprint.

    Separate arrays for ctrl, slots, and bucket_hashes: ctrl is contiguous
    for SIMD, slots (Int32 bucket_id) is compact, and bucket_hashes is a
    dense array indexed by bucket_id (N entries vs 2N slots).

    Row indices stored in flat chain arrays (no per-bucket heap alloc).
    """

    var _ctrl: UnsafePointer[UInt8, MutExternalOrigin]
    var _slots: UnsafePointer[Int32, MutExternalOrigin]   # bucket_id per slot
    var _capacity: Int
    var _mask: Int
    var _count: Int
    var _max_count: Int  # precomputed threshold: capacity * 7 / 8

    # Chain storage (flat arrays).
    var _heads: List[Int32]
    var _rows: List[Int32]
    var _next: List[Int32]
    var _bucket_hashes: List[UInt64]   # dense: bucket_id → hash

    def __init__(out self, capacity: Int = 0):
        var cap = Int(next_power_of_two(max(capacity * 2, _GROUP_WIDTH)))
        self._capacity = cap
        self._mask = cap - 1
        self._count = 0
        self._max_count = cap * 7 // 8
        self._ctrl = alloc[UInt8](cap + _GROUP_WIDTH)
        self._slots = alloc[Int32](cap)
        memset(self._ctrl, _CTRL_EMPTY, cap + _GROUP_WIDTH)
        self._heads = List[Int32](capacity=capacity)
        self._rows = List[Int32](capacity=capacity)
        self._next = List[Int32](capacity=capacity)
        self._bucket_hashes = List[UInt64](capacity=capacity)

    def __del__(deinit self):
        if self._capacity > 0:
            self._ctrl.free()
            self._slots.free()

    @always_inline
    def _set_ctrl(mut self, slot: Int, val: UInt8):
        """Set control byte and mirror into padding if slot < _GROUP_WIDTH."""
        self._ctrl[slot] = val
        if slot < _GROUP_WIDTH:
            self._ctrl[self._capacity + slot] = val

    def reserve(mut self, n: Int):
        """Pre-allocate slot and chain capacity for n expected entries."""
        var needed = Int(next_power_of_two(max(n * 2, _GROUP_WIDTH)))
        if needed > self._capacity:
            var old_ctrl = self._ctrl
            var old_slots = self._slots
            var old_cap = self._capacity
            self._capacity = needed
            self._mask = needed - 1
            self._max_count = needed * 7 // 8
            self._ctrl = alloc[UInt8](needed + _GROUP_WIDTH)
            self._slots = alloc[Int32](needed)
            memset(self._ctrl, _CTRL_EMPTY, needed + _GROUP_WIDTH)
            for i in range(old_cap):
                if old_ctrl[i] != _CTRL_EMPTY:
                    var bid = Int(old_slots[i])
                    var h = self._bucket_hashes[bid]
                    var h2 = _h2(h)
                    var pos = Int(h) & self._mask
                    while True:
                        var group = _Group(self._ctrl + pos)
                        var empty_mask = group.match_empty()
                        if empty_mask != 0:
                            var bit = count_trailing_zeros(Int(empty_mask))
                            var slot = (pos + bit) & self._mask
                            self._set_ctrl(slot, h2)
                            self._slots[slot] = Int32(bid)
                            break
                        pos = (pos + _GROUP_WIDTH) & self._mask
            old_ctrl.free()
            old_slots.free()
        self._heads.reserve(n)
        self._rows.reserve(n)
        self._next.reserve(n)
        self._bucket_hashes.reserve(n)

    def hash_keys(
        self, keys: StructArray
    ) raises -> PrimitiveArray[uint64]:
        return Self.hash_fn(keys)

    def build(mut self, hashes: PrimitiveArray[uint64]) raises:
        """Batch insert all rows. Reserves capacity upfront so the inner
        loop never needs to check load factor or grow.

        Optimized for the common case: mostly unique keys, so most
        inserts hit the empty-slot branch (no H2 match loop).
        """
        var n = len(hashes)
        self.reserve(n)
        var hash_ptr = hashes.buffer.unsafe_ptr[uint64.native](hashes.offset)

        # Track next bucket_id and entry_id locally to avoid repeated
        # len() calls on Lists in the hot loop.
        var next_bid = len(self._heads)
        var next_entry = len(self._rows)

        for i in range(n):
            var h = UInt64(hash_ptr[i])
            var h2 = _h2(h)
            var pos = Int(h) & self._mask

            # Prefetch next iteration's control bytes.
            if i + 8 < n:
                prefetch(self._ctrl +(Int(UInt64(hash_ptr[i + 8])) & self._mask))

            # Probe for existing bucket (duplicate key).
            var found = False
            while True:
                var group = _Group(self._ctrl + pos)
                var match_mask = group.match_h2(h2)
                while match_mask != 0:
                    var bit = count_trailing_zeros(Int(match_mask))
                    var slot_idx = (pos + bit) & self._mask
                    var bid = Int(self._slots[slot_idx])
                    if self._bucket_hashes[bid] == h:
                        self._rows.append(Int32(i))
                        self._next.append(self._heads[bid])
                        self._heads[bid] = Int32(next_entry)
                        next_entry += 1
                        found = True
                        break
                    match_mask &= match_mask - 1
                if found:
                    break

                var empty_mask = group.match_empty()
                if empty_mask != 0:
                    var bit = count_trailing_zeros(Int(empty_mask))
                    var slot = (pos + bit) & self._mask
                    self._set_ctrl(slot, h2)
                    self._slots[slot] = Int32(next_bid)
                    self._count += 1
                    self._bucket_hashes.append(h)
                    self._rows.append(Int32(i))
                    self._next.append(Int32(-1))
                    self._heads.append(Int32(next_entry))
                    next_bid += 1
                    next_entry += 1
                    break

                pos = (pos + _GROUP_WIDTH) & self._mask

    def insert(mut self, h: UInt64, row: Int32) -> Int:
        var h2 = _h2(h)
        var pos = Int(h) & self._mask

        while True:
            var group = _Group(self._ctrl + pos)
            var match_mask = group.match_h2(h2)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & self._mask
                var bid = Int(self._slots[slot_idx])
                if self._bucket_hashes[bid] == h:
                    var entry_idx = Int32(len(self._rows))
                    self._rows.append(row)
                    self._next.append(self._heads[bid])
                    self._heads[bid] = entry_idx
                    return bid
                match_mask &= match_mask - 1

            var empty_mask = group.match_empty()
            if empty_mask != 0:
                if self._count >= self._max_count:
                    self.reserve(self._capacity)
                    return self.insert(h, row)

                var bit = count_trailing_zeros(Int(empty_mask))
                var slot = (pos + bit) & self._mask
                var bid = len(self._heads)
                self._set_ctrl(slot, h2)
                self._slots[slot] = Int32(bid)
                self._bucket_hashes.append(h)
                self._count += 1

                var entry_idx = Int32(len(self._rows))
                self._rows.append(row)
                self._next.append(Int32(-1))
                self._heads.append(entry_idx)
                return bid

            pos = (pos + _GROUP_WIDTH) & self._mask

    def probe_pairs(
        self,
        hashes: PrimitiveArray[uint64],
        mut left_out: PrimitiveBuilder[int32],
        mut right_out: PrimitiveBuilder[int32],
        single_match: Bool = False,
    ) raises:
        """Pipelined probe: find matching (build_row, probe_row) pairs.

        Prefetches ahead to hide memory latency. Writes directly into
        the caller's pair builders.
        """
        var n = len(hashes)
        var hash_ptr = hashes.buffer.unsafe_ptr[uint64.native](hashes.offset)
        var ctrl = self._ctrl
        var mask = self._mask

        for probe_row in range(n):
            if probe_row + 8 < n:
                prefetch(ctrl + (Int(UInt64(hash_ptr[probe_row + 8])) & mask))

            var bid = self.find(UInt64(hash_ptr[probe_row]))
            if bid == -1:
                continue

            var entry = Int(self._heads[bid])
            while entry != -1:
                left_out.unsafe_append(Scalar[int32.native](self._rows[entry]))
                right_out.unsafe_append(Scalar[int32.native](probe_row))
                if single_match:
                    break
                entry = Int(self._next[entry])

    @always_inline
    def find(self, h: UInt64) -> Int:
        var h2 = _h2(h)
        var pos = Int(h) & self._mask
        while True:
            var group = _Group(self._ctrl + pos)
            var match_mask = group.match_h2(h2)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & self._mask
                var bid = Int(self._slots[slot_idx])
                if self._bucket_hashes[bid] == h:
                    return bid
                match_mask &= match_mask - 1
            if group.match_empty() != 0:
                return -1
            pos = (pos + _GROUP_WIDTH) & self._mask

    def find_or_insert(mut self, h: UInt64) -> Int:
        var h2 = _h2(h)
        var pos = Int(h) & self._mask

        while True:
            var group = _Group(self._ctrl + pos)
            var match_mask = group.match_h2(h2)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & self._mask
                var bid = Int(self._slots[slot_idx])
                if self._bucket_hashes[bid] == h:
                    return bid
                match_mask &= match_mask - 1

            var empty_mask = group.match_empty()
            if empty_mask != 0:
                if self._count >= self._max_count:
                    self.reserve(self._capacity)
                    return self.find_or_insert(h)

                var bit = count_trailing_zeros(Int(empty_mask))
                var slot = (pos + bit) & self._mask
                var bid = len(self._heads)
                self._set_ctrl(slot, h2)
                self._slots[slot] = Int32(bid)
                self._bucket_hashes.append(h)
                self._count += 1
                self._heads.append(Int32(-1))
                return bid

            pos = (pos + _GROUP_WIDTH) & self._mask

    def num_buckets(self) -> Int:
        return len(self._heads)
