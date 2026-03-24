"""Shared hash table infrastructure for join and groupby kernels.

Provides:
  - ``HashTable`` trait — core interface for hash-based row indexing
  - ``SwissHashTable`` — flat open-addressing hash table (primary, fast)
  - ``DictHashTable`` — Dict-backed implementation (fallback)
  - ``Partition`` / ``Partitioner`` / ``NoPartition`` — partitioning layer

Architecture:
  Hash Function  →  Partitioner  →  HashTable  →  Operator (join / groupby)
  Each layer is independently swappable.
"""

from std.bit import count_trailing_zeros, next_power_of_two
from std.hashlib import Hasher
from std.memory import Span, alloc, memset, pack_bits

from ..arrays import PrimitiveArray, StructArray
from ..builders import PrimitiveBuilder
from ..dtypes import int32, uint64
from .hashing import rapidhash


# ---------------------------------------------------------------------------
# IdentityHasher — avoids double-hashing pre-computed UInt64 keys
# ---------------------------------------------------------------------------


struct IdentityHasher(Hasher):
    """Hasher that returns the input UInt64 unchanged.

    Used with ``Dict[UInt64, V, IdentityHasher]`` to avoid re-hashing
    keys that are already high-quality 64-bit hashes produced by our
    column-wise hash functions (``hash_``, ``hash_identity``).
    """

    var _value: UInt64

    def __init__(out self):
        self._value = 0

    def _update_with_bytes(mut self, data: Span[Byte, _]):
        pass

    def _update_with_simd(mut self, value: SIMD[_, _]):
        self._value = UInt64(value[0])

    def update[T: Hashable](mut self, value: T):
        value.__hash__(self)

    def finish(var self) -> UInt64:
        return self._value


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

    Each unique hash gets a sequential bucket ID (0, 1, 2, ...).
    Buckets are iterated via chain pointers (head → next → next → -1).

    Shared by join and groupby:
      - Join uses ``insert()`` to build, chain iteration to probe.
      - GroupBy uses ``find_or_insert()`` for group_id assignment
        (bucket_id IS the group_id).
    """

    def hash_keys(
        self, keys: StructArray
    ) raises -> PrimitiveArray[uint64]:
        """Batch hash key columns using the table's hash function."""
        ...

    def reserve(mut self, n: Int):
        """Pre-allocate for n expected entries to avoid reallocs during build."""
        ...

    def build(mut self, hashes: PrimitiveArray[uint64]) raises:
        """Batch insert all rows from a pre-computed hash array."""
        ...

    def insert(mut self, h: UInt64, row: Int32) -> Int:
        """Append row to bucket for hash h. Create bucket if new.
        Return bucket_id."""
        ...

    def find(self, h: UInt64) -> Int:
        """Find bucket_id for hash. Return -1 if not found."""
        ...

    def find_or_insert(mut self, h: UInt64) -> Int:
        """Find or create bucket for hash. Return bucket_id.
        Does NOT store a row index. Used by groupby: bucket_id IS group_id.
        """
        ...

    def bucket_head(self, bid: Int) -> Int32:
        """First entry index in bucket, -1 if empty."""
        ...

    def entry_row(self, entry: Int) -> Int32:
        """Row index stored at this entry."""
        ...

    def entry_next(self, entry: Int) -> Int32:
        """Next entry in chain, -1 if end of bucket."""
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


struct SwissHashTable[
    hash_fn: def (StructArray) raises -> PrimitiveArray[uint64] = rapidhash
](HashTable):
    """Swiss Table hash table with SIMD group matching.

    Adopts Mojo Dict's design (16-slot groups, 7-bit H2 fingerprints,
    SIMD parallel matching via ``pack_bits``). Control bytes use Dict
    convention: 0xFF = EMPTY, 0x00-0x7F = H2 fingerprint.

    Capacity is always ≥ ``_GROUP_WIDTH`` and a power of 2. The control
    array is padded by ``_GROUP_WIDTH`` bytes to handle wrap-around
    without branching (first group mirrored at the end).

    Row indices stored in flat chain arrays (no per-bucket heap alloc).
    """

    alias _LOAD_FACTOR_NUM: Int = 7
    alias _LOAD_FACTOR_DEN: Int = 8

    var _ctrl: UnsafePointer[UInt8, MutExternalOrigin]
    var _slots: UnsafePointer[Int32, MutExternalOrigin]
    var _capacity: Int
    var _mask: Int
    var _count: Int

    # Chain storage (flat arrays).
    var _heads: List[Int32]
    var _rows: List[Int32]
    var _next: List[Int32]
    var _bucket_hashes: List[UInt64]

    def __init__(out self, capacity: Int = 0):
        var cap = Int(next_power_of_two(max(capacity * 2, _GROUP_WIDTH)))
        self._capacity = cap
        self._mask = cap - 1
        self._count = 0
        # Allocate ctrl + _GROUP_WIDTH padding for wrap-around mirroring.
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
            # Grow slot array.
            var old_ctrl = self._ctrl
            var old_slots = self._slots
            var old_cap = self._capacity
            self._capacity = needed
            self._mask = needed - 1
            self._ctrl = alloc[UInt8](needed + _GROUP_WIDTH)
            self._slots = alloc[Int32](needed)
            memset(self._ctrl, _CTRL_EMPTY, needed + _GROUP_WIDTH)
            # Re-insert existing entries.
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
        # Pre-allocate chain arrays.
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
        loop never needs to check load factor or grow."""
        var n = len(hashes)
        self.reserve(n)
        for i in range(n):
            var h = UInt64(hashes.unsafe_get(i))
            var h2 = _h2(h)
            var pos = Int(h) & self._mask

            # Probe for existing bucket.
            var found = False
            while True:
                var group = _Group(self._ctrl + pos)
                var match_mask = group.match_h2(h2)
                while match_mask != 0:
                    var bit = count_trailing_zeros(Int(match_mask))
                    var slot_idx = (pos + bit) & self._mask
                    var bid = Int(self._slots[slot_idx])
                    if self._bucket_hashes[bid] == h:
                        var entry_idx = Int32(len(self._rows))
                        self._rows.append(Int32(i))
                        self._next.append(self._heads[bid])
                        self._heads[bid] = entry_idx
                        found = True
                        break
                    match_mask &= match_mask - 1
                if found:
                    break

                var empty_mask = group.match_empty()
                if empty_mask != 0:
                    # New bucket — no load factor check (pre-reserved).
                    var bit = count_trailing_zeros(Int(empty_mask))
                    var slot = (pos + bit) & self._mask
                    var bid = len(self._heads)
                    self._set_ctrl(slot, h2)
                    self._slots[slot] = Int32(bid)
                    self._count += 1
                    self._bucket_hashes.append(h)
                    var entry_idx = Int32(len(self._rows))
                    self._rows.append(Int32(i))
                    self._next.append(Int32(-1))
                    self._heads.append(entry_idx)
                    break

                pos = (pos + _GROUP_WIDTH) & self._mask

    def insert(mut self, h: UInt64, row: Int32) -> Int:
        var h2 = _h2(h)
        var pos = Int(h) & self._mask

        # Probe for existing bucket with matching hash.
        while True:
            var group = _Group(self._ctrl + pos)
            var match_mask = group.match_h2(h2)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & self._mask
                var bid = Int(self._slots[slot_idx])
                if self._bucket_hashes[bid] == h:
                    # Existing bucket — append to chain.
                    var entry_idx = Int32(len(self._rows))
                    self._rows.append(row)
                    self._next.append(self._heads[bid])
                    self._heads[bid] = entry_idx
                    return bid
                match_mask &= match_mask - 1

            # If any empty slot, hash is absent — insert here.
            var empty_mask = group.match_empty()
            if empty_mask != 0:
                # Check load factor before inserting.
                if self._count * Self._LOAD_FACTOR_DEN >= self._capacity * Self._LOAD_FACTOR_NUM:
                    self.reserve(self._capacity)
                    return self.insert(h, row)

                var bit = count_trailing_zeros(Int(empty_mask))
                var slot = (pos + bit) & self._mask
                var bid = len(self._heads)
                self._set_ctrl(slot, h2)
                self._slots[slot] = Int32(bid)
                self._count += 1
                self._bucket_hashes.append(h)

                var entry_idx = Int32(len(self._rows))
                self._rows.append(row)
                self._next.append(Int32(-1))
                self._heads.append(entry_idx)
                return bid

            pos = (pos + _GROUP_WIDTH) & self._mask

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
                if self._count * Self._LOAD_FACTOR_DEN >= self._capacity * Self._LOAD_FACTOR_NUM:
                    self.reserve(self._capacity)
                    return self.find_or_insert(h)

                var bit = count_trailing_zeros(Int(empty_mask))
                var slot = (pos + bit) & self._mask
                var bid = len(self._heads)
                self._set_ctrl(slot, h2)
                self._slots[slot] = Int32(bid)
                self._count += 1
                self._bucket_hashes.append(h)
                self._heads.append(Int32(-1))
                return bid

            pos = (pos + _GROUP_WIDTH) & self._mask

    def bucket_head(self, bid: Int) -> Int32:
        return self._heads[bid]

    def entry_row(self, entry: Int) -> Int32:
        return self._rows[entry]

    def entry_next(self, entry: Int) -> Int32:
        return self._next[entry]

    def num_buckets(self) -> Int:
        return len(self._heads)


# ---------------------------------------------------------------------------
# DictHashTable — Dict-backed HashTable implementation (fallback)
# ---------------------------------------------------------------------------


struct DictHashTable[
    hash_fn: def (StructArray) raises -> PrimitiveArray[uint64] = rapidhash
](HashTable):
    """Dict-backed hash table with flat chain-pointer storage.

    Fallback implementation using Mojo's stdlib Dict. Slower than
    SwissHashTable due to Dict API overhead, but simpler.
    """

    var _map: Dict[UInt64, Int, IdentityHasher]
    var _heads: List[Int32]
    var _rows: List[Int32]
    var _next: List[Int32]

    def __init__(out self, capacity: Int = 0):
        self._map = Dict[UInt64, Int, IdentityHasher]()
        self._heads = List[Int32](capacity=capacity)
        self._rows = List[Int32](capacity=capacity)
        self._next = List[Int32](capacity=capacity)

    def reserve(mut self, n: Int):
        self._heads.reserve(n)
        self._rows.reserve(n)
        self._next.reserve(n)

    def build(mut self, hashes: PrimitiveArray[uint64]) raises:
        var n = len(hashes)
        self.reserve(n)
        for i in range(n):
            _ = self.insert(UInt64(hashes.unsafe_get(i)), Int32(i))

    def hash_keys(
        self, keys: StructArray
    ) raises -> PrimitiveArray[uint64]:
        return Self.hash_fn(keys)

    def insert(mut self, h: UInt64, row: Int32) -> Int:
        var entry_idx = Int32(len(self._rows))
        self._rows.append(row)

        var existing = self._map.get(h)
        if existing:
            var bid = existing.value()
            self._next.append(self._heads[bid])
            self._heads[bid] = entry_idx
            return bid

        var bid = len(self._heads)
        self._map[h] = bid
        self._next.append(Int32(-1))
        self._heads.append(entry_idx)
        return bid

    def find(self, h: UInt64) -> Int:
        var existing = self._map.get(h)
        if existing:
            return existing.value()
        return -1

    def find_or_insert(mut self, h: UInt64) -> Int:
        var existing = self._map.get(h)
        if existing:
            return existing.value()
        var bid = len(self._heads)
        self._map[h] = bid
        self._heads.append(Int32(-1))
        return bid

    def find_batch(
        self, hashes: PrimitiveArray[uint64]
    ) raises -> PrimitiveArray[int32]:
        """Batch find: return bucket_id per hash (-1 if not found)."""
        var n = len(hashes)
        var builder = PrimitiveBuilder[int32](capacity=n)
        for i in range(n):
            var h = UInt64(hashes.unsafe_get(i))
            var existing = self._map.get(h)
            if existing:
                builder.append(Scalar[int32.native](existing.value()))
            else:
                builder.append(Scalar[int32.native](-1))
        return builder.finish()

    def bucket_head(self, bid: Int) -> Int32:
        return self._heads[bid]

    def entry_row(self, entry: Int) -> Int32:
        return self._rows[entry]

    def entry_next(self, entry: Int) -> Int32:
        return self._next[entry]

    def num_buckets(self) -> Int:
        return len(self._heads)
