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

from std.bit import next_power_of_two
from std.hashlib import Hasher
from std.memory import Span, alloc, memset

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


struct SwissHashTable[
    hash_fn: def (StructArray) raises -> PrimitiveArray[uint64] = rapidhash
](HashTable):
    """Flat open-addressing hash table with 7-bit stamps.

    Same design as Arrow's SwissTable / Rust's hashbrown:
    - Flat slot array with open addressing and linear probing
    - 7-bit stamp per slot for fast rejection (avoid full key compare)
    - Slot stores bucket_id; chain pointers stored separately

    Memory layout (per slot):
      - ``_ctrl[slot]``: UInt8 control byte (0x80 = empty, low 7 bits = stamp)
      - ``_slots[slot]``: Int32 bucket_id

    Row indices stored in separate flat chain arrays (same as DictHashTable).
    """

    alias EMPTY: UInt8 = 0x80
    alias _LOAD_FACTOR_NUM: Int = 7
    alias _LOAD_FACTOR_DEN: Int = 8

    var _ctrl: UnsafePointer[UInt8, MutExternalOrigin]    # control bytes
    var _slots: UnsafePointer[Int32, MutExternalOrigin]  # bucket_id per slot
    var _capacity: Int                 # number of slots (power of 2)
    var _mask: Int                     # _capacity - 1
    var _count: Int                    # number of occupied slots

    # Chain storage (flat arrays).
    var _heads: List[Int32]            # bucket_id → first entry index (-1 = empty)
    var _rows: List[Int32]             # entry → row index
    var _next: List[Int32]             # entry → next entry in chain
    var _bucket_hashes: List[UInt64]   # bucket_id → original hash (for rehash)

    def __init__(out self, capacity: Int = 0):
        var cap = Int(next_power_of_two(max(capacity * 2, 16)))
        self._capacity = cap
        self._mask = cap - 1
        self._count = 0
        self._ctrl = alloc[UInt8](cap)
        self._slots = alloc[Int32](cap)
        for i in range(cap):
            self._ctrl[i] = Self.EMPTY
        self._heads = List[Int32](capacity=capacity)
        self._rows = List[Int32](capacity=capacity)
        self._next = List[Int32](capacity=capacity)
        self._bucket_hashes = List[UInt64](capacity=capacity)

    def __del__(deinit self):
        if self._capacity > 0:
            self._ctrl.free()
            self._slots.free()

    @always_inline
    def _stamp(self, h: UInt64) -> UInt8:
        """Extract 7-bit stamp from hash (high bits, avoids correlation with slot)."""
        return UInt8(h >> 57)  # top 7 bits

    @always_inline
    def _slot_index(self, h: UInt64) -> Int:
        """Slot index from hash (low bits)."""
        return Int(h) & self._mask

    def hash_keys(
        self, keys: StructArray
    ) raises -> PrimitiveArray[uint64]:
        return Self.hash_fn(keys)

    def _grow(mut self):
        """Double capacity and rehash all occupied slots."""
        var old_cap = self._capacity
        var old_ctrl = self._ctrl
        var old_slots = self._slots
        var new_cap = old_cap * 2
        self._capacity = new_cap
        self._mask = new_cap - 1
        self._ctrl = alloc[UInt8](new_cap)
        self._slots = alloc[Int32](new_cap)
        for i in range(new_cap):
            self._ctrl[i] = Self.EMPTY

        # Re-insert all occupied slots. We need the original hashes, but
        # we only stored stamps. Reconstruct slot positions from bucket
        # heads — each bucket's hash can be recovered by scanning the
        # _heads array and finding which old slot held each bucket_id.
        # Simpler: just scan old table and re-probe using stamp + slot.
        # Since stamp = top 7 bits and slot = hash & old_mask, we can't
        # fully recover the hash. Instead, store the full hash per bucket.
        #
        # WORKAROUND: We keep a _hashes list (one per bucket) for rehash.
        # This is populated during insert.
        for i in range(old_cap):
            if old_ctrl[i] != Self.EMPTY:
                var bid = Int(old_slots[i])
                var h = self._bucket_hashes[bid]
                var stamp = self._stamp(h)
                var slot = self._slot_index(h)
                while self._ctrl[slot] != Self.EMPTY:
                    slot = (slot + 1) & self._mask
                self._ctrl[slot] = stamp
                self._slots[slot] = Int32(bid)
        old_ctrl.free()
        old_slots.free()

    def insert(mut self, h: UInt64, row: Int32) -> Int:
        var stamp = self._stamp(h)
        var slot = self._slot_index(h)

        # Probe for existing entry with matching stamp.
        while True:
            var ctrl = self._ctrl[slot]
            if ctrl == Self.EMPTY:
                break  # not found
            if ctrl == stamp:
                # Stamp matches — check if hash fully matches (trust 64-bit hash).
                var bid = Int(self._slots[slot])
                if self._bucket_hashes[bid] == h:
                    # Existing bucket — append to chain.
                    var entry_idx = Int32(len(self._rows))
                    self._rows.append(row)
                    self._next.append(self._heads[bid])
                    self._heads[bid] = entry_idx
                    return bid
            slot = (slot + 1) & self._mask

        # New bucket — insert into empty slot.
        if self._count * Self._LOAD_FACTOR_DEN >= self._capacity * Self._LOAD_FACTOR_NUM:
            self._grow()
            # Re-probe after grow.
            slot = self._slot_index(h)
            while self._ctrl[slot] != Self.EMPTY:
                slot = (slot + 1) & self._mask

        var bid = len(self._heads)
        self._ctrl[slot] = stamp
        self._slots[slot] = Int32(bid)
        self._count += 1
        self._bucket_hashes.append(h)

        var entry_idx = Int32(len(self._rows))
        self._rows.append(row)
        self._next.append(Int32(-1))
        self._heads.append(entry_idx)
        return bid

    def find(self, h: UInt64) -> Int:
        var stamp = self._stamp(h)
        var slot = self._slot_index(h)
        while True:
            var ctrl = self._ctrl[slot]
            if ctrl == Self.EMPTY:
                return -1
            if ctrl == stamp:
                var bid = Int(self._slots[slot])
                if self._bucket_hashes[bid] == h:
                    return bid
            slot = (slot + 1) & self._mask

    def find_or_insert(mut self, h: UInt64) -> Int:
        var stamp = self._stamp(h)
        var slot = self._slot_index(h)

        while True:
            var ctrl = self._ctrl[slot]
            if ctrl == Self.EMPTY:
                break
            if ctrl == stamp:
                var bid = Int(self._slots[slot])
                if self._bucket_hashes[bid] == h:
                    return bid
            slot = (slot + 1) & self._mask

        # New bucket — empty slot, no row stored.
        if self._count * Self._LOAD_FACTOR_DEN >= self._capacity * Self._LOAD_FACTOR_NUM:
            self._grow()
            slot = self._slot_index(h)
            while self._ctrl[slot] != Self.EMPTY:
                slot = (slot + 1) & self._mask

        var bid = len(self._heads)
        self._ctrl[slot] = stamp
        self._slots[slot] = Int32(bid)
        self._count += 1
        self._bucket_hashes.append(h)
        self._heads.append(Int32(-1))
        return bid

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
