"""Shared hash table infrastructure for join and groupby kernels.

Provides:
  - ``HashTable`` trait — core interface for hash-based row indexing
  - ``DictHashTable`` — Dict-backed implementation (first concrete backend)
  - ``Partition`` / ``Partitioner`` / ``NoPartition`` — partitioning layer

Architecture:
  Hash Function  →  Partitioner  →  HashTable  →  Operator (join / groupby)
  Each layer is independently swappable.
"""

from std.hashlib import Hasher
from std.memory import Span

from ..arrays import PrimitiveArray, StructArray
from ..builders import PrimitiveBuilder
from ..dtypes import int32, uint64
from .hashing import hash_


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
        # UInt64.__hash__ goes through _update_with_simd, not bytes.
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
    Each bucket holds a list of Int32 row indices.

    Shared by join and groupby:
      - Join uses ``insert()`` to build, ``find()`` + ``bucket_*`` to probe.
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

    def bucket_len(self, bid: Int) -> Int:
        """Number of row indices in bucket bid."""
        ...

    def bucket_row(self, bid: Int, k: Int) -> Int32:
        """k-th row index in bucket bid."""
        ...

    def num_buckets(self) -> Int:
        """Number of unique keys (buckets created so far)."""
        ...


# ---------------------------------------------------------------------------
# DictHashTable — Dict-backed HashTable implementation
# ---------------------------------------------------------------------------


struct DictHashTable[
    hash_fn: def (StructArray) raises -> PrimitiveArray[uint64] = hash_
](HashTable):
    """Dict-backed hash table.

    Maps UInt64 hash → bucket_id via ``Dict[UInt64, Int]``.
    Buckets stored in ``List[List[Int32]]`` indexed by bucket_id.
    Hash function passed as comptime parameter (default: ``hash_``).

    Future alternatives (same HashTable trait):
      - SIMDHashTable — open addressing with SIMD probing
      - PerfectHashTable — direct array indexing for small-domain keys
    """

    var _map: Dict[UInt64, Int, IdentityHasher]
    var _buckets: List[List[Int32]]

    def __init__(out self):
        self._map = Dict[UInt64, Int, IdentityHasher]()
        self._buckets = List[List[Int32]]()

    def hash_keys(
        self, keys: StructArray
    ) raises -> PrimitiveArray[uint64]:
        return Self.hash_fn(keys)

    def insert(mut self, h: UInt64, row: Int32) -> Int:
        var existing = self._map.get(h)
        if existing:
            var bid = existing.value()
            self._buckets[bid].append(row)
            return bid
        var bid = len(self._buckets)
        self._map[h] = bid
        var bucket = List[Int32]()
        bucket.append(row)
        self._buckets.append(bucket^)
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
        var bid = len(self._buckets)
        self._map[h] = bid
        self._buckets.append(List[Int32]())
        return bid

    def bucket_len(self, bid: Int) -> Int:
        return len(self._buckets[bid])

    def bucket_row(self, bid: Int, k: Int) -> Int32:
        return self._buckets[bid][k]

    def num_buckets(self) -> Int:
        return len(self._buckets)
