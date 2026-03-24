"""Shared hash table infrastructure for join and groupby kernels.

Provides:
  - ``HashTable`` trait — core interface for hash-based row indexing
  - ``SwissHashTable`` — SIMD group matching with pipelined probing
  - ``Partition`` / ``Partitioner`` / ``NoPartition`` — partitioning layer

Architecture:
  Hash Function  →  Partitioner  →  HashTable  →  Operator (join / groupby)
  Each layer is independently swappable.
"""

from std.bit import count_trailing_zeros, next_power_of_two
from std.memory import alloc, memset, pack_bits
from std.sys.intrinsics import prefetch, PrefetchOptions

from ..arrays import PrimitiveArray
from ..builders import PrimitiveBuilder
from ..buffers import BufferBuilder
from ..dtypes import int32, uint64



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
    """Maps UInt64 hashes → sequential key IDs with row index storage.

    Shared by join and groupby:
      - Join uses ``build()`` + ``probe()``
      - GroupBy uses ``insert()`` + ``num_keys()``
    """

    def insert(
        mut self, hashes: PrimitiveArray[uint64]
    ) raises -> PrimitiveArray[int32]:
        """Insert hashes into the table, returning a key ID per hash.
        Existing hashes return their previous ID; new hashes get
        sequential IDs starting from ``num_keys()``."""
        ...

    def build(mut self, hashes: PrimitiveArray[uint64]) raises:
        """``insert`` + CSR row-index construction.
        Required before ``probe``."""
        ...

    def probe(
        self,
        hashes: PrimitiveArray[uint64],
        num_build_rows: Int,
        single_match: Bool = False,
    ) raises -> Tuple[PrimitiveArray[int32], PrimitiveArray[int32]]:
        """Look up hashes against the built index, returning matching
        (build_row, probe_row) index pairs. If single_match, emit at
        most one match per probe row."""
        ...

    def num_keys(self) -> Int:
        """Number of unique keys inserted so far."""
        ...


# ---------------------------------------------------------------------------
# SwissHashTable — flat open-addressing hash table
# ---------------------------------------------------------------------------


comptime _GROUP_WIDTH: Int = 16
"""Number of control bytes per group (matches Mojo Dict / abseil)."""

comptime _CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""

comptime _PIPE_DEPTH: Int = 16
"""Number of probes pipelined in a single batch (prefetch window)."""


@always_inline
def _h2(h: UInt64) -> UInt8:
    """Extract 7-bit H2 fingerprint from hash (top 7 bits)."""
    return UInt8(h >> 57)


struct SwissHashTable(HashTable):
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

    var _bucket_hashes: List[UInt64]   # dense: bucket_id → hash
    var _num_buckets: Int

    # CSR (Compressed Sparse Row) storage for row indices.
    # After build(): _offsets[bid]..._offsets[bid+1] is the range in _rows.
    var _offsets: List[Int]            # len = num_buckets + 1
    var _rows: List[Int32]             # flat, grouped by bucket

    def __init__(out self, capacity: Int = 0):
        var cap = Int(next_power_of_two(max(capacity * 2, _GROUP_WIDTH)))
        self._capacity = cap
        self._mask = cap - 1
        self._count = 0
        self._max_count = cap * 7 // 8
        self._ctrl = alloc[UInt8](cap + _GROUP_WIDTH)
        self._slots = alloc[Int32](cap)
        memset(self._ctrl, _CTRL_EMPTY, cap + _GROUP_WIDTH)
        self._bucket_hashes = List[UInt64](capacity=capacity)
        self._num_buckets = 0
        self._offsets = List[Int]()
        self._rows = List[Int32]()

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

    @staticmethod
    @always_inline
    def _match_h2(ctrl: UnsafePointer[UInt8, _], pos: Int, h2: UInt8) -> UInt16:
        """SIMD bitmask of slots in the group at ``pos`` matching ``h2``."""
        var group = (ctrl + pos).load[width=_GROUP_WIDTH]()
        return pack_bits(group.eq(SIMD[DType.uint8, _GROUP_WIDTH](h2)))

    @staticmethod
    @always_inline
    def _match_empty(ctrl: UnsafePointer[UInt8, _], pos: Int) -> UInt16:
        """SIMD bitmask of empty slots in the group at ``pos``."""
        var group = (ctrl + pos).load[width=_GROUP_WIDTH]()
        return pack_bits(group.eq(SIMD[DType.uint8, _GROUP_WIDTH](_CTRL_EMPTY)))

    @always_inline
    def _find(
        self,
        h: UInt64,
        ctrl: UnsafePointer[UInt8, _],
        slots: UnsafePointer[Int32, _],
        mask: Int,
    ) -> Int:
        """Probe for a matching bucket. Returns bucket_id or -1."""
        var h2 = _h2(h)
        var pos = Int(h) & mask
        while True:
            var match_mask = Self._match_h2(ctrl, pos, h2)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & mask
                var candidate = Int(slots[slot_idx])
                if self._bucket_hashes[candidate] == h:
                    return candidate
                match_mask &= match_mask - 1
            if Self._match_empty(ctrl, pos) != 0:
                return -1
            pos = (pos + _GROUP_WIDTH) & mask

    @always_inline
    def _find_empty_slot(
        self,
        h: UInt64,
        ctrl: UnsafePointer[UInt8, _],
        mask: Int,
    ) -> Int:
        """Find the first empty slot for hash ``h``. Used during resize."""
        var pos = Int(h) & mask
        while True:
            var empty_mask = Self._match_empty(ctrl, pos)
            if empty_mask != 0:
                var bit = count_trailing_zeros(Int(empty_mask))
                return (pos + bit) & mask
            pos = (pos + _GROUP_WIDTH) & mask

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
                    var slot = self._find_empty_slot(
                        h, self._ctrl, self._mask
                    )
                    self._set_ctrl(slot, _h2(h))
                    self._slots[slot] = Int32(bid)
            old_ctrl.free()
            old_slots.free()
        self._bucket_hashes.reserve(n)

    def insert(
        mut self, hashes: PrimitiveArray[uint64]
    ) raises -> PrimitiveArray[int32]:
        """Insert hashes, returning a key ID per hash.

        Prefetches ctrl bytes ``_PIPE_DEPTH`` iterations ahead to hide
        memory latency. Existing hashes return their previous ID; new
        hashes get sequential IDs starting from ``num_keys()``.
        """
        var n = len(hashes)
        self.reserve(n)
        var hash_ptr = hashes.buffer.unsafe_ptr[uint64.native](hashes.offset)
        var ctrl = self._ctrl
        var slots = self._slots
        var mask = self._mask

        var bid_builder = PrimitiveBuilder[int32](capacity=n, zeroed=False)

        # Warm up the prefetch pipeline.
        for i in range(min(_PIPE_DEPTH, n)):
            prefetch(ctrl + (Int(UInt64(hash_ptr[i])) & mask))

        for i in range(n):
            var h = UInt64(hash_ptr[i])
            var h2 = _h2(h)

            # Prefetch ctrl bytes _PIPE_DEPTH iterations ahead.
            var ahead = i + _PIPE_DEPTH
            if ahead < n:
                prefetch(ctrl + (Int(UInt64(hash_ptr[ahead])) & mask))

            # Find existing bucket.
            var bid = self._find(h, ctrl, slots, mask)

            # Insert new bucket if not found.
            if bid == -1:
                if self._count >= self._max_count:
                    self.reserve(self._capacity)
                    ctrl = self._ctrl
                    slots = self._slots
                    mask = self._mask

                var slot = self._find_empty_slot(h, ctrl, mask)
                bid = self._num_buckets
                self._set_ctrl(slot, h2)
                slots[slot] = Int32(bid)
                self._count += 1
                self._bucket_hashes.append(h)
                self._num_buckets += 1

            bid_builder.unsafe_append(Scalar[int32.native](bid))

        return bid_builder.finish()

    def build(mut self, hashes: PrimitiveArray[uint64]) raises:
        """``insert`` + CSR row-index construction.

        After this call, ``_offsets[bid]..._offsets[bid+1]`` gives the
        row range in ``_rows`` for each bucket. Required before
        ``probe``.
        """
        var bids = self.insert(hashes)
        var n = len(bids)

        # Count rows per bucket.
        var counts = List[Int](length=self._num_buckets, fill=0)
        for i in range(n):
            counts[Int(bids.unsafe_get(i))] += 1

        # Prefix sum → offsets.
        self._offsets = List[Int](length=self._num_buckets + 1, fill=0)
        for b in range(self._num_buckets):
            self._offsets[b + 1] = self._offsets[b] + counts[b]

        # Scatter row indices into CSR rows array.
        var total = self._offsets[self._num_buckets]
        self._rows = List[Int32](length=total, fill=Int32(0))
        for b in range(self._num_buckets):
            counts[b] = self._offsets[b]
        for i in range(n):
            var bid = Int(bids.unsafe_get(i))
            self._rows[counts[bid]] = Int32(i)
            counts[bid] += 1

    def probe(
        self,
        hashes: PrimitiveArray[uint64],
        num_build_rows: Int,
        single_match: Bool = False,
    ) raises -> Tuple[PrimitiveArray[int32], PrimitiveArray[int32]]:
        """Look up hashes against the built index, returning matching
        ``(build_row, probe_row)`` index pairs.

        Prefetches ctrl bytes ``_PIPE_DEPTH`` iterations ahead to hide
        memory latency. Uses raw-pointer access to slots, bucket hashes,
        offsets, and rows to avoid bounds-check overhead.
        """
        var n = len(hashes)
        var est = min(n, num_build_rows)
        var left_out = PrimitiveBuilder[int32](capacity=est, zeroed=False)
        var right_out = PrimitiveBuilder[int32](capacity=est, zeroed=False)

        if n == 0:
            return (
                left_out.finish(shrink_to_fit=False),
                right_out.finish(shrink_to_fit=False),
            )

        var hash_ptr = hashes.buffer.unsafe_ptr[uint64.native](hashes.offset)
        var ctrl = self._ctrl
        var mask = self._mask
        var slots = self._slots
        var off_ptr = self._offsets.unsafe_ptr()
        var rows_ptr = self._rows.unsafe_ptr()

        # Warm up the prefetch pipeline.
        for i in range(min(_PIPE_DEPTH, n)):
            prefetch(ctrl + (Int(UInt64(hash_ptr[i])) & mask))

        for probe_row in range(n):
            var h = UInt64(hash_ptr[probe_row])

            # Prefetch ctrl bytes _PIPE_DEPTH iterations ahead.
            var ahead = probe_row + _PIPE_DEPTH
            if ahead < n:
                prefetch(ctrl + (Int(UInt64(hash_ptr[ahead])) & mask))

            var bid = self._find(h, ctrl, slots, mask)
            if bid == -1:
                continue

            var start = off_ptr[bid]
            var end = off_ptr[bid + 1]
            for j in range(start, end):
                left_out.unsafe_append(Scalar[int32.native](rows_ptr[j]))
                right_out.unsafe_append(Scalar[int32.native](probe_row))
                if single_match:
                    break

        return (
            left_out.finish(shrink_to_fit=False),
            right_out.finish(shrink_to_fit=False),
        )

    def num_keys(self) -> Int:
        return self._num_buckets
