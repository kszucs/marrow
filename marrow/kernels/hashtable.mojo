"""Swiss Table hash table for join and groupby kernels.

Provides:
  - ``SwissHashTable`` — SIMD group matching with pipelined probing
  - ``Partition`` / ``Partitioner`` / ``NoPartition`` — partitioning layer

Architecture:
  Hash Function  →  Partitioner  →  SwissHashTable  →  Operator (join / groupby)
  Each layer is independently swappable.
"""

from std.bit import count_trailing_zeros, next_power_of_two
from std.gpu.host import DeviceContext
from std.memory import pack_bits
from std.sys import size_of
from std.sys.intrinsics import prefetch

from ..arrays import PrimitiveArray, AnyArray, StructArray
from ..builders import PrimitiveBuilder
from ..buffers import Buffer
from ..dtypes import int32, uint64
from .compare import equal
from .filter import take, filter_
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
# SwissHashTable — flat open-addressing hash table
# ---------------------------------------------------------------------------


comptime _GROUP_WIDTH: Int = 16
"""Number of control bytes per group (matches Mojo Dict / abseil)."""

comptime _CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""

comptime _PIPE_DEPTH: Int = 16
"""Number of probes pipelined in a single batch (prefetch window)."""


struct SwissHashTable[
    hasher: def(StructArray, Optional[DeviceContext]) raises -> PrimitiveArray[
        uint64
    ] = rapidhash
](Movable):
    """Swiss Table hash table with SIMD group matching.

    Open-addressing hash table using the Swiss Table design (abseil /
    Mojo Dict / hashbrown). Accepts ``StructArray`` key columns and
    handles hashing internally via the ``hash_fn`` parameter.

    Provides two main operations:

    - **insert** — hash keys and batch find-or-insert, returning a bucket
      ID per row.  Used by groupby to assign group IDs.
    - **build + probe** — two-phase hash join.  ``build`` hashes build-side
      keys, inserts them, and creates a CSR row index.  ``probe`` hashes
      probe-side keys, looks them up, verifies key equality (filtering
      hash collisions), and returns matching ``(build_row, probe_row)``
      index pairs.

    Terminology
    -----------
    - **slot**: Position in the flat ctrl/slots arrays (size = capacity).
      Each slot holds one control byte and one bucket ID.
    - **bucket** (or **key ID**): Sequential integer (0, 1, 2, ...) for
      each unique hash ever inserted. Stored in ``_slots[slot]`` and
      returned by ``insert()``.  ``_bucket_hashes[bucket]`` stores the
      original hash for collision verification.
    - **row**: Original input row index. After ``build()``, rows are
      stored in the CSR arrays ``_offsets`` / ``_rows``, grouped by bucket.

    Storage layout
    --------------
    ::

        ctrl   [ 0xFF | 0xFF | h2=0x3A | 0xFF | h2=0x1B | ... ]   capacity + 16 bytes
        slots  [  --  |  --  |   bid=0 |  --  |   bid=1 | ... ]   capacity × 4 bytes
                                  │                  │
                  ┌───────────────┘                  │
                  ▼                                  ▼
        _bucket_hashes  [ hash_for_bid_0, hash_for_bid_1, ... ]   num_buckets × 8 bytes

        After build():
        _offsets  [ 0, 2, 5, ... ]     CSR offsets per bucket (len = num_buckets + 1)
        _rows     [ r3, r7, r1, r4, r9, ... ]   row indices, grouped by bucket

    Control bytes
    -------------
    Each slot has a 1-byte control tag:

    - ``0xFF`` = EMPTY (available for insertion)
    - ``0x00..0x7F`` = H2 fingerprint (top 7 bits of the hash)

    Probing loads 16 control bytes at once (one "group") via SIMD and
    uses ``pack_bits`` to get a bitmask of matching H2 fingerprints.
    This checks 16 slots in a single comparison.

    Prefetch pipeline
    -----------------
    Both ``insert`` and ``probe`` issue ``prefetch`` for the ctrl group
    of the hash ``_PIPE_DEPTH`` (16) iterations ahead, warming L1 cache
    before the actual lookup.

    Parameters
    ----------
    ``hash_fn``
        Hash function mapping ``StructArray`` → ``PrimitiveArray[uint64]``.
        Defaults to ``rapidhash``.
    """

    comptime H = UInt64
    """Hash scalar type."""

    var _ctrl: Buffer[mut=True]
    """Control bytes: 1 byte per slot + ``_GROUP_WIDTH`` padding for
    SIMD loads at the end. ``0xFF`` = empty, ``0x00–0x7F`` = H2."""

    var _slots: Buffer[mut=True]
    """Bucket ID (Int32) stored at each slot position."""

    var _capacity: Int
    """Total number of slots (always a power of 2)."""

    var _mask: Int
    """``_capacity - 1``, used for fast modulo via ``hash & _mask``."""

    var _count: Int
    """Number of occupied slots."""

    var _max_count: Int
    """Resize threshold: ``_capacity * 7 / 8``."""

    var _bucket_hashes: Buffer[mut=True]
    """Dense array of hashes indexed by bucket ID. Used to verify
    H2-matching candidates against the full hash."""

    var _num_buckets: Int
    """Number of unique keys (buckets) inserted so far."""

    var _offsets: Buffer[mut=True]
    """CSR offsets: ``_offsets[bid]`` .. ``_offsets[bid+1]`` is the range
    in ``_rows`` for bucket ``bid``. Populated by ``build()``."""

    var _rows: Buffer[mut=True]
    """CSR row indices, grouped by bucket. Populated by ``build()``."""

    def __init__(out self, capacity: Int = 0):
        """Create an empty hash table.

        Args:
            capacity: Expected number of unique keys. The table
                pre-allocates ``2 * capacity`` slots (rounded up to the
                next power of 2) and reserves space for ``capacity``
                bucket hashes.
        """
        var cap = Int(next_power_of_two(max(capacity * 2, _GROUP_WIDTH)))
        self._capacity = cap
        self._mask = cap - 1
        self._count = 0
        self._max_count = cap * 7 // 8
        self._ctrl = Buffer.alloc_filled(cap + _GROUP_WIDTH, fill=_CTRL_EMPTY)
        self._slots = Buffer.alloc_uninit[DType.int32](cap)
        self._bucket_hashes = Buffer.alloc_uninit[DType.uint64](
            max(capacity, 16)
        )
        self._num_buckets = 0
        self._offsets = Buffer.alloc_uninit(0)
        self._rows = Buffer.alloc_uninit(0)

    # ------------------------------------------------------------------
    # Internal helpers — hash, ctrl, slot, bucket, CSR accessors
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def _h2(h: Self.H) -> UInt8:
        """Extract a 7-bit fingerprint from the top bits of ``h``.

        The fingerprint occupies bits ``[bit_width-7 : bit_width)`` of
        the hash.  For 64-bit hashes this is ``h >> 57``, yielding a
        value in ``0x00..0x7F`` — disjoint from ``_CTRL_EMPTY (0xFF)``.
        """
        comptime shift = size_of[Self.H]() * 8 - 7
        return UInt8(UInt64(h) >> UInt64(shift))

    @always_inline
    def _prefetch_ctrl(self, h: Self.H):
        """Prefetch the ctrl group for hash ``h`` into L1 cache."""
        prefetch(
            self._ctrl.view[DType.uint8]().unsafe_ptr()
            + Int(h & Self.H(self._mask))
        )

    @always_inline
    def _get_offset(self, index: Int) -> Int:
        """Read CSR offset at ``index``."""
        return Int(self._offsets.unsafe_get[DType.int64](index))

    @always_inline
    def _set_offset(mut self, index: Int, val: Int):
        """Write CSR offset at ``index``."""
        self._offsets.unsafe_set[DType.int64](index, Int64(val))

    @always_inline
    def _get_row(self, index: Int) -> Int32:
        """Read row index at CSR position ``index``."""
        return self._rows.unsafe_get[DType.int32](index)

    @always_inline
    def _set_row(mut self, index: Int, val: Int32):
        """Write row index at CSR position ``index``."""
        self._rows.unsafe_set[DType.int32](index, val)

    @always_inline
    def _get_hash(self, index: Int) -> Self.H:
        """Read the full hash stored for bucket ``index``."""
        return self._bucket_hashes.unsafe_get[DType.uint64](index)

    @always_inline
    def _set_hash(mut self, index: Int, h: Self.H):
        """Write the full hash for bucket ``index``."""
        self._bucket_hashes.unsafe_set[DType.uint64](index, h)

    @always_inline
    def _get_slot(self, index: Int) -> Int:
        """Read the bucket ID stored at slot ``index``."""
        return Int(self._slots.unsafe_get[DType.int32](index))

    @always_inline
    def _set_slot(mut self, index: Int, bid: Int):
        """Write bucket ID ``bid`` into slot ``index``."""
        self._slots.unsafe_set[DType.int32](index, Int32(bid))

    @always_inline
    def _set_ctrl(mut self, slot: Int, h: Self.H):
        """Write the H2 fingerprint of ``h`` into ctrl at ``slot``.

        Also mirrors the byte into the trailing padding region when
        ``slot < _GROUP_WIDTH`` so that SIMD loads at the end of the
        ctrl array see consistent data.
        """
        var h2 = Self._h2(h)
        self._ctrl.unsafe_set[DType.uint8](slot, h2)
        if slot < _GROUP_WIDTH:
            self._ctrl.unsafe_set[DType.uint8](self._capacity + slot, h2)

    # ------------------------------------------------------------------
    # SIMD group matching
    # ------------------------------------------------------------------

    @always_inline
    def _match_ctrl(self, pos: Int, h: Self.H) -> UInt16:
        """Load the 16-byte ctrl group at ``pos`` and return a bitmask
        of slots whose H2 fingerprint matches ``h``.

        Each set bit ``k`` means ``ctrl[pos + k]`` has the same H2 as
        ``h``.  Use ``count_trailing_zeros`` to iterate the matches.
        """
        var group = self._ctrl.view[DType.uint8]().load[_GROUP_WIDTH](pos)
        return pack_bits(group.eq(SIMD[DType.uint8, _GROUP_WIDTH](Self._h2(h))))

    @always_inline
    def _match_empty_ctrl(self, pos: Int) -> UInt16:
        """Load the 16-byte ctrl group at ``pos`` and return a bitmask
        of empty (``0xFF``) slots.
        """
        var group = self._ctrl.view[DType.uint8]().load[_GROUP_WIDTH](pos)
        return pack_bits(group.eq(SIMD[DType.uint8, _GROUP_WIDTH](_CTRL_EMPTY)))

    # ------------------------------------------------------------------
    # Probing primitives
    # ------------------------------------------------------------------

    @always_inline
    def _find_slot(self, h: Self.H) -> Int:
        """Probe the table for ``h``, returning its bucket ID or ``-1``.

        Walks groups starting at ``h & mask``. For each group:
        1. SIMD-match the H2 fingerprint against all 16 ctrl bytes.
        2. For each H2 hit, verify the full hash via ``_get_hash``.
        3. If an empty ctrl byte is found, the key is absent → return -1.
        4. Otherwise advance to the next group (linear probing).
        """
        var pos = Int(h & Self.H(self._mask))
        while True:
            var match_mask = self._match_ctrl(pos, h)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & self._mask
                var candidate = self._get_slot(slot_idx)
                if self._get_hash(candidate) == h:
                    return candidate
                match_mask &= match_mask - 1
            if self._match_empty_ctrl(pos) != 0:
                return -1
            pos = (pos + _GROUP_WIDTH) & self._mask

    @always_inline
    def _find_empty_slot(self, h: Self.H) -> Int:
        """Find the first empty slot for ``h`` (used during insert/resize).

        Same linear-probing walk as ``_find_slot`` but only checks for
        empty ctrl bytes — no H2 or full-hash comparison.
        """
        var pos = Int(h & Self.H(self._mask))
        while True:
            var empty_mask = self._match_empty_ctrl(pos)
            if empty_mask != 0:
                var bit = count_trailing_zeros(Int(empty_mask))
                return (pos + bit) & self._mask
            pos = (pos + _GROUP_WIDTH) & self._mask

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def reserve(mut self, n: Int) raises:
        """Ensure the table can hold at least ``n`` entries without resize.

        If the current capacity is insufficient, allocates new ctrl/slots
        buffers with ``2n`` slots (next power of 2), then re-inserts all
        existing entries. Also grows ``_bucket_hashes`` if needed.
        """
        var needed = Int(next_power_of_two(max(n * 2, _GROUP_WIDTH)))
        if needed > self._capacity:
            var old_cap = self._capacity
            self._capacity = needed
            self._mask = needed - 1
            self._max_count = needed * 7 // 8

            # Swap in new buffers, re-insert from old.
            var old_ctrl = self._ctrl^
            var old_slots = self._slots^

            self._ctrl = Buffer.alloc_filled(
                needed + _GROUP_WIDTH, fill=_CTRL_EMPTY
            )
            self._slots = Buffer.alloc_uninit[DType.int32](needed)

            for i in range(old_cap):
                if old_ctrl.unsafe_get[DType.uint8](i) != _CTRL_EMPTY:
                    var bid = Int(old_slots.unsafe_get[DType.int32](i))
                    var h = self._get_hash(bid)
                    var slot = self._find_empty_slot(h)
                    self._set_ctrl(slot, h)
                    self._set_slot(slot, bid)

        # Grow _bucket_hashes if needed.
        if n * size_of[Self.H]() > len(self._bucket_hashes):
            self._bucket_hashes.resize[DType.uint64](n)

    # ------------------------------------------------------------------
    # Hash-level operations (private)
    # ------------------------------------------------------------------

    def _insert_hashes(
        mut self, hashes: PrimitiveArray[uint64]
    ) raises -> PrimitiveArray[int32]:
        """Batch insert hashes, returning a bucket ID per input hash.

        For each hash:
        1. Look up via ``_find_slot``. If found, reuse the existing
           bucket ID.
        2. If not found, claim an empty slot via ``_find_empty_slot``,
           assign the next sequential bucket ID, and record the hash
           in ``_bucket_hashes``.

        Prefetches ctrl groups ``_PIPE_DEPTH`` iterations ahead to hide
        memory latency on the critical lookup path.

        Returns:
            ``PrimitiveArray[int32]`` of length ``len(hashes)`` where
            element ``i`` is the bucket ID for ``hashes[i]``.
        """
        var n = len(hashes)
        self.reserve(n)

        var bid_builder = PrimitiveBuilder[int32](capacity=n, zeroed=False)

        # Warm up the prefetch pipeline.
        for i in range(min(_PIPE_DEPTH, n)):
            self._prefetch_ctrl(UInt64(hashes.unsafe_get(i)))

        for i in range(n):
            var h = UInt64(hashes.unsafe_get(i))

            # Prefetch ctrl bytes _PIPE_DEPTH iterations ahead.
            var ahead = i + _PIPE_DEPTH
            if ahead < n:
                self._prefetch_ctrl(UInt64(hashes.unsafe_get(ahead)))

            # Find existing bucket.
            var bid = self._find_slot(h)

            # Insert new bucket if not found.
            if bid == -1:
                if self._count >= self._max_count:
                    self.reserve(self._capacity)

                var slot = self._find_empty_slot(h)
                bid = self._num_buckets
                self._set_ctrl(slot, h)
                self._set_slot(slot, bid)
                self._count += 1
                self._set_hash(self._num_buckets, h)
                self._num_buckets += 1

            bid_builder.unsafe_append(Scalar[int32.native](bid))

        return bid_builder.finish()

    def _build_hashes(mut self, hashes: PrimitiveArray[uint64]) raises:
        """Insert hashes and build a CSR row index.

        Calls ``insert()`` to populate the hash table, then constructs
        Compressed Sparse Row (CSR) storage so that ``probe()`` can
        iterate all build-side rows for a given bucket in a contiguous
        memory range::

            _offsets[bid] .. _offsets[bid+1]  →  range in _rows
            _rows[j]                          →  original build-side row index

        Must be called before ``probe()``.
        """
        var bids = self._insert_hashes(hashes)
        var n = len(bids)
        var nb = self._num_buckets

        # Count rows per bucket.
        var counts = List[Int](length=nb, fill=0)
        for i in range(n):
            counts[Int(bids.unsafe_get(i))] += 1

        # Prefix sum → offsets.
        self._offsets = Buffer.alloc_uninit[DType.int64](nb + 1)
        self._set_offset(0, 0)
        for b in range(nb):
            self._set_offset(b + 1, self._get_offset(b) + counts[b])

        # Scatter row indices into CSR rows array.
        var total = self._get_offset(nb)
        self._rows = Buffer.alloc_uninit[DType.int32](total)
        for b in range(nb):
            counts[b] = self._get_offset(b)
        for i in range(n):
            var bid = Int(bids.unsafe_get(i))
            self._set_row(counts[bid], Int32(i))
            counts[bid] += 1

    def _probe_hashes(
        self,
        hashes: PrimitiveArray[uint64],
        num_build_rows: Int,
        single_match: Bool = False,
    ) raises -> Tuple[PrimitiveArray[int32], PrimitiveArray[int32]]:
        """Look up probe-side hashes and return matching row index pairs.

        For each probe hash, finds the corresponding bucket via
        ``_find_slot()``, then iterates the CSR row range for that bucket
        to emit ``(build_row, probe_row)`` pairs.

        Equivalent Python logic::

            for probe_row, h in enumerate(hashes):
                bid = table.find(h)
                if bid != -1:
                    for build_row in rows[offsets[bid]:offsets[bid+1]]:
                        emit(build_row, probe_row)

        Args:
            hashes: Probe-side hash array.
            num_build_rows: Number of build-side rows (for output
                capacity estimation).
            single_match: If True, emit at most one match per probe row
                (used for ``JOIN_ANY`` / semi-join semantics).

        Returns:
            ``(left_indices, right_indices)`` — parallel int32 arrays
            where ``left_indices[i]`` is a build-side row and
            ``right_indices[i]`` is the corresponding probe-side row.
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

        # Warm up the prefetch pipeline.
        for i in range(min(_PIPE_DEPTH, n)):
            self._prefetch_ctrl(UInt64(hashes.unsafe_get(i)))

        for probe_row in range(n):
            var h = UInt64(hashes.unsafe_get(probe_row))

            # Prefetch ctrl bytes _PIPE_DEPTH iterations ahead.
            var ahead = probe_row + _PIPE_DEPTH
            if ahead < n:
                self._prefetch_ctrl(UInt64(hashes.unsafe_get(ahead)))

            var bid = self._find_slot(h)
            if bid == -1:
                continue

            var start = self._get_offset(bid)
            var end = self._get_offset(bid + 1)
            for j in range(start, end):
                left_out.unsafe_append(Scalar[int32.native](self._get_row(j)))
                right_out.unsafe_append(Scalar[int32.native](probe_row))
                if single_match:
                    break

        return (
            left_out.finish(shrink_to_fit=False),
            right_out.finish(shrink_to_fit=False),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(mut self, keys: StructArray) raises -> PrimitiveArray[int32]:
        """Hash keys and insert, returning a bucket ID per row.

        Used by groupby to assign group IDs.  Does not store keys or
        build a CSR index.
        """
        return self._insert_hashes(Self.hasher(keys, None))

    def build(mut self, keys: StructArray) raises:
        """Hash keys, insert, and build a CSR row index for ``probe()``.

        Must be called before ``probe()``.
        """
        self._build_hashes(Self.hasher(keys, None))

    def probe(
        self,
        build_keys: StructArray,
        probe_keys: StructArray,
        num_build_rows: Int,
        single_match: Bool = False,
    ) raises -> Tuple[PrimitiveArray[int32], PrimitiveArray[int32]]:
        """Hash probe keys, look up matches, verify key equality.

        1. Hash ``probe_keys`` via ``hash_fn``.
        2. Probe the hash table for candidate ``(build_row, probe_row)``
           pairs.
        3. Gather build/probe key values at matched indices and compare
           with vectorized equality. Filter out hash-collision false
           positives.

        Args:
            build_keys: Left (build) side key columns for equality verification.
            probe_keys: Right (probe) side key columns to look up.
            num_build_rows: Number of build-side rows (for capacity estimation).
            single_match: If True, emit at most one match per probe row.

        Returns:
            ``(left_indices, right_indices)`` — verified matching row pairs.
        """
        var probe_hashes = Self.hasher(probe_keys, None)
        var indices = self._probe_hashes(
            probe_hashes, num_build_rows, single_match
        )
        ref build_indices = indices[0]
        ref probe_indices = indices[1]

        # Filter hash-collision false positives by key equality.
        var mask = equal(
            take(build_keys, build_indices), take(probe_keys, probe_indices)
        )
        return (
            filter_[int32](build_indices, mask),
            filter_[int32](probe_indices, mask),
        )

    def num_keys(self) -> Int:
        """Number of unique keys (buckets) inserted so far."""
        return self._num_buckets
