"""Swiss Table hash table for join and groupby kernels.

Provides:
  - ``SwissHashTable`` — SIMD group matching with pipelined probing
  - ``Partition`` / ``Partitioner`` / ``NoPartition`` — partitioning layer

Architecture:
  Hash Function  →  Partitioner  →  SwissHashTable  →  Operator (join / groupby)
  Each layer is independently swappable.
"""

from std.algorithm.functional import sync_parallelize
from std.bit import count_trailing_zeros, next_power_of_two
from std.gpu.host import DeviceContext
from std.memory import pack_bits
from std.sys import size_of
from ..arrays import (
    PrimitiveArray,
    AnyArray,
    StructArray,
    Int32Array,
    UInt64Array,
)
from ..builders import PrimitiveBuilder, Int32Builder
from ..buffers import Buffer
from ..dtypes import int32, uint64, UInt64Type
from ..views import BufferView
from .compare import equal
from .execution import ExecutionContext
from .filter import take, filter_
from .hashing import rapidhash


# ---------------------------------------------------------------------------
# Partitioner — splits rows into partitions by hash
# ---------------------------------------------------------------------------


struct Partition(Copyable, Movable):
    """A subset of rows with pre-computed hashes.

    ``row_indices = None`` means all rows in order (NoPartition fast-path,
    avoids allocating an identity index array).
    """

    var row_indices: Optional[Int32Array]
    var hashes: UInt64Array

    def __init__(
        out self,
        var hashes: UInt64Array,
        var row_indices: Optional[Int32Array] = None,
    ):
        self.hashes = hashes^
        self.row_indices = row_indices^

    def __init__(out self, *, copy: Self):
        self.hashes = copy.hashes.copy()
        self.row_indices = copy.row_indices.copy()

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

    def partition(self, var hashes: UInt64Array) raises -> List[Partition]:
        ...


struct NoPartition(Partitioner):
    """Single partition containing all rows (default, current behavior)."""

    def __init__(out self):
        pass

    def num_partitions(self) -> Int:
        return 1

    def partition(self, var hashes: UInt64Array) raises -> List[Partition]:
        var result = List[Partition]()
        result.append(Partition(hashes^))
        return result^


struct RadixPartitioner(Partitioner):
    """Partition rows by the top ``num_bits`` of their hash.

    The partitioner is the key enabler of partition-parallel joins: each
    partition is independent, so per-partition hash-table builds and probes
    run in parallel with zero cross-thread synchronization.

    Partition count is ``2^num_bits``.  Default (``num_bits=6`` → 64
    partitions) is chosen so each partition's hash table tends to fit in
    L2 cache on typical build sides.

    Top bits are used for partitioning (``h >> (64 - num_bits)``) while the
    ``SwissHashTable`` probes with low bits (``h & mask``). This split
    keeps the partition router and the per-table probe order independent,
    avoiding double-hashing.

    Parallelism of the partitioning pass itself is deliberately deferred:
    the scatter loop is a memory-bandwidth-bound pass that's already quick
    relative to the build phase, and the win from parallel scatter is
    modest compared to the partition-parallel build/probe it enables.
    """

    var num_bits: Int
    """Number of top hash bits consumed by partition routing."""

    var _num_partitions: Int
    """Cached ``1 << num_bits``."""

    var num_threads: Int
    """Workers used for the histogram + scatter passes. ``1`` forces
    serial; ``>1`` uses per-thread histograms and parallel scatter."""

    def __init__(out self, num_bits: Int = 6, num_threads: Int = 1):
        self.num_bits = num_bits
        self._num_partitions = 1 << num_bits
        self.num_threads = max(1, num_threads)

    def __init__(out self, *, copy: Self):
        self.num_bits = copy.num_bits
        self._num_partitions = copy._num_partitions
        self.num_threads = copy.num_threads

    def num_partitions(self) -> Int:
        return self._num_partitions

    def partition(self, var hashes: UInt64Array) raises -> List[Partition]:
        """Split ``hashes`` into ``num_partitions()`` partitions by top bits.

        Each returned ``Partition`` carries the per-partition hash array
        and an ``Int32`` ``row_indices`` mapping partition-local rows back
        to the original input row number.

        Implementation: per-thread histogram → prefix-sum per (thread,
        partition) → parallel scatter into two shared flat buffers (one
        for Int32 row indices, one for UInt64 hashes). Each partition is
        then exposed as a zero-copy ``PrimitiveArray`` slice with
        ``offset`` baked in — ref-counted via ``ArcPointer`` on the
        immutable buffer, so all partitions share the same backing
        storage.  Total allocation: 2 flat buffers of N elements each.
        No atomics: each (thread, partition) writes into a distinct
        contiguous slot computed by the prefix sum.
        """
        var n = len(hashes)
        var p = self._num_partitions
        var shift = UInt64(64 - self.num_bits)
        var src = hashes.values()

        var nt = self.num_threads
        if n < _MIN_PARALLEL_PARTITION_ROWS:
            nt = 1  # dispatch overhead would dominate
        var chunk = (n + nt - 1) // nt

        # 1. Per-thread histogram — ``histograms[t * p + pid]`` is the
        # count of rows for partition ``pid`` handled by thread ``t``.
        var histograms = List[Int](length=nt * p, fill=0)

        @parameter
        def hist_worker(t: Int):
            var start = t * chunk
            if start >= n:
                return
            var end = min(start + chunk, n)
            var base = t * p
            for i in range(start, end):
                var pid = Int(UInt64(src.load[1](i)) >> shift)
                histograms[base + pid] += 1

        sync_parallelize[hist_worker](nt)

        # 2. Partition-major prefix sum → per-thread write offsets into
        # the flat buffers.  ``write_offsets[t * p + pid]`` is where
        # thread ``t`` starts writing partition ``pid``'s rows.
        var write_offsets = List[Int](length=nt * p, fill=0)
        var partition_offsets = List[Int](length=p + 1, fill=0)
        var counts = List[Int](length=p, fill=0)
        var running = 0
        for pid in range(p):
            partition_offsets[pid] = running
            for t in range(nt):
                write_offsets[t * p + pid] = running
                running += histograms[t * p + pid]
            counts[pid] = running - partition_offsets[pid]
        partition_offsets[p] = running

        # 3. Allocate the two flat buffers (N rows total each).
        var row_buf = Buffer.alloc_uninit[int32.native](n)
        var hash_buf = Buffer.alloc_uninit[uint64.native](n)
        var row_view = row_buf.view[int32.native](0, n)
        var hash_view = hash_buf.view[uint64.native](0, n)

        # 4. Parallel scatter — each thread scans its chunk and writes
        # into its precomputed per-partition slots.  No cross-thread
        # contention: every thread ``t`` owns indices ``t * p .. (t+1) * p``
        # of ``write_offsets`` exclusively, so we mutate that array in
        # place as the cursor — avoiding a per-worker ``List[Int]``
        # allocation (each alloc contends on tcmalloc's page heap
        # spinlock, which showed up as ~11% of worker time in profiling).
        @parameter
        def scatter_worker(t: Int):
            var start = t * chunk
            if start >= n:
                return
            var end = min(start + chunk, n)
            var base = t * p
            for i in range(start, end):
                var h = UInt64(src.load[1](i))
                var pid = Int(h >> shift)
                var pos = write_offsets[base + pid]
                row_view.store[1](pos, Int32(i))
                hash_view.store[1](pos, h)
                write_offsets[base + pid] = pos + 1

        sync_parallelize[scatter_worker](nt)

        # 5. Freeze buffers once, then expose per-partition slices via
        # ref-counted shares (ArcPointer bumps — O(1)).
        var row_imm = row_buf^.to_immutable()
        var hash_imm = hash_buf^.to_immutable()

        var result = List[Partition](capacity=p)
        for pid in range(p):
            var sz = counts[pid]
            var off = partition_offsets[pid]
            var row_arr = Int32Array(
                length=sz,
                nulls=0,
                offset=off,
                bitmap=None,
                buffer=row_imm.copy(),
            )
            var hash_arr = UInt64Array(
                length=sz,
                nulls=0,
                offset=off,
                bitmap=None,
                buffer=hash_imm.copy(),
            )
            result.append(Partition(hash_arr^, row_arr^))
        return result^


# ---------------------------------------------------------------------------
# SwissHashTable — flat open-addressing hash table
# ---------------------------------------------------------------------------


comptime _GROUP_WIDTH: Int = 16
"""Number of control bytes per group (matches Mojo Dict / abseil)."""

comptime _MIN_PARALLEL_PARTITION_ROWS: Int = 65_536
"""Row count below which the partitioner collapses to a single worker —
dispatch + per-thread histogram overhead would dominate."""

comptime _CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""

comptime _PIPE_DEPTH: Int = 16
"""Number of probes pipelined in a single batch (prefetch window)."""


struct SwissHashTable[
    hasher: def(StructArray, ExecutionContext) thin raises -> PrimitiveArray[
        UInt64Type
    ] = rapidhash
](Copyable, Movable):
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
        Hash function mapping ``StructArray`` → ``UInt64Array``.
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
        self._ctrl.view[DType.uint8]().prefetch_at(Int(h & Self.H(self._mask)))

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
    # Hash-level operations — public for callers that have pre-computed
    # hashes (e.g. the partition-parallel HashJoin, which hashes the
    # entire build side once up-front and then builds per-partition
    # tables without re-hashing).
    # ------------------------------------------------------------------

    def insert_hashes(mut self, hashes: UInt64Array) raises -> Int32Array:
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
            ``Int32Array`` of length ``len(hashes)`` where
            element ``i`` is the bucket ID for ``hashes[i]``.
        """
        var n = len(hashes)
        self.reserve(n)

        var bid_builder = Int32Builder(capacity=n, zeroed=False)

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

    def build_hashes(mut self, hashes: UInt64Array) raises:
        """Insert hashes and build a CSR row index.

        Calls ``insert()`` to populate the hash table, then constructs
        Compressed Sparse Row (CSR) storage so that ``probe()`` can
        iterate all build-side rows for a given bucket in a contiguous
        memory range::

            _offsets[bid] .. _offsets[bid+1]  →  range in _rows
            _rows[j]                          →  original build-side row index

        Must be called before ``probe()``.
        """
        var bids = self.insert_hashes(hashes)
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

    def probe_hashes(
        self,
        hashes: UInt64Array,
        num_build_rows: Int,
        single_match: Bool = False,
    ) raises -> Tuple[Int32Array, Int32Array]:
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
        var left_out = Int32Builder(capacity=est, zeroed=False)
        var right_out = Int32Builder(capacity=est, zeroed=False)

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

    def insert(
        mut self,
        keys: StructArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises -> Int32Array:
        """Hash keys and insert, returning a bucket ID per row.

        Used by groupby to assign group IDs.  Does not store keys or
        build a CSR index.

        ``ctx`` is forwarded to the hasher only — the insert loop itself
        is serial (concurrent inserts would require atomic slot claiming;
        the designed concurrency pattern is partition-parallel with one
        table per partition).
        """
        return self.insert_hashes(Self.hasher(keys, ctx))

    def build(
        mut self,
        keys: StructArray,
        ctx: ExecutionContext = ExecutionContext.serial(),
    ) raises:
        """Hash keys, insert, and build a CSR row index for ``probe()``.

        Must be called before ``probe()``.
        """
        self.build_hashes(Self.hasher(keys, ctx))

    def probe(
        self,
        build_keys: StructArray,
        probe_keys: StructArray,
        num_build_rows: Int,
        single_match: Bool = False,
        ctx: ExecutionContext = ExecutionContext.serial(),
        hashes: Optional[UInt64Array] = None,
    ) raises -> Tuple[Int32Array, Int32Array]:
        """Hash probe keys, look up matches, verify key equality.

        1. Hash ``probe_keys`` via ``hasher`` (driven by ``ctx``), unless
           ``hashes`` is provided — callers that have already computed
           probe-side hashes (e.g. the partition-parallel HashJoin) pass
           them in to skip re-hashing.
        2. Probe the hash table for candidate ``(build_row, probe_row)``
           pairs via ``probe_hashes``.
        3. Gather build/probe key values at matched indices and compare
           with vectorized equality. Filter out hash-collision false
           positives.

        Args:
            build_keys: Left (build) side key columns for equality verification.
            probe_keys: Right (probe) side key columns to look up.
            num_build_rows: Number of build-side rows (for capacity estimation).
            single_match: If True, emit at most one match per probe row.
            ctx: Execution context — threads through to the hasher. The
                lookup loop itself is serial (parallel lookups happen at
                the partition granularity in HashJoin).
            hashes: Optional pre-computed probe-side hashes. When
                provided, ``ctx`` is ignored for the hashing step.

        Returns:
            ``(left_indices, right_indices)`` — verified matching row pairs.
        """
        var resolved = hashes.value().copy() if hashes else Self.hasher(
            probe_keys, ctx
        )
        var indices = self.probe_hashes(resolved, num_build_rows, single_match)
        ref build_indices = indices[0]
        ref probe_indices = indices[1]

        # Filter hash-collision false positives by key equality.
        var mask = equal(
            take(build_keys, build_indices), take(probe_keys, probe_indices)
        )
        return (
            filter_[Int32Type](build_indices, mask),
            filter_[Int32Type](probe_indices, mask),
        )

    def num_keys(self) -> Int:
        """Number of unique keys (buckets) inserted so far."""
        return self._num_buckets
