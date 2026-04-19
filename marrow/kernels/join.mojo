"""Join kernels for Arrow StructArrays.

Public API
----------
``hash_join``   — equijoin two StructArrays on positional key columns.
``HashJoin``    — hash join using SwissHashTable; reusable across morsels.

Traits
------
``Join``        — abstract join algorithm (build + probe → IndexPairs).

Internal types
--------------
``IndexPairs``  — (left_indices, right_indices) result of a probe phase.

Supported join kinds (pass the JOIN_* constants from relations.mojo):
  JOIN_INNER  — only matched rows
  JOIN_LEFT   — all left + matched right (NULLs for non-matches)
  JOIN_RIGHT  — all right + matched left (NULLs for non-matches)
  JOIN_FULL   — all rows from both sides
  JOIN_SEMI   — left rows with at least one match (left columns only)
  JOIN_ANTI   — left rows with no match (left columns only)

Supported strictness:
  JOIN_ALL    — default: return all matching pairs (Cartesian for multi-match)
  JOIN_ANY    — return at most one matching right row per left row

Future join algorithms (implement the Join trait):
  RadixHashJoin   — partitioned hash join (SwissHashTable + RadixPartitioner)
  SortMergeJoin   — sort both sides, two-pointer merge (no hash table)

Performance / Optimization Notes
--------------------------------
At 10M × 10M INNER join on Apple Silicon (10-core P, parallel path), the
time budget looks roughly like this (from ``sample``-based profiling):

  20%  SwissHashTable.probe       — probe-loop CSR walk
  25%  take SIMD gather           — random gather of output columns
  14%  SwissHashTable.build_hashes — insertion + slot claim
  11%  nested semaphore waits     — tcmalloc spinlock on take's output buf
   8%  take(AnyArray) dispatch    — runtime type dispatch
   7%  RadixPartitioner scatter   — histogram + scatter passes
   2%  tcmalloc::PageHeap::New    — allocator
   ... dispatch + allocator small ops

The hot paths are memory-latency / bandwidth bound; further single-digit
percent wins are available but require structural work.  Ordered by
expected payoff for the next round of optimization:

1. **Fused equality verification**
   Current ``SwissHashTable.probe`` does
   ``take(build_keys, bi) + take(probe_keys, pi) + equal(...)`` which
   allocates three intermediate Arrays (two gathered keys + one mask)
   then ``filter_`` rebuilds two Int32 arrays from the candidates.
   A fused kernel could walk ``(bi[i], pi[i])`` once, load
   ``build_keys[bi[i]]`` and ``probe_keys[pi[i]]`` into registers,
   compare, and emit verified pairs into preallocated Int32 buffers —
   no intermediate gathered-key arrays, no bitmap.  Expected saving:
   3–6 ms at 10M (hot inside the per-partition probe worker).

2. **Buffer reuse / arena allocator for ``take()``**
   Every ``take()`` inside a partition worker allocates a fresh
   ``Buffer`` via tcmalloc.  With 64 partitions × (probe-key gather +
   equality takes) = ~200 heap allocations per join, contending on
   tcmalloc's page-heap spinlock.  The profile shows ~11% of worker
   time in nested semaphore waits that are largely this contention.
   A thread-local arena (bump allocator reset between joins) or a
   small-object pool inside ``ExecutionContext`` would eliminate it.
   Expected saving: ~5–8 ms at 10M.

3. **Software Write-Combine Buffers (SWWCB) in ``RadixPartitioner``**
   Classic PRO radix-join trick (Balkesen et al.): instead of scattering
   one row at a time (each write touches a different cache line across
   64 partitions), each worker keeps a small per-partition staging
   buffer (cache-line-sized) and flushes when full.  Reduces cross-core
   cache-line ping-pong and TLB pressure.  Expected saving at 10M:
   1–3 ms.  Bigger win at 100M+ rows where the scatter dominates.

4. **Fused probe + output materialization**
   Currently the probe emits ``IndexPairs`` (two global row-index arrays),
   which are concatenated across partitions, then ``_assemble`` re-scans
   them to gather the output columns.  A fused pass would emit output
   rows directly from the per-partition probe worker into a preallocated
   output StructArray — skipping ``_concat_int32`` and the per-column
   ``take()`` in ``_assemble`` entirely.  Largest refactor of the four;
   potentially eliminates the 25% take-gather cost.  Expected saving:
   10–15 ms at 10M, but needs careful output-row-count handling for
   LEFT / FULL / SEMI / ANTI (where matched count isn't known up front).

5. **Deeper prefetch pipeline in probe / build**
   ``_PIPE_DEPTH = 16`` in ``SwissHashTable`` — try 24 or 32 for larger
   build sides where DRAM latency (~100 ns) exceeds the 16-iteration
   compute window.

6. **Adaptive radix bits**
   ``_DEFAULT_RADIX_BITS`` is fixed at 6 (64 partitions).  Sweep at 10M
   showed 32/64/128 all within 1 ms — but at 1M the parallel path barely
   beats serial because of dispatch overhead.  An adaptive choice
   (log2(build_rows) − 10) would auto-tune across scales.

7. **Parallel ``_emit_unmatched`` for LEFT / FULL / SEMI / ANTI**
   Currently serial.  Doesn't affect INNER-join benchmarks but needed
   for full parallelism on outer-join workloads.

See ``docs/joins-design.md`` for the high-level architecture and the
``Phase 1b`` performance table.
"""

from std.algorithm.functional import sync_parallelize
from std.gpu.host import DeviceContext
from std.sys.info import num_physical_cores

from ..arrays import PrimitiveArray, AnyArray, StructArray
from ..buffers import Buffer
from ..builders import PrimitiveBuilder
from ..dtypes import (
    AnyDataType,
    Field,
    int32,
    uint64,
    UInt64Type,
    bool_ as bool_dt,
    struct_,
    null,
)
from .boolean import and_
from .compare import equal
from .execution import ExecutionContext
from .filter import take, filter_
from .hashtable import SwissHashTable, RadixPartitioner
from .hashing import rapidhash
from ..expr.relations import (
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_FULL,
    JOIN_SEMI,
    JOIN_ANTI,
    JOIN_MARK,
    JOIN_ALL,
    JOIN_ANY,
)


comptime IndexPairs = Tuple[
    PrimitiveArray[Int32Type], PrimitiveArray[Int32Type]
]
"""Parallel (left_indices, right_indices) arrays from the probe phase."""


def _concat_int32(
    var parts: List[Optional[PrimitiveArray[Int32Type]]],
) raises -> PrimitiveArray[Int32Type]:
    """Concatenate a list of Int32 index arrays into one.

    Used by the parallel probe path to merge per-partition pair arrays
    into a single ``IndexPairs``. Direct buffer-level memcpy rather than
    going through the generic ``concat(AnyArray)`` path — the per-
    partition pair arrays are always valid dense Int32 buffers with
    ``nulls == 0``, so we can skip bitmap and type-dispatch overhead.
    """
    var total = 0
    for ref p in parts:
        if p:
            total += len(p.value())
    if total == 0:
        var empty = PrimitiveBuilder[Int32Type](capacity=0)
        return empty.finish()

    var out_buf = Buffer.alloc_uninit[int32.native](total)
    var out_view = out_buf.view[int32.native](0, total)
    var write = 0
    for ref p in parts:
        if not p:
            continue
        ref arr = p.value()
        var n = len(arr)
        if n == 0:
            continue
        var src = arr.values()
        out_view.slice(write, n).copy_from(src, n)
        write += n

    return PrimitiveArray[Int32Type](
        length=total,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=out_buf^.to_immutable(),
    )


# ---------------------------------------------------------------------------
# Join trait — abstract join algorithm
# ---------------------------------------------------------------------------


trait Join(Movable):
    """Abstract join algorithm: build from left side, probe with right side.

    All join algorithms (hash, radix-hash, sort-merge) implement this.
    The trait is for static dispatch — operators use concrete types directly.
    Runtime algorithm selection uses if/elif at the call site, not type erasure.
    """

    def build(mut self, data: StructArray, key_indices: List[Int]) raises:
        """Build the join index from the left (build) side."""
        ...

    def probe(
        self,
        data: StructArray,
        key_indices: List[Int],
        kind: UInt8,
        strictness: UInt8,
    ) raises -> StructArray:
        """Probe with right (probe) side data.  Return assembled output."""
        ...

    def build_dtype(self) -> AnyDataType:
        """DataType of the build side (for output schema construction)."""
        ...

    def num_left_rows(self) -> Int:
        """Number of build-side rows."""
        ...


# ---------------------------------------------------------------------------
# HashJoin — hash join using SwissHashTable
# ---------------------------------------------------------------------------


comptime _PARALLEL_THRESHOLD = 100_000
"""Below this build-side row count the parallel path falls back to serial —
partitioning overhead dominates below ~100k rows on typical inputs."""

# TODO(partitioned-op): the divide/map/gather logic in `build_parallel`
# and `probe_parallel` is duplicated (hash → partition → per-partition
# parallel work → merge). A reusable `PartitionedOp[T]` trait with a
# `partition_apply` driver would collapse both call sites and be reusable
# for future partition-parallel kernels (groupby, radix sort). Deferred
# until Mojo's generics make variable-output-shape abstractions
# practical. See docs/joins-design.md → "Known limits / future work".

comptime _DEFAULT_RADIX_BITS = 6
"""Default radix fanout for ``RadixPartitioner`` (64 partitions).

Sweep at 10M INNER join on Apple Silicon shows 32 / 64 / 128 partitions
all land within ~1 ms of each other — the hash table per partition
already exceeds L2 at these sizes, so reducing fanout further doesn't
help cache locality but does reduce sync_parallelize dispatch overhead.
64 is the default; fanout is a runtime parameter on ``RadixPartitioner``
and can be tuned per workload."""


struct HashJoin[
    hasher: def(
        StructArray, ExecutionContext
    ) thin raises -> PrimitiveArray[UInt64Type] = rapidhash
](Join):
    """Hash join using SwissHashTable.

    Build phase: hash left-side key columns, insert rows into hash table.
    Probe phase: hash right-side key columns, look up in hash table,
    emit index pairs, verify key equality (filter hash collisions).

    Supports two execution paths, chosen by ``num_threads``:

    * **Serial** — a single ``SwissHashTable`` over the full build side.
      Used when ``num_threads == 1`` or the build side is below
      ``_PARALLEL_THRESHOLD``.
    * **Partition-parallel** — rows are split by the top bits of their
      hash into ``2^radix_bits`` independent ``SwissHashTable`` instances,
      built and probed concurrently via ``sync_parallelize``. No atomics,
      no locks: each partition is fully independent.

    The public ``build`` / ``probe`` entry points are thin dispatchers over
    ``build_serial`` / ``build_parallel`` and ``probe_serial`` /
    ``probe_parallel``. The serial implementation is unchanged from the
    pre-parallel version; the parallel path reuses the same
    ``SwissHashTable`` primitive per partition.
    """

    # Global state (shared by both paths)
    var _num_threads: Int
    var _left_key_indices: List[Int]
    var _left_dtype: AnyDataType
    var _left_data: Optional[StructArray]
    var _left_rows: Int

    # Serial path state
    var _table: SwissHashTable[Self.hasher]

    # Parallel path state (populated by build_parallel)
    var _tables: List[SwissHashTable[Self.hasher]]
    """One SwissHashTable per partition (parallel path only)."""
    var _left_partition_keys: List[StructArray]
    """Per-partition build-side keys, used for equality verification."""
    var _left_partition_rows: List[PrimitiveArray[Int32Type]]
    """Per-partition original row indices — maps partition-local row
    numbers back to the original build-side row index after probe."""
    var _radix_bits: Int

    def __init__(out self, num_threads: Int = 1):
        """Create a HashJoin.

        Args:
            num_threads: ``1`` for serial (default), ``>1`` for partition-
                parallel build + probe, ``0`` means auto-pick via
                ``num_physical_cores()``.
        """
        var nt = num_threads
        if nt == 0:
            nt = num_physical_cores()
        self._num_threads = nt
        self._left_key_indices = List[Int]()
        self._left_dtype = null
        self._left_data = None
        self._left_rows = 0
        self._table = SwissHashTable[Self.hasher]()
        self._tables = List[SwissHashTable[Self.hasher]]()
        self._left_partition_keys = List[StructArray]()
        self._left_partition_rows = List[PrimitiveArray[Int32Type]]()
        self._radix_bits = _DEFAULT_RADIX_BITS

    # ------------------------------------------------------------------
    # Public dispatchers — route to serial or parallel implementations.
    # ------------------------------------------------------------------

    def build(
        mut self, left: StructArray, left_key_indices: List[Int]
    ) raises:
        if self._num_threads <= 1 or left.length < _PARALLEL_THRESHOLD:
            self.build_serial(left, left_key_indices)
        else:
            self.build_parallel(left, left_key_indices)

    def probe(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: UInt8 = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> StructArray:
        if self._num_threads <= 1 or self._left_rows < _PARALLEL_THRESHOLD:
            return self.probe_serial(
                right, right_key_indices, kind, strictness
            )
        return self.probe_parallel(
            right, right_key_indices, kind, strictness
        )

    # ------------------------------------------------------------------
    # Serial path — one SwissHashTable over the whole build side.
    # ------------------------------------------------------------------

    def build_serial(
        mut self, left: StructArray, left_key_indices: List[Int]
    ) raises:
        self._left_dtype = left.dtype.copy()
        self._left_rows = left.length
        self._left_data = left.copy()
        self._left_key_indices = left_key_indices.copy()
        var ctx = ExecutionContext.parallel(self._num_threads)
        self._table.build(left.select(left_key_indices), ctx)

    def probe_serial(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: UInt8,
        strictness: UInt8,
    ) raises -> StructArray:
        var left_keys = self._left_data.value().select(self._left_key_indices)
        var right_keys = right.select(right_key_indices)
        var pairs = self._table.probe(
            left_keys,
            right_keys,
            self._left_rows,
            single_match=strictness == JOIN_ANY,
            ctx=ExecutionContext.parallel(self._num_threads),
        )
        var verified = (pairs[0].copy(), pairs[1].copy())
        var final = self._emit_unmatched(
            verified^, len(right), kind, strictness
        )
        return self._assemble(right, final, kind)

    # ------------------------------------------------------------------
    # Parallel path — radix-partitioned, one table per partition.
    # ------------------------------------------------------------------

    def build_parallel(
        mut self, left: StructArray, left_key_indices: List[Int]
    ) raises:
        """Radix-partitioned build.

        1. Hash the full build side once (parallel SIMD over key columns).
        2. Partition rows by the top ``_radix_bits`` of their hash.
        3. For each partition *in parallel*: gather the partition's keys
           via ``take``, build an independent ``SwissHashTable`` against
           the pre-computed hashes, and store per-partition state back
           on ``self``. No cross-partition synchronization: each worker
           writes to a distinct index slot.
        """
        self._left_dtype = left.dtype.copy()
        self._left_rows = left.length
        self._left_data = left.copy()
        self._left_key_indices = left_key_indices.copy()

        var left_keys = left.select(left_key_indices)

        # 1. Hash once, in parallel.
        var hashes = Self.hasher(
            left_keys, ExecutionContext.parallel(self._num_threads)
        )

        # 2. Partition.
        var partitioner = RadixPartitioner(
            num_bits=self._radix_bits, num_threads=self._num_threads
        )
        var partitions = partitioner.partition(hashes^)
        var p = len(partitions)

        # 3. Allocate per-partition slots — pre-sized so workers can
        # assign by index without racing on list growth.
        var tables = List[SwissHashTable[Self.hasher]](capacity=p)
        for _ in range(p):
            tables.append(SwissHashTable[Self.hasher]())
        var part_keys = List[Optional[StructArray]](length=p, fill=None)
        var part_rows = List[Optional[PrimitiveArray[Int32Type]]](
            length=p, fill=None
        )

        # 4. Parallel per-partition work: gather keys + build table.
        @parameter
        def build_worker(i: Int) raises:
            var rows = partitions[i].row_indices.value().copy()
            var k = take(left_keys, rows)
            tables[i].build_hashes(partitions[i].hashes.copy())
            part_keys[i] = k^
            part_rows[i] = rows^

        sync_parallelize[build_worker](p)

        # 5. Unwrap Optionals into dense lists (order preserved).
        var keys_out = List[StructArray](capacity=p)
        var rows_out = List[PrimitiveArray[Int32Type]](capacity=p)
        for i in range(p):
            keys_out.append(part_keys[i].value().copy())
            rows_out.append(part_rows[i].value().copy())

        self._tables = tables^
        self._left_partition_keys = keys_out^
        self._left_partition_rows = rows_out^

    def probe_parallel(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: UInt8,
        strictness: UInt8,
    ) raises -> StructArray:
        """Radix-partitioned probe.

        1. Hash the full probe side once in parallel.
        2. Partition probe rows by the same radix bits used at build time.
        3. For each partition: gather probe-side keys, look up in the
           matching partition's hash table, remap partition-local row
           indices to original row indices. Partitions probe concurrently.
        4. Concatenate per-partition index pairs, then run the shared
           ``_emit_unmatched`` + ``_assemble`` steps.
        """
        var right_keys = right.select(right_key_indices)
        var right_n = len(right)

        # 1. Hash probe side in parallel.
        var probe_hashes = Self.hasher(
            right_keys, ExecutionContext.parallel(self._num_threads)
        )

        # 2. Partition probe rows.
        var partitioner = RadixPartitioner(
            num_bits=self._radix_bits, num_threads=self._num_threads
        )
        var probe_partitions = partitioner.partition(probe_hashes^)
        var p = len(probe_partitions)

        # 3. Parallel probe — each worker gathers its probe keys, looks
        # them up, and remaps partition-local row indices to global row
        # numbering. Pre-sized Optional slots let workers assign by
        # partition index without racing on list growth.
        var part_build_idx = List[Optional[PrimitiveArray[Int32Type]]](
            length=p, fill=None
        )
        var part_probe_idx = List[Optional[PrimitiveArray[Int32Type]]](
            length=p, fill=None
        )
        var single = strictness == JOIN_ANY

        @parameter
        def probe_worker(i: Int) raises:
            var rows = probe_partitions[i].row_indices.value().copy()
            var probe_keys_i = take(right_keys, rows)
            var pairs = self._tables[i].probe(
                self._left_partition_keys[i],
                probe_keys_i,
                len(self._left_partition_keys[i]),
                single_match=single,
                hashes=probe_partitions[i].hashes.copy(),
            )
            # Remap partition-local indices → original row indices.
            part_build_idx[i] = take(
                self._left_partition_rows[i], pairs[0]
            )
            part_probe_idx[i] = take(rows, pairs[1])

        sync_parallelize[probe_worker](p)

        # 5. Concat per-partition pairs into a single IndexPairs.
        var combined_build = _concat_int32(part_build_idx^)
        var combined_probe = _concat_int32(part_probe_idx^)
        var verified = (combined_build^, combined_probe^)

        var final = self._emit_unmatched(
            verified^, right_n, kind, strictness
        )
        return self._assemble(right, final, kind)

    def _emit_unmatched(
        self,
        var pairs: IndexPairs,
        right_rows: Int,
        kind: UInt8,
        strictness: UInt8,
    ) raises -> IndexPairs:
        """Phase 3: add unmatched rows for outer/semi/anti joins.

        Scans the verified pairs to determine which build/probe rows
        were matched, then appends unmatched rows as needed.
        INNER: returns pairs unchanged.
        SEMI: emits matched build rows only.
        ANTI: emits unmatched build rows only.
        LEFT/RIGHT/FULL: appends unmatched rows from the appropriate side.
        """
        if kind == JOIN_INNER:
            return pairs^

        # Compute which build/probe rows appear in the verified pairs.
        var matched_build = List[Bool](length=self._left_rows, fill=False)
        var matched_probe = List[Bool](length=right_rows, fill=False)
        var n_pairs = len(pairs[0])
        for i in range(n_pairs):
            var lid = Int(pairs[0].unsafe_get(i))
            var rid = Int(pairs[1].unsafe_get(i))
            if lid >= 0:
                matched_build[lid] = True
            if rid >= 0:
                matched_probe[rid] = True

        if kind == JOIN_SEMI:
            var lb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            var rb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            for i in range(self._left_rows):
                if matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return (lb.finish(), rb.finish())

        if kind == JOIN_ANTI:
            var lb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            var rb = PrimitiveBuilder[Int32Type](capacity=self._left_rows)
            for i in range(self._left_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return (lb.finish(), rb.finish())

        # LEFT / RIGHT / FULL: matched pairs + unmatched rows.
        var lb = PrimitiveBuilder[Int32Type](capacity=n_pairs + self._left_rows)
        var rb = PrimitiveBuilder[Int32Type](capacity=n_pairs + right_rows)
        for i in range(n_pairs):
            lb.append(pairs[0].unsafe_get(i))
            rb.append(pairs[1].unsafe_get(i))
        if kind == JOIN_LEFT or kind == JOIN_FULL:
            for i in range(self._left_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
        if kind == JOIN_RIGHT or kind == JOIN_FULL:
            for i in range(right_rows):
                if not matched_probe[i]:
                    lb.append_null()
                    rb.append(Scalar[int32.native](i))
        return (lb.finish(), rb.finish())

    def build_dtype(self) -> AnyDataType:
        return self._left_dtype.copy()

    def num_left_rows(self) -> Int:
        return self._left_rows

    def output_dtype(self, probe: StructArray, kind: UInt8) -> AnyDataType:
        """Build the output struct DataType for a join result."""
        var fields = List[Field]()
        for ref f in self._left_dtype.as_struct_type().fields:
            fields.append(f.copy())

        if kind != JOIN_SEMI and kind != JOIN_ANTI:
            var left_names = List[String]()
            for ref f in self._left_dtype.as_struct_type().fields:
                left_names.append(f.name)
            for ref f in probe.dtype.as_struct_type().fields:
                var name = f.name
                var collides = False
                for ref ln in left_names:
                    if ln == name:
                        collides = True
                        break
                if collides:
                    name = name + "_right"
                fields.append(Field(name, f.dtype.copy()))

        return struct_(fields^)

    def _assemble(
        self, right: StructArray, pairs: IndexPairs, kind: UInt8
    ) raises -> StructArray:
        """Gather left + right columns using index pairs.

        After ``sync_parallelize`` in ``probe_parallel`` has finished
        there's no outer parallel region, so each per-column ``take``
        can safely fan its SIMD gather loop across workers internally.
        We pass an ``ExecutionContext.parallel(num_threads)`` through,
        and ``take`` decides per-column whether it's big enough to
        stripe (its own grain threshold inside ``apply``).
        """
        ref left = self._left_data.value()
        var out_cols = List[AnyArray]()
        var ctx = ExecutionContext.parallel(self._num_threads)

        for c in range(len(left.children)):
            out_cols.append(take(left.children[c].copy(), pairs[0], ctx))

        if kind != JOIN_SEMI and kind != JOIN_ANTI and kind != JOIN_MARK:
            for c in range(len(right.children)):
                out_cols.append(take(right.children[c].copy(), pairs[1], ctx))

        var out_length = out_cols[0].length() if len(out_cols) > 0 else 0
        return StructArray(
            dtype=self.output_dtype(right, kind),
            length=out_length,
            nulls=0,
            offset=0,
            bitmap=None,
            children=out_cols^,
        )


# ---------------------------------------------------------------------------
# Future join algorithms (stubs — implement the Join trait)
# ---------------------------------------------------------------------------

# struct RadixHashJoin(Join):
#     """Radix-partitioned hash join.
#
#     Partitions both sides by hash prefix bits using RadixPartitioner,
#     then runs a standard hash join (SwissHashTable) per partition.
#     Enables partition-parallel execution and better cache locality.
#
#     Uses the SAME SwissHashTable as HashJoin — only the Partitioner differs.
#     """
#     var _partitioner: RadixPartitioner    # from hash_table.mojo
#     var _tables: List[SwissHashTable]      # one per partition
#     var _build_dtype: DataType
#     var _left_data: Optional[StructArray]
#     var _num_rows: Int


# struct SortMergeJoin(Join):
#     """Sort-merge join.
#
#     Sorts both sides by key columns, then two-pointer linear merge.
#     O(N log N + M log M) time, zero hash table memory.
#
#     Does NOT use HashTable — proves the Join trait is not hash-specific.
#     """
#     var _sort_order: Optional[PrimitiveArray[Int32Type]]
#     var _sorted_keys: Optional[StructArray]
#     var _build_dtype: DataType
#     var _left_data: Optional[StructArray]
#     var _num_rows: Int


# ---------------------------------------------------------------------------
# hash_join — top-level public API
# ---------------------------------------------------------------------------


def hash_join(
    left: StructArray,
    right: StructArray,
    left_on: List[Int],
    right_on: List[Int],
    kind: UInt8 = JOIN_INNER,
    strictness: UInt8 = JOIN_ALL,
    ctx: ExecutionContext = ExecutionContext.serial(),
    num_threads: Int = 0,
) raises -> StructArray:
    """Equijoin two StructArrays on positional key column indices.

    The left side is always the build side; the right side is the probe side.

    Args:
        left: Build-side data as a StructArray (one child per column).
        right: Probe-side data as a StructArray (one child per column).
        left_on: Positional column indices in ``left`` to join on.
        right_on: Positional column indices in ``right`` to join on.
        kind: Join direction (JOIN_INNER, JOIN_LEFT, JOIN_RIGHT, JOIN_FULL,
              JOIN_SEMI, JOIN_ANTI).
        strictness: JOIN_ALL (default) or JOIN_ANY.
        ctx: Optional GPU device context (future GPU acceleration hook).
        num_threads: Worker count for the partition-parallel path.
            ``0`` (default) auto-picks ``num_physical_cores()``; ``1`` forces
            the serial single-table path; ``>1`` runs radix-partitioned
            parallel build + probe. Builds smaller than
            ``_PARALLEL_THRESHOLD`` always fall back to serial regardless.

    Returns:
        Output StructArray:
        * INNER/LEFT/RIGHT/FULL: left columns + right columns.
        * SEMI/ANTI: left columns only.
    """
    if len(left_on) != len(right_on):
        raise Error("hash_join: len(left_on) != len(right_on)")

    var join = HashJoin(num_threads=num_threads)
    join.build(left, left_on)
    return join.probe(right, right_on, kind, strictness)
