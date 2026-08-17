"""Radix partitioning layer for partition-parallel kernels.

Splits rows into independent partitions by the top bits of a precomputed hash,
so per-partition work (hash-table build/probe, grouped aggregation) runs in
parallel with zero cross-thread synchronization:

  Hash Function  ->  RadixPartitioner  ->  per-partition parallel op  ->  merge

``RadixPartitioner.map_partitions`` is the reusable driver that ties the middle
two steps together — hash once, split, run a worker per partition on its own
thread, collect the results — shared by the hash join (build + probe) and the
radix group-by path.
"""

from max.algorithm.functional import sync_parallelize

from ..arrays import Int32Array, UInt64Array
from ..buffers import Buffer
from ..dtypes import int32, uint64
from ..execution import ExecContext


comptime _MIN_PARALLEL_PARTITION_ROWS: Int = 65_536
"""Row count below which the partitioner collapses to a single worker —
dispatch + per-thread histogram overhead would dominate."""


def radix_histogram[
    BucketOf: def(Int) -> Int
](
    n: Int,
    num_buckets: Int,
    bucket_of: BucketOf,
    ctx: ExecContext,
    min_parallel_size: Int = _MIN_PARALLEL_PARTITION_ROWS,
) -> Tuple[List[Int], List[Int]]:
    """One counting/radix pass' histogram + partition-major prefix sum.

    Buckets ``n`` items (``bucket_of(i)`` gives item ``i``'s bucket in
    ``[0, num_buckets)``) with per-thread histograms, then prefix-sums them into
    per-thread write cursors — the shared front half of the radix *sort* and the
    radix *partitioner* (only the scatter payload differs, so that stays with
    each caller). Column-oriented, no atomics.

    Returns ``(write_offsets, bucket_start)``:
    - ``write_offsets[t * num_buckets + b]`` — where thread ``t`` starts writing
      bucket ``b``. Each ``(t, b)`` owns a disjoint slot, so the caller's scatter
      mutates this in place as a cursor without synchronization.
    - ``bucket_start[b]`` — global start of bucket ``b`` (``bucket_start[0]==0``,
      ``bucket_start[num_buckets]==n``); ``bucket_start[b+1]-bucket_start[b]`` is
      bucket ``b``'s size.

    The returned ``write_offsets`` is indexed by stripe, so the caller's scatter
    **must** stripe the same way: call ``ctx.stripe_workers(n, min_parallel_size)``
    with this same ``min_parallel_size``. Disagree and the scatter reads a cursor
    belonging to another stripe, which duplicates and drops rows rather than
    failing — see the coverage assertions in ``test_partition.mojo``.
    """
    var num_threads = ctx.stripe_workers(n, min_parallel_size)
    var hist = List[Int](length=num_threads * num_buckets, fill=0)

    @always_inline
    def hist_worker(t: Int, start: Int, end: Int) {mut hist, imm}:
        var base = t * num_buckets
        for i in range(start, end):
            hist[base + bucket_of(i)] += 1

    ctx.stripe(n, hist_worker, min_parallel_size)

    var write_offsets = List[Int](length=num_threads * num_buckets, fill=0)
    var bucket_start = List[Int](length=num_buckets + 1, fill=0)
    var running = 0
    for b in range(num_buckets):
        bucket_start[b] = running
        for t in range(num_threads):
            write_offsets[t * num_buckets + b] = running
            running += hist[t * num_buckets + b]
    bucket_start[num_buckets] = running
    return (write_offsets^, bucket_start^)


# ---------------------------------------------------------------------------
# Partitioner — splits rows into partitions by hash
# ---------------------------------------------------------------------------


struct Partition(Copyable, Movable):
    """A subset of rows with pre-computed hashes.

    ``row_indices`` maps each partition-local row back to the original input
    row number it came from.
    """

    var row_indices: Int32Array
    var hashes: UInt64Array

    def __init__(
        out self,
        var hashes: UInt64Array,
        var row_indices: Int32Array,
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
        return Int(self.row_indices.unsafe_get(i))


struct RadixPartitioner(Movable):
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

    var ctx: ExecContext
    """How the histogram + scatter passes execute. Held whole rather than
    reduced to a worker count, so a caller's device survives and both passes
    derive their stripe count from the same place."""

    def __init__(
        out self,
        num_bits: Int = 6,
        var ctx: ExecContext = ExecContext(),
    ):
        self.num_bits = num_bits
        self._num_partitions = 1 << num_bits
        self.ctx = ctx^

    def __init__(out self, *, copy: Self):
        self.num_bits = copy.num_bits
        self._num_partitions = copy._num_partitions
        self.ctx = copy.ctx.copy()

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

        # The histogram and the scatter both index `write_offsets` by stripe, so
        # they must stripe identically. What guarantees that is passing the same
        # `(ctx, _MIN_PARALLEL_PARTITION_ROWS)` pair to both — `stripe_workers`
        # and `stripe` agree by construction given equal arguments, and
        # `test_partition.mojo` asserts the row coverage that would break first
        # if they ever drifted.

        # 1-2. Histogram rows by top-bit partition id + prefix-sum into per-thread
        # write cursors (shared with the radix sort, cf. ``radix_histogram``).
        def bucket_of(i: Int) {imm} -> Int:
            return Int(UInt64(src.load[1](i)) >> shift)

        var offsets = radix_histogram(
            n, p, bucket_of, self.ctx, _MIN_PARALLEL_PARTITION_ROWS
        )
        var write_offsets = offsets[0].copy()
        var partition_offsets = offsets[1].copy()  # bucket_start

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
        @always_inline
        def scatter_worker(
            t: Int, start: Int, end: Int
        ) {mut write_offsets, imm}:
            var base = t * p
            for i in range(start, end):
                var h = UInt64(src.load[1](i))
                var pid = Int(h >> shift)
                var pos = write_offsets[base + pid]
                row_view.store[1](pos, Int32(i))
                hash_view.store[1](pos, h)
                write_offsets[base + pid] = pos + 1

        self.ctx.stripe(n, scatter_worker, _MIN_PARALLEL_PARTITION_ROWS)

        # 5. Freeze buffers once, then expose per-partition slices via
        # ref-counted shares (ArcPointer bumps — O(1)).
        var row_imm = row_buf^.to_immutable()
        var hash_imm = hash_buf^.to_immutable()

        var result = List[Partition](capacity=p)
        for pid in range(p):
            var off = partition_offsets[pid]
            var sz = partition_offsets[pid + 1] - off
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

    def map_partitions[
        # `Deinitable` is required since mojo 1.0.0b3.dev2026072406:
        # `Optional[R]` (used for the per-worker result slots below) only
        # conditionally conforms to it, so an unconstrained `R` makes the
        # slot list linear and unable to be dropped.
        R: Copyable & Deinitable,
        Op: def(Int, Int32Array, UInt64Array) raises -> R,
    ](self, var hashes: UInt64Array, op: Op) raises -> List[R]:
        """Run ``op`` on every partition in parallel and collect the results.

        Partitions ``hashes`` (one radix pass), then dispatches one worker per
        partition via ``sync_parallelize`` — each worker calls
        ``op(partition_index, row_indices, hashes)`` and produces one ``R``. The
        index lets an op correlate its partition with a paired structure (e.g.
        the probe side pairs partition ``i`` with build-side table ``i``, since
        both were split on the same radix bits). Partitions are independent, so
        there are no locks: workers write into distinct pre-sized slots, and the
        results are *moved* out in partition order (never copied — ``R`` may own
        a `SwissHashTable`).

        This is the shared skeleton behind every partition-parallel kernel: the
        hash join builds a table per partition (``R`` = table + keys + rows),
        probes per partition (``R`` = index pairs), and the radix group-by
        aggregates per partition (``R`` = first-rows + aggregate column).
        """
        var partitions = self.partition(hashes^)
        var p = len(partitions)
        var slots = List[Optional[R]](length=p, fill=None)

        # One slot per worker, like the result slots above: a single shared
        # `Optional[Error]` would be written by every failing thread at once.
        var worker_errs = List[Optional[Error]](length=p, fill=None)

        def worker(i: Int) {mut worker_errs, mut slots, imm}:
            # `sync_parallelize`'s value form takes a non-raising worker. The
            # body still unwinds at its first error; the other workers cannot be
            # cancelled, so their errors are collected and raised after the join.
            try:
                slots[i] = op(
                    i,
                    partitions[i].row_indices.copy(),
                    partitions[i].hashes.copy(),
                )

            except e:
                worker_errs[i] = e

        sync_parallelize(worker, p)
        for err in worker_errs:
            if err:
                raise err.value()

        var out = List[R](capacity=p)
        for i in range(p):
            out.append(slots[i].take())
        return out^
