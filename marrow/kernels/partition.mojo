# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Radix partitioning layer for partition-parallel kernels.

Splits rows into independent partitions by the top bits of a precomputed hash,
so per-partition work (hash-table build/probe, grouped aggregation) runs in
parallel with zero cross-thread synchronization:

  Hash Function  ->  RadixPartitioner  ->  per-partition parallel op  ->  collect

``RadixPartitioner.map_partitions`` is the reusable driver that ties the middle
two steps together — hash once, split, run a worker per partition on its own
thread, collect the results. Its one user is ``HashIndex`` (``hashtable.mojo``),
whose radix placement every hash consumer reaches: the group-by, the join,
``is_in`` and ``count_distinct``.
"""

from ..utils.stats import count
from ..arrays import Int32Array, StructArray, UInt64Array
from ..buffers import Buffer
from ..dtypes import int32, uint64
from ..execution import ExecContext
from ..views import BufferView
from ..utils import KeyHash
from .filter import TakeKernel
from .hashing import HashKernel


comptime _MIN_PARALLEL_PARTITION_ROWS: Int = 65_536
"""Row count below which the partitioner collapses to a single worker —
dispatch + per-thread histogram overhead would dominate."""


struct RadixHistogram[mut: Bool, //, origin: Origin[mut=mut]](Movable):
    """One counting pass of a radix scatter over 64-bit keys: how many keys
    hold each value of one bit field, prefix-summed into a write cursor per
    (stripe, digit) — the shared front half of the radix *sort* and the radix
    *partitioner*. Column-oriented, no atomics.

    A key's digit is its ``bits`` bits starting at bit ``shift``; there are
    ``2**bits`` digits. ``scatter`` then hands every key its destination,
    digit by digit and, within a digit, in key order.

    The cursors are indexed by stripe, so the count and the scatter must
    stripe the keys identically or the scatter duplicates and drops keys
    rather than failing. The histogram therefore keeps the ``ctx`` and
    ``min_parallel_size`` it counted with and drives the scatter through the
    same ``ctx.stripe``.
    """

    var _keys: BufferView[DType.uint64, Self.origin]
    var _shift: UInt64
    var _mask: UInt64
    var _cursors: List[Int]
    """Where stripe ``t`` starts writing digit ``d``, at
    ``t * num_digits + d``; each slot is disjoint."""
    var _starts: List[Int]
    """Where digit ``d`` starts, ``num_digits + 1`` entries ending at the key
    count."""
    var _ctx: ExecContext
    var _min_parallel_size: Int

    def __init__(
        out self,
        keys: BufferView[DType.uint64, Self.origin],
        shift: Int,
        bits: Int,
        ctx: ExecContext,
        min_parallel_size: Int = _MIN_PARALLEL_PARTITION_ROWS,
    ):
        self._keys = keys
        self._shift = UInt64(shift)
        self._mask = UInt64((1 << bits) - 1)
        self._ctx = ctx.copy()
        self._min_parallel_size = min_parallel_size

        var n = len(keys)
        var num_digits = 1 << bits
        var num_threads = ctx.stripe_workers(n, min_parallel_size)
        var hist = List[Int](length=num_threads * num_digits, fill=0)
        var key_shift = self._shift
        var key_mask = self._mask

        @always_inline
        def hist_worker(t: Int, start: Int, end: Int) {mut hist, imm}:
            var base = t * num_digits
            for i in range(start, end):
                hist[
                    base + Int((keys.unsafe_get(i) >> key_shift) & key_mask)
                ] += 1

        ctx.stripe(n, hist_worker, min_parallel_size)

        self._cursors = List[Int](length=num_threads * num_digits, fill=0)
        self._starts = List[Int](length=num_digits + 1, fill=0)
        var running = 0
        for d in range(num_digits):
            self._starts[d] = running
            for t in range(num_threads):
                self._cursors[t * num_digits + d] = running
                running += hist[t * num_digits + d]
        self._starts[num_digits] = running

    def num_digits(self) -> Int:
        """How many digit values there are: ``2**bits``."""
        return len(self._starts) - 1

    def digit_start(self, d: Int) -> Int:
        """Where the keys with digit ``d`` start in the scattered order."""
        return self._starts[d]

    def digit_count(self, d: Int) -> Int:
        """How many keys have digit ``d``."""
        return self._starts[d + 1] - self._starts[d]

    def distinct_digits(self) -> Int:
        """How many digit values occur among the keys."""
        var count = 0
        for d in range(self.num_digits()):
            if self.digit_count(d) > 0:
                count += 1
        return count

    def scatter[Place: def(Int, UInt64, Int) -> None](self, place: Place):
        """Call ``place(i, key, pos)`` once per key: ``key`` is key ``i`` and
        ``pos`` its destination."""
        var cursors = self._cursors.copy()
        var num_digits = self.num_digits()
        var keys = self._keys
        var key_shift = self._shift
        var key_mask = self._mask

        @always_inline
        def scatter_worker(t: Int, start: Int, end: Int) {mut cursors, imm}:
            var base = t * num_digits
            for i in range(start, end):
                var key = keys.unsafe_get(i)
                var slot = base + Int((key >> key_shift) & key_mask)
                var pos = cursors[slot]
                place(i, key, pos)
                cursors[slot] = pos + 1

        self._ctx.stripe(len(keys), scatter_worker, self._min_parallel_size)


# ---------------------------------------------------------------------------
# Partitioner — splits rows into partitions by hash
# ---------------------------------------------------------------------------


@fieldwise_init
struct Partition(Copyable, Movable):
    """A subset of rows with pre-computed hashes.

    ``row_indices`` maps each partition-local row back to the original input
    row number it came from.
    """

    var hashes: UInt64Array
    var row_indices: Int32Array


struct RadixPartitioner(Movable):
    """Partition rows by the top ``num_bits`` of their hash.

    The partitioner is the key enabler of partition-parallel joins: each
    partition is independent, so per-partition hash-table builds and probes
    run in parallel with zero cross-thread synchronization.

    Partition count is ``2^num_bits``; the caller picks it
    (``hashtable.mojo``'s ``_RADIX_BITS``).

    Top bits are used for partitioning (``h >> (64 - num_bits)``) while the
    ``SwissHashTable`` probes with low bits (``h & mask``). This split
    keeps the partition router and the per-table probe order independent,
    avoiding double-hashing.

    The partitioning pass is striped too: a per-stripe histogram, a prefix
    sum, and a scatter each stripe writes into its own slots (``partition``).
    """

    var num_bits: Int
    """Number of top hash bits consumed by partition routing."""

    var ctx: ExecContext
    """How the histogram + scatter passes execute. Held whole rather than
    reduced to a worker count, so a caller's device survives and both passes
    derive their stripe count from the same place."""

    def __init__(
        out self,
        num_bits: Int,
        var ctx: ExecContext = ExecContext(),
    ):
        self.num_bits = num_bits
        self.ctx = ctx^

    def num_partitions(self) -> Int:
        return 1 << self.num_bits

    @staticmethod
    def bits_for(workers: Int) -> Int:
        """How many hash bits split a parallel merge across ``workers``:
        enough for about four partitions per worker, so a slow partition holds
        up little, and at least 16 and at most 512 partitions."""
        var bits = 4
        while (1 << bits) < 4 * workers and bits < 9:
            bits += 1
        return bits

    def repartition(
        self, batches: List[StructArray], key_indices: List[Int]
    ) raises -> List[List[StructArray]]:
        """Every row of ``batches`` in one of the partitions, by the hash of
        its key columns: partition ``p``'s rows, batch by batch, are
        ``result[p]``. Equal keys land in one partition, whichever batch
        holds them, and a batch's rows keep their order within a partition.

        Each batch is gathered once, partition-major, so its partitions are
        slices of one copy. Batches are spread over this context's workers,
        each partitioned on its own thread.
        """
        var num_batches = len(batches)
        var parts = List[List[StructArray]](capacity=num_batches)
        for _ in range(num_batches):
            parts.append(List[StructArray]())

        def split(wid: Int, b: Int) raises {mut parts, imm}:
            ref batch = batches[b]
            var serial = ExecContext.serial()
            var hashes = HashKernel[KeyHash].apply(
                batch.select(key_indices), serial
            )
            var routed = RadixPartitioner(
                self.num_bits, serial.copy()
            ).partition(hashes^)
            # Every partition's rows slice one buffer, partition after
            # partition: one gather in that order gives every partition as a
            # slice of one copy.
            var order = Int32Array(
                length=len(batch),
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=routed[0].row_indices.buffer.copy(),
            )
            var gathered = TakeKernel.apply(batch, order, serial)
            var out = List[StructArray](capacity=self.num_partitions())
            var start = 0
            for p in range(self.num_partitions()):
                var size = len(routed[p].row_indices)
                out.append(gathered.slice(start, size))
                start += size
            parts[b] = out^

        self.ctx.fan_out(num_batches, split, num_batches)
        var result = List[List[StructArray]](capacity=self.num_partitions())
        for p in range(self.num_partitions()):
            var partition = List[StructArray](capacity=num_batches)
            for b in range(num_batches):
                partition.append(parts[b][p].copy())
            result.append(partition^)
        return result^

    @staticmethod
    @always_inline
    def partition_of(h: UInt64, num_bits: Int) -> Int:
        """The partition ``h`` routes to among ``2^num_bits``: its top
        ``num_bits`` — the rule ``partition`` applies to every row."""
        return Int(h >> UInt64(64 - num_bits))

    def partition(self, var hashes: UInt64Array) raises -> List[Partition]:
        """Split ``hashes`` into ``2^num_bits`` partitions by top bits.

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
        count["partition.rows"](n)
        var p = 1 << self.num_bits
        var bits = self.num_bits
        var src = hashes.values()

        # 1-2. Histogram rows by their top `bits` hash bits.
        var histogram = RadixHistogram(
            src, 64 - bits, bits, self.ctx, _MIN_PARALLEL_PARTITION_ROWS
        )

        # 3. Allocate the two flat buffers (N rows total each).
        var row_buf = Buffer.alloc_uninit[int32.native](n)
        var hash_buf = Buffer.alloc_uninit[uint64.native](n)
        var row_view = row_buf.view[int32.native](0, n)
        var hash_view = hash_buf.view[uint64.native](0, n)

        # 4. Parallel scatter of every row and its hash into its partition.
        @always_inline
        def place(i: Int, h: UInt64, pos: Int) {imm}:
            row_view.store[1](pos, Int32(i))
            hash_view.store[1](pos, h)

        histogram.scatter(place)

        # 5. Freeze buffers once, then expose per-partition slices via
        # ref-counted shares (ArcPointer bumps — O(1)).
        var row_imm = row_buf^.to_immutable()
        var hash_imm = hash_buf^.to_immutable()

        var result = List[Partition](capacity=p)
        for pid in range(p):
            var off = histogram.digit_start(pid)
            var sz = histogram.digit_count(pid)
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
    ](self, var hashes: UInt64Array, op: Op) raises -> Tuple[
        List[Partition], List[R]
    ]:
        """Run ``op`` on every partition in parallel; return the split and the
        results.

        Partitions ``hashes`` (one radix pass), then dispatches one worker per
        partition via ``fan_out`` — each worker calls
        ``op(partition_index, row_indices, hashes)`` and produces one ``R``. The
        index lets an op correlate its partition with a paired structure (e.g.
        the probe side pairs partition ``i`` with build-side table ``i``, since
        both were split on the same radix bits). Partitions are independent, so
        there are no locks: workers write into distinct pre-sized slots, and the
        results are *moved* out in partition order (never copied — ``R`` may own
        a `SwissHashTable`).

        **The partitions come back too**, because every caller needs them after
        the join and this used to drop them: both the hash join's build and the
        radix group-by returned ``rows`` as part of their ``R`` purely to get
        the mapping back out, which put an input into a structure named for
        results. They are ref-counted offset slices over the two buffers
        ``partition`` already allocated, so handing them back costs a refcount
        bump apiece.

        This is the skeleton behind ``HashIndex``'s radix placement: a table
        filled per partition, and a batch looked up per partition.
        """
        var partitions = self.partition(hashes^)
        var p = len(partitions)
        var slots = List[Optional[R]](length=p, fill=None)

        def run(w: Int, i: Int) raises {mut slots, imm}:
            slots[i] = op(
                i,
                partitions[i].row_indices.copy(),
                partitions[i].hashes.copy(),
            )

        self.ctx.fan_out(p, run, p)

        var out = List[R](capacity=p)
        for i in range(p):
            out.append(slots[i].take())
        return (partitions^, out^)
