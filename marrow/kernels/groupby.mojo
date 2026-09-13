"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouping`` hashes the key columns and resolves every row
     to a dense group index, storing the unique key rows.
  2. **Phase 2** — aggregate accumulation, layered on top by the caller through
     an ``AggKernel`` (``aggregate.mojo``).

The grouper is **aggregate-agnostic**: aggregates are ``AggKernel`` types, and
mapping a runtime function *name* onto one lives in the expression layer
(``marrow/expr``). ``HashGrouping`` is the keyed placement, owned concretely by
``GroupByOperator`` (``marrow/expr/physical.mojo``), which evaluates the key
expressions once per morsel and hands the resulting ``Groups`` to every
aggregate. The keyless case has no conformer here at all — it is
``Groups.single``, an empty id array, and no placement object exists to hold.

**Parallelism lives in phase 1 only, and that is the whole design.** Both
phases could in principle be parallelised, and the two options are not
equivalent:

- *Thread-local partial aggregation* — split rows into T ranges, aggregate each
  into its own table, then merge the partial states. This needs a ``merge`` on
  every ``AggKernel``, and the merges are not uniform: ``sum``/``count``/
  ``min``/``max`` combine pointwise, ``mean`` must combine ``(sum, count)`` and
  never two means, and ``Dispersion`` keeps a Welford ``(n, mean, M2)`` triple
  that needs the Chan/Golub/LeVeque formula rather than ``M2_a + M2_b``. Worse,
  exact ``count_distinct`` has **no** correct merge at all: its state is one
  hash table over ``(group, value)`` pairs, so two thread-local tables carry
  incompatible bucket numbering and a value seen by two threads is counted
  twice. It also needs a second, key-level merge — independent groupers number
  the same key differently, so slot ``g`` of one partial is not slot ``g`` of
  another.
- *Radix-partitioned placement* — what is implemented here. Rows are split by
  the **top bits of the key hash**, so every row of a group lands in exactly one
  partition and therefore in exactly one table. The grouper still emits a single
  global dense numbering, so each aggregate still sees every one of its rows,
  exactly once, through the unchanged ``Groups`` contract.

The second needs no aggregate merge, and so it is correct for every fold —
including the two the first gets wrong. ``mean`` and the variance family are not
*handled*; they are never split. Aggregation is untouched by this file.

**Grouping is on the 64-bit hash, not on the key values.** ``SwissHashTable``
resolves a key by hash alone, so two distinct keys whose rapidhashes collide
land in one group and ``key_columns`` reports whichever arrived first. That
makes ``GROUP BY`` here probabilistic rather than exact, at roughly ``n²/2⁶⁴``
— a coin-flip somewhere around four billion distinct keys. It predates the
parallel path and is unchanged by it: the radix split routes on the same hash,
so the two paths collide identically.
"""

from max.algorithm.functional import sync_parallelize

from ..arrays import (
    DynArray,
    UInt64Array,
    Int32Array,
)
from ..buffers import Buffer
from ..builders import DynBuilder, Int32Builder, UInt64Builder
from ..dtypes import DynType, Field, int32
from ..execution import ExecContext
from .groups import Groups
from .hashtable import SwissHashTable
from .hashing import RapidHashKernel
from .filter import TakeKernel
from .partition import RadixPartitioner
from ..utils import RapidHash64


comptime _PARALLEL_GROUPBY_MIN_ROWS: Int = 60_000
"""Batch size below which placement stays serial.

Matches the number ``ExecContext.worth_parallel``'s own docstring records for
group-by. Below it the radix pass, the 64 tables and the two extra row-order
passes cost more than the probe loop they replace.
"""

comptime _GROUPBY_RADIX_BITS: Int = 6
"""64 partitions — the same fan-out the hash join uses, chosen so a partition's
table tends to stay in L2.

**Raising it does not buy load balance, and was measured.** A quarter to a
third of thread-time on this path is spent parked in `semaphore_wait_trap`, and
the obvious reading — 64 work items split across 16 threads is too coarse — is
wrong. At 10M rows / 5M groups, profiling every thread and taking the minimum
of two benchmark runs:

    bits   partitions   idle share   work samples   par8 min
      6            64        31.7%           3799    22.40 ms
      7           128        29.6%           4248    24.18 ms
      8           256        35.1%           4513    28.24 ms

Idle barely moves while *work* grows 12–19%: four times the work items left the
waiting exactly where it was. The extra cost is the scatter, which went 493 ->
760 samples at 256 buckets — past 64 output streams the write cursors stop
fitting, which is the classic radix fan-out limit and the reason this constant
is 6 rather than a tuning knob.

The other tuning lever fails too: `MODULAR_THREAD_BUSY_WAIT_US`, which trades
semaphore sleeps for spinning at each barrier, is best left at its default —
200 is a wash and 2000 costs 50%, because spinning threads take cycles from the
ones still working. So the waiting is genuine idleness at ~5 barriers per
grouping rather than wakeup latency.

**Fusing the phases does not work either, and the reason is the id space.** The
obvious remedy for a barrier is to delete it — let a worker carry its partition
from insert straight through to write-back. It cannot: `base[i]` is a prefix sum
over every *preceding* partition's new-key count, and partition `i`'s count is
only known once its inserts are done, so the last partition's write-back depends
on all 64 inserts. Removing that dependency means giving each partition a fixed
id block instead of a measured one, which makes the global numbering sparse —
and `Groups` is dense precisely because an accumulator allocates one slot per
id. A wavefront (partition `i` writing back as soon as inserts `0..i` finish)
would recover part of it and is not expressible through `sync_parallelize`.

So the idle is a cost of dense ids, not an implementation defect. Anyone
attacking it should start there, not at the fan-out or the pool."""

comptime _SAMPLE_ROWS: Int = 4096
"""How many rows the cardinality probe looks at. Small enough that its table
stays in L1 and the probe costs tens of microseconds on a million rows."""

comptime _MIN_DISTINCT_RATIO: Float64 = 0.9
"""How nearly-distinct the sample must be before radix placement is worth it.

**Row count alone is the wrong question.** Radix reads the rows three extra
times — histogram, scatter, and the id write-back — where the serial path
probes once. At 1M rows and 1,000 groups the serial table never leaves L1, so
there is little for those passes to win back; at 500,000 groups it does not fit
in cache, the probe dominates, and splitting it 64 ways is what the
partitioning is for.

0.9 keeps radix off unless the sample is *almost all distinct*, which happens
only when cardinality is large relative to `_SAMPLE_ROWS`. A 1,000-group column
samples at ~0.24 and stays serial; a 500,000-group one samples at ~1.0.

**The threshold is not calibrated against this implementation.** It was set
from a 2.7 ms vs 6.4 ms measurement quoted in `_consume_keys_radix` below, and
that number came from a version whose *serial* O(rows) numbering pass has since
been removed — along with, later, the table pre-size and the write-back's id
loads. The structural argument above survives all three; the crossover point
does not, and 0.9 is now a conservative guess rather than a measured edge.
Re-measure before moving it.
"""


def _looks_high_cardinality(hashes: UInt64Array) raises -> Bool:
    """Is this batch distinct enough that radix placement pays for itself?

    Probes ``_SAMPLE_ROWS`` hashes into a throwaway table and asks what
    fraction came back distinct. The sample is taken on a large odd stride
    modulo ``n`` rather than every ``n / _SAMPLE_ROWS``-th row: a fixed stride
    aliases with any periodic key pattern — including the ones the tests and
    benchmarks build — and can report a handful of distinct values for a
    column that has millions.
    """
    var n = len(hashes)
    var want = min(_SAMPLE_ROWS, n)
    # The probe is asking whether the sample is *almost all* distinct, so size
    # the table for `want` keys up front. Left adaptive it grows out of
    # `_GROUP_WIDTH`, and every doubling re-inserts everything already in it —
    # about nine rehashes to answer one Bool, repeated on every serial batch
    # over the row threshold.
    var table = SwissHashTable[RapidHash64](capacity=want)
    var sample = UInt64Builder(capacity=want, zeroed=False)
    # Same indices as `(k * C) % n`, with one division instead of `want` of
    # them: consecutive terms differ by `C % n`, and two values below `n` sum
    # to less than `2n`, so a compare-and-subtract closes the modulo.
    var step = UInt64(0x9E3779B97F4A7C15) % UInt64(n)
    var idx = UInt64(0)
    for _ in range(want):
        sample.unsafe_append(hashes.unsafe_get(Int(idx)))
        idx += step
        if idx >= UInt64(n):
            idx -= UInt64(n)
    _ = table.insert_hashes(sample.finish(), grow_adaptively=False)
    return Float64(table.num_keys()) >= _MIN_DISTINCT_RATIO * Float64(want)


# ---------------------------------------------------------------------------
# HashGrouping — keys-only hash grouping
# ---------------------------------------------------------------------------


struct HashGrouping(Movable):
    """Keys-only hash grouper (ClickHouse-style, ``SwissHashTable``-backed).

    ``consume_keys`` hashes a batch of key rows, returns their dense group ids,
    and appends newly-seen key rows to a per-column builder. Call it repeatedly
    to accumulate groups across batches. NULL keys are treated as equal (same
    group), matching SQL GROUP BY semantics (unlike join, where NULL != NULL).

    Aggregate state is owned by the caller, not the grouper — see
    ``GroupByOperator`` in ``marrow/expr/physical.mojo``, which hashes the keys
    once and forwards the resulting ``Groups`` to each aggregate's own
    operator.

    Two placement paths, never mixed:

    * **Serial** — one ``SwissHashTable`` over the whole batch. Unchanged from
      the pre-parallel grouper, and what a serial context, a GPU context or a
      batch below ``_PARALLEL_GROUPBY_MIN_ROWS`` gets.
    * **Radix** — ``2 ** _GROUPBY_RADIX_BITS`` persistent tables, one per
      partition of the key hash's top bits, filled by one worker each.

    **Serial until a batch earns radix, then radix for good.** The two paths
    keep different state, so a grouper cannot alternate between them — the
    serial table and the partitioned ones would each hold half the keys and
    the ids already handed out would stop meaning anything. That is a
    constraint on *mixing*, though, not on *deciding*, and reading it as the
    latter was a bug: the choice used to be made from the first non-empty
    batch alone, so one small morsel — a selective ``Filter``, a small Parquet
    chunk — pinned a whole query to the serial path. Every serial batch now
    re-asks, and the crossing is a real migration (``_migrate_to_radix``) that
    rebuilds the serial table's keys as partitioned ones while preserving
    every group id. ``_built_parallel`` on ``HashJoin`` is the same one-way
    switch without the migration, because a join builds once.
    """

    var _table: SwissHashTable[RapidHash64]
    """Serial-path table — the only one, until the migration empties it."""

    var _key_builders: List[DynBuilder]

    var _ctx: ExecContext
    """How placement executes. Held whole, never reduced to a worker count, so
    a caller's device survives — see ``ExecContext.with_threads``."""

    var _parts: List[SwissHashTable[RapidHash64]]
    """Radix-path tables, one per partition. Empty until the switch.

    **Persistent across batches**, which is what makes the radix path work for a
    streaming operator: partition ``i`` always routes to table ``i``, so a key
    keeps its partition-local id no matter which batch it reappears in.
    """

    var _local_to_global: List[List[Int32]]
    """``_local_to_global[p][lid]`` is the global group id of partition ``p``'s
    local id ``lid``. Append-only, and dense: entry ``lid`` is written when
    local id ``lid`` is created, so it is never read unset. The indirection is
    what lets partitions number independently (in parallel) while the grouper
    still emits one dense global numbering."""

    var _num_groups: Int
    """Global group count — authoritative on both paths."""

    def __init__(out self, var ctx: ExecContext = ExecContext()):
        self._table = SwissHashTable[RapidHash64]()
        self._key_builders = List[DynBuilder]()
        self._ctx = ctx^
        self._parts = List[SwissHashTable[RapidHash64]]()
        self._local_to_global = List[List[Int32]]()
        self._num_groups = 0

    def _is_radix(self) -> Bool:
        """Which placement path is running. One-way: serial until a batch is
        big and distinct enough, radix from then on.

        Read off ``_parts`` rather than tracked in a flag beside it — the
        tables are only ever created by ``_migrate_to_radix``, so a separate
        Bool would be a second copy of the same bit to keep in step.
        """
        return len(self._parts) > 0

    def num_groups(self) -> Int:
        return self._num_groups

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        """Place this batch's rows, extending the grouping with any new slots.

        The batch-facing entry point, and the only one `GroupByOperator` uses.
        Ids are dense and stable across calls, so an accumulator that folded an
        earlier batch keeps its slots when a later one introduces new groups.
        New keys get new (dense, contiguous) ids; existing keys return the id
        they already had.

        Returning `Groups` rather than a bare id array is what keeps the ids and
        the slot count together; they are only meaningful as a pair.

        Takes **already-evaluated key columns**, never a `RecordBatch`:
        `kernels` must not depend on the expression layer, and evaluating a key
        expression is the caller's job. That is also what lets one grouping
        serve every aggregate in a query — the keys are hashed once, not once
        per aggregate.

        It hashes the keys itself. There used to be an optional pre-computed
        ``hashes`` parameter, and with it a check that its length matched
        ``keys`` — the radix path sizes its id buffer from one and scatters at
        rows derived from the other, so a mismatch wrote past the end. No
        caller ever passed it, which made both the parameter and the check
        unreachable; hashing here is the only behaviour that ever ran.

        Everything after the hashing is one of two paths, which take the same
        arguments and answer the same thing — see `_consume_keys_serial` and
        `_consume_keys_radix`.
        """
        if num_rows == 0:
            var empty = Int32Builder(0)
            return Groups(empty.finish(), self._num_groups)

        # Hashing is per-row and independent, so it stripes on both paths. The
        # grouper used to hash on the calling thread whatever context it was
        # given, because it was never handed one.
        #
        # **On the CPU even under a device context.** `RapidHashKernel.apply`
        # allocates its output with `alloc_device` when the context is a GPU
        # one, and everything downstream — `insert_hashes`' probe loop, the
        # radix scatter — reads it with host loads. There is no GPU grouping to
        # hand those hashes to, so asking for them on the device would be a
        # host read of device memory for no benefit. `worth_parallel` already
        # answers False on a GPU context, so this is the same decision the
        # placement gate makes, applied one step earlier.
        var hash_ctx = ExecContext.parallel(
            self._ctx.resolved_num_threads()
        ) if self._ctx.is_gpu() else self._ctx.copy()
        var batch_hashes = RapidHashKernel.apply(keys, num_rows, hash_ctx^)

        # Asked on *every* serial batch, not just the first. Deciding once and
        # for all off batch one looked equivalent and is not: a source hands
        # this operator morsels, not the whole input, and one small first
        # morsel used to pin an entire query to the serial path. A selective
        # `Filter` above the aggregate is enough to produce one, and
        # `ParquetScanOperator` emits a morsel per chunk rather than per row
        # group, so it can too. The test is cheap to repeat — `worth_parallel`
        # short-circuits on row count, so the 4,096-row sample is only drawn
        # for a batch already large enough to qualify.
        if (
            not self._is_radix()
            and self._ctx.worth_parallel(num_rows, _PARALLEL_GROUPBY_MIN_ROWS)
            and _looks_high_cardinality(batch_hashes)
        ):
            self._migrate_to_radix()

        var ids: Int32Array
        if self._is_radix():
            ids = self._consume_keys_radix(keys, num_rows, batch_hashes^)
        else:
            ids = self._consume_keys_serial(keys, num_rows, batch_hashes^)
        return Groups(ids^, self._num_groups)

    def _consume_keys_serial(
        mut self, keys: List[DynArray], n: Int, var hashes: UInt64Array
    ) raises -> Int32Array:
        """Placement through one table, and the counterpart to
        ``_consume_keys_radix``: same arguments, same answer, different shape.

        What a serial context, a GPU context, or a batch below
        ``_PARALLEL_GROUPBY_MIN_ROWS`` gets.

        Bucket ids are dense and assigned in row order, so first occurrences
        appear in increasing id order — one forward scan collects them all and
        stops as soon as the last new group is found (near-instant when the
        groups all appear early, which is the low-cardinality case). The radix
        path runs the same scan per partition inside ``_consume_keys_radix``,
        where it appends the *original* row rather than the batch-local one —
        sharing it would cost a second gather over every new group, so the
        seven lines are deliberately written twice.
        """
        var prev = self._table.num_keys()
        var bids = self._table.insert_hashes(hashes, grow_adaptively=True)
        var num_now = self._table.num_keys()
        # `num_groups` reads `_num_groups` rather than the table, because on
        # the radix path there is no single table to read.
        self._num_groups = num_now

        if num_now > prev:
            var first_rows = Int32Builder(capacity=num_now - prev, zeroed=False)
            var next_new = prev
            for i in range(len(bids)):
                if Int(bids.unsafe_get(i)) == next_new:
                    first_rows.unsafe_append(Int32(i))
                    next_new += 1
                    if next_new == num_now:
                        break
            self._register_new_groups(keys, first_rows.finish())
        return bids^

    def _migrate_to_radix(mut self) raises:
        """Switch placement to the 64 partitioned tables, carrying over
        whatever the serial table already holds.

        **A migration and not a flag flip**, because the two paths keep
        different state (see the struct docstring). Flipping with keys already
        in ``_table`` would strand them: the partitioned tables would never see
        them, so a key from an earlier batch would be issued a *second* id and
        an accumulator already folding into the first would answer for only
        part of its group.

        **The invariant is that no global id moves.** A bucket exists per
        distinct hash, so ``bucket_hashes()`` is a duplicate-free list of the
        keys held, in global-id order; re-inserting one gives it a
        partition-local id, and seeding ``_local_to_global`` with the id it
        already had keeps every issued id valid. ``_num_groups`` and the key
        builders are indexed by global id, so neither needs adjusting, and the
        caller's accumulators never learn that placement changed.

        The routing is ``RadixPartitioner``'s, not a second copy of it. The
        index of a hash in ``bucket_hashes()`` *is* its global id, and a
        ``Partition``'s ``row_indices`` are the input positions its hashes came
        from — so partitioning that array yields, per partition, exactly the
        hashes to insert and the global ids to record, in the same order and
        on the same top bits ``_consume_keys_radix`` will use afterwards.
        """
        if self._is_radix():
            # Correctness rests on the one call site testing this. Entering
            # again would install 64 empty tables while `_num_groups` kept its
            # value, so every id already handed out would be re-issued to a
            # different key — and `_table` is empty by then, so there would be
            # nothing to migrate and no failure either.
            raise Error("_migrate_to_radix: already on the radix path")

        var p = 1 << _GROUPBY_RADIX_BITS
        var parts = List[SwissHashTable[RapidHash64]](capacity=p)
        var l2g = List[List[Int32]](capacity=p)
        for _ in range(p):
            parts.append(SwissHashTable[RapidHash64]())
            l2g.append(List[Int32]())

        var existing = self._table.bucket_hashes()
        if len(existing) > 0:
            var routed = RadixPartitioner(
                num_bits=_GROUPBY_RADIX_BITS, ctx=self._ctx.copy()
            ).partition(existing^)
            for i in range(p):
                ref rows = routed[i].row_indices
                if len(rows) > 0:
                    var bids = parts[i].insert_hashes(
                        routed[i].hashes.copy(), grow_adaptively=True
                    )
                    l2g[i].reserve(len(bids))
                    for k in range(len(bids)):
                        # Distinct hashes into a table built empty two lines
                        # above, so `insert_hashes` hands out 0, 1, 2, ... in
                        # order and the local id is the loop counter. Appending
                        # in that order is the seeding, with no scatter.
                        debug_assert(
                            Int(bids.unsafe_get(k)) == k,
                            "migration expects identity bucket ids",
                        )
                        l2g[i].append(rows.unsafe_get(k))

        # Committed last, and together: `_parts` being non-empty is what says
        # this grouper is on the radix path, so a raise anywhere above leaves
        # it untouched and still serial rather than half-migrated.
        self._parts = parts^
        self._local_to_global = l2g^
        # The serial table is no longer the source of truth, and at the sizes
        # this runs at it is not worth keeping around.
        self._table = SwissHashTable[RapidHash64]()

    def _consume_keys_radix(
        mut self, keys: List[DynArray], num_rows: Int, var hashes: UInt64Array
    ) raises -> Int32Array:
        """Placement across ``2 ** _GROUPBY_RADIX_BITS`` independent tables.

        Partitioning on the *key hash* is what makes this aggregate-agnostic —
        see the module docstring for why that removes the merge step entirely.

        **Nothing here is O(rows) and serial**, which is the whole performance
        argument and was got wrong once. The first version assigned global ids
        by scanning rows in order, so the numbering matched the serial path
        exactly — and that scan, one row at a time through a
        `List[List[Int32]]`, cost more than the parallel insert saved: 1M rows
        at 1,000 groups went from 2.7 ms serial to 6.4 ms on 8 workers. The
        assignment now runs once per *new group* instead of once per row, and
        the only remaining row-order pass is the scatter, which is per
        partition and therefore parallel.

        The price is that ids come out **partition-major** rather than in
        first-appearance order, so the radix path renumbers relative to the
        serial one. `GROUP BY` has no defined row order, every golden
        aggregate case either sorts or returns a single row, and the serial
        path — which every order-sensitive test in the tree is small enough to
        take — is untouched.
        """
        var p = 1 << _GROUPBY_RADIX_BITS

        # Every partition's id map must be exactly as long as its table has
        # keys, since step 3 appends this batch's ids onto the end of it.
        #
        # **Checked, not asserted, because the way it breaks is silent.**
        # `insert_partition` commits into `self._parts[i]` from inside the
        # worker, and `map_partitions` runs the remaining workers to
        # completion before re-raising, so one worker failing — an allocation
        # inside `reserve`, say — leaves this batch's keys in some tables with
        # no matching ids, no `_num_groups` advance, and no way to tell later.
        # The next batch would then number from the wrong local index and
        # alias every stranded key onto another group's id: wrong aggregates,
        # never an error. 64 comparisons a batch is nothing to fail closed on.
        var prev_key_counts = List[Int](capacity=p)
        for i in range(p):
            var keys_here = self._parts[i].num_keys()
            if len(self._local_to_global[i]) != keys_here:
                raise Error(
                    "group-by placement is inconsistent: partition ",
                    i,
                    " holds ",
                    keys_here,
                    " keys against ",
                    len(self._local_to_global[i]),
                    (
                        " ids. A previous batch failed partway through and this"
                        " grouper cannot be reused."
                    ),
                )
            prev_key_counts.append(keys_here)

        # 1. One radix pass, then one worker per partition inserting into its
        #    own persistent table. Partitions are disjoint, so `self._parts[i]`
        #    is touched by exactly one worker — the same disjoint-slot
        #    discipline `map_partitions` already uses for its result slots.
        #
        #    Each worker also finds, for every group it creates, the *original
        #    batch row* it first appeared at. That is the same forward scan
        #    `consume_hashes` does, but it happens inside the worker, so
        #    locating new groups costs no serial time either.
        def insert_partition(
            i: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises {mut self, imm} -> Tuple[Int32Array, Int32Array]:
            var prev = self._parts[i].num_keys()
            # Size the table once rather than letting it double out of
            # `_GROUP_WIDTH`. Every doubling re-inserts every entry already
            # present, which `benchmarks/profiles/profile_groupby.mojo` put at
            # 13.3% of this path's work — second only to the inserts
            # themselves and the id write-back.
            #
            # **Half the partition's rows, not all of them.** The gate above
            # only says the sample came back at least `_MIN_DISTINCT_RATIO`
            # distinct, and a 4,096-row sample saturates: 5M groups over 10M
            # rows and 10M over 10M both sample at ~1.0, so the ratio cannot
            # be used as a key-count estimate. Half is within one doubling of
            # either — `insert_hashes` still grows adaptively when the guess
            # is low, and a high guess costs one power of two and no rehash.
            self._parts[i].reserve(prev + len(part_hashes) // 2)
            var bids = self._parts[i].insert_hashes(
                part_hashes, grow_adaptively=True
            )
            var now = self._parts[i].num_keys()
            var firsts = Int32Builder(capacity=now - prev, zeroed=False)
            var next_new = prev
            if now > prev:
                for j in range(len(bids)):
                    if Int(bids.unsafe_get(j)) == next_new:
                        firsts.unsafe_append(rows.unsafe_get(j))
                        next_new += 1
                        if next_new == now:
                            break
            return (bids^, firsts.finish())

        var split = RadixPartitioner(
            num_bits=_GROUPBY_RADIX_BITS, ctx=self._ctx.copy()
        ).map_partitions[Tuple[Int32Array, Int32Array]](
            hashes^, insert_partition
        )
        # `routed[i].row_indices` is where this partition's rows came from —
        # an input to the worker, so it comes back with the split rather than
        # being echoed through the result.
        ref routed = split[0]
        ref per_part = split[1]

        # 2. Lay out the global id space. Partition ``i`` takes a contiguous
        #    block starting at ``base[i]``, so the *assignment* is a prefix sum
        #    over 64 counts — the only serial arithmetic left, and it does not
        #    scale with rows or with groups.
        #
        #    Deriving the blocks up front is what makes step 3 parallel. Handing
        #    out ids one at a time from a running counter instead is O(groups)
        #    serial, which at high cardinality is O(rows/2) — and high
        #    cardinality is the only case this path runs in.
        var g0 = self._num_groups
        var base = List[Int](capacity=p)
        var running = g0
        for i in range(p):
            base.append(running)
            running += self._parts[i].num_keys() - prev_key_counts[i]
        var new_total = running - g0
        # Ids are Int32 all the way into the accumulators' scatter. Past 2^31
        # they truncate negative while `_num_groups` — an Int — keeps counting
        # correctly, so the slots would be sized right and indexed wrong: a
        # negative write, not a detectable mismatch.
        if running > Int(Int32.MAX):
            raise Error(
                "group-by: ",
                running,
                " groups exceeds the Int32 id space",
            )
        self._num_groups = running

        # 3. One parallel pass per partition doing all three remaining jobs:
        #    record each new local id's global id, copy its first row into the
        #    shared new-group block, and scatter every row's global id back to
        #    its original position. Partitions own disjoint tables, disjoint
        #    id blocks and disjoint rows, so none of it synchronizes.
        var n = num_rows
        var id_buf = Buffer.alloc_uninit[int32.native](n)
        var id_view = id_buf.view[int32.native](0, n)
        var fcap = max(new_total, 1)
        var first_buf = Buffer.alloc_uninit[int32.native](fcap)
        var first_view = first_buf.view[int32.native](0, fcap)

        def finish_partition(i: Int) {mut self, imm}:
            var pk = prev_key_counts[i]
            var b = base[i]
            # `foff[i]`, before it was derived: partition `i`'s block within
            # the shared new-group buffer sits at the same offset from `g0`
            # that its ids sit at from the global count.
            var fo = b - g0
            # Built locally and spliced in once, not appended in place. The 64
            # inner `List` headers sit contiguously in the outer list, 24 bytes
            # apart, so ~2.7 of them share a cache line — and `append` writes a
            # length field on every call. Appending straight into
            # `self._local_to_global[i]` therefore bounces a line between cores
            # once per *new group*, which on this path is ~rows/2 times.
            ref l2g = self._local_to_global[i]
            # `l2g` is grown to its table's key count by whichever pass created
            # those keys, so it holds exactly `pk` entries here and this
            # batch's ids append in bucket order. It used to be pre-filled with
            # -1 and then overwritten by the same loop — two stores per new
            # group, which at this path's design point is row-scale.
            debug_assert(len(l2g) == pk, "l2g out of step with its table")
            ref firsts = per_part[i][1]
            var mine = List[Int32](capacity=len(firsts))
            for k in range(len(firsts)):
                mine.append(Int32(b + k))
                first_view.store[1](fo + k, firsts.unsafe_get(k))
            l2g.extend(mine^)
            ref rows = routed[i].row_indices
            ref bids = per_part[i][0]
            for j in range(len(rows)):
                # A bucket id at or above `pk` was created in *this* batch, so
                # its global id is the arithmetic the loop above just wrote:
                # `base[i]` plus its offset within the block. Recomputing it
                # beats loading it back — `l2g` holds one Int32 per group and
                # is indexed in bucket order, so at high cardinality nearly
                # every read was its own cache miss on top of the scattered
                # store. Only ids carried in from an earlier batch, where the
                # numbering is not derivable, still need the table.
                var bid = Int(bids.unsafe_get(j))
                var g = Int32(b + bid - pk) if bid >= pk else l2g[bid]
                id_view.store[1](Int(rows.unsafe_get(j)), g)

        sync_parallelize(finish_partition, p)

        var ids = Int32Array(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=id_buf^.to_immutable(),
        )

        if new_total > 0:
            var new_rows = Int32Array(
                length=new_total,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=first_buf^.to_immutable(),
            )
            self._register_new_groups(keys, new_rows)
        return ids^

    def key_columns(mut self, key_fields: List[Field]) raises -> List[DynArray]:
        """The unique group-key columns (empty arrays when no groups yet).
        Finishes the per-column key builders — call once, at emit time."""
        var cols = List[DynArray]()
        # Asked once, not once per column: the builders are created together
        # by `_register_new_groups`, so either all of them exist or none does.
        if len(self._key_builders) == 0:
            for k in range(len(key_fields)):
                var empty = DynBuilder(key_fields[k].dtype)
                cols.append(empty.finish())
        else:
            for k in range(len(key_fields)):
                cols.append(self._key_builders[k].finish())
        return cols^

    def _register_new_groups(
        mut self, keys: List[DynArray], rows: Int32Array
    ) raises:
        """Append the key rows for newly-created groups to the per-column
        builders. Gathers all new rows in one ``take`` + one bulk ``extend`` per
        column instead of a slice/extend per group.

        **``rows`` must arrive in global-id order** — ``rows[j]`` is the
        first-appearance row of global group ``num_groups_before + j``, because
        the builders are indexed by global id and this only appends. The serial
        path gets that from its forward scan; the radix path gets it because
        ``finish_partition`` writes each partition's first rows at
        ``base[i] - g0``, which is the partition's own block of the id space.
        Nothing enforces it, and getting it wrong stores the wrong key value
        against a group rather than failing."""
        if len(self._key_builders) == 0:
            for k in range(len(keys)):
                self._key_builders.append(DynBuilder(keys[k].dtype()))
        for k in range(len(keys)):
            self._key_builders[k].extend(
                TakeKernel.dispatch(keys[k], rows, self._ctx.copy())
            )
