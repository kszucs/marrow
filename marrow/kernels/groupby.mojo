"""Hash grouping — keys-only.

Two-phase group-by:
  1. **Phase 1** — ``HashGrouper`` hashes the key columns and resolves every row
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
"""

from max.algorithm.functional import sync_parallelize

from ..arrays import (
    StructArray,
    DynArray,
    UInt64Array,
    Int32Array,
)
from ..buffers import Buffer
from ..builders import DynBuilder, Int32Builder, UInt64Builder
from ..dtypes import DynType, Field, struct_, int32
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
table tends to stay in L2."""

comptime _SAMPLE_ROWS: Int = 4096
"""How many rows the cardinality probe looks at. Small enough that its table
stays in L1 and the probe costs tens of microseconds on a million rows."""

comptime _MIN_DISTINCT_RATIO: Float64 = 0.9
"""How nearly-distinct the sample must be before radix placement is worth it.

**Row count alone is the wrong question, and measuring proved it.** Radix reads
the rows three extra times — histogram, scatter (12 bytes per row into a
partition-major buffer), and the id write-back — where the serial path probes
once. At 1M rows and 1,000 groups the serial table never leaves L1 and the
whole grouping takes ~2.7 ms, so those extra passes cannot be paid back no
matter how many workers run them: the first version of this file measured
2.7 ms serial against 6.4 ms on 8 workers. At 500,000 groups the table does not
fit in cache, the probe dominates, and splitting it 64 ways is what the
partitioning is for.

0.9 keeps radix off unless the sample is *almost all distinct*, which happens
only when cardinality is large relative to `_SAMPLE_ROWS`. A 1,000-group column
samples at ~0.24 and stays serial; a 500,000-group one samples at ~1.0.
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
    var table = SwissHashTable[RapidHash64]()
    var sample = UInt64Builder(capacity=want)
    for k in range(want):
        var idx = (UInt64(k) * 0x9E3779B97F4A7C15) % UInt64(n)
        sample.append(hashes.unsafe_get(Int(idx)))
    var bids = table.insert_hashes(sample.finish(), grow_adaptively=True)
    _ = bids
    return Float64(table.num_keys()) >= _MIN_DISTINCT_RATIO * Float64(want)


# ---------------------------------------------------------------------------
# HashGrouper — keys-only hash grouping
# ---------------------------------------------------------------------------


struct HashGrouper(Movable):
    """Keys-only hash grouper (ClickHouse-style, ``SwissHashTable``-backed).

    ``consume_keys`` hashes a batch of key rows, returns their dense group ids,
    and appends newly-seen key rows to a per-column builder. Call it repeatedly
    to accumulate groups across batches. NULL keys are treated as equal (same
    group), matching SQL GROUP BY semantics (unlike join, where NULL != NULL).

    Aggregate state is owned by the caller, not the grouper — see
    ``GroupByOperator`` in ``marrow/expr/physical.mojo``, which hashes the keys
    once and forwards the resulting ``Groups`` to each aggregate's own
    operator.

    Two placement paths, latched on the first non-empty batch and never mixed:

    * **Serial** — one ``SwissHashTable`` over the whole batch. Unchanged from
      the pre-parallel grouper, and what a serial context, a GPU context or a
      batch below ``_PARALLEL_GROUPBY_MIN_ROWS`` gets.
    * **Radix** — ``2 ** _GROUPBY_RADIX_BITS`` persistent tables, one per
      partition of the key hash's top bits, filled by one worker each.

    The mode is latched rather than re-decided per batch because the two paths
    keep *different state*: the serial table and the per-partition tables would
    each hold half the keys if a grouper ever switched, and the ids it had
    already handed out would stop meaning anything. ``_built_parallel`` on
    ``HashJoin`` exists for the same reason.
    """

    var _table: SwissHashTable[RapidHash64]
    """Serial-path table — the only one, when ``_radix`` is False."""

    var _key_builders: List[DynBuilder]

    var _ctx: ExecContext
    """How placement executes. Held whole, never reduced to a worker count, so
    a caller's device survives — see ``ExecContext.with_threads``."""

    var _parts: List[SwissHashTable[RapidHash64]]
    """Radix-path tables, one per partition. Empty until latched parallel.

    **Persistent across batches**, which is what makes the radix path work for a
    streaming operator: partition ``i`` always routes to table ``i``, so a key
    keeps its partition-local id no matter which batch it reappears in.
    """

    var _local_to_global: List[List[Int32]]
    """``_local_to_global[p][lid]`` is the global group id of partition ``p``'s
    local id ``lid``, or -1 before one is assigned. The indirection is what lets
    partitions number independently (in parallel) while the grouper still emits
    one dense global numbering."""

    var _num_groups: Int
    """Global group count — authoritative on both paths."""

    var _radix: Bool
    """Which placement path was latched."""

    var _latched: Bool
    """Whether the path has been chosen yet."""

    def __init__(out self, var ctx: ExecContext = ExecContext()):
        self._table = SwissHashTable[RapidHash64]()
        self._key_builders = List[DynBuilder]()
        self._ctx = ctx^
        self._parts = List[SwissHashTable[RapidHash64]]()
        self._local_to_global = List[List[Int32]]()
        self._num_groups = 0
        self._radix = False
        self._latched = False

    def num_groups(self) -> Int:
        return self._num_groups

    def consume_hashes(
        mut self, hashes: UInt64Array, grow_adaptively: Bool = True
    ) raises -> Tuple[Int32Array, Int32Array]:
        """Resolve a batch of key hashes to dense group ids, and report where
        each *newly created* group first appeared.

        The core both grouping paths share. Returns ``(group ids per row, the
        rows — indices into this batch — at which the new groups first showed
        up)``. What a caller does with those rows is what distinguishes them:
        ``consume_keys`` gathers the key values immediately, because it groups
        batch after batch; a one-shot partitioned grouping keeps them and
        gathers every partition's keys in a single pass at the end.

        Bucket ids are dense and assigned in row order, so first occurrences
        appear in increasing id order — one forward scan collects them all and
        stops as soon as the last new group is found (near-instant when the
        groups all appear early, which is the low-cardinality case)."""
        var prev = self._table.num_keys()
        var bids = self._table.insert_hashes(
            hashes, grow_adaptively=grow_adaptively
        )
        var num_now = self._table.num_keys()

        var first_rows = Int32Builder(capacity=num_now - prev, zeroed=False)
        var next_new = prev
        if num_now > prev:
            for i in range(len(bids)):
                if Int(bids.unsafe_get(i)) == next_new:
                    first_rows.unsafe_append(Int32(i))
                    next_new += 1
                    if next_new == num_now:
                        break
        return (bids^, first_rows.finish())

    def consume_keys(
        mut self, keys: StructArray, hashes: Optional[UInt64Array] = None
    ) raises -> Int32Array:
        """Hash keys and resolve group indices, materializing the unique key
        rows as it goes. Returns the per-row group ids.

        New keys get new (dense, contiguous) group ids; existing keys return
        their previous id. Safe to call across multiple batches — this is the
        incremental entry point, which is why it gathers each batch's new key
        rows straight away rather than deferring. Pass ``hashes`` when the
        caller already computed them to skip the re-hash."""
        var n = len(keys)
        if n == 0:
            var empty = Int32Builder(0)
            return empty.finish()

        # Hashing is per-row and independent, so it stripes on both paths. The
        # grouper used to hash on the calling thread whatever context it was
        # given, because it was never handed one.
        var batch_hashes = (
            hashes.value().copy() if hashes else RapidHashKernel.apply(
                keys, self._ctx.copy()
            )
        )

        if not self._latched:
            self._latched = True
            self._radix = self._ctx.worth_parallel(
                n, _PARALLEL_GROUPBY_MIN_ROWS
            ) and _looks_high_cardinality(batch_hashes)
            if self._radix:
                for _ in range(1 << _GROUPBY_RADIX_BITS):
                    self._parts.append(SwissHashTable[RapidHash64]())
                    self._local_to_global.append(List[Int32]())

        if self._radix:
            return self._consume_keys_radix(keys, batch_hashes^)

        var grouped = self.consume_hashes(batch_hashes)
        if len(grouped[1]) > 0:
            self._register_new_groups(keys, grouped[1])
        self._num_groups = self._table.num_keys()
        return grouped[0].copy()

    def _consume_keys_radix(
        mut self, keys: StructArray, var hashes: UInt64Array
    ) raises -> Int32Array:
        """Placement across ``2 ** _GROUPBY_RADIX_BITS`` independent tables.

        Partitioning on the *key hash* is what makes this aggregate-agnostic:
        equal keys hash equally, so they route to one partition and one table.
        No group is ever split across workers, so no aggregate state is ever
        split, so none has to be merged — which is why ``mean`` and the Welford
        variance triple need nothing here.

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

        var prev_keys = List[Int](capacity=p)
        for i in range(p):
            prev_keys.append(self._parts[i].num_keys())

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
        ) raises {mut self, imm} -> Tuple[Int32Array, Int32Array, Int32Array]:
            var prev = self._parts[i].num_keys()
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
            return (rows.copy(), bids^, firsts.finish())

        var per_part = RadixPartitioner(
            num_bits=_GROUPBY_RADIX_BITS, ctx=self._ctx.copy()
        ).map_partitions[Tuple[Int32Array, Int32Array, Int32Array]](
            hashes^, insert_partition
        )

        # 2. Lay out the global id space. Partition ``i`` takes a contiguous
        #    block starting at ``base[i]``, so the *assignment* is a prefix sum
        #    over 64 counts — the only serial arithmetic left, and it does not
        #    scale with rows or with groups.
        #
        #    Deriving the blocks up front is what makes step 3 parallel. Handing
        #    out ids one at a time from a running counter instead is O(groups)
        #    serial, which at high cardinality is O(rows/2) — and high
        #    cardinality is the only case this path runs in.
        var base = List[Int](capacity=p)
        var foff = List[Int](capacity=p)
        var running = self._num_groups
        var new_total = 0
        for i in range(p):
            base.append(running)
            foff.append(new_total)
            var fresh = self._parts[i].num_keys() - prev_keys[i]
            running += fresh
            new_total += fresh
        self._num_groups = running

        # 3. One parallel pass per partition doing all three remaining jobs:
        #    record each new local id's global id, copy its first row into the
        #    shared new-group block, and scatter every row's global id back to
        #    its original position. Partitions own disjoint tables, disjoint
        #    id blocks and disjoint rows, so none of it synchronizes.
        var n = len(keys)
        var id_buf = Buffer.alloc_uninit[int32.native](n)
        var id_view = id_buf.view[int32.native](0, n)
        var first_buf = Buffer.alloc_uninit[int32.native](max(new_total, 1))
        var first_view = first_buf.view[int32.native](0, max(new_total, 1))

        def finish_partition(i: Int) {mut self, imm}:
            var now = self._parts[i].num_keys()
            while len(self._local_to_global[i]) < now:
                self._local_to_global[i].append(Int32(-1))
            var firsts = per_part[i][2].copy()
            for k in range(len(firsts)):
                self._local_to_global[i][prev_keys[i] + k] = Int32(base[i] + k)
                first_view.store[1](foff[i] + k, firsts.unsafe_get(k))
            var rows = per_part[i][0].copy()
            var bids = per_part[i][1].copy()
            for j in range(len(rows)):
                var g = self._local_to_global[i][Int(bids.unsafe_get(j))]
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

    @staticmethod
    def key_fields(keys: StructArray) -> List[Field]:
        """The key columns' fields, read off the keys struct's own dtype."""
        var fields = List[Field]()
        ref st = keys.dtype.as_struct()
        for k in range(len(st.fields)):
            fields.append(Field(st.fields[k].name, st.fields[k].dtype.copy()))
        return fields^

    def key_columns(mut self, key_fields: List[Field]) raises -> List[DynArray]:
        """The unique group-key columns (empty arrays when no groups yet).
        Finishes the per-column key builders — call once, at emit time."""
        var cols = List[DynArray]()
        for k in range(len(key_fields)):
            if len(self._key_builders) == 0:
                var empty = DynBuilder(key_fields[k].dtype)
                cols.append(empty.finish())
            else:
                cols.append(self._key_builders[k].finish())
        return cols^

    def _register_new_groups(
        mut self, keys: StructArray, rows: Int32Array
    ) raises:
        """Append the key rows for newly-created groups to the per-column
        builders. Gathers all new rows in one ``take`` + one bulk ``extend`` per
        column instead of a slice/extend per group."""
        if len(self._key_builders) == 0:
            for k in range(len(keys.children)):
                self._key_builders.append(DynBuilder(keys.children[k].dtype()))
        var gathered = TakeKernel.apply(keys, rows, self._ctx.copy())
        for k in range(len(keys.children)):
            self._key_builders[k].extend(gathered.children[k])


# ---------------------------------------------------------------------------
# HashGrouping — the keyed placement
# ---------------------------------------------------------------------------
struct HashGrouping(Deinitable, Movable):
    """Dense ids from a keys-only hash table, accumulated across batches.

    Wraps `HashGrouper`, which already owns the hashing, the dense-id
    assignment and the unique-key materialisation. This adds the batch-facing
    `assign` / `num_groups` / `key_columns` surface `GroupByOperator` drives it
    through.

    **A concrete struct, not a conformer.** It used to implement a `Grouping`
    trait alongside a `ScalarGrouping` that stood for "one implicit slot", on
    the theory that window partitions and a sorted or radix placement would
    arrive as further conformers. Neither ever did, and two measurements say
    they will not arrive here: `ScalarGrouping` was **never constructed
    anywhere** — every occurrence was a type argument pinning the aggregate
    operator's unused `G` parameter — and parameterising `GroupByOperator`
    itself on the trait cost **+24,432 bytes for no benefit**
    (`expr/physical.mojo`). The only member any consumer read was
    `comptime scatters: Bool`, and the two answers to it are now two separate
    operators — `ScatteredAggregateOperator` and `RegisterAggregateOperator`.
    The radix placement that did arrive confirms the shape: it is a *runtime*
    branch inside `HashGrouper`, selected per grouper from the `ExecContext`,
    and it added no conformer and no comptime parameter anywhere.

    Takes **already-evaluated key columns**, never a `RecordBatch`: `kernels`
    must not depend on the expression layer, and evaluating a key expression is
    the caller's job. That is also what lets one grouping serve every aggregate
    in a query — the keys are hashed once, not once per aggregate.
    """

    var _grouper: HashGrouper

    def __init__(out self, var ctx: ExecContext = ExecContext()):
        self._grouper = HashGrouper(ctx^)

    def assign(mut self, keys: List[DynArray], num_rows: Int) raises -> Groups:
        """Place this batch's rows, extending the grouping with any new slots.

        Ids are dense and stable across calls, so an accumulator that folded an
        earlier batch keeps its slots when a later one introduces new groups.

        The field names built here are positional and deliberately arbitrary —
        `HashGrouper` keys on the *values*, and the caller supplies the real
        fields at `key_columns` time, where they reach the output schema.
        """
        var fields = List[Field](capacity=len(keys))
        for i in range(len(keys)):
            fields.append(Field("k" + String(i), keys[i].dtype()))
        var st = StructArray(
            dtype=struct_(fields^),
            length=num_rows,
            nulls=0,
            offset=0,
            bitmap=None,
            children=keys.copy(),
        )
        var ids = self._grouper.consume_keys(st)
        return Groups(ids^, self._grouper.num_groups())

    def num_groups(self) -> Int:
        """How many slots exist so far — the size of a per-slot accumulator."""
        return self._grouper.num_groups()

    def key_columns(mut self, fields: List[Field]) raises -> List[DynArray]:
        """One column per key field, one row per slot.

        Call once, at emit time — it finishes the key builders.
        """
        return self._grouper.key_columns(fields)
