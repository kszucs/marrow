"""Join kernels for Arrow StructArrays.

Public API
----------
``hash_join``   — equijoin two StructArrays on positional key columns.
``HashJoin``    — hash join using SwissHashTable; reusable across morsels.

Internal types
--------------
``IndexPairs``  — (left_indices, right_indices) result of a probe phase.

Supported join kinds (pass the JOIN_* constants defined below):
  JOIN_INNER  — only matched rows
  JOIN_LEFT   — all left + matched right (NULLs for non-matches)
  JOIN_RIGHT  — all right + matched left (NULLs for non-matches)
  JOIN_FULL   — all rows from both sides
  JOIN_SEMI   — left rows with at least one match (left columns only)
  JOIN_ANTI   — left rows with no match (left columns only)

Supported strictness:
  JOIN_ALL    — default: return all matching pairs (Cartesian for multi-match)
  JOIN_ANY    — return at most one matching right row per left row

Future join algorithms (see backlog M3.1); operators name the concrete
algorithm, so a new one is a new struct, not a conformance:
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
   8%  take(DynArray) dispatch    — runtime type dispatch
   7%  RadixPartitioner scatter   — histogram + scatter passes
   2%  tcmalloc::PageHeap::New    — allocator
   ... dispatch + allocator small ops

The hot paths are memory-latency / bandwidth bound; further single-digit
percent wins are available but require structural work.  Ordered by
expected payoff for the next round of optimization:

1. **Fused equality verification**
   Current ``SwissHashTable.probe`` does
   ``take(build_keys, bi) + take(probe_keys, pi) + EqKernel.apply(...)`` which
   allocates three intermediate Arrays (two gathered keys + one mask)
   then ``filter`` rebuilds two Int32 arrays from the candidates.
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
   small-object pool inside ``ExecContext`` would eliminate it.
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

See ``docs/architecture.md`` for the layering this sits in, and
``docs/backlog.md`` §8 for the designs this replaced — the original spec's
``JoinHashTable`` with an intrusive ``_chain_next`` list was superseded by
``SwissHashTable`` plus a CSR ``_offsets``/``_rows`` index.
"""


from ..arrays import (
    DynArray,
    StructArray,
    Int32Array,
    UInt64Array,
)
from ..buffers import Buffer
from ..builders import Int32Builder
from ..dtypes import (
    DynType,
    Field,
    int32,
    struct_,
    null,
)
from ..execution import ExecContext
from .filter import TakeKernel, filter, take
from .hashtable import SwissHashTable
from .partition import RadixPartitioner
from .hashing import HashKernel
from ..utils import Hasher, RapidHash64

# ---------------------------------------------------------------------------
# Join kind constants — what rows appear in output
#
# Owned here, not in the plan layer, since these describe the join kernel's own
# algorithm/behavior; the relational-plan layer is a consumer of this
# vocabulary, not its owner. `marrow/expr/logical.mojo` imports these.
# ---------------------------------------------------------------------------


struct JoinKind(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which rows a join emits — and therefore which *columns*.

    A value type rather than a bare `UInt8` for two reasons, both of which had
    already cost something:

    1. **The column question had four answers.** "Does this kind emit the right
       side's columns?" was re-derived inline at `output_dtype`, `_assemble`,
       `relations.Join.schema` and `tabular.join`, and they did not agree — the
       first two differed on MARK, so a MARK join built a `StructArray`
       declaring the right side's fields while carrying only the left's. That is
       a corrupt array, and nothing checked. It is one method now.
    2. **`kind` and `strictness` were both `UInt8`.** Passing them swapped
       compiled silently, and the numbering makes it invisible rather than merely
       plausible: `JOIN_INNER` and `JOIN_ALL` are both 0, `JOIN_LEFT` and
       `JOIN_ANY` are both 1. Strictness stays `UInt8` for now, but the two
       are no longer interchangeable at a call site.

    Both references put these predicates on the type — polars has
    `JoinType::is_semi_anti()` / `is_equi()`, ClickHouse a set of `constexpr
    isLeft(kind)` free functions. Neither answers the question inline at a use
    site. marrow follows polars' *flat* model, where SEMI and ANTI are kinds;
    ClickHouse instead files them under strictness.
    """

    var code: UInt8
    """The wire value. Stable — `expr.relations` and the Python bindings both
    round-trip it."""

    @implicit
    def __init__(out self, code: UInt8):
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        return self.code != other.code

    def emits_right_columns(self) -> Bool:
        """Whether the output carries the right side's columns.

        False only for the existence filters, which project the left side and
        use the right purely as a predicate. **This is the single source for
        the join's output width** — `output_dtype` and `_assemble` must agree or
        the result `StructArray` is malformed.
        """
        return self != JOIN_SEMI and self != JOIN_ANTI

    def emits_unmatched_left(self) -> Bool:
        """Whether unmatched *build*-side rows appear, padded with nulls."""
        return self == JOIN_LEFT or self == JOIN_FULL

    def emits_unmatched_right(self) -> Bool:
        """Whether unmatched *probe*-side rows appear, padded with nulls."""
        return self == JOIN_RIGHT or self == JOIN_FULL

    def is_supported(self) -> Bool:
        """Whether a kernel actually implements this kind.

        CROSS, MARK and SINGLE have constants and no implementation. They used
        to fall through to the outer-join arm and silently produce wrong output;
        `hash_join` now rejects them.
        """
        return (
            self == JOIN_INNER
            or self == JOIN_LEFT
            or self == JOIN_RIGHT
            or self == JOIN_FULL
            or self == JOIN_SEMI
            or self == JOIN_ANTI
        )

    def write_to[W: Writer](self, mut writer: W):
        """PyArrow's spelling, so error messages and `tabular.join`'s `how=`
        argument use one vocabulary."""
        if self == JOIN_INNER:
            writer.write("inner")
        elif self == JOIN_LEFT:
            writer.write("left outer")
        elif self == JOIN_RIGHT:
            writer.write("right outer")
        elif self == JOIN_FULL:
            writer.write("full outer")
        elif self == JOIN_SEMI:
            writer.write("left semi")
        elif self == JOIN_ANTI:
            writer.write("left anti")
        elif self == JOIN_CROSS:
            writer.write("cross")
        elif self == JOIN_MARK:
            writer.write("mark")
        elif self == JOIN_SINGLE:
            writer.write("single")
        else:
            writer.write("join kind ", self.code)

    def write_repr_to[W: Writer](self, mut writer: W):
        self.write_to(writer)

    @staticmethod
    def parse(how: String) raises -> Self:
        """PyArrow's `how=` spelling, with the short forms also accepted.

        The inverse of `write_to`, and it lives here so the name-to-kind mapping
        has one owner. `tabular.join` had its own copy — a fourth place that
        knew which kinds exist, and the one that would silently disagree when a
        kind was added."""
        if how == "inner":
            return JOIN_INNER
        elif how == "left outer" or how == "left":
            return JOIN_LEFT
        elif how == "right outer" or how == "right":
            return JOIN_RIGHT
        elif how == "full outer" or how == "full":
            return JOIN_FULL
        elif how == "left semi" or how == "semi":
            return JOIN_SEMI
        elif how == "left anti" or how == "anti":
            return JOIN_ANTI
        else:
            raise Error("join: unknown join type '", how, "'")


comptime JOIN_INNER = JoinKind(0)
"""INNER JOIN: only rows with matching keys on both sides."""

comptime JOIN_LEFT = JoinKind(1)
"""LEFT JOIN: all left rows + matched right rows; NULLs for non-matches."""

comptime JOIN_RIGHT = JoinKind(2)
"""RIGHT JOIN: all right rows + matched left rows; NULLs for non-matches."""

comptime JOIN_FULL = JoinKind(3)
"""FULL OUTER JOIN: all rows from both sides; NULLs for non-matches."""

comptime JOIN_SEMI = JoinKind(4)
"""LEFT SEMI JOIN: left rows that have at least one match in right (left columns only)."""

comptime JOIN_ANTI = JoinKind(5)
"""LEFT ANTI JOIN: left rows with no match in right (left columns only)."""

comptime JOIN_CROSS = JoinKind(6)
"""CROSS JOIN: Cartesian product; no key columns required. **Not implemented** —
`is_supported()` is False and `hash_join` rejects it."""

# Internal join kinds — generated by the planner for subquery decorrelation.
# Not intended for direct use. **Neither is implemented**; both are rejected.
comptime JOIN_MARK = JoinKind(10)
"""MARK JOIN: adds a boolean marker column for EXISTS/IN subquery rewriting."""

comptime JOIN_SINGLE = JoinKind(11)
"""SINGLE JOIN: at-most-1 right row per left row; for scalar subqueries."""

# ---------------------------------------------------------------------------
# Join strictness constants — how many matches are used
# ---------------------------------------------------------------------------

comptime JOIN_ALL: UInt8 = 0
"""ALL strictness (default): return all matching rows (Cartesian product for multi-match)."""

comptime JOIN_ANY: UInt8 = 1
"""ANY strictness: return at most one matching right row per left row (no row duplication)."""


@fieldwise_init
struct JoinIndex(Copyable, Movable):
    """Which build row pairs with which probe row, one entry per output row.

    A named pair rather than `Tuple[Int32Array, Int32Array]`, which is what this
    was. The tuple spelling means every consumer writes `pairs.build` and
    `pairs.probe`, and nothing distinguishes them -- reading the build side as the
    probe side is a silent wrong answer, and `_assemble` gathers the two sides
    from different arrays, so getting them the wrong way round produces a
    plausible-looking result with the columns crossed.

    A null entry on either side means "no match": an outer join emits an
    unmatched build row as `(row, null)` and an unmatched probe row as
    `(null, row)`.
    """

    var build: Int32Array
    """Row indices into the build (left) side."""
    var probe: Int32Array
    """Row indices into the probe (right) side."""

    def __len__(self) -> Int:
        return len(self.build)


comptime IndexPairs = JoinIndex
"""Parallel (left_indices, right_indices) arrays from the probe phase."""


def _concat_int32(
    var parts: List[Optional[Int32Array]],
) raises -> Int32Array:
    """Concatenate a list of Int32 index arrays into one.

    Used by the parallel probe path to merge per-partition pair arrays
    into a single ``IndexPairs``. Direct buffer-level memcpy rather than
    going through the generic ``concat(DynArray)`` path — the per-
    partition pair arrays are always valid dense Int32 buffers with
    ``nulls == 0``, so we can skip bitmap and type-dispatch overhead.
    """
    var total = 0
    for ref p in parts:
        if p:
            total += len(p.value())
    if total == 0:
        var empty = Int32Builder(capacity=0)
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

    return Int32Array(
        dtype=int32,
        length=total,
        nulls=0,
        offset=0,
        bitmap=None,
        buffer=out_buf^.to_immutable(),
    )


# ---------------------------------------------------------------------------
# HashJoin — hash join using SwissHashTable
# ---------------------------------------------------------------------------


comptime _PARALLEL_THRESHOLD = 100_000
"""Below this build-side row count the parallel path falls back to serial —
partitioning overhead dominates below ~100k rows on typical inputs."""

# The hash → partition → per-partition parallel work → merge skeleton shared by
# `build_parallel`, `probe_parallel`, and the radix group-by lives in
# `RadixPartitioner.map_partitions` (partition.mojo); each call site supplies
# only its per-partition op and its own merge.

comptime _PROBE_STRIPE_THRESHOLD = 32_768
"""Below this *probe-call* row count the probe hashes on the calling thread.

`ExecContext.parallel(n)` is a forced count, and a forced count is an
*instruction*: `stripe` splits a 1,000-row loop `n` ways because the caller
asked for `n` workers. That is the right reading for a caller who sized the
work, and the wrong one for a kernel splitting whatever batch it was handed —
so the probe asks `worth_parallel`, which treats a forced count as a *budget*,
about the rows in this call. Measured: an 8192-row probe hashed across 8
forced workers costs ~1213 us against ~76 us on the calling thread.

Set to `stripe`'s own default `min_parallel_size` — this is the same crossover,
just asked about the probe batch rather than about a whole column."""

comptime _DEFAULT_RADIX_BITS = 4
"""Default radix fanout for ``RadixPartitioner`` (16 partitions).

Was 6 (64 partitions), chosen from a sweep of a **one-shot** 10M INNER join
where 32 / 64 / 128 all landed within ~1 ms: partitioning is a per-*call* cost,
and one call over 10M rows amortizes any fanout. The plan layer streams the
probe side in 8192-row morsels, so it pays that cost ~122 times at 1M rows
instead of once, and the fanout stops being free.

Re-swept on the morselized shape (build + 8192-row probes, 8 workers, Apple
Silicon), times for the whole join:

    bits   1M      4M       10M
    3      20.3    61.7     158.9
    4      16.8    68.5     181.1
    6      30.1    93.5     238.9   (serial: 17.4 / 101.8 / 360.5)

3 is faster at 4M and 10M but loses to the serial baseline at 1M; 4 is the
only setting that beats serial at every size, and at 8 workers it is also the
principled one — 2 partitions per worker, enough to balance skew without
paying for 8x oversubscription. Fanout stays a runtime parameter on
``RadixPartitioner`` and can be tuned per workload."""


def _key_struct(source: StructArray, indices: List[Int]) raises -> StructArray:
    """The key columns, renamed to **positional** field names.

    `StructArray.select` keeps each field's original name, and a struct's
    dtype includes those names — so a build side of `struct<dept: int64>` and
    a probe side of `struct<did: int64>` are *different dtypes*, and the
    `EqKernel.apply` that filters hash collisions rejects the pair through
    `expect_same_dtype`. Join keys are matched by position, never by name, so
    the names are normalised away here.

    Without this, `left_on="dept", right_on="did"` — the ordinary shape, since
    a foreign key rarely shares its referent's name — raised
    `equal: dtype mismatch: struct<dept: int64> vs struct<did: int64>`. Only
    joins whose key columns happened to share a name worked.
    """
    var selected = source.select(indices)
    ref st = selected.dtype.as_struct()
    var fields = List[Field]()
    for i in range(len(st.fields)):
        fields.append(
            Field(String(i), st.fields[i].dtype.copy(), st.fields[i].nullable)
        )
    return StructArray(
        dtype=struct_(fields^),
        length=selected.length,
        nulls=selected.null_count(),
        offset=selected.offset,
        bitmap=selected.bitmap,
        children=selected.children.copy(),
    )


struct HashJoin[Hash: Hasher = RapidHash64]:
    """Hash join using SwissHashTable.

    Build phase: hash left-side key columns, insert rows into hash table.
    Probe phase: hash right-side key columns, look up in hash table,
    emit index pairs, verify key equality (filter hash collisions).

    Supports two execution paths, chosen by ``ctx.worth_parallel``:

    * **Serial** — a single ``SwissHashTable`` over the full build side.
      Used when the context resolves to one worker, targets a GPU, or the
      build side is below ``_PARALLEL_THRESHOLD``.
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
    var _ctx: ExecContext
    """How this join executes — held whole rather than destructured to a worker
    count. It used to be a bare `_num_threads: Int`, which five internal sites
    then rebuilt into `ExecContext.parallel(n)`; every one of those
    silently dropped the caller's GPU device, since that factory sets
    `device=None`."""
    var _left_key_indices: List[Int]
    var _left_dtype: DynType
    var _left_data: Optional[StructArray]
    var _left_rows: Int

    # Serial path state
    var _table: SwissHashTable[Self.Hash]

    # Parallel path state (populated by build_parallel)
    var _tables: List[SwissHashTable[Self.Hash]]
    """One SwissHashTable per partition (parallel path only)."""
    var _left_partition_keys: List[StructArray]
    """Per-partition build-side keys, used for equality verification."""
    var _left_partition_rows: List[Int32Array]
    """Per-partition original row indices — maps partition-local row
    numbers back to the original build-side row index after probe."""
    var _radix_bits: Int

    var _built_parallel: Bool
    """Which layout `build` produced — the *correctness* constraint.

    `probe_serial` reads `_table`, `probe_parallel` reads `_tables`, and only
    the matching `build_*` populates either. So the probe path is not a free
    choice: it is dictated by what build did. This used to be re-derived by
    asking `worth_parallel` about `_left_rows` a second time and trusting the
    two calls to agree, which conflated it with the throughput decision below.
    """

    def __init__(out self, var ctx: ExecContext = ExecContext()):
        """Create a HashJoin.

        Args:
            ctx: How to execute. ``ExecContext.serial()`` forces the serial
                single-table path; ``.parallel(n)`` runs radix-partitioned
                parallel build + probe across ``n`` workers; ``.parallel()`` /
                ``.auto()`` picks ``num_physical_cores()``. Builds smaller than
                ``_PARALLEL_THRESHOLD`` fall back to serial regardless.
        """
        self._ctx = ctx^
        self._left_key_indices = List[Int]()
        self._left_dtype = null
        self._left_data = None
        self._left_rows = 0
        self._table = SwissHashTable[Self.Hash]()
        self._tables = List[SwissHashTable[Self.Hash]]()
        self._left_partition_keys = List[StructArray]()
        self._left_partition_rows = List[Int32Array]()
        self._radix_bits = _DEFAULT_RADIX_BITS
        self._built_parallel = False

    # ------------------------------------------------------------------
    # Public dispatchers — route to serial or parallel implementations.
    # ------------------------------------------------------------------

    def build(mut self, left: StructArray, left_key_indices: List[Int]) raises:
        if not self._ctx.worth_parallel(left.length, _PARALLEL_THRESHOLD):
            self.build_serial(left, left_key_indices)
        else:
            self.build_parallel(left, left_key_indices)

    def probe(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: JoinKind = JOIN_INNER,
        strictness: UInt8 = JOIN_ALL,
    ) raises -> StructArray:
        # Layout, not throughput: `probe_parallel` reads the per-partition
        # tables that only `build_parallel` populates, and `probe_serial` reads
        # the single table that only `build_serial` populates. Whichever build
        # ran decides this, and nothing else may.
        #
        # Throughput is a separate question, and asking it here was the bug:
        # `worth_parallel(self._left_rows, ...)` let one row count answer both,
        # so a 1M-row build put every 8192-row morsel the plan layer streams
        # through the partitioned path. The throughput levers live where the
        # per-call cost actually is — `_DEFAULT_RADIX_BITS` (how much work each
        # probe call must repeat) and `_probe_ctx` (whether a call is big
        # enough to stripe) — and both are sized by the probe batch, never by
        # the build side. Measurement says the partition *fan-out* itself is
        # not a lever: it beats running the same partitions serially at every
        # batch size tested, 8192 rows included.
        if self._built_parallel:
            return self.probe_parallel(
                right, right_key_indices, kind, strictness
            )
        else:
            return self.probe_serial(right, right_key_indices, kind, strictness)

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
        self._built_parallel = False
        var ctx = self._ctx.copy()
        self._table.build(_key_struct(left, left_key_indices), ctx)

    def _probe_ctx(self, probe_rows: Int) -> ExecContext:
        """The context to spend on a probe call of `probe_rows` rows.

        Splits the two questions a single `ExecContext` otherwise answers at
        once: *how many workers may this join use* (the caller's budget, held
        in `self._ctx`) versus *is this particular call big enough to spend
        them* (a property of the batch, which only the call site knows).
        `worth_parallel` is the right predicate because it reads a forced
        thread count as a budget rather than as an instruction.
        """
        if self._ctx.worth_parallel(probe_rows, _PROBE_STRIPE_THRESHOLD):
            return self._ctx.copy()
        else:
            return ExecContext.serial()

    def probe_serial(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: JoinKind,
        strictness: UInt8,
    ) raises -> StructArray:
        var left_keys = _key_struct(
            self._left_data.value(), self._left_key_indices
        )
        var right_keys = _key_struct(right, right_key_indices)
        # Sized by *this call's* probe rows, not by the build side and not by
        # the raw worker count: `SwissHashTable.probe` spends `ctx` on hashing
        # the probe keys, and striping 8192 of them across a forced 8 workers
        # costs ~16x what hashing them on the calling thread does.
        var pairs = self._table.probe(
            left_keys,
            right_keys,
            self._left_rows,
            single_match=strictness == JOIN_ANY,
            ctx=self._probe_ctx(len(right)),
        )
        # `SwissHashTable.probe` still returns a bare tuple -- it cannot name
        # `JoinIndex`, since `join` imports `hashtable` and not the other way
        # round. Named at this boundary instead, which is where the two sides
        # stop being interchangeable.
        var verified = JoinIndex(pairs[0].copy(), pairs[1].copy())
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

        var left_keys = _key_struct(left, left_key_indices)

        # Pre-size one table per partition; each is built *in place* by the
        # matching worker (avoids moving/copying a SwissHashTable out of a
        # result), so the op only returns the cheap (keys, rows) per partition.
        var partitioner = RadixPartitioner(
            num_bits=self._radix_bits,
            ctx=self._ctx.copy(),
        )
        var p = partitioner.num_partitions()
        var tables = List[SwissHashTable[Self.Hash]](capacity=p)
        for _ in range(p):
            tables.append(SwissHashTable[Self.Hash]())

        def build_partition(
            i: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises {mut tables, imm} -> Tuple[StructArray, Int32Array]:
            var k = TakeKernel.apply(left_keys, rows)
            tables[i].build_hashes(part_hashes)
            return (k^, rows.copy())

        var hashes = HashKernel[Self.Hash].apply(left_keys, self._ctx.copy())
        var parts = partitioner.map_partitions[Tuple[StructArray, Int32Array]](
            hashes^, build_partition
        )

        var keys_out = List[StructArray](capacity=p)
        var rows_out = List[Int32Array](capacity=p)
        for i in range(len(parts)):
            keys_out.append(parts[i][0].copy())
            rows_out.append(parts[i][1].copy())

        self._tables = tables^
        self._left_partition_keys = keys_out^
        self._left_partition_rows = rows_out^
        self._built_parallel = True

    def probe_parallel(
        self,
        right: StructArray,
        right_key_indices: List[Int],
        kind: JoinKind,
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
        var right_keys = _key_struct(right, right_key_indices)
        var right_n = len(right)
        var single = strictness == JOIN_ANY

        # Per-partition probe: gather this partition's probe keys, look them up
        # in the matching build-side table `i` (same radix bits → same
        # partition), and remap partition-local indices to global row numbers.
        def probe_partition(
            i: Int, rows: Int32Array, part_hashes: UInt64Array
        ) raises {imm} -> IndexPairs:
            var probe_keys_i = TakeKernel.apply(right_keys, rows)
            var pairs = self._tables[i].probe(
                self._left_partition_keys[i],
                probe_keys_i,
                len(self._left_partition_keys[i]),
                single_match=single,
                hashes=part_hashes.copy(),
            )
            return JoinIndex(
                TakeKernel.apply(self._left_partition_rows[i], pairs[0]),
                TakeKernel.apply(rows, pairs[1]),
            )

        # 1. Hash probe side in parallel; 2-3. partition + parallel probe.
        var probe_hashes = HashKernel[Self.Hash].apply(
            right_keys, self._ctx.copy()
        )
        var pairs_per_partition = RadixPartitioner(
            num_bits=self._radix_bits,
            ctx=self._ctx.copy(),
        ).map_partitions[IndexPairs](probe_hashes^, probe_partition)

        # 4. Concat per-partition pairs into a single IndexPairs.
        var p = len(pairs_per_partition)
        var part_build_idx = List[Optional[Int32Array]](length=p, fill=None)
        var part_probe_idx = List[Optional[Int32Array]](length=p, fill=None)
        for i in range(p):
            part_build_idx[i] = pairs_per_partition[i].build.copy()
            part_probe_idx[i] = pairs_per_partition[i].probe.copy()
        var combined_build = _concat_int32(part_build_idx^)
        var combined_probe = _concat_int32(part_probe_idx^)
        var verified = JoinIndex(combined_build^, combined_probe^)

        var final = self._emit_unmatched(verified^, right_n, kind, strictness)
        return self._assemble(right, final, kind)

    def _emit_unmatched(
        self,
        var pairs: IndexPairs,
        right_rows: Int,
        kind: JoinKind,
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
        var n_pairs = len(pairs.build)
        for i in range(n_pairs):
            var lid = Int(pairs.build.unsafe_get(i))
            var rid = Int(pairs.probe.unsafe_get(i))
            if lid >= 0:
                matched_build[lid] = True
            if rid >= 0:
                matched_probe[rid] = True

        if kind == JOIN_SEMI:
            var lb = Int32Builder(capacity=self._left_rows)
            var rb = Int32Builder(capacity=self._left_rows)
            for i in range(self._left_rows):
                if matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return JoinIndex(lb.finish(), rb.finish())

        if kind == JOIN_ANTI:
            var lb = Int32Builder(capacity=self._left_rows)
            var rb = Int32Builder(capacity=self._left_rows)
            for i in range(self._left_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
            return JoinIndex(lb.finish(), rb.finish())

        # LEFT / RIGHT / FULL: matched pairs + unmatched rows.
        var lb = Int32Builder(capacity=n_pairs + self._left_rows)
        var rb = Int32Builder(capacity=n_pairs + right_rows)
        for i in range(n_pairs):
            lb.append(pairs.build.unsafe_get(i))
            rb.append(pairs.probe.unsafe_get(i))
        if kind.emits_unmatched_left():
            for i in range(self._left_rows):
                if not matched_build[i]:
                    lb.append(Scalar[int32.native](i))
                    rb.append_null()
        if kind.emits_unmatched_right():
            for i in range(right_rows):
                if not matched_probe[i]:
                    lb.append_null()
                    rb.append(Scalar[int32.native](i))
        return JoinIndex(lb.finish(), rb.finish())

    def build_dtype(self) -> DynType:
        return self._left_dtype.copy()

    def num_left_rows(self) -> Int:
        return self._left_rows

    def built_parallel(self) -> Bool:
        """Whether `build` produced the radix-partitioned layout.

        Exposed so a test can prove it exercised the partitioned probe rather
        than passing vacuously on the serial one — the two paths are supposed
        to be indistinguishable in their results, which is exactly what makes
        an accidental fallback invisible.
        """
        return self._built_parallel

    def output_dtype(self, probe: StructArray, kind: JoinKind) -> DynType:
        """Build the output struct DataType for a join result."""
        var fields = List[Field]()
        for ref f in self._left_dtype.as_struct().fields:
            fields.append(f.copy())

        if kind.emits_right_columns():
            var left_names = List[String]()
            for ref f in self._left_dtype.as_struct().fields:
                left_names.append(f.name)
            for ref f in probe.dtype.as_struct().fields:
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
        self, right: StructArray, pairs: IndexPairs, kind: JoinKind
    ) raises -> StructArray:
        """Gather left + right columns using index pairs.

        After ``sync_parallelize`` in ``probe_parallel`` has finished
        there's no outer parallel region, so each per-column ``take``
        can safely fan its SIMD gather loop across workers internally.
        We pass this join's own ``ExecContext`` through, and ``take``
        decides per-column whether it's big enough to stripe (its own grain
        threshold inside ``apply``).
        """
        ref left = self._left_data.value()
        var out_cols = List[DynArray]()
        var ctx = self._ctx.copy()

        for c in range(len(left.children)):
            out_cols.append(take(left.children[c].copy(), pairs.build, ctx))

        if kind.emits_right_columns():
            for c in range(len(right.children)):
                out_cols.append(
                    take(right.children[c].copy(), pairs.probe, ctx)
                )

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
# hash_join — top-level public API
# ---------------------------------------------------------------------------


def hash_join(
    left: StructArray,
    right: StructArray,
    left_on: List[Int],
    right_on: List[Int],
    kind: JoinKind = JOIN_INNER,
    strictness: UInt8 = JOIN_ALL,
    ctx: ExecContext = ExecContext.auto(),
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
        ctx: How to execute. ``.auto()`` (default) picks
            ``num_physical_cores()`` workers; ``.serial()`` forces the serial
            single-table path; ``.parallel(n)`` runs radix-partitioned parallel
            build + probe across ``n``. Builds smaller than
            ``_PARALLEL_THRESHOLD`` always fall back to serial regardless.
            Any GPU device on the context now survives into the join's internal
            dispatches; it previously did not.

    Returns:
        Output StructArray:
        * INNER/LEFT/RIGHT/FULL: left columns + right columns.
        * SEMI/ANTI: left columns only.
    """
    if len(left_on) != len(right_on):
        raise Error("hash_join: len(left_on) != len(right_on)")
    if not kind.is_supported():
        # CROSS, MARK and SINGLE have constants but no implementation. They used
        # to fall through to the outer-join arm, and MARK additionally built a
        # result whose declared schema had more fields than it had columns.
        raise Error(
            "hash_join: join kind '",
            kind,
            "' is not implemented",
        )

    var join = HashJoin(ctx.copy())
    join.build(left, left_on)
    return join.probe(right, right_on, kind, strictness)
