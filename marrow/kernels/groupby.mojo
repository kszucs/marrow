# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Group-by placement — which slot each row of a batch contributes to.

`Groups` is the value type: a batch's rows assigned to dense group ids. It
lives here, apart from its producer, so the aggregate path imports one name
from one place.

A group-by is two phases:

1. **Placement** — every row's key to a dense group id. A group id is a
   dictionary code (``dictionary.mojo``): ``GroupedAggregateOperator``
   (``marrow/expr/physical.mojo``) holds a ``DictionaryEncoder`` over the key
   columns, encodes each morsel's keys once, and hands the resulting
   ``Groups`` to every aggregate. Encoding is exact — two distinct keys never
   share a group, however their hashes fall — and the key of every group is
   kept by the encoder.
2. **Accumulation** — each aggregate folds its rows into its slots, through an
   ``AggKernel`` (``aggregate.mojo``). The keyless case has no encoder at all:
   ``Groups.single``, an empty id array.

**Placement is parallel.** Placing by the **top bits of the key hash**
(``HashIndex`` in ``hashtable.mojo``) puts every row of a group in one
partition and still hands out one dense numbering, so each aggregate sees each
of its rows once, in one slot.

**Aggregates over separate parts of an input combine.** Every ``AggKernel``
exports its per-slot state as columns (``partials``) and folds another's back
in (``merge``): ``mean`` as ``(sum, count)``, the variance family by Chan's
formula, and exact ``count_distinct`` as each slot's distinct *values*, which
two numberings of the same value agree on where their codes would not.
"""

from std.math import ceildiv

from ..arrays import Int32Array
from ..buffers import Buffer
from ..dtypes import int32
from ..execution import ExecContext


comptime _PARALLEL_GROUP_ROWS = 65_536
"""Rows from which grouping them by group runs on several threads."""

comptime _GROUP_RANGES = 256
"""How many ranges of groups the parallel grouping deals out to threads."""


# ---------------------------------------------------------------------------
# Groups -- a batch's rows assigned to slots
# ---------------------------------------------------------------------------


struct Groups(Copyable, Movable):
    """A batch's rows assigned to dense group ids, with how many groups exist.

    The two always travel together: `ids[i]` is row `i`'s group, and
    `num_groups` sizes every per-group accumulator the ids then scatter into.
    They were passed as two parameters through ~22 signatures across `groupby`,
    `aggregate`, `distinct` and `expr.aggregates`, which let a caller size an
    accumulator from one grouping and index it with another's ids — an
    out-of-bounds scatter rather than a type error, and silent when the
    mismatched count happens to be larger.

    Sibling of `JoinIndex`, which named `Tuple[Int32Array, Int32Array]` for the
    same reason.

    Named `Groups` rather than `Grouping` because it is the *assignment*, not
    the strategy that produced it — a `DictionaryEncoder` over the keys is
    one, and `expr` picks between hashing and the implicit one slot at plan
    time.
    """

    var ids: Int32Array
    """Dense group id per row of the batch."""

    var num_groups: Int
    """How many distinct groups exist — the size of a per-group accumulator."""

    var _single: Bool
    """Whether this is the implicit one-slot assignment — no `GROUP BY`.

    **Stated, not inferred.** It used to be derived as
    `len(ids) == 0 and num_groups == 1`, which is the same predicate for two
    different assignments: a keyless query, and a *keyed* query whose morsel
    happens to carry zero rows while exactly one group has been seen so far.
    A zero-row morsel in a grouped query therefore took the ungrouped branch
    in every kernel — benign today only because every such branch folds an
    empty extent into slot 0 with the fold's identity, which is a coincidence
    of the current kernels rather than a property of the contract. The flag
    costs one byte per morsel and makes the two cases distinguishable.

    Private, with `is_single()` the only reader, so the factory below can keep
    the name `single`: the constructor states the case and nothing outside can
    contradict it after the fact.
    """

    def __init__(out self, var ids: Int32Array, num_groups: Int):
        """A keyed assignment: `ids[i]` names row `i`'s slot."""
        self.ids = ids^
        self.num_groups = num_groups
        self._single = False

    @staticmethod
    def single(num_rows: Int) raises -> Groups:
        """The one-slot assignment: every row contributes to group 0.

        `ids` is **empty**, not a `num_rows`-long run of zeros. Materialising
        one `Int32` per row to communicate a constant is exactly the cost this
        assignment exists to avoid, and `Morsel.ungrouped` already establishes
        the convention. `num_rows` is therefore accepted and not stored — it
        says what extent the caller is asserting over, the same way
        `Morsel.ungrouped` takes the batch whose length it never records.

        Read it back with `is_single`, never with `len(self.ids)`.
        """
        var g = Groups(Int32Array.empty(int32), 1)
        g._single = True
        return g^

    def is_single(self) -> Bool:
        """Whether this is the one-slot assignment — no `GROUP BY`.

        **Every implementation that loops over rows must branch on this
        first.** A per-group loop is written `for i in range(len(self.ids))`,
        and the one-slot assignment holds no ids at all, so such a loop does
        not execute and answers `[0]` or `[null]` instead of the whole-input
        aggregate. That is a wrong answer rather than a crash, which is why
        `__len__` was removed from this struct: `len(groups)` returned
        `len(self.ids)` and read as "how many rows", so `range(len(groups))`
        was a silently empty loop waiting to be written.
        """
        return self._single

    @always_inline
    def slot(self, row: Int) -> Int:
        """Row `row`'s slot: slot 0 for every row of the one-slot assignment,
        which holds no ids."""
        return 0 if self._single else Int(self.ids.unsafe_get(row))

    @no_inline
    def rows_by_group(
        self, mut offsets: Buffer[mut=True], ctx: ExecContext
    ) raises -> Int32Array:
        """Every row grouped by its group, ascending within a group, with
        `offsets[g]` where group `g`'s rows start and `offsets[num_groups]`
        their total; `offsets` holds `num_groups + 1` `Int32`s. A counting
        sort, on `ctx`'s threads when the rows are many enough: the answer is
        the same either way.

        In parallel, the groups are cut into contiguous ranges. Each thread
        counts its stripe of rows per range, and deals them out by range into
        the slots a prefix sum gave it, so each range's rows stay in row
        order; then each range is counted and placed on its own thread, into
        its own slice. It makes more passes than the serial sort, which is
        why one thread takes the serial one."""
        debug_assert(not self._single, "the one-slot assignment has no ids")
        var n = len(self.ids)
        var groups = self.num_groups
        var rows = Buffer.alloc_uninit[int32.native](max(n, 1))
        var ov = offsets.view[int32.native](0, groups + 1)
        var rv = rows.view[int32.native](0, n)
        if not ctx.worth_parallel(n, _PARALLEL_GROUP_ROWS):
            var at = List[Int](length=groups, fill=0)
            for i in range(n):
                at[Int(self.ids.unsafe_get(i))] += 1
            var pos = 0
            for g in range(groups):
                ov.unsafe_set(g, Int32(pos))
                var count = at[g]
                at[g] = pos
                pos += count
            ov.unsafe_set(groups, Int32(n))
            for i in range(n):
                var g = Int(self.ids.unsafe_get(i))
                rv.unsafe_set(at[g], Int32(i))
                at[g] += 1
            return Int32Array(
                length=n,
                nulls=0,
                offset=0,
                bitmap=None,
                buffer=rows^.to_immutable(),
            )
        ref ids = self.ids
        var ranges = min(_GROUP_RANGES, groups)
        var threads = max(1, min(ctx.resolved_num_threads(), n // 4096))
        var stripe = ceildiv(n, threads)
        var counts = Buffer.alloc_zeroed[DType.int64](threads * ranges)
        var cv = counts.view[DType.int64](0, threads * ranges)
        var starts = Buffer.alloc_uninit[DType.int64](ranges + 1)
        var sv = starts.view[DType.int64](0, ranges + 1)
        var dealt = Buffer.alloc_uninit[int32.native](n)
        var dv = dealt.view[int32.native](0, n)
        var phase = Buffer.alloc_zeroed[DType.int64](1)
        var pv = phase.view[DType.int64](0, 1)

        # One body for the three steps, so the pool is instantiated once.
        def step(i: Int) {imm}:
            var which = pv.unsafe_get(0)
            if which < 2:
                var base = i * ranges
                for row in range(i * stripe, min(n, (i + 1) * stripe)):
                    var r = Int(ids.unsafe_get(row)) * ranges // groups
                    var at = Int(cv.unsafe_get(base + r))
                    if which == 1:
                        dv.unsafe_set(at, Int32(row))
                    cv.unsafe_set(base + r, Int64(at + 1))
            else:
                var lo = ceildiv(i * groups, ranges)
                var hi = ceildiv((i + 1) * groups, ranges)
                var first = Int(sv.unsafe_get(i))
                var last = Int(sv.unsafe_get(i + 1))
                var at = List[Int](length=hi - lo, fill=0)
                for p in range(first, last):
                    at[Int(ids.unsafe_get(Int(dv.unsafe_get(p)))) - lo] += 1
                var pos = first
                for c in range(hi - lo):
                    ov.unsafe_set(lo + c, Int32(pos))
                    var count = at[c]
                    at[c] = pos
                    pos += count
                for p in range(first, last):
                    var row = dv.unsafe_get(p)
                    var c = Int(ids.unsafe_get(Int(row))) - lo
                    rv.unsafe_set(at[c], row)
                    at[c] += 1

        ctx.run(threads, step)
        # Range-major, thread-minor: thread t's rows of range r follow every
        # earlier thread's, which is what keeps each range in row order.
        var running = 0
        for r in range(ranges):
            sv.unsafe_set(r, Int64(running))
            for t in range(threads):
                var count = Int(cv.unsafe_get(t * ranges + r))
                cv.unsafe_set(t * ranges + r, Int64(running))
                running += count
        sv.unsafe_set(ranges, Int64(running))
        pv.unsafe_set(0, 1)
        ctx.run(threads, step)
        pv.unsafe_set(0, 2)
        ctx.run(ranges, step)
        ov.unsafe_set(groups, Int32(n))
        _ = counts^
        _ = starts^
        _ = dealt^
        _ = phase^
        return Int32Array(
            length=n,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=rows^.to_immutable(),
        )
