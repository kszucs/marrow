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

**Parallelism lives in placement only, and that is the whole design.** The
alternative — thread-local partial aggregation merged at the end — needs a
``merge`` on every ``AggKernel``, and the merges are not uniform: ``mean``
combines ``(sum, count)``, the variance family needs the Chan/Golub/LeVeque
formula, and exact ``count_distinct`` has **no** correct merge, since two
thread-local tables number the same value differently. Placing by the **top
bits of the key hash** (``HashIndex`` in ``hashtable.mojo``) puts every row of
a group in one partition and still hands out one dense numbering, so each
aggregate sees each of its rows once, in one slot, and nothing is merged.
"""

from ..arrays import Int32Array
from ..dtypes import int32


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
