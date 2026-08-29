"""`Groups` — which slot each row of a morsel contributes to.

Its own module, and not `core.mojo`, because `core.mojo` is the `Kernel` base
trait: 25 modules import `Kernel` and 7 import `Groups`, and the two sets are
disjoint in what they are asking for. One is the contract every kernel
implements; the other is a value type the aggregate path threads through.

Not `groupby.mojo` either, which is where the *producer* lives. Putting it there
would make `kernels/aggregate.mojo` -- which only needs the assignment -- depend
on `HashGrouping` and `SwissHashTable`, inverting a dependency to save a file.
"""

from ..arrays import Int32Array
from ..builders import Int32Builder


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
    the strategy that produced it — `HashGrouping` in `groupby.mojo` is one
    such strategy, and `expr` picks between hashing and the implicit one slot
    at plan time.
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
        var empty = Int32Builder(0)
        var g = Groups(empty.finish(), 1)
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
