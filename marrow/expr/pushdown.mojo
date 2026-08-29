"""Delivery: getting a predicate from a `Filter` down to the source that can
use it, and turning it into a read plan.

Pruning (`pruning.mojo`) is *evaluation over a different domain*. This module is
the other half — **getting the predicate to where the statistics are**, and it
is a separate problem with a separate failure mode.

# The mechanism: information travels down the lowering, not into a copy

`bindings.mojo`'s central claim is that a plan is immutable and per-execution
information travels *through* the execution. Pushdown gets the identical
treatment: `Relation.to_operator` already threads its arguments down a
recursive descent from root to source, so a `Pushdown` rides the same descent.

There is no separate pass, no plan walk, no `children()`, no rewrite and no
rebuilt node -- and a rewrite is not merely unnecessary but unavailable:
`DynRelation(copy=self)` copies trampolines bound to `R`, so a rewritten node
would have to have the same concrete type, and returning `Optional[DynRelation]`
from a trampoline field makes the struct recursive, which the compiler rejects.

One capability falls out for free: the previous expression package pushed only
into an **adjacent** scan, so `Filter(Sort(ParquetScan))` pruned nothing. Here
the predicate rides the lowering all the way down.

# Per-node rules, and the one that is a correctness trap

| node | what it does with the pushed predicate |
|---|---|
| `Filter` | **conjoin** its own and forward |
| `Sort` | forward unchanged |
| `Limit` | **clear** — see below |
| `Project` | clear (a predicate names output columns, which may be computed) |
| `Aggregate` | clear (a `Filter` above it is `HAVING`) |
| `Join` | clear (needs column provenance) |
| `InMemoryTable` | ignore |
| `ParquetScan` | **consume** |

**`Limit` must clear the predicate, and getting this wrong silently changes
results.** `filter(p)` above `limit(10)` means: take the first 10 rows, *then*
apply `p`. If the scan below skips a row group that cannot match `p`, the
"first 10 rows" are different rows — and rows the correct query would have
returned disappear. That is a false negative, the one error class the whole
subsystem forbids.

`Sort` is safe by contrast, and the contrast is the proof: sorting drops no
rows, so the surviving set of `filter(p)` above `sort` is identical either way.

**Everything above the source is unchanged.** The `FilterOperator` still
evaluates the predicate exactly, on every row the scan produced. Pruning is
conservative, so the exact predicate must still run — which makes this
correctness-neutral by construction: delete every line of this module and
`pruning.mojo` and every answer is identical.

# What is not here

`Pushdown` carries a predicate and nothing else. **Projection pushdown is the
obvious second field** -- and, at a measured 3.6x against pruning's 1.04x, the
more valuable of the two. It is a field on this plain struct, not a slot on a
box; it is left to its own change so the two measurements stay readable. The
design and its one invariant (`needed` may only be *established* by a node that
also replaces the schema, and only ever *widened* by pass-through nodes) are in
`docs/backlog.md`.

Also absent: conjunction splitting, to push half a predicate below a `Join`.
`prune()` already handles a whole conjunction compositionally, so the only
consumer of a `conjuncts()` surface would be this module, which does not need
it.
"""

from ..parquet.reader import LeafSet, ParquetFile
from ..parquet.source import ByteSource
from .bindings import Bindings
from .pruning import PrunePredicate, PruneStats, Truth


# ---------------------------------------------------------------------------
# Pushdown — what the nodes above told the source
# ---------------------------------------------------------------------------
struct Pushdown(Copyable, Movable, Sized):
    """The predicates a source may use to skip data it does not have to read.

    A **list**, not one predicate, so stacked filters compose without a node
    that ANDs two erased boxes: `Filter(Filter(scan))` forwards two entries and
    a granule survives only if every one of them says `maybe`. That is exactly
    conjunction, and it needs no new type.

    Empty is the default and means "no pruning" — which is what every plan does
    today and what every plan does when the predicate arrives already boxed as
    a `DynValue`.
    """

    var _predicates: List[PrunePredicate]

    def __init__(out self):
        self._predicates = List[PrunePredicate]()

    def __init__(out self, var predicates: List[PrunePredicate]):
        self._predicates = predicates^

    def __len__(self) -> Int:
        return len(self._predicates)

    def conjoined(self, predicate: PrunePredicate) -> Self:
        """This pushdown plus one more predicate. Returns a new value; a
        `Pushdown` is as immutable as the plan it descends through."""
        var out = List[PrunePredicate](capacity=len(self._predicates) + 1)
        for ref p in self._predicates:
            out.append(p.copy())
        out.append(predicate.copy())
        return Self(out^)

    @staticmethod
    def cleared() -> Self:
        """Nothing pushed — what `Limit`, `Project`, `Aggregate` and `Join`
        hand to their inputs. Spelled as a verb at the call site so the two
        cases read differently and a reviewer can see which is which."""
        return Self()

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """Could **every** pushed predicate be true for some row of this
        granule?

        `Truth.never` from any one of them proves the granule holds no row the
        query wants, because the predicates are conjoined by construction: they
        came from nested `Filter`s, and a row must satisfy all of them.
        """
        var answer = Truth.maybe
        for ref p in self._predicates:
            answer = answer & p.prune(stats, bindings)
        return answer


# ---------------------------------------------------------------------------
# The Parquet bridge
# ---------------------------------------------------------------------------
def row_group_stats[
    S: ByteSource, leaves: LeafSet
](file: ParquetFile[S, leaves]) raises -> List[PruneStats]:
    """One `PruneStats` per row group, keyed by the file's top-level column
    names.

    **The leaf-alignment guard is the load-bearing line.**
    `ParquetFile.statistics()` is indexed by *leaf* position, and an expression
    names a *top-level column*. Those two agree only when every top-level field
    contributes exactly one leaf — and since every field contributes at least
    one and leaves are emitted in field order, "as many leaves as fields"
    proves that one-to-one correspondence outright. Anything else (a struct
    with two members, a list) yields more leaves than fields, and this returns
    statistics-free granules rather than handing field `i` the bounds of
    somebody else's leaf. That misattribution is a live defect class here: D14
    in `2026-08-25-pruning-indexing-findings.md` was `struct<x>`'s bounds being
    handed to a top-level `x`.

    Two further protections are already in the reader and are relied on rather
    than repeated: `_trusted_leaf` returns no statistics for a leaf that is not
    a whole top-level column or whose ColumnOrder the footer never declared,
    and `ColumnStatistics.from_metadata` keeps min/max only when *both* decode.

    **This reads every leaf's statistics, not just the predicate's.**
    `ParquetFile` exposes only whole-file `statistics()`; the per-`(row group,
    leaf)` narrowing is private, so a three-column predicate on a 105-column
    file decodes 210 chunk statistics to use 6. That is footer work, not column
    data, and it is bounded by the footer's size — but it is the first thing to
    fix if pruning ever shows up in a profile.
    """
    var arrow = file.schema()
    var meta = file.metadata()
    var stats = file.statistics()
    var out = List[PruneStats](capacity=len(stats))

    var aligned = len(stats) > 0 and len(arrow.fields) == len(stats[0])
    for rg in range(len(stats)):
        var rows = meta.row_groups[rg].num_rows
        if not aligned:
            out.append(PruneStats(rows))
        else:
            var g = PruneStats(rows, capacity=len(arrow.fields))
            for i in range(len(arrow.fields)):
                g.add(
                    arrow.fields[i].name.copy(),
                    stats[rg][i].min.copy(),
                    stats[rg][i].max.copy(),
                    stats[rg][i].null_count,
                )
            out.append(g^)
    return out^


def read_plan(
    granules: List[PruneStats],
    pushed: Pushdown,
    bindings: Bindings = Bindings(),
) -> List[Int]:
    """Which granules a source still has to read, in order.

    Non-generic on purpose: `row_group_stats` is the only part that has to know
    what a `ParquetFile` is, so it is the only part instantiated per
    `(ByteSource, LeafSet)` pair. Splitting the two also makes the decision
    itself testable with hand-written statistics and no file at all — which is
    how the soundness property in `test_pruning.mojo` is exercised.
    """
    var keep = List[Int](capacity=len(granules))
    for i in range(len(granules)):
        if pushed.prune(granules[i], bindings):
            keep.append(i)
    return keep^
