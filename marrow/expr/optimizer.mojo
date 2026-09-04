"""The optimizer: rules that rewrite a plan into a simpler plan.

`plan.optimize[Rules]()` returns a **new `DynRelation`**. It is an ordinary
plan — printable, executable, and comparable against the one it came from:

```mojo
var plan = scan("hits.parquet", s).filter(p).sort_by(...).limit(10)
print(plan)                       # Limit(Sort(Filter(ParquetScan(...))))
print(plan.optimize[AllRules]())  # Sort(Filter(ParquetScan(...)) top 10)
```

That is the property the whole file exists for, and it is what a demand
channel riding `to_operator` cannot give: there is no plan value to look at, so
nothing can be printed, diffed, or matched across more than one node.

# Reading this file

Every rule is a struct with one method:

    def apply(node: DynRelation) raises -> DynRelation

It answers the node **unchanged** for "I do not apply here", and a rewritten
node otherwise — never an `Optional`, so no caller unwraps and the identity is
just the plan it was given. The
driver walks the plan bottom-up, offers each node to every rule, and repeats
until nothing changes. **The rules below are the complete list** — there is no
behaviour hidden in the relation nodes, which is the mistake this replaced.

# How a rule reads a node

`DynRelation` is variant-backed, so a rule asks what a node is and then reads
it:

    if not node.isa[Limit]():
        return node.copy()
    ref outer = node.get[Limit]()
    if not outer.input[].copy().isa[Sort]():
        return node.copy()

No downcast, no flattened payload union, no accessor protocol — `get[R]()`
borrows the real node and its own methods answer. Rules construct nodes
directly too, which is what makes rewrites that *introduce* a node expressible
at all.

**This is why `DynRelation` is a variant.** The cost is that naming the eight
node types in one box makes every operator reachable from any plan; the
trampoline design that avoided it could not let a rule read a node or build
one, which left the rules scattered across the nodes themselves and unreadable.

# What a rule may assume

Rules run **bottom-up**: a node's children are already in final form when it is
offered. That is what lets `PushFilterBelowSort` and `TopN` compose in one pass
instead of needing the fixpoint to rediscover them.

Soundness is by construction, not by review:

- A rule that cannot prove its precondition returns `None`. Every default in
  the protocol answers "I do not know", so an unrecognised node is inert.
- No rule inspects a `DynValue`'s internals, so **no rule can lower a comptime
  expression into the runtime lane**. A fused predicate stays fused through
  every rewrite here; rules move the box, never its contents.
- Row *order* is meaningful below a `Limit`, out of a `Sort`, and — since the
  window node landed — into a `Window` that names no `ORDER BY`. The two rules
  that exploit ordering (`TopN`, `PushLimitBelowProject`) each state the
  argument at their definition.

  **The third case is why `Window` is deliberately absent from this file.**
  `WindowOperator._permutation` returns the identity when a window names no
  keys, so `ROW_NUMBER() OVER ()` reads its answer off input row order — and
  an ordered window is **not** safe either, because that permutation is
  `stable=True`, so input order still decides ties. `row_number`, `lag`,
  `lead`, `first_value` and `last_value` all change answer if tie order
  changes, and `RemoveRedundantSort` trades tie order away by design. No rule can reach a
  `Window` today — the node is not imported here, so no `isa[Window]()`
  exists — and `test_optimizer.mojo::test_no_rule_rewrites_a_plan_containing_a_window`
  pins that. Adding one means proving the rule preserves the order a window
  below it may be reading. SQL calls `ROW_NUMBER() OVER ()` nondeterministic,
  so this is an invariant to keep rather than a wrong answer to fix.
"""

from ..kernels.join import (
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from ..schema import Field, Schema, schema
from ..tabular import RecordBatch
from .logical import (
    Aggregate,
    DynRelation,
    DynValue,
    EmptyRelation,
    Filter,
    InMemoryTable,
    Join,
    Limit,
    ParquetScan,
    Project,
    Sort,
)


# ---------------------------------------------------------------------------
# Rule — one rewrite
# ---------------------------------------------------------------------------
trait Rule(Copyable, Movable):
    """One plan-to-plan rewrite.

    `apply` answers `None` for "not my shape" and a node for "replace this
    subtree with that one".

    A rule must be **semantics-preserving on its own**. The driver composes
    rules in listed order and to a fixpoint, so a rule that is only correct
    when another ran first is a rule that is not correct.
    """

    comptime NAME: String
    """What this rule is called, for `explain` and for test assertions."""

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Rules — removals
# ---------------------------------------------------------------------------
struct RemoveNoOpProject(Rule):
    """`Project` that reproduces its input's schema exactly -> the input.

    Frontends emit these constantly — a `select` of every column, or a
    `with_columns` whose expressions all folded away — and each still costs an
    operator and a materialisation per morsel.

    Matching the **schema** is what makes it safe: two projections producing
    identical fields in identical order are interchangeable however they spell
    themselves, and the child's schema is already computed and stored.
    """

    comptime NAME = "RemoveNoOpProject"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Project]():
            return node.copy()
        ref project = node.get[Project]()
        var input = project.input[].copy()
        if node.schema() != input.schema():
            return node.copy()
        for ref n in project.names.copy():
            if not project.passes_through(n):
                return node.copy()
        return input^


struct EliminateFilter(Rule):
    """`Filter(FALSE)` -> `Empty`, and `Filter(TRUE)` -> its input.

    **This is what makes constant folding worth doing.** Folding turns
    `x AND FALSE` into `FALSE` in the value constructors, and without this rule
    that only saves evaluating a comparison per row. With it, the whole subtree
    below collapses — scan, join, sort and all — and `PropagateEmpty` carries
    the emptiness up through whatever sits above.

    Real predicates fold to constants far more often than anyone writes
    `LIMIT 0`: a parameter bound to an impossible range, a generated `WHERE`
    with a contradictory pair, a frontend appending `AND true` per clause.

    The constant is read off `Filter.constant`, decided at the `.filter()` verb
    where the predicate's type was still concrete. A rule cannot ask a
    `DynValue` what it is, and the alternative — a slot on that box — is paid
    for by every projection value and sort key in the program.

    A **null** constant is not a constant here. A filter keeps rows where the
    predicate is `TRUE`, and a null predicate is not `FALSE`; it merely fails
    to select. `constant_bool` already answers `None` for it.
    """

    comptime NAME = "EliminateFilter"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref f = node.get[Filter]()
        if not f.constant:
            return node.copy()
        if f.constant.value():
            return f.input[].copy()
        var out: DynRelation = EmptyRelation(RecordBatch.empty(node.schema()))
        return out^


struct RemoveEmptyLimit(Rule):
    """`Limit(x, length=0)` -> `Empty`.

    A zero-length window returns nothing whatever `x` is, so the whole subtree
    below can be discarded — scan, join, sort and all. `LIMIT 0` is not a silly
    query: it is how a frontend asks for a schema without data, and how a UI
    renders headers before a result arrives.

    This is the rule `EmptyRelation` exists for, and it is the reason a rule
    never needs to answer "no relation": the empty case is a plan like any
    other, so it composes, prints and executes without a single caller
    unwrapping anything.
    """

    comptime NAME = "RemoveEmptyLimit"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Limit]():
            return node.copy()
        ref limit = node.get[Limit]()
        if limit.length != 0:
            return node.copy()
        var out: DynRelation = EmptyRelation(RecordBatch.empty(node.schema()))
        return out^


struct MergeLimits(Rule):
    """`Limit(Limit(x))` -> one `Limit`.

    Not `min` of the two lengths — offsets accumulate. The outer limit selects
    rows `[o2, o2+l2)` *of what the inner produced*, which is `[o1+o2, ...)` of
    the original, and it cannot reach past the inner window. Getting this wrong
    returns rows the query excluded, so the arithmetic is written out rather
    than folded into one expression.
    """

    comptime NAME = "MergeLimits"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Limit]():
            return node.copy()
        ref outer = node.get[Limit]()
        var input = outer.input[].copy()
        if not input.isa[Limit]():
            return node.copy()
        ref inner = input.get[Limit]()

        var offset = inner.offset + outer.offset
        var length: Int
        if inner.length < 0:
            length = outer.length
        else:
            var room = inner.length - outer.offset
            if room < 0:
                room = 0
            if outer.length < 0:
                length = room
            else:
                length = outer.length if outer.length < room else room
        var built: DynRelation = Limit(inner.input[].copy(), offset, length)
        return built^


struct RemoveRedundantSort(Rule):
    """`Sort(Sort(x))` -> the outer sort.

    The outer ordering wins outright: it reorders every row the inner sort
    produced, and sorting drops nothing, so the inner pass cannot affect the
    result. It can affect *ties* — marrow's sorts are stable — but only where
    the outer sort leaves rows equal, and the query then depends on an order it
    never specified. Dropping a whole buffering pipeline breaker is worth more
    than preserving that.

    **Not applied when the inner sort carries a TopN bound**, which does drop
    rows and is load-bearing.
    """

    comptime NAME = "RemoveRedundantSort"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Sort]():
            return node.copy()
        ref outer = node.get[Sort]()
        var input = outer.input[].copy()
        if not input.isa[Sort]():
            return node.copy()
        ref inner = input.get[Sort]()
        if inner.limit:
            return node.copy()
        var built: DynRelation = Sort(
            inner.input[].copy(),
            outer.keys.copy(),
            outer.ascending.copy(),
            outer.nulls_first,
            outer.limit,
        )
        return built^


struct PropagateEmpty(Rule):
    """Anything over `Empty` is `Empty`.

    Once one rule proves a subtree empty, that fact should travel: a filter of
    nothing is nothing, an ordering of nothing is nothing, a window onto
    nothing is nothing. Without this, `RemoveEmptyLimit` collapses one node and
    leaves a tower of operators above it, each of which is still built, still
    scheduled, and still processes an empty stream.

    **A `Project` over `Empty` keeps the projection's own schema**, not the
    input's — the columns still change even when no row does, and everything
    above it reads that schema. Getting this backwards would produce an empty
    result with the wrong columns, which is the kind of wrong that only shows
    up in a frontend rendering headers.

    **`Join` depends on the kind, and only some kinds collapse.** An `INNER`
    join with either side empty is empty, and so is a `SEMI`. A `LEFT` join
    with an empty *left* is empty, but with an empty *right* it still emits
    every left row padded with NULLs — collapsing that would delete rows the
    query asked for. `ANTI` with an empty right emits **all** of the left. So
    only the cases that are provably empty are taken, and the rest are left
    alone.

    `Aggregate` is deliberately **not** included: an ungrouped aggregate over
    zero rows produces one row (`count(*) = 0`, `sum = NULL`), not zero rows.
    Collapsing it would turn a valid answer into no answer at all.
    """

    comptime NAME = "PropagateEmpty"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if node.isa[Filter]():
            if node.get[Filter]().input[].copy().isa[EmptyRelation]():
                var out: DynRelation = EmptyRelation(
                    RecordBatch.empty(node.schema())
                )
                return out^
        if node.isa[Sort]():
            if node.get[Sort]().input[].copy().isa[EmptyRelation]():
                var out: DynRelation = EmptyRelation(
                    RecordBatch.empty(node.schema())
                )
                return out^
        if node.isa[Limit]():
            if node.get[Limit]().input[].copy().isa[EmptyRelation]():
                var out: DynRelation = EmptyRelation(
                    RecordBatch.empty(node.schema())
                )
                return out^
        if node.isa[Project]():
            if node.get[Project]().input[].copy().isa[EmptyRelation]():
                var out: DynRelation = EmptyRelation(
                    RecordBatch.empty(node.schema())
                )
                return out^
        return node.copy()


struct MergeProjects(Rule):
    """`Project(Project(x))` -> one projection.

    Fires only when **every** outer value is a bare pass-through of a column
    the inner projection produces, in which case the outer is doing nothing but
    selecting and reordering, and its selection can be answered from the
    inner's expressions directly.

    Restricting to pass-through outers is what keeps this sound and cheap. A
    computed outer — `Project(total * 2)` over `Project(qty * price AS total)`
    — would need the inner expression *substituted into* the outer one, which
    means rewriting inside a `DynValue` and would lower a fused comptime
    subtree into the runtime lane. Column selection needs no substitution.
    """

    comptime NAME = "MergeProjects"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Project]():
            return node.copy()
        ref outer = node.get[Project]()
        var input = outer.input[].copy()
        if not input.isa[Project]():
            return node.copy()
        ref inner = input.get[Project]()

        var outer_names = outer.names.copy()
        if not outer.passes_through_all(outer_names):
            return node.copy()

        var inner_names = inner.names.copy()
        var inner_values = inner.values.copy()
        var merged = List[DynValue](capacity=len(outer_names))
        for ref want in outer_names:
            var found = False
            for i in range(len(inner_names)):
                if inner_names[i] == want:
                    merged.append(inner_values[i].copy())
                    found = True
                    break
            if not found:
                # The outer names a column the inner does not produce, which
                # `Project` would have rejected — bail rather than guess.
                return node.copy()

        var out: DynRelation = Project(
            inner.input[].copy(), outer_names.copy(), merged^
        )
        return out^


struct RemoveSortBeforeAggregate(Rule):
    """`Aggregate(Sort(x))` -> `Aggregate(x)`.

    A sort feeding an aggregate is wasted work: every fold marrow has — `sum`,
    `product`, `mean`, `count`, `count_distinct`, `min`, `max`, the variance
    family — is order-insensitive, and there is no `first`/`last` in the kernel
    set that would not be. The aggregate's own output ordering is unaffected
    because it never depended on input order to begin with.

    This is a common frontend artifact: `order_by(...).aggregate(...)` is what
    a user writes when they mean to sort the *result*.

    **The honest caveat.** Floating-point addition is not associative, so
    consuming rows in a different order can change the last bits of a float
    `sum` or `mean`. That is not a new exposure — `GroupByOperator` already
    aggregates in parallel across morsels, so the summation order is not
    guaranteed by the unoptimized plan either. This rule does not introduce
    nondeterminism; it removes a sort that never constrained it.

    **Not applied when the sort carries a TopN bound**, which drops rows and so
    changes *which* rows are aggregated, not merely their order.
    """

    comptime NAME = "RemoveSortBeforeAggregate"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Aggregate]():
            return node.copy()
        ref agg = node.get[Aggregate]()
        var input = agg.input[].copy()
        if not input.isa[Sort]():
            return node.copy()
        ref sort = input.get[Sort]()
        if sort.limit:
            return node.copy()
        var out: DynRelation = Aggregate(
            sort.input[].copy(), agg.keys.copy(), agg.aggs.copy()
        )
        return out^


# ---------------------------------------------------------------------------
# Rules — reordering
# ---------------------------------------------------------------------------
struct PushFilterBelowSort(Rule):
    """`Filter(Sort(x))` -> `Sort(Filter(x))`.

    Always sound, and the contrast with `Limit` is the proof: sorting drops no
    rows, so the set surviving the filter is identical either way. What it buys
    is real — a sort buffers every morsel, so filtering first shrinks what is
    buffered and ordered.

    Note the direction. A `Filter` above a `Limit` may **not** be pushed below
    it: `limit(10)` then `filter(p)` means "the first ten rows, of which those
    matching p", where filtering first yields ten *matching* rows — a different
    and larger answer. That rule is absent deliberately.
    """

    comptime NAME = "PushFilterBelowSort"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref filter = node.get[Filter]()
        var input = filter.input[].copy()
        if not input.isa[Sort]():
            return node.copy()
        ref sort = input.get[Sort]()
        if sort.limit:
            # A `TopN` sort drops rows, so filtering below changes which rows
            # the bound keeps — the same hazard as `Limit`. `Sort.to_operator`
            # refuses to forward a pushdown for the same reason.
            return node.copy()
        var built: DynRelation = Sort(
            filter.with_input(sort.input[].copy()),
            sort.keys.copy(),
            sort.ascending.copy(),
            sort.nulls_first,
            None,
        )
        return built^


struct PushFilterBelowProject(Rule):
    """`Filter(Project(x))` -> `Project(Filter(x))`, for pass-through columns.

    The precondition is the whole rule. A predicate naming a *computed* output
    — `Filter(total > 100)` over `Project(qty * price AS total)` — cannot move
    below the node that defines `total`, because `total` does not exist in the
    input. `Project.passes_through_all` answers that by name and rejects anything
    renamed, cast or computed.
    """

    comptime NAME = "PushFilterBelowProject"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref filter = node.get[Filter]()
        var input = filter.input[].copy()
        if not input.isa[Project]():
            return node.copy()
        ref project = input.get[Project]()
        if not project.passes_through_all(filter.predicate.copy().columns()):
            return node.copy()
        var built: DynRelation = Project(
            filter.with_input(project.input[].copy()),
            project.names.copy(),
            project.values.copy(),
        )
        return built^


struct SplitConjunction(Rule):
    """`Filter(a AND b)` -> `Filter(a)` over `Filter(b)`.

    Stacked filters are not tidier — they are what lets every *other* filter
    rule work per conjunct. `PushFilterBelowJoin` cannot move `a AND b` when
    `a` names the left side and `b` the right; split, it moves `a` into the
    left and `b` into the right. `PushFilterBelowProject` cannot move a
    predicate that mentions one computed column; split, it moves the half that
    does not. And each conjunct carries its own `PrunePredicate`, where a
    compound `AND` prunes only as well as its weaker half.

    The split itself is decided at the `.filter()` verb, where the predicate's
    concrete type is visible, and each conjunct is boxed whole so a comptime
    subtree stays fused. Rebuilding through `.filter()` re-derives each
    conjunct's own pruner and constant for free.

    Answers unchanged, nulls included: a row survives `a AND b` under Kleene
    semantics exactly when both are `TRUE`, which is the row set two stacked
    filters keep — a null selects in neither.
    """

    comptime NAME = "SplitConjunction"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref f = node.get[Filter]()
        if len(f.conjuncts) < 2:
            return node.copy()
        var out = f.input[].copy()
        for ref c in f.conjuncts:
            out = out.filter(c.copy())
        return out^


struct PushFilterBelowJoin(Rule):
    """`Filter(Join(L, R))` -> the filter moved into whichever side it reads.

    The single most valuable reordering on a join-heavy workload: a predicate
    that only touches one input shrinks that input *before* it is hashed or
    probed, rather than after the join has already produced the rows it will
    throw away.

    **Only for `INNER`.** An outer join manufactures NULL rows for
    non-matches, and a predicate evaluated before that step never sees them —
    `LEFT JOIN ... WHERE r.x IS NULL` is the canonical anti-join idiom and
    pushing its predicate into the right side silently returns nothing. `SEMI`
    and `ANTI` are excluded for the same reason on the right, and are not worth
    a special case on the left.

    **The side is decided by name, not by position.** `Join` stores its keys as
    names and both children carry schemas, so "does this predicate read only
    left columns" is a set question with an exact answer. Positional indices
    could not answer it after any rewrite had touched a child, which is the
    defect that kept this rule out of the file until the keys changed.

    A predicate spanning *both* sides stays put here — but `SplitConjunction`
    runs first, so `a AND b` arrives as two filters and each half is placed
    independently. Only a genuinely inseparable predicate (`l.x + r.y > 5`)
    remains above the join.
    """

    comptime NAME = "PushFilterBelowJoin"

    @staticmethod
    def _reads_only(names: List[String], schema: Schema) -> Bool:
        for ref n in names:
            if schema.get_field_index(n) < 0:
                return False
        return True

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref f = node.get[Filter]()
        var input = f.input[].copy()
        if not input.isa[Join]():
            return node.copy()
        ref j = input.get[Join]()
        if j.kind != JOIN_INNER:
            return node.copy()

        var reads = f.predicate.columns()
        var left_schema = j.left[].schema()
        var right_schema = j.right[].schema()
        var left_only = Self._reads_only(reads, left_schema)
        var right_only = Self._reads_only(reads, right_schema)

        # A key column appears on both sides by name, so a predicate on one
        # could look like either. Ambiguity is left alone rather than guessed.
        if left_only == right_only:
            return node.copy()

        if left_only:
            var out: DynRelation = Join(
                f.with_input(j.left[].copy()),
                j.right[].copy(),
                left_names=j.left_keys.copy(),
                right_names=j.right_keys.copy(),
                kind=j.kind,
                strictness=j.strictness,
            )
            return out^
        var out: DynRelation = Join(
            j.left[].copy(),
            f.with_input(j.right[].copy()),
            left_names=j.left_keys.copy(),
            right_names=j.right_keys.copy(),
            kind=j.kind,
            strictness=j.strictness,
        )
        return out^


struct PushFilterBelowAggregate(Rule):
    """`Filter(Aggregate(x))` -> `Aggregate(Filter(x))`, for group keys only.

    `GROUP BY region ... WHERE region = 'west'` is written as a filter above
    the aggregate by every frontend that has a `HAVING`, and grouping every row
    before discarding most of them is pure waste. A predicate on a **group
    key** can move below, because grouping does not change a key's value — the
    rows that would have been grouped and then dropped are simply never
    grouped.

    **Only group keys.** A predicate naming an aggregate's *output* — `HAVING
    sum(x) > 100` — cannot move below the node that computes it, and neither
    can one naming a column the aggregate does not emit. Both are caught by
    requiring every column the predicate reads to be a group key by name.

    A grouped aggregate only. A **keyless** aggregate emits exactly one row, so
    a filter above it either keeps that row or drops it, and pushing the
    predicate down would filter the *input* instead — a different question with
    a different answer.
    """

    comptime NAME = "PushFilterBelowAggregate"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref f = node.get[Filter]()
        var input = f.input[].copy()
        if not input.isa[Aggregate]():
            return node.copy()
        ref agg = input.get[Aggregate]()
        if len(agg.keys) == 0:
            return node.copy()

        var key_names = List[String]()
        for ref k in agg.keys:
            var n = k.name()
            if n == "":
                return node.copy()  # an unnamed key cannot be matched by name
            key_names.append(n^)
        for ref want in f.predicate.columns():
            var found = False
            for ref have in key_names:
                if have == want:
                    found = True
                    break
            if not found:
                return node.copy()

        var out: DynRelation = Aggregate(
            f.with_input(agg.input[].copy()),
            agg.keys.copy(),
            agg.aggs.copy(),
        )
        return out^


struct PushLimitBelowProject(Rule):
    """`Limit(Project(x))` -> `Project(Limit(x))`.

    A projection is row- and order-preserving — one output row per input row,
    in order — so the window selects the same rows either side of it. Taking it
    first means the projection evaluates on `length` rows instead of all of
    them, which for `LIMIT 10` over a scan is ten evaluations against every one.

    **Only when no projected value is an aggregate**, which would collapse its
    input to one row and make a limit below it bound the wrong thing. `Project`
    rejects aggregates at construction, so this is belt-and-braces.
    """

    comptime NAME = "PushLimitBelowProject"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Limit]():
            return node.copy()
        ref limit = node.get[Limit]()
        var input = limit.input[].copy()
        if not input.isa[Project]():
            return node.copy()
        ref project = input.get[Project]()
        if project.computes_an_aggregate():
            return node.copy()
        var built: DynRelation = Project(
            Limit(project.input[].copy(), limit.offset, limit.length),
            project.names.copy(),
            project.values.copy(),
        )
        return built^


# ---------------------------------------------------------------------------
# Rules — reparameterization
# ---------------------------------------------------------------------------
struct TopN(Rule):
    """`Limit(Sort(x))` -> `Limit(Sort(x, top=offset+length))`.

    A full sort of N rows to return ten is the most wasteful shape a plan
    reaches, and `sort_indices` has accepted `limit=` all along —
    `SortOperator` never passed one, because only the plan knows whether a
    bound is safe.

    **The bound is `offset + length`, not `length`**: the `Limit` above still
    skips `offset` rows of the ordered result, so the sort must retain
    everything to the end of that window.

    **The `Limit` stays.** It applies the offset and reports `done` to stop the
    source early; the bound is an optimization under it, not a replacement.

    Only a `Limit` *directly* above a `Sort` matches. Anything between them that
    drops rows runs after the sort, so a k-row sort feeds it fewer than k and
    the query silently returns too few. Requiring adjacency makes that
    unrepresentable rather than merely avoided — and `PushFilterBelowSort` runs
    first, moving the common offender out of the way.
    """

    comptime NAME = "TopN"

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Limit]():
            return node.copy()
        ref limit = node.get[Limit]()
        if limit.length < 0:
            return node.copy()
        var input = limit.input[].copy()
        if not input.isa[Sort]():
            return node.copy()
        ref sort = input.get[Sort]()

        var bound = limit.offset + limit.length
        if sort.limit and sort.limit.value() <= bound:
            return node.copy()  # already this tight — do not loop forever
        var built: DynRelation = Limit(
            Sort(
                sort.input[].copy(),
                sort.keys.copy(),
                sort.ascending.copy(),
                sort.nulls_first,
                bound,
            ),
            limit.offset,
            limit.length,
        )
        return built^


# ---------------------------------------------------------------------------
# Column pruning — the one pass that travels downward
# ---------------------------------------------------------------------------
struct ColumnPruning(Copyable, Movable):
    """Narrow every source to the columns the plan above it actually reads.

    **Measured by this project at 3.6x**, against 1.04x for row-group pruning —
    the most valuable rewrite in the file, and the only one that is not a
    `Rule`. Every other rule matches a shape and rebuilds it locally; this one
    needs to know what the *whole plan above* a node reads, which is
    information that only exists travelling from the root down.

    So it is a second traversal with an accumulator, not another entry in
    `AllRules`. The accumulator is the set of column names still needed:

    - `Project` and `Aggregate` **replace** it — they name their inputs
      explicitly, and nothing above them can reach a column they do not emit.
    - `Filter` and `Sort` **widen** it: the rows they read are needed *in
      addition* to whatever the consumer wanted.
    - `Limit` passes it through untouched.
    - `Join` widens it with both key sets, because a key is read even when it
      is not emitted.
    - the sources **consume** it: a `ParquetScan` narrows its schema, an
      `InMemoryTable` selects its columns.

    **The empty set is never pushed to a source**, and that is a correctness
    rule rather than an optimization. `count_star()` desugars to
    `lit(1, int64).count()`, whose `columns()` is empty, so a plan that is
    nothing but `COUNT(*)` demands no columns at all — and a `RecordBatch`
    carries its row count in its columns, so a zero-column batch reports
    `num_rows() == 0` and every row of the query silently disappears. When the
    demand is empty a source keeps its first column, which is the narrowest
    thing that still counts.
    """

    @staticmethod
    def _widened(var into: List[String], extra: List[String]) -> List[String]:
        for ref name in extra:
            var seen = False
            for ref have in into:
                if have == name:
                    seen = True
                    break
            if not seen:
                into.append(name.copy())
        return into^

    @staticmethod
    def _narrowed(schema: Schema, needed: List[String]) -> List[String]:
        """`needed`, restricted to what `schema` has and in *its* order.

        Order matters: a source must not reorder its own columns just because
        a consumer happened to name them differently, or every positional
        reference above it moves.
        """
        var out = List[String]()
        for ref f in schema.fields:
            for ref want in needed:
                if f.name == want:
                    out.append(f.name.copy())
                    break
        if len(out) == 0 and len(schema.fields) > 0:
            out.append(schema.fields[0].name.copy())
        return out^

    @staticmethod
    def apply(node: DynRelation, needed: List[String]) raises -> DynRelation:
        """`node`, with its sources narrowed to `needed`."""
        if node.isa[ParquetScan]():
            ref scan = node.get[ParquetScan]()
            var keep = Self._narrowed(scan.schema(), needed)
            if len(keep) == len(scan.schema().fields):
                return node.copy()
            var fields = List[Field](capacity=len(keep))
            for ref name in keep:
                fields.append(scan.schema().field(name=name).copy())
            var out: DynRelation = ParquetScan(
                scan.path.copy(), schema(fields^)
            )
            return out^

        if node.isa[InMemoryTable]():
            ref src = node.get[InMemoryTable]()
            var keep = Self._narrowed(src.schema(), needed)
            if len(keep) == len(src.schema().fields):
                return node.copy()
            var out: DynRelation = InMemoryTable(src.batch.select(keep))
            return out^

        if node.isa[Filter]():
            ref f = node.get[Filter]()
            var below = Self._widened(needed.copy(), f.predicate.columns())
            var out: DynRelation = f.with_input(Self.apply(f.input[], below))
            return out^

        if node.isa[Sort]():
            ref t = node.get[Sort]()
            var below = needed.copy()
            for ref k in t.keys:
                below = Self._widened(below^, k.columns())
            var out: DynRelation = Sort(
                Self.apply(t.input[], below),
                t.keys.copy(),
                t.ascending.copy(),
                t.nulls_first,
                t.limit,
            )
            return out^

        if node.isa[Limit]():
            ref l = node.get[Limit]()
            var out: DynRelation = Limit(
                Self.apply(l.input[], needed), l.offset, l.length
            )
            return out^

        if node.isa[Project]():
            ref p = node.get[Project]()
            # A projection *replaces* the demand: only the columns its own
            # values read can matter below it.
            var below = List[String]()
            for ref v in p.values:
                below = Self._widened(below^, v.columns())
            var out: DynRelation = Project(
                Self.apply(p.input[], below), p.names.copy(), p.values.copy()
            )
            return out^

        if node.isa[Aggregate]():
            ref a = node.get[Aggregate]()
            var below = List[String]()
            for ref k in a.keys:
                below = Self._widened(below^, k.columns())
            for ref g in a.aggs:
                below = Self._widened(below^, g.columns())
            var out: DynRelation = Aggregate(
                Self.apply(a.input[], below), a.keys.copy(), a.aggs.copy()
            )
            return out^

        if node.isa[Join]():
            ref j = node.get[Join]()
            # Keys are read even when they are not emitted, so both sides'
            # keys join the demand before it descends. Names, not indices —
            # which is the whole reason `Join` stores names.
            var below = Self._widened(needed.copy(), j.left_keys)
            below = Self._widened(below^, j.right_keys)
            var out: DynRelation = Join(
                Self.apply(j.left[], below),
                Self.apply(j.right[], below),
                left_names=j.left_keys.copy(),
                right_names=j.right_keys.copy(),
                kind=j.kind,
                strictness=j.strictness,
            )
            return out^

        return node.copy()


# ---------------------------------------------------------------------------
# Rule sets
# ---------------------------------------------------------------------------
trait RuleSet(Copyable, Movable):
    """A comptime-selected list of rules, applied in order to a fixpoint.

    Comptime so a binary links exactly the rules it names. `execute()` on its
    own optimizes nothing.
    """

    @staticmethod
    def prepare(plan: DynRelation) raises -> DynRelation:
        """Whatever this set wants done **once**, before the rewrite loop.

        Column pruning lives here rather than in `rewrite` because it is a
        downward pass with an accumulator, where every `Rule` is a local
        bottom-up match — running it to a fixpoint would re-derive the same
        answer every time.

        A hook returning a plan rather than a `Bool` the driver branches on:
        neither `Self.R.PRUNE_COLUMNS` nor `Self.R.prune_columns()` resolves
        off a struct parameter inside a `comptime if`. A plain method call
        sidesteps that, and DCE still holds — a rule set whose `prepare` is the
        identity never mentions `ColumnPruning`, so nothing links it.
        """
        ...

    @staticmethod
    def rewrite(node: DynRelation) raises -> DynRelation:
        """Every rule in this set, applied to one node."""
        ...


struct NoRules(RuleSet):
    """The identity — `optimize[NoRules]()` returns the plan unchanged, which
    makes it the control arm of an equivalence test."""

    @staticmethod
    def prepare(plan: DynRelation) raises -> DynRelation:
        return plan.copy()

    @staticmethod
    def rewrite(node: DynRelation) raises -> DynRelation:
        return node.copy()


struct AllRules(RuleSet):
    """Every rule in this file.

    **Order is chosen, not incidental.** Removals run before reorderings, so no
    rule bothers moving a node another is about to delete, and
    `PushFilterBelowSort` runs before `TopN` so a filter between a limit and a
    sort is relocated *before* `TopN` checks adjacency and gives up.
    """

    @staticmethod
    def prepare(plan: DynRelation) raises -> DynRelation:
        """Seeded from the plan's **own output schema** — the columns a caller
        can actually observe. Anything else is dead by definition, however deep
        the plan is."""
        var wanted = List[String]()
        for ref f in plan.schema().fields:
            wanted.append(f.name.copy())
        return ColumnPruning.apply(plan, wanted)

    @staticmethod
    def rewrite(node: DynRelation) raises -> DynRelation:
        """Each rule in turn, every one seeing the previous rule's output.

        Composing rather than short-circuiting on the first match is what lets
        a single pass do real work: `PushFilterBelowSort` relocates a filter and
        `TopN` immediately sees the `Limit` and `Sort` it left adjacent. Rules
        answer unchanged when they do not apply, so threading the value through
        all of them is free.
        """
        var out = EliminateFilter.apply(node)
        out = RemoveEmptyLimit.apply(out)
        out = PropagateEmpty.apply(out)
        out = RemoveNoOpProject.apply(out)
        out = MergeProjects.apply(out)
        out = RemoveSortBeforeAggregate.apply(out)
        out = MergeLimits.apply(out)
        out = RemoveRedundantSort.apply(out)
        out = SplitConjunction.apply(out)
        out = PushFilterBelowProject.apply(out)
        out = PushFilterBelowSort.apply(out)
        out = PushFilterBelowJoin.apply(out)
        out = PushFilterBelowAggregate.apply(out)
        out = PushLimitBelowProject.apply(out)
        return TopN.apply(out)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
struct Optimizer[R: RuleSet](Copyable, Movable):
    """Applies `R` to a plan until nothing changes.

    A type rather than a pair of free functions, so the traversal, the pass cap
    and the convergence test are one thing with one name — and so the recursion
    reads as `Self.rewrite` rather than as mutually-referring module-level
    helpers.
    """

    comptime MAX_PASSES = 16
    """How many whole-plan passes before stopping.

    A backstop, not a budget: every rule shrinks the plan or moves a node
    downward, so the fixpoint arrives in a pass or two. It exists because a
    future rule that grows a plan, or two that undo each other, would otherwise
    spin with no diagnostic — and a half-optimized plan is a slow answer, where
    a hang is no answer at all.
    """

    @staticmethod
    def _rewritten_children(node: DynRelation) raises -> DynRelation:
        """`node` with every child already rewritten.

        Bottom-up, because every rule reads its child's type: rewriting
        children first means a rule sees the child's *final* form, so
        `Limit(Sort(Filter(...)))` collapses in one pass instead of waiting for
        the fixpoint to rediscover it.

        `traverse` is what makes this three lines instead of an arm per node —
        a node knows its own children and how to put itself back together, so
        adding a relation adds no code here. It also means `Aggregate` and
        `Join` are descended into, which a hand-written ladder had quietly
        skipped.
        """

        def descend(child: DynRelation) raises {imm} -> DynRelation:
            return Self.rewrite(child)

        return node.traverse(descend)

    @staticmethod
    def rewrite(node: DynRelation) raises -> DynRelation:
        """One bottom-up pass over `node`: children first, then `R`'s rules."""
        var current = Self._rewritten_children(node)
        return Self.R.rewrite(current)

    @staticmethod
    def run(plan: DynRelation) raises -> DynRelation:
        """`plan` rewritten until it stops changing.

        **Convergence is detected on the rendered plan**, not on a change flag
        threaded through the rules. A plan is `Writable` and its rendering is
        total and structural, so two passes producing the same string produced
        the same plan — which makes the check independent of whether every rule
        remembered to report that it fired. A rule that lies costs one extra
        pass here instead of an infinite loop, and it is also why a rule
        returns the node unchanged rather than an `Optional`.
        """
        var current = plan.copy()
        current = Self.R.prepare(current)
        var rendered = String(current)
        for _ in range(Self.MAX_PASSES):
            var next = Self.rewrite(current)
            var next_rendered = String(next)
            if next_rendered == rendered:
                return next^
            current = next^
            rendered = next_rendered^
        return current^


def optimize[R: RuleSet](plan: DynRelation) raises -> DynRelation:
    """`plan` rewritten by `R`. The one free function here, because it is the
    entry point `DynRelation.optimize` forwards to."""
    return Optimizer[R].run(plan)


# ---------------------------------------------------------------------------
# What is deliberately absent
# ---------------------------------------------------------------------------
#
# Recorded rather than left as a silent gap, because two of these look like
# oversights:
#
# - **Merging `Filter(Filter(x))`.** It needs the conjunction of two erased
#   `DynValue`s, and the box exposes no way to build one without lowering a
#   comptime predicate into the runtime lane, which would discard the fusion
#   that lane exists for. Stacked filters already prune and evaluate
#   identically, so the merge is cosmetic.
#
# - **Pushing a predicate below a `Join`.** Now *expressible* — a rule can
#   construct the two `Filter`s — but still unsafe: `Join` keys are positional
#   `List[Int]` indices into each child's schema, so any rewrite that changes a
#   child renumbers them silently. Fix the keys first
#   (fixed 2026-08-31: `Join` stores names, resolved from the caller's
#   indices at construction and back at lowering).
#
# - **Projection pushdown / column pruning.** The highest-value rule missing,
#   measured by this project at 3.6x against pruning's 1.04x. It needs a
#   *downward* needed-column set rather than a bottom-up rewrite, so it is a
#   second traversal rather than another entry above, and it carries a
#   correctness trap that must land with it: `count_star()` desugars to
#   `lit(1).count()`, whose `columns()` is empty, so a naive needed-set prunes
#   every column and `RecordBatch.num_rows()` then reports 0.
#
# - **Constant folding.** Belongs in the `RuntimeValue` constructors, where
#   `and_(x, lit(False))` folds as it is built — not in a plan rule that would
#   have to inspect inside a `DynValue`.
