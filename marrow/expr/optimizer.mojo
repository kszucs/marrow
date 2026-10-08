# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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
  argument at their definition. The order a *join* emits is not part of any
  answer — it follows whichever input is probed — so the join search in
  `finish` changes it below a `Limit` or a `Window` as freely as anywhere
  else.

  **The third case is why only one rule reads a `Window`.**
  `WindowOperator._permutation` returns the identity when a window names no
  keys, so `ROW_NUMBER() OVER ()` reads its answer off input row order — and
  an ordered window is **not** safe either, because that permutation is
  stable, so input order still decides ties. `row_number`, `lag`,
  `lead`, `first_value` and `last_value` all change answer if tie order
  changes, and `RemoveRedundantSort` trades tie order away by design. So no
  rule moves a node into a window's input: `MergeWindows` folds stacked
  windows together without moving a row, no other rule matches a `Window`, and
  `test_optimizer.mojo::test_no_rule_moves_a_node_into_a_window` pins that.
  A rule that does would have to prove it preserves the order a window below
  it may be reading. SQL calls `ROW_NUMBER() OVER ()` nondeterministic, so
  this is an invariant to keep rather than a wrong answer to fix.
"""

from std.collections import Dict, Set
from std.memory import ArcPointer

from ..kernels.join import (
    BUILD_LEFT,
    BUILD_RIGHT,
    JOIN_ALL,
    JOIN_INNER,
    JoinBuildSide,
    JoinKind,
)

from ..schema import Field, Schema, schema
from ..tabular import RecordBatch
from .estimates import Cost, Estimate
from .physical import JoinOrder, PlannedJoin
from ..errors import InternalError, KeyError
from .logical import (
    Aggregate,
    JoinLink,
    JoinPricing,
    JoinRules,
    JoinChain,
    JoinFilter,
    JoinRef,
    DynRelation,
    DynValue,
    EmptyRelation,
    FileScan,
    Filter,
    IcebergScan,
    InMemoryTable,
    IpcScan,
    JsonScan,
    Limit,
    ParquetScan,
    Difference,
    Intersection,
    Project,
    Sort,
    Union,
    Window,
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

    **A rule has no name field.** There was one, declared on this trait and
    spelled out by all eighteen conformers, and nothing ever read it; when a
    rule does need naming, `reflect[Self].name()` derives it. A trait
    requirement no caller consumes is eighteen places to keep in sync with
    nothing.
    """

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        ...


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

    **The set relations collapse when their answer is provably empty**: a
    `Union` with both sides empty, a `Difference` with an empty left, an
    `Intersection` with either. A `Union` with one side empty is *not*
    replaced by the other side, because the output takes the left side's
    names.

    `Aggregate` is deliberately **not** included: an ungrouped aggregate over
    zero rows produces one row (`count(*) = 0`, `sum = NULL`), not zero rows.
    Collapsing it would turn a valid answer into no answer at all.
    """

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        var empty = False
        if node.isa[Filter]():
            empty = node.get[Filter]().input[].isa[EmptyRelation]()
        elif node.isa[Sort]():
            empty = node.get[Sort]().input[].isa[EmptyRelation]()
        elif node.isa[Limit]():
            empty = node.get[Limit]().input[].isa[EmptyRelation]()
        elif node.isa[Project]():
            empty = node.get[Project]().input[].isa[EmptyRelation]()
        elif node.isa[Union]():
            ref u = node.get[Union]()
            empty = (
                u.left[].isa[EmptyRelation]() and u.right[].isa[EmptyRelation]()
            )
        elif node.isa[Intersection]():
            ref i = node.get[Intersection]()
            empty = (
                i.left[].isa[EmptyRelation]() or i.right[].isa[EmptyRelation]()
            )
        elif node.isa[Difference]():
            empty = node.get[Difference]().left[].isa[EmptyRelation]()
        if empty:
            var out: DynRelation = EmptyRelation(
                RecordBatch.empty(node.schema())
            )
            return out^
        else:
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
    `sum` or `mean`. That is not a new exposure — `GroupedAggregateOperator`
    already aggregates in parallel across morsels, so the summation order is
    not guaranteed by the unoptimized plan either. This rule does not introduce
    nondeterminism; it removes a sort that never constrained it.

    **Not applied when the sort carries a TopN bound**, which drops rows and so
    changes *which* rows are aggregated, not merely their order.
    """

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
        var out: DynRelation = agg.with_input(sort.input[].copy())
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
    rule work per conjunct. `PushFilterIntoJoin` cannot move `a AND b` into a
    participant when `a` names one input and `b` another; split, it moves each
    into the input it names. `PushFilterBelowProject` cannot move a
    predicate that mentions one computed column; split, it moves the half that
    does not. And each conjunct prunes on its own, where a compound `AND`
    prunes only as well as its weaker half.

    The split itself is decided at the `.filter()` verb, where the predicate's
    concrete type is visible, and each conjunct is boxed whole so a comptime
    subtree stays fused. Rebuilding through `.filter()` re-derives each
    conjunct's own constant for free.

    Answers unchanged, nulls included: a row survives `a AND b` under Kleene
    semantics exactly when both are `TRUE`, which is the row set two stacked
    filters keep — a null selects in neither.
    """

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


struct PushFilterIntoJoin(Rule):
    """`Filter(JoinChain)` -> the filter inside the chain: onto the participant
    it reads, or among the chain's own filters.

    The single most valuable reordering on a join-heavy workload: a predicate
    that only touches one input shrinks that input *before* it is hashed or
    probed, rather than after the join has already produced the rows it will
    throw away.

    **Onto a participant** when the predicate reads one participant under that
    participant's own column names, and the participant is never padded with
    NULLs nor picks a `JOIN_ANY` match (`JoinRules.holds`) —
    `LEFT JOIN ... WHERE r.x IS NULL` is the canonical anti-join idiom, and
    pushing its predicate into the right side would silently return nothing.

    **Among the chain's filters** otherwise: a predicate spanning
    participants, one reading a renamed output, or one reading no column at
    all. The chain evaluates it after the lowest join of its tree holding its
    columns that it may cross — whichever tree the planner chose — and never
    after a join appended later.

    **Above** when it reads a column the chain does not answer with. The
    predicate is never rewritten: it reads its columns by the names it was
    written with, and the chain supplies them.
    """

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref f = node.get[Filter]()
        var input = f.input[].copy()
        if not input.isa[JoinChain]():
            return node.copy()
        ref chain = input.get[JoinChain]()
        var names = chain.names()
        var refs = List[JoinRef]()
        var renamed = False
        for ref name in f.predicate.columns():
            if name not in names:
                return node.copy()
            var r = chain.ref_of(name)
            renamed = renamed or r.name != name
            refs.append(r^)
        var filter = JoinFilter(
            f.predicate.copy(), refs^, len(chain.inputs) - 1
        )
        var reads = filter.participants()
        # Onto the participant it reads when every tree may filter that
        # participant alone first.
        if (
            len(reads) == 1
            and not renamed
            and chain.rules().holds(filter, reads)
        ):
            var p = filter.refs[0].input
            var inputs = chain.inputs.copy()
            var moved: DynRelation = f.with_input(inputs[p][].copy())
            inputs[p] = ArcPointer(moved^)
            return input.with_chain(chain.with_inputs(inputs^))
        var filters = chain.filters.copy()
        filters.append(filter^)
        return input.with_chain(chain.with_filters(filters^))


struct MergeJoinChains(Rule):
    """`JoinChain` over a participant that is itself a `JoinChain` -> one
    chain.

    The `join` verb takes its right side as one participant, a chain
    included, as ibis does — so `a.join(b.join(c))` nests, and so does a
    filtered chain once `PushFilterIntoJoin` has taken its filter in. Spliced,
    the nested links join the outer chain's (`inlined`) and one
    planner sees every join: the bushy spelling reorders exactly as the
    left-deep one does. A chain whose joins must keep their place is not
    spliced.
    """

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[JoinChain]():
            return node.copy()
        ref chain = node.get[JoinChain]()
        for p in range(len(chain.inputs)):
            ref input = chain.inputs[p][]
            if not input.isa[JoinChain]():
                continue
            var order = Self.layout(chain, p)
            if len(order) == len(input.get[JoinChain]().inputs):
                return node.with_chain(Self.inlined(chain, p, order))
        return node.copy()

    @staticmethod
    def layout(chain: JoinChain, p: Int) raises -> List[Int]:
        """The order participant `p`'s own participants are laid out in when
        `p`, itself a join chain, is spliced — short of all of them when it
        cannot be.

        At participant 0 they keep their order: the nested links come first,
        as they were. Anywhere else only when the link joining `p` and every
        nested link are inner and `JOIN_ALL`, the joins that may run in any
        order: each, lowest first, the first with a key pair into one already
        placed — the participants before `p` included, through the link
        joining `p`.
        """
        ref inner = chain.inputs[p][].get[JoinChain]()
        var order = List[Int](capacity=len(inner.inputs))
        if p == 0:
            for i in range(len(inner.inputs)):
                order.append(i)
            return order^
        if not chain.links[p - 1].is_inner():
            return order^
        for ref l in inner.links:
            if not l.is_inner():
                return order^
        var reach = Set[Int]()
        for ref r in chain.links[p - 1].right_keys:
            reach.add(inner.ref_of(r.name).input)
        var laid = Set[Int]()
        while len(order) < len(inner.inputs):
            var next = -1
            for i in range(len(inner.inputs)):
                if next < 0 and i in reach and i not in laid:
                    next = i
            if next < 0:
                return order^
            order.append(next)
            laid.add(next)
            for j in range(len(inner.links)):
                for ref r in inner.links[j].left_keys:
                    if r.input == next:
                        reach.add(j + 1)
                    if j + 1 == next:
                        reach.add(r.input)
        return order^

    @staticmethod
    def inlined(chain: JoinChain, p: Int, order: List[Int]) raises -> JoinChain:
        """`chain` with participant `p`, itself a join chain, replaced by that
        chain's participants, laid out in `order` (`layout`), and its links —
        one chain, so one planner sees every join in it. Past participant 0
        each key pair is compared by the link of its later participant.

        The answer does not move. Nested filters come along — bounded by the
        last nested participant, since every join they cross is inner — and
        whatever read one of `p`'s outputs reads the participant column
        behind it.
        """
        ref inner = chain.inputs[p][].get[JoinChain]()
        var k = len(inner.inputs)
        var at = List[Int](length=k, fill=0)
        for i in range(k):
            at[order[i]] = p + i

        var inputs = List[ArcPointer[DynRelation]](
            capacity=len(chain.inputs) + k - 1
        )
        for i in range(p):
            inputs.append(chain.inputs[i].copy())
        for i in order:
            inputs.append(inner.inputs[i].copy())
        for i in range(p + 1, len(chain.inputs)):
            inputs.append(chain.inputs[i].copy())

        var links = List[JoinLink](capacity=len(inputs) - 1)
        for i in range(p - 1):
            links.append(chain.links[i].copy())
        if p == 0:
            links.extend(inner.links.copy())
        else:
            # Every key pair of the link joining `p` and of the nested links,
            # each compared by the link of its later participant.
            ref outer = chain.links[p - 1]
            var left = outer.left_keys.copy()
            var right = Self.spliced(chain, outer.right_keys, p, at)
            for ref l in inner.links:
                left.extend(Self.nested(l.left_keys, at))
                right.extend(Self.nested(l.right_keys, at))
            for q in range(p, p + k):
                var lk = List[JoinRef]()
                var rk = List[JoinRef]()
                for i in range(len(left)):
                    if right[i].input == q and left[i].input < q:
                        lk.append(left[i].copy())
                        rk.append(right[i].copy())
                    elif left[i].input == q and right[i].input < q:
                        lk.append(right[i].copy())
                        rk.append(left[i].copy())
                var nested = order[q - p]
                var side = (
                    outer.build_side if nested
                    == 0 else inner.links[nested - 1].build_side
                )
                links.append(JoinLink(JOIN_INNER, JOIN_ALL, side, lk^, rk^))
        for i in range(p, len(chain.links)):
            ref l = chain.links[i]
            links.append(
                JoinLink(
                    l.kind,
                    l.strictness,
                    l.build_side,
                    Self.spliced(chain, l.left_keys, p, at),
                    Self.spliced(chain, l.right_keys, p, at),
                )
            )

        var filters = List[JoinFilter](
            capacity=len(inner.filters) + len(chain.filters)
        )
        for ref f in inner.filters:
            filters.append(
                f.with_bounds(
                    Self.nested(f.refs, at), f.bound if p == 0 else p + k - 1
                )
            )
        for ref f in chain.filters:
            filters.append(
                f.with_bounds(
                    Self.spliced(chain, f.refs, p, at),
                    f.bound + k - 1 if f.bound >= p else f.bound,
                )
            )
        return JoinChain(
            inputs^,
            links^,
            Self.spliced(chain, chain.refs, p, at),
            chain.names(),
            filters^,
        )

    @staticmethod
    def nested(refs: List[JoinRef], at: List[Int]) -> List[JoinRef]:
        """A nested chain's columns, where their participants now stand."""
        var out = List[JoinRef](capacity=len(refs))
        for ref r in refs:
            out.append(JoinRef(at[r.input], r.name.copy()))
        return out^

    @staticmethod
    def spliced(
        chain: JoinChain, refs: List[JoinRef], p: Int, at: List[Int]
    ) raises -> List[JoinRef]:
        """Outer columns once participant `p` is spliced: a column of `p`
        becomes the participant column behind that output."""
        ref inner = chain.inputs[p][].get[JoinChain]()
        var out = List[JoinRef](capacity=len(refs))
        for ref r in refs:
            if r.input == p:
                var column = inner.ref_of(r.name)
                out.append(JoinRef(at[column.input], column.name.copy()))
            elif r.input < p:
                out.append(r.copy())
            else:
                out.append(JoinRef(r.input + len(at) - 1, r.name.copy()))
        return out^


struct MergeProjectIntoJoin(Rule):
    """`Project(JoinChain)` of bare column reads -> the chain answering with
    them.

    `select`, `rename`, `drop` and a `project` of reads over a chain all
    leave one, and so does SQL's `SELECT` over a `WHERE` once
    `PushFilterIntoJoin` has taken the filter in. A projection that only reads
    columns is the chain's own output under other names, one node rather than
    two. A value that computes, or two sharing a name, keep the projection:
    a read is a value naming exactly the one column it reads, and an aliased
    aggregate names its alias, so it is not one.
    """

    # Folded here rather than by the verbs, so a binary that projects and
    # never optimizes does not link the chain.

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Project]():
            return node.copy()
        ref p = node.get[Project]()
        var input = p.input[].copy()
        if not input.isa[JoinChain]():
            return node.copy()
        ref chain = input.get[JoinChain]()
        var names = chain.names()
        var refs = List[JoinRef](capacity=len(p.values))
        for ref v in p.values:
            var read = v.read_column()
            if not read:
                return node.copy()
            if not (read.value() in names):
                return node.copy()
            refs.append(chain.ref_of(read.value()))
        for i in range(len(p.names)):
            for j in range(i):
                if p.names[i] == p.names[j]:
                    return node.copy()
        return input.with_chain(chain.with_output(refs^, p.names))


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

        var out: DynRelation = agg.with_input(f.with_input(agg.input[].copy()))
        return out^


struct PushFilterIntoScan(Rule):
    """`Filter(ParquetScan)` -> the same filter over a scan that prunes.

    **The filter stays.** Pruning is conservative — it answers "could any row
    of this chunk match", never "every row matches" — so the exact predicate
    must still run on every row the scan produces. This rule makes the scan
    read less, and it is the `Filter` above that decides what survives. That is
    what makes it safe to get wrong: a pruner that should not be here costs
    time, and dropping one costs nothing but time either.

    **Reachability is the correctness argument.** `Filter(p)` above `Limit(10)`
    means "the first ten rows, then `p`" — a scan that skipped a row group
    would hand `Limit` a *different* first ten, and rows the query should have
    returned disappear. `_grown` descends through `Filter` and nothing else, so
    `Limit`, `Window`, `Project`, `Aggregate` and `Join` stop it by *being
    themselves*. The retired `to_operator` descent had to remember to clear at
    each of the five, one arm at a time, and adding a node meant remembering
    again; here a new node stops the predicate unless somebody deliberately
    teaches this method to walk it.

    The rules that legitimately move a filter closer to a scan —
    `PushFilterBelowSort`, `PushFilterBelowProject`, `PushFilterBelowAggregate`,
    `PushFilterIntoJoin` — each prove their own case first, and all four
    rebuild with `with_input`, so the pruner they carry still names the columns
    it named above. That is where this rule gets reach the descent never had:
    the descent cleared at `Project`, `Aggregate` and `Join` unconditionally.

    **Ordered after `SplitConjunction`** so it lands each conjunct separately:
    a compound `AND` prunes only as well as its weaker half. The predicate it
    reads is the `Filter`'s own `DynValue`, boxed or not, so a split's output
    prunes exactly as well as the filter it came from.
    """

    @staticmethod
    def _grown(
        node: DynRelation, predicate: DynValue
    ) raises -> Optional[DynRelation]:
        """`node` with `predicate` added to the `ParquetScan` at the bottom of
        its `Filter` chain, or `None` when there is no scan down there.

        **`Filter` is the one node this descends through**, and it is what
        keeps `Filter(a, Filter(b, scan))` pruning on both halves — the shape
        the retired `to_operator` descent conjoined and plain adjacency would
        miss. It is sound for the same reason the outer filter is: a chunk
        that cannot match `predicate` holds no row this filter keeps, and the
        filters in between only ever drop more.

        Every other node stops it by not being a `Filter` — `Limit`, `Window`,
        `Project`, `Aggregate` and `Join`, which is the whole of the table the
        descent used to spell out arm by arm.

        **A scan that already carries this predicate answers `None`**, which is
        what makes the rule idempotent: the filter above stays, so without the
        check a second pass would push the same predicate again and the driver
        would never see two renderings agree.
        """
        if node.isa[ParquetScan]():
            ref source = node.get[ParquetScan]()
            for ref carried in source.pruners:
                if String(carried) == String(predicate):
                    return None
            # A new list rather than a mutated one: the plan being rewritten
            # still holds the scan this came from, and the driver compares the
            # two renderings to decide it has converged.
            var pruners = source.pruners.copy()
            pruners.append(predicate.copy())
            var grown = ParquetScan(
                source.path.copy(),
                source.schema(),
                pruners^,
                source.statistics.copy(),
            )
            var out: DynRelation = grown^
            return out^
        if node.isa[IcebergScan]():
            ref source = node.get[IcebergScan]()
            for ref carried in source.pruners():
                if String(carried) == String(predicate):
                    return None
            var pruners = source.pruners().copy()
            pruners.append(predicate.copy())
            var out: DynRelation = source.with_pruners(pruners^)
            return out^
        if node.isa[Filter]():
            ref inner = node.get[Filter]()
            var below = Self._grown(inner.input[].copy(), predicate)
            if below:
                var out: DynRelation = inner.with_input(below.value().copy())
                return out^
        return None

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Filter]():
            return node.copy()
        ref filtered = node.get[Filter]()
        var below = Self._grown(filtered.input[].copy(), filtered.predicate)
        if not below:
            return node.copy()
        var out: DynRelation = filtered.with_input(below.value().copy())
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
# Rules — sharing a sort
# ---------------------------------------------------------------------------
struct MergeWindows(Rule):
    """`Window(Window(x))` -> one `Window`, when the outer reads nothing the
    inner adds.

    `with_columns` builds a `Window` per window value, and each one sorts. A
    node sorts once per distinct spec among its values, so folding stacked
    nodes together is what lets `rank()` and `dense_rank()` over one ordering
    share a sort. Bottom-up and to a fixpoint, a whole chain of independent
    window values folds into one node.

    **Sound because a window leaves its rows in input order**, so both nodes
    saw the same rows in the same order, and an outer value that reads none of
    the inner's columns computes the same answer beside them. The merged node
    appends the inner's columns, then the outer's — the order they had.
    """

    @staticmethod
    def apply(node: DynRelation) raises -> DynRelation:
        if not node.isa[Window]():
            return node.copy()
        ref outer = node.get[Window]()
        var input = outer.input[].copy()
        if not input.isa[Window]():
            return node.copy()
        ref inner = input.get[Window]()
        for ref v in outer.values:
            for ref c in v.columns():
                for ref n in inner.names:
                    if c == n:
                        return node.copy()
        var names = inner.names.copy()
        names.extend(outer.names.copy())
        var values = inner.values.copy()
        values.extend(outer.values.copy())
        var out: DynRelation = Window(inner.input[].copy(), names^, values^)
        return out^


# ---------------------------------------------------------------------------
# Rules — cost-based: join order
#
# A `JoinChain`'s links say what a join answers; this chooses the tree that
# computes it: the cheapest cross-product-free tree its `JoinRules` allows,
# kept only when strictly cheaper than the written order, every join's build
# side picked. The result is attached to the chain (`JoinChain.with_order`),
# which checks it (`JoinOrder.verify`); `JoinRules` says why every such tree
# returns the same answer.
#
# The search is exact under the model because a set of participants has one
# estimate whatever tree produced it (`JoinPricing.estimate`), and every tree
# is priced by one definition, `JoinPricing`.
#
# `EnumerateCsg` / `EnumerateCmp` (Moerkotte & Neumann, *Analysis of Two
# Existing and One New Dynamic Programming Algorithm for the Generation of
# Optimal Bushy Join Trees without Cross Products*, VLDB 2006) visit every
# connected subgraph and connected complement once; sorting the pairs by the
# size of their union solves both halves of a pair before it. A participant
# joined by its own link neighbours what must be in first, and
# `JoinRules.joinable` keeps the pairs a tree may join. Past `PAIR_BUDGET`
# the dynamic program is abandoned for greedy operator ordering (Fegaras,
# 1998).
# ---------------------------------------------------------------------------
comptime PAIR_BUDGET = 10_000
"""How much enumeration a chain may take before the search turns greedy: the
connected-subgraph / complement pairs emitted, and separately the subsets
visited to find them, each capped here. The pair figure is DuckDB's."""


@fieldwise_init
struct _Choice(Copyable, Movable):
    """The cheapest way found to join one set of participants: its cost, the
    subset on the left (empty for a participant) and the side to index."""

    var cost: Cost
    var left: Set[Int]
    var side: JoinBuildSide


struct _Choices(Movable):
    """The cheapest way found so far to join each set of a chain's
    participants. Costs leave out the participants' own work, which every
    tree pays alike."""

    var best: Dict[Set[Int], _Choice]

    def __init__(out self, mut pricing: JoinPricing) raises:
        self.best = Dict[Set[Int], _Choice]()
        for p in range(pricing.rules.participants()):
            var one: Set[Int] = {p}
            self.best[one^] = _Choice(pricing.leaf(p), Set[Int](), BUILD_LEFT)

    @staticmethod
    def search[budget: Int](mut pricing: JoinPricing) raises -> Self:
        """The cheapest trees found — over every participant only when some
        tree can be priced. Exhaustive when the pairs fit in `budget`, greedy
        otherwise."""
        var choices = Self(pricing)
        var pairs = _Pairs(budget)
        pairs.run(pricing.rules)
        if pairs.over:
            choices.greedy(pricing)
        else:
            for ref bucket in pairs.by_size(pricing.rules.participants()):
                for ref pair in bucket:
                    choices.offer(pricing, pair[0], pair[1])
        return choices^

    def offer(
        mut self, mut pricing: JoinPricing, a: Set[Int], b: Set[Int]
    ) raises:
        """Price joining `a` to `b`, keeping it if it is the cheapest way yet to
        join their union. A pair that cannot be priced is left unpriced:
        preferring the side that *can* be priced would hash a fact table
        rather than a dimension whose string column has no width. A
        participant joined by its own link goes on the right."""
        if not pricing.rules.joinable(a, b):
            return
        var left_choice = self.best.get(a)
        var right_choice = self.best.get(b)
        if not left_choice or not right_choice:
            return
        var s = a | b
        var left = a.copy()
        var right = b.copy()
        var own = pricing.rules.attaching(a, b)
        if own >= 0:
            right = {own}
            left = s - right
        var link = pricing.rules.link(left, right)
        if link.strictness == JOIN_ALL and link.kind.commutes():
            # The other side only when strictly cheaper, and neither when
            # either cannot be priced: a side that can be priced is not
            # cheaper than one nobody can bound.
            var l = pricing.size(left)
            var r = pricing.size(right)
            var other = (
                BUILD_RIGHT if link.build_side == BUILD_LEFT else BUILD_LEFT
            )
            var here = Cost.hash_join(l, r, link.build_side, link.kind)
            var there = Cost.hash_join(l, r, other, link.kind)
            var here_total = here.total().known()
            var there_total = there.total().known()
            if not here_total or not there_total:
                return
            if there_total.value() < here_total.value():
                link.build_side = other
        var cost = (
            left_choice.value().cost
            + right_choice.value().cost
            + pricing.join(left, right, link)
        )
        var total = cost.total().known()
        if not total:
            return
        var incumbent = self.best.get(s)
        if incumbent:
            var held = incumbent.value().cost.total().known()
            if held and held.value() <= total.value():
                return
        self.best[s^] = _Choice(cost, left^, link.build_side)

    def greedy(mut self, mut pricing: JoinPricing) raises:
        """Greedy operator ordering: repeatedly join the connected pair of parts
        whose join is expected to be smallest, pricing each join exactly as the
        dynamic program would. A pair that cannot be estimated or priced is
        passed over for the next; the search fails only once no pair is left
        to join."""
        var parts = List[Set[Int]]()
        for p in range(pricing.rules.participants()):
            parts.append({p})
        while len(parts) > 1:
            var ranked = List[Tuple[Int, Int, Int]]()
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    if not pricing.rules.joinable(parts[i], parts[j]):
                        continue
                    var n = pricing.size(parts[i] | parts[j]).rows.known()
                    if not n:
                        continue
                    var at = len(ranked)
                    while at > 0 and ranked[at - 1][0] > n.value():
                        at -= 1
                    ranked.insert(at, (n.value(), i, j))
            var joined = False
            for ref candidate in ranked:
                var i = candidate[1]
                var j = candidate[2]
                var merged = parts[i] | parts[j]
                self.offer(pricing, parts[i], parts[j])
                if merged in self.best:
                    parts[i] = merged^
                    _ = parts.pop(j)
                    joined = True
                    break
            if not joined:
                return

    def emit(
        self, rules: JoinRules, s: Set[Int], mut order: JoinOrder
    ) raises -> Int:
        """The tree chosen for `s`, added to `order` bottom-up; answers its
        node."""
        if len(s) == 1:
            for p in s:
                return p
        ref choice = self.best[s]
        ref left = choice.left
        var right = s - left
        var l = self.emit(rules, left, order)
        var r = self.emit(rules, right, order)
        return order.add(
            PlannedJoin.of(rules.link(left, right), l, r, choice.side)
        )


struct JoinOrdering:
    """Every join chain of a plan computed by the cheapest tree the search
    finds, when that is not the written one.

    Participants first, so a chain nested inside another is ordered too. The
    output of a chain never moves, so nothing above it needs to know.
    """

    @staticmethod
    def apply[
        budget: Int = PAIR_BUDGET
    ](node: DynRelation) raises -> DynRelation:
        # The recursion calls `apply` itself rather than through a closure
        # handed to `traverse`, as `Optimizer.rewrite` does: that shape
        # deadlocked the compiler.
        var children = List[DynRelation]()
        for ref child in node.children():
            children.append(Self.apply[budget](child))
        var rebuilt = node.with_children(children)
        if not rebuilt.isa[JoinChain]():
            return rebuilt^
        ref chain = rebuilt.get[JoinChain]()
        var order = Self.order[budget](chain)
        if order == JoinOrder.written(len(chain.inputs), chain.links):
            return rebuilt^
        return rebuilt.with_chain(chain.with_order(order^))

    @staticmethod
    def order[budget: Int = PAIR_BUDGET](chain: JoinChain) raises -> JoinOrder:
        """The tree to compute `chain` by: the cheapest the search finds when
        strictly cheaper than the written one, which is otherwise kept."""
        var pricing = JoinPricing(chain)
        var written = JoinOrder.written(len(chain.inputs), chain.links)
        var choices = _Choices.search[budget](pricing)
        var everything = pricing.rules.everything()
        if everything in choices.best:
            var found = choices.best[everything].cost.total().known()
            var held = pricing.tree(written).total().known()
            if found and held and found.value() < held.value():
                var order = JoinOrder(len(chain.inputs))
                _ = choices.emit(pricing.rules, everything, order)
                return order^
        return written^


struct _Pairs(Movable):
    """`EnumerateCsg` and `EnumerateCmp`, collecting every pair of disjoint,
    connected, adjacent participant sets once, until the budget runs
    out."""

    var budget: Int
    var visited: Int
    var over: Bool
    var pairs: List[Tuple[Set[Int], Set[Int]]]

    def __init__(out self, budget: Int):
        self.budget = budget
        self.visited = 0
        self.over = False
        self.pairs = List[Tuple[Set[Int], Set[Int]]]()

    def _visit(mut self):
        self.visited += 1
        if self.visited > self.budget:
            self.over = True

    def _emit(mut self, a: Set[Int], b: Set[Int]):
        self.pairs.append((a.copy(), b.copy()))
        if len(self.pairs) > self.budget:
            self.over = True

    def run(mut self, rules: JoinRules):
        for i in reversed(range(rules.participants())):
            if self.over:
                return
            var excluded = Set[Int]()
            for j in range(i + 1):
                excluded.add(j)
            self._emit_csg(rules, {i})
            self._csg(rules, {i}, excluded)

    @staticmethod
    def _next(mut sub: Set[Int], of: Set[Int], n: Int):
        """Step `sub` to the next subset of `of`, the whole set first and the
        empty one last: every member below `sub`'s lowest joins it, and that
        lowest leaves."""
        for i in range(n):
            if i in of:
                if i in sub:
                    sub.discard(i)
                    return
                sub.add(i)

    def _csg(mut self, rules: JoinRules, s: Set[Int], excluded: Set[Int]):
        """Every connected set grown from `s` through participants not in
        `excluded`."""
        var n = rules.participants()
        var grow = rules.adjacent(s) - excluded
        if len(grow) == 0:
            return
        var sub = grow.copy()
        while len(sub) > 0 and not self.over:
            self._visit()
            self._emit_csg(rules, s | sub)
            Self._next(sub, grow, n)
        sub = grow.copy()
        while len(sub) > 0 and not self.over:
            self._visit()
            self._csg(rules, s | sub, excluded | grow)
            Self._next(sub, grow, n)

    def _emit_csg(mut self, rules: JoinRules, s1: Set[Int]):
        """Every connected complement of `s1` whose lowest participant is above
        `s1`'s."""
        var low = 0
        while low not in s1:
            low += 1
        var excluded = s1.copy()
        for j in range(low + 1):
            excluded.add(j)
        var start = rules.adjacent(s1) - excluded
        for i in reversed(range(rules.participants())):
            if self.over:
                break
            if i not in start:
                continue
            self._emit(s1, {i})
            var below = excluded.copy()
            for j in start:
                if j <= i:
                    below.add(j)
            self._cmp(rules, s1, {i}, below)

    def _cmp(
        mut self,
        rules: JoinRules,
        s1: Set[Int],
        s2: Set[Int],
        excluded: Set[Int],
    ):
        var n = rules.participants()
        var grow = rules.adjacent(s2) - excluded
        if len(grow) == 0:
            return
        var sub = grow.copy()
        while len(sub) > 0 and not self.over:
            self._visit()
            self._emit(s1, s2 | sub)
            Self._next(sub, grow, n)
        sub = grow.copy()
        while len(sub) > 0 and not self.over:
            self._visit()
            self._cmp(rules, s1, s2 | sub, excluded | grow)
            Self._next(sub, grow, n)

    def by_size(self, n: Int) -> List[List[Tuple[Set[Int], Set[Int]]]]:
        """The pairs bucketed by the size of their union, smallest first, so
        both halves of a pair are always solved before it."""
        var buckets = List[List[Tuple[Set[Int], Set[Int]]]](capacity=n + 1)
        for _ in range(n + 1):
            buckets.append(List[Tuple[Set[Int], Set[Int]]]())
        for ref pair in self.pairs:
            buckets[len(pair[0]) + len(pair[1])].append(pair.copy())
        return buckets^


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
    - `Union`, `Intersection` and `Difference` **replace** it with every column
      of each side: they match rows by position, so every column counts.
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
    def _narrowed_chain(
        chain: JoinChain, needed: List[String]
    ) raises -> JoinChain:
        """`chain` answering with only the outputs `needed` names, in its own
        order — at least one, since a batch carries its row count in its
        columns."""
        var names = List[String]()
        var refs = List[JoinRef]()
        var all = chain.names()
        for i in range(len(chain.refs)):
            if all[i] in needed:
                names.append(all[i].copy())
                refs.append(chain.refs[i].copy())
        if len(names) == 0 and len(chain.refs) > 0:
            names.append(all[0].copy())
            refs.append(chain.refs[0].copy())
        return chain.with_output(refs^, names)

    @staticmethod
    def _demand(chain: JoinChain) -> List[List[String]]:
        """Per participant, the columns `chain` reads of it: its outputs,
        every link's keys and every filter's columns, each once."""
        var out = List[List[String]](capacity=len(chain.inputs))
        for _ in range(len(chain.inputs)):
            out.append(List[String]())

        def take(mut out: List[List[String]], refs: List[JoinRef]):
            for ref r in refs:
                if r.name not in out[r.input]:
                    out[r.input].append(r.name.copy())

        take(out, chain.refs)
        for ref l in chain.links:
            take(out, l.left_keys)
            take(out, l.right_keys)
        for ref f in chain.filters:
            take(out, f.refs)
        return out^

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
    def _narrow_scan[
        S: FileScan
    ](node: DynRelation, needed: List[String]) raises -> DynRelation:
        """`node`, an `S`, reading only the columns `needed` leaves it."""
        ref scan = node.get[S]()
        var source = scan.schema()
        var keep = Self._narrowed(source, needed)
        if len(keep) == len(source.fields):
            return node.copy()
        var fields = List[Field](capacity=len(keep))
        for ref name in keep:
            fields.append(source.field(name=name).copy())
        var out: DynRelation = scan.with_schema(schema(fields^))
        return out^

    @staticmethod
    def apply(node: DynRelation, needed: List[String]) raises -> DynRelation:
        """`node`, with its sources narrowed to `needed`."""
        if node.isa[ParquetScan]():
            return Self._narrow_scan[ParquetScan](node, needed)
        if node.isa[IpcScan]():
            return Self._narrow_scan[IpcScan](node, needed)
        if node.isa[JsonScan]():
            return Self._narrow_scan[JsonScan](node, needed)
        if node.isa[IcebergScan]():
            return Self._narrow_scan[IcebergScan](node, needed)

        if node.isa[InMemoryTable]():
            ref src = node.get[InMemoryTable]()
            var keep = Self._narrowed(src.schema(), needed)
            if len(keep) == len(src.schema().fields):
                return node.copy()
            var out: DynRelation = src.with_batch(src.batch.select(keep))
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
            var out: DynRelation = a.with_input(Self.apply(a.input[], below))
            return out^

        if node.isa[JoinChain]():
            # The output narrows to what is needed, and each participant to
            # what the chain then reads of it — by participant, so a name two
            # inputs share narrows each separately.
            var c = Self._narrowed_chain(node.get[JoinChain](), needed)
            var demand = Self._demand(c)
            var inputs = List[ArcPointer[DynRelation]](capacity=len(c.inputs))
            for p in range(len(c.inputs)):
                inputs.append(ArcPointer(Self.apply(c.inputs[p][], demand[p])))
            return node.with_chain(c.with_inputs(inputs^))

        # The set relations are positional and match whole rows, so each side
        # keeps all of its own columns. Descending anyway still prunes beneath
        # them.
        # Each side keeps every column it produces. The recursion calls
        # `apply` itself: routing it through a helper that calls back into
        # `apply` deadlocked the compiler.
        if node.isa[Union]():
            ref u = node.get[Union]()
            var out: DynRelation = Union(
                Self.apply(u.left[], u.left[].schema().names()),
                Self.apply(u.right[], u.right[].schema().names()),
            )
            return out^

        if node.isa[Intersection]():
            ref i = node.get[Intersection]()
            var out: DynRelation = Intersection(
                Self.apply(i.left[], i.left[].schema().names()),
                Self.apply(i.right[], i.right[].schema().names()),
                i.all,
            )
            return out^

        if node.isa[Difference]():
            ref d = node.get[Difference]()
            var out: DynRelation = Difference(
                Self.apply(d.left[], d.left[].schema().names()),
                Self.apply(d.right[], d.right[].schema().names()),
                d.all,
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

    @staticmethod
    def finish(plan: DynRelation) raises -> DynRelation:
        """Whatever this set wants done **once, after** the rewrite loop has
        converged — the counterpart of `prepare`.

        Join ordering lives here: it is a search over a whole region rather
        than a local match, and it has to price a region after the filters
        above it have been pushed onto its leaves, which the rewrite loop is
        what does. Run in `prepare` it would order joins it could not see the
        selectivity of.
        """
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

    @staticmethod
    def finish(plan: DynRelation) raises -> DynRelation:
        return plan.copy()


struct ScanPruning(RuleSet):
    """The smallest set that makes a scan skip row groups.

    **What an AOT binary names when pruning is all it wants.** Row-group
    pruning used to ride `Relation.to_operator`, so it happened whether or not
    anything optimized; now it is a rewrite, and a plan nobody rewrites reads
    every row group. Nothing applies this set behind the author's back — a plan
    prunes because its author wrote `.optimize[ScanPruning]()`, exactly as it
    merges projections because they wrote `.optimize[AllRules]()`.

    Three rules, chosen to cover what the retired descent covered and no more:
    `PushFilterBelowSort` puts a filter under an unbounded sort — the one node
    the descent forwarded through besides `Filter` — `SplitConjunction` breaks
    an `AND` into conjuncts the scan can use separately, and
    `PushFilterIntoScan` walks the remaining `Filter` chain and lands them on
    the scan.

    `prepare` is the identity, so `ColumnPruning` is never mentioned and never
    links. A binary that wants row-group pruning does not pay for the
    projection pass it did not ask for, which is the whole point of the rule
    set being a comptime parameter.
    """

    @staticmethod
    def prepare(plan: DynRelation) raises -> DynRelation:
        return plan.copy()

    @staticmethod
    def rewrite(node: DynRelation) raises -> DynRelation:
        return PushFilterIntoScan.apply(
            SplitConjunction.apply(PushFilterBelowSort.apply(node))
        )

    @staticmethod
    def finish(plan: DynRelation) raises -> DynRelation:
        return plan.copy()


struct AllRules(RuleSet):
    """Every rule in this file.

    **Order is chosen, not incidental.** Removals run before reorderings, so no
    rule bothers moving a node another is about to delete, and
    `PushFilterBelowSort` runs before `TopN` so a filter between a limit and a
    sort is relocated *before* `TopN` checks adjacency and gives up.

    `SplitConjunction` runs before `PushFilterIntoScan` so the scan learns each
    conjunct on its own. A compound `AND` prunes only as well as its weaker
    half, so `a > 60 AND a < 140` reaching the scan whole would skip a chunk
    only when *neither* bound can — where the two halves separately skip
    everything outside `[60, 140]`. The reverse order was forced until `mask`
    became a slot on `DynValue`: a split rebuilds its conjuncts through the
    erased `.filter()` overload, and an erased predicate used to have no
    pruning method at all.
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
    def finish(plan: DynRelation) raises -> DynRelation:
        """Every join chain ordered by `JoinOrdering`, over sources pruned
        again first: the rewrite loop can free columns — a sort it removed
        read one — and a tree is priced by the widths of what its inputs still
        hold."""
        return JoinOrdering.apply(Self.prepare(plan))

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
        out = MergeJoinChains.apply(out)
        out = MergeProjectIntoJoin.apply(out)
        out = RemoveSortBeforeAggregate.apply(out)
        out = MergeLimits.apply(out)
        out = RemoveRedundantSort.apply(out)
        out = SplitConjunction.apply(out)
        out = PushFilterIntoScan.apply(out)
        out = PushFilterBelowProject.apply(out)
        out = PushFilterBelowSort.apply(out)
        out = PushFilterIntoJoin.apply(out)
        out = PushFilterBelowAggregate.apply(out)
        out = PushLimitBelowProject.apply(out)
        out = TopN.apply(out)
        # Join *order* is not a rule: it is `finish`, a search over each
        # chain's tree once everything above has settled.
        return MergeWindows.apply(out)


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
    def rewrite(node: DynRelation) raises -> DynRelation:
        """One bottom-up pass over `node`: children first, then `R`'s rules.

        Bottom-up, because every rule reads its child's type: rewriting
        children first means a rule sees the child's *final* form, so
        `Limit(Sort(Filter(...)))` collapses in one pass instead of waiting for
        the fixpoint to rediscover it.
        """
        # The recursion calls `rewrite` itself rather than through a closure
        # handed to `traverse`: that shape deadlocked the compiler.
        var rewritten = List[DynRelation]()
        for ref child in node.children():
            rewritten.append(Self.rewrite(child))
        return Self.R.rewrite(node.with_children(rewritten))

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
        var settled = Self._settle(Self.R.prepare(plan))
        return Self._settle(Self.R.finish(settled))

    @staticmethod
    def _settle(plan: DynRelation) raises -> DynRelation:
        """`plan` rewritten pass after pass until a pass changes nothing.

        Run after `prepare`, and again after `finish`, whose output the rules
        have not seen — its pruning can leave a projection `RemoveNoOpProject`
        removes, and a plan's rendering does not show a narrowed source, so
        whether `finish` changed anything cannot be read off one.
        """
        var current = plan.copy()
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
