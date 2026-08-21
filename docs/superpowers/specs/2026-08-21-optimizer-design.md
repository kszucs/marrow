# Query optimizer — design

**Status:** design, not started. **Date:** 2026-08-21.
**Tree:** `bc72d5f` (`alpha`).

## Goal

Replace the two ad-hoc plan rewrites in `marrow/expr/relations.mojo` with a
mechanism whose cost is **constant in the number of rules**, so that adding the
third, fourth and tenth rewrite does not add a third, fourth and tenth virtual
slot to `DynRelation` — and does not cost binary size in the AOT lane, which is
the standing constraint `marrow.expr` is built around.

The optimizer is not the point. The *mechanism* is: today's mechanism admits
exactly two rules and, as shown in §1.2, is structurally incapable of expressing
a third of a different shape.

## Non-goals

- **No cost-based search.** No Volcano/Cascades memo, no plan enumeration, no
  cardinality estimation. See §2.9 for why join reordering is out of every
  phase, not merely deferred.
- **No SQL frontend, and no decorrelation in this spec.** Correlated-subquery
  rewriting is a *consumer* of this framework. §6 states what it demands so the
  framework is not designed into a corner; the rule itself belongs to the
  concurrent subquery spec and to a later phase.
- **No expression-interior rewriting in phase 1.** Constant folding and CSE are
  argued down in §2.6 and §2.10.
- **No new relational nodes.** `Distinct`, `Union`, `Values`, `EmptyRelation`
  are M2.1 and unstarted; several attractive rules are blocked on them and are
  named as blocked rather than designed around.

---

## 0. What exists today

`marrow/expr/relations.mojo`, 1744 lines, holds the plan IR: eight `Relation`
conformers (`InMemoryTable`, `ParquetScan[leaves]`, `Filter`, `Project`,
`Limit`, `Sort`, `Aggregate`, `Join`), erased behind `DynRelation`, an
`ArcPointer` plus eight function-pointer trampoline slots (`:422-436`).

Four of those slots are plan-analysis, not execution:

| slot | line | who overrides it |
|---|---|---|
| `_virt_kind` | `:429` | `ParquetScan`, `Sort` |
| `_virt_children` | `:436` | every non-leaf |
| `_virt_with_predicate` | `:430` | `ParquetScan` only |
| `_virt_with_projection` | `:433` | `Filter`, `Project`, `Limit`, `Sort`, `Aggregate`, `ParquetScan` |

`optimize()` (`:550`) is projection pushdown and nothing else: it seeds
`with_projection` with the root's own column names and lets each node widen the
set with the columns its own expressions read. `execute()` (`:571`) calls it.
`with_predicate` is not in `optimize()` at all — it fires from the *builder*,
inside `DynRelation.filter()` (`:696`), and therefore only when `filter()` is
called directly on a scan.

`marrow/expr/pruning.mojo` (52 lines) is `PruneStats` plus the `Value.prune()`
protocol: statistics-based row-group and page skipping, evaluated per row group
inside `ParquetScanProcessor`, not at plan time.

Projection pushdown is the largest performance change of the alpha
(17.7x → 5.0x versus polars; `COUNT(*)` 271 → 9.9 ms — `docs/backlog.md:482-490`).
This spec must not regress it, and every rewrite in it is behaviour-preserving
in the same sense: **same rows, same schema, same order.**

---

## 1. The rule mechanism

### 1.1 Requirements

R1. A rule must be able to **replace a node with a node of a different type**
    (a `Filter` becomes two `Filter`s; a `Sort` disappears; a `Filter` becomes a
    `Join`). This is the requirement today's mechanism fails.

R2. **Adding a rule must not add a virtual slot**, or the box grows without
    bound and every binary pays for rules it never reaches.

R3. **No open dispatcher.** No `List[DynRule]`, no `trait OptimizerRule` erased
    into a box. `CLAUDE.md`'s closed-erasure/DCE property is the constraint, and
    `execute()` calls `optimize()`, so anything reachable from `optimize()` is
    reachable from *every* AOT gate binary.

R4. It must survive `ParquetScan[leaves: LeafSet]` being **comptime
    parameterized**. A generic rule cannot name `ParquetScan[Self.leaves]`, and a
    `downcast[ParquetScan]()` silently rebuilds a narrow scan as the default
    full-ladder one — the defect `with_predicate` was introduced to fix
    (`relations.mojo:1265-1278`).

### 1.2 Why `with_predicate` / `with_projection` do not generalise

Not "they would need a third slot" — that is the symptom. The mechanism is
*structurally* incapable of R1, and the reason is three lines in `DynRelation`:

```
var data = self._virt_with_projection(self._data, needed)
if data:
    var out = DynRelation(copy=self)   # copies the trampolines
    out._data = data.take()            # swaps only the payload pointer
```
`relations.mojo:541-546`

`DynRelation(copy=self)` copies eight function pointers that were bound to `T`
at construction. The rewritten node therefore **must have the same concrete
type** — the protocol's own docstring says so (`:132-136`): *"The rebuilt node
has the same concrete type as this one, so the caller can keep its own
trampolines and swap only the pointer."*

That is not an implementation shortcut. It is forced: returning
`Optional[DynRelation]` from a trampoline **field** puts `DynRelation` inside its
own field's function type and Mojo rejects the struct as recursive. So the
protocol can only ever *narrow a node's payload in place*.

Every rule anyone actually wants violates that:

| rule | what it does to node identity |
|---|---|
| predicate pushdown | removes a `Filter` here, inserts one there |
| conjunction splitting | one `Filter` becomes N |
| filter merging | N `Filter`s become one |
| redundant-sort elimination | a `Sort` becomes its own child |
| top-K folding | `Limit(Sort)` becomes `Sort(limit=k)` |
| decorrelation | a unary node becomes a binary `Join` |

None of the six is expressible. Projection pushdown is expressible *only*
because it is the degenerate case: every node keeps its type and narrows its
own payload. A third rule of that exact shape would fit; there are no more of
that shape worth having.

Note what the recursion restriction actually forbids: a trampoline field
returning `Optional[DynRelation]` (an inline layout, needs `DynRelation`'s size)
is rejected; a field returning or taking `List[DynRelation]` (a pointer) is
fine. `_virt_children` (`:436`) already proves the return direction compiles.

### 1.3 The mechanism: rules are free functions over a closed node set

**Rules are ordinary functions.** A rule is
`def rule_name(rel: DynRelation) raises -> DynRelation` in a new module
`marrow/expr/optimize.mojo`. It is not a trait, not a struct, not a box, and it
is never stored in a list. It recognises the nodes it cares about with the
existing `kind()` discriminant plus `downcast[T]()`, rebuilds them by naming the
concrete constructor, and recurses through the framework's one new primitive.

Three pieces make that work.

**(a) `kind()` becomes a complete, closed discriminant.** Today there are three
values (`RELATION_GENERIC`, `_PARQUET_SCAN`, `_SORT`, `:114-116`). Widen to one
per node type. This is the closed-world assumption the whole design rests on and
it is the same one DuckDB makes: `PushDownCorrelatedNode` is a 26-case
`switch (plan->type)` in which four operator types explicitly `throw` and
anything else is an `InternalException` — the operator set is closed and every
member needs an explicit rule
(`../duckdb/src/planner/subquery/flatten_dependent_join.cpp:1002-1071`).

**(b) One new virtual: `with_children`.** Abstract on `Relation`:

```
def with_children(self, kids: List[DynRelation]) raises -> ArcPointer[NoneType]
```

Same erased-pointer protocol as today — same concrete type, so the caller keeps
its trampolines — but now it is a *generic structural* operation rather than a
rule-specific one, so one slot serves all rules forever. It is what lets a rule
handle its two interesting arms and delegate the other six to a shared
`_rewrite_children(rel, rule)` driver instead of repeating an eight-arm rebuild
ladder per rule. `ParquetScan[leaves]` implements it as a one-line "I am a leaf,
here I am again" that keeps `Self.leaves` in scope, satisfying R4.

Making it **abstract, not defaulted**, is deliberate: a new `Relation` conformer
then cannot compile without stating how it rebuilds, which is the compiler
enforcement a `kind()` ladder alone cannot give. The alternative — default to
"return self unchanged" — fails safe (a new node blocks rewrites below it rather
than corrupting them) but fails silently, and this is exactly the class of
mistake `docs/backlog.md` records eighteen instances of.

*Verified.* This shape is not obvious under Mojo's recursion rules, so it was
compiled before being specified. A standalone probe reproducing the erasure box
(`ArcPointer[NoneType]` payload, five trampoline slots, a
`def(ArcPointer[NoneType], List[Erased]) thin raises -> ArcPointer[NoneType]`
field, a comptime-parameterized leaf `Leaf[tag: Int]`, a generic bottom-up
`rewrite_children` driver) **compiles and runs**, and the parameterized leaf
keeps its comptime parameter across the rebuild. The `List[…]` argument position
behaves exactly like `_virt_children`'s `List[…]` return position.

**(c) `with_predicate` and `with_projection` collapse into one slot.** Both are
the same thing — "rebuild this *scan* around new read options" — and only
`ParquetScan` ever answers either non-trivially:

```
def with_scan_options(self, opts: ScanOptions) raises -> Optional[ArcPointer[NoneType]]
```

with `ScanOptions` a plain struct carrying an optional projected column list and
an optional pruning predicate. Default `None` — "I am not a scan". This is the
only node-local rewrite hook the design keeps, and it is kept for the one reason
a generic rule cannot do the job: `Self.leaves` is in scope inside the node and
nowhere else.

**Slot arithmetic. This is the headline.**

| | today | proposed |
|---|---|---|
| `_virt_kind` | ✓ | ✓ |
| `_virt_children` | ✓ | ✓ |
| `_virt_with_predicate` | ✓ | — |
| `_virt_with_projection` | ✓ | — |
| `_virt_with_children` | — | ✓ |
| `_virt_with_scan_options` | — | ✓ |
| **total** | **4** | **4** |
| **rules expressible** | **2, of one shape** | **any number, any shape** |

Rule count and slot count are decoupled. That is the whole design.

### 1.4 The pipeline is straight-line code

```
def optimize(self) raises -> DynRelation:
    var p = DynRelation(copy=self)
    p = scan_predicate_pushdown(p)
    p = topk_fold(p)
    p = projection_pushdown(p, _schema_names(self.schema()))
    return p^
```

A literal sequence of direct calls. No list, no registry, no erasure, no
indirect call. The linker sees a static call graph from `optimize()` and can
attribute every byte to a named rule.

### 1.5 What was rejected

**A `trait OptimizerRule` erased into a `DynRule`, held in a `List[DynRule]`** —
the DataFusion/Calcite shape, and the obvious one. Rejected on three grounds,
in increasing order of severity:

1. *It adds a second erasure box.* Two or more trampolines per rule, plus a
   `List` whose element type is a new struct — the exact "open dispatcher"
   `CLAUDE.md` names.
2. *It makes every rule permanently live.* Iterating a `List[DynRule]` means the
   linker cannot prove any element dead. Straight-line calls are also all live
   *today*, but they can be gated individually with `comptime if` later
   (`GPU_ENABLED`, `CLI_WRITERS_ENABLED` are the tree's two precedents); a list
   cannot be gated element-wise without becoming a `comptime if` ladder that
   builds the list — i.e. straight-line code with extra steps.
3. *It is one refactor away from the +662,740-byte shape.* The moment a rule
   wants to be generic over the node type it rewrites, a closure or adapter has
   to sit between the driver and the rule and bind on a weaker trait. That is
   precisely the `variant_dispatch` narrowing adapter that cost **+662,740 bytes
   (+31.9% of `__text`)** on `query_streaming_agg_fused` and whose removal
   recovered it (`benchmarks/binary_size/baseline.json`, `_comment`). A
   free-function pipeline has no place to put such an adapter.

**A pattern-matcher rule registry** (DuckDB's `Rule` + `ExpressionMatcher`,
29 rules in a `vector`, `../duckdb/src/optimizer/optimizer.cpp:63-92`). Note
that DuckDB confines this to *expression* rules, whose `Apply` returns
`unique_ptr<Expression>` and which structurally cannot change the operator tree;
its ~40 *plan-level* passes are hand-written straight-line calls with bespoke
`switch (op->type)` bodies
(`../duckdb/src/optimizer/optimizer.cpp:178-443`). Marrow's phase-1 rules are
all plan-level. Importing the matcher machinery would buy nothing and cost an
open dispatcher; importing the straight-line half is exactly §1.4.

**`inputs()` + `with_inputs()` + `required_columns()`** — already rejected once
(`docs/backlog.md:985-986`, and `:493-499`) because three read-only virtuals do
not add up to one rewrite. Still true: `Aggregate` *discards* the required set
and `Project` *renames* through it, so no generic driver can compute
`needed_below` from a `required_columns()` answer alone. Per-node projection
semantics have to live somewhere; §1.3 puts them in one `kind()` ladder inside
one rule rather than in a virtual on every node.

---

## 2. Which rules, and in what order

The tree already contains measurements that kill several textbook rules. They
are used rather than re-derived.

### 2.1 Phase-1 set

| # | rule | status |
|---|---|---|
| 1 | `scan_predicate_pushdown` | existing behaviour, **generalised** |
| 2 | `topk_fold` | existing behaviour, **relocated** |
| 3 | `projection_pushdown` | existing behaviour, re-expressed |

Three rules, of which **zero are new capabilities**. That is deliberate and it
is the honest YAGNI answer: phase 1's deliverable is the mechanism, and the
strongest possible evidence that a mechanism is sound is that it reproduces the
existing behaviour bit-for-bit at no size cost. Rules that pay are phase 2, when
the framework is proven and the size budget is known rather than guessed.

The engine with the largest rule set says the same thing about adding to it.
DataFusion's rule list carries an inline warning: *"The order of rules in this
list is important, as it determines the order in which they are applied. Adding
a new rule here is expensive as it will be applied to all queries. Please extend
existing rules when possible."*
(`../datafusion/datafusion/optimizer/src/optimizer.rs:280-317`). That is 25 rules
guarding against a 26th. Marrow is guarding against a fourth.

**1. `scan_predicate_pushdown`.** Today this fires from `DynRelation.filter()`
(`:696`) and therefore only when `filter()` is called *directly* on a scan. Moved
into the pipeline it walks down through pass-through nodes to find the scan, so
`parquet_scan(...).select("a","b").filter(col("a") > 0)` gets row-group pruning
where today it gets none. That shape is what the Python lazy frontend produces
and what a `select`-then-`filter` chain in the Mojo DSL produces. It is the one
piece of genuinely new *value* in phase 1 and it costs one rule body.

Correctness conditions, all of which must be checked and none of which the
current builder-embedded version needs to check because it only ever sees a
directly-adjacent scan:
- Only descend through nodes that neither reorder nor rename: `Filter`, `Sort`,
  `Limit`, and a `Project` **all of whose surviving outputs are pass-throughs
  under their own names**. `docs/backlog.md:515-519` records the trap: through a
  `Project` a rename makes the scan prune on *another column's* statistics — a
  wrong answer, not an error. Requiring identity-named pass-throughs is the
  cheap sound condition; anything else stops the walk.
- Stop at `Aggregate` and `Join` unconditionally.
- The predicate is only ever pushed **as pruning metadata**; the `Filter` node
  always stays. Correctness never depends on pruning
  (`marrow/expr/pruning.mojo`, module docstring), so an over-approximate push is
  safe and a missed push costs only time.
- Descending through `Limit` is safe *for pruning* — pruning cannot change rows
  — even though moving a `Filter` through a `Limit` is not.

**2. `topk_fold`.** `Limit(Sort)` → `Sort(limit=k)`, today inlined in
`DynRelation.limit()` (`:1170-1183`). Moving it changes no behaviour; it is
included because leaving a rewrite in the builder is the specific
leaky-abstraction finding `docs/backlog.md:1242-1246` records against
`DynRelation` ("the erasure box **and the entire plan builder/binder** … the
planner exists, and it is fused into the box"). Phase 1 is the cheapest moment
to fix it, and it makes the fold fire when the `Limit` was not adjacent at build
time.

**3. `projection_pushdown`.** Same algorithm, same two correctness guards
(`docs/backlog.md:501-506`): seed with the root's own schema so the output
schema is invariant, and never narrow a scan to zero columns. Re-expressed as
one function with a `kind()` ladder instead of six `with_projection` overrides.
This is the pure refactor and the one that proves the mechanism.

### 2.2 Ordering, and why this one

`scan_predicate_pushdown` → `topk_fold` → `projection_pushdown`.

Only one edge is load-bearing: **projection pushdown must run last**, because
the columns a pushed predicate reads must be in the required set when the scan
narrows. The current code gets this for free (the predicate is attached at build
time, and `Filter.with_projection` widens by
`self.predicate.referenced_columns()`, `:1379`); in a pipeline it becomes an
ordering constraint, and it must be stated because violating it produces a scan
that does not read the column its own pruning predicate references.

This is not marrow-specific. polars documents the identical edge in its own
pipeline — predicate pushdown *"should be run before projection pushdown. This
allows columns only needed for filters to be dropped early"*
(`../polars/crates/polars-plan/src/plans/optimizer/mod.rs:76-77`) — and
DataFusion orders `PushDownFilter` (#19) before `OptimizeProjections` (#25).

`topk_fold` before `projection_pushdown` because folding removes a `Limit` node
and the projection walk should not have to handle the pre-fold shape.
`scan_predicate_pushdown` before `topk_fold` is arbitrary; they do not interact.

Both DuckDB and DataFusion carry ordering rationale as inline comments at the
call site — `TYPE_PUSHDOWN` "must run before FILTER_PUSHDOWN"
(`../duckdb/src/optimizer/optimizer.cpp:217-221`); *"Filters can't be pushed
down past Limits, we should do PushDownFilter after PushDownLimit"*
(`../datafusion/datafusion/optimizer/src/optimizer.rs`, before rule #18). The
lesson taken is that pass order *is* the strategy and must be commented where
the call is, not inferred from the rule bodies.

### 2.3 Conjunction splitting — treated as a first-class problem

`filter(a AND b)` cannot be partially pushed unless the `AND` comes apart. Every
prior-art optimizer assumes a decomposable expression tree; marrow has two
expression lanes and only one of them is a tree at runtime:

- **runtime lane** — `DynValue` (`dynamic.mojo:257`) is a tag, a
  `List[ArcPointer[Self]]` of children and a payload. Decomposable.
- **AOT lane** — `values.mojo`'s nodes are *types*.
  `BoolBinary[AndKernel, AndInterval, Gt[…], Lt[…]]` has no runtime structure;
  its operands are comptime type parameters. `BoxedValue` erases it behind seven
  function pointers (`values.mojo:512-521`) and none of them exposes structure.

`golden/cases/filter_and.mojo` is exactly this: `(col("v") > lit(2)) & (col("w")
< lit(60))` runs as a fused `BoolBinary` in the Mojo lane and as a tagged
`DynValue` in the Python lane, from one source file.

**Finding: the fused lane *can* be split, and it was verified by compiling it.**
The trick is that the split happens where the operand types are still known — at
comptime, inside the node — and only the *result* is erased:

```
# on `Value`, defaulted: "I am one indivisible conjunct"
def conjuncts(self) -> List[BoxedValue]:
    return [BoxedValue(self)]

# on BoolBinary, when K is the conjunction kernel: split at comptime
def conjuncts(self) -> List[BoxedValue]:
    var out = self.l.conjuncts()   # Self.L is a concrete type here
    out.extend(self.r.conjuncts())
    return out^
```

A standalone probe of this exact shape — trait default constructing a box of
`Self`, an eighth trampoline `def(ArcPointer[NoneType]) thin -> List[BoxV]`, and
a parameterized `AndNode[L: V, R: V]` overriding the default — **compiles and
returns three conjuncts from `(a AND b) AND c`.** The two failures this was
expected to hit did not occur:

- It is *not* the ambiguous-overload trap. `CLAUDE.md`'s rule is that a struct
  method does not override a trait default — but that case
  (`Value.isnull` vs `DynValue.is_null`) had a *different* signature. A
  same-signature override is what `Value.prune` / `name` / `bound_column`
  already do throughout `values.mojo`, and it works.
- It is *not* the `DynArray.__eq__` instantiation deadlock. That cycle was
  mutually recursive across two erased containers; here
  `BoxedValue.__init__[V]` → `_conjuncts_tramp[V]` → `V.conjuncts()` →
  `BoxedValue.__init__[V]` resolves, because the leaf default's box
  instantiation is the one already in flight.

**Cost, stated honestly.** An eighth slot on `BoxedValue` is paid in every
binary: 8 bytes per instance, plus one `_conjuncts_tramp[V]` per boxed value
type. The instantiation cost is bounded and small — a leaf's trampoline needs
`BoxedValue.__init__[V]`, which was already needed; only an *interior*
conjunction operand forces a box that would not otherwise exist, so
`a AND b` costs two extra box instantiations. It is nonetheless a slot on the
hottest size-gated struct in the tree and it must be measured on
`query_streaming` and `query_dynvalue` before it lands.

**Two cheaper alternatives, and why they lose.**

- *A comptime overload on `filter()`*:
  `def filter[L: BoolValue, R: BoolValue](self, var pred: And[L, R])` splitting
  at the build site. Zero slots. Loses because it fires only where the static
  type is visible: it misses a predicate that arrives already boxed (from
  `ParquetScan.predicate`, from the Python frontend, from a subquery rewrite)
  and it misses a conjunction *synthesized by a later rule*, which is exactly
  what decorrelation does.
- *Callers pass `List[BoxedValue]`*: `filter([a, b])` instead of `filter(a & b)`.
  Zero slots, but it makes the API state the rewrite, and `filter_and.mojo`
  already exists in the golden corpus in the `&` spelling.

**Recommendation: the `conjuncts()` virtual, in phase 2, not phase 1** — because
its *consumers* are phase 2. Today the only consumer would be scan pruning, and
`prune()` already handles a whole conjunction compositionally through
`AndInterval` (`values.mojo:1302-1307`), so splitting buys nothing there. The
consumers that pay are predicate pushdown below `Join` (a conjunct referencing
only left columns) and below `Aggregate` (a `HAVING` conjunct on a group key),
both of which are M2/M3 workloads. Building the split before its consumer would
be paying a per-binary slot for a rewrite whose only effect is to create stacked
`Filter`s that `filter_merge` then puts back together.

**Which kernel is the conjunction.** Prefer a `comptime is_conjunction: Bool`
on `BoolBinaryKernel` (default `False`, `True` on `AndKernel`) over a
`comptime if Self.K.name == "and"` string match. `BoolBinary.prune`'s own
docstring gives the reason — "`P` is this operator read over intervals — see
`NumericCompare.prune` for why this is a kernel rather than a match on
`Self.K.name`" (`values.mojo:1302-1304`). `render()` matching on the name is
tolerable because it *is* printing the name; behaviour selection is not.

### 2.4 Lane classification of every candidate rule

| rule | lane category | fused lane |
|---|---|---|
| projection pushdown | relocates whole expressions; reads `referenced_columns()` | full |
| scan predicate pushdown | relocates a whole expression | full |
| top-K fold | node-shape only, no expression contact | full |
| filter merging | relocates whole expressions | full |
| limit pushdown | node-shape only | full |
| redundant-sort elimination | node-shape only | full |
| join reordering | manipulates key indices | full (but out — §2.9) |
| **conjunction splitting** | **decomposes an expression** | **full, via §2.3's `conjuncts()`** |
| constant folding | rewrites an expression interior | **unavailable** |
| CSE | rewrites an expression interior | **unavailable, and counterproductive** |

The line is not "moves vs inspects" — `referenced_columns()` already inspects
both lanes truthfully, and `conjuncts()` shows a *comptime-decomposable*
operation can cross the boundary too. The line is **rebuilding an arbitrary
interior node**: `And` can hand back its two halves because it knows their types
at the split point, but nothing can hand back "this subtree with the third
literal replaced by 5", because that names a type that does not exist yet.
Constant folding and CSE both need that, and that is why both are runtime-lane
only. See §2.6 and §2.10 for why neither is worth having on those terms.

### 2.5 Is a rule that fires in one lane only acceptable?

**Only if it cannot change results, and phase 1 has no such rule.** The rule for
the future:

- A rewrite that changes **results** in either lane is a bug, full stop. The
  golden corpus is the guard: 154 cases, each run through both lanes from one
  source, results asserted against a declared table (`golden/runner.py`).
- A rewrite that changes **plan shape** in only one lane is acceptable, and is
  in fact unavoidable — a fused `And` and a tagged `DynValue` are different
  objects. This is why §8 recommends the golden corpus keep asserting results
  only, and why plan-shape assertions live in a Mojo test that fixes the lane.
- The dangerous middle is a rewrite that is *legal* in one lane and *illegal* in
  the other. None is known; the safeguard is that every rule's correctness
  condition must be stated in terms of `Relation` node structure and
  `referenced_columns()`, both of which are lane-agnostic by construction.

The concurrent subquery spec sidesteps this by giving its predicate a distinct
non-`Value` type, so nothing needs inspecting — "the box is the erasure
boundary; a node never needs an erased variant." That generalises to §2.3's
finding but not to constant folding: a conjunction's halves are *existing*
nodes that merely need boxing, whereas a folded constant is a *new* node whose
type must be computed. The precedent holds exactly as far as the split does.

**One correctness note that must not be lost.** Splitting `a AND b` into
`filter(a).filter(b)` is sound under Kleene logic *because `FilterProcessor`
keeps only `True`*: `filter()` honours the mask's validity
(`marrow/expr/execution.mojo:551-563`, and the `fix(kernels): honour mask
validity in filter` commit), so `NULL` is dropped. Every three-valued
combination agrees between the fused and the split form. Marrow has a kleene
golden area; a `filter_and_with_nulls` case belongs alongside the rule.

### 2.6 Constant folding — **out, and probably permanently**

For the **fused lane** LLVM already does it: a `NumericLiteral` is a broadcast
constant inside the SIMD lane body, so `lit(2) + lit(3)` folds during codegen
with no plan rewrite. A plan-level folder would duplicate the backend.

For the **runtime lane** it is expressible (`DynValue` is a tree) and would save
per-morsel interpreted work, but it needs to *construct* a folded literal node
of the right dtype, which means a `dispatch_*` ladder inside the optimizer — the
`from_data` 30-arm-ladder shape that `docs/backlog.md:988-993` measures at
**+106,276 bytes** when pulled into a binary that does not otherwise link it.

And the workload does not want it. ClickBench predicates are
`column op literal`; there is no constant arithmetic to fold. Out.

### 2.7 Filter merging — **phase 2, paired with splitting**

Collapsing stacked `Filter`s saves one full materialization pass (each
`FilterProcessor.pull` rebuilds every column, `execution.mojo:556-561`). It is
lane-agnostic and cheap. But it requires `Filter` to hold `List[BoxedValue]`
rather than one predicate — `BoxedValue` is deliberately *not* a `BoolValue`
operand (`values.mojo:502-510`), so two boxed predicates cannot be combined into
an `And`. That is the same representation change conjunction splitting needs,
and the two are each other's inverse: shipping the merge without the split
collapses filters a user wrote separately, which is a small win; shipping the
split without the merge is a small loss. Ship them together.

### 2.8 Limit pushdown — **out, on inspection of the engine**

DuckDB has a `LIMIT_PUSHDOWN` pass
(`../duckdb/src/optimizer/limit_pushdown.cpp`, step 27). Marrow does not need
one, because the engine is **pull-based**: `LimitProcessor.pull` raises
`Exhausted` as soon as `_remaining <= 0` and never pulls its input again
(`marrow/expr/execution.mojo:619-637`), and `ParquetScanProcessor` decodes a row
group only when pulled. Laziness already delivers the entire benefit, and the
one place a limit *could* usefully move — below a `Sort` or `Aggregate` — is
where moving it is illegal. `topk_fold` is the legal special case and already
exists.

This is a rule that is in every textbook and every prior-art pipeline and is
worth exactly nothing here. It is listed to record the reasoning, so it is not
re-proposed.

### 2.9 Join reordering — **out of every phase**

Three independent blockers, any one sufficient:

1. **No plan-time statistics.** DuckDB's is the only cost-based pass it has, and
   it needs a `CardinalityEstimator` (39 KB of source) plus a `CostModel`
   (`../duckdb/src/optimizer/join_order/`). Marrow has no catalog and reads no
   Parquet footer at plan time. Wiring one in would invert the dependency
   described in §5 and is the boundary condition for ever reconsidering this.
2. **No join graph to reorder.** Marrow's joins come from explicit `.join()`
   calls in a programmatic API — the user wrote the order. Reordering is only
   meaningful when a frontend produces an unordered set of join predicates,
   i.e. after a SQL frontend, which `docs/backlog.md:902-906` records as Won't.
3. **Positional join keys.** `Join.left_key_indices` are positions into the
   child schemas (`relations.mojo:1724-1730`); reordering invalidates them all.

DuckDB's DP is additionally bounded by a recursion-depth guard, a 10,000-pair
emission cap, and a relation-count threshold past which it degrades to greedy
(`../duckdb/src/optimizer/join_order/plan_enumerator.cpp:227-241, 532-543`).
That is the complexity being declined.

### 2.10 CSE — **out of phase 1, and the fused lane makes it ambiguous**

In the fused lane a shared subexpression is *not* obviously a win: materializing
it into a `Project` column breaks the fusion that makes the lane fast, trading
one extra SIMD pass over a fused loop for one buffer write plus one buffer read.
Whether that pays is a measurement nobody has made. In the runtime lane
`DynValue` subtrees are already `ArcPointer`-shared, so an identical subtree
shares memory but is still *evaluated* twice — there the win is real but it
needs interior rewriting (§2.4).

`docs/backlog.md:547-549` already scopes CSE as M2/M3 with a design in
`design-expression-evaluation.md`. Left there. If it is ever built, the first
deliverable is a benchmark showing the fused lane does not regress.

### 2.11 Redundant-sort elimination — **phase 2, low value**

A `Sort` below a hash `Aggregate` is dead work (the grouper destroys order), as
is a `Sort` below another `Sort`. Cheap to detect (`kind()` on the child) and
node-shape only. But in the target workloads `ORDER BY` is last, so it fires
rarely. It is a good *second* rule for the framework precisely because it is the
first one that deletes a node — it exercises R1 and cannot be expressed by
today's mechanism at all.

### 2.12 Projection pushdown through `Join` — **phase 2, unblocks more than itself**

The largest *known* remaining plan gap (`docs/backlog.md:511-514`,
`test_projection_pushdown_stops_at_a_join`). Blocked on `Join.left_key_indices`
being positional: narrowing a child renumbers them.

The fix is to make join keys **name-based**, and it is worth doing for a second
reason: §6 shows that stable column identity across rewriting is decorrelation's
deepest structural demand. One change unblocks both. It is not free — `HashJoin`
does `_right` collision renaming inside the kernel
(`docs/backlog.md:1253-1256`), so "name" is not currently unique across a join's
output — and that is why it is phase 2 with its own review gate, not a
paragraph in phase 1.

---

## 3. Fixed pipeline vs iterate-to-fixpoint

**Fixed pipeline. One pass per rule. No fixpoint.**

Termination is then not a property to be guaranteed at runtime — it is
structural. `optimize()` is a finite sequence of total functions, each a
structural recursion over a finite tree. There is no loop to bound.

The three engines studied disagree about this, and the disagreement is
instructive:

- **DuckDB has no generic fixpoint engine at all.** ~40 passes, each invoked
  exactly once, termination proved per-component — structural recursion for
  plan walkers, per-rule discipline for the expression rewriter, explicit
  counters for the DP search (`../duckdb/src/optimizer/optimizer.cpp:178-443`).
  Two passes run twice; they are *named twice in the list* rather than looped.
- **DataFusion iterates**, capped at `max_passes = 3`, detecting convergence by
  inserting a `LogicalPlanSignature` into a `HashSet` of every previously-seen
  plan — so it detects *cycles*, not merely stability, with regression tests for
  a rule set that rotates a projection `[1,2,3]→[2,3,1]→[3,1,2]→[1,2,3]`
  (`../datafusion/datafusion/optimizer/src/optimizer.rs:611-612, 739-746,
  909-934`). Non-convergence at the cap is **silent** — there is no warning that
  a rule kept changing the plan.
- **polars is a hybrid, and shows the failure mode.** Its top-level `optimize()`
  is a fixed single-shot sequence, but `StackOptimizer::optimize_loop` runs its
  six light rules to fixpoint at two nesting levels with **no iteration cap** —
  a non-converging rule hangs rather than warning
  (`../polars/crates/polars-plan/src/plans/optimizer/stack_opt.rs:20-93`).

Marrow takes DuckDB's position, and gets DataFusion's convergence *evidence*
without its loop.

**How the "would a second pass have helped?" question is answered without
paying for one:** an assertion, not an iteration. Under `-D ASSERT=all` (which
is how `test_*.mojo` already builds), `optimize()` runs the pipeline a second
time and asserts the plan is unchanged. A rule that would have needed another
pass fails a test rather than silently under-optimizing in release. The check
needs a plan fingerprint, which is the recursive `explain()` in §4 — the two
should land together.

**Additionally, assert schema invariance inside `optimize()` itself.**
DataFusion runs `assert_expected_schema(prev_schema, plan)` after *every single
rule*, **in release builds, not just debug**
(`../datafusion/datafusion/optimizer/src/optimizer.rs:768-782`), on the grounds
that a rule silently changing a plan's output schema is a class of bug worth
paying for continuously. Marrow's equivalent is cheaper than DataFusion's
because the invariant is stronger here — *no* rule in any phase may change the
root schema — so it is one `Schema` comparison per rule. Recommend
`debug_assert` (which `-D ASSERT=all` turns on for every test run) rather than
an unconditional check, since marrow's rule count is three rather than
twenty-five and the gate in §7 is measured in kilobytes.

If a future rule genuinely needs iteration (predicate pushdown creating fresh
pushdown opportunities is the classic case), bound it explicitly and locally:
`for _ in range(MAX_PASSES)` with an early break on an unchanged fingerprint,
inside that one rule. Do not add a driver loop over the whole pipeline.
DataFusion's `max_passes` exists because its 25-rule set has mutually-enabling
rules discovered after the fact; marrow's ordering constraint (§2.2) has exactly
one edge today, and a cap that is never reached is a cap nobody maintains.

---

## 4. Where it runs, and how a caller inspects the result

**`optimize()` stays public and `execute()` keeps calling it.** Two reasons:
all 154 golden cases exercise the optimizer only because `execute()` calls it,
and the Python lazy frontend has no other hook. Its signature and its contract —
"same rows, same schema" — are unchanged.

**No `execute(optimize=False)` flag.** The unoptimized path already exists and
is spelled `plan.to_processor(ctx).collect()`, since `execute` is exactly
`optimize().to_processor(ctx).collect()` (`relations.mojo:571-593`). Adding a flag
would be a second way to say the same thing.

**Inspection.** Tests already assert on rewritten plans by walking `children()`
and reading the scan's schema (`_scan_columns_of`,
`marrow/expr/tests/test_pushdown.mojo:275-283`). That surface is sufficient and
stays. Two additions:

- Widening `kind()` (§1.3a) makes `downcast[T]()` safe to use from a test
  without a schema-shaped proxy — `assert_equal(node.kind(), RELATION_FILTER)`
  rather than inferring node type from field names.
- **A recursive `explain()`.** `write_to` renders one shallow label because no
  node renders its children (`docs/backlog.md:521-524`, open). It is now
  needed twice over: as the readable assertion surface for plan-shape tests, and
  as the fingerprint the idempotence assertion in §3 compares. It should land in
  phase 1 with the framework rather than as a separate card.

---

## 5. Interaction with `pruning.mojo`

**`pruning.mojo` is not in the pipeline, and should not be.**

It is a *physical, data-dependent* concern: `PruneStats` carries one row group's
or one page's `[min, max]` bounds, and `Value.prune(stats)` is evaluated inside
`ParquetScanProcessor` once per row group, against statistics the optimizer has
never read. The optimizer has no `PruneStats` to give it and no file to open.

The division is clean and the dependency is one-directional:

```
optimize.mojo  →  relations.mojo  →  values.mojo  →  pruning.mojo  →  kernels.interval
   (plan-time)                                          (run-time)
```

The optimizer's entire responsibility toward pruning is **delivery**: put a
predicate somewhere `pruning.mojo` can reach it. That is rule 1 (§2.1), and
generalising it from "adjacent scan only" to "walk down to the scan" is the
phase-1 win. Nothing about how a predicate prunes belongs in `optimize.mojo`.

**The boundary condition, recorded so it is not crossed by accident:** any
*cost-based* rule — join reordering, build/probe side selection, choosing
between a hash and a sort aggregate — needs statistics at *plan* time, which
means opening a Parquet footer from inside `optimize()`. That inverts the arrow
above and turns a pure plan rewrite into an I/O-performing pass. DuckDB accepts
this (its `StatisticsPropagator` is a pipeline pass and its join-order DP
consumes a cardinality estimator); marrow should not, until something forces it.
This is the same conclusion §2.9 reaches from the other direction.

---

## 6. What decorrelation demands

Decorrelation is the most structurally invasive rewrite any engine has, and it
is the framework's stress test. `docs/backlog.md:912-915` currently lists it
under **Won't** ("presupposes the SQL frontend"); a concurrent spec is
revisiting the uncorrelated and semi/anti-join cases. This section states what
the *correlated* case would demand, so the framework is not built into a corner.

**Where it should run: at plan-build time, not in `optimize()`. This is a
genuine trade-off and the engines disagree.**

- **DuckDB and Calcite decorrelate in the planner.** `Binder::PlanSubquery`
  builds a `LogicalDependentJoin` marker during planning
  (`../duckdb/src/planner/binder/query_node/plan_subquery.cpp:589-612`) and
  `FlattenDependentJoins::DecorrelateIndependent` runs once from
  `Planner::CreatePlan` (`../duckdb/src/planner/planner.cpp:131`) — **before the
  optimizer exists**; `OptimizerType` has no decorrelation member. Calcite has
  the same two-stage shape: `SubQueryRemoveRule` converts a correlated subquery
  into an explicit `Correlate` node, and a separate `RelDecorrelator` stage
  removes the correlation.
- **DataFusion decorrelates as optimizer rules**, and early ones —
  `DecorrelatePredicateSubquery` (#7), `ScalarSubqueryToJoin` (#8),
  `DecorrelateLateralJoin` (#9) of 25, ahead of equijoin extraction and both
  pushdowns, because decorrelation *produces* joins that later rules clean up
  (`../datafusion/datafusion/optimizer/src/optimizer.rs:280-317`).

The two are not equally capable, and that is the second half of the trade-off.
DuckDB's dependent-join flattening (Neumann & Kemper) is **total** — there is a
push-down rule per operator type, so every correlated query unnests. DataFusion's
`PullUpCorrelatedExpr` lifts correlated conjuncts to the subquery root and
**refuses** when something structurally blocks the lift: four explicit
`can_pull_up = false` sites cover `Union`/`Sort`/`Extension` holding an outer
reference, a `Limit` holding one outside an `EXISTS`, and any node whose
expressions contain outer references (window functions are named in the comment)
— `../datafusion/datafusion/optimizer/src/decorrelate.rs:137-173`. On refusal
the correlated subquery is simply left in the plan. DuckDB pays for totality
with a 26-case closed switch; DataFusion pays for the smaller rule with holes in
its coverage.

Three reasons marrow should nonetheless take the planner side, one of them
marrow-specific and decisive:

1. Binary size. Decorrelation *constructs* joins, so putting it in `optimize()`
   makes `Join`, `JoinProcessor`, `SwissHashTable`, `rapidhash` and radix
   partitioning reachable from every binary that calls `execute()` — the
   `query_join` gate's whole delta, paid by programs with no join. At the
   builder it is linked only when a subquery is actually constructed. This is
   the `MARROW_CLI_WRITERS` situation exactly (`relations.mojo:255-262`: the
   Parquet and IPC writers cost **+768,988 bytes** on a gate that never calls
   them, and were gated off).
2. It needs the outer plan's schema, which the builder has and a bottom-up
   rewrite would have to reconstruct.
3. A later *optimizer* pass then removes what the planner introduced — DuckDB's
   `Deliminator` demotes a `LOGICAL_DELIM_JOIN` back to a plain join when every
   `DelimGet` was eliminated (`../duckdb/src/optimizer/deliminator.cpp:85-88`).
   That half *is* a pipeline rule and this framework hosts it fine.

Reason 1 is the decisive one and it is the reason a marrow-specific answer
differs from DataFusion's. DataFusion is a dynamically-linked library where an
unused rule costs a `Vec` entry; marrow compiles a query to a standalone binary
where an unused rule costs its entire transitive closure.

**What it demands of the framework — and what it demands of marrow.**

| demand | framework? | marrow today |
|---|---|---|
| replace a node with a different type / arity (unary → binary join) | **yes, and §1.3 provides it**; today's mechanism cannot | — |
| a closed per-node-type case analysis, one arm per operator | **yes** — `kind()` ladder; DuckDB throws on 4 of 26 rather than handling generically | — |
| stable column identity across rewriting | no | **blocked**: names, plus positional join keys |
| mintable fresh scopes for generated relations | no | **missing**: no `table_index` concept |
| a correlated column reference with a `depth` | no | **missing**: a `Value` node addition |
| `DISTINCT` over the correlated columns (`DelimGet`) | no | **missing**: no `Distinct` node (M2.1) |
| null-propagation ("is this expression strict?") to pick INNER vs LEFT | no | **missing** |
| conjunction splitting of synthesized predicates | **yes** — §2.3 | phase 2 |

The framework's obligations are the first two and it meets both. **Everything
else that blocks decorrelation is outside it**, which is the result this section
exists to establish: the framework is not the bottleneck and does not need to be
designed around one.

The deepest of the blockers is column identity. DuckDB identifies every column
as `ColumnBinding{table_index, column_index}`;
`CorrelatedColumnInfo::operator==` compares *only* the binding and treats the
name as decorative (`../duckdb/src/planner/binder.hpp:109-111`), because after
pushdown the same logical column lives at several different bindings and two
distinct correlated columns may share a name. Marrow pushes projections **by
name** and joins **by position** — the worst of both. §2.12 already needs
name-based join keys; decorrelation would need something stronger still. That
is a phase-3 question and it should be answered before, not during, a
decorrelation attempt.

---

## 7. Binary-size impact

### The gate

`pixi run binary_size`, and `benchmarks/binary_size/check_gate.py` for the
enforced threshold. Current baseline (`baseline.json`, osx-arm64, `__text`,
`-O3 -g0`, stripped), `threshold_pct` **0.5**:

| gate | `__text` | 0.5% |
|---|---|---|
| `query_streaming` | 1,484,652 | 7,423 |
| `query_join` | 1,507,836 | 7,539 |
| `query_streaming_agg_fused` | 1,417,476 | 7,087 |
| `query_streaming_agg` | 1,932,404 | 9,662 |
| `query_dynvalue` | 4,871,156 | 24,356 |

**Add `query_runtime` and `query_scan_typed` to `baseline.json` as part of phase
1.** Neither is gated today, and they are the two binaries this work most
directly changes: `query_runtime` is the full erased entry point
(`in_memory_table(...).filter(...).select(...).execute()`) and `query_scan_typed`
pins a comptime `LeafSet`, which is the parameterized-node case R4 exists for.
The baseline `_comment` already records the failure mode — a change that added
+435,072 bytes (+7.09%) to `query_dynvalue` "measured 0.00% on every gate the CI
actually watches" — and this spec would repeat it exactly.

### Expected cost, and the number that rejects the design

**Phase 1 adds no new behaviour**, so the budget is tight on purpose:

- `_virt_with_children`: one pointer per `DynRelation` (8 bytes, not `__text`)
  plus eight `_tramp_with_children[T]` instantiations. Each is a `rebind`, a
  constructor call and a return — order 100 bytes. Estimate **~1 KB**.
- `_virt_with_predicate` + `_virt_with_projection` → `_virt_with_scan_options`:
  removes eight trampolines, adds eight. Net ~0, minus the six
  `with_projection` override bodies that move from per-node code into one rule
  body — which should be a small *reduction*.
- `optimize.mojo`'s rule bodies plus a recursive `explain()`: new code, ~2-3 KB.

**Reject the design if phase 1 exceeds +0.5% on any gate** — i.e. more than
7,423 bytes on `query_streaming`. That is the existing threshold and it is not
raised; a mechanism swap that adds no behaviour has no claim on a raise. If it
lands between 0.25% and 0.5% the `with_children` trampolines should be
re-examined before merging, because that is more than the estimate and means
something is being instantiated per node type that should be shared.

**Phase 2's `conjuncts()` slot is measured separately and has its own veto.**
An eighth trampoline on `BoxedValue` touches every binary. Budget it at
**+0.25% on `query_streaming` and `query_streaming_agg_fused`** (the two fused
gates) — if a slot that exists to serve a rule those binaries never execute
costs more than half the whole-phase budget, take the comptime-`filter()`
overload from §2.3 instead and accept its narrower reach.

**What must be checked with `nm`, not just `size`.** The `_comment` records that
four gates linked *zero* symbols from `marrow.kernels.cast` while a change added
+435,072 bytes to an ungated binary. The analogous risk here is
`optimize.mojo` making `Join`/`JoinProcessor` reachable from a binary with no
join (see §6). Phase 1 has no such rule; the check is
`nm -C` on every gate binary for `marrow::expr::optimize` symbols and for any
`Join` symbol in `query_streaming`, and it should be run once at phase-1 review
to establish the clean baseline against which phase 2 is judged.

---

## 8. Testing

### Unit tests: `marrow/expr/tests/test_optimizer.mojo`

One new file. It cannot share a compilation unit with `test_pushdown.mojo`
cheaply anyway (one selection = one unit), and the existing file's subject is
Parquet pushdown end-to-end, which stays where it is.

Each rule gets three kinds of case:

1. **It fires.** Build a plan, call the rule (not `optimize()`), assert the
   rewritten shape with `kind()` + `children()` + `downcast[T]()`, the way
   `test_pushdown.mojo` already does.
2. **It does not fire when it must not.** One case per correctness condition in
   §2.1 — the renaming `Project` that stops the predicate walk, the `Aggregate`
   that stops it, the `HAVING` filter that must not sink below its aggregate.
   These are the cases that catch a wrong answer rather than a missed
   optimization, and they are the reason each condition is written down
   individually.
3. **Results are unchanged.** Execute both the raw and the optimized plan over a
   real file and compare the batches. `test_projection_pushdown_preserves_results`
   is the template.

Plus two properties asserted over every plan the file builds:
`optimize().schema() == schema()` (the invariant that makes projection pushdown
sound) and `explain(optimize(p)) == explain(optimize(optimize(p)))` (the
idempotence check from §3).

### Golden corpus: **results only — do not add plan-shape assertions**

The corpus asserts results and should keep doing exactly that, for a reason
specific to marrow: **one case source runs in two lanes** (`golden/runner.py`
transpiles the Mojo body into Python against `helpers.NAMESPACE`), and the two
lanes legitimately produce *different plan shapes* for the same source.
`golden/cases/filter_and.mojo`'s `&` is a fused `BoolBinary` in one lane and a
tagged `DynValue` in the other. Freezing plan shape there would either fail on
the divergence or force a per-lane expected block, and the corpus's entire value
is that one file states one truth for both lanes.

`optimize()` runs inside `execute()`, so **all 154 cases already exercise the
optimizer indirectly**, and that is the property to lean on: any rule that
changes results in either lane fails the corpus, in the lane where it broke,
with a diffed table. That is a stronger guard than a plan-shape assertion and it
is free.

**Two things worth adding to the corpus harness, neither of them per-case:**

- A **global structural invariant** in `runner.py`, applied to every case in
  both lanes: assert `plan.optimize().schema()` equals `plan.schema()`, and that
  optimizing twice is a fixpoint. 154 free checks of the two properties most
  likely to break silently, with no expected-output maintenance.
- A **`filter_and_with_nulls` case**, when §2.3 lands, pinning the Kleene
  argument from §2.5 in both lanes.

### What is deliberately not built

No plan-shape golden files, no `EXPLAIN` output snapshots. An `explain()` string
is a rendering, and snapshotting renderings makes every cosmetic change to
`write_to` a corpus-wide diff. `explain()` is for humans and for the idempotence
fingerprint; assertions go through `kind()`.

---

## 9. Phasing

**Phase 1 — the mechanism, at parity.**
`with_children` (abstract, all 8 nodes) · `kind()` widened to a complete
discriminant · `with_scan_options` replacing `with_predicate` +
`with_projection` · `marrow/expr/optimize.mojo` with three rules ·
`scan_predicate_pushdown` generalised to walk to the scan · recursive
`explain()` · `test_optimizer.mojo` · `query_runtime` and `query_scan_typed`
added to `baseline.json`.

*Review gate:* `pixi run binary_size` at **≤ +0.5% on every gate**; `nm -C`
baseline recorded per §7; `pixi run -e dev pytest marrow/expr/tests golden` green
with all 154 golden cases passing in both lanes; `mojo precompile marrow` at 0
errors and 0 warnings. **If the size gate fails, phase 2 does not start** — a
mechanism that costs binary size at zero new behaviour is the wrong mechanism,
and the fallback is to keep `with_projection` and stop here rather than to raise
the threshold.

**Phase 2 — the first rules that pay, and the first that deletes a node.**
`redundant_sort_elimination` (the R1 exercise: it removes a node, which phase
1's mechanism could not) · `Filter` holding `List[BoxedValue]` ·
`conjuncts()` on `Value` / `BoolBinary` / `DynValue` · `filter_merge` ·
predicate pushdown below `Aggregate` for group-key conjuncts.

*Review gate:* the `conjuncts()` slot measured **separately** against the phase-1
baseline on `query_streaming` and `query_streaming_agg_fused` (§7's +0.25%
veto); `filter_and_with_nulls` green in both lanes; a benchmark showing
`filter_merge` is not a regression on a single-filter plan.

**Phase 3 — column identity, and what it unblocks.**
Name-based join keys (with the `_right` collision-renaming question in
`HashJoin` answered first) · projection pushdown through `Join` · predicate
pushdown below `Join`.

*Review gate:* this phase changes a node's contract, so it needs its own design
note before it starts — specifically, whether name-based keys are sufficient or
whether marrow needs a binding identity (§6). **That question should be answered
in writing before any decorrelation work begins**, not discovered during it.

**Not scheduled:** join reordering (§2.9), constant folding (§2.6), CSE
(§2.10), limit pushdown (§2.8), correlated decorrelation (§6).

---

## 10. What was taken from each prior-art optimizer

**DuckDB** (`../duckdb`, HEAD `985b2f2`) — the pipeline shape, and the sharpest
negative result. Its ~40 passes are a **flat, hand-written, straight-line
sequence of `RunOptimizer(...)` calls** with no driver loop and no fixpoint
(`src/optimizer/optimizer.cpp:178-443`); §1.4 and §3 are that, scaled down. Its
two rule mechanisms are disjoint — a `vector` of 29 pattern-matched *expression*
rules whose `Apply` returns `unique_ptr<Expression>` and structurally cannot
touch the operator tree (`src/optimizer/rule.hpp:17-32`), versus plan-level
passes that are hand-written `switch (op->type)` walks — and §1.5 takes the
second and declines the first. Decorrelation runs in the **planner, before the
optimizer** (`src/planner/planner.cpp:131`), which is §6's central
recommendation; `PushDownCorrelatedNode`'s 26-case closed switch that *throws*
on four operator types (`flatten_dependent_join.cpp:1002-1071`) is the evidence
that a closed `kind()` ladder is not a marrow limitation but the normal shape.
`ColumnBinding{table_index, column_index}` with the name treated as decorative
(`planner/binder.hpp:109-111`) is what §6 measures marrow's name-based scheme
against. And its `Deliminator` (`src/optimizer/deliminator.cpp:85-88`) is the
model for "the planner introduces a marker, a pipeline rule removes it when
provably redundant" — the one half of decorrelation this framework does host.

**DataFusion** (`../datafusion`, 54.1.0) — the shape §1.5 rejects, plus two
primitives worth stealing and one warning. Its `OptimizerRule` trait
(`rewrite(&self, plan: LogicalPlan, config) -> Result<Transformed<LogicalPlan>>`
plus `apply_order() -> Option<ApplyOrder>`) held in a
`Vec<Arc<dyn OptimizerRule>>` of 25 rules is the canonical embedded-engine design
and what a reader would expect this spec to copy. Declined on binary size only:
`dyn` dispatch over a rule vector is an open dispatcher by construction, and
`execute()` calls `optimize()`, so the whole catalogue would be live in every
AOT binary. Stolen anyway:

- `Transformed<T>`'s `transformed: bool` — "did this rule change anything" — is
  the right primitive, and reappears in §3 as the fixpoint *assertion* rather
  than as a driver loop.
- `assert_expected_schema` running after **every rule in release builds** is a
  better idea than the same check in a test, and §3 adopts it as a
  `debug_assert`.

The warning is its convergence story: `max_passes = 3` with cycle detection via
a `HashSet<LogicalPlanSignature>`, and **silent** non-convergence at the cap.
Also note its name-based `Column { relation: Option<TableReference>, name }`
forces a documented discipline marrow would inherit — *"it is critical that the
names of expressions are not changed by the optimizer when it rewrites
expressions"*, with `1 + 2` folding to `3` internally while still displaying as
`1 + 2` — and an invariant that every `(relation, name)` in a schema be unique.
That is the tax §6 weighs against DuckDB's positional bindings.

**Apache Calcite** — concepts only, and one decision. `HepPlanner` (an explicitly
ordered `HepProgram`, no cost) versus `VolcanoPlanner` (Cascades-style
branch-and-bound over a memo of `RelSet`/`RelSubset` with trait-satisfying
enforcers) is the fork in the road, and §2.9 takes the `HepPlanner` side
permanently. `HepMatchOrder::{ARBITRARY, BOTTOM_UP, TOP_DOWN, DEPTH_FIRST}` is
what DataFusion shipped as `ApplyOrder::{TopDown, BottomUp}`, skipping Volcano
entirely. The sourced reasons production engines run a fixed pipeline are: (i)
optimization time traded explicitly against plan quality; (ii) DataFusion's own
statement that *"any one particular set of heuristics and cost model is unlikely
to work well for the wide variety of DataFusion users"*; (iii) plan-space
explosion, severe enough that even Cascades implementations gate rule firing
heuristically; (iv) cost-based search is only as good as its statistics, which
file-scan engines with no catalog do not have — marrow exactly (§5); and (v)
debuggability: in a memo there is no "the plan after rule N" to print.

One correction worth recording, because it changes what "fixed pipeline" implies:
**"fixed pipeline" and "no cost model" are separable.** DuckDB has a genuinely
cost-based DP/dphyp join reorderer sitting *inside* an otherwise fixed pipeline;
DataFusion has only physical build-side swapping; polars has essentially none
(a `ROW_ESTIMATE` flag used solely to pick which join side to keep in memory).
Marrow declines both, but for the reasons in §2.9, not because one implies the
other.

Calcite's `SubQueryRemoveRule` — uncorrelated subqueries converted directly to
joins, correlated ones converted to a `Correlate` node and decorrelated in a
separate stage — is the concept behind the concurrent subquery spec's
IN/EXISTS → semi/anti-join lowering, not this one. Its three-valued-logic
handling of `IN`/`NOT IN` (`count(*)`, `count(key)` and a `true` literal
aggregated over the subquery, then case-analysed) is the same null hazard §2.5
raises for conjunction splitting, in a harder setting.

**polars** (`../polars`) — the closest engine in spirit, and the confirmation
that §1.3's mechanism is normal rather than a compromise. Its
`predicate_pushdown` and `projection_pushdown` are **single recursive `push_down`
functions whose body is one big `match` over the `IR` enum** — 17 and 16 arms
respectively, each ending in `Invalid => unreachable!()`
(`../polars/crates/polars-plan/src/plans/optimizer/predicate_pushdown/mod.rs:243`,
`projection_pushdown/mod.rs:368`). That is a closed `kind()` ladder by another
name, chosen in a language that would happily support the registry: polars *has*
an `OptimizationRule` trait, and uses it for six *light* rules while writing the
two heavy pushdowns by hand. §1.3 is that same split, with the light half
omitted because marrow has no rules that want it yet.

Its arena representation (`Arena<IR>` + `Arena<AExpr>`, `Node(usize)` handles,
`arena.replace(idx, val)` as a `mem::replace`) is the one thing marrow does *not*
get. There, replacing a node is O(1) and every parent holding that `Node`
automatically sees the new one. Marrow is in DataFusion's position instead:
`DynRelation` is an `ArcPointer`, so a plan *copy* is O(1) and untouched subtrees
are shared, but rewriting a child still requires rebuilding every ancestor on
the spine — which is what `with_children` (§1.3b) is for. At plan depths under
ten this is not worth engineering around; it is recorded so the arena is not
proposed as a fix for a cost nobody has measured. Its `OptFlags` bitflags
(`PREDICATE_PUSHDOWN`, `PROJECTION_PUSHDOWN`, `SLICE_PUSHDOWN`, … defaulting to
everything-on) are the precedent weighed in open question 6.

---

## 11. Open questions

Genuinely unsettled. Each is a thing to measure or decide, not a thing to
assume.

1. **Does `with_children` actually come in under 1 KB?** The probe compiled the
   shape; it did not measure eight trampolines in a real binary. This is the
   phase-1 gate and the answer changes the design if it is wrong.

2. **Is `with_children` worth its slot at all, versus an eight-arm `_rebuild`
   ladder inside `optimize.mojo`?** The ladder costs zero slots and zero
   trampolines. It loses on compiler enforcement — a new `Relation` conformer
   silently falls through a missing arm — and on repetition across rules. The
   spec picks the virtual for enforcement, but a measurement showing the
   trampolines cost more than ~1 KB should reopen it.

3. **Should `with_scan_options` be one slot or stay two?** Merging is tidy and
   slot-neutral, but it changes a working, tested protocol for no functional
   gain. A reviewer who prefers minimal diff should be able to keep
   `with_predicate` and `with_projection` as-is and add only `with_children`;
   the rule mechanism does not depend on the merge. Slot count would then be 5,
   not 4, which weakens §1.3's headline but not the argument.

4. **Does splitting a conjunction actually help scan pruning?** §2.3 argues no,
   because `AndInterval` composes. Not measured. If a conjunct references a
   column with no statistics while its sibling has good ones, the composed
   `Interval` may be less selective than the better half alone — in which case
   splitting pays *today*, and §2.3's phasing is wrong. One experiment on a
   ClickBench file with a mixed-statistics predicate settles it.

5. **Are name-based join keys sufficient, or does marrow need a binding
   identity?** §2.12 and §6 both land here and neither answers it. `HashJoin`
   renames colliding right-side columns inside the kernel, so "name" is not
   currently unique across a join's output. This is the single largest
   unanswered structural question and it gates phase 3.

6. **Should any rule be gateable, and at comptime or runtime?** Two precedents
   pull different ways. The `MARROW_GPU` / `MARROW_CLI_WRITERS` pattern says
   *comptime*, for anything that drags in a subsystem — that is the only kind of
   gate that saves bytes, and §6 argues decorrelation will need one. polars says
   *runtime*: an `OptFlags` bitfield defaulting to everything-on, with
   `LazyFrame::with_predicate_pushdown(false)` and friends, which is excellent
   for debugging ("is this rule what broke my query?") and saves nothing. They
   are not alternatives and marrow may eventually want both. Neither is needed
   in phase 1. The unresolved part is the comptime one's failure mode: an
   optimizer flag changes performance, not capability, so a gated-off build
   silently runs slower rather than raising — unlike `MARROW_CLI_WRITERS`, which
   raises a named error. Whether that is acceptable is a judgement call not made
   here.

7. **Does moving `topk_fold` out of the builder change any observable plan?**
   It should only ever fire in more places, never fewer, but `Sort` stores its
   own schema rather than deriving it (`relations.mojo:1573-1577` documents the
   care this needs), and a fold in a different position may expose that. Needs
   the existing sort/limit tests run against the relocated rule specifically.


---

## Verification note (main tree, 2026-08-21)

Two checks run against the real tree rather than a synthetic probe, because the
spec's `conjuncts()` argument is load-bearing and its probe modelled a
purpose-built `AndNode[L, R]`.

**The gap:** marrow has no `And` *type*. `And` is
`comptime And = BoolBinary[AndKernel, AndInterval, _, _]` (`values.mojo:1446`)
— a partial specialisation. An override on `BoolBinary` would therefore also
catch `Or` and `Xor`, so `conjuncts()` has to discriminate on the kernel
parameter, which the probe never had to do.

**The risk:** `CLAUDE.md` warns that "a `comptime name: T` trait requirement
does not resolve reliably as `E.name` off an externally-bound generic
parameter", which is exactly the shape such a guard needs.

**Result: it resolves.** A standalone program (since deleted) declaring
`def discriminates[K: BoolBinaryKernel]() -> Bool: return K.name == "and"`
compiled and printed `AndKernel -> True`, `OrKernel -> False`. This agrees with
the parenthetical in that same CLAUDE.md entry — "`Self.K.name` on a kernel
parameter does resolve" — so the guard is on the working side of a documented
limit, not straying past it.

So `BoolBinary.conjuncts()` guarded by `comptime if K.name == "and"` is
available, and the spec's conclusion holds for marrow's actual node shape. The
size cost of the eighth `BoxedValue` slot is still unmeasured and still gated at
+0.25% on the two fused binaries.


## Reconciliation with the AOT-rewrite research (2026-08-21)

`2026-08-21-aot-rewrite-research.md` measured the `conjuncts()` slot that this
spec estimates at "~1 KB" and gates at +0.25% on the fused binaries. Both
numbers were wrong in the same direction, and phase 2 as written **fails its own
gate**:

| | `query_streaming` `__text` | verdict |
|---|---|---|
| slot + trampoline only | +1,060 (+0.073%) | passes |
| slot + an actual conjunctive predicate | **+9,176 (+0.606%)** | **2.4x the +0.25% veto** |

The shipped gates carry no conjunctive predicate, so the first row — which is
what a naive gate run reports — misses the real cost. The research rebuilt
`query_streaming` with its predicate changed to `(a > b) & (a < 100)`, both
ways from identical source.

The cost model: splitting an N-conjunct predicate instantiates N additional
complete `BoxedValue` erasures, nine trampolines each, for sub-nodes that
previously existed only *inside* the `And`'s type and were never boxed. It
scales with the number of distinct conjunct **shapes** across the program, not
with the number of queries.

**This does not invalidate the rule mechanism** — phase 1 adds no `conjuncts()`
and is unaffected. It invalidates the phase-2 costing, and turns
"is conjunction splitting worth it?" into a question with a number attached:
0.6% of `__text` for join-level predicate pushdown. That is a judgement for the
user, not a gate to quietly raise.

Two further corrections from the research:

- The boundary this spec draws — "rebuilding an arbitrary interior node names a
  type that does not exist yet" — is in the wrong place. Naming a new type is
  fine; **conditionality** is the constraint. A type-changing recursive rewrite
  (De Morgan, double-negation elimination) works in the fused lane when each
  node states its own image as an associated type. It breaks the moment a node
  must branch on what its child *is*.
- The discriminator must be a `comptime is_conj: Bool` parameter rather than a
  `where` clause: a `where` clause does defeat overload rule 4, but the
  constraint solver rejects `StaticString.__eq__`. That upgrades this spec's
  style preference into a hard requirement.
