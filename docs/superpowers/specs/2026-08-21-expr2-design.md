# `marrow.expr2` — the second expression system

*2026-08-21*

## Why a second system rather than a refactor

`marrow/expr/` works and is fast. What it lacks is a spine: seven trampoline
slots on `BoxedValue` and eight on `DynRelation` accreted one at a time, each
added to serve one caller, none of them designed together. Two of those slots
(`with_predicate`, `with_projection`) are an optimizer that was never built,
and they cannot express any rule that changes a node's type or arity — which is
every rule worth having.

Refactoring in place means every step must preserve behaviour *and* keep the
tree green, so the accreted shape survives one apology at a time. Building
`expr2/` alongside lets the design be stated once and judged whole, with
`expr/` untouched until `expr2` passes the same tests.

**The golden corpus is the oracle.** 154 cases run in both lanes against
DuckDB-derived expectations. `expr2` is done when it passes all of them, and
the corpus needs no changes to say so — its case bodies name builders, not
internals.

## The one-line frame

```
                              marrow/kernels/
                  ┌────────────────────────────────────────┐
                  │  typed    filter(PrimitiveArray[T], m)  │
                  │  erased   filter(DynArray, DynArray)    │
                  └────────────────────────────────────────┘
                       ▲               ▲               ▲
           ┌───────────┘               │               └───────────┐
   comptime lane                runtime lane                   processors
   And[Gt[Column, Lit]]         RuntimeValue                   SortProcessor
   per element, fused           per node, interpreted          per morsel
```

`expr2` imports `kernels`. `kernels` never imports `expr2`. That rule is why
the entire `expr` package is 6.6% of a query binary and must survive.

## Responsibilities

The leaky-abstraction test from `CLAUDE.md`: name each type's single
responsibility in a few words. Anything needing "and" is a design smell.

| type | responsibility |
|---|---|
| `Value` | an expression — *composed of the three traits below, adds nothing* |
| `Analyzable` | answer questions a rewriter asks |
| `Evaluable` | produce a column from a batch |
| `Writable` | render itself (stdlib trait, not ours) |
| `DynValue` | erase which lane a `Value` came from |
| `Bound` | one fused subtree's column references, resolved against one batch |
| `Relation` | describe a query, immutably |
| `DynRelation` | erase which operator a `Relation` is |
| *(a rule)* | a free function `DynRelation -> DynRelation`; **not** a type |
| `Processor` | pull morsels, owning execution state |
| `DynProcessor` | erase which operator a `Processor` is |
| `AggExpr` | erase whether an aggregate is fused or named |

Twelve types, twelve single responsibilities. Compare with today's `Value`,
which executes, materialises, prunes, names itself, renders itself, lists its
columns, reports whether it is a bare column, reports its validity, *and*
constructs two kinds of distinct-count aggregate.

## The glue is the optimizer's interface, and it gets named

Today's `bound_column`, `referenced_columns`, `name`, `prune` and
`resolve_names` exist because **a rewriter must ask questions of an expression
it cannot open**. That is unavoidable for the comptime lane: the expression's
structure is its type, so nothing outside can inspect it. The methods must be
answered by the node.

What is avoidable is scattering them on `Value` as though they were part of
being an expression. They are the **optimizer's interface**, so they get their
own trait, named for the asker:

```mojo
trait Analyzable:
    def columns(self) -> List[String]
    """Every column name this expression reads."""

    def as_column(self) -> Int
    """This node's column position if it is a bare column reference, else -1."""

    def interval(self, stats: PruneStats) raises -> Interval
    """The range this expression can produce, for statistics pruning."""
```

`Value = Analyzable & Evaluable & Writable`. Three askers, three traits, and a
node implements exactly what someone asks of it.

`conjuncts()` is **not** here. It is measured at **+0.606%** on the comptime
gates — a slot every binary pays for — and its only consumer is Phase 4, which
happens only if conjunction splitting is shown to improve pruning. It is added
with that phase or not at all.

Two of today's methods do **not** survive this test and move out:

- `Value.count_distinct` / `approx_count_distinct` **construct aggregates**.
  Construction belongs in `builders`, not on the thing being constructed.
- `resolve_names` is a **rewrite**, and it is a no-op in the comptime lane
  (`return ptr.copy()`). It moves onto `RuntimeValue`, the only place it does
  work, and stops costing every boxed value a slot.

Net: `DynValue` carries five slots instead of seven, and each is traceable to a
named asker.

### Rules are free functions, not slots and not a trait

```mojo
def push_predicate_to_scan(plan: DynRelation) raises -> DynRelation
```

A straight-line pipeline calls them in order. **Deliberately not a `Rule`
trait**: erasing rules into a `List[DynRule]` would add a second erasure box,
keep every rule permanently live in every binary, and sit one refactor away
from the narrowing-adapter shape that cost **+662,740 bytes**. DuckDB's ~40
passes are likewise a fixed sequence of calls, not a registry.

`with_predicate` and `with_projection` disappear. A rule reads a node through
`to_view()` and rebuilds through `from_view()` — both per-type, so nothing
names all node types in one place.

**That last clause is load-bearing and measured.** `kernels::sort` occupies
**0 bytes** in a binary that never sorts, with dynamic relations, because
`DynRelation.__init__[T]` wires trampolines per constructed type and there is
no registry. A `match` over all kinds would make every operator reachable and
was projected at **+237%**. No construct in `expr2` may name all node types in
one place.

## Layout

```
marrow/expr2/
├── core.mojo              Analyzable · Evaluable · Value · DynValue
│
├── comptime/              structure lives in the TYPE
│   ├── __init__.mojo      re-exports; plain relative imports inside
│   ├── leaves.mojo        Column[T] · Literal[T] · Param[T]
│   ├── operators.mojo     And[L,R] · Gt[L,R] · arithmetic · string · temporal
│   └── reductions.mojo    Reduction[K,In] · WindowFunction
│
├── runtime/               structure lives in FIELDS
│   ├── values.mojo        RuntimeValue
│   ├── reductions.mojo    named aggregates
│   └── rewrite.mojo       constant folding · CSE — runtime-lane only
│
├── builders.mojo          col · lit · count_star — ONE overload set
├── aggregates.mojo        AggExpr
│
├── plan.mojo              Relation nodes · to_view / from_view
├── optimize.mojo          Rule · the pipeline
├── physical.mojo          Processor · plan_to_processor
├── pruning.mojo · params.mojo
└── __init__.mojo          the public surface; the one file that escapes a keyword
```

Placement rule:

| what it is | where |
|---|---|
| knows exactly one lane | that lane's directory |
| **spans** both — a box, an overload set | top level |
| **above** both — a trait | `core.mojo` |
| relational | top level |

`comptime` is a reserved word, so `expr2/__init__.mojo` writes
``from .`comptime` import ...``. Verified: that is the *only* line that needs
it, plus whatever imports both lanes directly. Inside `comptime/` the package
name never appears, and consumers import from `marrow.expr2`.

## Pipelines

```
Relation ──optimize()──► Relation ──plan_to_processor(ctx)──► DynProcessor ──pull()──► RecordBatch
 immutable   Rule values   immutable                            owns state             per morsel

Value ──to_evaluator(schema)──► DynEvaluator ──bind(batch)──► Bound ──lane[W](i)──►
logical                          physical       per morsel             per element
```

Expressions have one stage more than relations because they descend to
per-element. `Bound` names that stage — provisionally; whether it is worth
renaming today's `State`/`state()` is open question 5.

Splitting `Analyzable` from `Evaluable` means **the optimizer cannot reach
execution** — not by convention, but because the type it holds has no
`execute`. The one rule that would need to is constant folding, which is also
the one LLVM already performs for the comptime lane, and which therefore lives
in `runtime/rewrite.mojo`.

## Why the optimizer is lane-asymmetric

A fused subtree compiles to one inlined SIMD loop, so LLVM already applies
constant folding, GVN/CSE, instcombine and DCE to its interior. `RuntimeValue`
returns a `DynArray` **per node, per morsel**, so LLVM sees nothing across
nodes.

| | comptime lane | runtime lane |
|---|---|---|
| value of interior rewrites | ~0 — LLVM did it | high — one fewer materialised column each |
| cost of enabling them | **+0.606%** `__text` | ~0 |

So constant folding and CSE are runtime-only. The comptime lane's cost is only
ever justified by rewrites LLVM *cannot* do — those changing data flow, not
instruction selection: pushing a predicate to a scan (changes what is read),
splitting a conjunction across a join (changes cardinality).

**This is sound in a way lane-asymmetric semantics never is.** An optimization
must not change results, so a rule firing in one lane only is invisible except
in speed. The golden corpus catches a *feature* present in one lane; it has
nothing to say about a rewrite that changes neither answer.

## Rejected, with evidence

| proposal | why not |
|---|---|
| unify `Relation` + `Processor` | `DynProcessor`'s 3 clean slots vs `DynRelation`'s 8 accreted ones — the mess is logical. Welds shut the seam where algorithm choice goes. |
| collapse relations to an enum | DCE is worth **3.5×**; one dispatch switch ≈ **+237%** against a 0.5% gate |
| comptime relations | ceiling **measured at 3.3%**; the size win is already banked by mixing; instantiates per plan *shape* (a product) where today is per node *kind* (a union); no Python path; rules written twice |
| move processors under `kernels` | processors hold `DynValue` — kernels would import `expr2`. Cycle. |
| `Value` → `ComptimeValue` | `RuntimeValue` conforms to `Value`; the rename asserts `RuntimeValue: ComptimeValue` |
| split `builders` across lanes | **verified**: a second import shadows rather than merging; `col("a")` resolved against the 2-arg overload and failed |
| generic fold replacing the traversals | needs a closure generic over its own trait bound — the recorded **+662,740-byte** shape |

## Verified this session

Every row compiled, none recalled.

| claim | result |
|---|---|
| `comptime` as a module name | ✅ with backticks; **one line**, not every import |
| backticked subpackage + `__init__` re-export | ✅ end to end |
| circular imports across subpackages | ✅ |
| overload set spanning modules | ❌ silently shadows |
| struct-with-`List` as comptime parameter | ✅ |
| `comptime assert` on an unknown column | ✅ named compile error |
| one `def` at comptime *and* runtime | ✅ same function, both worlds |
| conditional recursive type rewrite | ✅ fixpoint; **totality** is the enabling trick |
| DCE erases unused operators | ✅ `kernels::sort` = **0 bytes** |
| expression lane worth | **3.4×** — 1.46 MB vs 4.91 MB, same plan |
| plan machinery share | **3.3%**; `builders`+`dtypes`+`arrays` = 73.7% |

Four `CLAUDE.md` entries were wrong or overbroad and are corrected in the file.

## Phases

Each ends at a review gate. `expr/` is untouched until the last.

**0 — probe the two open questions.** Neither blocks the skeleton, both change
scope:

- **`col["a", SCHEMA]()`** — compile-time validation with the dtype *derived*
  rather than retyped. Comptime schemas and `comptime assert` both work;
  unprobed is deriving a *type* from a schema *value*, which may hit the
  reflection limit that deferred `Table[T]`.
- **The physical plan layer.** Deferred earlier on "one join algorithm, nothing
  to choose between" — too weak. Selection-vector propagation and
  downstream-column awareness are *also* physical properties, and
  `FilterProcessor` compacts every column including ones only the predicate
  read. Decide whether `Relation → PhysicalPlan → Processor` goes in now.

**1 — the spine.** `core.mojo`, both lane directories, `builders`, `plan`,
`physical`. No optimizer. Gate: the golden corpus passes against `expr2`.

**2 — `Rule` and the pipeline.** `to_view`/`from_view`; re-express projection
and scan-predicate pushdown as rules; relocate `topk_fold` out of the builder.
Gate: plans identical to `expr`'s, ≤ +0.5%.

**3 — the rules that pay.** Generalised scan-predicate pushdown, so
`parquet_scan(…).select(…).filter(…)` gets pruning it gets none of today.
Narrow sort elimination only — the general case needs the physical layer.

**4 — conditional.** `conjuncts()` at a measured **+0.606%** against a +0.25%
veto, only if Phase 0 shows splitting improves pruning.

**5 — migrate and delete.** Python bindings and golden move to `expr2`; `expr`
is deleted. `expr2` is renamed `expr` in its own commit, so the diff that moves
code and the diff that renames it are never the same diff.

## Success criteria

- The golden corpus passes in both lanes with no case-body changes.
- `pixi run binary_size` within +0.5% of `expr`'s numbers, with `query_runtime`
  and `query_scan_typed` added to `baseline.json` — neither is gated today and
  both are what this work most changes.
- Every type in the responsibility table still has one responsibility.
- No construct names all node types in one place.
- `mojo precompile marrow` at 0 errors, 0 warnings.

## Open questions

1. Does `col["a", SCHEMA]()` compile? (Phase 0)
2. Does the physical plan layer go in now? (Phase 0)
3. Does conjunction splitting improve pruning *today*? If not, Phase 4 has
   nothing to buy and should not happen.
4. Is name-based join-key identity sufficient? `HashJoin` renames colliding
   right-side columns, so "name" is not unique across a join's output. Gates
   decorrelation.
5. Do `Bound` / `bind()` replace `State` / `state()`, or is that churn?
6. From the subquery spec: is `exists(sub, left_on=…, right_on=…)` honest
   naming when the caller supplies the key, and should `IsInKernel` move from
   PyArrow's `MATCH` to SQL's `INCONCLUSIVE`? No golden case can pin the
   latter — golden's oracle is DuckDB.
