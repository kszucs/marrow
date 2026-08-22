# Design B — one plan value, two readers

Status: proposed, 2026-08-23. An independent alternative to
`2026-08-23-expr-design-a.md`, written against the same three requirements.
Every mechanism claimed below was verified with throwaway compile probes on
`mojo 1.1.0.dev2026081705`; the numbers are from those probes and are labelled
where they are.

## The organising principle

**An expression is not a type and not a node graph — it is a value.** One
`Expr`/`Plan` struct holds a flat arena of `Node`s (opcode, child indices, a
payload index, a type code) plus a string table, and it holds *no types at
all*. Every question anyone asks of a query — which columns it reads, what it
is named, what type it produces against a schema, whether it is well-formed,
how it prints, constant folding, predicate pushdown, projection pruning,
aggregate extraction — is answered by a **non-raising `def` on that value**,
written exactly once. Execution is then a *reader* of the value, and there are
exactly two: `Compiled[p, i]`, a recursive comptime type family that folds a
**comptime** plan value into one monomorphic, fully fused struct with no
dispatch anywhere inside it, and `Interp`, an ordinary runtime walker over the
**same** value. The two lanes are therefore not two node sets, nor one node set
with two leaf kinds — they are *one representation with two evaluation
strategies*, and which you get is decided by one keyword at the call site:
`comptime P = …` compiles the query into the binary, `var p = …` interprets it.

Design A bets that the node set can be shared and the leaves can differ.
Design B bets that **nothing needs to be shared, because there is only one
thing** — and that the AOT lane is a compile-time *fold over data*, not a
hand-built tower of types.

## Why this is worth the change

CLAUDE.md records that a raising `def` cannot run at comptime but a
non-raising one can, and that *the same* `index_of` ran in both worlds. Design
B takes that one observation seriously and makes it the architecture: if the
plan is data and the analysis is non-raising, **the optimizer is one function
that runs at compile time for the AOT lane and at run time for the runtime
lane.**

`marrow/expr2/comptime/core.mojo` states the opposite as a law of the AOT lane:
*"Nothing outside can inspect the structure. A rewriter cannot open a type …
Interior rewrites belong to the runtime lane alone."* That is true of design A
and false of design B. It is the single largest capability difference between
them.

Verified (probe 3): `fold_constants(self) -> Expr`, an ordinary non-raising
`def`, ran at comptime over a plan built by DSL operator overloads, rewrote
`mul(add(a, mul(2, 3)), b)` into `mul(add(a, 6), b)`, and the two plans
compiled to two different monomorphic types that both produced the correct
answer. The same function is callable on a runtime `Expr`.

## Type inventory

Each type has one job. Anything not on this list is not written.

### The representation — `marrow/expr/plan.mojo`

| type | single responsibility |
|---|---|
| `Op` | the closed opcode set, one `UInt16` per operation. **Names** an operation; never selects one. |
| `Node` | one node's fields — `op`, three child indices, `payload`, `tcode`. Fixed size, no heap, no pointers. |
| `Lit` | one literal's bytes and type code. Lives in a side table, indexed by `payload`. |
| `Expr` | arena (`List[Node]`, `List[String]`, `List[Lit]`) + root index. Owns **every** structural query and **every** rewrite. |
| `Plan` | arena of relational nodes over the *same* expression arena, so a whole query is one comptime parameter. |
| `Builder` | appends nodes; `col`/`lit`/`__add__`/`__gt__`/… . The only place a node is created. |

Indices, not pointers: a plan is one flat, copyable, cycle-free value. Verified
(probe 1/2): a struct holding `List[Node]` and `List[String]` **is** usable as a
comptime struct parameter, its methods run at comptime, and reading a field into
runtime code needs an explicit `comptime(…)` wrapper — which is the guarantee
we want, not a wart.

### The compile-time reader — `marrow/expr/compiled.mojo`

| type | single responsibility |
|---|---|
| `NumLeaf[p,i]` / `BoolLeaf[p,i]` | bind one column or one constant, produce a lane. |
| `NumBinary[p,i]` | fuse two numeric operands under one arithmetic kernel. |
| `NumCompare[p,i]` | fuse two numeric operands into packed bits. |
| `Kleene[p,i]` | three-valued `AND`/`OR`, which does **not** fuse — it calls `kernels.boolean._kleene`. |
| `comptime NumNode[p,i]` | *selects* which of the above a numeric node is. |
| `comptime BoolNode[p,i]` | the same, for a boolean node. |
| `comptime TypeOf[c]` | integer type code → Mojo `DataType` type. One conditional chain, one place. |
| `comptime KernelOf[op]` | opcode → kernel type. One conditional chain, one place. |

The recursion is the whole mechanism:

```mojo
struct NumBinary[p: Plan, i: Int](NumericValue):
    comptime L = NumNode[Self.p, Self.p.a(Self.i)]
    comptime R = NumNode[Self.p, Self.p.b(Self.i)]
    comptime Type = TypeOf[Self.p.tcode(Self.i)]
    comptime Bound = Tuple[Self.L.Bound, Self.R.Bound]

    @staticmethod
    def bind(batch: RecordBatch) raises -> Self.Bound:
        return (Self.L.bind(batch), Self.R.bind(batch))

    @always_inline
    @staticmethod
    def lane[W: Int](bound: Self.Bound, k: Int) -> SIMD[Self.Type.native, W]:
        var a = Self.L.lane[W](bound[0], k).cast[Self.Type.native]()
        var b = Self.R.lane[W](bound[1], k).cast[Self.Type.native]()
        comptime if Self.p.op(Self.i) == Op.add:
            return a + b
        elif Self.p.op(Self.i) == Op.mul:
            return a * b
        …
```

Three properties, each verified rather than assumed:

- **The conditional family reduces and carries its trait bound.**
  `comptime NumNode[p,i] = NumLeaf[p,i] if p.is_leaf(i) else NumBinary[p,i]`
  reduces at every instantiation and the result satisfies a trait bound at a
  generic call site (`def run[E: Lane](…)`). Totality makes it terminate:
  index `-1` is a well-formed null leaf, so `NumBinary` at a leaf index still
  names something that exists. This is exactly the discipline CLAUDE.md
  already records for conditional associated types.
- **`Self.L.Bound` composes.** `Tuple[Self.L.Bound, Self.R.Bound]` over a
  *computed* child type reduces. (CLAUDE.md's "chained projection does not
  reduce" caveat is about `Self.Assoc.Assoc` off a trait-bound parameter; a
  projection off a concrete computed type is fine.)
- **An untaken `comptime if` arm never reaches codegen.** Probe 3 put a
  `comptime assert False` poison inside the `Op.div` arm of a plan that names
  only `add` and `mul`. It compiled and ran. So the opcode ladder is a
  *source-level* ladder that costs elaboration time and **zero binary**. It is
  not a dispatch ladder in the sense R1 forbids: no runtime branch, no
  emitted arm for an unselected op.

**Compiled nodes are stateless.** They have no fields at all — every operand,
name, column index and type lives in `p`. A fused subtree is a zero-sized type,
its column indices are resolved at *compile* time (`batch.columns[comptime(p.col_index(i))]`,
not `batch.column(name)`), and `bind` touches nothing but the batch. Design A's
nodes each carry their operands plus a `String` name per leaf.

This is not a stylistic choice: probe 1 established that **a field whose type
is a conditional comptime type is rejected** (`cannot synthesize move
constructor because field 'l' has non-movable … type`), because the conditional
cannot be reduced at the struct's *definition* site where `p` and `i` are still
free. Statelessness is the constraint that makes the family legal, and it turns
out to be the better design anyway.

### The runtime reader — `marrow/expr/interp.mojo`

| type | single responsibility |
|---|---|
| `EvalFn` | `def(List[DynArray], Lit, RecordBatch) thin raises -> DynArray` — how one opcode computes. |
| `Interp` | walk the arena bottom-up, materialising a `DynArray` per node. |

The evaluator table is **a parallel list on `Expr` itself**, filled in by the
builder and left empty when the plan is comptime. Verified (probe 4): a struct
with a `List[def(…) thin -> …]` field *is* usable as a comptime parameter when
that list is empty, so one arena type genuinely serves both lanes.

This is what preserves closed erasure. `Interp` contains **no switch over
`Op`**: the pointer for a node was bound when `Builder.add(…)` created it, so a
program that never calls `Builder.add` never links `AddKernel`. It is the same
property `expr2/runtime/values.mojo` gets from its per-node `EvalFn`, kept
verbatim.

It is also the seam for **lane mixing**, and design B gets that for free rather
than as a feature: an AOT-compiled subtree is registered as an ordinary node
whose `EvalFn` is `NumNode[P, root].evaluate`. A runtime-composed plan holding
comptime-fused expressions — expr2's headline 1.46 MB vs 4.91 MB result — is
just a plan with some compiled ops in its table. **There is no `DynValue`.**

### Execution — `marrow/expr/pipeline.mojo`

| type | single responsibility |
|---|---|
| `Grouping` | *placement*: which slot a row contributes to. `Scalar`, `Hash`, `Partition`. |
| `Fold[p,i,G]` | *composition*: algebra × fused input × placement, as one type. |
| `run_pipeline[p, k]` | compile one linear pipeline of `p` into one function body. |
| `DynOperator` | the **only** erased box: a pipeline, for the runtime lane and the source boundary. |

`run_pipeline` is the relational counterpart of `Compiled`, and it needs no
types at all — a pipeline is linear, so it unrolls:

```mojo
def run_pipeline[p: Plan, k: Int](var src: DynOperator, mut sink: …) raises:
    comptime for s in range(p.pipeline_len(k)):
        comptime if p.step_op(k, s) == Op.filter:      …
        elif       p.step_op(k, s) == Op.project:      …
        elif       p.step_op(k, s) == Op.aggregate:    …
```

Verified (probe 4): a `comptime for` over a comptime plan value, with a
`comptime if` per step and `comptime(p.payload(k))` materialising constants,
unrolls filter + project + fold into **one function body**. There is no
recursive operator type, no per-operator box, and no intermediate
`RecordBatch` between filter and project — LLVM sees straight-line code over
one batch. That is the operator fusion `2026-08-22-push-engine.md` wants, and
it arrives from the same mechanism as expression fusion rather than from a
second design.

Cross-pipeline state (a join's build side, a two-phase aggregate) is
`run_pipeline[p, j]` called from `run_pipeline[p, k]` with `j < k` — comptime
recursion on an `Int`, which terminates by construction.

## Meeting the three requirements

**R1 — monomorphizable AOT, no ladders, small binaries.** A comptime plan
compiles to exactly one type per node, all `@staticmethod`, all `@always_inline`
lanes, feeding the existing `views.apply` driver. Nothing is looked up at run
time: not the operator (comptime `if`), not the dtype (comptime conditional
type), not the column (comptime index), not the kernel (comptime type
parameter). The only ops, dtypes and kernels present in the binary are the ones
some comptime plan names — proven by the poison probe, and by the fact that
`KernelOf[op]` *selects* one type rather than calling through a table.

**R2 — a runtime counterpart sharing as much as possible.** What is shared:
the entire representation, the entire analysis surface (`columns`, `name`,
`dtype`, `render`, `validate`), **the entire optimizer**, the schema resolution,
the plan printer, the aggregate algebra (`AggKernel`, `AggState`), the placement
layer (`Grouping`), and every kernel. What is written twice, and only this: one
`comptime if` arm per opcode in `Compiled.lane`, and one `EvalFn` per opcode in
the builder — each about one line, each calling the same kernel.

Compare what design A shares. Its node set is shared, but `Analyzable` is
implemented *per node struct*: `expr2/comptime/numeric.mojo` today has
`NumericBinary.columns` and `NumericCompare.columns` as byte-identical
twenty-line duplicates, and each new node adds another. Design B has one
`Expr.columns(i)`. And design A has no shared optimizer at all, by its own
statement.

**R3 — clean abstractions, no ad-hoc structs.** Six roles, and nothing else is
written: **representation** (`Node`/`Expr`/`Plan`), **analysis** (methods,
not types), **selection** (the three `comptime` conditional families),
**algebra** (`AggKernel` and its combinators), **placement** (`Grouping`),
**erasure** (`DynOperator`, one box). The count that motivated
`2026-08-22-aggregation-architecture.md` — fifteen aggregate type names — comes
out at three: `AggKernel`, `AggState`, `Fold`, with `Grouping` shared with the
relational layer. `DynValue`, `DynAggregate` and `DynAggregateState` all
disappear on the AOT path, because a comptime plan's heterogeneous aggregate
list is `comptime for`-unrolled into locals of one function rather than
collected into a `List`.

## Aggregation and window

The four axes of `2026-08-22-aggregation-architecture.md` are right and are
kept unchanged — algebra × input × placement compose, emission belongs to the
operator. Design B changes only where each axis comes from:

| axis | design B source |
|---|---|
| algebra | `comptime KernelOf[p.op(i)]` — one conditional chain, one selected type |
| input | `NumNode[p, p.a(i)]` — the fused subtree, so `sum(a*2+b)` never materialises |
| placement | `comptime if p.num_keys(rel) == 0: ScalarGrouping else HashGrouping` |
| emission | the `comptime if` arm in `run_pipeline` |

So `Fold[p, i, G]` is one struct, and the three shapes are three call sites:

- **full reduction** — `Fold[p, i, ScalarGrouping]`, emitted once at end of
  stream. The register fold, which the repo measured at **14.6x** over a
  one-group scatter, is a `comptime if Self.G.is_scalar` inside `Fold`, so
  neither instantiation compiles the other's loop.
- **grouped** — `Fold[p, i, HashGrouping]`, keys evaluated by
  `NumNode[p, key_root]` (fused too), emitted per slot at end of stream.
- **window** — `Fold[p, i, PartitionGrouping]`, emitted **per row**. The moving
  frame picks its strategy with `comptime if conforms_to(KernelOf[…],
  InvertibleKernel)`: a running fold with `remove` where the algebra is
  invertible, recompute or a segment tree where it is not. One path compiled
  per instantiation.

`State[K]` / `Merge[K]` / `If[K]` / `Distinct[K]` remain algebra combinators, so
two-phase aggregation, `FILTER (WHERE …)` and `COUNT(DISTINCT x)` are the same
`Fold` differently composed. They are unaffected by this design; they compose
with `KernelOf` because `KernelOf` yields a type.

The runtime lane folds through the erased `Aggregation` path that
`kernels/aggregate.mojo` already has, sharing `AggKernel`, `AggState` and
`Grouping` in full. The only difference is that its input arrives as a
`DynArray` instead of a lane — which is precisely the fusion the AOT lane
exists to buy.

## What it costs

**Compile time is superlinear in expression size — and it is not this design's
fault.** Measured on the probes (`mojo run`, wall clock, single expression
chain):

| expression nodes | design B (plan value) | design A (nested types) |
|---|---|---|
| 21 | 1.05 s | 0.93 s |
| 101 | 3.06 s | 2.20 s |
| 201 | 19.9 s | 15.4 s |
| 401 | > 120 s (timeout) | > 120 s (timeout) |

The curve is the same shape for both; design B pays about **1.3x** on it for
re-evaluating `p.op(i)` and the conditional chains per node. Real queries are
10–40 expression nodes, where both are 1–3 s. The lesson for either design is
the same: deep monomorphic fusion has a cliff around 200 nodes, and a plan
above that must fall back to the runtime lane. Design B can *detect* that at
comptime (`comptime if p.size() > LIMIT`) and pick the reader automatically;
design A cannot, because it has no plan-size number to test.

Other costs, honestly:

- **An explicit lift.** An AOT query is `comptime P = (col("a", int64) >
  lit(5)).plan()`. Design A's expression *is* its type, with no lift at all.
  This is one line per query and is arguably a feature — it marks exactly which
  queries are compiled in — but it is unambiguously more ceremony.
- **Literals must default to the runtime side table**, or every constant change
  is a recompile. That is not a hardship (it also buys prepared statements and
  plan-type caching: a hundred queries differing only in constants share one
  instantiation), but `Op.const` — a comptime-baked literal, which enables
  strength reduction — becomes an explicit opt-in rather than the default.
- **Every `Expr` method must be non-raising**, so "not found" answers `-1` or
  `""` and all diagnostics funnel through `validate()`. This is the totality
  discipline CLAUDE.md already prescribes; it is still a constraint that will
  be violated by accident.
- **No per-node runtime state.** Because a field of a conditional type is
  rejected, anything a node would want to carry at run time — a compiled
  pattern, an `IsIn` hash set — must live in the runtime side table, indexed by
  `payload`. That is a real restriction on future ops.
- **Symbol names carry the serialized plan.** Every `Compiled[p, i]` mangles
  `p` into its name, so object-file string tables and debug info grow with plan
  size. Design A's type names grow too, but with type names rather than data.
  **Unmeasured**; it should be measured on the `benchmarks/binary_size/` gate
  before this is built.
- **A closed op set becomes a central file.** Adding an operator means an
  opcode, a `comptime if` arm, and a builder entry — three edits to shared
  files, versus design A's "write a struct". Closed-world is what DCE needs, so
  this is the price of R1, but it makes merge conflicts likelier and it means
  no out-of-tree operator is possible without touching marrow.

## Where design B is worse than design A

Stated plainly, because these are the reasons to pick A.

1. **Ergonomics of the AOT DSL.** Design A's `col("a", int64) * lit(2)` is
   already a fused type. Design B needs `comptime P = …` and a `.plan()` lift,
   and any helper that wants to be generic over "an expression" must take
   `[p: Plan, i: Int]` instead of `[V: NumericValue]`. Two parameters where one
   would do, at every generic boundary in the AOT lane.
2. **Diagnostics on the fused path.** A compiler error inside
   `NumBinary[p, 7]` names an index. Design A names
   `Add[Column[Int64Type], Literal[Int64Type]]`. Design B answers this only
   *partly*: `comptime assert p.validate() == "", p.validate()` catches
   plan-level type errors up front with a rendered expression (verified in
   probe 3 — the assert printed `mul(add(a, 6), b)`), but a genuine
   signature mismatch inside a kernel still reports an index.
3. **Readability of the operator semantics.** Design A puts one operator in one
   struct: readable, individually testable, and a mistake is local. Design B
   concentrates every numeric operator into one `comptime if` ladder in one
   `lane`, where an arm nobody instantiates is *never compiled* and therefore
   never even syntax-checked beyond parsing. That is a genuine correctness
   hazard: an untested arm is invisible. It needs a test that instantiates
   every opcode, and that test is the compile-time cliff above.
4. **Error location.** Design A's operand bound (`L: NumericValue`) rejects a
   bad operand at the construction site. Design B rejects it at `comptime
   assert` time — later, coarser, and only if `validate()` covers the case.
5. **Migration risk.** Design A is an incremental refinement of `expr2`, which
   exists and compiles today. Design B replaces `comptime/leaves.mojo`,
   `comptime/numeric.mojo`, `comptime/boolean.mojo`, `comptime/aggregates.mojo`,
   `runtime/values.mojo` and `core.mojo`'s `DynValue` — most of the package —
   and the parts that survive (`Grouping`, `AggKernel`, `AggState`, the kernels)
   are the parts neither design touches. That is a rewrite, and the compile-time
   cliff means it cannot be validated incrementally on small tests alone.
6. **Design A composes with hand-written Mojo.** A user can write their own
   `NumericValue` conformer and drop it into a fused tree. Design B's closed op
   set forbids that outright — the only extension point is the runtime `EvalFn`
   table, which does not fuse.

## Build order

1. `Expr`/`Node`/`Op`/`Builder` + `columns`/`name`/`render`/`validate`, with
   comptime and runtime tests of the *same* methods. No execution yet.
2. `NumLeaf`/`NumBinary`/`NumNode` + the `views.apply` driver. Gate on
   `benchmarks/binary_size/` before anything else lands — including a
   symbol-table size measurement, which is the one unmeasured cost above.
3. `Interp` + `EvalFn`, and the compiled-subtree-as-op registration that
   replaces `DynValue`.
4. `type_of(schema)` and `fold_constants`, run at comptime and at runtime from
   one implementation. This is the requirement-2 proof; it should be a test.
5. `run_pipeline` with `filter`/`project`, and `DynOperator`.
6. `Grouping` + `Fold[p,i,G]` at `Scalar` and `Hash`; then
   `PartitionGrouping` and window; then the `State`/`Merge`/`If`/`Distinct`
   combinators, unchanged from the aggregation architecture.

## Probes

Four throwaway programs verified, and were deleted after:

1. a heap-holding plan value as a comptime struct parameter; the failure mode
   for a field of a conditional type;
2. the recursive `Compiled[p,i]` family end to end — conditional type reduces,
   carries its trait bound at a `[E: Lane]` call site, `Tuple[L.Bound, R.Bound]`
   composes, `comptime(…)` materialises plan data, result correct;
3. DSL construction at comptime via operator overloads, a comptime rewrite
   (`fold_constants`) changing the compiled type, `comptime assert` rendering
   the offending expression, and the poison proving untaken arms are not
   elaborated; plus the compile-time curve against a design-A-shaped equivalent;
4. a `List[EvalFn]` field surviving comptime parameterisation, and a
   `comptime for` unrolling a filter/project/fold pipeline into one function.
