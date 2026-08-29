# Architecture

How marrow's compute stack is organized, and why it is shaped this way. This is
a **living document**: it describes what the code does, not a plan. It is the
single "what the system is" reference for the kernel, expression and relational
layers; unbuilt specs live in the other files under `docs/`, and the open work
lives in `docs/backlog.md`.

§3 was re-verified against the tree on 2026-08-17 and had drifted badly: it
still described the `Breaker` / `Context` / `prepare` staging model that
`7d57398` deleted. Treat a section's line references as evidence it was checked,
and re-check anything that cites a symbol you cannot find.

**§3-§7 were repointed on 2026-08-29 after the expression layer was rewritten**
into `marrow/expr/` (`logical.mojo` + `physical.mojo`, with `comptime/` and
`runtime/` for the two lanes); the package it replaced has been deleted. Names and paths were corrected;
line references into the old `values.mojo` were dropped rather than re-guessed,
and the sections have not otherwise been re-derived against the new tree.
Statistics-based pruning, predicate/projection pushdown and the CLI-output layer
did not survive the rewrite and are not in the tree today.

Three claims carry the whole design:

1. **One core, two runners.** A kernel defines its per-lane functor once; eager
   execution and fused execution both consume that one definition.
2. **The erasure boundary is the fusion boundary.** `DynValue` is the single
   box; inside it a subtree is monomorphized and fused, outside it the plan is a
   walkable runtime tree.
3. **The plan self-executes.** Every relational node builds its own operator.
   There is no central planner switching over node kinds, which is what lets an
   AOT binary dead-code-eliminate everything the query does not mention.

Five standing invariants, in force for any change to this stack:

- **AOT fusion is the differentiator** — a comptime-typed expression
  monomorphizes to one straight-line SIMD loop, `lane[W]` inlining across
  the whole subtree with no dispatch. Preserve it for every node that fuses.
- **No feature may live in only one lane** — the runtime lane
  (`marrow/expr/runtime/values.mojo`) must reach parity with the comptime lane
  (`marrow/expr/comptime/`).
- **Kernels are the shared substrate; only the driver differs.**
- **Fusion is the default; *breaking* is what a trait marks** — the minority is
  marked, so a new node fuses unless it says otherwise.
- **Shape is first-class**, and scalars broadcast by splat, never by eagerly
  building an N-row array.

---

## 1. Kernels — three tiers, one functor

`trait Kernel` (`marrow/kernels/core.mojo:16`) is the root: it fixes
`comptime name` — identity for display and diagnostics, *never* dispatch — and
owns the argument checks (`error`, `expect_same_length`, `expect_same_dtype`)
every family would otherwise re-spell. Family traits add the call shape. Each
family exposes up to three tiers:

```
Tier 0  core      — pure per-lane functor. No allocation, no I/O. THE fusion atom.
Tier 1  apply     — eager, typed. Runs the functor over full buffers via views.apply
                    (vectorized, null-propagating). One overload per type family.
Tier 2  dispatch  — eager, type-erased. Runtime dtype -> the right typed apply.
```

Tiers 1 and 2 are **trait defaults**, derived from tier 0 (plus a scalar
`predicate` for the string family) — a concrete kernel is usually a name and a
functor and nothing else:

```mojo
trait BinaryKernel(Kernel):                       # numeric.mojo:59
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]: ...
    # apply (numeric.mojo:85) and dispatch are defaulted

trait BinaryNumericKernel(BinaryKernel):          # numeric.mojo:117
    # defaults dispatch over dispatch_numeric

struct AddKernel(BinaryNumericKernel):            # numeric.mojo:233
    comptime name = "add"
    # core = a + b — the whole kernel
```

The families and their tier-0 shapes:

| Trait | Where | Tier 0 |
|---|---|---|
| `BinaryKernel` → `BinaryNumericKernel` / `BinaryFloatKernel` | `numeric.mojo:59,117,137` | `core[T,W](a, b) -> SIMD[T,W]` |
| `UnaryKernel` → `UnaryNumericKernel` / `UnaryFloatKernel` | `numeric.mojo:157,198,213` | `core[T,W](a) -> SIMD[T,W]` |
| `NumericCompareKernel` | `numeric.mojo:528` | `core[T,W](a, b) -> SIMD[bool,W]` |
| `StringPredicateKernel` | `string.mojo:297` | `predicate(StringSlice, StringSlice) -> Bool` (scalar) |
| `StringMapKernel` | `string.mojo:126` | per-row transform |
| `LengthKernel` | `string.mojo:47` | `core[T,W](hi, lo) -> SIMD[int32,W]` over offsets |

Dispatch on the widest family the typed leaf accepts: a leaf bound on
`PrimitiveType` already covers temporal, interval and decimal, so it needs one
`dispatch_primitive` arm, not one per family.

### The shape of `core` is set by the *output* width

A per-lane functor constrains only the **output** — it must be `W` fixed-width
lanes. It says nothing about the input. So variable-length *inputs* (strings) do
not disqualify a lane; only variable-length *outputs* do. Three regimes fall
out:

1. **Fixed-width output from a fixed-stride source** — genuinely SIMD, joins the
   numeric/bool loop unchanged. **Shipped** for numeric and bool. `LengthKernel`
   (`string.mojo:47`) is the interesting member: it loads `W+1` entries from the
   **offsets** buffer (fixed-stride `int32`) and subtracts shifted lanes
   (`string.mojo:59-65`), so the string *bytes* are never touched.
2. **Fixed-width output from variable-length content** (`s1 == s2`, `contains`).
   No data parallelism is available over variable-width bytes, so tier 0 is a
   **scalar** `predicate` and `apply` walks rows into a bit-packed `BoolArray`
   (`string.mojo:309-328`). **A pseudo-SIMD lane filling `SIMD[bool, W]` from `W`
   scalar `predicate` calls did not ship** — `StringPredicate` is a breaker.
3. **Variable-width output** (`s1 + s2`, `upper`, `substr`). There is no
   `SIMD[string, W]`, so the lane is per-row: `StringValue.elementwise(...) ->
   String` (`values.mojo:1388`) yields one row and the family driver
   (`:1393`) appends into a builder. **Shipped, and the strongest result here**:
   `upper(col) || "!"` composes in one builder pass and never materializes
   `upper(col)` (`values.mojo:1562`).

**Width caveat.** The fused loop runs at one width `W` per tree, driven by a
single `NativeType`. `BoolValue` declares `comptime NativeType: DType` for
exactly this (`values.mojo:880`): it is the *operand* width that sizes the SIMD
lane, not the output. Nodes whose operands may differ pick `wider` of the two
(`values.mojo:271`) — a distinct question from `promote` (`:262`), which decides
the *value* domain, where every float outranks every integer. A bool breaker
with no numeric operand picks a width outright: `StringPredicate` and `IsIn`
declare `NativeType = DType.int32` (`values.mojo:1690`, `:1754`).

## 2. One generic node per protocol, parameterized by the kernel

Fusion works by encoding the expression tree in **type parameters**, so the
compiler inlines the whole `core` chain into one loop. A kernel is a childless
functor and structurally cannot carry a tree. So the kernel owns the functor and
the expression layer owns the composition — and the composing node is made
generic over the kernel, so the functor still exists exactly once:

```mojo
struct NumericBinary[K: BinaryNumericKernel, L: NumericValue, R: NumericValue](
    NumericValue
):                                                     # values.mojo:664
    comptime OutType = promote[Self.L.OutType, Self.R.OutType]
    ...
    comptime State = Pair[Self.L.State, Self.R.State]
    def state(self, batch) raises -> Self.State: ...
    def lane[W: Int](self, state: Self.State, idx: Int) -> ...:
        var a = self.l.lane[W](state[0], idx).cast[...]()
        var b = self.r.lane[W](state[1], idx).cast[...]()
        return Self.K.core[Self.OutType.native, W](a, b)

comptime Add = NumericBinary[AddKernel, _, _]          # values.mojo:857
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
```

Adding an operation is writing the kernel struct: it is instantly eager
(`apply`/`dispatch`) *and* fusable (through the generic node). The same shape
covers `NumericUnary` (`:716`), `StringUnary` (`:1562`), `TemporalExtract`
(`:2236`), `Reduction` (`:1922`) and the rest.

**One node per *output* family.** A single `FusedBinary[K, L, R]` conditionally
conforming to `NumericValue` *or* `BoolValue` through `where`-guarded witnesses,
and a single input-family-unified `Equal[L, R]` branching on
`conforms_to(L, NumericValue)`, were both probed and **not taken**. The
per-output-family fallback shipped instead: `NumericBinary` (`:664`),
`FloatBinary` (`:780`, float-forcing `/` and `**`), `NumericCompare` (`:932`),
`BoolBinary` (`:1027`), `StringPredicate` (`:1685`, a breaker).

`NumericCompare` (`values.mojo:932`) is the one node carrying **two** kernel
parameters — `K: NumericCompareKernel` for fixed-width lanes and
`S: StringPredicateKernel` for strings — so `comptime Gt = NumericCompare[
GtKernel, StringGtKernel, _, _]` (`:1015`) names both halves of the operator in
one place. That is what replaced the input-family unification: the *operator*,
not the operand family, carries both meanings of `a < b`. `NumericCompareKernel`
deliberately knows nothing about strings (`numeric.mojo:528`). The runtime lane
spells the same thing as `DynValue._compare[N, S]`.

Recursive and nested operations stay out of kernel structs: struct equality is a
recursive `AND` over child comparisons, not a per-lane op, so it belongs at the
composition layer as a free function reusing `EqKernel.dispatch` for leaves.
`equal_any` (`numeric.mojo:583`) is the shipped version — hash-join row
verification and `nullif` both need equality over an *arbitrary* dtype.

### The value families

Five families sit under `Value`, each providing its own fused driver:
`NumericValue` (`:433`), `BoolValue` (`:879`, bit-packed), `StringValue`
(`:1384`), `TemporalValue` (`:2162`, materialize-only — no lane) and `ListValue`
(`:2310`). **The bucket a node falls into is which trait it conforms to, never a
runtime tag.**

`NumericValue` and `BoolValue` are **siblings and stay disjoint** — never merged
into a shared execution trait, never one a subtype of the other. They have
disjoint operator surfaces (`+`/`<` versus `&`/`~`) and disjoint packaging
(`PrimitiveArray` versus bit-packed `BoolArray`); `BoolType` was never made a
`PrimitiveType`. "Bool is a number" is recovered **only** by explicit bridges,
and all four shipped:

| bridge | site | note |
|---|---|---|
| `NumToBool[A: NumericValue](BoolValue)` | `values.mojo:1255` | `x != 0`, pure lane |
| `BoolToNum[To: NumericType, A: BoolValue](NumericValue)` | `:1286` | `True→1`, pure lane |
| `StringToNum[To: NumericType, A: StringValue](NumericValue)` | `:1316` | no value lane → breaker |
| `StringToBool[A: StringValue](BoolValue)` | `:1349` | no value lane → breaker |

**There is no struct value family.** No `StructValue`, no `StructField`, no
reflection of a child dtype into a family-typed leaf: zero occurrences in the
tree. `ListValue` / `ListColumn` / `ListLength` (`values.mojo:2310-2351`) are the
only nested nodes.

## 3. Staging — `Value`, per-node `State`, and the three drivers

Not every operation can be evaluated a lane at a time. The model that resolves
this is **per-node typed state**: each node declares what it needs resolved once
per pass, and a driver then walks the batch calling a pure lane function.

> **Superseded design.** This section used to describe `trait Breaker(Value)`,
> a `prepare(batch, ctx)` pre-pass, and a `Context` holding stage results in a
> positional `List[Datum]`. `7d57398` replaced all three. There is no `Breaker`
> trait, no `Context`, and no `prepare`; `vectorwise` is now `lane`. The
> positional-slot hazard that section warned about at length — "`prepare` and
> `vectorwise` must walk the tree in exactly the same DFS order, by hand … no
> diagnostic" — **is gone by construction**, because state is per-node and
> typed rather than appended to a shared list.

`trait Value` (`values.mojo:396`) is every node. It fixes **one** comptime
member, `OutShape` (0 scalar, 1 columnar), plus runtime methods: `materialize`
/ `execute` returning a `Datum`, and the plan-analysis surface `name`,
`referenced_columns`, `render`, `bound_column`, `prune`, `validity`.

`comptime Datum = Variant[DynScalar, DynArray]` (`:217`) is a stdlib `Variant`,
not a bespoke struct — no `load[W]`, no `is_scalar`. A scalar stays a scalar
until something needs a column; `into_array(d, n)` (`:220`) is the one
lazy-broadcast forcing point.

`Value` deliberately does **not** declare an output dtype. `OutType` lived here
until the `Dyn*` conformance removal and was read by no `[V: Value]` code: the
three nodes bound on plain `Value` — `NullPredicate`, `IsIn`, `WindowFunction` —
each declare their own, and every family trait redeclares it with a tighter
bound.

### The fusion contract lives on the family traits, not on `Value`

`NumericValue` (`:503`), `BoolValue` (`:972`) and `StringValue` (`:1497`) each
add four members:

```mojo
comptime State: Copyable & Deinitable      # resolved once per pass
def state(self, batch) raises -> Self.State
def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[...]
def state_validity(self, batch, state) raises -> Optional[Bitmap[mut=False]]
```

and default `materialize` to their driver. `lane` reads `state` and `idx` and
nothing else — that removal is the optimization, worth 30x on `a + 1` over 1M
rows.

Three drivers, one per family: `_drive_numeric` (`:317`) fills a `Buffer`,
`_drive_bool` (`:345`) bit-packs a `Bitmap`, `_drive_string` (`:367`) appends
into a builder. The numeric and string drivers carry the scalar-eval-once
branch — at `OutShape == 0` they run the lane once and return a scalar, so
`lit(1) + lit(2)` folds with no loop.

`state_validity` exists because a node's result validity and its state are often
the same computation: a breaker whose `State` is its materialized column reads
the bitmap straight off that array instead of re-running the kernel (FU-7a).

**`TemporalValue` (`:2399`) and `ListValue` (`:2549`) declare no fusion contract
at all** — only a fluent surface. Their columns hand themselves back from
`materialize`, and every temporal or list operation is a breaker into the
numeric family. So `Value` has two kinds of sub-trait under one name: a fused
family, and an operator namespace.

### A breaker is a node, not a conformance

There is no marker trait. A breaker is simply a node whose `State` *is* its
materialized result: `state()` runs the eager kernel once into a column and
`lane` loads from it, which is exactly what a column leaf does. So to its
parent a breaker is indistinguishable from a column, and fusion happens *above*
it: `length(s) + 1` is one numeric pass over a materialized length column, and
`rank(a) + b + c` is **one** fused pass, not three. The tree splits into stages
at each breaker; each stage is one fused loop.

Materialization is therefore the universal fallback and fusion is never
all-or-nothing: an operation with no per-lane functor runs eagerly in `state`
and its consumer loads the result per lane, so every kernel is immediately
usable from the expression layer. What does *not* exist is a `Materialized` leaf
adapter — an arbitrary `DynArray` produced outside the tree cannot re-enter a
fused subtree without a node for the operation.


### Classification, as shipped

| Expression | Node | Status |
|---|---|---|
| `a + b`, `a * 2`, `a / b` | `NumericBinary` / `FloatBinary` | fused SIMD |
| `-a`, `abs(a)`, `sqrt(a)` | `NumericUnary` / `FloatUnary` | fused SIMD |
| `a < b`, `a == b` (numeric) | `NumericCompare` | fused SIMD → bit-packed |
| `p and q`, `not p` | `BoolBinary` / `BoolUnary` | fused SIMD |
| `a.cast(int64)`, num ↔ bool | `NumericCast`, `NumToBool`, `BoolToNum` | fused SIMD |
| `is_nan(a)`, `is_inf(a)` | `NumericPredicate` | fused SIMD |
| `upper(s)`, `s1 + s2` | `StringUnary`, `Concat` | fused **per-row** (one builder pass) |
| `date_trunc(ts)` | `DateTrunc` | materialize-only, like a column |
| `is_null(x)`, `not_null(x)` | `NullPredicate` | breaker — reads validity through the kernel |
| `s.len()` | `StringLength` | breaker → `Int32Array` (two passes, Q7.1) |
| `s1 == s2`, `s.contains(x)`, `LIKE` | `StringPredicate` | breaker → full `BoolArray` (Q7.1) |
| `x IN (…)` | `IsIn` | breaker → `BoolArray` |
| `coalesce`, `nullif`, `CASE WHEN` | `ConditionalBinary`, `CaseWhen` | breaker |
| `year(ts)` and the extract family | `TemporalExtract` | breaker → `Int32Array` |
| string → num/bool parse; casts *to* string | the cast nodes in `casts.mojo` | breaker |
| `sum(a)`, `mean(a)`, `min(a)` | `Reduction` | breaker, `OutShape == 0` |
| `any(p)`, `all(p)` | `BoolReduce` | breaker, scalar |
| `row_number()` | `WindowFunction` | breaker, columnar |
| `filter`, `take`, `sort`, `group_by`, `join`, `concat` | relational operators | outside the expression layer |

The nodes live in `marrow/expr/comptime/` (`numeric.mojo`, `boolean.mojo`,
`strings.mojo`, `casts.mojo`, `aggregates.mojo`). Line references were dropped
rather than re-guessed after the rewrite, and `IsIn` / `WindowFunction` no
longer exist as nodes. Worked examples:

```
upper(s1) + s2 + "!"
  Concat[ Concat[ StringUnary[UpperKernel, StrCol s1], StrCol s2 ], StringLiteral ]
  → ONE builder pass, no intermediate string arrays.

s1 == s2  and  a > b
  BoolBinary[AndKernel,
             StringPredicate[StringEqKernel, StrCol s1, StrCol s2],   # breaker
             NumericCompare[GtKernel, …, NumCol a, NumCol b]]         # fused
  → two passes: the string predicate materializes a full BoolArray in prepare,
    then the AND fuses over that mask and the numeric compare.
```

Known follow-ups on this model were recorded in the previous `values.mojo`:
`Context.get` copies a `Datum` per lane; positional slots forgo CSE, so `sum(a)`
used twice recomputes; and independent breakers run sequentially in `prepare`
when they could be scheduled concurrently.

## 4. Two lanes, one box

The expression layer is **two lanes that share no node types**.

| | AOT lane — `marrow/expr/comptime/` | Runtime lane — `marrow/expr/runtime/values.mojo` |
|---|---|---|
| Node | one struct per protocol, parameterized by kernel and operands | one struct, `RuntimeValue` |
| Operand types | bound on a family trait (`L: NumericValue`) | not known until execute |
| Output dtype | a comptime type (`Self.OutType`) | `DynType`, answered at run time |
| Shape | `comptime OutShape ∈ {0, 1}` | the active variant member of `Datum` |
| Evaluation | subtree fuses into one SIMD loop | one call per node into `EvalFn` |
| Built by | `col("a", int64)`, `lit(3, int64)` | `col("a")`, `lit[Int64Type](3)` |

**What is runtime in the runtime lane is the operand *dtype*, not the
operation.** A `RuntimeValue` carries a tag, its children behind `ArcPointer`,
and an optional payload. The per-node evaluator function pointer this section
used to describe was removed: it triggered a compiler miscompile in a
self-referential struct, and the lane now interprets by switching on the tag
(`docs/backlog.md`). That cost is paid only by a binary that builds a runtime
expression at all — the comptime lane never reaches the interpreter, so both
AOT size gates measured **+0 bytes** when the pointer went away.

**`DynValue` (`marrow/expr/logical.mojo`) is the erasure box — the one box both
lanes erase into.** It is a wrapper, not an interpreter: its trampolines call
through to the *concrete* node, so a fused expression stays monomorphized and its
SIMD loop is entered through one indirect call per morsel. The constructor is
generic; the struct is not — which is why `Filter`/`Project`/`FilterOperator`
compile exactly **once** no matter how many expression types exist.
Parameterizing the operators instead (`Filter[P]`) would fuse just as well and
duplicate the whole operator per predicate.

The box carries five function slots — `columns`, `name`, `dtype`, `write`,
`to_operator` — plus a comptime `shape` read once at construction. That list is
deliberately short: it is what a plan needs and nothing that would require seeing
inside a fused subtree.

> **A node never needs an erased variant.** `DynValue` conforms to *nothing* it
> erases — not `Value`, not the family traits. It once claimed
> `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue` so fused nodes would
> take it as an *operand*; that was unsound — those traits promise a comptime
> `OutType: NumericType` and a lane the box could only stub. A box may hold a
> trait-bound value; it should not be one.

## 5. Relations — a walkable plan that executes itself

`marrow/expr/logical.mojo` holds the plan IR. `trait Relation` nodes are **pure,
immutable descriptions**: they hold their parameters and child relations, and no
execution state. `DynRelation` erases a node behind an `ArcPointer`, so copying a
plan is an O(1) share and the plan is a reusable, inspectable template.

Execution is a separate layer, `marrow/expr/physical.mojo`.
`Relation.to_operator(ctx)` builds the stateful `Operator` that runs, opening its
children recursively; the operator owns *all* mutable state — scan offset, built
hash index, grouper, child operators. `DynOperator` erases it. **The engine
pushes**: `push(batch)` answers with what that batch produced and `drain()` with
whatever the operator was still holding; `collect()` runs the whole tree down to
one `StructArray`. `DynRelation.execute()` is `to_operator(ctx).collect()`
wrapped back into a `RecordBatch`, and it never mutates the plan — so a plan runs
repeatedly and concurrently.

**There is no `Planner`.** Each node builds its own operator through its own
`to_operator`. A central builder switching over node kinds would make
`GroupByOperator` and `JoinOperator` — and therefore `kernels/join`,
`kernels/groupby`, `kernels/hashing` — reachable from *every* plan, including one
that never aggregates or joins. That single open dispatcher is what the closed
design exists to avoid.

Plans are built through the verbs on `DynRelation` (`select`, `project`, `filter`,
`aggregate`, `sort`, `limit`, `join`), not by constructing nodes: every verb
*derives* its output schema, whereas the node constructors take one, so a
hand-built plan can declare a schema its own expressions do not produce.

## 6. Why the AOT binary stays small

The size win is a property of **the closed self-executing driver plus a fused-only
value box** — not of encoding the plan shape in the type system. Both halves are
required, and either one alone buys nothing:

- No open dispatcher anywhere on the path (no `Planner`, no tag switch in the
  value box, no `eval` interpreter).
- Per-dtype kernel dispatch is closed: `dispatch_*` resolves a runtime dtype to a
  comptime parameter, and a program that never erases a dtype never links the
  ladder.
- The box holds a monomorphized node and trampolines into it, so nothing about
  per-node compute goes through runtime dispatch.

Consequently an AOT query links the kernels it mentions and nothing else, and the
linker discards the rest. Measured at `b2e7dae`, `__text`: `query_streaming`
(AOT) 1,309,032 versus `query_dynvalue` (runtime) 3,984,756. Both readings
are 2026-08-05; the AOT figure was 1,302,900 before B12.

**Trust the gate, not the prose.** `benchmarks/binary_size/` is the live
measurement and ratios quoted in any document go stale. `pixi run binary_size`
runs the sweep; `size -m <binary>` → `Section __text` is the number that matters,
because a stripped binary's *file* size is quantized to 16 KB pages on Apple
Silicon and will invent or hide a regression.

## 7. What the boundary admits

> **erasure boundary = fusion boundary = rewrite granularity.**

Above the boundary — relations, projection lists, predicates — everything is
runtime, walkable and rewritable, and does not fuse (it is already columnar and
`DynArray`-erased). Below it, inside one `DynValue`, is a single monomorphized
fused kernel, opaque to rewrites. That partitions the rewrites cleanly:

- **Move, drop or reorder whole sub-expressions** — projection pushdown,
  predicate pushdown, conjunct splitting, join reordering. These live *above* the
  boundary and need only metadata from each box (`columns`, `name`, `dtype`),
  never its internals, plus O(1) cloning, which an `ArcPointer` already gives.
  **Admitted by the design, but not implemented** — no rewrite pass, no
  pushdown and no statistics-based pruning exists in the tree today; they went
  with the previous expression package and have not been ported back.
- **Restructure the inside of an expression** — CSE across expressions,
  reassociating `a + b + c`, constant folding inside a fused tree. These need to
  see *through* the box, so they must happen **before boxing** or they cost
  fusion.

Boxing loses zero fusion, because the relational layer never fused across
operators in the first place: a `Filter` materializes its mask and then filters
each column. All fusion is intra-expression, and it lives entirely inside one box.

One property makes relation-level rewrites cheap: columns resolve **by name**
against the input schema at bind time, so narrowing a scan changes column
positions without rewriting a single expression. There is no `resolve_names`
rewrite — parameter values travel *through* an execution as `Bindings` rather
than being substituted into a copy of the plan, so the box never hands back a
re-boxed `DynValue`.

Granularity is a lowering choice, not a fixed rule: one box per predicate gives
maximum fusion and opaque internals; one box per conjunct keeps full fusion
inside each conjunct and enables conjunct-level pushdown; one box per node is the
runtime lane's model. Boxing *is* the lowering step.

---

## Where to look next

| Topic | File |
|---|---|
| Fused-expression internals: `traverse`, slot binding, CSE, scheduling | `docs/design-expression-evaluation.md` |
| Aggregate kernel inversion and the erased-box cost rule | `docs/aggregate-kernel-inversion.md` |
| Window functions (current toy + forward spec, unimplemented) | `docs/window-functions.md` |
| Milestones, defects, open work | `docs/backlog.md` |
