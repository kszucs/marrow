# Architecture

How marrow's compute stack is organized, and why it is shaped this way. This is
a **living document**: it describes what the code does at `b2e7dae`
(2026-08-03), not a plan. Design records and unbuilt specs live in the other
files under `docs/`; the open work lives in `docs/backlog.md`.

Three claims carry the whole design:

1. **One core, two runners.** A kernel defines its per-lane functor once; eager
   execution and fused execution both consume that one definition.
2. **The erasure boundary is the fusion boundary.** `BoxedValue` is the single
   box; inside it a subtree is monomorphized and fused, outside it the plan is a
   walkable runtime tree.
3. **The plan self-executes.** Every relational node builds its own operator.
   There is no central planner switching over node kinds, which is what lets an
   AOT binary dead-code-eliminate everything the query does not mention.

---

## 1. Kernels — three tiers, one functor

`trait Kernel` (`marrow/kernels/core.mojo:16`) is the root: it fixes
`comptime name` and owns the argument checks (`error`, `expect_same_length`,
`expect_same_dtype`) every family would otherwise re-spell. Family traits add
the call shape. Each family exposes up to three tiers:

```
Tier 0  core      — pure per-lane functor. No allocation, no I/O. THE fusion atom.
Tier 1  apply     — eager, typed. Runs the functor over full buffers via views.apply
                    (vectorized, null-propagating). One overload per type family.
Tier 2  dispatch  — eager, type-erased. Runtime dtype -> the right typed apply.
```

Tiers 1 and 2 are **trait defaults**, derived from tier 0 — a concrete kernel is
usually a name and a functor and nothing else:

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

Not every family has a SIMD `core`. Variable-width data has no `SIMD[string, W]`,
so `StringPredicateKernel` (`string.mojo:297`) requires a scalar
`predicate(StringSlice, StringSlice) -> Bool` and defaults an `apply` that walks
rows into a bit-packed `BoolArray`. `LengthKernel` (`string.mojo:47`) is the one
string kernel with a genuine `core[T, W]`, because it reads **offsets** — an
ordinary fixed-stride numeric buffer — rather than bytes.

Dispatch on the widest family the typed leaf accepts: a leaf bound on
`PrimitiveType` already covers temporal, interval and decimal, so it needs one
`dispatch_primitive` arm, not one per family.

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
    def vectorwise[W: Int](self, batch, ctx, mut slot: Int, idx: Int) -> ...:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx).cast[...]()
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[...]()
        return Self.K.core[Self.OutType.native, W](a, b)

comptime Add = NumericBinary[AddKernel, _, _]          # values.mojo:857
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
```

Adding an operation is writing the kernel struct: it is instantly eager
(`apply`/`dispatch`) *and* fusable (through the generic node). The same shape
covers `NumericUnary` (`:716`), `BoolBinary` (`:1027`), `StringUnary` (`:1562`),
`TemporalExtract` (`:2236`) and the rest.

`NumericCompare` (`values.mojo:932`) is the one node carrying **two** kernel
parameters — `K: NumericCompareKernel` for fixed-width lanes and
`S: StringPredicateKernel` for strings — so `comptime Gt = NumericCompare[
GtKernel, StringGtKernel, _, _]` (`:1015`) names both halves of the operator in
one place. `NumericCompareKernel` deliberately knows nothing about strings
(`numeric.mojo:528`): which family `a < b` means is a question about the
operands, and it belongs to whoever interprets the operator.

## 3. Staging — `Value`, `Breaker`, `Context`

Not every operation can be evaluated a lane at a time. The model that resolves
this is **polarity, expressed as conformance**, not a flag.

`trait Value` (`values.mojo:304`) is every node. It fixes two comptime members:

```mojo
comptime OutType: DataType
comptime OutShape: Int   # 0 scalar, 1 columnar   (values.mojo:306)
```

and a `Datum` result — `comptime Datum = Variant[DynScalar, DynArray]`
(`values.mojo:198`). A scalar stays a scalar until something needs a column;
`into_array(d, n)` (`:201`) broadcasts at that point. `OutShape` is what lets a
node know statically whether its child folds to one value (`Reduction.OutShape ==
0`, `:1928`) or a column, so `sum(a) + b` sizes its lane correctly.

`trait Breaker(Value)` (`values.mojo:417`) marks a cross-row or materializing
node. **Conformance is the marker** — it adds no method, replacing a
`comptime IsBreaker: Bool` every node had to set by hand (the same hazard the
deleted `IsErased` posed).

Two hooks implement the staging:

- **`materialize(batch, ctx) -> Datum`** (`:310`) — the family driver. The
  numeric one (`:535`) runs one fused vectorized pass into a `Buffer`; the bool
  one (`:909`) bit-packs into a `Bitmap`; the string one (`:1393`) appends rows
  into a builder.
- **`prepare(batch, ctx)`** (`:334`) — a pre-pass. A breaker runs its own
  `materialize` and appends the result to the `Context`; composites override to
  recurse into their children; a leaf does nothing.

`execute` (`:317`) is the one dispatch shared by every family:

```mojo
comptime if conforms_to(Self, Breaker):
    var i = ctx.size()
    self.prepare(batch, ctx)
    return ctx.get(i)
else:
    return self.materialize(batch, ctx)
```

`Context` (`values.mojo:215`) is per-execute scratch holding stage results
**positionally**: `prepare` appends in DFS order, and each node's `vectorwise`
reads back with the same walking `slot` counter. Keeping results in the context
rather than on the nodes is what keeps expressions immutable and re-executable.

**Fusion happens *above* a breaker, not around it.** `NumericBinary`'s docstring
(`values.mojo:667-670`) states the rule: there is no "materialized" counterpart
node, because a breaker operand *is* a fused leaf — its `vectorwise` just loads
its own stage result out of `ctx`. So `length(s) + 1` is one numeric pass over a
materialized length column, and `sum(a) * 2` is one pass over a splatted scalar.
The tree splits into stages at each breaker; each stage is one fused loop.

Known follow-ups on this model are recorded at `values.mojo:238-246`: `Context.get`
copies a `Datum` per lane; positional slots forgo CSE, so `sum(a)` used twice
recomputes; and independent breakers run sequentially in `prepare` when they
could be scheduled concurrently.

## 4. Two lanes, one box

The expression layer is **two lanes that share no node types**.

| | AOT lane — `marrow/expr/values.mojo` | Runtime lane — `marrow/expr/dynamic.mojo` |
|---|---|---|
| Node | one struct per protocol, parameterized by kernel and operands | one struct, `DynValue` (`dynamic.mojo:236`) |
| Operand types | bound on a family trait (`L: NumericValue`) | not known until execute |
| Output dtype | a comptime type (`Self.OutType`) | `DynType`, answered at run time |
| Evaluation | subtree fuses into one SIMD loop | one call per node into `EvalFn` |
| Built by | `col("a", int64)`, `lit(3, int64)` | `col("a")`, `lit[Int64Type](3)` |

**What is runtime in the runtime lane is the operand *dtype*, not the
operation.** `DynValue` carries a pointer to its evaluator —
`comptime EvalFn = def(List[DynArray], DynPayload, RecordBatch) thin raises ->
DynArray` (`dynamic.mojo:209`) — so `__sub__` names `_binary[SubKernel]` and a
binary links exactly the kernels its expressions mention. The tag string it also
carries drives only `render`/`prune`/`name` and never selects a kernel. Written
first as a single `_eval` switch over ~70 tags, it cost **+1,807,168 bytes of
`__text` (+45.7%)** on `query_dynvalue`, because every arm became reachable from
every node.

**`BoxedValue` (`relations.mojo:155`) is the erasure box — the one box both lanes
erase into.** It is a wrapper, not an interpreter: `_exec_tramp[V]` calls
`V.execute` on the *concrete* node, so a fused expression stays monomorphized and
its SIMD loop is entered through one indirect call per morsel. The constructor is
generic; the struct is not — which is why `Filter`/`Project`/`FilterProcessor`
compile exactly **once** no matter how many expression types exist.
Parameterizing the operators instead (`Filter[P]`) would fuse just as well and
duplicate the whole operator per predicate.

The box exposes only *metadata* beyond `execute`: `prune`, `name`, `render`,
`referenced_columns`, `bound_column`, `resolve_names`. That list is deliberate —
it is exactly what a plan rewrite needs and nothing that would require seeing
inside a fused subtree.

> **A node never needs an erased variant.** `DynValue` conforms to `Value` and to
> nothing else, because `Value`'s members are all runtime methods. It used to also
> claim `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue` so fused nodes
> would take it as an *operand*; that was unsound — those traits promise a comptime
> `OutType: NumericType` and a `vectorwise` lane the box could only stub. Erase into
> a trait of methods, never into one with comptime members you cannot supply.

## 5. Relations — a walkable plan that executes itself

`marrow/expr/relations.mojo` holds the plan IR. `trait Relation`
(`relations.mojo:103`) nodes are **pure, immutable descriptions**: they hold their
parameters and child relations, and no execution state. `DynRelation`
(`relations.mojo:291`) erases a node behind an `ArcPointer`, so copying a plan is
an O(1) share and the plan is a reusable, inspectable, rewritable template.

Execution is a separate layer. `Relation.to_processor(ctx)` (`relations.mojo:135`)
builds the stateful `Processor` (`execution.mojo:78`) that runs, opening its
children recursively; the processor owns *all* mutable state — scan offset, built
hash index, grouper, child processors. `DynProcessor` (`execution.mojo:94`) erases
it and drives the pull loop; `collect()` (`execution.mojo:135`) drains it into one
`RecordBatch`. `DynRelation.execute()` (`relations.mojo:393`) is
`to_processor(ctx).collect()`, and it never mutates the plan — so a plan runs
repeatedly and concurrently.

**There is no `Planner`.** Each node builds its own operator through its own
`to_processor`. A central builder switching over node kinds would make
`AggregateProcessor` and `JoinProcessor` — and therefore `kernels/join`,
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
(AOT) 1,302,900 versus `query_dynvalue` (runtime) 3,984,756.

**Trust the gate, not the prose.** `benchmarks/binary_size/` is the live
measurement; ratios quoted in any document (including this one) go stale.
`pixi run binary_size` runs the sweep, and `size -m <binary>` → `Section __text`
is the number that matters — a stripped binary's *file* size is quantized to
16 KB pages on Apple Silicon and will invent or hide a regression.

## 7. What the boundary admits

> **erasure boundary = fusion boundary = rewrite granularity.**

Above the boundary — relations, projection lists, predicates — everything is
runtime, walkable and rewritable, and does not fuse (it is already columnar and
`DynArray`-erased). Below it, inside one `BoxedValue`, is a single monomorphized
fused kernel, opaque to rewrites. That partitions the rewrites cleanly:

- **Move, drop or reorder whole sub-expressions** — projection pushdown,
  predicate pushdown, conjunct splitting, join reordering. These live *above* the
  boundary and need only metadata from each box (`referenced_columns`,
  `bound_column`, `name`), never its internals, plus O(1) cloning, which an
  `ArcPointer` already gives. **Fully supported.**
- **Restructure the inside of an expression** — CSE across expressions,
  reassociating `a + b + c`, constant folding inside a fused tree. These need to
  see *through* the box, so they must happen **before boxing** or they cost
  fusion.

Boxing loses zero fusion, because the relational layer never fused across
operators in the first place: a `Filter` materializes its mask and then filters
each column. All fusion is intra-expression, and it lives entirely inside one box.

Two properties make relation-level rewrites cheap. Columns resolve **by name**
against `batch.schema` at execute time (`values.mojo:575`, `:1479`), so narrowing
a scan changes column positions without rewriting a single expression. And
`BoxedValue.resolve_names` swaps only the erased pointer, keeping the node's type
— which is why binding names is not a re-boxing.

Granularity is a lowering choice, not a fixed rule: one box per whole predicate
gives maximum fusion and opaque internals; one box per conjunct keeps full fusion
inside each conjunct and enables conjunct-level pushdown; one box per node is the
runtime lane's model — no fusion, maximal structural visibility. Boxing *is* the
lowering step.

---

## Where to look next

| Topic | File |
|---|---|
| Kernel tiers, fusability classification, lane protocols | `docs/kernel-fusion-architecture.md` |
| Fused-expression internals: `traverse`, slot binding, CSE, scheduling | `docs/design-expression-evaluation.md` |
| Aggregate kernel inversion and the erased-box cost rule | `docs/aggregate-kernel-inversion.md` |
| Window functions (design, unimplemented) | `docs/lane-shape-window-design.md` |
| Milestones, defects, open work | `docs/backlog.md` |
