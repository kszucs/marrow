# Reductions and aggregation in `expr2`

Status: proposed, 2026-08-22. Supersedes the aggregate sketch in
`2026-08-21-expr2-design.md` (Phase 1a follow-on).

## The one-line frame

A reduction is a **value folded to a scalar**, and marrow can do something no
other engine can: fold a *fused expression* without materialising it.

## Requirements

The design is judged against these. Each is testable; none is aspirational.

| # | requirement | why it is here |
|---|---|---|
| **R1** | `sum(a * 2 + b)` compiles to **one loop with no intermediate array**, accumulator in a register | the only structural advantage marrow has over DataFusion/Polars/ClickHouse, all of which take an already-materialised column |
| **R2** | grouped and ungrouped use the same machinery; ungrouped is `num_groups == 1` | two paths is how the empty-input crash appeared in the first sketch |
| **R3** | streaming: memory is `O(groups)`, never `O(rows)` | `SUM(x)` over a 1B-row scan must not buffer |
| **R4** | one plan holds N reductions of differing types | `sum(int64)` beside `mean(float64)` |
| **R5** | a plan stays copyable and rewritable after processors exist | the optimizer moves aggregates (partial aggregation below a join) |
| **R6** | a reduction used as a value is a **compile** error | `project([sum(a)])` must not silently yield per-morsel partials |
| **R7** | closed erasure: unused kernels are dropped by DCE; `pixi run binary_size` within +0.5% | standing constraint on `marrow.expr` |
| **R8** | the runtime lane can be added later without redesigning | Python-driven aggregates land in Phase 5 |
| **R9** | thread-local partials + merge is expressible | `ExecContext` owns thread count; the fold must be able to use it |
| **R10** | empty input and all-null input both yield NULL | SQL semantics, not the kernel's identity |
| **R11** | `count(*)`, `count_distinct`, `any`/`all` fit without a second architecture | their state is not `AggState[K, V: NumericType]` |
| **R12** | kernels stay typed-first; erasure lives with whoever holds the list | every other kernel file has exactly one thin erased overload; `aggregate.mojo` has four |

## Prior art, and what it does not do

Verified 2026-08-22 against local checkouts.

| engine | accumulator shape | input |
|---|---|---|
| **DataFusion** | `GroupsAccumulator::update_batch(values, group_indices, opt_filter, total_num_groups)`, boxed | `&[ArrayRef]` — materialised |
| **ClickHouse** | `IAggregateFunction` + arena; `sizeOfData()`, `create(place)`, `add(place, columns, row_num)` | `IColumn**` — materialised |
| **Polars** | `evaluate_on_groups` → `AggregationContext`, then column-at-a-time over `GroupsProxy` | `Series` — materialised |

All three erase **at the input boundary**, because none has comptime types: they
*must* compute `a * 2` into an array before folding it. R1 is the requirement
that says marrow does not have to, and it is the one that dictates the rest of
this design. An earlier draft of this spec copied DataFusion's shape and
thereby gave up the only thing the comptime lane is for.

## Placement

Three rules decide where anything goes.

1. **Relations never enter a lane.** A plan holds `DynValue`/`DynReduction` and
   does not know which lane produced them.
2. **Traits and boxes at the top; implementations in a lane.**
3. **Inside a lane, description and execution share a struct** — that is what a
   lane *is*.

```
expr2/
├── core.mojo       Analyzable · Evaluable · Shape · Datum
│                   Value · DynValue
│                   Reduction · DynReduction · Accumulator · DynAccumulator
├── builders.mojo   col · lit · sum · min · max · mean
├── logical.mojo    Relation · DynRelation · InMemoryTable · Filter · Project · Aggregate
├── physical.mojo   Processor · DynProcessor · … · AggregateProcessor
├── comptime/
│   ├── core.mojo   ComptimeValue · NumericValue · BoolValue
│   ├── leaves.mojo · numeric.mojo · boolean.mojo · rules.mojo
│   └── reductions.mojo   Reduction[K, A] + its fused Accumulator
└── runtime/
    ├── values.mojo
    └── reductions.mojo   (deferred — R8)
```

## The logical/physical symmetry

```
                  logical (pure, copyable)          physical (state, move-only)
  relations       Relation ──── to_processor(ctx) ──►  Processor
                    └ DynRelation                        └ DynProcessor
  values            Value ─────── (no separate object) ──┤
                    └ DynValue                           │
  reductions        Reduction ─ to_accumulator() ──────►  Accumulator
                    └ DynReduction                        └ DynAccumulator
```

**A separate physical object exists iff execution owns state outliving a
batch.** Values do not: their physical half is compiled *into* the type
(`bind`/`lane`) or into a function pointer (`RuntimeValue.EvalFn`). Relations
and reductions do, so theirs is a distinct object.

`Bound`/`bind()` is the near-miss — within-batch scratch, destroyed with the
batch, which is why it is an associated type rather than an object.

**Erasure exists iff something holds a heterogeneous collection**, and every box
sits at the *outside* of a monomorphized subtree, so the inside stays fused: one
indirect call per morsel, never per row.

Copy semantics fall out of the same line: `DynRelation`/`DynValue`/
`DynReduction` are `Copyable`; `DynProcessor`/`DynAccumulator` are move-only,
because copying one forks an execution or double-counts a fold.

## The types

```mojo
# core.mojo — shared vocabulary
trait Reduction(Analyzable, Copyable, Deinitable, Writable):
    def to_accumulator(self) raises -> DynAccumulator: ...

trait Accumulator(Deinitable, Movable):
    def update(mut self, batch: RecordBatch,
               groups: Int32Array, num_groups: Int) raises: ...
    def merge(mut self, remap: Int32Array, deinit other: Self) raises: ...
    def finish(mut self, num_groups: Int) raises -> DynArray: ...
```

`Reduction` extends `Analyzable` and **not** `Evaluable` — that is R6. It cannot
reach `DynValue.__init__[V: Value]`, so `project([sum(a)])` fails to compile
rather than broadcasting a partial sum.

`update` takes the **batch**, not a column: the fused accumulator binds the
input subtree itself and reads lanes. That is R1, and it is the single most
important line in this spec.

```mojo
# comptime/reductions.mojo — the monomorphization
struct Reduction[K: AggKernel, A: NumericValue]:
    comptime Type = Self.K.AccType[Self.A.Type]
    var _input: Self.A
    var _name: String
```

Its accumulator's inner loop, ungrouped:

```mojo
var bound = self._input.bind(batch)
var acc = SIMD[Acc, W](Self.K.identity[Acc]())
for i in range(0, n, W):
    acc = Self.K.combine(acc, self._input.lane[W](bound, i).cast[Acc]())
```

`AggKernel.combine` is already SIMD, so the pieces exist. Grouped scatters into
slots instead of a register, still reading `lane[W]` and never an array.

## What gets deleted

From `kernels/`: `Aggregation`, `NumericAgg`, `ColumnAggregator`,
`OneAggregation`, and `GroupBy`'s aggregate-driving. All erase at the *input*
boundary, which is where erasure destroys fusion (R1), and they are why
`aggregate.mojo` has four erased layers where every other kernel file has one
(R12). `HashGrouper` stays — it produces group ids and owes nobody more.

`expr/aggregates.mojo`'s `FoldedAggregates` implements `ColumnAggregator`, so
the deletion is gated on Phase 5. Until then `DynAccumulator` lands **beside**
the existing types, not in place of them.

## Open risks

1. **Grouped scatter from a SIMD lane.** Ungrouped keeps the accumulator in a
   register; grouped must scatter `W` lanes to `W` different slots. Whether that
   beats materialise-then-scatter is **unmeasured**, and R1 only claims the
   ungrouped win outright.
2. **`merge`'s remap.** Threads group independently, so partials merge through a
   remap. Kept in the signature rather than promising pairwise merge.
3. **R11.** `count_distinct` needs entirely different state. The `Reduction`/
   `Accumulator` split is what should absorb that — same logical node, different
   physical fold — but the factory is then not always `AggState`-backed, and
   that shape is unwritten.
4. **`count(*)`** has no input value. Desugar to `count(lit(1))` in the builder
   (`Shape.scalar`, reads no columns, so projection pushdown stays correct)
   rather than adding a second node.
