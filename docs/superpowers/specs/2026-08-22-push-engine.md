# The push engine, and three `Dyn*` boxes

> **Built, and the design moved. Read
> `docs/superpowers/plans/2026-08-23-expr2-aggregation-plan.md` for what
> `expr2` actually is.** This spec is kept for its reasoning, not as a
> description of the tree. Four things below are now stale: `finish` is
> `drain` and is *repeatable* (which is what let `Source` be deleted rather
> than kept); `to_processor` is `to_operator`; `DynAggValue` is gone entirely,
> because an aggregate turned out to be a `Value` rather than a sibling of
> one; and `DynProcessor` is `Pipeline`, which is itself an `Operator`. The
> box count landed at **three**, as this spec predicted, but by a different
> route than it describes.

Status: proposed, 2026-08-22. **Amended 2026-08-23** — the uniform
`to_processor` surface below returns an *associated type*, not `DynProcessor`;
what an operator owns is spelled out; and the claim that the logical layer is
untouched is corrected. Supersedes `physical.mojo`'s pull design and simplifies
`2026-08-22-aggregation-architecture.md` by collapsing two of its types.

## Why

Today's engine is pull: each operator calls `pull()` on its input and raises
`Exhausted` at end of stream. That forces three different executor shapes —
`Processor` (pull), `Fold` (update/finish), and values (pure) — and therefore
five `Dyn*` boxes. Push collapses them, because **blocking stops being a type
distinction and becomes *when you return `Some`***.

It also buys what pull cannot: fusion across relational operators
(`filter().project()` in one pass, no intermediate `RecordBatch`) and clean
partitioning for parallelism. This is why DuckDB and Velox are push-based.

## The interface

```mojo
trait Operator(Deinitable, Movable):
    comptime Out: Copyable
    def push(mut self, batch: RecordBatch) raises -> Optional[Self.Out]
    def finish(mut self) raises -> Optional[Self.Out]
```

**`Out` is an associated type, and that correction came from implementing
this.** The draft fixed `Out = RecordBatch` and claimed one trait covers
relations and values alike. It does not: a relational stage produces a batch, a
*value*'s stage produces a **column**. Forcing `RecordBatch` makes every value
wrap its column in a one-column batch — a `Schema` allocated per value per
batch — only for `ProjectOperator` to unwrap N of them and reassemble one.
`DynOperator[Out]` keeps the erasure surface single: `DynOperator[RecordBatch]`
and `DynOperator[Datum]` are two instantiations of one definition, not two
hand-written boxes. Measured **size-neutral** — byte-identical on both `expr2`
gates.

| | `push` | `finish` |
|---|---|---|
| `Filter`, `Project` | `Some(out)` | `None` |
| a fold (`sum`, `GROUP BY`) | `None` — accumulating | `Some(result)` |
| `Join` build side | `None` | `None`, then probe streams |

**Sources stay pull.** A scan is I/O and naturally a generator, so the source
drives: it produces batches and pushes them through the chain. That keeps the
Parquet and IPC readers unchanged, and it is the same split DuckDB makes.

`Exhausted` is **deleted**. End of stream is `finish()`, not an exception —
which also removes the `String(e) == "Exhausted"` comparison in `collect()`.

## Three boxes, not five

```
DynValue      any value node — elementwise or reduction — via `to_processor`
DynRelation   any relation
DynProcessor  any Operator
```

Gone: `DynAggValue` and `DynAggregateState`/`DynFold`. (The value-level
aggregate trait was renamed `Aggregate` -> `AggValue` on 2026-08-23, freeing
`Aggregate` for the relational node; both its docstrings already claimed the
new name.)

They disappear for a reason rather than by fiat. Once **every logical node has
`to_processor(ctx)`**, an aggregate is no longer a different *kind* of node —
it is a value whose operator returns `None` until `finish`. `DynValue` needs no
extra slot to accommodate it, because `to_processor` is the uniform method, not
an aggregate-specific one. And `Fold` is an `Operator`, so `DynFold` was
`DynProcessor` all along.

The uniform surface — **an associated type, not an erased return**:

```mojo
trait Logical:
    comptime Processor: Operator
    def to_processor(self, ctx: ExecContext) raises -> Self.Processor
```

implemented by `Relation` and by `Value` alike.

**`-> DynProcessor` would have been wrong**, and it is the one correction this
draft needed. A fused subtree's whole value is that it is *one type*, so
erasing at `to_processor` ends fusion at exactly the boundary the AOT lane
exists to cross. With an associated type, `Column[Int64Type].Processor` is
`FusedOperator[Column[Int64Type]]` — concrete, monomorphic, DCE-able — while
`DynValue.Processor` is `DynProcessor`, the same box relations use. The count
is still three; only the *runtime* lane pays for a box.

### What the operator owns, and what stays logical

`bind`, `lane[W]` and `validity` **stay on the logical node**. They are
compile-time composition — `Add[L, R].lane[W]` calls `L.lane[W]` — and moving
them would dissolve the fusion they exist for. What becomes physical is the
loop that *drives* them, together with everything whose lifetime is an
execution:

```mojo
struct FusedOperator[V: NumericValue](Operator):
    var _node: V              # the node itself, never a callback
    var _ctx: ExecContext
```

Holding the node rather than a callback is not a style choice: the interposed
`narrow` closure adapter in the old `variant_dispatch` helper measured
**+662,740 bytes**, and removing it is what recovered that regression. A
callback-parameterised operator is the same shape and should be expected to
cost the same.

This is also what finally gives per-execution state a home. `evaluate` was a
pure function of (node, batch), so `IsIn`'s hash set and a compiled `LIKE`
automaton had nowhere to live but a per-batch rebuild — and `ExecContext` was
held by `FilterProcessor` and never reached the value at all. `push(mut self)`
fixes both.

Two binding levels, not three: per-execution (the operator) and per-batch
(`Bound`). A third, `Prepared`, is **deferred until it has a conformer** —
`IsIn` and `LIKE` are not ported yet, and a level built for nodes that do not
exist cannot be tested.

### `Evaluable` dissolves

`evaluate(batch) -> Datum` is the physical contract inlined into the logical
node; `push` replaces it. `comptime shape` moves to `Analyzable`, joining the
three analysis questions it belongs with, and `Value` becomes
`Analyzable & Logical & Writable & Copyable & Deinitable`.

**There is no incremental rollout.** A trait default cannot return
`Self.AssocType` unless that type is `ImplicitlyCopyable`, and marrow's array
types deliberately are not — so every conformer gains `to_processor` in the
same commit that declares it. Roughly nine source files, one commit.

### What this costs

`project([col("a").sum()])` becomes a **plan-time** error rather than a compile
error — `Project` checks `kind == elementwise` when it is built and raises
"aggregate not permitted here; use .aggregate()". That is what DuckDB,
DataFusion and Polars all do, and it is the price of one box instead of two.
Recorded explicitly because the previous spec claimed the compile-time version
as a requirement (R6); it is now a *preference* that lost to minimality.

## What stays

`logical.mojo` and both lanes keep their **node types**: every `Filter`,
`Project`, `Column`, `Add` and `Sum` survives unchanged in what it *describes*,
so no test that builds a plan changes shape.

**`core.mojo` is not untouched, and this draft's original claim that it was is
withdrawn.** `Evaluable` dissolves, `shape` moves to `Analyzable`, and every
value node gains `Processor` + `to_processor`. The payoff of the logical /
physical split is therefore smaller than first claimed, but it is real and it
is the useful half: the engine rewrite does not reach the *plan a caller
writes*.

The aggregation architecture's four axes are unchanged: algebra x input x
placement compose into a monomorphized fold, emission belongs to the operator,
and `State`/`Merge`/`If`/`Distinct` remain algebra combinators. The fold is now
spelled as an `Operator` instead of its own trait.

## Order

1. ~~**`expr2` binary-size gate.**~~ **Done 2026-08-23.**
   `query_expr2_agg_fused` (1,320,356) and `query_expr2_streaming` (1,358,480)
   are in `baseline.json`. The planned third gate, a runtime-named aggregate,
   **cannot be written yet**: `NumericAggregate[K, A: NumericValue]` accepts
   only a fused input, so `expr2` has no runtime aggregate to measure.
   Running the gate immediately exposed a **pre-existing** +450,112-byte
   regression on `query_join`, bisected to `6c570eb` and filed as backlog
   **S20**. The gate is red for that reason and for nothing in this work; do
   not clear it by re-baselining.
2. **`Operator` + `DynProcessor`**, with `BatchSource` as driver and `collect()`
   as the driver loop. Port `Filter`/`Project`. Delete `Exhausted`.
3. **`Kind` on `Value`**; `Project`/`Filter` validate at construction.
4. **`Logical.to_processor`** with the associated type; delete `DynAggValue`
   and `DynAggregateState`.
5. **`Grouping`** (`Scalar`, `Hash`), then the fold as an `Operator`.
   **This blocks step 4's deletions**, which the draft did not record.
   `GroupByOperator.push` resolves group ids from its grouper and calls
   `state.update(batch, gids, num_groups)`; a fold spelled as an `Operator`
   sees only `push(batch)` and has nowhere to get `gids`. It becomes possible
   once grouping is a *type* parameter — the aggregation architecture's
   `Fold[K, A, G: Grouping]` — so `DynAggValue` and `DynAggregateState` cannot
   be deleted until `Grouping` exists. Step 4 must therefore follow step 5, not
   precede it.
6. ~~`Aggregate` relation~~ **landed 2026-08-23, pull-based.** `Aggregate` +
   `AggregateProcessor`, buffering nothing, with 7 cases covering the ungrouped
   fold, grouping, schema order, positional key naming, a fold over zero
   morsels, a fused subtree, and `HAVING`. It is correct under the pull engine
   and should be **ported** at step 2 rather than rewritten — those tests are
   behaviour specs and should pass unchanged. Then combinators; then
   `PartitionGrouping` + `Window`.

## Carried over — correct under either engine, and now committed

All of this is **in HEAD** as of 2026-08-23; the draft described it as
uncommitted, which is stale.

- `AggState._grow`, and `finish` growing before it reads — a **live
  out-of-bounds**, silent in release builds.
- `AggState.accumulate[W]` — the public lane-shaped scatter, so a fused caller
  never reaches `_mark`.
- `AggState.combine_at` — additive hand-off, keeping the register fold as
  per-batch scratch.
- The verified fold body: masked lanes, mandatory scalar tail, int64 valid
  count.

One correction to the record: the fused aggregate's `update` was reported as
failing to instantiate for a reason "bisected to the *body*, not the plumbing".
That was wrong. The cause was `SIMD[DType.bool, W](True)`, whose positional
constructor asserts `size == 1`; `fill=True` is the whole fix and all 13 cases
pass. **`fill=` is declared only for `SIMD[DType.bool, size]`** — numeric
splats use the positional `Scalar[Self.dtype]` constructor and are already
correct.

## Open

- **`Join`'s two sides** are the case that most tests the interface: the build
  side consumes to completion before the probe side streams. A pipeline
  breaker with *two inputs* is what `Operator` must be checked against before
  step 2 is finished.
- **Backpressure / multi-output.** `push` returning one `Optional` assumes an
  operator emits at most one batch per input. A `Filter` that splits, or a
  `Join` that fans out, may need `List[RecordBatch]` or a callback sink.
  Decide before porting operators.
