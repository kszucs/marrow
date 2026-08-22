# The push engine, and three `Dyn*` boxes

Status: proposed, 2026-08-22. Supersedes `physical.mojo`'s pull design and
simplifies `2026-08-22-aggregation-architecture.md` by collapsing two of its
types.

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
    def push(mut self, batch: RecordBatch) raises -> Optional[RecordBatch]
    def finish(mut self) raises -> Optional[RecordBatch]
```

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

Gone: `DynAggregate` and `DynAggregateState`/`DynFold`.

They disappear for a reason rather than by fiat. Once **every logical node has
`to_processor(ctx)`**, an aggregate is no longer a different *kind* of node —
it is a value whose operator returns `None` until `finish`. `DynValue` needs no
extra slot to accommodate it, because `to_processor` is the uniform method, not
an aggregate-specific one. And `Fold` is an `Operator`, so `DynFold` was
`DynProcessor` all along.

The uniform surface:

```mojo
trait Logical:
    def to_processor(self, ctx: ExecContext) raises -> DynProcessor
```

implemented by `Relation` and by `Value` alike.

### What this costs

`project([col("a").sum()])` becomes a **plan-time** error rather than a compile
error — `Project` checks `kind == elementwise` when it is built and raises
"aggregate not permitted here; use .aggregate()". That is what DuckDB,
DataFusion and Polars all do, and it is the price of one box instead of two.
Recorded explicitly because the previous spec claimed the compile-time version
as a requirement (R6); it is now a *preference* that lost to minimality.

## What stays

The logical layer is **untouched** — `logical.mojo`, `core.mojo`, and both
lanes. That is the first real payoff from having kept logical and physical
separate: an engine rewrite that does not reach the plan.

The aggregation architecture's four axes are unchanged: algebra x input x
placement compose into a monomorphized fold, emission belongs to the operator,
and `State`/`Merge`/`If`/`Distinct` remain algebra combinators. The fold is now
spelled as an `Operator` instead of its own trait.

## Order

1. **`expr2` binary-size gate.** Still first — nothing here is measurable
   without it, and this change moves enough code to need a before/after.
2. **`Operator` + `DynProcessor`**, with `BatchSource` as driver and `collect()`
   as the driver loop. Port `Filter`/`Project`. Delete `Exhausted`.
3. **`Kind` on `Value`**; `Project`/`Filter` validate at construction.
4. **`DynValue.to_processor`**; delete `DynAggregate`.
5. **`Grouping`** (`Scalar`, `Hash`), then the fold as an `Operator`.
6. `Aggregate` relation; then combinators; then `PartitionGrouping` + `Window`.

## Carried over — correct under either engine

The uncommitted kernel work stands on its own and should be reviewed and
committed independently:

- `AggState._grow`, and `finish` growing before it reads — a **live
  out-of-bounds**, silent in release builds.
- `AggState.accumulate[W]` — the public lane-shaped scatter, so a fused caller
  never reaches `_mark`.
- `AggState.combine_at` — additive hand-off, keeping the register fold as
  per-batch scratch.
- The verified fold body: masked lanes, mandatory scalar tail, int64 valid
  count.

## Open

- **`Join`'s two sides** are the case that most tests the interface: the build
  side consumes to completion before the probe side streams. A pipeline
  breaker with *two inputs* is what `Operator` must be checked against before
  step 2 is finished.
- **Backpressure / multi-output.** `push` returning one `Optional` assumes an
  operator emits at most one batch per input. A `Filter` that splits, or a
  `Join` that fans out, may need `List[RecordBatch]` or a callback sink.
  Decide before porting operators.
