# Aggregation architecture — composable, monomorphized

Status: proposed, 2026-08-22. Supersedes `2026-08-22-reductions-design.md`
(both revisions). That document specced the *reduction* layer; this one
overhauls the whole stack, kernels included, because the name proliferation it
tried to manage is a symptom of three axes being tangled rather than composed.

## The problem, counted

`SUM(x) GROUP BY k` passes through fifteen type names: `AggKernel`, `AggState`,
`Aggregation`, `AggFunction`, `NumericAgg`, `TemporalMinMax`, `StringMinMax`,
`CountAgg`, `DistinctAgg`, `ColumnAggregator`, `OneAggregation`, `AggFunc`,
`AggFold`, `AggExpr`, `FoldedAggregates`. Each is a **partial product** of the
same few axes — `NumericAgg` is algebra x dtype, `ColumnAggregator` is algebra x
placement erased, `FoldedAggregates` is that again for N columns.

Separate the axes and compose them instead.

## The axes

| axis | question | conformers |
|---|---|---|
| **algebra** | how do two values combine? | `sum`, `min`, `mean`, `count`, … |
| **input** | what is folded? | any monomorphized `Value` subtree |
| **placement** | which slot does a row contribute to? | `Scalar`, `Hash`, `Partition`, `Sorted` |
| **emission** | when is a slot read out? | at end · per slot · **per row** |

The first three parameterise the fold. **Emission belongs to the relational
operator**, not the fold — which is what lets one architecture cover all three
shapes:

| | placement | emission |
|---|---|---|
| full column reduction | `ScalarGrouping` | once |
| grouped aggregation | `HashGrouping` | per slot |
| **window aggregation** | `PartitionGrouping` | **per row** |

## The stack

```mojo
# kernels/aggregate.mojo — algebra. Pure math, no state.
trait AggKernel:
    comptime AccType[V: NumericType]: NumericType
    comptime needs_count: Bool
    comptime empty_is_null: Bool
    def identity[W]() -> SIMD ; def combine[W](a, b) -> SIMD
    def finalize(acc, count) -> Scalar

trait InvertibleKernel(AggKernel):          # sum/count/mean have it; min/max do not
    def remove[W](acc, value) -> SIMD

# kernels/aggregate.mojo — slots. The ONLY state in the system.
struct AggState[K: AggKernel, V: NumericType]

# kernels/groupby.mojo — placement, extracted from GroupBy's tangle
trait Grouping
struct ScalarGrouping(Grouping)      # one slot; no ids, no scatter
struct HashGrouping(Grouping)        # HashGrouper's ids
struct PartitionGrouping(Grouping)   # window partitions
# SortedGrouping / radix land here as conformers, not as branches

# expr/comptime/aggregates.mojo — the composition
struct Fold[K: AggKernel, A: NumericValue, G: Grouping]

# expr/core.mojo — one value hierarchy
trait Value(Analyzable, Copyable, Deinitable, Writable):
    comptime shape: Shape    # scalar | columnar   — what it produces
    comptime kind: Kind      # elementwise | reduction | analytic — what it reads
```

`Fold[SumKernel, Mul[Column[Int64Type], Literal[Int64Type]], ScalarGrouping]` is
**one type**: algebra, whole input subtree, and placement all comptime. One loop,
no dispatch inside it.

## Combinators — the ClickHouse move

`-State`/`-Merge` are not new machinery. They are **transformers over the
algebra**, so a wrapped kernel is still an `AggKernel` and composes everywhere
one does:

```mojo
struct State[K: AggKernel](AggKernel)     # skips finalize; AccType is K's state layout
struct Merge[K: AggKernel](AggKernel)     # input IS state; combine is state-merge
struct If[K: AggKernel](AggKernel)        # FILTER (WHERE …)
struct Distinct[K: AggKernel](AggKernel)  # COUNT(DISTINCT x)
```

Two-phase aggregation is then the *same* `Fold`, differently composed:

```mojo
Fold[State[SumKernel], Mul[…], HashGrouping]    # partial → a state column
Fold[Merge[SumKernel], Column[…],  HashGrouping]    # final   → a value
```

**This deletes the `state()`/`absorb()` slots** the previous spec put on the
accumulator trait — which repeated an experiment this repo already measured at
**+3.2 MB (+24%)** (`expr/aggregates.mojo:250-253`). The box keeps `update` and
`finish`, two slots.

It is also R11's answer: `count_distinct` stops being a special case that needs
its own state design and becomes `Distinct[CountKernel]`.

## Erasure

Exactly one boundary, at the plan:

```mojo
DynValue.__init__[V: Value]:      comptime assert V.kind != Kind.reduction
DynAggregate.__init__[V: Value]:  comptime assert V.kind == Kind.reduction
```

One `Value` hierarchy — as DuckDB (`BoundAggregateExpression : Expression`),
DataFusion (`Expr::AggregateFunction`), Polars (`AExpr::Agg`), ClickHouse
(aggregates are functions) and marrow's own `expr/` all have. The split moves to
the **boxing site**, where comptime makes it free: `project([col("a").sum()])`
stays a compile error with a written message, `DynValue` stays at five slots,
and `x - avg(x)` becomes expressible once aggregate extraction lands as a
Phase 2 rule.

`DynAggregateState` erases the fold so `AggregateProcessor` can hold
`List[…]` of differing types. Two slots. **The alternative**, opened up by the
combinators: hold `List[DynArray]` of running *state columns* instead, folding
each morsel to state and merging. That removes the box entirely and is
ClickHouse's distributed model — but it costs `O(groups)` merge work per morsel
against in-place scatter, which loses when groups >> rows-per-morsel. In-place
locally, state columns across a boundary; same `Fold` either way.

## Window

Window needs one thing the other two do not: a **moving frame**. `Window` picks
a running fold when `conforms_to(K, InvertibleKernel)` and a
recompute/segment-tree strategy when it does not — a comptime check, so each
instantiation compiles one path. DuckDB and ClickHouse make the same choice at
runtime.

## What this replaces

| deleted | by |
|---|---|
| `Aggregation`, `NumericAgg`, `AggFunction`, `AggKernel.Grouped` | `Fold` composes the same product, in the type |
| `TemporalMinMax`, `StringMinMax`, `CountAgg`, `DistinctAgg` | `Fold` over the matching value family; `Distinct[K]` |
| `ColumnAggregator`, `OneAggregation`, `FoldedAggregates` | `List[DynAggregateState]` |
| `GroupBy`'s strategy tangle | `Grouping` conformers |

Fifteen names to **six plus combinators**, each with one job: algebra, slots,
placement, composition, description, erasure.

**Deletion is not part of this work.** `marrow/tabular.mojo:22-23` imports
`expr/aggregates.mojo` to back `RecordBatch.group_by()`, a shipped PyArrow-mirror
API; `python/bindings/compute.mojo:75-88` uses the kernel traits directly; and
`benchmarks/binary_size/query_streaming_agg_fused.mojo:19` imports `NumericAgg`.
This lands **additively**; the old path retires in a later commit that must
include those three.

## Carried over, already built and verified

- `AggState._grow` + `finish` growing before it reads — a **live
  out-of-bounds**, silent in release builds, that only became reachable when an
  accumulator could see zero batches.
- `AggState.accumulate[W]` — the public lane-shaped scatter, so a fused caller
  never touches `_mark`.
- `AggState.combine_at` — additive hand-off, so the ungrouped register fold
  stays *per-batch scratch* and `AggState` remains the only cross-batch state.
- The fold body: masked lanes (`lane[W]` is null-blind), a mandatory scalar
  tail (a `range(0, n, W)` loop aborts the process), and an int64 valid-count
  as a second accumulator (`sum` of nothing and `sum` of zeros are both 0, and
  it is `mean`'s divisor).

## Measurements this rests on

| | |
|---|---|
| fused vs materialise, grouped, 1M rows | **1.17-1.68x** across g10/g1k/g100k |
| `lane[W]` vs `lane[1]`, grouped | 1.09-1.37x — the scatter stays scalar, the loads do not |
| scatter at one group vs register fold | **14.6x** — why `ScalarGrouping` is a separate conformer |

## Build order

1. `expr2` binary-size gate — R7 measures nothing until it exists.
2. `Grouping` + `ScalarGrouping`/`HashGrouping` — smallest, unblocks the rest.
3. `Fold[K, A, G]`.
4. `Kind` on `Value`; the two comptime-asserted boxing sites.
5. `Aggregate` relation + processor.
6. `State`/`Merge` combinators, with two-phase aggregation.
7. `PartitionGrouping` + `Window`; `If`/`Distinct`.
