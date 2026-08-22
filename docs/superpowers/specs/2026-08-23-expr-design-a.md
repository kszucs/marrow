# Design A — one node set, two leaf kinds

Against three requirements, everything else negotiable:

1. **Monomorphizable AOT** — no downcast ladders, best-in-class fused
   performance, small binaries.
2. **A runtime counterpart sharing as much code as possible.**
3. **Clean abstractions — no ad-hoc structs.**

## The central move

Do not build two lanes. Build **one set of nodes** and **two kinds of leaf**.

```mojo
Add[Column[Int64Type], Literal[Int64Type]]   # AOT leaves      → fuses
Add[RuntimeLeaf, Literal[Int64Type]]         # one erased leaf → that operand materialises
```

Every composite node is generic over its operands' trait. A `RuntimeLeaf`
conforms to the same base trait but answers `comptime IsErased = True`, and each
node keys off the propagated `IsErased` to pick fusion or materialisation:

```mojo
comptime IsErased = Self.L.IsErased or Self.R.IsErased
```

**R2 falls out**: `Add`, `Gt`, `And`, `Fold`, every relation and the whole
optimizer exist once. The lanes differ only in which leaf you built from —
which is `col("a", int64)` versus `col("a")`, the distinction the user already
makes.

**R1 is preserved**: an all-AOT tree has `IsErased = False` everywhere, so the
erased branches are dead-code-eliminated and the tree fuses to one loop.

**R3 is served**: there is one node per operation, not one per lane.

## The taxonomy

Two orthogonal comptime properties on one `Value` trait:

| | |
|---|---|
| `shape` | `scalar` \| `columnar` — what it produces |
| `kind` | `elementwise` \| `reduction` \| `analytic` — what it reads |

`sum(a)` is `scalar` x `reduction`; `sum(a) OVER (…)` is `columnar` x
`analytic`; `a * 2` is `columnar` x `elementwise`. This is ibis's and
DataFusion's model, and marrow's own `expr/`.

## The stack

```
kernels/   AggKernel            algebra: AccType · identity · combine · finalize
           InvertibleKernel     + remove, for moving window frames
           State[K] Merge[K]    combinators — two-phase aggregation, free
           If[K] Distinct[K]    FILTER, COUNT(DISTINCT) — no new hierarchy
           AggState[K,V]        slots — the only aggregate state in the system
           Grouping             placement: Scalar · Hash · Partition · Sorted

expr/      Value                shape x kind — one hierarchy, both lanes
           Column · Literal · RuntimeLeaf          leaves
           Add · Gt · And · Fold[K,A,G] · …        nodes, generic over operands
           Relation                                 plan
           Operator                                 push: push/finish
           DynValue · DynRelation · DynProcessor    three boxes, no more
```

## Physical: push

```mojo
trait Operator(Deinitable, Movable):
    def push(mut self, batch) raises -> Optional[RecordBatch]
    def finish(mut self) raises -> Optional[RecordBatch]
```

Blocking is a *return value*, not a type: a fold answers `None` until `finish`.
Sources stay pull and drive. `Exhausted` is deleted.

That is what makes three boxes possible: `Fold` is an `Operator`, so `DynFold`
is `DynProcessor`, and once every logical node has `to_processor`, an aggregate
is a value whose operator answers late — so `DynAggregate` is `DynValue`.

## What R3 forbids

Every type is one of six roles: **algebra · state · placement · composition ·
description · erasure**. A type that is not one of those is ad hoc and does not
get written. The fifteen aggregate names in the tree today are all partial
products of the first three, which is why they proliferated.

## Costs, stated

- `project([col("a").sum()])` is a **plan-time** error, not a compile error —
  the price of one value box. Every other engine does this.
- `IsErased` propagation means every composite node carries a comptime branch.
  Unmeasured; the mitigation if it costs is that the erased branch is DCE'd in
  all-AOT trees, which is the common case.
- Nodes generic over operand traits cannot always name the tightest bound
  (`Fold` needs `lane`, so its operand bound is the family trait, not `Value`).
  Where a node must accept an erased operand it binds on `Value` and dispatches.
