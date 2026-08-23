# expr2 — architecture and remaining work

Rewritten 2026-08-23 after the consolidation. The original document was a plan
for the aggregation work; that work is done, and the layer was restructured
around it. This now describes **what expr2 is** and **what is left**.

## The architecture

Two modules, one edge, no cycle.

```
logical.mojo   Shape, Analyzable, Executable, Value, DynValue,
               Relation, DynRelation, and the plan nodes
       │
       ▼
physical.mojo  Datum, Morsel, Operator, DynOperator[Out], Pipeline,
               Evaluable, EvalOperator, and the operators
```

`physical` imports **nothing** from inside the package. The split is by
meaning: `Shape` is a *description* (does this yield one value or one per row,
knowable without running), `Datum` is a *result* (what a running operator
produced).

### Four rules the layer now obeys

1. **A logical node is stateless; a physical one owns state.** `to_operator()`
   is the *only* door between them, for relations and values alike. There is no
   `evaluate` on `Value` or on `DynValue` — a description has no business
   exposing a way to run itself. `evaluate` survives only inside a lane, on
   `Evaluable`, as the fused driver that lane's operator calls.

2. **One executor contract.**

   ```mojo
   trait Operator(Deinitable, Movable):
       comptime Out: Copyable
       def push(mut self, morsel: Morsel) raises -> Optional[Self.Out]
       def drain(mut self) raises -> Optional[Self.Out]
       def done(self) -> Bool
   ```

   Blocking is not a type distinction — it is *when you answer `Some`*. A
   filter answers from `push`; an aggregate accumulates and answers from
   `drain`; a **source** is simply the operator whose `push` is never called
   and whose `drain` answers until it runs dry. `Out` is what lets one trait
   cover both a relational stage (`RecordBatch`) and a value's (`Datum`).

   `drain` is **repeatable** — the driver calls it until `None`. An operator
   that cannot say "spent" hangs the driver; `FoldOperator` violated this and
   it was a latent infinite loop.

3. **An aggregate is a `Value`.** Not a sibling of one. `sum(x)` conforms to
   exactly what `x + 1` conforms to and is boxed by the same `DynValue`; it is
   simply the conformer that answers from `drain`. There is no `AggValue`
   trait and no `DynAggValue` box. Projecting or filtering on an aggregate is a
   plan-time error naming it, which is what DuckDB, DataFusion and Polars do.

4. **A `Pipeline` is an `Operator`.** A chain of stages pushes, drains and
   finishes like a single stage, so it is a *composite*, not a second
   abstraction. That is what will let `Join` hold two whole sub-plans.
   Relations still build it concretely, because `append` needs the concrete
   type and composing through the box would nest one pipeline per stage.

### Naming

Every `Operator` conformer ends in `Operator`: `BatchSourceOperator`,
`FilterOperator`, `ProjectOperator`, `LimitOperator`, `SortOperator`,
`AggregateOperator`, `FoldOperator`, `EvalOperator`.

### The lanes

`comptime/` is the fused lane — a node's type *is* the expression, and
`bind`/`lane[W]`/`validity` are compile-time composition, so a subtree inlines
into one loop. `runtime/` resolves dtypes from a schema. Both reach the
physical layer through the same `to_operator`, and `builders.mojo` holds the
one surface spanning them: `col("a", int64)` fuses, `col("a")` does not. That
overload set **cannot** be split across the lane packages — Mojo resolves
overloads from candidates visible at one name, so splitting gives two functions
that shadow rather than overload.

## What landed

- **Tier 1.1** — the first binary-size gates that build anything from `expr2`
  (`query_expr2_agg_fused`, `query_expr2_streaming`). Running them immediately
  exposed a pre-existing **+450 KB** regression on `query_join`, bisected to
  `6c570eb` and since fixed (backlog S20, −439,232 bytes recovered).
- **Tier 1.2** — the `AggState` fixes were already in HEAD.
- **Tier 1.3** — `Grouping` / `ScalarGrouping` / `HashGrouping`, the placement
  axis as a trait. `kernels.core.Grouping` was renamed `Groups` to free the
  name: the trait is the grouping, `Groups` is what it assigned.
- **Tier 2.5** — `FoldOperator[K, A, G]`. Algebra, input subtree and placement
  all comptime.
- **Tier 2.7** — the push engine, then the consolidation above.
- **Tier 2.8** — the `Aggregate` relation, plus `Limit` and `Sort`. Six of
  `expr/`'s eight relations.
- **The string family** — `StringValue`, the one that cannot vectorise
  (`lane` has no `W`). Fusion survived it, which is the evidence that fusion
  removes *dispatch*, not width.

**Nine types deleted**: `Exhausted`, `Processor`, `FilterProcessor`,
`ProjectProcessor`, `AggregateState`, `DynAggregateState`, `AggValue`,
`DynAggValue`, and `core.mojo` itself.

## Standing debt — the gate

`query_expr2_agg_fused` **+6.339%**, `query_expr2_streaming` **+3.268%**,
against a 0.5% threshold. Attributed, not guessed:

| | |
|---|---|
| ~44 KB | `Morsel` carrying `Groups` through every stage (`marrow::kernels::groupby` 1 → 13 linked symbols). **Inherent** to one executor contract. |
| ~30 KB | The `DynValue`/`DynAggValue` merge. **Cause unknown — see below.** |

**The `Kind` lever was measured and does not apply.** The plan previously said
this half was recoverable by giving `DynValue` a `kind` field so elementwise
values share one `EvalOperator[DynValue]` instead of instantiating one per
boxed type. A bucket diff across the merge (`5247a0e` → `3e0fd16`) falsifies
it: `__text` grew **+30,432** while the symbol count went **down 7**
(`marrow::arrays` −14, `dtypes` +4, `tabular` +2, `scalars` +1). Per-type
instantiation would appear as *many new symbols*; instead the same symbols got
**larger**, which is inlining, not monomorphisation. Collapsing instantiations
that do not exist would recover nothing.

Whoever picks this up should start from that measurement rather than from the
`EvalOperator` theory. The next probe worth running is *which* symbols grew —
`nm -S` sorted by size across the two commits — not another guess.

**Do not clear the gate by re-baselining.** §0 of the backlog already records
one +55% regression that survived ten commits for exactly that reason.

## Next steps, in order

1. ~~**Spend the `Kind` lever.**~~ **Dropped** — measured and falsified, see
   "Standing debt". The debt is real but its cause is not what was assumed.
2. ~~**Two audit leftovers**~~ **Done** — — a direct test for `Pipeline.drain`/`_stage` (new,
   non-trivial, covered only indirectly), and `Pipeline.push`'s docstring,
   which claims a capability the constructor forbids.
3. **`Join`.** The largest gap and the one that tests the design: a pipeline
   breaker with **two inputs**. Unblocked only now — `Pipeline` is an
   `Operator` so a join can hold two, and repeatable `drain` lets it emit
   several batches.
4. **`ParquetScan`** — a `BatchSourceOperator` sibling. With `Join` that is
   8/8 relations and the first point where `expr2` runs a real query.
5. **Thread the dtype *instance* into `AggState`.** This is the real
   prerequisite for temporal, and it is not what this plan previously said.

   Widening `AccType` to `PrimitiveType` was attempted and is blocked on
   something else entirely: `NumericType` is `Defaultable`, while
   `TemporalType` and `DecimalType` are **not** — a timestamp carries a unit
   and timezone, a decimal a precision and scale, so neither can be built from
   its type. `AggState` constructs its accumulator with
   `PrimitiveBuilder[Acc]()`, the no-dtype constructor that exists only for
   numeric types. Widening the bound without threading the dtype compiles and
   unlocks nothing.

   In the fused lane the dtype value is not known until `bind(batch)`, so the
   accumulator must either take it there or defer construction.

6. **Then widen `AccType`**, which activates the domain markers already in
   `kernels/aggregate.mojo` (`OrderedAgg` / `ArithmeticAgg` / `IntegralAgg`).
   A fold constrains itself with `where conforms_to(Self.K, OrderedAgg)` —
   probed and working, rejecting `sum(date)` at compile time. Then delete
   `TemporalMinMax`, which exists only because `MinMax` could not take a
   temporal type.

   **`StringMinMax` is not duplication and stays.** It keeps a per-group
   *index* rather than a scalar accumulator, because strings are
   variable-width — a state-shape difference, not a domain one.
7. **Temporal and list families**, then `param` (`expr/params.mojo`, 564 lines,
   no counterpart) and `if_else` / `coalesce` / `case_when`.
8. **Repoint the Python bindings.** Nothing outside `marrow/expr2/` imports it
   today, so it ships to no one. Until that changes `expr/` is the product and
   `expr2` is a parallel tree.

### Parity, measured

| | `expr/` | `expr2` |
|---|---|---|
| relations | 8 | **8** |
| value families | numeric, bool, string, temporal, list | **numeric, bool, string** |
| `col`/`lit` entry points | 18 | **8** |
| `col`/`lit` entry points | 18 | **7** |
| subsystems | pruning, params, pushdown, bindings | **none** |

Full parity is many sessions, not many steps.

## Open design questions

- **`x - avg(x)`** is still unresolved — see "Do not repeat".
- **`Pipeline.push` is vestigial.** The constructor guarantees stage 0 is a
  source, which ignores `push`, so it exists to satisfy the trait.
- **`physical.mojo` is ~820 lines** across the contract and six operators.
  Splitting the concrete operators into `operators.mojo` would mirror
  `logical.mojo`; deferred because it is cosmetic.

## The fold body — verified, transcribe carefully

Four things, each of which cost a wrong answer or a crash when missing:

1. **A scalar tail is mandatory.** `for i in range(0, n, W)` reads past the view
   on the final chunk and **aborts the process** — buffer *size* rounds up to 64
   bytes and that is not slack. One compilation unit, so it fails every case in
   the file.
2. **`lane[W]` is null-blind.** It returns data bits regardless of validity, so
   a null must become the identity via `mask.select(value, ident)`. The trap: an
   unmasked `sum` is *silently correct* whenever the null slot's payload is 0 —
   only `min`/`max` expose it.
3. **The valid count is a second accumulator**, and **int64, not the
   accumulator type** (`mean` accumulates in float64). `sum` of nothing and
   `sum` of zeros are both 0, and it is `finalize`'s divisor. Reduce it once at
   the end, never a horizontal reduce per chunk.
4. **`AccType` must never appear unerased in a signature.** It is a comptime
   conditional type: it reduces inside a struct but fails to unify at a return
   site. That is why `FoldOperator.Out` is `Datum` and not the accumulator
   type — forced, not chosen. `AggKernel.combine` also will not infer `W` from
   a `Scalar` — spell `combine[acc, 1](…)`.

## Measurements this rests on

| | |
|---|---|
| fused vs materialise, grouped, 1M rows | **1.17-1.68x** (g10 / g1k / g100k) |
| `lane[W]` vs `lane[1]`, grouped | 1.09-1.37x — the scatter stays scalar, the loads do not |
| scatter at one group vs register fold | **14.6x** — why `ScalarGrouping` is its own conformer |
| per-lane Kleene vs `_kleene`'s bitmap algebra | **4-10x worse** — do not fuse boolean validity |
| adding slots to the aggregate box | **+3.2 MB (+24%)**, recorded at `expr/aggregates.mojo:250-253` |

## Do not repeat

- **`Column[T: PrimitiveType]` does not work — probed and reverted
  2026-08-23.** The idea was sound and is what CLAUDE.md advises ("dispatch on
  the widest family the typed leaf accepts"): `PrimitiveType` carries
  `comptime native: DType`, which is all `lane[W]` needs, and `TemporalType` /
  `DecimalType` / `IntervalType` all conform to it — so **one** leaf should
  cover every fixed-width type and `expr/`'s separate `TemporalColumn` should be
  unnecessary duplication. It is not achievable as a widening: **19 errors**,
  because the aggregate chain is bound on `NumericType` the whole way down
  (`AggState[K, V: NumericType]`, `AggKernel.AccType[V: NumericType]`), so
  widening `NumericValue.Type` breaks every fold.

  The clean form needs `PrimitiveValue` as the family and `NumericValue`
  refining it — but **Mojo has no conditional conformance**, so a single
  `Column` struct cannot be a `PrimitiveValue` when `T` is a date and a
  `NumericValue` when `T` is an int. Temporal support therefore costs either a
  second leaf type (`expr/`'s answer) or widening `AccType` to `PrimitiveType`
  in `kernels/aggregate.mojo` first. **Do the kernels widening first** if this
  is attempted again; the expression layer is not where it is blocked.


- **`precompile` does not elaborate function bodies.** A clean `precompile` is
  not evidence a test will build, and `comptime assert` in an uninstantiated
  body reports *nothing*. Only `pytest` proves it.
- **Deletion is not part of this work.** `marrow/tabular.mojo:22-23` imports
  `expr/aggregates.mojo` to back `RecordBatch.group_by()`, a shipped
  PyArrow-mirror API; `python/bindings/compute.mojo:75-88` uses the kernel
  traits directly; `benchmarks/binary_size/query_streaming_agg_fused.mojo:19`
  imports `NumericAgg`. `Aggregation` has **five** conformers, and deleting it
  also deletes `AggFunction` and `AggKernel.Grouped`. This lands additively.
- **`x - avg(x)` is unresolved.** `expr/tests/test_values.mojo:166,175,182,192`
  cover it today. Any design forbidding aggregates in value position deletes
  four tested behaviours; the fix is aggregate extraction as a planning rule,
  not a value-level mechanism. Decide before `expr/` is deleted.
