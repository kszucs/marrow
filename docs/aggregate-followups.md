# Aggregation & group-by — follow-ups

Companion to `docs/expr-kernels-layering-tasks.md` (the full expr/kernels
layering audit, L1–L9). That document owns the *layering* tasks; this one owns
what is specific to aggregation and group-by, records what the `Aggregation`
inversion already changed, and flags where the two disagree.

**State:** 1866 passed / 314 skipped / 0 failed · binary size fused **7.6×**,
runtime-named **7.9×** (baseline 7.6× / 7.8×) · group-by competition **11/15**
wins, two runs on an idle machine.

---

## 0. Already done — do not re-open these

- **The perf regression is found and fixed**, so
  `docs/expr-kernels-layering-tasks.md` L5's guess ("most likely source of the
  unresolved regression … `aggregate[A]` now falls through to the shared
  `_by_partition` driver") is **not** the cause. The two real causes were, in
  the unified driver: an identity row array built once per serial query, and
  stitching a single partition under a *parallel* `ExecutionContext` — thread
  dispatch for a `num_groups`-row `take`. `sum[10k_g10]` 375 µs → **47.8 µs**,
  `sum[100k_g10]` 708 µs → **414 µs**, wins 9/15 → **11/15**. Details and the
  full table: `docs/aggregate-kernel-inversion.md` §8.
- `GroupPartitioner` / `WholeRows` / `ByKeyHash` are gone — one driver,
  `_by_partition[col_agg](..., partition: Bool)`. Inverting control flow through
  a `work` callback bought nothing for two cases.
- `slice_struct` is gone: `RapidHash.apply(StructArray)` now honours the
  struct's own offset/length, so `keys.slice(start, length)` is the API.
- `AggFold` no longer carries `whole` (a box resolved and called once buys
  nothing), and `_thread_local`'s three parallel `List[Optional[...]]` arrays
  indexed `[t * na + j]` are one `ThreadPartials` value.
- A keyless `aggregate(...)` — `SELECT sum(x)` with no `GROUP BY` — now
  executes; it used to raise inside the grouper.
- `marrow/expr/tests/test_aggregates.mojo` (18 tests) covers the aggregate
  surface through plan-build + `execute` only, so the machinery underneath stays
  refactorable, plus three AOT/fused cases.

## 1. Ibis-style aggregate expressions — **owner-requested, not started**

Today an aggregate is three positionally-correlated lists:

```mojo
rel.aggregate(keys=[col("region")],
              values=[col("amount"), col("amount")],
              funcs=["sum", "max"],
              names=["total", "biggest"])
```

Target (ibis: `t.group_by("region").agg(total=t.amount.sum())`):

```mojo
rel.aggregate(keys=[col("region")],
              aggs=[col("amount").sum().alias("total"),
                    col("amount").max().alias("biggest")])
```

- `DynValue.sum()/min()/max()/count()/mean()/product()/count_distinct()/
  approx_count_distinct()` + `.alias()` → an aggregate expression (input
  expression + function + optional name). The dynamic form.
- **The AOT form is the preference**, so it needs a typed spelling too —
  `Sum(col("amount", int64))` — not `Aggregates.append[NumericAgg[...]]`. This
  is where the owner's "aggregates are still values" note lands; `values.mojo`
  already has `Reduction[K, A]` for the whole-array case and the grouped case is
  its sibling.
- `Aggregate` / `AggregateProcessor` then hold **one** list instead of `inputs`
  + `aggs` in parallel; `python/bindings/tabular.mojo` builds the dynamic form
  from the names Python passes.
- Interacts with layering-audit **L3** (what `AggFunc` stores) and **L5** (who
  owns the strategy) — settle the expression shape first, then those.

## 2. Consolidate the two hash groupers

Hash is the only grouping mechanism (no sort-based grouping), but
"resolve rows to dense group ids and remember each group's first row" is
implemented twice, differing only in how keys are materialized:

| | `HashGrouper.consume_keys` | `GroupBy._by_partition` |
|---|---|---|
| across batches | incremental | one shot per partition |
| unique keys | per-column builders, as it goes | first-occurrence row numbers, one `take` at the end |

One grouper with two key-materialization modes. `AggregateProcessor` needs the
incremental mode, `GroupBy` the deferred one. Do this with, or right after,
audit **L4** (`GroupBy` stops speaking `RecordBatch`) — both touch the same
return shape.

## 3. `count` disagrees with itself on all-null groups

`count` over a **numeric** column resolves to `NumericAgg[CountKernel, V]`,
whose `AggState.finish` emits NULL for a group with no valid rows; over any
**other** column it resolves to `CountAgg`, which emits 0. SQL says 0. This
predates the inversion (the old code had the same split) and is currently pinned
by `test_nulls_are_excluded_and_empty_groups_are_null`, so fixing it is a
deliberate behaviour change: either `AggState` emits the kernel's identity
rather than NULL when the kernel says so, or `count` gets one dtype-independent
implementation.

## 4. Benchmark coverage for the radix path

`marrow/kernels/tests/bench_groupby.mojo` only covers `g10` — serial and
thread-local. The high-cardinality radix path, which is what the fusion work
targets, is reachable only through the Python competition harness (Python
overhead, needs an idle machine). Add `g1k` / `g100k` cases at the Mojo level so
a regression there shows up without the harness. Prerequisite for measuring
audit **L5** honestly.

## 5. Strategy thresholds are unre-tuned

`GroupBy(keys, ctx, strategy)` can now force a strategy — the stated
prerequisite in `docs/aggregate-kernel-inversion.md` §7 for evaluating a
`GroupStats` / `suitable` / `rank` policy. `_PARALLEL_MIN_ROWS`,
`_PARALLEL_ALWAYS_ROWS` and the cardinality sample have not been re-measured
since the driver changed.

## 6. Conflict to resolve: where does the `AggFunction` catalog live?

The catalog (`Sum`, `Product`, `Mean`, `Min`, `Max`, `Count`, `CountDistinct`,
`ApproxCountDistinct`) is currently in **`kernels/aggregate.mojo`**, moved there
on the owner's instruction ("are these actually kernels? if yes then move them
under kernels") and because it removed a real cycle:
`expr.aggregates ↔ expr.dynamic`.

`docs/expr-kernels-layering-tasks.md` **L1** asks for the reverse — catalog *and*
`resolve_agg` in `expr/aggregates.mojo` — which also removes that cycle, by
deleting the `expr.aggregates → expr.dynamic` edge instead.

Both are acyclic; they disagree on whether "which implementation for which
dtype" is kernel selection (like `cast.mojo`'s dispatch) or frontend vocabulary.
Pick one and make the three headers say it. Note the remaining edge under the
current arrangement (`expr.aggregates → expr.dynamic`, for `resolve_agg`) is
structural only — the fused binary links **zero** `marrow::expr::dynamic`
symbols, so DCE already removes the interpreter.

## 7. Smaller items

- `python/bindings/tabular.mojo`'s `group_by` / `aggregate` predate the plan
  API. Now that keyless `aggregate(...)` executes, both could route through a
  plan — one path to maintain — but only once `AggregateProcessor` can use the
  eager thread-local strategy, or it is a performance regression.
- `marrow.compute.sum` over narrow integers now widens to int64 (matching
  PyArrow and the grouped path). Worth a parity test against `pyarrow.compute`
  for the whole-column aggregates.
- The aggregate tests in `marrow/expr/tests/test_streaming.mojo` now overlap
  `test_aggregates.mojo` (min/max over strings and dates, count over strings,
  out-dtypes). Keep the plan-shaped ones there (computed keys, HAVING, morsel
  boundaries), drop the duplicated semantics.
- Still open from the inversion doc: **step 4** `FusedAggregation` (validate at
  `g100k`, not `g10`), **step 2b** `AggState` widening, **Q1.1** comptime key
  spec (`kernels::cast` is still ~20 % of the fused binary via `hashing.mojo`).
