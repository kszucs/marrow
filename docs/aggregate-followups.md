# Aggregation & group-by — follow-ups

Companion to `docs/expr-kernels-layering-tasks.md` (the full expr/kernels
layering audit, L1–L9). That document owns the *layering* tasks; this one owns
what is specific to aggregation and group-by, records what the `Aggregation`
inversion already changed, and flags where the two disagree.

**State:** 1865 passed / 317 skipped / 0 failed · binary size fused **7.6×**,
runtime-named **7.9×** (baseline 7.6× / 7.8×) · group-by competition **12/15**
wins, `sum[1m_g100k]` **1.74 ms** and `multi[1m_g100k]` **2.94 ms** — both now
under the pre-refactor baseline (1.87 / 3.02). §1, §3 and §4 are done — see the notes under
each; §2, §5, §6 and §7 remain.

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

## 1. Ibis-style aggregate expressions — **done**

```mojo
rel.aggregate(keys=[col("region")],
              aggs=[col("amount").sum().alias("total"),
                    col("amount").max().alias("biggest")])
```

`col("x").sum()` on a `DynValue` gives a `DynAgg` (the function's name plus its
input); on a fused node it is the existing `Reduction`, which converts to the
same `AggExpr` with its `Aggregation` already named — so the AOT spelling is the
identical call one lane down and resolves nothing at run time. Both lanes mix in
one list. `.count_distinct()` / `.approx_count_distinct()` have no fused
reduction and go straight to an `AggExpr`.

Left over:

- The **node still holds two parallel lists** (`inputs: List[AnyValue]` +
  `aggs: Aggregates`); `aggregate()` splits the `AggExpr`s at plan build. Fine
  for now — the split is where the input expression and the resolved aggregate
  genuinely part ways — but it is the obvious next simplification if the node
  ever needs to carry the pair around together.
- **A fused plan built through `rel.aggregate(...)` still links the catalog.**
  `AggExpr.resolve` has a dynamic arm, so the name ladder is reachable from any
  plan built with the fluent API. Only a plan that constructs the `Aggregate`
  node directly (as `benchmarks/binary_size/query_streaming_agg_fused.mojo`
  does) gets the 7.6× binary. Closing that needs the aggregate list to be
  comptime-known, which is `FusedAggregation` (step 4) territory.
- `python/bindings/tabular.mojo` is unaffected — it drives `Aggregates`
  directly, not the plan API.

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

## 3. `count` disagrees with itself on all-null groups — **done**

`AggKernel.empty_is_null` states it on the kernel (`count` is the one aggregate
that answers 0 rather than NULL when a group has no valid rows), and
`AggState.finish` finalizes the untouched identity instead of appending NULL
when it is false. `count` is now 0 over every dtype, and
`test_nulls_are_excluded_and_empty_groups_are_null` asserts the *validity* too,
which it previously did not.

## 4. Benchmark coverage for the radix path — **done**

`bench_groupby.mojo` gains `sum[1m_g1k]`, `sum[1m_g100k]` and `mean[1m_g100k]`;
`_bench_group_by` takes the group count. Cardinality is what picks the strategy,
so g100k is the radix path — previously reachable only through the Python
competition harness. Not yet used as a gate: no baseline numbers recorded.

## 5. Strategy thresholds are unre-tuned

`GroupBy(keys, ctx, strategy)` can now force a strategy — the stated
prerequisite in `docs/aggregate-kernel-inversion.md` §7 for evaluating a
`GroupStats` / `suitable` / `rank` policy. `_PARALLEL_MIN_ROWS`,
`_PARALLEL_ALWAYS_ROWS` and the cardinality sample have not been re-measured
since the driver changed.

## 6. Where the `AggFunction` catalog lives — **decided: `expr`** (audit L1)

The catalog (`Sum`, `Product`, `Mean`, `Min`, `Max`, `Count`, `CountDistinct`,
`ApproxCountDistinct`) and `resolve_agg` are in `expr/aggregates.mojo`. The
criterion that settled it: **nothing in `marrow/kernels` maps a name to
behaviour.** `Aggregation.name` and `AggKernel.name` stay — every kernel in the
package carries a name as identity, and removing them would only push a name
argument through `AggFunc.of[A]` / `Aggregates.append[A]` / `AggExpr.of[A]` for
no gain. The invariant is "no name→behaviour mapping", not "no strings".

Consequences that landed with it: `expr.aggregates → expr.dynamic` is gone (the
ladder moved out of `dynamic.mojo`), and `GroupBy` no longer names output
columns — see below.

### Old text (for the record)

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

## 6b. Done alongside — audit L4, K1, K2

- **L4:** `GroupBy` returns `GroupedColumns` (unique key columns + one per
  aggregate) instead of a named `RecordBatch`. `marrow/kernels` imports neither
  `..tabular` nor `..schema` anywhere now. Naming moved to `Aggregates._named`.
- **K1 — one `count`:** `AggKernel.Grouped[V]: Aggregation` lets a kernel name
  its own grouped implementation, so `CountKernel.Grouped[V] = CountAgg` and the
  validity scan serves every dtype. `CountValid.resolve` no longer branches on
  numeric-vs-not, and `CountAgg` is now mergeable (counts add), which it was
  not. The fused lane resolves through the same member.
- **K2 — the scatter loop:** `AggKernel.needs_count` (true only for `mean` and
  `count`) picks `AggState`'s second column — `Int64Type` when the count is
  read, `UInt8Type` when only "was this group touched" matters. For
  `sum`/`min`/`max`/`product` the per-row work drops from a read-add-write on 8
  bytes to a plain 1-byte store, and the column is 100 KB rather than 800 KB at
  100k groups. Partial exchange stays int64; the widening is O(groups), once.
  Measured on a *loaded* machine: `sum[1m_g100k]` 1.89 → **1.74 ms**,
  `multi[1m_g100k]` 3.06 → **2.94 ms**, `sum[1m_g1k]` 812 → 775 µs.

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
