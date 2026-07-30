# Aggregation & group-by — follow-ups

Companion to `docs/tasks-expr-kernels-layering.md` (the full expr/kernels
layering audit, L1–L9). That document owns the *layering* tasks; this one owns
what is specific to aggregation and group-by.

Sections 0–4, 6 and 6b were completed and have been removed; git history has the
detail. The one invariant worth carrying forward is below.

**From the catalog decision (§6, closed):** the `AggFunction` catalog (`Sum`,
`Product`, `Mean`, `Min`, `Max`, `Count`, `CountDistinct`,
`ApproxCountDistinct`) lives in `expr/aggregates.mojo`, because **nothing in
`marrow/kernels` maps a name to behaviour.** `Aggregation.name` and
`AggKernel.name` stay — every kernel carries a name as identity. The rule is "no
name→behaviour mapping in kernels", not "no strings".

---

## 5. Strategy thresholds are unre-tuned

`GroupBy(keys, ctx, strategy)` can now force a strategy — the stated
prerequisite in `docs/aggregate-kernel-inversion.md` §7 for evaluating a
`GroupStats` / `suitable` / `rank` policy. `_PARALLEL_MIN_ROWS`,
`_PARALLEL_ALWAYS_ROWS` and the cardinality sample have not been re-measured
since the driver changed.

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
