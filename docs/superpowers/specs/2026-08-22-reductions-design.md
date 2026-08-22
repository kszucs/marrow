# Reductions and aggregation in `expr2`

Status: **revision 2**, 2026-08-22. Rewritten after four adversarial reviews,
three of which refuted claims in revision 1. Everything below marked *measured*
or *verified* was executed; everything else is explicitly labelled unproven.

## The one-line frame

A reduction is a **value folded to a scalar**, and marrow can do something no
other engine can: fold a *fused expression* without materialising it.

That claim is now proven (§Evidence), and it is the only part of revision 1 that
survived unchanged.

## Requirements

| # | requirement | status |
|---|---|---|
| **R1** | `sum(a*2+b)` folds with **no intermediate array** | **proven** — 8/8 probe cases |
| **R2** | grouped and ungrouped share node types, semantics, and the empty/all-null rule — but **not the inner loop** | corrected; see below |
| **R3** | streaming: `O(groups)` memory, never `O(rows)` | design |
| **R4** | one plan holds N reductions of differing types | design |
| **R5** | plan stays copyable and rewritable after processors exist | design |
| **R6** | a reduction is not `Evaluable`, so it cannot be used where a per-batch value is expected | mechanism **verified**; **scope unresolved** |
| **R7** | closed erasure; binary size within +0.5% | **currently unfalsifiable** — no gate builds `expr2` |
| **R8** | the runtime lane can be added later without redesign | design |
| **R9** | intermediate state is a **column**, so partial/final aggregation is one mechanism | corrected |
| **R10** | empty input and all-null input both yield NULL | **refuted in code** — live bug |
| **R11** | the aggregate *catalogue* is expressible: numeric, bool, string, temporal, distinct | **redesigned** — was the weakest claim |
| **R12** | kernels stay typed-first | withdrawn as stated; see §Scope |

### R2 — shared semantics, separate loops

Revision 1 said "ungrouped is `num_groups == 1`". **Measured: that costs 14.6x**
(2538.6 us vs 173.9 us on 1M rows). A million serially dependent
read-modify-writes through a builder slot cannot compete with a register
accumulator. R2 means shared *node types and semantics*; the ungrouped fold is a
distinct inner loop.

### R6 — the mechanism holds; the scope is an open design question

Conformance in Mojo is **nominal, not structural** (verified): a struct that
declares only `Reduction` but structurally satisfies `Value` is still rejected
by `DynValue.__init__[V: Value]`. So the compile-time guarantee is real.

**But revision 1 used a streaming argument to delete a whole-batch capability,
and did not notice.** These pass today in `expr/`, where `Reduction[K, A]` *is* a
`NumericValue` with `OutShape = 0`:

| test | expression |
|---|---|
| `test_values.mojo:166` | `a + sum(a)` broadcasts |
| `:175` | `sum(a) + max(a)` stays scalar |
| `:182` | `(a + b) * sum(a)` |
| `:192` | `x - avg(x)` — mean-centering |

Mean-centering and share-of-total are why users write
`col("a") / col("a").sum()`. **This spec does not yet have an answer**, and
will not claim one. The candidates are a scalar-subquery rewrite (`Aggregate`
then `Project` against the resulting literal — what SQL does) or a windowed
`sum() OVER ()`. Whichever is chosen is a *plan-level* mechanism, not a
value-level one, and it must be decided before `expr/` is deleted.

Separately, the diagnostic is poor. `rel.project([col("a",int64).sum()])`
reports *"cannot be converted from list literal to `List[DynValue]`"* without
naming the offending element.

### R11 — the catalogue does not fit one parametric node

Revision 1 proposed `Reduction[K: AggKernel, A: NumericValue]` and filed the
gaps as "open risks". **Both bounds are wrong, and four of six affected
families went unmentioned.** Verified:

| family | why the revision-1 bound rejects it |
|---|---|
| `any`/`all` | `BoolReduceKernel` is **not** an `AggKernel` (no `identity`/`combine`/`finalize`/`AccType`); `BoolType` is **not** a `NumericType`. Fails both bounds. No grouped bool fold exists in kernels at all. |
| `min`/`max` over strings | `StringMinMax` — its per-group state is a **row index into the input batch** |
| `min`/`max` over temporal | `TemporalMinMax` |
| `count` over non-numeric | `CountAgg` |
| `count_distinct` | no incremental kernel form: `count_distinct` builds its `SwissHashTable` *inside the call* and returns a finished answer |
| `count(*)` | has no input value at all |

**Resolution: reductions are family-split, exactly as values already are.** The
value layer has `NumericValue`/`BoolValue`/`StringValue` because the families do
not share a lane shape; reductions inherit that for the same reason — a bool
fold folds bit-packed masks, a string min/max tracks a row index, a numeric fold
accumulates in a register.

```
core.mojo          trait Reduction          the erasure boundary; one per lane-family below
comptime/          NumericReduction[K: AggKernel,       A: NumericValue]   ← increment 1
                   BoolReduction   [K: BoolReduceKernel, A: BoolValue]     ← later
                   StringReduction, TemporalReduction                       ← later
```

All box into one `DynReduction`, so a plan holds them uniformly.

**`count(*)` does not desugar to `count(lit(1))`.** Revision 1 claimed that was
safe *because* `Literal.columns()` is empty. That reasoning is inverted:
empty `columns()` means projection pushdown prunes every column, the scan yields
column-less batches, and `RecordBatch.num_rows()` returns **0** when there are no
columns (`tabular.mojo:80-84`). The property cited as the safety argument is the
bug mechanism. `count(*)` needs its own node that reads `batch.num_rows()`.

### R10 — live bug, in the code today

Executed: `test_a_reduction_over_no_rows_is_null` fails with
`Buffer index 0 out of bounds for length 0`. Two independent causes:

1. **Kernel.** `AggState.finish` loops `for g in range(num_groups)` reading
   `cnt.unsafe_get(g)`, but only `update` ever grows the builders. Zero updates
   → length 0 → out of bounds. Under `-D ASSERT=all` this aborts; **in a release
   build it is a silent out-of-bounds read.**
2. **Expression.** `NumericReductionState.update` early-returns on `n == 0`.

Fix: `AggState.finish` grows to `num_groups` before the loop. This is a kernel
bug that predates the design and is only reachable because `Accumulator` makes
"zero updates" possible — `AggKernel.reduce` always calls `update` once.

## Evidence

Measured 2026-08-22. `sum(a*2+b)`, 1M rows, float64, medians.

**R1 is real.** A `FusedFold[K, A]` probe passed 8/8 including nulls, a ragged
tail (n=1003), two morsels, empty and all-null. `update` performs **no
allocation**: `bind` once, then `lane[W]` per chunk into a SIMD local.

**Fusion also wins grouped**, which revision 1 listed as an open risk:

| groups | materialise-then-scatter | fused, W=4 | |
|---|---|---|---|
| 10 | 684.0 us | 444.0 us | **1.54x** |
| 1 000 | 495.4 us | 294.3 us | **1.68x** |
| 100 000 | 1514.4 us | 1293.3 us | **1.17x** |

The accumulator scatter stays scalar and cannot vectorise — duplicate group ids
in one vector lose updates without conflict detection, and `BufferView` has
`gather` but no scatter. **`lane[W]` still pays 1.09-1.37x over `lane[1]`**: the
vector loads and arithmetic earn their keep before the unpack.

Ungrouped, register accumulator: **173.9 us** — the 14.6x behind R2.

## The fold loop, correctly

Revision 1's snippet was wrong three ways. All three are load-bearing.

```mojo
comptime W = simd_width_of[Scalar[Self.acc]]()
var bound = self._input.bind(batch)
var v     = self._input.validity(bound)          # 1. nulls
var ident = Self.K.identity[Self.acc]()
var vec   = SIMD[Self.acc, W](ident)
var cnt   = SIMD[Self.acc, W](0)                 # 3. the valid count
var simd_end = (n // W) * W                      # 2. the tail
var i = 0
if v:
    var vb = v.value()                           # named local: view() borrows it
    var bits = vb.view()
    while i < simd_end:
        var mask = bits.load[W](i)
        vec = Self.K.combine[Self.acc, W](
            vec, mask.select(self._input.lane[W](bound, i).cast[Self.acc](),
                             SIMD[Self.acc, W](ident)))
        cnt += mask.select(SIMD[Self.acc, W](1), SIMD[Self.acc, W](0))
        i += W
    while i < n:                                 # scalar tail
        if bits.load[1](i)[0]: ...
else:
    ...same without the mask
```

1. **`for i in range(0, n, W)` aborts the process.** Buffer *size* rounds to 64
   bytes and CLAUDE.md says that is not slack:
   `Assert Error: BufferView range [100, 102) out of bounds for length 101`.
   One compilation unit, so it took the whole runner down. A scalar tail is
   mandatory and costs nothing.
2. **`lane[W]` is null-blind.** It returns data bits regardless of validity, so
   the fold needs a per-lane `mask.select(value, identity)`.
   `views.reduce`'s null-aware overload already uses this trick.
   **The trap:** an unmasked `sum` over a test batch answers *correctly* when
   the null slot's payload happens to be 0. Only `min`/`max` expose it.
3. **The valid count is a second accumulator, which revision 1 never
   mentioned.** R10 cannot be decided from the value: `sum` of nothing and `sum`
   of zeros are both 0. And `AggKernel.finalize(acc, count)` takes it as an
   argument — it is `mean`'s divisor. Count it as a SIMD accumulator reduced
   once at the end, **not** a `reduce_add` per chunk, which would put a
   horizontal reduce on the critical path.

Two further constraints, both verified:

- **`AccType` must never appear unerased in a signature.**
  `Widening.AccType[V] = Int64Type if V.native.is_integral() else Float64Type`
  is a comptime *conditional* type. It reduces fine inside the struct; it fails
  to unify at a return site — CLAUDE.md's documented limit, hit live. Controlled
  comparison: `MinKernel.AccType[V] = V` is unconditional and compiled; `Sum`
  did not. So `finish() -> DynArray` is correct — by necessity, not by taste.
  This will recur on any *typed* state column.
- **`AggKernel.combine` will not infer `W` from a `Scalar`.** Every
  scalar-width call needs explicit `combine[Self.acc, 1](...)`.

## Intermediate state as a column

ClickHouse's `-State`/`-Merge` makes the accumulator's intermediate state a
first-class typed value. The idea is right. Three of revision 1's four factual
claims about it were wrong.

```mojo
trait Accumulator(Deinitable, Movable):
    def update(mut self, batch, groups: Int32Array, num_groups: Int) raises
    def state(mut self, num_groups: Int) raises -> DynArray    # intermediate
    def absorb(mut self, state: DynArray, groups: Int32Array, num_groups: Int) raises
    def finish(mut self, num_groups: Int) raises -> DynArray   # finalized
```

**Corrections.**

- ~~"`sum` is its own state"~~ — `AggState` holds `acc` **and** `cnt` for every
  kernel, and the count *is* R10. A one-column state works only if the seen-flag
  rides as the state column's **validity bitmap** — the natural Arrow encoding,
  free to produce, but it must be said or R10 breaks.
- ~~"`mean` is `(sum, count)`, real work, may raise `unimplemented`"~~ — wrong
  twice. That pair **is** marrow's existing state layout for every numeric
  kernel including `mean`: `AggState.into_partials` produces it, `merge`
  consumes it, `NumericAgg.is_mergeable` is `True` for all `K`, and
  `test_groupby_thread_local_mean_nulls_match_serial` covers it. Deferring
  `mean` would delete a tested capability. Packaging as `struct<sum, count>` is
  an O(1) child-list assembly — `groupby.mojo:561` already does it.
- ~~"the remap disappears"~~ — false. The remap already *is* group ids from the
  same grouper (`groupby.mojo:591`); `absorb` takes the identical array in the
  identical position. A state column is **not self-describing** — it carries no
  keys, so the caller still needs the partial's keys to derive `groups`.

**What `state`/`absorb` actually buys**, which is stronger than the rename
revision 1 claimed:

1. **It unconflates merge from finalize.** `NumericAgg.merge` ends with
   `state.finish(num_groups)` — merging *finalizes*, so a merged result cannot
   be merged again. Hierarchical, incremental and three-way merge are impossible
   today.
2. Three parallel lists (`remap`, `accs`, `cnts`) collapse to one column.
3. **At the plan level it is ClickHouse's model**: `Aggregate(mode=Partial)`
   emits `(key columns…, state columns…)`; `Aggregate(mode=Final)` groups those
   keys like any ordinary input, and there the remap stops being a special
   argument because it becomes Final's normal grouping.

**The genuine gaps** are the three `is_mergeable = False` types — `StringMinMax`
(state is a row index into the input batch, portable across no boundary),
`TemporalMinMax`, `DistinctAgg`.

**Contract to specify:** `PrimitiveBuilder.finish()` calls `reset()`, so a naive
`state(mut self)` **silently empties the accumulator**. `state` must be declared
either terminal (like `finish`) or flush-and-continue, and the reset documented
as the contract.

## Scope: what is added, and what is *not* deleted

Revision 1 proposed deleting `Aggregation`, `NumericAgg`, `ColumnAggregator`,
`OneAggregation` and `GroupBy`'s aggregate-driving, "gated on Phase 5 because of
`FoldedAggregates`". **That was wrong in five independent ways** and is
withdrawn.

1. `Aggregation` has **five** conformers, not one: `NumericAgg`,
   `TemporalMinMax`, `StringMinMax`, `CountAgg`, `DistinctAgg`. Deleting it also
   deletes `AggFunction` and `AggKernel.Grouped` — a member of the trait this
   spec *keeps*.
2. `marrow/tabular.mojo:22-23` — **core imports `expr/aggregates.mojo`**,
   backing `RecordBatch.group_by()`, a shipped PyArrow-mirror API with tests and
   benches. Outside Phase 5's stated scope.
3. `python/bindings/compute.mojo:75-88` uses the kernel traits directly,
   independently of `expr/`.
4. `benchmarks/binary_size/query_streaming_agg_fused.mojo:19` — **the gate
   imports `NumericAgg`**. R7's own instrument would be rewritten by the change
   it polices.
5. Strategy selection is **not being relocated**. Measured with `nm`: the plan
   lane does not use `GroupBy` at all — `AggregateProcessor` holds a bare
   `HashGrouper` and does a serial scalar scatter. `GroupBy`'s thread-local and
   radix strategies have exactly one caller, `tabular.mojo:310`. Deleting its
   aggregate-driving removes parallel group-by from the only path that has it,
   and radix would need ~250 lines re-implemented.

**Therefore: this design is purely additive.** The fused path lands beside the
existing erased one. Nothing in `kernels/` is deleted in this work. Whether the
erased path is later retired is a separate decision that must include
`tabular.mojo` and `python/bindings/compute.mojo`, neither of which Phase 5
covers. R12 is withdrawn as stated — `Aggregation`/`AggFunction` are *typed*
trait layers, not erased ones; the erased boxes are `AggFunc`/`AggFold`, and
they live in `expr/`.

## R1 versus R7 — the unresolved tension

`NumericReduction[K, A]` monomorphizes on the **expression subtree**, so
`sum(a*2+b)` and `sum(a+b*2)` are distinct types with distinct accumulators and
distinct trampoline sets. `NumericAgg[K, V]` instantiates over a bounded 6x12
grid; this is unbounded in query shape. `baseline.json` records the precedent:
*"1,476 bytes of added source cost 371,584 bytes of `__text` through that cross
product, a 250x multiplier that nothing would have reported."*

Two mitigations, neither measured:

- **Keep the state keyed on dtype, not on the subtree.** The fold *loop*
  monomorphizes on `A`; the `AggState[K, V]` it accumulates into does not, so it
  is shared across every expression shape reducing to the same type.
- **Keep `DynAccumulator` at fewer slots.** `expr/aggregates.mojo:250-253`
  records that folding whole-table reduce and partial/merge into the box
  measured **+3.2 MB (+24%)**; today's plan path deliberately pays one slot. A
  four-slot `Accumulator` repeats that experiment. `state`/`absorb` may need to
  arrive with two-phase aggregation rather than before it.

**R7 is unfalsifiable until an `expr2` gate exists.** No binary-size gate builds
anything from `expr2`; since this design lands beside `expr/`, `pixi run
binary_size` will report ~0.00% regardless — the same blind spot that let a
+7.09% regression measure zero. **`query_expr2_agg_fused` must be added to
`baseline.json` before any of this code lands.**

## Increment 1

1. Add the `expr2` binary-size gate. Without it nothing below is measurable.
2. Fix `AggState.finish` to grow to `num_groups` (R10; a real out-of-bounds read).
3. `core.mojo`: `Reduction`, `DynReduction`. `physical.mojo`: `Accumulator`,
   `DynAccumulator` — `update`/`finish` only; `state`/`absorb` deferred with
   two-phase aggregation. **No new state type**: `AggState` gains one public
   `accumulate[W]`, and the ungrouped fold is two registers.
4. `comptime/reductions.mojo`: `NumericReduction[K: AggKernel, A: NumericValue]`
   with the corrected fold loop — masked, counted, scalar tail, register
   accumulator ungrouped and scatter grouped.
5. The fluent surface CLAUDE.md mandates: `col("amount", int64).sum().alias("total")`.
   **Verified to compile**, and better-behaved than `expr/`'s — because the
   result conforms to `Reduction` and not `NumericValue`, `x.sum().sum()` is a
   compile error.
6. `Aggregate` / `AggregateProcessor`, ungrouped and grouped.

Deferred with the reason recorded: the other reduction families (R11), the
runtime lane (R8), `state`/`absorb` (R7 risk), and R6's whole-batch scope —
which must be settled before `expr/` can be deleted, not before this lands.
