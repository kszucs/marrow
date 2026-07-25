# Q2.5 step 3b — invert the aggregate layer onto the kernels

**Status:** specified, not started. Blocks `FusedAggregation` (step 4).
**Base:** `complete` @ `15af282`. Suite green (1848 passed / 314 skipped / 0 failed).

This file is the working spec for the next round. It exists because step 3a
(`2ecc58f`) delivered the plumbing but not the inversion, and shipped a measured
performance regression — both must be resolved before anything is built on top.

---

## 1. Why this step exists — two defects, one cause

### Defect A — the tag came back as a string

`marrow/expr/aggregates.mojo:104`:

```mojo
comptime if K.name == "min" or K.name == "max":
```

Step 3a deleted `comptime AGG_MIN: UInt8 = 1` and replaced it with `K.name == "min"`.
Same dispatch, same coupling, now **stringly-typed**: no exhaustiveness check, typo-able,
and a new kernel silently falls through to the `else` branch. The switch moved and the tag
was renamed; neither was removed.

### Defect B — 18 module-level functions in one 560-line file

```
agg_out_dtype[K]     _acc_dtype[K]        agg_grouped[K]       _fold_grouped[K]
_fold_grouped_typed  agg_whole[K]         _partials[K]         _partials_typed
_merge_partials[K]   _merge_partials_typed  dispatch_agg       _typed_out_dtype[K]
aggregate_grouped    _relabel_temporal    _thread_local_multi  aggregate_whole
```

Including three `_x` / `_x_typed` pairs — the same shape as the `reduce_widened` /
`reduce_widened_typed` workaround already flagged in the Q2.5 card.

### The single cause

**The kernel does not carry its own behaviour.** So the behaviour lives in free functions,
and those functions must re-identify the kernel by name to know what to do. Defect A is the
*consequence* of Defect B, not a separate problem — which is why one change fixes both.

---

## 2. Target design

Push behaviour onto `AggKernel` as required members. The trait becomes the single home of
what an aggregate *is*; the free functions dissolve into it.

```mojo
trait AggKernel(Kernel):
    comptime name: String                    # already inherited from Kernel
    comptime is_distinct: Bool = False       # replaces agg_is_distinct(tag)

    @staticmethod
    def out_dtype(value_dtype: AnyDataType) raises -> AnyDataType
```

`out_dtype` is a **property of the aggregate**, with exactly four answers:

| kernel | rule |
|---|---|
| `Widening[SumOp]`, `Widening[ProductOp]` | widen to `AccType` (so narrow integers don't overflow) |
| `MinMax[MinOp]`, `MinMax[MaxOp]` | **preserve** the input dtype — numeric, string, or temporal (unit/tz included) |
| `CountKernel`, the distinct counts | `int64` |
| `MeanKernel` | `float64` |

Confirmed with the owner 2026-07-25: *min/max preserve, sum widens.* Making `min(int32)`
return int64 would be a visible semantic break and is **not** wanted.

Requiring it on the trait means a new kernel **cannot** forget the rule — as opposed to a
name-string ladder in one central function that must know every kernel that will ever exist.

### Where each free function goes

| free function | becomes |
|---|---|
| `agg_out_dtype[K]`, `_acc_dtype[K]`, `_typed_out_dtype[K]` | `K.out_dtype(value_dtype)` |
| `agg_grouped[K]`, `agg_whole[K]` | `K`'s own execution methods |
| `_partials[K]`, `_merge_partials[K]` | private members of `AggFold` |
| `dispatch_agg` | the **one** runtime name → comptime kernel boundary; keep, but it should be the only place a name is ever compared |

The `_x_typed` inner halves are genuinely forced by Mojo's `@parameter` dispatch pattern
(outer resolves the dtype, inner does the work) and should **stay** — but as private helpers
of a struct, not module-level functions.

---

## 3. The regression this must also fix

Two independent `--competition` runs after `2ecc58f`:

| row | baseline (`319c0ca`) | after 3a |
|---|---|---|
| `groupby_multi[1m_g100k]` marrow | 3.02 ms | **3.28 / 3.29 ms** (+9%) |
| `groupby_sum[1m_g100k]` marrow | 1.87 ms | 1.89 / 1.89 ms (flat) |

Wins fell **12/15 → 11/15**. Marginal cost per added aggregate at g100k: **288 → 350 µs**
(polars 145 µs), widening the deficit we set out to close from 2.3× to 2.4×.

**Single-aggregate flat + multi-aggregate regressed** isolates the cause to `AggFunc`: each
aggregate dispatches through an erased box holding a `grouped_fn` pointer — an indirect call
per aggregate per column, where the tag switch previously resolved once and then ran direct.

**This is why the two problems are one change.** Pushing `out_dtype`/`grouped` onto the kernel
as comptime members is also what removes the `grouped_fn` indirection. Fixing the design fixes
the regression.

Candidate shapes, cheapest first:
1. Hoist `grouped_fn` resolution out of the per-column loop — paid once per aggregate, not per chunk.
2. Better: `AggFunc.typed` (comptime, zero indirection) is the **only** thing the fused spec uses;
   the dynamic path resolves directly to `K`'s method inside its own switch. The erased box then
   exists only where a runtime name genuinely does.

---

## 4. Hard constraints (owner-set — do not relitigate)

- **Do not touch `AggState` / `AccType` / the `NumericType` bounds.** Widening cascades into the
  fused `Value` tower (`Reduction` conforms to `NumericValue`, whose `OutType` must be
  `NumericType`) and was reverted once already. That is **step 2b**, separate.
  When it is attempted: pick the **largest type in the family** for the accumulator (collapses
  monomorphisation from per-dtype to per-family), narrow on emit for min/max.
- **Do not change array/scalar/builder layout** (fields). Adding methods is fine.
- **No backward-compat shims.** Owner: "no backward compat must be kept, we can break anything."
  Deleting is preferred to adding parameters or flags.
- Keep `marrow.aot`/`marrow.expr` small-binary: closed erasure, no open dispatchers.

### The erased-box cost rule (measured, expensive to rediscover)

Putting `whole`/`partials`/`merge` on `AggFunc` next to `out_dtype`/`grouped` cost
**+3.2 MB, +24%** (7.8× → 10.3×). **An erased box is live code for every field × every kernel
its name switch can produce**, and a relational plan never calls those three. Splitting them
into `AggFold` restored 7.8× exactly.

Implication for this refactor: prefer **fewer, narrower boxes**. Every field added to an erased
box multiplies by the kernel count.

---

## 5. Measured facts — do not re-derive

### Binary size is not an aggregate problem

| binary | ratio |
|---|---|
| `query_streaming` (filter+project) | 1.0× |
| `query_streaming_agg_fused` (comptime kernels) | **7.6×** |
| `query_streaming_agg` (runtime names) | **7.8×** |
| `query_dynvalue` / `query_runtime` | 12.8× |

Making the aggregate identity comptime is worth **2.9%**. The comptime spec provably works —
`AggState` symbols 22 → 1, exactly two leaves, no name switch reachable — the effect is just small.

**The real driver is grouping.** `HashGrouper.consume_keys` → `kernels/hashing.mojo:44`
(`from .cast import cast`) pulls **797 `kernels::cast` symbols — 20% of the fused binary**
(628 `NumericCast`, 79 `DecimalCast`, 55 `BoolToNum`, 27 `TemporalCast`).

> **Do not chase the size gate with aggregate work.** It needs a comptime *key* spec
> (Q1.1's hashing/sort dtype-ladder rework). Aggregates cannot touch that 20%.

### Three predictions, three misses

| predicted | measured |
|---|---|
| relocating the tag switch shrinks the binary | 0% |
| comptime aggregate identity shrinks it | 2.9% |
| the dynamic path won't regress | **+9%** at g100k multi |

The through-line: reasoning about *where code lives* rather than *what executes and what gets
instantiated*. Measure before believing a structural argument.

### Performance standing (baseline `319c0ca`)

marrow-dynamic wins **12/15** group-by rows vs pyarrow/polars/duckdb. **polars is the only close
competitor** (1.2–1.5× at 1M rows). Marginal cost per added aggregate, `(multi − sum)/4` at 1M:

| cardinality | marrow | polars |
|---|---|---|
| g10 | **56 µs** | 399 µs |
| g1k | **111 µs** | 160 µs |
| **g100k** | 288 µs | **125 µs** ← the only row we lose |

**Grouping is already amortised** — `AggregateProcessor` computes gids *once*, then aggregates
per column. Only the value scan repeats, not the hash lookup. So fusion **cannot** claim an
"N passes → 1" win; the real target is value-scan and accumulator traffic at **high cardinality**,
where per-aggregate output arrays stop fitting in cache.

> **Validate step 4 at g100k, not g10.** An implementation that improves g10 and leaves g100k
> at 288 µs has missed the point.

---

## 6. Gates

Run all; report real numbers; a negative result is valuable.

1. `pixi run -e dev check_lib` — known false positives: `'main()' is not supported within
   packages`, some `invalid call to 'bitmap_and'`. Nothing else.
2. `pixi run -e dev pytest marrow/kernels/tests/test_groupby.mojo marrow/kernels/tests/test_aggregate.mojo` — baseline **55 passed**.
3. `pixi run -e dev check marrow/expr/tests/test_streaming.mojo` **and** `check marrow/expr/tests/test_values.mojo`.
   A green `check` on **one** file is not proof the library builds — this has bitten us twice.
4. `pixi run -e dev pytest marrow/expr/tests/test_streaming.mojo` — baseline **43 passed**.
5. `pixi run test_parallel` — baseline **1848 passed / 314 skipped / 0 failed**, ~5–6 min
   (vs 38 min serial). Drive this yourself; do not delegate it.
6. `pixi run -e bench pytest --benchmark --competition python/marrow/tests/bench_groupby.py` —
   **must restore 12/15 wins** and `multi[1m_g100k]` to ≤ 3.02 ms. Run it **twice**: single runs
   drift ~8–10%, which is enough to hide or invent a regression.
7. `pixi run binary_size` — expect no change (see §5); report if it moves.

---

## 7. Sequencing after this

1. **3b — this file.** Inversion + regression fix.
2. **4 — `FusedAggregation`.** Single pass, AoS accumulator blob, comptime offsets per kernel,
   zero dispatch. Validate at **g100k** first. This is the step that must show the win.
3. **2b — `AggState` widening** (family-max accumulator). Independent; needs a new
   `PrimitiveType`-bounded tier in the `Value` tower designed *up front*, not discovered mid-refactor.
4. **Q1.1 — comptime key spec** for grouping. The only thing that moves the binary-size gate.
5. Pluggable grouping strategies (`GroupStats` / `suitable` / `rank`) — after fusion, since fusion
   changes the per-row costs the policy thresholds encode. Requires the bench harness to be able to
   **force** a strategy, or "improve X without hurting Y" is unverifiable.
