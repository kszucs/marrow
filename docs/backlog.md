# Backlog

The single source of truth for what is open. Verified against the code at
**`b2e7dae` (2026-08-03)**, then re-verified item by item through **2026-08-05**
as each was worked — not read off a header.

**Goal: marrow is a usable single-node columnar query engine.** Arrow-spec
completeness is not the objective — layout and kernel gaps are scheduled only
when a milestone query needs them. The deferred Arrow-parity list is §7.

Resolved items are **deleted, not struck through** — git history has them.
The exception is an item whose *diagnosis* turned out wrong: those are kept and
marked `done`, because the correction is the part worth reading. B22 ("a core
transfer defect, not a test bug" — it was a test bug), B24, Q7.5 and Q5.1 are all
of that kind, and A3 has now been cut down the same way without being started.

> **Re-verify before trusting any status line here.** This file replaces seven
> task documents that had drifted so far apart that a 2026-07-30 consolidation
> pass still left the index and `tasks-code-quality.md` disagreeing about five
> tasks. A 2026-08-03 audit found **18 wrong statuses**: eight tasks marked open
> that were done, four marked done that were not, and six whose premise the
> two-lane refactor had destroyed. Check with `grep`, not with a header.
>
> It also replaces `execution-engine-roadmap.md` — a second living plan is the
> same drift pattern one level up — and the four feature design docs
> (`sort-design.md`, `joins-design.md`, `groupby-design.md`,
> `decimal-type-design.md`). What each of those designed and the tree then built
> differently is §8; what they designed and nobody built is §4 and §7.

---

## 0. Standing constraints

Each of these cost real time to find and invalidates an approach that looks
obvious. Read before planning anything.

### Architectural invariants (gate every merge)

1. **Small-binary DCE.** Preserve the closed-erasure property: no open
   dispatchers, fused-only value boxes, closed per-dtype kernels. Gate on
   `pixi run binary_size`.
2. **One engine, two drivers.** No feature may exist in only one lane. Windows
   currently violate this (AOT-only) — see M2.3.
3. **PyArrow-shaped naming** in the core types and the bindings.
4. **Code quality is an acceptance criterion**, not a follow-up. Behaviour lives
   on the type or trait, not in free functions. A wave does not open until the
   prior wave's quality pass is green — it is a gate, not cleanup-later.

### Do not change

- **Array, scalar and builder layout.** Adding methods and accessors is fine;
  adding, removing, reordering or re-typing fields is **out of scope, not
  deferred**. Note this constraint has been read too broadly before: B13 (a
  sliced `BoolArray` double-applying its offset) was filed as blocked by it,
  when the fix was deleting a redundant `+ self.offset` and touched no field at
  all. Check whether a fix actually needs a layout change before deferring it.

### Measurement traps

- **The binary-size gate's file-size number is quantized to 16 KB.** Apple
  Silicon uses 16 KB pages, so a stripped binary's *file size* moves in 16,384-byte
  steps — a real +1,728-byte change once showed up as +16,504 with one *fewer*
  symbol. Measure `size -m <binary>` → `Section __text`.
- **Measure one gate binary directly** (`mojo build -O3 -g0 -I . …query_dynvalue.mojo`,
  ~2.5 min), not the whole `pixi run binary_size` sweep (~10 min).
- **A generic wrapper around an already-erased dispatch is not free.** Folding
  twelve promote-then-dispatch sites into one `_arith[K]` helper cost
  **+115,600 bytes**; writing the same four lines inline in each arm cost a
  fraction. A parameterised method is instantiated per kernel and each
  instantiation carries its own copy of what it touches.
- **A runtime switch over tags is the anti-pattern.** Rewriting the runtime lane
  as one `_eval` switch over ~70 tags cost **+1,807,168 bytes of `__text`
  (+45.7%)** on `query_dynvalue`, because every arm became reachable from every
  node. The fn-pointer `EvalFn` (`dynamic.mojo:209`) exists for this reason.
- **Reachability intuitions about erased paths are usually wrong; stub and
  measure.** Stubbing both `cast_array` calls out of the expression layer left
  the gate binary byte-identical.
- **An operator with no benchmark has no performance.** T2.4's per-row-group scan
  shipped a **4.7x** regression that every test passed through, because nothing
  benched the scan operator.
- **A benchmark whose captured value is not `keep()`-alive after `b.iter[call]()`
  measures nothing** — one reported 17,774 GElems/s. Check throughput for
  physical plausibility before believing a flat A/B.
- **The pytest-benchmark table prints mixed units per row.** Compare
  `--benchmark-json` medians (seconds), never the table's numbers.
- **Benchmarks here vary 10–18% run to run.** Interleave repeats across refs
  (nesting them concentrates machine drift on whichever ref is measured last and
  *invents* regressions), use five or more, compare ranges.
- **A fixed per-call cost hides until you change how often the call happens.**
  `ParquetFile.read` built a `CompressionLibs` per worker and the first
  decompress `dlopen`s the codec library — invisible at one read per file, 4.7x
  at one read per row group.

### Compiler and platform facts

- **`Buffer` requires 64-byte pointer alignment**, so `read_at` cannot return a
  sub-`Buffer` at an arbitrary file offset, and neither can IPC's `_slice_body`:
  Arrow IPC pads to 8, not 64. A source owns *one* whole-file `Buffer` and hands
  out `BufferView`s. This has blocked two separate designs.
- **A comptime conditional type carries no trait conformance** and does not
  reduce at a return site, even inside a `comptime if` that selected the branch;
  `rebind` does not rescue it.
- **A capturing closure's type is parameterised by its creating scope**, so it
  cannot be stored in a struct field and outlive that scope. Every stored
  callback must be `thin`.
- **`ctx.stripe` bodies may not raise**, and widening it miscompiles: the
  parameter form of `sync_parallelize` that accepts a raising worker needs an
  implicitly-capturing closure whose captures are silently not made. Watch for
  "assignment was never used" warnings on buffers the body writes.
- **`origin_of(a, b)` is an origin union**, which is what lets a function return
  values borrowed from either of two storages.
- **`.claude/worktrees/` contains two stale worktrees** (`docs-revamp`,
  `q25-aggregates`) holding pre-Q2.5 `AGG_*` and `reinterpret_array` code.
  Exclude them from every grep or you will get false positives.

---

## 1. Wave 1 — Correctness

Defects that produce **wrong answers with no error**. These come first because
the M1 gate is "results cross-checked against DuckDB": a wrong multi-key sort or
an unpruned date predicate corrupts exactly the thing the milestone measures.

**Every item opens with a failing test.** The remaining entries were found by
reading code paths and are unverified until that test exists — write it and
watch it fail before touching production code, because at least one of these
turned out to behave differently from how it read. (B20's missing-column `-1`
did not silently return the wrong column, as recorded: it tripped a bounds
assert that killed the whole test binary, which also masked B14 and B15 until
it was fixed.)

Two root causes account for most of this list, and fixing either collapses a
group rather than a line:

- **`offset` has two conventions** — views index logically from zero, owning
  `Buffer`/`Bitmap` do not — and array code mixes them. **All of this group is
  now fixed** (B11, B13, B16, B17); the convention split itself is not, so a new
  array method that indexes a bitmap can still pick the wrong one.

| ID | Defect | Evidence | Size |
|---|---|---|---|
| **B4** | **BIT_PACKED Parquet levels are mis-decoded.** `definition_level_encoding` / `repetition_level_encoding` are parsed (`format.mojo:576-578`) then never consulted — `_data_page_v1` (`reader.mojo:240-270`) applies `Rle.decode` unconditionally, and also reads RLE's 4-byte length prefix, which BIT_PACKED pages do not have. **Blocked on a fixture, not on the fix.** The guard is a few lines, but BIT_PACKED is deprecated: arrow-cpp never writes it and PyArrow exposes no option for it, so there is no reference writer to produce a test file. Doing this means hand-assembling a Parquet file with Thrift page headers declaring BIT_PACKED. Do not land the guard untested — a decode path that silently changes behaviour is exactly what this card is about. | as cited | S fix, M fixture |
| **B23** | **RESOLVED for marrow; the toolchain bug is characterised and ready to report.** The harness half landed first: `--mojo-timeout` (default 1800) turns a hang into an ordinary failure, because `MojoRunner.collect` splits and retries on a compiler *crash*, which emits a signal, while a hang emits none. **Then the deadlock itself was bisected, and my earlier note that it was "the shape of the single-file unit" was wrong.** It is one call: the *erased* `take(DynArray, Int32Array)`. Minimal reproduction, ~10 lines — import `DynArray`, `array`, `int32`, `take`; call `take(a.copy(), array([2, 0, 3], int32))`. Deadlocks after ~9 s of CPU with the main thread in `_dispatch_semaphore_wait_slow` and all 11 Mojo worker threads parked at the same frame, every frame inside the `mojo` binary. Evidence it is the *call*, not the imports: importing `take` without calling it passes in 5.9 s; `test_inner_join_basic` from the same module passes alone in 41 s; and — counter-intuitively — `test_filter.mojo:583` makes the byte-identical call in a *larger* unit and compiles fine, so a bigger compilation unit is what avoids it. **marrow's fix:** the two `take` tests were living in the *join* test file; moved to `test_filter.mojo` where they belong. `test_join.mojo` now passes on its own in **48 s**, against 7 hours before. **Still to do:** file the reproduction upstream. | `conftest.py`, `test_filter.mojo` (done); upstream report (open) | S |
| **B24** | **RESOLVED — deleted.** `is_fixed_size()` answered `is_bool() or is_primitive()`, so it covered neither `fixed_size_binary` nor `fixed_size_list`, the two dtypes its name promises. It had **zero production callers** — only tests and one docstring cross-reference. Arrow C++ spells the real predicate `is_fixed_width` = `is_primitive || is_dictionary || is_fixed_size_binary` (`type_traits.h:1400`), which notably excludes `FIXED_SIZE_LIST`; marrow's matched none of that. Removed rather than widened: adding dictionary/fsb coverage for no caller is speculative, and the two real uses are already spelled `is_bool() or is_primitive()` inline where they are needed (`ipc.mojo:1941`). `dtypes.mojo` now records Arrow's definition so anyone who needs it adds the right thing deliberately. | `dtypes.mojo` | done |
| **B25** | **`test_views_gpu.mojo` fails 13 of 15 with a Metal codegen error**, and has for at least the whole of this branch. Every failure is `error: Metal Compiler failed to compile metallib. Please submit a bug report.` followed by `mojo: error: failed to run the pass manager` — a backend crash, not an assertion. Verified 2026-08-05 at both `d565c1a` and `3c747c0` (pre-block), so nothing in the Q7.4/Q7.3/B8 work caused it. `test_buffers_gpu.mojo` (4/4) and `test_arrays_gpu.mojo` (4/4) pass, so the accelerator itself works — it is these kernels. Compiling all three GPU files as one selection fails the same way, so it is not a combined-unit effect either. **This invalidates an earlier claim in this repo's history that `test_views_gpu` passes 15/15.** Needs: minimise which of the 13 cases trips the backend, then either work around it or file upstream. | `marrow/tests/test_views_gpu.mojo`; `views.mojo` `_apply_dispatch`/`reduce` | M |
| **B26** | **RESOLVED.** `__eq__` compared the bitmaps themselves — presence against presence, then whole bitmap against whole bitmap — so an all-valid array carrying a bitmap was unequal to one carrying none, and two slices whose logical validity matched were unequal whenever their offsets differed. **Six** array types shared the shape, not just `PrimitiveArray`. New `_validity_equal(length, a, b)` compares null *positions* through offset-applied views; a missing bitmap means all-valid, which is a value rather than a distinguishing representation. This mattered beyond equality: CLAUDE.md tells you to write `assert_true(result == expected)` instead of an element loop, and every kernel that intersects validity emits an array with a bitmap while `array([...])` emits one without — so the recommended assertion was unreliable for exactly the outputs a kernel test checks. Found while writing Q2.3's test, which had to compare elementwise and has since been restored to `==`. | `arrays.mojo` `_validity_equal` + 6 `__eq__` | done |
| **B28** | **The fused expression lane costs ~2 ns/element for a trivial op — ~6x off memory bandwidth.** Measured 2026-08-05 while re-pricing Q7.1: `a + 1` over a 1M-row int32 column, entirely fused, no breaker, takes **1.99 ms** (`bench_b27_probe_plain_fused_add_1m`). That moves 4 MB in and 4 MB out; at even 20 GB/s it should be under 0.4 ms. For comparison `LengthKernel.apply`, a hand-written kernel over the same row count, runs at 68.8 µs. So the per-lane machinery — `vectorwise` threading `batch`, `ctx` and a `mut slot` through every node, per SIMD chunk — dominates the arithmetic by an order of magnitude. B27 removed an atomic from that path and bought 2x; this is what is left. **This is the strongest evidence yet for A1**: a typed `comptime State` returned by `prepare` and passed into `vectorwise` is exactly the change that stops re-deriving per-chunk state. Price A1 against this number. | `expr/values.mojo` fused driver; see A1 | L |
| **B22** | **RESOLVED, and the diagnosis in this card was wrong.** The two failing cases in `test_buffers_gpu.mojo` were a *test* bug, not the "core transfer defect, not a test bug" this card asserted. `Buffer.unsafe_set[T]` infers `T` from its value argument, so a bare literal (`unsafe_set(1, 13)`) widens and strides by the wider type; `unsafe_get[T]` has no argument to infer from and defaults to `uint8`. The write landed at byte 8 and the read looked at byte 1 — hence "reads back 0 where 99 was written", and hence index 0 passing while index 1 failed, which is the tell nobody followed. Typing the literals makes all 4 pass. Verified 2026-08-05: the synchronization hypothesis was tested first and **refuted** — adding `ctx.synchronize()` to `to_device` and `alloc_host` changed nothing. `to_device`/`to_cpu` are fine. `unsafe_set` now documents the asymmetry. **What remains true and is now the real item:** CI passes `--no-gpu` on all four jobs, so the five `*_gpu.mojo` files never run there — see I-notes. | `marrow/tests/test_buffers_gpu.mojo`; `buffers.mojo` `unsafe_set` | done |

**Also small and worth clearing in this wave:**



---

## 2. Wave 2 — Infrastructure

**CI has not run since 2026-05-11**, and on that last run everything except Lint
was already failing. None of the work below is verified by anything but local
runs. This wave is cheap and it is what makes every later wave believable.

| ID | Item | Evidence | Size |
|---|---|---|---|
| **I2** | **RESOLVED.** All three of the card's claims held: the `ma.*` compute functions had moved to `marrow.compute`, `Array.is_valid()` is elementwise and takes no index, and `sort()` lost `ascending=` in favour of `sort_keys=`. Build goes from **7 failing pages to 0**, 15/15 rendered. Two things the card did not have: the `is_valid(index)` misuse was also in `guide/arrays.qmd`, not only `compute.qmd`; and `sorting.qmd` carried a stale `argsort`/plain-array `sort()` snippet, renamed to `sort_indices` back in `f043246`. The one remaining `cell-output-error` is a deliberate `#| error: true` demo in `guide/types.qmd`. Left alone and noted: `examples/datafusion.qmd`'s prose about a trailing `None` selecting CPU execution is a separate pre-existing inaccuracy on a page with `execute: enabled: false`. | done |
| **I3** | **RESOLVED.** `.github/workflows/binary_size.yml` runs `check_gate.py`, which builds the four gate binaries and compares `__text` against a checked-in `benchmarks/binary_size/baseline.json` at a **0.5%** threshold — tighter than 1% on purpose, since B12's 0.63% is the regression that motivated this. Baseline is JSON rather than the `BASELINE.md` this card's sibling mentioned, so it is machine-updatable (`check_gate.py --update`) and cannot drift as prose — which is exactly I4's complaint. macOS-only, because `compare.py` reads `size -m`, which is Mach-O. Cannot be validated by pushing (CI is dark), so it was validated locally: the gate runs clean at 0.000% delta, and `--update` was exercised for real when `num_buffers`' always-on validation moved all four numbers. | done |
| **I4** | **RESOLVED.** The machine-readable baseline is `benchmarks/binary_size/baseline.json`, regenerated by `check_gate.py --update` and enforced in CI, so prose numbers can no longer be the source of truth. The README's remaining stale figures were verified against a real sweep and corrected: 5 of 7 quantitative claims were wrong, including the whole current-baselines table, the `query_arith` demonstration, and the `query_dynvalue`/`query_runtime` comparison — whose **direction had flipped**, not just its magnitude (stripped files are now byte-identical while `__text` differs by 1,600 B). **This card's own premise was also stale**: the "7.6x / 7.8x / 12.8x table" it names no longer existed, having been replaced in `c72144d`; what was actually stale was the replacement. The ~5.27M -> ~4.08M drop in the erased binaries since 2026-07-29 is the interpreter deletion, a real shrink rather than drift. | done |
| **I5** | **RESOLVED, with one claim found stale.** `benchmarks/binary_size/README.md` documented `query_comptime.mojo`, `query_erased_aot.mojo` and `query_hybrid.mojo`, plus `marrow/aot/relations.mojo` and `Planner.build()` — all verified genuinely absent, all removed. **The card's "`BASELINE.md` is referenced three times" was already false**: zero hits by grep. Stale docstrings in `compare.py` and `query_runtime.mojo` fixed alongside. | done |

**Not scheduled, but know the gaps:** GPU is never exercised (every job passes
`--no-gpu`; the five `*_gpu.mojo` files, 39 cases, never run). ASAN on Linux is
hard-disabled (`test.yml:55 if: false`). `precompile` — the warning-clean gate —
is not in CI. There is no separate `test_python` job.

---

## 3. Wave 3 — M1, the ClickBench milestone

**M1 = 42 of the 43 ClickBench queries** (Q29 `REGEXP_REPLACE` deferred to M2)
over the single flat `hits` table, run through marrow's frontend, results
cross-checked against DuckDB, with the binary-size gate green. It is the first
"ship it" line: it proves the whole analytical loop — scan → filter →
hash-aggregate → group-by → order-by-limit → string filters → date parts — on
both frontends.

The queries are the ones in `~/Workspace/ClickBench`, driven through marrow's
API rather than through SQL; `hits` is a single wide flat table (~105 columns,
no joins, no nesting), which is why it is the first target. **Acceptance is
three things, not one**: results match DuckDB, wall-clock is
competitive-or-better against polars and duckdb on the same box, and the AOT
lane's `__text` stays inside the `benchmarks/binary_size/` budget. Reading the
43 queries once already promoted five capabilities out of M2 — min/max on
string and date, `count_distinct` as a relational aggregate, `HAVING`, computed
group keys and computed aggregate inputs, and `date_trunc` — and all five have
landed.

Today `python/marrow/tests/clickbench.py` is **11 queries, eager, PyArrow doing
the I/O**, restricted to six numeric columns and GROUP BY. Its own docstring
concedes the restriction.

This path is almost entirely sequential and is the whole critical path.

### M1.0 — Widen the numeric dispatch bound — **DONE 2026-08-05**

`NumericCompareKernel.apply` is bound on `PrimitiveType` but `dispatch` narrows
to `NumericType` (`numeric.mojo:552` vs `:570`). Consequences, all live:

- runtime-typed comparison on **timestamp, date, time, duration, interval,
  decimal** raises;
- `equal_any` (`numeric.mojo:602-605`) raises, so **hash joins and `nullif` on
  those key types fail**;
- `pruning.mojo:115-135` mirrors the bound, so **no row group or page is ever
  pruned on a temporal or decimal predicate** — ClickBench Q's filtering on
  `EventDate`/`EventTime` get no pushdown at all;
- the same bound blocks decimal and temporal aggregates
  (`aggregate.mojo:102`, `:864`).

This is precisely the defect class CLAUDE.md's *"dispatch on the widest family
the typed leaf accepts"* rule was written for. It is already fixed in
`filter`/`take` (`filter.mojo:97`) and `sort` (`sort.mojo:433`).
**Done.** `NumericCompareKernel.dispatch` now dispatches on `PrimitiveType`,
and `pruning.mojo` with it — the card was right that pruning mirrored the bound,
and fixing only the kernel would have left the headline benefit (date pushdown)
unrealised. Arithmetic's own `dispatch_numeric` at `numeric.mojo:134` was left
alone: adding two dates is not meaningful, and widening it is a separate
question with its own semantics.

Cost, caught by the new CI gate on its first real use: `query_streaming`
+0.512%, `query_join` +1.154%, aggregates unchanged. That is comparison leaves
being emitted for temporal, interval and decimal — the code that supports those
dtypes — and the baseline was raised deliberately.

### M1.1 — Optimizer v1 — **L**

No `optimize.mojo` exists. Two ad-hoc rewrites live in the *builder* instead:
predicate → `ParquetScan` (`relations.mojo:437-443`, non-recursive, fires only
when `Filter` sits directly on the scan) and `Limit` → `Sort` top-K (`:723-735`).

Deliver a `DynRelation → DynRelation` rewrite pass with:

- **conjunct splitting** — `Filter` holds one `predicate: BoxedValue`
  (`relations.mojo:866`), not a `List[BoxedValue]`; splitting `AND` is the
  precondition for partial pushdown;
- **predicate pushdown** through `Project`/`Sort`/`Limit`, recursively;
- **projection pushdown** — a `ParquetScan`'s schema *is* its projection, so this
  is a schema rewrite. `referenced_columns()` is implemented on both lanes and
  the box (`values.mojo:348`, `dynamic.mojo:532`, `relations.mojo:273`) and is
  **currently called only by tests** — this is its consumer;
- **limit pushdown** into the scan;
- constant folding.

Structure it as a rule list over the immutable IR, each rule a pure
`DynRelation -> DynRelation` rewrite, mirroring DataFusion. Two rules are M2/M3
work rather than v1 and should not hold it up: **predicate pushdown below
joins** (H2O/TPC-H) and **CSE**, whose design is in
`design-expression-evaluation.md`. `is_deterministic()` was removed as a
default-`True` method with no caller; do not reintroduce it until a rule
actually needs it.

Gate: the rewrite must not make the fused lane's binary grow — an open dispatcher
in the optimizer would breach invariant 1.

### M1.2 — Python lazy bindings — **L**

`python/bindings/lazy.mojo` does not exist. `lib.mojo:17-30` registers eight
modules and **no plan or expression type is bound**; the only expr contact is
`bindings/compute.mojo:17`, importing aggregate functors. Bind `DynValue`,
`BoxedValue`, `DynRelation` and the builder methods.

### M1.3 — ibis-flavoured `Table` / `Column` — **L, blocked on M1.2**

`python/marrow/expr.py` does not exist. A thin ibis-*flavoured* native
`Table`/`Column` over `DynRelation`/`DynValue`, `.collect()` at the end — **not**
a real ibis backend and **no `ibis` dependency**. ibis is a naming guideline.
Today's `python/marrow/__init__.py:340 class Table` is the eager PyArrow-style
wrapper and stays as-is.

Ships with it: `marrow.read_parquet(path)` and `marrow.table(schema)` as the two
entry points that return a lazy `Table` (glob support arrives with M2.8). Decide
at the same time whether Python's eager `join`/`group_by`/`sort_by` — which
bypass the expression layer entirely today (`tabular.mojo`) — route through
`execute(plan)` or stay documented eager shortcuts. Two divergent paths is the
outcome to avoid; picking either deliberately is fine.

### M1.4 — Kernel gaps M1 actually needs — **`date_trunc` DONE 2026-08-05**

`date_trunc` now supports **month, quarter and year** alongside second/minute/hour/day, so ClickBench Q35/Q36 are no longer blocked. These are *calendar* units with no fixed length, so they cannot floor by dividing the tick count the way the others do; they go through `_civil_from_days`/`_days_from_civil`, the Hinnant algorithms the extraction kernels already use, so no new date arithmetic was added.

Fixed on the way: `DateTruncKernel` short-circuited **every** unit for `date32` on the grounds that date32 is day-granular. True for units up to a day; for month/quarter/year it silently returned the input unchanged. Both expression lanes route through `CalendarUnit.parse`, so they picked the new units up without change.

Whatever else this card lists is unaffected and still open.

### M1.5 — ClickBench through the lazy plan — **M, the M1 sign-off**

Rewrite `clickbench.py` against the lazy frontend, all 42 queries, cross-checked
against DuckDB, wall-clock compared to polars/duckdb on the same box, binary-size
gate green.

### M1.6 — AOT DSL docs — **stale API references FIXED 2026-08-05**

`docs/guide/expressions.qmd` documented a `Planner` type (**zero hits** in the tree), an `AnyValue` type (**zero hits**), and `from marrow.expr import execute` — `execute` is a *method* on the plan (`relations.mojo:394`), not a free function, so every example calling `execute(plan)` was wrong. All corrected, and the page now describes what the tree does: plans are immutable `Relation` trees and `.execute()` opens one into a processor tree, so a plan is a reusable template. The site still builds clean.

These blocks are illustrative (plain ```python```, not executed), which is exactly why the errors survived I2 — the docs build cannot catch prose that names types that do not exist.

**Still open on this card:** the runnable AOT example it also asks for.


---

## 4. Wave 4 — M2 and M3 enablers

**M2 = the H2O.ai db-benchmark** — its 10 group-by and 5 join queries at 0.5, 5
and 50 GB. It is the bar for hash group-by and hash join across sizes and key
types; the 50 GB size is what forces spill. **Gate: group-by and join pass at
5 GB, spill demonstrably works, wall-clock competitive with polars and duckdb.**

**M3 = TPC-H** — all 22 queries. It is the bar for multi-way joins, join
reordering and full relational breadth. **Gate: the 22 queries run, and join
reordering is measurable on the multi-join ones.**

Both inherit M1's standing acceptance criteria: results cross-checked against
DuckDB, and the binary-size gate green. Scheduled below are only the
capabilities those two milestones require.

| ID | Item | State | Size |
|---|---|---|---|
| **M2.1** | **`Distinct` and `Union` relational nodes.** Neither exists; `relations.mojo` has 8 nodes and no `RELATION_*` discriminant for either. Needs a `unique` kernel (below). `Values`/`EmptyRelation` literal sources are the same shape and are worth landing alongside, as optimizer rewrite targets. Set-op *kernels* (`intersect`/`except`) are post-M3 and not scheduled. | not started | M |
| **M2.2** | **`unique` / `value_counts` / `dictionary_encode`.** `distinct.mojo` is cardinality-only and there is no `def unique` anywhere under `marrow/kernels/`. **marrow consumes dictionaries but can never produce one** — not from a kernel, and not from Parquet, whose reader materialises dictionary pages into plain arrays. | not started | M |
| **M2.3** | **Real window functions.** Today: `row_number` only, **AOT lane only** (violating invariant 2), `WindowSpec` carries frame bounds but no PARTITION BY and no ORDER BY, `FrameBound.kind` is an untyped `UInt8` never read, `RowNumberKernel` ignores its `values` argument, and nothing outside `values.mojo` references any of it. Sequence: move to `marrow/kernels/window.mojo` → give `WindowSpec` `how`/`partition_by`/`order_by` → partition (reuse `groupby` hashing) + sort within partition (reuse `sort`) + scatter back → ranking family → navigation family (`Lag`/`Lead`/`NthValue`) → `RunningAgg[K: AggKernel]` → `.over()` on both lanes → wire through `relations.mojo`. `docs/window-functions.md` is the forward spec; this card owns the sequencing. | 2-node toy, `values.mojo:1975-2039` | L |
| **M2.4** | **Statistical aggregates** — `variance`, `stddev`, `quantile`, `approximate_median`, `mode`, `first`, `last`. `resolve_agg` is a closed list of exactly 8 (`expr/aggregates.mojo:194-221`). TODOs already acknowledge the variance gap at `aggregate.mojo:563,574,589`. | not started | M |
| **M2.5** | **Spill.** Zero occurrences of `spill` in the tree; no memory-budget tracking and no disk I/O anywhere. Required by H2O at 50 GB. Grace hash join, a spilling grouper, and a memory budget on `ExecContext` to trigger either. Note both blocking operators buffer unboundedly today: `AggregateProcessor` (`execution.mojo:699`) keeps every morsel's group ids and evaluated value columns, and `JoinProcessor` collects the whole left side. | not started | L |
| **M2.6** | **String manipulation and regex — the single largest kernel hole.** There is no regex engine in the repo. Missing: `match_substring_regex`, `replace_substring(_regex)`, `extract_regex`, `split_pattern(_regex)`, `count_substring`, `find_substring`, `utf8_slice_codeunits`/substring, `lpad`/`rpad`, `binary_join`, the whole `utf8_is_*` classification family, trim-with-charset. Also: string kernels dispatch on `is_string_like()` only, so `binary`/`large_binary` are excluded from string comparison. | not started | L |
| **M2.7** | **Temporal completeness** — `strftime`/`strptime` (and **string↔timestamp cast raises**, `cast.mojo:1028`), timezone-aware extraction (everything decomposes as UTC and a non-UTC `tz` is silently ignored, `temporal.mojo:36-39`), `week`/`iso_week`/`iso_year`, `millisecond`/`microsecond`/`nanosecond`, `is_leap_year`, `ceil_temporal`/`round_temporal`, and the `*_between` family. Temporal **arithmetic** belongs here too — date ± interval, `date_diff`, `now` — which H2O and TPC-H date logic both need and which nothing implements. | not started | M |
| **M2.8** | **Multi-file / dataset scan.** `ParquetScan.path` is a single `String`. No glob, no dataset, no partition discovery, no fan-out. Also: **bloom filters are fully implemented in the reader and never consulted by the scan** (zero `bloom` hits in `marrow/expr/`) — cheapest remaining pruning tier, do it with this. Two known-safe-but-lossy behaviours ride along: predicate pruning switches *off* entirely for nested files rather than risk misaligning statistics with the projection, and Hive-style `col=val` directory discovery does not exist. | not started | M |
| **M2.9** | **Join on computed keys.** Every join key must be a bare column reference; a computed expression raises at `relations.mojo:597` and `:606`. H2O and TPC-H both need it. | raises today | M |
| **M2.10** | **Plan-level parallelism.** The kernels are parallel; the pull loop is not — `collect()` drains one morsel at a time on the calling thread and nothing schedules operators across workers. Pairs with M3.3, which is the one-line half of the same gap. | not started | L |
| **M2.11** | **`CoalesceBatches`.** Nothing compacts small morsels after a selective `Filter`, so every downstream operator pays vector-at-a-time overhead on sparse batches. No such node or processor exists. | not started | S |
| **M2.12** | **A remote `ByteSource`.** The seam is in place and has exactly one implementation: `trait ByteSource` (`parquet/source.mojo:20`), `MappedFile` (`:49`), `ParquetFile[S: ByteSource]`. The OpenDAL Mojo binding (`~/Workspace/opendal/bindings/mojo` — operator verbs, seek-based ranged reads, fs/s3/http/memory, blocking only) is a capable WIP with **zero integration**: `opendal` appears nowhere under `marrow/` except one comment. Mind the 64-byte `Buffer` alignment constraint in §0 — it has already blocked two designs. | not started | M |
| **M2.13** | **`EXPLAIN` / plan pretty-printer.** `BoxedValue.render()` (`relations.mojo:270`) and `DynRelation.write_to` exist; nothing composes them into a plan dump on either frontend. Debuggability blocker for M1.1 the moment rewrites start moving nodes. | not started | S |
| **M3.0** | **Stream the probe side for LEFT/FULL/SEMI/ANTI.** These block on the whole probe side today: `JoinProcessor` collects it and probes once, because the kernel recomputes build-side matches per probe and `probe()` returns an assembled `StructArray` rather than pairs, so a caller cannot accumulate them. Streaming needs `HashJoin` to carry the matched-build set across probes and emit the tail on drain. Correct but blocking is where B5 left it; this makes it correct *and* streaming. | `expr/execution.mojo` `JoinProcessor.pull`, `join.mojo:569` `_emit_unmatched` | M |
| **M3.1** | **Join completeness** — `JOIN_CROSS`, `JOIN_MARK`, `JOIN_SINGLE` and `JOIN_ASOF` are declared constants that are never implemented; a CROSS join currently falls into the LEFT/RIGHT/FULL tail and produces wrong output rather than a cartesian product. All five `JOIN_ALGO_*` constants are dead and `struct Join(Relation)` has no `algorithm` field. Sort-merge join is a commented-out stub (`join.mojo:697-709`). Non-equi joins are absent as a *class*, not just as a kind: no nested-loop, no piecewise-merge/IEJoin. | constants only | L |
| **M3.2** | **Join reordering and build-side selection.** `hash_join` always builds on `left` (`join.mojo:754`); `JoinProcessor` always builds on `self.left` (`execution.mojo:878`). No cardinality estimation feeds the plan. | not started | L |
| **M3.3** | **`JoinProcessor` discards the plan's execution context.** `Join.to_processor(ctx)` receives a ctx and constructs `JoinProcessor` without it, which then builds `HashJoin[rapidhash]()` with the default — **the relational join never uses the parallel path**. Small, high-value, could move to Wave 1. | `relations.mojo:1114-1123`, `execution.mojo:879` | S |
| **M3.4** | **O(N) top-K.** `sort_indices(limit=…)` is a full sort then truncation; the docstring concedes it (`sort.mojo:379`). No `select_k_unstable`, no quickselect, no streaming heap. Directly relevant to every ORDER BY … LIMIT in ClickBench and TPC-H. | not started | M |
| **M3.5** | **`Scan` trait above the file formats** (was L6). `RELATION_PARQUET_SCAN` is an IR discriminant; `execution.mojo:37-43` imports six symbols from `..parquet`; there is no `trait Scan`/`trait Source` — only `trait ByteSource`, a *byte*-level seam. `ParquetScanProcessor` does four jobs. **Do this before adding CSV or IPC sources, not after.** | not started | L |
| **M3.6** | **`Table` and `ChunkedArray` are thin.** `ChunkedArray` has only `chunk()`/`combine_chunks()` — no `slice`/`__getitem__`/`null_count`/`filter`/`cast`/`take`; `Table` has no `slice`/`select`/`filter`/`sort`/`rename`/`concat_tables` — all of those are `RecordBatch`-only in the Mojo core too. The engine operates on tables, so columns have to behave like columns. | not started | M |
| **M3.7** | **Decimal arithmetic.** Nothing was built: no `add`/`sub`/`mul`/`div`, no scale-alignment or result-precision rule, no 256-bit intermediate promotion. Rescale exists only inside `cast` (`cast.mojo:855-869`) and its up-scale multiply has **no overflow check**; precision is never validated, so `decimal32(40, 0)` constructs. TPC-H money math needs it, which is why it is here and not in §7. The result-type rules (`result_scale = max(s1, s2)`, `result_precision = max(p1-s1, p2-s2) + result_scale + 1`) and the scalar-loop constraint — there are no 128-bit SIMD lanes, so `vectorize`/`elementwise` does not apply — are stated here because they were the only parts of the deleted design doc worth carrying forward. | not started | M |
| **M3.8** | **Late materialization / selection vectors on the scan** — decode the filter columns, filter, then decode only the survivors. DataFusion ships the equivalent *off* by default because it is subtle; sequence it after M2.8. Async/prefetch ranged reads for remote scans (M2.12) belong with it — the OpenDAL C ABI is blocking, so it needs several readers rather than one. | not started | L |

---

## 5. Quality debt

Surviving items from the Q/L/V backlogs, with their original IDs kept so git
history and CHANGELOG references still resolve. Everything else in those files
is done and has been deleted.

| ID | What is left | Size |
|---|---|---|
| **Q0.5** | **ATTEMPTED 2026-08-05 and blocked on a real limit — read this before retrying.** The plan was to give `BoxedValue` a `dtype_hint()`: `Some` for a fused node, `None` for the runtime lane, so the 0-row probe only runs where the answer genuinely is not known. Telling the lanes apart is easy — `DynValue` is concrete, so `__init__(out self, value: DynValue)` is a more specific overload than the generic `__init__[V: Value]`, and that works. **The blocker is that a fused node knows its output dtype's *type* but not its *parameters*.** `V.OutType` is a comptime type; the dtype *value* for `decimal128(p, s)`, `timestamp(unit, tz)` or `fixed_size_binary(w)` carries fields the type does not, and nodes do not store a dtype instance — `NumericLiteral[T]` holds only `_value: Scalar[T.native]`. So `DynType(V.OutType())` fails to compile for parameterised dtypes and would silently report `decimal128(0, 0)` if it did. **This is also a second reason the probe exists**, alongside "keeps a caller-supplied schema honest": executing is how a parameterised dtype gets its parameters. Retrying means either giving all 37 nodes a dtype instance — which grows the fused structs, and they are size-gated — or restricting the hint to parameter-free dtypes and keeping the probe for the rest. Neither is obviously worth the claimed ~16 KB. | M, and less attractive than it looked |
| **Q2.3** | **RESOLVED.** All three sites now build their output bitmap from offset-applied views: `numeric.mojo:95` (arithmetic), `numeric.mojo:502` (comparison) and `string.mojo:314` (`StringPredicateKernel.apply`). New `Bitmap.intersect_views(a, b)` takes two `Optional[BitmapView]` and handles the one-sided cases through `to_owned()` — `Bitmap.intersect`'s pass-through returned the lone operand raw, offset included, which is the case that looked safe and was not. Both overloads now document which to use where. `BinaryLikeArray` had no `validity()` at all — the only array type missing it, and the reason the string kernels reached for `.bitmap`; added. Red first: three cases in `test_arithmetic.mojo` failed on the parent's null positions leaking into a slice, then passed. | done |
| **Q2.4** | **RESOLVED as miscounted — do NOT do what this card asked.** It said one `sync_parallelize` loop remained to convert, `views.mojo`'s `_reduce_dispatch`, with "the other three deliberate and documented". That loop is the *fourth* deliberate one, and its own comment says so with measurements: routing it through `ctx.stripe` forces the serial arm to allocate a one-slot partials buffer it does not need, costing ~55% on small reductions (`sumint64_1k` 0.19-0.20 -> 0.30-0.32 us, `sumfloat64_1k` 0.23-0.24 -> 0.34-0.37 us, five interleaved repeats, ranges disjoint). A reduce is a fold plus a merge, and a serial fold should not pay for the merge's scratch. Converting it would have been a measured regression shipped as a cleanup. | done |
| **Q2.5** | Aggregates, remaining steps: **2b** — `AggState[K, V: NumericType]` was never widened to `PrimitiveType`, so temporal reductions do not work natively; **3** — `count_distinct`/`approx_count_distinct` (+`_grouped`) are still four free functions (`distinct.mojo:87,133,167,214`); **4** — `FusedAggregation` (single pass, AoS accumulator, comptime offsets, zero dispatch) has zero occurrences. Gated on Q6.1. Validate at `g100k`, never `g10`. **Re-scope in light of Q6.1 (2026-08-05):** the AOT aggregate path shows **no measurable runtime advantage** over the runtime-named one (under 2%, sign flips between cardinalities). So `FusedAggregation` should not be pitched as a speed win without a measurement that shows one — the demonstrated benefit is binary size. Validate any round of this against `bench_aggregate_aot.mojo` at g100k, and be prepared for the answer to be "no change". | L |
| **Q4.1** | **`JoinKind` and `JoinIndex` done; `Grouping` and `BuildPartition` remain.** `IndexPairs = Tuple[Int32Array, Int32Array]` is now `struct JoinIndex` with named `build`/`probe` fields, so consumers stop writing `pairs[0]`/`pairs[1]` for two arrays nothing distinguishes — and `_assemble` gathers the two sides from *different* arrays, so crossing them yields a plausible result with the columns swapped rather than an error. `IndexPairs` survives as an alias. One seam is deliberate and commented: `SwissHashTable.probe` still returns a bare tuple, because it cannot name `JoinIndex` (`join` imports `hashtable`, not the reverse), so the naming happens at that boundary. **`Grouping` is the bigger half and is still open**: `(gids: Int32Array, …, num_groups: Int)` appears at 23 sites across `groupby`, `aggregate`, `distinct` and `expr/aggregates`. All four size gates unchanged. | M |
| **Q4.3** | Parquet leaf visitor. **Zero uses of the `dispatch_*` family anywhere in `marrow/parquet/`**; hand-written runtime ladders at `statistics.mojo:302-377` (22 arms) and two near-identical 13-arm ladders at `writer.mojo:122-151` and `:249-273` — the duplicated writer pair is the highest-value target. The reader's 28-arm comptime-gated ladder is deliberate. | M |
| **Q4.4** | `ipc.mojo` → package. Single file, 2,342 lines. | M |
| **Q4.6** | **Localised 2026-08-05; no Q4.7-style single lever exists.** Current `__text`: `query_scan` **2,367,336**, `query_scan_typed` **1,839,632**, `query_streaming` **1,332,456** — so the 1.78x ratio in the original card still holds. The cost splits almost evenly in two, which is the useful finding: **~507 KB is the Parquet machinery itself** (scan_typed over streaming — `parquet::reader` 32 symbols, `codecs` 9, `schema` 18, `format` 15, plus dtypes +21 and arrays +34), and **~528 KB is the untyped fanout** (scan over scan_typed — reader 32 -> 107 symbols, codecs 9 -> 32). The second half is a scan that does not know its schema instantiating every dtype's decode path, i.e. the 28-arm comptime-gated ladder CLAUDE.md calls deliberate. **Checked and ruled out:** `kernels::cast` is no longer reachable from these binaries at all after Q4.7, and `marrow/parquet/` imports it nowhere — so the single-import trick that took 55-65% off the hashing binaries does not apply. ~7 KB per parquet symbol says the weight is in large monomorphised decode bodies, not symbol count. Reducing it is a design change to how the reader dispatches, not a lever to pull. Candidate: make the typed path (schema known at comptime) avoid instantiating the arms it cannot reach. | L, and genuinely L |
| **Q4.7** | **RESOLVED, and it was far larger than the card estimated.** The card said `kernels::cast` is "roughly 20% of the fused binary". Measured: **55-65%**. `RapidHash.dispatch` decoded dictionary keys with `cast(keys, value_type)`, and that one call at `hashing.mojo:308` made the entire cast fanout reachable from every binary that hashes. But `DictionaryCast` (`cast.mojo:976-991`) *is* just normalise-indices + `take` + a re-cast that is a no-op when the target is the value type — which is the only way hashing ever called it. Doing the gather directly, with a narrow integer widening in place of the index cast, removes the import entirely. **query_join 3,860,660 -> 1,410,788 (-63.5%), agg_fused 3,786,228 -> 1,338,468 (-64.6%), agg 4,159,796 -> 1,858,644 (-55.3%)** — about 2.4 MB off each. `query_streaming` is unchanged, as it does not hash. Two tests added: the existing dictionary test used int32 indices, which take the fast path, so the widening branch had no coverage, and a null index had none either. | done |
| **Q5.1** | **RESOLVED.** Fixed: `TagValue` -> `DynValue` (3 sites in `relations.mojo`/`pruning.mojo`); `faszom.mojo` -> `marrow/expr/values.mojo`; `arithmetic.mojo` -> `numeric.mojo` (2 sites — the card listed neither); `marrow.bitmap` -> "this module"; the `elementiwise` typo; `LengthKernel.core`'s claim that `StringLength` builds on it, when `StringLength` is a breaker calling `dispatch` (now says so, and points at Q7.1); `expr/__init__`'s "one-way `relations` -> `execution`", which is a cycle in both directions; and `Sort`/`Limit`, now re-exported. **The README claim is stale** — `README.md` already describes the two lanes correctly, and its `Filter`/`Project` mentions are *plan nodes*, which exist. Follow-up left open: the orphaned docstring at `values.mojo:581` asserting `NumericColumn[DynType]` is the erased column leaf is a dangling string expression between `OutShape` and `var _name`, and removing it is a code edit rather than a comment fix — folded into Q7.2, which already covers a dead parameter in that file. | done |
| **Q6.1** | **RESOLVED, and the result contradicts the differentiator claim.** `marrow/expr/tests/bench_aggregate_aot.mojo` runs the AOT-resolved aggregate (`AggFunc.of[NumericAgg[K, V]]`) against the runtime-named one (`AggFunc("sum", …)`) on identical data, in **one binary** so the harness interleaves them. Measured 2026-08-05, 1M rows: at **g100k** fused 13.943 ms (sd 0.374) vs named 13.712 ms (sd 0.278); at **g1k** fused 11.219 ms (sd 0.240) vs named 11.312 ms (sd 0.163). **Under 2%, and the sign flips between cardinalities — there is no measurable runtime advantage.** That is explicable rather than surprising: an aggregate's identity is resolved once per plan execution, not once per row, so the interpretation cost is O(1) per query while the per-row work is the same `AggState` fold either way. **What the AOT path actually buys is binary size** (see the two gate binaries and Q4.7), not speed. Any doc or card asserting a *speed* differentiator for fused aggregates is unsupported and should say size instead. | done |
| **B27** | **FIXED — and the original filing had the direction backwards.** It was filed as "`s.len()` is 25x slower than `s.len() + 1`". That came from comparing pytest-benchmark rows scaled in **different units** (µs against ms) — the real relationship was the opposite: `s.len()` 69.3 **µs**, `s.len() + 1` 2.75 **ms**, so adding a fused op was ~40x *slower*. `LengthKernel` was never slow: 68.8 µs for 1M rows is ~116 GB/s. **Root cause:** `Context.get[A]` ends in `.copy()`, an atomic ref-count bump, and `vectorwise` calls it **once per SIMD chunk** — 250,000 atomics at 1M rows and width 4, to re-read one array that never changes. `as_type` was already a borrow; only the copy was added. **Fix:** `Context.get_ref[A]`, a borrowing read, used by all 16 typed slot reads in fused lanes. `s.len() + 1` 2.75 -> **1.32 ms**, `s.len() + s.len()` 5.34 -> **2.60 ms** — ~2x on every fused expression over a breaker, and every breaker-backed lane benefits (`StringPredicate`, `BoolReduce`, `NullPredicate`, `StringToNum`, `DateTrunc`, …). All four size gates unchanged. This is the same shape as A1's "per-SIMD-chunk schema lookup", so A1 remains the structural fix; this removes the atomic from the hot loop now. | done |
| **Q7.1** | **RESOLVED as not worth doing — measured 2026-08-05, after B27.** The card's framing is "`s.len() + a` is two passes, not one", which is true and turns out not to matter. Aligned units, 1M rows: `s.len()` (the materialise alone) **69.8 µs**; `s.len() + 1` **1.34 ms**; and the control — `a + 1` over a plain int32 column with **no breaker at all** — **1.99 ms**. The breaker version is *faster* than the breaker-free one, so the breaker is not the bottleneck: fusing `StringLength` could recover at most the 69.8 µs materialise, ~5% of the total, in exchange for changing `StringValue`'s trait surface, which the size gate watches. Not worth it. **What the measurement did find is bigger and is now B28**: a fused pass over 1M int32 doing one add costs ~2 ms, about 2 ns/element and roughly 6x off memory bandwidth — paid on every fused expression, breaker or not. That is the thing to fix, and it is A1's territory. | done, superseded by B28 |
| **Q7.5** | **RESOLVED, with three of its claims corrected.** Removed: the five `JOIN_ALGO_*` hints, `JOIN_ASOF`, `RAPID_SECRET0/3/4/5/6`, `comptime lo32`, `_promote_operands`, `std.math as math` and `sync_parallelize` from `filter.mojo`, `sync_parallelize` from `sort.mojo`, `trait Partitioner` (one conformer, no generic consumer) and `NoPartition` (no caller, and `map_partitions:321` unwraps `row_indices` unconditionally so it would have crashed). **Corrections to the card:** (a) `JOIN_SINGLE` and `JOIN_CROSS` are no longer dead — `JoinKind.is_supported`/`write_to` name them, so they stayed; (b) of the four dtype imports in `filter.mojo`, only `int64` and `string` were unused — `uint8` (10 uses) and `uint64` (7) are live, and `int64`'s other hit was `DType.int64`, a different name; (c) the "Temporal reinterpret helpers" banner was not merely mistitled — it documented a reinterpret-through-signed-integer-backing design that the single `is_primitive()` dispatch arm replaced, so it was deleted rather than retitled. Follow-up left open: with `NoPartition` gone, `Partition.row_indices` has one construction site that always supplies it, so the `Optional` and its default are now dead too. | done |
| **Q7.2** | **RESOLVED.** `NumericCompare`'s `S: StringPredicateKernel` parameter had **zero** `Self.S` references — dead since the erased arm went with `IsErased` — and is removed, along with the string-kernel argument from all six operator aliases (`Lt`/`Le`/`Gt`/`Ge`/`Eq`/`Ne`). The orphaned docstring at `values.mojo:581` went too: a dangling string between `OutShape` and `var _name`, asserting `NumericColumn[DynType]` is the erased column leaf, which stopped being true when `DynType` dropped its family conformances. | done |
| **Q-NEW** | **`marrow.expr` now has three import cycles**, introduced by the two-lane refactor: `values ⟷ dynamic`, `values ⟷ relations`, `relations ⟷ execution`. Mojo resolves them, so this is a layering question, not a build break — but `tasks-expr-kernels-layering.md` asserted "zero cycles" and that is now false. Decide whether to restore the DAG or to document the cycles as intended. | S to decide |
| **L2** | **ATTEMPTED 2026-08-05 and reverted — the blocker is the shared *name*, not the file.** `values.mojo` is 2,607 lines and the residue is real: the runtime-lane builders `col(String)`, `lit[T](Scalar)`, `if_else`, `coalesce`, `case_when` all return `DynValue` and sit at `values.mojo:2549` beneath the AOT `col[T]`/`lit[T]` overloads. Moving them to `dynamic.mojo` **compiles**, but `marrow/expr/__init__.mojo` must then re-export `col` and `lit` from two modules, and Mojo emits *"importing 'col' from multiple modules is deprecated; import 'col' from a single module"* — which violates the standing 0-warning rule. The lanes deliberately overload the same verb (`col("a")` runtime vs `col("a", int64)` fused), and an overload set cannot span modules. So the move costs either an API rename or dropping `marrow.expr.col("a")` from the public surface, for no functional gain. **Do not retry as a file move.** If `values.mojo`'s size is the actual concern, split it along a boundary that does not share a public name — the node families (numeric / bool / string / temporal) are candidates. Reverted cleanly; nine files were touched and all restored. | S to re-scope |
| **V0** | **IPC half RESOLVED; two smaller gaps remain.** `map` now writes and reads through IPC in both directions: type code 17, the `keysSorted` flag in the Map table, the entries field as the single child, and the child recursion on decode. The type-code arm goes **before** `is_list()` — a map is a list of entry structs, so `is_list()` answering first would have written it as a plain list and read it back as one. The buffer walk needed nothing: a map owns one offsets buffer like any list, which `DynType.num_buffers()` already answers (A3's minimal form). Four tests, and the one that matters is the **PyArrow cross-check**: a self-round-trip only proves marrow agrees with itself, and writing type code 12 for a map would have passed every other test. PyArrow reads it as `map<string, int64>` with the right entry counts. **Still open:** no `MapScalar` (a scalar taken from a `MapArray` reports `list<…>`), and `cast` has no map arm. | `arrays.mojo`/`scalars.mojo` (open) | S |
| **FU-5** | Fused `IsIn` under bool logic — `test_values.mojo:839` is still prefix-disabled. | S–M |
| **FU-6** | `sort_indices` without key re-gather. `SortIndices.multi` exists but there is no public `sort_indices(StructArray, key_indices)`; `SortProcessor` still calls `sort_by_keys` and discards key fields, and `sort.mojo:507` re-gathers keys on every pass. | M |
| **FU-7** | (a) `ConditionalBinary.validity` and `materialize` both call `_result`, running the kernel **twice** (`values.mojo:2072-2084`; same on `CaseWhen` `:2124-2142`) — **M**. (b) fused `IsIn._value_set: DynArray` survives the payload cleanup (`values.mojo:1756`) — S. (d) `ConditionalBinary` is 2-ary and `CaseWhen` is 1-branch and numeric-only, while the kernels and runtime builders are variadic — **L**. |  |

**Test-coverage gaps worth closing alongside the above:** `test_utils.mojo` is
**1 case / 12 lines** for all of `utils.mojo`; `test_tabular.mojo` is 9 cases for
~20 public methods; `test_boolean.mojo` is 11 cases for the entire Kleene
surface; `test_nested.mojo` 4, `test_rapidhash.mojo` 4, `test_partition.mojo` 6.
Python `test_dtypes.py` is 2 cases. No bench exists for `concat`, `conditional`,
`membership`, `nested`, `temporal`, `distinct`, `boolean`, `partition`, or IPC —
and per the standing constraint, an operator with no benchmark has no
performance. GPU `cast` and `boolean` paths have no GPU test.

**Spikes filed outside this page**, kept as their own documents because each is
a prototype proposal rather than a task, and both accurately describe their own
status:

- `todo/reflect-schema-from-struct.md` — derive an Arrow `Schema` from a Mojo
  struct via `reflect[T]`. Not started; purely additive. It is the foundation
  `docs/late-binding.md` needs, and the reason `Table[T]` is deferred (see
  CLAUDE.md, "Reflection, packs, and comptime aliases").
- `todo/warp-match-any-gpu-hash-join.md` — `warp.match_any()` for a GPU hash
  join. Speculative; its own precondition (a GPU hash join existing) is unmet,
  and GPU is outside the engine-first scope.

**Both-lane parity tests are missing entirely, and they are the only enforcement
invariant 2 has.** For each operation, a fused `Value` result and the equivalent
`DynValue` result must be asserted equal. The two-lane refactor removed the
structural reason the lanes diverged; nothing currently stops them diverging
again.

---

## 6. One-time: documentation disposition

Do this in a single commit. Seven task files, one roadmap and ten design docs are
replaced or deleted by this page.

**Delete** (superseded; git history keeps them):

| File | Why |
|---|---|
| `tasks-code-quality.md`, `tasks-execution-engine.md`, `tasks-expr-kernels-layering.md`, `tasks-expr-simplification.md`, `tasks-type-coverage.md`, `tasks-aggregate-followups.md`, `tasks-backlog-status.md` | replaced by this file |
| `dynamic-dispatch-design.md` | specifies fn-pointer vtables and a `DataTypeVisitor`; the tree uses inline `Variant` and there is no visitor module. ~0% implemented, actively misleading. |
| `aot-query-compilation.md` | thesis shipped, **every named artifact is wrong**, and its central construct (`Binary[op: UInt8]` + `comptime if op == ADD`) is now a measured anti-pattern in this repo (+45.7% `__text`). Salvage two items into M1.1: predicate normalization at construction, and an `InlineArray`-backed `Schema`. |
| `unified-plan-hierarchy.md` | its central mechanism — erase into a family trait via default type parameters — is exactly what `7d57398` proved **unsound** and deleted. `fn`/`alias` throughout. |
| `expr-unification-plan.md` | completed-migration record; every identifier, path, layout and measurement stale. |
| `ibis-fusion-design.md` | strict subset of `ibis-expression-design.md`; its opening premise ("nothing executes yet") is false; no inbound links. |
| `lane-shape-window-skeleton.mojo` | 401 lines prototyping **three rejected designs** (the `Fusable`/`Value` split, `MatBinary`, a fn-ptr `DynValue` box). A *runnable* artifact of a rejected design is a live trap. |
| `execution-engine-roadmap.md` | **Two living plans is the drift pattern this page exists to end** — it said so itself ("where the two disagree, `backlog.md` wins"), which is an admission that it could not be trusted alone. Everything load-bearing is folded in: the M1/M2/M3 definitions and their gates now head §3 and §4, the four invariants were already §0, the capability rows became §4 cards (M2.9–M2.13, M3.7, M3.8), and the **Won't** list is §7. Dropped: its §2 current-state inventory (`docs/architecture.md`'s job), its G1–G10 gap list (every gap is a card here), and its external reference-engine appendix. |
| `sort-design.md`, `joins-design.md`, `groupby-design.md`, `decimal-type-design.md` | Feature specs whose designs the tree then diverged from. Each mixed three things: what shipped (belongs in docstrings and `architecture.md`, not here), what is still wanted (now §4 and §7), and what was designed then replaced (now §8). Keeping them as "specs" meant four documents describing algorithms the code does not use — `joins-design.md`'s `JoinHashTable`/`_chain_next` and `groupby-design.md`'s seven groupers do not exist at all. |

**Rewrite:**

| File | Action |
|---|---|
| `aot-relations-design.md` | **Split.** Extract §"Erased relations over fused values" into a short living architecture doc — it is the charter of the current code, and both `README.md:60` and `benchmarks/binary_size/README.md:231` link here. Cut the first-slice record (`Table[T]`, `Project[*Es]`, `marrow/aot/`, all deleted) to a paragraph of surviving compiler findings. Decide `Env`/`Param` in or out. |
| `kernel-fusion-architecture.md` | Keep §1–§6 (the trait organization and "one core, two runners" thesis are load-bearing and validated); add the `Breaker`/`Context` staging model, which the doc predates entirely; demote §8–§10 to backlog items — no `Materialized` leaf adapter, `StringPredicate` still materializes a full `BoolArray`, `StringLength` is two passes not one, reductions still consume materialized arrays. |
| `lane-shape-window-design.md` | **Split.** Rewrite §1–§5 to describe what shipped (`Value`/`Breaker` polarity, `Datum`, `OutShape`, `Context` staging, fuse-above-breaker) — corrected, it becomes the only doc covering the current execution model. Keep §7 as a forward spec, retitled "Window functions — design (unimplemented)"; it feeds M2.3. Delete §8–§9: their stated targets (delete `prepare`, no `Context`) are the **opposite** of what the codebase decided. |
| `ibis-expression-design.md` | Reduce to a ~30-line record: the fusability taxonomy, the "bucket = which trait, never a runtime tag" rule, the `NumericValue`/`BoolValue` disjointness decision, and the fact that dual conditional conformance was probed and the per-family **fallback** shipped. As a spec it forbids editing the very file that implements it. |
| `aggregate-kernel-inversion.md` | Keep — the most valuable of the design docs; §5's "three predictions, three misses" and §4's erased-box cost rule are hard-won measured facts. Three-line correction: `resolve_agg` is in `expr/aggregates.mojo:194` not `dynamic.mojo`; the `AggFunction` catalog is in `expr/aggregates.mojo:84-191` while the trait is in `kernels/aggregate.mojo:846`; §6's gate commands (`check_lib`, `check <file>`, `test_parallel`) no longer exist. |
| `aot-jit-research-notes.md` | Keep as a **dated** record — the literature survey and the AOT-vs-JIT reasoning are the intellectual justification for the two-lane architecture and exist nowhere else. Header note: it describes `faszom.mojo` (→ `lane.mojo` → `values.mojo`), its reproduction command cannot run, and its 21× figure is superseded by 4.2×. |
| `tasks-step3-expression-nodes.md` | If kept at all, it needs a **superseded-by banner**: its entire design rationale was reversed four days later. It argues the two lanes *do* share one node set, that the box implementing every family trait "is what lets the node bounds stay as they are", and that "the bet is the erased instantiation never reaches `vectorwise`". The bet lost. |

**Keep unchanged:** `design-expression-evaluation.md` — written 2026-08-03, fully
accurate, its gate number matches the CHANGELOG, and it is the actionable
backlog for `values.mojo` internals (visitor-driven `traverse` to replace ~96
hand-recursions, explicit slot binding, CSE, parallel stage scheduling).
Pairs with L2.

**One dangling reference is left behind and is not this page's to fix:**
`marrow/kernels/join.mojo:105` points at `docs/joins-design.md`, and
`todo/reflect-schema-from-struct.md:118` and
`todo/warp-match-any-gpu-hash-join.md:65` cite `joins-design.md` /
`groupby-design.md` as models. Redirect all three to `docs/architecture.md` and
§8 below.

---

## 7. Deferred — Arrow parity

Listed once so the gaps are known. **Not scheduled**: nothing here blocks M1,
M2 or M3. Promote an item only when a milestone query needs it.

- **Layouts with zero implementation**: sparse union, dense union, run-end
  encoded, BinaryView/StringView, ListView/LargeListView, `large_map`. All three
  of C-Data import, C-Data export and IPC raise on them today, which is the
  correct behaviour.
- **Scalar fidelity**: six types have no dedicated scalar — `binary`,
  `large_binary`, `large_string` collapse to `StringScalar`; `large_list`, `map`,
  `fixed_size_list` collapse to `ListScalar`. A scalar taken from a `MapArray`
  reports `list<…>`.
- **Parquet**: encryption is completely absent (an encrypted file fails with a
  Thrift parse error, not a diagnostic); LZO missing; `fixed_size_list` cannot be
  written; a nullable struct containing a repeated group is silently demoted to
  REQUIRED; UUID/JSON/BSON/ENUM/INTERVAL logical types are silently downgraded on
  read; Arrow `dictionary`, `null` and interval columns cannot be written.
- **IPC**: no zero-copy read (the body is copied byte-by-byte into a `List[UInt8]`
  then again per buffer); writers buffer the whole file in RAM; endianness is
  hardcoded LE on write and never checked on read; the file writer omits the
  trailing EOS marker; buffer alignment is 8, not 64.
- **C-Data**: device-array *export* is not implemented (import-shaped struct
  only); `__arrow_c_device_array__` absent; stream `get_last_error` always returns
  null. **Correct CLAUDE.md's "Known Limitation #2" — release callbacks *are*
  implemented and invoked** (`c_data.mojo:221,834,820,1234` plus three PyCapsule
  destructors); the double-free guard is the spec's null-release handshake.
- **Kernels with no marrow equivalent**: all bitwise ops (`bit_wise_and/or/xor/not`,
  `shift_left/right`); `tan`/`asin`/`acos`/`atan`/`atan2`/hyperbolics/`cbrt`;
  `list_flatten`, `list_parent_indices`, `list_slice`, `list_element`,
  `make_struct`, `struct_field`, `map_lookup`; `replace_with_mask`, `choose`,
  `inverse_permutation`; `run_end_encode`/`decode`, `index_in`.
- **Python surface**: ~60 implemented kernels are unreachable from Python — the
  entire string family (18 incl. LIKE/ILIKE), the entire temporal family (9
  extractors + `date_trunc`), all boolean/validity kernels, 17 numeric kernels,
  all conditional kernels, `is_in`, `array_length`, `array_contains`, `concat`.
  Also missing: `ChunkedArray` as a type (imported but never registered),
  `Table.from_pydict`/`from_arrays`/`concat_tables`, `Array.cast`/`unique`/
  `value_counts`, all `Schema` manipulation (the Mojo methods exist at
  `schema.mojo:114-164`, none are bound), all numpy/pandas/buffer-protocol
  interop, and the `dictionary()`/`map_()`/`decimal*()`/`large_*()` type
  factories.
- **Group-by strategies designed but unbuilt**: `DirectMapGrouper`,
  `PackedKeyGrouper`, `RowEncodedGrouper` (+ `kernels/row.mojo`), sorted-input
  run detection. The shipped strategy set is a different axis entirely — see §8.
- **Sort features designed but unbuilt**: `SortOptions` with per-column
  `nulls_first` (today one `nulls_first: Bool` covers every key —
  `sort.mojo:449`), GPU radix, 4-byte string prefix comparison, scatter
  prefetch, parallel comparison sort, batch-level K-way merge.

### Won't — deliberately out of scope

Recorded so they stop being re-proposed. Each is a decision, not an oversight.

- **A SQL string parser or SQL frontend.** Both frontends are programmatic: the
  Python lazy API and the Mojo AOT DSL. Revisit only after M3.
- **A real ibis backend** (`ibis.backends.marrow`) or any `ibis` runtime
  dependency. ibis is a *naming* guideline and nothing more. A backend drags in
  ibis's op catalog, its type-coercion rules and its backend test contract; M1.3
  ships a thin native `Table`/`Column` instead.
- **A pandas-style eager `DataFrame` API.** The existing eager `RecordBatch`
  surface stays as it is.
- **Subquery decorrelation and outer-join elimination.** Both presuppose the SQL
  frontend above. This is also why `JOIN_MARK` and `JOIN_SINGLE` (M3.1) have no
  producer even once they are implemented — a planner emits them, users do not.
- **Distributed or multi-node execution, and server mode.** Local single-node
  only; that is the whole premise.
- **Run-end-encoded arrays and the View layouts** (`BinaryView`/`StringView`,
  `ListView`/`LargeListView`). Compression and performance optimizations with a
  large surface. The parity gaps are listed above; the work is not planned.

---

## 8. Rejected and replaced designs

Each row is a design that was written down, then **not** built — because the tree
built something different and better, or because the premise turned out to be
wrong. They are here so nobody re-litigates them from a stale document. Every
citation below was checked against the code, not copied from the design it
replaces.

### Sort

- **Permutation refinement (`getPermutation` / `updatePermutation` /
  `EqualRanges`) was never built.** Multi-key sort is a **column-oriented LSD**:
  `SortIndices.multi` (`sort.mojo:449`) stable-sorts by the *least*-significant
  key, then for each more-significant key gathers the column under the running
  permutation, sorts that, and composes (`sort.mojo:493-515`). There is no
  equal-range bookkeeping anywhere. *Why it stands:* one stable pass per key
  yields the same order without tracking ranges, and it reuses `Take` and the
  single-column sorters unchanged. *Its one cost is real*: the composition
  depends on every pass being stable — the `stable` flag was accepted and never
  forwarded, so this silently returned wrong orders until B1 was fixed by making
  the comparison path's comparator break ties on the original row index. It also
  re-gathers each key column per pass (**FU-6**).
- **8-bit radix passes / 256-bucket histograms were replaced by 11-bit passes /
  2048 buckets** (`_BITS_PER_PASS` `sort.mojo:76`; `bucket_count`, `:252`).
  *Why:* 6 passes instead of 8 for 64-bit keys with a histogram that still fits
  L1 per thread — measured ~6.7× at N=10M against 8-bit serial; 16-bit thrashes
  L1 (`sort.mojo:77-86`).
- **The comparison-vs-radix cutoff is not `N < 64`.** It is
  `_RADIX_THRESHOLD = 32_768` (`sort.mojo:59`). *Why:* PDQsort measured faster
  all the way to ~28K on int64; below the threshold the pair-buffer setup
  dominates. Decimal128/256 take the comparison path at any size, because their
  native width exceeds the UInt64 radix key.
- **`argsort` as a free function plus a `SortOptions` struct** became the
  `SortIndices` kernel struct (`sort.mojo:344`) with `dispatch`/`apply`/`multi`
  and loose parameters. Per-column `nulls_first` is still wanted — §7.

### Group-by

- **Of the seven designed groupers, exactly one shipped, and it is not one of
  them.** There is a single type-agnostic `HashGrouper` over a `SwissHashTable`
  (`groupby.mojo:43`). `DirectMapGrouper`, `PackedKeyGrouper`,
  `RowEncodedGrouper`, `TwoLevelGrouper` and `SpillingGrouper` do not exist, and
  neither does `kernels/row.mojo`. *Why:* the Swiss table's SIMD probing made
  per-key-type tables not worth their combinatorics; the design's own premise —
  "ClickHouse has 40+, so we need several" — did not survive contact with one
  table that handles every key type through `StructArray`.
- **The dispatch axis is inverted.** The design selects a grouper by **key
  type** (bool/uint8 → direct map, ≤16 fixed bytes → packed, else row-encoded).
  The shipped `_choose_strategy` (`groupby.mojo:402-412`) never looks at the key
  type at all: it selects on **row count and a sampled cardinality estimate**.
  Reading the design as a guide to the code inverts the whole decision.
- **`GROUP_THREAD_LOCAL` (`groupby.mojo:312`) is a strategy the design never
  proposed** — a DuckDB-style thread-local partial aggregation that splits by
  row range and merges partials, chosen for large *low*-cardinality inputs where
  radix cannot use more threads than there are distinct keys. Conversely
  ClickHouse's 256-bucket two-level merge, the design's headline recommendation,
  was not built; `GROUP_RADIX` uses 64 partitions (`RADIX_BITS = 6`,
  `groupby.mojo:304`).
- **The `group_id(keys) -> PrimitiveArray[uint32]` public API does not exist.**
  The entry point is the `GroupBy` struct (`groupby.mojo:321`) with
  `aggregate[A]` / `apply[F]` / `aggregate_columns`. The companion `unique` was
  never written and is still wanted — **M2.2**.

### Joins

- **`JoinHashTable` with an intrusive `_chain_next` collision list was never
  built.** Neither identifier appears anywhere in the tree. It was replaced by
  `SwissHashTable` plus a **CSR index** — `_offsets` / `_rows`, built after
  insertion and grouped by bucket (`hashtable.mojo:76-87`) — so a probe walks
  `[offset[bid], offset[bid+1])` contiguously (`:523-532`) instead of chasing
  pointers. *Why:* same ALL-strictness multi-match capability, sequential memory
  access, and one shared table type for join and group-by.
- **`IndexPairs` is not a struct.** It is
  `comptime IndexPairs = Tuple[Int32Array, Int32Array]` (`join.mojo:201`) — two
  Arrow arrays rather than two `List[Int32]`, which is what lets per-partition
  results merge by buffer memcpy.
- **The `PartitionedOp[T]` trait + `partition_apply` driver, recorded as
  "non-trivial in Mojo's generics, tracked as a follow-up", shipped** as
  `RadixPartitioner.map_partitions` (`partition.mojo:288`). It is done, not
  deferred; do not schedule it again.

### Decimal

- **The custom `Int256 { low: UInt128, high: Int128 }` struct is unnecessary and
  was never written.** Its premise — "Mojo has no native `DType.int256`" — is
  false: `Decimal256Type = _DecimalType[DType.int256]` (`dtypes.mojo:252`), and
  `Decimal128Type` likewise uses `DType.int128` (`:251`). `struct Int256` has
  zero occurrences.
- **Decimal *arithmetic* was never built at all** — not rejected, just never
  started. It is **M3.7**, and the design's result-type rules are the part worth
  keeping.

---

## 9. Architectural debt

From a three-package responsibility audit (2026-08-03), applying CLAUDE.md's
own method: name each type's single responsibility; where one cannot be named,
that is a leaky abstraction; dependencies must form a one-directional tree.

**What the audit confirmed as sound, and should be defended rather than
"improved":** trait-derived family dispatch (`comptime if conforms_to` derives
the nine `dispatch_*` families from the trait hierarchy, so adding a dtype
extends every family it conforms to at once); peer erasure (`DynArray: Array`,
so generic code takes either and there is no parallel erased overload set);
parametric mutability on `Buffer`/`Bitmap` with a `comptime assert` making a
mutable copy a *compile* error; `DevicePassable` views with logical zero-based
indexing, which is what lets one kernel body serve CPU-serial, CPU-parallel and
GPU; kernel-parameterized generic nodes (10 structs, ~45 operators);
`Relation`-pure vs `Processor`-mutable, verified unviolated across all six
pairs; `Breaker` as marker-by-conformance; and `BoxedValue` as an erasure
boundary that is also the fusion boundary. `marrow/kernels` is a **verified DAG
with no up-edges into `expr` or `tabular`**.

### The highest-leverage fixes

A2, A4 and A6 are done. A2 landed as B12 (eager recount at the slice
boundary); A4 moved `ExecContext` to `marrow/execution.mojo`, so `marrow/kernels`
no longer has an inbound edge from core; A6 added `worth_parallel`/`with_threads`
and closed six sites where a GPU device was silently dropped.

A1 and A3 remain, and they are not comparable. A1 is specced and spiked. A3 was
researched on 2026-08-05 and came back **smaller than written** — see its row.

| ID | Fix | What it removes |
|---|---|---|
| **A1** | **Replace the positional `Context` with a typed `comptime State` per node**, returned by `prepare` and passed into `vectorwise`. **Design validated by spike 2026-08-03** — see `docs/design-expression-evaluation.md`, which carries the protocol, the spike results, the sequencing and the gates. | The `Context` correctness hazard below; six methods and the `Breaker` marker trait collapse to two methods; ~84 hand-written recursion bodies (17 shapes); the per-SIMD-chunk schema lookup in `NumericColumn.vectorwise`; the double kernel run in `ConditionalBinary`/`CaseWhen` and the subtree re-execution in `BoolBinary.validity`; and — once validity moves into the state — B14, B15 and the class B20 belongs to. **First gate is binary size**, not tests: convert three nodes, hold `query_streaming` `__text` at **1,309,032** (live reading 2026-08-05; the 1,302,900 this used to quote predates B12), stop if it regresses. |
| **A3** | **Give `DataType` a `layout()`.** **Researched against both references 2026-08-05, and it cuts the original claim down — read this before scheduling it.** Both have exactly this type. Arrow C++: `struct DataTypeLayout { vector<BufferSpec> buffers; bool has_dictionary; optional<BufferSpec> variadic_spec; }` with `BufferKind {FIXED_WIDTH, VARIABLE_WIDTH, BITMAP, ALWAYS_NULL}`, reached through a **pure virtual `layout()` on `DataType`** that each concrete type overrides (`type.h:93-178`); marked EXPERIMENTAL. arrow-rs: the same shape as a **free `fn layout(&DataType)`** — one `match` over the enum (`arrow-data/src/data.rs:1787`), explicitly ported from C++ with the source commit linked, plus an `alignment` field on `FixedWidth` that C++ lacks, and `Dictionary` delegating to its key type's layout. **The correction: neither reference drives its serializers from it.** `layout()` in Arrow C++ is consumed by `array/data.cc`, `array/validate.cc`, `compute/exec.cc` and `extension_type.*` — **zero hits under `ipc/` or `c/`**. In arrow-rs there is exactly one IPC use, and it is narrow: `get_or_truncate_buffer` reads `layout.buffers[0]` for an element width when slicing (`arrow-ipc/src/writer.rs:2346`); `ffi.rs` does not use it at all. So this card's "removes 17 dtype ladders across `c_data` + `ipc`" does **not** follow from the design both references chose, and the `map`-absent-from-IPC claim is overstated with it: IPC additionally needs type code 17 in its type-writing and type-reading switches, which a buffer-layout description does not supply. A codec needs more than buffer structure — which flatbuffer table to write, which child to recurse into, what parameters (precision/scale, list size, timezone) to emit. **What it does buy**, on both references' evidence: validation (does this `ArrayData` have the right buffer count, kinds and widths?) and the `ArrayData` round-trip — which is the *other* half of the audit finding, the 11 array types that are each their own `ArrayData` codec, 22 methods of duplicated layout knowledge. Schedule it for that, not for the ladders. marrow's shape suits either spelling: it has a `DataType` trait with concrete structs (C++'s form) *and* a `DynType` variant (arrow-rs's form). | S-M for validation + `ArrayData`; the ladder collapse is **not** on offer |

### The `Context` positional-slot invariant

**The single most dangerous thing in `marrow/expr`.** Correctness requires that
`prepare` (`values.mojo:334-339`, appending breaker results to `Context._slots`)
and `vectorwise` (threading `mut slot: Int`, each breaker consuming one) walk
the tree in **identical DFS order, per node, by hand**. `NumericBinary` does it
right — `prepare` l-then-r at `:710-712`, `vectorwise` l-then-r at `:697-703` —
and nothing enforces it.

Counts: **42 structs (37 of them value nodes), 29 with `vectorwise`, `prepare`
at 15 sites** (the trait default plus 14 node overrides).

Three failure modes, and the one that matters is silent:

1. **Order swapped, slot types identical** → the tree compiles, both
   `ctx.get[…]` calls succeed, and the query returns **the wrong column with no
   error**. `coalesce(a,b) + nullif(a,b)` is a two-int64-slot instance.
2. Order swapped, slot types differ → the `Variant` accessor trips at run time.
   Loud, but by luck of the operand types, not by design.
3. A non-`Breaker` composite forgets to override `prepare` → no slot is
   appended, `vectorwise` still increments → out-of-bounds. **`DateTrunc`
   (`values.mojo:2271`) is exactly this case today** — latently wrong, currently
   unreachable only because no temporal breaker exists to sit under it.

There is no compile-time signal for any of the three; slot consumption is a
property of a *traversal*, not of a type. Breakers are insulated only by
accident: `materialize` runs its operand through the single-argument `execute`
(`:329-332`), which allocates a fresh `Context` — but the two-argument overload
is equally in scope, equally callable, and would corrupt the numbering.

**Cheapest mitigation, pending A1:** a debug-only `ASSERT` that `prepare`
appended exactly as many slots as the root's `vectorwise` consumed. That turns
mode 1 from silent into loud.

### Both-lane parity is a hand-maintained list, and it has already drifted

Invariant 2 ("no feature in only one lane") is enforced by `test_parity.mojo`
alone: 47 cases, of which **39 assert both lanes** and 13 pin one lane against a
literal. Ops with a genuine cross-lane assertion: **19 of the 55 shared ops
(35%)**. No parity for `** <= >= != ^`, any of `abs sign floor ceil round sqrt
exp ln`, the seven string maps, `length`, `startswith`/`contains`/`like`, eight
of nine temporal extracts, `date_trunc`, or any aggregate. The file names the
hole itself — it is how the Q0.4 divergence survived.

**The op-name strings have already diverged**: `mod`/`modulo`, `pow_`/`power`,
`neg`/`negate`, `and_`/`and`, `or_`/`or`, `not_`/`not`, `abs_`/`abs`,
`log`/`ln`. `render()` therefore differs across lanes for the same expression —
and more seriously, **`prune` correctness keys on these strings against two
hand-maintained literal sets** (`values.mojo:1043-1046` vs
`dynamic.mojo:592-601`). A typo falls through to `PruneBound.unknown()`, which
is *sound*, so it fails as silently-disabled row-group skipping, never as an
error. The two lanes also disagree structurally: `prune` and `bound_column`
exist on `NumericColumn` only, so in the AOT lane a string column cannot be a
join key and a string predicate prunes nothing — while the runtime lane keys on
`_tag == "column"` regardless of dtype and does prune it.

**A5 — enumerate the op set and assert parity over it mechanically**, rather
than by hand-written case. The model already exists in the same file:
`test_numeric_rank_agrees_across_lanes` loops all 11 numeric dtypes through
`dispatch_numeric`. That is the right shape, applied to the wrong axis.

### Types carrying a second and third responsibility

Each of these is a leak, not a bug; they are what make the surrounding code hard
to change. Listed with the competing responsibilities, most costly first.

- **`SwissHashTable`** (`hashtable.mojo:42`) — slot management, CSR row index,
  hashing, **and key-equality verification**, which is why a hash table imports
  `EqKernel`, `Take` and `Filter`. It also has an **unenforced build→probe
  lifecycle**: `_offsets`/`_rows` are `alloc_uninit(0)` until `build_hashes`
  runs, and `probe_hashes` reads them unconditionally. `insert()` then `probe()`
  is an out-of-bounds read on a zero-length buffer, guarded only by prose. The
  two clients happen not to mix them, which is why it has never fired.
- **`DynRelation`** (`relations.mojo:291`) — the erasure box **and the entire
  plan builder/binder** (`:406-736`: schema derivation, dtype probing, join
  name-collision suffixing, top-K folding, predicate pushdown). "There is no
  `Planner`" is true of *dispatch*; the planner exists, and it is fused into the
  box.
- **`RecordBatch`/`Table`** (`tabular.mojo:32`, `:444`) — container **and query
  surface** (`join`, `group_by`, `aggregate`, `sort_by`), which is what creates
  the `core → expr → core` cycle via `FoldedAggregates`. `join` even parses
  PyArrow's `how` strings into `JOIN_*` inline. Neither constructor validates
  anything: no check that `len(columns) == len(schema.fields)`, that column *i*
  has field *i*'s dtype, or that columns are equal-length.
- **`HashJoin`** (`join.mojo:309`) — algorithm, strategy choice, **output schema
  construction with `_right` collision renaming**, and materialization.
  Relational naming policy inside a kernel.
- **`SortIndices`** (`sort.mojo:344`) — dispatch, algorithm selection, null
  placement (with the policy re-inlined twice more at `:649-671` and `:730-738`),
  multi-key LSD composition, and top-K truncation.
- **11 array types are also their own `ArrayData` codec** (`__init__(data)` +
  `to_data()`), 22 methods of duplicated layout knowledge. **This — not the
  codec ladders — is what A3 addresses**; both references use `layout()` for
  `ArrayData` construction and validation, and neither drives IPC or the C ABI
  from it.
- **`Value`** (`values.mojo:304`) is the union of four consumers' protocols —
  executor, printer, optimizer, pruner — and all 37 nodes pay for all four.

### `ExecContext`'s vocabulary is one concept short — **A6, done**

Closed. `worth_parallel(n, min)` is the missing predicate and `with_threads(n)`
the missing derivation; every hand-rolled copy and every `num_threads: Int`
boundary is gone. Two counts in the original card were low: the predicate had
**four** copies, not three (`HashJoin.probe` was the fourth, and it *must* agree
with `build`), and **five** API boundaries destructured to `Int`, not two —
`tabular.join`/`group_by`/`sort_by` as well as `Aggregation.whole` and
`GroupBy`. The one live defect was in the bindings: `compute.mojo` had a
Python-supplied `ExecContext` and passed `ctx.resolved_num_threads()`, so
`ma.compute.sum` on a GPU context ran on the CPU.

### The three `marrow.expr` import cycles

`values ⟷ dynamic`, `values ⟷ relations`, `relations ⟷ execution`. All three are
accidents of file placement, and **one move breaks all three**: extract `Value`,
`Breaker`, `Context`, `Datum`, `into_array`, `_union_columns` and `BoxedValue`
into `marrow/expr/core.mojo`; move `AggExpr` to `aggregates.mojo`; move the five
untyped builders (`col(String)`, `lit`, `if_else`, `coalesce`, `case_when`,
`values.mojo:2476-2513`) to `dynamic.mojo`. Result:

```
core        -> (kernels only)
pruning     -> (leaf)
values      -> core, pruning, kernels     # no longer imports dynamic or relations
dynamic     -> core, pruning, kernels
aggregates  -> core, kernels
execution   -> core, aggregates, pruning
relations   -> core, dynamic, aggregates, execution
```

`values.mojo` — the 2,534-line file — becomes a **leaf**. Today, touching
`relations.mojo` invalidates the AOT lane. This subsumes **L2**, whose stated
definition of done was written before the two-lane split and is unachievable as
worded. Note `execution.mojo:16-18` currently *denies* the cycle it participates
in, and names the type `DynValue` rather than `BoxedValue`.

Two smaller, genuinely free ones: `views → buffers` exists only because
`BufferView.filter` and `BitmapView.to_owned` *allocate* — they are kernels
living on a view type. `arrays → builders` is three convenience constructors;
`arrays.mojo:2085` already imports `DynBuilder` function-locally, so the author
knew.
