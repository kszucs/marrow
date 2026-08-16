# Backlog

The single source of truth for what is open. Verified against the code at
**`b2e7dae` (2026-08-03)**, re-verified item by item as each was worked, and
**pruned of everything resolved on 2026-08-16** — not read off a header.

**Goal: marrow is a usable single-node columnar query engine.** Arrow-spec
completeness is not the objective — layout and kernel gaps are scheduled only
when a milestone query needs them. The deferred Arrow-parity list is §6.

Resolved items are **deleted, not struck through** — git history and the
CHANGELOG have them. What a resolved item taught survives only where it changes
a future decision: a trap that invalidates an obvious approach goes to §0, a
design that was written down and then not built goes to §7.

> **Re-verify before trusting any status line here.** This file replaces seven
> task documents that had drifted so far apart that a 2026-07-30 consolidation
> pass still left the index and `tasks-code-quality.md` disagreeing about five
> tasks. A 2026-08-03 audit found **18 wrong statuses**: eight tasks marked open
> that were done, four marked done that were not, and six whose premise the
> two-lane refactor had destroyed. Check with `grep`, not with a header — and
> exclude `.claude/worktrees/`, which holds stale pre-refactor code.

---

## 0. Standing constraints

Each of these cost real time to find and invalidates an approach that looks
obvious. Read before planning anything.

### Architectural invariants (gate every merge)

1. **Small-binary DCE.** Preserve the closed-erasure property: no open
   dispatchers, fused-only value boxes, closed per-dtype kernels. Gate on
   `pixi run binary_size`.
2. **One engine, two drivers.** No feature may exist in only one lane. Windows
   currently violate this (AOT-only) — see M2.3. Enforced by
   `marrow/expr/tests/test_parity.mojo` across four axes — op names read off the
   kernel, pruning, values, aggregates through a keyless plan — so a new op adds
   its case there.
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
- **`views._reduce_dispatch` staying on `sync_parallelize` is deliberate, not an
  unconverted leftover** — it is the fourth documented one, and its comment says
  so. Routing it through `ctx.stripe` forces the serial arm to allocate a
  one-slot partials buffer it does not need: `sumint64_1k` 0.19-0.20 → 0.30-0.32
  µs, `sumfloat64_1k` 0.23-0.24 → 0.34-0.37 µs (five interleaved repeats,
  disjoint ranges). A serial fold should not pay for the merge's scratch.
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
- **`.claude/worktrees/` contains two stale worktrees** (`docs-revamp`, `q25`) holding pre-Q2.5 `AGG_*` and `reinterpret_array` code.
  Exclude them from every grep or you will get false positives.

---

## 0.5 Priority — MoSCoW against M1

**The frame matters more than the labels.** These are prioritised against
M1's own acceptance criteria, which the milestone section states as three
things, not one: *results cross-checked against DuckDB*, *wall-clock
competitive-or-better against polars and duckdb on the same box*, and *the
binary-size gate green*. "Must" means M1 cannot be claimed without it. Anything
whose absence would not stop that claim is Should or below, however appealing.

### Must — M1 cannot ship without these

| ID | Why it blocks M1 |
|---|---|
| **M1.1** Optimizer v1 | Head of a strictly sequential chain; nothing below it can start |
| **M1.2** Python lazy bindings | No plan or expression type is bound today |
| **M1.3** ibis-flavoured `Table`/`Column` | Blocked on M1.2 |
| **M1.4** Kernel gaps M1 needs | Named by the 43 queries |
| **M1.5** ClickBench through the lazy plan | *Is* the milestone |
| **Size-gate resolution** | Acceptance says "gate green". `query_streaming_agg_fused` is at +0.449% of 0.5% and the next aggregate change will not fit. Shrink or re-baseline **deliberately** |
| **Push, and run Benchmarks + Wheels** | Acceptance includes a *measured* wall-clock comparison and a green gate. Both currently rest on one unpushed laptop, ~580 commits ahead of `origin/main`. Those two jobs have never run at all |

### Should — real value, M1 does not block on it

| ID | Note |
|---|---|
| **V0** `MapScalar` | User-visible wrong type. Blocked only by the size gate, so it lands free once that is resolved |
| **M2.11** `CoalesceBatches` | ClickBench is filter-heavy; morsel compaction feeds the wall-clock criterion |
| **M3.4** O(N) top-K | Most queries are `GROUP BY … ORDER BY … LIMIT`. Grouped output is small, so this is not a blocker — but it is the obvious next perf lever |
| **M3.3** `JoinProcessor` drops the exec context | S, and a silent correctness-of-configuration bug |
| **FU-6** `sort_indices(StructArray, keys)` | Removes a re-gather |
| **Q4.4** `ipc.mojo` → package | 2,342 lines |
| **Q-NEW** three `marrow.expr` import cycles | A decision, then possibly nothing |
| **L2** split `values.mojo` | Re-scope first; the blocker was the shared name |

### Could — genuine, nothing depends on it

`M2.1` Distinct/Union · `M2.2` unique/value_counts · `M2.4` statistical aggregates ·
`M2.13` EXPLAIN · `M3.6` Table/ChunkedArray depth · `M3.7` decimal arithmetic ·
`Q2.5` aggregates (a size play, and the size gate is the thing that is blocked) ·
`Q4.6` Parquet untyped fanout · `Q0.5` · `FU-7(b)` · `FU-7(d)`

### Won't — this cycle, with the reason

| ID | Reason |
|---|---|
| **B4** | Needs a hand-assembled BIT_PACKED file; no reference writer emits them |
| **B25** | Metal backend crash, and every CI job passes `--no-gpu`, so it is unobserved either way |
| ASAN on Linux | `test.yml if: false`; the macOS job now carries the signal |
| `interval` YEAR_MONTH/DAY_TIME in archery | pyarrow has no type for either unit and the harness bridges through pyarrow |
| **M2.5** spill · **M3.1/M3.2** join completeness/reordering · **M3.8** late materialization | Post-M1 by construction — M2 and M3 milestones |

---

## 1. Wave 1 — Correctness

Defects that produce **wrong answers with no error**. These come first because
the M1 gate is "results cross-checked against DuckDB": a wrong multi-key sort or
an unpruned date predicate corrupts exactly the thing the milestone measures.

**Every item opens with a failing test.** A card found by reading a code path is
unverified until that test exists — write it and watch it fail before touching
production code, because more than one card here turned out to behave
differently from how it read.

**The wave is nearly clear: what is left is one blocked fixture and one backend
crash.** Two things it cleared are worth keeping in mind, because neither is
structurally prevented:

- **`offset` has two conventions** — views index logically from zero, owning
  `Buffer`/`Bitmap` do not — and array code mixes them. Every known instance is
  fixed; the convention split is not, so a new array method that indexes a
  bitmap can still pick the wrong one.
- **A bounds assert kills the whole test binary, not one case**, so one bad card
  can mask several others. That is how B20 hid B14 and B15.

| ID | Defect | Evidence | Size |
|---|---|---|---|
| **B4** | **BIT_PACKED Parquet levels are mis-decoded.** `definition_level_encoding` / `repetition_level_encoding` are parsed (`format.mojo:576-578`) then never consulted — `_data_page_v1` (`reader.mojo:240-270`) applies `Rle.decode` unconditionally, and also reads RLE's 4-byte length prefix, which BIT_PACKED pages do not have. **Blocked on a fixture, not on the fix.** The guard is a few lines, but BIT_PACKED is deprecated: arrow-cpp never writes it and PyArrow exposes no option for it, so there is no reference writer to produce a test file. Doing this means hand-assembling a Parquet file with Thrift page headers declaring BIT_PACKED. Do not land the guard untested — a decode path that silently changes behaviour is exactly what this card is about. | as cited | S fix, M fixture |
| **B25** | **`test_views_gpu.mojo` fails 13 of 15 with a Metal codegen error**, and has for at least the whole of this branch. Every failure is `error: Metal Compiler failed to compile metallib. Please submit a bug report.` followed by `mojo: error: failed to run the pass manager` — a backend crash, not an assertion. Verified 2026-08-05 at both `d565c1a` and `3c747c0` (pre-block), so nothing in the Q7.4/Q7.3/B8 work caused it. `test_buffers_gpu.mojo` (4/4) and `test_arrays_gpu.mojo` (4/4) pass, so the accelerator itself works — it is these kernels. Compiling all three GPU files as one selection fails the same way, so it is not a combined-unit effect either. **This invalidates an earlier claim in this repo's history that `test_views_gpu` passes 15/15.** Needs: minimise which of the 13 cases trips the backend, then either work around it or file upstream. | `marrow/tests/test_views_gpu.mojo`; `views.mojo` `_apply_dispatch`/`reduce` | M |

---

## 2. Wave 2 — Infrastructure

**CI has not run since 2026-05-11**, and on that last run everything except Lint
was already failing. Nothing here is verified by anything but local runs. This
wave is cheap and it is what makes every later wave believable.

**Every job was run locally on 2026-08-16** — the first time any of them has
been exercised since 2026-05-11, and still with nothing pushed. Results:

| job | local result |
|---|---|
| Lint | **was red**; nine files had format drift. Fixed (`0bf29da`) |
| Tests (macos, arm64) | green, 2034 passed |
| Docs | green, 15/15 pages — surfaced five compiler warnings in `python/bindings`, fixed |
| Binary size | green, but see the standing note below |
| ASAN (macos, arm64) | **could not pass as configured**; re-scoped to `test_asan_core` |
| Integration | green — `map`, `map_non_canonical` and `interval_mdn` were skipped under a false reason, un-skipped, and score **14/14** (`interval` YEAR_MONTH/DAY_TIME stays skipped: a pyarrow limit) |
| Benchmarks, Wheels | **still never run** |

**Still true:** GPU is never exercised (every job passes `--no-gpu`; five
`*_gpu.mojo` files, 39 cases). ASAN on Linux is hard-disabled
(`test.yml if: false`). `precompile` — the warning-clean gate — is not in CI.
There is no separate `test_python` job. And **nothing has been pushed**:
`backlog-wave1` is ~580 commits ahead of `origin/main` with no upstream, so all
of the above is local evidence only.

> **Standing constraint, and it is now binding.** `query_streaming_agg_fused`
> sits at **+0.449%** against the gate's 0.5% ceiling, up from 0.000% at the
> start of this branch: the A1 state refactor took it to +0.416%, the `Grouping`
> parameter to +0.449%. V0's `MapScalar` half measured +0.137% more — 0.586%,
> over the line — and was reverted for that reason alone. **The next change touching
> the aggregate or compare/bool nodes will not fit.** Either shrink something
> or re-baseline deliberately; do not discover this by having a gate fail.

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

### M1.4 — Kernel gaps M1 actually needs — **open, scope from the queries**

The five capabilities the 43 queries promoted out of M2 have all landed
(min/max on string and date, `count_distinct` as a relational aggregate,
`HAVING`, computed group keys and aggregate inputs, `date_trunc` down to
month/quarter/year). What remains on this card is whatever M1.5 turns up when
the queries are actually run: re-read the 43 against the kernel surface as part
of that work and schedule the gaps here rather than guessing them now. Q29
`REGEXP_REPLACE` is deferred to M2 by definition and is not one of them.

### M1.5 — ClickBench through the lazy plan — **M, the M1 sign-off**

Rewrite `clickbench.py` against the lazy frontend, all 42 queries, cross-checked
against DuckDB, wall-clock compared to polars/duckdb on the same box, binary-size
gate green.

### M1.6 — AOT DSL docs — **S, one runnable example**

`docs/guide/expressions.qmd`'s prose is correct as of 2026-08-05; what it still
lacks is a **runnable** AOT example. Its blocks are illustrative (plain
```` ```python ````, not executed), which is why they could name types that did
not exist and the docs build stayed green — an executed example is the only
thing that keeps this page honest.


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
| **M3.7** | **Decimal arithmetic.** Nothing was built: no `add`/`sub`/`mul`/`div`, no scale-alignment or result-precision rule, no 256-bit intermediate promotion. Rescale exists only inside `cast` (`cast.mojo:855-869`) and its up-scale multiply has **no overflow check**; precision is never validated, so `decimal32(40, 0)` constructs. TPC-H money math needs it, which is why it is here and not in §6. The result-type rules (`result_scale = max(s1, s2)`, `result_precision = max(p1-s1, p2-s2) + result_scale + 1`) and the scalar-loop constraint — there are no 128-bit SIMD lanes, so `vectorize`/`elementwise` does not apply — are stated here because they were the only parts of the deleted design doc worth carrying forward. | not started | M |
| **M3.8** | **Late materialization / selection vectors on the scan** — decode the filter columns, filter, then decode only the survivors. DataFusion ships the equivalent *off* by default because it is subtle; sequence it after M2.8. Async/prefetch ranged reads for remote scans (M2.12) belong with it — the OpenDAL C ABI is blocking, so it needs several readers rather than one. | not started | L |

---

## 5. Quality debt

Surviving items from the Q/L/V backlogs, with their original IDs kept so git
history and CHANGELOG references still resolve. Everything else in those files
is done and has been deleted.

| ID | What is left | Size |
|---|---|---|
| **Q0.5** | **ATTEMPTED 2026-08-05 and blocked on a real limit — read this before retrying.** The plan was to give `BoxedValue` a `dtype_hint()`: `Some` for a fused node, `None` for the runtime lane, so the 0-row probe only runs where the answer genuinely is not known. Telling the lanes apart is easy — `DynValue` is concrete, so `__init__(out self, value: DynValue)` is a more specific overload than the generic `__init__[V: Value]`, and that works. **The blocker is that a fused node knows its output dtype's *type* but not its *parameters*.** `V.OutType` is a comptime type; the dtype *value* for `decimal128(p, s)`, `timestamp(unit, tz)` or `fixed_size_binary(w)` carries fields the type does not, and nodes do not store a dtype instance — `NumericLiteral[T]` holds only `_value: Scalar[T.native]`. So `DynType(V.OutType())` fails to compile for parameterised dtypes and would silently report `decimal128(0, 0)` if it did. **This is also a second reason the probe exists**, alongside "keeps a caller-supplied schema honest": executing is how a parameterised dtype gets its parameters. Retrying means either giving all 37 nodes a dtype instance — which grows the fused structs, and they are size-gated — or restricting the hint to parameter-free dtypes and keeping the probe for the rest. Neither is obviously worth the claimed ~16 KB. | M, and less attractive than it looked |
| **Q2.5** | Aggregates, remaining steps: **2b** — `AggState[K, V: NumericType]` was never widened to `PrimitiveType`, so temporal reductions do not work natively; **3** — `count_distinct`/`approx_count_distinct` (+`_grouped`) are still four free functions (`distinct.mojo:87,133,167,214`); **4** — `FusedAggregation` (single pass, AoS accumulator, comptime offsets, zero dispatch) has zero occurrences. **Pitch it as a size play, not a speed one.** Q6.1 measured the AOT-resolved aggregate against the runtime-named one in one binary (1M rows: g100k 13.943 ms vs 13.712 ms, g1k 11.219 vs 11.312) — under 2%, sign flipping between cardinalities, which is what you expect when an aggregate's identity resolves once per plan and the per-row fold is the same `AggState` either way. Validate any round of this against `bench_aggregate_aot.mojo` at `g100k`, never `g10`, and be prepared for the answer to be "no change". | L |
| **Q4.4** | `ipc.mojo` → package. Single file, 2,425 lines. | M |
| **Q4.6** | **Localised 2026-08-05; no single lever exists here.** Current `__text`: `query_scan` **2,367,336**, `query_scan_typed` **1,839,632**, `query_streaming` **1,332,456** — so the 1.78x ratio in the original card still holds. The cost splits almost evenly in two, which is the useful finding: **~507 KB is the Parquet machinery itself** (scan_typed over streaming — `parquet::reader` 32 symbols, `codecs` 9, `schema` 18, `format` 15, plus dtypes +21 and arrays +34), and **~528 KB is the untyped fanout** (scan over scan_typed — reader 32 -> 107 symbols, codecs 9 -> 32). The second half is a scan that does not know its schema instantiating every dtype's decode path, i.e. the 28-arm comptime-gated ladder CLAUDE.md calls deliberate. **Checked and ruled out:** `kernels::cast` is no longer reachable from these binaries at all, and `marrow/parquet/` imports it nowhere — so the single-import trick that took 55-65% off the hashing binaries (removing one `cast` call from `RapidHash.dispatch`) does not apply here. ~7 KB per parquet symbol says the weight is in large monomorphised decode bodies, not symbol count. Reducing it is a design change to how the reader dispatches, not a lever to pull. Candidate: make the typed path (schema known at comptime) avoid instantiating the arms it cannot reach. | L, and genuinely L |
| **Q-NEW** | **`marrow.expr` now has three import cycles**, introduced by the two-lane refactor: `values ⟷ dynamic`, `values ⟷ relations`, `relations ⟷ execution`. Mojo resolves them, so this is a layering question, not a build break. Decide whether to restore the DAG — §8 has the one move that breaks all three — or to document the cycles as intended. | S to decide |
| **L2** | **ATTEMPTED 2026-08-05 and reverted — the blocker is the shared *name*, not the file.** `values.mojo` is 2,765 lines and the residue is real: the runtime-lane builders `col(String)`, `lit[T](Scalar)`, `if_else`, `coalesce`, `case_when` all return `DynValue` and sit at `values.mojo:2549` beneath the AOT `col[T]`/`lit[T]` overloads. Moving them to `dynamic.mojo` **compiles**, but `marrow/expr/__init__.mojo` must then re-export `col` and `lit` from two modules, and Mojo emits *"importing 'col' from multiple modules is deprecated; import 'col' from a single module"* — which violates the standing 0-warning rule. The lanes deliberately overload the same verb (`col("a")` runtime vs `col("a", int64)` fused), and an overload set cannot span modules. So the move costs either an API rename or dropping `marrow.expr.col("a")` from the public surface, for no functional gain. **Do not retry as a file move.** If `values.mojo`'s size is the actual concern, split it along a boundary that does not share a public name — the node families (numeric / bool / string / temporal) are candidates. Reverted cleanly; nine files were touched and all restored. | S to re-scope |
| **V0** | **`MapScalar` — BLOCKED on the size gate.** (The `cast` map arm, V0's other half, landed 2026-08-16 and was free in binary size.) The right fix is not a new scalar type but making `ListScalar` carry its own dtype instead of rebuilding `list_(child.dtype())` — that reconstruction reports the *shape*, so a `map` element answers `list<struct<key, value>>` and a `large_list` element answers `list<…>`; one field fixes all three. It works and is tested, but **a `DynType` field on `ListScalar` costs +0.137% on `query_streaming_agg_fused`, taking it from +0.449% to +0.586% and past the 0.5% ceiling** (measured 2026-08-16 by reverting that half alone). Reverted, not landed. Two ways forward: a 1-byte kind tag rebuilding `list_`/`large_list_`/`map_` from the child — cheaper, but loses `keysSorted` and the entries field name — or re-baselining the gate deliberately. That gate has been the binding constraint since the `Grouping` change and this is the second change to hit it. `scalars.mojo`. | S, gated |
| **FU-6** | `sort_indices` without key re-gather. `SortIndices.multi` exists but there is no public `sort_indices(StructArray, key_indices)`; `SortProcessor` still calls `sort_by_keys` and discards key fields, and `sort.mojo:507` re-gathers keys on every pass. | M |
| **FU-7** | **(b) re-sized 2026-08-16, and it is not S.** `IsIn._value_set: DynArray` is not a stray payload — it is a *runtime-dtype dispatch* (`IsInKernel.dispatch`) inside an otherwise comptime-typed lane. Removing it means parameterising `IsIn` over the value-set's type, while `IsIn[A: Value]` today accepts any family and is used for both numeric and string sets. That is a design change, **M–L**, not a cleanup. **(d)** `ConditionalBinary` is 2-ary and `CaseWhen` 1-branch/numeric-only while the kernels and runtime builders are variadic — L, open. | (b) M–L, (d) L |

**One small leftover from the dead-code sweep (was Q7.5):** with `NoPartition`
removed, `Partition.row_indices` (`partition.mojo:94`) has a single construction
site that always supplies it, and `map_partitions:296` unwraps it
unconditionally — so its `Optional`, its `None` default and the `if` at `:114`
are dead. **XS.**

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

## 6. Deferred — Arrow parity

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
  null. Release callbacks are **not** a gap — they are implemented and invoked
  (`c_data.mojo:221,834,820,1234` plus three PyCapsule destructors).
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
  run detection. The shipped strategy set is a different axis entirely — see §7.
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

## 7. Rejected and replaced designs

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
  and loose parameters. Per-column `nulls_first` is still wanted — §6.

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

## 8. Architectural debt

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

**The A-series is closed.** A1 (per-node `comptime State`, which took `a + 1`
over 1M rows from 2.04 ms to 70.9 µs and deleted the `Context` positional-slot
invariant), A2 (eager recount at the slice boundary, landed as B12), A4
(`ExecContext` to `marrow/execution.mojo`, removing `marrow/kernels`' inbound
edge from core), A5 (both-lane parity, now four axes) and A6
(`worth_parallel`/`with_threads`) all landed. A3 (`DataType.layout()`) was
removed without being started: neither Arrow C++ nor arrow-rs drives its
serializers from `layout()`, so the "17 dtype ladders across `c_data` + `ipc`"
collapse it promised does not follow. What it *would* have bought —
`ArrayData` construction and validation — survives as the array-codec bullet
below, to be picked up on its own merits if ever.

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
  `to_data()`), 22 methods of duplicated layout knowledge. A per-dtype
  `layout()` descriptor is how both references collapse this, and it is the
  *only* thing it buys them — neither drives IPC or the C ABI from it, which is
  why the card proposing it (A3) was removed rather than scheduled.
- **`Value`** (`values.mojo:304`) is the union of four consumers' protocols —
  executor, printer, optimizer, pruner — and all 37 nodes pay for all four.

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
