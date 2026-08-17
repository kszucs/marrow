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
- **A narrow test selection can hang the compiler where a wide one does not**,
  and the diagnostic advice for it is a trap. See §2 *`dispatch` hangs a narrow
  unit*. Two things cost hours on 2026-08-17: `mojo run` leaves an **idle parent
  process pinned at ~0:07.9 CPU** while a child does the real work, so
  `pgrep -x mojo | head -1` (and the harness's own "compare elapsed against CPU
  time" hint) reports a frozen CPU clock and looks exactly like a deadlock —
  sum CPU across *all* mojo processes instead, and it climbs past 17 CPU-minutes.
  And `ps -eo comm` prints `mojo` for the parent but the **full path** for the
  child, so `awk '$2=="mojo"'` silently misses the one that matters.
- **`ctx.stripe` bodies may not raise**, and widening it miscompiles: the
  parameter form of `sync_parallelize` that accepts a raising worker needs an
  implicitly-capturing closure whose captures are silently not made. Watch for
  "assignment was never used" warnings on buffers the body writes.
- **`origin_of(a, b)` is an origin union**, which is what lets a function return
  values borrowed from either of two storages.
- **A closure type cannot be generic over its own trait bound**, so a *shared*
  dispatch loop would have to bind `func` on `Movable` and let the caller narrow
  through an extra closure. That adapter inlines into every arm: it measured
  **+662,740 bytes (+31.9% of `__text`)** on `query_streaming_agg_fused`. Each
  erased box writes its own `isa` ladder instead — do not refactor them back
  onto a common helper.
- **Two closure arguments to the same call may not both capture mutably**, nor
  mut+imm over one origin. An API taking several closures over shared mutable
  state must thread that state through as an explicit `mut` parameter of each
  closure — which, when there are several callbacks over one object, is a trait
  (`parquet`'s `LeveledSink`). A state *struct* handed to separate closures does
  not work: it makes them parametric over the enclosing generic parameter.
- **macOS needs the Metal toolchain installed separately** —
  `xcodebuild -downloadComponent MetalToolchain`. Without it every GPU test dies
  with `Metal Compiler failed to compile metallib`, which reads like a Mojo bug
  and is not one (a marrow-free three-line `elementwise` program fails
  identically). This was tracked as B25 and blamed on a backend crash for
  months; resolved 2026-08-16, `test_views_gpu` is 15/15.
- **A trait requirement cannot name a field**, which is why `slice()` (7 bodies)
  and `validity()` (7 byte-identical bodies) in `arrays.mojo` are unfactorable.
  `validity()` returns `Optional[BitmapView[origin_of(self.bitmap._value)]]` — the
  return type names a field, so it is neither expressible as a requirement nor
  reachable from generic code over `A: Array`. The `ArrayData` round-trip default
  that would replace the seven `slice` bodies makes every typed `slice`
  **raising**, and keeping `slice` non-raising was decided 2026-08-16. The same
  wall blocks a `trait ValidityBuilder` default (needs `self._bitmap`) and a
  `CReleasable` trait for `c_data`'s release slot (needs `self.release`). These
  are language limits, not design debt — stop re-filing them as opportunities.
- **Erase into a trait whose members are all runtime methods**, and only where
  the conformance is *honest* **and** has a consumer outside its own loop. A
  comptime member has no execution point at which to raise, so a box can only
  supply a plausible constant and the failure mode is a **wrong answer, not an
  exception**. Four sound-but-unconsumed conformances were removed for the second
  half of that rule at zero binary cost; see `docs/dyn-conformance-removal.md`.
- **A trait *default method* is not the `_arith[K]` shape.** The +115,600-byte
  trap above is about a *generic wrapper instantiated per kernel*; a trait default
  is instantiated once per conforming struct, exactly as the hand-written copy
  was, so DCE sees the same reachability graph. If that holds, deduplicating
  behind a trait default is source-level only. **It is an argument, not a
  measurement** — one ~30-line spike settles it for every item in §5.1's first
  bullet, and should run before any of them.
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
| **Size-gate resolution** | Acceptance says "gate green". `query_streaming_agg_fused` is at +0.449% of 0.5% and the next aggregate change will not fit. **Decided 2026-08-17: re-baseline deliberately**, as **S17**, at the end of the simplification wave so the wave's own cost is measured against the current baseline first |
| **Push, and run Benchmarks + Wheels** | Acceptance includes a *measured* wall-clock comparison and a green gate. Both currently rest on one unpushed laptop, ~580 commits ahead of `origin/main`. Those two jobs have never run at all |

### Should — real value, M1 does not block on it

| ID | Note |
|---|---|
| **V0** `MapScalar` | User-visible wrong type. Blocked only by the size gate, so it lands free once that is resolved |
| **M2.11** `CoalesceBatches` | ClickBench is filter-heavy; morsel compaction feeds the wall-clock criterion |
| **M3.4** O(N) top-K | Most queries are `GROUP BY … ORDER BY … LIMIT`. Grouped output is small, so this is not a blocker — but it is the obvious next perf lever |
| **M3.3** `JoinProcessor` drops the exec context | S, and a silent correctness-of-configuration bug |
| **FU-6** `sort_indices(StructArray, keys)` | Removes a re-gather |
| **S2–S16** the simplification wave (§5) | Runs before feature work resumes. S2 is an M1 wall-clock item. S1 landed 2026-08-17 and subsumed Q-NEW and L2 |
| **Q4.4** `ipc.mojo` → package | 2,342 lines. Deferred out of the wave — §5.1 |

### Could — genuine, nothing depends on it

`M2.1` Distinct/Union · `M2.2` unique/value_counts · `M2.4` statistical aggregates ·
`M2.13` EXPLAIN · `M3.6` Table/ChunkedArray depth · `M3.7` decimal arithmetic ·
`Q2.5` aggregates (a size play, and the size gate is the thing that is blocked) ·
`Q4.6` Parquet untyped fanout · `Q0.5` · `FU-7(b)` · `FU-7(d)`

### Won't — this cycle, with the reason

| ID | Reason |
|---|---|
| **B4** | Needs a hand-assembled BIT_PACKED file; no reference writer emits them |
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

### `dispatch` hangs a narrow compilation unit — **open, blocks `test_distinct`**

`RapidHashKernel.dispatch` (the 37-arm runtime dtype fan-out) hangs the Mojo
toolchain when it is the *only* substantial thing in a compilation unit. Eight
lines reproduce it:

```mojo
from marrow.arrays import DynArray
from marrow.builders import array
from marrow.dtypes import int32
from marrow.kernels.hashing import RapidHashKernel

def main() raises:
    var a: DynArray = array([1, 2, 1], int32)
    var h = RapidHashKernel.dispatch(a)      # never finishes
    if len(h) != 3:
        raise Error("bad")
```

Replace `dispatch(a)` with `apply(a)` on a typed array and it compiles in **10
seconds**. `test_hash__dispatch` alone (its only distinguishing feature is the
`dispatch` call) hangs; `test_hash__int32_deterministic` alone compiles in 10s.

**The paradox, and the reason this is not simply "too much code":** the same
case inside the full 27-case `test_hashing.mojo` driver compiles in **88 s from
a cold cache** and passes. More code compiles *faster*. This is also why §2's
2026-08-16 full-suite run went green at 2034 passed — a whole-suite unit
compiles what a narrow one cannot.

Ruled out on 2026-08-17, each by direct experiment rather than inference:

| hypothesis | how it was tested | verdict |
|---|---|---|
| recent marrow changes | git worktrees at `5435f59` and `32d6b65` | predates all of it |
| the pytest harness | plain `mojo build` / `mojo run` | no |
| optimisation level | `-O0`, `-O1`, `-O3` | no |
| `ASSERT=all` | with and without | no |
| toolchain version | `dev2026081605` and `dev2026081705` | no |
| the empty-array input | `[1, 2, 1]` hangs identically | no |
| `print` / formatting | removed | no |
| the artifact cache | `MODULAR_CACHE_DIR` to a fresh dir → 88 s, green | no |
| driver size | N = 1, 4, 8, 16, 27 cases → all pass, 10-80 s | no |
| CPU contention | retested on a clean process table | no |
| `-j 1` (serial compile) | blocks earlier still, at 5.4 s CPU | no help |

`SwissHashTable` + `insert_hashes`, and `RadixPartitioner.map_partitions` with a
closure, each compile fine in isolation — the trigger is `dispatch` alone.

Sampling the stuck process (`sample <pid>`) shows it reaches *execution*: eleven
`🔥 Thread` runtime workers exist and every one, plus the main thread, is parked
in `semaphore_wait_trap`. Under `-j 1` it stops earlier — three threads, no
runtime, main thread in `mach_msg2_trap`.

`test_distinct.mojo` on its own does **not** finish given 90 minutes
(`--mojo-timeout 5400` → `✗ compiling 11 tests from 1 files — 5400s`), so this
is a hang and not merely a slow compile.

**Workaround:** do not run `test_distinct` (or a single `::case` that reaches
`dispatch`) as its own selection; fold it into a wider one. **Next step:** confirm
`pytest marrow/kernels/tests` (whole directory) goes green, which would make the
workaround a rule rather than a guess, then file the eight-line repro upstream.
Reducing `dispatch`'s instantiation footprint is the marrow-side fix if upstream
is slow, and it would help the size gate too.

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

### Simplification wave — scheduled, runs before feature work resumes

Decided 2026-08-17 from a cross-read of the abstraction, organizational and
duplication audits. **Those five documents have been folded into this one and
deleted** — their open items are the `S`-IDs below, their traps are in §0, their
ruled-out designs are in §7, and their defend-this findings are in §8. The filter
applied was "does this make M1 cheaper or safer", not "is this written twice":
roughly half of what the audits listed is deliberately **not** scheduled, and
§5.1 says which half and why.

The `Source` column cites the audit section an item came from. To read one:
`git show 26e85d3:docs/duplication-audit.md` (likewise `-proposals`, `-1.4`,
`abstraction-audit`, `organizational-audit`).

Three findings came out of the cross-read that no single document had:

1. **§8's cycle fix contained the exact step L2 reverted**, and moving the whole
   `col`/`lit` overload set to one new module instead of splitting it is what made
   L2 achievable. **Landed 2026-08-17**; two of the three cycles are gone and the
   third is structural — see §8 for the surviving one and why no placement breaks
   it.
2. **`5b14bfa` already landed half of a "take both or neither" pair.** The
   proposals doc pairs `ExecContext.alloc_*` with moving `GPU_ENABLED` out of
   `utils.mojo`; the second half landed independently, so S8 is now the cheap
   remainder of a coherent move rather than half of one.
3. **The size gate is vetoing a correctness fix.** Its job is catching drift, and
   it is currently the sole reason a `MapArray` element reports the wrong type.
   Resolved as S17: re-baseline **once, deliberately, at the end of this wave**,
   so the wave's own cost is measured against the current baseline first.

| ID | Item | Source | Gate | Size |
|---|---|---|---|---|
| **S2** | **`bound_column` and `prune` on `TemporalColumn` and `ListColumn`.** Verified absent 2026-08-17 (`values.mojo:2452`, `:2565`); both inherit the conservative defaults at `:449`/`:460`, so **a date predicate prunes nothing in the fused lane**. ClickBench's `hits` is date-filtered, so this is an M1 wall-clock item, not just a gap. For a list column "no information" may genuinely be right — add `bound_column` only if so. | dup §1.2 A | tests + `binary_size` | S |
| **S3** | **Parity test for the operator↔interval-kernel pairing.** 20 pairings encoded by hand in the AOT lane and re-encoded as a name ladder at `dynamic.mojo:610-626`; a mismatch is silently wrong pruning, not an error. Assert name equality per pairing and both-directions coverage in `test_parity.mojo`. The associated-type fix (Angle A) is **not** scheduled — the test is most of the safety for none of the size risk. | dup §1.3 B | tests | S |
| **S4** | **`CastKernel(Kernel)` family trait**, one `dispatch(array, to, safe, ctx)`. 15 structs across six signatures today, so `cast()`'s ladder drops the arguments an arm does not take: `cast(x, decimal128(38, 2), safe=True)` **wraps on overflow**, and four casts never see `ctx`. Largest open defect in the abstraction audit. | abs §1.4 | tests + `binary_size` | M |
| **S5** | **`_validity_equal` rewrite and the two `DictionaryArray` bugs.** Rewrite around the eagerly-maintained null count (removes two popcounts and a bit-by-bit loop per `__eq__`; `Bitmap.__eq__` measured word-level XOR ~64x faster), fold `BoolArray` in as a seventh caller, fix `slice` not recounting nulls and `__eq__` ignoring `_offset`. **Order: the `slice` recount first — the rewrite depends on that invariant.** Two questions the card must settle, not skip: (a) how to recount `DictionaryArray`'s nulls on slice — it has no `bitmap` field, its validity lives in `_indices`, and `DynArray.slice` raises, so the options are accept `raises`, add a non-raising `null_count_in_range`, or recount at construction; (b) what correct dictionary equality *is* — comparing `_indices`/`_values` structurally is representation equality, the same mistake B26 fixed one level up, so settle it against Arrow C++/arrow-rs first. `DictionaryScalar.__eq__` (`scalars.mojo:544`) has the same shape and rides along with (b). | dup-1.4 §3 | tests | M |
| **S6** | **Delete all 26 `write_repr_to`.** Verified 26 definitions, requirement of no trait, and `DynArray.write_repr_to` does not even dispatch — it calls `write_to`, so the 10 array implementations are unreachable through the only handle callers hold. Since the boxes conform to neither `Array` nor `ArrowScalar`, promoting it to a requirement can no longer reach them; deletion is the only remaining option. | abs §1.1 | tests | S |
| **S7** | **Delete `trait Join`.** Three abstract methods, one conformer, one commented out; its own docstring says operators use concrete types directly. Sort-merge join is post-M1 by construction (§0.5 Won't). | dup §4 | judgement | XS |
| **S8** | **`ExecContext.alloc_buffer[T](n)` / `.alloc_bitmap(n)`**, collapsing the 10-site GPU-or-host preamble in `numeric`/`cast`/`hashing`/`boolean`. `execution.mojo` already owns `GPU_ENABLED` after `5b14bfa`, and `views` already imports both modules, so no new cycle. | dup §1.5 B | tests | S |
| **S9** | **`Bitmap.extend_validity`**, collapsing the 11-line reserve-then-propagate block at `builders.mojo:696, 827, 1022, 1229, 1370`. It is bitmap logic, not builder logic — it already calls `Bitmap.extend` and `set_range`. Not a hot path. | dup §1.7 B | tests | S |
| **S10** | **`c_data.mojo` release-slot helpers.** `is_released`/`mark_released` verbatim on three structs, six copies of the same `unsafe_bitcast` slot arithmetic. This is the spec's double-free guard — the one place three copies drifting is a memory-safety bug, and CLAUDE.md restricts `unsafe_ptr`-class code precisely so it is not reasoned about six times. | dup §1.8 A | tests | S |
| **S11** | **Two placement moves.** `equal_any` → a neutral `kernels/compare.mojo`, deleting the `kernels.numeric → kernels.string` edge; `Grouping` → `kernels/grouping.mojo` (a leaf five files import — check the import direction before preferring `groupby.mojo`). | org §1.8, dup §2.4 | import check | S |
| **S12** | **`kernels/tests/test_execution{,_gpu}.mojo` → `marrow/tests/`.** They test `marrow/execution.mojo` and import `...execution`, three levels up and out of their own package. Left behind by A4. | dup §2.5 | tests | XS |
| **S13** | **Make `kernels/__init__` say what it does.** The docstring lists 17 submodules and promises direct use; 8 are re-exported, so `mk.cast` works and `mk.concat` does not, with no principle separating them. Either export the rest or document the boundary. User-visible. | org §1.2 | tests | S |
| **S14** | **Suite-wide `tempfile.mkstemp` in `parquet/tests/`.** ~15 fixed `/tmp/marrow_*.parquet` paths across `test_codecs`, `test_bloom`, `bench_parquet`, `test_parquet`; two concurrent `pytest` invocations — which the harness explicitly supports — collide. Fixed paths are the prevailing convention there, so this is suite-wide or nothing. | dup §1.12 | tests | S |
| **S15** | **`Partition.row_indices`' dead `Optional`.** One construction site always supplies it and `map_partitions:296` unwraps unconditionally, so the `Optional`, the `None` default and the `if` at `:114` are dead. Leftover from the `NoPartition` removal (was Q7.5). | §5 leftover | tests | XS |
| **S16** | **Re-verified 2026-08-17 and mostly evaporated — recommend dropping.** The audit's citations predate `5b14bfa` and `e3a6cd0`. What survives: `_format_ns` is still defined after its only caller, but at `utils/testing.mojo:542` → `:429`, not `testing/bench.mojo`; the three-helper chain in `kernels/string.mojo:523/536/546` is unmoved. What is **gone**: three of the four `hashing.mojo` helpers (`_rapid_mix_wide`, `_rapidhash_bool`, `_rapidhash_primitive_masked` — removed by the pluggable-hash and wide-multiply work), leaving only `_indices_as_int32` (`kernels/hashing.mojo:59` → `:228`). **`_rapidhash_bool_masked` — the one part of this card that could have been a real defect, "dead or a masked-hash gap for boolean columns" — is resolved by deletion; it has zero hits under `marrow/`.** The residue is three cosmetic inlines, and the string trio is named in two docstrings and a comment, which is the tell that it names a step rather than fragmenting one. | dup §3 B | tests | XS, or drop |
| **S18** | **`Partition.original_row` is dead** — `kernels/partition.mojo:112`, zero callers repo-wide (the only other hit, `sort.mojo:146`, is a docstring mentioning `original_row_index`). Found while landing S15, which de-guarded the method rather than deleting it because removing a method is an API change and S15's card ruled that out. `Partition` is not re-exported from `kernels/__init__`, so the surface is internal and the deletion is safe. | S15 fallout | tests | XS |
| **S17** | **Re-baseline the binary-size gate, deliberately, and land V0.** `query_streaming_agg_fused` sits at +0.449% of 0.5%; `MapScalar` measured +0.137% more and was reverted for that alone. **Runs last**, so the wave's own cost is measured against the current baseline first. Record the new ceiling and the reason in `benchmarks/binary_size/README.md`; the gate's job is catching drift, not vetoing correctness fixes. | §0.5 Must, V0 | `pixi run binary_size` | S |

**Order.** S1 landed first, 2026-08-17 — it was the conflict-heaviest change and
every later `expr` edit would otherwise have been rebased onto it. Then the
correctness group S5 → S2 → S3 → S4. Then the free subtraction batch —
everything except S1–S5 and S17 — which is independent and can land in any
order. S17 last. Rows are deleted as they land, so the batch develops holes; do
not read the remaining numbering as a range.

### 5.1 Deliberately not scheduled

Recorded so the audits stop pulling. Each is real; none pays into M1.

- **The `expr/values.mojo` validity/state delegation** (dup §1.1, ~200 lines over
  37 nodes). Needs a spike — whether a sub-trait default returning `Self.State`
  recurses — and then a size measurement on the gated surface, and the honest
  expectation is that it comes back "no". Revisit after S17.
- **`ipc.mojo` → package (Q4.4) and the `marrow/io` grouping** (org §1.4). The
  seams are real and already labelled, but §1.10 is right that there is **no
  build-time argument**: splitting a file does not shrink the compilation unit.
  Navigation-only, wide diff.
- **`Column` (Array ∪ ChunkedArray ∪ DynArray)** (abs §1.12), **`TableProvider` /
  `ScanSource`** (abs §1.8, M3.5), **one `LeafBuilder` covering leveled decode**
  (abs §1.10), **`Named` split from `Kernel`** (abs §1.6). All sound; all with
  M2/M3 consumers, and M3.5 is already scheduled where it belongs.
- **`slice()`'s seven bodies and `validity()`'s seven bodies** (dup §1.4, §1.11).
  Language limits, not design differences — a return type naming a field cannot
  be a trait requirement, and the `ArrayData` round trip would make every typed
  `slice` raising. Settled 2026-08-16; do not re-open.
- **`parquet/format.mojo`'s 12 Thrift read loops** (dup §1.10). A wire-format
  decoder where per-field explicitness is the point; both references generate the
  equivalent. The *inconsistency* is the only live part — `read` is not a trait
  requirement while `write` is, and 5 of 12 structs do not conform to
  `ThriftWritable` at all.
- **`trait ByteSource`, `trait WindowKernel`, `trait ListValue`** — one conformer
  each, and each has a scheduled second conformer (M2.12, M2.3, the runtime list
  lane). Deleting them now would only have to be undone.
- **`tabular → expr`, the one cross-layer import edge** (abs §1.7, org §1.7).
  S1 did **not** fix this: it restructured inside `marrow.expr` and left
  `tabular.mojo:23`'s `from .expr.aggregates import FoldedAggregates` alone. The
  fix is to move the aggregate *catalog* (`Sum`, `Min`, `Count`, … and the fold)
  down to `kernels/aggregate.mojo`, which already owns `AggFunction` — nothing
  about "the aggregate named `sum` over an int64 column" is an expression
  concept. Deferred because it is an M-size move into the size-gated kernels
  layer and no M1 item blocks on it. Worth doing before M1.3, which adds a second
  consumer of the same catalog.

### Closure migration — done bar an upstream block

**Done (2026-08-16):** 288 of 292 closures are value-taking. Four remain, in
`views._reduce_dispatch`; they feed `_reduce_generator_wrapper`, which has one
definition upstream and it is comptime-`capturing[_]` only. Removing them needs
an upstream value overload or dropping GPU reduce — not a local change.

Surviving items from the Q/L/V backlogs, with their original IDs kept so git
history and CHANGELOG references still resolve. Everything else in those files
is done and has been deleted.

| ID | What is left | Size |
|---|---|---|
| **Q0.5** | **ATTEMPTED 2026-08-05 and blocked on a real limit — read this before retrying.** The plan was to give `BoxedValue` a `dtype_hint()`: `Some` for a fused node, `None` for the runtime lane, so the 0-row probe only runs where the answer genuinely is not known. Telling the lanes apart is easy — `DynValue` is concrete, so `__init__(out self, value: DynValue)` is a more specific overload than the generic `__init__[V: Value]`, and that works. **The blocker is that a fused node knows its output dtype's *type* but not its *parameters*.** `V.OutType` is a comptime type; the dtype *value* for `decimal128(p, s)`, `timestamp(unit, tz)` or `fixed_size_binary(w)` carries fields the type does not, and nodes do not store a dtype instance — `NumericLiteral[T]` holds only `_value: Scalar[T.native]`. So `DynType(V.OutType())` fails to compile for parameterised dtypes and would silently report `decimal128(0, 0)` if it did. **This is also a second reason the probe exists**, alongside "keeps a caller-supplied schema honest": executing is how a parameterised dtype gets its parameters. Retrying means either giving all 37 nodes a dtype instance — which grows the fused structs, and they are size-gated — or restricting the hint to parameter-free dtypes and keeping the probe for the rest. Neither is obviously worth the claimed ~16 KB. | M, and less attractive than it looked |
| **Q2.5** | Aggregates, remaining steps: **2b** — `AggState[K, V: NumericType]` was never widened to `PrimitiveType`, so temporal reductions do not work natively; **3** — `count_distinct`/`approx_count_distinct` (+`_grouped`) are still four free functions (`distinct.mojo:87,133,167,214`); **4** — `FusedAggregation` (single pass, AoS accumulator, comptime offsets, zero dispatch) has zero occurrences. **Pitch it as a size play, not a speed one.** Q6.1 measured the AOT-resolved aggregate against the runtime-named one in one binary (1M rows: g100k 13.943 ms vs 13.712 ms, g1k 11.219 vs 11.312) — under 2%, sign flipping between cardinalities, which is what you expect when an aggregate's identity resolves once per plan and the per-row fold is the same `AggState` either way. Validate any round of this against `bench_aggregate_aot.mojo` at `g100k`, never `g10`, and be prepared for the answer to be "no change". | L |
| **Q4.4** | `ipc.mojo` → package. Single file, 2,425 lines. **Not scheduled in the simplification wave — see §5.1**: the seams are labelled, but there is no build-time argument and the diff is wide. | M |
| **Q4.6** | **Localised 2026-08-05; no single lever exists here.** Current `__text`: `query_scan` **2,367,336**, `query_scan_typed` **1,839,632**, `query_streaming` **1,332,456** — so the 1.78x ratio in the original card still holds. The cost splits almost evenly in two, which is the useful finding: **~507 KB is the Parquet machinery itself** (scan_typed over streaming — `parquet::reader` 32 symbols, `codecs` 9, `schema` 18, `format` 15, plus dtypes +21 and arrays +34), and **~528 KB is the untyped fanout** (scan over scan_typed — reader 32 -> 107 symbols, codecs 9 -> 32). The second half is a scan that does not know its schema instantiating every dtype's decode path, i.e. the 28-arm comptime-gated ladder CLAUDE.md calls deliberate. **Checked and ruled out:** `kernels::cast` is no longer reachable from these binaries at all, and `marrow/parquet/` imports it nowhere — so the single-import trick that took 55-65% off the hashing binaries (removing one `cast` call from `RapidHash.dispatch`) does not apply here. ~7 KB per parquet symbol says the weight is in large monomorphised decode bodies, not symbol count. Reducing it is a design change to how the reader dispatches, not a lever to pull. Candidate: make the typed path (schema known at comptime) avoid instantiating the arms it cannot reach. | L, and genuinely L |
| **V0** | **`MapScalar` — BLOCKED on the size gate.** (The `cast` map arm, V0's other half, landed 2026-08-16 and was free in binary size.) The right fix is not a new scalar type but making `ListScalar` carry its own dtype instead of rebuilding `list_(child.dtype())` — that reconstruction reports the *shape*, so a `map` element answers `list<struct<key, value>>` and a `large_list` element answers `list<…>`; one field fixes all three. It works and is tested, but **a `DynType` field on `ListScalar` costs +0.137% on `query_streaming_agg_fused`, taking it from +0.449% to +0.586% and past the 0.5% ceiling** (measured 2026-08-16 by reverting that half alone). Reverted, not landed. Two ways forward: a 1-byte kind tag rebuilding `list_`/`large_list_`/`map_` from the child — cheaper, but loses `keysSorted` and the entries field name — or re-baselining the gate deliberately. That gate has been the binding constraint since the `Grouping` change and this is the second change to hit it. **Unblocked by S17**, which re-baselines deliberately; land the `ListScalar` dtype field with it and drop the 1-byte kind tag, which loses `keysSorted` and the entries field name. `scalars.mojo`. | S, lands with S17 |
| **FU-6** | `sort_indices` without key re-gather. `SortIndices.multi` exists but there is no public `sort_indices(StructArray, key_indices)`; `SortProcessor` still calls `sort_by_keys` and discards key fields, and `sort.mojo:507` re-gathers keys on every pass. | M |
| **FU-7** | **(b) re-sized 2026-08-16, and it is not S.** `IsIn._value_set: DynArray` is not a stray payload — it is a *runtime-dtype dispatch* (`IsInKernel.dispatch`) inside an otherwise comptime-typed lane. Removing it means parameterising `IsIn` over the value-set's type, while `IsIn[A: Value]` today accepts any family and is used for both numeric and string sets. That is a design change, **M–L**, not a cleanup. **(d)** `ConditionalBinary` is 2-ary and `CaseWhen` 1-branch/numeric-only while the kernels and runtime builders are variadic — L, open. | (b) M–L, (d) L |

**One small leftover from the dead-code sweep (was Q7.5):** with `NoPartition`
removed, `Partition.row_indices` (`partition.mojo:94`) has a single construction
site that always supplies it, and `map_partitions:296` unwraps it
unconditionally — so its `Optional`, its `None` default and the `if` at `:114`
are dead. **Scheduled as S15.**

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

### Abstractions and deduplication

Folded in from the abstraction, organizational and duplication audits
(2026-08-17), which were deleted once their open items became §5's `S`-IDs.
Every row is an approach that looks obvious and does not work.

- **One `Column[T: DataType]` struct** replacing the four `*Column` nodes. It
  would have to satisfy `NumericValue.lane` (returns `SIMD[T.native, W]`) and
  `StringValue.lane` (returns `String`) at once — different return types, and a
  `comptime if` does not rescue it, per §0's conditional-type note.
- **Promoting `write_repr_to` to an `Array`/`ArrowScalar` requirement.** Was the
  filed fix; the conformance removal made it unreachable — a requirement now binds
  only the nine typed arrays and nine typed scalars, leaving the erased handles
  (the only ones callers hold) still forwarding to `write_to`. Deletion is what
  remains, and is **S6**.
- **`Buffer.alloc_for[T](ctx, n)` in `buffers.mojo`** for the ten-site GPU-or-host
  preamble. It points the tree's lowest-level module at device policy. The
  decision belongs on `ExecContext`, which *is* the policy type and already owns
  `GPU_ENABLED` after `5b14bfa` — **S8**.
- **Moving the `AggFunction` trait down to `expr/aggregates.mojo`** to reunite it
  with its four conformers. It inverts the dependency: `groupby.mojo` would import
  `expr`, breaking the verified "`marrow/kernels` has no up-edges into `expr`"
  property. The catalog moving the *other* way is the live option — §5.1.
- **A generic `_header_equal[A: Array](a, b)`**, and the `unsafe_get`-on-`Array`
  fix that was filed for `__eq__`'s five copies. Both die on §0's
  trait-cannot-name-a-field limit; the element loops are also genuinely different
  (three index types, and `unsafe_get` raises for some arrays and not others).
  What *is* reachable is **S5**, which removes work rather than lines.
- **An embedded `ArrayHeader` field** collapsing `length`/`nulls`/`offset`/
  `bitmap` across six arrays. Forbidden by §0's *Do not change* — array layout.
- **`kernels/interval.mojo`'s placement — examined and upheld.** Consumed
  exclusively by `expr/` and it never touches an array, which usually means
  misfiled; the module docstring argues it, and the argument holds. Its one real
  consequence is that `IntervalKernel(Kernel)` inherits `expect_same_length` /
  `expect_same_dtype` it can never call, which is the `Named`-split finding in
  §5.1. Recorded so it is not re-opened.
- **The two `execution.mojo` files and the two `struct Filter`** are namespaced
  and correct. The only option is a rename and neither name is wrong. Leave both.

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
extends every family it conforms to at once);
parametric mutability on `Buffer`/`Bitmap` with a `comptime assert` making a
mutable copy a *compile* error; `DevicePassable` views with logical zero-based
indexing, which is what lets one kernel body serve CPU-serial, CPU-parallel and
GPU; kernel-parameterized generic nodes (10 structs, ~45 operators);
`Relation`-pure vs `Processor`-mutable, verified unviolated across all six
pairs; and `BoxedValue` as an erasure boundary that is also the fusion boundary.
`marrow/kernels` is a **verified DAG with no up-edges into `expr` or
`tabular`**.

Two items from that list did not survive later work and have been struck above.
**Peer erasure** (`DynArray: Array`, `DynScalar: ArrowScalar`,
`DynBuilder: Builder`, `DynType: DataType`) was removed: the premise "generic
code takes either" was false — no generic code was ever bound on those traits
outside the boxes' own dispatch, and the four conformances were load-bearing
only for each other. See `docs/dyn-conformance-removal.md`. **`Breaker` as
marker-by-conformance** was removed by `7d57398`; there is no `Breaker` trait,
and a breaker is now simply a node whose `State` is its materialized column.
`marrow/tabular` also has an up-edge into `marrow.expr.aggregates`, so the
one-directional claim holds for `kernels` but not tree-wide — §5.1 has the fix
and why it is deferred.

A second audit (2026-08-17, over the 52 traits under `marrow/` and
`python/bindings/`) added to the defend-this list, and it is folded in here
because those documents are gone:

- **`DataType`'s deliberate minimality** — five inherited traits, one defaulted
  method, no associated types. Companion `ScalarType`/`ArrayType` members were
  removed in `63b93aa` because they forced an import cycle, and `DynType` being a
  *peer* rather than a supertype is the right call.
- **The nine `dispatch_*` narrowing adapters** over one `DynType._dispatch` loop,
  with the "a closure type cannot be generic over its own trait bound" limit
  recorded at the one place it bites.
- **`Aggregation` / `AggFunction` / `ColumnAggregator`** — three genuinely
  distinct responsibilities, with the optional-capability problem solved the right
  way: a predicate the caller checks first (`is_mergeable` comptime,
  `mergeable()` runtime), so the grouper never picks a strategy it cannot run.
  **That is the pattern to copy** the next time a capability is optional —
  `Array.to_device`/`to_cpu`, where 6 of 9 conformers inherit a raise, is the
  counter-example and the one place the tree does it the worse way.
- **`DynValue` conforms to `Value` and nothing else**, with the rule stated in
  the code. Best-articulated abstraction boundary in the tree.
- **`ByteSource`** and **`PyConverter`** — one responsibility each, no pretence.

**Two erasure mechanisms coexist deliberately, and this was written down nowhere:**
closed `Variant` + `isa[T]()` for the data types (`DynArray`, `DynBuilder`,
`DynScalar`, `DynType`) and open `ArcPointer[NoneType]` + `thin` function-pointer
trampolines for the plan layer (`DynRelation`, `DynProcessor`, `BoxedValue`). The
split is sound — a plan's node set is open and recursive, the set of Arrow layouts
is closed by the spec. Note CLAUDE.md's "no `rebind` casts, no function-pointer
trampolines" reads as a global rule when it describes only the first mechanism.

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

### The `marrow.expr` import cycles — two broken, one structural

**Resolved as far as code motion reaches, 2026-08-17 (S1).** There were three:
`values ⟷ dynamic`, `values ⟷ relations`, `relations ⟷ execution`. The last two
were placement accidents and are gone; the first is not, and the reason is
recorded here so it stops being re-filed.

What landed: `expr/core.mojo` (a leaf) holds `Datum`, `into_array` and
`_union_columns`; `expr/builders.mojo` holds the **entire** `col`/`lit`/
`if_else`/`coalesce`/`case_when` overload set, typed and untyped together;
`BoxedValue` moved from `relations.mojo` down to `values.mojo`, beside the
`Value` it erases. Current graph:

```
core        -> (arrays, scalars)          # leaf
pruning     -> (leaf)
aggregates  -> (kernels only)             # no intra-expr imports
values      -> core, pruning, aggregates, dynamic
dynamic     -> core, pruning, values      # `Value` only
builders    -> values, dynamic
execution   -> core-free: values (BoxedValue), aggregates, pruning
relations   -> values, dynamic, builders, aggregates, execution, pruning
```

**Why `values ⟷ dynamic` survives, and why moving `AggExpr` does not help.**
The original plan was to extract `Value` into `core.mojo` and move `AggExpr` to
`aggregates.mojo`. Neither is possible, because these three references close a
loop that spans the modules whatever the placement:

- `Value` defaults `count_distinct`/`approx_count_distinct` to an **`AggExpr`**,
  and defaults `isnull`/`notnull` to the fused **`NullPredicate`** — so `Value`
  cannot leave the node zoo, and `core.mojo` cannot hold it.
- `AggExpr` converts implicitly from **`Reduction`** (`values.mojo`) and holds an
  unresolved **`DynValue`** (`dynamic.mojo`), while `NumericValue.sum()` returns
  a `Reduction` and `Reduction.alias()` returns an `AggExpr`.
- `DynValue` **conforms to `Value`**.

So `{Value, NumericValue, Reduction, AggExpr, DynValue}` is one strongly
connected component. Putting `AggExpr` in `aggregates.mojo` only drags
`aggregates` into it (a fresh `aggregates ⟷ values`); putting it in `core.mojo`
drags in `core` and adds `core -> dynamic`. It stays in `values.mojo` because
that is the placement with the fewest edges. Breaking this needs a design
change — `Value` shedding its aggregate/null fluent defaults, or `AggExpr`
holding a `BoxedValue` rather than a `DynValue` — not a file move. `Breaker` and
`Context` no longer exist, so the original wording named two types that are gone.

Two smaller, genuinely free ones: `views → buffers` exists only because
`BufferView.filter` and `BitmapView.to_owned` *allocate* — they are kernels
living on a view type. `arrays → builders` is three convenience constructors;
`arrays.mojo:2085` already imports `DynBuilder` function-locally, so the author
knew.
