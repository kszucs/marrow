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

**Cleared 2026-08-19** — verified against the code, not read off a header:
`S4` (the `CastKernel` trait landed in `17488cd`), `S13`, `S18`, `A-1`, `A-2`,
`A-3`, `A-7`, `A-12` are implemented; `S19` is measured impossible and closed
as Won't; the nested-array-equality wedge is fixed and `test_arrays.mojo`
passes 167/167; and `A-9` no longer reproduces — all 41 `test_groupby.mojo`
cases now run whole in **213 s**, against a card claiming they exceed 1800 s.
Their rows are deleted rather than struck through, per this file's own rule.

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
  measure.** Stubbing the `cast_array` calls out of the expression layer left
  the gate binary byte-identical. (The card that recorded this said "both";
  `expr/dynamic.mojo` has **seven** — `:187, :189, :203, :216, :218, :359,
  :373` — so re-count before repeating the experiment.)
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

### Trait and compiler limits measured 2026-08-17

Each is a hard error, and each rules out an approach that looks obvious.

- **A trait default body cannot name a field of `Self`** —
  `error: '_Self' value has no attribute 'a'`. The operand must be reached
  through a by-value accessor requirement.
- **A ref-returning accessor is not expressible at trait level** —
  `cannot return 'self's origin, because it might expand to a RegisterPassable
  type`. Same origin-widening wall as `arrays.validity()`; this is what forces
  the `.copy()`.
- **A sub-trait default returning `Self.State` does not recurse — it fails to
  reduce**: `cannot implicitly convert '_Self.Operand.State' value to
  '_Self.State'`. `rebind` cannot bridge it, because `rebind` takes its argument
  by implicitly-copyable borrow and `State` is not. **This settles §5.1's
  validity/state delegation question as impossible, not merely costly.**
- **A struct parameter does not satisfy an associated-type requirement** —
  `struct Neg[A: Leafy](UnaryDirect)` gives `required member 'A' is not
  specified`. The trait needs a differently-named member the struct binds
  explicitly.
- **A trait default is size-free; the boilerplate is the cost.** Hoisting the 22
  verbatim `referenced_columns` copies to a `UnaryValue` default compiles clean
  and measures **+0 bytes on `query_streaming_agg_fused`, +128 on `query_exprs`**
  — and the +128 is not the trait mechanism but +48 bytes per boxed
  instantiation, from the forced `.copy()`. It is still not worth doing: the
  dedup makes the file **58 lines longer**.

### Measurement traps found in the alpha wave (2026-08-18)

Each of these produced a **confident wrong conclusion** before it was caught.
Sources: `git show c0831f5^:docs/alpha-findings/<name>.md`.

- **A sampled profile tells you where time goes *under the profiler*.** `sample`
  and Instruments inflate image load/unload, so anything dominated by `dlopen`,
  `dlclose` or loader work is over-weighted. Measured on one `-O1 -g` build:
  **37.83 ms/run under the sampler, 7.85 ms/run without it** — a ~5x
  over-attribution that produced the claim "80% of `COUNT(*)` is `dlopen`". The
  real saving from fixing it was ~0.9 ms/query. Confirm a profile-derived
  hypothesis with a wall-clock A/B before believing its percentage.
  (`o1-codec-caching`)
- **The profile build is `-O1`; the benchmark build is `-O3`.** `scripts/profile.py`
  rebuilds with debug info because `-O3` inlines away the frames you are reading.
  So a trace shows *where*, never *how much* — absolute numbers come from
  `bench_clickbench.py`.
- **Perturbing one `-O3` unit can move an unrelated kernel.** An exact-size
  `self.tokens.reserve(n)` — an upper bound, apparently free — cost **+43% on
  `bench_contains_1m`**, which shares no code with the changed type, plus +35%
  and +21% on other untouched scan paths. Reproduced two runs per side; vanished
  when the one line was removed. **Only the drift controls caught it**: without
  benchmarks the change cannot touch, a 20.9x win would have shipped with a 40%
  regression underneath. This is the concrete case for CLAUDE.md's "always
  include rows the change cannot touch". (`o3-string-alloc`)
- **A passing size gate is not "no regression".** `check_gate.py` compares to the
  recorded `baseline.json`, **not** to the branch under test. Four of five
  recorded values sat above the tree on *both* branches, so gates read as
  shrinking 2.4-4.1% while the branch-to-branch measurement showed every gate
  **grew** ~16 KB. Ask which question you are answering. (`g3-regression-check`)
- **ASAN is not usable as evidence on this tree.** A deliberate-overflow probe
  built with the harness's own flags **hangs before producing a report**. This is
  stronger than CLAUDE.md's existing "ASAN can hide a heap bug": here it cannot
  be run at all. (`f1-distinct-segfault`)
- **`pixi run -e dev python script.py` does not rebuild `libmarrow.so`** — only
  pytest's `conftest.py` does. The natural A/B (checkout old, run script,
  checkout new, run script) therefore measures the *new* library twice. It
  produced a confident "no improvement, revert it" on a change that was a 20.7x
  win, and separately made an already-fixed bug still look broken. Rebuild with
  `pixi run build_python` between variants, or drive the comparison through
  pytest. (`o2-cast-utf8`)
- **A mass failure at exactly the harness deadline is the harness, not your
  change.** Five separate runs reported every case failed with empty messages at
  precisely 1800.0s. It nearly caused a correct fix to be reverted. Distinguish
  the two shapes: a *slow* unit burns CPU; a *wedged* one freezes — check that
  accumulated CPU `time` is advancing, not just `%cpu`.
- **A clean `mojo precompile marrow` is not evidence a test file will build.** It
  compiles the library, not the test's instantiations. Both compiler hangs in §2
  are invisible to it. (`o2-cast-utf8`, `h2-nested-equality-wedge`)

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
  **Probed directly on 2026-08-17** for `validity()`, and it fails twice over:
  declaring the requirement with a self-wide `origin_of(self)` is rejected
  because Mojo does not widen an implementation's narrower origin at the
  conformance site, and `NullArray` cannot implement it at all — it is all-null
  with no bitmap, and `None` already means all-valid. Even with the origin
  solved it would be a promise 2 of 9 conformers could not keep, which is the
  `to_device` shape the abstraction audit called the tree's one leaky
  abstraction.
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

**Workaround, now a rule rather than a guess:** do not run `test_distinct` (or a
single `::case` that reaches `dispatch`) as its own selection; fold it into a
wider one. The confirmation this card asked for arrived 2026-08-17 while landing
S7 — `pytest marrow/kernels/tests` over the whole directory is **544 passed, 162
skipped, 0 failed**, including all 26 `test_join.mojo` cases, while
`test_join.mojo` alone burned 30 minutes to a timeout that reports as 26
`TIMEOUT` results and reads exactly like mass failure. **What remains:** file the
eight-line repro upstream. Note the marrow-side mitigation this card proposed —
reducing `dispatch`'s instantiation footprint — was independently carried out for
size reasons (§0, the `variant_dispatch` removal, −662,740 bytes and roughly half
the cold build time), so re-check whether the hang still reproduces before
spending anything more on it. **Re-checked 2026-08-19: it does.**
`pytest marrow/kernels/tests/test_distinct.mojo` on its own burned the full
900 s deadline and reported all 11 cases as failures — the TIMEOUT-reads-like-
mass-failure signature described above — while the whole `marrow/kernels/tests`
directory passes. So the `variant_dispatch` removal did not fix it and the
workaround still stands: never select this file alone.

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

### M1.1 — Optimizer v1 — **partly landed 2026-08-18, remainder M**

**Projection pushdown shipped** (`docs/alpha-findings/p1-pushdown.md`) and was
the single largest performance change of the alpha: **17.7x -> 5.0x** vs polars,
3.6x overall, with `COUNT(*)` 271 -> 9.9 ms. `DynRelation.optimize()` walks from
the root carrying the columns each parent reads and rewrites `ParquetScan`'s
schema — which *is* its projection. `execute()` calls it, so both the Mojo verbs
and the Python `LazyTable` get it.

Two design points from that work, worth not re-deriving:

- **Traversal is `children()` + `with_projection()`, not `inputs()`.** Erasure
  means a generic optimiser cannot rebuild a node whose type it does not know, so
  `inputs()` alone would need `with_inputs()` *and* a per-child
  `required_columns()` — three virtuals, of which the third *is* the rewrite.
  `with_projection` collapses them into one and reuses `with_predicate`'s
  existing erased-pointer protocol rather than adding a third incompatible
  rewrite mechanism. A trampoline *field* mentioning `DynRelation` is what Mojo
  rejects as recursive; a field returning `List[DynRelation]` compiles, so
  read-only traversal costs one trampoline.
- **Two guards carry the correctness**: `optimize()` seeds the required set with
  the *root's own schema*, so a passthrough node can only narrow to a subset its
  parent asked for and the plan's output schema is invariant; and a scan never
  narrows to nothing, because `COUNT(*)` references no column and a zero-column
  read yields zero-row batches the scan's loop reads as EOF.

`Join` is deliberately excluded: its key indices are *positions* into its
children's schemas, so narrowing a child would silently join on the wrong column.

**Still open on this card:**

- **projection pushdown through `Join`** — needs the key indices rewritten
  alongside the narrowed schema, which is why it was skipped;
- **recursive predicate pushdown.** Measured as worth nothing today (every
  ClickBench query is already `read_parquet(...).filter(...)`) and two of the
  three directions are unsafe: through `Limit` it changes what the limit counts
  from, and through `Project` a rename can prune on another column's statistics —
  a wrong answer, not an error. Only `Sort` and nested `Filter` are safe;
- **conjunct splitting**, still the precondition for partial pushdown;
- **limit pushdown** into the scan, and **constant folding**;
- **a recursive `write_to`** — no node renders its children, so `explain()`
  prints one shallow label rather than a tree. `children()` is now the primitive
  that makes a real EXPLAIN possible; writing the renderer was out of P1's scope.

---

Original card, for the parts not yet done. No `optimize.mojo` existed; two ad-hoc
rewrites lived in the *builder* instead: predicate → `ParquetScan`
(`relations.mojo:437-443`, non-recursive, fires only when `Filter` sits directly
on the scan) and `Limit` → `Sort` top-K (`:723-735`).

Deliver a `DynRelation → DynRelation` rewrite pass with:

- **conjunct splitting** — `Filter` holds one `predicate: BoxedValue`
  (`relations.mojo:866`), not a `List[BoxedValue]`; splitting `AND` is the
  precondition for partial pushdown;
- **predicate pushdown** through `Project`/`Sort`/`Limit`, recursively;
- ~~**projection pushdown**~~ — **done**, see above. `referenced_columns()` was
  implemented on both lanes and the box and called only by tests; it now has its
  consumer;
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
| **M2.6** | **String manipulation and regex — the single largest kernel hole.** There is no regex engine in the repo. Missing: `match_substring_regex`, `replace_substring(_regex)`, `extract_regex`, `split_pattern(_regex)`, `count_substring`, `find_substring`, `utf8_slice_codeunits`/substring, `lpad`/`rpad`, `binary_join`, the whole `utf8_is_*` classification family, trim-with-charset. Also: string kernels dispatch on `is_string_like()` only, so `binary`/`large_binary` are excluded from string comparison. **Engine chosen 2026-08-18 (`docs/alpha-findings/g2-regex-evaluation.md`): `dlopen` PCRE2, on the pattern `parquet/codecs.mojo` already uses for zstd/snappy/lz4/brotli.** `mojo-regex` was evaluated and **rejected on correctness**, not on version drift — it builds fine against our pinned Mojo, but an optional group is never entered, so `(?:www\.)?` is skipped and ClickBench Q29 returns `www.example.com` where pyarrow and CPython return `example.com`; 10 of 25 `sub()` cases disagree with CPython (minimal repro `sub("(?:foo)?bar", "B", "foobar") -> "fooB"`). It is also on no conda channel (vendoring 15,433 lines) and costs +493,792 bytes of `__text`, +10.1% on `query_dynvalue` against a 0.5% gate. PCRE2 measured correct on every case, **10.6x faster** (5,850,314 vs 551,315 rows/s), **zero `__text`** since the engine lives in the shared library, and resolves from conda-forge for osx-arm64 and linux-64. Estimated 2-3 days: `utils/regex.mojo` FFI shim, `kernels/regex.mojo` typed-first kernels, runtime-lane-only wiring so the fused gates keep contributing zero symbols. Q29 is the forcing function, not the benefit — marrow has no regex kernel at all and PyArrow ships seven. | engine chosen, not started | L |
| **M2.7** | **Temporal completeness** — `strftime`/`strptime` (and **string↔timestamp cast raises**, `cast.mojo:1028`), timezone-aware extraction (everything decomposes as UTC and a non-UTC `tz` is silently ignored, `temporal.mojo:36-39`), `week`/`iso_week`/`iso_year`, `millisecond`/`microsecond`/`nanosecond`, `is_leap_year`, `ceil_temporal`/`round_temporal`, and the `*_between` family. Temporal **arithmetic** belongs here too — date ± interval, `date_diff`, `now` — which H2O and TPC-H date logic both need and which nothing implements. | not started | M |
| **M2.8** | **Multi-file / dataset scan.** `ParquetScan.path` is a single `String`. No glob, no dataset, no partition discovery, no fan-out. Also: **bloom filters are fully implemented in the reader and never consulted by the scan** (zero `bloom` hits in `marrow/expr/`) — cheapest remaining pruning tier, do it with this. Two known-safe-but-lossy behaviours ride along: predicate pruning switches *off* entirely for nested files rather than risk misaligning statistics with the projection, and Hive-style `col=val` directory discovery does not exist. | not started | M |
| **M2.9** | **Join on computed keys.** Every join key must be a bare column reference; a computed expression raises at `relations.mojo:597` and `:606`. H2O and TPC-H both need it. | raises today | M |
| **M2.10** | **Plan-level parallelism.** The kernels are parallel; the pull loop is not — `collect()` drains one morsel at a time on the calling thread and nothing schedules operators across workers. Pairs with M3.3, which is the one-line half of the same gap. | not started | L |
| **M2.11** | **`CoalesceBatches`.** Nothing compacts small morsels after a selective `Filter`, so every downstream operator pays vector-at-a-time overhead on sparse batches. No such node or processor exists. | not started | S |
| **M2.12** | **A remote `ByteSource`.** The seam is in place and has exactly one implementation: `trait ByteSource` (`parquet/source.mojo:20`), `MappedFile` (`:49`), `ParquetFile[S: ByteSource]`. The OpenDAL Mojo binding (`~/Workspace/opendal/bindings/mojo` — operator verbs, seek-based ranged reads, fs/s3/http/memory, blocking only) is a capable WIP with **zero integration**: `opendal` appears nowhere under `marrow/` except one comment. Mind the 64-byte `Buffer` alignment constraint in §0 — it has already blocked two designs. | not started | M |
| **M2.13** | **`EXPLAIN` / plan pretty-printer.** `BoxedValue.render()` (`relations.mojo:270`) and `DynRelation.write_to` exist; nothing composes them into a plan dump on either frontend. Debuggability blocker for M1.1 the moment rewrites start moving nodes. | not started | S |
| **M3.0** | **Stream the probe side for LEFT/FULL/SEMI/ANTI.** These block on the whole probe side today: `JoinProcessor` collects it and probes once, because the kernel recomputes build-side matches per probe and `probe()` returns an assembled `StructArray` rather than pairs, so a caller cannot accumulate them. Streaming needs `HashJoin` to carry the matched-build set across probes and emit the tail on drain. Correct but blocking is where B5 left it; this makes it correct *and* streaming. | `expr/execution.mojo` `JoinProcessor.pull`, `join.mojo:569` `_emit_unmatched` | M |
| **M3.1** | **Join completeness** — `JOIN_CROSS`, `JOIN_MARK`, `JOIN_SINGLE` and `JOIN_ASOF` are declared constants that are never implemented; a CROSS join currently falls into the LEFT/RIGHT/FULL tail and produces wrong output rather than a cartesian product. All five `JOIN_ALGO_*` constants are dead and `struct Join(Relation)` has no `algorithm` field. Sort-merge join **had** a commented-out field-list stub in `join.mojo`; S7 deleted it along with `trait Join` (the kernel-side algorithm trait, unrelated to the `Join` plan node) — design sort-merge fresh, and note that operators name the concrete algorithm type, so a new algorithm is a new struct rather than a conformance. Non-equi joins are absent as a *class*, not just as a kind: no nested-loop, no piecewise-merge/IEJoin. | constants only | L |
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

**Verified card-by-card on 2026-08-17, and the schedule did not survive intact.**
The verification document has been folded into this section and deleted; read the
original with `git show f38f3c5:docs/simplification-wave-plan.md`.
Five of the thirteen cards had a false or vacuous premise, two collide with
designs §7 had already rejected, and one had already landed. **Read this table
before acting on any `S` row below it** — the row text is the original filing,
kept so the correction is legible.

| Card | The row says | Verified |
|---|---|---|
| **S2** | "a date predicate prunes nothing in the fused lane" | **Vacuous** — the fused lane cannot *express* a date predicate. The real defect is `bound_column`: a fused join on a temporal column **raises**. Re-scoped as A3 below. |
| **S4** | 15 structs, "four casts never see `ctx`" | **Six** never see `ctx`, and there are **two** behavioural defects, not one — `TemporalCast` drops `safe` too. An existing test encodes the bug. **Landed 2026-08-17 in `17488cd`** — `trait CastKernel(Kernel)` (`cast.mojo:72`) gives all 20 kernels one `dispatch(array, to, safe, ctx)` and `cast()` hands every arm all four. The scheduled row below was left behind and has now been deleted. |
| **S6** | "deletion is the only remaining option" | **Premise false.** `write_repr_to` is a stdlib `Writable` member whose default is field reflection, not `write_to`. Deleting all 26 is a user-visible Python regression. Re-scoped as C8. |
| **S8** | "no new cycle" | **False** — it closes `execution → buffers → views → execution`. Resolved by D1 below. |
| **S11** | two placement moves | The `equal_any` half **creates a `numeric ↔ compare` cycle**. Rejected as D2; only the `Grouping` half survives. |
| **S13** | 17 submodules, 8 re-exported | **19 named, 7 re-exported**, and `interval.mojo` is missing from the docstring entirely. |
| **S14** | ~15 paths, 4 files | **119 paths, 14 files**, and the fix is *not* `mkstemp` — Mojo tests cannot reach pytest's `tmp_path`. |
| **S16** | scheduled | **Evaporated** — the card says so itself; its citations predate `5b14bfa`/`e3a6cd0`. |
| **S17** | scheduled last | **Already landed** at `0e552a7`. |

Four cards were promoted *into* the wave that the original schedule does not
list, and one was demoted out:

| New | Item |
|---|---|
| **A2** | **`is_in` never verifies key equality.** Membership is decided on the 64-bit hash alone, so a collision is a silent wrong answer (~2⁻⁶⁴, but wrong with no error) and Arrow's `is_in` is exact. Fix is the seven lines `SwissHashTable.probe` already pays (`hashtable.mojo:619-626`): a `Take` + `EqKernel` on probe hits only. See D3 for sequencing against B2. |
| **A4** | **`RecordBatch`/`Table` validate nothing** — promoted from §8. |
| **B1** | **Move the aggregate catalog down to `kernels/`** — S–M. |
| **B2** | **Enforce `SwissHashTable`'s build→probe lifecycle** — XS. |
| *demoted* | **The `DynRelation` planner split moves out of the wave and into M1.1**, as its *first commit* — M1.1 needs the binder callable from `optimize.mojo` or it will duplicate it. On its own it is tidying: 410 lines, two files, zero behaviour change, **0-byte** expected size delta, and nothing in M1.2–M1.6 unblocks. Prefer **Option B** (keep the fluent surface — invariant 3, and M1.3 is specified over it; extract only the binder). Option A costs ~174 call-site rewrites and partially reverts `53f7be3`, which moved `execute` *onto* the box deliberately. |

**Decisions taken 2026-08-17**, each recorded with its alternative because each
reverses or upholds something already written down:

- **D1 — S8's allocation helper: `Buffer.alloc_for[T](ctx, n)` in `buffers.mojo`,
  and §7 is amended.** The card's "no new cycle" reasoning is wrong:
  `marrow/execution.mojo` imports *nothing* from marrow — it is a leaf — so
  putting `Buffer`/`Bitmap` on `ExecContext` closes a cycle. The acyclic
  alternative is the one §7 rejected for pointing the tree's lowest module at
  device policy; that objection is aesthetic, the cycle is structural, and §8
  tracks cycles as debt. The helper takes `ctx` as a *parameter*, so
  `buffers.mojo` names the policy type without owning the policy. Two things ride
  along: the 10 sites are **not one preamble** (7 buffer / 3 bitmap, two use
  `alloc_zeroed`), so it is two helpers or a flag; and **the GPU arm of the two
  `alloc_zeroed` sites calls `alloc_device`, which does not zero** — a latent bug
  centralising would fix.
- **D2 — `equal_any` does not move; `kernels/compare.mojo` is not created.** See §7.
- **D3 — `is_in` gets the `EqKernel` verification** (A2 above), matching `join`.
  **Sequencing matters:** B2's optional half deletes `probe()` and inlines those
  seven lines into its two `join.mojo` callers. Do **A2 first** so B2's inlining
  covers three call sites, or keep `probe()` and drop B2's optional half — the
  other order writes A2 against a method B2 is about to delete. `membership` also
  has no benchmark, so the verification's cost is unmeasurable until one exists:
  a reason to write one, not to defer the correctness fix.

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
| **S2** | **`bound_column` and `prune` on `TemporalColumn` and `ListColumn`.** Verified absent, re-checked 2026-08-19 (`values.mojo:2688`, `:2830`); both inherit the conservative defaults at `:421`/`:432`, so **a date predicate prunes nothing in the fused lane**. ClickBench's `hits` is date-filtered, so this is an M1 wall-clock item, not just a gap. For a list column "no information" may genuinely be right — add `bound_column` only if so. | dup §1.2 A | tests + `binary_size` | S |
| **S3** | **Parity test for the operator↔interval-kernel pairing.** 20 pairings encoded by hand in the AOT lane and re-encoded as a name ladder at `dynamic.mojo:610-626`; a mismatch is silently wrong pruning, not an error. Assert name equality per pairing and both-directions coverage in `test_parity.mojo`. The associated-type fix (Angle A) is **not** scheduled — the test is most of the safety for none of the size risk. | dup §1.3 B | tests | S |
| **S6** | **Delete all 26 `write_repr_to`.** Verified 26 definitions, requirement of no trait, and `DynArray.write_repr_to` does not even dispatch — it calls `write_to`, so the 10 array implementations are unreachable through the only handle callers hold. Since the boxes conform to neither `Array` nor `ArrowScalar`, promoting it to a requirement can no longer reach them; deletion is the only remaining option. | abs §1.1 | tests | S |
| **S8** | **`ExecContext.alloc_buffer[T](n)` / `.alloc_bitmap(n)`**, collapsing the 10-site GPU-or-host preamble in `numeric`/`cast`/`hashing`/`boolean`. `execution.mojo` already owns `GPU_ENABLED` after `5b14bfa`, and `views` already imports both modules, so no new cycle. | dup §1.5 B | tests | S |
| **S9** | **`Bitmap.extend_validity`**, collapsing the 11-line reserve-then-propagate block at `builders.mojo:696, 827, 1022, 1229, 1370`. It is bitmap logic, not builder logic — it already calls `Bitmap.extend` and `set_range`. Not a hot path. | dup §1.7 B | tests | S |
| **S10** | **`c_data.mojo` release-slot helpers.** `is_released`/`mark_released` verbatim on three structs, six copies of the same `unsafe_bitcast` slot arithmetic. This is the spec's double-free guard — the one place three copies drifting is a memory-safety bug, and CLAUDE.md restricts `unsafe_ptr`-class code precisely so it is not reasoned about six times. | dup §1.8 A | tests | S |
| **S11** | **Two placement moves.** `equal_any` → a neutral `kernels/compare.mojo`, deleting the `kernels.numeric → kernels.string` edge; `Grouping` → `kernels/grouping.mojo` (a leaf five files import — check the import direction before preferring `groupby.mojo`). | org §1.8, dup §2.4 | import check | S |
| **S12** | **`kernels/tests/test_execution{,_gpu}.mojo` → `marrow/tests/`.** They test `marrow/execution.mojo` and import `...execution`, three levels up and out of their own package. Left behind by A4. | dup §2.5 | tests | XS |
| **S14** | **Suite-wide `tempfile.mkstemp` in `parquet/tests/`.** ~15 fixed `/tmp/marrow_*.parquet` paths across `test_codecs`, `test_bloom`, `bench_parquet`, `test_parquet`; two concurrent `pytest` invocations — which the harness explicitly supports — collide. Fixed paths are the prevailing convention there, so this is suite-wide or nothing. | dup §1.12 | tests | S |
| **S16** | **Re-verified 2026-08-17 and mostly evaporated — recommend dropping.** The audit's citations predate `5b14bfa` and `e3a6cd0`. What survives: `_format_ns` is still defined after its only caller, but at `utils/testing.mojo:542` → `:429`, not `testing/bench.mojo`; the three-helper chain in `kernels/string.mojo:523/536/546` is unmoved. What is **gone**: three of the four `hashing.mojo` helpers (`_rapid_mix_wide`, `_rapidhash_bool`, `_rapidhash_primitive_masked` — removed by the pluggable-hash and wide-multiply work), leaving only `_indices_as_int32` (`kernels/hashing.mojo:59` → `:228`). **`_rapidhash_bool_masked` — the one part of this card that could have been a real defect, "dead or a masked-hash gap for boolean columns" — is resolved by deletion; it has zero hits under `marrow/`.** The residue is three cosmetic inlines, and the string trio is named in two docstrings and a comment, which is the tell that it names a step rather than fragmenting one. | dup §3 B | tests | XS, or drop |
| **S17** | **Re-baseline the binary-size gate — the premise it was filed on was false, and V0 has landed.** The card said `query_streaming_agg_fused` sat at +0.449% of 0.5% and that V0's `MapScalar` (+0.137%) could not fit. Investigating before resetting anything showed the gate was actually **+55%**, and not from accumulated drift: `f5226d5`'s closure migration cost +739,316 bytes on this gate while its own commit message and CHANGELOG recorded "+0.36%". The Mojo 1.1 / MAX 26.6 upgrade accounted for +0.39% of it, GPU gating was intact. §0 has the attribution; the fix recovered 662,740 bytes (89.6%). **V0 landed 2026-08-17** at its measured +0.136%, so what is left is only the reset: regenerate `baseline.json` on the post-fix tree with `check_gate.py --update`, keep `threshold_pct` at **0.5** (do not raise it — it caught this), and write the history into the `_comment` in the style of the existing entries: what the old numbers were, that the reset absorbs the ~4.8% residue that is *not* attributable to `f5226d5`, and that the gate is not in CI, which is why a +55% regression survived for ten commits. Fixing that last part is the durable lesson, not the reset. | §0.5 Must | `pixi run binary_size` | S |
| **S19** | **A GPU-off binary links `libAsyncRTMojoBindings.dylib`, and — measured 2026-08-19 — it cannot stop doing so. Recommend closing as Won't.** The card claimed 10 `_AsyncRT_*` symbols survive DCE, all of them `DeviceBuffer`/`DeviceContext` device calls, and that eliminating them would let `-Xlinker -dead_strip_dylibs` drop the dylib for **1,156,592 bytes, 42% of the runtime dylib closure**. Four things were checked by direct experiment; the first three confirm the mechanism and the fourth kills the win. (1) It is **14** undefined symbols, not 10 — eleven `_AsyncRT_Device*` plus three `_KGEN_CompilerRT_AsyncRT_*CPUDevice`. (2) The three `_KGEN_CompilerRT_*` do not matter: `def main(): print("hi")` pulls all three, yet `-dead_strip_dylibs` drops the dylib from it, so they resolve elsewhere. (3) The `Optional[DeviceBuffer]` / `Optional[HostBuffer]` / `Optional[DeviceContext]` **fields are not the floor** — the obvious objection, that Mojo has no conditional struct fields so `Allocation.__deinit__` must always emit `_AsyncRT_DeviceBuffer_release`, is **false**: a probe struct holding `Optional[DeviceBuffer[uint8]]`, constructed and destroyed, emits no `_AsyncRT_Device*` symbol and drops the dylib, because the optimizer proves the slot is always `None`. **(4) But six of the eleven are not GPU code at all — they are the CPU thread pool.** A program whose entire content is `sync_parallelize(body, 4)`, importing no marrow and no GPU module, pulls `_AsyncRT_DeviceContext_{create,deviceApi,enqueueHostFunctionRange,release,strfree,synchronize}` and **keeps the dylib under `-dead_strip_dylibs`**. `sync_parallelize` is what `ExecContext.stripe` / `views._cpu_striped` are built on, so marrow cannot shed the dylib without giving up parallel CPU execution. Only five symbols are marrow's own (`DeviceBuffer_{context,release,retain}`, `DeviceContext_{createHostBuffer,retain}`) and removing them cannot drop the dylib on its own. **Also ruled out:** gating `DynArray.to_device` / `to_cpu` — the erased entry points, and the obvious reachability root since `_dispatch` instantiates every arm — behind `comptime if GPU_ENABLED` changed the symbol count by **zero** (tried and reverted; it restricted a public API for nothing). CLAUDE.md's note that gating `GPU_ENABLED` "does not shed the `libmax`/AsyncRT runtime dependency" is therefore not a gap to close but a property of how Mojo implements CPU parallelism. Reopen only if `max.algorithm` gains a thread pool that does not route through `DeviceContext`. | re-measured 2026-08-19 | `nm -u` + `otool -L` probes | Won't — premise measured false |
| **S20 — FIXED 2026-08-23** | **`SwissHashTable.probe` reached the erased `filter`, costing 450,112 bytes on `query_join` (+29.7%).** Found 2026-08-23 by the first run of the new `expr2` gates, which failed the *existing* gates. Bisected over 183 commits to **`6c570eb`** ("fix(kernels): a NULL join key matches nothing, and bool keys are supported", 2026-08-20): parent `b79087a` measures **1,513,404**, `6c570eb` measures **1,963,516**. The fix itself is correct and must not be reverted — a NULL key matching another NULL was a real defect and it cleared four golden `-- xfail` markers. **What costs the bytes is how the rule is applied.** `hashtable.mojo:648-649` calls `filter(build_indices.copy().to_dyn(), verifier)`; `build_indices`/`probe_indices` are `Int32Array`, but `filter` exists **only** as a `DynArray` free function (`filter.mojo:1102`) — the typed layer is `Filter.apply` — so `.to_dyn()` instantiates the whole per-dtype ladder and makes it reachable from every binary that joins. Symbol buckets confirm it: `marrow::kernels::filter` 98 → 121, and it drags `marrow::views` 112 → 148, `marrow::arrays` 289 → 315, `marrow::execution` 237 → 258. **This is the exact shape §0 already records once** — Q4.7, where hashing decoded dictionary keys via `cast` and made all of `kernels.cast` reachable from every binary that hashes, for ~2.4 MB. **Proposed fix:** the null rule is just *select where data AND valid*, so fold validity into the mask and stay on the typed path — `Filter.apply(build_indices, combined)` where `combined` is `mask.values()` AND'ed with `mask`'s validity bitmap (`Bitmap` already exposes bitwise ops; do not hand-roll the loop). That keeps the corrected semantics, instantiates `Filter.apply` for `Int32Type` alone, and should recover most of the 450 KB. Verify with `pixi run binary_size` **and** `marrow/kernels/tests/test_join.mojo` plus the four golden join cases. **Do not clear this by re-baselining** — `--update` would erase the signal, which is the failure §0 already documents. **Fixed as proposed**: validity is ANDed into the mask and selection goes through the typed `Filter.apply`. `query_join` 1,967,052 -> 1,527,820, recovering **439,232 bytes**; +30.455% becomes +1.325%. The residue is the bool-key arm from the same commit, which is new functionality. Verified by 49 `test_join.mojo` cases and 32 golden join cases. | found by the Tier 1.1 gates | done | S |
| **S21 — FIXED 2026-08-23** | **`expr2` could not compare a temporal column, so it could not filter one.** `TemporalColumn` lands 2026-08-23 and projects and groups correctly, but `NumericCompare` is still bound on `NumericValue`. The blocker is **not** the bound: `NumericCompare.ArgType` is `promote[L.Type, R.Type]`, and `promote` (`expr2/comptime/rules.mojo:64`) is bound on `NumericType` because it encodes *numeric* widening — signedness and int-to-float. `wider[L.native, R.native]` is **not** a substitute: it picks by width and would silently change what `int32 < float32` compares in. Generalising `promote` is a decision about **promotion semantics**, and the interesting case is not width but **unit coercion**: `timestamp[s] < timestamp[ms]` needs one side *scaled*, not widened, and `date32 < timestamp[s]` needs a common unit chosen. Arrow C++ resolves this in `common_temporal_resolution` / `CastTemporal`; consult it before inventing rules. Until then `WHERE d > '2020-01-01'` is unexpressible in the fused lane, which blocks ClickBench (its `hits` table is date-filtered). The runtime lane is unaffected — `col("d")` compares fine, because it resolves dtypes at run time. **Fixed by `TemporalCompare`, a separate node rather than a generalised `promote`.** Generalising promotion was the wrong lever: the temporal question is *unit*, not width, and inventing coercion rules was not required to unblock filtering. `TemporalCompare` compares only operands that already share a representation — a `comptime assert` catches a width mismatch, and `bind` compares the dtypes once per batch to catch a *unit* mismatch the widths agree on (`date32` and `time32[s]` are both int32). Cross-unit comparison raises with a named error instead of silently comparing raw integers; adding it still means choosing coercion rules, and Arrow C++'s `common_temporal_resolution` remains the prior art. | found while landing TemporalColumn | done | M |

**Order.** S1 landed first, 2026-08-17 — it was the conflict-heaviest change and
every later `expr` edit would otherwise have been rebased onto it. Then the
correctness group S2 → S3 → S4. Then the free subtraction batch —
everything except S2–S4 and S17 — which is independent and can land in any
order. S17 last. Rows are deleted as they land, so the batch develops holes; do
not read the remaining numbering as a range.

### Alpha wave leftovers — open, 2026-08-18

Found while building the Python lazy frontend and the optimisation wave. Each is
diagnosed with a named cause; none is speculative.

The `Evidence` column cites the per-agent log it came from. Those twenty logs
(5,201 lines) have been **folded into this file and deleted** — their open items
are the `A`-IDs below, their measurement traps are in §0, their ruled-out
designs are in §7, and their defend-this findings are in §8. To read one:
`git show c0831f5:docs/alpha-findings/README.md` (likewise `a1-null-ops`,
`c1-binary-groupby`, `f1-distinct-segfault`, `p1-pushdown`, `o1-codec-caching`
… — the README indexes all twenty).

| ID | Item | Evidence | Size |
|---|---|---|---|
| **A-4** | **`COUNT(*)` has no representation and materialises a column.** `count_star()` is `lit(1).count().alias("count_star")`, so `function()` returns `"count"` and the marker is a default alias the first `.alias()` erases — a plan cannot be inspected for it, so reading the row count from the Parquet footer is unavailable. Measured 5.4% of q1: it allocates an N-element array of `1` to count rows the grouper already counted. Root cause is A-2's shape — `DynAgg.input` is mandatory, so a nullary aggregate must lie about having one. | `a1`, `d1`, `alpha-perf-baseline.md` | S |
| **A-5** | **`JoinProcessor` streams 8192-row morsels; probe cost flattens at ~256k.** Per-row probe falls **23.7 -> 3.3 ns/row** from 8192 to 262144 rows/call. Raising the morsel would cut per-row cost ~7x and erase the probe-only shortfall that remains after the radix fix. Plan-layer change, deliberately not made by the kernel agent. | `o5-join-threshold.md` | S |
| **A-6** | **The runtime lane splats a constant string literal to n rows.** `DynValue._literal` builds an n-row copy of e.g. `"%google%"` per morsel (4.2% + 2.9% self). `dynamic.mojo::_string_binary` should detect a literal right operand and call `apply_scalar`, which every `StringPredicateKernel` already has and `values.mojo` already does. Helps `startswith`/`endswith`/`contains` and string comparison too. | `o3-string-alloc.md` §6 | S |
| **A-8** | **`BitmapView.load_bits` over-reads 3-7 bytes past a bitmap.** Benign for heap integrity today (out-of-range bytes are written back unchanged) and now assertion-bounded, but optimistic for FOREIGN buffers, which the producer allocated and which the spec does not require to be padded. Tapering costs a per-lane branch; recommendation is to copy imported bitmaps that land exactly on a 64-byte boundary. | `g1-buffer-invariants.md` | S |
| **A-10** | **The binary-size baseline is stale in both directions.** `check_gate.py` compares to `baseline.json`, not to the branch under test, and four of five recorded values sit above the current tree — so gates read as "shrinking" while the branch-to-branch measurement showed every gate **grew** ~16 KB (~15.3 KB of it C1's builder fix, i.e. the price of fixing a process-killing abort). A gate that passes is not the same as no regression. Re-record deliberately, as a decision, not as a side effect — an agent attempted the latter and it was reverted. | `g3-regression-check.md` | S |
| **A-11** | **A3's temporal branch is held out of the alpha.** `worktree-agent-af8dec5bed6238e2e` adds working `uint16 -> date32` and ISO-8601 string<->temporal casts, but pushes `query_dynvalue` +113,472 bytes (+2.33%) against a 0.5% budget. Not on any ClickBench query's path. The residual is structural — `cast()` is one ladder reachable from `_promote_operands`, so any new cast kernel taxes every binary that promotes operands. Merge once that ladder is split (its own §1 finding). | `a3-temporal.md` | M |

**Not ours:** `mojo-regex`'s optional-group defect is upstream and appears
unreported. Filing it would be a courtesy; nothing has been sent.

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
| **FU-6** | `sort_indices` without key re-gather. `SortIndices.multi` exists but there is no public `sort_indices(StructArray, key_indices)`; `SortProcessor` still calls `sort_by_keys` and discards key fields, and `sort.mojo:507` re-gathers keys on every pass. | M |
| **FU-7** | **(b) re-sized 2026-08-16, and it is not S.** `IsIn._value_set: DynArray` is not a stray payload — it is a *runtime-dtype dispatch* (`IsInKernel.dispatch`) inside an otherwise comptime-typed lane. Removing it means parameterising `IsIn` over the value-set's type, while `IsIn[A: Value]` today accepts any family and is used for both numeric and string sets. That is a design change, **M–L**, not a cleanup. **(d)** `ConditionalBinary` is 2-ary and `CaseWhen` 1-branch/numeric-only while the kernels and runtime builders are variadic — L, open. | (b) M–L, (d) L |

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

### Simplification wave (2026-08-17)

- **`equal_any` → a neutral `kernels/compare.mojo` — rejected (D2).**
  `numeric.mojo:49` imports `StringEqKernel` solely for `equal_any`, so the move
  *does* delete the `numeric → string` edge. But `EqKernel.apply(StructArray, …)`
  (`:617-645`) calls `equal_any` twice and **cannot move**, because Mojo cannot
  add a static method to `EqKernel` from another module. The result is a
  `numeric ↔ compare` **cycle replacing an acyclic edge** — worse, immediately
  after S1 finished removing cycles. Revisit only if the `StructArray`
  row-equality relocates for an independent reason.
- **Amended (D1): the rejection of `Buffer.alloc_for[T](ctx, n)` in
  `buffers.mojo` is overruled.** The original row rejected it for pointing the
  tree's lowest-level module at device policy. It was re-opened on 2026-08-17
  because the `ExecContext` alternative turned out to close
  `execution → buffers → views → execution`, which the original row did not know.
  A rejected-designs list that is quietly contradicted is worse than no list, so
  the reversal is recorded here rather than left implicit.

### Alpha wave (2026-08-18)

- **`mojo-regex` — rejected on correctness, not on version drift.** It builds
  against our pinned Mojo (13/14 of its own tests pass) so the expected blocker
  was not the real one. It **never enters an optional group**: `(?:www\.)?` is
  skipped, so Q29 returns `www.example.com` where pyarrow and CPython return
  `example.com`; 10 of 25 `sub()` cases disagree with CPython. Minimal repro:
  `sub("(?:foo)?bar", "B", "foobar") -> "fooB"`. It survived upstream because
  their tests use `(?:` seventeen times and never with a trailing `?`. Adopting
  it would have converted an honest 42/43 gap into a **silently wrong answer on
  the majority of rows**, since `www.`-prefixed referers dominate `hits`. See
  M2.6 for the replacement.
- **A hand-written `extract_host` for Q29 — rejected.** It buys a DEVIATED row,
  generalises to nothing, and gets deleted the moment real regex lands.
- **Over-allocating buffers so "64-byte padded" becomes literally true —
  rejected, and the premise was wrong anyway.** `Columnar.rst:264-273` says pad
  "to a length that is a multiple of 8 or 64 bytes", i.e. *round the size up* —
  exactly `align_up(bytes, 64)`, which is byte-for-byte Arrow C++'s
  `PoolBuffer::RoundCapacity`. `arrow::AllocateBuffer(64)` allocates 64 bytes
  with zero slack. Marrow was already conformant; the false step was inferring
  that padding implies slack past the logical end. Over-allocating was rejected
  because **it cannot deliver the invariant**: FOREIGN buffers come from the
  producer, and pyarrow allocates exactly 64 bytes for a 512-row bitmap — the
  guarantee would hold on half the buffers and make the other half harder to
  find. Memory cost was explicitly *not* the deciding argument.
- **Rewriting `DynArray.__eq__` to dispatch once — tried, reverted.** O(n) arms
  instead of O(n²) and exactly as strict, and it compiles; but the hang in §2 is
  recursion through `ListLikeArray.__eq__`, not the squared ladder, so it fixes
  nothing. Not left in as an unmeasured change to a hot, size-gated file.
- **A fan-out threshold on probe rows for the parallel join — tried, measured,
  removed.** It would have *doubled* the regression: within the partitioned
  layout, fanning out beats serial partitions at **every** size, 8192 rows
  included (194 us vs 388 us). The expensive thing is the partitioning, not the
  `sync_parallelize`. What fixed it was `_DEFAULT_RADIX_BITS` 6 -> 4 — the
  64-partition default came from a *one-shot* 10M sweep, and morsel streaming
  pays it per call.
- **`inputs()`-based optimizer traversal — rejected for `children()` +
  `with_projection()`.** See M1.1.
- **Building a temporal literal through the storage integer and relabelling with
  `relabel_array` — rejected on measurement.** It avoids six
  `PrimitiveBuilder`/`PrimitiveArray` monomorphisations (+23,668 bytes on
  `query_streaming`) but drags in `DynArray.from_data`'s 30-arm ladder that these
  binaries do not otherwise link: **+106,276 bytes, 4.5x worse**. The trick is
  right in `kernels.cast`, where `from_data` is reachable anyway; it is wrong in
  `scalars.mojo`.

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
  What was reachable landed instead (2026-08-17): the shared validity helper
  was inlined into all seven `__eq__` bodies, which removed two popcounts and
  a bit-by-bit loop rather than removing lines.
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

### Confirmed sound by the alpha wave — defend, do not "simplify"

- **The `AggFunc` / `AggFold` / `FoldedAggregates` split is justified.** Two
  agents checked it independently and both concluded the binary-size
  measurements behind it are real (+3.2 MB / +1.2x on the aggregate gate if
  collapsed). The complaint is only that **nothing in the names says so**. Do
  not merge these. The genuinely redundant carriers are `DynAgg` and `AggExpr`
  (§5, A-2).
- **The closed-erasure / DCE property holds.** `marrow::expr::dynamic`
  contributes **0 symbols** to every fused/AOT target, so building comptime
  expressions still does not link the interpreter. `DynAgg`'s string-tag
  dispatch (A-2) is confined to the runtime-lane targets, which are the
  interpreter by definition.

### Binding-layer constraints discovered 2026-08-18

- **`add_type[T]` rejects any struct holding a function pointer.** It installs a
  default `tp_repr` calling `repr()`, and deriving `Writable` reflects over every
  field: *"Could not derive Writable for DynValue - member field `_eval_fn` does
  not implement Writable"*. `DynRelation` fails identically through
  `_virt_with_predicate`. So **any marrow struct that grows a function-pointer
  field becomes silently un-bindable** — `AggFunc._grouped_fn` and all three
  `AggFold` fields are already in that position. The binding layer works around
  it with one-field boxes (`Expr`/`Agg`/`Plan`); a two-line `write_repr_to` on
  each `Dyn*` box would delete all three.
- **`def_method` fills `tp_dict`, not the CPython slots.** Measured on the built
  `.so`: `e.__str__()` returns `'a'` while `str(e)` returns the derived repr. So
  operator dunders defined at the Mojo layer are dead weight — this is why the
  bindings expose named methods and pure-Python `Column` maps them onto dunders.
  The hazard is latent in every bound type that registers `__str__`, and **a
  substring assertion cannot detect it** (one shipped that way and was fixed).
- **`DynScalar` is `ConvertibleToPython` but not `FromPython`**, so `lit(3)` has
  to allocate a one-element Arrow array to reach a scalar.


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
- **`DynRelation`** (`relations.mojo:413`) — the erasure box **and the entire
  plan builder/binder** (`:643-1178`: schema derivation, dtype probing, join
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
