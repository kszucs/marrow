# Backlog

The single source of truth for what is open. Everything here was verified
against the code at **`b2e7dae` (2026-08-03)**, not read off a header.

**Goal: marrow is a usable single-node columnar query engine.** Arrow-spec
completeness is not the objective — layout and kernel gaps are scheduled only
when a milestone query needs them. The deferred Arrow-parity list is §7.

Resolved items are **deleted, not struck through** — git history has them.

> **Re-verify before trusting any status line here.** This file replaces seven
> task documents that had drifted so far apart that a 2026-07-30 consolidation
> pass still left the index and `tasks-code-quality.md` disagreeing about five
> tasks. A 2026-08-03 audit found **18 wrong statuses**: eight tasks marked open
> that were done, four marked done that were not, and six whose premise the
> two-lane refactor had destroyed. Check with `grep`, not with a header.

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
   on the type or trait, not in free functions.

### Do not change

- **Array, scalar and builder layout.** Adding methods and accessors is fine;
  adding, removing, reordering or re-typing fields is **out of scope, not
  deferred**. This is why B8 and B9 are recorded as accepted defects.

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

Nine defects that produce **wrong answers with no error**. None has a test.
These come first because the M1 gate is "results cross-checked against DuckDB":
a wrong multi-key sort or an unpruned date predicate corrupts exactly the thing
the milestone measures.

**Every item opens with a failing test.** All nine were found by reading code
paths; only B9 has been reproduced end-to-end.

| ID | Defect | Evidence | Size |
|---|---|---|---|
| **B1** | **Multi-key sort can return a wrong order.** `SortIndices.apply[T: PrimitiveType]` accepts `stable` and never forwards it — `_sort_valid` has no such parameter, so below `_RADIX_THRESHOLD = 32_768` the unstable stdlib PDQsort runs. `SortIndices.multi` is a column-oriented LSD sort that passes `stable=True` and *depends* on that stability ("every pass is stable, so a less-significant key's order is preserved"). Ties in a more-significant key therefore scramble. Decimal128/256 take the comparison path at any size. Existing tests use N=5, inside the stdlib insertion-sort cutoff, and pass by accident. | `sort.mojo:529`, `:553`, `:561`, `:188`, `:499-513`, `:575` | S |
| **B2** | **Compressed IPC decodes to garbage, silently.** `_read_record_batch_meta` reads FlatBuffer slots 0/1/2 only; `RecordBatch.compression` is slot 3 and is never inspected, so LZ4_FRAME/ZSTD bodies are read as raw buffers. Minimum fix: detect and raise. Full fix: implement both codecs (they are already `dlopen`ed for Parquet). | `ipc.mojo:1354-1384`, writer `:709-723` | S to raise, M to implement |
| **B3** | **Delta dictionary batches truncate instead of append.** `isDelta` (DictionaryBatch slot 2) is never read and dictionaries are unconditionally overwritten. | `ipc.mojo:1269-1276`, `:2191`, `:2263` | S |
| **B4** | **BIT_PACKED Parquet levels are mis-decoded.** `definition_level_encoding` / `repetition_level_encoding` are parsed then never consulted — `_data_page_v1` applies `Rle.decode` unconditionally. | `format.mojo:575-578` vs `reader.mojo:244-270` | S |
| **B5** | **Outer/semi/anti joins are wrong across multiple probe morsels.** `JoinProcessor.pull` calls `probe()` once per right-side morsel, and `_emit_unmatched` recomputes `matched_build` from *that morsel's* pairs alone. A build row matched only in morsel 2 is still emitted as unmatched by morsel 1: LEFT/FULL/ANTI over-produce, SEMI under-produces. Only single-morsel inputs are tested. Fix: hoist the matched bitmap into `JoinProcessor` and emit unmatched rows at exhaustion. | `execution.mojo:874-895`, `join.mojo:587-632` | M |
| **B6** | **`JOIN_MARK`'s declared schema disagrees with its columns.** The "emits right columns?" predicate is re-derived three times with different membership: `join.mojo:682` excludes MARK, `join.mojo:646` (`output_dtype`) and `relations.mojo:615` do not. A fourth copy parses strings at `tabular.mojo:265`. Fix by introducing the `JoinKind` value type (was Q4.1) rather than patching three sites. | `join.mojo:646`, `:682`, `relations.mojo:615` | S |
| **B7** | **`DictionaryBuilder.finish()` silently drops `ordered`.** The builder stores the flag, then calls `from_arrays(indices, values)` whose `ordered` defaults to `False`. | `builders.mojo:1428`, `:1482-1486`, `arrays.mojo:1902` | XS |
| **B8** | **`BoolType` is not a `PrimitiveType` while `is_primitive()` returns True for bool.** `DynType.byte_width()` guards on `is_primitive()` then calls `variant_dispatch[PrimitiveType]`, which **aborts** for bool. Latent — every current caller happens to branch on `dt == bool_` first, and `test_dtypes.mojo:116` skips bool. | `dtypes.mojo:184`, `:1062`, `:981-988` | S |
| **B9** | **The built wheel is unimportable.** `build.py` force-includes only `marrow/__init__.py` and the `.so`, but `__init__.py:492` does `from . import compute`. Neither `compute.py` nor `parquet.py` is packaged. Confirmed against the checked-in `python/dist/marrow-0.1.0-cp314-*.whl`. | `python/build.py:41-45` | XS |

**Also small and worth clearing in this wave:**

- **B10** — `Array.is_valid()` can never work: the Python wrapper calls
  `self._binding.is_valid()` with no argument, but the binding wraps
  `DynArray.is_valid(self, index: Int)`. It is also semantically wrong versus
  PyArrow, where `is_valid()` returns a BooleanArray.
  (`python/marrow/__init__.py:168`, `bindings/arrays.mojo:1156`) — XS
- **B11** — `arrays.mojo:366` `BoolArray.write_to` carries the same double-offset
  bug as B13 below. Unlike B13 it needs no layout change. — XS

**Accepted, blocked by the layout freeze** (documented, not scheduled):

- **B12** — `slice()` reports the parent's null count; `nulls=self.nulls` in eight
  `slice()` bodies (`arrays.mojo:350, 576, 803, 1048, 1339, 1507, 1758, 1988`).
- **B13** — indexing a sliced `BoolArray` double-applies the offset:
  `self.values().test(self.offset + index)` where `values()` already returns an
  offset-applied view (`arrays.mojo:387`).

---

## 2. Wave 2 — Infrastructure

**CI has not run since 2026-05-11**, and on that last run everything except Lint
was already failing. None of the work below is verified by anything but local
runs. This wave is cheap and it is what makes every later wave believable.

| ID | Item | Evidence | Size |
|---|---|---|---|
| **I1** | **The main test job cannot run.** `test.yml:29` and `:50` invoke `pixi run -e dev test_parallel --no-gpu`; that task was removed in `2aa1954` ("one compilation unit per selection") and `pixi run -e dev test_parallel` exits 127. Both the linux and macos jobs fail at that step, and `release.yml` gates on them. Fix: `test`. Note `test_mojo_asan_parallel` still exists, so the asan job is fine. | `.github/workflows/test.yml:29,50`; `pixi.toml:54-80` | XS |
| **I2** | **The docs site does not build.** 7 of 10 executed pages raise. Root cause: `81fa29a` moved every compute function to `marrow.compute` and no page was updated, so `ma.add`, `ma.sum`, `ma.filter`, `ma.sort`, `ma.greater`, `ma.cast`, `ma.take`, `ma.sort_indices`, `ma.drop_null` are all gone. Two further breaks: `is_valid(1)` arity (see B10) and `sort(ascending=…)`, which is now `sort(input, sort_keys=(), *, null_placement=…)`. `_freeze` is gitignored so nothing masks it. | `docs.yml`; `guide/compute.qmd` (18 of 20 cells fail) | S |
| **I3** | **The binary-size gate is not in CI at all** — zero hits for `binary_size` under `.github/`, despite it being the project's central architectural invariant. It is enforced only by hand. Add a job that runs `pixi run binary_size` and fails on a `__text` regression beyond a threshold. | `.github/` | S |
| **I4** | **Re-baseline the binary-size numbers.** The recorded 7.6× / 7.8× / 12.8× table predates the interpreter deletion; a local sweep gives 2.83× / 3.12× / 3.01×. Using the written numbers as a gate would invent or hide a regression — the exact failure the docs warn about. Also: `BASELINE.md` is referenced three times and **does not exist**. | `benchmarks/binary_size/README.md:110,154` | S |
| **I5** | **`benchmarks/binary_size/README.md` is half-updated** — it documents `query_comptime.mojo`, `query_erased_aot.mojo` and `query_hybrid.mojo`, none of which exist, and references `marrow/aot/relations.mojo` and `Planner.build()`. The newer table at `:75-84` is current. | as cited | XS |

**Not scheduled, but know the gaps:** GPU is never exercised (every job passes
`--no-gpu`; the five `*_gpu.mojo` files, 39 cases, never run). ASAN on Linux is
hard-disabled (`test.yml:55 if: false`). `precompile` — the warning-clean gate —
is not in CI. There is no separate `test_python` job.

---

## 3. Wave 3 — M1, the ClickBench milestone

**M1 = 42 of the 43 ClickBench queries** (Q29 `REGEXP_REPLACE` deferred to M2)
over the single flat `hits` table, run through marrow's frontend, results
cross-checked against DuckDB, with the binary-size gate green.

Today `python/marrow/tests/clickbench.py` is **11 queries, eager, PyArrow doing
the I/O**, restricted to six numeric columns and GROUP BY. Its own docstring
concedes the restriction.

This path is almost entirely sequential and is the whole critical path.

### M1.0 — Widen the numeric dispatch bound *(do first; smallest change, largest blast radius)*

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
**Size: S. Do it before the optimizer** — pruning correctness is a prerequisite
for measuring the optimizer.

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

### M1.4 — Kernel gaps M1 actually needs — **M**

- `date_trunc` at **month, quarter, year** — only second/minute/hour/day exist
  (`temporal.mojo:332-361`). Without these, ClickBench Q35/Q36 fail as queries
  rather than as compile errors.
- `regexp_replace` for Q29 — **or** formally defer Q29 to M2 and record it.

### M1.5 — ClickBench through the lazy plan — **M, the M1 sign-off**

Rewrite `clickbench.py` against the lazy frontend, all 42 queries, cross-checked
against DuckDB, wall-clock compared to polars/duckdb on the same box, binary-size
gate green.

### M1.6 — AOT DSL docs and a runnable example — **S**

`docs/guide/expressions.qmd` exists but documents a `Planner` type (zero hits),
`from marrow.expr import execute` (it is a method, `relations.mojo:393`),
`AnyValue`, `Binary(ADD)` node types, and a `parquet_scan("f.parquet")` that
requires a `schema`. It predates the two-lane split entirely and must be
rewritten, not patched. **The two-lane split, `BoxedValue`, and the
lane-choice/binary-size trade-off — the central design fact — are undocumented
anywhere user-facing.**

---

## 4. Wave 4 — M2 and M3 enablers

**M2 = H2O.ai db-benchmark** (10 group-by + 5 join queries at 0.5/5/50 GB, with
spill). **M3 = TPC-H** (22 queries, join-heavy). Scheduled here are only the
capabilities those milestones require.

| ID | Item | State | Size |
|---|---|---|---|
| **M2.1** | **`Distinct` and `Union` relational nodes.** Neither exists; `relations.mojo` has 8 nodes and no `RELATION_*` discriminant for either. Needs a `unique` kernel (below). | not started | M |
| **M2.2** | **`unique` / `value_counts` / `dictionary_encode`.** `distinct.mojo` is cardinality-only. **marrow consumes dictionaries but can never produce one** — not from a kernel, and not from Parquet, whose reader materialises dictionary pages into plain arrays. | not started | M |
| **M2.3** | **Real window functions.** Today: `row_number` only, **AOT lane only** (violating invariant 2), `WindowSpec` carries frame bounds but no PARTITION BY and no ORDER BY, `FrameBound.kind` is an untyped `UInt8` never read, `RowNumberKernel` ignores its `values` argument, and nothing outside `values.mojo` references any of it. Sequence: move to `marrow/kernels/window.mojo` → give `WindowSpec` `how`/`partition_by`/`order_by` → partition (reuse `groupby` hashing) + sort within partition (reuse `sort`) + scatter back → ranking family → navigation family (`Lag`/`Lead`/`NthValue`) → `RunningAgg[K: AggKernel]` → `.over()` on both lanes → wire through `relations.mojo`. `docs/lane-shape-window-design.md` §7 is a competent design for this and nothing supersedes it. | 2-node toy, `values.mojo:1975-2039` | L |
| **M2.4** | **Statistical aggregates** — `variance`, `stddev`, `quantile`, `approximate_median`, `mode`, `first`, `last`. `resolve_agg` is a closed list of exactly 8 (`expr/aggregates.mojo:194-221`). TODOs already acknowledge the variance gap at `aggregate.mojo:563,574,589`. | not started | M |
| **M2.5** | **Spill.** Zero occurrences of `spill` in the tree; no memory-budget tracking and no disk I/O anywhere. Required by H2O at 50 GB. Grace hash join and a spilling grouper. | not started | L |
| **M2.6** | **String manipulation and regex — the single largest kernel hole.** There is no regex engine in the repo. Missing: `match_substring_regex`, `replace_substring(_regex)`, `extract_regex`, `split_pattern(_regex)`, `count_substring`, `find_substring`, `utf8_slice_codeunits`/substring, `lpad`/`rpad`, `binary_join`, the whole `utf8_is_*` classification family, trim-with-charset. Also: string kernels dispatch on `is_string_like()` only, so `binary`/`large_binary` are excluded from string comparison. | not started | L |
| **M2.7** | **Temporal completeness** — `strftime`/`strptime` (and **string↔timestamp cast raises**, `cast.mojo:1028`), timezone-aware extraction (everything decomposes as UTC and a non-UTC `tz` is silently ignored, `temporal.mojo:36-39`), `week`/`iso_week`/`iso_year`, `millisecond`/`microsecond`/`nanosecond`, `is_leap_year`, `ceil_temporal`/`round_temporal`, and the `*_between` family. | not started | M |
| **M2.8** | **Multi-file / dataset scan.** `ParquetScan.path` is a single `String`. No glob, no dataset, no partitioning. Also: **bloom filters are fully implemented in the reader and never consulted by the scan** (zero `bloom` hits in `marrow/expr/`) — cheap win, do it with this. | not started | M |
| **M3.1** | **Join completeness** — `JOIN_CROSS`, `JOIN_MARK`, `JOIN_SINGLE` and `JOIN_ASOF` are declared constants that are never implemented; a CROSS join currently falls into the LEFT/RIGHT/FULL tail and produces wrong output rather than a cartesian product. All five `JOIN_ALGO_*` constants are dead and `struct Join(Relation)` has no `algorithm` field. Sort-merge join is a commented-out stub (`join.mojo:697-709`). | constants only | L |
| **M3.2** | **Join reordering and build-side selection.** `hash_join` always builds on `left` (`join.mojo:754`); `JoinProcessor` always builds on `self.left` (`execution.mojo:878`). No cardinality estimation feeds the plan. | not started | L |
| **M3.3** | **`JoinProcessor` discards the plan's execution context.** `Join.to_processor(ctx)` receives a ctx and constructs `JoinProcessor` without it, which then builds `HashJoin[rapidhash]()` with the default — **the relational join never uses the parallel path**. Small, high-value, could move to Wave 1. | `relations.mojo:1114-1123`, `execution.mojo:879` | S |
| **M3.4** | **O(N) top-K.** `sort_indices(limit=…)` is a full sort then truncation; the docstring concedes it (`sort.mojo:379`). No `select_k_unstable`, no quickselect, no streaming heap. Directly relevant to every ORDER BY … LIMIT in ClickBench and TPC-H. | not started | M |
| **M3.5** | **`Scan` trait above the file formats** (was L6). `RELATION_PARQUET_SCAN` is an IR discriminant; `execution.mojo:37-43` imports six symbols from `..parquet`; there is no `trait Scan`/`trait Source` — only `trait ByteSource`, a *byte*-level seam. `ParquetScanProcessor` does four jobs. **Do this before adding CSV or IPC sources, not after.** | not started | L |
| **M3.6** | **`Table` and `ChunkedArray` are thin.** `ChunkedArray` has only `chunk()`/`combine_chunks()`; `Table` has no `slice`/`select`/`filter`/`sort`/`concat_tables` — all of those are `RecordBatch`-only in the Mojo core too. | not started | M |

---

## 5. Quality debt

Surviving items from the Q/L/V backlogs, with their original IDs kept so git
history and CHANGELOG references still resolve. Everything else in those files
is done and has been deleted.

| ID | What is left | Size |
|---|---|---|
| **Q0.5** | `project`/`aggregate` derive output dtypes by executing against a 0-row batch (`relations.mojo:485,499,559,562,669,672`). A fused value's `OutType` is statically known, so this is unnecessary for that lane; ~16 KB, and it retires a hand-built `benchmarks/binary_size` exception. **Reword before scheduling** — the card says "`DynValue` should answer its dtype", but the box is now `BoxedValue`. | M |
| **Q2.3** | Validity plumbing. `bitmap_and` is gone and `Bitmap.intersect(Optional[Bitmap], Optional[Bitmap])` replaced it — but **the named defect survives**: kernels still pass the raw offset-unaware `.bitmap` (`numeric.mojo:95`, `:502`, `string.mojo:314`), yielding misaligned validity for sliced inputs. Only the expr layer passes `.validity()`. | S |
| **Q2.4** | One `sync_parallelize` loop left to convert: `views.mojo:1809` (`_reduce_dispatch`, a wid-shaped loop). The other three are deliberate and documented. | S |
| **Q2.5** | Aggregates, remaining steps: **2b** — `AggState[K, V: NumericType]` was never widened to `PrimitiveType`, so temporal reductions do not work natively; **3** — `count_distinct`/`approx_count_distinct` (+`_grouped`) are still four free functions (`distinct.mojo:87,133,167,214`); **4** — `FusedAggregation` (single pass, AoS accumulator, comptime offsets, zero dispatch) has zero occurrences. Gated on Q6.1. Validate at `g100k`, never `g10`. | L |
| **Q4.1** | Missing value types: `Grouping` (`(gids, num_groups)` is still two parameters in eight places), `JoinKind` (**pull forward — it fixes B6**), `JoinIndex`, `BuildPartition`. | M |
| **Q4.3** | Parquet leaf visitor. **Zero uses of the `dispatch_*` family anywhere in `marrow/parquet/`**; hand-written runtime ladders at `statistics.mojo:302-377` (22 arms) and two near-identical 13-arm ladders at `writer.mojo:122-151` and `:249-273` — the duplicated writer pair is the highest-value target. The reader's 28-arm comptime-gated ladder is deliberate. | M |
| **Q4.4** | `ipc.mojo` → package. Single file, 2,342 lines. | M |
| **Q4.5** | **Needs a test, and a comment deleted.** The fused `prune` overrides are live (`values.mojo:596,645,960,1039`; `BoxedValue.prune` delegates at `relations.mojo:264`), but all seven cases in `test_pruning.mojo` build **runtime** predicates, so the fused path is unasserted. Worse: `test_pruning.mojo:15-19` asserts the fused overrides "were not ported" and that `Value.prune` returns unknown — **that is factually false**. Cheapest item on this page. | S |
| **Q4.6** | A Parquet scan still nearly doubles a minimal AOT binary: `query_scan_stripped` 2,449,024 vs `query_streaming_stripped` 1,373,704 = **1.78×**. | L |
| **Q4.7** | **`kernels::cast` is roughly 20% of the fused binary, pulled in by `from .cast import cast` at `kernels/hashing.mojo:44`** — about 797 symbols reachable from any binary that hashes. `docs/aggregate-kernel-inversion.md` §5 calls this the only finding that meaningfully moves the size gate; the `rapidhash`/`sort_indices` ladder collapse took −66 KB off the *erased* gates and **exactly 0** off the fused ones, so it did not touch this. Give hashing a comptime key spec so it does not reach the cast fanout. (This is the surviving half of the old Q1.1, whose dispatch-family half is done.) | M |
| **Q5.1** | Doc drift in code: `expr/pruning.mojo:14,61` and `relations.mojo:167` name `TagValue`, deleted three commits earlier; `AggregateProcessor`'s docstring (`expr/execution.mojo:~710`) says its keys and inputs are "arbitrary `DynValue` expressions" — they are `BoxedValue`; `expr/__init__.mojo:31` claims `relations → execution` is one-way (it is a cycle — see Q-NEW below) and says `DynValue` "evaluates by dispatching on the tag", contradicting `dynamic.mojo:39-43`; `Sort` and `Limit` are not re-exported from `expr/__init__.mojo`; `buffers.mojo:103` points at "`marrow.bitmap`"; `kernels/boolean.mojo:6` references `faszom.mojo`, deleted two renames ago; `kernels/core.mojo:42` has a typo'd TODO; `values.mojo:675-677` has an **orphaned docstring** that documented the deleted `comptime IsErased` and now reads as false documentation of `OutShape`; a second one at `values.mojo:581-583` asserts "`NumericColumn[DynType]` *is* the erased column leaf", false since `DynType` dropped its family conformances; `LengthKernel.core`'s docstring (`string.mojo:63-65`) claims the expression layer's `StringLength` builds on it — it calls `LengthKernel.dispatch`, never `core`. **`README.md:60-62` claims the runtime lane "is what the Python bindings drive" — nothing is bound — and describes `Table`/`Column`/`Project`/`Filter`, all deleted.** | S |
| **Q6.1** | Cross-engine aggregate benchmark with the AOT path measured. Present: `bench_groupby.py`, `bench_clickbench.py`, `--competition`, the two agg gate binaries. **Absent**: any `*_aot.mojo` benchmark and the merged one-command table — so every table is still one row (`marrow-dynamic`) and **the differentiator is unmeasured**. Gates every Q2.5 round. | M |
| **Q7.1** | **Fusion gaps** (from `docs/kernel-fusion-architecture.md`'s corrected "Open" section): there is no `Materialized` leaf adapter, so an arbitrary eager kernel result cannot re-enter a fused subtree without writing a `Breaker` node; `StringPredicate` (`values.mojo:1685`) is a `Breaker` that materializes a full `BoolArray` in `prepare` (`:1710`), so `s1 == s2 and a > b` is two passes, not the claimed one; `StringLength` (`:1785`) likewise calls `LengthKernel.dispatch` into an `Int32Array` (`:1806`), so `s.len() + a` is two passes. | M each |
| **Q7.2** | **`NumericCompare`'s `S` (string kernel) parameter is dead in the fused lane.** Its docstring (`values.mojo:947-949`) says "It exists for the erased arm" — that arm was deleted with `IsErased`, and `Self.S` has zero occurrences in `values.mojo`. Either remove the parameter or correct the docstring. | S |
| **Q-NEW** | **`marrow.expr` now has three import cycles**, introduced by the two-lane refactor: `values ⟷ dynamic`, `values ⟷ relations`, `relations ⟷ execution`. Mojo resolves them, so this is a layering question, not a build break — but `tasks-expr-kernels-layering.md` asserted "zero cycles" and that is now false. Decide whether to restore the DAG or to document the cycles as intended. | S to decide |
| **L2** | Split `values.mojo` — 2,534 lines. **Rewrite the card first**: its stated DoD ("`DynValue` lives in its own module; `values.mojo` no longer imports `.dynamic`") is unachievable, since the box is `BoxedValue` in `relations.mojo` and `values.mojo` now imports both. The real residue is the runtime-lane builders `col`/`lit`/`if_else`/`coalesce`/`case_when` still sitting at `values.mojo:2476-2513`. | M |
| **V0** | Map is half-shipped: **no IPC in either direction** (zero `Map` hits in `ipc.mojo`; type code 17 absent from `:43-65` and `:785-827`), no `MapScalar`, no `cast` arm. Map *is* supported in dtype/array/builder/C-Data/Parquet. CLAUDE.md's layout-coverage claim is true everywhere except IPC. | S |
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

---

## 6. One-time: documentation disposition

Do this in a single commit. Seven task files and six design docs are replaced or
deleted by this page.

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

**Rewrite:**

| File | Action |
|---|---|
| `execution-engine-roadmap.md` | Keep — it is the only live plan, and the milestone structure, the ClickBench-as-M1 decision and the Won't list are all still right. Purge 14 `TagValue` mentions and `is_deterministic`; invert `DynValue`↔`BoxedValue` throughout (9 mentions); rewrite G1–G9 and D1/D4; refresh the stale line cites. |
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

`sort-design.md`, `joins-design.md`, `groupby-design.md` and
`decimal-type-design.md` stay as feature specs; their unbuilt sections are the
source of M2.\*/M3.\* above. `decimal-type-design.md` needs one correction: it
proposes a custom `Int256` because "Mojo has no native `DType.int256`" — Mojo
has one and the tree uses it (`dtypes.mojo:252`).

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
  run detection. Note the shipped `_choose_strategy` dispatches on **row count and
  estimated cardinality only** — the key *type* never enters the decision, which
  is the opposite of the design.
- **Sort features designed but unbuilt**: `SortOptions` with per-column
  `nulls_first` (today one flag covers all keys), permutation refinement with
  equal ranges, GPU radix, 4-byte string prefix comparison, scatter prefetch,
  parallel comparison sort, batch-level K-way merge.
- **Decimal arithmetic**: nothing was built — no `add`/`sub`/`mul`/`div`, no
  scale-alignment or result-precision rule, no 256-bit intermediate promotion.
  Rescale exists only inside `cast` (`cast.mojo:855-869`) and its up-scale
  multiply has **no overflow check**. Precision is never validated —
  `decimal32(40, 0)` constructs.
