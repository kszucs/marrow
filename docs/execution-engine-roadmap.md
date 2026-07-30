# Marrow Execution-Engine Roadmap

**Status:** planning document (2026-07-24). **Scope:** the plan of record for turning
marrow from "Arrow-in-Mojo + kernels + a prototype relational layer" into a *usable
local (single-node) columnar query engine* with two frontends:

- **(F1) Python lazy API** — ibis/polars/datafusion-style: build a lazy expression /
  plan, `.collect()` at the end. Type-erased, interpreted (vectorized) execution.
- **(F2) Mojo AOT DSL** — the same query shape written in Mojo, monomorphized at
  `mojo build` time into a small, fast, dependency-light binary (HyPer/Umbra-style
  compiled query, but "for free" via Mojo `comptime`).

Local execution only. No distributed, no server, no SQL-string parser (initially).

This document is **prescriptive and task-oriented**: features are tagged with
**MoSCoW** priority (Must / Should / Could / Won't) and grouped into ordered
**milestones** so the work can be handed to coding agents week-by-week. Every table
row is meant to become one or a few concrete tasks.

---

## 1. Objectives & success criteria

The engine is "usable" when it can run recognized local-analytics benchmarks end to
end through *both* frontends. We adopt the standard sequencing (see the research
appendix, §9):

| Milestone | Benchmark bar | What it proves |
|---|---|---|
| **M1 — Scan/agg core** | **ClickBench subset** (single flat table, *no joins*): scan → filter → hash-aggregate → group-by → order-by-limit → string filters → date parts | The MVP analytical loop, both frontends |
| **M2 — Join/agg at scale** | **H2O.ai db-benchmark** group-by (10 q) + join (5 q) | Hash group-by + hash join across sizes/key types, with spill |
| **M3 — Full relational** | **TPC-H** (22 q, join-heavy) | Multi-way joins, join reordering, subqueries, full SQL breadth |

M1 is the first "ship it" line. M2/M3 are follow-through. Success at each milestone
means: correct results (cross-checked against PyArrow/DuckDB), competitive-or-better
wall-clock vs polars/duckdb on the same box, **and** (for F2) a binary whose
`__TEXT` stays within the `benchmarks/binary_size/` budget.

> **M1 is defined concretely by the 43 ClickBench queries** (`~/Workspace/ClickBench`)
> run through marrow's frontend, not SQL. The `hits` table is a single wide flat table
> (~105 cols, no joins/nesting) — ideal first target. A per-query feature→task coverage
> map is in [`tasks-execution-engine.md`](tasks-execution-engine.md) §6. Reading the
> queries **promoted several features from M2 into M1** (min/max on string+date,
> `count_distinct` as a relational agg, `HAVING`, computed group keys/agg inputs,
> `date_trunc`) — the rows below and the tasks doc reflect that. Q29 (`REGEXP_REPLACE`)
> is the one query deferred to M2 (needs a regex engine); the M1 subset is the other 42.

### Guiding invariants (do not regress)

1. **Small-binary DCE property** (`benchmarks/binary_size/`, `pixi run binary_size`).
   The fused-only value box must never make `TagValue`'s interpreter or the open
   per-dtype kernel fanout reachable. Re-verify at *every* milestone gate, not just
   the end. This is the whole value proposition of F2.
2. **One engine, two drivers.** Every feature is implemented once in the kernels /
   relational layer and reached by *both* frontends — the fused comptime `Value`
   path (F2) and the interpreted `TagValue` path (F1). No feature may live in only
   one driver. (Current code violates this — see §3.)
3. **PyArrow-shaped naming** everywhere the surface is user-facing.
4. **Code quality is an acceptance criterion, not a follow-up.** The codebase must stay
   in great shape: **sound abstractions, minimal boilerplate, minimal free-standing
   functions** (behaviour belongs on the type/trait it operates on — reach for a free
   function only when it genuinely spans types). Every task's Definition of Done includes
   a quality pass; every wave/milestone ends with a dedicated review (`/simplify` over the
   diff, plus an abstraction/duplication audit). A feature that lands as a pile of
   free functions or copy-pasted per-dtype boilerplate is **not done** — it is refactored
   before merge. Prefer reusing existing building blocks (e.g. bitmap bitwise ops, view
   abstractions, kernel `core`/`apply` tiers) over new one-offs. This invariant outranks
   schedule: a wave does not open until the prior wave's quality gate is green.

---

## 2. Current state (grounded, corrects stale docs)

A full subsystem audit (2026-07-24) found the codebase **substantially more advanced
than `CLAUDE.md` claims**. Two `CLAUDE.md` statements are stale and should be fixed:
"Table: Not yet implemented" (it exists at `tabular.mojo:268`) and "only bool,
numeric, string, list, fixed-size list, struct types" (temporal, decimal 32/64/128/256,
map, dictionary, binary variants, interval are all present).

### 2.1 What already works (the foundation is strong)

- **Data model** — `RecordBatch`, `Table` (schema + `List[ChunkedArray]`),
  `ChunkedArray`; broad `DataType` coverage across array/builder/scalar tiers:
  all numerics, bool, string/large_string, binary/large_binary,
  fixed_size_binary, list/large_list/fixed_size_list, struct, **map**,
  **dictionary**, **date32/64, time32/64, timestamp(tz), duration, interval**,
  **decimal32/64/128/256**, null.
- **Interop** — C Data Interface (schema/array/stream, device), IPC file+stream
  read+write (hand-rolled FlatBuffers, no dep), PyArrow bridge on every type.
- **Kernels** — filter / take / drop_null (broadest type coverage, incl. nested);
  multi-column sort (asc/desc per key, nulls first/last, stable, top-K);
  hash group-by (sum/mean/min/max/count/product + count_distinct/HLL, serial /
  thread-local / radix-parallel strategies); hash join (inner/left/right/full/
  semi/anti, ALL/ANY strictness, multi-key, radix-parallel); comprehensive casts
  (numeric/temporal/decimal/string/bool/list/struct/dictionary); numeric
  arithmetic + math + comparisons; string maps/predicates; HLL distinct.
- **Expression engine** (`marrow/expr/values.mojo`) — the dual-representation design
  is real: a rich **fused comptime algebra** (`Value` families: numeric/bool/string/
  list, reductions, casts, pipeline-breaker staging) boxed alongside a **runtime
  interpreter** (`TagValue`) into one `DynValue`. Verified byte-identical `__TEXT`
  for the fused path vs the fully-typed layer, ~30× smaller than the interpreter.
- **Relational engine** (`marrow/expr/relations.mojo` + `execution.mojo`) — immutable
  plan IR + pull-based Volcano processors (morsel size 65 536): `InMemoryTable`,
  `ParquetScan`, `Filter`, `Project`, `Aggregate`, `Join`. `execute(plan)` drains a
  reusable plan. Parquet **row-group + page-level predicate pruning** is wired
  (`pruning.mojo`, `test_pushdown.mojo`).
- **Parquet** (`marrow/parquet/`) — production-grade *decode/encode breadth*: all
  encodings (PLAIN, RLE/dict, DELTA_*, BYTE_STREAM_SPLIT), all codecs (snappy/gzip/
  zstd/lz4/lz4_raw/brotli via runtime dlopen), DataPage V1+V2, full Dremel nested
  read+write, complete Thrift metadata, statistics, page index, bloom filters.
  Cross-validated against PyArrow both directions.
- **OpenDAL Mojo binding** (`~/Workspace/opendal/bindings/mojo`) — capable WIP:
  operator verbs (read/write/list/stat/delete/copy/rename), **seek-based ranged
  reads** (`Reader.seek` + `read_exact`), backends compiled into `libopendal_c`
  (fs/s3/http/memory). Blocking-only (no async).

### 2.2 The gaps that block "usable engine" status

Ordered by how hard they block the milestones.

**G1 — No lazy Python frontend at all.** The entire Python surface
(`python/marrow/`) is *eager*: PyArrow-shaped compute functions + `RecordBatch`
methods that execute immediately over a single in-memory batch. There is **no**
`Expr`, `col()`, `LazyFrame`, or deferred plan. The lazy engine exists in Mojo but
**zero of it is bound to Python**. ClickBench today runs through eager
`group_by().aggregate()` on one batch — no filter, no join, no SQL.

**G2 — The runtime (`TagValue`) driver is thin.** F1 must drive `TagValue`, but it
supports only ~20 op tags (arithmetic, compares, and/or, neg/abs/not, is_null,
if_else, length, cast). It lacks: mod/floordiv/pow, float math, xor, isnan/isinf,
notnull, any/all, **almost all string ops (only LENGTH)**, all list ops, windows.
The fused comptime algebra is far richer but **cannot be built dynamically**. This
asymmetry violates invariant #2 and must be closed op-by-op.

**G3 — Missing relational operators.** No `Sort`, `Limit`/`TopK`, `Distinct`,
`Union`, `HAVING`, or `Window` relational node. Join keys are **bare column indices
only** (computed keys raise). `Project.select()` is column-passthrough — no
computed-column projection sugar. These are all MVP-tier for ClickBench (`ORDER BY …
LIMIT`) and beyond.

**G4 — No optimizer / plan-rewrite layer.** Only the Parquet-scan pruning
special-case exists. No projection pushdown, no general predicate pushdown, no
pushdown below joins, no constant folding, no CSE, no limit pushdown, no join
reordering. `DynValue` lacks `referenced_columns()` — the prerequisite for
projection pushdown. Designed in `aot-relations-design.md` but unbuilt.

**G5 — Scan is not streaming and not remote.** `ParquetScanProcessor` reads the
**entire pruned file into memory** on first pull, then slices morsels — memory scales
with file size, defeating streaming. The reader is **local-mmap-only**: no
byte-range source abstraction, so OpenDAL cannot be plugged in without downloading
whole files. **Projection is not pushed into the scan** (reader supports `columns=`,
engine never passes it). Bloom filters are not consulted. Nested-file pruning is
disabled. No multi-file / dataset / glob scan.

**G6 — Blocking operators buffer everything; no spill; no plan-level parallelism.**
`AggregateProcessor` buffers all group-ids + value columns; `JoinProcessor` collects
the entire left side. No spill-to-disk, no bounded memory. The pull loop is
single-threaded — the parallel kernels exist but the `Processor` tree doesn't use
them.

**G7 — Kernel correctness & coverage holes.** Boolean `and/or/not/xor` **drop the
validity bitmap** — no Kleene 3-valued logic (SQL-incorrect). `select` (CASE/WHEN) is
numeric-only. No `coalesce`/`nullif`/`fill_null`, no `is_in`. String ops missing
`substring`/`like`/`ilike`/`regexp_*`/pad/replace/split and `<`/`>` ordering. No
temporal **extraction** (year/month/day/hour…) or **arithmetic** (date ± interval,
date_diff) or string↔timestamp parsing. No **decimal arithmetic**. Aggregates lack
`stddev`/`variance`/`median`/`quantile`/`first`/`last` and grouped min/max on
non-numeric columns. `LengthKernel`/`ArrayLengthKernel` don't propagate nulls.

**G8 — AOT relational monomorphization deferred.** F2 works today only at the
*expression* level (a fused `Value` boxed into a runtime `DynRelation`). The
fully-typed relational plan (`Table[T]`, `Project[*Es]`, typed `HashJoin[L,R,LK,RK]`,
`Env`/`Param` late binding) that yields a fully-DCE'd *relational* binary is unbuilt,
partly blocked on a Mojo `reflect` resolution bug (`values.mojo:1595`).

**G9 — Data-model computational thinness.** `ChunkedArray` is a length-aware
container only (no `slice`/`__getitem__`/`null_count`/`filter`/`cast`); `Table` has no
`slice`/`select`/`filter`/`sort`/`concat_tables`. Kernels target `DynArray`, not
`ChunkedArray`.

---

## 3. Architecture decisions (settle before building wide)

These shape many tasks; decide them first.

- **D1 — F1 drives `TagValue`; F2 drives fused `Value`. Both go through the same
  `DynRelation` plan and `execute()`.** Do *not* build a second execution path for
  Python. The Python `LazyFrame.collect()` maps to `execute(plan, ctx)`. Confirmed
  viable by the audit; the only new work is binding + Python wrappers + growing
  `TagValue`.
- **D2 — Grow the relational layer as *erased* nodes (`DynRelation`) first.** Add
  `Sort`/`Limit`/`Distinct`/`Union`/`Window` as runtime relational nodes over
  `DynValue`. Defer the fully-typed `Project[*Es]`/`Table[T]` monomorphized
  relational plan (G8) to M3+ — it is an optimization of F2, not a correctness gate,
  and it is blocked on a compiler bug. F2's binary-size win is already delivered at
  the expression level.
- **D3 — Introduce a `ByteSource` seam in the Parquet reader** (`read_at(offset,
  len)` + `size()`), implemented by both `MappedFile` (local) and an OpenDAL-backed
  source. This unblocks streaming row-group iteration *and* remote scans with one
  abstraction. Do this before wiring OpenDAL.
- **D4 — Optimizer is a rule-list over the immutable `DynRelation` IR**, mirroring
  DataFusion. Each rule is a pure `DynRelation -> DynRelation` rewrite. Add
  `referenced_columns()` / `is_deterministic()` metadata to `DynValue` first
  (prerequisite for pushdown).
- **D5 — Fix Kleene null semantics in the boolean kernels before M1 ships.** Wrong
  results are worse than missing features; ClickBench filters exercise this.
- **D6 — Vectorized batch size for the interpreted path** stays at the current
  morsel granularity for operators, but expression evaluation should process
  cache-resident sub-batches (~2K rows) internally where it matters (matches
  DuckDB's `STANDARD_VECTOR_SIZE`). Not an M1 blocker; revisit at M2 perf tuning.

---

## 4. Feature backlog (MoSCoW, by workstream)

Priority is relative to reaching **M1 (ClickBench)** unless noted. **Must** = required
for M1. **Should** = required for M2 or a correctness/UX necessity. **Could** = M3 or
polish. **Won't** = explicitly out of scope for this roadmap.

### 4.1 Frontends & API surface

| Feature | MoSCoW | Notes / entry points |
|---|---|---|
| Python `Column` wrapper over `TagValue` (forward `+ - * / == < > & \| ~`, `.cast`, `.is_null`, `.isin`, reductions, string/temporal methods) | **Must** | `TagValue` already has the dunders (`dynamic.mojo:413-468`); pure-Python `Column`. |
| Python `col()`, `lit()` (with runtime dtype inference incl. string/temporal), `if_else` | **Must** | `lit` is parametric on numeric type today — needs a runtime-dtype `lit`. |
| Python **ibis-flavored** `Table` over `DynRelation` (`.filter/.select/.mutate/.group_by/.aggregate/.order_by/.limit/.join`) building, not executing; `.execute()`/`.to_pyarrow()` | **Must** | Thin native wrapper — **not** an ibis backend, no `ibis` dep. `DynRelation` fluent API exists (`relations.mojo:211-335`). See tasks §3.1. |
| Bind `DynValue`/`DynRelation` through `PythonModuleBuilder.add_type` | **Must** | New binding territory; watch trait/associated-type elaboration hazards (CLAUDE.md). |
| `marrow.read_parquet(path/glob)` / `marrow.table(schema)` returning a `Table` | **Must** | Wraps `parquet_scan`; see scan workstream. |
| Route Python eager `join/group_by/sort_by` through `execute(plan)` (unify paths) or keep documented eager shortcuts | **Should** | Today they bypass the expr layer (`tabular.mojo`). Avoid two divergent code paths. |
| Mojo AOT DSL: ergonomic `col("a", int64)`-style builders documented end-to-end for M1 query shapes | **Must** | Expression-level fused builders exist (`values.mojo:1605`); document + fill gaps. |
| Mojo AOT DSL: fully-typed relational plan (`Table[T]`, `Project[*Es]`, typed joins) | **Could** | G8; blocked on `reflect` bug; F2 relational-binary optimization, not correctness. |
| SQL string parser / frontend | **Won't** | Out of scope. Both frontends are programmatic. Revisit post-M3. |
| Full **ibis backend** (`ibis.backends.marrow`, translating the ibis op graph) or an `ibis` runtime dependency | **Won't** | ibis is a *loose naming guideline* only. A real backend drags in ibis's op catalog, type coercion, and backend test contract — over-engineering. Ship the thin native `Table`/`Column`. |
| `pandas`-style eager `DataFrame` API | **Won't** | Deferred-only per the brief; the existing eager RecordBatch surface stays as-is. |

### 4.2 Relational operators (the `DynRelation` IR + `Processor` tree)

| Operator | MoSCoW | Notes |
|---|---|---|
| `Sort` node + processor (wraps existing multi-col sort kernel; nulls, top-K) | **Must** | ClickBench needs `ORDER BY … LIMIT`. No relational Sort node today. |
| `Limit` / `Offset` / `TopK` node (fuse limit into sort as top-K) | **Must** | Enables limit pushdown (§4.4). |
| Computed-column `Project` (project arbitrary expressions, not just column passthrough) | **Must** | Today `select()` is passthrough; must evaluate `DynValue`s and name outputs. |
| `Aggregate`: `HAVING` (post-agg filter), `count_distinct` as an agg func, computed group keys + computed agg inputs | **Must** | ClickBench Q5/6/9–14/19/23/28/30/36/40/43. Node exists; extend. See tasks §6. |
| Join on **computed keys** (not just bare column index) | **Should** | `column_index` raises on computed keys today (`relations.mojo:355`). H2O/TPC-H. |
| `Distinct` / `unique` node | **Should** | Needs a distinct-values kernel (§4.5). H2O. |
| `Union` / `concat` relational node | **Should** | TPC-H, set ops. |
| `CrossJoin` / `NestedLoopJoin` (non-equi / theta joins) | **Could** | TPC-H. Join constants declared but unimplemented. |
| `Window` relational node (partition/order/frame) | **Could** | M3; see §4.5 window kernels + `lane-shape-window-design.md`. |
| `Values` / `EmptyRelation` literal sources | **Could** | Convenience / optimizer targets. |

### 4.3 Execution runtime

| Feature | MoSCoW | Notes |
|---|---|---|
| **Streaming row-group iteration** in `ParquetScanProcessor` (per-row-group / per-morsel decode, not whole-file) | **Must** | G5. Memory must not scale with file size. Prereq for large ClickBench. |
| Push **projection** into the scan (`read_table(columns=...)`) | **Must** | G5. Biggest scan win; reader already supports it. |
| Plan-level parallelism (drive the parallel join/groupby kernels; partition or morsel-driven) | **Should** | G6. Kernels are parallel; the pull loop isn't. M2 perf. |
| Spill-to-disk for `Aggregate` build + `Join` build (bounded memory) | **Should** | G6. H2O larger-than-RAM sinks (cf. Polars). |
| `CoalesceBatches` (compact small morsels after filter) | **Should** | Vectorized-execution efficiency. |
| Late materialization / selection vectors on the scan (decode filter cols, filter, then decode survivors) | **Could** | DataFusion keeps this off-by-default; subtle. M3. |
| Async / prefetch ranged reads for remote scans | **Could** | OpenDAL C ABI is blocking; needs multiple readers. M3. |

### 4.4 Optimizer / plan rewrite

| Rule | MoSCoW | Notes |
|---|---|---|
| `referenced_columns()` / `is_deterministic()` metadata on `DynValue` | **Must** | D4 prerequisite for all pushdown. `TagValue` has `resolve_names`; extend. |
| **Projection pushdown** (prune unused columns to the scan) | **Must** | Highest ROI on columnar/Parquet. Feeds §4.3 projection-into-scan. |
| **Predicate pushdown** (filters toward the scan; split conjuncts on AND) | **Must** | Feeds Parquet row-group/page pruning (already wired below the scan). |
| **Constant folding / expression simplification** | **Should** | Cheap; enables other rules. |
| **Limit pushdown** (limit → top-K in sort; limit into scan) | **Should** | Turns full sort into bounded top-N. |
| Predicate pushdown **below joins** | **Should** | H2O/TPC-H. |
| **CSE** (common-subexpression elimination) | **Could** | Matters for generated/repeated subexprs. |
| **Join reordering** (needs cardinality estimation + stats) | **Could** | TPC-H; advanced. Defer until multi-join benchmarks. |
| Subquery decorrelation, outer-join elimination | **Won't** (this roadmap) | Needs SQL frontend; revisit post-M3. |

### 4.5 Kernels & expressions

Every kernel must expose **both** `core[W]` (fused, for F2) *and* `apply`/`dispatch`
(eager, for F1), and be reachable through a `TagValue` tag (F1) *and* a fused `Value`
node (F2). Closing the `TagValue`/fused asymmetry (G2) is a running theme.

| Feature | MoSCoW | Notes |
|---|---|---|
| **Kleene 3-valued boolean logic** in `and/or/not/xor` (stop dropping validity) | **Must** | G7/D5. SQL-correctness. `boolean.mojo`. |
| `CASE`/`when`/`select` for **all** types (not numeric-only), multi-branch | **Must** | ClickBench + general. `boolean.mojo` `select` is numeric-only. |
| `coalesce`, `nullif`, `ifnull`/`nvl`, `fill_null` | **Must** | Ubiquitous. |
| `is_in` / `isin` (scalar-set membership) | **Must** | ClickBench filters. |
| String `<`/`<=`/`>`/`>=` ordering (dispatch compares to strings) | **Must** | ClickBench string filters + string sort/group. |
| `like` / `ilike` pattern match | **Must** | ClickBench URL filters. |
| `substring`/`slice`, `replace`, `char_length`, `lower`/`upper` compare | **Should** | ClickBench + general string work (some maps exist). |
| `regexp_match` / `regexp_replace` / `regexp_extract` | **Should** | ClickBench proper uses regex. Needs a regex engine decision. |
| Temporal **extraction** (year/month/day/hour/minute/second/weekday/quarter) | **Must** | ClickBench date columns. `cast.mojo` only unit-scales. |
| Temporal `date_trunc(unit)` | **Must** | ⬆ ClickBench Q43 (`date_trunc('minute', …)`). |
| Temporal **arithmetic** (date ± interval, `date_diff`, `now`) | **Should** | H2O/TPC-H date logic. |
| String↔timestamp `strftime`/`strptime` cast | **Should** | Parsing/formatting; `cast.mojo` gap. |
| **Decimal arithmetic** (add/sub/mul/div with scale rules) + decimal compare/agg | **Should** | TPC-H money math. Casts exist; arithmetic doesn't. |
| Aggregates: grouped **and** whole-table `min`/`max` on **string/date/temporal** | **Must** | ⬆ ClickBench Q7 (date), Q22/23 (string). Value path is numeric-only today. |
| Aggregates: `first`/`last` | **Should** | H2O advanced group-by. |
| Aggregates: `stddev`/`variance`/`median`/`quantile`/`mode` | **Should** | H2O advanced group-by (`median`/`sd`). |
| `unique()` / distinct-values kernel (not just count) | **Should** | Powers `Distinct` node. |
| **Window kernels** (`row_number`/`rank`/`dense_rank`/`lag`/`lead`/`ntile`, running/rolling aggs) | **Could** | M3. `values.mojo` has only a toy `RowNumber`. Design in `lane-shape-window-design.md`. |
| Null propagation fix in `LengthKernel`/`ArrayLengthKernel` | **Should** | Correctness. |
| Integer overflow checking; `atan2`/`tan`/hyperbolic; bitwise/shift kernels | **Could** | Long tail. |
| `greatest`/`least` over N args | **Could** | Convenience. |
| Set ops kernels (intersect/except) | **Could** | Post-M3. |

### 4.6 Scan / IO / Parquet / OpenDAL

| Feature | MoSCoW | Notes |
|---|---|---|
| `ByteSource` trait (`read_at`, `size`) in the reader; `MappedFile` implements it | **Must** | D3. Prereq for streaming + remote. `reader.mojo`. |
| Streaming row-group decode via `ByteSource` (see §4.3) | **Must** | G5. |
| Projection pushdown into scan (see §4.3/§4.4) | **Must** | G5. |
| Multi-file / directory / glob dataset scan (list → fan-out → concat schema) | **Should** | `Operator.list` gives listing; no consumer yet. ClickBench-at-scale / H2O. |
| **OpenDAL-backed `ByteSource`** (whole-object read, then ranged reads via `Reader.seek`) | **Should** | G5. Multi-filesystem scan. Zero integration exists today. |
| Bloom-filter pushdown in the scan (equality predicates) | **Should** | `ParquetFile.bloom_filter` exists; scan never consults it. |
| Enable predicate pruning for **nested** files | **Could** | Disabled today (`_read_plan`). |
| Reduce the 3–4× file re-open in the scan (single metadata read) | **Could** | Perf. |
| Hive-style partition discovery (`col=val` dirs) | **Could** | TPC-H/dataset ergonomics. |
| Async/prefetch remote reads (multiple OpenDAL readers) | **Could** | M3. Blocking C ABI constraint. |
| Preserve struct-level nulls over repeated groups on write; `large_*` offset width | **Could** | Known writer asymmetries; not engine blockers. |

### 4.7 Data model / types

| Feature | MoSCoW | Notes |
|---|---|---|
| `ChunkedArray` computational surface (`slice`, `__getitem__`, `null_count`, `filter`, `cast`, `take`) | **Should** | G9. Engine operates on Tables → columns must behave like columns. |
| `Table`-level ops (`slice`/`select`/`filter`/`sort`/`rename`/`concat_tables`) | **Should** | G9. RecordBatch has these; Table doesn't. |
| Python `ChunkedArray` wrapper class | **Could** | Interop completeness. |
| `union` type (sparse/dense) | **Could** | Only needed for specific plans; not a first-cut blocker. |
| Run-end-encoded (REE) arrays | **Won't** (this roadmap) | Compression optimization. |
| `string_view` / `binary_view` layouts | **Won't** (this roadmap) | Perf optimization; big surface. |
| Fix stale `CLAUDE.md` type/Table claims + `similarity.mojo`/`sum.mojo` refs | **Must** | Docs hygiene; cheap; avoids misleading future agents. |

### 4.8 Cross-cutting

| Item | MoSCoW | Notes |
|---|---|---|
| Keep `benchmarks/binary_size/` green at every gate; add a relational-plan size bench | **Must** | Invariant #1. |
| End-of-wave **quality review** (`/simplify` + abstraction/duplication/free-function audit) gating the next wave | **Must** | Invariant #4. Not cleanup-later — an acceptance gate. |
| Both-driver parity tests (fused `Value` result == `TagValue` result) per op | **Must** | Invariant #2. Extend `test_values`/`test_runtime`. |
| ClickBench-subset harness runs *through the lazy plan* (not eager one-batch) | **Must** | Currently eager single-batch (`clickbench.py`). |
| PyArrow/DuckDB cross-check in the test suite for new kernels/ops | **Should** | Existing pattern. |
| `EXPLAIN` / plan pretty-printer (leverage `write_to`) for both frontends | **Should** | Debuggability; `DynValue.write_to` exists. |
| Spill/mem-budget config on `ExecutionContext` | **Could** | M2. |

---

## 5. Milestone plan (ordered)

Each milestone lists its ordered workstreams. Land features so the suite + binary-size
gate stay green after every step.

### M0 — Unblock (foundations, ~week 1)

Prerequisites everything else builds on. Small, high-leverage.

1. **Fix Kleene null semantics** in `boolean.mojo` (D5) + null-propagation in
   `LengthKernel`/`ArrayLengthKernel`. *(Must — correctness before features.)*
2. **`ByteSource` seam** in the Parquet reader (D3); `MappedFile` implements it.
   No behavior change yet — pure refactor with tests. *(Must.)*
3. **`referenced_columns()` / `is_deterministic()`** on `DynValue`/`TagValue` (D4).
   *(Must — unblocks pushdown.)*
4. **Grow `TagValue`** to reach parity with the fused algebra on the ops M1 needs
   (compares incl. strings, and/or Kleene, is_null/notnull, cast, if_else,
   arithmetic). Add a **both-driver parity test** harness. *(Must — G2.)*
5. Fix stale `CLAUDE.md` claims. *(Must — cheap.)*

### M1 — ClickBench: the usable analytical core (the "ship it" line)

Ordered:

1. **Relational operators**: computed-column `Project`; `Sort` node; `Limit`/`TopK`
   node. *(§4.2 Must.)*
2. **Scan**: streaming row-group iteration + projection-into-scan over `ByteSource`.
   *(§4.3/§4.6 Must — makes large single-table scans real.)*
3. **Optimizer v1**: projection pushdown + predicate pushdown (conjunct splitting),
   feeding the already-wired Parquet row-group/page pruning; limit pushdown.
   *(§4.4 Must/Should.)*
4. **Kernels for ClickBench**: `is_in`, `coalesce`/`nullif`, `CASE` for all types,
   string ordering compares, `like`/`ilike`, temporal extraction (year/month/day/
   hour…). *(§4.5 Must.)*
5. **Frontends**: bind `Expr`/`col`/`lit`/`LazyFrame`/`scan_parquet`; wire the
   ClickBench harness through the lazy plan for **both** F1 (Python) and F2 (Mojo
   AOT DSL). *(§4.1 Must.)*
6. **Gate**: ClickBench subset passes on both frontends, results match DuckDB,
   `binary_size` green.

### M2 — H2O: joins & aggregates at scale

1. Join on computed keys; grouped `first`/`last` + min/max on non-numeric;
   `stddev`/`variance`/`median`/`quantile`; `Distinct` + distinct-values kernel;
   `Union`. *(§4.2/§4.5 Should.)*
2. Plan-level parallelism (drive parallel join/groupby kernels; morsel or partition).
   *(§4.3 Should.)*
3. Spill-to-disk for aggregate/join build; `CoalesceBatches`; mem budget on
   `ExecutionContext`. *(§4.3 Should.)*
4. Multi-file/glob dataset scan + **OpenDAL-backed `ByteSource`** (remote scan) +
   bloom pushdown. *(§4.6 Should.)*
5. `ChunkedArray`/`Table` computational surface. *(§4.7 Should.)*
6. Predicate pushdown below joins; constant folding. *(§4.4 Should.)*
7. **Gate**: H2O group-by + join pass at 5 GB, spill works, competitive wall-clock.

### M3 — TPC-H: full relational breadth

1. `CrossJoin`/`NestedLoopJoin` (non-equi/theta); temporal arithmetic + decimal
   arithmetic; `Window` node + window kernel family; string `substring`/`replace`/
   `regexp_*`. *(§4.2/§4.5 Could/Should.)*
2. CSE; **join reordering** with cardinality estimation. *(§4.4 Could.)*
3. Late materialization / selection vectors; async prefetch remote reads. *(§4.3
   Could.)*
4. **F2 relational monomorphization** (`Table[T]`/`Project[*Es]`/typed joins) once
   the `reflect` bug clears — the fully-DCE'd relational binary. *(§4.1 Could; G8.)*
5. **Gate**: TPC-H runs; join reordering measurable on multi-join queries.

### Beyond (explicitly deferred)

SQL-string frontend; distributed/multi-node; server mode; `union`/REE/view types;
subquery decorrelation; set-op kernels; Hive partition discovery at scale.

---

## 6. How to use this document

- Each **table row** in §4 is a task seed. Pull rows tagged **Must** for the current
  milestone into the sprint; expand into concrete implementation tasks against the
  cited files (`marrow/expr/relations.mojo`, `execution.mojo`, `values.mojo`,
  `dynamic.mojo`, `marrow/kernels/*`, `marrow/parquet/reader.mojo`,
  `python/bindings/*`).
- Enforce the **invariants** (§1) as CI/review gates: `pixi run binary_size` and the
  both-driver parity tests are non-negotiable at every merge.
- Prefer the **prior-art path** (CLAUDE.md): consult Arrow C++/Rust and DataFusion/
  Polars/DuckDB for kernel semantics, null handling, offset rules, and operator
  design before writing new code.

---

## Appendix — reference-engine notes (external research)

- **Minimum usable engine** = `{TableScan (+Parquet projection/predicate pushdown),
  Projection, Filter, HashAggregate, HashJoin, Sort, Limit/TopK}` operators +
  `{projection pushdown, predicate pushdown, constant folding, limit pushdown}`
  optimizer rules + vectorized batches (~2048) with morsel-driven or partition
  parallelism.
- **DataFusion pipeline**: LogicalPlan → OptimizerRules → ExecutionPlan (pull-based
  async `RecordBatch` streams, partition-parallel). Baseline logical rules:
  `push_down_filter`, `push_down_limit`, `optimize_projections`,
  `simplify_expressions`, `common_subexpr_eliminate`, `eliminate_cross_join`,
  trivial-node eliminations, `type_coercion` (analyzer). Physical: join-selection,
  sort enforcement/removal, repartition, coalesce-batches, filter-into-scan.
- **Parquet pushdown tiers** (implement 1→2→4 first): (1) column projection,
  (2) row-group stats pruning, (3) page-index pruning, (4) bloom filters,
  (5) row-level late-materialization pushdown (subtle; DataFusion ships it *off* by
  default). marrow already has 1–3 at the reader and 2–3 wired into the scan; **1
  (projection) and 4 (bloom) are not wired into the scan.**
- **Vectorized execution**: DuckDB `STANDARD_VECTOR_SIZE = 2048` (L1/L2-resident);
  morsels ~10K–100K for parallelism; push-based pipelines fuse non-materializing
  operators; pipeline breakers = agg/join build, sort. Pull-based (DataFusion) is
  fully competitive and maps naturally onto Mojo iterators.
- **Compiled vs vectorized** (Kersten et al., VLDB 2018): neither dominates;
  compilation wins compute-bound/cache-resident (fewer instructions, register-resident
  tuples), vectorization wins memory-bound (latency hiding). Modern systems go hybrid.
  **This maps cleanly onto marrow's two frontends**: F2 (Mojo `comptime`
  monomorphization) = HyPer/Umbra-style compiled small binaries; F1 (type-erased
  `TagValue` interpreter over `DynArray`) = the vectorized path. Small binaries come
  from embedding only the operators/types the query uses + DCE — exactly the
  `benchmarks/binary_size/` property.
- **Benchmarks**: ClickBench (single flat table, ~100M rows, 43 q, **no joins** —
  scan/filter/group-by/top-n/string/date) is the ideal *first* target; H2O.ai
  (10 group-by + 5 join, 0.5/5/50 GB) is the join+agg bar; TPC-H (22 q, join-heavy)
  is the full-SQL bar.

Sources: DataFusion query-execution & optimizer docs, DataFusion Parquet
pruning/pushdown blog (2025), DuckDB vector internals & morsel-driven parallelism
(Leis et al.), Polars streaming-engine docs, Kersten et al. VLDB 2018, ClickBench /
H2O / TPC-H specs.
