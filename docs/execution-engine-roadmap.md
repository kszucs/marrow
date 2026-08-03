# Marrow Execution-Engine Roadmap

**Status:** living plan. Written 2026-07-24; re-grounded against the tree at
**`b2e7dae` (2026-08-03)**, after the two-lane expression refactor. **Scope:** the
plan of record for turning marrow from "Arrow-in-Mojo + kernels + a prototype
relational layer" into a *usable local (single-node) columnar query engine* with two
frontends:

- **(F1) Python lazy API** — ibis/polars/datafusion-style: build a lazy expression /
  plan, `.collect()` at the end. Runtime-typed, vectorized execution.
- **(F2) Mojo AOT DSL** — the same query shape written in Mojo, monomorphized at
  `mojo build` time into a small, fast, dependency-light binary (HyPer/Umbra-style
  compiled query, but "for free" via Mojo `comptime`).

Local execution only. No distributed, no server, no SQL-string parser (initially).

This document states **scope and sequencing**: capabilities are tagged with **MoSCoW**
priority (Must / Should / Could / Won't) and grouped into ordered **milestones**. It is
not the task tracker — **[`backlog.md`](backlog.md) is**, and it is the only file that
carries per-item IDs, sizes and statuses. Where the two disagree, `backlog.md` wins;
fix this file.

---

## 1. Objectives & success criteria

The engine is "usable" when it can run recognized local-analytics benchmarks end to
end through *both* frontends. We adopt the standard sequencing (see the research
appendix):

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
> (~105 cols, no joins/nesting) — ideal first target. Reading the queries **promoted
> several features from M2 into M1** (min/max on string+date, `count_distinct` as a
> relational agg, `HAVING`, computed group keys/agg inputs, `date_trunc`) — the rows
> below reflect that, and all five of those have since landed. Q29 (`REGEXP_REPLACE`)
> is the one query deferred to M2 (needs a regex engine); the M1 subset is the other
> 42. The per-query work items are `backlog.md` §3.

### Guiding invariants (do not regress)

1. **Small-binary DCE property** (`benchmarks/binary_size/`, `pixi run binary_size`).
   Preserve closed erasure: no open dispatchers, fused-only value boxes, closed
   per-dtype kernels. Two measured traps guard this — a runtime switch over tags cost
   **+1,807,168 bytes of `__text` (+45.7%)**, and a generic wrapper around an already
   erased dispatch cost **+115,600 bytes**. Re-verify at *every* milestone gate, not
   just the end. This is the whole value proposition of F2.
2. **One engine, two drivers.** Every feature is implemented once in the kernels /
   relational layer and reached by *both* lanes — the fused comptime `Value` lane (F2)
   and the runtime `DynValue` lane (F1). No feature may live in only one lane.
   Windows currently violate this (AOT-only) — see §4.5 and `backlog.md` M2.3.
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

## 2. Current state

### 2.1 What already works

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
  arithmetic + math + comparisons; **Kleene 3-valued** `and`/`or`/`not`/`xor`
  (`boolean.mojo:141`); `case_when`/`coalesce`/`nullif`/`fill_null`
  (`conditional.mojo:145,257,314,374`); `is_in` (`membership.mojo:36`); string
  maps, predicates, ordering compares and `LIKE`/`ILIKE`
  (`string.mojo:404-434,725,753`); temporal extraction + `date_trunc`
  (`temporal.mojo:236-330`, `:439`); HLL distinct.
- **Expression layer** — **two lanes that share no node types** (`marrow/expr/`):
  - the **AOT lane** (`values.mojo`) — every operand bound on a family trait
    (`L: NumericValue`), output dtype a comptime type, a subtree fusing into one
    SIMD loop. Builders `col(name, dtype)` / `lit(value, dtype)` at
    `values.mojo:2426-2470`.
  - the **runtime lane** (`dynamic.mojo:236`) — `DynValue` is one struct holding a
    tag, its children, an optional payload, and a **pointer to its evaluator**, so
    the *operation* stays comptime and a binary links exactly the kernels its
    expressions mention. Fluent dunders at `dynamic.mojo:634-691`; builders
    `col`/`lit`/`if_else`/`coalesce`/`case_when` at `values.mojo:2476-2513`.
  - **`BoxedValue`** (`relations.mojo:155`) is the one erasure box both lanes
    convert into, so each relational operator compiles exactly once. It conforms to
    `Value` and nothing else.
- **Relational engine** (`relations.mojo` + `execution.mojo`) — immutable plan IR +
  pull-based Volcano processors (morsel size 65 536, `execution.mojo:50`). Eight
  nodes: `InMemoryTable`, `ParquetScan`, `Filter`, `Project`, `Limit`, `Sort`,
  `Aggregate`, `Join` (`relations.mojo:744,783,866,892,934,965,1030,1081`). Fluent
  builder at `relations.mojo:406-735`. Computed-column `project`, computed group
  keys and computed aggregate inputs all work; `HAVING` is a `Filter` above an
  `Aggregate` and needs no node. Parquet **row-group + page-level predicate pruning**
  is wired (`pruning.mojo`, `test_pushdown.mojo`).
- **Streaming, projecting Parquet scan** — `ParquetScanProcessor`
  (`execution.mojo:190`) opens the file once, fixes the pushdown plan once, and
  decodes a bounded 64 MB window of row groups at a time (`_WINDOW_BYTES`,
  `execution.mojo:55`). The scan's schema **is** its projection, so unselected column
  chunks are never decoded. Measured 0.75x–0.93x (faster) vs reading whole files.
- **`ByteSource` seam** — `parquet/source.mojo:20`, with `MappedFile` (`:49`) as the
  local implementation and `ParquetFile[S: ByteSource]` (`reader.mojo:1950`)
  parameterized over it. Nothing in the decode path asks for the whole file.
- **Parquet** (`marrow/parquet/`) — production-grade *decode/encode breadth*: all
  encodings (PLAIN, RLE/dict, DELTA_*, BYTE_STREAM_SPLIT), all codecs (snappy/gzip/
  zstd/lz4/lz4_raw/brotli via runtime dlopen), DataPage V1+V2, full Dremel nested
  read+write, complete Thrift metadata, statistics, page index, bloom filters.
  Cross-validated against PyArrow both directions.
- **OpenDAL Mojo binding** (`~/Workspace/opendal/bindings/mojo`) — capable WIP:
  operator verbs (read/write/list/stat/delete/copy/rename), **seek-based ranged
  reads** (`Reader.seek` + `read_exact`), backends compiled into `libopendal_c`
  (fs/s3/http/memory). Blocking-only (no async). **Zero integration with marrow.**

### 2.2 The gaps that block "usable engine" status

Ordered by how hard they block the milestones. Each was re-verified against the tree
at `b2e7dae`. Four gaps from the 2026-07-24 list are **closed and deleted**: the
lane op-asymmetry gap (the 41-tag interpreter it described no longer exists — a
runtime node reaches the same kernels the fused node does), whole-file scan reads and
projection-into-scan (T2.4), the `ByteSource` seam, and most of the kernel-coverage
hole (Kleene, `case_when`, `coalesce`/`nullif`/`fill_null`, `is_in`, string ordering
and `LIKE`/`ILIKE`, temporal extraction, `date_trunc`, string/temporal min-max, null
propagation in `length`/`array_length`).

**G1 — No lazy Python frontend at all.** The entire Python surface is *eager*:
`python/marrow/` is `__init__.py` + `compute.py` + `parquet.py`, PyArrow-shaped
compute functions and `RecordBatch` methods that execute immediately over a single
in-memory batch. `python/bindings/` has ten files and **binds no plan or expression
type** — there is no `bindings/lazy.mojo`, and the only expr contact is
`bindings/compute.mojo:17` importing aggregate functors. There is **no** `Expr`,
`col()`, `Column`, `LazyFrame`, or deferred plan in Python. The lazy engine exists in
Mojo and **zero of it is reachable from Python.** ClickBench today
(`python/marrow/tests/clickbench.py`) is **11 eager queries** over six numeric
columns with PyArrow doing the I/O — its own docstring concedes the restriction.

**G2 — No optimizer / plan-rewrite layer.** No `optimize.mojo` exists. Two ad-hoc
rewrites live in the *builder* instead: predicate → `ParquetScan`
(`relations.mojo:437-443`, non-recursive, fires only when a `Filter` sits directly on
the scan) and `Limit` → `Sort` top-K (`:723-735`). Missing: conjunct splitting
(`Filter` holds one `predicate: BoxedValue`, `relations.mojo:866`, not a list),
recursive predicate pushdown, projection pushdown, pushdown below joins, constant
folding, CSE, limit pushdown into the scan, join reordering.
`referenced_columns()` — the prerequisite — **is implemented** on both lanes and the
box (`values.mojo:348`, `dynamic.mojo:532`, `relations.mojo:273`) and is **currently
called only by tests.** The optimizer is its consumer.

**G3 — The runtime-typed dispatch bound is too narrow, so pruning silently never
fires on dates.** `NumericCompareKernel.apply` is bound on `PrimitiveType` but
`dispatch` narrows to `NumericType` (`numeric.mojo:552` vs `:570`). Runtime-typed
comparison on timestamp/date/time/duration/interval/decimal raises; `equal_any`
(`:602-605`) raises, so hash joins and `nullif` on those key types fail; and
`pruning.mojo:115-135` mirrors the bound, so **no row group or page is ever pruned on
a temporal or decimal predicate** — every ClickBench query filtering `EventDate` /
`EventTime` gets no pushdown at all. This is the defect class CLAUDE.md's *"dispatch
on the widest family the typed leaf accepts"* rule exists for; `filter`/`take`
(`filter.mojo:97`) and `sort` (`sort.mojo:433`) are already correct.

**G4 — Missing relational operators.** No `Distinct`, `Union`, `CrossJoin`/
`NestedLoopJoin`, `Values`/`EmptyRelation`, or `Window` node — `relations.mojo` has
eight nodes and no discriminant for any of them. Join keys are **bare column
references only**: a computed key raises (`relations.mojo:597,606`).

**G5 — Scan is single-file and local.** `ParquetScan.path` is a single `String`: no
glob, no dataset, no partition discovery, no multi-file fan-out. The `ByteSource`
seam exists but has **no remote implementation** — OpenDAL is unwired. **Bloom
filters are fully implemented in the reader and never consulted by the scan** (zero
`bloom` hits under `marrow/expr/`) — the cheapest pruning tier still on the table.
Predicate pruning turns *off* for nested files rather than misaligning statistics
with the projection, which is safe but leaves nested files unpruned.

**G6 — Blocking operators buffer everything; no spill; no plan-level parallelism.**
`AggregateProcessor` (`execution.mojo:699`) groups incrementally but buffers every
morsel's group ids and evaluated value columns; `JoinProcessor` (`:837`) collects the
entire left side (`:878`) and always builds on it. Zero occurrences of `spill` in the
tree — no memory budget, no disk I/O. The pull loop is single-threaded, and
`Join.to_processor` **discards the `ExecutionContext` it is handed**
(`relations.mojo:1114-1123`), so the relational join never takes the parallel path
even though the kernel has one.

**G7 — Remaining kernel holes.** No regex engine anywhere in the repo, and with it no
`replace_substring(_regex)` / `extract_regex` / `split_pattern` / `count_substring` /
`find_substring` / substring-slice / `lpad`/`rpad` / `utf8_is_*` family. No **decimal
arithmetic** at all (rescale exists only inside `cast`, and its up-scale multiply has
no overflow check). `date_trunc` covers only second/minute/hour/day
(`temporal.mojo:402-405`) — **month/quarter/year are missing**, which is how
ClickBench Q35/Q36 fail. No `strftime`/`strptime` (string↔timestamp cast raises,
`cast.mojo:1028`), no timezone-aware extraction, no temporal arithmetic. `resolve_agg`
is a closed list of exactly eight functions (`expr/aggregates.mojo:194-221`) — no
`variance`/`stddev`/`quantile`/`median`/`mode`/`first`/`last`. No `unique` /
`value_counts` / `dictionary_encode`: marrow consumes dictionaries and **can never
produce one**. `sort_indices(limit=…)` is a full sort then truncation
(`sort.mojo:379`). No bitwise/shift kernels, no `greatest`/`least`, no overflow
checking.

**G8 — Window functions are AOT-only and a toy** (invariant #2 violation).
`values.mojo:1975-2039` has `RowNumber` and nothing else: `WindowSpec` carries frame
bounds but no PARTITION BY and no ORDER BY, `FrameBound.kind` is an untyped `UInt8`
never read, `RowNumberKernel` ignores its `values` argument, and nothing outside
`values.mojo` references any of it. `lane-shape-window-design.md` §7 is the design.

**G9 — AOT relational monomorphization deferred.** F2 works today only at the
*expression* level (a fused `Value` erased into `BoxedValue` inside a runtime
`DynRelation`). The fully-typed relational plan (`Table[T]`, `Project[*Es]`, typed
`HashJoin[L,R,LK,RK]`, `Env`/`Param` late binding) that would yield a fully-DCE'd
*relational* binary is unbuilt and partly blocked on a Mojo `reflect` resolution bug
(recorded at `values.mojo:2416-2421`).

**G10 — Data-model computational thinness.** `ChunkedArray` (`arrays.mojo:2040`) has
only `chunk()` and `combine_chunks()` — no `slice`/`__getitem__`/`null_count`/
`filter`/`cast`/`take`. `Table` (`tabular.mojo:444`) has no `slice`/`select`/
`filter`/`sort`/`concat_tables`; all of those are `RecordBatch`-only. Kernels target
`DynArray`, not `ChunkedArray`.

---

## 3. Architecture decisions

These shape many tasks; they are settled unless a measurement overturns one.

- **D1 — F1 drives the runtime lane (`DynValue`); F2 drives the fused AOT lane. Both
  erase into `BoxedValue`, and both go through the same `DynRelation` plan and
  `execute()`.** Do *not* build a second execution path for Python. The Python
  `LazyFrame.collect()` maps to `plan.execute(ctx)`. Since the two-lane refactor there
  is **no interpreter to grow**: a runtime node carries a pointer to its evaluator, so
  it reaches the same kernels the fused node does. The remaining work is entirely
  binding + Python wrappers (G1), not op-by-op parity.
- **D2 — Grow the relational layer as *erased* nodes (`DynRelation`) first.** Add
  `Distinct`/`Union`/`Window` as runtime relational nodes over `BoxedValue`. Defer the
  fully-typed `Project[*Es]`/`Table[T]` monomorphized relational plan (G9) to M3+ — it
  is an optimization of F2, not a correctness gate, and it is blocked on a compiler
  bug. F2's binary-size win is already delivered at the expression level.
- **D3 — The `ByteSource` seam is the one abstraction for streaming and remote.**
  *Done*: `parquet/source.mojo:20`, `MappedFile` local, `ParquetFile[S: ByteSource]`.
  What remains is the OpenDAL-backed implementation. Note the standing constraint —
  `Buffer` requires 64-byte pointer alignment, so `read_at` cannot hand back a
  sub-`Buffer` at an arbitrary file offset; a source owns one whole-object `Buffer` and
  hands out `BufferView`s. That constraint has already blocked two designs.
- **D4 — Optimizer is a rule-list over the immutable `DynRelation` IR**, mirroring
  DataFusion. Each rule is a pure `DynRelation -> DynRelation` rewrite.
  `referenced_columns()` already exists on both lanes and the box and needs no new
  metadata; `is_deterministic()` was removed as a default-True method with no caller
  and should not be reintroduced until a rule actually needs it. The one structural
  prerequisite is **splitting `Filter.predicate` into a conjunct list** — partial
  pushdown is not expressible while it is a single `BoxedValue`. Gate: the rewrite must
  not grow the fused lane's binary; an open dispatcher in the optimizer breaches
  invariant #1.
- **D5 — Correctness before features.** *Kleene null semantics are done*
  (`boolean.mojo:141`). The live instance of this rule is `backlog.md` Wave 1: nine
  defects that produce wrong answers with no error (multi-key sort instability,
  multi-morsel outer/semi/anti joins, silent compressed-IPC garbage). ClickBench's
  gate is "results match DuckDB", so these outrank every feature below.
- **D6 — Vectorized batch size for the runtime lane** stays at the current morsel
  granularity for operators, but expression evaluation should process cache-resident
  sub-batches (~2K rows) internally where it matters (matches DuckDB's
  `STANDARD_VECTOR_SIZE`). Not an M1 blocker; revisit at M2 perf tuning.

---

## 4. Capability backlog (MoSCoW, by workstream)

Priority is relative to reaching **M1 (ClickBench)** unless noted. **Must** = required
for M1. **Should** = required for M2 or a correctness/UX necessity. **Could** = M3 or
polish. **Won't** = explicitly out of scope for this roadmap. Completed rows are
deleted, not struck through. Concrete work items, sizes and statuses live in
[`backlog.md`](backlog.md); the IDs in the Notes column point there.

### 4.1 Frontends & API surface

| Feature | MoSCoW | Notes / entry points |
|---|---|---|
| Bind `DynValue`, `BoxedValue` and `DynRelation` through `PythonModuleBuilder.add_type` | **Must** | M1.2. `python/bindings/lazy.mojo` does not exist. Watch trait/associated-type elaboration hazards (CLAUDE.md). |
| Python `Column` wrapper over `DynValue` (forward `+ - * / == < > & \| ~`, `.cast`, `.is_null`, `.isin`, reductions, string/temporal methods) | **Must** | M1.3. The dunders already exist in Mojo (`dynamic.mojo:634-691`); the wrapper is pure Python. |
| Python `col()`, `lit()`, `if_else`, `coalesce`, `case_when` | **Must** | M1.3. The Mojo runtime-lane builders exist (`values.mojo:2476-2513`) and `lit` already takes a runtime dtype — this is binding, not new capability. |
| Python **ibis-flavored** `Table` over `DynRelation` (`.filter/.select/.mutate/.group_by/.aggregate/.order_by/.limit/.join`) building, not executing; `.execute()`/`.to_pyarrow()` | **Must** | M1.3. Thin native wrapper — **not** an ibis backend, no `ibis` dep. The fluent Mojo API is complete (`relations.mojo:406-735`). |
| `marrow.read_parquet(path/glob)` / `marrow.table(schema)` returning a `Table` | **Must** | Wraps `parquet_scan`; glob needs §4.6. |
| Route Python eager `join/group_by/sort_by` through `execute(plan)` (unify paths) or keep documented eager shortcuts | **Should** | Today they bypass the expr layer (`tabular.mojo`). Avoid two divergent code paths. |
| Mojo AOT DSL: `col("a", int64)`-style builders documented end-to-end for M1 query shapes | **Must** | M1.6. Builders exist (`values.mojo:2426-2470`); `docs/guide/expressions.qmd` predates the two-lane split entirely and must be rewritten, not patched. **The two-lane split, `BoxedValue`, and the lane-choice/binary-size trade-off are undocumented anywhere user-facing.** |
| Mojo AOT DSL: fully-typed relational plan (`Table[T]`, `Project[*Es]`, typed joins) | **Could** | G9; blocked on the `reflect` bug (`values.mojo:2416-2421`); an F2 binary optimization, not correctness. |
| SQL string parser / frontend | **Won't** | Out of scope. Both frontends are programmatic. Revisit post-M3. |
| Full **ibis backend** (`ibis.backends.marrow`, translating the ibis op graph) or an `ibis` runtime dependency | **Won't** | ibis is a *loose naming guideline* only. A real backend drags in ibis's op catalog, type coercion, and backend test contract — over-engineering. Ship the thin native `Table`/`Column`. |
| `pandas`-style eager `DataFrame` API | **Won't** | Deferred-only per the brief; the existing eager RecordBatch surface stays as-is. |

### 4.2 Relational operators (the `DynRelation` IR + `Processor` tree)

| Operator | MoSCoW | Notes |
|---|---|---|
| Join on **computed keys** (not just a bare column reference) | **Should** | G4; raises today at `relations.mojo:597,606`. H2O/TPC-H. |
| `Distinct` / `unique` node | **Should** | M2.1. Needs the `unique` kernel (§4.5). H2O. |
| `Union` / `concat` relational node | **Should** | M2.1. TPC-H, set ops. |
| `CrossJoin` / `NestedLoopJoin` (non-equi / theta joins) | **Could** | M3.1. `JOIN_CROSS`/`JOIN_MARK`/`JOIN_SINGLE`/`JOIN_ASOF` are declared constants that are never implemented — a CROSS join currently falls into the LEFT/RIGHT/FULL tail and produces wrong output. |
| `Window` relational node (partition/order/frame) | **Could** | M2.3/G8; see §4.5 and `lane-shape-window-design.md` §7. |
| `Values` / `EmptyRelation` literal sources | **Could** | Convenience / optimizer targets. |
| `Scan` trait above the file formats | **Could** | M3.5. `RELATION_PARQUET_SCAN` is an IR discriminant and `execution.mojo` imports six symbols straight from `..parquet`; `ParquetScanProcessor` does four jobs. **Do this before adding CSV or IPC sources, not after.** |

### 4.3 Execution runtime

| Feature | MoSCoW | Notes |
|---|---|---|
| Give `JoinProcessor` the plan's `ExecutionContext` | **Must** | G6. `Join.to_processor(ctx)` drops it (`relations.mojo:1114-1123`), so the relational join never uses the parallel path. Small, high value. |
| Plan-level parallelism (drive the parallel join/groupby kernels; partition or morsel-driven) | **Should** | G6. Kernels are parallel; the pull loop isn't. M2 perf. |
| Spill-to-disk for `Aggregate` build + `Join` build (bounded memory) | **Should** | M2.5. Zero occurrences of `spill` in the tree. H2O at 50 GB needs a grace hash join and a spilling grouper. |
| `CoalesceBatches` (compact small morsels after filter) | **Should** | Vectorized-execution efficiency; nothing exists today. |
| O(N) top-K (`select_k_unstable` / quickselect / streaming heap) | **Should** | M3.4. `sort_indices(limit=…)` is a full sort then truncation; the docstring concedes it (`sort.mojo:379`). Every ORDER BY … LIMIT in ClickBench and TPC-H. |
| Late materialization / selection vectors on the scan (decode filter cols, filter, then decode survivors) | **Could** | DataFusion keeps this off-by-default; subtle. M3. |
| Async / prefetch ranged reads for remote scans | **Could** | OpenDAL C ABI is blocking; needs multiple readers. M3. |

### 4.4 Optimizer / plan rewrite

Nothing in this section exists; see G2. `referenced_columns()` is already implemented
and unconsumed — these rules are its first caller.

| Rule | MoSCoW | Notes |
|---|---|---|
| **Conjunct splitting** — `Filter.predicate` becomes a list | **Must** | D4. The structural precondition for partial pushdown (`relations.mojo:866`). |
| **Projection pushdown** (prune unused columns to the scan) | **Must** | Highest ROI on columnar/Parquet. A `ParquetScan`'s schema *is* its projection, so this is a schema rewrite — the mechanism already landed with T2.4. |
| **Predicate pushdown** (recursive, through `Project`/`Sort`/`Limit`) | **Must** | Replaces the non-recursive builder special-case at `relations.mojo:437-443`. Feeds the already-wired Parquet row-group/page pruning — but see G3: without the widened compare bound, temporal predicates prune nothing. |
| **Limit pushdown** (limit → top-K in sort; limit into scan) | **Should** | The `Limit`→`Sort` fold exists in the builder (`relations.mojo:723-735`); generalize it into a rule. |
| **Constant folding / expression simplification** | **Should** | Cheap; enables other rules. Salvage predicate-normalization-at-construction from the deleted `aot-query-compilation.md`. |
| Predicate pushdown **below joins** | **Should** | H2O/TPC-H. |
| **CSE** (common-subexpression elimination) | **Could** | Matters for generated/repeated subexprs. See `design-expression-evaluation.md`. |
| **Join reordering** + build-side selection (needs cardinality estimation) | **Could** | M3.2. `hash_join` always builds on `left` (`join.mojo:754`). Defer until multi-join benchmarks. |
| Subquery decorrelation, outer-join elimination | **Won't** (this roadmap) | Needs SQL frontend; revisit post-M3. |

### 4.5 Kernels & expressions

Every kernel must expose **both** `core[W]` (fused, for F2) *and* `apply`/`dispatch`
(eager, for F1), and be reachable from a fused `Value` node *and* a `DynValue` node.

| Feature | MoSCoW | Notes |
|---|---|---|
| Widen `NumericCompareKernel.dispatch` from `NumericType` to `PrimitiveType` | **Must** | G3/M1.0. **Do this first** — smallest change, largest blast radius; pruning correctness is a prerequisite for measuring the optimizer. Also unblocks decimal/temporal aggregates (`aggregate.mojo:102`, `:864`). |
| `date_trunc` at **month, quarter, year** | **Must** | G7/M1.4. Only second/minute/hour/day exist (`temporal.mojo:402-405`); ClickBench Q35/Q36 fail as *queries* without them. |
| `regexp_replace` (ClickBench Q29) | **Should** | M1.4 — or formally defer Q29 to M2 and record it. Needs a regex engine decision; there is none in the repo. |
| `regexp_match` / `regexp_extract` / `split_pattern(_regex)` and the rest of the string hole (`replace_substring`, `count_substring`, `find_substring`, substring-slice, `lpad`/`rpad`, `binary_join`, `utf8_is_*`, trim-with-charset) | **Should** | M2.6, the single largest kernel hole. Also: string kernels dispatch on `is_string_like()` only, so `binary`/`large_binary` are excluded from string comparison. |
| Temporal completeness — `strftime`/`strptime` (string↔timestamp cast raises, `cast.mojo:1028`), timezone-aware extraction (everything decomposes as UTC and a non-UTC `tz` is silently ignored), `week`/`iso_week`/`iso_year`, sub-second extractors, `is_leap_year`, `ceil`/`round_temporal`, the `*_between` family | **Should** | M2.7. |
| Temporal **arithmetic** (date ± interval, `date_diff`, `now`) | **Should** | H2O/TPC-H date logic. |
| **Decimal arithmetic** (add/sub/mul/div with scale rules) + decimal compare/agg | **Should** | TPC-H money math. Nothing was built — no scale-alignment rule, no 256-bit intermediate promotion; rescale exists only inside `cast` and its up-scale multiply has **no overflow check**. |
| Aggregates: `variance`/`stddev`/`quantile`/`approximate_median`/`mode`/`first`/`last` | **Should** | M2.4. `resolve_agg` is a closed list of exactly 8 (`expr/aggregates.mojo:194-221`); TODOs acknowledge the variance gap at `aggregate.mojo:563,574,589`. |
| `unique` / `value_counts` / `dictionary_encode` | **Should** | M2.2. `distinct.mojo` is cardinality-only; **marrow consumes dictionaries but can never produce one**, not from a kernel and not from Parquet. Powers the `Distinct` node. |
| **Window kernels** (`row_number`/`rank`/`dense_rank`/`lag`/`lead`/`ntile`, running/rolling aggs) | **Could** | M2.3/G8. Today a 2-node AOT-only toy (`values.mojo:1975-2039`) — an invariant #2 violation. Design in `lane-shape-window-design.md` §7. |
| Widen the fused conditional nodes | **Could** | `ConditionalBinary` is 2-ary and `CaseWhen` is 1-branch and numeric-only (`values.mojo:2102`) while the kernels and runtime builders are variadic and type-general. |
| Integer overflow checking; `tan`/`asin`/`acos`/`atan`/`atan2`/hyperbolics/`cbrt`; bitwise/shift kernels | **Could** | Long tail; none exist. |
| `greatest`/`least` over N args | **Could** | Convenience. |
| Set-op kernels (intersect/except) | **Could** | Post-M3. |

### 4.6 Scan / IO / Parquet / OpenDAL

| Feature | MoSCoW | Notes |
|---|---|---|
| Bloom-filter pushdown in the scan (equality predicates) | **Should** | G5. Fully implemented in the reader, **never consulted** — zero `bloom` hits under `marrow/expr/`. Cheapest remaining pruning tier. |
| Multi-file / directory / glob dataset scan (list → fan-out → concat schema) | **Should** | M2.8. `ParquetScan.path` is a single `String`. `Operator.list` gives listing; no consumer. |
| **OpenDAL-backed `ByteSource`** (whole-object read, then ranged reads via `Reader.seek`) | **Should** | G5. The seam exists (`parquet/source.mojo:20`); zero integration. Mind the 64-byte `Buffer` alignment constraint (D3). |
| Enable predicate pruning for **nested** files | **Could** | Pruning turns off rather than misaligning statistics with the projection — safe, but nested files are unpruned. |
| Reduce the AOT scan's binary cost | **Could** | Q4.6: `query_scan_stripped` 2,449,024 vs `query_streaming_stripped` 1,373,704 = **1.78×**. |
| Hive-style partition discovery (`col=val` dirs) | **Could** | TPC-H/dataset ergonomics. |
| Async/prefetch remote reads (multiple OpenDAL readers) | **Could** | M3. Blocking C ABI constraint. |
| Preserve struct-level nulls over repeated groups on write; `large_*` offset width | **Could** | Known writer asymmetries; not engine blockers. |

### 4.7 Data model / types

| Feature | MoSCoW | Notes |
|---|---|---|
| `ChunkedArray` computational surface (`slice`, `__getitem__`, `null_count`, `filter`, `cast`, `take`) | **Should** | G10/M3.6. Only `chunk()`/`combine_chunks()` today (`arrays.mojo:2040`). The engine operates on Tables, so columns must behave like columns. |
| `Table`-level ops (`slice`/`select`/`filter`/`sort`/`rename`/`concat_tables`) | **Should** | G10/M3.6. `RecordBatch` has these; `Table` (`tabular.mojo:444`) doesn't. |
| Python `ChunkedArray` wrapper class | **Could** | Imported but never registered. Interop completeness. |
| `union` type (sparse/dense) | **Could** | Only needed for specific plans; not a first-cut blocker. |
| Run-end-encoded (REE) arrays | **Won't** (this roadmap) | Compression optimization. |
| `string_view` / `binary_view` layouts | **Won't** (this roadmap) | Perf optimization; big surface. |

### 4.8 Cross-cutting

| Item | MoSCoW | Notes |
|---|---|---|
| Get CI running again, and add the binary-size gate to it | **Must** | `backlog.md` Wave 2. CI has not passed since 2026-05-11; the main test job invokes a task that no longer exists, and `binary_size` — the project's central architectural invariant — has **zero hits under `.github/`**. |
| Re-baseline the binary-size numbers | **Must** | I4. The recorded 7.6×/7.8×/12.8× table predates the interpreter deletion. Current `__text`: `query_streaming` (AOT) 1,302,900 vs `query_dynvalue` (runtime) 3,984,756 ≈ **3.06×**. Using the written numbers as a gate would invent or hide a regression. |
| Keep `benchmarks/binary_size/` green at every gate; add a relational-plan size bench | **Must** | Invariant #1. |
| End-of-wave **quality review** (`/simplify` + abstraction/duplication/free-function audit) gating the next wave | **Must** | Invariant #4. Not cleanup-later — an acceptance gate. |
| Both-lane parity tests (fused `Value` result == `DynValue` result) per op | **Must** | Invariant #2. The refactor removed the structural reason they diverged; the tests are what keep them from diverging again. |
| Cross-engine aggregate benchmark **with the AOT path measured** | **Must** | Q6.1. Every comparison table is still one row (`marrow-dynamic`) — no `*_aot.mojo` benchmark exists, so **F2, the differentiator, is unmeasured.** |
| ClickBench-subset harness runs *through the lazy plan* (not eager one-batch) | **Must** | M1.5, the M1 sign-off. Currently 11 eager queries, PyArrow doing the I/O. |
| PyArrow/DuckDB cross-check in the test suite for new kernels/ops | **Should** | Existing pattern. |
| `EXPLAIN` / plan pretty-printer for both frontends | **Should** | Debuggability; `BoxedValue.render()` and `DynRelation.write_to` exist. |
| Spill/mem-budget config on `ExecutionContext` | **Could** | M2. |

---

## 5. Milestone plan (ordered)

Each milestone lists its ordered workstreams. Land features so the suite + binary-size
gate stay green after every step.

### M0 — Unblock (foundations)

Prerequisites everything else builds on. Small, high-leverage.

1. **Clear `backlog.md` Wave 1** — the nine wrong-answer defects, each opening with a
   failing test. *(Must — D5. "Results match DuckDB" is the M1 gate; a wrong multi-key
   sort corrupts exactly what the milestone measures.)*
2. **Widen `NumericCompareKernel.dispatch` to `PrimitiveType`** (G3). *(Must — smallest
   change, largest blast radius; without it no temporal predicate ever prunes, so the
   optimizer cannot be measured.)*
3. **Restore CI and put `pixi run binary_size` in it**; re-baseline the size table
   (`backlog.md` Wave 2). *(Must — none of the work below is verified by anything but
   local runs.)*
4. **Give `JoinProcessor` its `ExecutionContext`** (G6). *(Must — one-line class of fix,
   unlocks the parallel join the kernel already implements.)*

### M1 — ClickBench: the usable analytical core (the "ship it" line)

Ordered:

1. **Kernels for ClickBench**: `date_trunc` at month/quarter/year; decide `regexp_replace`
   vs formally deferring Q29. *(§4.5 Must.)*
2. **Optimizer v1**: conjunct splitting, then projection pushdown + recursive predicate
   pushdown + limit pushdown, feeding the already-wired Parquet row-group/page pruning.
   *(§4.4 Must — M1.1.)*
3. **Frontends**: bind `DynValue`/`BoxedValue`/`DynRelation` to Python; build the thin
   `Table`/`Column` wrapper and `read_parquet`. *(§4.1 Must — M1.2, M1.3.)*
4. **Docs**: rewrite `docs/guide/expressions.qmd` around the two lanes, `BoxedValue`,
   and the lane-choice/binary-size trade-off, with a runnable AOT example.
   *(§4.1 Must — M1.6.)*
5. **Gate**: all 42 ClickBench queries pass on both frontends through the lazy plan,
   results match DuckDB, wall-clock compared to polars/duckdb on the same box,
   `binary_size` green, and the AOT path actually measured. *(M1.5 + Q6.1.)*

### M2 — H2O: joins & aggregates at scale

1. Join on computed keys; `variance`/`stddev`/`quantile`/`median`/`mode`/`first`/`last`;
   `unique`/`value_counts`/`dictionary_encode`; `Distinct` + `Union` nodes.
   *(§4.2/§4.5 Should.)*
2. Plan-level parallelism (drive parallel join/groupby kernels; morsel or partition).
   *(§4.3 Should.)*
3. Spill-to-disk for aggregate/join build; `CoalesceBatches`; mem budget on
   `ExecutionContext`; O(N) top-K. *(§4.3 Should.)*
4. Multi-file/glob dataset scan + **OpenDAL-backed `ByteSource`** (remote scan) +
   **bloom pushdown**. *(§4.6 Should.)*
5. String/regex family and temporal completeness. *(§4.5 Should.)*
6. `ChunkedArray`/`Table` computational surface. *(§4.7 Should.)*
7. Predicate pushdown below joins; constant folding. *(§4.4 Should.)*
8. Real window functions, on **both** lanes — closing the invariant #2 violation.
   *(§4.5 Could, but the invariant makes it a debt, not a feature.)*
9. **Gate**: H2O group-by + join pass at 5 GB, spill works, competitive wall-clock.

### M3 — TPC-H: full relational breadth

1. `CrossJoin`/`NestedLoopJoin` (non-equi/theta); temporal arithmetic + decimal
   arithmetic. *(§4.2/§4.5 Could/Should.)*
2. `Scan` trait above the file formats, before any second format lands.
   *(§4.2 Could — M3.5.)*
3. CSE; **join reordering** with cardinality estimation and build-side selection.
   *(§4.4 Could.)*
4. Late materialization / selection vectors; async prefetch remote reads. *(§4.3
   Could.)*
5. **F2 relational monomorphization** (`Table[T]`/`Project[*Es]`/typed joins) once
   the `reflect` bug clears — the fully-DCE'd relational binary. *(§4.1 Could; G9.)*
6. **Gate**: TPC-H runs; join reordering measurable on multi-join queries.

### Beyond (explicitly deferred)

SQL-string frontend; distributed/multi-node; server mode; `union`/REE/view types;
subquery decorrelation; set-op kernels; Hive partition discovery at scale. The
Arrow-parity gaps that no milestone query needs are enumerated once in
[`backlog.md`](backlog.md) §7 and are not scheduled here.

---

## 6. How to use this document

- This file is **scope and sequencing**. It says *what* a milestone requires and *why*
  in that order. It does not track work — [`backlog.md`](backlog.md) does, with IDs,
  sizes and statuses, and it is verified against the code rather than against a header.
  When a capability here lands, delete its row; when the two disagree, `backlog.md`
  wins.
- Enforce the **invariants** (§1) as CI/review gates: `pixi run binary_size` and the
  both-lane parity tests are non-negotiable at every merge.
- **Re-verify a status line with `grep` before acting on it.** A 2026-08-03 audit of
  the predecessor task documents found 18 wrong statuses; this document's own §2.2 was
  four gaps out of date within ten days.
- Prefer the **prior-art path** (CLAUDE.md): consult Arrow C++/Rust and DataFusion/
  Polars/DuckDB for kernel semantics, null handling, offset rules, and operator
  design before writing new code.

---

## Appendix — reference-engine notes (external research)

- **Minimum usable engine** = `{TableScan (+Parquet projection/predicate pushdown),
  Projection, Filter, HashAggregate, HashJoin, Sort, Limit/TopK}` operators +
  `{projection pushdown, predicate pushdown, constant folding, limit pushdown}`
  optimizer rules + vectorized batches (~2048) with morsel-driven or partition
  parallelism. marrow has every operator on that list; it has **none** of the
  optimizer rules.
- **DataFusion pipeline**: LogicalPlan → OptimizerRules → ExecutionPlan (pull-based
  async `RecordBatch` streams, partition-parallel). Baseline logical rules:
  `push_down_filter`, `push_down_limit`, `optimize_projections`,
  `simplify_expressions`, `common_subexpr_eliminate`, `eliminate_cross_join`,
  trivial-node eliminations, `type_coercion` (analyzer). Physical: join-selection,
  sort enforcement/removal, repartition, coalesce-batches, filter-into-scan.
- **Parquet pushdown tiers** (implement 1→2→4 first): (1) column projection,
  (2) row-group stats pruning, (3) page-index pruning, (4) bloom filters,
  (5) row-level late-materialization pushdown (subtle; DataFusion ships it *off* by
  default). marrow has 1–3 at the reader **and wired into the scan**; **4 (bloom) is
  implemented in the reader and never consulted.**
- **Vectorized execution**: DuckDB `STANDARD_VECTOR_SIZE = 2048` (L1/L2-resident);
  morsels ~10K–100K for parallelism; push-based pipelines fuse non-materializing
  operators; pipeline breakers = agg/join build, sort. Pull-based (DataFusion) is
  fully competitive and maps naturally onto Mojo iterators.
- **Compiled vs vectorized** (Kersten et al., VLDB 2018): neither dominates;
  compilation wins compute-bound/cache-resident (fewer instructions, register-resident
  tuples), vectorization wins memory-bound (latency hiding). Modern systems go hybrid.
  **This maps cleanly onto marrow's two lanes**: F2 (Mojo `comptime`
  monomorphization) = HyPer/Umbra-style compiled small binaries; F1 (the runtime
  `DynValue` lane over `DynArray`) = the vectorized path. Small binaries come
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
