# Marrow Execution-Engine — Task Breakdown & Orchestration

Companion to [`execution-engine-roadmap.md`](execution-engine-roadmap.md). This file
turns the roadmap into **discrete, worktree-ready tasks** and schedules them into
**waves whose in-flight tasks have disjoint write-sets**, so parallel subagents in
separate git worktrees don't collide.

Read the roadmap first for the *why*. This file is the *what/who/when* for handing work
to agents. **Nothing here is started yet** — it is the orchestration plan.

---

## 1. Orchestration model

**One task = one worktree = one branch = one owned write-set.** An agent may *read* any
file but may only *write* the files its card lists under **Owns**. The wave schedule
(§4) guarantees that within a wave, every owned file has exactly one writer.

### Ownership rules

1. **At most one in-flight task writes a given source file.** Enforced by the wave
   schedule. If two ready tasks want the same file, they go in different waves (or
   merge into one task).
2. **Prefer new files.** New kernels, new optimizer passes, and the whole Python lazy
   frontend land as *new* files so they parallelize freely. New-file tasks almost never
   conflict.
3. **Hotspot files get a single owner per wave** (see §2). The schedule names the owner.
4. **Registration/index files are trivial-conflict, resolve at merge — not blockers.**
   These are `marrow/expr/__init__.mojo`, `python/bindings/lib.mojo`,
   `python/marrow/__init__.py`, and `CHANGELOG.md`. Each task appends its own
   export/registration/changelog line; if two land in the same wave, the merge is a
   one-line union. Do **not** serialize a wave just for these.
5. **Tests live with their task.** Each task adds or edits *its own* `test_*.mojo` /
   `test_*.py`; never share a test file between concurrent tasks. The cross-driver
   parity harness (T0.6) is the one shared test surface — it is append-only and owned by
   whoever last touched it in a wave.

### Merge protocol

- Merge order within a wave: **new-file tasks first, hotspot-owner tasks second,
  registration-file union last.** Then run `pixi run -e dev test` + `pixi run
  binary_size` on the integrated branch before opening the next wave.
- Every task's Definition of Done includes: its own tests green, the full suite green,
  and `binary_size` within budget (the DCE gate — invariant #1).
- A task that must touch a hotspot it doesn't own is a **scheduling bug** — split it or
  move it to a later wave, don't edit across ownership.

### Worktree hygiene

- Spawn each task with `isolation: "worktree"` so file mutations are isolated.
- Give each agent: its card (goal, Owns, Reads, Acceptance), the roadmap section it
  implements, and the two prior-art pointers (Arrow C++/Rust, DataFusion/Polars/DuckDB).
- Kernel and node work must land **both** drivers where applicable (fused `Value` +
  `DynValue`) — but those are *different files*, so the schedule often splits "kernel",
  "fused wiring", and "dynamic wiring" into sibling tasks that merge together.

---

## 2. Hotspot files (contended — single owner per wave)

| File | Role | Wanted by |
|---|---|---|
| `marrow/expr/values.mojo` | fused comptime `Value` nodes (F2) | metadata, every new fused expr node |
| `marrow/expr/dynamic.mojo` | `DynValue` interpreter (F1) | op-parity growth, every new dynamic op |
| `marrow/expr/relations.mojo` | relational IR nodes | Sort/Limit/Project/Distinct/Union, optimizer hooks |
| `marrow/expr/execution.mojo` | pull-based processors | new operators **and** streaming scan (`ParquetScanProcessor`) |
| `marrow/expr/pruning.mojo` | pushdown stats | predicate-pushdown work |
| `marrow/parquet/reader.mojo` | Parquet read path | `ByteSource` seam, streaming, projection pushdown |
| `marrow/kernels/boolean.mojo` | bool kernels | Kleene fix, `select`/CASE |
| `marrow/kernels/compare.mojo` | compares | string ordering dispatch |

> **The sharpest conflict:** `execution.mojo` is wanted by both *relational-operator*
> tasks and the *streaming-scan* task (`ParquetScanProcessor` lives there). The schedule
> deliberately puts them in **different waves**. Likewise `values.mojo` (fused) and
> `dynamic.mojo` (dynamic) are separate files, so fused-wiring and dynamic-wiring of the
> same feature run in parallel.

New files introduced by this plan (conflict-free): `marrow/kernels/conditional.mojo`,
`marrow/kernels/membership.mojo`, `marrow/kernels/temporal.mojo`,
`marrow/kernels/window.mojo`, `marrow/parquet/source.mojo`, `marrow/expr/optimize.mojo`,
`python/bindings/lazy.mojo`, `python/marrow/expr.py`, plus their `test_*` files.

---

## 3. Task cards

Format: **ID — Title** · *milestone/priority* · **Depends:** … · **Owns:** … ·
**Reads:** … · **Done when:** …

### Wave 0 — M0 unblock (fully parallel; disjoint files)

**T0.1 — Kleene 3-valued boolean logic** · *M0/Must* · Depends: — · Owns:
`marrow/kernels/boolean.mojo`, `marrow/kernels/tests/test_boolean.mojo` · Reads:
`bitmap.mojo`, `helpers.mojo` · Done when: `and/or/xor/not` implement SQL Kleene truth
tables (`TRUE OR NULL = TRUE`, `FALSE AND NULL = FALSE`, else NULL), validity no longer
dropped; also fix `LengthKernel`/`ArrayLengthKernel` null propagation
(`string.mojo`/`nested.mojo` — **add to Owns** if touched, they're otherwise free this
wave); cross-checked vs PyArrow.

**T0.2 — `ByteSource` seam in the reader** · *M0/Must* · Depends: — · Owns:
`marrow/parquet/source.mojo` (new), `marrow/parquet/reader.mojo`,
`marrow/parquet/tests/test_reader.mojo` · Reads: `format.mojo`, `codecs.mojo` · Done
when: a `ByteSource` trait (`read_at(offset, len) -> Span`, `size()`) exists;
`MappedFile` implements it; `ParquetFile` can be constructed from any `ByteSource` (not
just a path); **pure refactor, zero behavior change**, existing Parquet tests green.
This unblocks streaming (T2.x) and OpenDAL (M2).

**T0.3 — Fused `Value` plan metadata** · *M0/Must* · Depends: — · Owns:
`marrow/expr/values.mojo`, `marrow/expr/tests/test_values.mojo` · Reads:
`schema.mojo` · Done when: `AnyValue`/fused `Value` expose `referenced_columns() ->
List[String]` and `is_deterministic() -> Bool`; correct for column/literal/binary/unary/
cast/reduce nodes. Prerequisite for projection & predicate pushdown.

**T0.4 — Grow `DynValue` op parity + metadata** · *M0/Must* · Depends: (soft) T0.1 ·
Owns: `marrow/expr/dynamic.mojo`, `marrow/expr/tests/test_runtime.mojo` · Reads:
`marrow/kernels/*` · Done when: `DynValue` reaches parity with the fused algebra on the
ops M1 needs whose kernels already exist — arithmetic (incl. mod/floordiv), numeric
compares, Kleene and/or/xor, is_null/notnull, cast, if_else; plus
`referenced_columns()`/`is_deterministic()` on `DynValue`. (String/temporal/is_in
dynamic wiring is T2.2, after their kernels land.)

**T0.5 — Doc hygiene** · *M0/Must* · Depends: — · Owns: `CLAUDE.md` (the stale
"Table not implemented" / type-coverage / `similarity.mojo`+`sum.mojo` claims) · Reads:
audit findings · Done when: `CLAUDE.md` reflects reality (Table exists; temporal/decimal/
map/dictionary present; those two kernel files gone).

**T0.6 — Cross-driver parity harness** · *M0/Must* · Depends: (soft) T0.3, T0.4 · Owns:
`marrow/expr/tests/test_parity.mojo` (new) · Reads: `values.mojo`, `dynamic.mojo` · Done
when: a reusable helper asserts `fused Value result == DynValue result` for a given
expression over a batch; seeded with the M0 ops. Every later op-adding task appends a
parity case here (append-only; single owner per wave).

*Wave-0 write-sets:* `boolean.mojo` / `source.mojo`+`reader.mojo` / `values.mojo` /
`dynamic.mojo` / `CLAUDE.md` / `test_parity.mojo` — **all disjoint. Run all six in
parallel.** (Merge T0.1 before asserting T0.4's Kleene parity cases.)

### Wave 1 — M1 kernels (fully parallel; all new/disjoint files)

**T1.1 — Conditional kernels** · *M1/Must* · Depends: — · Owns:
`marrow/kernels/conditional.mojo` (new), test · Reads: `boolean.mojo` (select pattern),
`filter.mojo` · Done when: `case_when` (multi-branch, all types), `coalesce`, `nullif`,
`fill_null` kernels with `apply`/`dispatch`, null-correct, all-type (not numeric-only),
PyArrow-checked.

**T1.2 — Membership kernel** · *M1/Must* · Depends: — · Owns:
`marrow/kernels/membership.mojo` (new), test · Reads: `hashtable.mojo`, `hashing.mojo` ·
Done when: `is_in(values, value_set)` over numeric/string/temporal via hash set,
null-correct.

**T1.3 — String ordering + pattern match** · *M1/Must* · Depends: — · Owns:
`marrow/kernels/compare.mojo` (string `<`/`<=`/`>`/`>=` dispatch), `marrow/kernels/string.mojo`
(`like`/`ilike`), tests · Reads: `views.mojo` · Done when: ordering compares dispatch to
string/large_string; `like`/`ilike` (SQL `%`/`_`) implemented; PyArrow-checked. (Regex is
M2 — needs an engine decision; scope this to LIKE.)

**T1.4 — Temporal extraction + `date_trunc` kernels** · *M1/Must* · Depends: — · Owns:
`marrow/kernels/temporal.mojo` (new), test · Reads: `dtypes.mojo` (TimeUnit),
`cast.mojo` · Done when: `year/month/day/hour/minute/second/weekday/quarter/day_of_year`
extraction over date32/64, timestamp(unit,tz), time32/64; **plus `date_trunc(unit)`**
(minute/hour/day…) — ClickBench Q19 (`extract(minute …)`) and Q43 (`date_trunc('minute',
…)`) need these. PyArrow-checked.

**T1.5 — Aggregate kernel completeness (ClickBench)** · *M1/Must* · Depends: — · Owns:
`marrow/kernels/aggregate.mojo`, `marrow/kernels/groupby.mojo`, their tests · Reads:
`hashtable.mojo`, `distinct.mojo` · Done when: **min/max over string and date/temporal**
(whole-table *and* grouped — ClickBench Q7 `MIN/MAX(EventDate)`, Q22/Q23 `MIN(URL)`;
today min/max are numeric-only), and grouped **count_distinct** is exposed through the
grouped-aggregate value path (kernel exists via `count_distinct_grouped`; make it a
first-class agg func the relational node can request). PyArrow-checked. (stddev/median/
first/last stay M2.)

*Wave-1 write-sets:* `conditional.mojo` / `membership.mojo` / `compare.mojo`+`string.mojo`
/ `temporal.mojo` / `aggregate.mojo`+`groupby.mojo` — **disjoint. Run all five in
parallel.** Safe to start alongside Wave 0 (no Wave-0 task owns any of these).

### Wave 2 — M1 wiring + operators + scan (mixed parallel/serial)

Two parallel lanes plus a serialized `execution.mojo` sub-chain.

**T2.1 — Fused wiring of M1 kernels** · *M1/Must* · Depends: T1.1–T1.4 · Owns:
`marrow/expr/values.mojo`, `test_values.mojo` · Reads: the new kernel files · Done when:
fused `Value` nodes for case/coalesce/nullif/is_in/string-compares/like/temporal-extract
exist and fuse; parity cases appended to `test_parity.mojo`.

**T2.2 — Dynamic wiring of M1 kernels** · *M1/Must* · Depends: T1.1–T1.4, T0.4 · Owns:
`marrow/expr/dynamic.mojo`, `test_runtime.mojo` · Reads: the new kernel files · Done
when: `DynValue` tags for the same ops; parity with T2.1 (append to `test_parity.mojo`).

> T2.1 (`values.mojo`) and T2.2 (`dynamic.mojo`) are **disjoint files → parallel**. Both
> append to `test_parity.mojo`; that file gets a single owner this wave (merge the other's
> cases at integration — trivial union).

**T2.3a — Relational Sort + Limit/Offset/TopK + computed Project** · *M1/Must* ·
Depends: — · Owns: `marrow/expr/relations.mojo`, `marrow/expr/execution.mojo`,
`test_streaming.mojo`, `test_plan.mojo` · Reads: `kernels/sort.mojo`, `kernels/filter.mojo`
· Done when: `Sort` node+processor (multi-col, nulls, wraps sort kernel — ClickBench
sorts by agg/column/string/timestamp, Q8/24/25/26/27), `Limit`/`Offset`/`TopK`
node+processor (`OFFSET` needed by Q39–43; limit fuses into sort as top-K), and
computed-column `Project` (evaluates arbitrary `AnyValue`s, names outputs — Q30/35/36/40)
all execute; tested.

**T2.3b — Relational Aggregate completeness + HAVING** · *M1/Must* · Depends: T2.3a,
T1.5 · Owns: `marrow/expr/relations.mojo`, `marrow/expr/execution.mojo`,
`test_streaming.mojo` · ⚠️ *serialize after T2.3a (same files)* · Done when: the
`Aggregate` node/`AggregateProcessor` accepts **computed group keys** (arithmetic/CASE/
extract/date_trunc/literal — Q19/35/36/40/43) and **computed aggregate inputs**
(`SUM(x+1)` Q30, `AVG(STRLEN(URL))` Q28), adds **`count_distinct`** as an agg func
(Q5/6/9–14/23) wiring T1.5's grouped kernel, min/max over string/date value columns
(Q7/22/23), and a post-aggregate **`HAVING`** filter (Q28/29). Tested against those query
shapes.

**T2.4 — Streaming Parquet scan + projection-into-scan** · *M1/Must* · Depends: T0.2,
T2.3b · Owns: `marrow/parquet/reader.mojo`, `marrow/expr/execution.mojo`
(`ParquetScanProcessor`) · ⚠️ *serialize after T2.3a/b (same file)* · Done when:
`ParquetScanProcessor` decodes per-row-group (memory no longer scales with file size)
over a `ByteSource`, and pushes `columns=` (projection) into `read_table`; tested for
memory + correctness.

> **Serialize T2.3a → T2.3b → T2.4** — all three own `execution.mojo` (and 2.3a/b own
> `relations.mojo`). Meanwhile T2.1 + T2.2 run in parallel to the whole chain (they own
> `values.mojo` / `dynamic.mojo`).

*Wave-2 schedule:* **[T2.1 ∥ T2.2 ∥ (T2.3a → T2.3b → T2.4)]**. Three lanes; the third is
a 3-step serial chain on `execution.mojo`. (The chain is the wave's critical path — if it
dominates, note that T2.3b's kernel dep T1.5 must be merged first.)

### Wave 3 — M1 optimizer + frontends (mixed)

**T3.1 — Optimizer v1** · *M1/Must* · Depends: T0.3, T0.4, T2.3a · Owns:
`marrow/expr/optimize.mojo` (new), `marrow/expr/tests/test_optimize.mojo` (new); minimal
accessor additions to `relations.mojo` **only if unavoidable** (prefer read-only over the
existing `kind()`/node API) · Reads: `relations.mojo`, `values.mojo`, `pruning.mojo` ·
Done when: a rule-list `AnyRelation -> AnyRelation` implements projection pushdown
(uses `referenced_columns()`), predicate pushdown with conjunct splitting on AND
(feeding the existing Parquet pruning), and limit pushdown; each rule unit-tested.

**T3.2 — Python bindings for the lazy engine** · *M1/Must* · Depends: T0.4, T2.2, T2.3a ·
Owns: `python/bindings/lazy.mojo` (new); registration line in
`python/bindings/lib.mojo` (trivial-conflict) · Reads: `expr/*` · Done when: `AnyValue`,
`AnyRelation`, `col`/`lit`/`if_else`, plan builders, and `execute(plan, ctx)` are exposed
through `PythonModuleBuilder.add_type`; smoke-tested from Python.

**T3.3 — Python ibis-flavored `Table`/`Column` API** · *M1/Must* · Depends: T3.2 · Owns:
`python/marrow/expr.py` (new), `python/marrow/tests/test_expr.py` (new); export lines in
`python/marrow/__init__.py` (trivial-conflict) · Reads: `lazy.mojo` surface · Done when:
a thin pure-Python `Column` (over `DynValue`) and `Table` (over `AnyRelation`) implement
the **ibis-flavored** surface in §3.1, deferred (build-not-execute), terminating in
`.execute()`/`.to_pyarrow()`. **No ibis dependency, no ibis backend — ibis is a loose
naming guideline, not a parity target.** `marrow.table(schema)` and
`marrow.read_parquet(path)` return a `Table`.

**T3.4 — Mojo AOT DSL docs + example** · *M1/Should* · Depends: T2.1, T2.3a · Owns:
`docs/guide/expressions.qmd`, `examples/` (new example) · Reads: `values.mojo` · Done
when: the M1 query shapes are documented as Mojo `comptime` DSL and a runnable example
builds a small binary within `binary_size` budget.

**T3.5 — ClickBench through the lazy plan** · *M1/Must (gate)* · Depends: T3.3, T2.4,
T3.1 · Owns: `python/marrow/tests/clickbench.py` · Reads: everything · Done when: the
ClickBench subset runs through the **lazy plan** on both F1 (Python) and F2 (Mojo),
results match DuckDB, `binary_size` green. **This is the M1 sign-off.**

*Wave-3 schedule:* **[T3.1 ∥ (T3.2 → T3.3 → T3.5) ∥ T3.4]**, with T3.5 also gated on
T3.1 + T2.4. T3.1 (`optimize.mojo`, new) is disjoint from the Python lane.

### 3.1 Frontend surface (ibis-flavored, deliberately minimal)

The Python F1 surface is a **thin native** API — two pure-Python wrappers over the
already-built engine. It borrows ibis *names and feel*, nothing more: no `ibis` import,
no operation graph, no backend contract, no type-coercion layer. Every method just
builds an `AnyRelation`/`DynValue` and defers; execution is one `execute(plan, ctx)`
call. Scope it to what M1 (ClickBench) needs; grow method-by-method later.

**`Column`** — wraps a `DynValue`; column refs resolve by name against the table schema
(marrow's `DynValue.resolve_names` already does this):

```python
t = ma.read_parquet("hits.parquet")     # -> Table
t.url                                     # -> Column   (attribute access)
t["url"]                                  # -> Column   (item access, for odd names)

# operators (forward to DynValue dunders that already exist)
t.a + t.b,  t.a * 2,  t.price > 30,  (t.a > 0) & (t.b < 10),  ~t.flag
# methods (ibis-ish names; each maps to a DynValue tag / kernel)
t.a.cast("int64"),  t.a.isin([1,2,3]),  t.a.is_null(),  t.a.fill_null(0)
t.a.sum(),  t.a.mean(),  t.a.count(),  t.a.nunique()          # reductions
t.url.like("%google%"),  t.url.contains("x"),  t.name.upper() # string
t.ts.year(),  t.ts.month(),  t.ts.day()                       # temporal
t.a.name("total")                                             # alias output
```

**`Table`** — wraps an `AnyRelation`; methods build and return a new `Table` (immutable,
cheap `ArcPointer` copy). ibis-style names, mapped to existing/near-term relational nodes:

| Method | Maps to | Milestone |
|---|---|---|
| `t.filter(pred)` (accepts `Column` or list, AND-ed) | `AnyRelation.filter` | M1 |
| `t.select(*cols, **named)` | computed `Project` (T2.3a) | M1 |
| `t.mutate(**named)` (add cols, keep the rest) | computed `Project` | M1 |
| `t.group_by(*keys).aggregate(**named)` **and** `t.aggregate(metrics, by=[...])` | `Aggregate` | M1 |
| `t.order_by(*cols)` (a col or `t.a.desc()`) | `Sort` (T2.3a) | M1 |
| `t.limit(n, offset=0)` | `Limit`/`TopK` (T2.3a) | M1 |
| `t.join(right, predicates, how="inner")` | `Join` | M1 (equi) / M2 (computed keys) |
| `t.distinct()` | `Distinct` | M2 |
| `t.union(other)` | `Union` | M2 |
| `t.execute()` / `t.to_pyarrow()` | `execute(plan)` → `RecordBatch` via `__arrow_c_*__` | M1 |
| `t.to_pandas()` | via pyarrow | M1 (thin) |
| `t.schema`, `t.columns`, `repr` (plan pretty-print via `write_to`) | metadata | M1 |

**Deliberately deferred (don't build for M1):** the ibis deferred `_` placeholder
(`from ibis import _`) — offer a tiny optional `ma.deferred`/`_` later if wanted, but
`t.col` access covers ClickBench; window `.over(...)`; `.sql()`; the `Value`/`Column`
type-family class tower (ibis has `IntegerColumn`/`StringColumn`/…) — marrow's one
`Column` class dispatching on the carried dtype is enough. **Don't replicate ibis's class
hierarchy or op catalog to "match" it.**

**Two-frontend symmetry.** This mirrors — loosely — the Mojo AOT DSL (F2), which already
builds the *same* logical query from `col("a", int64)`-style fused builders
(`values.mojo:1605`). Same query shape, two drivers: Python `Table`/`Column` →
interpreted `DynValue`; Mojo `col()` → fused comptime `Value`. Keep the method *names*
aligned across the two where cheap, but do not contort either to force identical spelling.

---

## 4. Wave schedule (at a glance)

```
Wave 0 (M0)   T0.1 boolean   ∥ T0.2 reader/source ∥ T0.3 values-meta ∥
              T0.4 dynamic    ∥ T0.5 CLAUDE.md     ∥ T0.6 parity-harness
              └ all disjoint files → 6-wide parallel
                 (merge T0.1 before T0.4 parity assertions)

Wave 1 (M1)   T1.1 conditional ∥ T1.2 membership ∥ T1.3 compare+string ∥
              T1.4 temporal+date_trunc ∥ T1.5 aggregate(min/max str+date, count_distinct)
              └ all disjoint → 5-wide parallel (may overlap Wave 0)

Wave 2 (M1)   T2.1 fused-wiring(values) ∥ T2.2 dyn-wiring(dynamic) ∥
              [T2.3a Sort/Limit/Project → T2.3b Aggregate+HAVING → T2.4 stream-scan]
              └ 3 lanes; lane 3 is a serial chain on execution.mojo

Wave 3 (M1)   T3.1 optimizer(new) ∥ [T3.2 bind → T3.3 py-frontend → T3.5 ClickBench] ∥ T3.4 docs
              └ T3.5 is the M1 gate

── ship M1 ──

Wave 4+ (M2/M3)  see §5 epics
```

**Dependency DAG (edges = "must merge before"):**

```
T0.2 ─────────────────────────► T2.4 ─────────────► T3.5
T0.3 ─┬───────────────────────► T3.1 ─────────────► T3.5
T0.4 ─┤                          ▲
T1.1..T1.4 ──► T2.1 ─┐           │
             └► T2.2 ┤           │
T1.5 ──► T2.3b       │           │
T2.3a ─► T2.3b ─► T2.4 ──────────┘
T2.2, T2.3a ─► T3.2 ─► T3.3 ─► T3.5
(T0.1 merges before T0.4's Kleene parity cases)
```

Critical path: **T0.2 → (T2.3a → T2.3b → T2.4) → T3.5** — the `execution.mojo` serial
chain now dominates. Keep T1.5 merged before T2.3b so the chain doesn't stall. The
optimizer (T3.1) and docs (T3.4) remain off the critical path.

---

## 5. M2 / M3 epics (decompose when M1 lands)

Detail these into cards at the start of each milestone (scope will shift). Owner-file
notes included now so the conflict structure is already visible.

### M2 — H2O (joins & aggregates at scale)

- **E-M2.1 Advanced aggregates** — `stddev`/`variance`/`median`/`quantile`, `first`/`last`,
  grouped min/max on string/temporal. *Owns* `kernels/aggregate.mojo` + `kernels/groupby.mojo`.
  Parallel to expr work.
- **E-M2.2 Distinct + Union relational nodes + distinct-values kernel.** *Owns*
  `relations.mojo`+`execution.mojo` (serialize vs any other relational-node epic) + new
  `kernels/unique.mojo`.
- **E-M2.3 Computed-key joins + non-equi (Cross/NestedLoop).** *Owns* `relations.mojo`+
  `execution.mojo`+`kernels/join.mojo`. Serialize vs E-M2.2 (shared relational files).
- **E-M2.4 Plan-level parallelism + spill + CoalesceBatches.** *Owns* `execution.mojo`.
  Serialize vs E-M2.2/E-M2.3.
- **E-M2.5 Multi-file/glob dataset scan + OpenDAL `ByteSource` + bloom pushdown.** *Owns*
  `reader.mojo` + new `parquet/dataset.mojo` + new `parquet/opendal_source.mojo`
  (implements T0.2's `ByteSource`). Parallel to expr/relational work.
- **E-M2.6 ChunkedArray/Table computational surface.** *Owns* `arrays.mojo`+`tabular.mojo`.
  Parallel.
- **E-M2.7 Optimizer: predicate-pushdown-below-joins, constant folding.** *Owns*
  `optimize.mojo`. Parallel (new file).

> M2 conflict hotspot is again `relations.mojo`+`execution.mojo` (E-M2.2/2.3/2.4).
> Serialize those three; run kernels (E-M2.1), scan (E-M2.5), data-model (E-M2.6), and
> optimizer (E-M2.7) in parallel around them.

### M3 — TPC-H (full relational breadth)

- Window relational node + `kernels/window.mojo` family (`lane-shape-window-design.md`).
- Temporal + decimal **arithmetic**; `substring`/`replace`/`regexp_*`.
- Optimizer: CSE, join reordering (needs cardinality estimation + stats).
- Late materialization / selection vectors; async prefetch remote reads.
- **F2 relational monomorphization** (`Table[T]`/`Project[*Es]`/typed joins) once the
  Mojo `reflect` bug clears — the fully-DCE'd relational binary.

Same rule: window-node and typed-relational work serialize on `relations.mojo`/
`execution.mojo`; kernel + optimizer + scan epics parallelize.

---

## 6. ClickBench coverage map (the M1 acceptance target)

The M1 bar is the 43 ClickBench queries (`~/Workspace/ClickBench`, `duckdb/queries.sql`)
run over the `hits` table through **marrow's own frontend** (not SQL) — F1 (Python
`Table`/`Column`) and F2 (Mojo DSL). The `hits` schema is a single wide flat table
(~105 cols: int/smallint, `TEXT`/`VARCHAR`, `DATE`, `TIMESTAMP`) — **no joins, no
nesting**, which is why ClickBench is the right first milestone.

Feature → query → owning task (reading the queries drove several **M2→M1 promotions**,
marked ⬆):

| Feature exercised | Example queries | Task |
|---|---|---|
| Whole-table aggregates `COUNT(*)/SUM/AVG/MIN/MAX` | Q1–4 | existing `Aggregate` |
| `COUNT(DISTINCT)` (whole-table + grouped) | Q5,6,9–14,23 | ⬆ T1.5 + T2.3b |
| **MIN/MAX on DATE and TEXT** | Q7 (date), Q22,23 (string) | ⬆ T1.5 + T2.3b |
| GROUP BY single/multi-key | Q8,12,15,17,31–34 | existing + T2.3b |
| ORDER BY agg / column / string / timestamp / multi-key | Q8,24,25,26,27 | T2.3a |
| `LIMIT` + `OFFSET` | Q9–19,37–43 | T2.3a |
| `HAVING COUNT(*) > n` | Q28,29 | ⬆ T2.3b |
| Filter `<>`,`=`,`>=`/`<=` (incl. **DATE literals**) | Q2,20,37–43 | T2.2 (compares) + `lit` |
| `LIKE` / `NOT LIKE` | Q21,22,23 | T1.3 |
| `IN (list)` | Q41 | T1.2 |
| `CASE WHEN … THEN … ELSE …` | Q40 | T1.1 |
| Computed projection / literal column (`x`, `x+1`, `1`) | Q30,35,36 | T2.3a (Project) |
| **Computed group keys** (arithmetic, extract, date_trunc, CASE) | Q19,35,36,40,43 | ⬆ T2.3b |
| **Computed aggregate inputs** (`SUM(x+1)`, `AVG(STRLEN(URL))`) | Q28,30 | ⬆ T2.3b |
| String length `STRLEN` | Q28,29 | existing (`LENGTH`) + T2.2 |
| Temporal `extract(minute)`, `date_trunc('minute')` | Q19,43 | T1.4 |
| Point lookup (filter, no agg) | Q20 | T2.3a (Project) |
| `REGEXP_REPLACE` | Q29 | **M2** (regex) — *defer; skip Q29 in the M1 subset* |

**M1 ClickBench subset = Q1–Q28, Q30–Q43** (all 43 except **Q29**, which needs regex →
M2). T3.5 signs off when that subset returns DuckDB-matching results on both frontends.

> **Net correction from reading the queries:** grouped/whole-table **min-max over
> string+date**, **count_distinct as a relational agg**, **HAVING**, **computed group
> keys/agg inputs**, and **`date_trunc`** were M2/Should in the first draft; ClickBench
> forces them into M1. That is why T1.5 and T2.3b exist and the `execution.mojo` serial
> chain grew a step. Everything else (joins, windows, decimal/temporal arithmetic, regex)
> stays M2/M3 — ClickBench doesn't need it.

---

## 7. Ready-to-dispatch checklist (per task)

Before spawning an agent for a card, confirm:

- [ ] All **Depends** tasks are merged to the integration branch.
- [ ] No other in-flight task lists the same file under **Owns** (check the wave).
- [ ] The agent gets: this card, the roadmap section, prior-art pointers, and the
      invariants (`binary_size` gate + cross-driver parity).
- [ ] Isolation is `worktree`.
- [ ] Definition of Done = own tests + full suite + `binary_size` all green.
