# `marrow.expr` / `marrow.kernels` — responsibility + layering audit

Findings and an executable task plan from a full read of both packages' type inventory and
module import graph (2026-07-26).

**Base:** `complete` @ `6e43acc` · **Scope:** `marrow/expr/*`, `marrow/kernels/*` (core
`arrays`/`buffers`/`scalars`/`builders` are out of scope).

**Status — re-verified against the code, 2026-07-27.** The header said "not started"; three
Tier-A/B tasks were in fact complete. Checked by grep, not by trusting a header:

| task | status | evidence |
|---|---|---|
| ~~**L1**~~ | done | `NumericFold` (`expr/aggregates.mojo:84`), the `Sum`/… aliases (`:184`) and `resolve_agg` (`:194`) all live in `expr`; the `expr.aggregates → expr.dynamic` edge is gone |
| ~~**L4**~~ | done | `kernels/groupby.mojo` imports only `..arrays`/`..builders`/`..dtypes` — no `..schema`, no `..tabular`. `marrow/kernels` is free of the tabular layer |
| ~~**L5**~~ | done | `GROUP_THREAD_LOCAL` no longer crosses into `marrow/expr`; the mergeability check is internal to `GroupBy` (`groupby.mojo:487`) |
| ~~**L9**~~ | done | no occurrences outside a comment |
| **L2** — split `values.mojo` | **reduced** | the lane dependency is gone (interpreter deleted); only `DynAgg` + `_promote_operands` are imported from `.dynamic`. What remains is file size — 3,153 lines |
| **L3** — `AggFunc`'s late binding | **open** | unchanged |
| **L6** — a `Scan` abstraction | **open** | `RELATION_PARQUET_SCAN` still an IR discriminant (`relations.mojo:78`) |
| ~~**L7**~~ | **closed 2026-07-30** | `Kernel` now carries `error`/`expect_same_length`/`expect_same_dtype` and every named kernel conforms — it is no longer a bare marker |
| ~~**L8**~~ | **done 2026-07-30** | `TagValue` deleted; `dynamic.mojo` 1,087 -> 113 lines |

> **L6/Q1.3's blocker was imaginary.** Both were held up by a "compiler crashes on
> `rel.filter(col > lit)` under `TestSuite`" cap. That cap was caused by tests building plan
> nodes by hand, not by the toolchain — see `docs/tasks-code-quality.md`. Both are unblocked.

Same conventions as `docs/tasks-code-quality.md`: one owner per file, worktree-ready, conventional
commits, `CHANGELOG.md` entry per meaningful change. Tasks marked **⚠️ BINSIZE** must run
`pixi run binary_size` and report the **fused (AOT) `query_streaming` stripped size** before/after,
re-measured on the task's own base commit.

**Guiding standard.** A task is done when the concept has **one owner**, its invariant is enforced
by construction rather than by a docstring, and call sites got *simpler*. A fix that adds a flag,
a parameter, or a second way to do the same thing is the wrong fix.

---

## 1. Dependency graph — no cycles, and it is a DAG (not a tree)

A cycle check over all 54 non-test modules under `marrow/` finds **zero cycles inside
`marrow.expr`, zero inside `marrow.kernels`, and zero between them**. The only cycles anywhere are
three pre-existing core ones — `buffers↔views`, `arrays↔builders`, `arrays↔scalars` — which are
intentional (Mojo resolves intra-package circular imports) and out of scope here.

### `marrow.kernels`

```
L0  execution (ExecutionContext) · helpers (Kernel, bitmap_and) · partition
L1  arithmetic · boolean · compare · temporal · nested · string · concat · filter
L2  cast            → filter
L3  hashing         → cast
L4  hashtable → hashing, compare, filter
    sort      → cast, filter, partition
    conditional → concat, filter, compare
L5  distinct · membership · join   → hashtable, hashing, partition
L6  aggregate       → distinct
L7  groupby         → aggregate, hashtable, filter, concat
```

### `marrow.expr`

```
L0  pruning
L1  dynamic     → pruning
L2  values      → dynamic, pruning
    aggregates  → dynamic
L3  execution   → values, aggregates, pruning, (marrow.parquet)
L4  relations   → execution, values, dynamic, aggregates
```

### A note on the target shape

**A strict *tree* is not achievable and is not the right target.** Both graphs are diamond-shaped
DAGs (`filter` has five distinct dependents; `values` and `aggregates` both sit on `dynamic`),
which is what shared building blocks look like — a tree would mean no shared utilities at all.
The property that actually matters, **layered acyclicity**, holds today.

Two edges are backwards **conceptually** even though they are forwards structurally
(`aggregates → dynamic`, `values → dynamic`); both are addressed by L1/L2 below.

---

## 2. Responsibility inventory

### `marrow.kernels` — array in, array out

| Type | Responsibility | Single? |
|---|---|---|
| `ExecutionContext` | how to dispatch work: thread budget + optional GPU device | ✔ |
| `Kernel` | *just* `comptime name: String` — a naming marker | ⚠ L7 |
| `bitmap_and` | validity AND (delegates to `Bitmap.intersect`) | ✔ |
| `Partition` / `Partitioner` / `NoPartition` / `RadixPartitioner` | split rows into independent subsets by hash prefix | ✔ |
| `SwissHashTable` | keyed row table: find-or-insert → bucket ids; build+probe CSR | ✔ |
| `RapidHash` | row hashing of key columns | ✔ |
| `Filter` / `Take` | selection / gather, typed leaves + dtype dispatch | ✔ |
| `SortIndices` | the permutation that sorts a column | ✔ |
| `*Cast` (14 structs) | one conversion family each | ✔ |
| `BinaryKernel` / `UnaryKernel` / `BinaryCompareKernel` / `Bool*Kernel` + leaves | pure element-wise SIMD algebra | ✔ |
| `StringMapKernel` / `StringPredicateKernel` / `LikePattern` | string transforms, predicates, compiled LIKE pattern | ✔ |
| `TemporalExtractKernel` + leaves | calendar-field extraction | ✔ |
| `AggKernel` | the fold **algebra**: `AccType`, `identity`, `combine`, `finalize` | ✔ |
| `AggState[K,V]` | fully typed per-group accumulator + valid count | ✔ |
| `Aggregation` | one aggregate **bound to one input dtype**: `grouped`/`whole`/`partials`/`merge` | ✔ |
| `AggFunction` | which `Aggregation` a function resolves to, per dtype | ✔ but misplaced (L1) |
| `HashGrouper` | key rows → dense group ids; deliberately aggregate-agnostic | ✔ |
| `GroupBy` | grouping strategy choice + the multi-column driver | ⚠ L4, L5 |
| `Join` / `HashJoin` | build/probe → index pairs | ✔ |
| `ArrayLengthKernel` / `ArrayContainsKernel` | list-element ops | ✔ |

### `marrow.expr` — plans, expressions, execution

| Type | Responsibility | Single? |
|---|---|---|
| `PruneBound` / `PruneStats` | interval arithmetic over column statistics | ✔ |
| `TagValue` | runtime tagged expression node **+** its interpreter | ⚠ L8 |
| `resolve_agg` | aggregate-name → `Aggregation` ladder | ⚠ L1 |
| `Datum` / `Context` | fusion wire format + breaker slot protocol | ✔ |
| `Value` + 5 family traits + ~45 node structs | comptime-typed fused expression nodes | ✔ (each) |
| `DynValue` | dual-lane erased expression handle (fused `Value` ∪ `TagValue`) | ⚠ L2 |
| `AggFunc` | one aggregate erased to a `grouped` function pointer | ⚠ L3 |
| `AggFold` | the partial/merge pair, erased — split out on purpose for binary size | ✔ |
| `ThreadPartials` | one worker's frozen per-group state | ✔ |
| `Aggregates` | the aggregate **set** + the drivers that share one grouping pass | ⚠ L5 |
| `Relation` | pure immutable IR node | ✔ |
| `DynRelation` | erasure box **+** fluent plan-building API **+** `kind()` RTTI | ⚠ L9 |
| `Processor` / `DynProcessor` | pull-based physical operator + drain driver | ✔ |
| 8 `*Processor`s | one per node kind | ✔ except `ParquetScanProcessor` (L6) |
| `Exhausted` | end-of-stream sentinel | ✔ |

---

## 3. Findings

Each finding gets a task in §4 with the matching id.

**L2 — `values.mojo` is three modules in one file.** *Reduced 2026-07-30, not closed.* The
original finding was that `DynValue` unioned fused nodes with a `TagValue` interpreter, so the
comptime lane had to import the runtime one. The interpreter is gone: `DynValue` boxes a single
trait, and `values.mojo` now imports just `DynAgg` and `_promote_operands` from `dynamic.mojo`.
The **file-size** half stands — 3,153 lines holding the `Value` hierarchy, ~45 nodes, the fusion
runtime (`Datum`, `Context`, `promote`, `wider`) and the box.

The prune asymmetry noted here is **fixed**: `_prune_tramp[V]` calls `V.prune(stats)`, so a fused
predicate can skip row groups where the box previously answered `unknown()` for every fused node.
That was Q4.5's core, delivered as a side effect — and it is **untested**, since every case in
`test_pruning.mojo` builds an erased predicate.

**L3 — `AggFunc.name` is documented as "never dispatch" but *is* dispatch.**
`expr/aggregates.mojo:94` says the name is "the default output column name, never dispatch". Two
of three execution paths re-resolve from that string at run time:

- `aggregates.mojo:381` — `resolve_agg[run](self._funcs[j].name, values[j].dtype())` in `whole()`
- `aggregates.mojo:398` — `AggFold(self._funcs[j].name, values[j].dtype())` per aggregate, inside
  `_thread_local`

So `AggFunc` is half an erased handle (`grouped` is a pointer) and half a serialized recipe
(`whole`/`partials`/`merge` are a string + a dtype re-parsed later). The split itself is a
deliberate binary-size trade — folding `AggFold` into `AggFunc` measured **+3.2 MB (+24%)** — and
that trade is fine. What leaks is that the name-as-key half is undocumented, and the ladder in the
thread-local path runs once per aggregate *per grouped call*, not once at plan build as the module
header claims.

**L6 — Parquet is hard-wired into both the IR and the operator layer.**
`expr/execution.mojo:34-42` imports 8 symbols from `..parquet`; `expr/relations.mojo` carries
`RELATION_PARQUET_SCAN` as an IR kind discriminant for pushdown. There is no `Source`/`Scan`
abstraction, so CSV or IPC means another processor, another import into the same module, and
another discriminant in the generic IR. `ParquetScanProcessor` itself does four things: file read,
row-group pruning, page selection, morsel slicing.

**~~L7 — `Kernel` is a non-abstraction.~~** **Resolved 2026-07-30, differently than proposed.**
The finding was that `Kernel` required only `comptime name: String` and so constrained nothing.
It now also provides `error`, `expect_same_length` and `expect_same_dtype`
(`kernels/core.mojo`), and every named kernel conforms — 25 that carried a name without the
conformance were fixed, which removed two drifted copies (`Filter._require_len`,
`SortIndices` attributing errors to `sort:`). The trait is no longer a bare marker, so
splitting or deleting it is moot.

## 4. Tasks

### Tier A — layering (do these first; they unblock the rest)

**L2 — Extract `DynValue` out of `values.mojo`** · *unblocks fused-lane independence* ·
Depends: — · ⚠️ BINSIZE · Owns: `marrow/expr/values.mojo`, new `marrow/expr/erased.mojo`,
`marrow/expr/execution.mojo`, `marrow/expr/relations.mojo`, `marrow/expr/__init__.mojo` ·
Done when:

- `DynValue` lives in its own module that imports *both* `values` and `dynamic`; the graph becomes
  `values → pruning`, `dynamic → pruning`, `erased → values, dynamic`.
- `values.mojo` no longer imports `.dynamic`.
- `execution.mojo` / `relations.mojo` import `DynValue` from the new module; the public surface in
  `expr/__init__.mojo` is unchanged.
- Report the fused stripped size. Expected: unchanged (DCE already removes the interpreter) — the
  win is structural, so a **regression** is the signal to stop and investigate.

> Optional follow-on, same task if it stays small: give the fused arm a real `prune`. Today
> `_prune_tramp[V]` returns `unknown()` for every comptime node, so a fused predicate can never
> prune a row group. If that is intentional (it is sound, just pessimistic), say so on the trait
> rather than only in the trampoline comment.

### Tier B — the aggregate boxes (depends on Tier A)

**L3 — Make `AggFunc`'s late binding either honest or gone** · Depends: L1 · ⚠️ BINSIZE ·
Owns: `marrow/expr/aggregates.mojo`, `marrow/expr/tests/test_aggregates.mojo` ·
Done when **one** of the following is true, whichever the measurement supports:

- **(a)** `AggFunc` carries `whole` and the `AggFold` pair as pointers resolved at construction, so
  no `resolve_agg` call survives on any execution path — and the fused stripped size is reported
  against the known **+3.2 MB / +24%** cost of the previous attempt; or
- **(b)** the string stays as the late-binding key, `AggFunc.name`'s docstring says so plainly
  (replacing "never dispatch"), the module header stops claiming resolution happens only at plan
  build, and `_thread_local` hoists the per-aggregate `AggFold(name, dtype)` construction **out of
  the per-grouped-call path** so the ladder runs once per plan rather than once per batch.

> (b) is the likely answer given the measured size cost — but the hoist is not optional in that
> case, and neither is the docstring. The current state is the worst of both: it pays the ladder
> repeatedly *and* documents that it doesn't.

### Tier C — abstractions that name nothing

### Tier D — extensibility (not urgent, but blocks new sources)

**L6 — A `Scan` abstraction above the file formats** · *do before adding CSV/IPC, not after* ·
Depends: L2 · Owns: `marrow/expr/execution.mojo`, `marrow/expr/relations.mojo` ·
Done when a scan source is a trait (open → schema; pull → morsel; optional statistics for pruning)
and `ParquetScanProcessor` is one implementation of it. `expr/execution.mojo`'s 8-symbol
`..parquet` import collapses to the parquet source module, and `RELATION_PARQUET_SCAN` becomes a
generic "scan with pushdown support" discriminant rather than a format name in the IR.

> Also split `ParquetScanProcessor`'s four jobs (file read / row-group pruning / page selection /
> morsel slicing) at the same time — pruning and morsel slicing are format-independent and belong
> to the scan wrapper, not the parquet implementation.
>
> Coordinate with **Q1.3** in `docs/tasks-code-quality.md` (one file handle per scan), which owns
> the same two files.

## 5. Suggested order

```
L1 ──┬──> L3 ──┐
     │         ├──> L5   (measure twice, idle machine)
L4 ──┴─────────┘
L2 ──────────────> L6
L7   (independent, small)
L8   (independent, large — schedule alone)
L9   (independent, cheap)
```

`L1` first: it is mechanical, deletes an import edge, and makes three documents true.
`L2` is independent of it and equally mechanical. `L5` is the only perf-sensitive task in the set
and must not be run in parallel with anything else on this machine (see the orchestration lessons
in `docs/tasks-code-quality.md` §0).
