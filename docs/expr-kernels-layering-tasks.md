# `marrow.expr` / `marrow.kernels` — responsibility + layering audit

Findings and an executable task plan from a full read of both packages' type inventory and
module import graph (2026-07-26).

**Base:** `complete` @ `6e43acc` · **Scope:** `marrow/expr/*`, `marrow/kernels/*` (core
`arrays`/`buffers`/`scalars`/`builders` are out of scope).

**Status — re-verified against the code, 2026-07-27.** The header said "not started"; three
Tier-A/B tasks were in fact complete. Checked by grep, not by trusting a header:

| task | status | evidence |
|---|---|---|
| **L1** — move the aggregate catalog up | **done** | `NumericFold` (`expr/aggregates.mojo:84`), the `Sum`/… aliases (`:184`) and `resolve_agg` (`:194`) all live in `expr`; the `expr.aggregates → expr.dynamic` edge is gone |
| **L4** — `GroupBy` stops speaking `RecordBatch` | **done** | `kernels/groupby.mojo` imports only `..arrays`/`..builders`/`..dtypes` — no `..schema`, no `..tabular`. `marrow/kernels` is free of the tabular layer |
| **L5** — one owner for the grouping strategy | **done** | `GROUP_THREAD_LOCAL` no longer crosses into `marrow/expr`; the mergeability check is internal to `GroupBy` (`groupby.mojo:487`) |
| **L9** — `bitmap_and` duplicate | **done** | no occurrences outside a comment |
| **L2** — extract `DynValue` | **open** | still at `values.mojo:2194`; `values.mojo` still imports `.dynamic`. Blocks L6 |
| **L3** — `AggFunc`'s late binding | **open** | unchanged |
| **L6** — a `Scan` abstraction | **open** | `RELATION_PARQUET_SCAN` still an IR discriminant (`relations.mojo:78`) |
| **L7** — split or delete `Kernel` | **re-scope** | **the premise no longer holds.** `Kernel` now carries `error[M](message)` (`kernels/core.mojo:22-25`) plus `expect_same_length` / `expect_same_dtype` — a genuinely shared member across all 73 conformers, which is exactly what the task asked for. "Prefer the delete" is now the wrong advice; close it, or re-scope to the remaining question of whether one trait should serve both element-wise SIMD ops and whole-array algorithm namespaces |
| **L8** — decompose `TagValue` | **open** | unchanged, and still the root cause of the Q0.0 class of bug |

> **L6/Q1.3's blocker was imaginary.** Both were held up by a "compiler crashes on
> `rel.filter(col > lit)` under `TestSuite`" cap. That cap was caused by tests building plan
> nodes by hand, not by the toolchain — see `docs/code-quality-tasks.md`. Both are unblocked.

Same conventions as `docs/code-quality-tasks.md`: one owner per file, worktree-ready, conventional
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

**L1 — `resolve_agg` and the `AggFunction` catalog are both one layer from where the docs say
they are.** `Sum`, `Product`, `Mean`, `Min`, `Max`, `Count`, `CountDistinct`,
`ApproxCountDistinct` are defined at `marrow/kernels/aggregate.mojo:1096-1203`, but
`kernels/aggregate.mojo`'s own header says they "live in `marrow.expr.aggregates`",
`expr/aggregates.mojo`'s header lists them as its first layer, and
`docs/aggregate-kernel-inversion.md` §8 claims `expr/aggregates.mojo` "holds the aggregations, the
catalog, the erased boxes". Three documents describe the intended layering; none matches the code.
Separately, `resolve_agg` (`expr/dynamic.mojo:914`) lives next to `TagValue`, whose 41 tags contain
**no aggregate tags at all** — so `expr.aggregates` imports the runtime interpreter module
(`expr/aggregates.mojo:53`) purely to reach one name ladder. That is the module the small-binary
gate exists to keep out of fused builds.

**L2 — `DynValue` inside `values.mojo` is why the comptime lane depends on the runtime lane.**
`values.mojo` is three modules in one 2380-line file: the `Value` trait hierarchy + ~45 fused
nodes; the fusion runtime (`Datum`, `Context`, `promote`, `wider`); and `DynValue`, the box that
unions fused nodes with `TagValue`. Only the third needs `from .dynamic import TagValue`
(`values.mojo:158`). `DynValue` belongs *above* both lanes, not inside one of them. Related
asymmetry: `DynValue._prune_tramp[V]` (`values.mojo:2240`) returns `unknown()` unconditionally —
the box advertises a `prune` capability that only one of its two arms implements.

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

**L4 — `GroupBy` is the only kernel that speaks `RecordBatch`.** `kernels/groupby.mojo:29-30` is
the sole `..schema` / `..tabular` import in `marrow/kernels`. `aggregate_columns(values, names)`
takes output *column names* and returns an assembled `RecordBatch`; table assembly and naming are
relational concerns. Every other kernel returns arrays and lets the caller build the batch.

**L5 — grouping strategy is chosen in `kernels`, but one of its three branches is implemented in
`expr`.** `GroupBy` documents that it picks serial / thread-local / radix once at construction. It
can only *execute* two of them: `expr/aggregates.mojo:347` does
`if gb.strategy() == GROUP_THREAD_LOCAL and self.is_mergeable()` and then runs its own
`Aggregates._thread_local`. A kernel-layer constant is imported upward for a caller-side
comparison, and mergeability — a property the kernel layer cannot see — is an unstated
precondition on a strategy the kernel layer has already committed to. This is also the most likely
source of the unresolved regression in `docs/aggregate-kernel-inversion.md` §8: `aggregate[A]`
now falls through to the shared `_by_partition` driver.

**L6 — Parquet is hard-wired into both the IR and the operator layer.**
`expr/execution.mojo:34-42` imports 8 symbols from `..parquet`; `expr/relations.mojo` carries
`RELATION_PARQUET_SCAN` as an IR kind discriminant for pushdown. There is no `Source`/`Scan`
abstraction, so CSV or IPC means another processor, another import into the same module, and
another discriminant in the generic IR. `ParquetScanProcessor` itself does four things: file read,
row-group pruning, page selection, morsel slicing.

**L7 — `Kernel` is a non-abstraction.** It requires only `comptime name: String`, and it is worn
by two unrelated kinds of thing: element-wise SIMD ops (`AddKernel`, `LtKernel`) and whole-array
algorithm namespaces (`Filter`, `Take`, `SortIndices`, `RapidHash`, `NumericCast`). It names
nothing about behaviour, so it cannot be used to constrain anything — every real constraint is
carried by a sub-trait (`BinaryKernel`, `AggKernel`, `StringMapKernel`, …).

**L8 — `TagValue` is a god-node.** 41 tags, 7 fields, four responsibilities (tag dispatch,
statistics pruning, schema/dtype resolution, display). Two fields are explicitly overloaded:
`_name` carries "column name **or** LIKE pattern **or** `date_trunc` unit" (its own docstring
flags the hazard) and `_kind_data` carries "column index or op kind". The known Q0.0 heap
corruption (`size_of` 416 vs ≥417 needed for `ArcPointer[TagValue]`) is a direct consequence of
this width.

**L9 — minor, batched.** `DynRelation` bundles erasure + the whole fluent plan-building API +
`kind()` RTTI. `kernels/distinct.mojo` hosts `count_distinct_grouped` /
`approx_count_distinct_grouped`, so "grouped aggregation" is split across `aggregate.mojo` and
`distinct.mojo`. `bitmap_and` exists twice (`kernels/helpers` alias over `Bitmap.intersect`) and is
the source of the (now removed) `check_lib` false positives.

---

## 4. Tasks

### Tier A — layering (do these first; they unblock the rest)

**L1 — Move the aggregate catalog and name ladder up into `expr` (as documented)** ·
*highest leverage, mechanical* · Depends: — · ⚠️ BINSIZE ·
Owns: `marrow/kernels/aggregate.mojo`, `marrow/expr/aggregates.mojo`, `marrow/expr/dynamic.mojo`,
`marrow/kernels/groupby.mojo`, `marrow/kernels/tests/test_aggregate.mojo`,
`marrow/expr/tests/test_aggregates.mojo` · Done when:

- `AggFunction` conformers (`NumericFold`, `OrderPreserving`, `CountValid`, `DistinctCount`) and
  the `Sum`/`Product`/`Mean`/`Min`/`Max`/`Count`/`CountDistinct`/`ApproxCountDistinct` comptime
  aliases live in `marrow/expr/aggregates.mojo`. The `AggFunction` *trait* may stay in
  `kernels/aggregate.mojo` (it is the contract `GroupBy.apply[F]` is bound on) — the *catalog*
  moves.
- `resolve_agg` moves from `expr/dynamic.mojo` to `expr/aggregates.mojo`.
- The import edge **`expr.aggregates → expr.dynamic` is gone** (verify with the cycle/edge script
  in §1, or `grep -n "^from \." marrow/expr/aggregates.mojo`).
- The three stale docstrings are now true: `kernels/aggregate.mojo`'s header,
  `expr/aggregates.mojo`'s header, and `docs/aggregate-kernel-inversion.md` §8.

> Do not "fix" the docstrings in place instead. The docs describe the layering that was actually
> wanted; the code is what drifted. Moving the catalog is the same amount of work and removes the
> edge as a side effect.

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

**L4 — `GroupBy` stops speaking `RecordBatch`** · Depends: L1 · ⚠️ BINSIZE ·
Owns: `marrow/kernels/groupby.mojo`, `marrow/expr/aggregates.mojo`,
`marrow/kernels/tests/test_groupby.mojo`, `marrow/kernels/tests/bench_groupby.mojo` ·
Done when:

- `aggregate_columns` / `_by_partition` / `aggregate[A]` / `apply[F]` return key columns +
  aggregate columns (e.g. `Tuple[List[DynArray], List[DynArray]]` or a small typed result struct),
  not `RecordBatch`, and take no `names: List[String]`.
- `marrow/kernels/groupby.mojo` no longer imports `..schema` or `..tabular` — making `marrow/kernels`
  free of the tabular layer entirely (verify: `grep -rn "tabular\|schema" marrow/kernels/*.mojo`
  returns nothing outside tests).
- `Aggregates.grouped` / `Aggregates.whole` do the naming and the `RecordBatch` assembly, which is
  where the output-schema knowledge already lives.

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

**L5 — One owner for the grouping strategy** · Depends: L3, L4 · ⚠️ BINSIZE · *perf-sensitive* ·
Owns: `marrow/kernels/groupby.mojo`, `marrow/expr/aggregates.mojo`,
`marrow/kernels/tests/test_groupby.mojo`, `marrow/kernels/tests/bench_groupby.mojo` ·
Done when:

- `GROUP_THREAD_LOCAL` is no longer imported by `marrow/expr` — the caller does not compare
  strategy enums to decide which driver to call.
- Either `GroupBy` executes all three strategies (taking a mergeable-fold callback so the
  thread-local path can live where the strategy is chosen), or `GroupBy` is told up front whether
  the aggregate set is mergeable and never picks a strategy it cannot run. One of the two — not a
  third flag.
- **Measure before merging.** Run `--competition` group-by **twice on an idle machine** and A/B
  against the task's own base commit in the same session. The single-aggregate
  `groupby_sum[1m_g100k]` row is the one under suspicion (see
  `docs/aggregate-kernel-inversion.md` §8) — this task either fixes it or must show it did not
  make it worse.

### Tier C — abstractions that name nothing

**L7 — Split or delete the `Kernel` marker trait** · Depends: — ·
Owns: `marrow/kernels/helpers.mojo` + every `struct X(Kernel)` declaration ·
Done when either: `Kernel` gains a member that all conformers genuinely share (and the two
categories are distinguished by sub-traits), **or** it is deleted and `comptime name: String` is
declared on the sub-traits that actually constrain behaviour (`BinaryKernel`, `AggKernel`,
`StringMapKernel`, `TemporalExtractKernel`, `BoolReduceKernel`, …). Prefer the delete: a trait
that requires only a name cannot be used to constrain anything, and every call site is already
bound on the sub-trait.

> Fold in the duplicate `bitmap_and` while touching this file: `kernels/helpers.bitmap_and` is a
> one-line alias for `Bitmap.intersect` and is the source of the documented `check_lib` false
> positives. Point callers at `Bitmap.intersect` and delete it.

**L8 — Decompose `TagValue`** · *large; schedule deliberately* · Depends: L1 (removes the
aggregate ladder from this file first) · ⚠️ BINSIZE ·
Owns: `marrow/expr/dynamic.mojo`, `marrow/expr/tests/*` · Done when the two overloaded fields are
gone: `_name` no longer triples as column name / LIKE pattern / `date_trunc` unit, and
`_kind_data` no longer doubles as column index / op kind. Prefer a payload variant per tag family
over adding more `Optional` fields — the struct is already 416 bytes and its width is what causes
the Q0.0 `ArcPointer[TagValue]` discriminant overflow.

> Sequencing: this is the *fix* for Q0.0's root cause, not a duplicate of it. Do the narrow Q0.0
> correctness fix first if it is still open; do this when the interpreter is otherwise stable,
> because it touches every tag.

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
> Coordinate with **Q1.3** in `docs/code-quality-tasks.md` (one file handle per scan), which owns
> the same two files.

**L9 — Minor, batchable** · Depends: — · Owns: as listed per item:

- `marrow/expr/relations.mojo` — separate `DynRelation`'s erasure mechanics from the fluent
  plan-building API (`select`/`filter`/`aggregate`/`join`). The box should be a box; the builder
  API can be free functions or a thin wrapper over it.
- `marrow/kernels/distinct.mojo`, `marrow/kernels/aggregate.mojo` — `count_distinct_grouped` /
  `approx_count_distinct_grouped` are grouped-aggregation implementations living below the
  aggregate layer. Either move them behind `DistinctAgg` or state in `distinct.mojo`'s header why
  the sketch algorithms are the exception.

---

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
in `docs/code-quality-tasks.md` §0).
