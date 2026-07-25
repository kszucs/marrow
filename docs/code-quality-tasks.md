# Code quality — actionable task plan

Companion to **`docs/code-quality-review.md`** (findings + evidence). This file is the
executable half: discrete, worktree-ready tasks with explicit file ownership so they can be
run in parallel without merge conflicts, following the same conventions as
`docs/execution-engine-tasks.md`.

**Base:** `complete` @ `867d9d6`+ · **Status:** not started (Q0.0 closed — fixed upstream).

**Guiding standard.** These tasks are not chores; the bar is *elegant, performant, encapsulated
abstractions*. A task is done when the concept has **one owner**, its invariants are enforced by
construction rather than convention, and the call sites got **simpler**. If a fix adds a
parameter, a flag, or a second way to do something, it is the wrong fix — prefer the change that
deletes code. Measure anything on a hot path; never trade a real speedup for tidiness without
saying so.

**Hard constraint: do not change the layout of arrays, scalars, or builders.** Their fields and
memory layout are fixed. Adding accessors/methods is fine; adding, removing, reordering, or
re-typing fields is not. Any task that would require it is out of scope, not deferred.

---

## 0. How to run these

Same protocol as the execution-engine waves:

```bash
git worktree add -b <branch> .claude/worktrees/<name> complete
```

Work pinned to the worktree; touch **only** the files under *Owns*; commit on the branch with a
conventional-commit message + `CHANGELOG.md` entry.

**Verification** (per task, before commit):
```bash
pixi run -e dev pytest <the owned test files>
pixi run -e dev fmt
```
Tasks marked **⚠️ BINSIZE** must additionally run and report:
```bash
pixi run binary_size
```
**The gate is the fused (AOT) binary — `query_streaming` stripped size.** That is the
number the small-binary/DCE property is about. `query_dynvalue` / `query_runtime` size is
**explicitly not a goal for now**; a change that shrinks the fused binary at the runtime
binary's expense is a *win*, not a trade-off. (It may become a goal in the distant future.)

Report the fused stripped size before/after. Re-measure the baseline on your own base commit —
the pre-upgrade figures were ~12% stale after a compiler bump.

> The pre-upgrade figures (1,538,824 / 11.3×) are **stale** — the compiler upgrade alone shrank
> both the fused and runtime binaries by ~10%. Always re-measure the baseline on your own base
> commit rather than trusting a number written down earlier.
```
```

### In-flight conflicts

None — `t2.3b-aggregate` and `fu4-like-scalar` are both merged. Re-check before dispatching.

### Orchestration lessons (learned the hard way, 2026-07-25)

- **Run ONE lane at a time on this machine.** Three parallel Mojo worktrees all died to
  watchdog timeouts waiting on compiles, made worse by concurrently running `binary_size` and the
  full suite. Two produced good work that then needed manual validation; one produced nothing.
- **Drive the gates yourself.** Mojo compile times make agent-run test suites a poor fit — an
  agent that stalls mid-verification leaves work that looks finished but is untested. Both stalled
  lanes had *uncommitted, unvalidated* changes; one of them was wrong.
- **Re-measure `binary_size` on your own base commit.** The pre-upgrade figures (1,538,824 /
  11.3×) were stale by ~12% after the compiler bump; gating against a written-down number would
  have hidden a regression or invented one.
- **`pytest` overstates breakage.** A whole-file runner that exits non-zero marks *every* test in
  the file failed — one run reported 32 failures where the binary reported 2.

### Contended files (single owner per wave)

`marrow/arrays.mojo`, `marrow/views.mojo`, `marrow/buffers.mojo`, `marrow/expr/values.mojo`,
`marrow/kernels/filter.mojo`, `marrow/dtypes.mojo`, `marrow/utils.mojo`.

---

## Tier 0 — correctness + trustworthy signal

> Q0.0 is closed (upstream fix), so the suite is trustworthy again and these are no longer
> gated — Q0.2/Q0.3 can run in parallel with everything else.

**Q0.0 — ~~Fix the one-byte heap overflow in `AnyValue`~~** · **CLOSED — fixed upstream
2026-07-25** · No work required.

Was: `ArcPointer[DynValue]` (`expr/values.mojo:2299`) wrote its trailing `Variant` discriminant
one byte past the allocation (`size_of` 416 vs ≥417 needed), silently corrupting the heap and
causing **every** full-suite failure.

**Resolved by upgrading Mojo `1.0.0b3.dev2026072217` → `1.0.0b3.dev2026072406`.** It was a
toolchain bug in the #6401 family (`Variant.__init__` performing an invalid write), never a
marrow logic error. Verified both ways:

| check | before | after |
|---|---|---|
| `expr/tests/test_streaming.mojo` (no ASAN) | 24 failed | **43 passed** |
| `parquet/tests/test_reader.mojo` (no ASAN) | 35 failed | **35 passed** |
| `test_streaming` ASAN `heap-buffer-overflow` hits | 86 | **0** |

Lessons worth keeping (they cost real time):
- **ASAN masked this bug** — `test_reader` passed 35/35 *under* ASAN while failing 35/35 without.
  Verify toolchain-level memory bugs **without** ASAN; ASAN support is itself an open upstream
  feature request (#4575).
- **A Mojo build failure emits no ASAN output**, which is indistinguishable from "no bug" if you
  only grep for `heap-buffer-overflow`. Always assert the test actually ran.
- **Minimal reproducers were useless here** — the fault depended on compilation context, so
  shrinking it produced contradictory results. Only the real suite was authoritative.

**Q0.2 — Fused expression correctness (D3 + D4)** · *Tier 0* · Depends: — ·
Owns: `marrow/expr/values.mojo`, `marrow/kernels/boolean.mojo`, `marrow/expr/tests/test_values.mojo`,
`marrow/expr/tests/test_parity.mojo` · ⚠️ BINSIZE · Done when:
- `NumericCompare` uses `promote[L, R]` and casts **both** operands, matching `NumericBinary`
  (`values.mojo:583`). Add a mixed-width parity case (`int32 > int64`) — none exists today.
- `Any`/`All` bind to the null-correct kernels. Delete the duplicate pair in `boolean.mojo:288-303`
  and re-point `values.mojo:138-153` at `..kernels.aggregate` (the pair `kernels/__init__.mojo`
  already re-exports). Add a parity case with nulls whose data bits are set.
- Both verified via `test_parity.mojo` (fused == dynamic).

**Q0.3 — `DynValue.name()` tag guard** · *Tier 0, trivial* · Depends: — ·
Owns: `marrow/expr/dynamic.mojo` · Done when: `name()` returns `String()` unless `_tag == LOAD`,
so a `LIKE` node stops reporting its pattern (`"%foo%"`) and a `DATE_TRUNC` node its unit as an
output column name. Add a test.

---

### Accepted known defects (not scheduled)

**D1** — `slice()` copies the parent's `nulls`, so a slice of an array with nulls reports the
parent's null count even when every element in the slice is valid (probe-confirmed). Read by
kernels and by three builder fast paths.

**D2** — indexing a sliced `BoolArray` applies the offset twice (`values()` already returns an
offset-applied view, then `__getitem__` adds it again), aborting on the bounds assert in debug and
reading the wrong bit in release (probe-confirmed).

Both are real. Every fix identified requires changing array internals, which the hard constraint
above puts out of scope — so they are **accepted, not deferred**. Recorded here so nobody
rediscovers them. If the constraint is ever relaxed, they are one task: a `Validity` value type
owning bitmap+offset+length makes both unrepresentable.

---

## Tier 1 — unblocks M1 and T2.4

**Q1.1 — Close the dtype dispatch ladders (D5 + RC4)** · *M1 blocker* · Depends: — ·
Owns: `marrow/utils.mojo`, `marrow/dtypes.mojo`, `marrow/kernels/hashing.mojo`,
`marrow/kernels/sort.mojo` + their tests · Done when:
- `dispatch_over_integer`, `dispatch_over_primitive`, `dispatch_over_temporal`,
  `dispatch_over_listlike` exist alongside the current four.
- `rapidhash` and `sort_indices` route through them and accept **temporal, large_string, decimal,
  dictionary**. This unblocks `GROUP BY date_trunc(...)` (Q19/35/36/40/43) and `ORDER BY` a
  timestamp (Q8/24–27) — both on the M1 ClickBench list, both currently raising.
- Tests covering group-by and sort on a date/timestamp/large_string column.
- Supersedes **FU-1** (which tracked only the `large_string` half).

> **Do the struct conversion in the same pass, not "if convenient".** `hashing.mojo` (17 free
> functions) and `sort.mojo` (8) are the only two kernel modules with *no* `Kernel` struct, and
> they are exactly the two with dtype-coverage gaps. That is not a coincidence: with no struct
> there is no single `dispatch` to extend, so each new type had to be remembered in a hand-written
> ladder — and wasn't. Converting them to `RapidHash` / `SortIndices` and routing through
> `dispatch_over_*` makes the gap structurally impossible rather than currently-fixed. Adding a
> dtype should then be a one-line change in one place; if it isn't, the abstraction is wrong.

**Q1.4 — Finish the import hygiene** · *small, high leverage* · Depends: — ·
Owns: `marrow/arrays.mojo`, `marrow/scalars.mojo`, `marrow/builders.mojo` · Done when the
remaining `from .dtypes import *` wildcards are replaced by explicit import lists.

> **Context — the cycle is already broken.** Removing `DataType.ScalarType`/`ArrayType` (they had
> **zero consumers** tree-wide) let `dtypes.mojo` drop its imports of `arrays` and `scalars`, so
> the dependency graph is now a DAG: `dtypes → utils`, with `arrays`/`scalars`/`builders` all
> importing `dtypes` one-directionally. That alone removed a whole class of failure — three
> separate incidents in a single day (a trait shadowing the builtin `Scalar`, `BoolArray` failing
> to resolve along one import path but not another, and a "fix" that made errors go 2 → 10)
> all had the same signature: **the same source compiling or failing depending on which file you
> entered through**.
>
> The remaining wildcards are no longer a cycle, but they are still opaque — you cannot tell what
> a module depends on by reading its head, and a name added to `dtypes` silently lands in three
> other namespaces. Replacing them with explicit lists is mechanical and makes the next collision
> impossible rather than merely unlikely.
>
> Note this contradicts CLAUDE.md's standing advice ("Mojo resolves circular imports — do not
> reorganize code to avoid them"). That was true and is now outdated; update it as part of this
> task.

**Q1.2 — `ByteSource.read_at` → `Buffer[mut=False]` (RC5)** · *T2.4 prerequisite* ·
Depends: — · Owns: `marrow/parquet/source.mojo`, `marrow/parquet/reader.mojo`,
`marrow/parquet/tests/test_reader.mojo` · Done when: `read_at` returns a ref-counted
`Buffer[mut=False]`; `MappedFile` wraps the mmap via `Buffer.from_foreign` (still zero-copy);
`Page.body` / `PageReader.data` become `BufferView`; the `_untracked()` `rebind` helper
(`reader.mojo:157`) is **deleted**.

> **Do this before any other parquet streaming work.** The current contract —
> "a borrowed non-owning span with `ImmUntrackedOrigin`" — is satisfiable *only* by a whole-file
> mmap. A streaming source must own recycled buffers, and the untracked origin removes the
> compiler's ability to catch the dangle. Doing `_span()` removal first would just move the
> dangling problem into every page decode.

**Q1.3 — One file handle per scan (RC8)** · *T2.4 prerequisite* ·
Depends: Q1.2, **T2.3b merged** · Owns: `marrow/parquet/reader.mojo`, `marrow/expr/execution.mojo` ·
Done when: `ParquetScanProcessor` opens the file **once**. Today `_read_plan`
(`execution.mojo:290-292`) calls `read_metadata`, `read_statistics`, `read_page_bounds` and then
`read_table` (`:317`) — each constructing its own `ParquetFile`, i.e. **four mmaps and four footer
parses per logical scan**. Delete the three one-line wrappers (`reader.mojo:2085,2111,2147`, which
duplicate `ParquetFile` methods and have no PyArrow equivalent) and thread one `ParquetFile`
through. Then remove `_span()` and make `PageReader` chunk-relative.

---

## Tier 2 — root causes (removes classes of future bugs)

**Q2.5 — Aggregates: kernels as the only representation** · *large, do in a worktree* ·
Depends: — · Owns: `marrow/kernels/aggregate.mojo`, `marrow/kernels/groupby.mojo`,
`marrow/expr/relations.mojo`, `marrow/expr/execution.mojo`, `python/bindings/tabular.mojo`
(+ their tests) · ⚠️ BINSIZE

An aggregate is currently represented **four** ways: a comptime `AggKernel` struct, a runtime
`AGG_*` `UInt8` tag, a `String` name, and — for output typing — a separate `agg_out_dtype(tag, dt)`
rule that duplicates the kernel's own `AccType` algebra. 51 references outside `aggregate.mojo`,
40 of them in `groupby.mojo`.

**Scope (owner directive, 2026-07-25).** Three things, together:

1. **Runtime aggregate dispatch leaves the kernel layer entirely.** The `AGG_*` tags,
   `agg_tag_from_name`, `for_agg_tag` and `agg_out_dtype` do not belong in
   `marrow/kernels/aggregate.mojo` — they are a *runtime plan* concern, so they belong with the
   expression layer (`marrow/expr/dynamic.mojo` or alongside it). `AggKernel`'s own docstring
   already states the rule and the file breaks it: *"a kernel is a pure type, so any runtime
   `name -> kernel` selection lives in the expression layer, **never here**."* After this, the
   kernel layer holds only comptime types; nothing in it knows about names, tags, or plans.
2. **`reinterpret_array` removed outright** — including `groupby`'s remaining uses. See Q2.6 for
   why it is unnecessary everywhere else; `groupby` is the last holdout and falls out of the
   `AggState` rework below.
3. **The free-standing functions become kernels.** `count_distinct` / `approx_count_distinct`
   (and their `_grouped` variants) get real kernel structs, so `dispatch` covers the whole
   aggregate surface rather than special-casing them.

**Target: the kernel is the only representation.** Exactly one runtime→comptime boundary,
keyed on the name directly with no tag in between:

```mojo
trait AggKernel(Kernel):
    comptime name: String                            # already from Kernel — the inverse of lookup
    comptime is_distinct: Bool = False               # replaces agg_is_distinct(tag)
    comptime AccType[V: PrimitiveType]: PrimitiveType
    @staticmethod
    def acc_dtype(input: AnyDataType) raises -> AnyDataType   # replaces agg_out_dtype(tag, dt)

def dispatch_agg[
    job: def[K: AggKernel]() raises capturing[_] -> None
](name: String) raises:
    if name == SumKernel.name: job[SumKernel]()
    elif name == MinKernel.name: job[MinKernel]()
    ...
```

Everything downstream is typed. Deleted: the eight `AGG_*` constants, `agg_tag_from_name`,
`for_agg_tag`, `agg_is_distinct`, `agg_out_dtype`, `_agg_name`.

**Wrinkle — the distinct aggregates.** `count_distinct` / `approx_count_distinct` currently have
tags but *no kernel struct*, because they carry a hash set / HLL sketch rather than a scalar
accumulator and so cannot use `AggState`. For `dispatch_agg` to cover the whole surface they need
kernel structs too, with `is_distinct = True` selecting the alternate execution path. Note
`groupby`'s `avoid_thread_local` guard currently re-lists "distinct, or string min/max" by hand;
both should become comptime properties so the guard is *derived*, not maintained in parallel.

**Prerequisite — widen `AggState` to `PrimitiveType`** so temporal min/max/count work natively:

```mojo
comptime AccType[V: PrimitiveType]: PrimitiveType     # was NumericType in both positions
struct AggState[K: AggKernel, V: PrimitiveType]
```

This does **not** compile on its own. Widening the *return* bound drops `Defaultable` (since
`trait NumericType(Defaultable, PrimitiveType)`), and `_reduce_widened`/`_reduce_widened_typed`
build accumulators with the single-argument `PrimitiveScalar[Acc](value)` constructor, which needs
it. That constraint is **correct**: temporal types carry a unit and timezone, so a temporal scalar
cannot be conjured without its dtype instance. The fix is to thread it:

```mojo
@staticmethod
def acc_dtype(input: AnyDataType) raises -> AnyDataType   # on AggKernel
```
— `Min`/`Max` → `input`; `Sum`/`Product` → `Int64Type()`/`Float64Type()` by physical width;
`Count` → `Int64Type()`; `Mean` → `Float64Type()`. Then use the two-argument
`PrimitiveScalar[Acc](value, dtype)`. `agg_out_dtype` is exactly this rule keyed by tag instead of
by type, so the two unify here rather than one being deleted.

**Payoff:** removes the last blocker to deleting `reinterpret_array` (see below), gives native
temporal reductions instead of reinterpret-and-relabel, and leaves one place to add an aggregate.

### ⚠️ Do NOT widen `AggState`'s logical bound — make the fold physical instead

Asking *what does `AggState` actually use* changes the prerequisite completely.

`AggState[K, V]` holds `acc: PrimitiveBuilder[K.AccType[V]]` + `cnt: Int64Builder`, but every
operation immediately projects to the physical type:

| method | works on |
|---|---|
| `update` | `comptime A = Self.Acc.native` — SIMD `K.combine` over physical values |
| `merge` | raw accumulators + counts — physical |
| `into_partials` | typed arrays purely as a carrier |
| `finish` | physical fold, then builds `PrimitiveArray[Self.Acc]` |

**The logical dtype is used in exactly one place: constructing the output array in `finish()`.**

So the `NumericType` bound is not load-bearing for the fold — it is a consequence of holding
*logical* builders eagerly. That is what produces the whole `Defaultable` wall
(`PrimitiveBuilder[Acc]()`, `PrimitiveScalar[Acc](value)`), which the reverted attempt tried to
work around by threading a dtype instance through `acc_instance` and eight construction sites,
and which cascaded into the `Value` tower via `Reduction`'s `NumericValue` conformance.

### Decided: split `aggregate_runtime`; names belong to `DynValue`

`GroupBy.aggregate_runtime(values, tags)` is the only reason the kernel layer knows about names
or tags, and it knows about them because it bundles **two** responsibilities:

1. **group once** — hash the keys a single time, and
2. **apply N heterogeneous aggregates** to that grouping.

Only (2) needs runtime dispatch (the N aggregates are different kernels chosen at runtime, so no
single comptime `K` covers them). (1) is why they were fused — to avoid re-hashing per aggregate.
**Splitting them removes the need for names in the kernel layer entirely:**

```mojo
# kernel layer — pure comptime, no names, no tags
def group(keys) -> Grouping                        # gids + num_groups, hashed once
def aggregate[K: AggKernel](g: Grouping, value)    # one typed aggregate
```

The caller groups once, then loops its aggregate list resolving **name → kernel in its own
layer**, calling the typed method per column. Hash-once is preserved because `Grouping` is passed
in — that property came from the bundling, not from the dispatch.

**`DynValue` owns the name/tag routing** (`marrow/expr/dynamic.mojo`). It is the runtime-plan
frontend, so this is where a `String` becomes a kernel; `AggKernel`'s docstring already says
runtime `name -> kernel` selection "lives in the expression layer, never here". The AOT frontend
names kernels as types and pays no dispatch at all, which is what the small-binary property wants.

Deleted from the kernel layer: the eight `AGG_*` constants, `agg_tag_from_name`, `for_agg_tag`,
`agg_is_distinct`, `agg_out_dtype`, `_agg_name`, **and `aggregate_runtime` itself**.
`python/bindings/tabular.mojo` calls `aggregate_runtime` directly today and must be re-pointed
through the expression layer.

**Prerequisite: the `Grouping` value type (RC7 / Q4.1).** `(gids, num_groups)` currently travels
as two loose parameters across 8+ signatures with nothing checking `num_groups > max(gids)`. It
was logged as an independent cleanup; it is actually the enabling piece here, so do it first.

#### Prior art — checked 2026-07-25 (DataFusion, DuckDB, Polars, ClickHouse)

| engine | grouping representation | grouping vs aggregation |
|---|---|---|
| **DataFusion** | no value type — `GroupValues` is a *mutable accumulator*: `intern(cols, &mut groups)` / `len()` / `emit(EmitTo)` | separate, but `emit` is **repeatable and partial** (`EmitTo::First(n)` emits n groups and shifts the rest down) so aggregation can spill/emit incrementally |
| **DuckDB** | no value type — `GroupedAggregateHashTable` + `RadixPartitionedHashTable` | **fused**: `AddChunk(groups, payload)` takes keys *and* aggregate payload together |
| **ClickHouse** | no group ids at all — `HashMap<Key, AggregateDataPtr>` maps a key straight to its aggregate-state blob | **fused**: `executeOnBlock(block, AggregatedDataVariants&, key_columns, aggregate_columns)`; heavy per-key-shape specialisation (`UInt8Key`, `StringKey`, `Keys128`, two-level…) |
| **Polars** | **has** one — `GroupsType::{ Idx, Slice { overlapping, monotonic } }` | fully separate; grouping is a first-class value |

**Conclusions for marrow:**

1. **Do not fold key emission into a one-shot `finish()`.** DataFusion's `EmitTo` and DuckDB's
   scan state both exist so grouped aggregation can emit *incrementally* under memory pressure.
   A one-shot finish forecloses that. Keep emission a separate, repeatable method on the grouper
   — moving toward `emit(EmitTo)` if anything.
2. **Keep the strategy hidden.** None of the four leaks partitioning to consumers; DuckDB has a
   whole `RadixPartitionedHashTable` and still presents one logical table. Marrow's `GroupBy`
   already does this — preserve it.
3. **A `Grouping` value type is the minority position, and Polars' is not our shape.** Polars
   stores *row-indices-per-group* (gather-oriented, with `overlapping`/`monotonic` flags for
   rolling windows); marrow stores *group-id-per-row* (scatter-oriented), like DataFusion — which
   deliberately does **not** wrap it. So the RC7 complaint (`(gids, num_groups)` as two unchecked
   parameters) should be fixed narrowly — have the grouper own the relationship and pass *it*,
   rather than inventing a completed-result type.
4. **Splitting grouping from aggregation has a real cost.** DuckDB and ClickHouse deliberately
   *fuse* them — ClickHouse skips group ids entirely, going key → state pointer. Our split (group
   once, then typed `aggregate[K]` per column) buys the name-free kernel layer, but costs an extra
   pass over the group ids. That is a defensible trade for the frontend split; it is **not** a
   free win, and should be stated as such rather than assumed.

The name/tag routing moving to `DynValue` is **independent of all this** and still stands.

#### Performance: where marrow can actually beat them

The survey shows all four engines paying **per-aggregate dynamic dispatch** in the inner loop:
ClickHouse calls virtual `IAggregateFunction::add` per aggregate per row, DuckDB goes through
function pointers, DataFusion through `dyn GroupsAccumulator`. None can inline the fold into the
scatter loop, because none knows the aggregate set until runtime.

**Marrow's AOT frontend does.** That is the differentiator worth building for: a grouped
aggregation whose entire multi-aggregate update is monomorphised and inlined, with zero dispatch.

**The fusion.** Today N aggregates mean N independent `AggState`s, so N passes over the group-id
array and N scatter streams. With the set known at comptime, one pass suffices:

```mojo
struct FusedAggState[*Ks: AggKernel, *Vs: PrimitiveType]:
    """One accumulator set per group, updated for every aggregate in a single
    pass over the group ids. Each `K.combine` inlines into the loop body."""
    def update(mut self, g: Grouping, inputs: Tuple[...]) raises
```

Feasible in Mojo today: trait-bounded variadic parameters already work here (`*Ts: Movable`,
`marrow/utils.mojo:200`), and `comptime for` unrolls the per-aggregate body.

**Layout is the second lever, and it should be comptime-selectable:**

- *Column-per-aggregate* (today): each accumulator is its own contiguous column. SIMD-friendly per
  aggregate; N random-access streams per row.
- *Struct-of-accumulators-per-group* (ClickHouse's blob): all of a group's accumulators adjacent,
  so one row touches **one** cache line for all N aggregates. ClickHouse pays runtime offsets and
  virtual calls to get this; marrow can generate the struct at comptime with fixed offsets and
  inlined combines — **strictly better than the thing being imitated**.

Choose by cardinality: low-cardinality accumulators fit in cache, so columnar SIMD wins;
high-cardinality is dominated by random access, so the blob wins. Both variants can be generated
from the same kernel set.

**The strongest form** drops group ids entirely on the AOT path: hash → slot → update the blob in
place, never materialising a gid array (what ClickHouse does, minus the indirection). That removes
the extra pass the grouping/aggregation split costs — see conclusion 4 above.

**This does not conflict with moving name routing to `DynValue` — it is the reason to.** The two
frontends want different execution:

| frontend | aggregate set | execution |
|---|---|---|
| **F1 `DynValue`** | runtime | resolve name → kernel, loop calling typed `aggregate[K]` per column |
| **F2 AOT** | comptime | one `FusedAggState[*Ks]`, single pass, zero dispatch |

A kernel layer that speaks only in comptime kernels serves both: F1 loops over it, F2 fuses across
it. A kernel layer that speaks in runtime tags can serve *only* F1's shape — which is exactly why
the tags have to go before the fusion can be built.

#### Fused multi-aggregate execution — detailed design

##### The shape of the problem

A grouped aggregate is a scatter-fold: for each input row, find its group slot and fold the value
into that group's accumulator. With N aggregates there are three costs, and today marrow pays all
three N times:

1. **Group-id traffic** — the gid array is re-read once per aggregate.
2. **Random access** — each aggregate scatters into its own accumulator column, so a single input
   row touches N unrelated cache lines.
3. **Dispatch** — resolving *which* fold to apply. Marrow pays this once per aggregate (not per
   row) because `AggState[K, V]` is monomorphised, which is already better than the surveyed
   engines; but the fold still cannot be inlined *into* a shared loop, because each aggregate owns
   its own loop.

Fusion collapses all three: one pass over the gids, one slot address per row, and every fold
inlined into that row's body.

##### Accumulator layout

Two viable layouts, and the choice is the main performance lever:

**SoA — column per aggregate** (today). `acc_j: Buffer[A_j]` indexed by gid.
*For*: each column is contiguous and same-typed, so a single aggregate vectorises cleanly.
*Against*: N independent random-access streams per row; N TLB/cache working sets.

**AoS — accumulator blob per group** (ClickHouse's `AggregateDataPtr`). All of a group's
accumulators laid out adjacently, so one row computes **one** address and touches **one** cache
line for all N aggregates.
*For*: random access is paid once per row rather than N times — decisive at high cardinality,
where the accumulators exceed cache and misses dominate.
*Against*: the per-aggregate inner op is scalar (mixed types in the blob), so no cross-row SIMD.

**Recommendation: build AoS first.** High-cardinality group-by is where real workloads hurt and
where ClickHouse's design has been validated; at low cardinality the whole accumulator set fits in
cache and the layouts converge. Add SoA later *only if* benchmarks justify it — generating both
doubles the monomorphised code and must be weighed against the AOT binary-size gate.

##### Why marrow's version is better than the thing it imitates

ClickHouse gets AoS locality but pays for it: the blob's field offsets are computed at runtime,
and each aggregate is invoked through `virtual IAggregateFunction::add`. So it wins on memory and
loses on dispatch.

Marrow can have both. With the aggregate set known at comptime the offsets are constants and the
folds inline:

```mojo
comptime OFFSETS = _prefix_sums[size_of[K.AccType[V].native]() for each (K, V)]
comptime STRIDE  = OFFSETS[N] rounded up to the accumulator alignment
```

Per row: one multiply-add for the slot, then N inlined `combine`s at constant offsets. No virtual
calls, no runtime offset table, no function pointers — none of the surveyed engines can do this,
because none knows the set before the query runs.

##### Type architecture

```mojo
# ── kernel layer: comptime only, no names, no tags ────────────────────────────

# single typed aggregate — the surface F1 loops over
def aggregate[K: AggKernel, V: PrimitiveType](
    gids: Int32Array, num_groups: Int, values: PrimitiveArray[V],
    ctx: ExecutionContext,
) raises -> AnyArray

# fused multi-aggregate — the surface F2 generates
struct FusedAggregation[*Ks: AggKernel, *Vs: PrimitiveType]:
    """One accumulator blob per group; every aggregate folded in a single pass.

    `Ks[j]` folds `Vs[j]`; both packs are comptime, so the slot arithmetic is
    constant-folded and each `combine` inlines into the loop body."""
    var _blobs: Buffer[mut=True]        # num_groups * STRIDE bytes
    var _counts: Buffer[mut=True]       # valid-count per group, drives NULL output

    def update(mut self, gids: Int32Array, inputs: Tuple[...], num_groups: Int) raises
    def merge(mut self, other: Self, remap: Int32Array) raises   # partitioned/parallel fold
    def finish(mut self, num_groups: Int) raises -> List[AnyArray]
```

**Mojo feasibility, checked:**
- Trait-bounded variadic parameters already work here (`*Ts: Movable`, `marrow/utils.mojo:200`).
- A struct cannot have a *variadic number of fields*, which is why the blob is a flat `Buffer`
  with comptime offsets rather than a generated struct — the same conclusion ClickHouse reached,
  minus the runtime offsets.
- `comptime for j in range(N)` unrolls the per-aggregate body; `Ks[j]` / `Vs[j]` index the packs.
- **Spike this before committing to the design.** Parameter-pack indexing inside a `comptime for`
  in a struct method is the one construct not already exercised in this codebase.

##### Vectorisation

Naive SIMD scatter is unsound here: several lanes may target the same group, and the folds would
race. Three options, in increasing order of payoff:

1. **Scalar row loop, unrolled aggregates** (start here). One slot computation, N inlined folds.
   This alone captures most of the win — it is what ClickHouse does, minus the virtual calls.
2. **Run-aware**: after the radix partition, rows for one group are often contiguous; detect runs
   and fold a whole run with a horizontal SIMD reduce before writing the slot once.
3. **Conflict-detected SIMD**: gather W slots, detect duplicate gids in the lane block, fold the
   conflict-free lanes vectorised and the rest scalar.

(2) composes naturally with the existing radix path and is the better second step; (3) is a
micro-optimisation to defer until profiles justify it.

##### How the two frontends share this

| frontend | aggregate set | path |
|---|---|---|
| **F1 `DynValue`** | runtime | resolve name → kernel in the expression layer, then loop `aggregate[K]` per column (SoA) |
| **F2 AOT** | comptime | one `FusedAggregation[*Ks, *Vs]`, single pass, zero dispatch |

Both consume the *same* kernel algebra (`AggKernel`), so a new aggregate is written once and both
paths gain it. This is why the tags must leave the kernel layer first: a layer that speaks in
runtime tags can only express F1's shape, and the fused path becomes unbuildable.

##### Group ids, revisited

The fused AOT path can skip materialising gids entirely — hash → slot → fold the blob in place —
which removes the extra pass that splitting grouping from aggregation otherwise costs. That makes
the split free on the path that matters and keeps it explicit on the path that needs it (F1).
It also means **`Grouping` must not become mandatory**: the fused path never wants one.

##### Pluggable grouping strategies

Grouping already has three implementations chosen at construction — serial, thread-local partial
aggregation, and radix partitioning — selected by `GroupBy._choose_strategy(keys, num_threads)`
from row count plus a strided-sample cardinality estimate. The mechanism exists; what is missing
is a **seam**, so a fourth strategy can be added for a skewed distribution without editing the
aggregate path, and so tuning one case cannot silently cost another.

**Separate the three things that are currently entangled:**

```mojo
struct GroupStats:
    """Everything the policy is allowed to look at. Measured once, cheaply."""
    var num_rows: Int
    var est_groups: Int          # existing strided sample
    var max_group_share: Float64 # NEW: largest sampled group as a fraction of the sample
    var num_key_cols: Int
    var num_threads: Int
    var has_unmergeable_agg: Bool  # distinct / string min-max cannot use the merge path

trait GroupStrategy:
    comptime name: String
    @staticmethod
    def suitable(stats: GroupStats) -> Bool   # correctness/capability filter
    @staticmethod
    def rank(stats: GroupStats) -> Int        # preference among the suitable ones
```

- **Stats** — what is measured, as a value rather than as arguments threaded through.
- **Policy** — `suitable`/`rank`, *pure functions over stats*. Today the policy is embedded in
  `_choose_strategy` and can only be exercised by running a whole group-by; as pure functions it is
  **unit-testable directly** ("skewed + high-cardinality + 8 threads selects X"), which is what
  makes tuning safe.
- **Mechanism** — one struct per strategy.

**Selection stays closed.** Iterate a comptime list of registered strategies and take the
highest-ranked suitable one. This must **not** become a runtime trait object: `marrow.aot`
depends on closed erasure and no open dispatchers, so an open registry would break the
binary-size property. Adding a strategy is a struct plus one entry in that list; removing one is a
deletion.

**Note this also fixes an existing wart.** `avoid_thread_local` today hand-lists "distinct, or
string min/max" at the call site — that becomes `has_unmergeable_agg` in the stats, derived from
comptime kernel properties, and the guard disappears into `suitable`.

**Skew is the motivating case.** The current estimator answers "how many groups?" but not "are
they even?" — a single dominant key defeats radix partitioning, since one partition gets most of
the rows. `max_group_share` from the same strided sample is nearly free to compute, and enables
strategies that specifically target skew (pre-aggregate hot keys before partitioning; salt a heavy
hitter across partitions and merge). None of that should be built now — the point is that the seam
makes it addable **without touching the aggregate path**.

**This has a hard dependency on Q6.1.** "Improve one scenario without hurting others" is
unverifiable unless each strategy can be benchmarked in isolation, so the harness must be able to
**force a strategy** rather than only exercising whatever the policy picks. Without that, a policy
change and a mechanism change are indistinguishable in the numbers, and a regression in a
rarely-selected path stays invisible until someone's data hits it. Add a `strategy=` override to
the bench entry points as part of Q6.1.

**Sequence it after fusion, not before.** Fusion changes the per-row cost that the policy's
thresholds encode, so the current cutovers (`_PARALLEL_MIN_ROWS`, `_PARALLEL_ALWAYS_ROWS`, the
cardinality split) will need re-measuring anyway. Introducing the seam first would mean tuning a
policy against costs that are about to change.

##### Risks

- **Binary size.** Fusion monomorphises per distinct aggregate-set. AOT programs contain few sets,
  so this should be small, but the gate is `query_streaming` stripped size and must be measured
  per step, not at the end.
- **Merge across partitions.** `merge` has to remap local group ids into global ones while folding
  blobs. Getting this wrong is silent wrong-answers, not a crash — it needs targeted tests before
  the parallel paths are switched over.
- **Numerical parity.** Fusing must not change fold order in a way that alters float results
  versus the unfused path; parity tests should compare fused and per-column results.

##### Two paths, one substrate — and both must win

Both frontends must be excellent, and they are excellent in *different comparisons*. Stating the
targets separately matters, because a design that is good for one can quietly be bad for the other.

| path | who it is compared against | target |
|---|---|---|
| **dynamic (F1)** — Python/ibis, runtime-planned | polars, pyarrow, duckdb, chdb — all themselves runtime-planned | **match or beat.** This is the apples-to-apples comparison; it is the one an evaluator will actually run. |
| **fused (F2)** — AOT comptime DSL | the same engines, none of which can do this | **decisively beat**, by a margin that grows with aggregate count. This is the differentiator. |

**The rule that keeps the dynamic path fast: dispatch is amortised over a vector, never paid per
row.** One resolution per `(chunk × aggregate)`, then a tight typed loop — which is precisely why
polars and pyarrow are fast, and where an interpreter naively written would lose by an order of
magnitude. The tag-removal step in Q2.5 is what makes this structural rather than incidental:
once the kernel *is* the aggregate, the dynamic path resolves a kernel once per chunk and then runs
the same typed code the fused path runs.

**The rule that makes the fused path decisive: zero dispatch and one pass.** All aggregates for a
group updated in registers, per-kernel comptime offsets into the accumulator blob, no per-aggregate
revisit of the input. ClickHouse reaches for a JIT to approximate this; marrow gets it at comptime,
which is the whole bet.

**One substrate, or the two paths drift.** Both must bottom out in the *same* `core[W]` SIMD
functors. If the dynamic path ever grows its own copy of the arithmetic, every optimisation has to
be done twice, they diverge in edge cases, and the benchmark stops comparing like with like. The
fused path is then "the same kernels with the dispatch removed and the loop fused" — not a
different implementation. This is the property that makes "great results on both" achievable rather
than a matter of maintaining two competing codebases.

**So the benchmark table carries both as separate rows, always** — `marrow-aot` *and*
`marrow-dynamic`, alongside every competitor. A change that speeds the fused path while regressing
the dynamic one is a **regression**, not a trade, and only a two-row table makes that visible.

**Q6.1 — Cross-engine aggregate benchmark, with the AOT path measured** · *gates every Q2.5 round* ·
Owns: `python/marrow/tests/bench_*.py`, `marrow/kernels/tests/bench_*.mojo`,
`benchmarks/aggregates/` (new), `pixi.toml` (bench tasks) ·

**Why this exists.** The fused-aggregate work (above) is justified entirely by a performance
claim — that comptime monomorphisation beats engines which dispatch per aggregate at runtime.
That claim has to be *measured against them*, before and after each round, or it is just an
assertion. "Refactor, then check nothing regressed" is not enough here: the point is to improve.

**What already exists.** `python/marrow/tests/bench_groupby.py` and `bench_clickbench.py` already
compare marrow against **pyarrow, polars and duckdb**, `--competition` prints a side-by-side
table, and the `bench` env carries `polars`/`duckdb`/`rich`. Reuse all of it.

**What is missing — and it is the important half.** Those benchmarks drive marrow through the
*Python bindings*, i.e. the **dynamic (F1)** path. The AOT (F2) fused path — the thing that is
supposed to be faster than everyone — is not measured at all. Each Python benchmark needs a
**paired AOT Mojo file expressing the same query through the fused comptime DSL**:

```
python/marrow/tests/bench_groupby.py        # pyarrow · polars · duckdb · marrow-dynamic
marrow/kernels/tests/bench_groupby_aot.mojo # marrow-AOT (fused)   ← new, the differentiator
```

Both already emit machine-readable output (`pytest-benchmark` on one side, `BenchSuite`'s `--json`
on the other), so a small merge script can print one table across both runners rather than asking
anyone to eyeball two.

**Done when:**
- Every table carries **both `marrow-aot` and `marrow-dynamic` rows** (see above) —
  a fused-path win that regresses the dynamic path is a regression, and one row would hide it.
- A **baseline is recorded in this document** — a committed table of marrow-AOT / marrow-dynamic /
  pyarrow / polars / duckdb across the group-by shapes that matter (low- vs high-cardinality keys,
  1 vs N aggregates, with and without nulls). N-aggregate cases are essential: they are precisely
  what fusion targets, and a 1-aggregate benchmark would show none of the win.
- Paired `*_aot.mojo` benchmarks exist for the group-by and ClickBench shapes, following the
  `BenchSuite`/`Benchmark` pattern in CLAUDE.md — **including `keep(data)` after `b.iter[call]()`**,
  or ASAP destruction frees the input mid-benchmark.
- One command prints the merged comparison.
- Consider adding **chdb** to the `bench` feature: ClickHouse is the engine whose aggregate design
  we are explicitly trying to beat, so it is the most informative competitor to have in the table.

**How it gates Q2.5.** Re-run after *each* step and append the numbers to the table:

| step | expectation |
|---|---|
| tags out of the kernel layer | neutral — behaviour-preserving |
| typed `aggregate[K]` + F1 loop | neutral to slightly better (one less indirection) |
| `FusedAggregation` (AoS, scalar unrolled) | **the step that must show a win**, growing with aggregate count |
| run-aware vectorisation | further win on clustered/partitioned inputs |

A step that does not move its number is a signal to stop and understand why, not to continue. Pair
each measurement with `pixi run binary_size` — fusion monomorphises per aggregate-set, and the AOT
binary is the gate.

##### Sequencing

0. **Q6.1 baseline first** — record the cross-engine table *before* any change, including the
   paired AOT benchmark. Without it there is nothing to prove improvement against.
1. Tags out of the kernel layer (prerequisite — without it the fused path cannot be expressed).
2. Typed `aggregate[K]` surface; F1 loops over it. **Behaviour-preserving, fully testable.**
3. Spike parameter-pack indexing; if it does not work, stop and redesign before writing more.
4. `FusedAggregation` with AoS + comptime offsets, scalar unrolled loop.
5. Benchmark against the per-column path across cardinalities; record the numbers in the card.
6. Only then: run-aware vectorisation, and SoA if the data asks for it.

#### `Grouping` — earlier design notes (superseded in part by the prior art above)

A naive `struct Grouping { gids, num_groups }` **preserves the bug it is meant to remove**.
`HashGrouper` is incremental: `consume_keys` is called once per batch and `num_groups` grows with
each call, so a `Grouping` built per batch carries a count that is already stale. That is exactly
today's hazard ("read `num_groups` *after* the last `consume_keys`") wearing a struct.

Constraints the design has to satisfy:

- **Streaming**: `AggregateProcessor` consumes N batches, buffering one gid array per batch, and
  only knows the final group count at the end.
- **Single-shot**: `GroupBy._serial` groups one input and aggregates immediately.
- **Partitioned**: the radix and thread-local paths run a grouper per partition and merge.
- **Invariant**: every gid `< num_groups`. Validating costs O(n), so it should hold **by
  construction** rather than by assertion — i.e. only `HashGrouper` may produce a `Grouping`.

Shape that satisfies all four: make `Grouping` the *completed* result, produced at finish time,
never mid-stream.

```mojo
struct Grouping(Movable):
    """One group id per input row, the final group count, and the unique key
    columns. Produced only by `HashGrouper.finish()`, so `gid < num_groups`
    holds by construction and the count can never be read early."""
    var _gids: List[Int32Array]      # one per consumed batch, in order
    var _num_groups: Int
    var _key_columns: List[AnyArray]
```

`HashGrouper.consume_keys` keeps returning the per-batch gids (the streaming path needs them as it
goes), but the *count* is only reachable through `finish()`. That removes the ordering hazard
structurally: there is no way to observe a partial count.

Open questions to settle while implementing — do **not** guess:
- Whether `finish()` should also absorb `key_columns()`, which is destructive today ("call once, at
  emit time"). Folding it in makes the once-only rule structural too; keeping it separate is a
  smaller change.
- Whether the partitioned paths produce one `Grouping` per partition and merge, or a single
  `Grouping` spanning partitions. The merge path remaps local group ids, so this decides where
  that remap lives.

 `(gids, num_groups)` currently travels
as two loose parameters across 8+ signatures with nothing checking `num_groups > max(gids)`. It
was logged as an independent cleanup; it is actually the enabling piece here, so do it first.

### Simplifications the new design should absorb

**`_reduce_widened` / `_reduce_widened_typed` are the same function twice.** The typed one's own
docstring calls it "the fully-typed counterpart of `_reduce_widened`" — yet the erased one
*re-implements* the reduce body rather than dispatching into it. It should be one line:

```mojo
# erased = resolve the dtype, then call the typed one
return array.dtype().dispatch_numeric[λ V: _reduce_widened_typed[K, V](...)]()
```

Both are also module-level free functions whose only callers are `AggKernel.reduce`'s defaults, so
they should be **static methods on the trait**, not free functions in the kernel module (they are
in the Q3.1 census for exactly this).

**The kernels are near-duplicate pairs.** `MinKernel` / `MaxKernel` differ only in `identity`
(`MAX_FINITE` vs `MIN_FINITE`) and `combine` (`math.min` vs `math.max`); `SumKernel` /
`ProductKernel` only in `identity` (0 vs 1) and `combine` (`+` vs `*`) — their `AccType`,
`finalize` and (post-Q2.5) `acc_dtype` bodies are identical. This is the same shape already
collapsed elsewhere in the codebase via an op-struct parameter (`ConditionalBinary[K]` with
`CoalesceOp`/`NullifOp`, and the fifteen shells in `expr/values.mojo`) — apply it here so a new
fold kernel is a few lines rather than a copied struct.

**Runtime `is_min: Bool` flags should become the kernel type.** `_minmax_temporal_scalar`,
`_minmax_string_scalar` and `min_max_string_grouped` all take a boolean to say which of min/max
they are, in a file that otherwise parameterises on `K: AggKernel`. Pass the kernel.

**Prerequisite decision — make this before touching `groupby.mojo`.** `AggState` currently holds
`PrimitiveBuilder[Self.Acc]`, a *logical* builder, and there is no `DType`-based builder. So the
physical rework needs one of: (a) add a physical `PrimitiveBuilder` over a `DType`, or (b) have
`AggState` own a `Buffer` + `BufferView[A]` and hand-manage growth. That choice determines all
four methods (`update`, `finish`, `into_partials`, `merge`) *and* their eight call sites, so pick
it first. Supporting evidence for the physical direction: the whole-array reduce helpers
(`aggregate.mojo:56, :82`) already project to `K.AccType[V].native` — only `AggState` still
carries the logical type.

**Preferred design — parameterise the state on the physical type and label at the boundary:**

```mojo
struct AggState[K: AggKernel, A: DType]        # physical accumulator
    def finish(mut self, num_groups: Int, out_dtype: ...) -> ...   # caller supplies the dtype
```

- No `Defaultable` problem — physical `DType`s are always constructible, so `acc_instance` and
  the dtype threading are unnecessary.
- **No cascade into the `Value` tower** — `Reduction`'s `OutType` is decided by the caller, so
  the `NumericValue` conformance failure never arises.
- Temporal min/max works because the fold is physical and the logical dtype is attached at the
  boundary — the same shape as the `filter`/`sort`/`hashing` fix in Q2.6, where the leaf only
  ever needed `T.native`.

This is the same root cause as `reinterpret_array`: **a layer demanding logical types when it
only uses physical ones.**

### Reverted attempt (2026-07-25) — kept for the traps it found

Stage A (the widening) was implemented and **compiled**, but broke the expression layer:

```
expr/values.mojo:1705: 'Reduction[K, A]' does not implement all requirements for 'NumericValue'
  comptime member 'OutType' type 'PrimitiveType' does not conform to required 'NumericType'
```

**The widening cascades into the fused expression tower.** `Reduction[K, A]` declares
`OutType = K.AccType[A.OutType]` and conforms to `NumericValue`, whose `OutType` must be a
`NumericType`. Widening `AccType`'s return to `PrimitiveType` breaks that conformance, and there
is no `PrimitiveValue` tier to move `Reduction` to. **Any retry must plan for a new tier in the
`Value` trait tower (`NumericValue` / `BoolValue` / `StringValue` / …), not just the kernel layer.**

What did work, and is worth keeping in a retry:
- The `Aggregate` / `AggKernel(Aggregate)` trait split.
- **`acc_instance[V](input) -> AccType[V]`** — the comptime-typed counterpart of `acc_dtype`. This
  is the piece that gets past the `Defaultable` wall without a downcast: `min`/`max` return the
  input's own dtype (unit and timezone intact), the widening aggregates return a defaultable
  numeric one. Routing through `AnyDataType` instead needs the private `_as`, i.e. trades one leak
  for another.
- Threading the accumulator instance through `AggState.__init__(input_dtype)` and its 8
  construction sites, plus `agg_out_dtype` / `agg_dtype`, which all default-construct `AccType[V]`.

**Process note:** the widening compiled cleanly against `test_aggregate` and `test_groupby` (23 and
32 passing) while the expr layer was broken — a per-file `check` is not proof the library builds.
Run `check_lib`, or at minimum `check marrow/expr/tests/test_streaming.mojo`, before believing a
kernel-layer change is done.

**Q2.6 — Delete `reinterpret_array`** · Depends: Q2.5 for the `groupby` half ·
Owns: `marrow/kernels/{filter,sort,hashing,groupby,aggregate}.mojo` ·

`reinterpret_array` (+ `AnyDataType.storage_type()` and `temporal_backing_dtype`) exists because
the *dispatch layer* routes temporal columns through the **numeric** branch — even though every
typed leaf is already bound on `PrimitiveType`, which temporal satisfies. It is not needed for
correctness anywhere:

| kernel | leaf bound | reinterpret needed? |
|---|---|---|
| `Filter` / `Take` | `T: PrimitiveType` | no — and removing it is *more* correct: today it strips the dtype and relabels it back, versus returning `PrimitiveArray[T]` with the dtype preserved by construction |
| `SortIndices` | `T: PrimitiveType` | no |
| `RapidHash` | `T: PrimitiveType` | no |
| `AggState` | `V: NumericType` | yes **until Q2.5** |

**Measured, and the monomorphization argument does not survive it.** Reinterpreting funnels ~15
logical primitive types into 4 integer widths, so removing it instantiates more kernel bodies —
but that cost lands almost entirely on the *runtime* binary, which is not what the project
optimises for:

| | before | after | delta |
|---|---|---|---|
| **fused** `query_streaming` | 1,357,176 | **1,274,584** | **−82,592 (−6.1%)** |
| runtime `query_dynvalue` | 15,859,016 | 16,684,776 | +825,760 (+5.2%) |
| ratio | 11.7× | **13.1×** | improved |

The fused binary *shrank*, because it no longer links `reinterpret_array` / `storage_type` at all.
So the reinterpret was buying runtime-binary size — which is out of focus — at the fused binary's
expense.



**Q2.1 — Add the missing accessors (RC1)** · Depends: — · Owns: `marrow/dtypes.mojo`,
`marrow/buffers.mojo`, `marrow/builders.mojo`, `marrow/utils.mojo`, `marrow/expr/values.mojo` ·
⚠️ BINSIZE · Done when no type reaches into another's `_`-prefixed fields:
`AnyDataType` exposes its variant (kills `dt._v` in `utils.mojo:287-316` and `arrays.mojo:447`);
`Allocation` exposes `is_device()`/`is_host()` (kills `Buffer`'s `_host`/`_device` probing — and
**fixes `Buffer.resize` mishandling DEVICE memory**); `PrimitiveBuilder.append_nulls(n)` replaces
`b._null_count = size`; `Context.get[A]` goes through `AnyArray._as[A]()`.

**Q2.2 — One concept, one owner (RC2)** · Depends: Q0.2 · Owns: `marrow/kernels/compare.mojo`,
`marrow/kernels/string.mojo`, `marrow/kernels/arithmetic.mojo`, `marrow/kernels/__init__.mojo`,
`marrow/buffers.mojo` (+ tests) · ⚠️ conflicts with `fu4-like-scalar` — **wait for it** ·
Done when: string comparison has one implementation (**this is FU-3**: add `StringLt/Le/Gt/Ge` to
`string.mojo`, give `BinaryCompareKernel` a `StringKernel` associated type, delete
`apply_string`/`str_predicate` and the legacy `equal` free functions); element-wise
`MinKernel`/`MaxKernel` are renamed so they stop colliding with the reducing pair and become
reachable through the package namespace; `Bitmap` forwards its bit operations to `self.view()`
instead of re-implementing them against `_buffer._ptr` (which is how `Bitmap.__eq__` drifted to
~64× slower than `BitmapView.__eq__`).

**Q2.3 — Validity plumbing hygiene (RC3, layout-preserving)** · Depends: Q2.1 ·
Owns: `marrow/buffers.mojo`, `marrow/views.mojo`, `marrow/kernels/helpers.mojo` ·
⚠️ **must not change array/scalar/builder layout** · Done when:
- `bitmap_and` takes `Optional[BitmapView]` (i.e. `.validity()`, **not** the raw `.bitmap` —
  today's signature yields offset-misaligned validity for sliced inputs) and returns
  `(bitmap, nulls)` so the count is computed once.
- The `nulls = length - count_set_bits()` incantation disappears from all 8 inline sites.
- `BitmapView.to_owned()` replaces the three ad-hoc "copy a view into an owned bitmap" idioms
  (`v.union(v)`, `~Bitmap.alloc_zeroed(n).view()`, and an identity SIMD functor).

> Scope note: an earlier draft proposed a `Validity` value type owning bitmap+offset+length and
> embedded it in every array. **Dropped** — array/scalar/builder layout is off-limits. Everything
> above is call-site and helper-level only.

**Q2.4 — `ExecutionContext` owns execution policy (RC6)** · Depends: **T2.3b merged** ·
Owns: `marrow/kernels/execution.mojo`, `marrow/views.mojo`, `marrow/buffers.mojo`,
`marrow/kernels/{filter,sort,partition,join,groupby,distinct}.mojo` · ⚠️ large, contended ·
Done when: `ctx.stripe[worker](n, min_parallel_size)` replaces the **7 hand-rolled
`sync_parallelize` chunk loops**; `Buffer.alloc_uninit[T](n, ctx)` replaces the **10-site**
`if ctx.is_gpu(): alloc_device else alloc_uninit` branch; `ExecutionContext` is threaded whole
rather than destructured to `num_threads: Int` and rebuilt (which currently **silently drops the
GPU device** at five sites in `HashJoin`); `hash_join`'s dead `ctx` parameter is removed; the three
incompatible parallel-gating idioms become one.

> Consider splitting into Q2.4a (context API: `execution.mojo` + `buffers.mojo` + `views.mojo`)
> and Q2.4b (migrate kernel call sites), so the API lands first and migration is mechanical.

---

## Tier 3 — free-function elimination (RC10)

### Schedule — three waves, no file conflicts

The Tier-3 tasks all want to *add accessors to the same core types*, so naively running them in
parallel collides on `views.mojo` / `buffers.mojo` / `arrays.mojo` / `utils.mojo` / `scalars.mojo`.
Sequencing Q3.2 first turns that contention into a dependency: it lands the accessors the others
consume, after which they are genuinely disjoint.

| wave | task | owns | why here |
|---|---|---|---|
| **1** | **Q3.2** core + memory | `views.mojo`, `buffers.mojo`, `arrays.mojo`, `utils.mojo`, `scalars.mojo`, `builders.mojo`, `c_data.mojo` | The hub. Lands `BitmapView.to_owned`, `Bitmap.unset_count`, `Bitmap.intersect`, `AnyArray.view(dtype)`, `ArrayData.owned_validity`, `PrimitiveArray.nulls/arange`, and moves `_apply_dispatch`/`_reduce_dispatch` onto `ExecutionContext`. **Run solo.** |
| **2a** | **Q3.1** kernels | `marrow/kernels/*` (+ tests) | Consumes `Bitmap.intersect` (for `bitmap_and`) and `AnyArray.view`. Biggest item: delete the **20** `filter`/`take` typed delegators, keep 3. |
| **2b** | **Q3.3** parquet + IPC | `marrow/ipc.mojo`, `marrow/parquet/*` | Consumes `LittleEndian.checked` and `AnyArray.view` from wave 1 — which is exactly what made `_read_le` and `_retag` cross-cutting before. |
| **2c** | **Q3.4 item 3** python | `python/bindings/*`, `marrow/tabular.mojo`, `marrow/scalars.mojo` (`as_py` only) | Items 1/2/4 already landed (`81fa29a`). ⚠️ touches `scalars.mojo`, so it must follow wave 1, not run beside it. |
| **3** | **Q3.5** expr | `marrow/expr/*` | ⚠️ BINSIZE. Consumes `owned_validity` / `to_owned` / `unset_count` from wave 1. Also contends with **Q2.5** on `expr/values.mojo` — pick one; do not run both. |

**Conflict matrix** (⚠ = same file, must not run concurrently):

| | Q3.1 | Q3.2 | Q3.3 | Q3.4·3 | Q3.5 |
|---|---|---|---|---|---|
| **Q3.1** | — | ⚠ `buffers` | · | · | · |
| **Q3.2** | ⚠ | — | ⚠ `utils`,`arrays` | ⚠ `scalars` | ⚠ `views`,`buffers`,`arrays` |
| **Q3.3** | · | ⚠ | — | · | · |
| **Q3.4·3** | · | ⚠ | · | — | · |
| **Q3.5** | · | ⚠ | · | · | — · but ⚠ **Q2.5** on `expr/values.mojo` |

Wave 2's three lanes are mutually disjoint, so they can run in any order or together — subject to
the machine limit in *Orchestration lessons* (one Mojo worktree at a time here; the disjointness
means they can also simply be done back-to-back without rebasing).



Rule to apply (from review §7). A module-level function survives **only** if:
1. **PyArrow/ibis parity** — must name the equivalent (`pc.filter`, `pa.list_`, `pq.read_table`);
2. **DSL entry point** (`col`, `lit`, `if_else`);
3. **No representable receiver** — C-ABI callbacks, or comptime adapters over *stdlib* types
   (`Variant`, `Span`, `UnsafePointer`) marrow cannot extend.

Everything else becomes a method, static factory, private method of its one owning type, or a
`Kernel` struct. Full per-function classification is in review §7.

**Q3.1 — Kernels (115 of 122)** · Depends: Q1.1, Q2.2 · Owns: `marrow/kernels/*` (+ tests) ·
⚠️ wait for `t2.3b-aggregate` and `fu4-like-scalar` · Highest-value order:
1. Delete the **20 typed `filter`/`take`/`drop_null` delegators** — keep exactly 3 free
   (`filter`, `take`, `drop_null`, all `pc.*`, all needed by the binding). Adding an array type
   drops from a six-site edit to two. Only 4 have production callers; each is a trivial rewrite.
2. `hashing.mojo` → `RapidHash` struct (17 fns) and `sort.mojo` → `SortIndices` struct (8 fns) —
   the two kernel modules with **no struct at all**. Includes renaming the public free function
   literally named `array()` (`sort.mojo:349`), which forces `import array as _primitive_array`
   in its own file. *(Fold into Q1.1 if done together.)*
3. Delete legacy `is_null`/`select`/`equal` free functions — `expr/dynamic.mojo` calls the **old,
   narrower** ones (numeric-only `is_null`; `select` silently drops validity).
4. `membership.mojo` → `IsInKernel`; `conditional.mojo` → `Multiplex` + kernel structs;
   `temporal.mojo` `date_trunc` → `DateTruncKernel` with a `TimeUnit` enum instead of a `String`.
5. Delete the **9 temporal delegators** (`year`, `month`, …) called only by tests.
6. Move `reinterpret_array` / `temporal_backing_dtype` out of `aggregate.mojo` onto
   `AnyArray` / `AnyDataType` (today `filter.mojo` imports from *aggregate* to filter a timestamp).
7. Delete verified-dead: `hash_identity` ×3, `_drop_null_bool`.

**Q3.2 — Core + memory (41 of 89)** · Depends: Q2.1, Q2.3 · Owns: `marrow/views.mojo`,
`marrow/utils.mojo`, `marrow/builders.mojo`, `marrow/scalars.mojo`, `marrow/c_data.mojo` ·
⚠️ BINSIZE · Key moves: `dispatch_over_*` → methods on `AnyDataType` (**64 call sites**, removes
the largest private-field reach-in); `_apply_dispatch`/`_reduce_dispatch` → `ExecutionContext`
(their bodies read *only* `ctx` accessors); the two bitmap↔bitmap `apply` overloads → private
methods on `BitmapView` (they read `_data`/`_offset`/`_length` from module scope); `nulls()` →
`PrimitiveArray[T].nulls()`; `arange` → `PrimitiveArray[T].arange` (no `pa.arange` exists);
delete `_invert/_and/_or/_xor/_and_not` (re-implement `SIMD` operators) and dead `scalar()` ×2.
Name `_heap_move` and `is_released()`/`mark_released()` in `c_data.mojo` — the C-ABI double-free
guard is currently open-coded **14 times**.
The 13 `KEEP-FREE` here (C-ABI callbacks + `variant_dispatch*`) are genuinely forced — leave them.

**Q3.3 — Parquet + IPC (24 of 27)** · Depends: Q1.2, Q1.3 · Owns: `marrow/ipc.mojo`,
`marrow/parquet/*` · Only 3 survive (`pq.read_table`, `pq.read_metadata`, `pq.write_table`).
Highlights: `_walk_slots` → `Page.scatter` (highest fan-in in the package); `_read_le` →
`LittleEndian.checked` (**28 call sites, 4 structs**); `xxh64` + 3 helpers → an `XxHash64`
namespace next to `Crc32`; `_retag` → `AnyArray.view(dtype)` (that is `pyarrow.Array.view`);
delete the 6 redundant `read_ipc_*`/`write_ipc_*` wrappers (each is one constructor call).

**Q3.4 — Python layer (65 delete, 19 relocate)** · Depends: — · Owns: `python/marrow/*.py`,
`python/bindings/*.mojo` · **Independent of all Mojo-core tasks — can run any time, and is the
best effort-to-value ratio in the plan.**

> **Target abstraction.** One home per function, PyArrow's home. The current state is not "some
> duplication" — it is *three* definitions of `filter`/`take`/`sort_indices` (top level,
> `compute.py`, and as `Array` methods) shipping **contradictory** `null_placement` defaults, so
> the same call means different things depending on which you reach for. Deleting the top-level
> copies is not cosmetic; it removes a live correctness hazard. While there: any kwarg that is
> accepted and ignored (`skip_nulls`, `mode`, `boundscheck`, `sort_keys`) must raise
> `NotImplementedError` — silently returning the wrong answer is worse than not offering the
> option. Highest
value in the whole plan for effort: delete the **24 duplicated compute functions** in
`python/marrow/__init__.py` (they ship *contradictory* `null_placement` defaults vs `compute.py`;
`filter`/`take`/`drop_null`/`sort`/`sort_indices` each exist **three** times — top-level,
`compute.py`, *and* as `Array` methods; `min`/`max`/`sum`/`any`/`all`/`filter` shadow builtins).
Then: `_as_py` → `AnyScalar.as_py()` (58 lines of core type dispatch stranded in bindings);
`_record_batch_join`/`group_by`/`aggregate`/`sort_by` (~265 lines of real semantics) → methods on
`RecordBatch` — they currently exist *only* for Python callers and import `marrow.expr.relations`
inside a function body, inverting CLAUDE.md's mandated split. Raise `NotImplementedError` on the
kwargs currently accepted and silently ignored (`skip_nulls`, `mode`, `boundscheck`, `sort_keys`).

**Q3.5 — Expr (~13 of 26)** · Depends: **T2.3b merged**, Q2.3 · Owns: `marrow/expr/values.mojo`,
`marrow/expr/dynamic.mojo`, `marrow/expr/relations.mojo` · ⚠️ BINSIZE · `_column_validity` +
`_result_validity` → one `ArrayData.owned_validity()`; `_view_to_owned` → `BitmapView.to_owned()`
(add the offset-0 fast path — it currently allocates and does a full pass **per column per batch**);
`_nulls_of` → `Bitmap.unset_count()`; `relations.execute` → `AnyRelation.execute(ctx)` (the only
plan verb that is not a method, and it collides with `Value.execute`); `slit` → a `lit` overload;
fix `lit`'s `value: Int` (today `lit(3.5, float64)` is unrepresentable).
**Measure before/after**: `_rank`/`promote` → `dtypes.mojo` and any `ColumnSet` type are the two
DCE-sensitive items; leave `into_array` free (promoting `Datum` to a struct is the highest binary-
size risk in the plan).

---

## Tier 4 — larger refactors (schedule deliberately)

- **Q4.1 — Missing value types (RC7).** `Grouping` (`gids` + `num_groups`, currently 2 parameters
  across 8+ signatures with no consistency check); `JoinKind` (bare `UInt8`s whose "emits right
  columns?" predicate is re-derived **three times with different membership**, and `JOIN_CROSS`
  silently falls into the LEFT/RIGHT/FULL branch); `JoinIndex` (split `SwissHashTable`'s two
  mutually-exclusive lifecycles); `BuildPartition` (replace three lockstep-indexed `List`s).
- **Q4.2 — Expr op registry (RC9).** ⚠️ BINSIZE. One `marrow/expr/ops.mojo` comptime registry
  driving F1's tags/names/uniform arms *and* F2's aliases. **F2 currently has ~30 operators F1
  lacks** and nothing enforces parity. ~80% of the wiring duplication is eliminable and is **not**
  load-bearing for DCE (the small-binary property comes from which trampoline `AnyValue.__init__`
  instantiates). Build the dynamic table only inside `DynValue.eval`.
- **Q4.3 — Parquet leaf visitor.** Collapse the **8 hand-written Arrow-type ladders** in
  `reader`/`writer`/`statistics`/`schema` into one `visit_leaf[V: LeafVisitor]`. They already drift
  (INT96 in one; `binary` missing from another).
- **Q4.4 — `ipc.mojo` (2318 lines) → a package**, mirroring `parquet/`. Natural split in review §2.
- **Q4.5 — Fused `prune`.** The AOT frontend **cannot prune row groups at all**
  (`_prune_tramp` always returns `unknown()`) while `pruning.mojo`'s docstring claims it can — so
  the performance-oriented frontend loses the biggest available win.

---

## Continuous / cheap

- **Q5.1 — Documentation drift.** CLAUDE.md lists `marrow/bitmap.mojo`, `marrow/visitor.mojo`,
  `BuilderData`, `ArrayVisitor`/`DataTypeVisitor`, `BitmapBuilder` — **none exist**.
  `buffers.mojo`'s own docstring points at "`marrow.bitmap`" for types defined 700 lines below in
  itself. `benchmarks/binary_size/README.md` documents four binaries against modules that no
  longer exist and quotes a ratio (30.9×) contradicting CLAUDE.md (~12×) — *the documentation of
  the project's central architectural gate does not describe the gate.* Also fix the three
  docstrings asserting capabilities the code lacks (`pruning.mojo:14`, `values.mojo:515`,
  `dynamic.mojo:5`).
- **Q5.3 — Finish the `1.0.0b3.dev2026072406` migration** · *blocker for the areas it touches* ·
  Owns: `marrow/arrays.mojo`, `marrow/builders.mojo`, `marrow/dtypes.mojo`,
  `marrow/kernels/tests/{bench_cast,profile_sort}.mojo`, `marrow/tests/test_views_gpu.mojo` ·
  Six files still fail to build:

  | file | error |
  |---|---|
  | `parquet/tests/{test_metadata,test_nested,test_writer}.mojo` | `types are not subscriptable` — see below |
  | `kernels/tests/bench_cast.mojo` | `apply`: unexpected keyword `safe` |
  | `kernels/tests/profile_sort.mojo` | `__floordiv__`: `UInt` vs `Int` |
  | `tests/test_views_gpu.mojo` | no matching `apply` overload |

  **Root cause of the parquet three (the interesting one):** `scalars.mojo` names a trait
  `Scalar`, colliding with the builtin `Scalar[_]`. `dtypes.mojo` already works around it
  (`from .scalars import Scalar as ScalarTrait  # \`Scalar\` alone = builtin`), but `arrays.mojo`
  and `dtypes.mojo` wildcard-import **each other**, so along that cycle the bare `Scalar` resolves
  to the trait and `Scalar[T.native]` stops parsing (~7 sites in `arrays.mojo`/`builders.mojo`).
  Rewriting each site to `SIMD[…, 1]` makes it *worse* (2 → 10 errors, measured) — it only moves
  the ambiguity. **Fix the cause, per the guiding standard:** rename the trait (e.g. `ScalarValue`)
  so one name means one thing, and drop the circular `import *` between `arrays` and `dtypes` in
  favour of explicit imports. That also removes a long-standing readability trap.

- **Q5.2 — Fold untracked items into `execution-engine-tasks.md`** so they are dated and
  closeable: D5 (was untracked), Q0.0, and the RC5 lifetime issue. Mark **FU-1 superseded by
  Q1.1** and **FU-3 absorbed into Q2.2**.

---

## Suggested first parallel batch

Three disjoint lanes. Q5.3 (compiler migration) is done, so nothing is blocked.

| lane | task | owns | why first |
|---|---|---|---|
| **A** | **Q0.2 + Q0.3** — fused expr correctness | `expr/values.mojo`, `expr/dynamic.mojo`, `kernels/boolean.mojo`, `expr/tests/{test_values,test_parity}.mojo` | Two proven wrong-answer bugs (D3, D4) |
| **B** | **Q1.1** — close the dtype ladders | `kernels/hashing.mojo`, `kernels/sort.mojo`, `dtypes.mojo`, `utils.mojo` + their tests | Only M1 blocker: `GROUP BY`/`ORDER BY` on temporal raises |
| **C** | **Q3.4** — Python layer dedup | `python/**` | Best effort-to-value; fully independent |

> Q0.2 and Q0.3 both touch `expr/values.mojo`/`dynamic.mojo` → **one lane**.
> Q1.1 must land before Q3.2 (both own `utils.mojo`/`dtypes.mojo`).
> All three are disjoint in *Owns*, so they can run concurrently in worktrees.

Q1.1 can join as lane E (owns `hashing.mojo`, `sort.mojo`, `dtypes.mojo`, `utils.mojo`) if
Q3.2 contends with Q1.1 on `utils.mojo`/`dtypes.mojo`, so run Q1.1 **before** Q3.2.
