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
pixi run binary_size      # fused ≈1.3MB vs runtime ≈15.7MB, ~12× — must not regress
```

### In-flight conflicts

None — `t2.3b-aggregate` and `fu4-like-scalar` are both merged. Re-check before dispatching.

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
