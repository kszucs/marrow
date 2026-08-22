# expr2 aggregation — implementation plan

Self-contained handoff. Written 2026-08-23 at the end of a long design session;
everything needed to continue is here or linked from here.

## Decision

Build **design A** (`docs/superpowers/specs/2026-08-23-expr-design-a.md`) plus
the aggregation architecture
(`docs/superpowers/specs/2026-08-22-aggregation-architecture.md`).

Design B (`…-expr-design-b.md`) is *better* on one axis — one implementation of
every operation, and rewrites that run at compile time — but costs the UX: an
explicit `comptime P = …` lift and comptime-parameterised builders. A synthesis
keeping both exists (typed nodes carry a `comptime PLAN`, the plan value is the
IR) but rests on one **unprobed** mechanism. See Tier 3.

**A is not a dead end.** Typed nodes are the surface in every version, so
adopting A forecloses nothing — a plan IR derived from those types is additive.

A's known weakness, stated plainly: a Python-built expression's *shape* cannot
be a type, so operations exist twice — typed nodes and runtime evaluators. That
is the status quo `expr/` already lives with. A does not fix it; B does.

## State of the tree

Branch `expr2`. **Updated 2026-08-23 during execution — the description below
replaces the original handoff text, which was stale.**

The work this plan described as "uncommitted and held for review" was **already
committed**; the tree was clean at the start of the session. `AggState._grow`,
`AggState.accumulate[W]` and `AggState.combine_at` are all in HEAD. Tier 1.2 was
therefore done before this plan was executed.

Landed since:

- `fix(expr2)` — the fused aggregate's mask splat. The plan recorded `update`
  as failing to instantiate, "bisected to the *body*, not the plumbing", to be
  fixed during the `Fold` rewrite. **That diagnosis was wrong.** The cause was
  `SIMD[DType.bool, W](True)`, whose positional constructor carries
  `comptime assert Self.size == 1` — *"use the `fill` keyword instead for
  explicit splatting"*. One line. All 13 cases pass. The verified fold body was
  never at fault, so there is nothing to carry into `Fold` beyond what is
  already written.
  **Note for the rewrite:** `fill=` is declared **only** for
  `SIMD[DType.bool, size]`. Numeric splats use the positional
  `Scalar[Self.dtype]` constructor and are already correct; applying `fill=`
  uniformly trades one error for two.
- `feat(expr2)` — the `Aggregate` relation and `AggregateProcessor` (Tier 2.8's
  first half). It **buffers nothing**: `update` takes the `RecordBatch`, so no
  per-aggregate chunk list is built. `HAVING` falls out as a `Filter` above it.
  Also renamed the value-level `Aggregate`/`DynAggregate` to
  `AggValue`/`DynAggValue`, which is what both docstrings already claimed and
  what frees `Aggregate` for the plan node.
- `test(binary_size)` — Tier 1.1, see below.

63 expr2 tests pass; `precompile` is clean at 0 errors, 0 warnings.

## Tier 1 — design-independent, build first

1. ~~**The `expr2` binary-size gate.**~~ **Done**, with one deviation. The plan
   said to mirror `query_streaming_agg_fused` *and* `query_streaming_agg`. The
   second could not be written: it measures the cost of a **runtime-named**
   aggregate identity, and `expr2` has no `AggFunc` equivalent —
   `NumericAggregate[K, A: NumericValue]` accepts only a fused input, so a
   runtime aggregate cannot be spelled at all yet. Shipped instead:
   `query_expr2_agg_fused` (1,320,356) and `query_expr2_streaming`
   (1,358,480), the latter covering filter + projection. Add the runtime-named
   gate when `expr2` grows a runtime aggregate.

   Both use an `int64` group key where their `expr/` twins use `string`:
   `Column[T]` is bound on `NumericType`, so a fused string column cannot be
   spelled. **The two packages' numbers are not comparable**; these gates catch
   `expr2` regressing against itself.

   **Running the gate immediately surfaced two pre-existing regressions**,
   neither caused by this work: `query_join` **+30.455%** (+459,216 bytes) and
   `query_dynvalue` **+0.862%**. Proven pre-existing by A/B — `query_join`
   measures 1,967,052 at `6b32d74`, byte-for-byte what HEAD measures. 59
   commits since the 2026-08-17 baseline reset touch what `query_join` links.
   **Do not `--update` the baseline to clear these**; that erases the signal,
   which is exactly the failure the baseline comment already documents once.
2. ~~**Commit the `AggState` fixes.**~~ **Done before this session** — already
   in HEAD, tree clean.
3. **`Grouping`** in `kernels/groupby.mojo` — `ScalarGrouping`, `HashGrouping`;
   placement extracted from `GroupBy`'s tangle. `PartitionGrouping`/`Sorted`
   later, as conformers rather than branches.
4. **Combinators** in `kernels/aggregate.mojo` — `State[K]`, `Merge[K]`,
   `If[K]`, `Distinct[K]`, each an `AggKernel` transformer. Two-phase
   aggregation becomes the same fold differently composed; `count_distinct`
   becomes `Distinct[CountKernel]` rather than a second state design.

## Tier 2 — A-shaped

5. `Fold[K: AggKernel, A: NumericValue, G: Grouping]` in
   `expr2/comptime/aggregates.mojo`, carrying the verified fold body below.
6. `Kind` on `Value` (`elementwise | reduction | analytic`) beside `Shape`.
7. **The push engine** (`…-push-engine.md`) — orthogonal to A vs B, and what
   cuts five `Dyn*` boxes to three. `Operator{push, finish}`; sources stay pull
   and drive; delete `Exhausted`.
8. ~~`Aggregate` relation + processor~~ **done**; then `PartitionGrouping` +
   `Window`.

## Tier 2.5 — `to_processor()` symmetry (added 2026-08-23, approved)

Agreed during execution, design approved, spec pending at
`docs/superpowers/specs/2026-08-23-expr2-processor-symmetry-design.md`.

**The rule: the physical counterpart of logical trait `X` is `XProcessor`,
erased as `DynXProcessor`, produced by `X.to_processor(ctx)`.** Three instances
— `Relation`/`RelationProcessor`, `AggValue`/`AggValueProcessor`,
`Value`/`ValueProcessor` — so today's `Processor`/`DynProcessor` are renamed to
`RelationProcessor`/`DynRelationProcessor` and `AggregateState` becomes
`AggValueProcessor`.

Four decisions, all settled:

- **The processor owns real per-execution state** — `IsIn`'s set, a compiled
  `LIKE` automaton, and the `ExecContext`, which today `FilterProcessor` holds
  and never hands to the value. `evaluate` takes `mut self`.
- **`to_processor` returns an associated type, and the processor holds the
  node** — `FusedProcessor[V]` owns `V` by value, so the comptime lane is never
  erased and fusion and DCE survive the boundary. **Not** a callback-based
  executor: the interposed closure adapter in `variant_dispatch` measured
  **+662,740 bytes**.
- **`bind`/`lane[W]`/`validity` stay logical.** They are compile-time
  composition — `Add[L, R].lane[W]` calls `L.lane[W]`. Only the driving loop
  becomes physical. `Evaluable` dissolves; `comptime shape` moves to
  `Analyzable`, where the other analysis questions live.
- **Two binding levels, not three.** `Prepared` waits for a real conformer.

**No incremental rollout is possible** — no trait default can return the
associated type unless it is `ImplicitlyCopyable`, which marrow's array types
deliberately are not. One commit, ~9 source files. The gates in Tier 1.1 exist
so this refactor's size cost is one falsifiable number.

## Tier 3 — the probe that could change the shape

Can a type carry `comptime PLAN` built recursively from its children's `PLAN`s
— `List` concatenation at comptime, upward through a generic parameter? The
pieces are individually verified (heap-holding structs as comptime parameters;
non-raising `def`s at comptime; conditional comptime types reducing *and*
carrying their trait bound when both branches are well-formed). Composing them
upward through a type tree is not.

If it works: the plan IR lands **under the existing surface**, rewrites run at
comptime, and the Python bindings share one set of operation implementations.
If it does not: A stands as-is.

## The fold body — verified, transcribe carefully

Four things, each of which cost a wrong answer or a crash when missing:

1. **A scalar tail is mandatory.** `for i in range(0, n, W)` reads past the view
   on the final chunk and **aborts the process** — buffer *size* rounds up to 64
   bytes and that is not slack. One compilation unit, so it fails every case in
   the file.
2. **`lane[W]` is null-blind.** It returns data bits regardless of validity, so
   a null must become the identity via `mask.select(value, ident)`. The trap: an
   unmasked `sum` is *silently correct* whenever the null slot's payload is 0 —
   only `min`/`max` expose it.
3. **The valid count is a second accumulator**, and **int64, not the
   accumulator type** (`mean` accumulates in float64). `sum` of nothing and
   `sum` of zeros are both 0, and it is `finalize`'s divisor. Reduce it once at
   the end, never a horizontal reduce per chunk.
4. **`AccType` must never appear unerased in a signature.** It is a comptime
   conditional type: it reduces inside a struct but fails to unify at a return
   site. `finish() -> DynArray` is forced, not chosen. `AggKernel.combine` also
   will not infer `W` from a `Scalar` — spell `combine[acc, 1](…)`.

## Measurements this rests on

| | |
|---|---|
| fused vs materialise, grouped, 1M rows | **1.17-1.68x** (g10 / g1k / g100k) |
| `lane[W]` vs `lane[1]`, grouped | 1.09-1.37x — the scatter stays scalar, the loads do not |
| scatter at one group vs register fold | **14.6x** — why `ScalarGrouping` is its own conformer |
| per-lane Kleene vs `_kleene`'s bitmap algebra | **4-10x worse** — do not fuse boolean validity |
| adding slots to the aggregate box | **+3.2 MB (+24%)**, recorded at `expr/aggregates.mojo:250-253` |

## Do not repeat

- **`Column[T: PrimitiveType]` does not work — probed and reverted
  2026-08-23.** The idea was sound and is what CLAUDE.md advises ("dispatch on
  the widest family the typed leaf accepts"): `PrimitiveType` carries
  `comptime native: DType`, which is all `lane[W]` needs, and `TemporalType` /
  `DecimalType` / `IntervalType` all conform to it — so **one** leaf should
  cover every fixed-width type and `expr/`'s separate `TemporalColumn` should be
  unnecessary duplication. It is not achievable as a widening: **19 errors**,
  because the aggregate chain is bound on `NumericType` the whole way down
  (`AggState[K, V: NumericType]`, `AggKernel.AccType[V: NumericType]`), so
  widening `NumericValue.Type` breaks every fold.

  The clean form needs `PrimitiveValue` as the family and `NumericValue`
  refining it — but **Mojo has no conditional conformance**, so a single
  `Column` struct cannot be a `PrimitiveValue` when `T` is a date and a
  `NumericValue` when `T` is an int. Temporal support therefore costs either a
  second leaf type (`expr/`'s answer) or widening `AccType` to `PrimitiveType`
  in `kernels/aggregate.mojo` first. **Do the kernels widening first** if this
  is attempted again; the expression layer is not where it is blocked.


- **`precompile` does not elaborate function bodies.** A clean `precompile` is
  not evidence a test will build, and `comptime assert` in an uninstantiated
  body reports *nothing*. Only `pytest` proves it.
- **Deletion is not part of this work.** `marrow/tabular.mojo:22-23` imports
  `expr/aggregates.mojo` to back `RecordBatch.group_by()`, a shipped
  PyArrow-mirror API; `python/bindings/compute.mojo:75-88` uses the kernel
  traits directly; `benchmarks/binary_size/query_streaming_agg_fused.mojo:19`
  imports `NumericAgg`. `Aggregation` has **five** conformers, and deleting it
  also deletes `AggFunction` and `AggKernel.Grouped`. This lands additively.
- **`x - avg(x)` is unresolved.** `expr/tests/test_values.mojo:166,175,182,192`
  cover it today. Any design forbidding aggregates in value position deletes
  four tested behaviours; the fix is aggregate extraction as a planning rule,
  not a value-level mechanism. Decide before `expr/` is deleted.
