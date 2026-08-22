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

Branch `expr2`. **Uncommitted work exists and is deliberate** — it was held for
review. `git diff` / `git status` before anything else.

Uncommitted and **correct under every design** (Tier 1; commit these first):

- `marrow/kernels/aggregate.mojo`
  - `AggState._grow`, and `finish` growing before it reads. Fixes a **live
    out-of-bounds**: `finish` looped `range(num_groups)` reading `cnt` while
    only `update` ever grew the builders, so an aggregate over zero batches read
    unallocated slots. Aborts under `ASSERT=all`; a **silent bad read in
    release**. Only reachable once an accumulator can see zero batches.
  - `AggState.accumulate[W]` — public lane-shaped scatter, so a fused caller
    never reaches `_mark` (CLAUDE.md forbids `_underscore` access from outside).
  - `AggState.combine_at` — additive hand-off for a register fold.
- `marrow/expr2/comptime/aggregates.mojo` + its tests — the fold. **`update`
  fails to instantiate**; bisected to the *body*, not the plumbing (stubbing
  the body out compiles clean). It is being rewritten as `Fold[K, A, G]`
  anyway, so fix it there.
- `marrow/expr2/core.mojo`, `physical.mojo` — renames from the design session.

Committed this session: five specs under `docs/superpowers/specs/`
(`2026-08-21-expr2-design`, `2026-08-22-reductions-design` (superseded),
`2026-08-22-aggregation-architecture`, `2026-08-22-push-engine`,
`2026-08-23-expr-design-a`, `-b`), the boolean family, `col`/`lit`, the test
reorganisation, and `conftest.py` backticking reserved words in generated
import paths.

## Tier 1 — design-independent, build first

1. **The `expr2` binary-size gate.** `benchmarks/binary_size/` has five gates;
   **none builds anything from `expr2`**, so `pixi run binary_size` reports
   ~0.00% no matter what this work does. Mirror `query_streaming_agg_fused.mojo`
   and `query_streaming_agg.mojo` as `query_expr2_agg_fused` /
   `query_expr2_agg`, add them to `baseline.json`. Until this exists no
   binary-size claim about this work is falsifiable.
2. **Commit the `AggState` fixes** (above). Run
   `pixi run -e dev pytest marrow/kernels/tests/test_aggregate.mojo
   marrow/kernels/tests/test_groupby.mojo`.
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
8. `Aggregate` relation + processor; then `PartitionGrouping` + `Window`.

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
