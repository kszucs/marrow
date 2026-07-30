# `marrow.expr` unification — merging `aot` + `dyn` into one runtime layer

**Status: consolidation complete (naming cleanup optional).** Delivered across
five commits on `deepseek`:

1. `aot`+`dyn` merged into one `marrow/expr/` package.
2. `Expr` merged into `TagValue` — one interpreter node.
3. `executor.mojo` + `Planner` deleted — processors folded into self-executing
   fat relation nodes.
4. Relation nodes migrated to hold `DynValue` — fused and Python-built
   (`TagValue`) values interchange through one value box.
5. `streaming.mojo` deleted — its fat nodes were a duplicate; the relational
   execution is now a single file (`plan.mojo`).

Net: one value box (`DynValue`), one interpreter (`TagValue`), one relational
layer (`plan.mojo`, self-executing fat nodes, no `Planner`). Full suite green
(123 passed, 18 skipped). Optional remaining cleanup: rename `plan.mojo` →
`relations.mojo` (moving `Table`/`col` out of today's `relations.mojo` into
`values.mojo` and dropping the typed `Project[*Es]`/`Filter`), and the
`marrow.kernels.concat` closedness follow-up noted below.

**Value-model clarification (confirmed with the user).** `TagValue` is what the
**Python bindings build** — it *is* the reworked `Expr` (there is no separate
`Expr`), with `List[DynValue]` children. `DynValue` boxes the typed/**fusable**
`Value` nodes (`Column`/`Add`/`Gt`) *and* `TagValue`, so relations hold
`List[DynValue]` uniformly and the fused-vs-interpreted choice is which node got
boxed.

## Why

`benchmarks/binary_size/` measured that a **runtime, walkable** relational tree
over **fused-only** value boxes (`query_erased_aot`) has a `__TEXT` segment
**byte-identical** to the fully-typed `Project[*Es]` layer (`query_comptime`),
and ~30x smaller than the `dyn` path. So the two packages no longer represent
two architectures — they collapse to **one axis: which value node you box**,
which is a per-node choice, not a package boundary.

The unification makes that explicit: one value box, one relational op set, and
**binary size = pay-for-what-you-construct**. "AOT vs dyn" becomes "which nodes
did you build," not "which package you imported." It also removes the current
cross-package dependency (`dyn.values` importing `aot.values` traits).

## Core idea

- **`DynValue`** — the one trampoline value box.
- **`TagValue`** (reworked from `Expr`) — *only* the tag-interpreter node, with
  `List[DynValue]` children. It is boxed into `DynValue` like any other node.
  It **drops `Expr`'s `FUSED` slots**: fusion is now "box a fused comptime node
  directly into `DynValue`," so the interpreter no longer carries a
  boxed-comptime bridge. `DynValue` = the box; `TagValue` = just the interpreter
  — the two roles `Expr` conflated, split apart.
- **Fused nodes** (`Column`, `Add`, `Gt`, …) box into `DynValue` and stay small.
  **`TagValue`** links the interpreter + its per-dtype kernel fanout *only when
  constructed* (parsed SQL, Python-driven plans).
- **Processors fold into relation nodes** ("fat nodes": node = operator =
  executor). The `Planner`'s open per-kind switch — the 30x size killer — is
  **deleted**; each relation self-executes, dispatched through the `DynRelation`
  vtable (references only its own `execute`, no central switch).

## Target structure

```
marrow/expr/
  values.mojo     # fused value nodes (Column/Add/Sub/Gt/Lt/Eq/Length),
                  #   the Table[T]/col() name-resolution handles,
                  #   DynValue + the boxing trampolines (typed fusable -> DynValue)
                  #   [merges aot/values.mojo + aot/relations.mojo leaves
                  #    + DynValue from aot/erased.mojo]
  runtime.mojo    # TagValue — the tag-interpreter node, boxable into DynValue
                  #   [was dyn/values.mojo; Expr -> TagValue, FUSED slots dropped]
  relations.mojo  # one self-executing relational op set (Scan/Filter/Project/Join)
                  #   + DynRelation box
                  #   [merges aot/erased.mojo relations + dyn/relations.mojo
                  #    + dyn/executor.mojo processors]
```

No `box.mojo` — boxing lives in `values.mojo` alongside the fused nodes. No
`typed.mojo` — the `Project[*Es]` type-pack layer is **dropped** (relational
fusion isn't worth pursuing now; see `docs/aot-relations-design.md`). `named.mojo`
is optional if `values.mojo` grows too large.

## Decisions to settle before/while doing it

1. **Streaming model of the fat nodes** — single-shot (`execute(batch) -> batch`,
   materializing; the `erased.mojo` model) vs pull-based (`next_batch()`, each
   node pulling its child; preserves `dyn`'s streaming/morsel/spill *and* removes
   the open dispatch). Pull-based is the better endpoint but more work; pick
   based on whether streaming must survive.
2. **`DynValue` boxing surface** — its constructors (in `values.mojo`) currently
   overload on `Column` / `BoolValue`. They need an overload (or a shared trait)
   covering the **erased** `TagValue`, which produces `DynArray` directly rather
   than being a statically-typed `Column`/`BoolValue`.
3. **Column consolidation** — today there are two `NumericColumn`s (positional in
   `aot/values.mojo`, name-resolved in `aot/relations.mojo`). Keep the
   **name-resolved** leaves (the pushdown/rewrite synergy depends on them); drop
   the positional duplicates.
4. **Benchmark set** — `query_comptime` (typed) retires with `typed.mojo`;
   `query_erased_aot` becomes the canonical *small* path (fused nodes through the
   unified ops); `query_runtime` becomes the *big* path (a `TagValue` tree through
   the *same* unified ops). Those two are the standing DCE gate.

## Migration phases (each gated on `binary_size` + tests)

1. **Rework `Expr` → `TagValue`** behind `DynValue`: children become
   `List[DynValue]`, drop the `FUSED` slots, move interpretation into
   `TagValue.to_array()`. **Gate:** re-run `binary_size` — the fused path stays
   ~250 KB, the `TagValue` path stays ~7.7 MB. Proves the unification preserves
   DCE *before* any files move.

   **Load-bearing invariant — verified (2026-07-07).** The riskiest assumption
   (one `DynValue` co-holding fused nodes *and* the interpreter, without the
   interpreter bloating the fused path) is confirmed, and confirmed to *scale*:
   `TagValue` (the reworked `Expr` — `to_array()` interpreter over
   `List[DynValue]` children, **no `FUSED` slots**) was added to
   `marrow/aot/erased.mojo` with an `DynValue` constructor, then grown from 2 ops
   (LOAD/GT) to `Expr.eval`'s full op set (arithmetic, comparisons, boolean,
   literal, length, if-else; CAST excepted, as in `Expr`).
   `benchmarks/binary_size/query_dynvalue.mojo` runs the same query and the same
   erased `Project`/`Filter` as `query_erased_aot`, with values boxed as
   `TagValue`. Measured `__TEXT`:

   | variant | `__TEXT` | note |
   |---|---:|---|
   | `query_erased_aot` (fused) | 229,376 B | **unchanged** — byte-identical before *and after* the interpreter grew ~9x |
   | `query_dynvalue` (seed: LOAD/GT) | 638,976 B | interpreter + `greater` fanout, on construction |
   | `query_dynvalue` (full op set) | 5,455,872 B | full per-dtype kernel fanout, **only** because a `TagValue` is constructed |

   The decisive point: the interpreter grew ~9x (639 KB → 5.46 MB of linked
   kernel surface) and the fused path stayed at **exactly 229,376 B**. So the
   fused-only program links *none* of the interpreter, no matter how large it
   becomes — the DCE boundary holds at full scale. Fusion still works (the fused
   path is byte-identical to the typed `query_comptime`). Correctness covered by
   `marrow/aot/tests/test_erased.mojo` (6 tests: fused path, interpreter
   arithmetic, LOAD name-resolution, fused-vs-interpreted equivalence, and a
   single `Project` co-holding a fused *and* an interpreter column); all aot
   tests pass. `TagValue` still lacks `Expr`'s *plan-manipulation* API
   (`kind`/`resolve_names`/`inputs`) — that lands with the relational-consumer
   repoint in Phase 2. It lives in `erased.mojo` until the Phase-3 relocation
   (renames last).
2. **Fold processors into fat relation nodes; delete `Planner`.** Re-run the full
   test suite (Python bindings + SQL) and `binary_size`. **Gate:** behavior
   identical, size unchanged.

   **Streaming fat-nodes — prototyped & verified (2026-07-07).** Decision:
   **pull-based** (streaming preserved). `marrow/aot/streaming.mojo` implements
   the fat-node design as a direct fold of `dyn.executor`'s processors: a
   `Source` trait (`schema()`/`pull() raises Exhausted`), an `AnySource`
   trampoline box, and `Scan`/`Filter`/`Project` where **the node is the
   processor** (holds its child `AnySource`, pulls morsels) — over `DynValue`
   values, with **no `Planner`**. `Exhausted` is defined locally so streaming
   doesn't import (and link) the dyn executor. Covered by
   `marrow/aot/tests/test_streaming.mojo` (4 tests incl. morsel-size
   independence and interpreter values); all aot tests pass.

   Measured `__TEXT` (`benchmarks/binary_size/query_streaming.mojo`):

   | variant | `__TEXT` | note |
   |---|---:|---|
   | pure streaming (pull morsels, no materialize) | 294,912 B | the fat-node design's true cost: +65 KB over single-shot for the pull machinery + `slice`; **no `Planner`** |
   | `query_streaming` (`collect()`, general `concat`) | 835,584 B | `DynBuilder(dtype)`'s open all-dtype switch |
   | `query_streaming` (`collect()`, local flat `concat`) | 425,984 B | bounded: flat typed builders only |
   | `query_runtime` (Planner executor) | 7,651,328 B | for reference |

   **The streaming design itself is size-safe** (~295 KB, no `Planner`, fusion
   intact). The `collect()` cost is materialization-only (pure morsel consumers
   never hit it). Root cause, corrected after measuring: the general
   `marrow.kernels.concat` routes through `DynBuilder(dtype)` — an open switch
   over *every* dtype (incl. nested list/struct) that is **never DCE'd** even
   for a provably-`int64` direct call (163 KB with the fallback replaced by a
   `raise`, vs 754 KB with it) because nested types genuinely need it. So
   `concat` can't be made closed without dropping nested support — the shared
   kernel keeps it. **Resolution (per the "exclude `concat` from expr"
   decision):** the expr layer uses its own *closed, flat-only* `_concat`
   (`marrow/aot/streaming.mojo`: typed primitive/bool/string builders, `raise`
   otherwise), dropping the general-kernel dependency → 835 → 426 KB. The
   residual ~131 KB over pure streaming is the flat typed builders: through the
   `pull` trampoline the collected column dtypes are **opaque**, so all flat
   builders link (bounded), not just the two used — inherent to type-erased
   streaming materialization, not the nested `DynBuilder` explosion. Remaining
   Phase-2 work: `Scan`/`Filter`/`Project` are folded; `Aggregate`/`Join` and
   the `TagValue` plan-manipulation API still to port.
3. **Physically consolidate** into `marrow/expr/`, do the renames
   (`values`/`runtime`/`relations`), delete `marrow/aot/` and `marrow/dyn/`,
   update all imports (Python bindings, tests). Renames last, once behavior and
   size are locked.

## Risks

- **Blast radius.** The Python bindings and SQL parser construct `Expr` today;
  their construction API shifts to `TagValue` / `DynValue`.
- **The whole value is a DCE property** — if `DynValue` ever makes
  `TagValue.execute` unconditionally reachable, the 250 KB win evaporates.
  Re-verify at every gate, not just at the end.
- **Reverses the earlier `expr/` deletion.** Justified: `expr/` "added nothing"
  when `aot`/`dyn` were *parallel* hierarchies; here they *unify* into one, so
  `expr/` becomes the real package rather than empty nesting. (It also holds
  relations, so `plan/` is an alternative name — open.)
