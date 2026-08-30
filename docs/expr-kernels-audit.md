# Audit: abstractions in `marrow/expr/` and `marrow/kernels/`

Branch `expr-kernels-audit`, based on `97eb4701` ("refactor(expr): finish the
aggregate rearchitecture, and add pruning"). Scope: `marrow/expr/**` (8,947
non-test lines, 91 types) and `marrow/kernels/*.mojo` (12,217 non-test lines,
164 types).

Method: CLAUDE.md's own *How to Identify Leaky Abstractions* procedure — a
one-line responsibility per type, then the module dependency graph — plus
experiments. **Five changes were applied, compiled and reverted** in this
worktree to turn "this looks removable" into "this compiles"; they are marked
**VERIFIED (compiled)**. Everything else is VERIFIED (grep/read/measured) or
INFERRED, and the distinction is stated at every finding.

---

## 1. Executive summary

**The architecture is sound and the comptime lane is in good shape.** The fused
inner loops are fully typed, the erasure boundary sits per-morsel where it
belongs, all four erased boxes carry their `_drop` trampoline, `marrow/kernels/`
has no code dependency on `marrow/expr/`, and the convention sweep came back
clean on every hard rule (`alias`, `fn`, `unsafe_ptr`, `AnyOrigin`,
`constrained`, legacy closures: **zero violations in scope**). Most of what
follows is cleanup, not repair.

The ten things that matter, in rough value order:

1. **A live, reachable process abort.** `Value.aggregates` exists to stop an
   aggregate reaching a per-row operator. Four nodes need the guard; **two have
   it**. `rel.sort_by([col("a", int64).sum()], [True])` and
   `rel.aggregate(keys=[col("a", int64).sum()], ...)` reach `.value()` on the
   `None` an aggregate returns from `push`, and abort. → **B1**

2. **`AggKernel.empty()` is dead** — zero production callers. Its last consumer
   was deleted in `97eb4701` *itself*, whose own tombstone comment says
   `reserve` replaced it. So are `AggKernel.grouped()` and
   `ValidCount._per_group`. → **F1**

3. **The comptime lane's string path copies every string, per row, per operand**
   — `StringValue.lane` returns an owned `String`. The kernel layer's equivalent
   takes `StringSlice` and copies nothing, so the "fused" AOT string path is
   *more* expensive per element than the erased path it exists to beat. → **P1**

4. **Six string-comparison kernels are duplicated inside `expr/`.**
   **VERIFIED (compiled)**: the node builds against the kernel-layer ones with a
   three-line change; 63 lines of kernel code leave the expression layer. → **F2**

5. **Prose is 36% of both trees, and a quantified part of it is *wrong*, not
   merely verbose.** `runtime/values.mojo`'s module docstring documents a
   function-pointer design that does not exist and asserts the *opposite* of
   what the code does; `physical.mojo` documents an `Operator` contract with the
   wrong argument type, the wrong return type and a method (`finish`) that does
   not exist. ~635 measured lines of archaeology, ~32 self-referential `expr/`
   mentions. → **D1–D4**

6. **Two module cycles are removable by splitting `params.mojo`**, and a third
   (runtime → comptime) by relocating one operator whose bound is
   over-constrained — **VERIFIED (compiled)**. → **F5, F6**

7. **16 identical `dtype` bodies collapse to 3 trait defaults** —
   **VERIFIED (compiled)**, including the exact error proving it cannot go one
   level higher. → **F7**

8. **`marrow/expr/__init__.mojo` is empty**, so the backtick escaping CLAUDE.md
   says costs "one line, not every import" appears at **35 sites in 20 files**.
   → **F8**

9. **`Fold._domain` duplicates a check its only caller already ran**, and ships
   it into the AOT binary where it provably cannot fire — a measured 4,588-byte
   contiguous `__text` run. → **F4**

10. **The comptime lane is 6.6x smaller than the runtime lane** —
    independently re-measured at 1,488,528 vs 9,859,844 bytes of `__text`. The
    lane split is earning far more than the 3.4x its own docstring claims. → **M1**

**Naming collisions to fix while touching these:** `Aggregate` is defined twice
inside `marrow.expr` ([logical.mojo:889](../marrow/expr/logical.mojo#L889), a
`Relation`; [comptime/aggregates.mojo:73](../marrow/expr/comptime/aggregates.mojo#L73),
a `Value`), `Filter` twice in the tree, and five cast kernels collide with five
cast nodes — worked around by five `as ...Kernel` renames in a single import.

### Verdicts on the seven seed findings

| # | Seed | Verdict |
|---|---|---|
| 1 | `in_dtype: DynType` is redundant | **Partly confirmed.** 4 of 5 conformers ignore it; `StringExtremum.dtype` is provably redundant (`StringLikeType` *is* `Defaultable`). But `Fold[MinMax[...], V]` over temporal/decimal genuinely needs the *value* — `TimestampType` carries unit+timezone and is not `Defaultable` — and the trait must keep a uniform `DynType` for the erasure boundary. The removable part is `_domain`, not the parameter. → **F4** |
| 2 | `empty()` is unused | **Confirmed, completely.** 0 production callers, 17 references all in 2 test files. → **F1** |
| 3 | Do hot loops run through erased paths? | **Refuted for the comptime lane, confirmed as a different problem.** Erasure is strictly per-morsel; the fused aggregate loops call `lane[W]` on a concrete type with no dispatch. But the *string* lane copies per element (**P1**) and `Groups.single` heap-allocates per morsel (**P2**). |
| 4 | `dispatch_agg_array` — "what is that?" | **Confirmed.** Takes no array, knows no aggregate, is the only free `dispatch_*` in the kernels package, has one caller, and lives in the wrong module. Coherent scheme proposed. → **F3** |
| 5 | Docstring/comment volume | **Confirmed and worse than stated.** 7,582/21,164 lines reproduced exactly; the real defect is that a measurable slice is *false*. → **D1–D4** |
| 6 | `agg_vocabulary` shouldn't be in kernels | **Confirmed on layering, refuted on size.** One consumer, in `expr/runtime/` — wrong layer. But **measured**: the AOT binary links neither the function (0 symbols) nor the long name strings (0 vs 1). → **F10, M1** |
| 7 | File/struct organization | **Confirmed**, with specific seams named for all six large files. → **F11** |

---

## 2. Type-responsibility inventory

Per CLAUDE.md: *"if a single responsibility cannot be identified, that is a
leaky abstraction."* All 91 `expr/` types and all 164 `kernels/` types were
walked. **Almost every type states a crisp single responsibility** — this is
the audit's most reassuring result. The exceptions:

| Type | Anchor | Responsibilities found | Verdict |
|---|---|---|---|
| `RuntimeValue` | [runtime/values.mojo:84](../marrow/expr/runtime/values.mojo#L84) | (1) plan node, (2) **interpreter** — 30 `_tag ==` comparisons across `evaluate`/`dtype`/`prune`, (3) pruner, (4) aggregate fluent surface (10 verbs), (5) writer | **Leaky, deliberately.** This is the interpreter CLAUDE.md sanctions for the runtime lane. Report, do not "fix" — and note that a fn-pointer split is a documented dead end. |
| `AggState[K,V]` | [kernels/aggregate.mojo:622](../marrow/kernels/aggregate.mojo#L622) | (1) owns acc + seen columns, (2) **sole owner of the group count**, (3) **is the driver** — the per-row scatter loop, the SIMD lane loop, `combine_at`, (4) **is the finalize policy** — the `empty_is_null` / count-is-zero rule | **Leaky. The main one in `kernels/`.** Storage + loops + policy in one struct. |
| `Fold[K,V]` | [kernels/aggregate.mojo:1077](../marrow/kernels/aggregate.mojo#L1077) | (1) adapt `FoldKernel` → `AggKernel`, (2) own the whole-column `views.reduce` fast path, (3) `_domain` runtime validation | **Leaky.** (3) is removable outright → **F4**. |
| `FoldKernel` | [kernels/aggregate.mojo:97](../marrow/kernels/aggregate.mojo#L97) | (1) scalar-fold algebra, (2) *state-shape policy* read by `AggState` not by the algebra (`empty_is_null`, `needs_count`), (3) *input-domain gate* (`check_domain`) | **Leaky.** Its own docstring already separates (1) from (3) in prose. |
| `DistinctCount[exact,A]` | [kernels/aggregate.mojo:1487](../marrow/kernels/aggregate.mojo#L1487) | two unrelated algorithms (exact hash table vs HLL registers), both field sets declared, one always empty | **Leaky but blocked** — a struct body admits no `comptime if`, so no conditional fields. Acknowledged in-tree. Leave alone. |
| `PruneStats` | [pruning.mojo:195](../marrow/expr/pruning.mojo#L195) | statistics container + three lane-specific readings (`bounds[T]` comptime, `bool_truth`, `dyn_bounds` runtime) | Borderline; `dyn_bounds` should move → **F5**. |
| `kernels/core.mojo` | — | holds `Kernel` (13 importers) and `Groups` (3 importers) — two disjoint consumer sets, unrelated concerns | Module-level leak → **F11**. |

**Single-conformer traits** (premature abstraction, VERIFIED tree-wide):
`BoolUnaryKernel` ([kernels/boolean.mojo:69](../marrow/kernels/boolean.mojo#L69) → only `NotKernel`),
`BinaryFloatKernel` ([kernels/numeric.mojo:143](../marrow/kernels/numeric.mojo#L143) → only `PowKernel`),
`ListValue` ([comptime/core.mojo:943](../marrow/expr/comptime/core.mojo#L943) → only `ListColumn`).
`Foldable` and `Evaluable` are also single-conformer but are `conforms_to`
capability markers, which is a legitimate use — leave them.

---

## 3. Dependency graph

### 3.1 `kernels/` → `expr/`: clean

**VERIFIED**: `grep -rn 'expr' marrow/kernels --include='*.mojo'` returns **11
hits, every one inside a comment or docstring**. No code edge. The layering
holds in the direction that matters most.

Two *exports* nonetheless exist only to serve `expr/`:
`agg_vocabulary()` and `dispatch_agg_array` (→ **F3**, **F10**). That is a
backwards-facing API surface, not a cycle.

### 3.2 `expr/` internal graph: **not a DAG — 9 cycle edges**

```
                        ┌─────────────┐
        ┌──────────────▶│  bindings   │  ← proposed leaf (does not exist yet)
        │               └─────────────┘
   logical ◀────┐
     │  ▲       │
     │  │ (C3)  │ (C1)         C1  logical ↔ params
     │  └───────┴── params     C2  params  ↔ pruning
     │              │  ▲       C3  logical ↔ runtime.values
     │              │  │ (C2)  C4  runtime.values ↔ runtime.aggregates
     │           pruning       C5  comptime.core ↔ {aggregates, numeric,
     ▼                                              boolean, strings, temporal}
  physical ──▶ pushdown ──▶ parquet.reader
     ▲
     └──── comptime.* , runtime.*        X   runtime.aggregates → comptime.aggregates
```

| Cycle | Edges | Removable? |
|---|---|---|
| **C1** `logical ↔ params` | [logical.mojo:33](../marrow/expr/logical.mojo#L33) ← [params.mojo:63](../marrow/expr/params.mojo#L63) | **Yes** → F5 |
| **C2** `params ↔ pruning` | [params.mojo:58](../marrow/expr/params.mojo#L58) ← [pruning.mojo:103](../marrow/expr/pruning.mojo#L103) | **Yes** → F5 |
| **C3** `logical ↔ runtime.values` | [logical.mojo:36](../marrow/expr/logical.mojo#L36) ← [runtime/values.mojo:58](../marrow/expr/runtime/values.mojo#L58) | Hard — one import (`column`), needed by `DynRelation.select`. Leave. |
| **C4** `runtime.values ↔ runtime.aggregates` | [values.mojo:71](../marrow/expr/runtime/values.mojo#L71) ← [aggregates.mojo:88](../marrow/expr/runtime/aggregates.mojo#L88) | Inherent — the fluent verbs return the aggregate node. Leave. |
| **C5** `comptime.core ↔ 5 node modules` | [core.mojo:57,73,104,105,126](../marrow/expr/comptime/core.mojo#L57) ← each node module's `from .core import` | **Inherent and correct.** Trait defaults return concrete node types (`def sum(self) -> Sum[Self]`) — that *is* the fluent API. Not a defect; document it. |
| **X** `runtime → comptime` (not a cycle, a lane leak) | [runtime/aggregates.mojo:87](../marrow/expr/runtime/aggregates.mojo#L87) | **Yes** → F6 |

**Verdict: the graph is not a DAG. 2 of 5 cycles plus the cross-lane edge are
removable with the changes in F5/F6; the remaining 3 are structural.**

### 3.3 The two-lane split: holds, with one leak

The `DynValue` boundary is respected. Runtime→comptime narrowing happens in
exactly **two** places in the whole of `expr/` —
[pruning.mojo:523](../marrow/expr/pruning.mojo#L523) (`_ord`) and the aggregate
ladder in `runtime/aggregates.mojo`. No comptime node takes a `DynValue`
operand. The only leak is **X** above, plus three runtime-lane types
(`DynBounds`, `compare_dyn`, `PruneStats.dyn_bounds`) living in the shared
`pruning.mojo` (→ **F5**).

---

## 4. Findings

### B — Bugs

#### B1. Two of four nodes are missing the aggregate guard — reachable process abort

- **Anchors**: [logical.mojo:910-919](../marrow/expr/logical.mojo#L910) (`Aggregate.__init__`),
  [logical.mojo:1031-1050](../marrow/expr/logical.mojo#L1031) (`Sort.__init__`);
  the unchecked unwraps at [physical.mojo:663](../marrow/expr/physical.mojo#L663)
  and [physical.mojo:822-825](../marrow/expr/physical.mojo#L822).
- **What**: `Value.aggregates` exists so a node can reject an aggregate in a
  per-row position. `Filter` ([logical.mojo:727](../marrow/expr/logical.mojo#L727))
  and `Project` check it and raise. `Aggregate`'s `keys` list and `Sort`'s
  `keys` list do not — `Sort.__init__` validates only length and non-emptiness.
- **Why a problem**: **correctness/robustness.** An aggregate's `push` returns
  `None` unconditionally ([comptime/aggregates.mojo:435,483,544,619,697](../marrow/expr/comptime/aggregates.mojo#L435)),
  and both consumers call `.value()` on it without checking. That aborts the
  process. [logical.mojo:117-127](../marrow/expr/logical.mojo#L117) documents
  this exact failure as already fixed — the fix reached two nodes of four.
- **Evidence**: VERIFIED by reading all four constructors and both call sites.
  **Not** verified by running (would need a test that currently aborts the
  runner, which CLAUDE.md warns takes the whole suite down).
- **Fix**: hoist the guard into a shared helper and call it from all four —
  `Aggregate` on `keys` (aggregates in `aggs` are the point), `Sort` on `keys`.
  Effort: ~20 lines. Risk: none.
- **Blocked?** No.

#### B2. `get_field_index` returns `-1`; two callers index with it unchecked

- **Anchors**: [comptime/leaves.mojo:148](../marrow/expr/comptime/leaves.mojo#L148)
  (`TemporalColumn.dtype`), [comptime/leaves.mojo:414](../marrow/expr/comptime/leaves.mojo#L414)
  (`ListColumn.dtype`); [schema.mojo:156-161](../marrow/schema.mojo#L156).
- **What**: both do `schema.fields[schema.get_field_index(name)]`.
  `get_field_index` answers `-1` for an unknown column, and `fields[-1]` yields
  the **last** field.
- **Why a problem**: **correctness.** An unknown column silently reports another
  column's dtype instead of raising — on two methods already declared `raises`.
- **Evidence**: VERIFIED by reading. Not exercised.
- **Fix**: raise on `-1`. Effort: 6 lines. Risk: none. **Blocked?** No.

### F — Structure, duplication, dead code

#### F1. `AggKernel.empty()`, `.grouped()` and `ValidCount._per_group` are dead

- **Anchors**: trait member [kernels/aggregate.mojo:1012](../marrow/kernels/aggregate.mojo#L1012);
  overrides at [:1269](../marrow/kernels/aggregate.mojo#L1269),
  [:1483](../marrow/kernels/aggregate.mojo#L1483),
  [:1546](../marrow/kernels/aggregate.mojo#L1546). Also
  [:994](../marrow/kernels/aggregate.mojo#L994) (`grouped`) and
  [:1474](../marrow/kernels/aggregate.mojo#L1474) (`_per_group`).
- **Why a problem**: **clarity + binary size.** Dead virtual surface on the
  trait every aggregate conforms to.
- **Evidence**: VERIFIED. `grep -rn "\.empty()" marrow/ python/ benchmarks/ golden/`
  → 21 hits; 4 are the unrelated `Array.empty()`; the other **17 are all in two
  test files**. All 39 `.grouped(` hits are tests/benches. `_per_group` has
  **zero callers anywhere**, and its docstring cites a `partials` that does not
  exist in the aggregate layer. History: `b66e2198` introduced `empty()` with
  consumers, `6866bad9` dropped the stored `_empty`, `c737411f` fixed a drift
  bug in it, and **`97eb4701` deleted the last caller** — leaving an 11-line
  tombstone at [runtime/aggregates.mojo:327-337](../marrow/expr/runtime/aggregates.mojo#L327)
  that states it "had **no production caller**" and that `reserve` covers the
  case for all ten names rather than three.
- **Fix**: delete the trait member, 3 overrides, `grouped`, `_per_group`, and
  the 17 test assertions. Effort: ~90 lines removed. Risk: none — `_emit_fold`
  ([comptime/aggregates.mojo:343](../marrow/expr/comptime/aggregates.mojo#L343))
  already produces the zero-input answer via `reserve` + `finish`.
- **Blocked?** No. **This is the highest-value/lowest-risk item in the audit.**

#### F2. Six string-comparison kernels duplicated inside the expression layer

- **Anchors**: [comptime/strings.mojo:59-121](../marrow/expr/comptime/strings.mojo#L59)
  (`StringCompareKernel` + `StrEqKernel`…`StrGeKernel`) vs
  [kernels/string.mojo:405-471](../marrow/kernels/string.mojo#L405)
  (`StringEqKernel`…`StringGeKernel` on `StringPredicateKernel`).
- **What**: the same six comparisons, implemented twice, on two traits, in two
  packages. The `expr/` copies do not conform to `Kernel` and use
  `comptime name: StaticString` where `Kernel` uses `String`. The runtime lane
  uses the kernel-layer six; the comptime lane uses the local six.
- **Why a problem**: **clarity + drift risk + layering.** Kernels in the
  expression layer, and two lanes comparing strings through different code.
  The in-file justification ([strings.mojo:61-66](../marrow/expr/comptime/strings.mojo#L61))
  explains why the *SIMD* shape cannot serve variable-width strings — it does
  not explain why the replacement lives in `expr/` rather than in
  `kernels/string.mojo`, which already holds the answer.
- **Evidence**: **VERIFIED (compiled).** I repointed
  `StringCompare[K: StringPredicateKernel, ...]`, changed `Self.K.core(...)` to
  `Self.K.predicate(...)`, and repointed the six `StrEq`…`StrGe` aliases at the
  kernel-layer structs. `pixi run -e dev precompile` → **exit 0, no errors, no
  warnings**. Reverted.
- **Fix**: as above; delete 63 lines from `comptime/strings.mojo`. Effort: small.
  Risk: low — but **run `marrow/expr/comptime/tests/test_strings.mojo` and
  `golden/`**, since `precompile` compiles the library, not the tests'
  instantiations.
- **Blocked?** No.

#### F3. `dispatch_agg_array` is misnamed and in the wrong module; kernel naming is 60% consistent

- **Anchor**: [kernels/aggregate.mojo:1650](../marrow/kernels/aggregate.mojo#L1650).
- **What**: it takes no array and knows no aggregate — it maps a runtime
  `DynType` onto the *array type* that holds it. It is the **only free
  `dispatch_*` function in the kernels package** (all 48 other `dispatch`
  definitions are either static methods on kernel structs or methods on
  `DynType`), and it has exactly **one** caller,
  [runtime/aggregates.mojo:176](../marrow/expr/runtime/aggregates.mojo#L176).
- **Why a problem**: **clarity.** It also reads as a sibling of `dispatch_agg`
  ([runtime/aggregates.mojo:150](../marrow/expr/runtime/aggregates.mojo#L150)),
  which is a completely different operation (name×dtype → kernel) in a
  different package.
- **Evidence**: VERIFIED by reading the body and grepping all 48 `def dispatch`
  sites. Naming survey: of 164 kernel types, **99 end in `Kernel`, 6 in `Op`,
  and 59 in nothing** — including all 19 cast kernels and `Filter`/`Take`.
- **Fix** (the user's stated intent — *"dispatch functions should have separate
  names and the comptime ones should just use the type directly"*, which the
  comptime lane already honours):

  | now | proposed | why |
  |---|---|---|
  | `kernels.aggregate.dispatch_agg_array` | `arrays.dispatch_array` (free fn beside `DynArray`) | the 10th member of the `dispatch_*` family; nothing aggregate-specific |
  | `expr.runtime.dispatch_agg` | `resolve_aggregate` | it is *name* resolution, not dtype narrowing |
  | `SumKernel`/`MinKernel`/… (folds) | `SumFold`/`MinFold`/… | says "algebra, not entry point", **and dissolves the `numeric.MinKernel` vs `aggregate.MinKernel` collision** that currently forces [kernels/__init__.mojo:24-28](../marrow/kernels/__init__.mojo#L24) to re-export neither |
  | 19 cast kernels, `Filter`, `Take` | add the `Kernel` suffix | removes the 5 `as ...Kernel` renames at [comptime/casts.mojo:34-40](../marrow/expr/comptime/casts.mojo#L34) and the `Filter` collision with [logical.mojo:714](../marrow/expr/logical.mojo#L714) |
  | `StringExtremum` | `LexicalExtremum` | the one conformer named for its input |

  **Rule to adopt: `dispatch_*` names runtime→comptime narrowing only, and lives
  on the type it narrows.** Effort: mechanical but wide. Risk: low; must land
  with `pixi run build_python` and `pytest golden` per CLAUDE.md, since renames
  under `marrow/` are exactly what `precompile` cannot catch.
- **Blocked?** No.

#### F4. `Fold._domain` duplicates its caller's check and ships into the AOT binary

- **Anchors**: [kernels/aggregate.mojo:1128-1154](../marrow/kernels/aggregate.mojo#L1128)
  vs [runtime/aggregates.mojo:125-133](../marrow/expr/runtime/aggregates.mojo#L125).
- **What**: both write the same `comptime if conforms_to(K, ArithmeticAgg)` /
  `if not in_dtype.is_numeric()` gate, with two hand-written sentences that
  differ only in prefix. `_fold_agg` runs first and binds `V` *from* `in_dtype`,
  so `_domain`'s family arms can only fire after a different caller already
  resolved wrong.
- **Why a problem**: **binary size** in the size-gated AOT lane. In the comptime
  lane it cannot fire at all: `Column[T].dtype`
  ([leaves.mojo:65-67](../marrow/expr/comptime/leaves.mojo#L65)) *ignores the
  schema* and returns `DynType(Self.T())`, so the dtype `_domain` validates was
  manufactured from `V` two frames earlier.
- **Evidence**: **VERIFIED (measured).** In the fused aggregate gate,
  `nm` shows two `_domain` instantiations (360 and 348 bytes) inside a
  4,588-byte contiguous `__text` run spanning `Kernel::error` and three
  `Error::__init__` formatted-raise instantiations, on a 1,488,528-byte
  `__text`. All three format strings are present in the binary.
- **Fix**: delete `_domain`'s two family arms; **keep** the `holds[V]` guard
  ([:1150-1153](../marrow/kernels/aggregate.mojo#L1150)), which *is* reachable —
  `TemporalColumn[T].dtype` genuinely reads the schema, so a
  `TemporalColumn[TimestampType]` over a `date32` field hits it. Separately,
  `StringExtremum.dtype` ([:1350](../marrow/kernels/aggregate.mojo#L1350)) can
  become `DynType(Self.T())` because `StringLikeType` **is** `Defaultable`
  ([dtypes.mojo:105](../marrow/dtypes.mojo#L105)).
- **Blocked?** Partly — **the `in_dtype` parameter itself cannot be removed.**
  `TemporalType`/`DecimalType` are not `Defaultable`, so `MinMax.acc_dtype` has
  no other way to learn a timestamp's unit and timezone; and `trait Array` fixes
  no `Type` companion, so `AggKernel` cannot narrow the signature per conformer
  while `dispatch_agg`'s job calls `Agg.dtype(d)` generically. **This is the
  answer to seed 1: the parameter stays, the dead validation goes.**

#### F5. Split `params.mojo` — removes two cycles

- **Anchors**: [params.mojo:67](../marrow/expr/params.mojo#L67) (`comptime Bindings = Dict[String, DynScalar]`),
  [params.mojo:85](../marrow/expr/params.mojo#L85) (`struct Param[T: NumericType](NumericValue)`).
- **What**: `params.mojo` holds two unrelated things — a **type alias with no
  dependencies** and a **comptime-lane leaf node** structurally identical to
  `Literal[T]` (same trait, same `shape = Shape.scalar`, same
  `Bound = Scalar[T.native]`). Because they share a file, `params` must import
  `logical.Shape` and `pruning.param_bounds`, and both import it back.
- **Why a problem**: **structure.** Two of the five cycles exist only because of
  this pairing.
- **Evidence**: VERIFIED by reading. `Bindings` needs only `std.collections.Dict`
  and `..scalars.DynScalar` — a genuine leaf.
- **Fix**: `Bindings` → a new `marrow/expr/bindings.mojo`; `Param[T]` →
  `comptime/leaves.mojo` (which already imports `..pruning`, so `param_bounds`
  needs no new edge). `params.mojo` disappears. **While there**, move
  `DynBounds`, `compare_dyn` and `PruneStats.dyn_bounds` out of `pruning.mojo`
  into `runtime/` — VERIFIED: their only consumer is `runtime/values.mojo`, so
  the comptime lane currently relies on DCE to avoid them.
- Effort: medium (mechanical move + import fixes). Risk: low. **Blocked?** No.

#### F6. Relocate `BufferedAggregateOperator` — removes the runtime→comptime edge

- **Anchor**: [comptime/aggregates.mojo:630](../marrow/expr/comptime/aggregates.mojo#L630),
  imported by [runtime/aggregates.mojo:87](../marrow/expr/runtime/aggregates.mojo#L87).
- **What**: this operator is not lane-specific — it evaluates any `Evaluable`
  operand to a column and hands it to the kernel. Both lanes use it. Its two
  siblings (`Scattered`, `Register`) bind on `PrimitiveValue` and genuinely
  belong in `comptime/`.
- **Why a problem**: **structure.** It is the only reason the runtime lane
  imports the comptime lane.
- **Evidence**: **VERIFIED (compiled).** Its declared bound
  `A: Evaluable & Value` is over-constrained — it never calls a `Value` member.
  I changed it to `A: Evaluable`; `precompile` → **exit 0**. Reverted. That
  matters because `physical.mojo` deliberately does not import `logical.mojo`
  (the one-directional `logical → physical` edge), so the looser bound is what
  makes the move possible without creating a new cycle.
- **Fix**: loosen the bound, move the struct to `physical.mojo` beside the other
  operators. Effort: small. Risk: low. **Blocked?** No.
- Note: this is **not** the merge CLAUDE.md forbids — moving is not merging, and
  the `Fused`/`Buffered` unification remains blocked for the recorded reason.

#### F7. 16 identical `dtype` bodies collapse to 3 trait defaults

- **Anchors**: 16 occurrences of `return DynType(Self.Type())` across
  `comptime/{boolean,casts,numeric,strings,temporal}.mojo`.
- **Why a problem**: **clarity.** `ComptimeValue` declares
  `comptime Type: DataType` and every conformer restates the one-line body that
  reads it.
- **Evidence**: **VERIFIED (compiled), both directions.**
  (a) Defaults added to `NumericValue`, `StringValue` **and** `BoolValue`
  simultaneously → `precompile` exit 0.
  (b) Removing the override from `NumericBinary` while the `NumericValue`
  default is present → `precompile` exit 0, so the default is genuinely used.
  (c) The default **cannot** go on `ComptimeValue` — reproduced error:
  `core.mojo:258:33: error: no matching function in initialization / return
  DynType(Self.Type())`, because `Self.Type` is bounded only by `DataType`,
  which is not `Defaultable`. `TemporalValue` must also keep its override for
  the same reason (unit + timezone).
- **Fix**: three defaults, 13 overrides deleted. Effort: small. Risk: low.
- **Blocked?** No — and note this is *not* the CLAUDE.md trap: the return type
  is the concrete `DynType`, not `Self.AssocType`, and a same-signature override
  is ordinary.

#### F8. The package re-export surface is empty and unused

- **Anchors**: `marrow/expr/__init__.mojo` — **0 bytes**;
  [comptime/__init__.mojo:6-17](../marrow/expr/comptime/__init__.mojo#L6) (13
  names, never imported); [runtime/__init__.mojo:1](../marrow/expr/runtime/__init__.mojo#L1).
- **Why a problem**: **clarity + a false claim in two places.**
  `comptime/__init__.mojo`'s own docstring says the escaping "only has to be
  spelled at the boundary, in the package `__init__.mojo`", and CLAUDE.md
  repeats it as "**the cost is one line, not every import**".
- **Evidence**: VERIFIED. `grep` for package-level `from marrow.expr import` →
  **0**. Every consumer reaches submodules and escapes for itself:
  `` from marrow.expr.`comptime`.numeric import Gt `` appears in
  [benchmarks/binary_size/](../benchmarks/binary_size/) 10 times, and
  `` `comptime` `` appears **35 times across 20 files** repo-wide.
  `comptime/__init__.mojo`'s own list is an arbitrary subset — it has `Add`,
  `Gt`, `Lt`, `Mul`, `Sub` but not `Div`, `Eq`, `Ne`, `Le`, `Ge`, `And`, `Or`.
- **Fix**: populate `marrow/expr/__init__.mojo` with the intended public surface
  (`col`, `lit`, `table`, `scan`, `if_else`, `param`, `count_star`, the node
  aliases, `DynValue`, `DynRelation`), then rewrite the 35 import sites.
  Effort: medium. Risk: low, but **verify with the size gate** that re-exporting
  costs nothing — these are comptime aliases and generic structs, so it should
  be free, but that is INFERRED, not measured.
- **Blocked?** No.

#### F9. Two transitively dead kernels, both with fictional docstring consumers

- **Anchors**: `ConcatKernel` [kernels/string.mojo:237](../marrow/kernels/string.mojo#L237);
  `ArrayContainsKernel` [kernels/nested.mojo:93](../marrow/kernels/nested.mojo#L93).
- **Evidence**: VERIFIED. Transitive reachability over 480 library names left
  exactly these two dead. `ConcatKernel`: 2 repo hits (definition + re-export);
  its docstring claims the expression layer's `Concat` builds on it —
  `grep -rn "Concat" marrow/expr` → **0**, and there is no string-concat node in
  either lane. `ArrayContainsKernel`: 3 hits, zero tests.
- **Fix**: either wire them up (a `Concat` node is a real gap in the expression
  surface) or delete them. **Do not leave a public kernel whose docstring names
  a consumer that does not exist.** Effort: small either way.

Also dead: **top-K.** `sort_indices(..., limit=)`
([kernels/sort.mojo:804](../marrow/kernels/sort.mojo#L804)) is passed non-`None`
at exactly two sites, both tests. `SortOperator`
([physical.mojo:773](../marrow/expr/physical.mojo#L773)) never passes it, so
`sort_by(...).limit(k)` — an exercised plan shape
(`golden/cases/sort_top_k.mojo`) — fully sorts and then discards. **A dead
parameter *and* a missed optimization.**

#### F10. `agg_vocabulary()` is a frontend catalog in the kernel layer

- **Anchor**: [kernels/aggregate.mojo:218](../marrow/kernels/aggregate.mojo#L218).
- **Evidence**: VERIFIED. 8 grep hits, 3 of them code: the definition, the
  re-export, and one consumer module — `expr/runtime/aggregates.mojo`, at
  `RuntimeAggregate.vocabulary()` and `_checked`. **Measured (M1): the AOT lane
  links neither it nor the names** — 0 vs 78 `RuntimeAggregate` symbols, 0 vs 1
  for the string `approx_count_distinct`. So the size objection is refuted; the
  layering objection stands.
- **The file's stated justification is false.** [:198-204](../marrow/kernels/aggregate.mojo#L198)
  argues the constants must live here because a second hand-written list "is not
  a build error". There are **18 more hand-written spellings** that derive from
  nothing: 10 at [runtime/values.mojo:367-412](../marrow/expr/runtime/values.mojo#L367)
  — which are the actual *resolver keys* — and 8 at
  [comptime/core.mojo:637-687](../marrow/expr/comptime/core.mojo#L637). And
  `core.mojo` is internally inconsistent: `NumericValue.sum()` passes
  `String("sum")` while `StringValue.min()` derives the alias from
  `Self.Agg.name`. Separately, `agg_vocabulary()`'s list and `dispatch_agg`'s
  ladder are two independent enumerations of the same ten names.
- **Fix**: move `agg_vocabulary()` into `RuntimeAggregate.vocabulary()` — its
  sole consumer and the layer that owns "what names a frontend may say". **Keep
  the ten constants in `kernels/`**: they are `Kernel.name`, a real kernel
  property the comptime lane does read. Then replace the 18 literals with the
  constants. While there: `_checked`
  ([runtime/aggregates.mojo:302](../marrow/expr/runtime/aggregates.mojo#L302))
  allocates a fresh 10-element `List[String]` and linear-scans it on **every**
  `RuntimeAggregate` construction, duplicating a gate `dispatch_agg` already has.
- **Blocked?** No.

#### F11. File and struct seams

| File | Lines | Seam |
|---|---|---|
| [kernels/aggregate.mojo](../marrow/kernels/aggregate.mojo) | 1,692 | fold algebras (`FoldKernel`+ops) \| `AggState` (storage+driver+policy — split these three) \| the 5 `AggKernel` conformers \| the vocabulary (→ F10) \| `dispatch_agg_array` (→ F3) |
| [kernels/cast.mojo](../marrow/kernels/cast.mojo) | 1,575 | primitive casts \| string/binary casts \| **the decimal family (~330 lines, 6 structs + 5 helpers)** \| nested+dictionary \| the `cast()` ladder. The decimal block is the obvious `cast_decimal.mojo`. |
| [kernels/filter.mojo](../marrow/kernels/filter.mojo) | 1,147 | **two structs of 531 and 515 lines** (`Filter`, `Take`) plus 3 free functions. Each is a per-layout dispatch table; split by layout, not by struct. |
| [expr/logical.mojo](../marrow/expr/logical.mojo) | 1,237 | `Shape`+`Value`+`DynValue`+`merged` (the value layer, ~300) \| `Relation`+`DynRelation`+the 8 fluent verbs (~340) \| the 8 relation nodes (~550). Three files. |
| [expr/physical.mojo](../marrow/expr/physical.mojo) | 1,159 | wire types (`Datum`, `Morsel`) \| contract (`Operator`, `DynOperator`, `Evaluable`, `EvalOperator`) \| `Pipeline` \| the 8 operators. `ParquetScanOperator` (~95) is the only thing pulling `parquet.reader` in. |
| [expr/comptime/core.mojo](../marrow/expr/comptime/core.mojo) | 1,166 | `ComptimeValue`+`PrimitiveValue`+`Unnamed`+`ColumnBound` (the machinery) \| the four family traits, which are ~85% *fluent surface* (NumericValue alone declares ~40 methods). The fluent surface is what forces cycle C5; isolating it makes that visible. |
| [expr/pruning.mojo](../marrow/expr/pruning.mojo) | 612 | `Truth` (algebra) \| `ColumnBounds`+`PruneStats` (container) \| `DynBounds`+`compare_dyn`+`_ord` (**runtime-lane only** → F5) \| `Prunable`+`PrunePredicate` (trait+box) |
| [kernels/core.mojo](../marrow/kernels/core.mojo) | 127 | `Kernel` (13 importers) \| `Groups` (3 importers) — two disjoint consumer sets. Move `Groups` to `groupby.mojo`. |

### P — Performance

#### P1. The comptime lane copies every string, per row, per operand

- **Anchors**: [comptime/core.mojo:491-493](../marrow/expr/comptime/core.mojo#L491)
  (`def lane(self, bound, idx) -> String`),
  [comptime/leaves.mojo:333-334](../marrow/expr/comptime/leaves.mojo#L333)
  (`return String(bound.unsafe_get(UInt(idx)))`),
  [comptime/strings.mojo:172-180](../marrow/expr/comptime/strings.mojo#L172)
  (`StringCompare.lane` calls it twice per lane element).
- **What**: `StringValue.lane` returns an **owned `String`**. `unsafe_get`
  already hands back a borrowed `StringSlice`; the leaf then constructs an owned
  copy of the bytes from it. `StringValue.evaluate`
  ([core.mojo:478](../marrow/expr/comptime/core.mojo#L478)) does the same once
  per row into a builder.
- **Why a problem**: **performance, in the lane whose entire purpose is to be
  faster.** The kernel layer's equivalent —
  `StringPredicateKernel.predicate[o1, o2](s: StringSlice[o1], pat: StringSlice[o2])`
  ([kernels/string.mojo:300-304](../marrow/kernels/string.mojo#L300)) — copies
  nothing. So the erased path the fused lane exists to beat does strictly less
  work per element on strings.
- **Evidence**: VERIFIED by reading all three sites. **Magnitude unmeasured** —
  whether each copy mallocs depends on Mojo's small-string threshold, which I
  could not inspect (stdlib source is not on disk in this environment). Even
  with SSO it is a byte copy where the kernel path does none.
- **Fix**: a **spike**, not a patch. `StringValue.lane` would need a parametric
  origin (`-> StringSlice[__origin_of(bound)]`), and that does not work
  uniformly: `StringUnary` (`upper`, `lower`, `strip`) genuinely produces a new
  string and must own. Likely shape: keep `lane -> String` on the trait, add a
  borrowed reading on `ColumnBound` string leaves, and have `StringCompare` use
  it. **Measure first** with a string-comparison bench.
- **Blocked?** Unknown — needs the spike. Land it *with* F2, which already moves
  `StringCompare` onto the slice-taking kernel trait.

#### P2. `Groups.single` heap-allocates once or twice per morsel

- **Anchors**: [kernels/core.mojo:96-108](../marrow/kernels/core.mojo#L96);
  callers [physical.mojo:170](../marrow/expr/physical.mojo#L170) (via
  `Morsel.ungrouped`, on the pipeline driver path at
  [:416](../marrow/expr/physical.mojo#L416) and [:467](../marrow/expr/physical.mojo#L467))
  and [comptime/aggregates.mojo:690](../marrow/expr/comptime/aggregates.mojo#L690).
- **What**: `Groups.single` builds `Int32Builder(0).finish()` — an allocation
  for a value that by construction carries no data. Its own docstring says the
  one-slot assignment exists precisely to *avoid* materialising per-row ids.
- **Why a problem**: **performance**, mildly. Per morsel, not per row, so this
  is a small constant — but it is on the driver path of every query.
- **Evidence**: VERIFIED by reading. Not measured.
- **Fix**: make `ids` an `Optional[Int32Array]`, or hold one shared empty array.
  Effort: small. Risk: low — but `is_single()` is documented as a correctness
  hazard ("every implementation that loops over rows must branch on this
  first"), so touch it carefully. **Blocked?** No.

#### P3. Where the erasure boundary actually sits — answering seed 3 directly

**VERIFIED by reading every `push`/`drain` on the aggregate path.** The tree
honours CLAUDE.md's stated shape:

- `GroupByOperator.push` ([physical.mojo:686-701](../marrow/expr/physical.mojo#L686))
  crosses `DynOperator` **once per morsel per fold**, never per row.
- `ScatteredAggregateOperator.push` ([comptime/aggregates.mojo:421-483](../marrow/expr/comptime/aggregates.mojo#L421))
  is a straight-line SIMD loop over `self._input.lane[W](bound, i)` on a
  concrete `A` — no `DynArray`, no `DynType`, no `DynScalar` inside it.
- `RegisterAggregateOperator` keeps the accumulator in registers for the whole
  morsel.
- `BufferedAggregateOperator.push` ([:685-698](../marrow/expr/comptime/aggregates.mojo#L685))
  narrows once per morsel via `Self.Agg.InArray(column.to_data())` — a
  comptime-resolved conversion, not a dispatch.
- `RuntimeAggregate.to_operator` resolves the name **at plan-build time** and
  constructs a fully typed `BufferedAggregateOperator[Fold[K,V], RuntimeValue]`,
  so even the runtime lane holds no erased aggregate state.

**Conclusion: the aggregate hot loops do take the lowest-level path.** The
performance problems are elsewhere — P1 and P2, plus the runtime lane's 30
`String` tag comparisons (per morsel, not per row, and a documented deliberate
choice).

### D — Prose

Reproduced exactly: **21,164 lines, 7,582 prose (35.8%; 41.1% of non-blank).**
The volume is a symptom; the defect is that a measurable slice is **false**.
`C` (restatement) turned out negligible — a strict scan found ~10 true
restatements. The mass is **archaeology (~635 measured lines)** and **design
essays (~330)**.

#### D1. Docstrings that describe code that does not exist

The two worst are at the top of the two most-read files:

- **[runtime/values.mojo:3-4, 21-25, 105, 217-219, 532](../marrow/expr/runtime/values.mojo#L3)** —
  the module docstring describes a **fourth field, `_eval`, a function pointer
  bound at construction**, and states the invariant "**A tag never selects a
  kernel.** `_tag` is how a node prints and how it prunes; `_eval` is how it
  computes." `RuntimeValue` has three fields; `evaluate` is nothing but 30
  `_tag ==` comparisons; `EvalFn` is undefined tree-wide. This documents the
  exact design CLAUDE.md records as **removed after a miscompile**, in the
  present tense, and asserts the opposite of what the code does. **Highest-value
  prose fix in the tree.**
- **[physical.mojo:10-11, 15, 17, 180-181](../marrow/expr/physical.mojo#L10)** —
  the `Operator` contract is documented as
  `push(batch) -> Optional[StructArray]` / `finish() -> Optional[StructArray]`.
  Three errors: the argument is a `Morsel`, the return is a `Datum`, and
  `finish` does not exist. "**Two methods, and that is the whole physical
  contract**" — there are three (`done` has a default at
  [:214](../marrow/expr/physical.mojo#L214) and three implementers). The phantom
  `finish` recurs at [physical.mojo:579](../marrow/expr/physical.mojo#L579),
  [:776](../marrow/expr/physical.mojo#L776),
  [logical.mojo:1022](../marrow/expr/logical.mojo#L1022) and
  [comptime/aggregates.mojo:118](../marrow/expr/comptime/aggregates.mojo#L118) —
  the last contradicting [logical.mojo:117](../marrow/expr/logical.mojo#L117),
  which says `drain`, correctly.

Others, each VERIFIED by grepping the named symbol tree-wide:

| Anchor | Names | Reality |
|---|---|---|
| [comptime/core.mojo:172](../marrow/expr/comptime/core.mojo#L172), [:835](../marrow/expr/comptime/core.mojo#L835) | `FusedAggregateOperator` | never existed under that name; and the whole block at 829-836 is **factually reversed** — it says temporal `min`/`max` cannot fuse "today", while [comptime/aggregates.mojo:145-152](../marrow/expr/comptime/aggregates.mojo#L145) says in the present tense that the obstacle "is gone" |
| [logical.mojo:162](../marrow/expr/logical.mojo#L162) | `AggregateOperator[..., G]` | a fourth name, with a third parameter no operator takes |
| [logical.mojo:87](../marrow/expr/logical.mojo#L87), [:790](../marrow/expr/logical.mojo#L790), [comptime/core.mojo:11-13](../marrow/expr/comptime/core.mojo#L11), [comptime/aggregates.mojo:122](../marrow/expr/comptime/aggregates.mojo#L122), [runtime/values.mojo:87](../marrow/expr/runtime/values.mojo#L87) | `Analyzable`, `Executable` | removed; three of the five are present tense. `aggregates.mojo:122` proposes "a comptime `kind` on `Analyzable`" as future work — the mechanism already exists 4 lines below as `comptime aggregates` |
| [comptime/core.mojo:18-20](../marrow/expr/comptime/core.mojo#L18) | `operators.mojo`, `reductions.mojo` | neither file exists |
| [kernels/string.mojo:239](../marrow/kernels/string.mojo#L239) | expression layer's `Concat` | no such node (see F9) |
| [kernels/cast.mojo:123](../marrow/kernels/cast.mojo#L123) | fused AOT `Cast` node | no such node |
| [kernels/join.mojo:148](../marrow/kernels/join.mojo#L148) | `relations.Join`, `tabular.join` | `relations.mojo` does not exist; `tabular` has no `join` |
| [kernels/core.mojo:53](../marrow/kernels/core.mojo#L53) | `expr.aggregates` | previous package's path |
| [runtime/values.mojo:381](../marrow/expr/runtime/values.mojo#L381) | `Dispersion[1, False]` | takes three parameters |
| [tests/test_erasure.mojo:3,15](../marrow/expr/tests/test_erasure.mojo#L3) | `DynAgg` | removed |
| [kernels/cast.mojo:484](../marrow/kernels/cast.mojo#L484) | `# TODO: remove this` on `_reinterpret` | has 2 live callers + a test |

#### D2. Member counts that contradict the struct they sit on

- [logical.mojo:91-93](../marrow/expr/logical.mojo#L91), [:101](../marrow/expr/logical.mojo#L101) —
  "**Five members** … five members in one trait is the honest count." `Value`
  has **six**; `comptime aggregates` (line 116) is missing from both lists.
- [logical.mojo:187-188](../marrow/expr/logical.mojo#L187) — "**Five function
  slots**". `DynValue` has **six** (`_drop`) plus two constant fields — and its
  own `shape()` docstring five lines of code away says "rather than a
  **seventh** trampoline". Replicated into [pushdown.mojo:27](../marrow/expr/pushdown.mojo#L27).
- [logical.mojo:10-11](../marrow/expr/logical.mojo#L10) — "**Two methods, not
  eight**", listing `write` and `drop` among the eight rejected. Both are
  present slots on the current `DynRelation`.

#### D3. `expr/` means the *previous* package, inside `marrow/expr/`

**32 mentions across 13 files**, in three spellings: `` `expr/` `` ×27,
`exprold` ×4, `expr2` ×1. Every one reads as a statement about the file it is
in. The worst is [physical.mojo:1087-1089](../marrow/expr/physical.mojo#L1087),
which packs all three into one paragraph — "(superseded) `expr/`'s scan skips
row groups … Both need `expr/pruning.mojo`, which has no `expr2` counterpart
yet" — where `` `expr/` `` means the old package and `` `expr/pruning.mojo` ``
means the current file, in adjacent clauses. **And the paragraph is false**: the
docstring six lines above it and the code 50 lines below both implement pruning.
Two `exprold` mentions ([pruning.mojo:398](../marrow/expr/pruning.mojo#L398),
[:430](../marrow/expr/pruning.mojo#L430)) compare to a baseline in files the
previous package never had.

**Fix: one pass replacing the three spellings with a single unambiguous name
(e.g. "the previous expression package"), and deleting the comparisons that no
longer have a referent.**

#### D4. What to cut, quantified

~635 measured archaeology lines + ~200 net design narrative ≈ **700-750 lines
removable or relocatable (≈10% of prose)**, plus ~70 lines of duplicated
fluent-verb docstrings. Largest single blocks:

| Block | Lines | What |
|---|---|---|
| [kernels/join.mojo:29-101](../marrow/kernels/join.mojo#L29) | 73 | perf profile + 7 future optimizations → `docs/backlog.md` |
| [comptime/aggregates.mojo:308-340](../marrow/expr/comptime/aggregates.mojo#L308) | 33 | "One struct held all three for a while" merge history |
| [kernels/bounds.mojo:21-49](../marrow/kernels/bounds.mojo#L21) | 29 | "# Why this replaces `kernels/interval.mojo`" — a deleted file with zero consumers |
| [pushdown.mojo:8-34](../marrow/expr/pushdown.mojo#L8) | 27 | "# The mechanism" essay → design doc |
| [kernels/aggregate.mojo:289-325](../marrow/kernels/aggregate.mojo#L289) | 37 | `AccType`-was-`NumericType` narrative |
| [kernels/hashtable.mojo:60-107](../marrow/kernels/hashtable.mojo#L60) | 48 | ASCII layout diagram + glossary → `docs/` |
| [pruning.mojo:66-96](../marrow/expr/pruning.mojo#L66) | 31 | scope/value sections → the plan doc they cite |
| [pushdown.mojo:65-85](../marrow/expr/pushdown.mojo#L65) | 21 | projection-pushdown proposal → backlog |

**Keep** (verified load-bearing): [pushdown.mojo:36-63](../marrow/expr/pushdown.mojo#L36)
(the `Limit`-must-clear correctness trap), [kernels/core.mojo:100-127](../marrow/kernels/core.mojo#L100)
(`is_single` silently-empty-loop hazard), [comptime/numeric.mojo:555-580](../marrow/expr/comptime/numeric.mojo#L555)
(three compiler diagnostics quoted verbatim), [kernels/hashtable.mojo:623-656](../marrow/kernels/hashtable.mojo#L623)
(+450,112 bytes measured), [kernels/cast.mojo:1078-1099](../marrow/kernels/cast.mojo#L1078),
[builders.mojo:1-18](../marrow/expr/builders.mojo#L1).

**Duplicated verbatim**: 9 docstrings, 11 redundant copies — including a
12-line `Bound` docstring byte-identical at
[comptime/core.mojo:310-321](../marrow/expr/comptime/core.mojo#L310) and
[:999-1010](../marrow/expr/comptime/core.mojo#L999). Worse, the 10 aggregate
fluent verbs are documented on **five** surfaces and have already **diverged
semantically** — `min` is described three different ways, and the runtime
`variance` lost the comptime one's `ddof` parameter entirely.

### C — Convention consistency

**Clean, zero violations in scope**: `alias`, `fn`, `unsafe_ptr()`,
`AnyOrigin`/`unsafe_origin_cast`, `constrained(`,
`@__parameter`/`capturing[`/`@parameter`, `PrimitiveArray[bool_]`. The closure
migration is complete here.

Violations, almost all in tests:

| Violation | Sites |
|---|---|
| Wildcard import | [kernels/tests/test_concat.mojo:30](../marrow/kernels/tests/test_concat.mojo#L30) — `from ...dtypes import *` |
| `_underscore` across types — **library** | [kernels/cast.mojo:453](../marrow/kernels/cast.mojo#L453), [:1154](../marrow/kernels/cast.mojo#L1154) reach `Optional._value`; [:30](../marrow/kernels/cast.mojo#L30) imports private `std...._is_valid_utf8` |
| `_underscore` across modules — test | [expr/tests/test_pushdown.mojo:28-34](../marrow/expr/tests/test_pushdown.mojo#L28) imports 5 private names from `test_pruning.mojo` |
| Spelled-out type over alias | 16 sites (`PrimitiveBuilder[TimestampType]` ×10, `BinaryLikeBuilder[BinaryType]` ×5, `PrimitiveArray[Int32Type]` ×1), all tests |
| Explicit `DynScalar(...)`/`DynArray(...)` wrap | 9 sites, all tests |
| `.as_primitive[Concrete]` where a shorthand exists | 3 sites, all tests |

**49 gratuitous `raises`** in `marrow/expr/` (224 of 611 library `def`s are
`raises`). The big cluster is the 26 `dtype` implementations of F7 — forced by
the `Value.dtype` trait requirement, which genuinely needs `raises` for the
runtime lane. **The comptime cost is real**: per CLAUDE.md a raising `def`
cannot run at comptime, so all 26 are disqualified from compile-time evaluation
in the lane whose purpose is compile-time resolution. Three are free-standing
and directly removable: `table()` and `scan()`
([builders.mojo:198,209](../marrow/expr/builders.mojo#L198)) — the entry points
every plan starts from, so today no plan can be built inside a non-raising
function — and `DynRelation.limit()` ([logical.mojo:637](../marrow/expr/logical.mojo#L637)).

### E — Erased boxes

**All four are correct; no leak.** `rebind[ArcPointer[NoneType]]` appears in
exactly 4 places, all in `expr/`, and each has a `_drop` field, a `_drop_tramp`,
a `__deinit__` that calls it, and exactly one `__init__` (no bypass path):
`DynValue` ([logical.mojo:205](../marrow/expr/logical.mojo#L205)), `DynRelation`
([:390](../marrow/expr/logical.mojo#L390)), `DynOperator`
([physical.mojo:240](../marrow/expr/physical.mojo#L240)), **`PrunePredicate`**
([pruning.mojo:557](../marrow/expr/pruning.mojo#L557)).

**Gap**: `test_erasure.mojo` covers three boxes. `PrunePredicate` — the newest —
is **untested**, and CLAUDE.md's own point is that this bug class is invisible
without counting destructions. Add the fourth case. Effort: ~15 lines.

---

## 5. Measurements

### M1. Comptime vs runtime lane, re-measured

Built at `-O3 -g0`, stripped, `__text` section (independently re-verified with
`size -m` and `nm`):

| gate | `__text` | `RuntimeAggregate`/`agg_vocabulary` syms | `"approx_count_distinct"` |
|---|---:|---:|---:|
| fused (comptime) aggregate | **1,488,528** | **0** | **0** |
| runtime name-resolved aggregate | **9,859,844** | 78 | 1 |
| ratio | **6.6x** | | |

Three conclusions:

1. **The lane split is earning much more than advertised.** [logical.mojo:181](../marrow/expr/logical.mojo#L181)
   cites "1.46 MB against 4.91 MB" (3.4x). The measured ratio is 6.6x. The
   docstring understates its own case.
2. **Seed 6's size concern is refuted.** The AOT binary links neither
   `agg_vocabulary()` nor the long aggregate names. (`sum`/`min`/`max`/`count`
   read 0 in *both* binaries — short strings are SSO immediates, so those four
   are inconclusive by construction; the six long names are decisive.)
3. **9.86 MB is large in absolute terms** and `baseline.json` records
   `query_streaming_agg` at 1,932,404 (2026-08-05, pre-port). The README warns
   the gates changed program, so this is **not** a like-for-like regression —
   but it is a strong argument for re-baselining before anything else lands.

### M2. `Fold._domain` in the AOT binary

`nm -n` on the fused gate shows two `_domain` instantiations (360 and 348 bytes)
inside a **4,588-byte contiguous `__text` run** from `Kernel::error` to `_main`,
covering both `_domain` bodies and three `Error::__init__` formatted-raise
instantiations. All three format strings are present. See F4.

### What I could not measure, and why

**A full binary-size sweep did not complete.** Three separate `mojo build -O3`
invocations of the gates parked at **0.0% CPU with flat RSS** — the signature
CLAUDE.md documents for an elaborator deadlock. One of them
(`benchmarks/binary_size/query_streaming.mojo`, in the **main checkout**, not
mine) had been stuck **5 hours 29 minutes** when I found it, predating this
session. My own two attempts stalled the same way at ~800 MB and ~770 MB RSS.

I could not separate three hypotheses: (a) a genuine hang building
`query_streaming.mojo`, (b) lock contention on a compilation cache shared
through the symlinked `.pixi`, (c) plain CPU starvation from concurrent builds.
Two gates *did* build successfully earlier in the session, which argues against
(a) being universal. **Recommendation: investigate independently — a stuck
`mojo build` on the floor gate would silently blind the size gate in CI.**

Consequently the following are **INFERRED, not measured**, and each should be
gated before it lands: the size effect of F1 (delete `empty()`), F7 (trait
defaults), F8 (populating `__init__.mojo`), and the size *saving* of F4 beyond
the 4,588 bytes localized above.

---

## 6. Prioritized remediation plan

### Wave 1 — do first: correctness, then free deletions (low risk, no size gate needed)

| # | Change | Effort | Risk |
|---|---|---|---|
| **B1** | Hoist the aggregate guard; call it from `Aggregate.__init__` (keys) and `Sort.__init__` | S | none |
| **B2** | Raise on `get_field_index() == -1` in the two `dtype` methods | S | none |
| **F1** | Delete `AggKernel.empty()` + 3 overrides, `grouped()`, `ValidCount._per_group`, 17 test assertions | S | none |
| **E** | Add the `PrunePredicate` case to `test_erasure.mojo` | S | none |
| **D1** | Rewrite the two false module docstrings (`runtime/values.mojo`, `physical.mojo`) and fix the 11 dangling symbol references | S | none |

These are independent and can land as five commits.

### Wave 2 — structure (land together; one `pytest` + `build_python` + `golden` run covers all)

| # | Change | Effort | Risk |
|---|---|---|---|
| **F5** | Split `params.mojo` → `bindings.mojo` + `Param` into `comptime/leaves.mojo`; move `DynBounds`/`compare_dyn`/`dyn_bounds` into `runtime/` | M | low |
| **F6** | Loosen `BufferedAggregateOperator` to `A: Evaluable`, move to `physical.mojo` | S | low |
| **F7** | Three `dtype` trait defaults; delete 13 overrides | S | low |
| **F2** | Repoint `StringCompare` at `kernels.string`'s predicates; delete the 7 duplicated kernel types | S | low |
| **F4** | Delete `_domain`'s two family arms (keep `holds[V]`); `StringExtremum.dtype` → `DynType(Self.T())` | S | low |

**Group F5+F6+F7 in one commit** — they all touch the comptime trait/module
layout and splitting them means three full recompiles. **F2 and F4 are
independent** and can land separately.

Result: cycles drop from 5 to 3, the cross-lane edge disappears, ~110 lines of
duplicated kernel and boilerplate code go, and the AOT binary sheds the dead
validation.

### Wave 3 — naming and layering (wide, mechanical, needs the full verification set)

| # | Change | Effort | Risk |
|---|---|---|---|
| **F3** | The rename scheme: `dispatch_agg_array` → `arrays.dispatch_array`; `dispatch_agg` → `resolve_aggregate`; `*Kernel` suffix on the 19 cast kernels + `Filter`/`Take`; folds → `*Fold`; `StringExtremum` → `LexicalExtremum` | L | med |
| **F10** | Move `agg_vocabulary()` into `RuntimeAggregate.vocabulary()`; replace the 18 literals with the constants; drop `_checked`'s per-construction `List` | M | low |
| **F8** | Populate `marrow/expr/__init__.mojo`; rewrite the 35 escaped import sites | M | low |
| **D3/D4** | The prose pass: one name for the previous package; relocate ~700 lines to `docs/` and `docs/backlog.md`; dedupe the fluent-verb docstrings into one authority per verb | M | none |

**F3 must land with `pixi run build_python` and `pixi run -e dev pytest golden`**
— CLAUDE.md is explicit that a public rename under `marrow/` leaves `precompile`
at 0/0 while the tree is broken.

### Status as of 2026-08-30 — read this before the plan below

Waves 1-3 all landed, plus most of the spike list. The plan below is kept as
the record of how the work was scoped; **`docs/backlog.md` is authoritative on
what is still open.** Differences:

| item | plan below says | actual |
|---|---|---|
| **P1** — the string `lane` copy | spike | **done.** `lane` borrows; `Bound` owns the bytes |
| **Top-K** | spike | **won't fix** — needs a `Pushdown` row-limit channel *and* a per-node rule table where a wrong rule silently returns wrong rows |
| **`ConcatKernel`/`ArrayContainsKernel`** | spike | **won't fix** — correct and tested, reachable from no node; wiring them is feature work |
| **Re-record `baseline.json`** | spike | **moot.** It was re-recorded 2026-08-29 and a full run on 2026-08-30 matches it. The compiler deadlock that blocked the gate is fixed (`42bbcb14`) |
| **F5b** — move `DynBounds` | Wave 2 | **won't fix** — moving it means exposing `PruneStats._cols` |
| **F11** — file seams | not scheduled | **partial.** `Groups` and the decimal casts split out; six files remain over 1,000 lines |

Two claims in §4 were **refuted by later measurement** and should not be acted
on: the audit's `grouped()` is *not* dead (~40 live call sites), and the 16
duplicate `dtype` bodies collapse to **14**, not 16 — `BoolBinary` and `Not`
conform to `ComptimeValue` directly, so a `BoolValue` default does not satisfy
the base requirement.

---

### Needs a spike, not a patch

- **P1 — the string `lane` copy.** Design a borrowed reading that coexists with
  `StringUnary`'s owning one. Bench first. Land after F2.
- **Top-K** (F9): wire `sort_indices`' `limit` through `SortOperator`, or delete
  the parameter. A real missed optimization on an exercised plan shape.
- **`ConcatKernel`/`ArrayContainsKernel`** (F9): add the expression nodes their
  docstrings already promise, or delete both.
- **Re-record `baseline.json`** and investigate the stalled `query_streaming`
  build **before** anything in Wave 2 or 3, so the size gate means something
  again.

### Leave alone

- `RuntimeValue`'s multi-responsibility interpreter — sanctioned, and the
  alternative is a documented miscompile.
- `DistinctCount`'s two algorithms in one struct — blocked by "a struct body
  admits no `comptime if`".
- Cycles C3, C4, C5 — structural.
- `PruneStats.bounds[T: NumericType]` excluding temporal/decimal — a documented,
  deliberately deferred gap with a stated reason.
- `kernels/cast.mojo`'s measured-byte tradeoff comments — genuinely load-bearing.

---

## 7. Considered and rejected

Checked against CLAUDE.md's recorded dead ends; none of the above re-proposes
one. Explicitly:

| Idea | Why rejected |
|---|---|
| A shared generic `variant_dispatch(v, func)` replacing the per-box `isa` ladders | **Measured +662,740 bytes (+31.9% `__text`)** and reverted. A closure type cannot be generic over its own trait bound, so the helper must bind on `Movable` and the narrowing adapter inlines into every arm. |
| Make `DynValue` conform to `Value` | Recorded dead end. The comptime nodes bind on `ComptimeValue`; a runtime operand inside one discards the fusion the lane exists for. F2/F6 keep the lanes meeting at the box. |
| A `DynColumn`/`DynLiteral`/`DynCast` node | All three were added and removed. Nothing in this audit needs one. |
| A per-node fn pointer in `runtime/values.mojo` to replace the `_tag` switch | Recorded miscompile. **D1 is the opposite move** — deleting the docstring that still describes it. |
| Merge `Fused*` and `Buffered*AggregateOperator` behind one node parameterised on its operand bound | Blocked: both must *store* their operand, and a trait-valued associated type cannot type a field. **F6 moves one of them; it does not merge them.** |
| Remove the `in_dtype: DynType` parameter (seed 1's literal ask) | **Blocked, and this is the honest answer to the seed.** `TemporalType`/`DecimalType` are not `Defaultable`, so `MinMax.acc_dtype` cannot recover a timestamp's unit and timezone from `V` alone; and `trait Array` fixes no `Type` companion, so `AggKernel` cannot narrow the signature per conformer while `dispatch_agg`'s job calls `Agg.dtype(d)` generically. F4 removes the dead validation instead. |
| Put the `dtype` default on `ComptimeValue` (one default instead of three) | **Blocked, error reproduced**: `Self.Type` is bounded by `DataType`, which is not `Defaultable`. Three family-level defaults it is. |
| A `Grouping` trait parameterising `GroupByOperator` on placement | Already tried in-tree: **+24,432 bytes** for a branch that runs once per batch. |
| Making `BufferedAggregateOperator`'s `scatters` a comptime parameter | Already tried: **+4.6%** for a branch that never reaches the inner loop. |

---

## 8. Stale or now-false statements in `CLAUDE.md`

Found while auditing. **Not edited** — listed for the user to apply.

1. **"Statistics-based pruning, predicate/projection pushdown and the CLI-output
   layer are unported… do not describe them as available."** Statistics pruning
   (`expr/pruning.mojo`, 612 lines) and **predicate** pushdown
   (`expr/pushdown.mojo`, 224 lines) both exist, with tests
   (`expr/tests/test_pruning.mojo`, `test_pushdown.mojo`). **Projection**
   pushdown is genuinely still absent — `pushdown.mojo`'s own docstring says so
   and measures it at 3.6x. Split the sentence.

2. **The `expr/` directory tree omits `pruning.mojo`, `pushdown.mojo` and
   `comptime/temporal.mojo`.**

3. **"`FusedAggregateOperator` and `BufferedAggregateOperator` (`expr/physical.mojo`)"** —
   `FusedAggregateOperator` does not exist (the three are
   `Scattered`/`Register`/`BufferedAggregateOperator`) and all three live in
   `expr/comptime/aggregates.mojo`, not `physical.mojo`. Appears twice: in the
   Mojo Gotchas associated-types entry and in the Expression layer section.

4. **The erased-box list** ("`DynValue`, `DynRelation`, `DynOperator`, and the
   aggregate box") is out of date: the aggregate box is gone and
   **`PrunePredicate` (`expr/pruning.mojo:557`) is a fourth box** — correct, but
   unlisted and untested.

5. **"And the cost is one line, not every import. … Only the boundary crossing
   escapes — the parent `__init__.mojo`"** — false. `marrow/expr/__init__.mojo`
   is **0 bytes**; `` `comptime` `` is escaped at **35 sites across 20 files**,
   10 of them in `benchmarks/binary_size/` alone. Either fix the claim or fix
   the code (F8).

6. **"the fused aggregate operators are parameterised on their input"** and the
   physical-layer operator list omit the three-way `Scattered`/`Register`/
   `Buffered` split introduced in `97eb4701`.

7. **The kernels list** describes `aggregate.mojo` as "sum, product, min/max,
   count, mean, any/all" without mentioning that it also owns the frontend name
   vocabulary and a dtype→array-type dispatcher (F3, F10) — the two exports that
   make it a backwards-facing layer.

8. **Worth adding, not a correction**: `mojo build -O3` on
   `benchmarks/binary_size/query_streaming.mojo` was observed **stalled at 0%
   CPU for 5.5 hours** in the main checkout during this audit. If that
   reproduces, the size gate is silently dark, which is the same failure mode
   the "CI is dark" note already records for the test workflow.
