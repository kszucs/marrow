# Abstraction audit — `marrow/expr/` and `marrow/kernels/`

**Date:** 2026-08-29
**Tree:** branch `expr2`, uncommitted working tree, immediately after `marrow/exprold/`
was deleted and its consumers (`golden/`, `benchmarks/binary_size/`) were ported.
**Build state at audit time:** `pixi run -e dev precompile` → exit 0, no errors, no warnings.
**Method:** read-only. No edits, no build beyond `precompile`.

---

## Headline

The aggregate layer is in genuinely good shape. `AggKernel`'s typed contract,
`dispatch_agg`'s parametric job, and the removal of `DynAgg` / `AggKernel.open` /
`ArrayInput` all landed correctly and should be **defended**, not revisited.

What remains is not conceptual confusion but **four concrete ragged seams**:

1. one quantity — "how many groups" — owned by six types;
2. the `comptime fuses` predicate written twice, identically;
3. a `Grouping` trait that is a `Bool` in a trench coat;
4. an `expr` package that is a single 11-module import cycle, because two ~30-line
   lane-agnostic vocabulary items live inside two ~1000-line modules.

Plus a substantial layer of docstrings describing types that no longer exist — one of
which is actively blocking an optimisation.

---

## 1. Responsibility table

Legend: ✅ one nameable responsibility · ⚠️ arguable · ❌ leak.

### `marrow/expr/` — lane-agnostic core

| Type | file:line | Verdict |
|---|---|---|
| `Shape` | `logical.mojo:53` | ⚠️ encodes two cases where there are three. `Aggregate.shape = Shape.scalar` (`comptime/aggregates.mojo:167`) yet a grouped aggregate produces one row *per group*; the docstring "One value for the whole batch" is false for every `GROUP BY`. |
| `Value` | `logical.mojo:87` | ✅ "what an expression is" — five members, honest count. |
| `DynValue` | `logical.mojo:175` | ✅ erase a `Value`. Five slots + two constants + drop trampoline. |
| `merged` | `logical.mojo:337` | ✅ order-preserving union. |
| `Relation` | `logical.mojo:349` | ✅ immutable description. |
| `DynRelation` | `logical.mojo:377` | ⚠️ erasure box **+** the fluent plan-builder (`:453-521`). Much smaller than the old binder — no schema derivation, no pushdown, no top-K folding — so a *mild* leak. The placement argument at `:446-451` holds. |
| `InMemoryTable` … `ParquetScan` | `logical.mojo:529-993` | ✅ one relational node each. |
| `Bindings` | `params.mojo:65` | ✅ but **misplaced** (§2). |
| `Param[T]` | `params.mojo:83` | ✅ but **misplaced** — a comptime-lane leaf living outside `comptime/`. |
| `Datum` | `physical.mojo:61` | ✅ `Scalar \| Array`. `is_scalar()` (`:96`) has no production caller except its own guard at `:103`. |
| `Morsel` | `physical.mojo:122` | ✅ batch + group assignment. |
| `Operator` | `physical.mojo:155` | ✅ push/drain/done. The `Datum`-as-output unification is right. |
| `DynOperator` | `physical.mojo:207` | ✅ erase an `Operator`, with the drop trampoline. |
| `Evaluable` | `physical.mojo:285` | ✅ the correct seam — it is what lets `AggregateOperator` accept a `RuntimeValue`. |
| `EvalOperator[V]` | `physical.mojo:297` | ✅ |
| `Pipeline` | `physical.mojo:334` | ⚠️ composite operator + driver. Defensible (`collect` is 12 lines); `push` is documented vestigial at `:404-414` and unreachable for every constructible pipeline. |
| `FilterOperator` … `BatchSourceOperator` | `physical.mojo:493-853` | ✅ |
| `GroupByOperator` | `physical.mojo:555` | ⚠️ key evaluation + grouping ownership + fold fan-out + **null-slot backfill from the plan schema** (`:688-697`). The fourth job is forced by `AggKernel.empty()` not being reachable per-conformer. |
| `JoinOperator` | `physical.mojo:855` | ❌ `_build_schema()` (`:951`) returns the **output** schema, is passed to `_concat_batches` for the **build side** (left columns only), and its docstring describes something else again. Saved today only by `_concat_batches`'s `len(batches)==1` shortcut (`:999`). |
| `ParquetScanOperator` | `physical.mojo:1009` | ✅ |
| `_struct_of` / `_concat_batches` | `physical.mojo:976`, `:995` | ⚠️ `_struct_of` validates nothing — backlog **AG-2**, still live. |

### `marrow/expr/comptime/`

| Type | file:line | Verdict |
|---|---|---|
| `ComptimeValue` | `core.mojo:81` | ✅ The defence at `:148-155` for *not* putting `Bound`/`bind`/`validity` here is correct. |
| `PrimitiveValue` | `core.mojo:192` | ✅ "fixed-width lane machinery". The split from `NumericValue` is the best refactor in the package. |
| `NumericValue` / `TemporalValue` | `core.mojo:449`, `:565` | ✅ domain markers. |
| `StringValue` | `core.mojo:338` | ✅ |
| `ListValue` | `core.mojo:599` | ✅ "consumed, never an operand". |
| `BoolValue` | `core.mojo:642` | ✅ |
| `Unnamed` / `ColumnBound` | `core.mojo:761`, `:785` | ✅ exemplary — the conformance *is* the documentation. |
| leaves (`Column` … `StringLiteral`) | `leaves.mojo:43-360` | ✅ |
| `ListLength` | `leaves.mojo:411` | ✅ |
| `NumericBinary` … `ConditionalBinary` | `numeric.mojo:52-401` | ✅ `TemporalCompare`'s three-way merge analysis (`:280-311`) is recorded work — do not re-open. |
| `BoolBinary` / `Not` / `NullPredicate` | `boolean.mojo:70-166` | ✅ |
| casts | `casts.mojo:56-246` | ✅ |
| `StringCompare` … `StringLength` | `strings.mojo:108-328` | ✅ |
| `StringCompareKernel` + 4 kernels | `strings.mojo:59-106` | ❌ **kernels defined in the expression layer** (§2). |
| `Aggregate[Agg, A]` | `aggregates.mojo:73` | ✅ one aggregate over one operand — correctly one struct. |
| `AggregateOperator[Agg, A, G]` | `aggregates.mojo:288` | ⚠️ two machines in one struct (§4). |
| `promote` / `wider` / `widest_shape` | `rules.mojo` | ✅ |

### `marrow/expr/runtime/`

| Type | file:line | Verdict |
|---|---|---|
| `Payload` | `values.mojo:65` | ✅ closed variant. |
| `RuntimeValue` | `values.mojo:75` | ⚠️ node + interpreter (`evaluate`, `:202-267`). Deliberate, defended by backlog §0 (+1.8 MB for a tag switch on the other side of the trade). Accept. |
| `RuntimeAggregate` | `aggregates.mojo:235` | ✅ name + operand, resolved at plan time. |
| `dispatch_agg` | `aggregates.mojo:163` | ✅ **the** name × dtype ladder. |
| `_fold_agg` | `aggregates.mojo:116` | ✅ |
| `SUM`…`MAX` + `vocabulary()` | `aggregates.mojo:104-113`, `:302` | ❌ a second catalog of names the kernels already own (§4). |

### `marrow/kernels/` — aggregate family

| Type | file:line | Verdict |
|---|---|---|
| `Kernel` | `core.mojo:18` | ⚠️ name + three argument checks; `expect_same_length`/`expect_same_dtype` are meaningless for `AggKernel` and `IntervalKernel`. |
| `Groups` | `core.mojo:48` | ⚠️ `is_single()` conflates two states — backlog **AG-1**, unfixed. `single()`'s docstring explains itself by reference to `Morsel.ungrouped`, a plan-layer type. |
| `FoldKernel` | `aggregate.mojo:94` | ✅ the pure algebra — **but its docstring is false twice** (§7). |
| `WideningOp` / `MinMaxOp` | `:190`, `:312` | ✅ |
| `ArithmeticAgg` | `:268` | ✅ domain marker. |
| `Widening` / `MinMax` / `CountKernel` / `MeanKernel` | `:277-419` | ✅ |
| `BoolReduceKernel` / `AnyKernel` / `AllKernel` | `:455-513` | ⚠️ a second, unrelated aggregate hierarchy in the same file; not `AggKernel`s, unreachable from either lane. Only consumers: `utils/testing.mojo:612`, `python/bindings/compute.mojo:93-94`. |
| `AggState[K,V]` | `:575` | ✅ |
| `AggKernel` | `:806` | ✅ **the contract is right.** |
| `Foldable` | `:945` | ✅ one responsibility — see §4. |
| `Fold[K,V]` | `:995` | ⚠️ carries `_slots`, duplicating `AggState.num_groups()`. |
| `Dispersion` / `StringExtremum` / `ValidCount` / `DistinctCount` | `:1131-1406` | ✅ each; but each re-implements `_grow` with a different length convention. |
| `dispatch_agg_array` | `:1569` | ✅ dtype → array type. |
| `HashGrouper` | `groupby.mojo:37` | ✅ |
| `Grouping` | `groupby.mojo:156` | ❌ §5 |
| `ScalarGrouping` | `groupby.mojo:206` | ❌ **never constructed anywhere.** |
| `HashGrouping` | `groupby.mojo:230` | ✅ used concretely by `GroupByOperator`. |
| `Interval` + `IntervalKernel` + 9 conformers | `interval.mojo:35-245` | ❌ **zero consumers** — no production code, no test, no benchmark. |

### Leaks, most severe first

1. **"How many groups" has six owners.** `HashGrouper._table.num_keys()` → `Groups.num_groups` (`core.mojo:70`) → `GroupByOperator._num_groups` (`physical.mojo:599`) → `AggregateOperator._num_groups` (`aggregates.mojo:340`) → `Fold._slots` (`aggregate.mojo:1029`) → `AggState.acc.length()`.
   **`Fold._slots` is provably redundant**: `max`'d at `:1072, :1109, :1121, :1127`, each immediately followed by a call that `_grow`s `AggState` to the same value; `_grow` is monotonic; therefore `_slots == self._state.num_groups()` invariantly.
   `AggKernel.finish`'s docstring (`:908-914`) says *"Takes no group count… Passing one here is how a caller and a state disagree"* — and `Fold.finish` (`:1104`) passes one.
   `AggregateOperator._num_groups` is **dead in the non-fusing arm**: written only under `comptime if Self.G.scatters` (`:374`), read only under `comptime if Self.fuses` (`:528`).
2. **`Grouping` is a one-bit trait with no polymorphic consumer.** §5.
3. **Two kernel families for string comparison, one in the wrong package.** `expr/comptime/strings.mojo:59-106` defines `StringCompareKernel` + `str_eq/ne/lt/gt`; `kernels/string.mojo` defines six, used by `runtime/values.mojo:271`. The expr-side trait does not conform to `Kernel` and types `name` as `StaticString` where `Kernel.name` is `String` (`kernels/core.mojo:21`).
4. **`comptime fuses` written twice, identically** — `aggregates.mojo:128` and `:322`. If they diverge, `Aggregate.to_operator` picks `HashGrouping` while the operator takes the buffered arm.
5. **The aggregate name catalog exists twice** — `Fold.name`/`Dispersion.name`/… in `kernels/aggregate.mojo`, and `SUM`…`MAX` + `vocabulary()` in `runtime/aggregates.mojo:104-113, :302`. Nothing enforces agreement. Backlog §5.1 left this open.
6. **`Aggregate` is two different types in one package** — `logical.mojo:698` (a `Relation`) and `comptime/aggregates.mojo:73` (a `Value`). `benchmarks/binary_size/query_expr2_agg_fused.mojo:32,34` imports both and must know which is which.
7. **`RuntimeAggregate.empty()` (`runtime/aggregates.mojo:344`) has no production caller** — `GroupByOperator.drain` fills empty slots from the plan schema instead (`physical.mojo:688`). Exercised only by a test whose own docstring describes a `resolve` method that no longer exists.
8. **`marrow/kernels/interval.mojo`** — 253 lines, one trait, nine conformers, zero consumers.

---

## 2. Dependency graph

### `marrow/kernels/` — clean, verified

**No edge from `kernels/` into `expr/` or `tabular/`, and no cycle.** Graph rebuilt
mechanically from every non-test `from … import` under `marrow/`. `kernels` is a DAG;
longest chain `sort → cast → filter → core`.

Two **spirit-level** wrong-way pulls, neither an import:

- `Grouping`/`ScalarGrouping` exist solely to serve `AggregateOperator`'s bound.
  `ScalarGrouping` is never constructed; `Grouping` has no polymorphic consumer.
- `Groups.single()`'s contract is documented by reference to `Morsel.ungrouped`
  (`kernels/core.mojo:79-83`) — the kernel root module explaining itself with a
  plan-layer type.

Mirror-image misplacement: `StringCompareKernel` and four kernel structs live in
`marrow/expr/comptime/strings.mojo`.

### `marrow/expr/` — one 11-module strongly connected component

Tarjan over the same graph:

```
SCC = { logical, physical, params,
        comptime.core, comptime.aggregates, comptime.boolean,
        comptime.numeric, comptime.strings, comptime.rules,
        runtime.values, runtime.aggregates }
```

Every non-test module except `builders.mojo` and `comptime/casts.mojo`. Eight mutual pairs:

```
comptime.core  <->  comptime.aggregates      (fluent defaults return node types)
comptime.core  <->  comptime.boolean         ( "  )
comptime.core  <->  comptime.numeric         ( "  )
comptime.core  <->  comptime.strings         ( "  )
runtime.values <->  runtime.aggregates       ( "  )
comptime.core  <->  params                   <- removable
logical        <->  params                   <- removable
logical        <->  runtime.values           <- removable
```

The first five are the structural "a trait default returns a concrete node type" shape
already recorded for `values ⟷ dynamic`. They are the price of the fluent API; keep them.

**The last three are accidents of placement:**

- 11 of the 12 modules importing `logical` want only `Shape` (± `Value`/`merged`) — a ~30-line value type.
- 11 of the 12 modules importing `params` want only `Bindings` — a `comptime` alias for `Dict[String, DynScalar]`.

So the package is one cycle because two tiny lane-agnostic items are parked inside the
two largest modules.

**One cross-lane edge:** `runtime.aggregates → comptime.aggregates`
(`runtime/aggregates.mojo:85`, for `AggregateOperator`). Not a cycle and not wrong —
reusing the operator is what deleted `RuntimeAggregateOperator` — but it makes
`AggregateOperator` misplaced: it is *the* aggregate operator for both lanes while
sitting in one lane's directory.

---

## 3. Lane symmetry

### Correct, keep

- Both nodes conform to `Value`; both are boxed by the same `DynValue`; both reach the
  same `EvalOperator`/`AggregateOperator` through `to_operator(schema, grouped, bindings)`.
  **This is the best thing about the current design.**
- Both aggregate nodes carry the `_name`/`_alias` split, `shape = Shape.scalar`,
  `aggregates = True`.
- `col(name, dtype)` vs `col(name)` selecting a lane by what the caller knows is a good
  API idea, and `builders.mojo`'s docstring on why the overload set cannot be split is correct.

### Divergent without a stated reason

**Operators.** The comptime lane uses dunders (`__add__`, all six comparisons,
`__and__/__or__/__xor__/__invert__`, `core.mojo:179-189, 512-548`). The runtime lane uses
**free functions only** (`eq/ne/lt/le/gt/ge/and_/or_/xor/not_`, `values.mojo:402-450`) and
has **no dunder at all**. Nothing explains this; `RuntimeValue` is not `Equatable`, so
`__eq__` returning a node is expressible exactly as on `StringValue`.

**Comparison coverage.** `NumericValue` has six. `StringValue` has **four**
(`core.mojo:419-428`) because only `StrEq/StrNe/StrLt/StrGt` exist. `kernels/string.mojo`
has all six and the runtime lane uses all six (`values.mojo:236-241`). `TemporalValue` has
**zero** — `TemporalCompare` exists (`numeric.mojo:270`) with all six aliases, but no
method produces one, so callers must name `TemporalGt(...)` as a type.

**18 comptime nodes have no fluent method.** `golden/prelude.mojo:45-63` imports as
*types*: `IsNull`, `NotNull`, `NumericCast`, `NumToBool`, `BoolToNum`, `StringToNum`,
`NumToString`, `CaseWhen`, `Coalesce`, `FillNull`, `EndsWith`, `ILike`, `Like`, `Lower`,
`StartsWith`, `StringLength`, `Strip`, `Upper`. None is reachable by a method on any value
trait, while every aggregate is. This contradicts CLAUDE.md's own "write with the fluent
API" instruction.

**Parity table.** Backlog §0 invariant 2 says no feature may exist in only one lane. There
is no `test_parity.mojo` replacement, and the invariant is violated in both directions.

| capability | comptime | runtime |
|---|---|---|
| arithmetic `+ - *` | ✅ dunder | ❌ absent |
| six numeric comparisons | ✅ dunder | ✅ free fn |
| string comparisons | 4 of 6, node-only | 6 of 6 |
| `and/or/xor/not` | ✅ dunder | ✅ free fn |
| `coalesce` | binary, numeric-only (`numeric.mojo:401`) | n-ary, any dtype (`values.mojo:452`) |
| `case_when` | 3-ary numeric (`builders.mojo:152`) | n-ary any dtype (`values.mojo:465`) |
| `nullif` / `fill_null` | ✅ node-only | ❌ absent |
| `is_null` / `not_null` | ✅ node-only | ❌ absent (no tag) |
| casts | ✅ 5 nodes, node-only | ❌ absent (payload exists, no tag) |
| string ops (upper/like/…) | ✅ node-only | ❌ absent |
| `array_length` | ✅ `builders.mojo:164` | ❌ absent |
| `count()` | `NumericValue` only | ✅ any |
| `variance`/`stddev` ddof | ✅ `[ddof: Int]` | ❌ pinned 0 (documented) |
| `count_star()` | ✅ `builders.mojo:219` | ❌ absent |
| `any`/`all` | ❌ | ❌ (kernels exist, `aggregate.mojo:475,513`) |
| parameters | ✅ `Param[T]` | ❌ documented as absent (`values.mojo:194`) |

**Count is implemented twice.** Comptime `Count[A] = Aggregate[Fold[CountKernel, A.Type], A]`
(`aggregates.mojo:259`) fuses; runtime `"count"` resolves to `ValidCount[A]`
(`runtime/aggregates.mojo:177`). Two kernels for one SQL verb, agreeing only by convention.

**The runtime lane has no frontend.** `python/bindings/*.mojo` contains zero references to
`marrow.expr`; `python/marrow/compile.py:12` records the CLI surface was deleted with the
old package. The lane's stated reason to exist is currently aspirational, which makes its
parity gaps easy to miss.

---

## 4. The aggregate layer

**Is `AggKernel`'s contract minimal and honest? — Yes. Defend this.**
`InArray`/`OutArray`/`dtype(in_dtype)`/`__init__(in_dtype)`/`update(groups, input)`/`finish()`
plus two statics. Six members, each pulling weight. The reasoning at `:826-851` (why `Array`
and not `DynArray`) and `:873-891` (why construction, not `open`) is correct and visible in
the code: `Fold._state` is a plain field, not an `Optional` (`:1022-1028`).

One honesty gap: `grouped()` (`:917`) is a static default nothing overrides and nothing in
production calls — only `kernels/tests/test_agg_kernels.mojo`. Test-only API on a
production trait.

**Is the `Foldable`/`AggKernel` split right? — Yes; `Foldable` has one responsibility.**
The docstring reads like two jobs, but the two are inseparable in use: the fused loop needs
`Agg.Lane.identity/combine` (statics) *and* `agg.scatter/combine_at` (state mutation) in the
same body (`aggregates.mojo:388-524`). One protocol: **"this aggregate can be driven from
registers."**

The one member that does not belong is **`grow`** (`:971`). It is not lane-facing — it is
called once, from `drain` (`aggregates.mojo:528`), purely to seed slots for a zero-row
input. Every other conformer solves that inside its own `finish`. `grow` exists because
`Fold._slots` and `AggState` disagree about who owns the count; delete `_slots` and `grow`
collapses into `finish`.

**Is `dispatch_agg`'s parametric job the right shape? — Yes, and do not spread it further.**
`def dispatch_agg[R, //, Func: def[Agg: AggKernel]() raises -> R](name, in_dtype, func)`
(`runtime/aggregates.mojo:163`) mirrors `DynType.dispatch_*`, is non-generic in itself so it
is one instantiation tree-wide (`:38-46`), and produces a **fully typed** operator — which is
what deleted `DynAgg` and `RuntimeAggregateOperator`. Both callers (`dtype` at `:337`,
`to_operator` at `:386`) take their answer off the same branch, so two-ladder drift is
unrepresentable.

The only tempting further target is `RuntimeValue.evaluate`'s tag switch, and that is bounded
on both sides by measurement already in the repo: a runtime tag switch cost +1,807,168 bytes
(backlog §0), the fn-pointer alternative was miscompiled, and resolving tags into typed nodes
at plan time would instantiate the entire comptime node zoo from the runtime lane — the
4.91 MB configuration `DynValue` exists to avoid. `dispatch_agg` is affordable *because* the
aggregate vocabulary is 10 names and the state must be typed to be fast.

**Does `AggregateOperator`'s two-inline-arm structure still earn itself? — No.**

- `comptime fuses` duplicated (`:128`, `:322`).
- **Two dead fields, one per arm**: the fused arm never reads `_scatters` (`:344`); the
  buffered arm never reads `_num_groups`.
- `G` is a lie in the buffered arm: both `Aggregate.to_operator` (`:214`) and
  `RuntimeAggregate.to_operator` (`:390`) pin `G = ScalarGrouping` while passing
  `grouped=True` — the type says "one slot", the field says "scatter".
- ~110 lines of fused loop and ~10 lines of buffered path share a `push` and nothing else.
- `_state`'s docstring (`:335-337`) still says *"`None` until the first morsel"*; the field
  is `Self.Agg`, not `Optional`.

The docstring at `:288-320` argues against splitting by citing the *node* split
(`FusedAggregate`/`BufferedAggregate`) that made "every fluent method restate the split by
hand". **That argument does not transfer.** Splitting the *operator* leaves `Aggregate` one
node and one fluent surface; `to_operator` already branches on `comptime if Self.fuses`, and
both halves are already boxed in `DynOperator`, so the return type is unchanged.

**A stale docstring is blocking temporal fusion.** `core.mojo:576-584` says temporal
`min`/`max` cannot fuse because *"`FusedAggregateOperator.__init__` builds its accumulator
dtype from `Self.A.Type()` and a `TemporalType` is not `Defaultable`."* No longer true:
`AggregateOperator.__init__` takes `in_dtype: DynType` (`:353`) and does `Self.Agg(in_dtype)`.
The fused arm calls only `bind`, `validity` and `lane[W]` — all `PrimitiveValue` members —
so `conforms_to(Self.A, NumericValue)` in both `fuses` predicates appears one token wider
than necessary. *(Unverified by build — §9.)*

---

## 5. `Grouping` / `ScalarGrouping` / `HashGrouping`

**A trait parameter carrying one comptime `Bool` is not justified here, and the docstring's
aspiration is not fact.**

- `AggregateOperator` reads exactly one member: `Self.G.scatters`, twice
  (`aggregates.mojo:370`, `:378`). It never constructs a `G`, never calls `assign`,
  `num_groups` or `key_columns`.
- **`ScalarGrouping` is never constructed anywhere** — every occurrence is a type argument
  (`aggregates.mojo:204, 214`; `runtime/aggregates.mojo:390`). Its `assign`/`num_groups`/
  `key_columns` (`groupby.mojo:220-229`) are unreachable.
- `HashGrouping` is used as a value, but **concretely**: `GroupByOperator._grouping:
  HashGrouping` (`physical.mojo:596`). No generic function is bound on `Grouping`.
- The aspiration ("window partitions and a sorted or radix placement arrive as
  **conformers**", `groupby.mojo:164-168`) is aspiration: no such conformer exists, and
  `physical.mojo:654` records that parameterising `GroupByOperator` on `Grouping` was
  **measured at +24,432 bytes for no benefit** — the one place a future conformer would plug
  in has already been tried and rejected.

Honest shape today: `comptime scatters: Bool` as a direct parameter of the fused operator.
That deletes `ScalarGrouping`, keeps `HashGrouping` as the concrete grouper `GroupByOperator`
owns, and removes the type-level lie. A future sorted/radix placement will be a *runtime*
strategy chosen inside `GroupByOperator` (which owns the keys), not a comptime parameter of
a fold — which is what the +24,432-byte measurement already says.

---

## 6. Kernel naming and shape consistency

Canonical three-level shape, stated at `numeric.mojo:66-72`:
**`core` (SIMD) → `apply` (typed arrays) → `dispatch` (erased `DynArray`)**.

| family | `core` | typed `apply` | erased `dispatch` | free verb |
|---|---|---|---|---|
| numeric | ✅ | ✅ | ✅ trait default | `equal` |
| boolean | ✅ | ✅ | ✅ | — |
| string | ✅ | ✅ | ✅ | — |
| temporal | ✅ | ✅ | ✅ (`temporal.mojo:164`) | — |
| cast | ✅ | ✅ | ✅ | `cast` |
| nested | — | ✅ | ✅ | — |
| membership | — | ✅ | ✅ | `is_in` |
| filter | — | ✅ | ✅ | `filter`/`take`/`drop_null` |
| sort | — | ✅ | ✅ | `sort_indices`/`sort` |
| hashing | — | ✅ | ✅ | — |
| **conditional** | — | ❌ `apply` is the *erased* one | ❌ named `combine` | `case_when`/`coalesce`/`nullif`/`fill_null` |
| **concat** | — | ✅ | ❌ none | `concat` only, no `Kernel` struct |
| **distinct** | — | — | ❌ none | free fns only, no `Kernel` struct |
| **aggregate** | ✅ `combine` | ✅ `update` | ❌ none | `dispatch_agg_array` (dtype→type) |
| **interval** | — | — | ❌ none | no consumer at all |

Divergences worth fixing, in order:

1. **`conditional.mojo` inverts the vocabulary.** `CoalesceKernel.apply(List[DynArray], ctx)`
   (`:270`) and `NullifKernel.apply(DynArray, DynArray, ctx)` (`:324`) take **erased** arrays
   where `AddKernel.apply[T](PrimitiveArray[T], …)` (`numeric.mojo:92`) takes typed ones. The
   same method name means the opposite thing across two families. The erased entry is
   additionally spelled `combine` (`:253, 263, 320, 378`), a name used nowhere else.
2. **`StringCompareKernel.name: StaticString`** (`expr/comptime/strings.mojo:71`) vs
   `Kernel.name: String` (`kernels/core.mojo:21`), and it does not conform to `Kernel`.
3. **`kernels/__init__.mojo`'s headline example is wrong.** Line 4 advertises
   `mk.SumKernel.dispatch`; `SumKernel = Widening[SumOp]` is a `FoldKernel`, which has no
   `dispatch`, `reduce` or `apply`. `SumKernel`/`ProductKernel`/`MeanKernel`/`CountKernel`
   are re-exported as fold algebras with **no array entry point at all**. The aggregate
   family's actual abstractions (`AggKernel`, `Fold`, `Foldable`, `dispatch_agg_array`) are
   not re-exported.
4. The same docstring references `expr/pruning.mojo` (line 47), which does not exist.

---

## 7. Docstrings contradicted by their own code

Not cosmetic — several are *why* a fix has not been made.

| # | Claim | Code | True |
|---|---|---|---|
| 1 | `runtime/values.mojo:22-26`: *"**A tag never selects a kernel.** … `_eval` is a function pointer bound at construction."* | `evaluate` (`:202`) is a switch on `_tag`. There is no `_eval` field. `:207` says *"A switch on `_tag`, not a per-node function pointer."* | **Code.** Two paragraphs of one file contradict each other. |
| 2 | `kernels/aggregate.mojo:781-786`: *"Its inputs and its output are **erased**… the runtime lane can store as a plain function pointer."* | `comptime InArray: Array` (`:827`), `OutArray: Array` (`:853`), whose docstring is *"**typed on what it consumes and produces**"*. | **Code.** The comment block and the trait 20 lines below say opposite things. |
| 3 | `aggregate.mojo:100-113` and `:84-88`: *"a default whole-array `reduce`"*, *"sum/min/max/product override `reduce` with the SIMD whole-array fast path"* | `FoldKernel` declares `AccType`, `check_domain`, `empty_is_null`, `needs_count`, `identity`, `acc_dtype`, `combine`, `finalize`. **No `reduce`, no `apply`.** | **Code.** |
| 4 | `aggregate.mojo:818`, `:823`, `:917`: *"`DynAgg` below is the one erased face"*, *"`open` / `update` / `finish` are instance methods"* | No `DynAgg`; no `open`. | **Code.** |
| 5 | `aggregate.mojo:569-572`: *"The runtime processor stores its accumulators erased and, per batch, resolves `(K, V)`…"* | No such processor. | **Code.** |
| 6 | `comptime/aggregates.mojo:335-337`: `_state` is *"`None` until the first morsel"* | `var _state: Self.Agg` | **Code.** |
| 7 | `comptime/core.mojo:576-584`: temporal `min`/`max` cannot fuse | `__init__` takes `in_dtype: DynType` | **Code** — and this stale reason is **blocking a real optimisation**. |
| 8 | `runtime/values.mojo:78-80`: *"Satisfies `Value` — `Analyzable & Executable & Writable & Copyable & Deinitable`"* | `Value` is one trait; `Analyzable`/`Executable` do not exist; the struct declares `(Evaluable, Movable, Value)`. | **Code.** |
| 9 | `physical.mojo:96-101`: `is_scalar()` — *"The planner asks…"* | Only caller is `struct_array()`'s own guard. | **Code.** |
| 10 | `physical.mojo:951`: `_build_schema` — *"the first `len(left_keys)`-agnostic prefix of the output schema"* | `return self._schema.copy()` | **Neither.** Docstring describes nothing; code is wrong for a multi-batch build side. |
| 11 | 12 references to `FusedAggregateOperator`/`BufferedAggregateOperator`/`RuntimeAggregateOperator`/`AggregateFn` (`physical.mojo:307,563,565,569,656`; `logical.mojo:293`; `comptime/aggregates.mojo:13,19,189,190,193`; `runtime/aggregates.mojo:29,33,36,254`; `comptime/core.mojo:107`) | None exists. | **Code.** |
| 12 | CLAUDE.md: *"consumers import from `marrow.expr`, which re-exports. Only the boundary crossing escapes"* | `marrow/expr/__init__.mojo` is **0 bytes**; `comptime/__init__.mojo` re-exports a stale 13-name subset nothing imports; **35 backticked `` `comptime` `` import sites** across `golden/`, `benchmarks/`, `marrow/`. | **Code.** |
| 13 | `runtime/tests/test_aggregates.mojo:207-260` describes `node.resolve(...)` and an int64-probing implementation | `RuntimeAggregate.empty()` is a four-arm switch on `_name`. | **Code.** Test passes; it tests a method with no production caller. |

---

## 8. Recommended target design

Sequenced so each step compiles on its own. Net: deletes 2 types, 1 trait, 1 module,
2 fields and ~14 stale claims; adds one module and one trait member.

### Step 1 — extract `expr/core.mojo` (a true leaf). Removes 3 of 8 cycles.

Move unchanged: `Shape`, `Value`, `DynValue`, `merged` (from `logical.mojo`) and `Bindings`
(from `params.mojo`). Move `Param[T]` into `comptime/leaves.mojo`, where it belongs — it is a
`NumericValue` leaf and the only reason `params.mojo` imports `comptime.core`. Delete
`params.mojo`.

```
core        (Shape, Value, DynValue, Bindings, merged)   [leaf]
physical    -> core, kernels, parquet
comptime.*  -> core, physical, kernels                   [SCC of 5, intrinsic]
runtime.*   -> core, physical, comptime.aggregation, kernels  [SCC of 2, intrinsic]
logical     -> core, physical, runtime.values, kernels
builders    -> all
```

Fill `expr/__init__.mojo` with the re-export CLAUDE.md already claims exists, collapsing the
35 backtick sites to one.

### Step 2 — split `AggregateOperator`; delete `ScalarGrouping` and `Grouping`

New lane-agnostic module `expr/aggregation.mojo`, between `physical` and the two lanes:

```mojo
struct FusedAggregateOperator[Agg: Foldable, A: PrimitiveValue, scatters: Bool](Operator):
    var _input: Self.A
    var _bindings: Bindings
    var _state: Self.Agg
    var _num_groups: Int
    var _emitted: Bool

struct BufferedAggregateOperator[Agg: AggKernel, A: Evaluable & Value](Operator):
    var _input: Self.A
    var _bindings: Bindings
    var _state: Self.Agg
    var _scatters: Bool
    var _emitted: Bool
```

`Aggregate.to_operator` keeps its `comptime if fuses` and picks; `RuntimeAggregate.to_operator`
always picks the buffered one.

Deleted: the duplicated `fuses`, `ScalarGrouping`, `trait Grouping`, `_scatters` from the
fused path, `_num_groups` from the buffered path, and the type-level lie. `HashGrouping`
stays, unparameterised, owned concretely by `GroupByOperator`.

`Aggregate.fuses` stays as the single definition and widens `NumericValue` → `PrimitiveValue`
so temporal `min`/`max` fuses. Fix `core.mojo:576-584` accordingly.

*Fallback if binding `Self.Agg` at the tighter `Foldable` bound at a struct-instantiation
site does not typecheck (§9): keep one struct, deduplicate `fuses` into a single
`comptime fuses_over[Agg, A]`, and drop the two dead fields. That alone removes the worst.*

### Step 3 — one owner for the group count

Delete `Fold._slots` and its four `max` sites. `AggState.finish(num_groups)` →
`AggState.finish()` reading `self.acc.length()`. `Fold.finish` becomes
`return self._state.finish()`. `Foldable.grow` then has no caller but `drain`'s empty-input
seed, which becomes `AggKernel.reserve(slots)` — a one-line `pass` default overridden by the
five conformers that already have a private `_grow`. That also unifies the three different
`_grow` length conventions.

Net: `Foldable` loses `grow` and becomes exactly "drivable from registers" — `Lane`, `Acc`,
`scatter`, `combine_at`. `AggKernel` gains `reserve`.

While here, close backlog **AG-1** and **AG-2**, the same defect one level down: give
`Groups` an explicit `single: Bool` instead of inferring it, and make `Datum.to_array(n)`
check the array's length against `n`.

### Step 4 — one catalog of aggregate names

Move `SUM`…`MAX` and `vocabulary()` into `kernels/aggregate.mojo`, beside the `name`
constants they duplicate. `dispatch_agg` stays in `expr` — it maps a name onto a kernel,
which is the expression layer's job. Delete `RuntimeAggregate.empty()` (no production caller)
and the test that exercises it, **or** wire `GroupByOperator.drain` to it — one or the other,
not neither.

### Step 5 — move the string-comparison kernels out of `expr`

`StringCompareKernel` + `StrEq/Ne/Lt/Gt` → `kernels/string.mojo`, conforming to `Kernel`,
`name: String`; add `Le`/`Ge` so `StringValue` can have all six. Delete
`kernels/interval.mojo` (zero consumers) or state in its docstring that pruning is coming back.

### Step 6 — close the fluent surface, then reinstate the parity test

Add methods for the 18 node types `golden/prelude.mojo` imports as types, and the
corresponding runtime tags. Then restore backlog §0 invariant 2's test — the table in §3 is
its first failing case list.

### Step 7 — docstring sweep

The 13 contradictions in §7 plus the 12 dangling type references. Item 7 matters most (it
blocks a real optimisation); item 3 most misleads a new reader of the kernel layer.

### Already right — leave alone

`Evaluable` as the shared "can run" seam; `DynValue` conforming to `Value` and nothing else;
`Unnamed`/`ColumnBound`; `PrimitiveValue` split from `NumericValue`; `Aggregate` as **one**
node; `dispatch_agg`'s parametric job; the `Relation`-pure / `Operator`-mutable split;
`Operator`'s two-method interface with `Datum` output; `kernels/` as a verified DAG;
`TemporalCompare` separate from `NumericCompare` (three merge attempts measured and recorded).

---

## 9. Not verified

- No edits; no build beyond `precompile` (exit 0, clean). Every claim is from reading the
  code as it stands on disk.
- **Temporal-fusion widening** (`NumericValue` → `PrimitiveValue` in `fuses`) is indicated by
  the member set the fused arm uses, but not compile-tested.
- **Step 2's viability** rests on whether Mojo accepts binding a parameter at a *tighter*
  trait bound at a struct-instantiation site inside `comptime if conforms_to(...)`. The
  current code proves it for *method calls* on a field (`aggregates.mojo:388-524`); it does
  not prove it for a struct's parameter list. This is the one step that could fail; the
  fallback is stated.
- **`Fold._slots == AggState.num_groups()`** is derived from all four write sites and
  `_grow`'s monotonicity, not from instrumenting a run.
- **The 14.6x / 1.17-1.68x fusion numbers** are quoted from docstrings dated 2026-08-22 and
  predate the `FusedAggregate`/`BufferedAggregate` merge. Not re-measured; nothing above
  depends on their exact value, only on fusion being worth keeping.
- **Binary size**: `pixi run binary_size` was not run. Steps 1, 3, 4, 5, 7 are code motion and
  deletion, so size-neutral or better. Step 2 replaces one two-armed instantiation with two
  one-armed ones — a plausible small win, but must be gated on `query_expr2_agg_fused`.
  Step 6 adds surface; a fluent method is a trait default, which backlog §0 measured at
  **+0 bytes** on `query_streaming_agg_fused`, so the risk is low but not zero.
- **`JoinOperator._build_schema`** looks wrong for a multi-batch build side; no failing case
  was constructed. Out of scope here, but it should not be left as it is.
