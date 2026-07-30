# Specification — expression execution: fusion, shape, reductions & windows

Authoritative spec for `marrow.expr.values` execution. Defines the target model,
the invariants, the node/kernel contracts, the window design, and the work plan to
get there from the current implementation. Grounded in ibis (`expr/datashape.py`),
DataFusion (`ColumnarValue`, `WindowExpr`) and Polars (`Column::Scalar`, `Expr::Over`).

**Validated by a standalone prototype** (compile & run: `mojo run
lane-shape-window-skeleton.mojo`):
[`lane-shape-window-skeleton.mojo`](lane-shape-window-skeleton.mojo) exercises the
whole hierarchy — the `Fusable`/`Value` trait split, `comptime Shape` + `max`
derivation, fused `core[W]`, `ColumnarValue = Scalar | Array` as the uniform
`execute` result (which also sidesteps the `-> Self.ArrayType` associated-type
recursion), `DynValue` erasure returning it, a dynamic interpreter sharing the same
kernels, and a `WindowFunction` carrying a runtime `WindowSpec` with `List[DynValue]`
keys — with no `ExecCtx`, no cursor, no per-node state.

## 1. Goals & invariants (normative)

1. **AOT-fused execution is the differentiator.** A comptime-typed fusable
   expression monomorphizes to a single straight-line SIMD loop: `core[W]` inlines
   across the whole subtree, no dispatch. Preserve this for every node that fuses.
2. **Dynamic dispatch reaches parity.** The Python bindings and any runtime-built
   query run through the `TagValue` interpreter with eager, type-erased kernels —
   like DataFusion/Polars. Same results, no fusion.
3. **Kernels are the shared substrate; only the driver differs.** Every kernel
   exposes `core[W]` (fused) **and** `apply` (eager). A feature is implemented once
   and both drivers get it. No feature may live in only one mode.
4. **Fusability is a trait, not a flag.** A node is either **`Fusable`** — implements
   `core[W]` and fuses — or a plain **`Value`** — implements `execute` and
   materializes. The fuse/materialize boundary *is* the trait line, resolved at
   comptime by node type. No `fusable` boolean to maintain or propagate.
5. **Shape is first-class.** Every value is *scalar* or *columnar* (ibis `DataShape`)
   — `comptime Shape` on nodes, the runtime tag inside `ColumnarValue`. Scalars
   broadcast by **splat**, never by eagerly building an N-row array.
6. **No execution state.** `execute` returns a `ColumnarValue`; a cross-row node
   *computes and returns* it, storing nothing. No `ExecCtx`, no cursor, no per-node
   cache — hence no cross-batch staleness by construction.

## 2. The model — one grid

Two axes: **shape** (one element vs N) × **locality** (computable per-lane from
batch data at/near `idx`, vs needs other rows). Locality is the `Fusable`/`Value`
line; shape is `comptime Shape`.

|                       | **fusable** — implements `core`, fuses            | **cross-row** — implements `execute`, materializes |
| --------------------- | -------------------------------------------------- | -------------------------------------------------- |
| **scalar** (ndim 0)   | `lit` → `core` splats the constant                 | **`Reduce`** → returns a **scalar** `ColumnarValue` |
| **columnar** (ndim 1) | col, arith, cast, cmp, **len / parse / bool→num**  | **`WindowFunction`** → returns an **array** `ColumnarValue` |

- **Broadcast = splat.** A scalar `core[W](idx)` returns `SIMD(value)`;
  `scalar ⊕ columnar → columnar` (`Shape = max(l, r)`), the scalar splats in the loop.
  In the eager path the scalar `ColumnarValue` broadcasts lazily (`into_array`).
- **Combining a cross-row operand** with more arithmetic is a **`MatBinary`** (a
  `Value`): its `execute` eagerly folds children's `ColumnarValue`s with the kernel's
  `apply`. So `sum(x) + col`, `rank().over(w) + 1` are eager one-`apply` ops.
- **No fuse-*above* a cross-row node** — the deliberate tradeoff for dropping all the
  ctx/cursor machinery. A single op above a reduction/window is one eager `apply`,
  the *same* O(n) cost a fused pass would have (scalar broadcast / array read).
  Only *chains* directly above a cross-row node (`rank(a) + b + c`) do several eager
  passes instead of one fused pass — rare, and cheap. Crucially, fuse-above for
  *fusable* boundaries is unaffected: `len` / `parse` / `bool→num` are `Fusable`
  (they have `core`), so `(len(a) + len(b)) + 1` is all-fusable and fuses fully.

## 3. Two drivers

| concept | AOT-fused (comptime) | dynamic (`TagValue`) |
| --- | --- | --- |
| result type | `ColumnarValue` | `ColumnarValue` |
| shape | `comptime Shape ∈ {0,1}` | runtime tag inside `ColumnarValue` |
| broadcast | scalar `core` splats | scalar stays length-1; `into_array(n)` only when forced |
| fusable op | `core[W]` inlined into one loop | `K.apply` eager |
| reduction | `Value.execute` → scalar `ColumnarValue` | `AggKernel.reduce` → scalar `ColumnarValue` |
| window | `Value.execute` → array `ColumnarValue` | `WindowKernel.apply` → array `ColumnarValue` |

Prior art confirms the shape split: DataFusion keeps `ColumnarValue::Scalar` lazy
(`(Scalar,Scalar)→Scalar`; `into_array` only when forced); Polars makes a repeated
scalar a first-class `Column::Scalar{value,length}`; both reuse the aggregate
accumulator for window running-aggs. marrow carries shape at **comptime** on the
fused path and as the `ColumnarValue` tag on the dynamic path.

## 4. Node contract

```mojo
trait Value(Copyable, ...):
    comptime OutType: NumericType
    comptime Shape: Int                          # 0 scalar, 1 columnar
    def execute(self, batch) raises -> ColumnarValue: ...    # every node

trait Fusable(Value):
    comptime NativeType: DType
    def core[W](self, batch, idx) -> SIMD[Self.NativeType, W]: ...
    # inherits a fused `execute` default — see §5. Safe to default the parent's
    # abstract `execute` because it returns a concrete `ColumnarValue`, not
    # `Self.ArrayType` (no associated-type recursion — the worst real-code gotcha).
```

`ColumnarValue = Scalar(value) | Array(buffer)`: `load[W](idx)` hides splat-vs-load;
`into_array(n)` is the lazy broadcast; carries its own `is_scalar`/`num_rows`.

Shape derivation (ibis): column → 1, literal → 0, len/parse/bool→num → 1,
unary → `arg.Shape`, binary → `max(L.Shape, R.Shape)`, reduction → 0, window → 1.

Per-node behaviour:

| node | trait | Shape | mechanism |
| --- | --- | --- | --- |
| `NumericColumn` | Fusable | 1 | `core` reads the batch column |
| `NumericLiteral` | Fusable | 0 | `core` splats the constant |
| arith / `Cast` / cmp (fusable operands) | Fusable | max(children) | `core` fuses children |
| `StringLength`, `Counting` | Fusable | 1 | `core` = `offsets[idx+1]-offsets[idx]` (vectorized) |
| `StringToNum` | Fusable | 1 | `core` = scalar `atol` per lane element, packed |
| `BoolToNum` | Fusable | 1 | `core` = `arg.core.cast[native]()` |
| `Reduce`, `Count` | Value | 0 | `execute` folds arg via `AggKernel.reduce` → scalar CV |
| `WindowFunction` | Value | 1 | `execute` = partition/sort/frame → array CV |
| `MatBinary[K,L,R]` | Value | max | `execute` = `K.apply(l.execute, r.execute)` |

## 5. Execution

```mojo
# Fusable.execute — default, one fused pass (or eval-once for a scalar)
def execute(self, batch) raises -> ColumnarValue:
    comptime if Self.Shape == 0:
        return ColumnarValue.scalar(self.core[1](batch, 0)[0])   # no loop
    else:
        # single W-blocked vectorized pass; self.core[W] inlines the whole subtree
        ... fill buffer via self.core[W](batch, i) ... -> ColumnarValue.columnar

# Value.execute — cross-row node: compute & return, store nothing
Reduce.execute:  AggKernel.reduce(self.arg.execute(batch)) -> scalar CV
WindowFunction.execute: partition/sort/frame(self.arg.execute(batch), keys) -> array CV
MatBinary.execute: K.apply(self.l.execute(batch), self.r.execute(batch))    # eager

# dynamic driver
TagValue.execute(batch) -> ColumnarValue     # node-by-node eager kernels; scalars lazy
```

**Operator dispatch** picks fused-vs-materialized by operand trait, via overload:
`Fusable.__add__[R: Fusable](o) -> Add[Self, R]` (fused) and
`Value.__add__[R: Value](o) -> MatBinary[AddK, Self, R]` (eager). `fusable ⊕ fusable`
resolves to the more-specific `Fusable` overload; anything touching a cross-row node
falls to `MatBinary`. (Verify Mojo's most-specific-overload resolution holds here.)

`eval_scalar` is just `Fusable.execute` at `Shape == 0` (or a `Value` scalar node's
`execute`): folds an all-scalar tree once (`lit(1)+lit(2)→3`; `sum(a)` → one
reduction) and returns a scalar `ColumnarValue` — never a length-1 array.

## 6. Kernels

Shared substrate. Fusable kernels already provide `core[W]` + `apply`
(arithmetic/compare/boolean). Add cores where missing (§4). New:

- `AggKernel` (exists): `reduce(array) -> scalar` reused by reductions **and** window
  running-aggs. Needs a `retract` to support true sliding frames (else recompute).
- `WindowKernel` (`marrow/kernels/window.mojo`), two eval modes (DataFusion):
  ```mojo
  trait WindowKernel(Kernel):
      comptime frame_dependent: Bool
      def evaluate_all(values, keys, orders, spec) raises -> PrimitiveArray[Out]: ...  # frame-independent
      def evaluate(values, range) raises -> Scalar[Out]: ...                            # per-row frame
  ```

## 7. Windows

The columnar cross-row quadrant. `WindowFunction` is a **`Value`** node (func + spec,
`Shape == 1`); it reuses the reduction kernel (ibis: `func: Analytic | Reduction`).
No `core`, no `prepare`, no `ctx` — its `execute` materializes and returns an array
`ColumnarValue`; anything above it is a `MatBinary`.

```mojo
struct FrameBound:                       # DF WindowFrameBound / ibis WindowBoundary
    var kind: UInt8                      # UNBOUNDED_PRECEDING | PRECEDING | CURRENT | FOLLOWING | UNBOUNDED_FOLLOWING
    var offset: Int64

struct WindowSpec:
    var how: UInt8                       # ROWS | RANGE
    var start: FrameBound
    var end: FrameBound
    var partition_by: List[DynValue]     # erased key sub-exprs
    var order_by: List[DynValue]         # (+ asc flags)

struct WindowFunction[Func: WindowKernel, A: Value](Value):
    comptime OutType = Func.OutType[A.OutType]
    comptime Shape   = 1
    var arg: A
    var spec: WindowSpec

    def execute(self, batch) raises -> ColumnarValue:
        var v      = self.arg.execute(batch).into_array(batch.num_rows())
        var keys   = [k.execute(batch) for k in self.spec.partition_by]
        var orders = [k.execute(batch) for k in self.spec.order_by]
        return ColumnarValue.columnar(Func.apply(v, keys, orders, self.spec))
```

Execution (per DataFusion `WindowAggExec` / Polars `.over()`): evaluate keys/arg →
partition (reuse `groupby` hashing) → sort each partition (reuse `sort`) →
frame-fold / rank / shift → **scatter back to original row order**. Spec keys are
erased sub-expressions materialized here; `Func` is the only comptime param.

Kernel families:
- **Ranking** (order only, `evaluate_all`): `RowNumber`, `MinRank`, `DenseRank`,
  `PercentRank`, `CumeDist`, `NTile`.
- **Navigation** (`evaluate_all`): `Lag`, `Lead`, `NthValue`.
- **Running/rolling aggregates**: `RunningAgg[K: AggKernel]` — reuse `AggKernel`.
  *Cumulative* (`UNBOUNDED PRECEDING → CURRENT`) = update-only (`combine`); *sliding*
  = needs `retract` (else `O(n·w)` recompute). `col.sum()` → scalar `Reduce[SumKernel]`;
  `col.sum().over(w)` → columnar `WindowFunction[RunningAgg[SumKernel]]`.

API: `.over(win)` on a reduction/analytic; `win = window(partition_by=[...],
order_by=[...], rows=(-2, 0))` — mirrors ibis/PyArrow.

## 8. Current state → target (the refactoring)

Current (committed): cross-family casts landed; the numeric lane carries a
`prepare`/per-node `_cache` on `StringLength`, `Counting`, `StringToNum`, `BoolToNum`
(over-classified as boundaries), and `Reduce`/`Count` are `Value` nodes returning
length-1 arrays via `.repeat(1)`. A short-lived `fusable` flag + materialize-fallback
was already removed.

Target changes:
1. **Split the trait**: `Value` (`execute -> ColumnarValue`) and `Fusable(Value)`
   (`core[W]` + fused `execute` default). Introduce `ColumnarValue = Scalar | Array`.
2. **Reclassify len/parse/bool→num as `Fusable`** with a real `core` (§4); **delete
   their `_cache` and `prepare`**. They rejoin the fused loop.
3. **Add `comptime Shape`** and the scalar-eval-once branch in `Fusable.execute`.
4. **Reductions become scalar `Value` nodes** (`Shape=0`): `execute` folds via
   `AggKernel.reduce` → scalar `ColumnarValue`; drop `.repeat(1)`. Add `MatBinary`
   for mixed arithmetic; wire operator dispatch (fusable→fused, else `MatBinary`).
5. **Add windows** (§7) as `Value` nodes whose `execute` materializes.
6. **Dynamic parity throughout**: `TagValue.execute -> ColumnarValue` with lazy
   broadcast, reductions/windows via the eager kernels.

Net: no `ExecCtx`, no cursor, no `prepare`, no per-node `_cache`, no flag — the
boundary machinery collapses to (a) `core` on fusable nodes and (b) `execute`
returning a `ColumnarValue` on cross-row nodes.

## 9. Work plan (each step lands both drivers)

1. **Trait split + `ColumnarValue` + Shape.** `Value`/`Fusable`, `ColumnarValue`,
   `comptime Shape`, `Fusable.execute` default (fused loop / eval-once), literal
   folding. Dynamic: `TagValue.execute -> ColumnarValue` + lazy broadcast.
   *Done when:* `col+lit` fuses, `lit+lit` folds to a scalar, both drivers agree;
   existing tests green.
2. **Fusable cores.** `StringLength`/`Counting` offset-subtract (easy, vectorized),
   `BoolToNum`, `StringToNum`; delete their `_cache`/`prepare`.
   *Done when:* `col.length() + 1` and `str.cast(int)*2` fuse; no `_cache` on them.
3. **Reductions + `MatBinary` + operator dispatch.** `Reduce`/`Count` scalar `Value`
   nodes; `MatBinary[K,L,R]`; `+`/etc. return `Add` for fusable operands, `MatBinary`
   otherwise; drop `.repeat(1)`.
   *Done when:* `x / sum(x)` runs (scalar broadcasts) in both drivers.
4. **Windows.** `WindowFunction` + `WindowSpec`; kernels starting with `RowNumber`
   and `RunningAgg[Sum]` (cumulative), then the full family; `.over()` API; `TagValue`
   window handling.
   *Done when:* `rank().over(w)` and `sum(x).over(w) + col` run in both drivers.

## 10. Deferred / open

- **Fuse-above cross-row nodes** — chains above a reduction/window run as several
  eager `apply`s. Recoverable later by wrapping a materialized `ColumnarValue` as a
  fusable leaf (`ColumnarValue` already has a `core`-shaped `load[W]`), *if* profiling
  ever justifies it. Not needed now.
- **Operator-overload dispatch** (fusable→`Add`, else→`MatBinary`). *Confirmed
  during implementation:* defining `+`/`*` on **both** `Value` and `Fusable` traits
  trips Mojo trait-elaboration — a param-name clash (rename `R`→`Rhs`), an `execute`
  recursion, and the `Fusable.execute` **default** ceasing to satisfy `Value`'s
  abstract `execute` for conforming structs. Fix path: stop *re-defaulting* `execute`
  on `Fusable` (the documented gotcha) — keep it abstract on `Value`, have each
  fusable node override with a one-liner delegating to a `_fused(self, batch)` free
  function — then the operator overloads can be added. Until then, use the explicit
  builders (`Add`, `MatAdd`, …), which work.
- **Batch-local partitions** in v1 (each batch = full partition set, sorted within);
  cross-batch/streaming windows (cf. DataFusion `BoundedWindowAggExec`) are a
  plan-layer concern.
- **Sliding frames** need `AggKernel.retract`; cumulative first.
- **Constant frame bounds** only (`rows=(-2,0)`); dynamic (`Value`) bounds later.
- **`ROWS` first**; `RANGE`/`GROUPS` reuse the same driver with a different range calc.
- **CSE / value-numbering** (content-addressed hashing to hoist repeated expensive
  subexpressions) is an *optional optimization*, not required for correctness.
- **`Shape` for bool/string families** (scalar string/bool, `any`/`all`) applies the
  same idea; out of scope here.
