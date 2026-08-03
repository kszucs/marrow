# Window functions — design (unimplemented)

**Nothing in the target design below is shipped.** What exists in the tree is a
two-node toy; §1 states exactly what that is, and §2 onward is a forward spec.
`docs/backlog.md` M2.3 owns the sequencing and is authoritative — it is not
restated here. The execution model this design plugs into (`Value`/`Breaker`
polarity, `Datum`, `OutShape`, `Context` staging, fuse-above-breaker) is
described in `docs/architecture.md` §3.

Grounded in ibis (`func: Analytic | Reduction`), DataFusion (`WindowExpr`,
`WindowAggExec`) and Polars (`Expr::Over`).

---

## 1. Current state — a two-node toy

Everything window-related in the tree is `marrow/expr/values.mojo:1975-2039`,
plus two cases in `marrow/expr/tests/test_values.mojo`. Verified at `265df9b`:

- **`WindowSpec` (`:1981`) carries frame bounds only** — `start` and `end`, and
  its own docstring says "the toy carries frame bounds only". There is **no
  `partition_by`, no `order_by`, and no `how` (ROWS/RANGE)**.
- **`FrameBound.kind` (`:1976`) is an untyped `UInt8` that is never read** by any
  code path. Neither is `offset`.
- **`trait WindowKernel` (`:1988`) has one method**, `evaluate_all(values:
  DynArray) -> DynArray`. There is no per-row `evaluate`, no
  `comptime frame_dependent`, and it takes no keys, no orders and no spec.
- **`RowNumberKernel` (`:1997`) is the only implementation**, and it **ignores
  its `values` argument** entirely — it returns `1..n` built from `len(values)`.
- **There is no `marrow/kernels/window.mojo`.** The kernel lives in the
  expression layer.
- **There is no `.over()`** anywhere in the tree — the string occurs once, in a
  docstring (`:2013`). A window is built by naming the node:
  `RowNumber(col("a", int64), spec)` (`comptime RowNumber` at `:2039`).
- **Nothing outside `values.mojo` and `test_values.mojo` references any of it** —
  not `relations.mojo`, not `execution.mojo`, not the Python bindings.
- **It exists only in the AOT lane**, violating the standing "one engine, two
  drivers" invariant that no feature may live in a single lane
  (`docs/architecture.md`, invariant 2).

What *is* right about the toy, and worth keeping: `WindowFunction`
(`values.mojo:2012`) is a `Breaker` with `OutShape == 1`; its `materialize`
(`:2024`) stages the whole output column into the `Context` and its `vectorwise`
(`:2028-2036`) loads that column per lane. So **arithmetic already fuses above a
window** — `row_number() + 1` is one fused pass over a staged column, not two
eager applies (`test_arithmetic_above_window_materializes`,
`marrow/expr/tests/test_values.mojo:226`).

## 2. Target design

The columnar cross-row quadrant. `WindowFunction` stays a `Breaker` (func + spec,
`OutShape == 1`) and reuses the reduction kernel.

```mojo
struct FrameBound:                       # DF WindowFrameBound / ibis WindowBoundary
    var kind: FrameBoundKind             # UNBOUNDED_PRECEDING | PRECEDING | CURRENT
                                         # | FOLLOWING | UNBOUNDED_FOLLOWING
    var offset: Int64

struct WindowSpec:
    var how: FrameMode                   # ROWS | RANGE
    var start: FrameBound
    var end: FrameBound
    var partition_by: List[BoxedValue]   # boxed key sub-exprs
    var order_by: List[BoxedValue]       # (+ asc flags)

struct WindowFunction[Func: WindowKernel, A: Value](Breaker, NumericValue):
    comptime OutType = Func.OutType
    comptime OutShape = 1
    var a: Self.A
    var spec: WindowSpec

    def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
        var v = into_array(self.a.execute(batch), batch.num_rows())
        var keys = [k.execute(batch) for k in self.spec.partition_by]
        var orders = [k.execute(batch) for k in self.spec.order_by]
        return Datum(Self.Func.evaluate_all(v, keys, orders, self.spec))
```

Two corrections against the toy are load-bearing:

1. **`kind` and `how` are typed value types, not raw `UInt8`** — the same shape
   of fix as `docs/backlog.md` Q4.1. The toy's untyped `kind` is precisely why it
   is never read: nothing can mean anything by it.
2. **The spec's key sub-expressions are `BoxedValue`** — the one box both lanes
   erase into (`relations.mojo:155`) — not `DynValue`, which names the runtime
   *node*. Boxing the keys is what lets the same `WindowSpec` serve both lanes.

`WindowKernel` needs DataFusion's two eval modes:

```mojo
trait WindowKernel(Kernel):
    comptime frame_dependent: Bool

    @staticmethod
    def evaluate_all(values, keys, orders, spec) raises -> DynArray: ...  # frame-independent

    @staticmethod
    def evaluate(values, range) raises -> DynScalar: ...                 # per-row frame
```

Note it should conform to `trait Kernel` (`kernels/core.mojo:16`) and live in
`marrow/kernels/window.mojo`, like every other kernel family — the toy's trait
conforms to nothing and sits in the expression layer.

Execution, per DataFusion `WindowAggExec` and Polars `.over()`: evaluate keys and
argument → partition (reuse `groupby` hashing) → sort each partition (reuse
`sort`) → frame-fold / rank / shift → **scatter back to original row order**.
None of the four reused pieces needs new kernel machinery.

### Kernel families

- **Ranking** (order only, `evaluate_all`): `RowNumber`, `MinRank`, `DenseRank`,
  `PercentRank`, `CumeDist`, `NTile`.
- **Navigation** (`evaluate_all`): `Lag`, `Lead`, `NthValue`.
- **Running / rolling aggregates**: `RunningAgg[K: AggKernel]`, reusing the
  existing `AggKernel`. *Cumulative* (`UNBOUNDED PRECEDING → CURRENT`) is
  update-only (`combine`); *sliding* needs a `retract` that `AggKernel` does not
  have today, or it degrades to `O(n·w)` recompute. The pairing is exact:
  `col.sum()` → scalar `Reduction[SumKernel, _]`; `col.sum().over(w)` → columnar
  `WindowFunction[RunningAgg[SumKernel], _]`.

### API

`.over(win)` on a reduction or analytic, with
`win = window(partition_by=[...], order_by=[...], rows=(-2, 0))` — mirroring
ibis and PyArrow. **Both lanes**, per the one-engine-two-drivers invariant: the
runtime lane needs the same builders and a `DynValue` evaluator, not just the
AOT nodes.

## 3. Deferred within this spec

- **Batch-local partitions** in v1 (each batch is the full partition set, sorted
  within). Cross-batch and streaming windows (cf. DataFusion
  `BoundedWindowAggExec`) are a plan-layer concern.
- **Sliding frames** need `AggKernel.retract`; do cumulative first.
- **Constant frame bounds** only (`rows=(-2, 0)`); dynamic (`Value`) bounds
  later.
- **`ROWS` first.** `RANGE` and `GROUPS` reuse the same driver with a different
  range calculation.
- **CSE / value-numbering** to hoist repeated expensive subexpressions is an
  optimization, not a correctness requirement; it is tracked in
  `docs/design-expression-evaluation.md`.
- **`OutShape` for the bool and string families** (scalar string/bool, `any`,
  `all`) applies the same idea and is out of scope here.
