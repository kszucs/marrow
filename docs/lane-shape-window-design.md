# Expression execution — fusion, shape, staging & windows

How `marrow.expr.values` executes: the trait polarity, the shape algebra, the
staging model, and the node/kernel contracts. §1–§6 describe **what is in the
tree at `b2e7dae` (2026-08-03)**, verified against the code, not a plan. §7 is a
**forward spec for window functions and is unimplemented** — it feeds
`docs/backlog.md` M2.3.

Grounded in ibis (`expr/datashape.py`), DataFusion (`ColumnarValue`,
`WindowExpr`) and Polars (`Column::Scalar`, `Expr::Over`).

> **History.** An earlier revision of this file specified a `Fusable`/`Value`
> trait split, a `MatBinary` node, a `ColumnarValue` struct, and "no execution
> state". Three of those four were rejected during implementation and the fourth
> shipped in a different shape; the polarity was inverted outright. The rejected
> designs were prototyped in `lane-shape-window-skeleton.mojo`, which is
> **deleted** — a runnable artifact of a rejected design is a live trap. What
> follows is the surviving design, corrected. §7 alone remains a proposal.

## 1. Goals & invariants (normative)

1. **AOT-fused execution is the differentiator.** A comptime-typed expression
   monomorphizes to a single straight-line SIMD loop: `vectorwise[W]` inlines
   across the whole subtree, no dispatch. Preserve this for every node that fuses.
2. **The runtime lane reaches parity.** `marrow.expr.dynamic`'s `DynValue`
   (`dynamic.mojo:236`) is the runtime *node* — its children, an optional payload
   and a pointer to its evaluator. What stays runtime is the *dtype* of the
   operands, not the operation. No feature may live in only one lane.
3. **Kernels are the shared substrate; only the driver differs.** A fused kernel
   exposes `core[W]` and `apply`; the fused node calls `core[W]` inside the loop,
   the runtime node calls the kernel eagerly. One implementation, two drivers.
4. **Fusion is the default; *breaking* is what a trait marks.** This is the
   inverse of the polarity originally specified here. Every node implements
   `materialize`; a node that cannot be evaluated a lane at a time additionally
   conforms to **`Breaker`** (`values.mojo:417`), an **empty marker trait** that
   adds no method. Its own docstring (`:421-423`) gives the rationale:
   "Conformance *is* the marker. It replaces a `comptime IsBreaker: Bool` that
   every node had to set by hand, the same hazard `IsErased` posed before it was
   deleted." Marking the minority means a new node fuses unless it says otherwise,
   and there is no boolean to propagate.
5. **Shape is first-class.** Every value is *scalar* or *columnar* (ibis
   `DataShape`) — `comptime OutShape: Int` on nodes (`values.mojo:306`), the
   variant tag inside `Datum` at run time. Scalars broadcast by **splat**, never
   by eagerly building an N-row array.
6. **Breakers stage into a `Context`; nodes stay immutable.** The original §1.6
   ("no execution state, no `prepare`") was **reversed**. `Context`
   (`values.mojo:215-235`) holds a positional `List[Datum]`, and `Value.prepare`
   (`:334`) fills it in DFS order. Results live in the `Context`, *not* on the
   nodes — which is precisely what keeps an expression immutable and re-runnable
   across batches, the property §1.6 wanted and tried to buy by having no state
   at all. There is no `ExecCtx` and no cursor.

## 2. The model — one grid

Two axes: **shape** (one element vs N) × **locality** (computable per-lane from
batch data at/near `idx`, vs needs other rows). Locality is the `Breaker` line;
shape is `comptime OutShape`.

|                       | **fuses** — `vectorwise[W]` runs in the loop        | **breaks** — conforms to `Breaker`, stages into `Context` |
| --------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| **scalar** (ndim 0)   | `lit` → `vectorwise` splats the constant             | **`Reduction`** (`:1928`) → a `DynScalar` `Datum`           |
| **columnar** (ndim 1) | col, arith, cast, cmp, bool logic, temporal extract  | `StringToNum`, `StringToBool`, `StringPredicate`, `ListLength`, `ConditionalBinary`, `CaseWhen`, `WindowFunction` |

- **Broadcast = splat.** A scalar node's `vectorwise[W](…)` returns
  `SIMD(value)`; `scalar ⊕ columnar → columnar` (`OutShape = max(l, r)`), and the
  scalar splats inside the loop. At the `Datum` boundary the scalar stays a
  `DynScalar` until `into_array` (`values.mojo:201`) forces it.
- **Fusion continues *above* a breaker.** The original §2 ruled this out and
  proposed `MatBinary` — an eager node for "arithmetic over a cross-row operand".
  It was rejected and **`MatBinary` was never needed**. A breaker's `vectorwise`
  reads its own staged slot out of the `Context` and returns a lane, so to its
  parent it is indistinguishable from a column leaf. `NumericBinary`'s docstring
  (`values.mojo:667-670`) states it: *"There is no 'materialized' counterpart: a
  breaker operand is itself a fused leaf (it reads its stage result from `ctx`),
  so it composes here like any column/literal."*
  `test_arithmetic_above_window_materializes`
  (`marrow/expr/tests/test_values.mojo:226`) pins it — `row_number() + 1` is one
  fused pass over a staged window column, not two eager `apply`s.
- The cost the rejected design was trying to avoid is therefore not paid: a chain
  above a breaker (`rank(a) + b + c`) is **one** fused pass, not several.

## 3. Two lanes

| concept | AOT lane (`values.mojo`) | runtime lane (`dynamic.mojo`) |
| --- | --- | --- |
| node | comptime-typed struct, operands bound on a family trait | `DynValue`: children + payload + `EvalFn` pointer |
| result type | `Datum` | `Datum` |
| shape | `comptime OutShape ∈ {0,1}` | the active variant member of `Datum` |
| broadcast | scalar `vectorwise` splats | scalar stays a `DynScalar`; `into_array(n)` only when forced |
| fusable op | `K.core[W]` inlined into one loop | `K`'s erased dispatch, eager, one array in / one array out |
| reduction | `Reduction` breaker → scalar `Datum` | resolved `Aggregation` → scalar |
| the box | `BoxedValue` (`relations.mojo:155`) — the one box **both** lanes erase into, so each relational operator compiles exactly once |

The two lanes **share no node types**. A fused node's operand is bound on its
family trait; a runtime node's operand is another `DynValue`. The operation is
comptime in both: `DynValue` carries a pointer to its evaluator, so a binary
links exactly the kernels its expressions mention. A runtime *switch over tags*
is the measured anti-pattern here — it cost **+1,807,168 bytes of `__text`
(+45.7%)** on `query_dynvalue`, because every arm became reachable from every
node.

Prior art confirms the shape split: DataFusion keeps `ColumnarValue::Scalar` lazy
(`(Scalar,Scalar)→Scalar`; `into_array` only when forced); Polars makes a repeated
scalar a first-class `Column::Scalar{value,length}`; both reuse the aggregate
accumulator for window running-aggs. marrow carries shape at **comptime** on the
fused lane and as the `Datum` variant tag on the runtime lane.

## 4. Node contract

```mojo
comptime Datum = Variant[DynScalar, DynArray]          # values.mojo:198

def into_array(d: Datum, n: Int) raises -> DynArray:   # values.mojo:201
    """Force `d` to an array of length `n` — broadcasting a scalar."""

trait Value(Copyable, ImplicitlyDeletable, Movable):   # values.mojo:304
    comptime OutType: DataType
    comptime OutShape: Int                             # 0 scalar, 1 columnar

    def materialize(self, batch, mut ctx: Context) raises -> Datum: ...
    def execute(self, batch, mut ctx: Context) raises -> Datum        # defaulted, §5
    def prepare(self, batch, mut ctx: Context) raises                 # defaulted, §5
    def referenced_columns(self) -> List[String]: ...
    def render(self) -> String

trait Breaker(Value):                                  # values.mojo:417
    pass                                               # adds nothing — conformance IS the marker
```

`Datum` is a stdlib `Variant`, not a bespoke struct: there is no `load[W]` on it
and no `is_scalar`/`num_rows` accessor. Splatting happens in the *fused loop*
(a scalar node's `vectorwise` returns `SIMD(value)`), and `into_array(d, n)` is
the one lazy-broadcast forcing point.

Five value families sit under `Value`, each providing its own fused driver:
`NumericValue` (`:433`), `BoolValue` (`:879`, bit-packed), `StringValue`
(`:1384`), `TemporalValue` (`:2162`), `ListValue` (`:2310`). `Breaker` is
orthogonal — a node conforms to a family *and* to `Breaker`
(e.g. `StringToNum(Breaker, NumericValue)`).

Shape derivation, as implemented:

| node kind | `OutShape` | site |
| --- | --- | --- |
| column (`NumericColumn`, …) | `1` | `values.mojo:580` |
| literal (`NumericLiteral`, …) | `0` | `:640` |
| unary / cast / bridge | `Self.A.OutShape` | `:720`, `:754`, `:1259`, `:1290` |
| binary (arith, compare, bool) | `max(Self.L.OutShape, Self.R.OutShape)` | `:673`, `:787`, `:952`, `:1031` |
| reduction | `0` | `:1928` |
| window | `1` | `:2017` |
| string/list breakers, predicates | `1` | `:1321`, `:1353`, `:1689`, `:2351` |

Per-node behaviour:

| node | breaker? | `OutShape` | mechanism |
| --- | --- | --- | --- |
| `NumericColumn[T]` | no | 1 | `vectorwise` reads the batch column |
| `NumericLiteral[T]` | no | 0 | `vectorwise` splats the constant |
| `NumericBinary` / `FloatBinary` / `NumericCompare` / `BoolBinary` | no | max(children) | `vectorwise` fuses children through `K.core[W]` |
| `NumToBool` / `BoolToNum` | no | `A.OutShape` | pure lane bridges (`:1255`, `:1286`) |
| `StringToNum` / `StringToBool` | **yes** | 1 | no value lane: parse the column once via the kernel, then load per lane (`:1316`, `:1349`) |
| `StringPredicate` | **yes** | 1 | variable-width input → fixed-width `BoolArray`, once (`:1685`) |
| `Reduction[K, A]` | **yes** | 0 | folds `A` via the aggregate kernel → `DynScalar` (`:1928`) |
| `WindowFunction[Func, A]` | **yes** | 1 | §7 — materializes the whole output column (`:2012`) |

## 5. Execution

Three methods, one dispatch. `materialize` is the abstract strategy hook every
node implements; `execute` and `prepare` are defaulted on `Value` and branch on
`Breaker` conformance at comptime:

```mojo
# Value.execute — values.mojo:317
def execute(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
    comptime if conforms_to(Self, Breaker):
        var i = ctx.size()
        self.prepare(batch, ctx)
        return ctx.get(i)
    else:
        return self.materialize(batch, ctx)

# Value.prepare — values.mojo:334
def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
    comptime if conforms_to(Self, Breaker):
        ctx.append(self.materialize(batch, ctx))
    # composites override to recurse into children; a leaf does nothing
```

`Context` slots are **positional**: `prepare` appends in DFS order and
`vectorwise` reads them back in the same order through a `mut slot: Int` cursor
threaded down the tree. `Context.get[A: Array](i)` is the typed read, pulling the
array straight out of the slot's `Datum`.

The family driver is the family's `materialize`. `NumericValue.materialize`
(`values.mojo:538-543`) carries the **scalar-eval-once** branch:

```mojo
def materialize(self, batch: RecordBatch, mut ctx: Context) raises -> Datum:
    self.prepare(batch, ctx)
    comptime native = Self.OutType.native
    comptime if Self.OutShape == 0:            # scalar → evaluate the lane once, then splat
        var slot = 0
        var v = self.vectorwise[1](batch, ctx, slot, 0)[0].cast[native]()
        return PrimitiveScalar[Self.OutType](v).to_dyn()
    else:                                      # columnar → one fused vectorized pass
        ...                                    # alloc, apply[native, producer], attach validity
```

So `lit(1) + lit(2)` folds with no loop and returns a scalar `Datum` — never a
length-1 array. `BoolValue` bit-packs into a `Bitmap` instead of a `Buffer`;
`StringValue` drives a builder; the shape branch is the same.

Validity threads alongside: each node exposes `validity(batch)` returning an
offset-0 owned bitmap (`None` = all valid), and the family driver folds it into
the finished array.

## 6. Kernels

Shared substrate. A fusable kernel provides `core[W]` (the SIMD functor) *and*
`apply` (eager, array-in/array-out); the *node* decides which one runs. That is
the load-bearing separation — "fusable" is a property of the expression node, not
of the kernel.

- Arithmetic, comparison and boolean kernels (`kernels/numeric.mojo`,
  `kernels/boolean.mojo`) already provide both.
- Variable-width-input kernels (string predicates, parses) provide `apply` only;
  the node conforms to `Breaker` and stages the result.
- `AggKernel` (`kernels/aggregate.mojo`) backs `Reduction` today and is intended
  to back window running-aggregates (§7). A true sliding frame needs a `retract`
  it does not have.

## 7. Window functions — design (UNIMPLEMENTED forward spec)

**Nothing in this section is shipped.** It is the design `docs/backlog.md` M2.3
sequences; nothing supersedes it. Read §7.0 first for what actually exists.

### 7.0 Current state — a two-node toy

Everything window-related in the tree is `values.mojo:1975-2039`, plus two cases
in `marrow/expr/tests/test_values.mojo`. Specifically:

- **`WindowSpec` (`:1981`) carries frame bounds only** — `start` and `end`, and
  its own docstring says "the toy carries frame bounds only". There is **no
  `partition_by`, no `order_by`, and no `how` (ROWS/RANGE)**.
- **`FrameBound.kind` (`:1976`) is an untyped `UInt8` that is never read** by any
  code path. Neither is `offset`.
- **`trait WindowKernel` (`:1988`) has one method**, `evaluate_all(values:
  DynArray) -> DynArray`. There is no per-row `evaluate`, no
  `comptime frame_dependent`, and it takes no keys, no orders and no spec.
- **`RowNumberKernel` (`:1997`) is the only implementation**, and it **ignores
  its `values` argument** entirely — it returns `1..n` from `len(values)`.
- **There is no `marrow/kernels/window.mojo`.** The kernel lives in the
  expression layer.
- **There is no `.over()`** anywhere in the tree. A window is built by naming the
  node: `RowNumber(col("a", int64), spec)`.
- **Nothing outside `values.mojo` and `test_values.mojo` references any of it** —
  not `relations.mojo`, not `execution.mojo`, not the Python bindings.
- **It exists only in the AOT lane**, which violates the standing invariant that
  no feature may live in only one lane (`docs/backlog.md` §0, invariant 2).

What *is* right about the toy, and worth keeping: `WindowFunction`
(`values.mojo:2012`) is a `Breaker` with `OutShape == 1`, its `materialize`
stages the whole output column into the `Context`, and its `vectorwise` loads
that column per lane — so **arithmetic fuses above a window** already (§2).

### 7.1 Target design

The columnar cross-row quadrant. `WindowFunction` stays a `Breaker` (func + spec,
`OutShape == 1`) and reuses the reduction kernel (ibis: `func: Analytic |
Reduction`).

```mojo
struct FrameBound:                       # DF WindowFrameBound / ibis WindowBoundary
    var kind: FrameBoundKind             # UNBOUNDED_PRECEDING | PRECEDING | CURRENT | FOLLOWING | UNBOUNDED_FOLLOWING
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

Note the two corrections against the toy: `kind`/`how` are **typed value types**,
not raw `UInt8` (`docs/backlog.md` Q4.1 is the same shape of fix), and the spec's
key sub-expressions are `BoxedValue` — the box both lanes erase into — not
`DynValue`, which now names the runtime *node*.

`WindowKernel` needs the two DataFusion eval modes:

```mojo
trait WindowKernel(Kernel):
    comptime frame_dependent: Bool
    @staticmethod
    def evaluate_all(values, keys, orders, spec) raises -> DynArray: ...   # frame-independent
    @staticmethod
    def evaluate(values, range) raises -> DynScalar: ...                   # per-row frame
```

Execution (per DataFusion `WindowAggExec` / Polars `.over()`): evaluate keys/arg
→ partition (reuse `groupby` hashing) → sort each partition (reuse `sort`) →
frame-fold / rank / shift → **scatter back to original row order**.

Kernel families:

- **Ranking** (order only, `evaluate_all`): `RowNumber`, `MinRank`, `DenseRank`,
  `PercentRank`, `CumeDist`, `NTile`.
- **Navigation** (`evaluate_all`): `Lag`, `Lead`, `NthValue`.
- **Running/rolling aggregates**: `RunningAgg[K: AggKernel]` — reuse `AggKernel`.
  *Cumulative* (`UNBOUNDED PRECEDING → CURRENT`) = update-only (`combine`);
  *sliding* = needs `retract` (else `O(n·w)` recompute). `col.sum()` → scalar
  `Reduction[SumKernel, _]`; `col.sum().over(w)` → columnar
  `WindowFunction[RunningAgg[SumKernel], _]`.

API: `.over(win)` on a reduction/analytic; `win = window(partition_by=[...],
order_by=[...], rows=(-2, 0))` — mirrors ibis/PyArrow. **Both lanes**, per
invariant 2: the runtime lane needs the same builders and a `DynValue` evaluator.

### 7.2 Sequencing

`docs/backlog.md` M2.3 owns the order and is authoritative:

> move to `marrow/kernels/window.mojo` → give `WindowSpec` `how`/`partition_by`/
> `order_by` → partition (reuse `groupby` hashing) + sort within partition (reuse
> `sort`) + scatter back → ranking family → navigation family
> (`Lag`/`Lead`/`NthValue`) → `RunningAgg[K: AggKernel]` → `.over()` on both lanes
> → wire through `relations.mojo`.

### 7.3 Deferred within this spec

- **Batch-local partitions** in v1 (each batch = full partition set, sorted
  within); cross-batch/streaming windows (cf. DataFusion
  `BoundedWindowAggExec`) are a plan-layer concern.
- **Sliding frames** need `AggKernel.retract`; cumulative first.
- **Constant frame bounds** only (`rows=(-2, 0)`); dynamic (`Value`) bounds later.
- **`ROWS` first**; `RANGE`/`GROUPS` reuse the same driver with a different range
  calculation.
- **CSE / value-numbering** (content-addressed hashing to hoist repeated
  expensive subexpressions) is an optional optimization, not required for
  correctness. Tracked in `docs/design-expression-evaluation.md`.
- **`OutShape` for the bool and string families** (scalar string/bool, `any`/`all`)
  applies the same idea; out of scope here.
