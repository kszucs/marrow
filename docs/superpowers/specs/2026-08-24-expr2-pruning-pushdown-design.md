# expr2 — pruning and pushdown design

**Status:** proposed, 2026-08-24. Target: `marrow/expr/`, branch `expr2`.

Companion to `2026-08-21-expr2-design.md` (the layer), `2026-08-21-optimizer-design.md` (§1.2 and §5 are load-bearing here) and `2026-08-22-push-engine.md` (the operator contract this hangs off).

---

## 0. Problem statement

`marrow/expr/` has 8/8 relations, 5/5 value families and `params`. It has none of the *subsystems*. `ParquetScanOperator`'s own docstring states the gap precisely (`marrow/expr/physical.mojo:1000-1004`):

> **No pruning.** `expr/`'s scan skips row groups whose statistics prove no row can match, and windows several groups at once. Both need `expr/pruning.mojo`, which has no `expr2` counterpart yet. Their absence costs speed and never correctness — a `Filter` above the scan applies the predicate exactly — so this is a smaller scan, not a wrong one.

Two capabilities are missing, and they are different problems that share one delivery mechanism:

- **Pruning** — *evaluation over a different domain*. Given only each column's `[min, max]` bounds and null count for a row group or a data page, could this predicate be `TRUE` for **some** row? A proven "no" lets the reader skip the group without decoding it.
- **Pushdown** — *delivery*. Getting the predicate (and the needed column set) from a `Filter` node down to the `ParquetScan` that can use them. `expr/` does this by rewriting the plan. This design does not.

The user's binding constraint: **a simple, clean and sound architecture, preferably avoiding runtime dispatching in the case of comptime expressions.** §3 is the mechanical answer; everything else is arranged around it.

### 0.1 What correctness means: one-sided error

Pruning may only ever produce **false positives**, never false negatives. Reading a group that turns out to contain no matching row costs time. Skipping a group that contained one changes the answer.

The consequence that shapes every type below: **the pruning domain is one-sided.** The only question a node ever answers is *"could this be TRUE here?"*. There is no "could this be FALSE" and no "is this definitely TRUE".

This is not a simplification, it is what keeps the algebra sound with two lines of proof per node. It is also why `NOT` and `XOR` prune nothing (§2.5) — negating "maybe true" is not "maybe false", and inventing that answer is the classic way a pruner goes silently wrong.

Consulted for the semantics:

- **Arrow C++** — `cpp/src/arrow/compute/expression.h:186-225`, `SimplifyWithGuarantee`: bounds are modelled as a *guarantee* (a predicate known true of every row), and simplification may only weaken the predicate, never strengthen it. Same one-sidedness, expressed as expression rewriting rather than interval evaluation.
- **Parquet statistics semantics** — `cpp/src/parquet/statistics.h:135-176`: `null_count`, `has_min_max` and `has_null_count` are independent flags; min/max are computed over **non-null** values only, and `HasMinMax()` false means no usable bound.
- **arrow-rs** — `parquet/src/arrow/arrow_reader/statistics.rs:1439-1471` carries `missing_null_counts_as_zero` as an explicit *option*, because treating an absent null count as zero is a soundness choice, not a default. `:1721-1740` exposes `is_min_value_exact` / `is_max_value_exact`, which matter for equality and for `IS NULL`, not for range pruning (a truncated bound only ever *widens* the interval).
- **Marrow's own writer already respects the float rule**: `parquet/statistics.mojo:101-119` computes float bounds with `skip_nan=True` and normalises signed zero so the bound brackets both `+0.0` and `-0.0`. The reader must mirror it — a NaN bound bounds nothing, so a bound that is NaN must be read as *unknown*, not compared.

### 0.2 The four hard constraints

1. **Binary size.** `marrow.expr` must keep the closed-erasure/DCE property. Gated by `benchmarks/binary_size/`, `threshold_pct` **0.5**. Priors that reject designs: an extra slot on `expr/`'s aggregate box cost **+3.2 MB / +24%**; a shared generic dispatch adapter cost **+662,740 bytes (+31.9% of `__text`)**.
2. **`DynValue` has five slots and no rewrite slot.** Its docstring is explicit that this is a *property*, not an omission: "parameter values travel *through* an execution rather than being substituted into a copy of the plan, so the box never has to hand back a re-boxed `DynValue`." §4 keeps that true.
3. **`physical.mojo` imports almost nothing from the package.** Exactly one intra-package import today: `from .params import Bindings` (`physical.mojo:51`). That is the precedent this design reuses — a *leaf* module that both `physical` and the lanes may depend on.
4. **Mojo's recorded limits** (CLAUDE.md "Associated types, traits, reflection"). Chained associated-type projections do not reduce at a call/return site; a trait default whose return type a conformer must change cannot be overridden; a closure type cannot be generic over its own trait bound; a function that can `raise` cannot run at comptime. §7 lists the designs each of these kills.

---

## 1. Architecture at a glance

Four new files, one new argument, one new field, zero new slots on either box.

```
marrow/kernels/bounds.mojo         NEW  Bounds[dt], BoundsKernel, the eight readings.
                                        Typed. No DynScalar. No dispatch ladder.
marrow/expr/pruning.mojo          NEW  Truth, PruneStats, Prunable, PrunePredicate.
                                        A leaf module, like params.mojo.
marrow/expr/pushdown.mojo         NEW  Pushdown — the lowering context.
marrow/expr/tests/test_pruning.mojo    NEW
marrow/expr/tests/test_pushdown.mojo   NEW

marrow/expr/comptime/core.mojo    +   ComptimeValue refines Prunable;
                                        PrimitiveValue gains `bounds`.
marrow/expr/comptime/*.mojo       +   `prune` / `bounds` overrides on the nodes
                                        that can do better than "maybe".
marrow/expr/logical.mojo          +   to_operator gains a `Pushdown` argument;
                                        Filter carries Optional[PrunePredicate];
                                        DynRelation.filter gains one overload.
marrow/expr/physical.mojo         +   ParquetScanOperator takes the predicate
                                        and computes a read plan.
marrow/expr/runtime/values.mojo   +   a `_prune` thin pointer (stage 6).
```

### 1.1 Responsibilities, one line each

| type | responsibility |
|---|---|
| `Bounds[dt]` | what one **typed** sub-expression's value can be, over one row group. Register-passable. |
| `BoundsKernel` | one operator read over bounds: *could it be true for some pair?* |
| `Truth` | the one-sided answer: `never` or `maybe`. |
| `PruneStats` | one row group's (or page's) per-column bounds and null counts, over the scan's schema. |
| `Prunable` | *"I can answer, conservatively, whether I could be true here."* Total default: `maybe`. |
| `PrunePredicate` | a `Prunable` erased behind **one** function pointer. Built at `filter()`, consumed by the scan. |
| `Pushdown` | what the nodes above told the source: an optional predicate and an optional column set. |

### 1.2 Dependency direction

Strictly one-directional. Arrows point *from* dependent *to* dependency.

```
        expr2/builders.mojo
                 |
        expr2/logical.mojo ──────────────┐
           |          |                  |
   comptime/*     runtime/values.mojo    |
       |    \        /                   |
       |     \      /                    |
       |   expr2/physical.mojo ──────────┤
       |          |                      |
       └──> expr2/pruning.mojo <─────  expr2/pushdown.mojo
                 |        \
        expr2/params.mojo  kernels/bounds.mojo, kernels/interval.mojo
             (Bindings)          |
                 |               |
             marrow core: schema, scalars, dtypes, tabular
```

`kernels/bounds.mojo` depends only on `std`. `expr2/pruning.mojo` depends on `..schema`, `..scalars`, `..dtypes`, `..kernels.bounds` and `.params`. Neither imports `logical`, `physical` or either lane. `physical.mojo` gains exactly one inbound edge — to `pruning`, a leaf, exactly the shape its existing edge to `params` already has.

**Two pre-existing cycles found while checking this, and one must be broken first.**

- `params.mojo:61` imports `.comptime.core` for `NumericValue` (because `Param[T]` lives there), and `comptime/core.mojo:52` imports `..params` for `Bindings`. That is a two-module cycle *today*.
- `logical.mojo:34` imports `.runtime.values.column`, and `runtime/values.mojo:38` imports `..logical`. A second two-module cycle.

Mojo resolves both (CLAUDE.md: "Mojo resolves circular imports between modules in the same package"), so neither is a build failure. But this design needs `pruning.mojo` to depend on `Bindings` (§5), and routing that through a module that itself depends on `comptime.core` would drag the whole comptime lane into `pruning`'s dependency set and put `physical -> pruning -> params -> comptime.core -> physical` on the graph. **Stage 0 breaks it**: move `Param[T]` into `comptime/leaves.mojo`, where the other comptime leaves already live and where its own docstring says it belongs ("`Param` mirrors `Literal`"). `params.mojo` is then genuinely what its name says — `Bindings` and nothing else — and the cycle is gone. The second cycle (`logical <-> runtime.values`) is untouched by this work; it is recorded here so it is not mistaken for something this design introduced.

---

## 2. Pruning

### 2.1 `Bounds[dt: DType]` — the typed interval

```
struct Bounds[dt: DType](Copyable, ImplicitlyCopyable, Movable):
    var lo: Scalar[dt]
    var hi: Scalar[dt]
    var known: Bool          # False => lo/hi are meaningless, treat as unbounded
    var all_null: Bool       # null_count == num_rows
    var may_be_null: Bool    # null_count != 0, or null_count unknown
```

Five register-sized fields. No `Optional[DynScalar]`, no heap, no `ArcPointer`, no `Variant`. `unknown()`, `point(v)`, `cast[to: DType]()`.

This is the whole reason the fused lane pays no dispatch, and it is the one place this design *deliberately does not reuse* `expr/`'s prior art. `kernels/interval.mojo`'s `Interval` holds `Optional[DynScalar]` and compares through `Interval._compare_scalar`, which calls `DynType.dispatch_primitive` (`interval.mojo:128-140`). That is an eleven-arm runtime ladder **per comparison node, per row group** — runtime dispatch inside a fused predicate, and structurally the same shape as the `+662,740` adapter. `Interval` stays exactly where it is and serves the runtime lane, which has no types to fuse with (§4.3).

### 2.2 `BoundsKernel` — the interval reading of an operator

`kernels/interval.mojo`'s docstring already settled *where* this belongs and why:

> `NumericCompareKernel` used to carry a `comptime StringKernel` naming its string counterpart, and it was removed because "which family `a < b` means is a question about the operands, and it belongs to whoever is interpreting the operator, not to the SIMD kernel". An interval reading is the same kind of claim, so it lives beside the SIMD kernels rather than inside them, **and the expression node pairs the two.**

So the reading is a second kernel, and the node names both — `NumericCompare[K: NumericCompareKernel, P: BoundsKernel, L, R]`, spelled at the named shapes so no call site changes:

```
comptime Gt = NumericCompare[GtKernel, GtBounds, _, _]
comptime Lt = NumericCompare[LtKernel, LtBounds, _, _]
```

The trait is the typed sibling of `IntervalKernel`:

```
trait BoundsKernel(Kernel):
    @staticmethod
    def maybe[dt: DType](l: Bounds[dt], r: Bounds[dt]) -> Bool: ...
```

`maybe[dt]` is parameterised exactly as `NumericCompareKernel.core[dt, W]` is. `LtBounds.maybe` is two comparisons and a flag test; it inlines to nothing. Eight conformers: `Lt`, `Le`, `Gt`, `Ge`, `Eq`, `Ne`, `And`, `Or` — with `Ne` answering `True` unconditionally for the reason `NeInterval` already records (`interval.mojo:211-213`).

**None of them raises.** Which is the next point, and it is the important one.

### 2.3 `prune` never raises — and that is a design rule, not an accident

`Value.prune` and `bound_column` in `expr/` declare `raises` and therefore cannot run at comptime (CLAUDE.md). This design makes `prune` and `bounds` **non-raising, total functions**, for three reasons that reinforce each other:

1. **"I don't know" is always a legal answer.** One-sided error means the conservative result is always available. A pruner that fails is a pruner that has a bug, because the fallback (`Truth.maybe`) is correct by construction.
2. **A pruning failure must never fail a query.** Pruning only ever reduces I/O. A missing column, a dtype the file disagrees about, an unbound parameter — each of those must degrade to "read the group", not abort a scan. If it can raise, one bad footer turns a working query into an error.
3. **Non-raising is comptime-eligible.** Nothing in this design needs comptime evaluation of `prune` (§3.4 explains why it would be meaningless), but keeping the door open costs nothing here and closing it costs a rewrite later.

Every primitive this needs is already non-raising, checked in the tree:

| call | file:line | raises? |
|---|---|---|
| `Schema.get_field_index` | `schema.mojo:141` | no |
| `DynScalar.type()` | `scalars.mojo:84` etc. | no |
| `DynScalar.is_valid()` | `scalars.mojo:87` etc. | no |
| `DynScalar.as_primitive[T]()` | `scalars.mojo:747` | no (`debug_assert`) |
| `PrimitiveScalar.value()` | `scalars.mojo:214` | no |
| `DynType.__eq__` | `dtypes.mojo:1317` | no |

The one requirement: `as_type` is a `debug_assert`, so a mismatch **aborts the process** rather than raising (the failure mode `comptime/boolean.mojo:41-58` documents at length: "the abort took down the whole test runner, failing all seven cases in the file rather than the one that was wrong"). Therefore the leaf must gate the unwrap on a type equality check, never on a hope:

```
# Column[T: NumericType] — T is Defaultable, so the leaf can name its own dtype.
if not s or s.value().type() != DynType(Self.T()) or not s.value().is_valid():
    return Bounds[Self.Type.native].unknown()
```

For `TemporalColumn[T]` — whose `dtype()` already reads from the schema at `leaves.mojo:138-145` because a temporal dtype is not `Defaultable` — the comparison is against `stats.dtype_at(i)`, the scan schema's own field dtype. That also catches the unit mismatch a width comparison cannot see (`timestamp[s]` vs `timestamp[ms]`), the same hazard `TemporalCompare.bind` guards against at `numeric.mojo:320-331`.

`Bindings` needs one addition: a non-raising `find(name) -> Optional[DynScalar]`. `Bindings.get` (`params.mojo:86-89`) raises because `Dict.__getitem__` does.

### 2.4 The trait surface

Two methods, one per question, declared where the question means something.

```
# pruning.mojo — a leaf module
trait Prunable(Copyable, Deinitable):
    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """Could this be TRUE for some row covered by `stats`?"""
        return Truth.maybe          # total default: every node conforms for free
```

```
# comptime/core.mojo
trait ComptimeValue(Evaluable, Value, Prunable): ...

trait PrimitiveValue(ComptimeValue):
    ...
    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        """What this sub-expression's value can be. Default: unknown."""
```

**`prune` and `bounds` are `bind` and `lane` in a different domain, and the signatures mirror them deliberately.** That is not aesthetics — it is how the design inherits type-level behaviour that has already been proven to compile:

| fusion | pruning | shared shape |
|---|---|---|
| `bind(batch, bindings) -> Self.Bound` | `bounds(stats, bindings) -> Bounds[Self.Type.native]` | resolve the subtree against a source, once |
| `lane[W](bound, i) -> SIMD[Self.Type.native, W]` | folded inside `bounds` | typed, register-passable result |
| `.cast[Self.ArgType.native]()` at a compare | `.cast[Self.ArgType.native]()` at a compare | the operand-promotion step |
| `validity(bound)` | `Bounds.may_be_null` / `all_null` | nullability answered from the bound, not the data |

`PrimitiveValue.lane` already returns `SIMD[Self.Type.native, W]` and compiles, so `Bounds[Self.Type.native]` is the same projection shape at the same kind of site. `NumericCompare` already binds `comptime ArgType = promote[L.Type, R.Type]` as a **member** specifically because `Self.L.Type.native` used directly "does not reduce — the compiler reports a type 'cannot be converted' to *itself*" (`numeric.mojo:283-291`). `prune` reuses that member rather than rediscovering the failure.

`Value` gains **nothing**. `ComptimeValue` gains a supertrait whose sole method has a total default. `DynValue` gains **no slot**.

### 2.5 The rules, per node

| node | `bounds` / `prune` | soundness |
|---|---|---|
| `Column[T]`, `TemporalColumn[T]` | typed unwrap of `stats.min/max` at its column index | min/max cover every **non-null** value; nulls are covered by `may_be_null` |
| `BoolColumn` | `prune`: `never` iff `all_null`, else `maybe` | an all-null bool column has no `TRUE` row |
| `Literal[T]`, `StringLiteral[T]` | `Bounds.point(v)`, `may_be_null=False` | exact |
| `Param[T]` | `Bounds.point` of the bound value or default; `unknown()` when neither | §5 |
| `NumericCompare[K, P, L, R]` | `P.maybe[ArgType.native](l.bounds().cast(), r.bounds().cast())` | §2.6 |
| `TemporalCompare[K, P, L, R]` | same, but `ArgType = L.Type`; `unknown` if the two dtypes differ | matches `bind`'s runtime dtype check |
| `BoolBinary[AndKernel, …]` | `l.prune() and r.prune()` | a conjunct proven false makes the whole conjunction false |
| `BoolBinary[OrKernel, …]` | `l.prune() or r.prune()` | both disjuncts must be provably false |
| `BoolBinary[XorKernel, …]`, `Not[A]` | `maybe`, unconditionally | **§2.7** |
| `NumericBinary`, `CaseWhen`, `ListLength` | `unknown()` (the default) | v1; §2.8 |
| `StringColumn`, string comparison | `unknown()` (the default) | v1; §2.8 |

**One arithmetic invariant makes the comparison rules sound**, and it is worth stating because it is the step where a careless implementation goes wrong.

`NumericCompare.lane` casts both operands to `ArgType.native` before comparing (`numeric.mojo:163-168`). `prune` casts both **bounds** with the identical cast. That is sound iff the cast is **monotone non-decreasing**, because monotonicity is exactly what preserves `cast(lo) <= cast(v) <= cast(hi)`. Every cast `promote` can produce is monotone: widening within a signedness class, and int->float (rounding is monotone). So the interval reading agrees with the lane's reading, row for row.

**The exception, which is a pre-existing defect and not one this design introduces.** `promote[Int64Type, UInt64Type]` resolves to `Int64Type` — `_outranks` (`rules.mojo:42-61`) sees neither operand is float and falls to `bit_width_of[int64]() >= bit_width_of[uint64]()`, which is `64 >= 64`, true. So `u64 > i64` compares a `uint64` *reinterpreted* as `int64`, and `UInt64(2**63) > Int64(0)` evaluates `False`. `lane` already does this. Because `prune` performs the *same* cast, pruning stays consistent with the filter — it cannot skip a group whose rows the filter would have kept — so this design is safe on top of the bug. It should be filed separately against `promote`.

### 2.6 Why the comparison rule is the whole pruning algorithm

`x > 5` over a group with bounds `[min, max]`:

- Every non-null `x` satisfies `min <= x <= max`.
- Null `x` makes `x > 5` evaluate to `NULL`, and `FilterOperator` keeps only valid `TRUE` — `filter()` honours mask validity. So a null row can never be kept.
- Therefore `x > 5` can be `TRUE` for some row **iff** `max > 5`.

That is `GtBounds.maybe`. Everything else in the algebra is `and`, `or`, and the conservative default. There is no fourth idea in this subsystem, which is the point of §0.1.

### 2.7 `NOT` and `XOR` prune nothing, and that is a correctness statement

A one-sided domain has no negation. `Not(p).prune()` needs *"could `p` be FALSE?"*, and `p.prune()` answers *"could `p` be TRUE?"*. Deriving one from the other is not possible, and inventing it produces false negatives: `p.prune() == maybe` says nothing at all about whether `p` is ever false.

`XorInterval` in `expr/` already reaches the same conclusion for the same reason (`interval.mojo:241-243`: "both operands being possible says nothing about them differing on any single row"). `NeInterval` likewise.

The cheap escape, if `NOT` pruning ever matters, is **not** a two-sided lattice — that doubles every node's obligation and doubles the surface where a sign error becomes a wrong answer. It is De Morgan at *construction* time in `builders.mojo`, where the operand types are still visible: `not_(Gt(a, b))` returns `Le(a, b)` by a comptime overload. That is a build-site rewrite in the one place where the types exist. It is not in v1.

### 2.8 What the defaults cost, honestly

`NumericBinary` returning unknown means `a * 2 > 10` prunes nothing. `StringColumn` returning unknown means `name = 'x'` prunes nothing, and ClickBench filters on strings. Both are *correct* and both are *slow*, which is the right side of the trade for a first landing, and both extend without touching anything else — a node that today inherits the default overrides it later and nothing above it changes. Interval arithmetic for `+`/`-` is straightforward with saturation; `*` needs the four-product min/max and overflow care; strings need a `StringBounds` carrying two heap `String`s, which is why they are not in the register-passable `Bounds[dt]`.

---

## 3. How the comptime lane avoids runtime dispatch

This is the requirement the design exists to satisfy. Mechanically, in five steps.

### 3.1 The one indirect call, and where it sits

`Filter` holds `Optional[PrunePredicate]`. `PrunePredicate` is a **one-slot** box:

```
struct PrunePredicate(Copyable, Movable):
    var _boxed: ArcPointer[NoneType]
    var _prune: def(ArcPointer[NoneType], PruneStats, Bindings) thin -> Truth

    @staticmethod
    def _prune_tramp[P: Prunable](
        ptr: ArcPointer[NoneType], stats: PruneStats, bindings: Bindings
    ) -> Truth:
        return rebind[ArcPointer[P]](ptr)[].prune(stats, bindings)
```

`ParquetScanOperator._read_plan` calls `_prune` **once per row group** (and, at stage 4, once per page per column). A row group is order 10^6 rows. One indirect call per 10^6 rows is not a dispatch cost; it is the erasure boundary, and it is placed at the coarsest possible granularity on purpose.

### 3.2 Everything below the call is monomorphic

`_prune_tramp[Gt[Column[Int64Type], Literal[Int64Type]]]` calls `Gt<…>.prune`, which is a concrete function on a concrete type. The compiler sees:

```
GtBounds.maybe[DType.int64](
    self.l.bounds(stats, bindings).cast[DType.int64](),
    self.r.bounds(stats, bindings).cast[DType.int64](),
)
```

`self.l` is `Column[Int64Type]`, `self.r` is `Literal[Int64Type]`, `Self.ArgType.native` is `DType.int64`. Every operand type, every kernel and the promotion are comptime parameters. `Bounds[DType.int64]` is register-passable. The whole predicate collapses to straight-line code: one `get_field_index`, one `DynType` compare, one `as_primitive` borrow, two `Scalar[int64]` loads, one compare, a return.

Concretely, the dispatch this design avoids relative to `expr/`:

| per comparison node, per row group | `expr/` (`Interval`) | expr2 fused (`Bounds`) |
|---|---|---|
| `DynType.dispatch_primitive` ladder | 1 (11 comptime arms, one taken) | **0** |
| `Optional[DynScalar]` copies | 4 | **0** |
| `DynScalar` type equality | 1 | 1 (moved to the leaf, once per leaf) |
| indirect calls | 1 per node (`_prune_fn` on the box) | **0** |

### 3.3 Nothing is *reachable* that is not used

Three separate mechanisms keep the fan-out closed, and each has a precedent in the tree:

1. **`Prunable` is a total default, not an abstract method.** A node that cannot prune inherits `Truth.maybe` — a `return` of a constant. No node is forced to name a kernel it does not use.
2. **`BoundsKernel` is a node parameter, not a registry.** `Gt` names `GtBounds`; a binary that never writes `Lt` never links `LtBounds`. Same closed-erasure property `numeric.mojo:12-14` states for the SIMD kernels.
3. **`_prune_tramp[P]` is instantiated at `filter()`, per *filter predicate type* — not at `DynValue.__init__`, per *boxed value type*.** This is the difference between paying for the projection expressions, the sort keys and the aggregate inputs of every plan in the program, and paying for the predicates. It is the single most important binary-size decision in this document and §7.2 records the alternative it rejects.

**The check that proves it, and it must be run:** `nm -C` on the `query_expr2_scan` gate binary (new, §8) must show **zero** symbols from `marrow::kernels::interval` and **zero** `dispatch_primitive` instantiations reached from the pruning path. The `baseline.json` `_comment` records exactly why `size` alone is not enough.

### 3.4 What "comptime" does and does not mean here

The prompt asks whether pruning should be a compile-time computation. **It cannot be, and the reason is worth stating so nobody tries.** Statistics are runtime data read from a file footer at execution time. There is no compile-time value to fold.

What *is* compile-time is everything else: the structure of the evaluation, every operand type, the promotion, the choice of interval rule, and the absence of any dispatch. That is what "avoiding runtime dispatching in the case of comptime expressions" means operationally, and §3.2 is the mechanical answer.

The one genuinely comptime-computable piece is **column resolution**. `Column[T].bounds` does `stats.index_of(self._name)` at run time. CLAUDE.md records that a struct holding heap fields can be a comptime parameter and that "the same `index_of` ran at comptime and at runtime". So a future `Column[T, s: MiniSchema, i: Int]` could resolve its index at elaboration and turn an unknown column into a `comptime assert`. **That is the seam this design leaves open and does not take.** It requires a comptime schema, which requires `Table[T]`, which CLAUDE.md records as deferred behind the reflection limit. `prune` being non-raising (§2.3) is what keeps it reachable.

---

## 4. Pushdown, without a rewrite

### 4.1 The mechanism: information travels down the lowering, not into a copy of the plan

`params.mojo`'s central claim is that a plan is immutable and per-execution information travels *through* the execution. Pushdown gets the identical treatment. `Relation.to_operator` already takes `bindings` and threads it down; it now also takes a `Pushdown`:

```
# pushdown.mojo
struct Pushdown(Copyable, Movable):
    var predicate: Optional[PrunePredicate]   # pruning hint for a source
    var needed: Optional[List[String]]        # column set for a source
```

```
def to_operator(
    self, ctx: ExecContext, bindings: Bindings = Bindings(),
    pushed: Pushdown = Pushdown(),
) raises -> Pipeline
```

Lowering is *already* a recursive descent from root to source. Pushdown is a second value carried down that same descent. There is no separate pass, no plan walk, no `children()`, no rewrite, and no rebuilt node.

This is one **signature** change on the existing `_virt_to_operator` slot, not a new slot. `Pushdown` contains no `DynRelation`, so it does not trip the recursion restriction that shaped `expr/`'s protocol.

### 4.2 Per-node rules, and the one that is a correctness trap

| node | predicate | needed |
|---|---|---|
| `Filter` | **conjoin** its own `PrunePredicate` and forward | union its predicate's `columns()` and forward |
| `Sort` | forward unchanged | union its keys' `columns()` |
| `Limit` | **clear** — see below | forward unchanged |
| `Project` | clear | replace with the union of its values' `columns()` |
| `Aggregate` | clear (a `Filter` above it is `HAVING`) | replace with keys' ∪ aggs' `columns()` |
| `Join` | clear (needs column provenance; §7.6) | clear |
| `InMemoryTable` | ignore | ignore |
| `ParquetScan` | **consume** | **consume** |

**`Limit` must clear the predicate, and getting this wrong silently changes results.** `filter(p)` above `limit(10)` means: take the first 10 rows, *then* apply `p`. If the scan below skips a row group that cannot match `p`, the "first 10 rows" become different rows — and rows that the correct query would have returned disappear. That is a false negative, the one error class §0.1 forbids. `Sort` is safe by contrast: sorting does not drop rows, so the surviving set of `filter(p)` above `sort` is identical either way.

`Project` clearing the predicate is conservative. Forwarding is legal exactly when every column the predicate reads is a bare pass-through of the same name, and `Project._output_schema` already computes bare-column-ness. Not in v1.

**Everything above the scan is unchanged.** The `FilterOperator` still evaluates the predicate exactly, on every row the scan produced. That is what makes the whole subsystem speed-only: delete every line of it and the answers are identical.

### 4.3 Where the `PrunePredicate` comes from

`Filter` holds a `DynValue`, and the box exposes no `prune`. The concrete type has to be captured **where it is still visible**, which is the plan-building verb:

```
def filter[V: Value & Prunable](self, var predicate: V) raises -> DynRelation:
    return DynRelation(
        Filter(self.copy(), DynValue(predicate.copy()), PrunePredicate(predicate^))
    )

def filter(self, var predicate: DynValue) raises -> DynRelation:   # unchanged
    return DynRelation(Filter(self.copy(), predicate^, None))
```

The two overloads are **disjoint by construction**, which is the property that makes this safe: `DynValue` deliberately does not conform to the traits it erases, so a `DynValue` argument matches only the second, and a concrete node matches only the first. `Value & Prunable` covers **both lanes** once `RuntimeValue` conforms to `Prunable` — one verb, one box, two lanes, exactly as `DynValue` itself works.

Note what falls out for free: `expr/` pushes only into an **adjacent** scan. Here the predicate rides the lowering all the way down, so `Filter(Sort(ParquetScan))` prunes. That is a capability `expr/` does not have, obtained by removing a mechanism rather than adding one.

### 4.4 Consuming it: `ParquetScanOperator`

On first `drain`, the operator opens the file and computes a read plan, as `expr/`'s `ScanProcessor._read_plan` does:

1. Build a **leaf map** from the scan's schema to file-leaf indices. Empty when any column has no leaf of its own name — a nested file turns pruning **off** rather than misaligning bounds against a projected schema.
2. Per row group, assemble a `PruneStats` and call the predicate once. `Truth.never` -> skip the group entirely.
3. Stage 4: per surviving group, per column with a page index, evaluate the predicate with **only that column's** page bounds known and every other column unknown; keep a page iff `maybe`; intersect the per-column `RowSelection`s.

Marrow's reader already supplies everything needed: `ParquetFile.statistics()` gives `null_count` and typed `min`/`max` per (row group, leaf); `page_bounds()` gives per-page bounds and correctly reports an all-null page as no bound.

**Windowing is out of scope.** `expr/` reads several row groups per call under a byte budget and records a 1.6x-4.7x win for it. That is a separate performance change with its own benchmark, and mixing it into this one would make the pruning measurement unreadable.

### 4.5 Projection pushdown

`ParquetScan`'s docstring already states the mechanism: "**The schema is the projection.**" So `ParquetScan.to_operator` intersects its schema with `pushed.needed` and hands the narrowed schema to `ParquetScanOperator`. Nothing is rebuilt; `ParquetScan.schema()` still reports the full schema.

That last point looks like an inconsistency and is not. `needed` is only ever *established* by a node that also replaces the schema (`Project`, `Aggregate`); pass-through nodes only ever *union in* their own read columns. So the fields that `Filter.schema()` reports but the runtime batch does not carry are, by construction, fields nothing above ever names. This invariant must be stated in `Pushdown`'s docstring, because it is the thing that would break if someone later let `Limit` or `Filter` *set* `needed` rather than widen it.

---

## 5. Interaction with `params`

**A late-bound predicate prunes exactly as well as a literal one, and that is a direct consequence of the no-rewrite decision.**

The framing "a predicate whose value is late-bound cannot be pruned at plan time" is right about plan time and does not bind here, because **pruning never happens at plan time.** It happens inside `ParquetScanOperator`, per row group, at execution time, after `Bindings` is in hand. `prune(stats, bindings)` mirrors `bind(batch, bindings)` for precisely this reason. `Param[T].bounds` returns:

- `Bounds.point(v)` when `bindings.find(name)` yields a value of the right dtype;
- `Bounds.point(default)` when it does not and a default exists;
- `Bounds.unknown()` otherwise — **it does not raise**, per §2.3.

`expr/` has a recorded bug here, and it is the reason this must be threaded rather than skipped. `expr/dynamic.mojo:670-679`:

> A bound parameter is a literal that arrived late, so it prunes […] a parameterised date filter read `Interval.unknown()` and decoded every row group, while the *fused* lane's `NumericParam.prune` returned a point interval.

`expr/` fixed it by reading a process-global parameter registry from inside `prune`. expr2 deleted that registry on purpose, so the value has to arrive as an argument. It does, on the same call, from the same `Bindings` the operator is already carrying.

**The unbound-and-no-default case.** `bounds` answers unknown, the scan reads everything, and then `FilterOperator` -> `Param.bind` raises **naming the parameter**. The error still surfaces; it surfaces after one row group of wasted I/O. That is the correct division: **pruning degrades, binding raises.**

---

## 6. The runtime lane

Same capability, different cost, and the cost is confined.

`RuntimeValue` gains a `_prune: PruneFn` thin pointer beside `_eval`, bound at construction by the leaf constructor — never a switch on `_tag`. The module docstring already fixes the rule and this obeys it:

> **A tag never selects a kernel.** `_tag` is how a node prints and how it prunes; `_eval` is how it computes […] Routing on the tag would put every kernel in every binary that builds any expression.

That last clause is what `expr/` violates: `DynValue.prune` (`expr/dynamic.mojo:647-707`) is a nine-branch `if/elif` over interval-kernel *name strings*, which makes the interval kernels — and through them `Interval._compare_scalar` and its `dispatch_primitive` ladder — reachable from **any** binary that builds any runtime value. A `_prune` pointer keeps the same closed-erasure property `_eval` has.

The runtime lane **does** use `kernels/interval.mojo`'s `Interval`, `DynScalar` bounds and `dispatch_primitive` compare, because it has no types to fuse with. That is the honest asymmetry, and it is the same one the lane already carries everywhere else (1.46 MB vs 4.91 MB for the same plan).

One prerequisite: expr2's runtime lane has **no comparison or boolean constructors today** — `runtime/values.mojo` exposes `column`, `literal`, `coalesce` and `case_when` only. Runtime pruning has nothing to prune until they land, which is why it is stage 6 and not stage 2.

---

## 7. Alternatives considered and rejected

### 7.1 Plan rewriting — `with_predicate` / `with_projection`, as `expr/` does

`logical.mojo`'s own docstring already refused the four rewrite methods:

> The last four exist for an optimizer that was never finished, and two of them cannot express any rewrite that changes a node's type or arity — which is every rewrite worth having. They are not reproduced here.

The optimizer spec §1.2 gives the mechanism: `DynRelation(copy=self)` copies trampolines bound to `T`, so a rewritten node **must have the same concrete type**, and returning `Optional[DynRelation]` from a trampoline *field* makes the struct recursive and is rejected by the compiler.

Rejected because it costs two slots on `DynRelation`, needs `children()` to reach a non-adjacent scan, and buys nothing this design does not already get. **Only a rewrite forces the plan to be copied; this does not, so `DynValue`'s "no rewrite slot" property survives untouched.**

### 7.2 A sixth slot on `DynValue` — `_prune`

The obvious design, and the one `expr/` uses. Rejected on two counts:

- It contradicts the box's stated contract. `DynValue`'s docstring counts five slots against `expr/`'s seven and names the two it dropped; adding one back for a concern only *predicates* have is the opposite of the reasoning that removed `resolve_names`.
- **It pays per boxed value type instead of per filter predicate type.** `_prune_tramp[V]` at `DynValue.__init__[V]` is instantiated for every projection value, every sort key and every aggregate input in the program. The optimizer spec §7 budgets an eighth `BoxedValue` slot at **+0.25% on the two fused gates** with an explicit veto, and CLAUDE.md records the aggregate-box slot addition at **+3.2 MB / +24%**.

### 7.3 Reusing `kernels/interval.mojo`'s `Interval` in the fused lane

The largest saving of new code, and the direct violation of the user's requirement. `Interval` carries `Optional[DynScalar]` and compares through a runtime dtype dispatch. That puts an eleven-arm ladder and heap-boxed scalars **inside a fused predicate**, per comparison node, per row group — the exact structural shape CLAUDE.md attributes `+662,740 bytes` to.

### 7.4 A `comptime Bounds: BoundsKernel` companion on `NumericCompareKernel`

Rejected because **the tree already removed exactly this and recorded why**: `NumericCompareKernel` carried a `comptime StringKernel` naming its string counterpart, removed because "which family `a < b` means is a question about the operands... belongs to whoever is interpreting the operator, not to the SIMD kernel".

### 7.5 Making `ParquetScanOperator` generic on the predicate type

Removes the last indirect call (§3.1). Rejected: it instantiates the entire Parquet decode ladder **once per predicate type**. That is the widest fan-out in the tree. The erasure boundary belongs *above* the scan operator.

### 7.6 Conjunction splitting, to push half a predicate below a `Join`

Out of v1. The optimizer spec §2.3 verified the fused lane *can* be split by a `conjuncts()` trait default, then recommended **not building it yet**: "the only consumer would be scan pruning, and `prune()` already handles a whole conjunction compositionally".

### 7.7 An optimizer pass over `children()`

Out of scope, and the optimizer spec §5 already rules pruning out of the optimizer from the other side: "The optimizer's entire responsibility toward pruning is **delivery**." This design *is* that delivery, done without an optimizer.

### 7.8 A two-sided truth lattice, to prune under `NOT`

Rejected: §2.7. Doubles every node's obligation and doubles the surface where a polarity error becomes a wrong answer.

---

## 8. Staged implementation plan

Each stage is independently testable and independently gated. Stages 1 and 2 add no reachable code to any existing binary.

### Stage 0 — break the `params <-> comptime.core` cycle *(prerequisite, severable)*

Move `Param[T]` from `params.mojo` into `comptime/leaves.mojo`. `params.mojo` keeps `Bindings` and gains a non-raising `find(name) -> Optional[DynScalar]`.

*Verify:* `pixi run -e dev precompile` clean; `pixi run -e dev pytest marrow/expr/tests`. Behaviour-neutral, so **0.00%** on all gates.

### Stage 1 — `marrow/kernels/bounds.mojo` *(smallest independently testable increment)*

`Bounds[dt]`, `BoundsKernel`, and `Lt/Le/Gt/Ge/Eq/Ne/And/Or`. Depends on nothing in `expr2`.

*Verify:* `pixi run -e dev pytest marrow/kernels/tests/test_bounds.mojo`. Table-driven, and it must include the one-sided rules explicitly: `Ne` and `Xor` answer `maybe` for every input; `unknown` on either side answers `maybe`; a `point ∩ point` mismatch answers `never` for `Eq` and `maybe` for `Ne`; `all_null` on either side of a comparison answers `never`. Gates: **0.00%** everywhere.

### Stage 2 — `marrow/expr/pruning.mojo` and the node rules

`Truth`, `PruneStats`, `Prunable` (total default), `PrunePredicate`. `ComptimeValue` refines `Prunable`; `PrimitiveValue` gains `bounds`. Overrides on `Column`, `TemporalColumn`, `Literal`, `Param`, `BoolColumn`, `NumericCompare`, `TemporalCompare`, `BoolBinary`.

**This is the correctness core and it is testable with no file I/O.** Cases that must exist:

- `col("a", int64) > lit(5)` against `[0, 3]` -> `never`; against `[0, 9]` -> `maybe`.
- Same predicate against an all-null column -> `never`.
- Same predicate with **no** statistic for `a` -> `maybe`.
- Same predicate where the statistic's dtype disagrees with `T` -> `maybe`, **and the process does not abort** (the `as_type` `debug_assert` guard from §2.3; the single most important test in the file).
- `And` of a prunable and an unprunable conjunct -> follows the prunable one.
- `Not(...)`, `Xor(...)` -> `maybe` for every input.
- `col("a", int64) > param("min-a", int64)` with a binding -> prunes; without a binding and without a default -> `maybe`, **no raise**.
- `col("d", date32) > lit(date32)` -> prunes (the ClickBench shape `expr/` regressed on).

*Verify:* `pixi run -e dev pytest marrow/expr/tests marrow/kernels/tests/test_bounds.mojo`. Gates: **0.00%**.

### Stage 3 — delivery, row-group skipping

`pushdown.mojo`; the `Pushdown` argument on `to_operator` across all eight relations; `Filter` carrying `Optional[PrunePredicate]`; the `filter[V: Value & Prunable]` overload; `ParquetScanOperator` computing a read plan and exposing `groups_read()`.

*Verify:* `test_pushdown.mojo` writes a multi-row-group Parquet into `tmp_path` and asserts:

1. **Results are byte-identical with and without pruning.** This is the one-sided-error property under test.
2. `groups_read()` is strictly smaller with the pruner. Without this the first assertion passes trivially.
3. `filter(p).limit(10)` vs `limit(10).filter(p)` — the second must read **every** group (§4.2's trap).
4. A nested-schema file prunes nothing and returns the right rows (the leaf-map guard).
5. A file whose footer carries no statistics prunes nothing and returns the right rows.

*Gates:* the first stage that can move a number. Budget `query_expr2_streaming` at **<= 0.1%**. **Add `query_expr2_scan`** (fused predicate over a `ParquetScan`, the DCE proof for the fused pruner); `nm -C` on it must show zero `marrow::kernels::interval` symbols.

### Stage 4 — page-level selection

`page_bounds()` -> per-column keep flags -> `RowSelection.intersect` -> `ParquetFile.read(row_selections=…)`.

*Verify:* plus a case where page pruning yields a group that decodes to **zero** rows — the operator must drop it rather than yield an empty morsel.

### Stage 5 — projection pushdown

`Pushdown.needed`; `Project`/`Aggregate` establish it, `Filter`/`Sort` widen it, `Limit` forwards it, `ParquetScan` consumes it.

*Verify:* a scan under `project(["a"])` reads one column chunk; results unchanged for every existing `test_relations.mojo` case.

### Stage 6 — the runtime lane

Comparison and boolean constructors for `RuntimeValue`; the `_prune` thin pointer; conformance over `kernels/interval.mojo`. New gate `query_expr2_runtime`.

*Verify:* the stage-2 test file re-run against runtime-lane predicates, asserting **the same prune/keep decisions**. A divergence between the lanes is the dangerous case.

### Stage 7 — future, not designed here

Bloom filters for equality · interval arithmetic for `+`/`-`/`*` · `StringBounds` · `Not` via comptime De Morgan · reinstating `ParquetScan[leaves]` · multi-file dataset partition pruning.

---

## 9. Open questions

1. **Overload resolution.** Does `filter[V: Value & Prunable](V)` beat `filter(DynValue)` for a concrete node, given `DynValue`'s `@implicit __init__[V: Value]`? Candidate sets should be disjoint, but this must be settled by a probe before stage 3, not assumed.
2. **A trait *default* returning `Bounds[Self.Type.native]`.** `PrimitiveValue.lane`'s *abstract* declaration returns `SIMD[Self.Type.native, W]` and works. A defaulted **body** at that return type is untested. Fallback: declare `bounds` abstract and write the four-line unknown body per struct.
3. **`Prunable` as a supertrait of `ComptimeValue` with a defaulted method, overridden per node.** CLAUDE.md says a *same-signature* override is ordinary; the ambiguity trap needs a *differing return type*.
4. **`Bindings.find`.** Does `std`'s `Dict` expose a non-raising lookup?
5. **`PruneStats` column lookup cost.** `get_field_index` is a linear scan. Per leaf per row group that is free; at stage 4 it could matter.
6. **Marrow's reader discards one-sided bounds.** `ColumnStatistics.from_metadata` keeps min/max "only when both decode". A file with only a max is currently unusable.
7. **`is_min_value_exact` / `is_max_value_exact` are not decoded.** Range pruning is sound without them; bloom-filter equality needs them.
8. **Missing `null_count`.** Treated as *unknown* -> `may_be_null = True` -> conservative. Should be checked against what marrow's own writer emits.
9. **`ParquetScan` has no `LeafSet` parameter in expr2** (`physical.mojo:1011` hardcodes `LeafSet.all()`).
10. **`promote[Int64Type, UInt64Type]` is wrong** (§2.5). Pre-existing; pruning inherits it *consistently*, so it introduces no new error — but it should be filed.
11. **Should `Truth` ever say "definitely TRUE"?** Deliberately no in v1; coupled to §2.7.

---

## 10. Summary

**Pruning is `bind`/`lane` in a second domain; pushdown is `bindings` on a second channel.** Neither is a new mechanism, and that is the argument for this shape over every alternative in §7.

- A node answers for itself through `prune`/`bounds`, typed by the same comptime parameters that type `lane`, folded by the same promotion, over `Bounds[dt]` instead of `SIMD[dt, W]`. The fused lane therefore pays **one** indirect call per row group and **zero** dispatch below it, with `nm -C` on a new `query_expr2_scan` gate as the standing proof.
- The predicate reaches the scan by riding the lowering recursion that `to_operator` already performs, exactly as parameter values do. No plan is rewritten, no node is rebuilt, `DynValue` keeps its five slots.
- Because pruning runs at execution time with `Bindings` in hand, a late-bound predicate prunes exactly as well as a literal one — the regression `expr/` had to fix with a process-global registry cannot occur.
- The whole subsystem is speed-only by construction: delete it and every answer is identical.

**The key tradeoff:** pruning capability is reachable only where the *concrete* predicate type is visible — at `DynRelation.filter`. A predicate that arrives already boxed (a Python frontend calling `filter(DynValue)`) gets no pruner and reads the whole file. That is the price of not putting a sixth slot on `DynValue`, and it is the right price.
