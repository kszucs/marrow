# Unified Plan Type Hierarchy

This document describes the design for a single expression and relation type
hierarchy that serves both the runtime-interpreted path (Python bindings,
dynamic SQL) and the AOT-compiled path (Mojo compile-time specialization)
without maintaining two parallel type hierarchies.

## Problem with Two Hierarchies

The earlier AOT design (`aot-query-compilation.md`) introduced a parallel set
of compile-time types alongside the existing runtime types:

| Runtime | AOT counterpart |
|---|---|
| `DynValue` / `Column` | `ColRef[idx, T]` |
| `Binary` heap node | `Binary[op, L, R]` zero-size struct |
| `DynRelation` / `Filter` | `CtFilter[Child, Pred]` |

Every new operator required two implementations plus a one-way bridge
(`to_runtime()`). Python bindings always produce the runtime types and can
never reach the AOT path — so the bridge is structurally useless in that
direction.

## Core Idea: Specific Types with Default Type Parameters

Instead of a single `Binary[op, L, R]` or two parallel hierarchies, each
operation is its own type with default type parameters:

```mojo
struct Add[L: Expr = DynValue, R: Expr = DynValue](Expr):
    var left:  L
    var right: R

    def __init__(out self, left: L, right: R):
        self.left  = left
        self.right = right

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return add(self.left.eval(batch), self.right.eval(batch))
```

| `L`, `R` bound to | `var left: L` size | `left.eval()` | Path |
|---|---|---|---|
| `ColRef[0, Int64Type]` | 0 bytes | inlined by compiler | AOT |
| `DynValue` | pointer + vtable refs | vtable dispatch | Runtime |

Same struct, same `eval()`, no switch on an `op` constant. The operation is
the type name. The type parameters decide which execution path is taken.

## The `Expr` Trait

Both `DynValue` and concrete zero-size expression types implement `Expr`:

```mojo
trait Expr:
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        """Evaluate against a batch. Inlined when Self is a concrete type;
        vtable-dispatched when Self is DynValue."""
        ...

    fn to_dyn(self) -> DynValue:
        """Wrap in the type-erased container. Used by plan serialization,
        the Python API, and the runtime bridge."""
        ...
```

### `DynValue` implements `Expr`

`DynValue` gains an `eval()` method that walks its own `ArcPointer` tree
through the interpreter — logic currently in `executor.mojo` that moves onto
the type itself:

```mojo
struct DynValue(Expr):
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return _interpret(self, batch)   # existing executor walk, relocated

    fn to_dyn(self) -> DynValue:
        return self
```

### Concrete expression types

```mojo
struct ColRef[idx: Int, T: DataType](Expr):
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return batch.columns[idx].copy()

    fn to_dyn(self) -> DynValue:
        return DynValue(Column(index=idx, dtype_=T().to_dyn()))

    fn __add__[R: Expr](self, rhs: R) -> Add[Self, R]:   return Add(self, rhs)
    fn __sub__[R: Expr](self, rhs: R) -> Sub[Self, R]:   return Sub(self, rhs)
    fn __mul__[R: Expr](self, rhs: R) -> Mul[Self, R]:   return Mul(self, rhs)
    fn __truediv__[R: Expr](self, rhs: R) -> Div[Self, R]: return Div(self, rhs)
    fn __eq__[R: Expr](self, rhs: R) -> Equal[Self, R]:  return Equal(self, rhs)
    fn __ne__[R: Expr](self, rhs: R) -> NotEqual[Self, R]: return NotEqual(self, rhs)
    fn __lt__[R: Expr](self, rhs: R) -> Less[Self, R]:   return Less(self, rhs)
    fn __le__[R: Expr](self, rhs: R) -> LessEq[Self, R]: return LessEq(self, rhs)
    fn __gt__[R: Expr](self, rhs: R) -> Greater[Self, R]: return Greater(self, rhs)
    fn __ge__[R: Expr](self, rhs: R) -> GreaterEq[Self, R]: return GreaterEq(self, rhs)
    fn __and__[R: Expr](self, rhs: R) -> And[Self, R]:   return And(self, rhs)
    fn __or__[R: Expr](self, rhs: R) -> Or[Self, R]:     return Or(self, rhs)
    fn __neg__(self) -> Negate[Self]:                    return Negate(self)
    fn __invert__(self) -> Not[Self]:                    return Not(self)
    fn cast[To: DataType](self) -> Cast[Self, To]:       return Cast(self)

struct Add[L: Expr = DynValue, R: Expr = DynValue](Expr):
    var left: L
    var right: R
    def __init__(out self, left: L, right: R):
        self.left = left; self.right = right
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return add(self.left.eval(batch), self.right.eval(batch))
    fn to_dyn(self) -> DynValue:
        return DynValue(RtBinary(op=ADD, left=self.left.to_dyn(), right=self.right.to_dyn()))
    # operator overloads — same set as ColRef above

struct Greater[L: Expr = DynValue, R: Expr = DynValue](Expr):
    var left: L
    var right: R
    def __init__(out self, left: L, right: R):
        self.left = left; self.right = right
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return greater(self.left.eval(batch), self.right.eval(batch))
    fn to_dyn(self) -> DynValue:
        return DynValue(RtBinary(op=GT, left=self.left.to_dyn(), right=self.right.to_dyn()))
    # operator overloads ...

# Sub, Mul, Div, Equal, NotEqual, Less, LessEq, GreaterEq, And, Or follow
# the same pattern. Each eval() calls the matching single kernel directly —
# no op switch.

struct Negate[C: Expr = DynValue](Expr):
    var child: C
    def __init__(out self, child: C): self.child = child
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return negate(self.child.eval(batch))
    fn to_dyn(self) -> DynValue:
        return DynValue(RtUnary(op=NEG, child=self.child.to_dyn()))

struct Cast[C: Expr = DynValue, To: DataType = DynType](Expr):
    var child: C
    def __init__(out self, child: C): self.child = child
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray:
        return cast(self.child.eval(batch), To().to_dyn())
    fn to_dyn(self) -> DynValue:
        return DynValue(RtCast(child=self.child.to_dyn(), to=To().to_dyn()))
```

## Usage: Same Constructor, Two Paths

```mojo
# Runtime — default DynValue params, heap-backed, interpreter-executed
var expr = col(0) + lit(1) > lit(0)
# type: Greater[Add[DynValue, DynValue], DynValue]

var plan = in_memory_table(batch).filter(expr)
var result = execute(plan, ctx)

# AOT — concrete params inferred from the chain, zero-size, fully inlined
alias expr = col[0, Int64Type]() + lit_i[1]() > lit_i[0]()
# type: Greater[Add[ColRef[0, Int64Type], IntLit[1]], IntLit[0]]

alias plan = Scan[orders_schema](batch).filter(expr)
# type: Filter[Scan[orders_schema], Greater[Add[...], IntLit[0]]]

var result = run_plan(plan)
```

The expression builder chain is identical. `col(0)` returns `DynValue`;
`col[0, Int64Type]()` returns `ColRef[0, Int64Type]`. Everything downstream
follows from that single choice.

## The `Relation` / `CtRelation` Trait Split

Relation operators need two levels of schema guarantee:

```mojo
trait Relation(Movable):
    fn schema(self) -> Schema: ...
    fn pull(mut self) raises -> Optional[RecordBatch]: ...
    fn to_any_relation(self) -> DynRelation: ...

trait CtRelation(Relation):
    alias schema: Schema               # compile-time — required by run_plan
    fn schema(self) -> Schema:
        return Self.schema             # default: alias satisfies the runtime method
```

`DynRelation` gains `pull()` that delegates to the interpreter, mirroring
what `DynValue.eval()` does for expressions.

### Unified relation operators

```mojo
struct Filter[
    Child: Relation = DynRelation,
    Pred:  Expr     = DynValue,
](Relation):
    var child: Child
    var pred:  Pred

    def __init__(out self, child: Child, pred: Pred):
        self.child = child; self.pred = pred

    fn schema(self) -> Schema:
        return self.child.schema()

    fn pull(mut self) raises -> Optional[RecordBatch]:
        while True:
            var maybe = self.child.pull()
            if not maybe: return None
            var batch = maybe.value()
            var mask  = self.pred.eval(batch)   # inlined or dispatched
            var out   = filter_batch(batch, mask)
            if out.num_rows() > 0: return out

    fn to_any_relation(self) -> DynRelation:
        return DynRelation(RtFilter(
            input=self.child.to_any_relation(),
            predicate=self.pred.to_dyn(),
            schema_=self.child.schema(),
        ))

    fn filter[P: Expr](owned self, pred: P) -> Filter[Self, P]:
        return Filter(child=self^, pred=pred)

    fn join[
        Right: Relation, LK: Expr, RK: Expr
    ](owned self, right: Right, left_on: LK, right_on: RK
    ) -> HashJoin[Self, Right, LK, RK]:
        return HashJoin(left=self^, right=right, left_on=left_on, right_on=right_on)

    fn limit(owned self, comptime n: Int) -> Limit[Self, n]:
        return Limit(child=self^)

struct HashJoin[
    Left:     Relation = DynRelation,
    Right:    Relation = DynRelation,
    LeftKey:  Expr     = DynValue,
    RightKey: Expr     = DynValue,
](Relation):
    var left:      Left
    var right:     Right
    var left_key:  LeftKey
    var right_key: RightKey
    ...

struct Scan[s: Schema](CtRelation):
    alias schema: Schema = s
    var source: RecordBatch
    var offset: Int
    def __init__(out self, source: RecordBatch):
        self.source = source; self.offset = 0
    fn pull(mut self) raises -> Optional[RecordBatch]: ...
    fn to_any_relation(self) -> DynRelation: ...
    fn filter[P: Expr](owned self, pred: P) -> Filter[Self, P]:
        return Filter(child=self^, pred=pred)
    fn join[...] ...; fn limit(...) ...
```

`Filter` satisfies `CtRelation` when `Child: CtRelation` — Mojo's constraint
propagation handles this automatically. `run_plan[P: CtRelation]()` therefore
works on any plan tree composed entirely of concrete types.

## `run_plan` and `execute`

```mojo
# AOT entry point — P must be CtRelation (schema known at compile time)
def run_plan[P: CtRelation](owned plan: P) raises -> RecordBatch:
    var out = RecordBatch.empty(P.schema)
    while True:
        var batch = plan.pull()
        if not batch: break
        out.append(batch.value())
    return out^

# Runtime entry point — any Relation, schema resolved at runtime
def execute(owned plan: DynRelation, ctx: ExecutionContext) raises -> RecordBatch:
    var out = RecordBatch.empty(plan.schema())
    while True:
        var batch = plan.pull()
        if not batch: break
        out.append(batch.value())
    return out^
```

`execute` becomes a trivial loop once `DynRelation.pull()` exists.

## What Changes in the Existing Code

### `expr/values.mojo`

- Add the `Expr` trait.
- Add `eval(batch) raises -> DynArray` and `to_dyn() -> DynValue` to `DynValue`.
- Replace the monolithic `Binary` / `Unary` concrete structs with the
  parameterized `Add[L, R]`, `Sub[L, R]`, `Greater[L, R]`, `Negate[C]`,
  etc. The existing `RtBinary` / `RtUnary` names can be kept as aliases or
  renamed to `_RtBinary` (internal use only, bridged via `to_dyn()`).
- `ColRef`, `IntLit`, `FloatLit`, `Cast` are new; `Column`, `Literal`,
  `Cast` (existing runtime nodes) remain but are now the `DynValue`-param
  defaults.

### `expr/relations.mojo`

- Add `Relation` and `CtRelation` traits.
- Add `pull()` to `DynRelation`.
- Replace `Filter`, `Join`, etc. with parameterized versions with default
  `DynRelation` / `DynValue` type params.
- Existing concrete runtime nodes (`RtFilter`, `RtJoin`, …) become the
  targets of `to_any_relation()`.

### `expr/executor.mojo`

- `execute(plan, ctx)` delegates to `plan.pull()` in a loop — the
  interpreter logic has moved onto `DynValue.eval()` and
  `DynRelation.pull()`.
- `run_plan[P: CtRelation](plan)` is additive.

### `expr/aot.mojo`

- Removed. `run_plan`, `Scan`, factory helpers (`col[idx, T]()`,
  `lit_i[val]()`) move to `relations.mojo` / `values.mojo`.

## Merging Kernels with Expressions

The current codebase separates concerns across three layers for every
element-wise operation:

1. **SIMD kernel** — `_add[T, W](SIMD[T,W], SIMD[T,W])` in
   `kernels/arithmetic.mojo`
2. **Typed wrapper** — `add[T: PrimitiveType](PrimitiveArray[T],
   PrimitiveArray[T])`
3. **Runtime dispatch** — `add(DynArray, DynArray)` via
   `binary_array_dispatch`
4. **Expression node** — `Add.eval()` calls the runtime dispatch version

The dispatch layer exists entirely because `DynValue` erases the dtype at
runtime. When the expression type parameters are concrete, the dtype is known
at compile time — the dispatch is unnecessary, and all four layers can
collapse into one struct method.

### Associated output array type on `Expr`

The mechanism that enables this is an associated output array type on the
`Expr` trait. Concrete expression types override it with a specific array
type; `DynValue` keeps the default `DynArray`:

```mojo
trait Expr:
    alias OutArray: AnyType = DynArray  # default: type-erased

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> Self.OutArray: ...
    fn to_dyn(self) -> DynValue: ...

struct DynValue(Expr):
    alias OutArray = DynArray           # type-erased, runtime dispatch in eval
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> DynArray: ...

struct ColRef[idx: Int, T: DataType](Expr):
    alias OutArray = PrimitiveArray[T]  # concrete — no dispatch needed
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> PrimitiveArray[T]:
        return batch.columns[idx].as_primitive[T]()

struct IntLit[val: Int](Expr):
    alias OutArray = PrimitiveArray[Int64Type]
    @always_inline
    fn eval(self, batch: RecordBatch) raises -> PrimitiveArray[Int64Type]: ...
```

### `BinaryExpr` sub-trait: `eval_arrays` receives typed operands

A `BinaryExpr` sub-trait exposes `eval_arrays` which receives the
already-evaluated left and right arrays. `eval()` in the default
implementation simply evaluates the two children and forwards them:

```mojo
trait BinaryExpr[L: Expr, R: Expr](Expr):
    @always_inline
    fn eval_arrays(
        self, l: L.OutArray, r: R.OutArray
    ) raises -> Self.OutArray:
        ...

    @always_inline
    fn eval(self, batch: RecordBatch) raises -> Self.OutArray:
        return self.eval_arrays(
            self.left.eval(batch),   # type: L.OutArray
            self.right.eval(batch),  # type: R.OutArray
        )
```

`Add` then contains only the operation itself:

```mojo
struct Add[L: Expr = DynValue, R: Expr = DynValue](BinaryExpr[L, R]):
    alias OutArray = _promote[L.OutArray, R.OutArray]()
    # OutArray = PrimitiveArray[Int64Type] when both sides are typed Int64
    # OutArray = DynArray when either side is DynValue

    var left:  L
    var right: R
    def __init__(out self, left: L, right: R):
        self.left = left; self.right = right

    @always_inline
    fn eval_arrays(
        self, l: L.OutArray, r: R.OutArray
    ) raises -> Self.OutArray:
        return add(l, r)
        # When L.OutArray = PrimitiveArray[Int64Type]:
        #   add(PrimitiveArray[T], PrimitiveArray[T]) — typed, no dispatch
        # When L.OutArray = DynArray:
        #   add(DynArray, DynArray) — runtime dtype switch

    fn to_dyn(self) -> DynValue: ...
    fn __add__[R2: Expr](self, rhs: R2) -> Add[Self, R2]: return Add(self, rhs)
    fn __gt__[R2: Expr](self, rhs: R2) -> Greater[Self, R2]: return Greater(self, rhs)
    ...
```

`Greater`, `Sub`, `Mul`, `Equal` etc. are identical in structure — only
`eval_arrays` differs, and each body is a single typed kernel call.

`_promote[L.OutArray, R.OutArray]()` is a comptime function that returns the
promoted array type: `PrimitiveArray[promoted_dtype]` when both sides are
typed primitives, `DynArray` when either side is type-erased.

### What this removes

| Current | Merged |
|---|---|
| `_add[T, W](SIMD[T,W], SIMD[T,W])` in `kernels/` | `Add.eval_arrays` body is `l + r` directly |
| `add[T](PrimitiveArray[T], PrimitiveArray[T])` typed wrapper | subsumed by `add(l, r)` overload resolution |
| `add(DynArray, DynArray)` runtime dispatch function | still needed for `DynArray` operands, but lives once in the `add` overload, not repeated per expression type |
| `binary_array_dispatch` helper | no longer needed as a separate helper — the overloaded `add(DynArray, DynArray)` already handles it |
| `Add.eval()` calls `add(l, r)` via dispatch | `BinaryExpr.eval()` default handles it; `Add` only implements `eval_arrays` |

### What stays in `kernels/`

Operations that are not scalar expression nodes — aggregations, sorting,
joins, filtering, string kernels — remain in `kernels/` as standalone
functions. The merge applies only to element-wise arithmetic, comparison, and
boolean operations that map one-to-one onto expression node types.

## Limitations

**Operator overload repetition.** `__add__`, `__gt__`, etc. must be defined
on every concrete `Expr` implementor because Mojo traits do not currently
support default method implementations whose return types reference structs
defined later in the same file. A future Mojo improvement to trait default
methods would allow a single definition on `Expr`.

**Schema compile-time availability.** `alias schema: Schema` on `CtRelation`
requires `Schema` to be evaluable by the Mojo interpreter at elaboration
time. `Schema` contains `List[Field]` (heap-allocated), which the interpreter
handles but which cannot be implicitly materialized — `materialize[s]()` must
be called in `to_any_relation()`. If `Schema` is later backed by a
self-contained `InlineArray[FieldDef, N]`, materialization becomes free.

**Dynamic SQL.** A query assembled at runtime from a parsed SQL string
produces `DynValue` / `DynRelation` nodes regardless. The default-param
structs handle this correctly — they just do not receive the AOT
specialization. The AOT path requires the plan to be expressed as Mojo types
before `mojo build` runs.

**One-way bridge.** `to_dyn()` / `to_any_relation()` convert comptime types
to runtime. The reverse is impossible: a type-erased `DynValue` tree cannot
be promoted to a parameterized comptime type at runtime because comptime types
must be fixed before compilation finishes.
