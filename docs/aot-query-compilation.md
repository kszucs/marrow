# AOT Query Compilation via Mojo Metaprogramming

This document describes the design for ahead-of-time (AOT) compilation of
marrow query plans using Mojo's compile-time metaprogramming system.

## Motivation

The current execution engine in `marrow/expr/executor.mojo` is a pull-based
Volcano interpreter:

- `AnyValue` — runtime expression tree, heap-allocated with vtable dispatch
  via function pointers
- `BinaryProcessor.eval()` — runtime `if self.op == ADD` switch over all 20
  op kinds
- `JoinProcessor` — key expressions are `List[Int]` at runtime
- Every morsel pays the full dispatch cost for every expression node

For a fixed, known query like `SELECT * FROM a JOIN b WHERE a.col + 1 = b.col
+ 2`, all of this dispatch is unnecessary. The compiler could emit a single
fused loop with no branches and no intermediate column allocations — if it knew
the plan at compile time.

Mojo's metaprogramming system makes this possible without any source code
generation.

## Core Idea

Encode the query plan as a **Mojo type** using compile-time parameters. The
compiler specializes every operator for that exact type, eliminating all
dispatch:

```mojo
comptime MyJoin = HashJoin[
    Scan[schema_a],
    Scan[schema_b],
    ColRef[0, DType.int64],                               # left key:  a.col
    Binary[ADD, ColRef[0, DType.int64], Literal[1_i64]],  # right key: b.col + 1
]

# Compiler specializes run_plan[MyJoin] — fully inlined, zero dispatch
var plan = MyJoin(left=Scan(a), right=Scan(b))
var result = run_plan(plan^)
```

`mojo build`-ing a file that calls `run_plan[MyJoin]` IS the AOT step. The
binary has no vtable, no runtime op switches, and the predicate `a.col + 1 =
b.col + 2` is fused directly into the probe loop — both key expressions
evaluated in registers.

The predicate is also normalized at type-construction time:
`a.col + 1 = b.col + 2` → left key `a.col`, right key `b.col + 1`, saving one
addition per probe row.

## New Files

### `expr/comptime_expr.mojo`

A compile-time expression type hierarchy that mirrors `expr/values.mojo` but
encodes nodes as Mojo parameter types:

```mojo
trait Expr:
    alias result_dtype: DynType
    fn eval(batch: RecordBatch) raises -> DynArray          # AOT path
    fn to_processor(self) -> AnyValueProcessor              # runtime bridge

struct ColRef[idx: Int, dt: DynType](Expr):
    alias result_dtype = dt
    fn eval(batch: RecordBatch) raises -> DynArray:
        return batch.columns[idx]
    fn to_processor(self) -> AnyValueProcessor:
        return AnyValueProcessor(ColumnProcessor(idx))

struct Literal[value: DynArray](Expr):
    alias result_dtype = value.dtype()
    fn eval(batch: RecordBatch) raises -> DynArray:
        return value
    fn to_processor(self) -> AnyValueProcessor:
        return AnyValueProcessor(LiteralProcessor(value))

struct Binary[op: UInt8, L: Expr, R: Expr](Expr):
    alias result_dtype = promote_dtype[L.result_dtype, R.result_dtype]()
    fn eval(batch: RecordBatch) raises -> DynArray:
        var l = L.eval(batch)
        var r = R.eval(batch)
        comptime if op == ADD:
            return add(l, r)
        elif op == SUB:
            return sub(l, r)
        elif op == EQ:
            return eq(l, r)
        # ... all dead branches eliminated at compile time
    fn to_processor(self) -> AnyValueProcessor:
        # op is a comptime UInt8 — materializes implicitly (trivially copyable)
        return AnyValueProcessor(BinaryProcessor(
            left=L.to_processor(),
            right=R.to_processor(),
            op=op,
        ))

struct Cast[Child: Expr, to: DynType](Expr):
    alias result_dtype = to
    fn eval(batch: RecordBatch) raises -> DynArray:
        return cast(Child.eval(batch), to)
    fn to_processor(self) -> AnyValueProcessor:
        return AnyValueProcessor(CastProcessor(Child.to_processor(), to))
```

The `comptime if op == ADD` in `Binary.eval()` is where the elimination
happens: the compiler sees a constant `op` for each specialization and emits
only the matching branch.

### `expr/relation_plan.mojo`

Parameterized relation types. Each type carries its output schema as an `alias`
and implements a morsel-pull interface:

```mojo
trait RelationPlan:
    alias schema: Schema
    fn pull(mut self) raises -> Optional[RecordBatch]       # AOT path
    fn to_processor(self) -> AnyRelationProcessor           # runtime bridge

struct Scan[s: Schema](RelationPlan):
    alias schema = s
    var source: RecordBatch
    var offset: Int

    fn pull(mut self) raises -> Optional[RecordBatch]:
        if self.offset >= self.source.num_rows():
            return None
        var morsel = self.source.slice(self.offset, MORSEL_SIZE)
        self.offset += morsel.num_rows()
        return morsel

    fn to_processor(self) -> AnyRelationProcessor:
        var rt_schema = materialize[s]()   # Schema may contain List — explicit
        return AnyRelationProcessor(ScanProcessor(self.source, rt_schema))

struct Filter[Child: RelationPlan, Pred: Expr](RelationPlan):
    alias schema = Child.schema
    var child: Child

    fn pull(mut self) raises -> Optional[RecordBatch]:
        while True:
            var batch = self.child.pull()
            if not batch:
                return None
            var mask = Pred.eval(batch.value())   # fully inlined
            var filtered = apply_boolean_mask(batch.value(), mask)
            if filtered.num_rows() > 0:
                return filtered

    fn to_processor(self) -> AnyRelationProcessor:
        return AnyRelationProcessor(FilterProcessor(
            child=self.child.to_processor(),
            predicate=Pred.to_processor(),
            schema_=Child.schema,
        ))

struct HashJoin[
    Left: RelationPlan,
    Right: RelationPlan,
    LeftKey: Expr,
    RightKey: Expr,
](RelationPlan):
    alias schema = concat_schemas[Left.schema, Right.schema]()
    var left: Left
    var right: Right
    var _ht: Optional[HashTable]

    fn pull(mut self) raises -> Optional[RecordBatch]:
        if not self._ht:
            self._ht = self._build()
        return self._probe_next()

    fn _build(mut self) raises -> HashTable:
        var ht = HashTable()
        while True:
            var batch = self.left.pull()
            if not batch:
                break
            # LeftKey.eval is inlined — no vtable, no dispatch
            ht.insert(LeftKey.eval(batch.value()), batch.value())
        return ht

    fn to_processor(self) -> AnyRelationProcessor:
        return AnyRelationProcessor(JoinProcessor(
            left=self.left.to_processor(),
            right=self.right.to_processor(),
            left_key=LeftKey.to_processor(),
            right_key=RightKey.to_processor(),
        ))
```

## Changes to `executor.mojo`

Additive only — add a single entry point for the comptime path:

```mojo
fn run_plan[P: RelationPlan](owned plan: P) raises -> RecordBatch:
    var out = RecordBatch.empty(P.schema)
    while True:
        var batch = plan.pull()
        if not batch:
            break
        out.append(batch.value())
    return out
```

`P` is specialized at compile time — `plan.pull()` inlines the entire operator
tree through `P`'s type.

The existing `AnyValue` / `AnyRelationProcessor` / `Planner` / `execute()`
machinery is unchanged. The runtime interpreted path stays intact.

## One Plan, Two Execution Modes

```mojo
comptime MyJoin = HashJoin[
    Scan[schema_a],
    Scan[schema_b],
    ColRef[0, DType.int64],
    Binary[ADD, ColRef[0, DType.int64], Literal[1_i64]],
]

# AOT path: compiler specializes run_plan for this exact plan type
var plan = MyJoin(left=Scan(a), right=Scan(b))
var result = run_plan(plan^)

# Runtime path: same plan definition, falls back to interpreter
var proc = plan.to_processor()
var result = execute(proc, ctx)
```

`to_processor()` is a deliberate downgrade — you lose fusion but get access to
the existing `execute()` path. Useful for:

- Debugging (step through morsel-by-morsel with the interpreter)
- One-shot queries where compile-time specialization cost isn't worth it
- Dynamic queries assembled at runtime that don't match a pre-compiled type

## Materialization Details

`materialize[comptime_value]()` bridges comptime values to runtime variables.
In the plan types, it comes up in two places:

**Trivial materialization (implicit):** `op: UInt8` in `Binary`, column
indices in `ColRef`, dtype constants — all trivially copyable. They flow into
runtime structs automatically when `to_processor()` constructs a
`BinaryProcessor(op=op, ...)`.

**Non-trivial materialization (explicit):** `Schema` contains `List[ColumnDef]`
(heap-allocated). `Scan.to_processor()` must call `materialize[s]()` explicitly
to convert the comptime `Schema` alias to a runtime variable.

**Zero-cost static path:** If `Schema` is backed by `InlineArray[ColumnDef, N]`
(self-contained, no pointers), `global_constant[s]()` stores it in static
memory once and returns an immutable reference — no allocation on each
`to_processor()` call. This is a strong reason to keep `Schema` as an
`InlineArray`-based type.

## Relationship to `max/kernels/src/pipeline/`

The pattern here is directly analogous to the pipeline scheduler in
`max/kernels/src/pipeline/`:

| Pipeline scheduler | marrow AOT plans |
|---|---|
| `PipelineSchedule` trait | `RelationPlan` trait |
| `build_body() -> List[OpDesc]` | `pull() -> Optional[RecordBatch]` |
| `compile_schedule[S: PipelineSchedule]` | `run_plan[P: RelationPlan]` |
| `comptime if` on resource kinds | `comptime if op == ADD` in `Binary.eval` |
| Schedule is a type parameter | Plan is a type parameter |

The key difference: the pipeline scheduler targets AMD CDNA3-specific hardware
counters and requires a separate CSP solver pass. The marrow plan types are
self-contained — the "scheduling" is just Mojo's normal inlining of
parameterized generics.

## What Stays Unchanged

- `expr/values.mojo` — the runtime `AnyValue` tree is still the right
  representation for dynamically assembled SQL (parsed strings, Python API
  calls). It remains the input to the interpreted executor.
- `expr/executor.mojo` — `Planner`, `AnyRelationProcessor`, `execute()`,
  all processor types — no changes. The new `run_plan` entry point is additive.
- The Python-facing API — `execute(plan, ctx)` continues to work as-is.

## Limitations

- **Dynamic SQL**: A query parsed from a string at runtime cannot become a
  comptime type. The comptime path requires the plan to be known at Mojo
  compile time. The runtime path handles dynamic queries.
- **Schema changes**: If the schema changes, the compiled binary is stale and
  must be rebuilt. The `alias schema: Schema` on each plan type encodes the
  schema into the type — a schema change is a type change.
- **`mojo build --shared-library`**: Shared library output from `mojo build`
  is not a documented public interface. If the goal is a separately loadable
  `.dylib`, this boundary needs to be prototyped before depending on it.
