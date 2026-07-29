# AOT Relations: Typed Tables, Named Columns, and Variadic Projection

This document designs a **fully-monomorphized relational layer** on top of the
existing comptime-typed expression nodes — one where a query plan's entire
shape lives in its type, so `.execute()` compiles into specialized, fused code
with no runtime tag dispatch and no runtime schema round-trips.

It is scoped to a **first slice: `Project` + `Filter`** over an in-memory
batch. Aggregations and joins are deliberately out of scope here (see
*Deferred* at the end).

**Status: implemented and tested (milestones 1–5 of 6).** Lives in
`marrow/schema.mojo` (`Schema.from_struct[T]()`) and `marrow/aot/relations.mojo`
(`Table`, `NumericColumn`, `StringColumn`, `Project`, `Filter`), tested in
`marrow/aot/tests/test_relations.mojo` and `marrow/tests/test_schema.mojo`. Two
things changed from the sketch below during implementation — both confirmed
against the pinned toolchain, not judgment calls:

1. **No runtime `Schema` object anywhere.** A `NumericColumn[T]`'s position is
   derived by reflecting the named field on the table struct
   (`reflect[Tbl].field_index[name]()`) in the `Table[Tbl]()` handle, not
   resolved via a `.from_schema(schema)` runtime lookup as first sketched. (The
   position/name are passed as *runtime* constructor args so the column type
   monomorphizes only on dtype — see *Column access* below.)
2. **`t.select(t.a, t.b)` doesn't compile as bare variadic args.** A
   `VariadicPack` captured by one function (`select`'s own `*exprs`) cannot
   be forwarded to another function's variadic parameter (`Tuple`'s
   constructor) in current Mojo — confirmed by triggering the actual compiler
   error, not a design guess. The supported call site is
   `Project(Tuple(t.a, t.b))`. Replicating `Tuple`'s own raw pack-storage
   (`!kgen.struct<... isParamPack>`, `__mlir_op` calls) would recover the bare
   syntax but was declined in favor of the safe, public-API-only path — this
   project restricts that kind of raw/unsafe code to a few vetted files
   (`buffers.mojo`, `views.mojo`, `c_data.mojo`), and `Project`/`Filter`
   wouldn't qualify. See *Project* below.
3. **The two implementations live in top-level `marrow.aot` / `marrow.dyn`
   packages, not nested under a shared `marrow.expr`.** They started as
   `marrow/expr/{values,typed}.mojo` (comptime) and
   `marrow/expr/{runtime,relations,executor}.mojo` (erased) under one parent
   package; once the split was real, keeping `expr` as a nesting level added
   nothing (`marrow.expr.aot.values` vs. `marrow.aot.values`), so the two
   moved up to be siblings of `marrow.kernels`, `marrow.dtypes`, etc., and
   `marrow/expr/` was deleted. `comptime` itself is a reserved Mojo keyword
   and can't be a module name (confirmed: `import marrow.expr.comptime`
   fails to parse) — `aot` / `dyn` were chosen instead, matching this doc's
   own title and `dynamic-dispatch-design.md`. The one cross-package
   dependency is `dyn.values` importing the `NumericValue`/`BoolValue` traits
   from `aot.values` to declare its `FUSED`-boxing constructors' generic
   bounds (see *Bridge to the erased layer*); `aot` imports nothing from
   `dyn`. File map: `marrow/aot/values.mojo`, `marrow/aot/relations.mojo`,
   `marrow/dyn/values.mojo`, `marrow/dyn/relations.mojo`,
   `marrow/dyn/executor.mojo`, each with a `tests/` subdirectory mirroring it.

## Where this sits relative to prior designs

Two earlier docs cover the scalar/plan AOT story:

- `aot-query-compilation.md` — the original two-hierarchy AOT design
  (`ColRef[idx, dt]`, `Binary[op, L, R]`, `Scan[s]`, `Filter[Child, Pred]`).
- `unified-plan-hierarchy.md` — supersedes it with a *single* hierarchy where
  each operator is its own struct with default type params
  (`Add[L = AnyValue, R = AnyValue]`) serving both the runtime and AOT paths.

What actually shipped (`marrow/aot/values.mojo`) took a third route: dedicated
comptime nodes (`NumericColumn[T]`, `Add[L, R]`, `Sub[L, R]`, `Length[S]`) whose type
parameters encode the tree, plus a **boxing bridge** — the `FUSED` tag in
`marrow/dyn/values.mojo` wraps a comptime node into a type-erased `Expr` via trampolines,
so a fused subtree keeps its single-pass execution even when driven through the
runtime path (Python bindings, dynamic SQL). That bridge is the model this
design reuses at the *relational* level.

Three things none of the prior docs cover — the substance of this design:

1. **Deriving a `Schema` from a Mojo struct via compile-time reflection**
   (a `Table` type), rather than hand-writing `Schema(Field(...), ...)`.
2. **Named columns** — `NumericColumn[name, T]` carrying the column *name* in its
   type (prior docs used positional `ColRef[idx, T]`), so output schemas are
   compile-time known and column access can key on the field name.
3. **Variadic projection** — `Project[*Es]` carrying one type parameter per
   output column via a parameter pack, which is the crux of
   `t.select(t.a, t.b)` tracking each column's type independently.

## Motivating payoff

The scalar layer already proves the fusion technique. The relational layer
extends the guarantee up to the plan and kills the **runtime dtype-resolution
gymnastics** the erased path is forced into. In `marrow/dyn/executor.mojo`, the aggregate
builder reaches back into the input schema at runtime just to pick an
accumulator type:

```mojo
# marrow/dyn/executor.mojo — runtime dtype resolution for an aggregate value
var dt = agg_expr.dtype()
if not dt and agg_expr.kind() == LOAD:
    dt = input_schema.fields[Int(agg_expr.kind_data())].dtype.copy()
...
else:
    value_dtypes.append(DynType(float64))   # fallback guess
```

In the typed layer the child's type *is* the answer — no schema round-trip, no
guess. Output-schema construction is likewise compile-time known from the
plan's type instead of assembled at runtime.

## Feasibility (verified against `modular` stdlib, Mojo 1.0.0b3 era)

- **Reflection** (`std/reflection/reflect.mojo`): `reflect[T].field_at[i].T`
  reads directly as a *type* and works even when `T` is a generic parameter
  (backed by `#kgen.struct_field_types`, evaluated after specialization).
  `field_names()`, `field_count()`, and `field_index[name]()` (needs concrete
  `T`) are available. This is the foundation for `Table` → `Schema`.
- **Variadic packs as fields**: `struct Foo[*Es: Trait]` storing
  `var storage: Tuple[*Es]` is exactly how `std/builtin/tuple.mojo` works
  (`!kgen.struct<... isParamPack>`). Iterate with
  `comptime for i in range(Es.__len__()): comptime E = Es[i]`. Confirmed
  *not* transitive, though: a `Tuple[*Es]` field can be built directly from
  fresh args (`Tuple(a, b, c)`), but a pack already captured by an enclosing
  function's own `*args` parameter cannot be forwarded into it — see
  *Project* below.
- **Comptime string params**: `struct NumericColumn[Tbl: AnyType, name:
  StringLiteral, T: ...]` is valid. Confirmed `StringLiteral` (not
  `StaticString` as first assumed) is what `reflect[Tbl].field_index[name]`
  /`.field[name]` actually require.
- **No runtime `__getattr__`** on ordinary structs — `t.a` is strictly the
  declared type of field `a`. There *is* a compile-time hook,
  `__getattr_param__[name: StaticString]()` (used by `AddressSpace` in
  `std/memory/pointer.mojo`), whose return type depends on the name — relevant
  to access style C below.

## Column access — the CRTP finding

The original sketch had three candidate styles (A: column-typed fields with a
runtime `.from_schema(schema)` resolve step; B: marker struct + reflection
accessor; C: marker struct + the `__getattr_param__` comptime hook). Once
milestone 1 (`Schema.from_struct[T]()`) proved `reflect[T].field_index[name]()`
works, the styles collapsed onto style C (marker struct + the
`__getattr_param__` comptime hook), which reaches the strongest ergonomics —
the table is a plain struct of dtype-tag fields, and a separate handle carries
the column accessors:

```mojo
struct Orders:
    var a: Int32Type
    var b: StringType

var t = Table[Orders]()
t.a.index   # 0 — a compile-time constant, baked directly into t.a's generated code
```

`Table[Orders]()` is a handle whose two
`__getattr_param__` overloads reflect the named field on `Orders`:
`reflect[Orders].field_index[name]()` gives the position (passed as a runtime
constructor arg to the returned `NumericColumn[T]` / `StringColumn`) and
`reflect[Orders].field[name].T` gives the dtype, with a `where` clause routing
numeric fields to `NumericColumn` and string fields to `StringColumn`. Both reflection
queries fold to builtin KGEN attributes, so the constraint solver can prove the
`where` clause *during overload selection* — the linchpin that makes the
numeric/string dispatch work.

Two hard constraints shaped this:

- **A handle is required; `Orders()` cannot be the accessor.** `Orders`'s real
  field `a` (of type `Int32Type`) shadows `__getattr_param__` — `Orders().a`
  reads the dtype value, not a column node. `__getattr_param__` only fires for
  *missing* attributes, so the accessors must live on a separate `Table[Orders]`
  handle. The dtype fields exist purely for reflection; they are never
  instantiated, so `Orders` needs no `__init__`.
- **`var` is mandatory.** Mojo rejects bare `a: Int32Type` field declarations,
  so `var a: Int32Type` is as terse as the struct gets.

An earlier sketch had `Column` reflect **on the enclosing table struct itself**
(`var a: Column[Orders, "a", Int32Type]`, a self-referential CRTP form). That
compiled, but forced the name and type to be repeated per field and required a
hand-written `__init__` — strictly more boilerplate than the dtype-tag struct.
It was dropped once the `__getattr_param__` handle (which needs neither) was
shown to dispatch numeric/string via the `where` clause.

A name that doesn't exist on `Orders` is a **compile error**
(`struct 'Orders' has no field named 'x'`), not a runtime exception.

**Rejected alternative — a struct-free `Schema[Field["a", T], ...]` form.** An
earlier iteration also offered a variadic `Schema[*Fields: FieldDescriptor]`
over `Field["a", T]` tags that *was itself* the handle (no companion struct).
It was prototyped and then **removed** — `Table[T]()` is the single, more
idiomatic surface. Two hard limits also made it strictly weaker, and are worth
recording because they constrain any future pack-based surface:

1. It could not do the numeric/string `where`-dispatch. Its field index comes
   from a recursive `def` (`_field_index[name, *Fields]()`), not a builtin
   reflection attribute, and the constraint solver **refuses to evaluate a
   non-builtin function inside a `where` clause** (unlike `Table[T]()`'s
   `reflect[T].field_index[name]()`, which folds). Returning a
   `comptime`-branched type from a helper is likewise rejected ("dynamic type
   values not permitted yet"). So it was **numeric-only** (`Field` bounded to
   `dt.NumericType`, single `__getattr_param__` always returning
   `NumericColumn`).
2. `Schema[*Fields]` has no reflectable named fields (the names live in the
   `Field` type params, not as struct fields), so it cannot use the same
   builtin reflection path as `Table[T]()` at all.

One real compiler limitation surfaced during this work: **a comptime alias
requirement (`comptime name: StringLiteral`) on a trait doesn't resolve
reliably when accessed from a separate generic function parameterized over
that trait** (`E: Named` in some other function, then `E.name`) — it works
fine accessed as `Self.name` from *inside* the concrete node's own method
body, but not from an external generic context. `Named` (used by `Project` to
derive output field names) is therefore declared as a **method**
(`def field_name(self) -> String`), not a bare comptime alias — sidesteps the
limitation entirely, since regular trait-bound method dispatch works fine
(same mechanism `NumericValue.execute()` already relies on).

## Design of the first slice (as implemented)

`marrow/schema.mojo` (`Schema.from_struct[T]()`) and `marrow/aot/relations.mojo`
(`Table`, `Named`, `NumericColumn`, `StringColumn`, `Relation`, `Project`,
`Filter`). Reuses `marrow/aot/values.mojo`'s `NumericValue`/`StringValue`/`BoolValue`
traits and `Add`/`Sub`/`Lt`/`Gt`/`Eq` nodes directly.

### `Schema.from_struct[T]()` — reflection foundation

```mojo
@staticmethod
def from_struct[T: AnyType]() -> Schema:
    comptime r = reflect[T]
    comptime assert r.is_struct(), "Schema.from_struct[T] requires a struct"
    var fields = List[Field]()
    comptime for i in range(r.field_count()):
        comptime FieldT = r.field_at[i].T
        comptime assert conforms_to(FieldT, DataType)
        comptime assert conforms_to(FieldT, Defaultable)
        var dt = _construct_default[FieldT]()   # see note below
        fields.append(Field(String(r.field_names()[i]), dt^, nullable=False))
    return Schema(fields=fields^)
```

For a marker struct whose fields are bare Arrow type markers
(`struct Orders: var a: Int32Type; var b: StringType`), not
`Column`-wrapped — this is a different, simpler shape than the `Table` structs
below, useful standalone.

**Compiler finding:** a bare `FieldT()` call inside the `comptime for` fails —
`FieldT` is only visible as an opaque type during generic-mode checking of the
enclosing generic function (`from_struct[T: AnyType]`), with no constructor
resolvable. Routing the construction through a *separately-instantiated*
generic function bound on `Defaultable & DataType`
(`_construct_default[D: Defaultable & DataType]() -> D: return D()`) makes the
zero-arg constructor visible via the trait witness instead. (An earlier
iteration reused this "route through a generic instantiation" trick in
`Project` too, via `_numeric_col_to_any` / `_string_col_to_any` /
`_named_field_name` helpers; those were removed once the `Column` base trait
let `Project` call `to_array()` / `field_name()` directly on pack-indexed
values — see *Project* below.)

### `Table[T]` handle, `Named` / `Column` traits

The user's schema is a plain struct (no marker trait) — `Table[T]` is the
column-access handle over it, and the two leaf column nodes share a `Column`
base trait so `Project` can treat them uniformly:

```mojo
struct Table[T: AnyType](Copyable, Movable):
    """Column-access handle: Table[Orders]().a -> NumericColumn[...]."""
    ...  # __getattr_param__ overloads (see below)

trait Named:
    def field_name(self) -> String:
        ...

trait Column(Named, Value):
    def to_array(self, batch: RecordBatch) raises -> DynArray:
        ...
```

### `NumericColumn[T]` / `StringColumn` — runtime name, no stored index

```mojo
struct NumericColumn[T: dt.NumericType](Column, Named, NumericValue):
    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var name: String

    def __init__(out self, var name: String):
        self.name = name^

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        var i = batch.schema.get_field_index(self.name)
        return batch.columns[i].as_primitive[Self.T]().values().load[W](idx)

    def field_name(self) -> String:
        return self.name.copy()
```

The column carries only its `name` (a runtime field); the sole type parameter is
the dtype `T`, because only the dtype drives the SIMD `core`. So a query with N
int64 columns instantiates `NumericColumn[Int64Type]` once, not N distinct types
(the field name is metadata that never affects the generated compute). The
position isn't stored — it's resolved by name against `batch.schema` at
execution (a loop-invariant lookup, hoisted out of the fused loop). `Table[Tbl]()`
reflects only the *dtype* (`reflect[Tbl].field[name].T`) to pick the column
type; `col(name, dtype)` takes it explicitly — both produce the same
name-carrying leaf. `StringColumn` mirrors this for the string path
(`StringValue` instead of `NumericValue`, no `T` param since string is a single
physical type). `Lt`/`Gt`/`Eq` (in `marrow/aot/values.mojo`, implementing the new
`BoolValue` trait — bit-packs a `SIMD[bool, W]` mask directly into a `Bitmap`,
same fused-vectorize-loop shape as `NumericValue.execute()`) give `Filter` a
predicate; both accept `NumericColumn`/`StringColumn` children interchangeably with
the unnamed `values.NumericColumn`, since they're generic over any `NumericValue`.

### `Project[*Es]` — the single deliberate erasure boundary

```mojo
struct Project[*Es: Column](Relation):
    var exprs: Tuple[*Self.Es]

    def __init__(out self, var exprs: Tuple[*Self.Es]):
        self.exprs = exprs^

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var cols = List[DynArray]()
        var fields = List[Field]()
        comptime for i in range(Self.Es.__len__()):
            ref e = self.exprs[i]
            var arr = e.to_array(batch)
            fields.append(Field(e.field_name(), arr.dtype()))
            cols.append(arr^)
        return RecordBatch(Schema(fields=fields^), cols^)

    def filter[P: BoolValue](var self, var predicate: P) -> Filter[Self, P]:
        return Filter(self^, predicate^)
```

Bounding on `*Es: Column` (not the broader `Value`) is what makes `execute`
this small: every element exposes `to_array()` and `field_name()`, so there is
no numeric-vs-string `comptime if` and no per-sub-trait erasure helper — each
column executes as its own monomorphized, fused kernel and erases its own
result. The **only** dynamic step is collecting heterogeneous columns into
`List[DynArray]` / `RecordBatch` — inherently heterogeneous and O(#columns),
so `DynArray` is the correct join point. Nothing about per-element compute
goes through tag dispatch.

**Construction takes a pre-built `Tuple[*Es]`, not bare variadic args** —
`Project(Tuple(t.a, t.b))`, not `Project(t.a, t.b)` / `t.select(t.a, t.b)`.
Confirmed by triggering the compiler's actual error (not inferred): a
`VariadicPack` captured by one function's own `*args` parameter cannot be
forwarded to a *different* function's variadic parameter — passing it through
produces `"assigning 1 operand to an unresolvable variadic pack argument"`.
`Tuple`, `Variant`, `Coord`, `UnsafeUnion` in the stdlib all sidestep this by
owning their pack storage directly (`!kgen.struct<... isParamPack>`,
raw `__mlir_op` calls in their own `__init__`/`__getitem_param__`) rather than
wrapping another pack-holding type. `Project` could do the same to recover
`Project(t.a, t.b)`, but that means hand-writing `mark_initialized`,
`kgen.struct.gep`, and friends — the project's own conventions restrict that
class of raw/unsafe code to a few vetted files (`buffers.mojo`, `views.mojo`,
`c_data.mojo`); `Project`/`Filter` don't qualify. Decided (user's call,
2026-07-06) to accept the `Tuple(...)` wrapper rather than replicate it.

An earlier iteration routed each call through per-sub-trait helpers
(`_numeric_col_to_any` / `_string_col_to_any` / `_named_field_name`) as the
same "separately-instantiated generic function" fix as `_construct_default`
above. That turned out to be unnecessary: `to_array()` / `field_name()` are
single `Column`-trait methods and resolve directly on the pack-indexed `e`
(`self.exprs[i]`) with no helper, so the `Column` bound removed all three
helpers *and* the numeric/string branch.

### `Relation` and `Filter[Input, Pred]`

```mojo
trait Relation(ImplicitlyDeletable, Movable):
    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        ...

struct Filter[Input: Relation, Pred: BoolValue](Relation):
    var input: Self.Input
    var predicate: Self.Pred

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var mask = self.predicate.execute(batch).to_dyn()   # fused BoolArray
        var projected = self.input.execute(batch)
        var cols = List[DynArray]()
        for i in range(len(projected.columns)):
            cols.append(filter(projected.columns[i].copy(), mask.copy()))
        return RecordBatch(projected.schema.copy(), cols^)
```

`predicate` evaluates against the *original* input `batch`, not `input`'s
projected output — its column nodes are typed against the source `Table`,
so a filter can reference a column absent from the projection, exactly like
SQL's `WHERE` referencing a column outside `SELECT`. Tested directly (see
`test_filter_predicate_references_dropped_column`).

### Chaining and data entry

`Project(Tuple(...))` builds the leaf; `.filter(pred)` on it returns
`Filter[Project[*Es], Pred]` (`Project` conforms to `Relation` so this
composes). The whole plan is one nested type; `execute(batch)` recurses. Data
enters through the `batch` argument (matching `marrow/aot/values.mojo`); the `Table` is
pure schema, and every column's position is a compile-time constant baked into
the node by the `Table[Tbl]()` handle — no `Schema` object is ever constructed
or consulted at runtime by `NumericColumn`/`Project`/`Filter` themselves.

### Bridge to the erased layer — not yet built (milestone 6)

Mirror the `FUSED` boxing that carries a comptime scalar node into `Expr`: give
the typed relation an `DynRelation(typed_plan)` boxing path so a
statically-built plan can still flow into the existing executor / Python
driver when upstream types aren't known. Harder than the scalar bridge: it
needs `Project`/`Filter` to expose a schema *without* executing (a
`Relation.schema()` computed purely from `Es`'s names/`OutType`s, no batch
touched) and a new `RelationProcessor` variant so `Planner.build()`'s
kind-dispatch (in `marrow/dyn/executor.mojo`) recognizes a boxed typed plan and drives it
through the pull-based morsel pipeline — reconciling the typed layer's
single-shot `execute(batch)` with the erased layer's streaming model. Not
started; the biggest remaining scope decision of the six milestones.

## Semantic note: `select` is row-wise in this slice

`t.select(t.a.sum(), t.b)` mixes a **reduction** (`sum(a)` → scalar) with a
**row-valued** column (`b`), which is ill-defined without grouping. The typed
layer will ultimately split:

- `select(*exprs)` — row-wise; output length = input length.
- `aggregate(keys, *aggs)` — reductions; output length = #groups.

For this slice, `select` is **row-wise only** (columns, arithmetic,
comparisons). `Sum`/`Mean` and an `Aggregate[Keys, *Aggs]` node are the next
slice.

## Milestones

1. ✅ `Schema.from_struct[T]()` in `schema.mojo` — the reflection foundation
   (see `reflect-schema-from-struct.md`). Tested in `test_schema.mojo`.
2. ✅ `Table`, `Named`, `Column`, `Table[Tbl]()`, `NumericColumn[T]` /
   `StringColumn` (runtime name/index) in `marrow/aot/relations.mojo` —
   reflected position via the `__getattr_param__` handle (access style C; see
   *Column access* above). Tested in `marrow/aot/tests/test_relations.mojo`.
3. ✅ `BoolValue` + comparison nodes (`Lt`/`Gt`/`Eq`) in `marrow/aot/values.mojo`,
   mirroring `Add`/`Sub`. Tested in `test_values.mojo`.
4. ✅ `Project[*Es].execute -> RecordBatch`, via `Project(Tuple(...))` (not
   bare variadic `select` — see the `VariadicPack`-forwarding finding above).
   Tested in `test_relations.mojo`.
5. ✅ `Filter[Input, Pred].execute -> RecordBatch` with `.filter(pred)`
   chaining off `Project`. Tested in `test_relations.mojo`, including the
   exact milestone DoD round trip below.
6. ⬜ `DynRelation(typed_plan)` bridge. Not started — see *Bridge to the
   erased layer* above for why it's a bigger scope decision than the rest.

## Definition of done

- ✅ `Orders(Table)` with two columns round-trips
  `Project(Tuple(t.a, t.b)).filter(Gt(t.a, t.b)).execute(batch)` and produces
  a `RecordBatch` equal to the hand-written `marrow/dyn/relations.mojo` equivalent —
  `test_filter_matches_hand_written_relations_equivalent` executes *both*
  paths (typed `execute()` and erased `execute(DynRelation)` via
  `in_memory_table(batch).select(...).filter(...)`) and asserts schema +
  column equality.
- ⬜ A test pinning the plan's *type* to the fully-specialized nested form —
  not yet written; the existing tests assert behavior (correct values,
  correct schema), not the type shape itself. Worth adding before widening
  scope further.
- ✅ Access style C (`__getattr_param__`) is the shipped form: the `Table[Tbl]()`
  handle reaches the target ergonomics (plain dtype-tag struct, no `__init__`)
  and dispatches numeric/string columns via a `where` clause on the reflected
  field type.
- Widen beyond `Project` + `Filter` only after milestone 6 (the bridge) lands
  or is explicitly deferred, and after this note is updated with the
  aggregate/join extension.

## Late binding: prepared plans, `Env`, `Param`, and joins

**Status: design, not yet implemented. Unverified against the toolchain** —
unlike the first slice above, nothing here has been compiled. The compiler
findings that shaped the first slice (reflection, `VariadicPack` forwarding,
`where`-clause folding) all still apply and are assumed, but the specific
shapes below are sketches, not confirmed forms.

### Objective

A plan should be a **precompiled artifact you invoke many times**, reusable
across *both* different data sources and different scalar values — as long as
the schema shape (field names + dtypes) matches. This is the prepared-statement
model: build the query once, bind data and parameters at execution.

The plan's **type is the compile cache key**. Two invocations with the same
schema shape and the same parameter dtypes resolve to the same fully-specialized
nested type, so they share one monomorphized artifact — the generated `core[W]`
machine code is identical across calls; only the bound environment differs.

Half of this is already true. The plan is data-free (`plan.execute(batch)`),
and `NumericColumn` resolves its position *by name* against `batch.schema` at
execution — so a plan already runs on any batch whose fields match by name and
dtype, regardless of column order or extra columns. The two missing pieces are
(1) binding **more than one** data source, and (2) binding **scalar
parameters** instead of baking constants into the plan.

### The generalization: thread an `Env`, not a `RecordBatch`

Both a named column and a bind parameter are the same thing — *a name resolved
at execute against the environment*. A column reads its array from a batch; a
param reads its scalar from a bindings map. So the value threaded down through
`core[W]` and `execute` stops being a bare `RecordBatch` and becomes an
execution environment:

```mojo
struct Env:
    var tables: Catalog    # name -> RecordBatch  (late-bound data sources)
    var params: Bindings   # name -> DynScalar     (late-bound scalar params)
```

(Distinct from the join kernel's `ExecutionContext`, which is the GPU/threading
handle — a different concept. `Env` is the *data + parameter* environment.)

`core[W](self, batch, idx)` becomes `core[W](self, env, idx)`; `execute(self,
batch)` becomes `execute(self, env)`. This ripples through every existing node,
but it is a clean generalization and it is the **load-bearing change** — both
joins and params depend on it. The single-source `execute(batch)` survives as a
one-line convenience wrapper (`execute(Env(Catalog(anon = batch)))`).

The key structural simplification: **only the join subtree is ever
two-input.** A `Join` executes its two children (each pulls its own source from
the catalog) and returns *one* joined batch; every node stacked above the join
(`filter`, `select`) is single-batch again over that joined output. So the
entire existing `col` / `Table` / fused-predicate machinery works unchanged
above a join — the multi-input concern is contained to the join node itself.

### `Param[T]` — scalar late binding, the analog of a named column

`Param[T]` is to a scalar what `NumericColumn[T]` is to a column: the **dtype
lives in the type**, the **name is a runtime field**, and the **value is read
from the environment at execute** — never baked into the plan. So the same
plan object, executed with different `Bindings`, yields different results with
no rebuild and no recompile.

```mojo
struct Param[T: dt.NumericType](NumericValue):
    comptime OutType = Self.T
    comptime NativeType = Self.T.native
    var name: String

    @always_inline
    def core[W: Int](self, env: Env, idx: Int) -> SIMD[Self.NativeType, W]:
        # loop-invariant: resolve once, splat — see "resolution" below
        return env.params.get(self.name).as_primitive[Self.T]().splat[W]()
```

This contrasts with a `Literal[T]` node, which bakes a constant into a runtime
field — fine for genuinely fixed constants, but not reusable across values
without rebuilding the node. `lit()` and `param()` coexist: `lit` for constant
folding, `param` for the late-bound case.

Parameters get the **same struct-reflection surface as tables**, so the mental
model is one rule applied twice — *dtype-tag struct → typed placeholder nodes →
resolved by name against the environment*:

```mojo
# polars-style leaf, dtype spelled explicitly
... .filter(o.amount > param("min_amount", float64))

# or the reflected handle, symmetric with Table[T]()
struct Args:
    var min_amount: Float64Type
    var region:     StringType

var p = Params[Args]()          # mirror of Table[Orders]()
... .filter(o.amount > p.min_amount)   # p.min_amount : Param[Float64Type]("min_amount")
```

`Params[Args]()` reflects the dtype tags off `Args` exactly as `Table[Orders]()`
reflects `Orders` — the same `__getattr_param__` + `reflect[T].field[name].T`
machinery, so it inherits the numeric/string `where`-dispatch for free.

### Resolution must stay loop-invariant

A param is a scalar constant for the whole run; a column's position is fixed for
the whole run. Neither may be resolved per SIMD lane — a `String`-keyed map
lookup inside `core[W]` would run once per lane. Column index resolution has the
same hazard today (`get_field_index` is called inside `core`; the doc asserts
the compiler hoists it, which holds for an integer field-index but is far less
certain for a `String`-keyed param lookup).

The safe design is a **per-execute resolve pass**: walk the typed tree once,
turn each column *name* into an index and each param *name* into its resolved
scalar, then run the fused loop over the resolved form. This keeps the inner
loop pure arithmetic while staying fully late-bound. Whether this can stay
implicit (trust the hoist) or needs an explicit resolved-node representation is
an **open question** to settle when prototyping.

### Joins — the two-input node

A `Join[L: Relation, R: Relation, ...]` is the binary node that collapses two
sources into one batch. Two surfaces for the keys:

```mojo
# Pragmatic (near-term): argument position encodes side, zero new machinery.
# Mirrors dyn's join() and the hash_join kernel (positional left_on/right_on: List[Int]).
o.join(c, left_on = Tuple(o.cust_id), right_on = Tuple(c.cust_id), how = INNER)

# Sugar (north star): needs an On[L, R] node (NOT the fused SIMD Eq — a join
# condition is a hash equijoin, not a per-row compare) with "left operand = left side".
o.join(c, on = o.cust_id == c.cust_id)
```

`left_on`/`right_on` needs nothing new — columns stay plain
`NumericColumn[T]("cust_id")`, the left key resolves against the left child's
output schema and the right against the right's. Composite keys are tuples of
equalities / two grouped tuples.

**Name collisions are the reason to invest in source-tagged columns.** After a
join, colliding names get suffixed (`cust_id` → `cust_id_right`, matching
`dyn`). With plain `col("name", string)` in a post-join `select`, the user must
know and spell the suffixed name — fragile. If columns are parameterized by
their source struct — `NumericColumn[T, Src]`, so `o.cust_id :
NumericColumn[Int64Type, Orders]` and `c.name : StringColumn[Customers]` — then
`c.name` *knows* it came from the right side and auto-resolves to the suffixed
name. `Src` is a phantom type param (like `name` today, it never touches the
SIMD `core`), adding one instantiation axis whose cardinality is just the number
of tables in the query. This is the difference between "works" and the target
ergonomics, and it also makes `on = a == b` unambiguous without a positional
convention and lets `join` statically verify at comptime that a left key's `Src`
is actually the left table.

### Filter/select ordering over a join

The natural order is **JOIN → filter → select** (SQL/polars), i.e. `Filter`
wraps the join and `Project` wraps the filter, each resolving against its
*child's output*. This is cleaner than the first slice's `Project(...).filter(...)`
nesting (where the filter's predicate deliberately evaluates against the
*source* batch — see *Relation and Filter* above). For the join story, flipping
to child-output semantics — the filter sees the full joined schema, the select
picks from filtered rows — is more predictable and generalizes to N-way. This is
a **semantics change to make deliberately**, not silently.

### The reuse pattern (the payoff)

```mojo
# compile ONCE — no data, no scalar values in the type or the object
var query = (
    o.join(c, on = o.cust_id == c.cust_id)
     .filter(o.amount > param("min_amount", float64))
     .select(o.order_id, c.name, o.amount)
)

# reuse the SAME compiled plan across data sources AND parameter values
var jan = query.execute(Env(
    Catalog(orders = jan_orders, customers = customers),
    Bindings(min_amount = 100.0),
))
var feb = query.execute(Env(
    Catalog(orders = feb_orders, customers = customers),
    Bindings(min_amount = 250.0),
))
```

`query` is a single value of a single monomorphized type; `jan`/`feb` differ
only in the bound `Env`. This goes a step beyond DataFusion/substrait
bound-vs-unbound plans or polars `pl.lit`/`pl.col`, which do late binding but
stay runtime-typed — here the late binding is statically typed and
monomorphized.

### Type-safety of the binding — open question

`Bindings(min_amount = 100.0)` hands a `Param[Float64Type]` node an `DynScalar`
at execute; a dtype mismatch is a **runtime** error. The `Params[Args]()` handle
closes half the gap (the node's dtype is comptime-checked against the plan), but
the *provided value* is still checked at execute. Fully-static binding would
make `Bindings` itself a typed struct keyed to `Args`. Same tension applies to
`Catalog`: matching a bound batch's schema against the plan's expected shape is
a runtime check unless the catalog is typed. Decide per surface when
prototyping; runtime-checked bindings are an acceptable first cut.

### Suggested build order

1. **`Env` foundation** — generalize `core[W]`/`execute` from `RecordBatch` to
   `Env` (tables + params). Load-bearing; do first. `Catalog` + single-source
   convenience wrapper.
2. **`lit` + `Param[T]` + `param()` + `Bindings`** — scalar late binding, and
   the constant `t.a > lit(0)` predicate that *Deferred* below still lists as
   missing.
3. **`Join` binary node + `left_on`/`right_on`** — working `join().filter().select()`
   on the `col()` surface, child-output semantics.
4. **Source-tagged columns + `Params[Args]()` + operator sugar** — the north
   star (`on = a == b`, `o.amount > p.min_amount`, auto-suffix resolution).

Prove the "one compiled plan, many executions" property with a reuse test plus a
binary-size comparison against the equivalent `dyn` join (mirroring
`benchmarks/binary_size/`).

## Erased relations over fused values: rewritable plans at comptime size

**Status: the binary-size result is measured (verified). The rewrite designs
(projection/predicate pushdown) are design, not yet implemented.** The prototype
lives in `marrow/aot/erased.mojo`; the measurement in
`benchmarks/binary_size/query_erased_aot.mojo`.

The `Project[*Es]` type-pack encoding makes the whole plan shape a type, which
is what buys the fully-monomorphized, tiny binary — but a *type* can't be
restructured from a runtime decision, so rule-based rewrites (predicate/
projection pushdown, join reordering) are effectively impossible on it. The
question this section answers is whether we can have a **runtime, walkable plan
tree** (rewritable) *without* paying the ~30x binary-size cost of the `dyn`
path.

### The measured result — yes, byte-for-byte

`marrow/aot/erased.mojo` replaces the type pack with plain runtime structs:
`Project`/`Filter` hold `List[AnyValue]`, and `AnyValue` is a **fused-only value
box** — an `ArcPointer` to the concrete node plus a thin trampoline into that
node's own fused `execute()`/`to_array()`. Crucially it carries **no `eval()`
tag-switch** (unlike `dyn.values.Expr`), and the operators execute themselves
single-shot (no `Planner`/`RelationProcessor`). Same query as the other
binary-size variants (`SELECT a, name WHERE a > b`):

| binary | stripped | `__TEXT` | ratio |
|---|---:|---:|---:|
| `query_comptime` (type-pack) | 250,120 B | 229,376 B | 1.0x |
| `query_erased_aot` (runtime tree) | 250,136 B | 229,376 B | 1.0x |
| `query_hybrid` | 7,734,104 B | 7,651,328 B | 30.9x |
| `query_runtime` | 7,734,088 B | 7,651,328 B | 30.9x |

The `__TEXT` is **byte-identical** to the type-pack layer. The per-module
breakdown confirms both open surfaces are absent: `kernels::arithmetic` 0 (no
`Expr.eval` interpreter), `kernels::join`/`groupby`/`hashing` 0 (no `Planner`),
`dyn::*` 0. The entire cost of the runtime plan tree is ~17 symbols (the box,
its trampolines, the two column types, `Gt`).

`query_hybrid` and `query_erased_aot` bracket exactly which knob controls size.
Both fuse the value; hybrid keeps the open driver and saves **nothing**, erased
closes the driver and gets the **full 30x**. So the size win is a property of
**the closed self-executing driver + fused-only value box**, not of encoding the
plan in the type system. **Rewritability and ~250 KB binaries are decoupled.**

### The box is the fusion boundary

Fusion is fully preserved *inside* a box. `AnyValue(Gt(Add(a, b), c))` holds the
root `Gt[Add[…], …]` node, whose type still encodes the whole subtree, so
`box.to_array(batch)` trampolines into one fused vectorize loop computing
`(a+b) > c` per SIMD lane with zero intermediate arrays. The only indirection is
the single root call through the trampoline — O(#boxes), not per-node,
definitely not per-row.

Boxing loses **zero** fusion, because the typed relational layer never fused
across operators in the first place: `Filter.execute` materializes the predicate
mask, then filters each projected column — two passes, two materializations. The
*only* fusion anywhere is intra-expression, and it lives entirely inside one
`AnyValue`. So:

> **erasure boundary = fusion boundary = rewrite granularity.**

Above the boundary — relations, conjunction lists, projection lists — everything
is runtime, walkable, and rewritable, and does not fuse (it is already columnar/
`DynArray`-erased). Below the boundary — inside one `AnyValue` — is a single
monomorphized fused kernel, opaque to rewrites.

This cleanly partitions which rewrites the design admits:

- **Move/drop/reorder whole sub-expressions** — projection pushdown, predicate
  pushdown, conjunction splitting, join reordering. These live *above* the
  boundary and need only *metadata* from each box, never its internals. **Fully
  supported.**
- **Restructure the inside of an expression** — common-subexpression elimination
  across expressions, reassociating `a+b+c`, constant-folding inside a fused
  tree. These need to see *through* the box. They must happen **before boxing**
  (in the typed construction layer, or a dedicated pre-boxing pass — see
  *granularity* below), or they cost fusion.

That partition is the whole trade: you give up cheap intra-expression rewrites
(rare, and recoverable before boxing) to keep fusion + a tiny binary, and you
keep every relation-level rewrite (the ones that matter for pushdown) for free.

### Projection pushdown

Goal: read/carry only the columns actually needed. Mechanism: give `AnyValue` a
`referenced_columns() -> List[String]` accessor (one more trampoline; each node
implements it — a column returns its own name, `Add` returns the union of its
children). Then, at the relational level:

1. Union `referenced_columns()` across the top `Project`'s expressions **and**
   any `Filter` predicates above the scan → the required column set.
2. Narrow the scan to emit only those columns; drop any intermediate projection
   output no expression above references.

The synergy with the **name-resolved columns** (the current leaf design —
`NumericColumn` carries a name, resolves its position against `batch.schema` at
execution) is what makes this cheap: narrowing the scan changes column
*positions*, but every surviving expression still resolves its columns **by
name**, so **the expressions themselves are never rewritten** — no re-indexing,
no touching the fused boxes. Pushdown is purely a relational-tree edit plus a
metadata union.

### Predicate pushdown

Goal: evaluate each filter as early (and on as narrow an input) as possible,
especially below a join. Mechanism:

1. Represent a `Filter` as a **list of conjuncts** (`List[AnyValue]`), not a
   single boxed `AND(...)`. Splitting a conjunction then costs nothing — the
   conjunction is modeled *in the relation*, above the boundary; each conjunct
   stays its own fully-fused box. (Modeling `AND` inside a box instead would
   force the rewrite to see through the box — the wrong side of the boundary.)
2. For each conjunct, test `referenced_columns()` against each side's schema. A
   conjunct referencing only left-side columns pushes into the left input; only
   right-side, into the right; mixed conjuncts stay above the join.
3. Relocate the conjunct box (an `ArcPointer` clone — O(1)). Name resolution
   again means the moved predicate resolves against its new, narrower input with
   no re-indexing.

Both rewrites need from `AnyValue` only: `referenced_columns()` (new),
`dtype()`/`field_name()` (already present), and O(1) cloning (already true — it's
`ArcPointer`-backed). **None of these de-fuse anything** — they are all metadata
over an opaque-but-fused box.

### Granularity is a lowering choice

The box boundary need not be "one box per whole predicate." It is a spectrum:
per-whole-expression (max fusion, expression internals opaque), per-conjunct
(full fusion within each conjunct, conjunct-level pushdown — the sweet spot
above), down to per-node (`dyn`'s model — no fusion, maximal structural
visibility). This suggests a two-level plan, matching how query compilers stage
optimization:

- **Logical plan** — fine-grained enough for the rewrites you want (relations +
  conjunct lists; expression internals visible if intra-expression rewrites are
  needed). All rule-based rewrites run here.
- **Physical plan** — after rewrites, **lower** each maximal fusible expression
  subtree into one `AnyValue`. Boxing *is* the lowering step: it trades
  structural visibility for fusion + the small binary, exactly at the granularity
  the optimizer chose.

For plans authored directly in Mojo (the AOT path's primary case), the "logical"
stage can be as simple as constructing the already-good plan and boxing it; the
staging only earns its keep once runtime/cost-based rewriting is in play.

## Deferred

- **Aggregations** (`Sum`/`Mean`/`Min`/`Max`, `aggregate(keys, *aggs)`).
- **Joins** — designed above (*Late binding*); `unified-plan-hierarchy.md` also
  sketches a typed `HashJoin[Left, Right, LK, RK]`. Fold in once milestone 6
  lands and the `Env` foundation is in place.
- **A comptime `Literal[value]` node.** `marrow/aot/values.mojo` has no comptime literal
  yet (only the runtime layer's `lit()`) — every `Filter` test so far
  compares two columns (`Gt(t.a, t.b)`); a predicate against a constant
  (`t.a > 0`) needs this first. See *Late binding* above, where `lit` sits
  alongside the late-bound `param()`.
- **Nested/`Optional` fields, variable-length types** in `Schema.from_struct`
  — see the open questions in `reflect-schema-from-struct.md`.
- **Bare variadic `select(a, b)` / `Project(a, b)`.** Recoverable only by
  giving `Project` its own raw pack storage (see the `VariadicPack`-forwarding
  finding above) — deliberately not pursued in this slice.

## Limitations

- **Operator-overload repetition.** `__add__`, `__gt__`, etc. must be defined
  on each concrete node (Mojo trait default methods can't yet return
  later-defined structs) — same constraint noted in
  `unified-plan-hierarchy.md`.
- **Dynamic SQL.** A plan parsed from a runtime SQL string produces erased
  `DynRelation` / `Expr` nodes regardless; it gets the runtime path, not AOT
  specialization. The AOT path requires the plan expressed as Mojo types before
  `mojo build`.
- **One-way bridge.** `DynRelation(typed_plan)` (once built, milestone 6)
  would box comptime → erased; the reverse is impossible, since comptime
  types are fixed before compilation finishes.
- **`VariadicPack` forwarding.** Confirmed, not just suspected: a pack
  captured by one function's `*args` cannot be forwarded to a different
  function's variadic parameter. Every "build a heterogeneous collection from
  variadic args and hand it to another type" API in this codebase will hit
  this — `Project(Tuple(...))` is the workaround, not a one-off.
- **Trait comptime-alias access from external generics.** A `comptime name:
  T` trait requirement doesn't resolve reliably when read as `E.name` from a
  separate function generic over `E: SomeTrait` — only from inside the
  concrete type's own method body (`Self.name`). Affects any future trait
  wanting to expose a comptime constant this way; the fix is always the same
  (expose it via a method instead, as `Named.field_name()` does).
