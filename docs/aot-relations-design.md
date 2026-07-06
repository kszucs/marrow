# AOT Relations: Typed Tables, Named Columns, and Variadic Projection

This document designs a **fully-monomorphized relational layer** on top of the
existing comptime-typed expression nodes — one where a query plan's entire
shape lives in its type, so `.execute()` compiles into specialized, fused code
with no runtime tag dispatch and no runtime schema round-trips.

It is scoped to a **first slice: `Project` + `Filter`** over an in-memory
batch. Aggregations and joins are deliberately out of scope here (see
*Deferred* at the end).

**Status: implemented and tested (milestones 1–5 of 6).** Lives in
`marrow/schema.mojo` (`Schema.from_struct[T]()`) and `marrow/aot/table.mojo`
(`Table`, `Column`, `StringColumn`, `Project`, `Filter`), tested in
`marrow/aot/tests/test_typed*.mojo` and `marrow/tests/test_schema.mojo`. Two
things changed from the sketch below during implementation — both confirmed
against the pinned toolchain, not judgment calls:

1. **No runtime `Schema` object anywhere.** `Column[Tbl, name, T]`'s position
   is a `comptime` constant derived by reflecting on the enclosing table
   struct directly (`reflect[Tbl].field_index[name]()`), not resolved via a
   `.from_schema(schema)` runtime lookup as first sketched. See *Column
   access* below — the actual implementation, not what's in the milestones
   section originally drafted.
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
   dependency is `dyn.expr` importing the `NumericValue`/`BoolValue` traits
   from `aot.values` to declare its `FUSED`-boxing constructors' generic
   bounds (see *Bridge to the erased layer*); `aot` imports nothing from
   `dyn`. File map: `marrow/aot/values.mojo`, `marrow/aot/table.mojo`,
   `marrow/dyn/expr.mojo`, `marrow/dyn/relations.mojo`,
   `marrow/dyn/executor.mojo`, each with a `tests/` subdirectory mirroring it.

## Where this sits relative to prior designs

Two earlier docs cover the scalar/plan AOT story:

- `aot-query-compilation.md` — the original two-hierarchy AOT design
  (`ColRef[idx, dt]`, `Binary[op, L, R]`, `Scan[s]`, `Filter[Child, Pred]`).
- `unified-plan-hierarchy.md` — supersedes it with a *single* hierarchy where
  each operator is its own struct with default type params
  (`Add[L = AnyValue, R = AnyValue]`) serving both the runtime and AOT paths.

What actually shipped (`marrow/aot/values.mojo`) took a third route: dedicated
comptime nodes (`Column[T]`, `Add[L, R]`, `Sub[L, R]`, `Length[S]`) whose type
parameters encode the tree, plus a **boxing bridge** — the `FUSED` tag in
`marrow/dyn/expr.mojo` wraps a comptime node into a type-erased `Expr` via trampolines,
so a fused subtree keeps its single-pass execution even when driven through the
runtime path (Python bindings, dynamic SQL). That bridge is the model this
design reuses at the *relational* level.

Three things none of the prior docs cover — the substance of this design:

1. **Deriving a `Schema` from a Mojo struct via compile-time reflection**
   (a `Table` type), rather than hand-writing `Schema(Field(...), ...)`.
2. **Named columns** — `Column[name, T]` carrying the column *name* in its
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
    value_dtypes.append(AnyDataType(float64))   # fallback guess
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
- **Comptime string params**: `struct Column[Tbl: AnyType, name:
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
works, a stronger option emerged that supersedes all three: have
`Column[Tbl, name, T]` reflect **on the enclosing table struct itself**,
self-referentially:

```mojo
struct Orders(Table):
    var a: Column[Orders, "a", Int32Type]
    var b: StringColumn[Orders, "b"]

    def __init__(out self):
        self.a = {}
        self.b = {}

var t = Orders()
t.a.index   # 0 — a compile-time constant, baked directly into t.a's generated code
```

This compiles and works *exactly* as hoped for style A — `t.a` is the column
node, `t.a.index` is `Self.T.index = reflect[Self.Tbl].field_index[Self.name]()`
evaluated at compile time — **with zero runtime `Schema` lookup**, stronger
than the original style A sketch (which resolved `index` via a runtime
`.from_schema(schema)` call). Confirmed against the pinned toolchain: `Orders`
referencing itself as a field's type parameter while `Orders` is still being
defined is legal, because `reflect[Tbl].field_index[name]()` only needs
`Tbl`'s field *names*, available independent of resolving each field's own
type (a `#kgen.struct_field_types` property, evaluated after specialization).
A name that doesn't exist on `Tbl` is a **compile error**
(`struct 'Bad' has no field named 'x'`), not a runtime exception — strictly
better than the runtime-raise a `.from_schema()` step would have given.

Given this, style B (marker struct + reflection accessor) and style C (the
`__getattr_param__` spike) are no longer necessary to reach the target
ergonomics — style A already gets there with zero remaining language risk.
Both stay theoretically available (`Column[Tbl, name, T]`'s `Tbl` parameter
accepts *any* reflectable struct, not just a self-referential one), but
neither was built; not needed once the CRTP form worked.

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

`marrow/schema.mojo` (`Schema.from_struct[T]()`) and `marrow/aot/table.mojo`
(`Table`, `Named`, `Column`, `StringColumn`, `TypedRelation`, `Project`,
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
zero-arg constructor visible via the trait witness instead — same fix reused
throughout `marrow/aot/table.mojo` (see `_numeric_col_to_any` /
`_named_field_name` below).

### `Table` and `Named`

```mojo
trait Table:
    """Marker for a struct whose fields are Column[Self, name, T] /
    StringColumn[Self, name] nodes."""
    pass

trait Named:
    def field_name(self) -> String:
        ...
```

### `Column[Tbl, name, T]` / `StringColumn[Tbl, name]` — zero runtime fields

```mojo
struct Column[Tbl: AnyType, name: StringLiteral, T: dt.NumericType](
    NumericValue, Named
):
    comptime OutType = Self.T
    comptime NativeType = Self.T.native
    comptime index = reflect[Self.Tbl].field_index[Self.name]()

    def __init__(out self):
        pass

    @always_inline
    def core[W: Int](self, batch: RecordBatch, idx: Int) -> SIMD[Self.NativeType, W]:
        return batch.columns[Self.index].as_primitive[Self.T]().values().load[W](idx)

    def field_name(self) -> String:
        return String(t"{Self.name}")
```

`index` is a `comptime` constant — no `var index: Int` field, no runtime
lookup, ever. `StringColumn[Tbl, name]` mirrors this for the string path
(`StringValue` instead of `NumericValue`, no `T` param since string is a
single type). `Lt`/`Gt`/`Eq` (in `marrow/aot/values.mojo`, implementing the new
`BoolValue` trait — bit-packs a `SIMD[bool, W]` mask directly into a `Bitmap`,
same fused-vectorize-loop shape as `NumericValue.execute()`) give `Filter` a
predicate; both accept `Column`/`StringColumn` children interchangeably with
the unnamed `values.Column`, since they're generic over any `NumericValue`.

### `Project[*Es]` — the single deliberate erasure boundary

```mojo
struct Project[*Es: Value](TypedRelation):
    var exprs: Tuple[*Self.Es]

    def __init__(out self, var exprs: Tuple[*Self.Es]):
        self.exprs = exprs^

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var cols = List[AnyArray]()
        var fields = List[Field]()
        comptime for i in range(Self.Es.__len__()):
            comptime E = Self.Es[i]
            ref e = self.exprs[i]
            comptime if conforms_to(E, NumericValue):
                cols.append(_numeric_col_to_any[E](e, batch))
            elif conforms_to(E, StringValue):
                cols.append(_string_col_to_any[E](e, batch))
            fields.append(Field(_named_field_name[E](e), cols[len(cols)-1].dtype()))
        return RecordBatch(Schema(fields=fields^), cols^)

    def filter[P: BoolValue](var self, var predicate: P) -> Filter[Self, P]:
        return Filter(self^, predicate^)
```

Every projected expression executes as its own monomorphized, fused kernel.
The **only** dynamic step is collecting heterogeneous columns into
`List[AnyArray]` / `RecordBatch` — inherently heterogeneous and O(#columns),
so `AnyArray` is the correct join point. Nothing about per-element compute
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

`_numeric_col_to_any` / `_string_col_to_any` / `_named_field_name` are the
same "separately-instantiated generic function" fix as
`_construct_default` above — needed again here because `e`'s type comes from
indexing the `Es` pack.

### `TypedRelation` and `Filter[Input, Pred]`

```mojo
trait TypedRelation(ImplicitlyDeletable, Movable):
    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        ...

struct Filter[Input: TypedRelation, Pred: BoolValue](TypedRelation):
    var input: Self.Input
    var predicate: Self.Pred

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        var mask = self.predicate.execute(batch).to_any()   # fused BoolArray
        var projected = self.input.execute(batch)
        var cols = List[AnyArray]()
        for i in range(len(projected.columns)):
            cols.append(filter(projected.columns[i].copy(), mask.copy()))
        return RecordBatch(projected.schema.copy(), cols^)
```

`predicate` evaluates against the *original* input `batch`, not `input`'s
projected output — its `Column` nodes are typed against the source `Table`,
so a filter can reference a column absent from the projection, exactly like
SQL's `WHERE` referencing a column outside `SELECT`. Tested directly (see
`test_filter_predicate_references_dropped_column`).

### Chaining and data entry

`Project(Tuple(...))` builds the leaf; `.filter(pred)` on it returns
`Filter[Project[*Es], Pred]` (`Project` conforms to `TypedRelation` so this
composes). The whole plan is one nested type; `execute(batch)` recurses. Data
enters through the `batch` argument (matching `marrow/aot/values.mojo`); the `Table` is
pure schema, and every column's position is a compile-time constant baked in
via the `Tbl` self-reference — no `Schema` object is ever constructed or
consulted at runtime by `Column`/`Project`/`Filter` themselves.

### Bridge to the erased layer — not yet built (milestone 6)

Mirror the `FUSED` boxing that carries a comptime scalar node into `Expr`: give
the typed relation an `AnyRelation(typed_plan)` boxing path so a
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
2. ✅ `Table`, `Named`, `Column[Tbl, name, T]` / `StringColumn[Tbl, name]` in
   `marrow/aot/table.mojo` — compile-time-resolved position (the CRTP finding
   above, a stronger result than the originally-sketched access style A).
   Tested in `marrow/aot/tests/test_typed.mojo`.
3. ✅ `BoolValue` + comparison nodes (`Lt`/`Gt`/`Eq`) in `marrow/aot/values.mojo`,
   mirroring `Add`/`Sub`. Tested in `test_bool_values.mojo`.
4. ✅ `Project[*Es].execute -> RecordBatch`, via `Project(Tuple(...))` (not
   bare variadic `select` — see the `VariadicPack`-forwarding finding above).
   Tested in `test_typed_project.mojo`.
5. ✅ `Filter[Input, Pred].execute -> RecordBatch` with `.filter(pred)`
   chaining off `Project`. Tested in `test_typed_filter.mojo`, including the
   exact milestone DoD round trip below.
6. ⬜ `AnyRelation(typed_plan)` bridge. Not started — see *Bridge to the
   erased layer* above for why it's a bigger scope decision than the rest.

## Definition of done

- ✅ `Orders(Table)` with two columns round-trips
  `Project(Tuple(t.a, t.b)).filter(Gt(t.a, t.b)).execute(batch)` and produces
  a `RecordBatch` equal to the hand-written `marrow/dyn/relations.mojo` equivalent —
  `test_filter_matches_hand_written_relations_equivalent` executes *both*
  paths (typed `execute()` and erased `execute(AnyRelation)` via
  `in_memory_table(batch).select(...).filter(...)`) and asserts schema +
  column equality.
- ⬜ A test pinning the plan's *type* to the fully-specialized nested form —
  not yet written; the existing tests assert behavior (correct values,
  correct schema), not the type shape itself. Worth adding before widening
  scope further.
- N/A — the access-style-C (`__getattr_param__`) spike is moot: the CRTP form
  found during milestone 2 already reaches the target ergonomics without it.
- Widen beyond `Project` + `Filter` only after milestone 6 (the bridge) lands
  or is explicitly deferred, and after this note is updated with the
  aggregate/join extension.

## Deferred

- **Aggregations** (`Sum`/`Mean`/`Min`/`Max`, `aggregate(keys, *aggs)`).
- **Joins** — `unified-plan-hierarchy.md` already sketches a typed
  `HashJoin[Left, Right, LK, RK]`; fold it in once milestone 6 lands.
- **A comptime `Literal[value]` node.** `marrow/aot/values.mojo` has no comptime literal
  yet (only the runtime layer's `lit()`) — every `Filter` test so far
  compares two columns (`Gt(t.a, t.b)`); a predicate against a constant
  (`t.a > 0`) needs this first.
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
  `AnyRelation` / `Expr` nodes regardless; it gets the runtime path, not AOT
  specialization. The AOT path requires the plan expressed as Mojo types before
  `mojo build`.
- **One-way bridge.** `AnyRelation(typed_plan)` (once built, milestone 6)
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
