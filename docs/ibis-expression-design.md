# marrow.expr.ibis — a fused, ibis-like typed expression system

## Context

`marrow/expr/values.mojo` today fuses expressions into a single vectorized pass, but the core
abstraction is **numeric-only**: `FusedBinary[K, L: NumericValue, R: NumericValue]` bakes "numeric"
into the node, forcing a parallel `FusedCompare`, and string comparison into a hand-rolled per-lane
scalar loop (pseudo-SIMD). There is no cross-family (`string→bool`, `temporal→int`), nested
(struct/list), or non-fusable-boundary story.

We are porting a **representative subset of ibis's typed expression system** into marrow's typesystem
and fusing everything fusable. Organizing principle:

- **`ibis.expr.types` → marrow traits** (the value families: `NumericValue`, `BoolValue`,
  `StringValue`, `TemporalValue`, `StructValue`, `ListValue`).
- **`ibis.expr.operations` → marrow structs** (the nodes — mostly `comptime` aliases of a few
  generic templates).
- **`marrow.kernels` → compute** (nodes are pure glue that route to a kernel).

Built **standalone** in a new `marrow/expr/ibis.mojo`, proven end-to-end on its own tests, meshed
with the relational/execution layers later. Ignore `marrow/expr/dynamic.mojo`.

## Non-negotiable constraints

1. **types→traits, operations→structs, kernels→compute.** All execution lives in `marrow.kernels`.
2. **No `.to_expr()`.** A node struct *statically conforms* to its family trait, so it **is** the
   typed API. `(a+b) < c` chains via trait operators — no wrapper classes, no dtype→class lookup.
3. **No pseudo-vectorized code.** Fixed-width-output / variable-length-input ops (string `==`,
   `startswith`, `contains`, `ArrayContains`) execute **eagerly once** via a kernel to a fixed-width
   array; that array re-enters fusion as a lane leaf. Fusion resumes at the first fixed-width boundary.
4. **No enums, no runtime tags.** Everything in the typesystem: `comptime` params, trait conformance,
   `comptime if`, comptime type equality (`==`). No `regime` field, no name-strings.
5. **Declarative / boilerplate-minimal.** Adding an operation = kernel functor + one `comptime` alias.

### Strictness rules (from review — apply throughout)

- **`Value` carries `comptime OutType` + a typed `execute()`.** `Value` declares an associated output
  dtype `comptime OutType: DataType` and an associated output-array type, with
  `execute(self, batch) -> <that typed array>` — so `execute()` is uniform on *every* `Value` and
  generic code can run any node. `core[W]` stays as the SIMD lane primitive (only on lane families).
- **No `DynArray`, no type-erased types.** `execute()` returns the family's concrete typed array
  (`PrimitiveArray[OutType]` / `BoolArray` / `StringArray` / `StructArray` / `ListArray`); typed
  scalars, not `DynScalar`. No `AnyValue` box. Where an existing `arrays.mojo` API returns `DynArray`
  (e.g. `StructArray.field`), downcast to the typed array immediately at the call site.
- **No `Optional` inputs/outputs.** Strict, concrete signatures everywhere (incl. the materialization
  cache — a typed array initialized 0-length, overwritten in `prepare`).
- **No pruning.** Out of scope.
- **`BoolValue` and `NumericValue` are entirely distinct families** — sibling traits, never merged
  into a shared execution trait, never one a subtype of the other. Disjoint operator surfaces
  (`+`/`<` vs `&`/`~`) and disjoint packaging (`PrimitiveArray` vs bit-packed `BoolArray`). "Bool is
  a number" is recovered only by explicit `BoolToNum`/`NumToBool` bridge nodes.
- **Do not modify existing files.** `ibis.mojo` is purely additive: it *imports and reuses* existing
  kernel structs (`AddKernel.core`, `EqKernel.core`, `equal(StringArray,…)`, …) but changes nothing
  in `dtypes.mojo`, `values.mojo`, or `marrow/kernels/*`. In particular **do not** add
  `BoolType : PrimitiveType` — `Value.OutType` is bound by the base `DataType` trait (which `BoolType`
  already satisfies), and bool packaging lives in `BoolValue`'s own `execute`. New kernels
  (`StartsWithKernel`, …) go in **new** files, not edits to existing ones.

## Fusability taxonomy (the design axis — bucket = which trait/branch, not a runtime tag)

| Bucket | Examples | Mechanism |
|---|---|---|
| **true-SIMD lane** (fixed→fixed) | Add/Sub/Mul/Div, Neg/Abs, compares over numeric, And/Or/Not, Cast, IsNull, temporal `Extract*`, `StringLength`/`ArrayLength` (read offsets) | per-lane `core[W]`; fuses |
| **materialize-once → lane leaf** (var-len in, fixed-width out) | string `==`, `StartsWith`, `Contains`, `ArrayContains` | eager kernel `apply` **once** → typed array → lane leaf via `prepare()` cache |
| **variable-length output** | `Upper`/`Lower`, `Substring` | `StringValue` node; `execute→StringArray`; not a lane |
| **reduction / boundary** (N→1) | `Sum`, `Mean`, `Min`, `Max` | consume a fused lane's `execute`, produce a typed scalar |
| **nested / structural** | `StructField` (compile-time name → child-array select, **free**), `ArrayLength` (offset lane), `MapGet` (boundary) | struct = static/free; map = compute |

## Trait tower (value families strictly distinct; `Value` carries `OutType` + `execute`)

```
Value (Copyable, ImplicitlyCopyable, Movable, Writable)
  ├─ comptime OutType: DataType                 # associated output dtype (BoolType ok — base DataType)
  ├─ comptime Output: AnyType                   # associated output ARRAY type
  ├─ execute(self, batch) -> Self.Output        # THE typed verb — uniform on every Value
  ├─ name(self) -> String                       # default ""
  └─ prepare(self, batch)                        # default no-op; composites recurse; boundaries fill cache
      ├─ NumericValue(Value)   OutType: NumericType; comptime OutNative = OutType.native
      │     Output = PrimitiveArray[OutType]
      │     core[W](batch,idx) -> SIMD[OutNative,W]   ;   execute default = vectorize core
      │     __add__/__sub__/__mul__/__lt__/… ; cast(...)
      ├─ BoolValue(Value)      OutType = BoolType ; comptime OutNative = DType.bool
      │     Output = BoolArray
      │     core[W](batch,idx) -> SIMD[DType.bool,W]  ;   execute default = vectorize core, bit-packed
      │     __and__/__or__/__invert__
      ├─ TemporalValue(Value)  OutType: TemporalType (int32/int64 native) ; like Numeric
      ├─ StringValue(Value)    Output = StringArray ; (no lane) execute; length()/upper()/…
      ├─ StructValue(Value)    Output = StructArray ; .field → child-family leaf
      └─ ListValue(Value)      Output = ListArray  ; length()/contains()
```

`NumericValue`/`BoolValue` are **separate** sub-traits (no shared lane super-trait). Each provides
its own `execute` default (numeric/temporal store into `PrimitiveArray`; bool bit-packs into
`BoolArray`). If the numeric and temporal loop bodies turn out identical, factor one private helper
*then* — an implementation detail, not an up-front architectural concept.

## Operation structs — ONE struct per operation via conditional conformance

The answer to "do we need `NumericBinary` vs `BoolBinary`, or `NumericEqual` vs `StringEqual`?" is
**no** — conditional conformance (a just-shipped Mojo nightly feature) collapses them:

**(a) Unify the OUTPUT family** — one generic binary/unary node whose family is chosen by the
kernel's **existing** trait (no new markers):

```mojo
struct FusedBinary[K: Kernel, L: Value, R: Value](
    NumericValue where conforms_to(Self.K, BinaryKernel),                       # core -> SIMD[T,W]
    BoolValue    where conforms_to(Self.K, BinaryCompareKernel)
                     or conforms_to(Self.K, BoolBinaryKernel),                  # core -> SIMD[bool,W]
):
    # Phase-0 finding: TWO disjoint `comptime OutType` witnesses are rejected
    # ("invalid redefinition"). Use ONE base `comptime OutType: DataType` computed
    # via a `comptime if` helper, plus a family-specific numeric type as a SINGLE
    # where-guarded witness (the proven `Foo.SIZE` pattern):
    comptime OutType: DataType = _pick_out[Self.K, Self.L]()          # single, computed
    comptime Num: NumericType where conforms_to(Self.K, BinaryKernel) = Self.L.Num
    # type refinement makes Self.K.core callable at the right signature in each branch

# Phase-0 validated on Mojo 1.0.0b3.dev2026070506 (no nightly update needed):
#   dual conditional conformance ✅ (shared ancestor Value listed with the disjoined
#   constraint); conditional methods ✅; associated Output type + uniform
#   execute()->Self.Output ✅; ArcPointer cache mutation from a borrowed self,
#   visible across .copy() ✅.
comptime Add  = FusedBinary[AddKernel, _, _]   # → NumericValue (exposes +, <)
comptime Less = FusedBinary[LtKernel,  _, _]   # → BoolValue    (exposes &, |)
comptime And  = FusedBinary[AndKernel, _, _]   # → BoolValue
```

**(b) Unify the INPUT family for ops that span families** — one `Equal` for numeric *and* string
operands, choosing lane-vs-boundary by `comptime if conforms_to(L, NumericValue)`:

```mojo
struct Equal[L: Value, R: Value](BoolValue):        # always a BoolValue
    var left: Self.L
    var right: Self.R
    var cache: ArcPointer[BoolArray]                 # used only on the string branch; 0-length otherwise
    def core[W](self, batch, idx) -> SIMD[DType.bool, W]:
        comptime if conforms_to(Self.L, NumericValue):
            return EqKernel.core[Self.L.OutNative, W](self.left.core[W](batch,idx),
                                                      self.right.core[W](batch,idx).cast[...]())
        else:                                        # StringValue operands
            return self.cache[].values().load[W](idx)         # filled in prepare (materialized once)
    def prepare(self, batch) raises:
        comptime if conforms_to(Self.L, NumericValue):
            self.left.prepare(batch); self.right.prepare(batch)
        else:
            self.cache[] = equal(self.left.execute(batch), self.right.execute(batch))  # kernel, once
```

So `Equal(a_i64, b_i64)` fuses as a lane; `Equal(s, t)` materializes once via the string kernel and
re-enters fusion as a lane leaf — **one struct**, no pseudo-SIMD, output family fixed (`BoolValue`),
value families never blurred. `StartsWith`/`Contains` (string-only, no numeric variant) are their own
`BoolValue` boundary structs with the same cache+`prepare`+`core` shape.

Leaves: `NumericColumn[T]`, `Literal[T]`, `StringColumn`, `StructColumn`, `ListColumn`, `col()`,
`Table[T]` (reflection handles). Existing kernel `core`s reused verbatim by import.

## Fusion lives in the expression system, not the kernels

"Fusable" is **not** a kernel property — a kernel is just compute (`core` SIMD functor + eager
`apply`). Whether an op *fuses* (the node calls `K.core` in the vectorize loop) or *materializes*
(the node calls `K.apply` once) is decided by the **expression node**. So there are **no new
kernel-side marker traits and no `fusion_markers.mojo`**:

- **Output family** is read off the kernel's **existing** trait: `FusedBinary` conditionally conforms
  to `NumericValue` when `conforms_to(K, BinaryKernel)`, to `BoolValue` when
  `conforms_to(K, BinaryCompareKernel) or conforms_to(K, BoolBinaryKernel)`. Type refinement makes
  `K.core` callable in each branch. No wrapper, no marker.
- **Lane vs boundary** is a per-node expression-layer choice (`comptime if conforms_to(L,
  NumericValue)` → lane via `K.core`; else → materialize via `K.apply`), as in `Equal`.
- **New boundary kernels** (`StartsWithKernel`, `ArrayContainsKernel`) are plain compute structs with
  an `apply(typed arrays) -> <typed array>` static method, in **one new** kernel file
  (`marrow/kernels/string_ops.mojo`) — additive, no existing-file edits. `equal(StringArray,…)` is
  reused for string `==`.

## Nested families (struct + list, typed)

- `StructField` reflects the child dtype (`reflect[T].field[name].T`, the `Table[T]` trick one level
  deeper) and returns a **child-family-typed leaf** (numeric child → `NumericValue`, string child →
  `StringValue`) that fuses like a column. Its `core`/`execute` gets the child via
  `parent.execute(batch).field(name)` downcast immediately to the typed array (no `DynArray` flow).
- `ArrayLength[L: ListValue](NumericValue)` reads list offsets (`off[i+1]-off[i]`, true-SIMD).
- `ArrayContains` = string-style materialize→`BoolValue` boundary.

## Compile-probe order (spike FIRST — this is the make-or-break)

Throwaway `.mojo` files under the scratchpad. The whole design rests on bleeding-edge conditional
conformance, so probe before building wide:

1. `Value` with associated `comptime OutType` + `comptime Output` + a `execute -> Self.Output` that
   families bind to different concrete arrays; generic code calls `.execute()`. *(foundational)*
2. **Output-family unification**: one `FusedBinary[K: Kernel, L, R]` with two disjoint `where`-guarded
   `comptime OutType` witnesses + dual conditional conformance to `NumericValue`/`BoolValue`, keyed on
   the **existing** `BinaryKernel` vs `BinaryCompareKernel`/`BoolBinaryKernel` traits (no new markers);
   use-site `Add(...) + x` resolves numeric ops, `Less(...) & y` resolves bool ops. *(top risk)*
3. **Input-family unification**: `Equal[L: Value, R: Value](BoolValue)` with `comptime if
   conforms_to(L, NumericValue)` lane branch vs string boundary branch; always-present typed
   `ArcPointer[BoolArray]` cache, interior mutation in `prepare` from borrowed `self`. *(top risk)*
4. Nested reflection: `StructField` returning a family-typed leaf from a 2-level schema struct.
5. Free-helper `execute` forwarder (`self.core[W]` in a `@parameter` closure → `_lane_execute`).

**If any probe fails on a conditional-conformance / type-refinement / typed-`comptime`-witness
feature, first update the Mojo nightly** (already fairly recent; these features are new) before
concluding it is unavailable. Only if it still fails do we retreat to per-output-family templates
(`FusedBinary`/`Compare`/`BoolBinary`) — a fallback that keeps every other decision intact.

## Implementation phases (each builds green in `ibis.mojo` + `test_ibis.mojo`)

- **Phase 0 — Spike**: probes 1–5. Confirm the conditional-conformance mechanics (update nightly if needed).
- **Phase 1 — Foundation**: `Value` + `NumericValue` + `NumericColumn`/`Literal`/`col` + `FusedBinary`
  arithmetic; test a numeric fused expr `.execute(batch)`.
- **Phase 2 — Cross-family lanes**: comparisons + boolean logic via the same `FusedBinary` +
  operators; test `(a+b) < c & d`.
- **Phase 3 — Input-family + materialization**: `Equal` over numeric AND string; `StartsWith`; test
  `Equal(s,t)` and `startswith(s,"x") & (a>b)` — boundary runs once + one fused pass.
- **Phase 4 — Nested**: `StringColumn`/`StructColumn`/`StructField` + `ListColumn`/`ArrayLength`/
  `ArrayContains`; test struct field arithmetic + list length + list contains.
- **Phase 5 — Var-len + reductions**: `Upper`/`Substring` (`StringValue`) + `Sum`/`Mean`/`Min`/`Max`.

## Files

- **New only**: `marrow/expr/ibis.mojo`, `marrow/expr/tests/test_ibis.mojo`,
  `marrow/kernels/string_ops.mojo` (new boundary compute: `StartsWithKernel`, `UpperKernel`,
  `ArrayContainsKernel`).
- **Reuse by import (no edits)**: `arithmetic.mojo` (`BinaryKernel`, `AddKernel`…), `compare.mojo`
  (`BinaryCompareKernel`, `EqKernel`, `equal(StringArray)`), `boolean.mojo` (`BoolBinaryKernel`,
  `AndKernel`), `aggregate.mojo`, `dtypes.mojo`, `arrays.mojo`. Output-family classification uses
  these **existing** kernel traits — no new markers.
- **Untouched**: `values.mojo`, `dynamic.mojo`, `dtypes.mojo` (no `BoolType : PrimitiveType`).

## Deferred

- Meshing `ibis.mojo` with `values.mojo`'s relational/execution layer + Python bindings; temporal
  `Extract*` beyond a smoke test; the Map family; pruning.

## Verification

- `pixi run -e dev pytest marrow/expr/tests/test_ibis.mojo` — build expressions (`Add`, `Less`, `And`,
  `Equal` over numeric AND string, `startswith`, struct field, list length/contains) and
  `.execute(batch)`, asserting results equal expected typed `array([...])` (cross-check vs PyArrow
  where an equivalent exists).
- Assert the materialization path runs the boundary kernel exactly once and that no scalar loop sits
  inside any `core`.
- Keep the small-binary property: `pixi run binary_size` (`benchmarks/binary_size/`).
