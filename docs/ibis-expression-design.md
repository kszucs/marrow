# Record — the ibis-flavoured typed expression system

**Status: landed, in `marrow/expr/values.mojo`.** Built standalone in
`marrow/expr/ibis.mojo` as specified (`b5b6687`), then merged into `values.mojo`
as the canonical expression system (`d2a2937`, −1,038 lines) — so the file this
spec named is gone, as is the `marrow/kernels/string_ops.mojo` it proposed (the
string kernels went into `kernels/string.mojo`). Its "do not modify existing
files / `values.mojo` untouched" constraint is void: as a spec it forbade editing
the very files that now implement it. Kept for four decisions worth not
re-deriving.

## 1. Fusability taxonomy — the design axis

The bucket a node falls into is **which trait it conforms to, never a runtime
tag**. That survived intact.

| Bucket | Examples | Mechanism |
|---|---|---|
| **true-SIMD lane** (fixed→fixed) | Add/Sub/Mul/Div, Neg/Abs, numeric compares, And/Or/Not, Cast, IsNull, temporal `Extract*` | per-lane `vectorwise[W]` over `K.core[W]`; fuses |
| **materialize-once → lane leaf** (var-len in, fixed-width out) | string `==`, `StartsWith`, `Contains`, `StringLength`, `ArrayLength` | eager kernel `apply` **once**, then loaded per lane |
| **variable-length output** | `Upper`/`Lower`, `Substring` | `StringValue` node producing a `StringArray`; not a lane |
| **reduction / boundary** (N→1) | `Sum`, `Mean`, `Min`, `Max` | consume a fused lane, produce a typed scalar |
| **nested / structural** | `ArrayLength` (offset read), `MapGet` (boundary) | list = compute; struct never shipped (§4) |

Only the mechanism drifted: the marker is `trait Breaker(Value)`
(`values.mojo:417`) and the cache is a positional `Context` slot, not a per-node
`ArcPointer`. See `docs/lane-shape-window-design.md` §2/§5 — including what this
spec did not anticipate, that **fusion continues above a boundary**.

## 2. `NumericValue` and `BoolValue` stay disjoint

Sibling traits: never merged into a shared execution trait, never one a subtype
of the other. Disjoint operator surfaces (`+`/`<` vs `&`/`~`) and disjoint
packaging (`PrimitiveArray` vs bit-packed `BoolArray`). "Bool is a number" is
recovered **only** by explicit bridges, and all four shipped:

| bridge | site | note |
|---|---|---|
| `NumToBool[A: NumericValue](BoolValue)` | `values.mojo:1255` | `x != 0`, pure lane |
| `BoolToNum[To: NumericType, A: BoolValue](NumericValue)` | `:1286` | `True→1`, pure lane |
| `StringToNum[To: NumericType, A: StringValue](Breaker, NumericValue)` | `:1316` | no value lane → breaker |
| `StringToBool[A: StringValue](BoolValue, Breaker)` | `:1349` | no value lane → breaker |

`BoolType` was never made a `PrimitiveType`, as required. The families today are
`NumericValue`, `BoolValue`, `StringValue`, `TemporalValue`, `ListValue`.

## 3. Dual conditional conformance was probed; the fallback shipped

The highest-risk mechanism — one `FusedBinary[K, L, R]` conditionally conforming
to `NumericValue` *or* `BoolValue` via `where`-guarded witnesses (probe 2), plus
one input-family-unified `Equal[L, R]` branching on `conforms_to(L,
NumericValue)` (probe 3) — was **not taken**. The per-output-family fallback the
spec named as its retreat is what shipped, with every other decision intact:

| node | site | family |
|---|---|---|
| `NumericBinary[K: BinaryNumericKernel, L, R]` | `values.mojo:664` | `NumericValue` |
| `FloatBinary[K: BinaryKernel, L, R]` | `:780` | `NumericValue` (float-forcing `/`, `**`) |
| `NumericCompare[K: NumericCompareKernel, S: StringPredicateKernel, L, R]` | `:932` | `BoolValue` |
| `BoolBinary[K: BoolBinaryKernel, L, R]` | `:1027` | `BoolValue` |
| `StringPredicate[K: StringPredicateKernel, L, R]` | `:1685` | `BoolValue`, `Breaker` |

What replaced the input-family unification: `NumericCompare` takes **two kernel
parameters** — `K` for fixed-width lanes, `S` for strings — so the *operator*,
not the operand family, carries both meanings of `a < b`. The runtime lane spells
the same thing as `DynValue._compare[N, S]`.

## 4. Phase 4 (nested) never shipped

There is **no struct value family**: no `StructValue`, no `StructField`, no
reflection of a child dtype into a family-typed leaf. `ListValue`/`ListColumn`/
`ListLength` (`values.mojo:2310-2351`) are the only nested nodes. Phases 0–3 and
5 landed; pruning, declared out of scope here, landed too
(`values.mojo:596,645,960,1039`).
