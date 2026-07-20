# Kernel & Fusion Architecture

Status: **design proposal** (draft). Scope: how compute kernels are organized as
structs, how they expose a single fusion primitive (`core`), and how the
comptime-typed expression layer (`marrow/expr/values.mojo`) composes them into
fused, zero-intermediate passes — including the string / variable-length regimes
and the materialization fallback.

This document supersedes the ad-hoc "typed overload first, type-erased blanket"
note in `CLAUDE.md` for compute kernels; the eager tiers still hold, this adds
the fusion axis and the trait organization.

---

## 1. Goals

1. **Encapsulation** — every kernel is a standalone struct owning one operation.
2. **One functor, two runners** — the per-lane functor (`core`) is defined *once*
   in the kernel and consumed by both eager execution (`apply` over buffers) and
   fused execution (expression nodes composing `core` chains). No duplication.
3. **Type-family organization via traits** — the type families an operation
   supports (numeric / bool / string / …) are expressed as the trait it conforms
   to; shared machinery (`apply`, `dispatch`, `execute`) lives on the traits.
4. **Typed *and* type-erased entry points** — `apply[T]` / `apply(StringArray)`
   for known types, `dispatch(AnyArray)` for runtime types.
5. **Graceful degradation** — an operation that can't express a per-lane `core`
   still participates in fused expressions by materializing (`apply`) and
   re-entering fusion as a leaf. Fusion is never all-or-nothing.
6. **Vectorized, allocation-light** — kernels operate through `marrow/views.mojo`
   primitives; per-element `get`/`set` loops are a last resort, flagged with a
   `TODO`.

---

## 2. The layered kernel model

Every kernel exposes up to three tiers (eager side):

```
Tier 0  core    — pure per-lane functor. No allocation, no I/O. THE fusion atom.
                  numeric:  core[T: DType, W: Int](a: SIMD[T,W], b) -> SIMD[R,W]
Tier 1  apply   — eager, typed. Runs core over full buffers via views.apply
                  (vectorized, null-propagating, hardware-portable). One overload
                  per supported type family.
Tier 2  dispatch — eager, type-erased. runtime dtype -> the right typed apply.
```

Everything above Tier 0 is *derivable* from `core` (+ a scalar `compare` for the
string family) and therefore lives as **trait defaults**, not per-kernel code.

---

## 3. What is fusable, and why

**An operation is fusable iff it is element-wise and length-preserving** — output
row `i` depends only on input row `i`, and `len(out) == len(in)`. Those ops
expose a per-lane functor and become the *interior* of a fused subtree.
Everything else is a *boundary*: it can **consume** a fused subtree as input but
cannot itself compose into a lane function.

### Classification

| Expression | Fusable? | Protocol / reason |
|---|---|---|
| `a + b`, `a * 2`, `-a`, `sqrt(a)` | ✅ SIMD | numeric → numeric |
| `a < b`, `a == b` | ✅ SIMD | numeric → bool |
| `p and q`, `not p` | ✅ SIMD | bool → bool |
| `a.cast(int64)` | ✅ SIMD | numeric → numeric |
| `s.len()` / `s.byte_length()` | ✅ SIMD | string → numeric — reads **offsets** (fixed-stride) |
| `is_null(s)` | ✅ SIMD | any → bool — reads validity bitmap |
| `(s.len() + a) * 2` | ✅ SIMD | cross-type, all numeric-lane after `len` |
| `if_else(m, a, b)` | ✅ SIMD | ternary numeric → numeric |
| `s1 == s2`, `s.contains(x)` | ⚠️ pseudo-SIMD | string → bool — W scalar `compare` calls fill `SIMD[bool,W]` |
| `hash(s)` | ⚠️ pseudo-SIMD | string → numeric |
| `s1 + s2 + s3`, `upper(s)`, `substr(s)` | ⚠️ per-row | string → **string** — `core[W]` inapplicable, needs builder emit |
| `sum(a)`, `mean(a)`, `min(a)` | ❌ boundary | reduction all → one (consumes a fused input) |
| `count_distinct(a)` | ❌ boundary | stateful reduction |
| `filter(t, m)`, `take(t, i)`, `drop_null` | ❌ boundary | length/position change (consumes a fused mask) |
| `sort`, `sort_indices` | ❌ boundary | global reorder |
| `group_by(…).agg`, `join` | ❌ boundary | multi-row / multi-table |
| `concat([a, b])` (arrays) | ❌ boundary | length change |

**Fused subtrees are bounded by non-fusable operators.** A plan is a DAG of
boundary operators, each fed by fusable expression subtrees.

---

## 4. Lane protocols — the shape of `core` is set by the *output* width

The critical realization: `core[W](batch, idx) -> SIMD[T, W]` constrains only the
**output** — it must be `W` fixed-width lanes. It says nothing about the input.
So variable-length *inputs* (strings) do **not** disqualify `core`; only
variable-length *outputs* do.

`Length` ([values.mojo](../marrow/expr/values.mojo)) is the proof: it is a
`NumericValue` whose `core[W]` loads `W+1` entries from the **offsets** buffer
(fixed-stride `int32`) and subtracts shifted lanes. The string *bytes* are never
touched — only their offsets, an ordinary numeric buffer. `strlen` is therefore
*truly SIMD* and drops straight into the numeric fusion loop.

Protocols, keyed by *(input family → output family)*:

```
numeric → numeric   core[T,W](a: SIMD[T,W], b) -> SIMD[T,W]            add, mul, min, neg…
numeric → bool      core[T,W](a: SIMD[T,W], b) -> SIMD[bool,W]         lt, eq, and…
string  → numeric   core[W](offsets: BufferView[i32], idx) -> SIMD[i32,W]   len   (true SIMD)
string  → bool      compare(a: String, b: String) -> Bool             str <,==   (scalar → filled SIMD[bool,W])
string  → string    emit(a: StringSlice, b, mut out: StringBuilder)   concat     (per-row, NO core[W])
```

Three regimes fall out — **and the boundary is the output width, not the input**:

1. **Fixed-width output from a fixed-stride source** (`strlen`, `is_null`):
   `core[W]` is *genuinely SIMD*. Joins the existing numeric/bool loop unchanged.
2. **Fixed-width output from variable-length content** (`s1 == s2`, `contains`,
   `hash`): `core[W]` is *pseudo-SIMD* — a scalar loop over the W lanes fills a
   `SIMD[bool,W]`/`SIMD[u64,W]`. You lose data-parallelism on the compare (there
   is none to be had on variable-length bytes) but keep the fusion win: **no
   intermediate array, one pass**. So `filter(t, s1 == s2 and a > b)` still fuses
   to a single mask.
3. **Variable-width output** (`s1 + s2`, `upper`, `substr`): `core[W]` is
   *inapplicable* — there is no `SIMD[string, W]`. Needs a separate per-row
   `emit(batch, idx, mut builder)` protocol with one output builder pass.

### Width caveat

The existing loop runs at **one width `W` per tree**, driven by a single
`NativeType` (see `Less` casting `r` to `Self.NativeType`). Consequences:

- Pseudo-SIMD string lanes coexist with true-SIMD numeric lanes only if they
  agree on `W`. In `s1 == s2 and a > b`, `W` is driven by the numeric side and
  the string node just loops `W` times internally.
- A **pure**-string predicate (`s1 == s2`) has no numeric `NativeType`; it must
  pick a `W` (bit-packing tolerates any — e.g. drive from `uint8`).

---

## 5. Materialization — the universal fusion fallback

**Key unifier:** any operation that cannot (or does not yet) express a per-lane
`core` still participates in a fused expression by being executed eagerly via
`apply`/`dispatch`, and its **result array re-enters fusion as a leaf** — a
column-like node whose `core[W](idx)` is just `array.values().load[W](idx)`.

```
struct Materialized(NumericValue | BoolValue | StringValue):
    var array: AnyArray                       # produced eagerly, once
    core[W](self, _, idx) = array.values().load[W](idx)    # leaf load
```

Consequences:

- **Fusion is never all-or-nothing.** A tree fuses its lane-expressible parts and
  inserts a materialization point at each boundary op. Exactly how real engines
  pipeline: fuse within a stage, materialize at stage edges.
- **Every op is immediately usable in the expr system** — natively via `core` if
  lane-expressible, otherwise via `apply` → `Materialized` leaf. New ops don't
  block on a fused-node implementation.
- **String-producing fusion becomes an *optimization*, not a prerequisite.**
  `s1 + s2` runs through the eager string-concat `apply` → `StringArray` →
  `Materialized`; `strlen(s1 + s2)` then reads the materialized offsets. The
  per-row `emit` protocol (regime 3) is a later speedup for hot concat chains,
  not a correctness gate.

This is the "execute as `apply()` then apply `core()` of the result" rule,
generalized: **`apply` is the adapter that turns any kernel into a fusion leaf.**

---

## 6. Struct architecture — one `core`, generic fused nodes

Composition must live in the expression layer: fusion works by encoding the tree
in **type parameters** (`Add[L: NumericValue, R: NumericValue]`) so the compiler
inlines the whole `core` chain. A kernel is a childless functor and structurally
cannot carry the tree. So: **the kernel owns the functor; the expr layer owns the
composition** — but the fused node is made *generic over the kernel* so the
functor is defined once.

The codebase already contains both styles; the design is to make them all look
like the good one:

```
# GOOD (already): expr node delegates to the kernel functor
Cast.core        -> NumericCast.core[...]      (values.mojo)
NumToBoolValue   -> NumToBool.core[...]
BoolToNumValue   -> BoolToNum.core[...]

# DUPLICATED (to fix): expr node re-inlines the op
Add.core   -> `l + r`     (vs AddKernel.core = a + b)
Less.core  -> `l.lt(r)`   (vs LtKernel.core = a.lt(b))
Equal.core -> `l.eq(r)`   (vs EqKernel.core = a.eq(b))
```

### Eager side — trait per protocol

```mojo
trait NumericBinaryKernel(Kernel):
    core[T: DType, W: Int](a: SIMD[T,W], b: SIMD[T,W]) -> SIMD[T,W]   # only requirement
    apply[T: PrimitiveType](l, r, ctx) -> PrimitiveArray[T]           # default (views.apply)
    dispatch(AnyArray, AnyArray, ctx) -> AnyArray                     # default (numeric)

struct AddKernel(NumericBinaryKernel):
    comptime name = "add"
    core[T,W](a, b) = a + b            # the whole kernel

trait CompareKernel(Kernel):
    core[T: DType, W: Int](a, b) -> SIMD[DType.bool, W]               # numeric predicate
    compare(a: String, b: String) -> Bool                            # string scalar predicate
    apply[T](num, num) / apply(str, str) / dispatch                  # defaults (numeric + string)
```

### Fused side — one generic node per protocol, parametrized by the kernel

```mojo
struct FusedBinary[K: NumericBinaryKernel, L: NumericValue, R: NumericValue](NumericValue):
    core[W](self, batch, idx) = K.core[NativeType, W](left.core[W](…), right.core[W](…))

struct FusedCompare[K: CompareKernel, L: NumericValue, R: NumericValue](BoolValue):
    core[W](self, batch, idx) = K.core[NativeType, W](left.core[W](…), right.core[W](…))

comptime Add   = FusedBinary[AddKernel, _, _]
comptime Mul   = FusedBinary[MulKernel, _, _]
comptime Less  = FusedCompare[LtKernel, _, _]
comptime Equal = FusedCompare[EqKernel, _, _]
```

`AddKernel.core` / `LtKernel.core` is now the single source of the functor,
consumed by eager `apply` *and* the fused node. Adding an op = write the kernel
struct; it is instantly eager (apply/dispatch) *and* fusable (via the generic
node). This is exactly what `Cast` already does, generalized to remove the
per-op fused-node boilerplate.

### Where recursive / nested ops go

Struct equality is **not** a per-lane op — it's a recursive `AND` over child
comparisons. It belongs at the composition layer as a free function reusing
`EqKernel.dispatch` for leaves, not as a method on the lane-functor struct. (This
also sidesteps a Mojo binding-compiler crash observed when mutually-recursive
`StructArray` static methods sit on a binding-reflected kernel struct — the
design and the compiler agree here.)

---

## 7. Cross-type worked examples

```
(s.len() + a) * 2
  Mul[ Add[ StrLen[LenKernel, StrCol("s")], NumCol[i32]("a") ], Lit[i32](2) ]
  → StrLen bridges string→i32 (offsets, true SIMD); all nodes emit i32 lanes
  → single fused SIMD i32 pass, zero intermediates
  → already expressible today once Mul + Lit nodes exist

s1 < s2  and  a > b
  And[ FusedCompare[LtKernel, StrCol s1, StrCol s2],   # pseudo-SIMD (scalar compare → bool lane)
       FusedCompare[GtKernel, NumCol a,  NumCol b] ]    # true SIMD
  → both fill SIMD[bool,W] at the numeric-driven W; one predicate pass, no intermediates

s1 + s2 + s3
  option A (now):   Materialized(concat_apply(concat_apply(s1,s2), s3))   # eager, correct
  option B (later): Concat[Concat[StrCol s1, StrCol s2], StrCol s3]       # per-row emit, fused

filter(t, a > b)     # boundary consumes a fused mask
sum(a * b)           # boundary consumes a fused numeric lane
```

---

## 8. Current state (what exists vs. what's missing)

Exists and correct in `values.mojo`:
- `NumericValue` / `BoolValue` protocols with shared `execute()` vectorize loops.
- `Add`, `Sub` (numeric); `Less`, `Greater`, `Equal` (comparison); `Cast`,
  `NumToBoolValue`, `BoolToNumValue` (casts, already delegating to kernel cores).
- `Length` (string→numeric bridge, true SIMD over offsets).
- `NumericColumn` / `StringColumn` leaves; `AnyValue` erasure box; `col()` /
  `Table[T]`.

Gaps:
- `Add`/`Sub`/`Less`/`Greater`/`Equal` re-inline the op instead of delegating to
  kernel `core` (the duplication to consolidate).
- Missing fused numeric ops: `Mul`, `Div`, `Min`, `Max`, unary (`neg`/`abs`/…),
  and comparisons `Le`/`Ge`/`Ne`. Only 5 of the ops have fused nodes.
- No `Literal` node (needed for `* 2`).
- `StringValue` has no per-lane / per-row protocol — only `resolve() -> StringArray`.
  So string-producing fusion and fused string comparison don't exist yet.
- No `Materialized` leaf adapter (§5) — so boundary/non-fused ops can't currently
  re-enter a fused subtree.

---

## 9. Refined plan (correctness-first, then push more into lane fusion)

Each phase builds and keeps tests green before the next.

| Phase | Deliverable | Notes |
|---|---|---|
| **1 — reference slice** | `NumericBinaryKernel` trait + generic `FusedBinary[K,…]`; port `Add`/`Sub` onto it, delegating to `AddKernel.core`/`SubKernel.core`. Prove "one core, two runners" against the real code. | Smallest change that removes duplication; template for the rest. |
| **2 — `Materialized` leaf** | The `apply → fusion leaf` adapter (§5). Any `AnyArray` becomes a `NumericValue`/`BoolValue`/`StringValue` leaf via buffer load. | Unlocks correctness for *every* op immediately; makes later phases optimizations, not gates. |
| **3 — numeric completeness** | Generic `FusedUnary[K,…]`, remaining `FusedBinary`/`FusedCompare` ops (`Mul`/`Div`/`Min`/`Max`, `neg`/`abs`/…, `Le`/`Ge`/`Ne`), `Literal`. | After this, `(s.len() + a) * 2` fuses end-to-end. |
| **4 — compare kernels as structs** | Consolidate `compare.mojo` onto the `CompareKernel` trait (numeric `core` + string `compare`), struct equality as a free function, callers + binding updated. | Also the fix for the removed free-function `equal` family. |
| **5 — string fixed-output fusion** | `s1 == s2` etc. as pseudo-SIMD `FusedCompare` (string path fills `SIMD[bool,W]`); resolve the per-tree width question for pure-string predicates. | Fuses string predicates into mixed numeric/string filters. |
| **6 — string-producing fusion** | Per-row `emit` protocol on `StringValue`; `Concat`/`upper`/`substr` fused. | Pure optimization over the Phase-2 materialization path; do last / as needed. |

Sequencing rationale: Phase 2 makes the whole system *correct and usable* early
(everything works via materialization); Phases 3–6 progressively convert
materialization boundaries into true lane fusion where it pays off.

---

## 10. Open questions

- **Per-tree width for pure-string predicates** (§4 caveat): pick a fixed `W`
  (drive from `uint8`), or special-case scalar (`W=1`) execution for
  content-reading subtrees?
- **`strlen` over computed strings**: `Length.core` needs offsets, so
  `strlen(s1 + s2)` only fuses if concat exposes a lazy offset stream (regime 3);
  otherwise it materializes (Phase 2). Is lazy-offset concat worth the
  complexity, or is materialize-then-strlen good enough?
- **Reductions consuming fused inputs** (`sum(a * b)`): should the reduction
  kernel accept a `NumericValue` and fold `core` into its accumulation loop, or
  always consume a materialized array? (Fusing avoids one pass; adds a generic
  parameter to the reduction.)
- **Binding reflection constraint**: pin down the exact construct that crashes
  the Python-binding compiler (mutually-recursive nested-type static methods on a
  reflected kernel) so the trait rules are grounded rather than avoided.
