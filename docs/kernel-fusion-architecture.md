# Kernel & Fusion Architecture

**Status: implemented.** How compute kernels are organized as structs, how they
expose a single fusion primitive (`core`), and how the comptime-typed expression
layer (`marrow/expr/values.mojo`) composes them into fused, zero-intermediate
passes — including the string and variable-length regimes.

This document supersedes the ad-hoc "typed overload first, type-erased blanket"
note in `CLAUDE.md` for compute kernels; the eager tiers still hold, this adds
the fusion axis and the trait organization.

**Scope.** This is the *kernel and fused-node* view. The erasure boundary, the
`Breaker`/`Context` staging model, the plan IR and the binary-size argument live
in **`docs/architecture.md`** and are not repeated here.

---

## 1. Goals

1. **Encapsulation** — every kernel is a standalone struct owning one operation.
2. **One functor, two runners** — the per-lane functor (`core`) is defined *once*
   in the kernel and consumed by both eager execution (`apply` over buffers) and
   fused execution (expression nodes composing `core` chains). No duplication.
3. **Type-family organization via traits** — the type families an operation
   supports (numeric / bool / string / …) are expressed as the trait it conforms
   to; shared machinery (`apply`, `dispatch`) lives on the traits.
4. **Typed *and* type-erased entry points** — `apply[T]` / `apply(StringArray)`
   for known types, `dispatch(DynArray)` for runtime types.
5. **Graceful degradation** — an operation that cannot express a per-lane `core`
   still participates in a fused expression by materializing once into a stage.
   Fusion is never all-or-nothing.
6. **Vectorized, allocation-light** — kernels operate through `marrow/views.mojo`
   primitives; per-element `get`/`set` loops are a last resort.

---

## 2. The layered kernel model

`trait Kernel` (`marrow/kernels/core.mojo:16`) is the root of the hierarchy. It
fixes `comptime name` — identity for display and diagnostics, *never* dispatch —
and owns the argument checks every family would otherwise re-spell (`error`,
`expect_same_length`, `expect_same_dtype`). Family traits add the call shape.

Every kernel family exposes up to three tiers on the eager side:

```
Tier 0  core     — pure per-lane functor. No allocation, no I/O. THE fusion atom.
                   numeric:  core[T: DType, W: Int](a: SIMD[T,W], b) -> SIMD[R,W]
Tier 1  apply    — eager, typed. Runs core over full buffers via views.apply
                   (vectorized, null-propagating, hardware-portable). One overload
                   per supported type family.
Tier 2  dispatch — eager, type-erased. Runtime dtype -> the right typed apply.
```

Everything above tier 0 is *derivable* from `core` (plus a scalar `predicate` for
the string family) and therefore lives as **trait defaults**, not per-kernel code.
A concrete kernel is usually a name and a functor:

```mojo
trait BinaryKernel(Kernel):                       # numeric.mojo:59
    @staticmethod
    def core[T: DType, W: Int](a: SIMD[T, W], b: SIMD[T, W]) -> SIMD[T, W]: ...
    # apply defaulted at numeric.mojo:85; dispatch declared here so a node
    # generic over BinaryKernel can reach it, and defaulted on each sub-trait

trait BinaryNumericKernel(BinaryKernel): ...      # numeric.mojo:117 — dispatch_numeric
trait BinaryFloatKernel(BinaryKernel): ...        # numeric.mojo:137 — dispatch_floating

struct AddKernel(BinaryNumericKernel):            # numeric.mojo:233
    comptime name = "add"
    # core = a + b
```

The families and their tier-0 shapes:

| Trait | Where | Tier 0 |
|---|---|---|
| `BinaryKernel` → `BinaryNumericKernel` / `BinaryFloatKernel` | `numeric.mojo:59,117,137` | `core[T,W](a, b) -> SIMD[T,W]` |
| `UnaryKernel` → `UnaryNumericKernel` / `UnaryFloatKernel` | `numeric.mojo:157,198,213` | `core[T,W](a) -> SIMD[T,W]` |
| `NumericCompareKernel` | `numeric.mojo:528` | `core[T,W](a, b) -> SIMD[bool,W]` |
| `StringPredicateKernel` | `string.mojo:297` | `predicate(StringSlice, StringSlice) -> Bool` (scalar) |
| `StringMapKernel` | `string.mojo:126` | per-row transform |
| `LengthKernel` | `string.mojo:47` | `core[T,W](hi, lo) -> SIMD[int32,W]` over offsets |

**`NumericCompareKernel` is numeric only** (`numeric.mojo:528`). It used to carry
a `comptime StringKernel` naming its string counterpart, so every numeric
comparison had to know about strings and `dispatch` branched on dtype at run time
between two unrelated implementations. Which family `a < b` means is a question
about the *operands*, and it belongs to whoever interprets the operator — see §6.

**Dispatch on the widest family the typed leaf accepts.** A leaf bound on
`PrimitiveType` already takes temporal, interval and decimal columns, so it needs
one `dispatch_primitive` arm, not one per family.

---

## 3. What is fusable, and why

**An operation is fusable iff it is element-wise and length-preserving** — output
row `i` depends only on input row `i`, and `len(out) == len(in)`. Those ops expose
a per-lane functor and become the *interior* of a fused subtree. Everything else
is a **breaker**: it can consume a fused subtree but cannot itself compose into a
lane function, so it materializes into a `Context` stage and its consumer reads
the slot back (`docs/architecture.md` §3).

Note that "boundary" is not "outside the expression tree". A `Reduction` is a
`Value` like any other; it just conforms to `Breaker` as well.

### Classification, as shipped

| Expression | Node | Status |
|---|---|---|
| `a + b`, `a * 2`, `a / b` | `NumericBinary` `:664` / `FloatBinary` `:780` | ✅ fused SIMD |
| `-a`, `abs(a)`, `sqrt(a)` | `NumericUnary` `:716` / `FloatUnary` `:826` | ✅ fused SIMD |
| `a < b`, `a == b` (numeric) | `NumericCompare` `:932` | ✅ fused SIMD → bit-packed |
| `p and q`, `not p` | `BoolBinary` `:1027` / `BoolUnary` `:1099` | ✅ fused SIMD |
| `a.cast(int64)`, num ↔ bool | `NumericCast` `:749`, `NumToBool` `:1255` | ✅ fused SIMD |
| `is_nan(a)`, `is_inf(a)` | `NumericPredicate` `:1182` | ✅ fused SIMD |
| `upper(s)`, `s1 + s2` | `StringUnary` `:1562`, `Concat` `:1533` | ✅ fused **per-row** (one builder pass) |
| `is_null(x)`, `not_null(x)` | `NullPredicate` `:1214` | ⚠️ breaker — reads validity through the kernel |
| `s.len()` | `StringLength` `:1785` | ⚠️ breaker → `Int32Array` |
| `s1 == s2`, `s.contains(x)`, `LIKE` | `StringPredicate` `:1685` | ⚠️ breaker → `BoolArray` |
| `x IN (…)` | `IsIn` `:1751` | ⚠️ breaker → `BoolArray` |
| `coalesce`, `nullif`, `CASE WHEN` | `ConditionalBinary` `:2056`, `CaseWhen` `:2102` | ⚠️ breaker |
| `year(ts)`, `date_trunc` | `TemporalExtract` `:2236` | ⚠️ breaker → `Int32Array` |
| string → num/bool parse; casts *to* string | `:1316`, `:1349`, `:1601`-`:1653` | ⚠️ breaker |
| `sum(a)`, `mean(a)`, `min(a)` | `Reduction` `:1922` | ⚠️ breaker, `OutShape == 0` |
| `any(p)`, `all(p)` | `BoolReduce` `:1137` | ⚠️ breaker, scalar |
| `row_number()` | `WindowFunction` `:2012` | ⚠️ breaker, columnar |
| `filter`, `take`, `drop_null`, `sort`, `group_by`, `join`, `concat` | relational operators | ❌ outside the expression layer |

All line numbers are `marrow/expr/values.mojo`.

**Fused subtrees are bounded by breakers**, and fusion continues *above* one: a
breaker's `vectorwise` just loads its own stage result out of the `Context`, so
`length(s) + 1` is one numeric pass over a materialized length column. The tree
splits into stages; each stage is one fused loop.

---

## 4. Lane protocols — the shape of `core` is set by the *output* width

The critical realization: a per-lane functor constrains only the **output** — it
must be `W` fixed-width lanes. It says nothing about the input. So
variable-length *inputs* (strings) do not disqualify a lane; only variable-length
*outputs* do.

Protocols, keyed by *(input family → output family)*:

```
numeric → numeric   core[T,W](a: SIMD[T,W], b) -> SIMD[T,W]              add, mul, min, neg…
numeric → bool      core[T,W](a: SIMD[T,W], b) -> SIMD[bool,W]           lt, eq, is_nan…
bool    → bool      core[T,W](a: SIMD[bool,W], b) -> SIMD[bool,W]        and, or, not
string  → numeric   core[T,W](hi, lo) -> SIMD[int32,W]                   length (over offsets)
string  → bool      predicate(a: StringSlice, b: StringSlice) -> Bool    str <, ==  (scalar)
string  → string    elementwise(...) -> String                           concat, upper (per-row)
```

Three regimes fall out — **and the boundary is the output width, not the input**:

1. **Fixed-width output from a fixed-stride source.** Genuinely SIMD; joins the
   numeric/bool loop unchanged. `LengthKernel` (`string.mojo:47`) is the
   interesting member: it loads `W+1` entries from the **offsets** buffer
   (fixed-stride `int32`) and subtracts shifted lanes, so the string *bytes* are
   never touched. `strlen` is therefore truly SIMD *at the kernel tier*.
2. **Fixed-width output from variable-length content** (`s1 == s2`, `contains`).
   There is no data parallelism to be had over variable-width bytes, so the
   kernel's tier 0 is a **scalar** `predicate` and `apply` walks rows into a
   bit-packed `BoolArray` (`string.mojo:309-328`).
3. **Variable-width output** (`s1 + s2`, `upper`, `substr`). There is no
   `SIMD[string, W]`, so the lane is per-row: `StringValue.elementwise(...) ->
   String` (`values.mojo:1388`) yields one row and the family driver
   (`values.mojo:1393`) appends into a builder.

**Regime 3 shipped, and it is the strongest result here.** `upper(col) || "!"`
composes in one builder pass and never materializes `upper(col)`
(`values.mojo:1562`). Regime 1 shipped for numeric and bool. **Regime 2 did
not** — see §8.

### Width caveat

The fused loop runs at **one width `W` per tree**, driven by a single
`NativeType`. `BoolValue` declares `comptime NativeType: DType` for exactly this
(`values.mojo:880`): it is the *operand* width that sizes the SIMD lane, not the
output. Nodes whose operands may differ in width pick `wider` of the two
(`values.mojo:271`) — a distinct question from `promote` (`:262`), which decides
the *value* domain, where every float outranks every integer. A bool breaker with
no numeric operand picks a width outright: `StringPredicate` and `IsIn` declare
`NativeType = DType.int32` (`values.mojo:1690`, `:1754`).

---

## 5. Materialization — the universal fallback

Any operation that cannot express a per-lane functor still participates in a
fused expression: it conforms to `Breaker`, runs eagerly through `apply` /
`dispatch` in a `prepare` pre-pass, and its result lands in a positional
`Context` slot that its consumer reads per lane.

Consequences:

- **Fusion is never all-or-nothing.** A tree fuses its lane-expressible parts and
  inserts a stage at each breaker. Exactly how real engines pipeline: fuse within
  a stage, materialize at stage edges.
- **Every kernel is immediately usable from the expression layer** — natively via
  a lane if it has one, otherwise as a breaker node wrapping `apply`. New kernels
  do not block on a fused-node implementation.

**This is not the `Materialized` leaf adapter this document originally proposed.**
The shipped mechanism attaches to a *node*: you write a `Breaker` node for the
operation. There is still no adapter that takes an arbitrary `DynArray` — say the
output of an eager kernel computed elsewhere — and drops it into a fused subtree
as a leaf. See §8, gap 1.

The full staging model — `Value` / `Breaker` polarity, `Datum`, `OutShape`,
`Context` slots, `prepare` DFS order, fuse-above-breaker — is documented in
`docs/architecture.md` §3.

---

## 6. Struct architecture — one `core`, generic fused nodes

Composition must live in the expression layer: fusion works by encoding the tree
in **type parameters** so the compiler inlines the whole `core` chain. A kernel is
a childless functor and structurally cannot carry the tree. So **the kernel owns
the functor and the expr layer owns the composition** — with the fused node made
*generic over the kernel*, so the functor is still written once.

```mojo
struct NumericBinary[K: BinaryNumericKernel, L: NumericValue, R: NumericValue](
    NumericValue
):                                                          # values.mojo:664
    comptime OutType = promote[Self.L.OutType, Self.R.OutType]

    @always_inline
    def vectorwise[
        W: Int
    ](self, batch: RecordBatch, ctx: Context, mut slot: Int, idx: Int) -> SIMD[
        Self.OutType.native, W
    ]:
        var a = self.l.vectorwise[W](batch, ctx, slot, idx).cast[...]()
        var b = self.r.vectorwise[W](batch, ctx, slot, idx).cast[...]()
        return Self.K.core[Self.OutType.native, W](a, b)

comptime Add = NumericBinary[AddKernel, _, _]               # values.mojo:857
comptime Sub = NumericBinary[SubKernel, _, _]
comptime Mul = NumericBinary[MulKernel, _, _]
```

`AddKernel.core` is the single source of the functor, consumed by eager `apply`
*and* the fused node. Adding an op is writing the kernel struct: it is instantly
eager (`apply`/`dispatch`) *and* fusable (through the generic node).

The same shape covers every protocol — `NumericUnary[K: UnaryNumericKernel, A]`
(`:716`), `FloatBinary[K: BinaryKernel, L, R]` (`:780`),
`BoolBinary[K: BoolBinaryKernel, L, R]` (`:1027`),
`StringUnary[K: StringMapKernel, A]` (`:1562`),
`StringPredicate[K: StringPredicateKernel, L, R]` (`:1685`),
`Reduction[K: AggKernel, A]` (`:1922`).

### The one node with two kernel parameters

`NumericCompare` (`values.mojo:932`) carries **both** kernels of a comparison
operator:

```mojo
struct NumericCompare[
    K: NumericCompareKernel,
    S: StringPredicateKernel,
    L: NumericValue,
    R: NumericValue,
](BoolValue): ...

comptime Lt = NumericCompare[LtKernel, StringLtKernel, _, _]   # values.mojo:1013
comptime Gt = NumericCompare[GtKernel, StringGtKernel, _, _]
comptime Eq = NumericCompare[EqKernel, StringEqKernel, _, _]
```

This is where the removed `comptime StringKernel` went: naming both halves of
`<` in one alias keeps the operator's two families together without making the
SIMD kernel know about strings. The fused lane only ever uses `K` — its operands
are `NumericValue` — while the string operators route to `StringPredicate`
through separate aliases (`StrLt` = `StringPredicate[StringLtKernel, _, _]`,
`values.mojo:1738`) and the runtime lane picks a family from operand dtypes in
`DynValue._compare[N, S]`.

### Where recursive / nested ops go

Struct equality is **not** a per-lane op — it is a recursive `AND` over child
comparisons. It belongs at the composition layer as a free function reusing
`EqKernel.dispatch` for leaves, not as a method on the lane-functor struct.
`equal_any` (`numeric.mojo:583`) is the shipped version of this: hash-join row
verification and `nullif` both need equality *over an arbitrary dtype* rather
than an operator they are interpreting, and it names that once.

---

## 7. Cross-type worked examples

```
(s.len() + a) * 2
  Mul[ Add[ StringLength[StringColumn], NumericColumn[i32] ], NumericLiteral[i32] ]
  → two passes: StringLength materializes an Int32Array in prepare,
    then Add/Mul fuse over it in one numeric pass.
  → one pass would need a fused offsets lane — §8 gap 3.

s1 == s2  and  a > b
  BoolBinary[AndKernel,
             StringPredicate[StringEqKernel, StrCol s1, StrCol s2],   # breaker
             NumericCompare[GtKernel, …, NumCol a, NumCol b]]         # fused
  → two passes: the string predicate materializes a full BoolArray in prepare,
    then the AND fuses over that mask and the numeric compare — §8 gap 2.

upper(s1) + s2 + "!"
  Concat[ Concat[ StringUnary[UpperKernel, StrCol s1], StrCol s2 ], StringLiteral ]
  → ONE builder pass, no intermediate string arrays. Regime 3, shipped.

filter(t, a > b)     # a relational operator consuming one fused mask
sum(a * b)           # Reduction consuming a fused numeric lane, via one
                     # materialized array — §8 gap 4
```

---

## 8. Open

Four gaps, verified against the code at `b2e7dae`. Everything else this document
once listed as missing has shipped.

1. **No `Materialized` leaf adapter.** An arbitrary eager kernel result — a
   `DynArray` produced outside the expression tree — cannot re-enter a fused
   subtree. The only way in is to write a `Breaker` node for the operation, which
   is a per-operation cost rather than a one-time adapter. A leaf whose
   `vectorwise[W](…, idx)` is `array.values().load[W](idx)` would close it for
   every fixed-width dtype at once.

2. **`StringPredicate` materializes a full `BoolArray`.** It is a `Breaker`
   (`values.mojo:1685`) whose `prepare` (`:1707-1711`) materializes *both* string
   operands and runs `K.apply` into a `BoolArray`. So `s1 == s2 and a > b` is
   **not** the single pass §4 regime 2 describes: the predicate is one pass and
   the AND is another. Closing it means a pseudo-SIMD lane that fills a
   `SIMD[bool, W]` from `W` scalar `predicate` calls, plus a decision about the
   per-tree width for a pure-string predicate.

3. **`StringLength` is two passes, not the offsets lane §4 calls the proof.**
   `LengthKernel.core` exists and is genuinely SIMD over offsets
   (`string.mojo:60`), and its own docstring claims "both `apply` and the
   expression layer's `StringLength` build on it" — but `StringLength`
   (`values.mojo:1785`) is a `Breaker` whose `materialize` (`:1803`) calls
   `LengthKernel.dispatch` into an `Int32Array`, and `vectorwise` loads that
   column. So `s.len() + a` is two passes. A `StringValue` that could expose its
   offsets view to a consumer would let `StringLength` fuse; that is exactly what
   the kernel's `core` was written for and nothing calls it from the expr layer.
   (The kernel docstring is stale and should be corrected with the fix.)

4. **Reductions always consume a materialized array.** `Reduction.materialize`
   (`values.mojo:1947`) does `into_array(self.a.execute(batch), …)` and hands the
   whole array to `K.reduce`. So `sum(a * b)` materializes `a * b` first. Folding
   the operand's `core` chain into the accumulation loop would save one pass and
   one allocation, at the cost of a generic parameter on the reduction. Related:
   `FusedAggregation` (single pass, AoS accumulator, comptime offsets, zero
   dispatch) has zero occurrences — backlog Q2.5 step 4.

Two smaller notes carried over, neither blocking:

- **`strlen` over computed strings.** `LengthKernel` needs offsets, so
  `strlen(s1 + s2)` cannot fuse unless concat exposes a lazy offset stream. It
  materializes today, which is correct and probably good enough.
- **Binding reflection constraint.** A Mojo binding-compiler crash was once
  observed on mutually-recursive nested-type static methods on a reflected kernel
  struct. The construct was never pinned down; the design avoids it by keeping
  recursive/nested ops out of kernel structs (§6), so the rule is grounded in
  layering rather than in the crash.
