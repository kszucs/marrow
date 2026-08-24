# Can one rewrite rule serve both expression lanes? — research findings

**Status:** research spike, complete. **Date:** 2026-08-21.
**Base:** `c0c9a94`. **Toolchain:** Mojo 1.1.0.dev2026081705 (18b45e5c).

All code written for this spike was **throwaway** and has been deleted. Every
claim below is marked **[compiled]** (I built and ran it), **[measured]** (a
`pixi run binary_size` number), or **[read]** (inferred from a cited file I
read but did not execute).

---

## The answer

**Yes, with a sharp and durable limit: a rule can be written once and serve both
lanes if and only if its output type is a *total function* of its input type.**

Concretely:

- **Decomposition** — splitting a fused node into already-existing sub-nodes,
  erased into `BoxedValue` — is uniform across both lanes. **[compiled]**
- **Rebuilding** is *not* categorically unavailable to the AOT lane, which is
  what I expected to find and did not. A rewrite that produces a new fused type
  can be written once, generically, and applied recursively over an arbitrary
  tree — **provided every node can state what it becomes without inspecting
  what its children are**. De Morgan plus double-negation elimination compiles
  and runs in that form. **[compiled]**
- What is genuinely impossible is a rewrite driver that **pattern-matches on
  node kind and returns a different type per branch**. The `comptime`
  conditional type does not reduce at the construction site, and `rebind`
  cannot bridge it. **[compiled]** — this is CLAUDE.md's "a comptime conditional
  type carries no trait conformance and does not reduce at a return site",
  confirmed for exactly this shape.

The original hypothesis in the brief — *decompose at `rel.filter(expr)` via
parameter deduction on an overload taking `And[L, R]`* — **is viable but
strictly weaker than the trait-virtual route, and I recommend against it.**
Deduction works; the overload is selected; but it splits **exactly one level**
and cannot recurse. Details in §3.

---

## 1. Parameter deduction — works, at arbitrary depth, with trait bounds

**[compiled]** An overload parameterised on a generic-struct instantiation
deduces the inner parameters from a concrete argument, and the deduced
parameters may be trait-bound:

```mojo
def top[L: BoolV, R: BoolV](v: And[L, R]) -> List[String]:      # And = BinOp[AndK, _, _]
    return [v.l.tag(), v.r.tag()]
```

called with `And(And(Leaf("a"), Leaf("b")), Leaf("c"))` yields
`and(a, b) | c`. Both the parametric-alias spelling `And[L, R]` and the direct
`BinOp[AndK, L, R]` deduce identically — which matters, because **marrow has no
`And` type**: `comptime And = BoolBinary[AndKernel, AndInterval, _, _]`
(`marrow/exprold/values.mojo:1446`).

Nesting is unbounded. **[compiled]** Two levels deduce in one signature:

```mojo
def demorgan[L: BoolV, R: BoolV](v: Not[And[L, R]]) -> Or[Not[L], Not[R]]:
    return Or(Not(v.a.l.copy()), Not(v.a.r.copy()))
```

`not(and(a, b))` → `or(not(a), not(b))`.

The compiler source agrees. `KGEN/lib/MojoParser/ParamMatcher.cpp:516-534`
unifies two `StructType`s by comparing symbols and then recursing pairwise over
parameter positions, so depth is unbounded. **[read]** Note the guard at
`:498-499`: if the struct *symbols* differ, deduction fails outright — there is
no subtyping search. Out-of-order dependent parameters (`T[x+1, x]`) are
unsupported (`ParamMatcher.cpp:464-470`, TODO). **[read]**

### But overload ranking has no specificity rule

**[compiled]** This is the trap. Given

```mojo
def split[V: BoolV](v: V) -> List[String]:              # 1 parameter binding
def split[L: BoolV, R: BoolV](v: And[L, R]) -> ...      # 2 parameter bindings
```

the **generic fallback wins**, and `And(And(a,b),c)` returns one conjunct, not
three. Declaration order does not matter — I tested the specialization declared
first. This is not a bug: `mojo/docs/reference/function-declarations.mdx:816-844`
states resolution rule 4 as *"Pick the candidate with a shorter parameter list…
The parameter list also counts implicit parameters synthesized from argument
types"*, with the explicit note *"a concrete function wins over a parameterized
one"*. `KGEN/lib/MojoParser/OverloadFitness.cpp:210-243` implements exactly
`paramBindings.size() < other.paramBindings.size()`, and
`OverloadSet.cpp:135-200` applies no C++-style partial ordering at all. **[read]**

Rule 1 (fewer implicit conversions) precedes rule 4, which is why the
specialization *does* win against an erasing overload reached by an `@implicit`
conversion — see §3.

### A `where` clause does defeat rule 4

**[compiled]** Rule 4 prunes by *count*, but a `where` clause removes a
candidate entirely, and that happens during resolution. Marking the fallback
excludes conjunctions makes the specialization reachable:

```mojo
def pick[L: BoolV, R: BoolV](v: And[L, R]) -> Int: return 2
def pick[V: BoolV](v: V) -> Int where (not V.IS_CONJ): return 1
```
```
And -> 2
Or  -> 1
```

Two constraints on this, both hit while getting it to compile:

- **The discriminator must be a `Bool`, not a string.** `comptime IS_CONJ =
  Self.K.name == StaticString("and")` fails in the `where` clause with
  *"cannot evaluate call to non-builtin function"* pointing at
  `StaticString.__eq__` (`std/collections/string/string_span.mojo:575`). That is
  CLAUDE.md's *"the constraint solver will not evaluate a non-builtin function
  in a `where` clause"*, confirmed. A `comptime is_conj: Bool` on the kernel
  trait works. **This upgrades the optimizer spec's style preference for
  `is_conjunction: Bool` over `Self.K.name == "and"` into a hard requirement
  the moment a `where` clause is involved.**
- **It still does not recurse** — see §3.

---

## 2. Type reflection — there is none over struct *parameters*

**[read]** `std.reflection` (`mojo/stdlib/std/reflection/reflect.mojo`) is real
and reasonably rich, but it reflects **fields**, not parameters:
`field_count()`, `field_names()`, `field_types()`, `field_offset[…]()`,
`is_struct()`, `name()`, `base_name()`.

The complete KGEN primitive set is enumerated in
`KGEN/include/KGEN/KGENDialect/KGENAttrs.td`: `struct_field_types` (:1295),
`struct_field_names` (:1323), `is_struct_type` (:1490), `get_base_type_name`
(:1568), `type_conforms_to_trait` (:607), and so on. **There is no
`struct_param_count` / `struct_param_names` / `struct_param_types` attribute.**
`get_function_parameter_count` (:1132) exists but applies to *function*
generators and is not exposed — see the TODO at
`mojo/stdlib/std/reflection/function.mojo:127-130`. Neither `parameters_of` nor
`fields_of` exists anywhere in the Mojo repo.

So there is no first-class *"is type X an instantiation of generic struct G?"*.
Three workable substitutes exist, in descending quality:

1. **Signature matching** — declare the argument as `G[_, _]` and let deduction
   answer the question. This is what §1 measures, and it is the intended
   mechanism.
2. **`where (T == G[a, b])`** with `a, b` infer-only. Type `==` landed in v1.0.0
   (`mojo/docs/releases/v1.0.0.md:200`); the stdlib pattern is
   `std/atomic/atomic.mojo:245-252`.
3. **`reflect[T].base_name()` string comparison** — used in the stdlib
   (`std/builtin/device_passable.mojo:107-108`) but explicitly a workaround, and
   it has a live bug: it never folds inside `comptime assert`
   (`max/kernels/src/layout/tensor_storage.mojo:88-93`, MOCO-4353). **Do not
   build on this.**

For marrow's actual need — discriminating `AndKernel` from `OrKernel` inside
`BoolBinary` — none of these is needed. `Self.K.name` resolves. **[compiled]**
`comptime if Self.K.name == AndKernel.name:` inside `BoolBinary.conjuncts`
compiles clean against the real tree (`pixi run -e dev precompile`, 0 errors,
0 warnings) and splits `and` while leaving `or`/`xor` atomic. This confirms the
optimizer spec's independent finding, and it means the `comptime for` +
`conforms_to` pattern of `DynArray._dispatch` is **not** the ceiling: for a
kernel-parameterised node, a direct comptime read of the kernel's own constant
is available and cheaper.

Per CLAUDE.md's own precedent (`BoolBinary.prune`'s docstring: *"a kernel rather
than a match on `Self.K.name`"*), a `comptime is_conjunction: Bool` on
`BoolBinaryKernel` is the better spelling. That is a style choice; both work.

---

## 3. The build-site overload — viable, but splits only one level

**[compiled]** With the competitor being the *erasing* overload reached by an
`@implicit` conversion (marrow's actual shape — `BoxedValue` does **not**
conform to `Value`), the specialization is selected, because resolution rule 1
(fewer implicit conversions) fires before rule 4:

```
And  -> splitting overload
Or   -> erasing overload
Leaf -> erasing overload
```

**But it does not recurse.** `And(And(a, Or(b,c)), d)` yields **2** conjuncts,
not 3:

```
conjuncts: 2
  - and(a, or(b, c))
  - d
```

The reason is precise and worth recording. Inside `split[L, R](p: And[L, R])`,
the expression `p.l` has declared type `L` — an opaque trait-bound parameter.
Overload resolution runs against the *declared* type, so the `And[_, _]`
candidate cannot match and the call falls to the erasing overload. Calling
`split` on that same inner node from a **concrete** site in `main` splits it
correctly. So the recursion dies at the first generic boundary.

Adding the `where` clause of §1 does not rescue it; it converts the silent wrong
answer into a compile error. **[compiled]**

```
error: invalid call to 'pick': lacking evidence to prove correctness
note: constraint declared here needs evidence for 'not R.IS_CONJ.__bool__()'
```

Inside the specialization neither candidate can be selected: the `And[_, _]`
candidate cannot match an opaque `L`, and the fallback's constraint cannot be
proven for it.

**Trait-method dispatch does not have this problem**, and that is the whole
difference between the two designs: a trait method call on `self.l` resolves
against the *monomorphized* conformer, so the override is found at every depth.
**[compiled]** The trait-default form returns 3 conjuncts for `(a AND b) AND c`
and 5 for a depth-5 left-deep chain, and correctly refuses to split
`And(Or(x,y), a)` past the `Or`.

**Recommendation: reject the build-site overload.** The optimizer spec already
rejects it for coverage reasons (misses already-boxed predicates, misses
conjunctions synthesized by a later rule). Add a stronger one: without a `where`
clause it is *silently wrong* for the common case — a left-deep
`a AND b AND c` is exactly what a multi-conjunct `WHERE` produces, and it
returns two conjuncts while compiling cleanly. With a `where` clause it does not
compile at all.

---

## 4. Rebuilding — the sharp line, and it is not where the optimizer spec drew it

The optimizer spec says the boundary is *"rebuilding an arbitrary interior
node"*, and that *"nothing can hand back 'this subtree with the third literal
replaced by 5', because that names a type that does not exist yet"*. The second
half is not the operative constraint — naming a not-yet-existing type is fine.
The operative constraint is **conditionality**.

### 4a. Unconditional rebuild: works, recursively, over arbitrary trees

**[compiled]** A per-node associated type plus a method returning it compiles
and runs, recursing to arbitrary depth:

```mojo
trait BoolV(Copyable, Deinitable, Movable):
    comptime Rewritten: BoolV
    def rewrite(self) -> Self.Rewritten: ...

struct BinOp[K: Kernel, L: BoolV, R: BoolV](BoolV):
    comptime Rewritten = BinOp[Self.K, Self.L.Rewritten, Self.R.Rewritten]
    def rewrite(self) -> Self.Rewritten:
        return Self.Rewritten(self.l.rewrite(), self.r.rewrite())
```

And a **genuinely type-changing, genuinely conditional-looking** rewrite works
too, as long as the conditionality is *distributed* — each node states its own
image rather than a driver testing its children's kind. De Morgan plus
double-negation elimination, in full:

```mojo
struct AndOp[L: BoolV, R: BoolV](BoolV):
    comptime Negated = OrOp[Self.L.Negated, Self.R.Negated]
    def negate(self) -> Self.Negated:
        return Self.Negated(self.l.negate(), self.r.negate())

struct OrOp[L: BoolV, R: BoolV](BoolV):
    comptime Negated = AndOp[Self.L.Negated, Self.R.Negated]
    ...

struct NotOp[A: BoolV](BoolV):
    comptime Negated = Self.A          # double-negation elimination
    def negate(self) -> Self.Negated:
        return self.a.copy()

struct Leaf(BoolV):
    comptime Negated = NotOp[Leaf]
```

Output **[compiled]**:

```
expr      : and(a, or(b, not(c)))
negated   : or(not(a), and(not(b), c))
double neg: and(a, or(b, not(c)))
```

That is a real optimizer rewrite — pushing negation to the leaves — executing
in the fused lane, producing new fused types, over a tree of arbitrary shape.
**So "AOT expressions cannot be rewritten" is false as stated.**

### 4b. Conditional rebuild: does not work, and `rebind` does not save it

**[compiled]** The moment a node must branch on *what its child is*, the output
type becomes a comptime conditional and the construction site rejects it:

```mojo
struct NotOp[A: BoolV](BoolV):
    comptime Rewritten = OrOp[
        NotOp[Self.A.Lhs], NotOp[Self.A.Rhs]
    ] if Self.A.KIND == KIND_AND else NotOp[Self.A.Rewritten]

    def rewrite(self) -> Self.Rewritten:
        comptime if Self.A.KIND == KIND_AND:
            return Self.Rewritten(NotOp(self.a.lhs()), NotOp(self.a.rhs()))
```

```
error: no matching function in initialization
    return Self.Rewritten(
note: candidate not viable: missing required argument: 'copy'
```

Inside the `comptime if` that *selected the branch*, `Self.Rewritten` still
reduces only to its trait bound, so the only visible constructors are
`Copyable`/`Movable`'s synthetic ones. The `rebind` escape fails identically,
and the diagnostic prints the unreduced type verbatim:

```
note: 'NotOp[A].Rewritten' is aka
      'OrOp[NotOp[A.Lhs], NotOp[A.Rhs]] if (A.KIND == Int(1)) else NotOp[A.Rewritten]'
```

This is CLAUDE.md's recorded limit — *"A comptime conditional type carries no
trait conformance and does not reduce at a return site, even inside a
`comptime if` that selected the branch. `rebind` cannot bridge it."* — now
confirmed for the rewrite shape specifically.

### 4c. The cost of the distributed form: no defaulting, every node in one commit

**[compiled]** The associated-type method **cannot be defaulted on the trait**,
even when the associated type has a default that is literally the returned type:

```mojo
trait BoolV(...):
    comptime Negated: BoolV = NotOp[Self]
    def negate(self) -> Self.Negated:
        return NotOp(self.copy())
```
```
error: cannot implicitly convert 'NotOp[_Self]' value to '_Self.Negated'
```

This is CLAUDE.md's *"a trait-level default method cannot return
`Self.AssocType`… every conformer must implement it in the same commit"*,
confirmed. **`marrow/exprold/values.mojo` has 34 `Value`-family conformer structs**
plus `DynValue`. Any distributed type-level rewrite is therefore a 35-struct
change with no incremental path — and each new node type added later must
implement it or fail to compile.

That is the honest price of 4a, and it is why I would not build one today. But
it is a *cost*, not an *impossibility*, and the distinction matters for the
roadmap: if predicate normalisation (NNF, negation pushdown) is ever wanted in
the fused lane, the mechanism exists.

---

## 5. Measured cost of `conjuncts()` — it exceeds the proposed +0.25% gate

Nobody had measured this. I implemented the full shape against the real tree —
`conjuncts()` on `Value` defaulting to `[BoxedValue(self)]`, overridden on
`BoolBinary` (gated on `Self.K.name == AndKernel.name`) and on `DynValue`
(gated on `_tag`), an eighth trampoline slot `_conjuncts_fn` on `BoxedValue`,
and `Relation.filter` splitting into one `Filter` per conjunct — and ran
`pixi run binary_size`. **The implementation was throwaway and has been
reverted**; `git status` is clean apart from this document.

Two things worth recording before the numbers:

- **A trait default *can* construct `BoxedValue(self)`.** **[compiled]** This
  was not obvious; it is the erasing box being built from inside the trait it
  erases.
- **A same-signature struct method *does* override a trait default.**
  **[compiled]** CLAUDE.md's *"a struct method does not override a trait default
  — the two become competing overloads and every call reports `ambiguous call to
  'x'`"* is too broad. That rule was learned from `Value.isnull` vs
  `DynValue.is_null`, which differ in **return type**. With identical
  signatures the override is selected, at every depth of a recursive call — it
  is what makes §5 work at all, and it is what `Value.prune` / `name` /
  `bound_column` already rely on. The CLAUDE.md entry should be narrowed to say
  *differing* signatures.
- **A trampoline field whose function type mentions `BoxedValue` compiles.**
  **[compiled]** `var _conjuncts_fn: def(ArcPointer[NoneType]) thin ->
  List[BoxedValue]` is accepted, despite `_resolve_names_fn`'s docstring warning
  that a field mentioning the erased type triggers *"struct has recursive
  reference to itself"*. The difference is that `List` is heap-indirect, so the
  struct's size stays computable. The existing warning is about `DynValue`
  specifically and should not be read as a general rule.

### Slot + trampoline only (no conjunction anywhere in the plan)

| gate | baseline `__text` | with `conjuncts()` | delta |
|---|---|---|---|
| `query_streaming` | 1,457,764 | 1,458,824 | **+1,060 (+0.073%)** |
| `query_exprs` | 1,538,224 | 1,538,116 | **−108 (−0.007%)** |
| `query_dynvalue` | 4,913,140 | 4,915,188 | **+2,048 (+0.042%)** |

Comfortably inside +0.25%. **[measured]**

### With one real two-conjunct AOT predicate

The gates ship no conjunctive predicate, so the table above does not measure the
thing that actually costs — boxing interior operands that would otherwise never
be boxed. I changed `query_streaming`'s predicate to
`(col("a") > col("b")) & (col("a") < lit(100, int64))` and built it **both
with and without** the machinery, same gate source in both:

| `query_streaming`, conjunctive predicate | `__text` | symbols |
|---|---|---|
| baseline (no `conjuncts()`) | 1,514,160 | 649 |
| with `conjuncts()` + splitting `filter` | 1,523,336 | 677 |
| **delta** | **+9,176 (+0.606%)** | **+28** |

**[measured]** That is **2.4x the +0.25% gate the optimizer spec proposes.**

The cost model this implies: splitting an N-conjunct predicate instantiates
**N additional complete `BoxedValue` erasures** — nine trampolines each — for
sub-nodes (`Gt[NumericColumn, NumericColumn]`,
`Lt[NumericColumn, NumericLiteral]`) that previously appeared only *inside* the
`And`'s type and were never boxed. It scales with the number of conjuncts and
with the number of distinct conjunct *shapes* across the program, not with the
number of queries.

**Implication for the optimizer spec.** Its recommendation — ship `conjuncts()`
in phase 2, when predicate pushdown below `Join`/`Aggregate` actually consumes
it — is reinforced, not weakened: the slot is nearly free, but *using* it is
not. If the +0.25% gate is to be kept, it must be measured against a gate that
contains a conjunction, and the gate number will need to move or the rule will
need to be justified on query-performance grounds that outweigh 0.6%.

I did not measure a 3+-conjunct predicate or the runtime-lane equivalent; the
model above predicts roughly linear growth but that is **inference, not
measurement**.

---

## 6. Recursion and instantiation depth — not a practical concern

**[read]** Two independent mechanisms exist, and neither bites here.

- `--elaboration-max-depth` defaults to `std::numeric_limits<unsigned>::max()`
  (`KGEN/tools/mojo/Common/CompilationOptions.td:132-138`,
  `mojo/docs/releases/v0.26.1.md:941-943`). The error, when it fires, is
  `"elaborator expansion is N levels deep - infinite recursion?"`
  (`KGEN/lib/Elaborator/Elaborator.cpp:697-705`).
- A separate SCC cycle detector catches mutual recursion regardless of depth:
  `"function instantiation in parameter domain that recursively requires
  itself"` (`KGEN/lib/Elaborator/ParametricElaborator.cpp:2316-2329`).

**[compiled]** A depth-5 left-deep conjunction split correctly with no
diagnostic, and the in-repo `precompile` of the full machinery was clean.

The important point is structural, not numeric: **decomposition creates no new
types.** `conjuncts()` adds one method to node types that already exist and
boxes sub-nodes into the single non-generic `BoxedValue`. The per-depth
instantiation blow-up the brief worried about belongs to §4a rebuilding, where
each level *does* name a new type — and even there the count is bounded by the
tree's node count, not exponential in it.

---

## 7. The honest fallback, and what marrow loses

The fallback is **not** needed for conjunction splitting — that works in both
lanes. It is needed for the rules that require conditional rebuilding:
**constant folding** and **CSE**. Both stay runtime-lane only, and I agree with
the optimizer spec that this costs marrow close to nothing:

- **Constant folding.** In the fused lane a `NumericLiteral` is already a
  broadcast constant inside the SIMD lane body, so LLVM folds `lit(2) + lit(3)`
  during codegen. A plan-level folder would duplicate the backend. **[read,
  from the optimizer spec's reasoning, which I did not independently benchmark.]**
- **CSE.** A fused subtree is one SIMD loop; hoisting a common sub-expression
  out of it *materialises an intermediate buffer that does not currently exist*.
  In the fused lane CSE is plausibly a pessimisation, not an optimisation.

So the answer to the brief's question 5 is: **yes, that is the right answer, and
it is cheaper than it sounds** — but the reason is not "the fused subtree is
already one SIMD loop, so interior rewrites buy little" in general. Negation
pushdown is an interior rewrite that would buy something (it enables pruning
through `NOT`), and §4a shows it *is* achievable. The rules that are genuinely
lost are exactly the two whose value the fused representation already captures
by other means.

### Consequence for the golden corpus

The corpus exists to catch a feature present in only one lane, so a rule firing
in one lane and not the other is a design consequence, not a detail. The
distinction that matters:

- A rule that changes **results** in only one lane is a bug. Nothing here
  proposes one.
- A rule that changes **plan shape** in only one lane is unavoidable — a fused
  `And` and a tagged `DynValue` are different objects. Constant folding will
  reshape a runtime plan and not a fused one.

Since the corpus asserts **results**, not plan shape, constant folding being
runtime-only is invisible to it, which is correct. The thing the corpus *must*
gain alongside conjunction splitting is a **Kleene case**: splitting
`a AND b` into `filter(a).filter(b)` is sound only because `FilterProcessor`
keeps `True` and drops `NULL`. A `filter_and_with_nulls` case belongs in the
kleene area, run through both lanes, before the rule lands.

---

## What I could not settle

- **Whether the cost in §5 grows linearly past two conjuncts**, and the
  runtime-lane equivalent of that measurement. Inferred, not measured.
- **Whether a *parametric* associated type** (`comptime AddWith[R: Value]`) is
  expressible in a Mojo trait. If it were, constant folding could in principle
  be encoded as unconditional double dispatch in the type domain — CPS-style,
  and almost certainly not worth it, but it is the one route I did not close.
- **The +0.25% gate number itself.** I measured that the rule exceeds it; I did
  not establish what the gate *should* be, which is a judgement about how much
  binary size a join-pushdown win is worth.
