# Simplification backlog — `marrow.expr`

Re-evaluated 2026-07-30 against the tree *after* Step 3 landed. Everything below
was measured mechanically — duplicate method bodies compared as text, callers
counted with grep — and each proposed fix was **tried**, not reasoned about. Two
of the four items in the first draft turned out not to be actionable at all, and
saying why is the point of this document.

Current: `values.mojo` is 3,153 lines with **93 duplicated method bodies**.

---

## Done

| what | result |
|---|---|
| **32 one-line `materialize` overrides** | **Removed.** The family traits default `materialize` again; only `DynValue` implements it. |
| `count_distinct`/`approx_count_distinct` defined 3× | Hoisted to `Value`. |
| `ConditionalBinaryKernel` + `CoalesceOp`/`NullifOp` — a second notion of "kernel" in the expression layer | Moved to `kernels/conditional.mojo`. |
| 14 repeated `comptime if Self.IsErased` blocks | Centralised into the three family drivers. |

The `materialize` one is the interesting one, because **it was not fixable when
first recorded**. The overrides existed because conforming `DynValue` to several
families made the defaults conflict, and implementing it manually then hit
`attempt to resolve a recursive reference to declaration 'DynValue.materialize'`.

Splitting `NumericOps` off `NumericValue` — done for an unrelated reason, to fix
a recursion that broke *external* builds — removed the cause. Re-testing an item
that had been ruled out is what found it. **+0 bytes on all eleven gates.**

---

## Not actionable — and why

### `referenced_columns` / `validity` / `prepare` — 69 duplicated bodies

The largest cluster by far, and there is no mechanism in Mojo to remove it:

| method | copies | body |
|---|---:|---|
| `referenced_columns` (unary) | 23 | `self.a.referenced_columns()` |
| `validity` (unary) | 9 | `self.a.validity(batch)` |
| `prepare` (unary) | 8 | `self.a.prepare(batch, ctx)` |
| `referenced_columns` (binary) | 7 | `_union_columns(l, r)` |
| `prepare` (binary) | 5 | `l.prepare(); r.prepare()` |
| `validity` (binary) | 3 | `Bitmap.intersect(l, r)` |
| column leaves | 14 | `_name`-based `__init__`/`name`/`validity`/`referenced_columns` |

The obvious fix is `UnaryNode`/`BinaryNode` traits supplying defaults. **Traits
cannot require `var` fields** — the compiler is explicit: `traits do not support
'var' fields; use 'comptime' to declare associated types`. A default body cannot
mention `self.a`, so it cannot be written.

A helper function does not help either: the bodies are *already* one-liners, so
`return _unary_columns(self.a)` trades one line for one line.

This is a language limitation, not a design flaw. **Accepted, not open.**

### Hoisting `count` to `Value`

Tried; it does not compile. The first draft said `count` was "byte-identical in
`StringValue` and `TemporalValue`" — true, but it compared only those two.
`NumericValue.count` returns `Count[Self]`, a `Reduction` node, not an
`AggExpr`. Hoisting the other two to `Value` makes a three-way conflict and
every node has to implement it manually — 30 new bodies to remove 1.

Same for `min`/`max`: three families, three genuinely different `Aggregation`s.

---

## Open

### `render` — 6 duplicated bodies

`String(Self.K.name, "(", self.l.render(), ", ", self.r.render(), ")")` ×4 and
the unary form ×2. Same field-access problem as above, so the same limit
applies — but only 6 bodies, so the payoff is small either way.

### Column leaves — 4 near-identical structs

`NumericColumn`, `StringColumn`, `TemporalColumn`, `ListColumn` differ only in
which family they report to; between them they repeat `__init__`, `name`,
`validity`, `referenced_columns` and `materialize`. A single generic leaf is
blocked by the same reason the families exist at all (`vectorwise` vs
`elementwise` have different signatures), but the *non-family* half — the four
`_name`-based methods — is 14 of the 93 duplicated bodies and is worth another
look if a mechanism ever appears.

---

## Non-findings, checked and dismissed

- **`_rank` / `_numeric_rank`** — cannot be one function (comptime type → `Int`
  vs runtime value → `Int`). Enforcement was the gap; `test_numeric_rank_agrees_across_lanes` closed it.
- **`_op_name`** — gone with the interpreter. It was never duplication of the
  kernels' names: the vocabularies deliberately differ (`sub` vs `subtract`).
- **`ConcatKernel.dispatch`** — not redundant with `apply`; `apply` is typed,
  `dispatch` is the erased entry the runtime `+` needs.

---

## The erased box must not claim the typed traits (landed 2026-08-03)

**The finding.** `DynValue: NumericValue` was **unsound**, and the long-running
`NumericOps` split was a symptom. `NumericValue` makes two comptime promises and
the box stubbed both:

- `comptime OutType: NumericType` — the box's is `DynType`, whose `native` is a
  placeholder `DType.bool` (`dtypes.mojo:190`), not a real lane type;
- `vectorwise[W] -> SIMD[OutType.native, W]` — the box has no lane; it `abort`s.

It satisfied the signature and none of the contract. The compiler's
`attempt to resolve a recursive reference to declaration 'DynValue.__gt__'` was
that lie surfacing as a conformance cycle — which is why five attempts to fix it
*as a compiler problem* (including a faithful minimal prototype in
`~/Workspace/dtype-proto`, which never reproduced it) all failed.

**The rule.** Erasing into a trait whose members are all *runtime methods* is
sound — the box dispatches. Erasing into one with *comptime* members the box can
only stub is not. So:

- `DynValue` — drop `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue`,
  keep `Value`. **Done**, and it dissolved the cycle outright.
- `DynType` — same lie across 8 family traits (`NumericType`, `IntegerType`,
  `FloatingType`, `DecimalType`, `IntervalType`, `TemporalType`,
  `StringLikeType`, `ListLikeType`). **Done**, along with the placeholder
  `comptime native = DType.bool` those conformances existed to satisfy.
- `DynArray: Array` — **sound, keep it.** `ScalarType = DynScalar`
  (`arrays.mojo:2225`) is genuine and `__getitem__` really returns one; every
  other `Array` member is a runtime method.

**What this bought.** `NumericOps` is **deleted** — one `NumericValue` again,
with all 31 members including the six comparisons and `materialize`.

### Split the weld (the sketch that became the decision)

The `comptime if conforms_to(Self.L, NumericValue)` gate is not a fix, it is the
defect restated. A node that has to *ask* whether its operand is numeric should
have demanded a numeric operand. The reason it cannot is that **each of these 14
structs is two nodes welded together**: a fused node with a lane, and an erased
node that dispatches on runtime dtype, chosen by `IsErased`. Neither the operand
bound nor the responsibility can be stated honestly while both live in one type.

    NumericBinary  NumericUnary  NumericCast   FloatBinary   NumericCompare
    BoolBinary     BoolUnary     StringPredicate  IsIn        StringLength
    Reduction      ConditionalBinary  CaseWhen  TemporalExtract

So split them:

- **fused node** — `NumericCompare[K, S, L: NumericValue, R: NumericValue]`. No
  `_erased`, no `IsErased`, no gate. Responsibility: evaluate a lane.
- **erased node** — holds `DynValue` operands, body is today's `_erased`
  verbatim, conforms to `Value` only. Responsibility: dispatch on runtime dtype.

`DynValue`'s 47 methods build the *erased* nodes, so `Gt[DynValue, DynValue]`
never forms and nothing needs a relaxed bound.

Consequences:

- **`IsErased` disappears entirely** — all 42 mentions. The type is the answer,
  so there is nothing to propagate and the failure its own docstring warns about
  ("not a wrong answer, it is a build failure") becomes unrepresentable.
- Every `conforms_to` gate and every `Value`-relaxed operand bound reverts.
- Most of the 28 errors listed above evaporate — they are all downstream of the
  dishonest bound.
- Parameterizing by kernel should collapse 14 into ~5-6 erased structs
  (`ErasedBinary[K]`, `ErasedUnary[K]`, `ErasedCompare[K, S]`, ...).

**Size, both directions — gate it.** Today a fused `Add[Int32Col, Int32Col]`
still carries an `_erased` body (`into_array` + kernel dispatch), dead weight in
an AOT binary unless DCE removes it; after the split a fused node cannot
reference the erased path at all. Against that, the -1.28 MB Step-3 win came
precisely from erased trees *reusing* fused node types, and this stops that
reuse. Measure `query_dynvalue` and a fused gate with `size -m` -> `Section
__text` before and after; do not trust stripped file size (16 KB page-quantized).

### DECIDED (2026-07-30) — two lanes, two files, no shared node types

Chosen after the soundness finding above; supersedes both the 28-error fix list
and the split-the-weld sketch. Size concern was raised and overruled — **gate it**
(below).

**`marrow/expr/values.mojo` — the AOT lane, strictly typed.**

- Every node operand bound on its family trait (`L: NumericValue`, `C: BoolValue`,
  ...). No `Value`-bounded operands, no `comptime if conforms_to` gates.
- **Delete `IsErased`** — all 42 mentions, including `Value.IsErased` and the
  `comptime if Self.IsErased` branch in `_numeric_fused`/`_bool_fused`/
  `_string_fused`.
- **Delete all 14 `_erased` methods** and `Value._erased`. A fused node loses its
  second half; the bodies move to the erased nodes below.
- `NumericOps` merges into `NumericValue` — trivial once `DynValue` is out of the
  file, since the conformance cycle was `DynValue` claiming the family traits.
  (Verified: with the box conforming to `Value` only, the merge builds externally.)

**`marrow/expr/dynamic.mojo` — the erased lane, tagged.**

- `DynValue` moves here from `values.mojo:2602-2929` and is rewritten around a
  tag rather than a fn-pointer trampoline.
- The tag stays **comptime where it can be**: `K.name` already exists on every
  kernel and `NumericCompare.prune` already switches on it
  (`comptime n = Self.K.name; comptime if n == "equal"`). Prefer `ErasedBinary[K]`
  over a runtime string switch wherever the operation is known at construction.
- A **runtime** string tag is warranted only at the frontend boundary, where the
  operation genuinely arrives as data (`DynValue.aggregate(func: String)`,
  `DynAgg`, a future SQL/wire frontend). Resolve the string to a kernel **once**,
  there, then build the comptime-tagged node. This is also the honest answer to
  **L3** (`AggFunc`'s late binding).

**`marrow/dtypes.mojo`.** `DynType` drops `NumericType`, `IntegerType`,
`FloatingType`, `DecimalType`, `IntervalType`, `TemporalType`, `StringLikeType`,
`ListLikeType` — same lie as `DynValue`'s, `comptime native: DType = DType.bool`
at `:190`. Keep `DataType`. Expect fallout at `promote`/`wider`/`_rank` and every
`dispatch_*` caller; those want the *runtime* dtype, not a fake comptime one.
`DynArray: Array` **stays** — `ScalarType = DynScalar` is genuine.

**Gate before believing any of it.** Measure `size -m <binary>` -> `Section
__text`, never stripped file size (16 KB page-quantized on Apple Silicon):

- `query_dynvalue` — the risk. Deleting the 41-tag `TagValue` interpreter won
  **-1,279,552 B (-24.4%)**; a runtime-tag design can hand that back. A comptime
  `ErasedBinary[K]` should not, because only named kernels link.
- a fused gate (e.g. `query_streaming`) — the expected *win*: a fused node no
  longer carries an `_erased` body reachable via `into_array` + kernel dispatch.

Build one gate directly (`mojo build -O3 -g0 -I . ...`, ~2.5 min) rather than the
full `pixi run binary_size` sweep (~10 min).

### Landed 2026-08-03 — built, tested, gated

**What is in the tree.**

1. `values.mojo` is the AOT lane only — 3,153 -> 2,558 lines. `IsErased`, all 14
   `_erased` methods and `Value._erased` are gone; `NumericOps` is merged back
   into `NumericValue`; no operand bound is relaxed to `Value` except
   `NullPredicate`'s, which is honest (see below).
2. `dynamic.mojo` is the runtime lane — `DynValue` is a tag, a
   `List[ArcPointer[Self]]` of children and a `DynPayload`, conforming to `Value`
   and nothing else. `resolve_names` is an ordinary recursive walk; the eight
   trampolines it needed are gone.
3. `relations.mojo` gains `BoxedValue`, the erasure box both lanes go through.
   Every operator (`Filter`/`Project`/`Sort`/`Aggregate`/`Join`, and the
   `Processor`s) holds one, so each compiles exactly once.
4. `DynType` drops the 8 family traits and the placeholder
   `comptime native = DType.bool` they existed to satisfy.

**Two defects the deleted `IsErased` left behind, both found by running the
suite, both fixed.**

- `StringPredicate` and `ConditionalBinary` lost their `Breaker` conformance
  when the conditional `IsBreaker` was removed, while keeping the `ctx.get(slot)`
  read in `vectorwise`. Nothing filled the slot, so *every* `coalesce`, `nullif`
  and string comparison aborted with `index 0 is out of bounds` — one runner
  crash reported as 771 failures. A struct that reads a slot must conform to
  `Breaker`; that invariant is now checkable by grep and holds for all 15.
- The runtime lane silently lost four behaviours the fused lane has, because a
  tag arm is written per operation and four were written to the wrong rule:
  `+` on strings (concatenate, not add), `/` and `**` (always float64 — the
  fused `FloatBinary`, so `5 / 2 == 2.5` rather than `2`), `sqrt`/`exp`/`ln`
  over an integer column (cast up, rather than raising from
  `dispatch_floating`), and `prune` (returned "unknown" for everything, which
  disables row-group and page skipping for every runtime predicate). Also
  restored: `length()` and `isin()`, which had no tag at all.

**Removed, deliberately.** `Value.is_deterministic` — a default returning True
with no caller in the library, asserted only by two tests.

**The tag must not select the kernel.** The runtime lane was first written with
one `_eval` switch of ~70 arms, each naming a `Kernel.dispatch`. Every arm is
then reachable from every `DynValue`, so building *one* linked *all* of them —
the 41-tag `TagValue` interpreter's cost re-created, measured at **+1,807,168
bytes of `__text` (+45.7%)**, the whole win Step 3 got for deleting it.
`kernels::execution` went 27 -> 1,452 symbols, `views` 38 -> 1,265,
`kernels::numeric` 8 -> 611.

The fix is this document's own prescription — make the operation comptime, since
it *is* known when the node is built — but it does not need a struct per
operation. The children, the payload and every walk over them already live on
`DynValue`; parameterising the struct would duplicate all of that per kernel to
buy one property. So the node carries a **function pointer** instead:

    comptime EvalFn = def (List[DynArray], DynPayload, RecordBatch)
        thin raises -> DynArray

`__gt__` names `_compare[GtKernel, StringGtKernel]`, `__sub__` names
`_binary[SubKernel]`, and `_eval` is three lines with no switch: evaluate the
children, call the pointer. A program links exactly the kernels its expressions
mention. The signature names no `Self` on purpose — a field whose function type
mentions its own struct is rejected — which is why children arrive
pre-evaluated. `_tag` survives for `render`/`prune`/`name`, none of which
reference a kernel.

**Gate (`size -m` -> `Section __text`, HEAD rebuilt in a worktree for the
baseline, not quoted from the log).**

| gate | HEAD | flat switch | comptime `EvalFn` |
|---|---:|---:|---:|
| `query_streaming` (AOT lane) | 1,303,028 | 1,302,900 | **1,302,900** (-128, -0.01%) |
| `query_dynvalue` (runtime lane) | 3,956,596 | 5,763,764 | **3,984,756** (+28,160, +0.71%) |

Both lanes hold. The AOT result also settles a question this document raised in
both directions: deleting the `_erased` half of every fused node gained 128
bytes, not a meaningful win — DCE was already removing it — and splitting the
lanes did *not* cost the AOT binary anything by ending node-type reuse, which
was the stated risk. The runtime lane's residual +0.71% is the `_eval_fn`
indirection plus the string halves of the comparison kernels; it is not the
45.7% cliff, which was the thing to avoid.

**Residue, all deliberate, none blocking.**

- `NullPredicate[K, DynValue]` is the one fused node still instantiated over the
  box, because `isnull`/`notnull` are defaults on `Value` itself. Its bound is
  `A: Value` and it calls only runtime methods on it, so this is sound — but it
  means `col("x").isnull()` is the one runtime-lane operation that hands back a
  fused node. Moving the two defaults onto the family traits would close it, at
  the cost of a default duplicated 4-5 times — the shape that already broke once
  (see `count_distinct`'s comment at `values.mojo:399`).
- The runtime-lane *builders* (`col(name)`, `lit[T](v)`, `if_else`, `coalesce`,
  `case_when`) still live in `values.mojo`, so the AOT-lane file imports
  `DynValue`. They are overloads of the fused builders and moving them would
  split one name across two modules; left alone until that is worth doing.

#### Shape constraint for the tagged node (measured, `gate_selfref.mojo`)

A tagged `DynValue` stores children instead of trampolines, so its children field
had to be settled first:

| field | result |
|---|---|
| `var children: List[Self]` | **rejected** — `field 'children' has non-implicitly deletable type 'List[Direct]'` |
| `var children: List[ArcPointer[Self]]` | **compiles and runs** (nested tree, `depth: 3`) |

So children are `List[ArcPointer[DynValue]]`. Note the failure is about
*deletability*, not the self-reference itself — a different mechanism from the
`struct has recursive reference to itself` that forces `_resolve_names_fn` to
hand back an erased pointer today, so removing that workaround needs its own
check rather than being assumed to fall out.

Worth carrying into the rewrite: with children stored directly, `_resolve_names_fn`
and the other seven trampolines have no reason to exist — `resolve_names` becomes
an ordinary recursive walk that rebuilds the tree.
