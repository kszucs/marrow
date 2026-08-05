# Expression evaluation: typed state, and one traversal

Design for the AOT lane, `marrow/expr/values.mojo`. Supersedes the ad-hoc
protocol described in the "Known follow-ups" comment at the top of that file.

Status: **design, with its central mechanism validated by spike (2026-08-03).**
Nothing is implemented. The binary-size question is open and is the first gate.

---

## 1. What is wrong today

A fused expression evaluates in two passes. `prepare` walks the tree and each
pipeline breaker appends its whole-column result to a `Context`; the fused loop
then walks the same tree calling `vectorwise[W](batch, ctx, mut slot, idx)`,
incrementing `slot` and reading `ctx` positionally. Note the loop resets
`var slot = 0` **inside** `producer[W](i)` (`values.mojo:551`), so the second
walk happens once per SIMD chunk, not once per batch.

### 1.1 The slot correspondence is unenforceable

No index is stored. `prepare` allocates by arrival order, `vectorwise` resolves
by visit order, and they agree only because both were hand-written to walk
identically — across 14 composite overrides. Three failure modes:

1. **Order swapped, slot types identical** → compiles, both reads succeed, the
   query returns **the wrong column with no error**. `coalesce(a,b) +
   nullif(a,b)` is a two-int64-slot instance.
2. Order swapped, types differ → the `Variant` accessor trips at run time. Loud,
   but by luck of the operand types.
3. A non-`Breaker` composite forgets to override `prepare` → nothing is
   appended, `vectorwise` still increments → out of bounds. **`DateTrunc`
   (`values.mojo:2271`) is this case today** — latently wrong, unreachable only
   because no temporal breaker exists to sit beneath it.

Counts: 42 structs (37 value nodes), 29 with `vectorwise`, `prepare` at 15 sites
(the trait default plus 14 overrides). Breakers are insulated only by accident —
`materialize` routes its operand through the single-argument `execute`
(`values.mojo:329-332`), which allocates a fresh `Context`; the two-argument
overload is equally in scope and would corrupt the numbering.

### 1.2 The protocol is six methods and a marker trait

`materialize`, `execute(batch, ctx)`, `execute(batch)`, `prepare`, `vectorwise`,
`validity`, plus `trait Breaker` — all of it in service of staging breakers.

### 1.3 `validity` is opt-in, and two nodes forgot

22 of 33 nodes override `validity`; the other 11 inherit "all valid". Nine of
those are correct by construction. Two are not, and both are **verified by
execution**: `StringValue.materialize` never calls `validity` at all, so every
string transformation drops nulls; and `StringToNum`/`StringToBool` do not
propagate parse failures, so `to_int("x") + 1` yields `0`, not null. Nothing
distinguishes "no nulls by construction" from "nobody wrote this method".

### 1.4 The same recursion is written many times

37 `referenced_columns` overrides collapse to 5 distinct bodies; 22 `validity`
to 4; 14 `prepare` to 2; 7 `render` to 2. **84 bodies, 17 shapes.**

### 1.5 Work is repeated

`ConditionalBinary` and `CaseWhen` call `_result` from *both* `validity` and
`materialize`, running their kernel twice per morsel. `BoolBinary.validity`
re-executes **both entire subtrees** with fresh contexts to read their bitmaps —
so `(s LIKE 'a%') AND (t LIKE 'b%')` over nullable strings is four scans for two
predicates. And `NumericColumn.vectorwise` resolves its column by name — a
string-comparison scan of the schema plus a `Variant` ladder — **once per SIMD
chunk**, inside the loop the design exists to fuse.

---

## 2. The design — a node's prepared state is its own typed value

Replace the positional `Context` with an associated type. The state **value**
lives outside the node, exactly as it does today; only its **type** is declared
by the node, which is what lets the compiler pair writer with reader.

```mojo
trait Value(Copyable, ImplicitlyDeletable, Movable):
    comptime OutType: DataType
    comptime OutShape: Int
    comptime State: Copyable & ImplicitlyDeletable

    def prepare(self, batch: RecordBatch) raises -> Self.State
    # family trait adds exactly one lane method:
    #   vectorwise[W](self, batch, state: Self.State, idx: Int) -> SIMD[...]
```

A fused node composes its children's states:

```mojo
struct NumericBinary[K: BinaryNumericKernel, L: NumericValue, R: NumericValue]:
    comptime State = Tuple[Self.L.State, Self.R.State]

    def prepare(self, batch) raises -> Self.State:
        return Tuple(self.l.prepare(batch), self.r.prepare(batch))

    def vectorwise[W: Int](self, batch, state: Self.State, idx: Int) -> ...:
        return Self.K.core[Self.OutType.native, W](
            self.l.vectorwise[W](batch, state[0], idx),
            self.r.vectorwise[W](batch, state[1], idx),
        )
```

A breaker is no longer special. Its state simply *is* the materialized column:

```mojo
struct StringLength[A: StringValue]:
    comptime State = Int32Array

    def prepare(self, batch) raises -> Self.State:
        return LengthKernel.dispatch(into_array(self.a.execute(batch), batch.num_rows()))

    def vectorwise[W: Int](self, batch, state: Self.State, idx: Int) -> ...:
        return state.values().load[W](idx)
```

A column leaf's state is its **resolved index**, computed once:

```mojo
struct NumericColumn[T: NumericType]:
    comptime State = Int
    def prepare(self, batch) raises -> Self.State:
        var i = batch.schema.get_field_index(self._name)
        if i == -1:
            raise Error("column not found: ", self._name)
        return i
```

The driver hoists `prepare` out of the chunk loop:

```mojo
var state = self.prepare(batch)                  # once per batch
@parameter
def producer[W: Int](i: Int) -> SIMD[native, W]:
    return self.vectorwise[W](batch, state, i)   # no slot, no ctx
```

### 2.1 Validity travels in the state

Each node's `prepare` returns its validity alongside whatever it needs to serve
lanes; the driver reads the root state's bitmap. This is not decoration — it is
what makes §1.3 unrepresentable, because a node **must** return a `State` and
cannot silently omit one. It also computes validity exactly once, bottom-up,
from already-prepared children, which removes the double-kernel-run and the
subtree re-execution in §1.5.

### 2.2 What this deletes

| Gone | Why |
|---|---|
| `Context`, `_slots`, `ctx.get`, `ctx.append` | no positional store |
| `mut slot: Int` on 29 lane signatures | no index to thread |
| `trait Breaker` | breaker-ness *is* having an array-shaped `State` |
| `materialize` | folds into `prepare` (breakers) or the driver (fused) |
| both `execute` overloads | one `execute` on the family driver |
| `validity` as a method | carried in `State` |
| the DFS-order invariant | **there is no order** |
| 14 hand-written `prepare` recursions | each becomes a one-line tuple build |

Six methods and a marker trait become **two methods**. Failure mode 1 becomes a
type error: a node can only read the state its own `prepare` produced.

### 2.3 Why not split logical from physical

The alternative considered — `Expr.compile() -> Plan`, with the plan owning its
state and the expression keeping no execution surface at all — was **spiked and
works** (`C1`–`C6` below). It is cleaner on paper: the logical node ends up with
zero execution methods, and expression reuse is safe by construction.

It was **rejected** because it creates 37 `Expr` + 37 `Plan` types mirroring each
other — structurally the same parallel hierarchy as `Relation`/`Processor`, in a
hotter path and at four times the node count. One struct per node is worth more
than the last increment of purity. State-threading keeps the node immutable
during execution, which was the actual concern.

Note this also supersedes the *previous* version of this document, whose §2.2
proposed storing a bound slot index **on the node** — either a mutated
`var _slot: Int` or a rebuild pass. That is strictly worse on the same criterion:
it puts execution state into the logical tree, which state-threading does not.

---

## 3. `traverse` — one declared child order (orthogonal, still wanted)

State-threading collapses `prepare` to a one-liner, but `referenced_columns`,
`render`, `prune` and `bound_column` still hand-recurse. A reflection-driven
visitor removes those too:

```mojo
trait Visitor:
    def visit[V: Value](mut self, v: V) raises

trait Value:
    def traverse[Vis: Visitor](self, mut vis: Vis) raises:
        comptime r = reflect[Self]
        comptime for i in range(r.field_count()):
            comptime FieldT = r.field_at[i].T
            comptime if conforms_to(FieldT, Value):
                vis.visit(r.field_ref[i](self))
```

`comptime for` unrolls, so this expands to what the hand-written
`vis.visit(self.l); vis.visit(self.r)` produces. `conforms_to` filters children
from ordinary fields (`_name`, `_unit`, kernel parameters) with no annotation.
`visit` must be **generic** — `Add[L, R]` holds two different types, and a
closure is monomorphic.

`vectorwise` does **not** use `traverse`: it runs per lane and needs values
inline, not through a callback.

---

## 4. What is verified, and what is not

### Verified by spike, 2026-08-03

Both variants were built and run against the pinned toolchain.

```
state-threading                        logical/physical split
Q1 fused binary   : 14 16 18 20        C1 fused binary   : 14 16 18 20
Q2 breaker fused  : 14 16 18 20        C2 breaker fused  : 14 16 18 20
Q3 nested asym    : 21 24 27 30        C4 breaker^2      : 14 16 18 20
Q4 breaker^2      : 14 16 18 20        C5 depth-4 mixed  : 28 32 36 40
Q5 depth-4 mixed  : 28 32 36 40        C6 reuse expr     : 14 20 | 14 18
```

- **`comptime State = Tuple[Self.L.State, Self.R.State]` with
  `def prepare(…) raises -> Self.State` compiles and reduces.** This was the
  headline risk: CLAUDE.md documents an associated-type default referencing
  another associated type as producing `attempt to resolve a recursive reference`.
  That hazard is narrower than it reads — it applies to a reference to a
  **sibling of `Self`**, not to composition through **type parameters**. Recorded
  in CLAUDE.md, "Reflection, packs, and comptime aliases".
- **The state type needs `ImplicitlyDeletable`.** Because `prepare` raises,
  omitting it fails every call site with *"abandoned without being explicitly
  destroyed"* on the throw path.
- **A node whose `State` is an owning container composes under a fused parent**,
  and **nests inside another such node** (`Q4`) — the case that today requires
  the fresh-`Context` convention. Under state-threading it needs no special
  handling, because there is no shared numbering to corrupt.
- **State survives capture by a `@parameter` producer closure.**
- Depth 4 with interleaved shapes works (`Q5`).

### Verified previously, still relevant to §3

- `reflect[Self].field_ref[i](self)` reaches a field's *value*.
- A generic `visit[V: Value]` on a visitor trait handles heterogeneous children.
- `comptime if conforms_to(Self, Trait)` inside a trait default works.

### Not verified — gates, not assumptions

- **Binary size. This is the first checkpoint and it can still kill the design.**
  `State` types are per-instantiation, and the standing invariant is the
  fused-binary DCE property. Convert `NumericBinary` + `NumericColumn` + one
  breaker, build `query_streaming`, compare `__text` against **1,309,032 (live 2026-08-05; the 1,302,900 previously written here predates B12, which added 8,260 to this gate)**. Flat
  → proceed; a regression → stop and reconsider, because the rest is mechanical
  and will only compound it. Measure one binary directly
  (`mojo build -O3 -g0 -I . …`, ~2.5 min), not the full sweep.
- **Raising visitors** (§3 only). CLAUDE.md records that widening `ctx.stripe` to
  a raising worker "needs an implicitly-capturing closure, whose captures are
  silently not made", with an "assignment was never used" warning as the only
  symptom. Spike before building §3.
- **`BoxedValue` at the boundary.** The box erases `Value`; `State` is per-node.
  Most likely the box prepares eagerly and holds a `Datum`, but it is a design
  question to settle during step 2, not a spike question.

---

## 5. What this costs

**CSE moves out of the execution protocol.** The superseded slot-binding design
got common-subexpression elimination nearly free: keyed binding could map a
repeated `sum(a)` to the first occurrence's slot. With typed state trees, two
identical subtrees produce two independent states and both compute.

This is judged an acceptable trade, because CSE belongs in the optimizer
(backlog M1.1) rather than in the evaluator — a plan-level rewrite that hoists a
shared subexpression is more general than a slot-level dedup, and applies to both
lanes rather than only the AOT one. Record it as a dependency: **do not close CSE
against this document.**

**Parallel stage scheduling gets harder.** The old design exposed stages as a
flat, pre-indexed list, which is schedulable. Here `prepare` for a binary is
`(l.prepare(), r.prepare())` — the concurrency is real but the structure is a
tree of typed values, not a work queue. Deferred; not part of this design.

---

## 6. Sequencing

Each step independently revertible and gate-checked.

1. **Binary-size checkpoint.** Convert `NumericBinary`, `NumericColumn` and one
   breaker only. Gate: `query_streaming` `__text` at 1,309,032 (live 2026-08-05; the 1,302,900 previously written here predates B12, which added 8,260 to this gate). **Stop here if it
   regresses.**
2. **Convert the numeric family**, then bool, string, temporal, list. Gate per
   family: `pixi run -e dev pytest marrow/expr/tests` plus the size number.
3. **Fold validity into `State`.** First behaviour change — it fixes the two
   verified null defects (backlog B14, B15). Add tests for both *first*.
4. **Delete `Context`, `Breaker`, `materialize`, the `execute` overloads.**
   Mechanical once 2 and 3 land.
5. **`traverse` + the metadata visitors** (§3), after its own raising spike.
   Removes the remaining ~70 recursion bodies.

Do not collapse 3 into 2, or 5 into 4. This codebase has repeatedly shown that
predictions about the compiler's instantiation and conformance behaviour are
unreliable; measure each step on its own.

---

## 7. Relationship to the two-lane split

Orthogonal, and it should stay that way. This concerns the **AOT lane** only.
The runtime lane (`DynValue`, `dynamic.mojo`) carries a function pointer per node
and has no fused loop — no slots, no breakers, nothing to stage. `BoxedValue`
(`relations.mojo`) wraps either lane and is affected only at the boundary noted
in §4.

Keeping the lanes as separate types is what holds `query_streaming` at parity;
nothing here should reintroduce a shared node type between them.
