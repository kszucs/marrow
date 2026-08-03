# Expression evaluation: traversal, slot binding, and stage scheduling

Design for `marrow/expr/values.mojo`. Supersedes the ad-hoc protocol described
in the "Known follow-ups" comment at the top of that file.

Status: **design only.** Two of its three mechanisms are verified by prototypes
(`~/Workspace/dtype-proto/`); one is not. Nothing here is implemented.

---

## 1. What is wrong today

A fused expression evaluates in two passes. `prepare` walks the tree and each
pipeline breaker appends its whole-column result to a `Context`; then the fused
loop walks the same tree calling `vectorwise[W](batch, ctx, mut slot, idx)`,
incrementing `slot` as it goes and reading `ctx` positionally.

Three problems follow from that.

**The slot correspondence is an unenforceable convention.** No index is ever
stored. `prepare` allocates by arrival order and `vectorwise` resolves by visit
order, so the two passes agree *only* because they happen to walk the tree
identically. A node that recurses `l, r` in one and reads `r, l` in the other
produces wrong numbers with no diagnostic — not a crash, not a type error, just
the wrong column.

**The same recursion is written four times per node.** `prepare`,
`referenced_columns`, `validity` and `render` each hand-recurse into children.
Across ~24 composites that is ~96 method bodies whose only content is "visit my
children", and each is a place to forget one.

**Independent breakers run sequentially.** `sum(a)` and `sum(b)` in one
projection are unrelated stages, and `prepare` runs them one after the other.
Already noted in-file as a follow-up; nothing exposes them for scheduling.

A fourth, related: identical breakers recompute. `sum(a)` used twice produces two
stages, because slots are positional and nothing can dedupe them.

---

## 2. The design

Three mechanisms, in dependency order. Each is useful alone; the third requires
the first two.

### 2.1 `traverse` — one declared child order

A single default on `Value`, driven by reflection. No node implements it.

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

`comptime for` unrolls, so this expands to the same code the hand-written
`vis.visit(self.l); vis.visit(self.r)` produces — it is not runtime iteration.
`conforms_to(FieldT, Value)` filters children from ordinary fields (`_name`,
`_unit`, kernel parameters) with no annotation required.

`visit` must be **generic**: `Add[L, R]` holds `l: Self.L` and `r: Self.R`, two
different types. A closure is monomorphic and cannot visit both — this is why the
visitor is a trait, not a callable.

Child order becomes exactly one fact: field declaration order. `prepare`,
`referenced_columns`, `validity` and `render` become four small visitors instead
of ~96 bodies, and cannot disagree about order because there is only one walk.

`vectorwise` does **not** use `traverse` — it runs per lane and needs values
inline, not through a callback. That is the reason §2.2 exists.

### 2.2 Explicit slot binding

Replace arrival-order allocation with a stored index.

```mojo
struct Bind(Visitor):
    var next: Int
    def visit[V: Value](mut self, v: V) raises:
        v.traverse(self)                       # post-order: children first
        comptime if conforms_to(V, Breaker):
            v.assign(self.next); self.next += 1
```

A breaker records its own index; `vectorwise` drops `mut slot: Int` entirely and
reads `self._slot`. The two passes no longer have to agree about order, because
only one pass assigns and the answer is stored.

This is what makes the positional read *safe* rather than merely conventional,
and it is a prerequisite for §2.3 — parallel stages cannot allocate by arrival
order, since arrival order is exactly what concurrency destroys.

It also enables CSE: binding keys each breaker subtree (its `render()` is already
a canonical structural string) and a repeated `sum(a)` binds to the first
occurrence's index instead of allocating a second stage.

Nodes are immutable during execution, so `assign` needs a mutable pass. Two
options, to be decided when implementing: a rebuild pass returning a new tree
(the shape `resolve_names` already uses), or a `var _slot: Int` field mutated
once at bind time. The rebuild is cleaner; the field is cheaper.

### 2.3 Stage scheduling, driven from outside

With slots pre-assigned, stages become independently executable work items.

```mojo
def run_stages(plan, batch, mut ctx, exec: ExecutionContext) raises
```

- **Post-order gives the dependency structure for free.** Children complete
  before parents, so siblings are the concurrency. `sum(a) + sum(b)` is two
  independent stages; `sum(x + sum(y))` is a chain. It is levels of a DAG, not a
  flat list, and `traverse` produces that shape naturally.
- **Each stage writes to its own pre-assigned index**, so concurrent completion
  cannot perturb the layout.
- **Policy stays external.** `ExecutionContext` is already how kernels take their
  parallelism (`ExecutionContext.serial()`, `ctx.stripe`), so scheduling is a
  parameter, not something baked into `prepare`. Serial remains the default.

### 2.4 Resulting method surface

| member | role |
|---|---|
| `materialize` | the one strategy hook — every node produces its whole-column result |
| `vectorwise` / `elementwise` | the fused lane, per SIMD step |
| `traverse` | the only recursion, defaulted on `Value`, never implemented per node |
| `Breaker` | method-less marker: *when* `materialize` runs, not *what* it does |

`prepare` disappears as a per-node method — it becomes the `Bind` visitor plus
`run_stages`. `IsBreaker` and `stage` are already gone.

---

## 3. What is verified, and what is not

Verified by prototype:

- **`reflect[Self].field_ref[i](self)` reaches a field's value.**
  `gate_traverse.mojo` builds and runs (`visited: 2 / a / b`). Only field *types*
  were previously known to work (`Schema.from_struct`, `schema.mojo:97`).
- **A generic `visit[V: Value]` on a visitor trait handles heterogeneous
  children.** Same prototype.
- **`comptime if conforms_to(Self, Trait)` inside a trait default works** — this
  is how `Breaker` already replaced `IsBreaker` in the working tree.

Not verified — spike before building on either:

- **Raising visitors.** `gate_traverse.mojo`'s `visit` does not raise; `prepare`
  and every stage do. CLAUDE.md records that widening `ctx.stripe` to a raising
  worker "needs an implicitly-capturing closure, whose captures are silently not
  made", with an "assignment was never used" warning as the only symptom. Both
  §2.1 and §2.3 depend on this. **Spike this first.**
- **Binary size.** `traverse[Vis]` instantiates per (node type x visitor), which
  is the shape that cost +115,600 bytes when Q0.4 folded erased arms into a
  generic helper. It *replaces* per-node code so it may be neutral or better, but
  that is a gate measurement, not an assumption.

---

## 4. Sequencing

Each step is independently revertible and gate-checked.

1. **Raising-visitor spike** (prototype). Decides whether §2.1 and §2.3 are
   possible at all.
2. **`traverse` + the four visitors** (serial, behaviour-identical). Gate:
   `query_streaming` `__text` must hold at 1,302,900; `pixi run -e dev pytest
   marrow/expr/tests`.
3. **Explicit slot binding.** Removes `mut slot` from `vectorwise`. Same gates,
   plus `bench_scan` — it changes nothing semantically but touches the hot loop.
4. **CSE during binding.** First behaviour change: fewer stages run.
   `bench_scan` + `BASELINE.md`.
5. **Parallel stage scheduling** as an `ExecutionContext` change. `bench_scan`,
   and interleave repeats across refs — benchmarks here vary 10-18% run to run.

Do not collapse 3 into 2, or 5 into 4. Today's session repeatedly showed that
predictions about this compiler's instantiation and conformance behaviour are
unreliable; each step should be measured on its own.

---

## 5. Relationship to the two-lane split

Orthogonal, and it should stay that way. This design concerns the **AOT lane**
only — `values.mojo`, where every node is strictly typed. The erased lane
(`DynValue` in `dynamic.mojo`) interprets a tag and has no fused loop, so it has
no slots, no breakers and nothing to schedule. `BoxedValue` (`relations.mojo`)
wraps either lane for the relational operators and is unaffected.

Keeping them separate types is what holds `query_streaming` at parity with HEAD;
nothing here should reintroduce a shared node type between the lanes.
