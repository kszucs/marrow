# A1 — typed per-node `State` for the fused lane

**Status:** EXECUTED 2026-08-06. `a + 1` over 1M rows 2.04 ms → 70.9 µs (28.8x,
within 2.4% of the 69.2 µs floor); `a + a` now costs the same as `a + 1`. Size
gate +0.056%; core+parquet+python 1145 passed; expr+kernels 845 of 846. The one
failure is **B29** in `docs/backlog.md` — a miscompile in `_drive_bool` that ten
source-level formulations did not move. `docs/backlog.md` is the live status;
this file is kept as the record of what was planned.
**Size:** one commit, ~27 structs. Not incremental — see "Why one commit".

## Why

The fused expression lane re-resolves everything it needs on **every SIMD chunk**.
`NumericColumn.vectorwise` runs a schema lookup by name (a string comparison over
every field), a `Variant` unwrap, and a `BufferView` reconstruction — 250,000
times over a million rows, to read a column that never moves.

Measured (`marrow/exprold/tests/bench_fusion_gaps.mojo`, 1M rows, `a + 1`):

| | |
|---|---|
| fused lane today | **2.04 ms** |
| hand-written, view hoisted | **66.8 µs** |
| A1 protocol, spiked | **67.4 µs** |

**30×**, within 1% of the floor. Cost scales with column-leaf count and nothing
else: `a + a` is exactly 2× `a + 1`, and `a + (1 + 2)` is free.

Secondary benefits, all previously the *stated* reason for A1 and now the side
effects: the `Context` positional-slot hazard disappears, six methods collapse to
two, ~84 hand-written recursion bodies go, and B14/B15 go with them.

## The protocol

```mojo
comptime State: Copyable & ImplicitlyDeletable      # per-node, comptime
def state(self, batch: RecordBatch) raises -> Self.State
@always_inline
def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]
```

`state()` runs once per pass, before the loop. `lane()` reads only `state` — no
`batch`, no `Context`, no `mut slot`. That removal *is* the optimisation.

**`State` holds the owned typed array, not a `BufferView`.** Confirmed by spike:
a view cannot be named, because `prepare` yields
`BufferView[..., origin_of(o.columns[...].buffer)]` — a nested projection of the
batch origin that will not unify with a plain `o` parameter. It does not matter:
`.values()` is pointer arithmetic on a `TrivialRegisterPassable` view and costs
nothing per chunk. The expensive parts were the lookup and the unwrap, and both
hoist into `state()`.

So: **no origin gymnastics, no `unsafe_origin_cast`, no parameterised associated
types.** All three were feared; none is needed.

## Why one commit

A defaulted `state()` does not compile:

```
cannot implicitly convert 'DynArray' value to '_Self.State'
value of type '_Self.State' cannot be implicitly copied      ← rebind cannot bridge
```

A trait default returning `Self.AssocType` requires that type's bound to be
`ImplicitlyCopyable`. The natural default is `DynArray`, which this codebase
deliberately keeps non-implicitly-copyable, and `rebind` needs the same
conformance. Recorded in CLAUDE.md's gotchas.

Consequence: there is no "convert three nodes and measure" path. Every conformer
lands together.

## Surface

27 structs implement `vectorwise`; 18 sites call a child's. The three lanes are
entangled — `NumericCompare` is a `BoolValue` consuming `NumericValue`s,
`BoolToNum` is a `NumericValue` consuming a `BoolValue` — so the split is by
*node shape*, not by lane.

**Leaves** — `State` is the resolved thing; `state()` does the lookup once.

| node | `State` |
|---|---|
| `NumericColumn[T]` | `PrimitiveArray[T]` |
| `NumericLiteral[T]` | `NoneType` (the value is in the node) |
| `StringColumn[T]`, `StringLiteral[T]` | as above, string-typed |

**Unary** — `State = Self.A.State`; `state()` delegates, `lane()` applies the op.
`NumericUnary`, `FloatUnary`, `BoolUnary`, `NumericCast`, `BoolToNum`,
`NumToBool`, `TemporalExtract`, `ListLength`.

**Binary** — `State = Tuple[Self.L.State, Self.R.State]`. The spike notes in
CLAUDE.md confirm this composes to at least depth 4 with mixed shapes, and that
`prepare` raising means `State` needs `ImplicitlyDeletable` or every call site
fails with *"abandoned without being explicitly destroyed"* on the throw path.
`NumericBinary`, `FloatBinary`, `BoolBinary`, `NumericCompare`,
`NumericPredicate`, `StringPredicate`, `ConditionalBinary`, `IsIn`,
`ListContains`.

**Breakers** — `State` is the materialised result, and it **replaces** that
node's `Context` slot rather than supplementing it. This is where the slot
protocol dies. `BoolReduce`, `NullPredicate`, `StringPredicate`, `StringLength`,
`StringToNum`, `StringToBool`, `Reduction`, `WindowFunction`.

**Variadic** — `CaseWhen` holds a collection of branch states.

## Order

1. Declare `State`/`state`/`lane` on `Value`, `NumericValue`, `BoolValue`,
   `StringValue`. Abstract — no defaults, they do not compile.
2. Leaves, then unary, then binary, then breakers, then `CaseWhen`. The tree will
   not compile until all are done; that is expected and is why this is one commit.
3. Convert the drivers (`NumericValue.materialize` and the bool/string
   equivalents) to `state()` once + `lane()` per chunk.
4. Delete `Context`, the `mut slot` threading, and `prepare`. Confirm nothing
   references `ctx.get`/`ctx.get_ref`/`ctx.append`.

## Gates

- `pixi run -e dev precompile` — 0 errors, **0 warnings**. An `assignment was
  never used` warning on a captured value means a capture was not made and the
  code reads garbage; treat it as an error.
- `bench_fusion_gaps.mojo` — `bench_b27_probe_plain_fused_add_1m` must approach
  **67 µs**, from 2.04 ms. If it does not, the conversion is not done.
  **Read the unit on each row** (ns/µs/ms vary per benchmark; strip ANSI first) —
  misreading them once already produced a conclusion that was exactly backwards.
- `pixi run -e dev python3 benchmarks/binary_size/check_gate.py` — hold
  `query_streaming` `__text` at **1,332,456**. A typed `State` per node could
  plausibly grow the fused binary; if it does, that is the trade to weigh and
  report, not to hide.
- Full suite: 844 kernels+expr, 1145 core+parquet+python.

## Watch for

- **The `Context` slot invariant is what this deletes**, so during conversion the
  tree has both mechanisms live. Do not try to keep them consistent — convert a
  node fully or not at all.
- `StringPredicate` and `StringLength` are `Breaker` *and* a value trait; their
  `State` is the materialised array and their `lane` is a load. That is the whole
  of Q7.1, obtained for free.
- `DynValue` (runtime lane) hardcodes `OutShape = 1` and must keep working; its
  `State` is its materialised result, like a breaker's.
