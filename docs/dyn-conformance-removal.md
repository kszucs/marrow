# Plan: remove the `Dyn*` base-trait conformances and their residue

Follows `docs/abstraction-audit.md` §3. Every claim below was measured in the
worktree `.claude/worktrees/dyn-conformances`, branched from `71f069c`, by
patching, running `pixi run -e dev precompile`, and restoring. **No source file
is currently modified.**

> **Revision.** An earlier draft kept `DynType: DataType`, arguing it was
> "earned" because `DynValue.OutType` consumed it. That was wrong: `Value.OutType`
> is read by **no** generic `[V: Value]` code, so the consumer was itself
> vestigial. All four conformances go. See §3.2.

---

## 1. Why these conformances exist, and why the reason is gone

`8334bf0` — *"feat: the erased containers now conform to the traits they erase"*:

> **Step 1 of the lane-unification plan.** `DynScalar: ArrowScalar`,
> `DynArray: Array`, `DynBuilder: Builder` — so the erased container is a *peer*
> of the typed ones, and generic code bound on a trait accepts either. That is
> the property the whole unification rests on: **the static/dynamic choice
> becomes a type argument instead of a second codebase.**

That plan was abandoned. `7d57398` — *"split the expression layer into two
lanes"* — concluded the opposite:

> `DynValue: NumericValue` was unsound. […] **So the lanes separate and share no
> node types.**

The conformances were the foundation of a unification that was then reversed.
Nothing replaced their purpose: the abstraction audit established that **every
`[T: Array]`, `[T: Builder]` and `[T: ArrowScalar]` bound in the tree lives inside
the erasure wrapper's own `_dispatch` closures**. There is no generic-over-trait
code for the boxes to be accepted by.

## 2. What they cost

Recorded in `8334bf0` itself, and still present:

| cost | detail |
|---|---|
| **+13,428 bytes `__text` (+1.1%)** on `query_streaming` | "from the raising trait requirements" — `Array.slice` and `Builder.reset` gained `raises` *solely* because the erased implementations dispatch over a variant and an uncovered member falls through |
| a **compiler-crash workaround** | `python/bindings/scalars.mojo:30-35` — `pymethod[DynScalar.X]()` crashes (`CallParamInf::inferForCall`) for `type`/`is_valid`/`is_null` once they are trait members; three hand-written wrappers |
| **duplicate spellings** | `DynArray.type()` beside `dtype()` ("~200 call sites say `dtype()`"); `DynArray.__init__(ArrayData)` beside `from_data` (abstraction audit §1.2 — a requirement nothing generic invokes) |
| a **weakened contract for 9 typed arrays** | `Array.slice` is `raises` for everyone, though "every typed implementation cannot fail" |
| **dead comptime members** | `DynType.offset` (a placeholder whose docstring cites `native` and two conformances that `7d57398` deleted); `Value.OutType` and its `DynValue.OutType = DynType` satisfier |

## 3. Measured results

All runs: `pixi run -e dev precompile` (compiles every module under `marrow/`,
tests and benches included).

| # | change | errors | warnings |
|---|---|---|---|
| E1 | `DynType.offset` placeholder removed | **0** | **0** |
| E5 | `DynBuilder` drops `Builder` | **0** | **0** |
| E3 | `DynArray` drops `Array` (alone) | 1 | — |
| E4 | `DynScalar` drops `ArrowScalar` (alone) | 1 | — |
| E2 | `DynType` drops `DataType` (alone) | 1 | — |
| C | all three data boxes together | **0** | **0** |
| P | C + `Array.slice`/`Builder.reset` non-raising | **0** | **0** |
| S1 | `Value` drops `OutType`; `DynValue` drops `OutType = DynType` | **0** | **0** |
| S2 | S1 + `DynType` drops `DataType` | **0** | **0** |

### 3.1 The data-box chain

Each single-removal error is *only the next box in the chain*:

```
DynBuilder: Builder      <- held by nothing
   requires ArrayType: Array
DynArray: Array          <- held only by DynBuilder.ArrayType   (builders.mojo:385)
   requires ScalarType: ArrowScalar
DynScalar: ArrowScalar   <- held only by DynArray.ScalarType    (arrays.mojo:2476)
```

### 3.2 The `DynType` chain is the same shape

```
Value.OutType: DataType  <- read by NO generic [V: Value] code   (values.mojo:397)
   satisfied by DynValue.OutType = DynType                        (dynamic.mojo:256)
DynType: DataType        <- held only by DynValue.OutType         (dtypes.mojo:772)
```

`Value.OutType` looked load-bearing and is not. The three fused nodes bound on
plain `Value` — `NullPredicate[K, A: Value]`, `IsIn[A: Value]`,
`WindowFunction[Func, A: Value]` — each declare their **own** output type
(`BoolType`, `BoolType`, `Self.Func.OutType`); none reads `A.OutType`. Only the
*family* traits read `OutType`, and each redeclares it with a tighter bound
(`NumericValue: comptime OutType: NumericType`), so removing it from the base
trait costs them nothing.

**All four conformances are load-bearing only for each other.** Two closed loops,
neither anchored to anything outside itself.

## 4. The plan — six stages, each independently green

Order is forced within each chain: `Builder` requires `ArrayType: Array`, which
requires `ScalarType: ArrowScalar`. Top-down keeps every stage compiling.

### Stage 1 — `DynBuilder` drops `Builder`
- `builders.mojo:161` — remove `Builder` from the conformance list.
- Delete `comptime ArrayType = DynArray` (`builders.mojo:385`) and its docstring.
- Keep `finish`, `reset`, `dtype`, `length`, `null_count`, `extend`,
  `append_null`, `reserve` — all are `DynBuilder`'s own API with real callers.
- **Verify:** `precompile` (measured clean).

### Stage 2 — `DynArray` drops `Array`
- `arrays.mojo:2345` — remove `Array` from the conformance list.
- Delete `comptime ScalarType = DynScalar` (`arrays.mojo:2476`).
- Delete `def __init__(out self, data: ArrayData) raises` (`arrays.mojo:2449`) —
  a pure delegate to `from_data`, which is "the implementation and the primary
  spelling".
- Delete `def type(self)` (`arrays.mojo:2502`) — `dtype()` is what ~200 call sites
  use. Checked: every `.type()` call site in the tree resolves to a *typed* array
  or to `DynScalar` (`filter.mojo:524,1023` are `DictionaryArray.type()`), so none
  should break — `precompile` is the check.
- Keep `to_dyn()` returning `self^` only if it has non-trait callers.
- **Verify:** `precompile`.

### Stage 3 — `DynScalar` drops `ArrowScalar`
- `scalars.mojo:563` — remove `ArrowScalar` from the conformance list.
- Re-check `to_dyn()` (`scalars.mojo:711`) as in Stage 2.
- **Do not rename `DynScalar.type()`.** `ArrowScalar.type()` stays the spelling for
  all 9 typed scalars; renaming only the box would create the divergence this work
  removes.
- **Verify:** `precompile` **and** build the bindings (§6).

### Stage 4 — reclaim the `raises` (the measurable payoff)
- `arrays.mojo:170` — `Array.slice` back to non-raising; delete the paragraph
  explaining the erased-dispatch exception.
- `builders.mojo:143` — `Builder.reset` back to non-raising, same.
- Generic `[T: Array]` callers that added `try`/`raises` for this can drop it.
  `DynArray.slice` / `DynBuilder.reset` stay `raises` — they are ordinary methods
  now, not trait implementations.
- **Verify:** `precompile` (measured clean), then the gate. **Outcome: 0 bytes
  recovered** — see §8. The gate command is
  `pixi run -e dev python3 benchmarks/binary_size/check_gate.py`;
  `pixi run binary_size` is `compare.py`, which prints per-module symbol counts
  and no `__text` totals.

### Stage 5 — `Value` drops `OutType`
- `values.mojo:397` — delete `comptime OutType: DataType`.
- `dynamic.mojo:256` — delete `comptime OutType = DynType`.
- Leave every family trait's `OutType` alone; they redeclare it and read it
  constantly.
- Check whether `BoxedValue` declares an `OutType` that also becomes dead.
- **Verify:** `precompile` (measured clean).

### Stage 6 — `DynType` drops `DataType`, and the residue goes with it
- `dtypes.mojo:772` — remove `DataType` from the conformance list.
- `dtypes.mojo:777` — delete `comptime offset = DType.int32` and its docstring.
  Measured dead independently (E1); its docstring already cites `native` and two
  conformances `7d57398` deleted.
- `dtypes.mojo:1018-1034` — rewrite `DynType.byte_width`'s docstring. It claims to
  be a load-bearing override of `PrimitiveType.byte_width` and calls
  `DynType.native` "the `bool` placeholder"; `DynType` conforms to neither and has
  no `native`. After this stage it overrides nothing at all. Keep the "returns 0
  for non-fixed-width" contract, re-derive the reason.
- Check `DynType.to_dyn()` (`dtypes.mojo`) — it exists to override a `DataType`
  default that no longer applies. Keep only if it has direct callers.
- **Verify:** `precompile`, then the full suite.

### Finally
- `python/bindings/scalars.mojo:30-35` — with `DynScalar` no longer conforming, try
  restoring `pymethod[DynScalar.type]()` etc. and delete the workaround comment.
  If the crash does not recur, that confirms the conformance caused it; if it does
  recur under Mojo 1.1, keep the wrappers and correct the comment.
- `CHANGELOG.md` — one `### Refactors` entry.
- `CLAUDE.md` — the "Type erasure" section describes erased values converting
  "transparently" and `DynType` as a *peer* of the concrete dtypes. Both need
  rewording: after this work the boxes are not peers, they are separate types that
  happen to expose a similar surface.

## 5. What stays: `DynValue: Value`

Removing it produces **179 errors**, against 0 for the four above. Three kinds:

- `relations.mojo:418` — `cannot be converted from 'DynValue' to 'BoxedValue'`.
  `BoxedValue.__init__` is bound on `Value`; without the conformance the runtime
  lane cannot reach the relational engine.
- `'DynValue' value has no attribute 'count_distinct'` — `Value`'s defaults are
  DynValue's real API. It inherits `validity`, `count_distinct`,
  `approx_count_distinct`, `isnull`, `notnull`.
- every `plan.aggregate(...)` / `.filter(...)` / `.join(...)` call site.

Strictly, each is replaceable — a non-generic `BoxedValue.__init__(DynValue)`
overload, three duplicate node bounds, five method definitions. But the
conformance buys something the boilerplate would not: it is the **one sound bridge
between the lanes**. `NullPredicate`, `IsIn` and `WindowFunction` are bound on
plain `Value` precisely because they need no typed lane from their operand, so a
runtime leaf can feed a *fused* tree:

```mojo
col("x").isnull() & col("y").notnull()   # untyped col() -> DynValue leaves
# => And[NullPredicate[IsNullKernel, DynValue],
#        NullPredicate[NotNullKernel, DynValue]]   -- fused
```

(The *typed* `col("x", int64)` factory is the AOT one and involves no `DynValue`
at all.) The bridge is opt-in and DCE-friendly: `DynValue` links only into a
binary that actually writes one of those three nodes over a runtime leaf.

After Stage 5, `Value` has exactly one comptime member left — `OutShape` — and
`DynValue.OutShape = 1` is true, since `execute` returns a `DynArray`
unconditionally. So the rule from `7d57398`, *"erase into a trait whose members
are all runtime methods"*, is satisfied as nearly as the design allows.

### Open thread

`OutShape` is now the last comptime member on `Value`, and its only generic reader
is `NullPredicate`'s `comptime OutShape = Self.A.OutShape`. If that propagation
can be replaced (or `NullPredicate` can hardcode `1`), `Value` becomes a pure
method trait and the erasure rule holds without qualification. Not in scope here;
worth its own probe.

## 6. Risks

1. **`precompile` does not compile `python/bindings/`.** `8334bf0` says so
   explicitly, and that is where this change's known hazard lives. Every clean
   result in §3 is necessary but **not sufficient**. Stages 3–6 must be validated
   with `pixi run -e dev test`, which rebuilds `python/marrow/libmarrow.so`.
2. **Implicit-conversion overload resolution.** `DynArray` has
   `@implicit __init__[T: Array]`. Once `DynArray` is not an `Array`, expressions
   like `var a: DynArray = some_dyn_array` resolve through `__init__(*, copy:)`
   instead.
3. **Losing a drift check.** The conformance is currently the only mechanical
   guarantee that the erased surface tracks the typed one. Small in practice — the
   surfaces have already diverged by consent (`dtype()` vs `type()`, `from_data`
   vs `__init__`), and the audit found the compiler was not catching the
   divergence that mattered (`write_repr_to`, §1.1). Mitigation: keep abstraction
   audit §3.3's table as the written contract.
4. **Reversing a deliberate decision.** `8334bf0` was considered, not accidental.
   The justification is that its stated goal was itself reversed by `7d57398`;
   that reasoning belongs in the commit message.

## 7. Would a shared `DynBox` base trait help? — No

The boxes plainly share a shape: a `Variant`, an `@implicit __init__[T: Trait]`,
`_dispatch` / `_dispatch_raises`, `as_type[T]`, per-type accessors, `write_to`.

**Every one of those members is generic over a *different* trait bound** — `Array`
vs `ArrowScalar` vs `Builder` vs `DataType`. Writing them once requires a trait
parameterised by a trait; Mojo has no higher-kinded traits.

Not speculation — **the shared implementation already exists and already
surrendered the bound**:

```mojo
def variant_dispatch[
    R: Movable, //, *Ts: Movable, Func: def[T: Movable](T) -> R
](ref v: Variant[*Ts], func: Func) -> R:
```

`utils.mojo:277`, bound on `Movable`, for the reason `CLAUDE.md` records: *"a
closure type cannot be generic over its own trait bound."* A `DynBox` could bind
no tighter.

What would be left to require is `Copyable`, `Movable`, `Writable` and a `dtype()`
accessor — the first three already stated individually, the last a naming
convention the boxes do not share (`DynScalar.type()` vs `DynArray.dtype()`), and
§4 Stage 3 explains why converging that would make things worse.

The one place two boxes are handled alike is already solved without a trait:

```mojo
comptime Datum = Variant[DynScalar, DynArray]        # values.mojo:217
def into_array(d: Datum, n: Int) raises -> DynArray  # values.mojo:220
```

**Verdict: no gain.** Recorded so it is not re-proposed. The genuine shared-contract
opportunities are elsewhere and unaffected: a `CastKernel` family trait
(abstraction audit §1.4), splitting `Named` out of `Kernel` (§1.6), and a `Column`
trait over `Array ∪ ChunkedArray` (§1.12).

## 8. Net effect

**Deleted:** 4 conformance entries, 3 companion `comptime` members
(`ArrayType`, `ScalarType`, `OutType`), 1 trait member (`Value.OutType`),
2 duplicate `DynArray` members, 1 dead `DynType` placeholder, 2 `raises` and their
explanatory docstrings, 1 stale docstring, 1 compiler workaround (confirmed
retired — `pymethod[DynScalar.is_valid]()` compiles again).

**Recovered: 0 bytes — the predicted ~13 KB did not materialise.**

Measured after Stage 6 by running `check_gate.py` on this branch and on the
unmodified branch point `71f069c` in a separate worktree:

| gate | `71f069c` | this branch | delta |
|---|---|---|---|
| `query_streaming` | 2,607,180 | 2,607,180 | **0** |
| `query_join` | 2,341,108 | 2,341,108 | **0** |
| `query_streaming_agg_fused` | 2,089,084 | 2,089,084 | **0** |
| `query_streaming_agg` | 2,637,536 | 2,637,536 | **0** |

Byte-identical. §2 quoted `8334bf0`'s recorded **+13,428 bytes** for *adding* the
raising trait requirements and this plan banked on reclaiming it; that was a
mistake in reasoning, not in execution. That figure was measured under
`mojo 1.0.0b3`, and a `raises` on a trait requirement only costs anything at
generic call sites — of which there are none, since nothing outside the boxes is
generic over `Array` or `Builder`. The conformances were free to hold and are
free to drop.

**Separately: the size gate is already red on `main`, and not because of this
work.** `71f069c` measures +95.7% / +65.9% / +56.1% / +41.9% against
`baseline.json`, which was last recorded at `8504152` — before `04d01e4`
(Mojo 1.0 / MAX 26.5) and `4a2c0a9` (Mojo 1.1). The baseline is two toolchain
upgrades stale and needs re-recording by someone who can attribute the jump. Not
in scope here; flagged because a red gate hides real regressions.

**Lost:** a compile-time drift check between the erased and typed surfaces that
was already only partially effective.

**Unchanged:** all runtime behaviour. Every method involved keeps its body and its
callers; only trait checking, two `raises`, and one unread comptime member change.

**Left standing:** `DynValue: Value` — the one conformance with a consumer outside
its own loop, and the seam the two lanes compose across.
