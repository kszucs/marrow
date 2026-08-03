# Step 3 — replace `TagValue`'s tag interpreter with node structs

> ## SUPERSEDED BY `7d57398` (2026-08-03) — historical record only
>
> **Phases 0–4 genuinely landed** — the 41-tag interpreter was deleted and
> `dynamic.mojo` went 1,087 → 113 lines, exactly as planned. But the **design
> rationale below was reversed four days later** by the two-lane split. Do not
> build on it. The current model is `docs/lane-shape-window-design.md`; status
> lives in `docs/backlog.md`.
>
> What this document asserts, and what is true at `b2e7dae`:
>
> 1. **"The two lanes *do* share one node set."** They do **not**. The lanes now
>    share **no node types**: `marrow/expr/values.mojo` is the AOT lane, every
>    operand bound on its family trait; `marrow/expr/dynamic.mojo` is the runtime
>    lane, where `DynValue` is its children, an optional payload and a pointer to
>    its evaluator. `a.add(b)` on two runtime operands does *not* build the fused
>    `Add`.
> 2. **"`TagValue` implements every family trait … that is what lets the node
>    bounds stay exactly as they are."** That conformance was **unsound and is
>    gone**. `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue` promise a
>    comptime `OutType: NumericType` and a `vectorwise` lane; the box supplied a
>    placeholder `native = DType.bool` and a stub returning zero. The compiler
>    reported it as `attempt to resolve a recursive reference to declaration
>    'DynValue.__gt__'` — which is what forced the fluent surface into a
>    `NumericOps` sub-trait (surprise 3 in the log below). §"The one thing that
>    has to be made to work" is the part that did not survive.
> 3. **"The bet is the erased instantiation never *reaches* `vectorwise`."** The
>    bet lost. The 14 `_erased` method bodies and the hand-propagated
>    `comptime IsErased` that selected them are **deleted**, along with
>    `NumericOps`. `DynType` **dropped all 8 family-trait conformances** (Phase 1
>    of this plan) — they told the same lie about `native`.
> 4. **Naming has inverted.** `TagValue` no longer exists. Today `DynValue`
>    (`dynamic.mojo:236`) names the runtime **node**, and the box both lanes
>    erase into is **`BoxedValue`** (`relations.mojo:155`) — which is what keeps
>    each relational operator compiling exactly once. Read every "`TagValue`" and
>    every "the box" below with that substitution.
> 5. **The final numbers are superseded.** `query_dynvalue` 3,956,596 →
>    **3,984,756** (+0.71%) and `query_streaming` 1,303,028 → **1,302,900** at
>    HEAD. The −24.4% this step delivered was kept; the two-lane split cost
>    +0.71% on top of it to buy back soundness.
>
> What is still worth reading: the isolated-gate measurement (§"What changed
> since the original sketch" item 3), the 41-tag → existing-node mapping table,
> and the five surprises in the log — three of those (the `materialize`
> catch-22, the triplicated aggregate sugar, "the box is the erasure boundary")
> are still live constraints and are restated in `CLAUDE.md`.

Status: **complete**, 2026-07-30. Supersedes the Step 3 sketch in
`~/Workspace/dtype-proto/PLAN.md`, which three findings have overtaken.

## What changed since the original sketch

**1. The two lanes *do* share one node set — the sketch was right.** It looked
otherwise at first: `NumericValue.vectorwise[W](…) -> SIMD[Self.OutType.native,
W]` (`values.mojo:379`) is the fused lane's reason for existing, and an erased
`DynType` has no `native`, which reads as the same structural wall the prototype
hit in the kernels (`PrimitiveArray[T]` stores `comptime scalar =
Scalar[Self.T.native]`).

It is not the same wall. A kernel *must* instantiate `PrimitiveArray[T]`; a node
need only satisfy the trait. `DynType` can carry a placeholder `native` it never
uses, because the erased instantiation never reaches `vectorwise` — the
`comptime if` takes the dispatch arm. Phase 0 confirmed this end to end.

**2. That mechanism already exists.** `DynValue` (`values.mojo:2270+`) already
boxes via an opaque `ArcPointer[NoneType]` plus `thin` trampolines — exactly the
"open family" canon. Step 3 is therefore not "build erasure". It is "stop
`TagValue` being a fat tagged union, and let it box node structs the way fused
nodes already box".

**3. The size objection is answered, and it was the real risk.** The codebase
records **+115,600 bytes** for folding `eval`'s twelve binary arms into one
`_arith[K]` helper (`dynamic.mojo:164`) — strong evidence that per-op
monomorphization explodes. Two isolated gates
(`~/Workspace/dtype-proto/gate_tag_interp.mojo`, `gate_node_interp.mojo`, neither
importing marrow) computing the same twelve ops:

| | `__text` | symbols |
|---|---:|---:|
| tag interpreter (fat node + 12-arm switch) | 41,576 | 79 |
| node structs + trampolines | **35,048** | 98 |
| | **−6,528 (−15.7%)** | +19 |

The node design is *smaller*. This does not contradict the `_arith[K]` number, it
explains it: `_arith[K]` added a monomorphized layer while **keeping** the switch,
so it paid for both. Removing the dispatcher is not the same as wrapping it.

Verified both gates compute the same result (outputs 443 vs 456; the 13-unit gap
is exactly the display-name length difference, 71 vs 84 chars — array checksums
identical).

**Caveat carried forward:** the gates model only the *uniform binary* case, the
shape most favourable to the design. `CASE_WHEN`, `COALESCE`, `CAST` and `LIKE`
will not collapse as cleanly. Measure at each phase; do not extrapolate.

## Design

**One node set. The erased lane is those same nodes, instantiated with `TagValue`
operands.** There are no `Dyn*` node types — a `DynUnary` would defeat the point.
`col("a").abs()` builds `Unary[AbsKernel, TagValue]`, and `a.add(b)` builds
`Add[TagValue, TagValue]`, which is the same `Add` the fused lane uses.

`TagValue` **is** a `Value`: it erases the plan *shape* (which node types the
tree is built from) while `DynArray` erases the data *lane*. It replaces
`DynValue` outright — that box exists only because the name `TagValue` was taken
by the tag interpreter, and it has to carry the union of two representations (a
fused node *or* an interpreter). Once the interpreter is gone there is one
representation, so the box erases a single trait and `TagValue` can hold a
`TagValue` like any other `Value`.

This is the shape `~/Workspace/dtype-proto/unified/expr.mojo` already
demonstrates: `TagValue[In, Out](Value)` over an opaque `ArcPointer[NoneType]`
plus `thin` trampolines, alongside `Add[L: Value, R: Value]`.

### The one thing that has to be made to work

`TagValue` implements **every family trait** — `NumericValue`, `BoolValue`,
`StringValue` — not just `Value`. That is what lets the existing node bounds stay
exactly as they are: `NumericBinary[K, L: NumericValue, R: NumericValue]` accepts
`TagValue` for `L` and `R` with no relaxation, so `a.add(b)` on two erased
operands builds the *same* `Add` the fused lane builds.

The node then picks its strategy at elaboration:

```mojo
comptime IsErased = Self.L.IsErased or Self.R.IsErased

def materialize(self, batch, mut ctx) raises -> Datum:
    comptime if Self.IsErased:
        ...                                 # K.dispatch over DynArray
    else:
        ...                                 # fused SIMD, as today
```

That is the "when the incoming type parameter is erased, fall back to a
dynamically dispatched comptime arm" idea, and it is what makes one node type
serve both lanes.

**The cost of admission is that `DynType` must satisfy the family traits too**,
because `NumericValue` requires `comptime OutType: NumericType` and
`NumericType` requires `comptime native: DType`. An erased dtype has no honest
`native`. The bet is that the erased instantiation never *reaches* `vectorwise`
— the `comptime if` takes the other arm — so a placeholder `native` is never
elaborated into a real `SIMD[…]`.

**This is the load-bearing assumption of the whole step, and it is not free.**
`CLAUDE.md` records two adjacent behaviours that bite exactly here: a chained
associated-type projection (`Self.OutType.ArrayType`) does not reduce, and
re-defaulting a trait method that returns `Self.ArrayType` triggers `attempt to
resolve a recursive reference to declaration`. `Self.OutType.native` inside the
fused arm is that shape.

Phase 0 probed it in isolation before a line of marrow moved, and it holds — see
the result below. (Had it not, the fallback was relaxing the node bounds to
`Value`; that is no longer needed, and every fused node keeps its signature.)

### What the box carries

`TagValue` keeps the public surface the tag interpreter has today — factories
(`col`, `lit`, `if_else`), operator overloads, `.cast()`, `.isin()`, `.year()`,
the aggregate sugar — so callers and tests are unaffected. What changes is that
each operator constructs a shared node rather than a tagged record: from

```mojo
var _tag: UInt8
var _args: List[TagValue]
var _kind_data: UInt8
var _value: Optional[DynScalar]
var _name: String
var _cast_to: Optional[DynType]
var _value_set: Optional[DynArray]
```

(seven fields, every node paying for all seven, and a 41-arm switch per method)
to an opaque pointer plus trampolines, over node structs that each own their
`eval`, `prune`, `op_name` and `referenced_columns`.

**No new node types are introduced.** All 41 tags map onto nodes that already
exist in `values.mojo`, instantiated with `TagValue` operands:

| tags | existing node |
|---|---|
| ADD SUB MUL MOD FLOORDIV | `NumericBinary[K, TagValue, TagValue]` |
| DIV | `FloatBinary[DivKernel, TagValue, TagValue]` |
| EQ NE LT LE GT GE | `NumericCompare[K, TagValue, TagValue]` |
| AND OR XOR | `BoolBinary[K, TagValue, TagValue]` |
| NEG ABS | `NumericUnary[K, TagValue]` |
| NOT | `BoolUnary[NotKernel, TagValue]` |
| IS_NULL NOT_NULL | `NullPredicate[K, TagValue]` |
| CAST | `NumericCast[To, TagValue]` and family |
| YEAR … DAY_OF_YEAR, DATE_TRUNC | the temporal nodes |
| LOAD LITERAL | the column and literal leaves |

**Every one of the 41 tags already has a fused counterpart** — checked
mechanically, not by eye. An earlier draft of this plan claimed eight did not
(LENGTH, LIKE, ILIKE, IS_IN, COALESCE, NULLIF, CASE_WHEN, IF_ELSE) and that was
simply wrong: `StringLength`, `Like`, `ILike`, `IsIn`, `Coalesce`, `Nullif` and
`CaseWhen` all exist in `values.mojo`, and `IF_ELSE` *is* `CaseWhen` — it is the
single-branch form. (`test_parity.mojo`'s docstring still says "there is no fused
`if_else` node"; that is stale and should go when Phase 3 touches it.)

So Phase 3 adds no fused nodes. It rewires the operators onto the ones already
there, which makes it substantially smaller than budgeted.

`_op_name` does **not** disappear into `K.name`: the vocabularies deliberately
differ (`sub` vs the kernel's `subtract`, `and` vs `and_`). It becomes a
`comptime display: String` per node — still one string per op, but attached to
the node that uses it rather than to a switch that can fall through.

### Constraints

- **No backward compatibility.** Tag constants (`LOAD`, `ADD`, …) are deleted
  from `dynamic.mojo` and from `expr/__init__.mojo`.
- **`relations.mojo:346` is the only external tag consumer** — it asks
  `k.kind() == LOAD` then `k.kind_data()` to name a sort/group key. It gets a
  proper accessor instead; no caller learns about node types.
- **Gate every phase** on `pixi run binary_size`, measured as `__text` (never
  stripped size — page-quantized to 16 KB), plus `marrow/expr/tests`.
- **The parity suite is the safety net.** `test_parity.mojo` asserts the fused
  and interpreted lanes agree element-for-element; it must stay green at every
  phase, and it is what proves a converted node still means what its tag meant.

## Phases

Each is independently revertible and independently valuable.

### Phase 0 — probe the conformance (blocking) — **DONE, passed**
Smallest standalone program that answers: can an erased value type conform to a
family trait whose associated type demands a `comptime native: DType`, and will
the existing generic node accept it, with the fused arm never elaborated?
Lives in `~/Workspace/dtype-proto/`, imports no marrow. **Nothing moves in
marrow until this builds.** A negative result selects the bound-relaxation
fallback instead, which is a different and larger diff.

### Phase 1 — `DynType` conforms to the family traits
`native` placeholder plus whatever else `NumericType`/`StringLikeType` require.
**Gate:** full suite plus every binary-size gate unchanged — this phase adds
conformances and must not change a single byte of generated code.

### Phase 2 — `TagValue` replaces `DynValue`
The box implements `Value` and the family traits, and takes over `DynValue`'s
role in `relations.mojo`, `execution.mojo` and the binary-size gate programs.
The tag interpreter still exists and still works; it is simply one more thing the
box can hold. **Gate:** suite green, gates within noise.

### Phase 3 — operators build shared nodes
`TagValue.__add__` returns `Add[TagValue, TagValue]` re-boxed, and so on through
the operator surface, each backed by the `comptime if` erased arm in the existing
node. Tags fall out of use group by group as their operator is converted:
binaries and compares first (the shape the gates measured), then unary and
temporal, then the payload-carrying ones (`cast`, `like`, `isin`, `coalesce`,
`nullif`, `case_when`, `date_trunc`). **Gate after each group:**
`query_dynvalue` / `query_runtime` `__text`.

### Phase 4 — delete the interpreter
Remove `_tag`, the tag constants, the three switches, and `DynValue`. Give
`relations.mojo` its accessor in place of `kind() == LOAD`. **Done when**
`dynamic.mojo` contains no `_tag` and no switch, and the full suite plus the
binary-size gate are green.

### Phase 5 — `DynRelation` (was Step 5)
Only reachable once plan *shape* is erased independently of data *lane*. Out of
scope here; tracked separately.

## Log

Measurements land here as each phase completes, so the next decision has numbers
rather than impressions.

| phase | `query_dynvalue` `__text` | `query_runtime` `__text` | tests |
|---|---:|---:|---|
| baseline (`a7524a7`) | 5,237,876 | 5,238,004 | 1000 passed |
| Phase 1 (`DynType` conformances) | 5,236,148 | 5,236,276 | 1288 passed |
| Phase 2a (rename + `Value`) | not measured | not measured | 315 passed |
| Phase 2b (family traits) | 5,236,148 | 5,236,276 | 315 passed |
| Phase 3 (erased arms, 13 nodes) | 5,236,148 | 5,236,276 | 806 passed |
| Phase 4 (interpreter deleted) | **3,956,596** | **3,957,620** | 1285 passed |

### Final result

    query_dynvalue   5,236,148 -> 3,956,596   -1,279,552  -24.4%
    query_runtime    5,236,276 -> 3,957,620   -1,278,656  -24.4%
    query_streaming  1,266,040 -> 1,303,028      +36,988   +2.9%
    query_arith      1,274,676 -> 1,310,732      +36,056   +2.8%
    query_join       3,762,420 -> 3,762,420           +0    0.0%

`dynamic.mojo`: **1,087 -> 113 lines**. 41 tag constants and 99 switch arms
gone. The erased lane loses **1.28 MB** — past the -15.7% the isolated gates
predicted.

The fused gates carry ~+37 KB, and ~30 KB of that is `Value.prune`: statistics
pruning the fused lane **did not have before**, since the box answered
`unknown()` for every fused node. That is part of Q4.5 delivered as a side
effect, so it is capability rather than pure cost; ~7 KB remains unaccounted.

Perf vs `BASELINE.md`: 57/57 rows, nothing attributable to this work.

### What the phases actually cost, in surprises

Worth recording because each cost real time and none was in the plan:

1. **The `materialize` catch-22** — every family defaults it, so conforming to
   two is a compile error, and implementing it manually recurses. Fixed by the
   `CLAUDE.md` workaround: family-specific helper names plus 32 one-line
   overrides.
2. **Triplicated aggregate sugar** — `count_distinct` was defined three times,
   `min`/`max`/`count` three times with *different* bodies. Duplication across
   family traits is not inert: it is invisible until one struct conforms to two.
3. **`__gt__` recursion, only from outside the package.** The box inheriting a
   fluent surface returning `Gt[Self, Rhs]` made an external program fail to
   compile while the test suite stayed green. Caught only when a binary-size
   gate was rebuilt. Fixed by splitting `NumericOps` off `NumericValue`.
4. **Three erased node types added and removed.** `DynColumn`, `DynLiteral`,
   `DynCast` were all unnecessary — the box is the erasure boundary. Only
   `RuntimeCast` survived, because `NumericCast[To]` binds `To` on
   `NumericType` and cannot express `cast(timestamp(second))`.
5. **A widened dispatch cost 34 KB.** Fixing `DynScalar.repeat`'s missing
   string arm by moving to `dispatch_primitive` added ~13 types that build an
   array each. Narrowed back to numeric + bool + string-like.

### Phase 2 blocker — **RESOLVED** in `b56b886`

`DynValue` conforms to `Value`. It does **not** yet conform to `NumericValue` /
`BoolValue` / `StringValue`, and the compiler gives both halves of the reason:

```
without an override:  trait method requirement 'materialize' has conflicting
                      default implementations in 'BoolValue' and 'StringValue';
                      you must implement it manually
with an override:     attempt to resolve a recursive reference to declaration
                      'DynValue.materialize'
```

Each family *defaults* `materialize` to drive its own fused loop, so one struct
conforming to all three inherits three conflicting defaults — and implementing it
manually hits the re-defaulted-method recursion `CLAUDE.md` documents.

The prescribed fix from that same note: keep the base method abstract, give each
family a differently-named helper (`_numeric_fused` / `_bool_fused` /
`_string_fused`), and have each node override `materialize` with a one-liner
delegating to it. **Cost: 37 family-conforming structs, 4 already define
`materialize`, so 33 need a new one-line override.**

The alternative is to leave `DynValue` on `Value` alone and relax the fused
nodes' bounds from their family trait to `Value`, dispatching on
`conforms_to(Self.L, NumericValue)`. That avoids 33 boilerplate overrides but
changes the signature of every fused node, and weakens the bound that currently
makes a mis-typed operand a compile error. **Took the overrides** — 32 in the end, not 33; the earlier count used a cruder
body-slice. Binary size moved **+0 on all eleven gates**, confirming it is a
compile-time reorganization only.

Two further conflicts surfaced while doing it, both duplication rather than
design: `count_distinct`/`approx_count_distinct` were defined **three** times
(`NumericValue`, `StringValue`, and `TemporalValue` with different formatting,
which is why a literal-block search found only two). `AggExpr.of` is bound on
`Value`, so none needed a family trait; they are now defined once on `Value`.
That duplication was never inert — it is exactly what makes a struct conforming
to two families fail to compile, and `DynValue` is the first struct to try.

Note there is a **fourth** family, `TemporalValue`, which the earlier survey
missed. It has no `materialize` default, so it needed no helper rename.

### Phase 0 result (2026-07-30)

`~/Workspace/dtype-proto/gate_dynvalue_family.mojo` builds and runs. One `Binary`
struct, three instantiations, all correct:

```
fused  : 42     Binary[AddKernel, Column[Int32Type], Column[Int32Type]]
erased : 42     Binary[AddKernel, TagValue, TagValue]
nested : 42     Binary[AddKernel, Binary[AddKernel, TagValue, TagValue], TagValue]
```

All three assumptions hold:

1. **`DynType` conforms to `NumericType` with a placeholder `native`.** There is
   no `DType.invalid` (checked — it does not exist), so `DType.bool` stands in:
   the one DType that is not a numeric lane, so a fused path that wrongly
   elaborated against it yields visibly bool-shaped results rather than a
   plausible-but-wrong integer width.
2. **`comptime OutType = Self.L.OutType` reduces**, including through nesting.
   The chained-projection gotcha in `CLAUDE.md` did not bite here — that warning
   is about a *doubly* chained projection (`Self.OutType.ArrayType`); a single
   projection off a direct trait-bound parameter reduces, as documented.
3. **The fused arm is not elaborated for an erased instantiation.**

`conforms_to` was not usable as the discriminator — its second argument must be a
trait, not a struct. A defaulted marker parameter is the idiomatic alternative and
already has precedent: marrow's `Value` carries `comptime IsBreaker: Bool = False`
for the same job. So `comptime IsErased: Bool = False` on `Value`, `True` on
`TagValue`.

**The finding worth carrying into Phase 1: the marker must propagate.**

```mojo
comptime IsErased = Self.L.IsErased or Self.R.IsErased
```

A composite whose operand is erased is itself erased — its `vectorwise` would
have to call the child's, and the child has no lane to fuse. Without that line
the nested case takes the fused arm over an erased child and **fails to
instantiate**, which is how the probe surfaced it. That failure is also the proof
that the marker genuinely controls elaboration rather than merely selecting a
branch at run time.
