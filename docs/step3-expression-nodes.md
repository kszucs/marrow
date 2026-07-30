# Step 3 — replace `DynValue`'s tag interpreter with node structs

Status: **in progress**, started 2026-07-30. Supersedes the Step 3 sketch in
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

**2. That mechanism already exists.** `AnyValue` (`values.mojo:2270+`) already
boxes via an opaque `ArcPointer[NoneType]` plus `thin` trampolines — exactly the
"open family" canon. Step 3 is therefore not "build erasure". It is "stop
`DynValue` being a fat tagged union, and let it box node structs the way fused
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

**One node set. The erased lane is those same nodes, instantiated with `DynValue`
operands.** There are no `Dyn*` node types — a `DynUnary` would defeat the point.
`col("a").abs()` builds `Unary[AbsKernel, DynValue]`, and `a.add(b)` builds
`Add[DynValue, DynValue]`, which is the same `Add` the fused lane uses.

`DynValue` **is** a `Value`: it erases the plan *shape* (which node types the
tree is built from) while `DynArray` erases the data *lane*. It replaces
`AnyValue` outright — that box exists only because the name `DynValue` was taken
by the tag interpreter, and it has to carry the union of two representations (a
fused node *or* an interpreter). Once the interpreter is gone there is one
representation, so the box erases a single trait and `DynValue` can hold a
`DynValue` like any other `Value`.

This is the shape `~/Workspace/dtype-proto/unified/expr.mojo` already
demonstrates: `DynValue[In, Out](Value)` over an opaque `ArcPointer[NoneType]`
plus `thin` trampolines, alongside `Add[L: Value, R: Value]`.

### The one thing that has to be made to work

`DynValue` implements **every family trait** — `NumericValue`, `BoolValue`,
`StringValue` — not just `Value`. That is what lets the existing node bounds stay
exactly as they are: `NumericBinary[K, L: NumericValue, R: NumericValue]` accepts
`DynValue` for `L` and `R` with no relaxation, so `a.add(b)` on two erased
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

`DynValue` keeps the public surface the tag interpreter has today — factories
(`col`, `lit`, `if_else`), operator overloads, `.cast()`, `.isin()`, `.year()`,
the aggregate sugar — so callers and tests are unaffected. What changes is that
each operator constructs a shared node rather than a tagged record: from

```mojo
var _tag: UInt8
var _args: List[DynValue]
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
exist in `values.mojo`, instantiated with `DynValue` operands:

| tags | existing node |
|---|---|
| ADD SUB MUL MOD FLOORDIV | `NumericBinary[K, DynValue, DynValue]` |
| DIV | `FloatBinary[DivKernel, DynValue, DynValue]` |
| EQ NE LT LE GT GE | `NumericCompare[K, DynValue, DynValue]` |
| AND OR XOR | `BoolBinary[K, DynValue, DynValue]` |
| NEG ABS | `NumericUnary[K, DynValue]` |
| NOT | `BoolUnary[NotKernel, DynValue]` |
| IS_NULL NOT_NULL | `NullPredicate[K, DynValue]` |
| CAST | `NumericCast[To, DynValue]` and family |
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

### Phase 2 — `DynValue` replaces `AnyValue`
The box implements `Value` and the family traits, and takes over `AnyValue`'s
role in `relations.mojo`, `execution.mojo` and the binary-size gate programs.
The tag interpreter still exists and still works; it is simply one more thing the
box can hold. **Gate:** suite green, gates within noise.

### Phase 3 — operators build shared nodes
`DynValue.__add__` returns `Add[DynValue, DynValue]` re-boxed, and so on through
the operator surface, each backed by the `comptime if` erased arm in the existing
node. Tags fall out of use group by group as their operator is converted:
binaries and compares first (the shape the gates measured), then unary and
temporal, then the payload-carrying ones (`cast`, `like`, `isin`, `coalesce`,
`nullif`, `case_when`, `date_trunc`). **Gate after each group:**
`query_dynvalue` / `query_runtime` `__text`.

### Phase 4 — delete the interpreter
Remove `_tag`, the tag constants, the three switches, and `AnyValue`. Give
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

### Phase 0 result (2026-07-30)

`~/Workspace/dtype-proto/gate_dynvalue_family.mojo` builds and runs. One `Binary`
struct, three instantiations, all correct:

```
fused  : 42     Binary[AddKernel, Column[Int32Type], Column[Int32Type]]
erased : 42     Binary[AddKernel, DynValue, DynValue]
nested : 42     Binary[AddKernel, Binary[AddKernel, DynValue, DynValue], DynValue]
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
`DynValue`.

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
