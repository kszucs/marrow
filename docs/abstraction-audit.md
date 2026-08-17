# Abstraction audit

Trait hierarchies, their conformers, what each is responsible for, what is
implemented that should not be, what is missing, and where an abstraction leaks
its implementation.

Verified against the working tree at `f5226d5` + uncommitted changes
(2026-08-16), over the 52 traits declared under `marrow/` and `python/bindings/`.

Companion to the duplication audit (`docs/duplication-audit.md` and
`-proposals.md`, uncommitted at the time of writing — they may not be present on
this branch). This document asks a different question:
not "is this written twice" but "does this type have one responsibility, and does
its trait promise only what it can deliver".

## Status

Findings list, not a plan — **except** where marked. One finding has been acted
on; the rest are open.

| finding | status |
|---|---|
| §3 — the erased boxes' base-trait conformances | **DONE**, see `docs/dyn-conformance-removal.md` |
| §1.2 — `Array.__init__(ArrayData)`, a requirement nothing invokes | **DONE** (removed with the `Array` conformance) |
| §1.14 — `ArrayData` in the `Array` contract | **half done** — the dead half went with §1.2 |
| §1.1 `write_repr_to`, §1.3 `to_device`, §1.4 `CastKernel`, §1.5–§1.13 | open |

§3 has been rewritten in place to describe the outcome rather than the proposal.

### Follow-ups this changeset creates or changes

Three open findings are no longer what they were when first written:

1. **§1.1 `write_repr_to` — the choice narrowed to one option.** The fix used to
   be "put it on `Array`/`ArrowScalar` and have `DynArray` dispatch it". The
   boxes implement neither trait now, so a requirement would bind only the typed
   types and leave the erased handles — the only ones callers hold — still
   forwarding to `write_to`. Deleting all 26 is what remains.
2. **§1.12 `Column` trait — became more valuable, not less.** `DynArray` now
   conforms to nothing either, so the set wanting a shared vocabulary is
   `Array` ∪ `ChunkedArray` ∪ `DynArray`, and all three can answer the same five
   methods. This is the one place where *adding* a conformance is justified by
   the same rule that removed four: it would have a consumer outside its own
   loop.
3. **`Value.OutShape` — the direct continuation.** After §3, `OutShape` is the
   last comptime member on `Value` and its only generic reader is
   `NullPredicate`'s `comptime OutShape = Self.A.OutShape`. Remove that
   propagation and `Value` becomes a pure method trait, so `7d57398`'s rule —
   "erase into a trait whose members are all runtime methods" — would hold with
   no qualification. Tracked in `docs/dyn-conformance-removal.md` §5.

§1.3 (`to_device`/`to_cpu`) and §1.4 (`CastKernel`) are untouched by this work
and stand as first written. §1.4 is the largest open defect in this document:
`safe=True` silently wraps on decimal casts.

---

## 0. The trait map

52 traits, in five groups. The layer column is where the trait's *contract*
lives, which is not always where its conformers live.

| Group | Traits | Conformers |
|---|---|---|
| **Type system** (`dtypes.mojo`) | `DataType` → `PrimitiveType` → `NumericType` → `IntegerType`/`FloatingType`; `TemporalType`, `IntervalType`, `DecimalType`; `BinaryLikeType` → `StringLikeType`; `ListLikeType` | ~40 zero-size structs (`DynType` **no longer conforms** — §3) |
| **Data** (`arrays`/`builders`/`scalars`) | `Array`, `Builder`, `ArrowScalar` | 9 / 37 / 9 typed only — the `Dyn*` boxes **no longer conform** (§3) |
| **Kernels** (`kernels/`) | `Kernel` → `BinaryKernel` → `BinaryNumericKernel`/`BinaryFloatKernel`; `UnaryKernel` → `UnaryNumericKernel`/`UnaryFloatKernel`; `NumericCompareKernel`, `BoolBinaryKernel`, `BoolUnaryKernel`, `UnaryPredicateKernel` → `NullPredicateKernel`/`ValuePredicateKernel`, `StringMapKernel`, `StringPredicateKernel`, `TemporalExtractKernel`, `BinaryConditionalKernel`, `IntervalKernel`, `AggKernel`, `WideningOp`, `MinMaxOp`, `BoolReduceKernel`, `Aggregation`, `AggFunction`, `ColumnAggregator`, `Join` | ~70 structs |
| **Expression / plan** (`expr/`) | `Value` → `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue`/`ListValue`; `WindowKernel`; `Relation`, `Processor` | ~60 fused nodes, `DynValue`, ~12 relations/processors |
| **IO** (`parquet/`, `python/bindings/`) | `ByteSource`, `LeafBuilder`, `LeveledSink`, `ThriftWritable`, `PyConverter` | 1 / 6 / 6 / ~20 / 8 |

Two erasure mechanisms coexist, deliberately: **closed `Variant` + `isa[T]()`**
for the data types (`DynArray`, `DynBuilder`, `DynScalar`, `DynType`) and **open
`ArcPointer[NoneType]` + `thin` function-pointer trampolines** for the plan layer
(`DynRelation`, `DynProcessor`, `BoxedValue`). The split is sound — the node set
of a plan is open and recursive, the set of Arrow layouts is closed by the spec —
but it is not written down anywhere, and `CLAUDE.md`'s "no `rebind` casts, no
function-pointer trampolines" reads as a global rule when it describes only the
first mechanism.

---

## 1. Findings

Ranked by how much they cost today.

### 1.1 `write_repr_to` — a 26-implementation protocol with zero consumers

`write_repr_to` is defined **26 times**: 10 in `arrays.mojo`, 8 in
`scalars.mojo`, and one each in `buffers`, `views`, `execution`, `dtypes` (×2),
`tabular` (×2), `kernels/join`. It is a requirement of **no trait**.

The only call site in the tree is `DynScalar.write_repr_to`
([scalars.mojo:856](marrow/scalars.mojo#L856)), which dispatches to the concrete
scalar's implementation — and `DynScalar.write_repr_to` itself has no callers.
The Python `__repr__` binding does not use it: `_scalar_repr`
([python/bindings/scalars.mojo:58](python/bindings/scalars.mojo#L58)) formats
with `String(ptr[])`, i.e. `write_to`.

Worse on the array side: `DynArray.write_repr_to`
([arrays.mojo:2598](marrow/arrays.mojo#L2598)) does **not** dispatch — it calls
`self.write_to(writer)`. So the 10 `write_repr_to` implementations on the
concrete array types are unreachable through the only handle anyone holds.

Because it is a convention and not a requirement, nothing enforces it either: a
new scalar type that omits `write_repr_to` breaks the build inside
`scalars.mojo`'s dispatch closure, not at the new type.

Either it is part of the display contract or it is not. **The conformance
removal (§3) settled which.** The old fix — put it on `Array` and `ArrowScalar`
and have `DynArray` dispatch it — no longer reaches the boxes at all, since they
implement neither trait. A trait requirement would now bind only the nine typed
arrays and nine typed scalars, leaving the erased handles (the only ones callers
hold) free to keep forwarding to `write_to`. So the remaining option is to
delete all 26.

### 1.2 `Array.__init__(data: ArrayData)` is a requirement nothing invokes

[arrays.mojo:140](marrow/arrays.mojo#L140) makes "constructible from the flat
layout" part of the `Array` contract. No generic code calls it.

Every `[T: Array]` bound in the entire tree lives inside `DynArray`'s own
`_dispatch` closures (`arrays.mojo:2413`–`2625`). Same for `[T: Builder]`
(`builders.mojo:319`–`391`) and `[T: ArrowScalar]` (`scalars.mojo:615`–`856`).
Nothing outside the erasure wrapper is written generically over these traits.
Construction from `ArrayData` goes through `DynArray.from_data` at all 10 call
sites (`ipc.mojo:1795,2021,2051`, `arrays.mojo:1184,1497,1588,1867,2139,2140`).

That reframes what belongs in these three traits: **a requirement earns its place
only if the erased wrapper forwards it, or a concrete caller uses it.** By that
test `__init__(ArrayData)` is dead weight carried by 9 structs, and
`write_repr_to` (§1.1) is the mirror-image mistake — forwarded but not required.

### 1.3 `Array.to_device` / `to_cpu` — a promise 6 of 9 conformers cannot keep

The `Array` trait gives both methods a
`raise Error("...: not supported for this array type")` default. Implemented by
`BoolArray`, `PrimitiveArray` and `FixedSizeListArray`. `NullArray`,
`BinaryLikeArray`, `ListLikeArray`, `FixedSizeBinaryArray`, `StructArray` and
`DictionaryArray` inherit the raise — 6 of the 9 typed conformers.

(`DynArray` also defines both, but since §3 it is a plain method rather than a
trait implementation, so the box is no longer part of this count.)

This is the textbook shape of a leaky abstraction: the interface promises
something the abstraction cannot deliver, and the failure surfaces as a runtime
error at the call site rather than as a type error. A caller cannot ask "can this
column move to a device?" without a `try`.

The kernel layer already solves this problem correctly, twice:
`Aggregation.is_mergeable` is a **comptime `Bool`** the driver checks before
calling `partials`/`merge`, and `ColumnAggregator.mergeable()` is its runtime
counterpart ([groupby.mojo:172](marrow/kernels/groupby.mojo#L172)) — "the grouper
will not pick a strategy it cannot run". The array side has no equivalent
predicate. Device transport is also the only *GPU* concept in the universal array
interface; every other GPU path in the tree sits behind `comptime GPU_ENABLED`.

### 1.4 No `CastKernel` family trait — and `safe`/`ctx` are silently dropped

`cast.mojo` declares 13 structs that conform to bare `Kernel`
(`NumericCast`, `NumToBool`, `BoolToNum`, `TemporalCast`, `StringToNum`,
`StringToBool`, `NumToString`, `BoolToString`, `BinaryLikeCast`,
`FixedSizeBinaryCast`, `NullCast`, `DecimalCast`, `ListCast`, `StructCast`,
`DictionaryCast`) with **six different `dispatch` signatures**:

| shape | kernels |
|---|---|
| `(array, to, safe, ctx)` | `NumericCast`, `DictionaryCast`, `ListCast`, `StructCast` |
| `(array, to, safe)` | `BinaryLikeCast` |
| `(array, to, ctx)` | `BoolToNum`, `TemporalCast` |
| `(array, safe, ctx)` | `StringToBool` |
| `(array, to)` | `NumToString`, `BoolToString`, `FixedSizeBinaryCast`, `DecimalCast`, `NullCast` |
| `(array, ctx)` | `NumToBool` |

Every other kernel family in the tree has exactly one `dispatch` shape declared on
the family trait. Here there is no family trait, so `cast()`
([cast.mojo:986](marrow/kernels/cast.mojo#L986)) is a 15-arm `if/elif` ladder that
hardcodes each kernel's signature and **drops the arguments the arm does not
take**.

The consequence is not stylistic. `DecimalCast`'s own docstring
([cast.mojo:742](marrow/kernels/cast.mojo#L742)) says *"Arithmetic is unchecked
(wrapping / truncating)"* — so `cast(x, decimal128(38, 2), safe=True)` wraps on
overflow instead of raising, silently, because the ladder never passes `safe`
down. `TemporalCast` rescales by `10^n` with the same exposure. `NumToString`,
`BoolToString`, `FixedSizeBinaryCast` and `BinaryLikeCast` likewise never see
`ctx`, so a parallel or GPU context is discarded for those casts with no
diagnostic.

A `CastKernel(Kernel)` trait with one `dispatch(array, to, safe, ctx)` requirement
would make each arm's decision to ignore a flag explicit and reviewable, and
reduce `cast()` to family selection.

### 1.5 `UnaryPredicateKernel` inverts the typed-leaf rule for its whole family

`CLAUDE.md`'s kernel architecture is "typed overloads first, one type-erased
overload on top". Every family trait follows it: `BinaryKernel.apply` takes
`PrimitiveArray[T]`, `StringMapKernel.apply` takes `BinaryLikeArray[T]`,
`TemporalExtractKernel.apply` takes `PrimitiveArray[T]`, and `dispatch` is the
erased wrapper.

`UnaryPredicateKernel.apply` takes a **`DynArray`**
([boolean.mojo:110](marrow/kernels/boolean.mojo#L110)), and `dispatch` is a
one-line `apply(...).to_dyn()`. So `is_null`, `not_null`, `is_nan` and `is_inf`
have no typed entry point at all: a caller holding a `Float64Array` must erase it
to ask whether its values are NaN, and the sub-traits' type resolution
(`dispatch_floating` inside `ValuePredicateKernel.apply`) happens one level below
where every sibling family puts it.

The two sub-traits also justify themselves differently — `NullPredicateKernel`
reads the bitmap and needs no dtype, `ValuePredicateKernel` scans values and
needs `FloatingType` — which is precisely why the shared parent should have been
a *typed* `apply` with the erased `dispatch` above it, not the reverse.

### 1.6 `Kernel` bundles two unrelated responsibilities, and already has a hole

[kernels/core.mojo:17](marrow/kernels/core.mojo#L17) mixes:

1. `comptime name` — identity, used by every conformer for diagnostics.
2. `expect_same_length(Int, Int)` / `expect_same_dtype(DynType, DynType)` —
   argument checks for a **binary array kernel**.

Six conformers can never use (2): `WideningOp` and `MinMaxOp` are pure SIMD
algebra fragments (`identity` + `combine`) passed as parameters to
`Widening[Op]`/`MinMax[Op]`; `IntervalKernel` folds two `Interval`s in the pruning
layer and never sees an array; `AggFunction` is a name→`Aggregation` resolver;
`BoolReduceKernel` and `Aggregation` are folds, not binary ops. They conform
solely to inherit `comptime name`.

And the property the trait exists to guarantee — "a kernel is nameable without
knowing its family" — already has a hole: `WindowKernel`
([values.mojo:2199](marrow/expr/values.mojo#L2199)) declares its own
`comptime name: String` and does **not** conform to `Kernel`.

Splitting `Named` (name only) from `Kernel(Named)` (name + the array checks)
would let the op fragments and the interval kernels say what they are, and let
`WindowKernel` rejoin the hierarchy.

The module already carries the related note:
`# TODO: have vectorwise and elementwise kernels conform to a common trait`
([core.mojo:43](marrow/kernels/core.mojo#L43)).

### 1.7 `marrow.tabular` depends on `marrow.expr` — the dependency tree is not one-directional

[tabular.mojo:23](marrow/tabular.mojo#L23): `from .expr.aggregates import FoldedAggregates`.

`RecordBatch._agg_columns` / `group_by` / `aggregate` resolve string aggregate
names through `FoldedAggregates`. That makes the edge **core → expr → kernels →
core**, the only cycle in the module graph that crosses a layer boundary rather
than sitting inside one package.

The misplaced piece is the *catalog*. `kernels/aggregate.mojo` already owns the
contract for name-based resolution — `AggFunction` is documented as "an aggregate
*function*: a name plus the input dtypes it supports" — but its catalog (`Sum`,
`Min`, `Count`, …) and the `FoldedAggregates` fold live in `expr/aggregates.mojo`.
Nothing about "the aggregate named `sum` over an int64 column" is an *expression*
concept; both `tabular` and `expr` are consumers of it.

Other cycles in the graph are intra-package and expected: `arrays ↔ scalars`,
`arrays ↔ builders`, `buffers ↔ views`, `expr.values ↔ expr.relations ↔
expr.dynamic`.

### 1.8 The plan layer hard-depends on Parquet, and one node's needs leak into `Relation`

`expr/relations.mojo` imports `LeafSet` from `..parquet` and defines `ParquetScan`
+ `parquet_scan()`; `expr/execution.mojo` imports `MappedFile` and the reader
types. There is a `ByteSource` trait abstracting *bytes*
([parquet/source.mojo:20](marrow/parquet/source.mojo#L20)) but **no trait
abstracting a scan** — so IPC, in-memory and any future format are not peers of
Parquet in the plan layer; only `InMemoryTableProcessor` exists alongside it,
hand-written.

The cost shows up in the `Relation` trait itself. `with_predicate`
([relations.mojo:113](marrow/expr/relations.mojo#L113)) returns
`Optional[ArcPointer[NoneType]]` — a **raw erased pointer in a trait
requirement**, imposed on all ~12 `Relation` conformers so that one node
(`ParquetScan`) can be rebuilt with pruning metadata while keeping its comptime
`LeafSet` parameter. The docstring is honest about why (returning
`Optional[DynRelation]` makes the struct recursive), but the shape is a
scan-specific concern that every relation now has in its vocabulary.

A `TableProvider`/`ScanSource` trait — schema, plus "open a `Processor` for this
projection and predicate" — would let `parquet` sit *beside* `expr` instead of
under it, and confine predicate pushdown to the providers.

### 1.9 `Relation.kind()` — RTTI by integer registry, and half of it is dead

[relations.mojo:146](marrow/expr/relations.mojo#L146) plus the
`RELATION_GENERIC`/`RELATION_PARQUET_SCAN`/`RELATION_SORT` constants make every
node responsible for knowing a global discriminant namespace in order to be
recognisable behind `DynRelation`.

It buys exactly one rewrite: `relations.mojo:723` fuses `limit` into a preceding
`Sort`. `RELATION_PARQUET_SCAN` is **never read in the library** — its only uses
are `expr/tests/test_pushdown.mojo:25,108`.

This is also the mechanism `with_predicate` (§1.8) was introduced to *replace* —
the docstring records that a `downcast` to `ParquetScan` stopped being correct
once the scan gained a comptime parameter. The virtual survived; so did the
integer tag it was meant to retire.

### 1.10 `LeafBuilder` and `LeveledSink` — two traits, one responsibility

Both accumulate the values of a Parquet column chunk into an Arrow array.
`LeafBuilder` ([reader.mojo:461](marrow/parquet/reader.mojo#L461)) has 6
conformers — `PrimitiveLeafBuilder`, `ByteArrayLeafBuilder`, `DecimalLeafBuilder`,
`Int96LeafBuilder`, `FixedSizeBinaryLeafBuilder`, `BoolLeafBuilder`.
`LeveledSink` ([reader.mojo:1226](marrow/parquet/reader.mojo#L1226)) has 6
conformers — `_PrimitiveSink`, `_BytesSink`, `_BoolSink`, `_DecimalSink`,
`_FsbSink`, `_Int96Sink`.

The same six physical types, twice, split by whether the column is nested. The
two contracts differ in shape (`consume(page)` / `consume_selected(page, mask)` /
`finish()` versus `handle_dict` / `decode_present` / `place_present` /
`place_null`), because the flat path decodes a whole page at once and the leveled
path is driven row-by-row by `_drive_leveled` — but the *decoding* half is the
same work in both, and adding a physical type means writing two structs that must
agree.

`LeveledSink`'s docstring justifies why it is a trait rather than four closures;
it does not address why it is a second trait rather than a second method set on
`LeafBuilder`.

### 1.11 `Value`'s sub-traits are two different kinds of thing

Three of the five family traits declare the fusion contract —
`comptime State`, `state(batch)`, `lane[W](state, idx)`, `state_validity`, and a
default `materialize` that runs the family driver:

- `NumericValue` ([values.mojo:493](marrow/expr/values.mojo#L493))
- `BoolValue` ([values.mojo:962](marrow/expr/values.mojo#L962))
- `StringValue` ([values.mojo:1487](marrow/expr/values.mojo#L1487))

Two declare **only a fluent surface** and no fusion contract at all:

- `TemporalValue` ([values.mojo:2389](marrow/expr/values.mojo#L2389)) — 12 methods, all returning `TemporalExtract[K, Self]` or an `AggExpr`
- `ListValue` ([values.mojo:2539](marrow/expr/values.mojo#L2539)) — 2 methods

So `Value` has two kinds of sub-trait under one name: a *fused family* and an
*operator namespace*. `TemporalColumn` and `ListColumn` implement `materialize`
directly (hand back their column) and every temporal/list operation is a breaker
into the numeric family. That is a reasonable implementation choice, but it is
invisible from the hierarchy — a reader cannot tell from `trait TemporalValue(Value)`
that no temporal expression ever fuses.

Related, and already noted in the duplication audit's §1.1: `state_validity`'s
12-line docstring and one-line body are **verbatim identical** in all three fused
families. A `FusedFamily(Value)` intermediate holding `State` + `state_validity`
is the obvious home — and is likely blocked by the limit `CLAUDE.md` records
("a trait-level default method cannot return `Self.AssocType` unless that type is
`ImplicitlyCopyable`"). Worth stating in the code so it is not rediscovered a
fourth time.

### 1.12 `ChunkedArray` conforms to nothing

[arrays.mojo:2282](marrow/arrays.mojo#L2282): `struct ChunkedArray(Copyable, Movable, Writable)`.

It is `Table`'s column type as `Array` is `RecordBatch`'s, but shares no trait
with it — not even `Sized` or `Equatable`. It genuinely cannot conform to `Array`
(no single validity bitmap, no single buffer, `ScalarType` undefined), so this is
not a missing conformance; it is a **missing narrower trait**.

The consequence is that no code can be written against "a column of a tabular
thing". `Table`-level operations either duplicate the `RecordBatch` version or
force `combine_chunks` first. A `Column` trait carrying only what both can answer
— `dtype()`, `__len__`, `null_count()`, `slice()`, and iteration over chunks —
would give the tabular layer one vocabulary.

**§3 widened this.** `DynArray` now conforms to nothing either, so the set
needing a shared vocabulary is `Array` ∪ `ChunkedArray` ∪ `DynArray` — and every
member can answer exactly the five methods above. If any follow-up here is worth
doing, this is the one the conformance removal made more valuable rather than
less: it replaces a conformance that existed for its own sake with one that has
a stated consumer.

### 1.13 `ExecContext` is a per-kernel accident, not a property of "a kernel"

Occurrences of `ctx: ExecContext` per kernel module:

| module | count | | module | count |
|---|---|---|---|---|
| `filter` | 25 | | `membership` | 4 |
| `aggregate` | 16 | | `distinct` | 4 |
| `cast` | 14 | | `nested` | 2 |
| `conditional` | 14 | | `string` | 1 |
| `numeric` / `boolean` | 12 | | `concat` | 1 |
| `sort` | 9 | | `temporal` | **0** |

It is visible in the traits: `BinaryKernel.apply` and `NumericCompareKernel.apply`
take a `ctx`; `StringMapKernel.apply`, `StringPredicateKernel.apply` and
`TemporalExtractKernel.apply` do not.

Some of this is correct — an elementwise UTF-8 walk has no SIMD lane to stripe.
But the result is that `ctx.with_threads(8)` silently does nothing for a temporal
or string projection, and a caller cannot tell which from the signature. Either
the family traits take a `ctx` uniformly and ignore it explicitly, or the ones
that cannot use it should say so where a reader will look.

### 1.14 Smaller items

- **`Builder` is not `Writable`** ([builders.mojo:115](marrow/builders.mojo#L115)),
  while `Array`, `ArrowScalar`, `DataType` and `Relation` all are. A `DynBuilder`
  cannot be printed in a diagnostic, for no stated reason.
- **`Join` is a trait for static dispatch with one conformer** (`HashJoin`). Its
  own docstring says runtime selection uses `if/elif` at the call site, so the
  trait is documentation of an intended future shape rather than a live
  abstraction. Fine to keep — worth knowing it constrains nothing today.
- **`Value.OutShape: Int` (`0` scalar, `1` columnar)** is an integer discriminant
  where a two-member enum or a `comptime is_scalar: Bool` would say the same
  thing; `values.mojo:1899-1916` needs 18 lines of comment to explain that
  `Concat.OutShape` is `max(L, R)` and why that preserves a null invariant.
- **`ArrayData` is in the `Array` contract but documented as an interop DTO.**
  **Half resolved:** `__init__(ArrayData)` was the dead half and went with §1.2.
  `to_data()` remains a requirement and is load-bearing — real callers in
  `c_data`, `ipc` and nested construction — so the contract now matches the
  struct's own docstring ("produced *on demand* by `to_data()` for interop").
- **`equal_any`** ([numeric.mojo:583](marrow/kernels/numeric.mojo#L583)) is the
  one place a runtime dtype selects between the numeric and string comparison
  families, and it lives in `numeric.mojo` — which is why `kernels.numeric`
  imports `kernels.string`. The function is well-justified and well-documented;
  its *placement* is what creates the edge. A neutral home (`kernels/compare.mojo`,
  or `kernels/__init__.mojo`) removes it.

---

## 2. Missing traits — summary

| Missing | Would fix | Cost of absence today |
|---|---|---|
| `CastKernel(Kernel)` | §1.4 | `safe`/`ctx` silently dropped on 5 of 15 cast arms; decimal casts wrap on overflow under `safe=True` |
| `Named` (split from `Kernel`) | §1.6 | 6 conformers inherit array checks they cannot use; `WindowKernel` sits outside the hierarchy |
| `TableProvider` / `ScanSource` | §1.8 | `expr` hard-depends on `parquet`; `Relation.with_predicate` carries a raw `ArcPointer[NoneType]` for all conformers |
| `Column` (Array ∪ ChunkedArray) | §1.12 | no code can be written against "a column of a table" |
| `FusedFamily(Value)` | §1.11 | `state_validity` verbatim in 3 traits; fused vs non-fused families indistinguishable |
| one `LeafBuilder` covering leveled decode | §1.10 | 12 structs for 6 physical types, which must agree pairwise |
| a device-capability predicate on `Array` | §1.3 | "can this move to a device" is answerable only by `try` |
| vectorwise/elementwise kernel base | §1.6 | already `# TODO`'d in `core.mojo:43` |

## 3. Why the erased boxes conform to some traits and not others

The question this section answers: `DynArray` conforms to `Array` — so should the
erased boxes also conform to the *typed* trait ladders, stubbing what they cannot
know and raising at run time?

It was tried, shipped, and reverted, and the reasoning is recorded.

### 3.1 The attempt and the revert

`b56b886` (2026-07-30, *"DynValue implements every value family trait"*) made
`DynValue` conform to `NumericValue`, `BoolValue` and `StringValue` alongside
`Value`, and made `DynType` conform to **eight** dtype sub-traits — `DecimalType`,
`FloatingType`, `IntegerType`, `IntervalType`, `ListLikeType`, `NumericType`,
`StringLikeType`, `TemporalType`. The goal was that the fused nodes would accept
an erased operand without relaxing a bound.

`7d57398` (2026-08-03, *"split the expression layer into two lanes"*) reverted it:

> `DynValue: NumericValue` was unsound. `NumericValue` promises a comptime
> `OutType: NumericType` and a `vectorwise` lane; the box supplied a placeholder
> `native = DType.bool` and a stub returning zero. It satisfied the signatures and
> none of the contract […]
>
> **The rule that follows: erase into a trait whose members are all runtime
> methods, never into one with comptime members the box can only stub.**

### 3.2 Why "raise at runtime" is not available

This is the crux. **A comptime member has no execution point at which to raise.**
`comptime native: DType`, `comptime offset: DType`, `comptime State`,
`comptime OutType: NumericType` are resolved during elaboration; the box cannot
answer "I don't know" there. What it can do is supply a plausible constant — which
is what happened: `native = DType.bool`, chosen deliberately as the least-harmful
lie ("the one `DType` that is not a numeric lane, so a path that wrongly
elaborated against it produces visibly bool-shaped results rather than a
plausible-but-wrong integer width").

So the failure mode of this design is **a wrong answer, not an exception** — the
opposite of what the proposal intends. And it does not even fail cleanly at the
conformance site: the compiler reported it as
`attempt to resolve a recursive reference to declaration 'DynValue.__gt__'`, an
error so far from the cause that it forced the whole fluent surface into a
`NumericOps` sub-trait for as long as the conformance existed.

### 3.3 The test, applied to every box in the tree

Does the trait have comptime members, and can the box supply them **truthfully**?

| trait | comptime members | box supplies | sound? | outcome |
|---|---|---|---|---|
| `DataType` | none | — | yes | **removed** — no consumer |
| `ArrowScalar` | none | — | yes | **removed** — no consumer |
| `Array` | `ScalarType: ArrowScalar` | `DynScalar` | yes | **removed** — no consumer |
| `Builder` | `ArrayType: Array` | `DynArray` | yes | **removed** — no consumer |
| `Value` | `OutShape: Int` | `1` | yes | **kept** — `BoxedValue` + the 3-node bridge |
| `PrimitiveType`/`NumericType`/… | `native: DType` | `bool` placeholder | **no** | reverted by `7d57398` |
| `BinaryLikeType`/`ListLikeType` | `offset: DType` | `int32` placeholder | **no** | reverted by `7d57398` |
| `NumericValue`/`BoolValue`/`StringValue` | `OutType`, `State`, `lane` | placeholder + stub returning zero | **no** | reverted by `7d57398` |

**Soundness turned out to be the floor, not the criterion.** The first four rows
were all sound — nothing stubbed — and all four were removed anyway, because
each was load-bearing only for the next box's companion member, in two closed
loops with no external anchor:

```
DynBuilder.ArrayType -> DynArray: Array -> DynArray.ScalarType -> DynScalar: ArrowScalar
Value.OutType        -> DynValue.OutType = DynType -> DynType: DataType
```

`Value.OutType` was read by no generic `[V: Value]` code at all, which is what
collapsed the second loop. Removing all four changed no behaviour and **no binary
size** (0 bytes on all four gates). See `docs/dyn-conformance-removal.md`.

What survives is the sharper rule: a conformance must be **honest** *and* have a
consumer outside its own loop. `DynValue: Value` is the only one that does.

### 3.4 There are no "typed array traits" to conform to

`Array` is **flat** — one trait, 9 concrete conformers. There is no
`PrimitiveArray` trait, no `StringArray` trait; `PrimitiveArray[T]` is a struct.
Same for `ArrowScalar` (9 conformers) and `Builder` (37). So on the array,
scalar and builder side there is nothing further to conform to.

The only place a sub-trait ladder exists is the **dtype** hierarchy — and that is
exactly the ladder `DynType` climbed and fell off.

### 3.5 One placeholder survived the revert — since removed

`7d57398` deleted the eight conformances and the `native` placeholder but left
`comptime offset = DType.int32` ([dtypes.mojo:777](marrow/dtypes.mojo#L777)),
whose docstring pointed at `native` and justified itself by conformances to
`BinaryLikeType`/`ListLikeType` — all three deleted by that same commit.
Measured dead (`precompile` clean without it), and removed alongside
`DynType: DataType`.

`DynType.byte_width`'s docstring was stale for the same reason — it described
itself as a load-bearing override of `PrimitiveType.byte_width` and called
`DynType.native` "the `bool` placeholder". Rewritten; the behaviour was always
correct, only the explanation was wrong.

### 3.6 Where the tree does do "conform and raise" — and what it costs

`Array.to_device`/`to_cpu` (§1.3) are *runtime methods*, so they pass the letter
of the rule. They are precisely the "conform and raise at run time" design, and
they show its actual price: 6 of 9 conformers inherit a raise, and a caller cannot
ask whether a column can move without a `try`.

So the honest summary is not "conform-and-raise is forbidden" — it is:

- **Comptime members: impossible.** No raise point exists; you get a silent wrong
  answer. This is what §3.1 reverted.
- **Runtime methods: possible, but it moves a compile-time error to run time.**
  Worth it when the erased box is the only handle callers hold; not worth it when
  it makes 6 typed structs carry a failure they could simply not have had. §1.3 is
  the second case.

The kernel layer's answer to the same problem is better than either and is already
in the tree: pair the capability with a **predicate the caller checks first** —
`Aggregation.is_mergeable` (comptime) and `ColumnAggregator.mergeable()`
(runtime), so "the grouper will not pick a strategy it cannot run".

## 4. What is in good shape

Stated so a future pass does not re-litigate it.

- **`DataType`'s deliberate minimality.** Five inherited traits, one defaulted
  method, no associated types — and the docstring records that companion
  `ScalarType`/`ArrayType` members were removed in `63b93aa` because they forced
  an import cycle. `DynType` being a *peer* rather than a supertype is the right
  call and is documented as such.
- **The `dispatch_*` family.** Nine narrowing adapters over one
  `DynType._dispatch` loop, each resolving a runtime dtype to a comptime
  parameter, with the "a closure type cannot be generic over its own trait bound"
  limitation recorded at the one place it bites.
- **`Aggregation` / `AggFunction` / `ColumnAggregator`.** Three genuinely distinct
  responsibilities — one resolved aggregate over one input type; a name plus the
  dtypes it supports; what to compute per value column for a grouping — with the
  optional-capability problem solved correctly via `is_mergeable`/`mergeable()`
  (see §1.3 for the contrast).
- **`DynValue` conforms to `Value` and nothing else**, with the rule stated
  explicitly: *"erase into a trait of methods, never into one with comptime
  members you cannot supply."* This is the single best-articulated abstraction
  boundary in the tree, and §1.3 is the one place the codebase violates it.
- **`ByteSource`.** One responsibility, origin-checked borrows instead of a
  convention, one conformer that does not pretend to be more.
- **`PyConverter`** in `python/bindings/arrays.mojo` mirrors the array family
  cleanly, and the wrapper layer's composition-over-inheritance rule holds.
