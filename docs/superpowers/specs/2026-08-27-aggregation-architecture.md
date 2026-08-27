# Aggregation architecture

**Supersedes `2026-08-22-aggregation-architecture.md`.** Written after four
independent reviews on 2026-08-27 (naming, layer purity, Mojo mechanics/size,
minimality). Every claim below was checked against the tree or against binaries
on disk; the ones that were checked and *failed* are recorded in §8, because
three of them were mine.

`marrow/exprold/` is being deleted. Nothing here ports from it; it is cited only
as evidence of what a concept was for, and as a working reference for two
measured incidents.

---

## 1. The rule

> **If changing this code can change the answer, it is a kernel.
> If it can only change the time, the memory, or the label, it is glue.**

That is the whole boundary, and it is derivable — a contributor can place a new
concept without asking. Four things follow immediately, and each is a live
violation today (§3):

- A kernel may not turn a **name** into behaviour, nor define the user-facing
  name something else turns into behaviour.
- A kernel may not **decide** anything whose inputs span more than one kernel
  invocation, or whose alternatives all produce the same answer.
- A kernel may not construct a `Field`, a `Schema`, or an output column name.
- A kernel may not name `expr` — in a signature, an import, or a docstring.

And the converse: **`expr/` may not contain a loop over lanes.** If a body reads
`identity`, `combine`, `finalize`, a validity mask or a SIMD tail, it is compute
and belongs in a kernel that takes those lanes *by value*.

---

## 2. What is actually true about size

**Deleting a trait is worth zero bytes.** Verified:

```
nm query_streaming_agg | grep -ci 'conformance|witness|vtable'   →  0
```

Mojo emits no witness tables; the specialisation key in every mangled symbol is
the concrete type. The trait bound appears only as mangling noise. So this
refactor must be justified by **layering and clarity, not by size** — any plan
that sells it as a size win is selling something that is not there.

**The DCE property is real and must become an asserted invariant.** Verified:

| gate | `distinct` symbols | `StringMinMax` |
|---|---|---|
| `query_expr2_agg_fused` | 0 | 0 |
| `query_streaming_agg_fused` | 0 | 0 |
| `query_streaming_agg` (runtime-named) | 4 | 6 |

It comes from **who calls the resolver**, not from any trait. It holds iff the
fluent comptime API never reaches one.

**What is gated and what is not.** Verified:

```
query_streaming_agg   aggregate_all:0   by_partition:0   AggFunc:95
libmarrow.so                            by_partition:7
```

The group-by drivers are in **no** AOT gate. De-genericising them shrinks only
the shipped `.so`, which nothing measures. The **resolver box** is the gated
cost.

---

## 3. Violations to fix

### 3.1 Glue in `kernels/`

| location | what | fix |
|---|---|---|
| `aggregate.mojo:234,425,439,511,556,1375` | the SQL vocabulary — `"sum"`, `"min"`, `"count_distinct"` — defined in kernels and used as a dispatch key | expr's resolver compares its own literals; `Kernel.name` becomes what `core.mojo:21` already claims: *"for display and diagnostics, never dispatch"* |
| `aggregate.mojo:1113` `AggFunction` | a trait whose whole job is name→behaviour | delete |
| `aggregate.mojo:113` `AggKernel.Grouped[V]` | a resolution table inside the algebra; **zero reads** outside `exprold` | delete — and it is the *only* forward reference from `AggKernel` to `Aggregation`, so removing it breaks the cycle inside the file |
| `aggregate.mojo:164-187` `AggKernel.reduce` default | allocates an `n`-element zeros array to say "everything is group 0" — the cost `ScalarGrouping` exists to avoid. Every conformer overrides it | make `reduce` abstract; delete the body |
| `aggregate.mojo:1174-1182` `NumericAgg.whole` | `ctx.with_threads(1)` — a kernel narrowing the caller's budget. Its own comment says the gating "belongs in the reduce primitive itself" | the reduce primitive gates on size; aggregates pass `ctx` through untouched |
| `groupby.mojo:418-440` | `_PARALLEL_MIN_ROWS`, `RADIX_BITS`, `GROUP_SERIAL/THREAD_LOCAL/RADIX` — strategy policy, public "so the expression layer can reuse the same strategy decision" | the *numbers* are measured facts and stay as named constants; the `if` moves to expr. Precedent: `ExecContext.worth_parallel(n, min_parallel_size)` requires the threshold from its caller for exactly this reason |
| `groupby.mojo:280,323` `ColumnAggregator`/`OneAggregation` | plan-shaped orchestration (`column: Int` indexes a *query's* aggregate list) plus its adapter | delete — see the ordering constraint in §7 |
| `groupby.mojo:259-269,506-527` | builds `Field("k0")`, `Field("key")` to feed a hash kernel | `RapidHashKernel` takes `List[DynArray]` |
| ten docstrings across `aggregate.mojo`, `groupby.mojo`, `distinct.mojo`, `core.mojo` | kernels prose naming `marrow.exprold.aggregates` / "the expression layer" | rewrite to state a compute contract |

### 3.2 Compute in `expr/` — the largest violation in either direction

**`FoldOperator.push` (`comptime/aggregates.mojo:239-365`) is ~130 lines of pure
SIMD compute in the expression layer**: identity fill, dual accumulator, masked
lane select, int64 count vector, horizontal reduce, mandatory scalar tail. The
only expr-shaped token in the body is `self._input.lane[W](bound, i)`.

The module docstring defends the location on lane-organisation grounds. That
argues *where within `expr/`*, not *whether in `expr/`*.

`AggState.accumulate[W]` already solves the identical problem for the scattering
arm — **the caller pulls lanes, the kernel takes them by value.** The scalar arm
simply has no counterpart. Add one:

These are **not** members of `FoldState` — `flush` takes a `FoldState`, so
`self` is something else. They belong to a new per-morsel, register-resident
scratch type: the four locals `FoldOperator.push` creates today at
`comptime/aggregates.mojo:302-306`, plus the `cnt` vector at `:315`.

```mojo
# kernels/aggregate.mojo — a per-morsel register accumulator
struct FoldRegister[K: FoldKernel, V: PrimitiveType](Movable):
    comptime Acc = Self.K.AccType[Self.V]
    comptime W   = simd_width_of[Scalar[Self.Acc.native]]()

    var _vec: SIMD[Self.Acc.native, Self.W]
    var _cnt: SIMD[DType.int64, Self.W]      # a count is a count, not an Acc
    var _acc: Scalar[Self.Acc.native]
    var _n: Int

    @always_inline
    def push_lane(mut self, values: SIMD[Self.Acc.native, Self.W],
                  mask: SIMD[DType.bool, Self.W])
    @always_inline
    def push_lane(mut self, values: SIMD[Self.Acc.native, Self.W])  # null-free
    @always_inline
    def push_scalar(mut self, value: Scalar[Self.Acc.native], valid: Bool)
    def flush(mut self, mut state: FoldState[Self.K, Self.V], slot: Int) raises
```

**`push_lane` must not be generic in `W`.** The register's width is the width of
its own `_vec`; a `[W: Int]` parameter would admit a lane it cannot combine.
The tail is `push_scalar`.

`FoldOperator.push` then becomes ~25 lines: bind, loop pulling `lane[W]`, push,
flush once per morsel. Identity handling, the count semantics and the horizontal
reduce move next to the algebra that defines them — which is the argument
`combine_at`'s own docstring already makes for itself.

**The scalar-tail bound does not move, and an earlier draft wrongly claimed it
would.** `simd_end = (n // W) * W` and both loops stay in `FoldOperator.push`,
because only the caller can call `self._input.lane[W](bound, i)`. The register
never sees `n` or a view. Moving that bound into kernels would need a
lane-pulling driver taking a closure — the +662,740-byte adapter shape CLAUDE.md
forbids.

Scope, corrected: the scatter arm (`aggregates.mojo:260-298`) is untouched — it
needs `gids` and already calls `accumulate[W]`. `push` goes ~127 → ~70 lines,
not ~25.

**Also:** `SortOperator.drain` (`physical.mojo:735-771`) reimplements multi-key
stable sort while `SortIndices.multi` already does it. Out of scope, same defect
class, worth a backlog entry. (It composes the permutation *correctly* — an
earlier draft claimed it contained the bug its own comment warns about, which it
does not.)

---

## 4. The design

### 4.1 Naming — one word per concept, one layer per word

`aggregate.mojo` currently spells one word six ways: `Aggregation`, `AggKernel`,
`AggState`, `AggFunction`, `NumericAgg`, `CountAgg`, `DistinctAgg`. That is not
a convention, it is inconsistency, and carrying it forward would make it a
convention by accident.

| suffix | means | examples |
|---|---|---|
| `*Kernel` | element-wise compute | `AddKernel`, `EqKernel` |
| `*Fold` | fold algebra, and folds bound to an input type | `SumFold`, `MinFold`, `PrimitiveFold`, `FoldState`, `FoldOperator` |
Never `Agg`. Never `-er`, `-Manager`, `-Data`, `-Info`, `-Helper`.

`Agg` is dropped because `aggregate.mojo` spells one word six ways today. Note
what is **not** a rename: `NumericAgg`, `CountAgg`, `StringMinMax`, `DistinctAgg`,
`Aggregation` and `AggFunction` are **deleted** (§4.2), not renamed, so no
`*Scan`/`*Sketch` tier is needed — free functions have function names.

`MinKernel`/`MaxKernel` → `MinFold`/`MaxFold` incidentally lets
`kernels/__init__.mojo` re-export the fold family, which it declines to do today
because `numeric.MinKernel` and `aggregate.MinKernel` collide. That is a
documented, argued decision rather than a defect — the rename simply removes the
reason for it, **and only if the re-export is added in the same commit.**

### 4.2 Kernels — what survives

```
FoldKernel          the algebra of a fold: AccType, acc_dtype, identity,
                    combine, finalize, empty_is_null, needs_count, reduce
FoldState[K,V]      per-group accumulator + counts; accumulate[W], push_lane,
                    push_scalar, combine_at, flush, merge, finish
Groups              rows assigned to dense ids; is_single() for the one-slot case
                    (and `__len__` DELETED — see below)
HashGrouping        the single producer of Groups (a plain struct, no trait)

# the general aggregate: one signature, no trait, no type
comptime ColumnFold = def(Groups, List[DynArray]) thin raises -> DynArray

count_distinct_column · approx_count_distinct_column
string_min_max_column[Op] · validity_count_column · fold_column[K]
```

**`List[DynArray]`, not `DynArray`, from the start.** No multi-input aggregate
is currently scheduled — `docs/backlog.md` M2.4 is `variance`, `stddev`,
`quantile`, `approximate_median`, `mode`, `first`, `last`, all single-input — so
this is not justified by a roadmap item. It is justified by cost asymmetry: the
signature is the expensive-to-change part (every implementation plus the box),
and widening it now is free because the operator owns the list and lends it.

**`Groups.is_single()` must be the first branch of every `ColumnFold`, not
documentation.** This is the design's sharpest hazard. Deleting
`Aggregation.whole` routes *everything* through `ColumnFold(Groups, …)`, and
every existing per-group implementation loops `range(len(groups.ids))` —
`count_distinct_grouped` (`distinct.mojo:179`), `approx_count_distinct_grouped`
(`:229`), `StringMinMax.grouped` (`aggregate.mojo:1255`), `CountAgg.grouped`
(`:1323`). Handed `Groups(empty, 1)` they return `[0]` or `[null]`: **the loop
simply does not execute.** Today they are never handed one, because
`Aggregation.whole` synthesises a real zeros array first.

That branch is also where `whole`'s three overrides must land, or they are lost
silently: `count_distinct`'s radix-parallel whole-array path (`distinct.mojo:89`)
and `CountAgg.whole`'s **O(1)** metadata answer (`aggregate.mojo:1332`). Losing
the latter turns `RecordBatch.aggregate(["x"], ["count"])` from O(1) into O(n).

**Delete `Groups.__len__` (`core.mojo:72-74`).** It has **zero callers**, its
docstring says *"Number of rows assigned"*, and it returns `len(self.ids)` —
which is **0** for an ungrouped morsel. Any `ColumnFold` author who writes
`for i in range(len(groups))` gets a silently empty result. A free cut of a live
trap, on a struct whose entire docstring is about preventing exactly this.

### 4.3 Expr — the glue

```mojo
# comptime lane: the fused specialisation. Earns its parameters at 14.6x.
struct NumericFold[K: FoldKernel, A: NumericValue](Evaluable, Value)
struct FoldOperator[K: FoldKernel, A: NumericValue, scatters: Bool](Operator)

# everything else: zero parameters, one instantiation for the whole program
struct RuntimeAggregate(Evaluable, Value):
    var _inputs: List[DynValue]
    var _func: String     # the resolver key: "count_distinct", "min", …
    var _alias: String    # Value.name(); empty until .alias()

struct RuntimeAggregateOperator(Operator)    # zero parameters
```

**Named for the lane, because that is what it is.** Nothing about
`RuntimeAggregate` is comptime: it holds erased operands and a function pointer,
and it materialises. `col("region", string).count_distinct()` therefore *is* the
point where the comptime lane hands off to the runtime one — the operand was a
comptime `StringColumn`, and an aggregate with no fold algebra cannot carry it
any further. Stating that in the name stops the next reader looking for a
fusion that was never possible.

**Two name fields, not one.** `_func` is the resolver key; `_alias` is what
`Value.name()` answers and what `Aggregate._output_schema` reads
(`logical.mojo:658`). `.alias("n")` must change the second and not the first —
one field would make `write_to` print `n(region)` for
`golden/cases/agg_count_distinct_string.mojo`. `NumericAggregate` already keeps
them apart: `_name` is the alias, the function name is `Self.K.name`.

**No `_dtype` and no `_fold` field.** Neither is available where the node is
built — see §4.4.

**The operator is not a fold**, so it is not named one: it buffers erased
columns and calls one thin pointer. Being parameterless, it also does not make
the lane-agnostic layer name a lane, so unlike `FoldOperator` it belongs in
`physical.mojo` beside `AggregateOperator`.

**Why two nodes, stated correctly.** Not "the operand bound differs and Mojo has
no conditional conformance" — conditional conformance *is* available
(`dtypes.mojo:852,904` uses `comptime if conforms_to(T, NumericType)` + `rebind`
today). The real reason: **`count_distinct` has no fold algebra**, so `K` cannot
be supplied at any operand bound. The two differ on the **algebra** axis. A
single node would need a `DistinctKernel` with a meaningless `identity`/
`combine`/`finalize` invented to fill a parameter slot — which is precisely the
ceremony `Aggregation` was.

**No `"agg"` tag on `RuntimeValue`.** The resolver returns a `DynValue`
directly; an aggregate always terminates a value tree and is stored straight
into `Aggregate._aggs`. `RuntimeValue` never holds one, `evaluate` gains no case,
and the zero-row-batch dtype probe never special-cases a node that cannot be
evaluated.

### 4.4 Resolution — two functions, at two different times

```mojo
# marrow/expr/aggregates.mojo — a top-level sibling of logical/physical,
# because it belongs to neither lane.
def aggregate_out_dtype(name: String, in_dtypes: List[DynType]) raises -> DynType
def resolve_fold(name: String, in_dtypes: List[DynType]) raises -> ColumnFold
```

**Not one function, and not one that constructs the node.** An earlier draft
said `resolve_aggregate(name, inputs, in_dtype) -> DynValue` was "the whole
surface". It cannot be, for two independent reasons:

- **`Aggregate._output_schema` runs before any batch exists.** It calls
  `a.dtype(schema)` at plan-build time (`logical.mojo:637-660`), so a
  dtype-only path is mandatory and must take a `Schema`.
- **The fluent API has no dtype to pass.** `col("ts", timestamp(us)).min()` is
  written where no schema exists, and a temporal dtype is *not constructible
  from its type* — `TemporalColumn.dtype` reads the schema for exactly this
  reason (`comptime/leaves.mojo:134-141`: *"a numeric dtype is `Defaultable`
  and can answer from its type, a temporal one cannot"*). So a node that stored
  a resolved `ColumnFold` could never be built for temporal `min`/`max` — which
  is a step the sequence schedules.

So the node stores only `_func` and its inputs. `aggregate_out_dtype` is called
by `_output_schema` with dtypes derived from the schema; `resolve_fold` is
called by the **operator, on first push**, from the morsel's real dtypes. That
is the same "resolve at first push, not at construction" move
`comptime/aggregates.mojo:230-234` already says the fused lane must eventually
make.

Both take `List[DynType]`, not one — `string_agg(x, sep)` has two different
input dtypes.

**Non-generic on purpose.** A `resolve[Job: def[A: …]()]` form is instantiated
per closure type — the `_arith[K]` shape, measured at **+115,600 bytes**.
A resolver with no type parameter is one instantiation for the whole tree.

**The cost of splitting, stated.** `Aggregation.out_dtype` is today a *second,
compiler-checked* statement of the (kernel, dtype) pairing — it must agree with
`grouped`'s return type or the code does not build. Two hand-written tables can
silently disagree, and `min` over `timestamp[us]` yielding a schema field of
`timestamp[s]` is a `Variant` misaccess at emit, not a raise. §6's test
requirement is sharpened accordingly.

**Mergeability is expressed by the operator, not by a flag.** Parallel
aggregation (§7 E) is N `FoldOperator`s over disjoint morsels plus
`FoldState.merge`; an aggregate that cannot merge is one the operator does not
run in parallel. No `is_mergeable` bit, no raising stubs.

---

## 5. Scope — what is actually missing

Corrected: it is **not** only `count_distinct`.

| aggregate | operand | new-lane path today |
|---|---|---|
| `sum` `product` `mean` `min` `max` `count` | numeric | ✅ |
| `count_star()` | — | ✅ (`lit(1,int64).count()`) |
| `min` / `max` | **string** (`agg_min_max_string`) | ❌ |
| `min` / `max` | **date32, timestamp** (`temporal_min_max_*`) | ❌ |
| `count_distinct` | **string** (`agg_count_distinct_string`) | ❌ |

The fluent surface at `comptime/core.mojo:405-428` exists **only on
`NumericValue`**. So 4 of golden's 6 `.max()` and 4 of its 5 `.min()` uses have
no path either.

Temporal min/max is a genuine fold blocked only by `FoldOperator.__init__`
calling `Self.A.Type()` — `TemporalType` and `DecimalType` are not `Defaultable`.
Routing it through `RuntimeAggregate` is correct but slow; the real fix is moving
the accumulator dtype out of `__init__` and into first push, which
`FoldOperator`'s own comment already anticipates. **Record it as owed.**

---

## 6. Soundness of the layering

**One aggregate-*resolution* trait is deleted, not "one trait remains."**
`kernels/` holds **30** traits and `aggregate.mojo` alone keeps seven after
every deletion here: `AggKernel`, `WideningOp`, `OrderedAgg`, `ArithmeticAgg`,
`IntegralAgg`, `MinMaxOp`, `BoolReduceKernel`, plus `Kernel` in `core.mojo`. An
earlier draft claimed "one trait in `kernels/`", which is flatly false.

What the design actually achieves is narrower and still worth having: **no trait
in `kernels/` resolves an aggregate any more.** `Aggregation` and `AggFunction`
go; `ArithmeticAgg` stays and earns it — it is what makes `sum(date)` a build
error via `_check_domain` (`aggregate.mojo:339-343`). Deleting
`FoldKernel.Grouped[V]` removes the only forward edge from the algebra to the
resolved layer, so `aggregate.mojo` has no internal cycle, and `expr → kernels`
is one-directional (verified: no `from ..expr` import exists in `kernels/`).

**Extension is honest.** A new fold is one `FoldKernel`. A new non-fold is one
function. Neither touches a trait.

**One regression, and it must be paid for explicitly.** `ColumnFold` takes
`List[DynArray]`, so a dtype mismatch is a `Variant` misaccess — an **abort**,
not a catchable `Error`. Today `Aggregation.from_any` is declared `raises`, so
the type system provides a second line of defence; after the cut, the resolver
is the *only* thing tying a kernel to a column's dtype. This is the failure mode
`boolean.mojo`'s `_as_bool` exists to prevent, and whose docstring records the
abort taking down a whole test file. Two requirements, not hopes:

1. **`resolve_aggregate` is the sole constructor of a `ColumnFold`.** Nothing
   else may build one, so there is exactly one place the pairing can be wrong.
2. **One test per (name, dtype-family) pair, asserting the *schema* dtype
   equals the *produced* dtype.** Not merely "the value is right": the specific
   new failure is `aggregate_out_dtype` and `resolve_fold` disagreeing, which
   the compiler used to prevent because `Aggregation.out_dtype` had to agree
   with `grouped`'s return type. `min` over `timestamp[us]` declaring
   `timestamp[s]` is a `Variant` misaccess at emit, not a raise.

3. **A `thin` fn carries no identity.** If `resolve_fold` returns the wrong arm,
   nothing downstream can name which aggregate it holds — `Kernel.name` is on
   the type, and the type is gone. The resolver is the only place this can be
   caught.

**Two stated properties this design reverses, and must say so.**

- **`AggregateOperator` buffers again.** `ColumnFold` is one-shot over the whole
  input, so the operator must accumulate each morsel's evaluated columns and
  ids, concat at `drain`, and call the fold once. `physical.mojo:534-541`
  currently claims *"**Nothing is buffered** … this one keeps only the grouper's
  key builders, which grow with the number of *groups* rather than the number of
  rows."* That becomes false for any query containing a `count_distinct`, and
  memory is O(rows) again. There is precedent (`SortOperator._batches`,
  `JoinOperator._buffered`) so it is acceptable — but the docstring must be
  corrected in the same commit, not left lying. Concat-across-morsels is sound:
  `HashGrouping` ids are dense and stable across batches.
- **`Grouping` becomes a `Bool`.** §4.3 replaces `G: Grouping` with
  `scatters: Bool`. Three places argue the opposite — `groupby.mojo:171-174`,
  `comptime/aggregates.mojo:184-187` and `:331-333`, all saying a sorted or
  radix placement should arrive as a *conformer* rather than another `Bool`.
  The counter-argument, which those predate: `ScalarGrouping` is **never
  constructed** outside tests, and a sorted or radix grouping changes `assign`
  while still scattering — so it instantiates the *same* `scatters=True`
  operator. The bit that varies and the object that varies sit on opposite
  sides of the `Morsel` boundary. That is why the trait never earned the
  parameter position; it is not a size argument (a `comptime Bool` specialises
  identically) and should not be sold as one.

**What the layering deliberately cannot express.** An aggregate that must see
rows in order — `lag`, `lead`, `first`/`last` without a sort. Those are window
functions: they emit one row per *input* row, so their operator differs
regardless, and no aggregate abstraction should stretch to cover them.

---

## 7. Sequence

No bridge types and no temporary surfaces: every step lands the shape the end
state keeps. `exprold` stays untouched until the step that replaces it, which is
not scaffolding — it is declining to delete something before its replacement
works.

| | step | gate |
|---|---|---|
| **A** | **Rename the survivors only**, before any new code: `AggKernel`→`FoldKernel`, `AggState`→`FoldState`, `NumericAggregate`→`NumericFold`, `MinKernel`/`MaxKernel`→`MinFold`/`MaxFold` (+ the `kernels/__init__` re-export the collision blocked). Every *other* name §4.1 calls wrong is **deleted** by G, not renamed — 12 references versus 77 — so doing the survivors first means B–G are authored in the final vocabulary instead of being re-touched six times | `precompile` clean |
| **B** | Add the `query_expr2_agg_named` gate and the two `nm` reachability assertions to `check_gate.py`. **Not the `compare.py:NAMES`/`MODULE_BUCKETS` churn** — `baseline.json` already holds both expr2 gates and `check_gate.py` already builds them; that part was gold-plating. The new gate is not: `fold_column[K]` reintroduces a per-kernel `dispatch_primitive` ladder and nothing measures it today | new gate builds and records |
| **C** | `ColumnFold` · `Groups.single()`/`is_single()` · **delete `Groups.__len__`** · `count_distinct_column` *with its `is_single` whole-array branch* · `RuntimeAggregate` · `RuntimeAggregateOperator` · `aggregate_out_dtype` · `resolve_fold` · `.count_distinct()` as **one** `ComptimeValue` trait default. Correct `AggregateOperator`'s "Nothing is buffered" docstring | `agg_count_distinct_string` green — note it is **keyless**, so it exercises exactly the empty-ids path; DCE assertions still 0 |
| **C2** | **A grouped `count_distinct` golden case.** None exists; C's only gate is the ungrouped path, which is the one that silently returns `[0]` | new case green |
| **D** | `string_min_max_column[Op]`, `validity_count_column`, `.min()`/`.max()` on `StringValue` | `agg_min_max_string` green |
| **D2** | `fold_column[K]` for the six numeric kernels, and `aggregate_array(name, arrays, ctx)` — **what G consumes**, and the only replacement for `marrow.compute`'s seven array-level entry points (`compute.mojo:134-153`), which take one erased array and a context and have no plan to be repointed at | `query_expr2_agg_named` recorded; ladder cost visible |
| **D3** | Temporal `min`/`max`, via `aggregate_out_dtype`'s schema-derived dtypes | `temporal_min_max_timestamp` green |
| **E** | The `FoldRegister` extraction (§3.2) | `__text` vs **the commit before E**, not vs merge-base — B–D have added symbols by then |
| **F** | Parallel aggregation in `AggregateOperator`; `FoldState.merge(other, remap)` | group-by benches |
| **G** | Repoint `tabular.mojo` and the seven `compute.mojo` entry points; delete `Aggregation`, `AggFunction`, `Grouped[V]`, `ColumnAggregator`, `OneAggregation`, `GroupBy.aggregate`/`apply`/`aggregate_all`; delete `NumericAgg.whole`'s `.with_threads(1)` **with a bench, not a correctness gate** — `aggregate.mojo:1178` asserts `wants_parallel`'s 32768 default is too low for a SIMD reduce, so removing it is a behaviour change | 676 Python tests green; benches no worse than F *except* the named regression below |

**F before G**, because `RecordBatch.group_by` is a shipped Python API and
`AggregateOperator` is single-threaded until F.

**An accepted regression, with a number required.** `GroupBy` has **two**
parallel paths, not one: `_thread_local_columns`, which F recovers, and
`_by_partition(partition=True)` (`groupby.mojo:730-747`, `:833-845`), which runs
the aggregate inside `RadixPartitioner.map_partitions`. The radix path is the
only one `count_distinct` can use — `DistinctAgg.is_mergeable = False`
(`aggregate.mojo:1378`) — and F does not recover it. So after G,
`count_distinct` is **single-threaded**. Either measure and accept it, or keep
`_by_partition` alive for the non-mergeable case; do not assert, as an earlier
draft did, that the thread-local path is the only one.

**Measurement protocol.** The baseline is **58 commits stale and red on 4 of 7
gates** (`query_expr2_streaming` by +7.3%). Measure **tip vs merge-base**, never
tip vs `baseline.json`, same machine and `pixi.lock`, `__text` only. Do **not**
`--update` to clear the pre-existing reds — that erases the signal.

---

## 8. Claims that failed review

Recorded because each was believed and acted on.

| claim | reality |
|---|---|
| "deleting the trait is a size win" | **zero bytes** — Mojo emits no witness tables (verified). Justify by layering only |
| "`ExecContext` is a kernels-layer type" | it is `marrow/execution.mojo:68`, a *core* module, re-exported for convenience |
| "two nodes because the operand bound differs and Mojo has no conditional conformance" | conditional conformance works (`dtypes.mojo:852`); the real reason is that `count_distinct` has **no algebra** |
| "only `count_distinct` is missing" | string and temporal `min`/`max` are missing too — off by two families |
| "the aggregate catalog should move down to `kernels/aggregate.mojo`" (`docs/backlog.md:990-999`) | wrong under §1. **Delete that backlog entry** rather than leave it to mislead |
| "one trait in `kernels/`" | **30** traits in `kernels/`; seven survive in `aggregate.mojo` alone. The true claim is narrower: no trait in `kernels/` *resolves an aggregate* any more |
| "`push_lane`/`push_scalar`/`flush` are the missing half of `accumulate[W]`" on `FoldState` | `flush` takes a `FoldState`, so they cannot all be its members. They belong to a new `FoldRegister` |
| "M2.4 already schedules multi-input aggregates" | M2.4 is `variance, stddev, quantile, approximate_median, mode, first, last` — **all single-input**. `List[DynArray]` is justified by cost asymmetry, not by a roadmap item |
| "`SortOperator.drain` contains the bug its comment warns about" | it composes correctly (`physical.mojo:762-768`) |
| "`resolve_aggregate` is the whole surface, and it constructs the node" | `_output_schema` runs before any batch exists, and a temporal dtype is not constructible from its type — so a dtype-only path is mandatory and the node cannot store a resolved fold |
| "`RecordBatch.group_by`'s only parallel path is `_thread_local_columns`" | `_by_partition(partition=True)` is a second one, and the only one `count_distinct` can use |
| "de-genericising the group-by drivers is the real size win" | they are in **no** AOT gate (verified: `aggregate_all:0`, `by_partition:0`); only in `libmarrow.so`, which nothing measures |

## 9. Settled by spike, and what is still open

A throwaway spike (`/tmp/agg_spike.mojo`, built `mojo build -I .`) settled the
three load-bearing mechanics questions. All three **compile and run**:

| | question | result |
|---|---|---|
| Q1 | `comptime ColumnFold = def(Groups, List[DynArray]) thin raises -> DynArray` — as an alias, as a **struct field type**, as a **parameter type**, and **callable through the field** | ✅ all four |
| Q2 | a **move-only** `DynOperator` constructed inside a `dispatch_numeric` closure captured `{mut box, imm}`, escaping via `List.pop()` | ✅ |
| Q3 | an `Operator` holding a move-only `AggState[SumKernel, Int64Type]` taken by `var`, then boxed into `DynOperator` — two levels of move | ✅ |

Q1 is the important one: the extension point is expressible exactly as written,
so `ColumnFold` needs no longhand repetition and no restatement. Q2 is what
`resolve_fold` is built on. Q3 is what `RuntimeAggregateOperator` is built on.

**Deliberately not tested, and still open:**

1. **`AggState[SumKernel, T]` with `T` narrowed by the dispatch.** The spike used
   a fixed `Int64Type` to isolate the *escape* question from the `AccType[V]`
   projection. `NumericAggregate.Type`'s docstring warns that `K.AccType[V]`
   "reduces inside the struct but fails to unify at a return site", and
   `FoldState.finish -> PrimitiveArray[Self.Acc]` proves only the struct-level
   alias form. A `List[PrimitiveArray[Self.Acc]]` **argument** is unproven.
2. **`min`/`max` over a `timestamp` column through `FoldState`** — the capability
   the whole materialising path exists to add, and nothing exercises it. A test,
   not a spike.
3. **The `ComptimeValue` trait default returning a concrete node type.** Legal
   today, but CLAUDE.md records that a default whose return type a conformer
   must change becomes an ambiguous overload at every call site — so
   `NumericValue` could never later specialise `.count_distinct()` to a fused
   form. Acceptable; worth knowing it is a one-way door.
4. **The pre-existing +10.1% on `query_expr2_agg_fused`** — undiagnosed, on a
   60-commit-stale baseline, and it is the gate whose DCE property §2 turns into
   an invariant. Making it an invariant on an untrusted instrument is not sound.
   One `git bisect run` against `check_gate.py` settles it.
