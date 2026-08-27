# Aggregation architecture

**Supersedes `2026-08-22-aggregation-architecture.md`.** Written after four
independent reviews on 2026-08-27 (naming, layer purity, Mojo mechanics/size,
minimality). Every claim below was checked against the tree or against binaries
on disk; the ones that were checked and *failed* are recorded in §7, because
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
| `groupby.mojo:280,323` `ColumnAggregator`/`OneAggregation` | plan-shaped orchestration (`column: Int` indexes a *query's* aggregate list) plus its adapter | delete — see the ordering constraint in §6 |
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

```mojo
# kernels/aggregate.mojo — the missing half of accumulate[W]
def push_lane[W: Int](mut self, values: SIMD[Acc, W], mask: SIMD[DType.bool, W])
def push_scalar(mut self, value: Scalar[Acc], valid: Bool)
def flush(mut self, mut state: AggState[K, V], slot: Int) raises
```

`FoldOperator.push` then becomes ~25 lines: bind, loop pulling `lane[W]`, push,
flush once per morsel. Identity handling, the count semantics and the horizontal
reduce move next to the algebra that defines them — which is the argument
`combine_at`'s own docstring already makes for itself.

The scalar-tail hazard documented at `:255-258` (*"a `range(0, n, W)` loop reads
past the view and **aborts the process**"*) is a second reason: it is a
`BufferView` invariant enforced by hand in an expr file, and should be enforced
once, in kernels.

**Also:** `SortOperator.drain` (`physical.mojo:744-771`) reimplements multi-key
stable sort — including the bug its own comment warns about — while
`SortIndices.multi` already does it. Out of scope, same defect class, worth a
backlog entry.

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
| `*Scan` / `*Sketch` | a per-group aggregate that is **not** a fold | `ValidityCountScan`, `DistinctCountSketch` |
| `*Op` | an algebra fragment | `SumOp`, `MinOp` |
| `*Grouping` / `Groups` | placement, and its result | unchanged |
| `Aggregate` | **the relational stage only** | `Aggregate`, `AggregateOperator` |

Never `Agg`. Never `-er`, `-Manager`, `-Data`, `-Info`, `-Helper`.

This fixes a **live defect**: `numeric.MinKernel` (element-wise) and
`aggregate.MinKernel` (whole-array fold) collide, and `kernels/__init__.mojo:12`
resolves it by re-exporting *neither* — so `mk.AddKernel` exists and
`mk.MinKernel` does not. `MinFold`/`MaxFold` closes it.

Two names are simply false and must change regardless: **`NumericAgg[K, V: PrimitiveType]**`
serves temporal and decimal (→ `PrimitiveFold`), and **`CountAgg`** claims *the*
count aggregate while deliberately not serving numeric columns
(→ `ValidityCountScan`).

### 4.2 Kernels — what survives

```
FoldKernel          the algebra of a fold: AccType, identity, combine, finalize
FoldState[K,V]      per-group accumulator + counts; accumulate[W], push_lane,
                    push_scalar, combine_at, flush, finish
Groups              rows assigned to dense ids
HashGrouping        the single producer of Groups (a plain struct, no trait)

# the general aggregate: one signature, no trait, no type
comptime ColumnFold = def(Groups, DynArray) thin raises -> DynArray

count_distinct_column · approx_count_distinct_column
string_min_max_column[Op] · validity_count_column · fold_column[K]
```

**`ColumnFold` is the entire extension point.** Every non-fold aggregate —
`count_distinct`, string and temporal min/max, non-numeric count, and the future
catalogue (median, quantile, `first`/`last`, `array_agg`) — is **one function
and zero new types**.

Deleted: `Aggregation`, `AggFunction`, `FoldKernel.Grouped[V]`,
`ColumnAggregator`, `OneAggregation`, `NumericAgg`/`StringMinMax`/`CountAgg`/
`DistinctAgg` *as types*, `OrderedAgg` (zero readers — `_check_domain` never
tests it), `IntegralAgg` (nothing conforms), `Grouping` as a trait,
`ScalarGrouping` (never constructed outside tests).

### 4.3 Expr — the glue

```mojo
# comptime lane: the fused specialisation. Earns its parameters at 14.6x.
struct NumericFold[K: FoldKernel, A: NumericValue](Evaluable, Value)
struct FoldOperator[K: FoldKernel, A: NumericValue, scatters: Bool](Operator)

# everything else: zero parameters, one instantiation for the whole program
struct ColumnAggregate(Evaluable, Value):
    var _input: DynValue
    var _name: String
    var _dtype: DynType
    var _fold: ColumnFold

struct ColumnFoldOperator(Operator)    # zero parameters
```

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

### 4.4 Resolution — split by capability

**One fat box reproduces a measured +3.2 MB (+24%)** (`exprold/aggregates.mojo:248-252`).
So four non-generic resolvers, and a binary links only the ladders it calls:

```mojo
def resolve_out_dtype(name: String, in_dtype: DynType) raises -> DynType
def resolve_grouped(name: String, in_dtype: DynType) raises -> AggColumn
def resolve_fold(name: String, in_dtype: DynType) raises -> Optional[FoldBox]
def resolve_stream(name, in_dtype, var input: DynOperator, grouped: Bool) raises -> DynOperator
```

**Non-generic on purpose.** A `resolve[Job: def[A: …]()]` form is instantiated
per closure type — the `_arith[K]` shape, measured at **+115,600 bytes** — and
there are ≥5 distinct `Job` types in-tree today. A resolver with no type
parameter is one instantiation for the whole tree.

**Mergeability by presence, not by a flag.** `resolve_fold` returns `None` for
non-mergeable aggregates, so they need no `partials`/`merge` bodies at all and a
binary that never merges links no `FoldState.into_partials` — an `nm`-assertable
property rather than a comment.

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
Routing it through `ColumnAggregate` is correct but slow; the real fix is moving
the accumulator dtype out of `__init__` and into first push, which
`FoldOperator`'s own comment already anticipates. **Record it as owed.**

---

## 6. Sequence, with the one hard ordering constraint

| | step | gate |
|---|---|---|
| A | **Fix the measurement tooling first.** Add both expr2 gates to `compare.py:NAMES` (today `compare.py query_expr2_agg_fused` exits *unknown gate*); replace the four dead `MODULE_BUCKETS` entries (`expr::values/dynamic/relations/execution`, 0 symbols each) with `expr::logical/physical/comptime/runtime` (29/28/37); add a `query_expr2_agg_named` gate — **the resolver ladder is ungated in every configuration today** | `compare.py` builds all gates; buckets non-empty |
| B | `ColumnFold` + `count_distinct_column` + `ColumnAggregate` + `ColumnFoldOperator` + `.count_distinct()` on `StringValue` | `agg_count_distinct_string` green |
| C | `string_min_max_column`, temporal min/max via the same node; `.min()/.max()` on `StringValue`/`TemporalValue` | 3 more golden cases green |
| D | Move `FoldOperator.push`'s SIMD body into `FoldState.push_lane/push_scalar/flush` | `__text` measured tip-vs-merge-base; behaviour unchanged |
| E | The four resolvers + `RuntimeAggregate`; repoint the Python bindings | 676 Python tests green |
| F | **Parallel aggregation in `AggregateOperator`** | group-by benches vs today |
| G | Repoint `tabular.mojo` at the plan lane; delete `GroupBy`'s aggregate surface | benches no worse than F |
| H | The renames (§4.1), one mechanical commit | `precompile` clean |

**The constraint: F before G.** `RecordBatch.group_by` is a shipped Python API,
and it currently reaches `GroupBy._thread_local_columns` and the radix path via
`FoldedAggregates` — **the only parallel group-by in the tree**.
`AggregateOperator` is single-threaded. Doing G first makes a shipped API
slower.

**Do not preserve `FoldState.merge`'s current signature to "keep the option
open".** `merge(remap, part_acc, part_cnt, num_groups)` derives from a
whole-input, columns-of-partials model; a morsel-parallel operator wants
`merge(other: FoldState[K,V], remap)`. Deleting it is better than keeping a
signature the future must break. Same for `_by_partition`, which takes the whole
keys `StructArray` at construction and cannot be lifted into a push engine.

**Measurement protocol.** The baseline is **58 commits stale and red on 4 of 7
gates** (`query_expr2_streaming` by +7.3%). Measure **tip vs merge-base**, never
tip vs `baseline.json`, same machine and `pixi.lock`, `__text` only. Do **not**
`--update` to clear the pre-existing reds — that erases the signal. Add the
reachability assertions to `check_gate.py`:

```
nm -C query_expr2_agg_fused | grep -ci distinct     # must stay 0
nm -C query_expr2_agg_fused | grep -c  resolve_     # must stay 0
```

---

## 7. Claims that failed review

Recorded because each was believed and acted on.

| claim | reality |
|---|---|
| "deleting the trait is a size win" | **zero bytes** — Mojo emits no witness tables (verified). Justify by layering only |
| "`ExecContext` is a kernels-layer type" | it is `marrow/execution.mojo:68`, a *core* module, re-exported for convenience |
| "two nodes because the operand bound differs and Mojo has no conditional conformance" | conditional conformance works (`dtypes.mojo:852`); the real reason is that `count_distinct` has **no algebra** |
| "only `count_distinct` is missing" | string and temporal `min`/`max` are missing too — off by two families |
| "the aggregate catalog should move down to `kernels/aggregate.mojo`" (`docs/backlog.md:990-999`) | wrong under §1. **Delete that backlog entry** rather than leave it to mislead |
| "de-genericising the group-by drivers is the real size win" | they are in **no** AOT gate (verified: `aggregate_all:0`, `by_partition:0`); only in `libmarrow.so`, which nothing measures |

## 8. Open, ranked by (chance the design dies) × (cost of finding out late)

1. **Can a move-only `DynOperator` leave a `dispatch_*` closure?** Every resolver
   arm constructs inside `{mut box, imm}` and hands the result out. Proven for
   *copyable* results (`GroupBy.apply:572-578`); `DynOperator` is move-only.
   ~25-line spike.
2. **Does `comptime ColumnFold = def(...) thin raises -> ...` work as a field
   type?** Nothing in the tree aliases a function type; every box writes it
   longhand. ~10-line spike.
3. **Does `ColumnFoldOperator` compile holding a move-only `FoldState`?**
   Two-level move through `DynOperator`. ~40-line spike.
4. **Does `min`/`max` over `timestamp` survive `FoldState`?** The capability the
   whole materialising path exists to add, and nothing exercises it. Test only.
5. **`K.AccType[V]` in a `List[PrimitiveArray[Self.Acc]]` argument position** —
   the struct-level alias form is proven (`aggregate.mojo:909`); this one is not.
