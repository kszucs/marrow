# marrow — open work

The epics and tasks worth doing, in priority order. **Nothing here describes
how the code is built** — for that read the code and `CLAUDE.md`, which is the
tracked home for architecture, coding rules, compiler gotchas and measurement
traps. This file is only what is missing, what is wrong, and what it would take.

Untracked (`notes/` is gitignored). Two things are recorded per item: why a
user cares, and what standing in the way is real rather than assumed. Claims
were re-verified against the tree on 2026-09-04; where a claim did not survive
that check it says so inline.

## What is missing, in priority order

One table for the whole backlog. **Priority** is what a first user hits soonest;
**complexity** is S (a day), M (a week), L (a couple of weeks), XL (a project
with its own design). "Blocked by" names a thing that must land first, not a
preference.

Detail for every row is in the tiers below; the standing constraints in §3 gate
all of them.

| # | Missing | Why it matters | Cx | Blocked by |
|---|---|---|---|---|
| 1 | **Float `/` by zero answers the dividend** — `10.0 / 0.0` is `10.0`, not NULL, not `inf` | `//` and `%` mask the zero-divisor rows and `/` does not, so the three operators disagree about the same divisor. Open deliberately: it is a choice about float division, and whichever answer is picked should be pinned by a case | **S** | — |
| 2 | **CI is dark** — no run since 2026-05-11, `test.yml` calls a deleted task, `binary_size` not wired in at all | Everything below is unverifiable in CI. A +55% size regression once survived ten commits for exactly this reason | **S** | — |
| 3 | **CSV reader**, then NDJSON | A first user arrives with a CSV, not a Parquet file. `find marrow -iname '*csv*'` is empty | **M** | — |
| 4 | **Error taxonomy** — 269 `raise Error` sites, zero typed exceptions | Cheap while the Python boundary is fresh, expensive to retrofit across 269 sites. Already a retrofit | **M** | — |
| 5 | **`scan(path)` without a hand-written schema**, then globs, directories, hive partitions | `scan()` takes one path *and* demands the schema by hand. Every real Parquet dataset is a directory | **M** | 3 |
| 6 | **Parallel group-by — landed 2026-09-01 and silently reverted the same day** | `f17045a9` took `groupby.mojo` 224 -> 517 lines with radix-partitioned placement. `bcbbd32e`, whose subject is *"delete StringArgs, one trait per string signature"*, put it back to 224 and deleted `test_groupby.mojo` (520 lines) and `bench_groupby.mojo` (157). 980 lines gone; nothing noticed, because the tests that would have caught it went in the same commit. `GROUP_RADIX`, `GROUP_THREAD_LOCAL` and `_choose_strategy` are zero grep hits today. Recover with `git show f17045a9:marrow/kernels/groupby.mojo` | **S** | — |
| 7 | **`distinct`, `union`, `except`, `intersect`** — no node exists for any of them | Table stakes for a SQL-shaped frontend, and `ReplaceDistinctWithAggregate` is a rule nobody can write without the node | **M** | — |
| 8 | **The string and temporal verbs cannot be reached from Python** | All 17 (`substr`, `lpad`, `replace`, `split_part`, `position`, `epoch`, `last_day`, `week`, …) exist in **both** lanes and evaluate: comptime methods, and runtime free functions in `runtime/values.mojo` with their tags handled in `evaluate`. What is missing is reach — they are not methods on `RuntimeValue`, `expr/__init__.mojo` re-exports only `RuntimeValue` and `RuntimeAggregate`, and **0 of 17** appear in `python/marrow/_expr.py`. The work is bindings and a re-export, not kernels or nodes | **S** | — |
| 9 | **Join output ordering** — `JoinOperator` hardcodes build=left, `_output_schema` is positional | Blocks *both* remaining optimizer rules. Not an optimizer change: the kernel must accept an output ordering | **M** | — |
| 10 | **Join reordering + build-side selection** | The largest TPC-H win available, and the only genuinely cost-based pass in any incumbent | **L** | 9, 11 |
| 11 | **Statistics propagation and a cost model** | Feeds 10. Not urgent on its own — it is the last piece, after 9 | **L** | — |
| 12 | **CSE and duplicate group/sort key elimination** | Both need `DynValue` equality — likely solvable at the verb, as `constant_bool` and `conjuncts` were, rather than with a box slot | **M** | — |
| 13 | **Larger-than-memory execution** — no spilling anywhere | Every aggregate and join is bounded by RAM. Changes the operator contract | **XL** | — |
| 14 | **Nested-loop / range joins** | Only equijoins exist, so a non-equi predicate has no plan at all | **M** | — |
| 15 | **UDFs** | The escape hatch that makes a missing kernel survivable rather than fatal | **M** | 4 |
| 16 | **A row format** | Needed by sort-merge join, spilling, and any wire protocol | **L** | — |

---

## 1. Open work

Ordered by value. Nothing below is started.

### 1.1 Correctness — known-wrong answers

**Float `/` by zero answers the dividend.** `DivKernel.core` is `a / b` with a
zero `b` replaced by 1, so `10.0 / 0.0` is `10.0` — not `inf`, not NULL, but
the numerator. SQL says NULL. This is the last survivor of the family fixed on
2026-09-04: `//` and `%` now mask the zero-divisor rows out, and `/` does not,
so the three operators disagree with each other about the same divisor.

Nothing pins it — no golden case asks, and the covered-arithmetic section of
`golden/COVERAGE.md` never named `/ 0`. The fix is mechanical on the erased
side (`DivKernel.extra_validity` returning `Self.nonzero_validity(right)`,
which already exists) and needs one more node on the fused side: `Div` is a
`FloatBinary`, which has no third `Bound` slot, so it needs the treatment
`DivisionBinary` gave `//` and `%`.

**SQL's null on a zero divisor costs 13-16% on `//` and `%`.** Measured
2026-09-04 on `bench_{floordiv,mod}_int32_{10k,100k,1m}`, drift-corrected
against a +1.0% control median. The rounding correction inside `core` is
free — stubbing the validity back out puts both kernels within +/-1.2% of the
old ones — so the whole cost is the extra pass over the divisor that the null
requires. Three variants of that scan were measured:

| variant | cost |
|---|---|
| per-chunk `reduce_or` | **+15%** (kept) |
| vector accumulator collapsed once, via `select` | +26% |
| vector accumulator collapsed once, via `cast` | +26% |
| `SIMD[DType.bool, width]` accumulator | does not compile — see §3 |

The accumulator losing is the counter-intuitive part, and it is specific to
this shape: ARM collapses a vector in one instruction, and an accumulator
sized to the *value* type is only `width` bytes wide, so it spends the loop in
partial registers.

Removing it means not scanning at all, which means computing the zero-mask
*during* the compute pass — an accumulator threaded through `views.apply`'s
driver, which today hands a lane a destination and nothing to fold into. That
is a core-API change and it would serve exactly two kernels, so it is recorded
rather than done. The fused lane already avoids the scan for a literal
divisor, which is the common shape; this is the erased path only.

**`count(*)` desugars to `count(lit(1))`.** `builders.mojo` returns
`lit(1, int64).count().alias("count_star")`, whose `columns()` is empty — so
anything that prunes by "what does this read" prunes every column, and
`RecordBatch.num_rows()` returns **0** when there are none (`tabular.mojo`).

**Projection pushdown landed 2026-08-31 and handles it rather than fixing it:**
`ColumnPruning` never narrows a source to zero columns, keeping the first when
the demand is empty. The desugaring is unchanged, so the trap still waits for
the next thing that reads `columns()` and believes the answer.

**Fixed 2026-09-04, kept here for the shape.** Three of the four `xfail`s were
one root cause — Mojo's `//` floors and its `%` takes the divisor's sign, and
both substituted 1 for a zero divisor — and the fourth was a *hash* bug
mistaken for a grouping one: `HashKernel` widened a float lane with
`cast[uint64]()`, a numeric conversion, so every value in (-1, 1) truncated to
0 and the hash-only table merged them. Two lessons worth keeping: a
convention bug hides wherever the two conventions agree (`math_mod_int64`
asked `((n % 3) + 3) % 3`, the agreeing form, for exactly that reason), and a
kernel that decides validity from a *value* cannot do it in the lane — the
mask belongs outside, in `BinaryKernel.extra_validity` and in the fused node's
`Bound`.

### 1.2 The published guides document APIs that do not exist

The only user-visible item here, and now one guide rather than two.
`docs/guide/expressions.qmd` documents `from marrow.expr import col, lit`
(lines 26 and 56) — there is no such Python module; the names are
`marrow.col` / `marrow.lit` / `marrow.read_parquet` — and line 5 calls the
engine "pull-based" when `physical.mojo` is push/drain. This is on the
rendered site.

**`docs/guide/compile.qmd` came off this list on 2026-09-03** with the CLI work: it
had documented `plan.execute_cli()`, deleted with the old CLI entry point, and
the branch rewrote all 377 lines against the `QueryCli` that now exists.

**`README.md` came off this list on 2026-08-30**, when the Python query API was
restored: its lazy-query section had described `morsel_size` and `strictness`
arguments that no longer exist, claimed `explain()` "renders one node, not the
tree" when plans render recursively, cited two deleted docs, and carried a
40/43 ClickBench claim whose harness went with them.

### 1.3 Latent compiler hazards

**~15 more `t"…{dtype}"` sites under `marrow/kernels/`.** A t-string
interpolating a recursive `Writable` value inside a function-level recursion
cycle deadlocks the compiler — the finding is in `CLAUDE.md`. Three sites were
fixed; the rest are untested and sit in the same shape.

**File the deadlock upstream.** `docs/repros/tstring_recursive_writable_cycle.mojo`
is a 55-line reproducer with no marrow import for a compiler hang that emits no
diagnostic. Modular should have it.

### 1.4 Engine capability the golden corpus measures as missing

`golden/COVERAGE.md` is authoritative and machine-checked: 278 cases, of which
**63 carry `-- skip mojo`** because marrow has no API for them. Their bodies
are never compiled, so they are proposals rather than verified spellings.

Counted 2026-09-03, by prefix:

| skipped | area |
|---|---|
| 13 | aggregates — median, quantile, first/last, arg_min/arg_max, string_agg, corr/covar, mode, skewness |
| 10 | temporal — date_diff, age, strftime/strptime, make_date, iso_week |
| 8 | nested — struct field, map lookup, list element/slice, unnest |
| 7 | math — atan2 and the rest of the trigonometric family |
| 5 | string — regexp, concat_ws |
| 4 | set operations — UNION / EXCEPT / INTERSECT |
| 4 | joins — cross, non-equi, asof |
| 3 | GROUPING SETS / ROLLUP / CUBE |
| 3 | filters — SQL `NOT IN` null semantics, `.is_in` as a method |
| 3 | decimals |
| 2 | subqueries |
| 1 | `DISTINCT ON` |

**Fifteen came off between 2026-08-31 and 2026-09-03** — seven window cases
(windows), `math_log_bases`, and the string and temporal surface, which
landed `substr`, `replace`, `lpad`, `position`, `split_part`, `last_day` and
`epoch`. `bool_and`/`bool_or` is still the odd one out among the aggregates:
`AnyKernel`/`AllKernel` are `BoolReduceKernel`s rather than `FoldKernel`s and
have no grouped variant, so they need an aggregate node in both lanes *plus* a
kernel change.

`temporal_epoch_seconds` is the sharpest instance and was re-measured on
2026-09-04: `EpochKernel` exists, the body compiles, and un-skipping it still
fails on one row. Un-skipping it is a change to the case (`CAST(FLOOR(...))`
and a regenerated expectation), not to marrow.

Three of the remaining skips are **not** missing API. `math_greatest_and_least`
and `filter_not_in_list_with_null` encode SQL null semantics marrow does not
implement — skip-nulls extrema, and `NOT IN` with a NULL matching nothing —
and `nested_list_contains` needs `.contains` as a method on `ListValue`, where
only the free `array_contains` exists. Re-checking a case by name is not enough
to un-skip it; §1.12 records four that were claimed unblocked and were not.

### 1.8 Test and infrastructure gaps

- **No cross-lane parity test.** "One engine, two drivers" was enforced by a
  `test_parity.mojo` across four axes; it went with the previous expression
  package and has no replacement. The invariant is currently unenforced.
- **`HashGrouper` has no dedicated test** — `kernels/tests/test_groupby.mojo`
  was deleted without replacement.
- **CI has not run since 2026-05-11.** `test.yml` calls a deleted task, the docs
  job cannot pass, and the binary-size gate is not in CI at all — which is how a
  +55% size regression once survived ten commits.

### 1.10 Python binding limits, measured 2026-08-30

Audited against the `std.python.bindings` surface at Mojo
1.1.0.dev2026083005 while restoring the Python query API. Each was verified by
reading `mojo/stdlib/std/python/{bindings,_python_func}.mojo` **and** by
running the built `.so`; none is a marrow bug, and each dictates a shape the
bindings currently have.

- **`PythonTypeBuilder.bind` installs four slots** -- `tp_new`, `tp_init`,
  `tp_dealloc`, `tp_repr` -- and nothing else. `def_method` fills the type's
  `tp_dict`, not a CPython slot, so a registered `__str__` is reachable as
  `obj.__str__()` and **not** as `str(obj)`, which falls back to `tp_repr`.
  Measured: `str(expr)` returns `"<marrow.Expr: gt(a, 1)>"` while
  `expr.__str__()` returns `"gt(a, 1)"`. Every Python wrapper that wants the
  real text calls `._binding.__str__()` explicitly (`LazyTable._plan_text`).
  The same limit is why operators live in Python: a registered `__add__` would
  never fire for `+`, and a registered `__eq__` would never fire for `==`.

- **No `__eq__` reaches `tp_richcompare`, and none is registered anyway.**
  `ma.int32() == ma.int32()` is `False` and `ma.array([1,2]) == ma.array([1,2])`
  is `False` -- both identity comparisons. The Python suite compares dtypes by
  `str()` for exactly this reason. Fixing it needs a slot, not a method.

- **`def_property` does not exist.** The `tp_getset` slot is unexposed (there
  is a design doc for it in the modular tree, unlanded). So `Array.type()` and
  `RecordBatch.num_rows()` are methods where PyArrow has properties. That is a
  real divergence from the "follow PyArrow closely" rule and it is not
  currently fixable from this side.

- **A dotted type name sets `__module__` but breaks attribute lookup.** CPython
  3.14 emits `DeprecationWarning: builtin type X has no __module__ attribute`
  once per registered type -- 11 per import today. Passing `"probe.Dotted"` to
  `add_type` does set `__module__` correctly, but `finalize(module)` uses the
  same string as the module *attribute* key, so the type lands at
  `vars(m)["probe.Dotted"]` and `m.Dotted` stops resolving. Verified both ways.
  The warning cannot be silenced without an upstream change that splits the
  `PyType_Spec` name from the attribute name. Worth reporting.

- **`PyObjectFunction` supports 8 positional arguments, kwargs, and a typed
  self** -- more than the bindings use. The typed-self form takes
  `Pointer[T, MutAnyOrigin]` as its first parameter and downcasts
  automatically, which would delete the `py_self.downcast_value_ptr[T]()` line
  at the top of most binding functions and most of `helpers.mojo`'s `pymethod`
  factory family. Not adopted here: it is a mechanical sweep across ten binding
  modules and belongs in its own change. The kwargs form is deliberately *not*
  adopted -- keyword sugar lives in pure Python by project rule.

- **Statistics pruning is unreachable from Python.** `DynRelation.filter` has
  two overloads: the erased one takes `DynValue` and reads every row group, and
  the pruning one takes `V: Value & Prunable` and captures the concrete type.
  A `PythonObject` has already erased that type by the time it crosses the
  boundary, so `Plan.filter` can only reach the first. `RuntimeValue` *does*
  conform to `Prunable`, so what is missing is a way to carry the unerased
  value across, not the pruning itself -- the `Expr` box holds a
  `RuntimeValue` and could call the pruning overload directly.

### 1.11 Undocumented subsystems

Nine substantial pieces of the codebase have no design document and never did:
the whole Parquet subsystem (nine modules, ~380 KB), the Arrow IPC layer, the C
Data Interface, the GPU execution model, `utils/argparse.mojo` (769 lines),
`kernels/groups.mojo`/`bounds.mojo`/`cast_decimal.mojo`, `Dispersion`, the
`comptime/temporal.mojo` nodes, and the `_drop` destructor trampoline on every
erased box. Listed so that "there is no doc" is not mistaken for "there is no
feature".

---

### 1.12 Found while landing the window, CLI and string/temporal work

Five findings that were not on this list and are not covered by any row above.
Ordered by what they cost.

**The binary-size gate cannot see a sort.** `sort_indices` decoded dictionary
keys through the type-erased `cast`, whose single non-generic body chains
`elif` over every type family — so **every AOT binary that sorted anything
linked 694 cast symbols**, about 3 MB, for a path most plans never take. Fixed
in `04d84cb8`: decode directly with `take` and convert the index through an
integer-only ladder. A sorting binary went 4,510,744 -> 1,441,112 of `__text`,
a 68% cut, and sorting now costs 12,440 bytes over a plan that does not sort.

This is the third instance of the same shape — `kernels::cast` reachable from
a plan that needs none — after the hashing fix and `ParquetScan`. What
let it survive is that **no gate program sorts**. Adding one is the actual
task here; without it the next instance is equally invisible.

**Windows read a pruned population.** `Window.to_operator` forwarded its
`Pushdown`, so a `Filter` above a window pushed its row-group predicate to the
scan and every rank and running total was computed over fewer rows than the
query selects — wrong numbers, no error. Fixed in `04d84cb8`. Nothing could
have caught it: all 21 window tests and 7 golden cases use in-memory tables,
and the bug needs a Parquet scan under a window under a filter.

**A bounded sort forwarded the same predicate.** `Sort.to_operator` forwarded
unconditionally while `PushFilterBelowSort` refused when `sort.limit` was set,
so the two mechanisms disagreed about a `TopN` sort. Unreachable only because
`TopN` leaves the `Limit` above, which clears. Fixed in `a85bd62d`.

**`COUNT` over an empty window frame answered NULL.** The identity of the empty
set is the aggregate's to decide and the aggregate operator already decided it
correctly; a short-circuit was discarding that. Fixed in `04d84cb8` by deleting
the special case.

**NaN ordering keys are never peers — still open.** `mark_changes` decides
peer identity with `equal()`, which is IEEE, while the sort maps all NaNs to
one key and places them adjacent. The docstring claims `IS NOT DISTINCT FROM`
and delivers it for NULL but not NaN.

**The root is not in the window path.** `sort` and `equal` disagree about NaN,
and anything pairing those two kernels inherits it — `distinct` and `group_by`
are the other candidates. Fixing it inside `mark_changes` would paper over
that, so the first task is to establish whether the divergence is general.

### 1.13 Framed window aggregates are O(n^2)

`WindowOperator._framed_aggregate` constructs **one aggregate operator per
row** and rescans that row's frame. Under SQL's default frame the frame grows
to the whole partition, so `SUM(x) OVER (ORDER BY y)` over 100k rows is about
5 billion element visits.

The implementation says so itself — *"one operator per row ... the honest price
of the reuse; a running accumulator would be a per-aggregate, per-dtype kernel
and is what to write when this shows up in a profile"* — and the trade it buys
is real: every aggregate is a window aggregate at once, with the kernel's own
null semantics rather than a second implementation of them.

**Nothing measures it.** The golden fixture is seven rows and there is no
window benchmark, so the quadratic is invisible. A `bench_window.mojo` belongs
with the fix, not after it.

Note this is the *one* window cost a comptime lane cannot address. Fusion has
nothing to fuse in a breaker, the per-row work is already typed kernels, and
`lag`/`lead`/`first_value`/`last_value` reduce to one `take`. The accumulator
is an algorithmic change that happens to want comptime as its mechanism.

## 2. Missing capabilities, in detail

The tiering is by *user impact*, and it
cuts across the priority table above: a Tier 1 item can be cheap and a Tier 2
item can be the largest thing here.

Each section states what exists, what is absent, and what it would take.

### Tier 1 — table stakes

A user rejects the library outright without these.

#### 1.1 A query API from Python

**What exists.** Nothing. `python/marrow/__init__.py` exposes `Array`, `Scalar`,
`RecordBatch`, `Table`, IPC read/write and `parquet.read_table`/`write_table`;
`python/marrow/compute.py` has 21 functions (`add`, `subtract`, `multiply`,
`divide`, six comparisons, `any`/`all`, `filter`, `take`, `cast`, `is_null`,
`is_valid`, `drop_null`, `sort_indices`, `sort`). `python/bindings/lib.mojo`
registers `dtypes, scalars, arrays, compute, schema, tabular, ipc, parquet` —
no expression module.

The frontend existed and was deleted on 2026-08-29 in commit `b2de85a0`
(message: "e"):

```
D  python/bindings/expressions.mojo
D  python/bindings/plan.mojo
D  python/marrow/lazy.py
D  python/marrow/_expr_column.py
D  python/marrow/tests/test_lazy.py
D  python/marrow/tests/test_expressions.py
D  python/marrow/tests/{bench_,test_,profile_}clickbench.py
```

`README.md:178-247` still documents `ma.read_parquet`, `ma.memtable`, `col`,
`lit`, `collect()`, `to_pyarrow()`, and claims 40/43 ClickBench queries pass.
None of it resolves today; the ClickBench harness went in the same commit and
`benchmarks/` no longer contains it.

**Done, 2026-08-30 — and it was not only re-wiring.** The estimate above said
"the runtime lane was built for exactly this shape", and that was half right.
The relation nodes and fluent verbs were indeed ready. The *expression* lane
was not: it could express comparisons, three-valued boolean logic, `coalesce`
and `case_when` and nothing else — no arithmetic, no strings, no temporal
extraction, no casts, no null predicates. A frontend on that could not have
written `col("a") + 1`. Restoring the API therefore meant adding ~45 verbs to
`RuntimeValue`, each a new tag over an existing kernel, plus the two binding
modules and the two Python modules.

It also surfaced one wrong answer: `_compare` promoted mixed dtypes by casting
the right operand to the left's type and falling back to the reverse, which
narrows rather than widens, so `int32_col > lit(2**40)` raised instead of
comparing. `promote_dyn` now states the same rule the comptime lane's
`promote[L, R]` does.

#### 1.2 CSV and JSON readers

**What exists.** Nothing. The only occurrence of "csv" anywhere under `marrow/`
is a test string in `marrow/kernels/tests/test_string.mojo`.

**What it would take.** marrow already has the hard parts: `Buffer`,
`BufferView`, every builder, and `LittleEndian.fixed` as the byte-order
primitive. What is new is a tokenizer, a sampling inference pass, and a
type-widening lattice. NDJSON is the same shape with a different tokenizer.
**This is the highest ratio of user value to engineering novelty on the whole
page.**

#### 1.3 Datasets: multi-file, partitioned, remote

**What exists.** `scan(path: String, schema: Schema)`
(`marrow/expr/builders.mojo:213`) — one file, and the caller supplies the schema
because "a `Relation` is a description and must not touch the filesystem to
exist". `ByteSource` is a deliberate seam whose docstring anticipates "a
streaming reader or a remote (OpenDAL) object store later", but `MappedFile` is
the only implementation (`marrow/parquet/source.mojo`).

**What it would take.** Three separable pieces. (a) Derive a `Schema` from the
Parquet footer so `scan(path)` needs no schema — small; everything needed is in
`marrow/parquet/schema.mojo`. (b) A `MultiFileScan` relation node owning a list
of sources and yielding row groups across them, plus hive-path parsing to
synthesise partition columns. (c) An object-store `ByteSource`, which is the
seam's stated purpose but needs an HTTP client marrow does not have.

#### 1.4 The optimizer: no cost model, no CSE

**What exists.** A plan-to-plan rewriter in `marrow/expr/optimizer.mojo` —
**16 rules and one downward pass**, invoked as `plan.optimize[AllRules]()`,
which returns an ordinary `DynRelation` that prints, diffs and executes:

    Limit(Sort(Filter(ParquetScan(...))))  ->  Sort(Filter(ParquetScan(...)) top 10)

| | |
|---|---|
| elimination | `EliminateFilter`, `RemoveEmptyLimit`, `PropagateEmpty`, `RemoveNoOpProject`, `RemoveRedundantSort`, `RemoveSortBeforeAggregate` |
| merging | `MergeProjects`, `MergeLimits` |
| splitting | `SplitConjunction` |
| pushdown | `PushFilterBelowProject`, `PushFilterBelowSort`, `PushFilterBelowJoin`, `PushFilterBelowAggregate`, `PushLimitBelowProject` |
| reparameterization | `TopN` |
| downward pass | `ColumnPruning` |

plus constant folding in the `RuntimeValue` constructors, and the original
Parquet statistics pushdown, which still rides `to_operator`'s descent unchanged.

The rule set is a comptime parameter, so a binary links exactly the rules it
names and `execute()` alone optimizes nothing. `DynRelation` became **a variant
for inspection and a trampoline for lowering**: `isa[R]()`/`get[R]()` let a rule
read a real typed node and construct one, while `to_operator` stays on a
per-type slot — routing it through the variant instead cost **+348%** of
`__text` on `query_streaming`, because that ladder instantiates every node's
lowering and `ParquetScan.to_operator` reaches `kernels::cast` in a plan with no
Parquet in it.

**The design note this section used to cite is wrong and has been corrected.**
`pushdown.mojo` claimed a rewrite was "not merely unnecessary but unavailable"
given `DynRelation`'s layout. Trampolines returning `List[Self]`, `Self` and
`Optional[Self]` all compile; what the compiler rejects is a by-value recursive
*field*, which is a different thing. See
`CLAUDE.md`'s Mojo gotchas, and `marrow/expr/pushdown.mojo`'s own
corrected docstring.

**Still absent:** common-subexpression elimination, duplicate group/sort key
elimination, statistics propagation, aggregate pushdown, and any cost model.

**Blocked in the kernel, not the optimizer:** join reordering and build-side
selection. `Join._output_schema` is positional (left fields then right) and
`JoinOperator` hardcodes build=left, so both rewrites change the output column
order and are not expressible as plan rewrites at all. They need
`kernels/join.mojo` to accept an output ordering.

**Two engine bugs surfaced by building it**, both fixed: an ungrouped aggregate
above a `Limit` returned zero rows (`Pipeline.drain` skipped every stage above
a finished one), and `RecordBatch.__eq__` was not reflexive (`Buffer.__eq__`
compared the 64-byte-aligned allocation past the logical end). Neither was
visible to a harness that compares an optimized plan against an unoptimized one
— both sides agree and pass.

A credible engine ships without a cost model.

It is also the mirror image of marrow's `count_star()` defect below — the same
expression that blocks projection pushdown is the one an optimizer most wants
to special-case. (marrow's projection pushdown now clamps rather than
special-cases: it never prunes a source to zero columns. Fast count-star
remains uncopied.)

**What it would take — done, and the estimate was wrong in an instructive
way.** This said projection pushdown was "a second field on the `Pushdown`
struct", and that anything beyond it needed "a real plan representation — a
structural change, not an increment." The structural change is what shipped:
`DynRelation` is variant-backed, nodes carry `traverse`, and rules rewrite
plans. Projection pushdown turned out **not** to fit the `Pushdown` struct at
all — it needs a downward pass with an accumulator, where `Pushdown` carries
per-node facts.

The `count_star()` hazard was real and is handled rather than fixed:
`ColumnPruning` never narrows a source to zero columns, keeping the first
column when the demand is empty, because a `RecordBatch` carries its row count
in its columns. The underlying desugaring (`lit(1, int64).count()` with empty
`columns()`) is unchanged.

#### 1.6 String and temporal function coverage

**What exists.** 20 string kernels (case, strip family, reverse, capitalize,
byte length, starts/ends/contains, six comparisons, `LIKE`/`ILIKE`, and a
`ConcatKernel` wired to no expression node) and 11 temporal extractors plus
`date_trunc`.

**Absent — 16 string cases:** `substr`, `replace`, `split_part`, `concat` and
`concat_ws` (the kernel exists; the node does not), `lpad`, `position`,
`repeat`, `left`/`right`, `trim(characters)`, character-length as distinct from
byte-length, `ascii`, and the whole regex family.

**Absent — 13 temporal cases**, of which the smallest is the most damaging: `lit`
has numeric and string-like overloads only (`marrow/expr/builders.mojo:123-153`),
so **no date or timestamp constant can be written**. `WHERE ts > TIMESTAMP
'2024-01-01'` is not expressible at all. Also missing: `date_diff`, interval
arithmetic, `last_day`, `epoch`, ISO week/year, `strftime`/`strptime`,
`make_date`, day/month names, `age`, timezone attachment. Timezones are carried
on the type (`dtypes.mojo:394`) and **ignored by every kernel** —
`marrow/kernels/temporal.mojo:37` states a non-UTC timestamp is decomposed in
UTC.

ibis's `strings.py` is the engine-level expectation: case, trim/pad,
substring/slice, find/predicate, pattern match, regex (extract/split/
replace), replace/split/join, and URL parsing.

**What it would take.** Most are ordinary kernels over machinery that exists.
Two are not: regex needs a real engine — `mojo-regex` was evaluated and rejected
on *correctness*, not availability (it never enters an optional group, so
`(?:www\.)?` is skipped) — and timezone conversion needs a
tz database. **Temporal literals are the cheapest fix on this page and unblock
an entire query class.**

#### 1.7 Known-wrong answers in core operations

- **`GROUP BY` on a float column merges distinct keys.** A hash bug, not a
  grouping one — the float lane widens by value instead of by bit pattern, and
  the table buckets on the hash alone. **Fixed on `19a386e9`, which is not
  merged**, so it is still wrong on `main`.
- **Integer `//`, `%` and division by zero follow Python, not SQL.** Same
  branch, same status: rounding inside `core` and the zero-divisor null outside
  the lane, **not merged**. Float `/` by zero is not fixed even there — see
  §1.1.
- **Integer overflow wraps where SQL raises** (`golden/COVERAGE.md`). The
  `edges` fixture already carries int64 max/min for the day a checked-arithmetic
  mode exists.

---

### Tier 2 — competitive

Needed to be chosen over an incumbent, but a user will trial the library without
them.

#### 2.1 Parallelism above the kernel

**What exists.** Data parallelism *inside* kernels only —
`sync_parallelize`/`ctx.stripe` appear in `partition.mojo` (6), `views.mojo` (3),
`sort.mojo` (2), `join.mojo` (2), `filter.mojo` (1), and nowhere else.
**Group-by aggregation is serial**: `marrow/kernels/groupby.mojo` is 224 lines
holding one `HashGrouper` and one `HashGrouping`, and `_choose_strategy`,
`GROUP_RADIX` and `GROUP_THREAD_LOCAL` — once described as
shipped — return **zero grep hits in the tree**. They were removed in the
aggregate rearchitecture; the backlog is stale. There is no pipeline
parallelism: `Pipeline._flow` pushes one morsel through the stages on the
calling thread.

**What it would take.** Restoring thread-local partial aggregation is the
highest-value single item and it was previously built. True pipeline parallelism
is larger: the push `Operator` contract is a good foundation, but nothing owns a
task queue today.

#### 2.2 Larger-than-memory execution

**What exists.** Nothing. No spill, no memory pool, no accounting, no limit —
`grep -in 'spill\|memory_pool\|memory_limit'` over `marrow/` returns only
unrelated bitmap-test strings. `execute()` drains the entire plan into one
`RecordBatch` (`marrow/expr/logical.mojo:710`), so even a streaming plan
materializes its full result, and **there is no batch-iterator result API** even
though `drain()` is exactly that shape internally.

no limit at all. The old spilling streaming engine was removed and not
replaced.

Then spilling variants of group-by and sort.

#### 2.3 Relational operations that have no node

Each is a missing `Relation`, not a missing kernel. ibis's `relations.py` is the
canonical list; marrow has 8 of it.

| Missing | Golden cases | Note |
|---|---|---|
| `UNION ALL` / `UNION` / `EXCEPT` / `INTERSECT` | 4 | ibis models these as one `Set(left, right, distinct: bool)`. They also treat NULL as equal to itself, which nothing else in marrow does |
| `Distinct` / `.unique()` | 1 (`DISTINCT ON`) | Expressible today as `aggregate(keys=[...], aggs=[])` (`logical.mojo:938` accepts empty aggs), but there is no verb and no `unique` kernel |
| `GROUPING SETS` / `ROLLUP` / `CUBE` | 3 | `Aggregate` carries one key list; `ROLLUP` also needs `GROUPING()`. Implementable as a rewrite into an aggregation cascade |
| `explode` / `unnest` | 1 | Row-multiplying, so a new operator shape. ibis has a dedicated `TableUnnest` with `offset` and `keep_empty` |
| `Sample`, `DropNull(how)`, `FillNull` as relations | — | ibis has all three as nodes |
| `top_k` / `bottom_k` as first-class | — | A dedicated streaming node beats rewriting sort+limit; marrow has a *dead* `limit=` parameter (§2.10) |
| `merge_sorted`, `rolling`, `group_by_dynamic`, `upsample` | — | Time-series reshaping — a common ask in that domain |

#### 2.4 Join breadth

**What exists.** Hash equi-join in seven kinds — inner, left, right, full, semi,
anti, mark — over a Swiss table with a CSR probe index
(`marrow/kernels/join.mojo:69`, `hashtable.mojo:76-87`), with radix partitioning
and parallel probing. Multi-column keys work because keys go through
`StructArray`.

**Absent:** cross join, non-equi / inequality join, asof join, and — the
semantically dangerous one — an outer join with a residual non-key `ON`
predicate, which must be applied *before* null-widening
(`golden/cases/join_left_with_residual_condition.mojo`). There is no
nested-loop, merge or IE-join operator, so a non-equi join has **no fallback
path at all**: the query is simply unexpressible rather than slow.

A generic nested-loop join gives marrow exactly that degradation path and
turns cross and non-equi joins from *unexpressible* into merely slow, which is
a categorical improvement for a small amount of code.

#### 2.5 Aggregate breadth

13 golden cases, in three distinct kinds of gap:

- **Missing kernels:** median, quantile, mode, skewness, kurtosis, bitwise
  and/or/xor.
- **Missing nodes over kernels marrow already has:** `bool_and`/`bool_or` over
  `AnyKernel`/`AllKernel`.
- **Missing *shapes*:** `Aggregate[Agg, A]` binds exactly one operand, so
  `arg_min`/`arg_max`, `corr`/`covar`, `ORDER BY`-carrying `first`/`last`,
  `string_agg`/`array_agg`, multi-column `count(DISTINCT a, b)`, the
  `FILTER (WHERE ...)` clause and the `DISTINCT` modifier have nowhere to
  attach. **This is a node redesign, not kernel work**, and ibis shows how far
  it must go: every reduction there inherits `Filterable`, which supplies
  `where: Optional[Value[Boolean]]` (`ibis/expr/operations/reductions.py:27-29`),
  i.e. the FILTER clause is not a special case but a property of the base class.

#### 2.6 Nested-type and decimal operations

**Storage is complete; expressions cannot reach it.**

- **Nested:** eight golden cases — list element access, `list_contains`, list
  slice, `unnest`, list sum, struct field access, map lookup, map cardinality.
  `list_contains` was closed on 2026-08-31 (§1.5): `ArrayContainsKernel` now
  has `ArrayContains` in the comptime lane, an `array_contains` tag in the
  runtime one, and a verb reaching both, so the nested verbs are
  `array_length` and `array_contains`. Everything else needs a kernel that
  does not exist. ibis's minimum here is `ArrayIndex`, `ArraySlice`,
  `ArrayContains`, `ArrayLength`, `Unnest`, `MapGet`/`MapContains`/`MapKeys`/
  `MapValues`/`MapLength`, and `StructField` — and struct is genuinely thin
  there too (`structs.py` defines exactly two nodes), so `StructField` alone
  closes most of the struct gap.
- **Decimal:** `Column[T]` binds `T: NumericType` and `DecimalType` is a separate
  trait (`marrow/dtypes.mojo:160`), so **no decimal column can enter an
  expression at all**, despite `Decimal128Array`, all four decimal widths and
  every decimal cast kernel existing. Decimal arithmetic was never written. For a library aimed at analytics, "cannot compute on a
  money column" is close to disqualifying.

#### 2.7 No row format

It is the shared primitive behind fast multi-column sort, sort-merge join, and
hash group-by keys.

marrow instead does column-oriented LSD multi-key sort — one stable pass per
key, re-gathering each key column per pass — and
routes group-by/join keys through `StructArray` with per-column hashing. Both
work and neither is wrong, but this is the structural reason a future
sort-merge join has no cheap path and why multi-key sort re-gathers. Worth
naming as a design decision rather than discovering it under a benchmark.

#### 2.8 UDFs

**What exists.** Nothing. No `map_elements`, no `map_batches`, no `apply`, no
native UDF registration, no plugin surface.

**What it would take.** In the runtime lane, a UDF is a new `RuntimeValue` tag
holding a callable — tractable. **In the comptime lane a native UDF is close
to free and is where marrow should be strongest**: a Mojo function is already
a comptime value, and a user-supplied `lane[W]` would fuse into the same loop
as the built-ins with no boundary and no dynamic library. This is a
differentiator hiding inside a table-stakes item.

#### 2.9 Interop and format gaps

- **Compressed Arrow IPC bodies are unsupported.** `marrow/ipc.mojo:1409`
  raises on LZ4_FRAME/ZSTD bodies. Marked *unverified* as to how much it would
  buy on marrow's reader. - **No SQL frontend.** No parser anywhere. The
  golden corpus is *written in SQL* and translated to marrow by hand, which is
  itself evidence of the impedance. - **No `__dataframe__` protocol**, though
  the PyCapsule/C Stream path marrow already has is the better-supported
  modern route.

#### 2.10 Operability

- **No `explain()` verb**, though plans render recursively through `Writable`
  (`Filter.write_to`, `logical.mojo:816`), so the string exists and only needs
  a name. `README.md:238`'s "explain() renders one node" is stale. - **No
  `EXPLAIN ANALYZE`, no per-operator metrics, no profiling hook, no progress,
  no cancellation.** An operator cannot be interrupted mid-`drain`. - **No
  error taxonomy.** 244 `raise Error(...)` sites across `marrow/` produce
  plain strings with no type. The messages themselves are good — they name the
  verb and the column (`"drop: column 'x' not found in schema"`) — but a
  Python frontend cannot map them to distinct exception classes. That is cheap
  to copy and should land with the frontend, not after it. - **Top-K is a dead
  parameter.** `sort_indices(..., limit=)` is passed non-`None` at exactly two
  sites, both tests; `SortOperator` never passes it, so
  `sort_by(...).limit(k)` performs a full sort and discards. Wiring it needs a
  `row_limit` channel through `Pushdown` *and* a per-node rule table, because
  `Limit(Filter(Sort(x)))` may not take the top K — the filter runs after the
  sort, so a K-row sort silently returns fewer than K rows. - **Per-key null
  placement is missing on sort:** `Sort` carries one `nulls_first: Bool` for
  all keys (`logical.mojo:1066`); ibis's `SortKey` carries it per key.

---

### Tier 3 — differentiating

Where marrow could be better than anything that exists.

#### 3.1 The comptime / AOT lane — the real one

**What it is.** In the comptime lane a node's operands are bound on a family
trait, its output dtype is a comptime type, and a whole subtree fuses into one
SIMD loop with nothing erased. `col("a", int64).sum()` resolves to
`Aggregate[Fold[SumFold, Int64Type], Column[Int64Type]]`, so the plan holds a
direct `AggState[SumFold, Int64Type]` and no per-dtype resolution ladder is
reachable in the binary at all.

**The measurement** (`benchmarks/binary_size/baseline.json`, `-O3 -g0`,
stripped, `__text`):

| Gate | Bytes | |
|---|---:|---|
| `query_streaming_agg_fused` (comptime) | 1,481,012 | |
| `query_streaming_agg` (runtime-named) | 9,940,868 | **6.71x** |
| `query_dynvalue` (erased values) | 6,227,524 | |
| `query_streaming` (fused filter + project floor) | 1,483,336 | |

The runtime lane's cost is not incidental: it links the whole name-resolution
ladder and, through it, `marrow.kernels.cast` — 693 cast symbols in
`query_dynvalue` alone.

**The second half nobody else has.**
`marrow/expr/comptime/tests/test_schema_handle.mojo` pins four compiler
contracts and proves that a schema can be a comptime parameter and that
`__getattr_param__` can return a *conditional* type carrying its trait bound.
So `t.amount` resolves to `Column[Float64Type]`, `t.qty` to
`Column[Int64Type]`, and `t.amont` is a compile error reading `constraint
failed: unknown column: amont`. None of them can do it at build time, because
none of them has a compile step to do it in.

**Is it a product advantage, and for whom?** Yes, and for a market nobody
serves:

- Fixed, known queries shipped into constrained targets — edge devices,
  embedded analytics, on-device telemetry rollups, per-tenant compiled
  reports. - Data-plane filters and ETL steps where the query is code, is
  reviewed, and never changes at run time. - Anywhere a wrong column name
  should fail in CI rather than at 3 a.m.

Which is exactly why the runtime lane exists and why the two lanes must stay
at parity — a point the project already holds as an architectural invariant
("one engine, two drivers") but currently does not
enforce, since `test_parity.mojo` was deleted with the old package and has no
replacement.

**What it would take to be a product.** The engine is done; the wrapper is not.
`execute_cli()` does not exist, nor the generated `--help`/`--describe` surface,
nor the Parquet/Arrow output writers — all lived in the deleted package
(`python/marrow/compile.py:9`, `marrow/utils/argparse.mojo:39`).
`python/marrow/compile.py` still builds a `.mojo` file and still passes
`-D MARROW_CLI_WRITERS=true`, a define that currently gates nothing. Restoring
that tail and promoting the schema handle from spike to public API is a small,
well-scoped amount of work for the only story here that is genuinely
unavailable elsewhere.

**Honest counterweights.** 1.48 MB of `__text` is small for a query engine and
not small in absolute terms; the binary still links `libmax`/AsyncRT with GPU
codegen off. The fused lane requires the schema at compile time, which most
workloads do not have. And the 6.71x is measured on one query shape on
osx-arm64 — a broader sweep across query shapes is *unverified*.

#### 3.2 One kernel, two targets

`apply` writes a single lane and dispatches it to CPU stripes, CPU serial, or a
GPU `elementwise` launch (`marrow/views.mojo`), with `Buffer`/`Array` carrying
explicit `to_device`/`to_cpu` and device-resident results. No CPU dataframe
library has this; the GPU dataframe libraries are not CPU libraries.

Today this is a research capability, not a product: transfer cost dominates, the
measured crossover was ~10K vectors at dim ≥ 384, and the expression layer does
not plan device placement. But "the same kernel source runs on both, and the
plan decides" is a defensible long-term position that Rust and C++ engines
cannot copy cheaply.

#### 3.3 Correctness discipline as a feature

Archery conformance against three other Arrow implementations.

This is not a user-visible feature on its own.

---

## 3. Unexplored — `warp.match_any()` for GPU hash join and group-by

### Correction to the original note

The first pass at this idea (written right after the b3 changelog scan)
claimed marrow's hash-join/group-by kernels "already run GPU probe/build
paths." That's wrong — checked more carefully:

- `marrow/kernels/hashtable.mojo` (`SwissHashTable`, the SIMD-group Swiss
  table every join/groupby is built on) imports `std.gpu.host.DeviceContext`
  but never calls anything from it — `grep -c DeviceContable hashtable.mojo`
  shows exactly one match, the import line itself. Dead import.
- `marrow/kernels/join.mojo`'s `ctx` parameter is always
  `ExecutionContext.parallel(self._num_threads)` — CPU multi-threading, not
  GPU. Its own docstring says the GPU context slot is a "future GPU
  acceleration hook" — i.e. not implemented.
- `marrow/kernels/aggregate.mojo`'s only GPU involvement is delegating
  simple reductions to `views.reduce` (sum/min/max over a single array) —
  unrelated to hash-based grouping.

So: **there is no GPU hash join or GPU group-by today.** `warp.match_any()`
/ `warp.match_all()` (new in b3 — portable same-value lane masks: NVIDIA
`match.any.sync`, AMD ballot fold, Apple shuffle emulation) can't be slotted
into an existing kernel. This note is about whether a GPU hash-join/group-by
would be worth building *and* would want these intrinsics — two decisions,
not one.

### What exists today (the thing a GPU port would replace/parallel, not patch)

`SwissHashTable` in `hashtable.mojo` is a from-scratch Swiss-table
implementation, CPU-only, SIMD-group matching with pipelined probing:

```
Hash Function  →  Partitioner  →  SwissHashTable  →  Operator (join / groupby)
```

Entry points: `insert_hashes`, `build_hashes`, `probe_hashes`, plus
`insert`/`build`/`probe` wrappers. `RadixPartitioner` splits rows across
partitions by hash before they reach the table, presumably to bound
per-partition working-set size for cache locality — the same reason a GPU
version would want partitioning too, probably per-threadblock rather than
per-CPU-core.

### Where `match_any`/`match_all` would actually help, if this gets built

The plausible use is inside a GPU probe kernel: when a warp of threads is
probing the same hash bucket (or a set of buckets that collide into the
same warp), `match_any()` gives you — for free, without shared-memory
traffic — a mask of which lanes in the warp are looking at equal keys. That
can shortcut redundant global-memory key comparisons when many probe rows
in a warp share a key (skewed join keys, or a `GROUP BY` on a
low-cardinality column). This is a real, well-known GPU hash-table
optimization pattern in principle — I have not verified it against any
GPU-Swiss-table reference implementation, and marrow's CPU Swiss table's
specific probing sequence (SIMD group matching) may or may not map cleanly
onto a warp-level equivalent.

### What the actual spike is

Not "add match_any to hashtable.mojo" — it's:

1. Decide whether a GPU hash-join/group-by path is worth building at all
   for marrow's workloads before anything else. This is a much bigger
   design question than the language-feature note it started as — probably
   deserves its own design doc rather than a todo item, if the answer is yes.
2. If yes: prototype a minimal GPU probe kernel (even a toy one, independent
   of `SwissHashTable`) using `warp.match_any()` for intra-warp key dedup,
   and benchmark it against the existing CPU `SwissHashTable::probe_hashes`
   on a skewed-key workload, since that's the specific case where this
   would pay off — a uniform-key workload probably won't show a difference.
3. Only then decide whether it's worth integrating into
   `kernels/join.mojo` / `kernels/hashtable.mojo` for real, following
   whatever GPU dispatch convention `views.reduce`/`views.apply` already
   established (`ExecutionContext.gpu(ctx)`, `has_accelerator_support[...]`
   gating, etc. — see `marrow/views.mojo`).

### Status

Speculative, two levels removed from "ready to prototype." The precondition
(GPU hash join existing) isn't met yet — resolve that design question
first, independently of whether `match_any`/`match_all` end up being useful
inside it.

---
