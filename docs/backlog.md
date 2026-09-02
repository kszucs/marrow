# marrow — open work

**The only document that says what is open.** Everything under `docs/*.md` and
`docs/superpowers/` was consolidated here on 2026-08-30 and the source files
deleted: 33 design documents, plans, audits and research notes, 15,694 lines.
They were audited file by file against the tree first — what had shipped, what
was false, what was superseded — and only two categories survived into this
file:

1. **work someone may still choose to do**, and
2. **constraints that bind that work** — measured dead ends, compiler limits and
   traps, each of which cost real time to find and each of which invalidates an
   approach that looks obvious.

Nothing here describes something that is already built. For what marrow *does*,
read the code, `CLAUDE.md`, `golden/COVERAGE.md` (the best statement of engine
capability) and `CHANGELOG.md`. For why a rejected design was rejected, see
§4 — that section exists so the same ideas stop being re-proposed.

The deleted set is recoverable in full: `git show 8f365d14:docs/` lists it, and
`git log --diff-filter=D --name-only -- docs/` finds any single file.

---

## What is missing, in priority order

One table for the whole backlog. **Priority** is what a first user hits soonest;
**complexity** is S (a day), M (a week), L (a couple of weeks), XL (a project
with its own design). "Blocked by" names a thing that must land first, not a
preference.

Detail for every row is in the tiers below; the standing constraints in §3 gate
all of them.

| # | Missing | Why it matters | Cx | Blocked by |
|---|---|---|---|---|
| 1 | **Two wrong answers**: float `GROUP BY` keys collapse; integer `//`, `%` and `/0` follow Python, not SQL | Wrong results beat every other consideration. Three strict xfails and one golden case already pin them | **S** | — |
| 2 | **CI is dark** — no run since 2026-05-11, `test.yml` calls a deleted task, `binary_size` not wired in at all | Everything below is unverifiable in CI. A +55% size regression once survived ten commits for exactly this reason | **S** | — |
| 3 | **CSV reader**, then NDJSON | A first user arrives with a CSV, not a Parquet file. `find marrow -iname '*csv*'` is empty | **M** | — |
| 4 | **Error taxonomy** — 269 `raise Error` sites, zero typed exceptions | Cheap while the Python boundary is fresh, expensive to retrofit across 269 sites. Already a retrofit | **M** | — |
| 5 | **`scan(path)` without a hand-written schema**, then globs, directories, hive partitions | `scan()` takes one path *and* demands the schema by hand. Every real Parquet dataset is a directory | **M** | 3 |
| 6 | **Parallel group-by** — thread-local partials plus a radix merge | Group-by is where analytical queries spend their time; single-threaded loses every benchmark. `groupby.mojo` is 224 lines with no parallelism | **M** | — |
| 7 | **`distinct`, `union`, `except`, `intersect`** — no node exists for any of them | Table stakes for a SQL-shaped frontend, and `ReplaceDistinctWithAggregate` is a rule nobody can write without the node | **M** | — |
| 8 | **String and temporal surface** — 16 + 13 skipped golden cases | The long tail a dataframe user hits immediately after the basics work | **M** | — |
| 9 | ~~**Cheap expression nodes over kernels that already exist**~~ — done 2026-08-31 | Nine kernels had no node. `is_in`, `array_contains`, `minimum`/`maximum` and six float unaries now reach both lanes; see §1.5 for what is left and why it is not cheap | **S** | — |
| 10 | **Join output ordering** — `JoinOperator` hardcodes build=left, `_output_schema` is positional | Blocks *both* remaining optimizer rules. Not an optimizer change: the kernel must accept an output ordering | **M** | — |
| 11 | **Join reordering + build-side selection** | The largest TPC-H win available, and the only genuinely cost-based pass in any incumbent | **L** | 10, 12 |
| 12 | **Statistics propagation and a cost model** | Feeds 11. Note two of three incumbents ship without one — DataFusion and polars both have none | **L** | — |
| 13 | **CSE and duplicate group/sort key elimination** | Both need `DynValue` equality — likely solvable at the verb, as `constant_bool` and `conjuncts` were, rather than with a box slot | **M** | — |
| 14 | **Window functions** — no node, no operator, no ranking kernel | Seven golden cases recorded unsupported. A whole subsystem, not an increment | **L** | — |
| 15 | **Larger-than-memory execution** — no spilling anywhere | Every aggregate and join is bounded by RAM. Changes the operator contract | **XL** | — |
| 16 | **Nested-loop / range joins** | Only equijoins exist, so a non-equi predicate has no plan at all | **M** | — |
| 17 | **UDFs** | The escape hatch that makes a missing kernel survivable rather than fatal | **M** | 4 |
| 18 | **A row format** | Needed by sort-merge join, spilling, and any wire protocol | **L** | — |
| 19 | **AOT lane product** — no CLI, no entry point, no output writers | The one thing nobody else offers, with no way to use it. `execute_cli()` does not exist | **M** | — |
| 20 | **Seven files over 1,000 lines** | `logical.mojo` at 1,672. Seams identified; deliberately deferred so churn does not swamp behaviour review | **M** | — |

**Not on this list, deliberately.** Limit pushdown into the scan — early
termination already achieves it at row-group granularity. Merging stacked
filters — the optimizer now deliberately *splits* them so each conjunct moves
independently. A `Distinct`-specific operator — it is an `Aggregate` with no
aggregates once the node exists.

---

## 1. Open work

Ordered by value. Nothing below is started.

### 1.1 Correctness — known-wrong answers

**Integer `//` and `%` follow Python, not SQL.** `FloordivKernel` and
`ModKernel` forward Mojo's operators: Python floors and takes the *divisor's*
sign, SQL truncates toward zero and takes the *dividend's*. So `-1 // 3` is `0`
in SQL and `-1` here, and `-1 % 3` is `-1` there and `2` here. Division by zero
returns a number where SQL returns NULL — the kernels substitute 1 for a zero
divisor inside a SIMD lane that can neither raise nor null.

Recorded as three strict xfails: `golden/cases/math_floordiv_truncates_toward_zero.mojo`,
`math_modulo_sign_follows_dividend.mojo`, `math_integer_division_by_zero.mojo`.
A fix needs a sign correction **and** a validity mask derived from the divisor.
It was invisible because `math_mod_int64` asks `((n % 3) + 3) % 3` — the one
form both rules agree on.

**`count(*)` desugars to `count(lit(1))`.** `builders.mojo` returns
`lit(1, int64).count().alias("count_star")`, whose `columns()` is empty — so
anything that prunes by "what does this read" prunes every column, and
`RecordBatch.num_rows()` returns **0** when there are none (`tabular.mojo`).

**Projection pushdown landed 2026-08-31 and handles it rather than fixing it:**
`ColumnPruning` never narrows a source to zero columns, keeping the first when
the demand is empty. The desugaring is unchanged, so the trap still waits for
the next thing that reads `columns()` and believes the answer.

### 1.2 The published guides document APIs that do not exist

The only user-visible item here. `docs/guide/expressions.qmd` documents
`from marrow.expr import col, lit, parquet_scan` — there is no such Python
module — and calls the engine "pull-based" when `physical.mojo` is push/drain.
`docs/guide/compile.qmd` documents `plan.execute_cli()`, deleted with the CLI
entry point. These are on the rendered site.

**`README.md` came off this list on 2026-08-30**, when the Python query API was
restored: its lazy-query section had described `morsel_size` and `strictness`
arguments that no longer exist, claimed `explain()` "renders one node, not the
tree" when plans render recursively, cited two deleted docs, and carried a
40/43 ClickBench claim whose harness went with them. The guides are still
wrong, and `docs/guide/expressions.qmd`'s import line is now *nearly* right —
the names are `marrow.col` / `marrow.lit` / `marrow.read_parquet`, not
`marrow.expr.*`.

### 1.3 Latent compiler hazards

**~15 more `t"…{dtype}"` sites under `marrow/kernels/`.** A t-string
interpolating a recursive `Writable` value inside a function-level recursion
cycle deadlocks the compiler — see §4 for the full finding. Three sites were
fixed; the rest are untested and sit in the same shape.

**File the deadlock upstream.** `docs/repros/tstring_recursive_writable_cycle.mojo`
is a 55-line reproducer with no marrow import for a compiler hang that emits no
diagnostic. Modular should have it.

### 1.4 Engine capability the golden corpus measures as missing

`golden/COVERAGE.md` is authoritative and machine-checked: 278 cases, of which
**85 carry `-- skip mojo`** because marrow has no API for them. Their bodies are
never compiled, so they are proposals rather than verified spellings. In rough
order of how often a real query wants them: set operations (UNION/EXCEPT/
INTERSECT), window functions (7 cases), 13 aggregates (median, quantile,
first/last, arg_min/arg_max, string_agg, bool_and/or, corr/covar, mode,
skewness), 16 string functions (substr, replace, regexp, lpad, position,
split_part, concat), 13 temporal (date_diff, age, strftime/strptime, last_day,
make_date, epoch, iso_week), nested types (struct field, map lookup, list
element/slice/contains, unnest), decimals, `DISTINCT ON`, GROUPING SETS/ROLLUP/
CUBE, and cross/non-equi/asof joins.

Two of these came off the list on 2026-08-31: `list_contains` and `IN` now
have nodes and verbs in both lanes (§1.5). `bool_and`/`bool_or` over
`AnyKernel`/`AllKernel` is the one that remains, and it is **not** the same
size of job — those two are `BoolReduceKernel`s rather than `FoldKernel`s and
have no grouped variant, so they need an aggregate node in both lanes plus a
kernel change.

### 1.5 ~~One kernel reachable from no expression node~~ — done 2026-08-31

The entry named one kernel. Grepping every public kernel in `marrow/kernels/`
against every import under `marrow/expr/` found **nine**, and all nine now
have a node and a verb in both lanes:

- `ArrayContainsKernel` — `ArrayContains` (`comptime/leaves.mojo`), the
  `array_contains` tag, `builders.array_contains`.
- `IsInKernel` — it had a runtime `isin` tag but no comptime node and no verb,
  so `IN (...)` could not be written from the one surface that spans both
  lanes. Now `IsIn` (`comptime/boolean.mojo`) and `builders.is_in`.
- `MinKernel` / `MaxKernel` — dead in both lanes. Now `Minimum` /
  `Maximum` and `builders.minimum` / `.maximum`.
  **Deliberately not spelled `least` / `greatest`**: those are SQL's names and
  SQL *skips* nulls (`LEAST(NULL, 3)` is 3), while `MinKernel` intersects
  validity like every other `BinaryNumericKernel`. Naming them for SQL would
  have put a wrong answer behind the name;
  `golden/cases/math_greatest_and_least.mojo` pins the skipping semantics and
  stays skipped, since providing them is a kernel change. `.min()` / `.max()`
  were unavailable anyway — those are the aggregates.
- `Exp2Kernel`, `Log2Kernel`, `Log10Kernel`, `Log1pKernel`, `SinKernel`,
  `CosKernel` — six `UnaryFloatKernel`s where `NumericValue` named three. Now
  `.exp2()` … `.cos()` and the matching runtime tags.

`array_length` was not on the list and should have been. It had a verb in each
lane, but its **overload set was split** across `builders.mojo` and
`runtime/values.mojo` — the failure `builders.mojo`'s own docstring warns
about — so `from marrow.expr import array_length` was comptime-only. The
runtime overload now sits beside the comptime one.

`ConcatKernel` (`kernels/string.mojo`) came off this list on 2026-08-30: the
runtime lane's `add` tag dispatches to it when the operands turn out to be
strings, which is the call its own docstring already said it existed for. The
comptime lane still has no `+` on `StringValue` -- a typed string column
concatenates only by leaving the lane.

**What is still unreachable, and why none of it is one alias away:**

- `AnyKernel` / `AllKernel` (`kernels/aggregate.mojo`) are `BoolReduceKernel`s,
  deliberately *not* `FoldKernel`s: they fold bit-packed masks and expose only
  `reduce(BoolArray) -> Bool`. Reaching them needs an aggregate node in both
  lanes **and** a grouped variant the kernel does not have — a subsystem, not a
  node. This is the one genuine coverage gap left here.
- `StringToBoolKernel` / `BoolToStringKernel` would be two more `casts.mojo`
  breakers. Both are already reachable through the runtime `cast()` verb, so
  what is missing is fusion, not coverage. The same holds for the nine
  remaining `cast.mojo` kernels (temporal, binary-like, fixed-size-binary,
  null, list, struct, dictionary) and all six of `cast_decimal.mojo` — with the
  extra note that no decimal column can enter the comptime lane at all, since
  `Column[T]` binds `T: NumericType` (§2.6).
- `drop_null`, `sort`, `filter` and `take` are **relational**: each needs a
  `Relation` node, not an expression one. `filter`/`take`/`sort_indices` are
  already reached from `physical.mojo`.
- `hashing.mojo`, `hashtable.mojo`, `partition.mojo` and `distinct.mojo`'s
  public defs are building blocks other kernels consume, not user-facing
  compute. `distinct` is reached through `aggregate.DistinctCount`.

**Of six skipped golden cases, exactly one was unblocked.** Checked case by
case, not by name:

- `math_log_bases` — **unblocked and un-skipped.** It needs only `.log2()` and
  `.log10()`, both of which are now methods on `NumericValue`.
- `math_trigonometry` — **still blocked.** It uses `sin`, `cos` *and*
  `atan2(y, n)`, and there is no `atan2` kernel. Two of three verbs is not
  enough to run a case.
- `nested_list_contains` — **still blocked.** It calls
  `col("l", list_(int64)).contains(...)`, a *method* on `ListValue`. Only the
  free `array_contains(list, elem)` was added; `.contains` exists solely on
  `StringValue`.
- `filter_in_literal_list` — **still blocked.** It calls
  `col("v", int64).is_in([1, 3, 99])`, a method taking a Mojo list. Only the
  free `is_in(a, DynArray)` exists.
- `filter_not_in_list_with_null` — **blocked on semantics, a second exception
  the previous version of this entry did not name.** It expects SQL's `NOT IN`
  with a NULL in the list to match *nothing*; marrow's `is_in` follows
  PyArrow's `MATCH`, so `~is_in(v, {1, NULL})` returns 5 rows where the case
  expects 0.
- `math_greatest_and_least` — **blocked on semantics.** Its expected rows
  encode SQL's null-skipping, which `MinKernel` / `MaxKernel` do not do; it
  needs a `skip_nulls=True` kernel first.

So three of the six need *methods* rather than free verbs (`.contains`,
`.is_in`) or a kernel that does not exist (`atan2`), and two need null
semantics marrow does not implement.

### 1.6 ~~Top-K is a dead parameter~~ — done 2026-08-31

`TopN` in `marrow/expr/optimizer.mojo` rewrites `Limit(Sort(x))` to a bounded
sort, and `SortOperator` now passes `sort_indices(limit=)`. Two details the
original entry got right and one it did not:

- The `Limit(Filter(Sort(x)))` hazard is real, and the rule requires the
  `Limit` to be **directly** above the `Sort` — which makes it unrepresentable
  rather than merely avoided. `PushFilterBelowSort` runs first and moves the
  common offender out of the way, so adjacency is less restrictive than it
  sounds.
- The bound is `offset + length`, not `length`: the `Limit` still skips
  `offset` rows of the ordered result.
- The bound applies to the **primary-key pass only**. Multi-key sort composes
  stable passes from least to most significant key, so truncating an earlier
  pass drops rows the later keys still have to order. The original entry did
  not mention this and it is the one way to get `TopN` silently wrong.

It did **not** need "a `row_limit` channel through `Pushdown` and a per-node
rule table" — that was the shape the demand-lattice design would have required.
It is a field on `Sort` and one rule.

### 1.7 Structure

**Seven files over 1,000 lines** (`expr/optimizer.mojo` joined at ~1,100, and
is the one file that should stay whole — it exists so every rule is readable in
one place), with seams identified: `kernels/aggregate.mojo`
(1,573 — fold algebras | `AggState`, whose storage/driver/policy should split
three ways | the `AggKernel` conformers), `expr/logical.mojo` (**1,672** — the value
layer | `Relation`+`DynRelation`+the fluent verbs | the eight relation nodes),
`kernels/cast.mojo` (1,246), `expr/physical.mojo` (1,281 — wire types |
contract | `Pipeline` | the operators), `comptime/core.mojo` (1,212 — machinery
| the family traits, ~85% fluent surface), `kernels/filter.mojo` (1,163 — two
structs of ~520 lines each; split by layout, not by struct).

Deliberately not done: the churn would swamp review of the behaviour changes it
would ride alongside. `Groups` and the decimal casts were split out because
they had hard internal boundaries.

**`DynBounds`/`compare_dyn`/`PruneStats.dyn_bounds` stay in `pruning.mojo`.**
Moving them to `runtime/` — their only consumer — means exposing
`PruneStats._cols` or adding an accessor for one caller. `PruneStats` hosts
three readings of itself; moving one and leaving two is arbitrary, and trading
encapsulation for a file boundary is not a simplification.

### 1.8 Test and infrastructure gaps

- **No cross-lane parity test.** "One engine, two drivers" was enforced by a
  `test_parity.mojo` across four axes; it went with the previous expression
  package and has no replacement. The invariant is currently unenforced.
- **`HashGrouper` has no dedicated test** — `kernels/tests/test_groupby.mojo`
  was deleted without replacement.
- **CI has not run since 2026-05-11.** `test.yml` calls a deleted task, the docs
  job cannot pass, and the binary-size gate is not in CI at all — which is how a
  +55% size regression once survived ten commits.

### 1.9 Measurement trap: `==` on an erased array

`DynArray.__eq__` routes through `ArrayData.__eq__`, which compares the
**layout** — dtype, length, null count, *offset*, validity, whole buffers,
children — and says so in its own docstring: "two layouts holding the same
values at different offsets are not equal here". Two consequences bite in
tests, and both cost time on 2026-08-30:

- **A filtered result over-allocates its buffer**, so comparing one to a fresh
  `array([...])` of the same values answers False. `PrimitiveArray[T].__eq__`
  loops over the valid elements and is immune; the erased one is not. Assert
  element by element, or compare typed arrays.
- **A value under a null is unspecified.** A kernel computes every lane and
  masks afterwards, so `add(a, b)` holds `a[i] + b[i]` under a null where a
  builder writes 0. Whole-array `==` sees the difference.

CLAUDE.md's "prefer `assert_true(result == expected)`" is about
`PrimitiveArray[T].__eq__` and holds there. It does not transfer to `DynArray`.

**Hit again on 2026-08-31, and the interesting half is which case *passed*.**
`least(a, b)` and `greatest(a, b)` over `a = [1, 2, null, 4]`,
`b = [10, 20, 30, 40]` were both compared erased. `greatest` failed, because
the byte under the null is `max(0, 30)` where the builder wrote 0. `least`
**passed**, because `min(0, 30)` happens to be 0. So a green erased comparison
is not evidence that erased comparison is safe here — it can be one operand
value away from failing, and a suite can carry the defect silently. Narrow to
the typed array.

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

## 2. Missing capabilities, in detail

Carried from `CAPABILITY-GAPS.md`, which this file absorbed on 2026-08-31 so
that one document says what is open. The tiering is by *user impact*, and it
cuts across the priority table above: a Tier 1 item can be cheap and a Tier 2
item can be the largest thing here.

Each section states what exists, what is absent, and what the incumbents do —
that last part is the most perishable and the most useful, because it is what
stops a gap being estimated from first principles.

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

**What the incumbents do.** ibis exists *only* as a frontend — 20 backends,
`pandas` and `dask` since removed, and its whole value is the API surface. That
is the clearest possible evidence that the frontend is the product and the
engine is the commodity. polars, DuckDB and DataFusion are all primarily
consumed from Python.

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

**What the incumbents do.** All of them read CSV and NDJSON with schema
inference. arrow-rs ships them as first-class crates with sampling-based
inference — `arrow-csv/src/reader/mod.rs:360` (`Format::infer_schema(reader,
max_records)`, plus `infer_schema_from_files` at :461, with a `Format` struct
carrying delimiter/quote/escape/null-regex) and
`arrow-json/src/reader/schema.rs:155,203,424`. DuckDB has an entire scanner
subsystem with a dialect sniffer, a state-machine parser and a rejects table
(`duckdb/src/execution/operator/csv_scanner/`), and exposes inference as a table
function (`sniff_csv.cpp`). polars is the best guide to the *option surface* a
user expects (`crates/polars-io/src/csv/read/options.rs`):
`infer_schema_length` (default 100 rows, `None` = whole file), `schema`,
`schema_overwrite` by name, `dtype_overwrite` positionally, `NullValues::{
AllColumnsSingle, AllColumns, Named}`, `ignore_errors`, `try_parse_dates`,
`decimal_comma`, `comment_prefix`, `truncate_ragged_lines`, `skip_rows` /
`skip_lines` / `skip_rows_after_header`. Note also that polars ships **both**
eager `read_csv` and lazy `scan_csv` for CSV, IPC, Parquet and NDJSON, but
whole-document JSON is eager-only — a reasonable scope line to copy.

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

**What the incumbents do.** DuckDB: `src/common/multi_file/` (list, reader,
column mapper at 49 KB, `union_by_name`), `src/common/hive_partitioning.cpp`,
`src/function/table/glob.cpp`, and partitioned writes with `PARTITION_BY`,
file-size rotation and filename patterns in `physical_copy_to_file.cpp`.
DataFusion: `catalog-listing/src/{helpers.rs,table.rs}` (`ListingTable`,
partition pruning from paths), `execution/src/object_store.rs`
(`ObjectStoreRegistry` over S3/GCS/Azure/HTTP), and hive-style write demux in
`datasource/src/write/demux.rs`. polars goes furthest: one generic multi-file
scan engine reused by every format
(`crates/polars-stream/src/nodes/io_sources/multi_scan/`) with a shared
`ScanOptions` surface — `row_index`, `pre_slice`, `hive_partitioning`,
`hive_schema`, `missing_columns={insert,raise}`, `extra_columns={ignore,raise}`,
`include_file_paths`, `deletion_files`, `table_statistics`
(`py-polars/src/polars/io/scan_options/_options.py`) — a cloud module
dispatching AWS/GCP/Azure/HTTP/HuggingFace with credential providers and an
on-disk remote-file cache (`crates/polars-io/src/{cloud,file_cache}/`), and
partitioned writes via `pl.PartitionBy(base_path, key=..., max_rows_per_file=...)`
accepted by every `sink_*`.

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

**What the incumbents do.** DuckDB's pipeline is ~45 individually-disable-able
passes (`src/optimizer/optimizer.cpp:171-445`), including `EXPRESSION_REWRITER`
(26 sub-rules), `FILTER_PUSHDOWN` (15 operator-specific handlers),
`UNUSED_COLUMNS` (projection pushdown, 48.5 KB), `COMMON_SUBEXPRESSIONS`,
`TOP_N`, `LIMIT_PUSHDOWN`, `LATE_MATERIALIZATION`, `STATISTICS_PROPAGATION`, and
`JOIN_FILTER_PUSHDOWN` (dynamic runtime min-max filters into scans). DataFusion
has 25 logical rules including `push_down_filter` (154 KB),
`optimize_projections`, `common_subexpr_eliminate`, `push_down_limit` and
`simplify_expressions`, plus 21 physical rules. polars runs 18
(`crates/polars-plan/src/plans/optimizer/mod.rs`): type coercion, fused
arithmetic, common *subplan* elimination, slice pushdown (plan and expression),
predicate pushdown — whose `join.rs` also collapses cross-join+filter into an
inner or IE join — projection pushdown, **fast count-star** (`select(len())`
over a scan becomes a metadata read, `optimizer/count_star.rs`), simplify
expressions, cluster `with_columns`, common *subexpression* elimination,
order-observation analysis (unset `maintain_order` where order is provably
unobserved), sortedness propagation, and dataset expansion *after* pushdown so
pushed predicates prune the file list.

**Useful nuance: do *not* prioritize join reordering or a cost model.** It is
the only genuinely cost-based piece in any of the three, and **two of the three
do not have it.** DuckDB does — DPccp/DPhyp enumeration with a greedy fallback
and HLL-based cardinality estimation
(`src/optimizer/join_order/plan_enumerator.cpp:234`,
`cardinality_estimator.cpp`). DataFusion has none, only build-side and
implementation selection in `join_selection.rs`. polars has none either, and
this is worth knowing precisely: its `OptFlags::ROW_ESTIMATE` is **dead code**
(set at `crates/polars-lazy/src/frame/mod.rs:201`, read nowhere), and
`FileInfo.row_estimation` is consumed only to print `ESTIMATED ROWS` in
`explain()`. What polars' own docs call "join ordering" and "cardinality
estimation" are *runtime-adaptive* decisions made with HyperLogLog sketches
during execution (`polars-stream/src/nodes/joins/equi_join.rs:204-350`,
`nodes/group_by.rs`), not planning passes. A credible engine ships without a
cost model.

**A cheap pass worth copying early:** polars' fast count-star. It is also the
mirror image of marrow's `count_star()` defect below — the same expression that
blocks projection pushdown is the one an optimizer most wants to special-case.
(marrow's projection pushdown now clamps rather than special-cases: it never
prunes a source to zero columns. Fast count-star remains uncopied.)

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

#### 1.5 Window functions

**What exists.** Nothing — no window node in `logical.mojo`, no windowed
operator in `physical.mojo`, no ranking or offset kernel. Seven golden cases are
recorded unsupported: `window_row_number`, `window_rank_and_dense_rank`,
`window_lag_and_lead`, `window_partitioned_running_sum`,
`window_explicit_rows_frame`, `window_first_and_last_value`, `window_qualify`.

**What the incumbents do.** ibis's `analytic.py` is the canonical minimum:
`MinRank`, `DenseRank`, `RowNumber`, `PercentRank`, `CumeDist`, `NTile`, `Lag`,
`Lead`, `NthValue` — plus, critically, **no separate cumulative nodes**: any
reduction under a `WindowFunction` with an unbounded-preceding frame *is* a
running aggregate (`ibis/expr/operations/window.py:29-120`). Frames are
`(how: rows|range, start, end, group_by, order_by)` with `None` for unbounded.
That design means marrow gets running sums for free once frames exist.

**What it would take.** A `Window` relation node, a blocking `WindowOperator`
that partitions and sorts (both kernels exist), and a frame evaluator. The rank
family and `lag`/`lead` need no new kernels. The golden cases already state the
semantics, including the `last_value` default-frame trap.

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

**Calibration.** arrow-rs deliberately stops where marrow does — it has
`substring`, `concat_elements`, `like`, `regexp`, `length`, and *no*
trim/pad/case/replace/split kernels, because those are engine-level, not
Arrow-level. ibis's `strings.py` is the engine-level expectation: case,
trim/pad, substring/slice, find/predicate, pattern match, regex (extract/split/
replace), replace/split/join, and URL parsing.

**What it would take.** Most are ordinary kernels over machinery that exists.
Two are not: regex needs a real engine — `mojo-regex` was evaluated and rejected
on *correctness*, not availability (it never enters an optional group, so
`(?:www\.)?` is skipped; `docs/backlog.md` §4) — and timezone conversion needs a
tz database. **Temporal literals are the cheapest fix on this page and unblock
an entire query class.**

#### 1.7 Two known-wrong answers in core operations

- **`GROUP BY` on a float column merges distinct keys.**
  `golden/cases/group_by_float_key.mojo` records `-1.25` and `0.5` collapsing
  into one group where DuckDB returns three distinct float groups. This is a
  wrong-answer bug in a primary relational operator, not an edge case.
- **Integer `//`, `%` and division by zero follow Python, not SQL.** `-1 // 3`
  is `-1` here and `0` in SQL; `-1 % 3` is `2` here and `-1` in SQL; `10 // 0`
  returns `10` because the kernels substitute 1 for a zero divisor inside a SIMD
  lane that can neither raise nor produce a null. Three strict xfails. A fix
  needs a sign correction *and* a validity mask derived from the divisor — the
  shape `NumericBinary.validity` already computes.
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
`GROUP_RADIX` and `GROUP_THREAD_LOCAL` — described in `docs/backlog.md` §4 as
shipped — return **zero grep hits in the tree**. They were removed in the
aggregate rearchitecture; the backlog is stale. There is no pipeline
parallelism: `Pipeline._flow` pushes one morsel through the stages on the
calling thread.

**What the incumbents do.** DuckDB is morsel-driven with its own work-stealing
scheduler (`src/parallel/task_scheduler.cpp`, `pipeline_executor.cpp`) over a
pipeline/event graph. polars' streaming engine is the same family and its
vocabulary is nearly marrow's: `Morsel {df, MorselSeq, SourceToken, WaitToken}`
(`crates/polars-stream/src/morsel.rs`), where the `WaitToken` is backpressure
and `SourceToken::stop()` is **exactly marrow's `done()`** — so marrow's
operator contract is already the right shape and what it lacks is the executor
underneath (polars has a custom work-stealing one in
`polars-stream/src/async_executor/mod.rs`). DataFusion is partition-based: the
plan declares `Partitioning`, `EnsureRequirements` inserts `RepartitionExec`,
and tasks run on the ambient Tokio runtime — **there is no DataFusion scheduler
at all**, so the bar is lower than it looks. All three parallelize aggregation
by radix-partitioned or thread-local partials
(`duckdb/src/execution/radix_partitioned_hashtable.cpp`,
`datafusion/physical-plan/src/aggregates/grouped_hash_stream.rs`,
`polars-stream/src/nodes/group_by.rs`).

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

**What the incumbents do — and the bar is much lower than reputation suggests.**
Only DuckDB genuinely spills everything: one buffer manager plus a
`TemporaryMemoryManager` that rebalances reservations across concurrent
operators (`src/storage/{standard_buffer_manager,temporary_file_manager,
temporary_memory_manager}.cpp`), hash joins included
(`physical_hash_join.cpp:1715`, `JoinHashTable::ProbeSpill`). DataFusion has
`MemoryPool` + `DiskManager` + `SpillManager` and spills sort, grouped
aggregation, sort-merge join and nested-loop join — but **not hash join**
(`hash_join/exec.rs` returns `ResourcesExhausted` on build-side OOM) — and its
default pool is `UnboundedMemoryPool`, i.e. no limit at all. **polars does not
spill:** every method of `crates/polars-ooc/src/spiller.rs` is
`unimplemented!()` and `SpillPolicy::NoSpill` is the default; what exists is
memory *accounting* — a global `MemoryManager` budgeted at 70% of system memory
with per-thread drift-synced counters
(`crates/polars-ooc/src/memory_manager.rs`), wired into group_by, joins,
multiplexer and the in-memory sink. The old spilling streaming engine was
removed and not replaced.

**What it would take.** Memory accounting first — nothing in marrow knows how
much it has allocated, and polars shows that accounting alone is a shippable
position. Then spilling variants of group-by and sort. Far cheaper and worth
doing much earlier: **expose `drain()` to the user as a batch iterator**, which
is exactly polars' `sink_batches(callback)` / `collect_batches()` escape hatch
(`py-polars/src/polars/lazyframe/frame.py:4126,4222`), so a large result never
has to be one `RecordBatch`.

#### 2.3 Relational operations that have no node

Each is a missing `Relation`, not a missing kernel. ibis's `relations.py` is the
canonical list; marrow has 8 of it.

| Missing | Golden cases | Note |
|---|---|---|
| `UNION ALL` / `UNION` / `EXCEPT` / `INTERSECT` | 4 | ibis models these as one `Set(left, right, distinct: bool)`. They also treat NULL as equal to itself, which nothing else in marrow does |
| `Distinct` / `.unique()` | 1 (`DISTINCT ON`) | Expressible today as `aggregate(keys=[...], aggs=[])` (`logical.mojo:938` accepts empty aggs), but there is no verb and no `unique` kernel |
| `GROUPING SETS` / `ROLLUP` / `CUBE` | 3 | `Aggregate` carries one key list; `ROLLUP` also needs `GROUPING()`. DuckDB rewrites these into an aggregation cascade (`grouping_sets_optimizer.cpp`) |
| `explode` / `unnest` | 1 | Row-multiplying, so a new operator shape. ibis has a dedicated `TableUnnest` with `offset` and `keep_empty` |
| `Sample`, `DropNull(how)`, `FillNull` as relations | — | ibis has all three as nodes |
| pivot / unpivot / transpose | — | Out of scope per `golden/COVERAGE.md`, but two of three incumbents have them: DuckDB (`bind_pivot.cpp`, 42.9 KB) and polars (`LazyFrame.pivot`, `unpivot`, plus eager-only `transpose`). DataFusion does not |
| `top_k` / `bottom_k` as first-class | — | polars has a dedicated streaming node (`nodes/top_k.rs`) rather than a sort+limit rewrite; marrow has a *dead* `limit=` parameter (§2.10) |
| `merge_sorted`, `rolling`, `group_by_dynamic`, `upsample` | — | polars-only; time-series reshaping is a large part of why users pick it |

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

**What the incumbents do.** DuckDB ships nine join operators (hash + perfect
hash, nested loop, blockwise NL, piecewise merge, IE-join, asof, cross,
positional, plus delim joins for decorrelation). polars ships seven streaming
join nodes (equi, merge, asof, cross, range/IEJoin, semi-anti) and exposes
`join_where` for inequality joins, collapsing up to two inequality predicates
into an IEJoin and falling back to cross+filter beyond that
(`optimizer/predicate_pushdown/join.rs`, `IEJOIN_MAX_PREDICATES = 2`); its asof
carries `strategy={backward,forward,nearest}`, `tolerance`, `by`, and
`allow_exact_matches`. DataFusion ships six (hash, sort-merge, piecewise merge,
nested loop, cross, symmetric hash) and has **no asof and no IE-join** — so the
bar is lower than DuckDB suggests.

**The cheap move is the fallback, not the specialised operator.** polars'
own strategy is instructive: beyond two inequality predicates it degrades to
cross-product plus filter. A generic nested-loop join gives marrow exactly that
degradation path and turns cross and non-equi joins from *unexpressible* into
merely slow, which is a categorical improvement for a small amount of code.

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
  every decimal cast kernel existing. Decimal arithmetic was never written
  (`docs/backlog.md` §4). For a library aimed at analytics, "cannot compute on a
  money column" is close to disqualifying.

#### 2.7 No row format

marrow has no equivalent of arrow-rs's `arrow-row` crate
(`arrow-rs/arrow-row/src/lib.rs`, 256 KB): a byte-normalized row encoding where
`memcmp` on the encoded bytes equals the multi-column lexicographic comparison,
respecting per-field `descending`/`nulls_first`, and where rows are also
hashable and `Eq`. It is the shared primitive behind fast multi-column sort,
sort-merge join, and hash group-by keys. **polars has one too**
(`crates/polars-row/`), used for exactly that — multi-key sort — so this is a
convergent design in both reference implementations rather than one project's
taste.

marrow instead does column-oriented LSD multi-key sort — one stable pass per
key, re-gathering each key column per pass (`docs/backlog.md` §4, "Sort") — and
routes group-by/join keys through `StructArray` with per-column hashing. Both
work and neither is wrong, but this is the structural reason a future
sort-merge join has no cheap path and why multi-key sort re-gathers. Worth
naming as a design decision rather than discovering it under a benchmark.

#### 2.8 UDFs

**What exists.** Nothing. No `map_elements`, no `map_batches`, no `apply`, no
native UDF registration, no plugin surface.

**What the incumbents do.** DataFusion's *entire standard library* is written
against its public extension API — `ScalarUDF`, `AggregateUDF`, `WindowUDF`,
async UDFs, `TableProvider`, `OptimizerRule`, `ExecutionPlan`, all registerable
on `SessionState`, plus a stable C ABI (`datafusion/ffi/src/`) that carries
extensions across a language boundary using Arrow C Data wrappers. polars has
`map_elements` (per-element Python, whose own docstring warns it is "much slower
than the native expressions API"), `map_batches` (whole-Series), a
`LazyFrame.map_batches`, a **native plugin system** loaded as a dynamic library
over a stable C ABI (`py-polars/src/polars/plugins.py`,
`crates/polars-ffi/src/version_0.rs`), and **IO plugins**
(`register_io_source`) that receive pushed-down projections and predicates.
ibis exposes four UDF input types (`BUILTIN`, `PYTHON`, `PANDAS`, `PYARROW`) and
offers *only* `builtin` for aggregates (`ibis/expr/operations/udf.py:684-701`) —
so a Python scalar UDF is table stakes and a Python aggregate UDF is not.

**What it would take.** In the runtime lane, a UDF is a new `RuntimeValue` tag
holding a callable — tractable. **In the comptime lane a native UDF is close to
free and is where marrow should be strongest**: a Mojo function is already a
comptime value, and a user-supplied `lane[W]` would fuse into the same loop as
the built-ins with no boundary and no dynamic library. Note what polars had to
build to approximate that — a versioned C ABI, a derive-macro crate, and a
`.so` per plugin — and that it still has **no vectorized Arrow-native Python UDF
path** at all. This is a differentiator hiding inside a table-stakes item.

#### 2.9 Interop and format gaps

- **Compressed Arrow IPC bodies are unsupported.** `marrow/ipc.mojo:1409` raises
  on LZ4_FRAME/ZSTD bodies. arrow-rs supports both
  (`arrow-ipc/src/compression.rs:142`), so this is a live interop failure with
  the reference implementation — and marrow already `dlopen`s both codecs for
  Parquet.
- **No late materialization / row filter in the Parquet reader.** marrow prunes
  row groups and pages by statistics and bloom filters, which is most of the
  win; what is missing is DataFusion's `row_filter.rs` — evaluate the predicate
  on a subset of columns, then decode the rest only for surviving rows. Marked
  *unverified* as to how much it would buy on marrow's reader.
- **No SQL frontend.** No parser anywhere. The golden corpus is *written in SQL*
  and translated to marrow by hand, which is itself evidence of the impedance.
  polars shows the cheap version: `crates/polars-sql/` over sqlparser-rs,
  exposed as `SQLContext` / `df.sql(...)`, covering CTEs, set operations and
  window functions without the engine having its own parser.
- **No Substrait**, so no plan interchange — but this is genuinely optional:
  DataFusion treats it as first-class (`datafusion/substrait/src/`, logical and
  physical), while **DuckDB keeps it out of tree and polars has zero
  occurrences of the word anywhere in its repo.** polars instead serializes its
  own DSL (`LazyFrame.serialize/deserialize`, binary or JSON, guarded by a
  schema-hash file), which is a much cheaper answer to the same need.
- **No `__dataframe__` protocol**, though the PyCapsule/C Stream path marrow
  already has is the better-supported modern route. polars implements both.
- **No one-call pandas/polars/numpy conversion** — reachable via PyCapsule, but
  ibis's backend contract expects `to_pandas`, `to_pyarrow`,
  `to_pyarrow_batches`, `to_polars` and `to_torch` as named methods.

#### 2.10 Operability

- **No `explain()` verb**, though plans render recursively through `Writable`
  (`Filter.write_to`, `logical.mojo:816`), so the string exists and only needs a
  name. `README.md:238`'s "explain() renders one node" is stale.
- **No `EXPLAIN ANALYZE`, no per-operator metrics, no profiling hook, no
  progress, no cancellation.** An operator cannot be interrupted mid-`drain`.
  All three incumbents treat metrics as core: DuckDB has a declarative metric
  catalog (`src/common/metrics.json`) and a Kalman-filtered progress bar;
  DataFusion has `BaselineMetrics` with per-partition attribution and pruning
  counters (`datasource-parquet/src/metrics.rs`); polars has `profile()`
  returning per-node microsecond timings as a DataFrame, `show_graph()`
  rendering the plan as Graphviz, and per-node poll/morsel/row/IO counters
  (`crates/polars-stream/src/metrics.rs`). On cancellation, polars has three
  mechanisms — SIGINT→`KeyboardInterrupt` bridging
  (`crates/polars-error/src/signals.rs`), `collect_concurrently()` returning an
  `InProcessQuery` with `.cancel()`, and `SourceToken::stop()` — and marrow has
  the third one already, under the name `done()`.
- **No error taxonomy.** 244 `raise Error(...)` sites across `marrow/` produce
  plain strings with no type. The messages themselves are good — they name the
  verb and the column (`"drop: column 'x' not found in schema"`) — but a Python
  frontend cannot map them to distinct exception classes. polars' answer is 17
  semantically distinct `PolarsError` variants plus two wrapping ones,
  `Context { error, msg }` and `ExprContext { error, expr }`, which attaches the
  offending expression to the message
  (`crates/polars-error/src/lib.rs:83`), all surfaced as a typed Python
  hierarchy. That is cheap to copy and should land with the frontend, not after
  it.
- **Top-K is a dead parameter.** `sort_indices(..., limit=)` is passed non-`None`
  at exactly two sites, both tests; `SortOperator` never passes it, so
  `sort_by(...).limit(k)` performs a full sort and discards. Both incumbents
  make this a named optimizer pass (`duckdb/src/optimizer/topn_optimizer.cpp`,
  `datafusion/physical-optimizer/src/topk_aggregation.rs`). Wiring it needs a
  `row_limit` channel through `Pushdown` *and* a per-node rule table, because
  `Limit(Filter(Sort(x)))` may not take the top K — the filter runs after the
  sort, so a K-row sort silently returns fewer than K rows.
- **Per-key null placement is missing on sort:** `Sort` carries one
  `nulls_first: Bool` for all keys (`logical.mojo:1066`); ibis's `SortKey`
  carries it per key.

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
`__getattr_param__` can return a *conditional* type carrying its trait bound. So
`t.amount` resolves to `Column[Float64Type]`, `t.qty` to `Column[Int64Type]`,
and `t.amont` is a compile error reading `constraint failed: unknown column:
amont`. Every incumbent discovers a bad column name at plan time at the earliest
— ibis at expression construction, DuckDB and DataFusion at bind time, polars at
`collect()`. None of them can do it at build time, because none of them has a
compile step to do it in.

**Is it a product advantage, and for whom?** Yes, and for a market nobody
serves:

- Fixed, known queries shipped into constrained targets — edge devices,
  embedded analytics, on-device telemetry rollups, per-tenant compiled reports.
  DuckDB, DataFusion and polars all ship a general interpreter whether or not
  the deployment uses one.
- Data-plane filters and ETL steps where the query is code, is reviewed, and
  never changes at run time.
- Anywhere a wrong column name should fail in CI rather than at 3 a.m.

It is **not** an advantage for the analyst reaching for polars: that user needs
a REPL, and a REPL has no compile step. Which is exactly why the runtime lane
exists and why the two lanes must stay at parity — a point the project already
holds as an architectural invariant ("one engine, two drivers",
`docs/backlog.md` §2) but currently does not enforce, since `test_parity.mojo`
was deleted with the old package and has no replacement.

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

278 golden cases whose expectations are generated *from DuckDB, never from
marrow*, with `xfail` reserved for wrong answers and `skip mojo` for absent APIs
— so the corpus states its own gaps rather than hiding them. Archery conformance
against three other Arrow implementations. A pruning module that cites
`parquet/statistics.h` and arrow-rs line-by-line on what an absent null count
means, and gets `Limit`-clears-the-predicate right, which is the one error class
that changes answers.

This is not a user-visible feature on its own. It is the reason the gap list
above can be trusted, and it is a credible thing to say to anyone evaluating an
alpha: *these are our 85 known gaps and our 5 known wrong answers, machine-
checked against DuckDB nightly.* Very few young engines can say that.

---

## 3. Standing constraints

Each of these cost real time to find and invalidates an approach that looks
obvious. Read before planning anything.

### Architectural invariants (gate every merge)

1. **Small-binary DCE.** Preserve the closed-erasure property: no open
   dispatchers, fused-only value boxes, closed per-dtype kernels. Gate on
   `pixi run binary_size`.

   **Amended 2026-08-31.** `DynRelation` is now a `Variant`, which does name
   all nine relation nodes in one place — the thing this invariant forbids.
   That is deliberate and its price is bounded by *where* the variant is used:
   `isa`/`get` instantiate nothing per member, so inspection is free, while
   `to_operator` stays on a per-type trampoline. Routing lowering through the
   variant's ladder instead measured **+348%** of `__text` on `query_streaming`,
   because it makes every node's operator — and therefore `kernels::sort`,
   `kernels::join` and the Parquet reader — reachable from any plan. The
   working rule is narrower than the original: **a closed type set may be a
   variant; what must never go through its ladder is anything that reaches a
   kernel.**
2. **One engine, two drivers.** No feature may exist in only one lane. This was
   enforced by a `test_parity.mojo` across four axes — op names read off the
   kernel, pruning, values, aggregates through a keyless plan — which went with
   the old expression package and **has no replacement under
   `marrow/expr/tests/`**. Until one exists the invariant is unenforced; a new
   op should still add both lanes.
3. **PyArrow-shaped naming** in the core types and the bindings.
4. **Code quality is an acceptance criterion**, not a follow-up. Behaviour lives
   on the type or trait, not in free functions. A wave does not open until the
   prior wave's quality pass is green — it is a gate, not cleanup-later.

### Do not change

- **Array, scalar and builder layout.** Adding methods and accessors is fine;
  adding, removing, reordering or re-typing fields is **out of scope, not
  deferred**. Note this constraint has been read too broadly before: B13 (a
  sliced `BoolArray` double-applying its offset) was filed as blocked by it,
  when the fix was deleting a redundant `+ self.offset` and touched no field at
  all. Check whether a fix actually needs a layout change before deferring it.

### Measurement traps

- **The binary-size gate's file-size number is quantized to 16 KB.** Apple
  Silicon uses 16 KB pages, so a stripped binary's *file size* moves in 16,384-byte
  steps — a real +1,728-byte change once showed up as +16,504 with one *fewer*
  symbol. Measure `size -m <binary>` → `Section __text`.
- **Measure one gate binary directly** (`mojo build -O3 -g0 -I . …query_dynvalue.mojo`,
  ~2.5 min), not the whole `pixi run binary_size` sweep (~10 min).
- **A generic wrapper around an already-erased dispatch is not free.** Folding
  twelve promote-then-dispatch sites into one `_arith[K]` helper cost
  **+115,600 bytes**; writing the same four lines inline in each arm cost a
  fraction. A parameterised method is instantiated per kernel and each
  instantiation carries its own copy of what it touches.
- **A runtime switch over tags is the anti-pattern.** Rewriting the runtime lane
  as one `_eval` switch over ~70 tags cost **+1,807,168 bytes of `__text`
  (+45.7%)** on `query_dynvalue`, because every arm became reachable from every
  node. The fn-pointer `EvalFn` (`dynamic.mojo:209`) exists for this reason.
- **Reachability intuitions about erased paths are usually wrong; stub and
  measure.** Stubbing the `cast_array` calls out of the expression layer left
  the gate binary byte-identical. (The card that recorded this said "both";
  `expr/dynamic.mojo` has **seven** — `:187, :189, :203, :216, :218, :359,
  :373` — so re-count before repeating the experiment.)
- **An operator with no benchmark has no performance.** T2.4's per-row-group scan
  shipped a **4.7x** regression that every test passed through, because nothing
  benched the scan operator.
- **A benchmark whose captured value is not `keep()`-alive after `b.iter[call]()`
  measures nothing** — one reported 17,774 GElems/s. Check throughput for
  physical plausibility before believing a flat A/B.
- **The pytest-benchmark table prints mixed units per row.** Compare
  `--benchmark-json` medians (seconds), never the table's numbers.
- **Benchmarks here vary 10–18% run to run.** Interleave repeats across refs
  (nesting them concentrates machine drift on whichever ref is measured last and
  *invents* regressions), use five or more, compare ranges.
- **`views._reduce_dispatch` staying on `sync_parallelize` is deliberate, not an
  unconverted leftover** — it is the fourth documented one, and its comment says
  so. Routing it through `ctx.stripe` forces the serial arm to allocate a
  one-slot partials buffer it does not need: `sumint64_1k` 0.19-0.20 → 0.30-0.32
  µs, `sumfloat64_1k` 0.23-0.24 → 0.34-0.37 µs (five interleaved repeats,
  disjoint ranges). A serial fold should not pay for the merge's scratch.
- **A fixed per-call cost hides until you change how often the call happens.**
  `ParquetFile.read` built a `CompressionLibs` per worker and the first
  decompress `dlopen`s the codec library — invisible at one read per file, 4.7x
  at one read per row group.

### Trait and compiler limits measured 2026-08-17

Each is a hard error, and each rules out an approach that looks obvious.

- **A trait default body cannot name a field of `Self`** —
  `error: '_Self' value has no attribute 'a'`. The operand must be reached
  through a by-value accessor requirement.
- **A ref-returning accessor is not expressible at trait level** —
  `cannot return 'self's origin, because it might expand to a RegisterPassable
  type`. Same origin-widening wall as `arrays.validity()`; this is what forces
  the `.copy()`.
- **A sub-trait default returning `Self.State` does not recurse — it fails to
  reduce**: `cannot implicitly convert '_Self.Operand.State' value to
  '_Self.State'`. `rebind` cannot bridge it, because `rebind` takes its argument
  by implicitly-copyable borrow and `State` is not. **This settles §5.1's
  validity/state delegation question as impossible, not merely costly.**
- **A struct parameter does not satisfy an associated-type requirement** —
  `struct Neg[A: Leafy](UnaryDirect)` gives `required member 'A' is not
  specified`. The trait needs a differently-named member the struct binds
  explicitly.
- **A trait default is size-free; the boilerplate is the cost.** Hoisting the 22
  verbatim `referenced_columns` copies to a `UnaryValue` default compiles clean
  and measures **+0 bytes on `query_streaming_agg_fused`, +128 on `query_exprs`**
  — and the +128 is not the trait mechanism but +48 bytes per boxed
  instantiation, from the forced `.copy()`. It is still not worth doing: the
  dedup makes the file **58 lines longer**.

### Measurement traps found in the alpha wave (2026-08-18)

Each of these produced a **confident wrong conclusion** before it was caught.
Sources: `git show c0831f5^:docs/alpha-findings/<name>.md`.

- **A sampled profile tells you where time goes *under the profiler*.** `sample`
  and Instruments inflate image load/unload, so anything dominated by `dlopen`,
  `dlclose` or loader work is over-weighted. Measured on one `-O1 -g` build:
  **37.83 ms/run under the sampler, 7.85 ms/run without it** — a ~5x
  over-attribution that produced the claim "80% of `COUNT(*)` is `dlopen`". The
  real saving from fixing it was ~0.9 ms/query. Confirm a profile-derived
  hypothesis with a wall-clock A/B before believing its percentage.
  (`o1-codec-caching`)
- **The profile build is `-O1`; the benchmark build is `-O3`.** `scripts/profile.py`
  rebuilds with debug info because `-O3` inlines away the frames you are reading.
  So a trace shows *where*, never *how much* — absolute numbers come from
  `bench_clickbench.py`.
- **Perturbing one `-O3` unit can move an unrelated kernel.** An exact-size
  `self.tokens.reserve(n)` — an upper bound, apparently free — cost **+43% on
  `bench_contains_1m`**, which shares no code with the changed type, plus +35%
  and +21% on other untouched scan paths. Reproduced two runs per side; vanished
  when the one line was removed. **Only the drift controls caught it**: without
  benchmarks the change cannot touch, a 20.9x win would have shipped with a 40%
  regression underneath. This is the concrete case for CLAUDE.md's "always
  include rows the change cannot touch". (`o3-string-alloc`)
- **A passing size gate is not "no regression".** `check_gate.py` compares to the
  recorded `baseline.json`, **not** to the branch under test. Four of five
  recorded values sat above the tree on *both* branches, so gates read as
  shrinking 2.4-4.1% while the branch-to-branch measurement showed every gate
  **grew** ~16 KB. Ask which question you are answering. (`g3-regression-check`)
- **ASAN is not usable as evidence on this tree.** A deliberate-overflow probe
  built with the harness's own flags **hangs before producing a report**. This is
  stronger than CLAUDE.md's existing "ASAN can hide a heap bug": here it cannot
  be run at all. (`f1-distinct-segfault`)
- **`pixi run -e dev python script.py` does not rebuild `libmarrow.so`** — only
  pytest's `conftest.py` does. The natural A/B (checkout old, run script,
  checkout new, run script) therefore measures the *new* library twice. It
  produced a confident "no improvement, revert it" on a change that was a 20.7x
  win, and separately made an already-fixed bug still look broken. Rebuild with
  `pixi run build_python` between variants, or drive the comparison through
  pytest. (`o2-cast-utf8`)
- **A mass failure at exactly the harness deadline is the harness, not your
  change.** Five separate runs reported every case failed with empty messages at
  precisely 1800.0s. It nearly caused a correct fix to be reverted. Distinguish
  the two shapes: a *slow* unit burns CPU; a *wedged* one freezes — check that
  accumulated CPU `time` is advancing, not just `%cpu`.
- **A clean `mojo precompile marrow` is not evidence a test file will build.** It
  compiles the library, not the test's instantiations. Both compiler hangs in §2
  are invisible to it. (`o2-cast-utf8`, `h2-nested-equality-wedge`)

### Compiler and platform facts

- **`Buffer` requires 64-byte pointer alignment**, so `read_at` cannot return a
  sub-`Buffer` at an arbitrary file offset, and neither can IPC's `_slice_body`:
  Arrow IPC pads to 8, not 64. A source owns *one* whole-file `Buffer` and hands
  out `BufferView`s. This has blocked two separate designs.
- **A comptime conditional type carries no trait conformance** and does not
  reduce at a return site, even inside a `comptime if` that selected the branch;
  `rebind` does not rescue it.
- **A capturing closure's type is parameterised by its creating scope**, so it
  cannot be stored in a struct field and outlive that scope. Every stored
  callback must be `thin`.
- **A narrow test selection can hang the compiler where a wide one does not**,
  and the diagnostic advice for it is a trap. See §2 *`dispatch` hangs a narrow
  unit*. Two things cost hours on 2026-08-17: `mojo run` leaves an **idle parent
  process pinned at ~0:07.9 CPU** while a child does the real work, so
  `pgrep -x mojo | head -1` (and the harness's own "compare elapsed against CPU
  time" hint) reports a frozen CPU clock and looks exactly like a deadlock —
  sum CPU across *all* mojo processes instead, and it climbs past 17 CPU-minutes.
  And `ps -eo comm` prints `mojo` for the parent but the **full path** for the
  child, so `awk '$2=="mojo"'` silently misses the one that matters.
- **`ctx.stripe` bodies may not raise**, and widening it miscompiles: the
  parameter form of `sync_parallelize` that accepts a raising worker needs an
  implicitly-capturing closure whose captures are silently not made. Watch for
  "assignment was never used" warnings on buffers the body writes.
- **`origin_of(a, b)` is an origin union**, which is what lets a function return
  values borrowed from either of two storages.
- **A closure type cannot be generic over its own trait bound**, so a *shared*
  dispatch loop would have to bind `func` on `Movable` and let the caller narrow
  through an extra closure. That adapter inlines into every arm: it measured
  **+662,740 bytes (+31.9% of `__text`)** on `query_streaming_agg_fused`. Each
  erased box writes its own `isa` ladder instead — do not refactor them back
  onto a common helper.
- **Two closure arguments to the same call may not both capture mutably**, nor
  mut+imm over one origin. An API taking several closures over shared mutable
  state must thread that state through as an explicit `mut` parameter of each
  closure — which, when there are several callbacks over one object, is a trait
  (`parquet`'s `LeveledSink`). A state *struct* handed to separate closures does
  not work: it makes them parametric over the enclosing generic parameter.
- **macOS needs the Metal toolchain installed separately** —
  `xcodebuild -downloadComponent MetalToolchain`. Without it every GPU test dies
  with `Metal Compiler failed to compile metallib`, which reads like a Mojo bug
  and is not one (a marrow-free three-line `elementwise` program fails
  identically). This was tracked as B25 and blamed on a backend crash for
  months; resolved 2026-08-16, `test_views_gpu` is 15/15.
- **A trait requirement cannot name a field**, which is why `slice()` (7 bodies)
  and `validity()` (7 byte-identical bodies) in `arrays.mojo` are unfactorable.
  `validity()` returns `Optional[BitmapView[origin_of(self.bitmap._value)]]` — the
  return type names a field, so it is neither expressible as a requirement nor
  reachable from generic code over `A: Array`. The `ArrayData` round-trip default
  that would replace the seven `slice` bodies makes every typed `slice`
  **raising**, and keeping `slice` non-raising was decided 2026-08-16. The same
  wall blocks a `trait ValidityBuilder` default (needs `self._bitmap`) and a
  `CReleasable` trait for `c_data`'s release slot (needs `self.release`). These
  are language limits, not design debt — stop re-filing them as opportunities.
  **Probed directly on 2026-08-17** for `validity()`, and it fails twice over:
  declaring the requirement with a self-wide `origin_of(self)` is rejected
  because Mojo does not widen an implementation's narrower origin at the
  conformance site, and `NullArray` cannot implement it at all — it is all-null
  with no bitmap, and `None` already means all-valid. Even with the origin
  solved it would be a promise 2 of 9 conformers could not keep, which is the
  `to_device` shape the abstraction audit called the tree's one leaky
  abstraction.
- **Erase into a trait whose members are all runtime methods**, and only where
  the conformance is *honest* **and** has a consumer outside its own loop. A
  comptime member has no execution point at which to raise, so a box can only
  supply a plausible constant and the failure mode is a **wrong answer, not an
  exception**. Four sound-but-unconsumed conformances were removed for the second
  half of that rule at zero binary cost; see `docs/dyn-conformance-removal.md`.
- **A trait *default method* is not the `_arith[K]` shape.** The +115,600-byte
  trap above is about a *generic wrapper instantiated per kernel*; a trait default
  is instantiated once per conforming struct, exactly as the hand-written copy
  was, so DCE sees the same reachability graph. If that holds, deduplicating
  behind a trait default is source-level only. **It is an argument, not a
  measurement** — one ~30-line spike settles it for every item in §5.1's first
  bullet, and should run before any of them.
- **`.claude/worktrees/` contains two stale worktrees** (`docs-revamp`, `q25`) holding pre-Q2.5 `AGG_*` and `reinterpret_array` code.
  Exclude them from every grep or you will get false positives.

---

---

## 4. The compiler deadlock — fixed 2026-08-30, kept because the shape recurs

`raise Self.error(t"unsupported dtype {dt}")` deadlocked the Mojo compiler:
~7 s of CPU, then parked forever at 0% CPU with flat RSS, main thread in
`semaphore_wait_trap`, **no diagnostic**. `String("unsupported dtype ", dt)` —
same message, same value — builds.

**Minimal trigger: a t-string interpolating a recursive `Writable` value inside
a function-level recursion cycle.** Three ingredients, each independently
necessary; remove any one and it builds in 5 s. Reproducer with no marrow
import: `docs/repros/tstring_recursive_writable_cycle.mojo`.

It had been open since 2026-08-17 under three separate backlog entries, and two
confident diagnoses were wrong before this one:

- **not** the `List`+`Variant` layout corruption — `DynScalar` measures 80/8 at
  HEAD and re-introducing the broken 96/32 shape left the deadlock unchanged;
- **not** the `dispatch` ↔ `apply(StructArray)` instantiation cycle — removing
  the `is_struct()` arm entirely still hung;
- **not** the erased ladder's fan-out width, which is irrelevant.

Two working rules came out of it:

1. **The compiler wedges *before* it diagnoses.** An ordinary type error
   produces an identical hang with an empty log. Always `precompile` first;
   "it still hangs" can just mean "it does not compile".
2. **Deadlock vs slow is CPU *time*, not elapsed.** Frozen CPU time plus flat
   RSS is a deadlock. A runaway elaboration burns CPU and grows memory.

The mechanism is still **unexplained** — the toolchain is stripped, and the
t-string/`String(...)` difference is inferred from the A/B, not observed.


## 5. Rejected and replaced designs

Each row is a design that was written down, then **not** built — because the tree
built something different and better, or because the premise turned out to be
wrong. They are here so nobody re-litigates them from a stale document. Every
citation below was checked against the code, not copied from the design it
replaces.

### Simplification wave (2026-08-17)

- **`equal_any` → a neutral `kernels/compare.mojo` — rejected (D2).**
  `numeric.mojo:49` imports `StringEqKernel` solely for `equal_any`, so the move
  *does* delete the `numeric → string` edge. But `EqKernel.apply(StructArray, …)`
  (`:617-645`) calls `equal_any` twice and **cannot move**, because Mojo cannot
  add a static method to `EqKernel` from another module. The result is a
  `numeric ↔ compare` **cycle replacing an acyclic edge** — worse, immediately
  after S1 finished removing cycles. Revisit only if the `StructArray`
  row-equality relocates for an independent reason.
- **Amended (D1): the rejection of `Buffer.alloc_for[T](ctx, n)` in
  `buffers.mojo` is overruled.** The original row rejected it for pointing the
  tree's lowest-level module at device policy. It was re-opened on 2026-08-17
  because the `ExecContext` alternative turned out to close
  `execution → buffers → views → execution`, which the original row did not know.
  A rejected-designs list that is quietly contradicted is worse than no list, so
  the reversal is recorded here rather than left implicit.

### Alpha wave (2026-08-18)

- **`mojo-regex` — rejected on correctness, not on version drift.** It builds
  against our pinned Mojo (13/14 of its own tests pass) so the expected blocker
  was not the real one. It **never enters an optional group**: `(?:www\.)?` is
  skipped, so Q29 returns `www.example.com` where pyarrow and CPython return
  `example.com`; 10 of 25 `sub()` cases disagree with CPython. Minimal repro:
  `sub("(?:foo)?bar", "B", "foobar") -> "fooB"`. It survived upstream because
  their tests use `(?:` seventeen times and never with a trailing `?`. Adopting
  it would have converted an honest 42/43 gap into a **silently wrong answer on
  the majority of rows**, since `www.`-prefixed referers dominate `hits`. See
  M2.6 for the replacement.
- **A hand-written `extract_host` for Q29 — rejected.** It buys a DEVIATED row,
  generalises to nothing, and gets deleted the moment real regex lands.
- **Over-allocating buffers so "64-byte padded" becomes literally true —
  rejected, and the premise was wrong anyway.** `Columnar.rst:264-273` says pad
  "to a length that is a multiple of 8 or 64 bytes", i.e. *round the size up* —
  exactly `align_up(bytes, 64)`, which is byte-for-byte Arrow C++'s
  `PoolBuffer::RoundCapacity`. `arrow::AllocateBuffer(64)` allocates 64 bytes
  with zero slack. Marrow was already conformant; the false step was inferring
  that padding implies slack past the logical end. Over-allocating was rejected
  because **it cannot deliver the invariant**: FOREIGN buffers come from the
  producer, and pyarrow allocates exactly 64 bytes for a 512-row bitmap — the
  guarantee would hold on half the buffers and make the other half harder to
  find. Memory cost was explicitly *not* the deciding argument.
- **Rewriting `DynArray.__eq__` to dispatch once — tried, reverted.** O(n) arms
  instead of O(n²) and exactly as strict, and it compiles; but the hang in §2 is
  recursion through `ListLikeArray.__eq__`, not the squared ladder, so it fixes
  nothing. Not left in as an unmeasured change to a hot, size-gated file.
- **A fan-out threshold on probe rows for the parallel join — tried, measured,
  removed.** It would have *doubled* the regression: within the partitioned
  layout, fanning out beats serial partitions at **every** size, 8192 rows
  included (194 us vs 388 us). The expensive thing is the partitioning, not the
  `sync_parallelize`. What fixed it was `_DEFAULT_RADIX_BITS` 6 -> 4 — the
  64-partition default came from a *one-shot* 10M sweep, and morsel streaming
  pays it per call.
- **`inputs()`-based optimizer traversal — rejected for `children()` +
  `with_projection()`.** Both were superseded 2026-08-31: the optimizer walks
  plans through `Relation.traverse(f)`, one method per node that applies `f` to
  its own children and rebuilds itself. No `children()`, no `with_projection()`,
  and no ladder over node types in the optimizer.
- **Building a temporal literal through the storage integer and relabelling with
  `relabel_array` — rejected on measurement.** It avoids six
  `PrimitiveBuilder`/`PrimitiveArray` monomorphisations (+23,668 bytes on
  `query_streaming`) but drags in `DynArray.from_data`'s 30-arm ladder that these
  binaries do not otherwise link: **+106,276 bytes, 4.5x worse**. The trick is
  right in `kernels.cast`, where `from_data` is reachable anyway; it is wrong in
  `scalars.mojo`.

### Sort

- **Permutation refinement (`getPermutation` / `updatePermutation` /
  `EqualRanges`) was never built.** Multi-key sort is a **column-oriented LSD**:
  `SortIndices.multi` (`sort.mojo:449`) stable-sorts by the *least*-significant
  key, then for each more-significant key gathers the column under the running
  permutation, sorts that, and composes (`sort.mojo:493-515`). There is no
  equal-range bookkeeping anywhere. *Why it stands:* one stable pass per key
  yields the same order without tracking ranges, and it reuses `Take` and the
  single-column sorters unchanged. *Its one cost is real*: the composition
  depends on every pass being stable — the `stable` flag was accepted and never
  forwarded, so this silently returned wrong orders until B1 was fixed by making
  the comparison path's comparator break ties on the original row index. It also
  re-gathers each key column per pass (**FU-6**).
- **8-bit radix passes / 256-bucket histograms were replaced by 11-bit passes /
  2048 buckets** (`_BITS_PER_PASS` `sort.mojo:76`; `bucket_count`, `:252`).
  *Why:* 6 passes instead of 8 for 64-bit keys with a histogram that still fits
  L1 per thread — measured ~6.7× at N=10M against 8-bit serial; 16-bit thrashes
  L1 (`sort.mojo:77-86`).
- **The comparison-vs-radix cutoff is not `N < 64`.** It is
  `_RADIX_THRESHOLD = 32_768` (`sort.mojo:59`). *Why:* PDQsort measured faster
  all the way to ~28K on int64; below the threshold the pair-buffer setup
  dominates. Decimal128/256 take the comparison path at any size, because their
  native width exceeds the UInt64 radix key.
- **`argsort` as a free function plus a `SortOptions` struct** became the
  `SortIndices` kernel struct (`sort.mojo:344`) with `dispatch`/`apply`/`multi`
  and loose parameters. Per-column `nulls_first` is still wanted — §6.

### Group-by

- **Of the seven designed groupers, exactly one shipped, and it is not one of
  them.** There is a single type-agnostic `HashGrouper` over a `SwissHashTable`
  (`groupby.mojo:43`). `DirectMapGrouper`, `PackedKeyGrouper`,
  `RowEncodedGrouper`, `TwoLevelGrouper` and `SpillingGrouper` do not exist, and
  neither does `kernels/row.mojo`. *Why:* the Swiss table's SIMD probing made
  per-key-type tables not worth their combinatorics; the design's own premise —
  "ClickHouse has 40+, so we need several" — did not survive contact with one
  table that handles every key type through `StructArray`.
- **The dispatch axis is inverted.** The design selects a grouper by **key
  type** (bool/uint8 → direct map, ≤16 fixed bytes → packed, else row-encoded).
  The shipped `_choose_strategy` (`groupby.mojo:402-412`) never looks at the key
  type at all: it selects on **row count and a sampled cardinality estimate**.
  Reading the design as a guide to the code inverts the whole decision.
- **The parallel strategies are gone, and this entry described a tree that no
  longer exists.** It cited `GROUP_THREAD_LOCAL` at `groupby.mojo:312`,
  `_choose_strategy` at `:402-412` and `GROUP_RADIX` at `:304` — in a file that
  is now **224 lines** with no parallelism in it at all: no `sync_parallelize`,
  no thread-local partials, no radix placement. The only mentions of "radix"
  are as a *future* runtime choice. Verified 2026-08-31.

  Parallel group-by is therefore **absent**, not "shipped differently from the
  design". §2 Tier 2.1 is the accurate description.

### Joins

- **`JoinHashTable` with an intrusive `_chain_next` collision list was never
  built.** Neither identifier appears anywhere in the tree. It was replaced by
  `SwissHashTable` plus a **CSR index** — `_offsets` / `_rows`, built after
  insertion and grouped by bucket (`hashtable.mojo:76-87`) — so a probe walks
  `[offset[bid], offset[bid+1])` contiguously (`:523-532`) instead of chasing
  pointers. *Why:* same ALL-strictness multi-match capability, sequential memory
  access, and one shared table type for join and group-by.
- **`IndexPairs` is not a struct.** It is
  `comptime IndexPairs = Tuple[Int32Array, Int32Array]` (`join.mojo:201`) — two
  Arrow arrays rather than two `List[Int32]`, which is what lets per-partition
  results merge by buffer memcpy.
- **The `PartitionedOp[T]` trait + `partition_apply` driver, recorded as
  "non-trivial in Mojo's generics, tracked as a follow-up", shipped** as
  `RadixPartitioner.map_partitions` (`partition.mojo:288`). It is done, not
  deferred; do not schedule it again.

### Abstractions and deduplication

Folded in from the abstraction, organizational and duplication audits
(2026-08-17), which were deleted once their open items became §5's `S`-IDs.
Every row is an approach that looks obvious and does not work.

- **One `Column[T: DataType]` struct** replacing the four `*Column` nodes. It
  would have to satisfy `NumericValue.lane` (returns `SIMD[T.native, W]`) and
  `StringValue.lane` (returns `String`) at once — different return types, and a
  `comptime if` does not rescue it, per §0's conditional-type note.
- **Promoting `write_repr_to` to an `Array`/`ArrowScalar` requirement.** Was the
  filed fix; the conformance removal made it unreachable — a requirement now binds
  only the nine typed arrays and nine typed scalars, leaving the erased handles
  (the only ones callers hold) still forwarding to `write_to`. Deletion is what
  remains, and is **S6**.
- **`Buffer.alloc_for[T](ctx, n)` in `buffers.mojo`** for the ten-site GPU-or-host
  preamble. It points the tree's lowest-level module at device policy. The
  decision belongs on `ExecContext`, which *is* the policy type and already owns
  `GPU_ENABLED` after `5b14bfa` — **S8**.
- **Moving the `AggFunction` trait down to `expr/aggregates.mojo`** to reunite it
  with its four conformers. It inverts the dependency: `groupby.mojo` would import
  `expr`, breaking the verified "`marrow/kernels` has no up-edges into `expr`"
  property. The catalog moving the *other* way is the live option — §5.1.
- **A generic `_header_equal[A: Array](a, b)`**, and the `unsafe_get`-on-`Array`
  fix that was filed for `__eq__`'s five copies. Both die on §0's
  trait-cannot-name-a-field limit; the element loops are also genuinely different
  (three index types, and `unsafe_get` raises for some arrays and not others).
  What was reachable landed instead (2026-08-17): the shared validity helper
  was inlined into all seven `__eq__` bodies, which removed two popcounts and
  a bit-by-bit loop rather than removing lines.
- **An embedded `ArrayHeader` field** collapsing `length`/`nulls`/`offset`/
  `bitmap` across six arrays. Forbidden by §0's *Do not change* — array layout.
- **`kernels/interval.mojo`'s placement — examined and upheld.** Consumed
  exclusively by `expr/` and it never touches an array, which usually means
  misfiled; the module docstring argues it, and the argument holds. Its one real
  consequence is that `IntervalKernel(Kernel)` inherits `expect_same_length` /
  `expect_same_dtype` it can never call, which is the `Named`-split finding in
  §5.1. Recorded so it is not re-opened.
- **The two `execution.mojo` files and the two `struct Filter`** are namespaced
  and correct. The only option is a rename and neither name is wrong. Leave both.

### Decimal

- **The custom `Int256 { low: UInt128, high: Int128 }` struct is unnecessary and
  was never written.** Its premise — "Mojo has no native `DType.int256`" — is
  false: `Decimal256Type = _DecimalType[DType.int256]` (`dtypes.mojo:252`), and
  `Decimal128Type` likewise uses `DType.int128` (`:251`). `struct Int256` has
  zero occurrences.
- **Decimal *arithmetic* was never built at all** — not rejected, just never
  started. It is **M3.7**, and the design's result-type rules are the part worth
  keeping.

---

---

## 6. What marrow has today, and what makes it different

Kept because a gap list without it reads as a list of failures, and
because the AOT lane is the thing every priority call above is weighed
against.

### The differentiation story

marrow's comptime lane is the one thing here that does not exist anywhere else:
an expression tree whose *structure lives in its type*, monomorphised and fused
into a single SIMD loop, with no interpreter reachable in the resulting binary.
Same query, `-O3 -g0`, stripped, `__text` section
(`benchmarks/binary_size/baseline.json`):

| Gate | Bytes | |
|---|---:|---|
| `query_streaming_agg_fused` (aggregate resolved at compile time) | 1,481,012 | |
| `query_streaming_agg` (same query, aggregate named at run time) | 9,940,868 | **6.71x** |

And `marrow/expr/comptime/tests/test_schema_handle.mojo` proves against the
compiler that `t.amount` can resolve to `Column[Float64Type]`, `t.qty` to
`Column[Int64Type]`, and `t.amont` to a **compile error**. No dataframe library
in existence type-checks column names.

That is a real advantage, and it is **not** an advantage for the polars user. It
is an advantage for someone shipping a fixed, known query into a constrained
target: edge devices, embedded analytics, data-plane filters, per-tenant
compiled reports. Nobody serves that market, because DuckDB, DataFusion and
polars all ship a general interpreter whether or not the deployment uses one.
The honest caveats: the AOT lane has **no product** — no CLI, no output writers,
no end-to-end `marrow compile` — and 1.48 MB still links `libmax`/AsyncRT, so
"tiny" is relative.

### The inventory

Established by reading, so the gap analysis is not argued against a straw man.

### Arrow core — strong

| Capability | Evidence |
|---|---|
| Arrays, builders, scalars, all bit-packed/offset layouts | `marrow/arrays.mojo` (2,941 lines), `builders.mojo`, `scalars.mojo` |
| Types: null, bool, all ints/floats, decimal 32/64/128/256, all temporal + interval, binary/large/fixed-size, string/large, list/large/fixed-size, struct, map, dictionary | `marrow/dtypes.mojo:177-767` |
| C Data Interface, **C Stream Interface**, device arrays | `marrow/c_data.mojo:268, 906, 1357, 1536` |
| PyCapsule protocol both directions from Python | `python/marrow/__init__.py:132, 278, 324` |
| Arrow IPC file + stream, read and write, with dictionary batches | `marrow/ipc.mojo` (2,425 lines) |
| Parquet reader: mmap, footer + page index, row-group **and page** pruning, bloom filters, SNAPPY/GZIP/BROTLI/LZ4/LZ4_RAW/ZSTD | `marrow/parquet/reader.mojo`, `codecs.mojo:1071-1077`, `bloom.mojo` |
| Parquet writer, nested Dremel shred | `marrow/parquet/writer.mojo` |
| GPU: one kernel serves CPU and device, opt-in at `-D MARROW_GPU=true` | `marrow/views.mojo`, `marrow/execution.mojo:182` |

**Worth stating plainly: marrow's Arrow *type* coverage is already better than
polars'.** polars has no `Union` at all (`crates/polars-core/src/datatypes/field.rs:307`
panics on it), flattens Arrow `Map` to `List(Struct{key,value})` lossily
(`field.rs:296`), widens `FixedSizeBinary` to variable `Binary` (`field.rs:295`),
converts `Interval` to a struct only behind an env var and only for two of three
units (`field.rs:299-306`), has no run-end encoding, and has only 128-bit
`Decimal` capped at precision 38. marrow has `map` as a first-class type —
which `CLAUDE.md:1053` records as passing the archery suite 14/14 against C++,
Rust and Go in both directions, *unverified here since no test was run* —
plus `fixed_size_binary`, all three interval variants, and all four decimal
widths. marrow's own gaps —
union, run-end-encoded, and the view layouts — are a *narrower* set than
polars'. Against arrow-rs, which has all of them
(`arrow-schema/src/datatype.rs:96-485`), marrow is behind on exactly those
three.

### Kernels — broad on numerics, thin elsewhere

~120 kernel structs (`marrow/kernels/*.mojo`): arithmetic and comparison across
every primitive family, boolean/Kleene, casts including every decimal
conversion, conditional (`case_when`/`coalesce`/`nullif`/`fill_null`),
aggregates (sum/product/min/max/count/mean/variance/stddev/any/all, exact and
HLL-approximate distinct count), filter/take/drop_null, radix + comparison sort,
Swiss-table hash join and group-by, rapidhash, radix partitioning, `is_in`, 20
string kernels including a backtracking `LIKE`/`ILIKE`, 11 temporal extractors
plus `date_trunc`, `array_length`/`array_contains`, `concat`.

### The engine — small but architecturally sound

- Eight relational nodes: `InMemoryTable`, `Filter`, `Project`, `Aggregate`,
  `Limit`, `Sort`, `Join`, `ParquetScan` (`marrow/expr/logical.mojo`).
- Fluent verbs: `.filter() .select() .project() .with_columns() .drop()
  .rename() .limit() .sort_by() .aggregate() .join() .execute()`.
- Push-based streaming operators with `push`/`drain`/`done`, so `LIMIT 10` over
  a large scan stops early (`marrow/expr/physical.mojo:182-232`). Parquet scans
  one row group per `drain` (`physical.mojo:1073`). **This is the same family as
  DuckDB's push/morsel model** (`duckdb/src/README.md`) and a better starting
  point than DataFusion's pull-based async streams for a single-process engine.
- Predicate pushdown to Parquet statistics, threaded through the lowering rather
  than as a rewrite pass, with correct per-node rules including the
  `Limit`-must-clear trap (`marrow/expr/pushdown.mojo:30-45`). It survived the
  optimizer rewrite unchanged and is independent of the rule set.
- A plan optimizer: 16 rules and a column-pruning pass in one file, producing an
  inspectable rewritten plan (`marrow/expr/optimizer.mojo`, §1.4). The rule set
  is a comptime parameter, so an AOT binary links only the rules it names.
- A one-sided pruning algebra: `Truth`/`Bounds[dt]` typed by the same comptime
  parameters as the fused `lane`, so pruning is the fusion mechanism read over a
  second domain (`marrow/expr/pruning.mojo:1-55`). It cites
  `parquet/statistics.h` and arrow-rs line-by-line on what an absent null count
  means. This is better-reasoned than most production pruners.
- Late-bound parameters carried *through* an execution rather than substituted
  into a plan copy (`marrow/expr/bindings.mojo`).
- Plans render recursively through `Writable`, so an EXPLAIN-shaped string
  already exists; only the verb is missing.

### Testing — a genuine strength

1,825 Mojo test cases across 73 files; 278 golden cases whose expectations are
generated **from DuckDB, never from marrow** (`golden/runner.py`), with a
three-marker discipline separating "wrong answer" (`xfail`, strict) from "no
API" (`skip mojo`); Arrow archery conformance against C++, Rust and Go
(`integration/`); an AddressSanitizer suite; a binary-size gate.

---

