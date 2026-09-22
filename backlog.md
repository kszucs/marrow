# marrow — open work

The epics and tasks worth doing, in priority order. **Nothing here describes
how the code is built** — for that read the code and `CLAUDE.md`, which is the
tracked home for architecture, coding rules, compiler gotchas and measurement
traps. This file is only what is missing, what is wrong, and what it would take.

Two things are recorded per item: why a user cares, and what standing in the
way is real rather than assumed. Claims were re-verified against the tree on
2026-09-14; where a claim did not survive that check it says so inline.

## What is missing, in priority order

One table for the whole backlog. **Priority** is what a first user hits soonest;
**complexity** is S (a day), M (a week), L (a couple of weeks), XL (a project
with its own design). "Blocked by" names a thing that must land first, not a
preference.

Detail for every row is in the tiers below; the standing constraints in §3 gate
all of them.

| # | Missing | Why it matters | Cx | Blocked by |
|---|---|---|---|---|
| 1 | **CSV reader**, then NDJSON | A first user arrives with a CSV, not a Parquet file. `find marrow -iname '*csv*'` is empty | **M** | — |
| 2 | **Error taxonomy** — 377 `raise Error` sites, zero typed exceptions | Cheap while the Python boundary is fresh, expensive to retrofit across 377 sites. Already a retrofit, and growing steadily: 269 on 2026-09-04, 337 on 2026-09-08, 366 on 2026-09-12, 373 on 2026-09-14, 377 on 2026-09-22 | **M** | — |
| 3 | **`scan(path)` without a hand-written schema**, then globs, directories, hive partitions | `scan()` takes one path *and* demands the schema by hand. Every real Parquet dataset is a directory | **M** | 1 |
| 4 | **`OpenDalSource.read_ranges` fetches serially** | One round-trip time per range where they could go out together. On a local file that is free; on S3 it is the difference between one RTT and N. The seam, the URI dispatch and `ParquetScanOperator` on `DynSource` are all in place, so this is the last piece of scanning `s3://` well. See §1.9 | **S** | — |
| 5 | **Parallel group-by gates are uncalibrated** — `_MIN_DISTINCT_RATIO` (0.9) and `_PARALLEL_GROUPBY_MIN_ROWS` (60,000) in `kernels/groupby.mojo` | Radix-partitioned placement landed in `dcef953a`, but when it engages is a guess: the 0.9 was set from a measurement of a version since made twice as fast, and nothing has re-measured the crossover. See §2.1 | **S** | — |
| 6 | **`distinct`, `union`, `except`, `intersect`** — no node exists for any of them | Table stakes for a SQL-shaped frontend, and `ReplaceDistinctWithAggregate` is a rule nobody can write without the node | **M** | — |
| 7 | **Join output ordering** — `JoinOperator` hardcodes build=left, `_output_schema` is positional | Blocks *both* remaining optimizer rules. Not an optimizer change: the kernel must accept an output ordering | **M** | — |
| 8 | **Join reordering + build-side selection** | The largest TPC-H win available, and the only genuinely cost-based pass in any incumbent | **L** | 7, 9 |
| 9 | **Statistics propagation and a cost model** | Feeds 8. Not urgent on its own — it is the last piece, after 7 | **L** | — |
| 10 | **CSE and duplicate group/sort key elimination** | Both need `DynValue` equality — likely solvable at the verb, as `constant_bool` and `conjuncts` were, rather than with a box slot | **M** | — |
| 11 | **Larger-than-memory execution** — no spilling anywhere | Every aggregate and join is bounded by RAM. Changes the operator contract | **XL** | — |
| 12 | **Nested-loop / range joins** | Only equijoins exist, so a non-equi predicate has no plan at all | **M** | — |
| 13 | **UDFs** | The escape hatch that makes a missing kernel survivable rather than fatal | **M** | 2 |
| 14 | **A row format** | Needed by sort-merge join, spilling, and any wire protocol | **L** | — |

---

## 1. Open work

Ordered by value. Some of it is partly done -- §1.9 in particular tracks a
subsystem that moved twice this week -- and each entry says which part.

### 1.1 Correctness — known-wrong answers

**`count(*)` desugars to `count(lit(1))`.** `builders.mojo` returns
`lit(1, int64).count().alias("count_star")`, whose `columns()` is empty — so
anything that prunes by "what does this read" prunes every column, and
`RecordBatch.num_rows()` returns **0** when there are none (`tabular.mojo`).

**Projection pushdown landed 2026-08-31 and handles it rather than fixing it:**
`ColumnPruning` never narrows a source to zero columns, keeping the first when
the demand is empty. The desugaring is unchanged, so the trap still waits for
the next thing that reads `columns()` and believes the answer.

**The vectorised zero-divisor scan has no caller left.** `//` and `%` answer
NULL by first asking "is there a zero in this column", and
`BufferView.__contains__` (`views.mojo:199`) is that question: SIMD, early
exit, per-chunk reduction — the variant measured fastest on 2026-09-04, where
the scalar form cost **+45%** on `bench_floordiv_int32_*`.
`marrow/tests/bench_views.mojo` exists to keep the scalar form from coming
back and says `RuntimeValue._null_zeros` and `DivisionBinary` both call it.

Neither does, and `grep -rn __contains__ marrow/` finds no production caller
at all. `DivisionBinary.bind` (`expr/comptime/numeric.mojo:214`) walks the
divisor a row at a time into a `Bitmap.alloc_zeroed(length)`, which is the
shape the +45% measured; `RuntimeValue._null_zeros` pays a `nullif` against a
broadcast zeros array instead. So the benchmark guards a helper nothing uses
while the regression it was written to catch is in the tree. Re-point the two
callers at `__contains__` before treating any cost here as measured — and a
kernel bench cannot see it, since no kernel passes over the divisor:
`BinaryKernel.apply` intersects the operands' validity and nothing else.

### 1.3 Latent compiler hazards

**More `t"…{dtype}"` sites under `marrow/kernels/`.** A t-string
interpolating a recursive `Writable` value inside a function-level recursion
cycle deadlocks the compiler. Three sites were fixed; the rest are untested and
sit in the same shape.

### 1.4 Engine capability the golden corpus measures as missing

`golden/COVERAGE.md` is authoritative and machine-checked: 279 cases, of which
**63 carry `-- skip mojo`** because marrow has no API for them. Their bodies
are never compiled, so they are proposals rather than verified spellings.

Counted 2026-09-03 and unchanged on 2026-09-14, by prefix:

| skipped | area |
|---|---|
| 13 | aggregates — median, quantile, first/last, arg_min/arg_max, string_agg, corr/covar, mode, skewness |
| 10 | temporal — date_diff, age, strftime/strptime, make_date, interval arithmetic, timezone attach |
| 8 | nested — struct field, map lookup, list element/slice, unnest |
| 7 | math — atan2 and the rest of the trigonometric family |
| 5 | string — regexp, concat_ws, null-skipping `concat` |
| 4 | set operations — UNION / EXCEPT / INTERSECT |
| 4 | joins — cross, non-equi, asof |
| 3 | GROUPING SETS / ROLLUP / CUBE |
| 3 | filters — SQL `NOT IN` null semantics, `.is_in` as a method |
| 3 | decimals |
| 2 | subqueries |
| 1 | `DISTINCT ON` |

`bool_and`/`bool_or` is the odd one out among the aggregates:
`AnyKernel`/`AllKernel` are `BoolReduceKernel`s rather than `FoldKernel`s and
have no grouped variant, so they need an aggregate node in both lanes *plus* a
kernel change.

`temporal_epoch_seconds` is the sharpest instance and was re-measured on
2026-09-04: `EpochKernel` exists, the body compiles, and un-skipping it still
fails on one row. Un-skipping it is a change to the case (`CAST(FLOOR(...))`
and a regenerated expectation), not to marrow.

Three of the remaining skips are **not** missing API, and two of those three
changed on 2026-09-07 without the skip count moving. `math_greatest_and_least`
and `filter_not_in_list_with_null` encode SQL null semantics — skip-nulls
extrema, and `NOT IN` with a NULL matching nothing — which the **SQL front end
now desugars**: `GREATEST`/`LEAST` become `coalesce(extremum, a, b)` and
`x IN (a, b)` becomes an `=` chain with NULLs lifted out, both documented in
`marrow/expr/sql.mojo`'s header. So the semantics exist through `Sql.plan` and
**not** through the expression API the cases are written against; closing them
means either the verbs or rewriting the cases in SQL. The third,
`nested_list_contains`, still needs `.contains` as a method on `ListValue`,
where only the free `array_contains` exists. Re-checking a case by name is not
enough to un-skip it: a case has been claimed unblocked, and found still
blocked, four separate times.

### 1.8 Test and infrastructure gaps

- **No cross-lane parity test.** "One engine, two drivers" was enforced by a
  `test_parity.mojo` across four axes; it went with the previous expression
  package and has no replacement. The invariant is currently unenforced.
- **Group-by is covered at the kernel, not through the engine.**
  `kernels/tests/test_groupby.mojo` (24 cases) pins both placement paths
  directly on `HashGrouping`, but nothing drives the radix path through
  `GroupByOperator`: every group-by case in `expr/tests`, `golden/` and
  `python/marrow/tests` is far under the 60,000-row gate, so the engine's
  wiring to the parallel path is untested end to end.

### 1.8b The dylib layer: what was measured and dropped

**Caching `dlsym` results in typed symbol tables.** Possible —
`_DLHandle.get_function[result_type]` returns a raw C-ABI function pointer
that *can* be a struct field, contrary to what `io/opendal.mojo` claimed for a
while (that claim was about the `OwnedDLHandle` overload, which returns a
borrowing callable). Implemented across all 37 call sites, then reverted: on
`bench_parquet` the snappy rows moved −3.6%, −0.2% and −0.4% while an
untouched uncompressed control moved +10%, so every number was noise; and the
size gate caught **+24,7xx bytes on `query_cli` (+0.79%)**, because a binary
that only reads Parquet links the compress symbols too where DCE previously
dropped the unused `call[...]` instantiations. What it *would* buy is
type-checked signatures — today a wrong return type in
`call["ZSTD_decompress", Int]` is silent. Revisit only with a plan for the
size, e.g. splitting each codec's table into decompress and compress halves.

One trap it surfaced, worth knowing before a second attempt: a typed symbol
field names an untracked pointer, which severs the compiler's reason to keep a
*local* struct argument materialised across the call — an `opendal_bytes`
passed that way faulted inside `Bytes::copy_from_slice`, silently writing zero
bytes before it crashed. A second: `_Global` keys against a registry shared
with the stdlib and MAX, so key uniqueness is a property of the whole process,
not of these specs. The key is the caller's to pick for that reason — deriving
it from a spec name let two specs sharing one silently alias each other's
storage.

### Link-time linking for the codecs — where this should end up

**The `dlopen` machinery is a workaround for a dependency we now declare.**
Linking the page codecs at build time is the better shape and should be the
target; it is deferred rather than rejected.

What it would delete outright: the candidate-path search and its documented
load-from-cwd surface (`_exe_dir` reads `argv()[0]`, which is caller-supplied),
`python/marrow/_dylibs.py` and `MARROW_DYLIB_DIR`, the wheel staging in
`python/build.py`, `compile.py`'s duplicated soname tables and the drift test
that polices them, and most of `utils/dylib.mojo`. `delocate`/`auditwheel`
would find the libraries in the load commands by themselves, which is the whole
reason that staging exists. Calls become `external_call["ZSTD_decompress", Int]`
or the typed `@extern("ZSTD_decompress") def ... abi("C")` form, so the
signature is checked where today `call["ZSTD_decompress", Int]` is not.

Mojo supports it: `-Xlinker -l<name>` reaches `ld`, and marrow already passes
`-Xlinker -lm` on Linux (`devkit/mojo.py`). Per-binary, so it is a
`BuildOptions` change.

**What blocked it, and what changed.** The objection was that there is no
portable optional link — a `DT_NEEDED` / `LC_LOAD_DYLIB` entry resolves before
`import marrow` returns, so a missing codec stops the process starting instead
of raising. macOS has `-weak-l`; ELF has no per-symbol equivalent and Mojo
exposes no `weak` attribute. That argument was strong when the codecs were
present only by luck. It is weaker now: they are `[package.run-dependencies]`,
so a conda install has them by construction, and graceful degradation is a
safety net rather than the mechanism. The remaining questions are the wheel
(which vendors its own copies today and would instead need them as real
linked deps) and anyone building from source without the dev libraries.

**`libopendal_c` cannot follow** and must stay `dlopen`ed: `publish = false`
upstream, no conda package, and genuinely optional. So this is a codecs-only
change and the two mechanisms would coexist — which is the honest cost, and
the reason it has not been done yet rather than a reason never to.

Order of work if picked up: link the codecs behind a `BuildOptions` flag,
measure the size gate and the wheel, then delete the staging only once both
platforms are green.

### 1.9 The Parquet reader, after page-level pruning landed

A `RowSelection` now narrows what is *fetched*, not just what is decoded: the
`OffsetIndex` plans the byte ranges, adjacent pages merge into one read, and a
skipped page is stepped over from the index rather than by parsing its header.
`marrow/parquet/tests/test_page_io.mojo` measures this with a recording
`ByteSource` -- the only way to tell "returned the right rows" from "did less
work". What is left:

- **A remote `read_ranges` fetches its ranges serially.** The storage seam is
  done -- `marrow/io/` owns `ByteSource`/`ByteSink`, both formats read and
  write through them, `DynSource`/`DynSink` pick a backend from the URI scheme,
  and `ParquetScanOperator` holds a `ParquetFile[DynSource]` so a *plan* can
  scan `s3://` (measured: `query_cli` +50,048, +1.64%; every other gate under
  800 bytes). `ParquetFile.read` also plans first and issues one `read_ranges`
  before the fan-out, so the decode workers borrow a value nobody mutates,
  which is the property `Fetched` exists to give.

  What is left is inside `OpenDalSource.read_ranges`: it issues its fetches one
  after another, where parquet-rs hands the whole set to
  `object_store::ObjectStore::get_ranges` and they go out together. On a local file that is
  free; on S3 it is the difference between one round-trip time and N. The shape
  to copy is the pattern at `reader.mojo`'s existing fan-out -- a pre-sized
  `List[Optional[Buffer]]`, disjoint slots, `sync_parallelize`, a per-worker
  `Optional[Error]`.

- **`pytest marrow/tests/test_ipc.mojo` on its own deadlocks the compiler.**
  `%cpu=0.0`, RSS flat at ~900 MB, CPU time frozen at ~13.7 s while elapsed
  grows, no diagnostic -- the signature CLAUDE.md records for the `__eq__`
  instantiation cycle. It reproduces on `cb296c82` with no local changes, so it
  is not new, and it is invisible day to day because every routine invocation
  selects that file *alongside* `marrow/parquet/tests`, and that larger unit
  compiles in ~110 s. One selection is one compilation unit, so the smaller
  selection is a different unit and only it deadlocks. Not yet narrowed to a
  case: the first 18 cases compile in 33 s, and both halves of the remaining 19
  hang. Anyone touching `ipc.mojo` must run it in the combined selection or
  they will read the timeout as their own breakage -- as happened here.

- **A Parquet file is still staged whole before it is written.** `ColumnWriter`
  records `data_page_offset`, `dictionary_page_offset` and every
  `PageLocation.offset` as `len(out)`, an absolute file offset, so
  `FileWriter`'s `BufferedSink` cannot flush between row groups: the staging
  buffer would restart at zero while the footer went on claiming file
  positions. Teaching `ColumnWriter` its base offset is what unlocks streaming,
  and would drop peak residency from the file to one row group. The IPC writers
  already stream, because their only absolute offsets are the `_Block`
  positions the writer itself computes.

- **The page index is fetched more than once.** `expr.page_selections` decodes
  the whole file's page index to choose pages, and then `read` fetches and
  decodes each chunk's `OffsetIndex` again (`_chunk_offsets`) to locate them --
  one extra round trip per (selected row group, leaf) on top of one for the
  file. Visible in `test_a_scattered_selection_fetches_one_range_per_run`,
  which has to exclude it to count the data reads. The fix is for the selection
  to carry the locations it already read, which `RowSelection` cannot do
  without learning about Parquet; a `PageIndex` cache on `ParquetFile` is the
  smaller change. It is coupled to hoisting the *planning* out of the parallel
  decode loop -- `locs` and the byte ranges are pure functions of the footer,
  so they can be computed once before dispatch while the fetches stay in the
  workers, and every chunk's `OffsetIndex` lives in one contiguous page-index
  region, so a hoisted plan is what makes one read replace N.

- **`LIMIT` never becomes a `RowSelection`.** `LimitOperator.done()` stops the
  driver, so row groups past the limit are never opened -- but within the first
  surviving group every row is decoded. `limit 10` over a million-row group
  reads the million. The `OffsetIndex` machinery to read ten rows' worth of
  pages now exists; what is missing is the limit reaching the scan as a row
  range, which is the same `row_limit` channel top-K needs.

- **The comptime lane prunes numerics only.** Temporal and decimal predicates
  prune through the runtime lane, which recovers the dtype from the index and
  dispatches. In the fused lane `TemporalCompare` inherits `Value.mask`'s
  default and keeps every chunk, and a `DecimalValue` column has no comparison
  node to prune with (§2.6). Temporal would need a dtype *instance* to
  prune with, since a temporal type carries a unit and `Stat()` does not exist
  where `NumericType(Defaultable, ...)` makes it free; that was judged not worth
  a required trait member for one dtype family.

- **A `RowSelection` is copied per (row group x leaf) and walked per page.**
  It is a `List[Bool]`, one byte per row, `.copy()`-ed into every
  `ColumnReader` -- a megabyte per leaf on a million-row group -- and
  `last_selected` rescans it per leaf. Sharing it behind an `ArcPointer` and
  caching `last_selected`/`num_selected` at construction removes both; a prefix
  sum would make `selected_in` O(1) as well. Nothing measured yet, so this is
  a shape complaint rather than a profile.

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

- **A dotted type name sets `__module__` but breaks attribute lookup.** CPython
  3.14 emits `DeprecationWarning: builtin type X has no __module__ attribute`
  once per registered type -- 13 per import today. Passing `"probe.Dotted"` to
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

### 1.11 Undocumented subsystems

Ten substantial pieces of the codebase have no design document and never did:
the whole Parquet subsystem (ten modules, ~490 KB), the Arrow IPC layer, the C
Data Interface, the GPU execution model, `utils/argparse.mojo` (769 lines),
`Groups` (in `kernels/groupby.mojo`), the decimal cast family in
`kernels/cast.mojo`, `Dispersion`, the
`comptime/temporal.mojo` nodes, and the `_drop` destructor trampoline on every
erased box. Listed so that "there is no doc" is not mistaken for "there is no
feature".

---

### 1.12 Two findings not covered by any row above

**The binary-size gate is blind to more than half its own programs.** Three
times a plan has linked `kernels::cast` without needing it — through hashing,
through `ParquetScan`, and through `sort_indices`, the last costing 694 cast
symbols and about 3 MB in every AOT binary that sorted anything. Each was found
by hand, because **no gate program sorts**: `benchmarks/binary_size/query_sort.mojo`
exists but `baseline.json` has no `query_sort` entry, so the gate never builds
or compares it. Fifteen sources, eight gated — `query_arith`, `query_exprs`,
`query_param`, `query_runtime`, `query_scan`, `query_scan_typed` and
`query_sort` are all ungated. Adding the baseline entries, not the programs, is
the task, and until it is done the next instance is equally invisible.

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

#### 1.2 CSV and JSON readers

**What exists.** No reader. The only CSV code under `marrow/` is `QueryCli`'s
output writer (`render_csv`, `marrow/expr/cli.mojo`).

**What it would take.** marrow already has the hard parts: `Buffer`,
`BufferView`, every builder, and `LittleEndian.fixed` as the byte-order
primitive. What is new is a tokenizer, a sampling inference pass, and a
type-widening lattice. NDJSON is the same shape with a different tokenizer.
**This is the highest ratio of user value to engineering novelty on the whole
page.**

#### 1.3 Datasets: multi-file, partitioned, remote

**What exists.** `scan(path: String, schema: Schema)`
(`marrow/expr/builders.mojo:727`) — one file, and the caller supplies the schema
because "a `Relation` is a description and must not touch the filesystem to
exist".

Storage itself is no longer the gap: `marrow/io/` owns the seam, and
`DynSource`/`DynSink` pick a backend from the URI scheme.

**What it would take — two pieces left, both local.** (a) Derive a `Schema`
from the Parquet footer so `scan(path)` needs no schema — small; everything
needed is in `marrow/parquet/schema.mojo`, and it is row 3 of the table.
(b) A `MultiFileScan` relation node owning a list of sources and yielding row
groups across them, plus hive-path parsing to synthesise partition columns.
Both are now strictly harder than the remote piece was, which inverts this
section's original ordering.

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

plus constant folding in the `RuntimeValue` constructors. Parquet statistics
pushdown is `PushFilterIntoScan`, in the same list.

The rule set is a comptime parameter, so a binary links exactly the rules it
names and `execute()` alone optimizes nothing. `DynRelation` became **a variant
for inspection and a trampoline for lowering**: `isa[R]()`/`get[R]()` let a rule
read a real typed node and construct one, while `to_operator` stays on a
per-type slot — routing it through the variant instead cost **+348%** of
`__text` on `query_streaming`, because that ladder instantiates every node's
lowering and `ParquetScan.to_operator` reaches `kernels::cast` in a plan with no
Parquet in it.

**Still absent:** common-subexpression elimination, duplicate group/sort key
elimination, statistics propagation, aggregate pushdown, and any cost model.

**Blocked in the kernel, not the optimizer:** join reordering and build-side
selection. `Join._output_schema` is positional (left fields then right) and
`JoinOperator` hardcodes build=left, so both rewrites change the output column
order and are not expressible as plan rewrites at all. They need
`kernels/join.mojo` to accept an output ordering.

A credible engine ships without a cost model, so none of this is urgent — but
the `count_star()` hazard in §1.1 is the mirror image of it: the same
expression that blocks projection pushdown is the one an optimizer most wants
to special-case. `ColumnPruning` clamps rather than special-cases, never
narrowing a source to zero columns, because a `RecordBatch` carries its row
count in its columns. Fast count-star remains uncopied and the desugaring is
unchanged.

#### 1.6 String and temporal function coverage

**What exists.** 31 string kernels (case, strip family, trim chars, reverse,
capitalize, byte and character length, ascii, starts/ends/contains, position,
six comparisons, `LIKE`/`ILIKE`, substr/left/right, repeat, pad, replace,
split_part, and `ConcatKernel` behind `||` in both lanes) and 15 temporal
extractors plus `date_trunc`.

Adding one is cheap and reaches every caller: `UNARY_VERBS`/`BINARY_VERBS`/
`TERNARY_VERBS` (`marrow/expr/runtime/values.mojo:1896`) is the single
vocabulary, and a verb added there is callable from Python without touching the
bindings or `python/marrow/expr.py`.

**Still absent — strings:** the `concat` function, which skips null
arguments where `||` propagates them, `concat_ws`, and the whole regex family.

**Still absent — temporal:** `date_diff`, interval arithmetic,
`strftime`/`strptime`, `make_date`, `age`, and timezone attachment. Timezones
are carried on the type (`dtypes.mojo:401`) and **ignored by every kernel** —
`marrow/kernels/temporal.mojo:8` states a non-UTC timestamp is decomposed in
UTC.

ibis's `strings.py` is the engine-level expectation: case, trim/pad,
substring/slice, find/predicate, pattern match, regex (extract/split/
replace), replace/split/join, and URL parsing.

**What it would take.** What remains is the hard half. Regex needs a real
engine — `mojo-regex` was evaluated and rejected on *correctness*, not
availability (it never enters an optional group, so `(?:www\.)?` is skipped) —
and timezone conversion needs a tz database. `concat`/`concat_ws` is the one
cheap item left: `||` already runs through `ConcatKernel` in both lanes, and
what is missing is the null-skipping variant.

#### 1.7 Known-wrong answers in core operations

The float group-key and integer `//`/`%` entries that stood here are **merged**,
in `136b3529`, and the golden corpus now carries **zero `xfail`s** — every case
it compiles, marrow answers the way DuckDB does. `/` by zero was not part of
that fix and landed separately on 2026-09-22 — floats answer `inf`/`-inf`/
`nan`, and `pc.divide` raises on an integer zero divisor as pyarrow does.

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
`groupby.mojo` (3), `parquet/reader.mojo` (3), `sort.mojo` (2), `join.mojo` (2)
and `filter.mojo` (1), outside `execution.mojo`, which defines them.
**Group-by placement is parallel**, as of the radix-partitioned `HashGrouping`
— one `SwissHashTable` per partition of the key hash's top 6 bits, so no
aggregate state is ever split and no merge step exists. Aggregate *accumulation*
is still serial, and deliberately: a thread-local partial would need a `merge`
on every `AggKernel`, which `mean` and the Welford triple make non-uniform and
exact `count_distinct` makes impossible. There is no pipeline parallelism:
`Pipeline._flow` pushes one morsel through the stages on the calling thread.

**What it would take.** True pipeline parallelism is the remaining item: the
push `Operator` contract is a good foundation, but nothing owns a task queue
today. Two group-by knobs are also uncalibrated — `_MIN_DISTINCT_RATIO` (0.9)
and `_PARALLEL_GROUPBY_MIN_ROWS` (60,000) have no measurement in the tree, and
`bench_groupby.mojo` has no row-count tier between 1M and 10M to find the
crossover with.

#### 2.2 Larger-than-memory execution

**What exists.** Nothing. No spill, no memory pool, no accounting, no limit —
`grep -in 'spill\|memory_pool\|memory_limit'` over `marrow/` returns only
unrelated bitmap-test strings. `execute()` drains the entire plan into one
`RecordBatch` (`marrow/expr/logical.mojo:1408`), so even a streaming plan
materializes its full result, and **there is no batch-iterator result API** even
though `drain()` is exactly that shape internally.

The old spilling streaming engine was removed and not replaced. What it would
take includes spilling variants of group-by and sort.

#### 2.3 Relational operations that have no node

Each is a missing `Relation`, not a missing kernel. ibis's `relations.py` is the
canonical list; marrow has 8 of it.

| Missing | Golden cases | Note |
|---|---|---|
| `UNION ALL` / `UNION` / `EXCEPT` / `INTERSECT` | 4 | ibis models these as one `Set(left, right, distinct: bool)`. They also treat NULL as equal to itself, which nothing else in marrow does |
| `Distinct` / `.unique()` | 1 (`DISTINCT ON`) | Expressible today as `aggregate(keys=[...], aggs=[])`, which is exactly what the SQL front end desugars `SELECT DISTINCT` into — so the semantics are reachable through `Sql.plan` and there is still no verb and no `unique` kernel |
| `GROUPING SETS` / `ROLLUP` / `CUBE` | 3 | `Aggregate` carries one key list; `ROLLUP` also needs `GROUPING()`. Implementable as a rewrite into an aggregation cascade |
| `explode` / `unnest` | 1 | Row-multiplying, so a new operator shape. ibis has a dedicated `TableUnnest` with `offset` and `keep_empty` |
| `Sample`, `DropNull(how)`, `FillNull` as relations | — | ibis has all three as nodes |
| `top_k` / `bottom_k` as first-class | — | A dedicated streaming node beats rewriting sort+limit: `TopN` bounds the sort, but `SortOperator` still buffers every row before ordering |
| `merge_sorted`, `rolling`, `group_by_dynamic`, `upsample` | — | Time-series reshaping — a common ask in that domain |

#### 2.4 Join breadth

**What exists.** Hash equi-join in six kinds — inner, left, right, full, semi,
anti — over a Swiss table with a CSR probe index
(`marrow/kernels/join.mojo:190`, `hashtable.mojo:76-87`), with radix partitioning
and parallel probing. Multi-column keys work because keys go through
`StructArray`.

**Declared but rejected:** `JOIN_CROSS`, `JOIN_MARK` and `JOIN_SINGLE` are
`JoinKind` constants whose `is_supported()` is False; `hash_join` raises for
all three, which `test_join.mojo` pins.

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

**Storage is complete, and every type enters an expression** — each has a
column, literal and parameter leaf — **but almost nothing computes on the
nested ones or on decimal.**

- **Nested:** eight golden cases — list element access, list slice, `unnest`,
  list sum, struct field access, map lookup, map cardinality. The nested verbs
  that exist are `array_length` and `array_contains`; everything else needs a
  kernel that does not exist. ibis's minimum here is `ArrayIndex`, `ArraySlice`,
  `ArrayContains`, `ArrayLength`, `Unnest`, `MapGet`/`MapContains`/`MapKeys`/
  `MapValues`/`MapLength`, and `StructField` — and struct is genuinely thin
  there too (`structs.py` defines exactly two nodes), so `StructField` alone
  closes most of the struct gap.
- **Decimal:** a decimal column, literal and parameter enter the comptime lane
  as `DecimalValue` (`col("price", decimal128(10, 2))`), but **nothing computes
  on one**: there is no decimal arithmetic, comparison, cast or aggregate node,
  though all four decimal widths and every decimal cast kernel exist. Each
  node has to align scales before it adds or compares. For a library aimed at
  analytics, "cannot compute on a money column" is close to disqualifying.

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

- **Compressed Arrow IPC bodies are unsupported.** `marrow/ipc.mojo:1417`
  raises on LZ4_FRAME/ZSTD bodies. Marked *unverified* as to how much it would
  buy on marrow's reader. - **No `__dataframe__` protocol**, though
  the PyCapsule/C Stream path marrow already has is the better-supported
  modern route.

#### 2.10 Operability

- **No `EXPLAIN ANALYZE`, no per-operator metrics, no profiling hook, no
  progress, no cancellation.** An operator cannot be interrupted mid-`drain`.
 - **No
  error taxonomy.** 373 `raise Error(...)` sites across `marrow/` produce
  plain strings with no type. The messages themselves are good — they name the
  verb and the column (`"drop: column 'x' not found in schema"`) — but a
  Python frontend cannot map them to distinct exception classes. That is cheap
  to copy and should land with the frontend, not after it. - **Per-key null
  placement is missing on sort:** `Sort` carries one `nulls_first: Bool` for
  all keys (`logical.mojo:1871`); ibis's `SortKey` carries it per key.

---

### Tier 3 — differentiating

Where marrow could be better than anything that exists.

#### 3.1 The comptime / AOT lane — the real one

**What it is.** In the comptime lane a node's operands are bound on a family
trait, its output dtype is a comptime type, and a whole subtree fuses into one
SIMD loop with nothing erased. `col("a", int64).sum()` resolves to
`Aggregate[Fold[SumFold, Int64Type], NumericColumn[Int64Type]]`, so the plan holds a
direct `AggState[SumFold, Int64Type]` and no per-dtype resolution ladder is
reachable in the binary at all.

**The measurement** (`benchmarks/binary_size/baseline.json`, `-O3 -g0`,
stripped, `__text`):

| Gate | Bytes | |
|---|---:|---|
| `query_streaming_agg_fused` (comptime) | 1,452,744 | |
| `query_streaming_agg` (runtime-named) | 13,204,780 | **9.09x** |
| `query_dynvalue` (erased values) | 9,888,172 | |
| `query_streaming` (fused filter + project floor) | 1,451,532 | |
| `query_cli` (the AOT lane as a program) | 2,973,356 | |

Re-read from `baseline.json` on 2026-09-14 (mojo 1.2.0.dev2026091405). In
2026-08 the ratio was 6.71x, with `query_dynvalue` at 6,227,524, so the gap has
**widened**, not narrowed. That is the erasure boundary doing its job, but the
gate compares each change only against the last recording, so the runtime
lane's floor drifts one accepted re-recording at a time — worth a note the day
a `query_dynvalue` regression matters.

The runtime lane's cost is not incidental: it links the whole name-resolution
ladder and, through it, `marrow.kernels.cast` — 693 cast symbols in
`query_dynvalue` alone.

**The second half nobody else has.**
`marrow/expr/comptime/tests/test_schema_handle.mojo` pins four compiler
contracts and proves that a schema can be a comptime parameter and that
`__getattr_param__` can return a *conditional* type carrying its trait bound.
So `t.amount` resolves to `NumericColumn[Float64Type]`, `t.qty` to
`NumericColumn[Int64Type]`, and `t.amont` is a compile error reading `constraint
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

**A parameter's command-line spelling stops at numeric, bool and string.**
`738146e8` made a plan's `param()`s its options, and `ParamSpec.parse` is
`None` for every other family — so a temporal or decimal parameter raises
naming itself rather than being readable from argv.

**What it would take to be a product.** One thing: **promote the schema handle
from spike to public API.** The wrapper it used to wait on is `marrow/expr/cli.mojo`
already. This is the only story on the page that is genuinely unavailable
elsewhere.

**Honest counterweights.** 1.45 MB of `__text` is small for a query engine and
not small in absolute terms; the binary still links `libmax`/AsyncRT with GPU
codegen off. The fused lane requires the schema at compile time, which most
workloads do not have. And the 9.09x is measured on one query shape on
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

### There is no GPU hash join or group-by today

Worth stating, because the plumbing reads as though there might be:

- `marrow/kernels/join.mojo` parallelises on the CPU only:
  `ctx.worth_parallel` picks serial or partition-parallel, a context that
  targets a GPU gets the serial path, and the device reaches only the
  join's internal kernel dispatches, never the hash table.
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
   established (`ExecContext.gpu(ctx)`, `has_accelerator_support[...]`
   gating, etc. — see `marrow/views.mojo`).

### Status

Speculative, two levels removed from "ready to prototype." The precondition
(GPU hash join existing) isn't met yet — resolve that design question
first, independently of whether `match_any`/`match_all` end up being useful
inside it.

---

## 4. Unexplored — `wrap_host_memory()` for zero-copy uploads

### What landed

`DeviceContext.wrap_host_memory[dtype](host_ptr, size) -> DeviceBuffer[dtype]`
arrived with the `dev2026091405` → `dev2026092105` toolchain bump (upstream
`7688979e18`, *Expose Host Memory Wrapping in DeviceContext*). It makes a range
of the caller's host memory device-accessible without allocating device memory
and copying into it.

### Why marrow would want it

`Buffer.to_device` (`marrow/buffers.mojo:910`) is `enqueue_create_buffer` plus
`enqueue_copy` — every upload allocates a device buffer and copies the whole
range into it, and `Bitmap.to_device` and `Array.to_device` all funnel through
it. CLAUDE.md's measured guidance is that transfer cost dominates and that
uploading per call is 2-3x slower than just staying on the CPU; this is the API
that removes the copy rather than amortising it.

### Why it does not work on this machine

**Metal requires a page-aligned base and a page-multiple length.** Marrow
allocates at `alignment=64` (`marrow/buffers.mojo:501`) because that is Arrow's
rule — `PoolBuffer::RoundCapacity` is `RoundUpToMultipleOf64` — and the page on
Apple Silicon is 16 KiB. So Metal rejects every buffer marrow owns, and the
development machine is Metal. It works on CUDA and HIP, where the wrap also
page-locks the range (`cuMemHostRegister` / `hipHostRegister`) and so buys DMA
overlap on top of the elided copy. Raising `Buffer`'s alignment to a page would
unblock Metal, but that is a layout decision with its own cost — it is not a
kernel change, and 64 is there for a reason.

### Why it is not a drop-in even where it is supported

The contract does not match what `Buffer` models today:

- It grants **access, not ownership**. The returned `DeviceBuffer` does not keep
  the host allocation alive — the origin is cast away — so the owner must
  outlive every enqueued transfer and kernel touching the range.
- The range must be addressed **through the returned buffer**, not through
  `host_ptr`. Only on CUDA are the two the same address.
- Dropping the buffer *queues* the release, so every context that copied the
  range needs `synchronize()` before the host memory is unmapped.

`Buffer` treats residency as a **kind** — CPU / FOREIGN / MAPPED / HOST or
DEVICE — with exactly one active release mechanism per `Allocation`, and
`to_device` answers with a new immutable `Buffer`. A wrapped range is neither
side of that: one allocation, two addresses, and a lifetime borrowed from
something else. That is a new `Allocation` kind carrying a borrow, not a sixth
enum value added to the existing five.

### What the actual spike is

1. Decide whether marrow wants a *borrowed-device* residency at all, or whether
   the answer to transfer cost stays "upload once, run several kernels
   device-resident, download at the end" — which is what the performance
   guidance already says and which needs no new API.
2. If yes: it is a Linux/NVIDIA-only path until the alignment question is
   settled, so it cannot be developed or benchmarked on the current machine.
   Model the lifetime first — `Allocation` already checks its release rules in
   `__del__`, and a borrowing kind has to answer them.
3. Only then wire it behind `comptime if GPU_ENABLED`, like every other device
   path.

### Status

Unexplored, and blocked on hardware before it is blocked on design. Recorded
because the API is new and the copy it removes is the one thing measurement
keeps pointing at — not because the precondition is met.

---
