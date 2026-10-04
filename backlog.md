<!--
Copyright 2024 Szűcs Krisztián
SPDX-License-Identifier: Apache-2.0
-->

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
| 1 | **CSV reader** | A first user arrives with a CSV, not a Parquet file. NDJSON reads (`marrow.json`, §1.2); CSV does not | **M** | — |
| 2 | **Declared error types** — every raise site raises an `ArrowError` kind, but ~1,800 signatures still declare bare `raises` | A bare frame keeps only an error's text, so a caller recovers the kind by parsing it (`DynError(e)`) instead of catching a type, and Python gets it through the same parse. Migrate bottom-up: a function declares its kind (`raises CorruptError`) or `DynError` once nothing it calls raises a bare `Error`; dispatch ladders forward `raises E`. It also pays back a size cost: a kind raised in a bare frame is converted to `Error` inline at the site, which put +21 KB (+1.5%) on `query_streaming_agg_fused` and `query_expr2_agg_fused`, mostly in the `dispatch_*` and `DynBuilder._dispatch_mut` ladders | **L** | — |
| 3 | **`scan(path)` without a hand-written schema**, then globs, directories, hive partitions | `scan()` takes one path *and* demands the schema by hand. Every real Parquet dataset is a directory | **M** | 1 |
| 4 | **Join reordering** — no *search* over a join tree | The largest TPC-H win available. Every precondition has landed and two rewrites spend the cost: `SelectBuildSide` picks the side to index, `JoinReassociation` does one local association, and a footer's `distinct_count` now reaches `ColumnEstimate.ndv` so the cardinality term is visible wherever a writer recorded one. What is left is the **enumeration** — choosing among the Catalan-many associations of an *n*-join chain — which is a `prepare` pass rather than a `Rule` | **L** | — |
| 5 | **CSE and duplicate group/sort key elimination** | Needs no `DynValue` equality slot: `WindowExpr.spec()` already compares erased expressions by rendering them through the existing, non-raising `_write` slot, so duplicate key elimination is a `Rule` comparing renderings. What blocks it is that rendering is not faithful — see §1.4 | **M** | — |
| 6 | **Larger-than-memory execution** — no spilling anywhere | Every aggregate and join is bounded by RAM. Changes the operator contract | **XL** | — |
| 7 | **Nested-loop / range joins** | Only equijoins exist, so a non-equi predicate has no plan at all | **M** | — |
| 8 | **UDFs** | The escape hatch that makes a missing kernel survivable rather than fatal | **M** | — |
| 9 | **A row format** | Needed by sort-merge join, spilling, and any wire protocol | **L** | — |

---

## 1. Open work

Ordered by value. Some of it is partly done -- §1.9 in particular tracks a
subsystem that moved twice this week -- and each entry says which part.

### 1.1 Latent traps — nothing currently answers wrongly

**`count(*)` reads no column, and a `RecordBatch` cannot tell that from "no
rows".** `builders.mojo` desugars it to `lit(1, int64).count()`, whose
`columns()` is correctly empty — the defect is that `RecordBatch` has no row
count and derives one from `columns[0]`, which `to_struct_array` copies into
`StructArray.length`. So a zero-column batch does not merely *report* zero
rows, it becomes zero rows one node into the plan. `ColumnPruning` never
narrows a source to zero columns, keeping the first when the demand is empty,
which is what keeps every answer right today at the cost of one decoded column
per count-star query.

**There is no contained fix.** Operators compute in `StructArray`, which
carries an explicit `length` already (`_struct_of` in `physical.mojo` takes
one), and `RecordBatch` appears at exactly two boundary functions —
`to_struct_array` in, `from_struct_array` out. So the in-memory half is
`tabular.mojo`: an explicit `num_rows` on `RecordBatch` (what Arrow specifies
and pyarrow does), additive enough not to touch its 56 construction sites,
carried through `from_struct_array`, `select` and `slice`. What that does *not*
cover is the source the clamp exists for: a `ParquetScan` narrowed to no
columns has to answer its row count from the footer rather than from a column
it no longer decodes. Dropping the clamp for the in-memory source alone would
make `ColumnPruning`'s rule depend on which source is below it, which is a
worse rule than the uniform clamp — so the two halves land together or not at
all.

### 1.2 Known-wrong answers

- **A NULL dictionary entry joins NULL to NULL** (unconfirmed: found by
  review, no test yet). `JoinHashTable.probe` (`kernels/join.mojo`) skips a
  null probe key through `key.null_count()` and `NotNullKernel`, and for a
  dictionary column both read only the *index* validity. A row whose index is
  valid but points at a NULL dictionary value decodes to NULL, and
  `DictionaryEncoder` treats NULL as equal to NULL, so it matches the build
  side's NULL key: an INNER join emits a NULL = NULL pair and LEFT/ANTI joins
  get the wrong unmatched rows. SQL `=` matches nothing on NULL. Not known
  whether the pre-encoder join behaved the same. `count_distinct` and the
  grouped `DistinctCount` share the index-only null test, so a NULL entry
  behind a valid index counts as a distinct value there too — that one
  predates the encoder. The fix is a logical null test for dictionary
  columns (index valid *and* entry valid), in one place both callers use,
  plus a test with a NULL dictionary entry behind a valid index.

### 1.2b Found by the differential property tests (2026-10-04)

`python/marrow/tests/test_properties_{compute,io,cdata}.py` run marrow against
PyArrow under Hypothesis. Every item below is a strict `xfail` there with its
minimal input, so fixing one flips its test; the properties themselves step
around the pinned inputs. Line numbers are as of `2c28f9bb`.

**Process crashes** (each reproduced in a child process):

- **Parquet write of any `large_list` column aborts** — `get: wrong variant
  type`. `parquet/schema.mojo:309` and `:428` read the column with
  `as_list()`, which holds only `ListArray`. (`test_parquet_write_large_list`)
- **Parquet read of a v2 page under a list over-reads the page.**
  `parquet/reader.mojo:529` takes the present-value count from the header's
  `num_nulls`; PyArrow's count leaves empty and null lists out, so for
  `list<struct<decimal128(9,2)>>` holding `[]` marrow decodes a value past the
  end (`codecs.mojo:504` / `reader.mojo:1824` asserts). The v1 path counts
  `def == max_def` instead (`reader.mojo:388`). Primitive leaves read past the
  values silently. (`test_parquet_read_v2_page_under_empty_list`)

**Silently wrong data:**

- **Parquet: a column whose values are all equal is unreadable by PyArrow** —
  every one-row column, among others. A one-entry dictionary gets index width
  0 (`parquet/writer.mojo:606`) and `Rle.encode` then emits no run at all
  (`parquet/codecs.mojo:321`); PyArrow wants a run header even at width 0.
  marrow reads its own file back. (`test_parquet_write_single_valued_column`)
- **Parquet: `timestamp[s]` is written under a nanosecond annotation and
  `time32[s]` under a millisecond one**, values unscaled — 1 s reads back as
  1 ns / 1 ms, in marrow and PyArrow alike (`parquet/schema.mojo:1123`, `:1097`).
  (`test_parquet_write_seconds_unit`)
- **Parquet: a null struct holding a list is written as a present struct with
  an empty list** — writer side, both readers agree. Suspected in the struct
  arm of `_shred_elem` (`parquet/schema.mojo:401`). (`test_parquet_write_null_struct_over_list`)
- **Parquet: dictionary encoding merges `0.0` and `-0.0`** — `Dict` keyed by
  float value (`parquet/codecs.mojo:622`), so the later sign is lost.
  (`test_parquet_write_keeps_signed_zero`)
- **Parquet: a zero-row list or map column reads back as one empty list** —
  `_fold_list_offsets` appends the closing offset unconditionally
  (`parquet/schema.mojo:256`). (`test_parquet_read_zero_row_list`)
- **IPC file writer reuses the first batch's dictionary for every batch**
  (`ipc.mojo:2234` skips a written id without comparing); later batches decode
  against the wrong dictionary. PyArrow refuses the replacement; the stream
  writer is fine. (`test_ipc_file_dictionary_replacement`)
- **IPC read of PyArrow's `float16` gives `float64`** with garbage values: an
  absent `FloatingPoint.precision` defaults to DOUBLE (`ipc.mojo:1566`), but
  the flatbuffer default PyArrow omits is HALF. (`test_ipc_read_float16`)
- **`cast(safe=True)` misses out-of-range integers.** The check is a
  round trip (`kernels/cast.mojo:164`), which a sign change survives: int8
  `-1` -> uint8/uint16 is 255/65535, uint8 `200` -> int8 is -56. For float ->
  int it depends on what an out-of-range `fptosi` returns: float64 `128.0` ->
  int8 gives -128, float64 `2**63` -> int64 saturates, float16 `inf` -> int32
  saturates. (`test_cast_sign_change_is_checked`,
  `test_cast_float_to_int_out_of_range`)
- **`cast` string -> integer wraps** — `'128'` -> int8 is -128 even with
  `safe=True`: `Scalar[native](atol(s))` truncates (`kernels/cast.mojo:1006`).
  (`test_cast_string_to_integer_out_of_range`)
- **Temporal casts only rescale the tick** (`TemporalCastKernel`,
  `kernels/cast.mojo:763`): timestamp -> date64 keeps the time of day,
  timestamp -> time does not reduce to a day (-1 s -> -1000 ms), timestamp ->
  date32 with `safe=True` raises where PyArrow drops the time of day, and an
  unsafe downscale floors a negative tick where Arrow truncates
  (`kernels/cast.mojo:954`). (`test_cast_timestamp_*`)
- **`cast` float -> decimal rounds** — `round(x * 10**scale)` in float64
  (`kernels/cast.mojo:557`), so an exact `14411518807587.0` -> decimal128(20,4)
  gains `.0016`. (`test_cast_float_to_decimal_is_exact`)
- **`sum`/`product` of unsigned integers accumulate as int64**
  (`kernels/aggregate.mojo:323`); PyArrow answers uint64, so a total past
  `2**63` comes back negative. (`test_unsigned_sum_is_uint64`)
- **`min`/`max` start from `±MAX_FINITE`** (`kernels/aggregate.mojo:372`,
  `:386`): all-NaN input answers ±FLT_MAX instead of NaN, `min([inf])` answers
  FLT_MAX. (`test_min_max_identity`)
- **`take` answers null for an out-of-bounds or negative index** instead of
  raising (`kernels/filter.mojo:1224`). (`test_take_out_of_bounds_raises`)
- **`divide` INT_MIN / -1** wraps to INT_MIN (`kernels/numeric.mojo:298`);
  PyArrow answers 0. Signed overflow in `sdiv` is undefined in LLVM and
  x86-64's `idiv` faults on it, so this may be a crash there (measured only on
  arm64). (`test_divide_int_min_by_minus_one`)
- **`sort_indices` is not stable from 32 elements on** — `stable` defaults to
  False (`kernels/sort.mojo:416`) and PDQsort takes over. Its float key orders
  NaN as the largest value (`:112`), where PyArrow keeps NaN beside the nulls,
  and orders `-0.0` before `0.0`, so `sort_by` lets the sign of zero decide
  instead of the next key. (`test_sort_indices_is_stable`,
  `test_sort_indices_nan_placement`, `test_sort_by_signed_zero_ties`)

**Divergences from PyArrow's defaults:**

- `RecordBatch.sort_by` puts nulls first by default
  (`python/bindings/tabular.mojo:397`); PyArrow puts them at the end.
- `compute.any`/`all` answer False/True for empty or all-null input
  (`kernels/aggregate.mojo:574`, `:612`); PyArrow's `min_count=1` answers null.
- `upper` and `capitalize` apply the full case mapping, `'ß'` -> `'SS'`
  (`kernels/string.mojo:226`, `:283`); PyArrow maps one code point to one
  (`'ẞ'`), and `capitalize` should give the titlecase `'Ss'` either way.

Not bugs, recorded so nobody re-derives them: `count_distinct` counts `0.0`
and `-0.0` once (hashing canonicalises on purpose, `kernels/hashing.mojo:163`);
float and timestamp -> string formatting differs from PyArrow's; unparseable
strings cast to null under `safe=False` where PyArrow raises; int -> decimal
checks values where PyArrow refuses by precision; decimal -> float is closer
to the nearest float than PyArrow's.

### 1.3 Latent compiler hazards

**More `t"…{dtype}"` sites under `marrow/kernels/`.** A t-string
interpolating a recursive `Writable` value inside a function-level recursion
cycle deadlocks the compiler. Three sites were fixed; the rest are untested and
sit in the same shape.

**`DynRelation`'s mangled names sit near Apple `ld`'s 1 MiB cap.** Mojo spells
every variant member's layout into some symbol names (`Optional[DynRelation]`'s
`Variant` constructor among them). With an extra scan node holding a path, a
schema, a string and a function pointer inline, two names reached 1,051,066
bytes, `ld` asserted (`name.size() <= maxLength`) and `libmarrow.so` failed to
link; the tree builds without it. A new relation node with large by-value
fields can trip it; holding them behind an `ArcPointer` is the known fix.

### 1.4 Engine capability the golden corpus measures as missing

`golden/COVERAGE.md` is authoritative: 286 cases, of which **51 carry
`-- skip mojo`** because marrow has no API for them. Their bodies
are never compiled, so they are proposals rather than verified spellings.

Recounted 2026-09-27, by prefix:

| skipped | area |
|---|---|
| 12 | aggregates — median, quantile, first/last, arg_min/arg_max, string_agg, corr/covar, mode, skewness |
| 9 | temporal — date_diff, age, strftime/strptime, make_date, interval arithmetic |
| 8 | nested — struct field, map lookup, list element/slice, unnest |
| 7 | math — atan2 and the rest of the trigonometric family |
| 2 | string — concat_ws, null-skipping `concat` |
| 4 | joins — cross, non-equi, asof |
| 3 | GROUPING SETS / ROLLUP / CUBE |
| 3 | filters — SQL `NOT IN` null semantics, `.is_in` as a method |
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

### 1.7b The view layouts, where they are not native

`string_view` / `binary_view` read, write and compute natively in the core,
the C Data Interface, IPC, Parquet, the structural kernels (filter, take,
concat, cast, hash, equality, sort, min/max) and the hot string kernels
(length, the map transforms, concat, the predicates, LIKE/ILIKE). What is not:

- **The comptime lane produces `string` from a view column.**
  `StringViewColumn` fuses over the views, but its `Type` is `StringType` and
  `evaluate` casts, because every breaker string node downcasts its operands'
  materialised columns to `BinaryLikeArray[X.Type]`. Projecting a view column
  in the comptime lane therefore changes its type; the runtime lane keeps it.
  Fixing it means binding those nodes on `BytesArray` rather than on the
  offsets array.
- **The SQL argument functions cast through `string`**: `substr`, `left`,
  `right`, `repeat`, `lpad`/`rpad`, `replace`, `split_part` and
  `trim(chars)` (`StringArgKernel.dispatch`). Each builds a new string per
  row anyway. The measures (`char_length`, `ascii`, `position`) read views
  natively.
- **No header fast path for a string key in the offsets layout.** A
  `string_view` key is compared on its view's length and four-byte prefix
  before its bytes (`KeyCompare`, `kernels/hashing.mojo`); a plain `string`
  key compares bytes on every row, which put plain-string grouping at ~1.7x
  the old hash-only grouper (1M rows, 10k groups). A prototype kept a 16-byte
  header per group — length plus the first eight bytes — and checked it
  first: 10.7 -> 6.6 ms serial (-39%) and 9.9 -> 5.9 ms on 8 threads (-41%).
  Give plain strings a header in the view's shape — 4-byte length plus 12
  inline bytes, which also decides 9-12-byte keys. **The header decides alone
  only when it holds the whole key** (<= 12 bytes); a longer key that matches
  it still compares every remaining byte, as DuckDB's `string_t` and
  DataFusion's `ByteViewGroupValueBuilder` do.
  `test_collisions_long_strings_sharing_a_prefix` pins that case.
- **A written view column reads back as `string`** unless the reader passes
  `binary_type=binary_view` or declares the column a view in its schema: the
  writer does not store `ARROW:schema`.

### 1.8 Test and infrastructure gaps

- **No cross-lane parity test.** "One engine, two drivers" was enforced by a
  `test_parity.mojo` across four axes; it went with the previous expression
  package and has no replacement. The invariant is currently unenforced.
- **Group-by is covered at the kernel, not through the engine.**
  `kernels/tests/test_groupby.mojo` (39 cases) pins both placement paths
  directly on `DictionaryEncoder`, but nothing drives the radix path through
  `GroupedAggregateOperator`: every group-by case in `expr/tests`, `golden/` and
  `python/marrow/tests` is far under the 50,000-row gate, so the engine's
  wiring to the parallel path is untested end to end.
- **`mojo-regex` and `morrow` resolve only from git.** `pixi.toml` takes
  each from the `marrow` branch of a fork that builds on the pinned nightly:
  `kszucs/mojo-regex` (of `msaelices/mojo-regex`) and `kszucs/morrow.mojo`
  (of `mojoto/morrow.mojo`). A git source resolves for a source build and not
  for a conda install from prefix.dev. So before the next `v*` tag, publish
  both forks to `mojo-community`, or return to upstream once each is
  published and builds on marrow's Mojo.

**The vectorised zero-divisor scan has no caller left.** `//` and `%` answer
NULL by first asking "is there a zero in this column", and
`BufferView.__contains__` (`views.mojo:202`) is that question: SIMD, early
exit, per-chunk reduction — the variant measured fastest on 2026-09-04, where
the scalar form cost **+45%** on `bench_floordiv_int32_*`.
`marrow/tests/bench_views.mojo` exists to keep the scalar form from coming
back and says `RuntimeValue._null_zeros` and `DivisionBinary` both call it.

Neither does, and `grep -rn __contains__ marrow/` finds no production caller
at all. `DivisionBinary.bind` (`expr/comptime/numeric.mojo:217`) walks the
divisor a row at a time into a `Bitmap.alloc_zeroed(length)`, which is the
shape the +45% measured; `RuntimeValue._null_zeros` pays a `nullif` against a
broadcast zeros array instead. So the benchmark guards a helper nothing uses
while the regression it was written to catch is in the tree. Re-point the two
callers at `__contains__` before treating any cost here as measured — and a
kernel bench cannot see it, since no kernel passes over the divisor:
`BinaryKernel.apply` intersects the operands' validity and nothing else.

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

### The Mojo LZ4 and Zstandard codecs — what is left

`marrow/utils/lz4.mojo` and `marrow/utils/zstd/` run Parquet's LZ4, LZ4_RAW
and ZSTD pages and Arrow IPC's LZ4_FRAME and ZSTD bodies, in both
directions; a reader or writer made with `native_codecs=False` runs them
through liblz4 and libzstd instead. Both compressors write their library's bytes: LZ4 blocks
and frames as liblz4 1.10.0's `LZ4_compress_default` and
`LZ4F_compressFrame`, Zstandard frames as libzstd 1.5.7's level 1, and the
tests hold them identical. Speeds below are `benchmarks/codecs/codec_ab.mojo`
-- thread CPU time, the two sides alternated, best of 25 -- native over the
library, under 1 being faster.

- **LZ4 decoding is 1.12-1.17x liblz4 on text**, 1.03-1.04x on int64s
  and 1.08-1.13x on floats (14 us a MiB, nearly all one literal copy).
  Profiles of both loops show the same mispredicted branch -- whether a
  match is longer than 18 -- taking about 29% of each, the same loads and a
  few more instructions on ours; liblz4's 18-byte copy, tried in SIMD and in
  integer registers, made it no faster. Compression is 0.92-1.03x.
- **Zstandard compression is 1.05-1.09x libzstd on 64 KiB of text and
  1.06x on 1 MiB of int64s**, decoding 1.04-1.05x on 64 KiB of text; every
  other case is 0.92-1.03x. Per-phase timings against libzstd's exported `HIST_*`,
  `HUF_*` and `FSE_*` functions were the way to find each gap so far
  (throwaway drivers, not kept); the end-to-end benchmark is too noisy on
  this machine to resolve a few percent.
- **Zstandard's sequence decoding runs short of registers.** On 64 KiB of
  text, 28% of the loop's samples are stack loads and stores, and small
  changes around it move that: a call anywhere in the loop, even in its
  rarely taken branch, took it to 36% and decoding 10-20% slower; the
  guard at the top of `Zstd.decompress_into` is worth 13-25% on text and
  int64s, though the length check after the loop catches the same input.
  Cutting what the loop keeps live -- three FSE tables behind one base
  pointer, the bounds as end positions -- is the open fix, so its speed
  stops depending on what surrounds it.
- **Zstandard compresses at level 1 only.** Neither writer takes a level
  yet; when one does, levels above 1 need libzstd's double-fast and lazy
  match finders, or `native_codecs=False` for those levels.
- **The size gate needs re-recording when this lands.** `query_cli` carries
  the native decoders where it had `dlopen` calls: +31,296 bytes of `__text`
  (+1.04%) over this work's merge base, `zstd` 28 KB of it and `lz4` 2.8 KB.
  No encoder links into a binary that never compresses. Keeping liblz4 and
  libzstd selectable (`native_codecs`) adds +5,632 to `query_cli` and about
  11 KB to `query_scan` and `query_param`; the whole sweep now reads
  `query_cli` at 3,056,172, +1.47% over the recorded baseline.

### The Mojo Snappy codec — what the experiment leaves open

`marrow/utils/snappy.mojo` is a port of libsnappy 1.2.2. Its module docstring
has the measurements. Parquet still calls libsnappy. Routing it through
`Snappy` is two lines in `Compression.decompress_into` and
`Compression.compress` (`parquet/codecs.mojo`); with that local patch all 279
`marrow/parquet/tests` passed once, and `bench_parquet` read as parity.

- **The reader should prefetch a chunk before decoding it.** A local file is
  memory-mapped and faulted in as the decoder touches it, so disk time and
  decode time add up. `benchmarks/codecs/snappy_disk.mojo` measures this on
  cold files. `posix_madvise(..., POSIX_MADV_WILLNEED)` over the chunk first
  made libsnappy 1.2x faster on `strings` and `ints` and several times faster
  on incompressible `floats`. It is POSIX (macOS and Linux), and it belongs
  in `BufferSource.read_ranges`, with the call in `buffers.mojo`. It helps
  every codec, uncompressed pages included.
- **Pair decoding needs the reader to hand it pairs.** `PageReader._body`
  decompresses one page at a time into one scratch buffer. Using
  `decompress_pair_into` means parsing the next page header before decoding
  this page, and keeping two scratch buffers. Without the prefetch above it
  helps less from a cold mapping, and on incompressible pages it hurts:
  alternating between two regions of the file seems to defeat read-ahead.
  With the prefetch it reached 1.76x on `strings` and 1.92x on `ints` against
  plain libsnappy from cold disk.
- **Before flipping the default:**
  - Run the `query_cli` size gate with Parquet routed through `Snappy`.
  - Run `bench_parquet` on a quiet machine. The end-to-end runs so far were
    under a load average of 30-60, and they read as parity within the noise.
  - Get a Linux x86-64 run. There the stdlib's shuffle is `pshufb`; only
    Linux arm64 has run, under Docker.
- **The compressor's 5-10% gap on tag-heavy data** (`ints`, `strings`) is not
  algorithmic. Its disassembly has the same CRC hash, 16-position probe and
  `rbit`/`clz` match finder as libsnappy. That gap was not chased.
- **`CompressionLibs` returns a fresh list per page, for every codec.** The
  writer then copies it again into the file buffer, and the decompress
  methods take a raw `Pointer` destination. Taking `mut out: List[UInt8]` to
  append to, and a `Span` destination, as `Snappy.compress` and
  `Snappy.decompress_into` do, would drop a copy and an allocation per page
  and take raw pointers out of every caller.
- **`bulk_copy` (`buffers.mojo`) is measured on one caller only.** The
  stdlib's `unsafe_memcpy` is a 32-byte loop above 16 bytes and never calls
  libc; handing copies of 256 bytes or more to libc `memcpy` took Snappy
  compression of incompressible data from 0.83-0.88x libsnappy to parity.
  PLAIN pages in `parquet/reader.mojo`, the uncompressed arm of
  `Compression.decompress_into`, `BufferView.copy_from` and `Buffer.resize`
  switched with it; that they gain too is inferred, not measured.

### 1.9 The Parquet reader, after page-level pruning landed

A `RowSelection` now narrows what is *fetched*, not just what is decoded: the
`OffsetIndex` plans the byte ranges, adjacent pages merge into one read, and a
skipped page is stepped over from the index rather than by parsing its header.
`marrow/parquet/tests/test_page_io.mojo` measures this with a recording
`ByteSource` -- the only way to tell "returned the right rows" from "did less
work". What is left:

- **A remote `read_ranges` fans out on threads, not asynchronously.**
  `OpenDalSource.read_ranges` issues its fetches through
  `ctx.fan_out_blocking`, on the shared I/O pool (`ThreadPool.shared_io()`,
  `MARROW_IO_THREADS` threads, else one per logical core). What it is not is
  asynchrony: no more requests are in flight than that pool has threads, where
  `object_store::get_ranges` hands the whole set to a runtime, and a thread
  blocked on a socket is a pool thread doing nothing. That is N/nt round trips
  instead of N. Closing the rest needs an async I/O primitive marrow does not
  have; widening the I/O pool raises nt but does not substitute for one.

- **`bench_read_selected_prefix_snappy_1m` reads an uncompressed file.**
  `_prepare_groups` in `marrow/parquet/tests/bench_parquet.mojo` takes a
  `compression` argument and writes `compression="none"` regardless, so the
  row its docstring calls "where page skipping shows" measures no codec at all.

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
  region, so a hoisted plan is what makes one read replace N. **Measured**: it
  is the whole ~350 us gap between a read with no selection and one that
  selects every row, on a 4-group x 3-leaf 1M-row file, and nothing is cached
  between reads -- a loop of 20 reads over one `ParquetFile` decodes all 12
  indexes 20 times.

- **Page skipping made a read slower, and the cause was `last_selected`.** Its
  cost is the rows *discarded* -- a backward walk over the deselected tail --
  and the read path asked it once per row group and once per leaf, so a `limit`
  selection walked the same tail sixteen times. That is why the cost was a step
  rather than a gradient: the walk is one iteration when the last row is
  selected and ~N when it is not, so it turns on the moment a selection stops
  being total and barely moves after. Finding it once, when the flags arrive,
  measured 2.09x at an eighth kept and 2.33x at one row, 1.02x when nothing is
  excluded -- and put the feature the right way round, skipping now costing
  less than reading everything where it used to cost 1.8x more.

  **The instrument is the lesson.** This fix was rejected twice on numbers that
  could not see it: `bench_read_selected_*` builds its selections *inside* the
  timed body, so caching relocated the walk rather than removing it and read as
  1.03x. `benchmarks/profiles/profile_page_skip.mojo` builds them outside,
  which is the difference. Before trusting a measurement of selection
  machinery, check which side of the timed body the selection is built on.

  Two facts about the local path still stand and bound what any of this can
  buy: a local `ByteSource` is `Buffer.mmap_file`, so a trimmed fetch skips no
  I/O, and an uncompressed all-present page decodes to a memcpy.
  `bench_read_selected_prefix_snappy_1m` is the row that can move when page
  pruning improves; its uncompressed sibling mostly cannot.

  **Done.** `RowSelection` stores runs, which is the shape `page_selections`
  always produced. Measured against the per-row form: 1.13x on a total
  selection, 1.11x at an eighth kept, 1.05x on a 1-in-8 per-row scatter -- the
  shape expected to regress, since per-row scatter is one run per row. The
  benchmark understates it, handing `read` a selection directly and so never
  paying the `1 + 2W` row-length allocations per row group that
  `page_selections` used to make.

  Still per-row on the decode side: a partially selected page builds a
  `List[Bool]` mask, and every `consume_selected` copies it again before
  placing values one at a time. Handing the runs down instead -- so a primitive
  builder can `unsafe_memcpy` a run -- retires `mask` and both allocations, at
  the cost of one signature across seven builders.
- **`LIMIT` never becomes a `RowSelection`.** `LimitOperator.done()` stops the
  driver, so row groups past the limit are never opened -- but within the first
  surviving group every row is decoded. `limit 10` over a million-row group
  reads the million. The `OffsetIndex` machinery to read ten rows' worth of
  pages now exists; what is missing is the limit reaching the scan as a row
  range, which is the same `row_limit` channel top-K needs.

- **The comptime lane prunes numerics only.** Temporal and decimal predicates
  prune through the runtime lane, which recovers the dtype from the index and
  dispatches. In the fused lane `TemporalCompare` inherits `Value.mask`'s
  default and keeps every chunk. Temporal would need a dtype *instance* to
  prune with, since a temporal type carries a unit and `Stat()` does not exist
  where `NumericType(Defaultable, ...)` makes it free; that was judged not worth
  a required trait member for one dtype family.

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

### 1.12 A finding not covered by any row above

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

### 1.13 Window `variance`, string `min`/`max` and `count_distinct` are O(n * frames)

An aggregate with a `Windowable.over` (`kernels/aggregate.mojo`) runs over all
its frames in one pass: a running fold while the frame's start holds still, a
`SegmentTree` query otherwise, as DuckDB's `WindowSegmentTree` does. `Fold`
(`sum`, `product`, `min`, `max`, `mean`, `count`) and `ValidCount` conform. The
other three do not, so `WindowOperator._per_frame` runs their operator once per
distinct frame, and a cumulative `variance` over 100k distinct keys is
quadratic.

Each is one more `Windowable` conformer. `Dispersion` needs only a Welford
`(n, mean, m2)` `Monoid` with Chan's merge — its math combines, it just has no
`over`. String `min`/`max` needs a `Monoid` over `Optional[String]`, or over row
indices plus one `take`. `count_distinct` needs a merge sort tree (DuckDB's
`WindowDistinctAggregator`) rather than a segment tree, since a set union is
not a cheap combine.

### 1.14 Readers that crash on malformed input — the fuzzing findings

**Epic: every reader must answer a malformed file with an `ArrowError`, never
an abort or a read past an allocation.** libFuzzer finds the bugs below within
seconds of a session starting. Each has a reproducer under `fuzz/corpus/` whose
`expected.toml` says `crash`; `pytest fuzz` fails when one of them stops
crashing, so fixing a bug includes moving its verdict to `reject`. An entry
marked `sanitizer = "address"` is one only `pixi run -e asan fuzz-replay-asan`
can judge: B6's string over-read is the one left. A reproducer must crash the
same way on every platform, so none relies on a wild read faulting -- glibc's
heap has mapped memory where macOS's has none. B1 (an IPC buffer outside the
body) and B17 (an uncompressed page copying more than it holds) were fixed
before the corpus landed, and so was B6's short-bitmap reproducer, whose
bitmap length is negative; those stay as `reject` entries. `read_array` still
checks no bitmap against `length`, so a short one of positive length should
read past it, but there is no reproducer for that yet.

| Bug | Where | What the input does |
|---|---|---|
| B2 | `ipc.mojo:2179`, `:2190`, `:2319` `read_array` / `_consume_buffer` | more field nodes or buffers consumed than the message carries |
| B3 | `ipc.mojo:696` `_FlatbufReader.read_string` | `String(unsafe_from_utf8=)` without validation; aborts in `_read_field` (`:1649`) and `_read_kv_vec` (`:1458`) |
| B4 | `dtypes.mojo:1456` `as_type` | a dictionary or child whose declared type disagrees with the schema reaches `as_type` as the wrong member |
| B5 | `ipc.mojo:701` `_FlatbufReader` | nested tables recurse without a depth limit: stack overflow |
| B6 | `ipc.mojo:2291` `read_array` | builds `ArrayData` without checking any buffer against `length` -- values, offsets and their contents, the validity bitmap -- so formatting the result reads past the allocation (`arrays.mojo:1146`, `:3348`, `views.mojo:908`, `buffers.mojo:858`); one input gets as far as reading a struct child's buffer as device memory (`buffers.mojo:904`) |
| B7 | `ipc.mojo:2194` `read_array` | a positive `null_count` with an empty validity buffer keeps the count and drops the bitmap; `PrimitiveArray.slice` then unwraps the absent bitmap (`arrays.mojo:805`) |
| B8 | `parquet/schema.mojo:596` `SchemaMapping.from_parquet` | trusts `schema[0]` exists and its `num_children` fits the schema list. 12 bytes reproduce it, and it is what almost every Parquet mutation hits first -- the `parquet_metadata` and `parquet_page_index` sessions found nothing else |
| B9 | `parquet/codecs.mojo:741` `Dictionary.byte_offsets` | a byte-array dictionary page holding fewer values than `num_values` reads past the page (`utils/byteorder.mojo:75`) |
| B10 | `parquet/reader.mojo:733` `PrimitiveLeafBuilder._scatter` | a page with more values than the chunk has rows **writes** past the values buffer (ASAN: heap-buffer-overflow, WRITE of size 8) |
| B11 | `parquet/reader.mojo:2980` | a row group with fewer column chunks than the schema has leaves |
| B12 | `parquet/reader.mojo:483` `PageReader.next` | a `DATA_PAGE_V2` page without its `data_page_header_v2` unwraps an empty `Optional` |
| B13 | `parquet/codecs.mojo:66` `Rle._run_value` | a truncated RLE run header reads past the data |
| B14 | `parquet/reader.mojo:427` | a negative `compressed_page_size` slices with start past end |
| B15 | `parquet/reader.mojo:385` | a v1 page's definition-level length is not checked against the page body |
| B16 | `utils/snappy.mojo` | a literal whose 4-byte length is `0xFFFFFFFF`: libsnappy wraps it to 0 and accepts, the native decoder refuses -- a parity gap, not a memory bug |

B6 and B7 are one fix in spirit: `read_array` should validate what it builds,
which is what Arrow C++'s `ValidateFull` and arrow-rs's `ArrayData::validate`
do on IPC read; `ArrayData.validate` checks only the buffer count today. B8
masks the rest of the Parquet footer and page index: fuzz those two targets
again once it is fixed.

## 2. Missing capabilities, in detail

The tiering is by *user impact*, and it
cuts across the priority table above: a Tier 1 item can be cheap and a Tier 2
item can be the largest thing here.

Each section states what exists, what is absent, and what it would take.

### Tier 1 — table stakes

A user rejects the library outright without these.

#### 1.2 CSV and JSON readers

**What exists.** NDJSON: `marrow/json/` reads (`read_json`, `open_json`) and
writes (`write_json`) like `pyarrow.json`, from Mojo and Python, and scans as
`scan_json`. No CSV: the only CSV code is `QueryCli`'s output writer
(`render_csv`, `marrow/expr/cli.mojo`); `QueryCli` has no `--format json`.

**CSV — what it would take.** A tokenizer (most of the work), a sampling
inference pass and a type-widening order. The value side exists: every builder,
`Iso8601` in `utils/datetime.mojo`, and the string-to-number/bool cast kernels;
`marrow.json`'s two-pass shape (infer, then parse into builders of the settled
schema) carries over. **This is the highest ratio of user value to engineering
novelty on the whole page.**

**JSON — open.**

- *EmberJson is not published.* `marrow.expr` imports `marrow.json`, so
  `package/marrow.mojoc`, the conda package and `marrow compile` need
  `emberjson`/`emberserde` built by the same nightly beside them, and a conda
  package cannot depend on a git source. Repository builds get it from the
  fork's `marrow` branch.
- *Speed: 1.7-2.5x single-threaded pyarrow* (`bench_json.py --competition`,
  2026-10-04: flat 1M rows 436 ms against 254 ms, nested 100k 115 ms against
  47 ms). Keys and strings are matched and appended in place, not copied, and
  key lookup tries the next expected field first. Left: `read_json` tokenizes
  the input twice (inference, then parsing); the per-column slot tree and its
  builders are rebuilt for every block.
- *Parallel reads.* Blocks are independent once the schema is settled, and
  pyarrow's threaded reader is ~10x its serial one on the bench file. Cut the
  block ranges first (the newline scan `Blocks` already does), run pass 1 per
  block on `utils/threads.mojo`'s pool with a `ColumnShape.merge` by the same
  promotion rules (what Arrow C++ does), keep per-block row counts so errors
  still say "in row N", then run pass 2 per block and reassemble in order.
  `ReadOptions(use_threads=...)` as pyarrow spells it. An OpenDAL source is
  not safe to `read_at` from several threads: fetch its ranges up front with
  `read_ranges`. Cover it with the `test_tsan` lane.
- *Remote sources fetch every block twice*: once per pass, since `read_json`
  infers before it parses. Each fetch is released when the next replaces it.
- *Differences from pyarrow*: `NaN`/`Infinity` are rejected; a file of empty
  objects reads as zero rows; a row longer than `block_size` grows the read
  instead of failing; an empty local file raises `IOError` (from
  `Buffer.mmap_file`); `newlines_in_values` and an explicit `date32` are
  unsupported.

**The conda package build is probably broken on 1.2, JSON or not.** `mojo
package -o x.mojopkg` now fails with `output path must have a '.mojoc'
extension` (measured 2026-09-25), and `pixi-build-mojo 0.1.*`, which
`pixi.toml` pins, writes `lib/mojo/marrow.mojopkg`. Not yet confirmed with
`pixi build` itself.

#### 1.3 Datasets: multi-file, partitioned, remote

**What exists.** `scan(path: String, schema: Schema)`
(`marrow/expr/builders.mojo:730`) — one file, and the caller supplies the schema
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
**18 rules and one downward pass**, invoked as `plan.optimize[AllRules]()`,
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
elimination, and aggregate pushdown.

**Rendering is the blocker for the first two, not equality.** Comparing two
boxed expressions needs no new slot — `WindowExpr.spec()` (`logical.mojo`)
already does it by comparing `String(k)`, because `DynValue` exposes `write`
and `write` is non-raising, so a rule can call it. That makes it exactly as
sound as the rendering, and the rendering is not faithful: `BinaryLiteral`
prints `lit(<N bytes>)` and never its value, so two different blobs compare
equal; every other literal prints `lit(v)` with no dtype, so `lit(1)` as int32
and as int64 are indistinguishable. A deduplicating rule written today would
merge keys that differ. The work is making every node's `write_to` injective
over what distinguishes it, then the rule is small.

**Still missing in the join path:** the **search**. Choosing among the
Catalan-many associations of an *n*-join chain is a `prepare` pass, not a
`Rule`. And a reassociation is only as well-informed as its sources' NDV:
over a pyarrow-written file, which records no `distinct_count`,
`max_distinct` falls back to the row count, every join estimates at
`min(|L|, |R|)` rows and only the intermediate's *width* is left to decide on.
From Python the rules see no source statistics at all: `parquet_scan` in
`python/bindings/plan.mojo` attaches none, and an in-memory table records no
distinct count.

A credible engine ships without a join-tree search, so none of this is urgent. But
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
split_part, and `ConcatKernel` behind `||` in both lanes), three regex kernels
(`regexp_matches`, `regexp_extract`, `regexp_replace`, in
`marrow/kernels/regex.mojo`) and 15 temporal extractors plus `date_trunc`.

Adding one is cheap and reaches every caller: `UNARY_VERBS`/`BINARY_VERBS`/
`TERNARY_VERBS` (`marrow/expr/runtime/values.mojo:1922`) is the single
vocabulary, and a verb added there is callable from Python without touching the
bindings or `python/marrow/expr.py`.

**Still absent — strings:** the `concat` function, which skips null
arguments where `||` propagates them, `concat_ws`, and the rest of the regex
family: `regexp_full_match`, a global `regexp_replace` (DuckDB's `'g'` flag),
`regexp_split_to_array`, and `regexp_extract` without a group argument.

**Still absent — temporal:** `date_diff`, interval arithmetic,
`strftime`/`strptime`, `make_date`, `age`, date to string casts, and the SQL
spellings of the zone verbs: `timezone(zone, ts)` and `AT TIME ZONE` have no
translation, and `TIMESTAMPTZ` is not a SQL type name. `assume_timezone` and
`convert_timezone` exist in both lanes and Python, and extraction and
`date_trunc` read a zoned timestamp in its zone, through morrow
(`WallClock` in `marrow/kernels/temporal.mojo`).

**No `ambiguous` / `nonexistent` choice.** pyarrow's `assume_timezone` and
polars' `replace_time_zone` raise by default on a wall-clock time a
transition repeats or skips. marrow's `assume_timezone` always resolves it as
ICU and DuckDB do: the earlier instant, and the offset before the gap. A
parameter taking pyarrow's values (`raise`, `earliest`, `latest`), defaulting
to today's behaviour, closes that gap; morrow's `resolve` already reports
both cases through `is_ambiguous` / `is_imaginary`.

**A zoned timestamp is Arrow's, not DuckDB's.** Arrow reads a zoned
timestamp's fields in the zone on its type; DuckDB reads a `TIMESTAMPTZ` in
the session's. A SQL frontend that wants DuckDB's answers has to convert to
the session zone first, which is why `timezone()` is not simply mapped to
`assume_timezone`. The golden corpus pins DuckDB's session to UTC.

**Each row's offset copies a zone.** morrow parses a zone once per process,
but its per-instant answer is `TimeZone.at(utc)`, which returns a whole
`TimeZone` value (two `String`s and a shared pointer) for one `Int`. A morrow
API answering the offset alone removes that per-row copy.

ibis's `strings.py` is the engine-level expectation: case, trim/pad,
substring/slice, find/predicate, pattern match, regex (extract/split/
replace), replace/split/join, and URL parsing.

**Known-wrong regex answers.** The engine is a fork of `mojo-regex` (see
1.8), whose capture-group path never enters an optional group: `(?:foo)?(bar)`
on `"foobar"` captures from offset 3, so `regexp_replace(s, '(?:foo)?(bar)',
'[\1]')` answers `foo[bar]` where DuckDB answers `[bar]`. The same path finds no
match for a pattern whose only match is empty (`x*`). Only `regexp_extract`
with a group above 0 and `regexp_replace` with a `\N` reference reach it;
`regexp_matches` and group 0 use the overall matcher, which gets both right.
`marrow/kernels/tests/test_regex.mojo` pins the wrong answer so a fixed engine
flips the test. Report both upstream with the reproductions from `ced18e32`.

**What it would take.** `concat`/`concat_ws` is the one cheap item left: `||` already runs through
`ConcatKernel` in both lanes, and what is missing is the null-skipping
variant.

#### 1.7 Known-wrong answers in core operations

One left; the golden corpus carries **zero `xfail`s** otherwise.

- **Integer overflow wraps where SQL raises** (`golden/COVERAGE.md`). The
  `edges` fixture already carries int64 max/min for the day a
  checked-arithmetic mode exists.

  **The machinery is already in the tree, in `cast`.**
  `NumericCast.apply[From, To, safe]` takes `safe` as a *comptime* parameter
  and, when it is on and `needs_check[In, Out]()` says the pair can overflow,
  swaps `views.apply` for `views.apply_checked` — a driver whose lane returns
  `(value, bad)` and raises on the first flagged lane, serial because an
  exception cannot cross a worker or a GPU launch. Arithmetic would copy that
  shape: a `core_checked` beside each `core`, a two-input `apply_checked`, and
  a selector that must be comptime in the fused lane (a runtime `if` there is
  not eliminable and every AOT gate pays for it) while the runtime lane can
  read it off `ExecContext`. `cast` already spells it both ways — comptime on
  `apply`, a runtime argument on `dispatch` — so the asymmetry has a
  precedent. A golden case still cannot express "raises".

---

### Tier 2 — competitive

Needed to be chosen over an incumbent, but a user will trial the library without
them.

#### 2.1 Parallelism above the kernel

**What exists.** Data parallelism *inside* kernels only — `ExecContext`'s
`stripe`, `run` and `fan_out` over marrow's own `ThreadPool`
(`utils/threads.mojo`), called from `partition.mojo`, `views.mojo`,
`hashtable.mojo`, `hashing.mojo`, `join.mojo`, `sort.mojo`, `filter.mojo` and
the Parquet and OpenDAL readers.
**Group-by placement is parallel**, as of the radix-partitioned `HashIndex`
— one `SwissHashTable` per partition of the key hash's top 6 bits, so no
aggregate state is ever split and no merge step exists. Aggregate *accumulation*
is still serial, and deliberately: a thread-local partial would need a `merge`
on every `AggKernel`, which `mean` and the Welford triple make non-uniform and
exact `count_distinct` makes impossible. There is no pipeline parallelism:
`Pipeline._flow` pushes one morsel through the stages on the calling thread.

**What it would take.** True pipeline parallelism is the remaining item: the
push `Operator` contract is a good foundation, and `ThreadPool` is the substrate
a scheduler would run on — its task queue, `TaskScope` spawning, and a `fan_out`
whose body can stop early for a `LIMIT`. What is missing is the scheduler: nothing
runs pipelines or morsels on it yet.

Group-by placement has three open costs, all measured by the "Calibration
sweeps" in `bench_groupby.mojo`:

- **String keys cross about twice as late** as the gates in
  `kernels/groupby.mojo` — near 65,000 groups and 75,000 rows, against
  int32's 30,000 and 50,000 — so one pair of constants costs a string batch
  between the edges up to 5%. A per-type gate would recover it, but the cause
  is unprofiled: from 2k to 100k groups at 1M rows radix's string time grows by
  1.3 ms and its int32 time by 0.3 ms, while serial's grows by 1.8 ms and
  1.4 ms. The key gather of the new groups' rows, which the radix path issues
  in partition-major rather than row order, is the first suspect.
- **Serial placement pays for table growth on insert-heavy batches.** At 15k
  rows and 7,500 groups the bare insert takes 102 us growing adaptively and
  29 us into a table reserved for the answer — most of the batch's 129 us. The
  radix path already reserves per partition; the serial one has no size hint,
  because the cardinality probe only runs above the row gate.
- **`ExecContext.parallel(N)` stripes every loop N ways however small.** That
  is its contract, but it made serial placement of a 15k-row batch 5x slower
  (675 us against 129 us) with nothing else changed, and every caller that
  passes an explicit worker count pays it. `auto()` does not.

#### 2.2 Larger-than-memory execution

**What exists.** Nothing. No spill, no memory pool, no accounting, no limit —
`grep -in 'spill\|memory_pool\|memory_limit'` over `marrow/` returns only
unrelated bitmap-test strings. `execute()` drains the entire plan into one
`RecordBatch` (`marrow/expr/logical.mojo:1411`), so even a streaming plan
materializes its full result, and **there is no batch-iterator result API** even
though `drain()` is exactly that shape internally.

The old spilling streaming engine was removed and not replaced. What it would
take includes spilling variants of group-by and sort.

#### 2.3 Relational operations that have no node

Each is a missing `Relation`, not a missing kernel. ibis's `relations.py` is the
canonical list.

| Missing | Golden cases | Note |
|---|---|---|
| `DISTINCT ON` | 1 | `distinct()` is an aggregate keyed by every column, so it cannot keep a whole row per key; `DISTINCT ON` needs a first-row-per-key node or a `row_number() = 1` rewrite |
| `GROUPING SETS` / `ROLLUP` / `CUBE` | 3 | `Aggregate` carries one key list; `ROLLUP` also needs `GROUPING()`. Implementable as a rewrite into an aggregation cascade |
| `explode` / `unnest` | 1 | Row-multiplying, so a new operator shape. ibis has a dedicated `TableUnnest` with `offset` and `keep_empty` |
| `Sample`, `DropNull(how)`, `FillNull` as relations | — | ibis has all three as nodes |
| `top_k` / `bottom_k` as first-class | — | A dedicated streaming node beats rewriting sort+limit: `TopN` bounds the sort, but `SortOperator` still buffers every row before ordering |
| `merge_sorted`, `rolling`, `group_by_dynamic`, `upsample` | — | Time-series reshaping — a common ask in that domain |

#### 2.4 Join breadth

**What exists.** Hash equi-join in eight *implemented* kinds — inner, left,
right, full, left semi, left anti, right semi, right anti — over a Swiss table
with a CSR probe index (`marrow/kernels/join.mojo`, `hashtable.mojo:79-90`),
with radix partitioning and parallel probing, and either input may be the build
side. Multi-column keys work because keys go through `StructArray`. `mark`,
`single` and `cross` have constants and no kernel; `is_supported()` says so and
`hash_join` rejects them.


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
- **Missing *shapes* — arity, ordering and `DISTINCT`.** `arg_min`/`arg_max`,
  `corr`/`covar`, `ORDER BY`-carrying `first`/`last`, multi-column
  `count(DISTINCT a, b)` and the `DISTINCT` modifier still have nowhere to
  attach, and the node now has the shape to take one more operand cheaply — a
  second `Evaluable & Value` parameter defaulting to `Nothing`, read under
  `comptime if`, exactly as `P` is. What blocks them is the **kernel**
  contract and, in the runtime lane, the instantiation space:

  - `AggKernel` declares `InArray` and `update(groups, input)`, so a
    two-operand aggregate needs a sibling trait (`dtype`, `name`, `reserve`
    and `finish` are common and would move to a shared base) plus a fourth
    physical operator that evaluates two operands per morsel. `ArgExtremum`'s
    state is a bounded per-slot pair — the best key and the value at it — so
    it streams like every other kernel here; it is `AggState[K, V]` with two
    accumulator columns rather than one, which is the same shape `Dispersion`
    already declined to fit.
  - **The comptime lane is where a two-operand aggregate is cheap and the
    runtime lane is where it is not.** A comptime plan names one `(value, key)`
    pair and links one `ArgExtremum[Op, V, K]`; `resolve_aggregate` would have
    to dispatch the key *inside* the value's dispatch, which is every primitive
    dtype squared — order 800 instantiations across `arg_min`/`arg_max`, on the
    two gates that are already the largest. Affording it there means erasing
    the key side (a `DynArray` key compared per row), which is a different
    kernel, not the same one resolved twice.
  - `string_agg`/`array_agg` remain a separate epic whatever the node does:
    their fold is neither associative nor bounded, so `Foldable`'s
    `combine_at` contract does not describe them.

#### 2.6 Nested-type operations

**Storage is complete, and every type enters an expression** — each has a
column, literal and parameter leaf — **but almost nothing computes on the
nested ones.**

- **Nested:** eight golden cases — list element access, list slice, `unnest`,
  list sum, struct field access, map lookup, map cardinality. The nested verbs
  that exist are `array_length` and `array_contains`; everything else needs a
  kernel that does not exist. ibis's minimum here is `ArrayIndex`, `ArraySlice`,
  `ArrayContains`, `ArrayLength`, `Unnest`, `MapGet`/`MapContains`/`MapKeys`/
  `MapValues`/`MapLength`, and `StructField` — and struct is genuinely thin
  there too (`structs.py` defines exactly two nodes), so `StructField` alone
  closes most of the struct gap.

#### 2.7 No row format

It is the shared primitive behind fast multi-column sort, sort-merge join, and
hash group-by keys.

marrow instead does column-oriented LSD multi-key sort — one stable pass per
key, re-gathering each key column per pass — and hashes join and group-by keys
a column at a time, as a struct array of the key columns
(`DictionaryEncoder.batch`). That works and is not wrong, but it is the structural reason a future sort-merge
join has no cheap path and why multi-key sort re-gathers. Worth naming as a
design decision rather than discovering it under a benchmark.

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

- **No `__dataframe__` protocol**, though
  the PyCapsule/C Stream path marrow already has is the better-supported
  modern route.
- **Avro, and Iceberg on top of it.** `marrow/avro/` reads and writes object
  container files (`read_avro`, `write_avro`, `AvroFile`, `AvroWriter`;
  Python's `marrow.avro`) with the `null`, `deflate`, `snappy` and
  `zstandard` codecs, and carries Iceberg's `field-id` / `element-id` /
  `key-id` / `value-id` as `field_id` field metadata. What it does not do:
  - **No Avro schema resolution.** A file is read with the writer's schema
    only. Iceberg does not need it -- it projects by field id -- but a
    generic consumer evolving an Avro schema would.
  - **Unions of several non-null branches** are refused: marrow has no union
    layout (§ Known Limitations in CLAUDE.md).
  - **`bzip2` and `xz`** raise `NotImplementedError`; neither library is in
    the `Codecs` set.
  - **Blocks decode serially.** They are independent once the sync markers
    are found, so `AvroFile.read` could fan them out over `ctx`; it also reads
    the whole data region in one `read_at`, which is right for manifests and
    wrong for a multi-gigabyte file on an object store.
  - **No `AvroScan`** in `marrow.expr`, and `AvroFile` does not implement
    `BatchReader`, so a plan cannot scan Avro.
  - **The Iceberg layer itself** -- the manifest-list / manifest model,
    field-id projection and evolution over the decoded tables, and
    `Index.from_iceberg_manifest` beside `Index.from_parquet`
    (`marrow/expr/index.mojo`) -- is unwritten. So are field ids on the
    *Parquet* reader, which Iceberg's data files need the same way.

#### 2.10 Operability

- **No `EXPLAIN ANALYZE`, no per-operator metrics, no profiling hook, no
  progress, no cancellation.** An operator cannot be interrupted mid-`drain`.
- **Per-key null placement is missing on sort:** `Sort` carries one `nulls_first: Bool` for
  all keys (`logical.mojo:1874`); ibis's `SortKey` carries it per key.

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
| `query_streaming_agg_fused` (comptime) | 1,459,112 | |
| `query_streaming_agg` (runtime-named) | 14,255,908 | **9.77x** |
| `query_dynvalue` (erased values) | 9,891,236 | |
| `query_streaming` (fused filter + project floor) | 1,452,940 | |
| `query_cli` (the AOT lane as a program) | 3,000,940 | |

Re-read from `baseline.json` on 2026-09-22 (mojo 1.2.0.dev2026092105). In
2026-08 the ratio was 6.71x, with `query_dynvalue` at 6,227,524, and on
2026-09-14 it was 9.09x, so the gap keeps **widening**. The last step is not
drift: `FILTER (WHERE ...)` on aggregates put **+1,119,360 bytes on
`query_streaming_agg` and zero on every AOT gate**, because the runtime lane
instantiates `BufferedAggregateOperator` once per kernel `resolve_aggregate`
can bind — 163 of them — where an unfiltered comptime aggregate is the
instantiation it always was. That is the erasure boundary doing its job, and
it is also the shape of what a second aggregate operand would cost there; see
§2.5. The gate compares each change only against
the last recording, so the runtime lane's floor still drifts one accepted
re-recording at a time — worth a note the day a `query_dynvalue` regression
matters.

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

**The comptime lane boxes, and should not.** Keys and aggregates become
`DynValue`s the moment they enter a `List[DynValue]`, and the plan is an erased
`DynRelation`, so `Aggregate` lowers to `GroupedAggregateOperator`, which hands
`DictionaryEncoder` the key columns, made one array that it dispatches on by
the key's *runtime* dtype. Only key evaluation and the folds are specialised. Exact
grouping made that visible: +231 KB (+14.9%) on `query_streaming_agg_fused`,
and the same on `query_expr2_agg_fused` and `query_decimal_agg_fused`. A
symbol diff of the first, which groups by one `string` key: ~70 KB was
`KeyCompare` for every key type, of which only the string leaf runs; ~55 KB
the dictionary-key encode and emit paths, which never run there; ~47 KB the
stored key values; ~60 KB the encoder, the hash index and the operator.

**Typed grouping was built and removed.** `aggregate(aggs, by=key)` encoded
one comptime key through a store over its own type (`PrimitiveKeys[T]`,
`BytesKeys[T]`), and the three fused gates measured 1,178,528 / 1,182,104 /
1,184,088 bytes of `__text`, about 34% under the erased encoder. It covered
one shape only — a group-by over a single primitive or byte-string key, in a
program that links `DictionaryEncoder` nowhere else (no join, `DISTINCT`,
`is_in` or `count_distinct`) — and needed one store per value family, so one
erased path replaced it. Typed keys worth having are one general design: a
store over a pack of key types, for every consumer of keys. Expect the
variadic-pack forwarding and reflected-field-type limits (CLAUDE.md,
"Associated types, traits, reflection") to shape it. Also open:

- **The aggregates are still boxed** (`List[DynValue]`), each lowered through
  its own trampoline to a typed fold, and `DistinctCount` encodes through a
  `DictionaryEncoder` — a grouped count's key is the `(group, value)` pair —
  so a comptime `count_distinct` still links the erased encoder.
- **`is_in` encodes its value set on every batch**, in both lanes:
  `IsInKernel` builds a `DictionaryEncoder` per call and both nodes call it
  per batch. The set is a plan constant, but a value has nowhere to keep
  state between batches — the slot `EvalOperator`'s docstring names for "an
  `IsIn` hash set" was never built — and the runtime lane only learns the
  set's dtype, unified with the operand's, when a batch arrives. Encoding it
  once is a per-execution state for values, in both lanes.
- **`DictionaryBuilder.extend` rebuilds the dictionary on each
  differently-encoded input**, so `concat` over N chunks with N dictionaries
  copies it N times, and slices sharing one dictionary repeat it. One pass
  over all the inputs, as arrow-rs's `concat_dictionaries`, would not; it
  needs `concat` to see the whole list rather than the builder one array at a
  time. The merge itself costs every size gate ~24 KB, because every binary
  that can extend a builder links the dictionary arm; raising on a second
  dictionary instead measured +13 KB of the same.
- **`DynRelation`'s symbol names are half of `ld`'s limit.** Mojo writes a
  generic type's full layout into its symbols, so `query_cli`'s longest is
  543,714 characters at the parent commit; `ld` asserts above about a million
  (*"name.size() <= maxLength"*). A field on a relation node whose type spells
  a `DynOperator` by value doubled it and broke the link — see
  `AggregateLowering`. Every relation field counts against this.

The same erasure applies to the lane's filters, projections and joins.

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

Entry points: `insert_hashes`, `find_hashes`, `find_one` and `insert_new`,
under `HashIndex` (radix placement), `DictionaryEncoder` (exact codes) and
`JoinHashTable` (the join's rows per code). `RadixPartitioner` splits rows across
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
   and benchmark it against the existing CPU `JoinHashTable.candidates`
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

`Buffer.to_device` (`marrow/buffers.mojo:913`) is `enqueue_create_buffer` plus
`enqueue_copy` — every upload allocates a device buffer and copies the whole
range into it, and `Bitmap.to_device` and `Array.to_device` all funnel through
it. CLAUDE.md's measured guidance is that transfer cost dominates and that
uploading per call is 2-3x slower than just staying on the CPU; this is the API
that removes the copy rather than amortising it.

### Why it does not work on this machine

**Metal requires a page-aligned base and a page-multiple length.** Marrow
allocates at `alignment=64` (`marrow/buffers.mojo:504`) because that is Arrow's
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
