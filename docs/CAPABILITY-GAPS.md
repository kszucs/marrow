# What marrow is missing to be a usable query / dataframe library

**Scope.** Capabilities and features only. Packaging, distribution, installation,
documentation-as-product, community and licensing are explicitly out of scope.
The question is the one a user asks when choosing between marrow and polars,
DuckDB, DataFusion, pandas or ibis for real work.

**Method.** Every claim about marrow was established by reading the tree at
`5ab349e6` — `git grep`, file reads, `git log` — not from memory and not from
`CLAUDE.md`, which is stale on at least one material point (see
[Corrections](#corrections-to-the-projects-own-documents)). Claims about the
incumbents were established by reading
`~/Workspace/{polars,duckdb,datafusion,ibis,arrow-rs}` at their current
checkouts (polars `cbad9d66a0`, duckdb `985b2f2515`, datafusion `44c6671b2`).
**Nothing was built or run** — the Mojo toolchain was mid-upgrade — so anything
that would need a build to confirm is marked *unverified*.

---

## Executive summary

marrow is a **very good Arrow implementation with a small, correct, unusually
well-tested query engine attached, and no user-facing product on top of it.**

The Arrow layer is close to complete and in places ahead of its peers — full C
Data *and* C Stream interfaces including device arrays (`marrow/c_data.mojo:268,
906, 1357, 1536`), a from-scratch Parquet reader with row-group *and* page-level
statistics pruning plus bloom filters, all six page codecs, IPC both directions,
and a type system covering the Arrow spec except union, run-end-encoded and the
view layouts. 1,825 Mojo test cases, 278 golden cases differentially checked
against DuckDB, Arrow archery conformance against C++/Rust/Go.

The engine layer is real but small: eight relational nodes, a push-based
streaming operator model with early termination, hash join in seven kinds, hash
group-by, radix sort, and a genuinely well-reasoned one-sided pruning algebra
(`marrow/expr/pruning.mojo`).

The product layer is one verb deep. **Between `b2de85a0` (2026-08-29) and
2026-08-30 there was no way to run a query from Python at all** —
`python/marrow/lazy.py`, `python/bindings/plan.mojo` and
`python/bindings/expressions.mojo` were deleted with the old expression package.
They have since been rebuilt on the runtime lane, so `read_parquet` /
`memtable` / `col` / `lit` and the nine relational verbs work again, and
`README.md` describes what exists. **Everything else on this page still
stands**: the frontend reads Parquet and IPC only, the optimizer has no cost
model
reachable from it, and the AOT lane still has no entry point — `execute_cli()`
does not exist (`python/marrow/compile.py:9`).

### The three things that most block adoption

1. **There is no CSV or JSON reader, and no dataset concept.** A first user
   reaches marrow with a CSV, not a Parquet file. (The Python query API, listed
   first here until 2026-08-30, is closed: `python/bindings/lib.mojo` registers
   ten submodules including the expression and plan layers, and the runtime
   lane matches the comptime lane verb for verb.)

   Parquet and Arrow IPC are the only formats — `find marrow -iname '*csv*' -o -iname '*json*'`
   returns empty. `scan()` takes exactly one file path *and requires the caller
   to write out the schema by hand* (`marrow/expr/builders.mojo:213`), and
   `ByteSource` has one implementation, a local `mmap`. No globs, no
   directories, no hive partitioning, no object storage, no schema inference.
   ibis's own backend contract makes the expectation explicit: `_FileIOHandler`
   (`ibis/backends/__init__.py:94`) requires `read_csv`, `read_json`,
   `read_parquet`, `read_delta`, `to_csv`, `to_json`, `to_parquet`,
   `to_parquet_dir`, `to_delta` from *any* engine it will drive. polars ships
   fourteen source formats and a shared multi-file scan engine with hive
   partitioning, globbing, cloud credential providers and an on-disk remote-file
   cache (`crates/polars-io/src/{cloud,file_cache,hive}.rs`,
   `crates/polars-stream/src/nodes/io_sources/multi_scan/`).

2. **There is no cost model, and no CSE.** The optimizer landed 2026-08-31 with 16
   rules and a column-pruning pass (`marrow/expr/optimizer.mojo`), covering
   every entry-level member of the set this item listed: projection pushdown,
   limit pushdown below a projection, TopN, constant folding, filter pushdown
   through project/sort/join/aggregate, and conjunction splitting. What remains
   absent is CSE, statistics propagation and a cost model — and join
   reordering, which is blocked by the join's positional output schema rather
   than by the optimizer. See §1.4.

Behind those sit two correctness defects that would fail a proof-of-concept:
`GROUP BY` on a float column silently merges distinct keys
(`golden/cases/group_by_float_key.mojo`), and integer `//`, `%` and
division-by-zero follow Python rather than SQL (three strict xfails).

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

### How far away is usable?

| Target | Distance |
|---|---|
| *A usable Arrow library for Python* (arrays, Parquet, IPC, interop) | **Already there**, minus CSV/JSON. |
| *A usable dataframe library* (a polars or ibis-backend alternative) | **Far.** ~85 measured expression gaps, no CSV/JSON reader, single-threaded group-by, and no set or window operations. |
| *A niche AOT query compiler nobody else serves* | **Closest of the three.** The engine works; what is missing is an entry point, output writers, and promoting the schema handle from spike to API. |

---

## What marrow has today

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

## Tier 1 — table stakes

A user rejects the library outright without these.

### 1.1 A query API from Python

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

### 1.2 CSV and JSON readers

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

### 1.3 Datasets: multi-file, partitioned, remote

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

### 1.4 The optimizer: no cost model, no CSE

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

### 1.5 Window functions

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

### 1.6 String and temporal function coverage

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

### 1.7 Two known-wrong answers in core operations

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

## Tier 2 — competitive

Needed to be chosen over an incumbent, but a user will trial the library without
them.

### 2.1 Parallelism above the kernel

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

### 2.2 Larger-than-memory execution

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

### 2.3 Relational operations that have no node

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

### 2.4 Join breadth

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

### 2.5 Aggregate breadth

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

### 2.6 Nested-type and decimal operations

**Storage is complete; expressions cannot reach it.**

- **Nested:** eight golden cases — list element access, `list_contains` (the
  kernel `ArrayContainsKernel` exists and is wired to nothing,
  `marrow/kernels/nested.mojo:93`), list slice, `unnest`, list sum, struct field
  access, map lookup, map cardinality. The only nested verb in the expression
  layer is `array_length`. ibis's minimum here is `ArrayIndex`, `ArraySlice`,
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

### 2.7 No row format

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

### 2.8 UDFs

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

### 2.9 Interop and format gaps

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

### 2.10 Operability

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

## Tier 3 — differentiating

Where marrow could be better than anything that exists.

### 3.1 The comptime / AOT lane — the real one

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

### 3.2 One kernel, two targets

`apply` writes a single lane and dispatches it to CPU stripes, CPU serial, or a
GPU `elementwise` launch (`marrow/views.mojo`), with `Buffer`/`Array` carrying
explicit `to_device`/`to_cpu` and device-resident results. No CPU dataframe
library has this; the GPU dataframe libraries are not CPU libraries.

Today this is a research capability, not a product: transfer cost dominates, the
measured crossover was ~10K vectors at dim ≥ 384, and the expression layer does
not plan device placement. But "the same kernel source runs on both, and the
plan decides" is a defensible long-term position that Rust and C++ engines
cannot copy cheaply.

### 3.3 Correctness discipline as a feature

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

## Corrections to the project's own documents

Found while establishing the inventory; each would mislead a reader.

1. **`CLAUDE.md:448` says statistics-based pruning and predicate pushdown are
   "unported".** They are in the tree, exported, tested and live:
   `marrow/expr/pruning.mojo` (592 lines), `marrow/expr/pushdown.mojo` (206),
   both re-exported from `marrow/expr/__init__.mojo`, both threaded through
   `Filter.to_operator` (`logical.mojo:790-805`), with
   `marrow/expr/tests/test_pruning.mojo` (31 KB) and `test_pushdown.mojo`
   (12 KB). Projection pushdown and the CLI-output layer *are* still absent.
2. **`docs/backlog.md` §4 describes group-by strategies that no longer exist.**
   `_choose_strategy`, `GROUP_RADIX` and `GROUP_THREAD_LOCAL` have zero
   occurrences; `groupby.mojo` is 224 lines and serial. Parallel aggregation is
   a regression to recover, not a shipped feature.
3. **`README.md:178-247` documents a deleted Python API in detail**, including a
   ClickBench claim whose harness was deleted in the same commit. This is on the
   public README; `docs/backlog.md` §1.2 records the equivalent problem for the
   rendered guides but not for this.
4. **`README.md:238`'s "`explain()` renders one node"** is stale — plans render
   recursively today; what is missing is the verb, not the rendering.

---

## What to build next, in order, and why

A sequence, not a wishlist. Each step is chosen because it unblocks the next, or
because nothing after it matters without it.

**1. Fix the two wrong answers.** Float group keys collapsing, and the three
integer-semantics xfails. A library that returns wrong `GROUP BY` results cannot
be trialled, and both are bounded fixes with golden cases already written that
turn red the moment they are correct.

**2. CSV reader with schema inference, then NDJSON.** The largest ratio of user
value to engineering novelty available, and it can run in parallel with 1 and 2
because nothing downstream depends on it. No user reaches marrow's Parquet path
without having first had a CSV.

**3. Cheap expression nodes over kernels that already exist.** `IN` over
`is_in`, `bool_and`/`bool_or` over `AnyKernel`/`AllKernel`, `list_contains` over
`ArrayContainsKernel`, `concat` over `ConcatKernel`, `greatest`/`least` over
`MinKernel`/`MaxKernel`, `StructField` for struct access, and — the one that
unblocks a whole query class — **temporal literals** in `lit`. Each is a node,
not a kernel, and each deletes a `skip mojo` line. Doing these early also
settles whether the two-lane parity invariant is cheap to maintain, which every
later feature depends on; restore `test_parity.mojo` here, not later.

**4. `scan(path)` without a hand-written schema, then multi-file scans.** Derive
the schema from the Parquet footer; then a `MultiFileScan` node over a glob with
hive partition columns synthesised from the path. Every real Parquet dataset is
a directory, and step 1's users hit this immediately.

**5. Parallel group-by.** Thread-local partials plus a radix merge — previously
built, since removed. Group-by is where analytical queries spend their time, and
running it on one core makes every benchmark comparison unwinnable.

**6. String and temporal function surface.** The 16 + 13 skipped cases, minus
regex and timezones. Ordinary kernels over existing machinery: high volume, low
risk, directly measured by the corpus.

**7. Window functions.** A `Window` node, a blocking partition-and-sort operator
over the existing kernels, and a frame evaluator — from which running aggregates
fall out free, because a reduction under an unbounded-preceding frame *is* one.
The largest single feature a user will ask for that has no partial answer today.

**8. A nested-loop join, then set operations and `distinct`.** The nested-loop
join is small and turns cross/non-equi joins from *unexpressible* into merely
slow, which is a categorical improvement. Set ops and `unique` follow.

Then, and separately, **the AOT product**: restore `execute_cli()` and the
output writers, promote the compile-time schema handle from spike to public API,
and write the one benchmark that makes the case — a compiled marrow query binary
against a DuckDB or DataFusion binary doing the same job, measuring size, cold
start and throughput together. That is the story nobody else can tell, and it
does not compete for the same engineering as steps 1-10.

**What to deliberately *not* build yet**, with the reason in each case being
that an incumbent already ships without it:

- **Cost-based join reordering.** Two of the three do not have one — DataFusion
  has no reordering at all, and polars' `ROW_ESTIMATE` flag is dead code with
  cardinality handled adaptively at run time.
- **Spill-to-disk.** polars' spiller is `unimplemented!()` and DataFusion does
  not spill hash joins. Memory *accounting* is the shippable half.
- **Substrait.** Absent from both DuckDB's tree and polars' entirely; polars
  serializes its own DSL instead, which is the cheaper answer to the same need.
- **SQL parsing.** Worth having eventually, and `polars-sql` shows it can be a
  thin layer over an off-the-shelf parser rather than engine work — but it buys
  nothing until the frontend and the expression surface exist.
- **Pivot / unpivot, run-end-encoded and view layouts, GPU plan placement.**
  Real, none of them what a first user rejects marrow for.

Each is much cheaper once there are users to say which of them they actually
hit — which is the entire argument for putting step 1 first.
