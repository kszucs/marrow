# Changelog

## [Unreleased]

### Features

- **Native Parquet reader/writer** (`marrow.parquet`): a from-scratch Parquet
  implementation that reads and writes Arrow directly, replacing the PyArrow
  bridge (`read_table`/`write_table` are now native; PyArrow is only a test
  oracle). Includes a hand-written Thrift Compact Protocol codec
  (`thrift.mojo`) and metadata structs (`format.mojo`) — no Thrift runtime or
  code generator — modelled on arrow-rs's `parquet_thrift.rs`; page/level
  decoding via the RLE/bit-packed hybrid and PLAIN encodings (`encoding.mojo`);
  and Snappy/Zstd compression through runtime `dlopen` FFI (`compression.mojo`,
  new `zstd`/`snappy` conda deps). Covers flat columns (all common primitives,
  string/binary), definition-level nullability, dictionary (RLE_DICTIONARY /
  PLAIN_DICTIONARY) and PLAIN encodings, v1 and v2 data pages, multiple row
  groups, and struct nesting. The reader additionally handles int8/16 &
  uint8/16, temporal (date32, timestamp incl. nanosecond, time32/64),
  binary/large variants, GZIP/LZ4_RAW compression, and single-level
  List/LargeList columns (Dremel repetition levels). The writer emits multiple
  row groups, per-column null-count statistics, and widens narrow ints. Map
  columns, struct-level nulls, list/temporal writing, dictionary-encoding on
  write, and min/max statistics are follow-ups (all raise a clear error where
  unsupported). The reader mmaps the file, decodes
  fixed-width PLAIN pages straight into the output buffer (memcpy fast path),
  counts definition levels without materializing them for no-null columns, and
  SIMD-unpacks RLE/dictionary index streams eight values at a time (one 64-bit
  load per lane, then a vector shift + mask) — matching single-threaded PyArrow
  on PLAIN data and beating both PyArrow (~3.1×) and polars (~1.6×) on
  dictionary-encoded columns.

### Fixes

- **Empty (zero-chunk) `ChunkedArray.combine_chunks()`** (`marrow.arrays`): a
  chunked array with no chunks — e.g. a column materialized from an Arrow C
  stream that yields zero batches, as PyArrow does for an empty table — combined
  into an `ArrayData` with no buffers, which is not a valid array for most dtypes
  (a primitive needs its data buffer, a string its offsets), so materializing it
  raised "PrimitiveArray requires exactly one buffer". It now builds a
  properly-structured empty array of the dtype. Surfaced by the new Parquet
  interop tests writing an empty table.

- **Relational plans are reusable templates** (`marrow.expr.relations`): the
  relation nodes carried mutable execution cursors (scan offset, built hash
  index, grouper, emitted flag) shared through `AnyRelation`'s `ArcPointer`, so a
  plan was single-use and copying it aliased the cursors — draining one corrupted
  the other. Plans are now immutable descriptions that `open()` into fresh
  operators (see the IR/operator split under Refactors), so a plan can be
  executed repeatedly and copies never share execution state.
- **Correct grouped-aggregation output schema** (`marrow.expr.relations`): a
  multi-key `aggregate` named every key field `"key"` (duplicate names) and fell
  back to the *first* input column's dtype for any key. Each key field is now
  named after its source column with that column's dtype, and key/value
  expressions are bound to positions once. The aggregate output columns are also
  relabelled to the plan's declared schema, so `plan.schema()` matches the
  executed result exactly (previously the grouper emitted `col{i}_{func}` names
  the declared schema did not carry).

### Tests

- **Parquet cross-compatibility suite** (`marrow/parquet/tests/test_interop.mojo`):
  asserts Marrow and PyArrow read each other's Parquet output identically,
  across three shapes — Marrow-reads-PyArrow, PyArrow-reads-Marrow, and
  Marrow-round-trip — over primitives, narrow ints, nullable columns, structs,
  empty tables, dictionary-encoded columns (all codecs incl. GZIP/LZ4), and the
  read-only surface (temporal, binary/large_string, lists). Tables are compared
  column-by-column via the Arrow C stream interface (values plus a type check
  that normalizes the Arrow-only distinctions Parquet drops), tolerating
  chunk-boundary and field-metadata differences.

### Refactors

- **Aggregate field cleanup + single `ExecutionContext`** (`marrow.expr`):
  `agg_exprs` renamed to `aggs`; the redundant `key_fields` field dropped (the
  key fields are the first `len(keys)` output-schema fields, so the processor
  derives them); and the unused expr-local `ExecutionContext` removed in favour
  of `marrow.kernels.execution.ExecutionContext` — one context type, threaded
  through `to_processor(ctx)` / `execute(ctx)`.
- **`Relation.open()` renamed to `to_processor()`** (`marrow.expr`): it converts a
  descriptive IR node into its executing `Processor`, matching the
  `.to_<type>()` convention.
- **`field_name()` renamed to `name()`** on the value interface
  (`marrow.expr.values`): columns/predicates/`DynValue` expose their output name
  as `name()`; the column leaf's field is now `_name`.
- **Removed dead expression API** (`marrow.expr`): `DynValue.inputs()`, the
  unimplemented `CAST` cast (tag + `.cast()` + eval branch), and the unbound
  `Scan` relation node (use `in_memory_table` / `parquet_scan` sources).
- **`collect()` uses the shared `kernels.concat`** (`marrow.expr.execution`)
  instead of a local closed copy — removes the duplication, at the cost of
  `query_streaming` growing 448 KB → 878 KB (the generic concat links
  `AnyBuilder`'s open per-dtype switch, incl. nested builders).
- **Column handles moved into `values.mojo`** (`marrow.expr`): the named
  `NumericColumn`/`StringColumn` leaves and the `Table[Tbl]()` handle now live
  beside the fused algebra and the `col(name, dtype)` factory, removing the
  `values`↔`relations` import cycle — the expr dependency graph is now acyclic
  (`dynamic`/`execution`/`relations` → `values`).
- **Comparison value nodes renamed** `Gt`/`Lt`/`Eq` → `Greater`/`Less`/`Equal`
  (`marrow.expr.values`), matching the compare-kernel vocabulary
  (`greater`/`less`/`equal`).
- **Fused `dtype()` is now total** (`marrow.expr.values`): the
  `NumericValue`/`BoolValue`/`StringValue` traits return `AnyDataType` (always
  known at compile time) instead of `Optional[AnyDataType]`; the runtime
  `DynValue.dtype()` stays optional (a column's type is unknown without a schema).
- **Dropped the relation `kind()` tag** (`marrow.expr`): the `Relation.kind()`
  method and the `*_NODE` constants had no consumer beyond a tautological test —
  node type is available through `write_to()` and `downcast[T]`. An optimizer can
  re-add a tag trivially if it ever wants fast dispatch.
- **`marrow.expr` submodules use relative imports** (`..dtypes`, `.values`, …)
  and `from .. import dtypes as dt`, matching the rest of the package.
- **Relational engine split into a descriptive IR and an execution layer**
  (`marrow.expr.relations` + new `marrow.expr.execution`): `Relation` nodes are
  now pure, immutable descriptions (`kind`/`schema`/`open`) with no execution
  state, so copying an `AnyRelation` is an O(1) share and a plan is an
  inspectable, rewritable, reusable template. `Relation.open(ctx)` builds a
  matching `Processor` (`InMemoryTableProcessor`/`FilterProcessor`/
  `ProjectProcessor`/`AggregateProcessor`/`JoinProcessor`, in `execution.mojo`)
  that owns all mutable state (scan offset, built hash index, grouper, child
  processors), erased behind `AnyProcessor` which drives `collect()`.
  `execute(plan)` opens a fresh processor tree and drains it, so the plan is
  never mutated and runs repeatedly or concurrently. The dependency is one-way
  (`relations` → `execution`; the execution layer needs only the value box and
  kernels). This supersedes the reset-on-copy fat-node mechanism — plan
  reusability now falls out of node immutability for free — and gives an
  optimizer a clean IR to rewrite. Fused path unchanged (`query_streaming`
  448 KB, interpreter DCE'd).
- **The value box erases behind `Value`** (`marrow.expr.values`): `to_array` and
  `field_name` moved onto the `Value` trait itself (the `NumericValue` /
  `BoolValue` / `StringValue` sub-traits default `to_array` via their fused
  `execute()`; the named column leaves override `field_name`), so `AnyValue`'s
  three `@implicit` constructors and six trampolines collapse to one
  `__init__[V: Value]` and one trampoline pair. Removes the separate `Boxable` /
  `Named` / `Column` traits. The fused path still dead-code-eliminates the
  interpreter — `query_streaming` stays 448 KB, 12.7× smaller than the `DynValue`
  paths.
- **Single named column representation** (`marrow.expr`): removed the positional
  `NumericColumn[T](index)` / `StringColumn(index)` leaves that duplicated the
  name-resolved leaves in `relations.mojo`; the fused algebra
  (`Add`/`Gt`/`Length`…) composes over the `col(name, dtype)` leaves the same
  way. Removed the stale `bench_fused.mojo`, which imported the long-removed
  `marrow.expr.fused` module and no longer compiled.
- **No per-morsel expression-tree reconstruction** (`marrow.expr.dynamic`):
  `DynValue.to_array` ran `resolve_names(schema).eval(batch)` on every morsel,
  deep-copying the whole expression tree each time; `eval` now resolves a named
  column reference inline (one schema lookup, no allocation) and `to_array`
  calls it directly.
- **Encapsulated join key extraction** (`marrow.expr`): the join builder read
  `DynValue._kind_data` directly and assumed each key was a bare column
  reference; a new public `DynValue.column_index(schema)` resolves a key to its
  position and raises on a computed (non-column) key.
- **Trimmed unimplemented join surface** (`marrow.expr.relations`): dropped the
  never-read `algorithm` parameter from `.join()` and stopped re-exporting
  `JOIN_ALGO_*`, `JOIN_ASOF`, and `JOIN_CROSS`/`JOIN_SINGLE`/`JOIN_MARK` (the
  hash-join path implements inner/left/right/full/semi/anti with all/any
  strictness).

### Features

- **AOT typed tables declared as plain dtype-tag structs** (`marrow.aot.relations`):
  a plain struct declares its columns as bare dtype fields (`var a: Int64Type`,
  `var name: StringType`) with no column-node wrappers and no `__init__`, and
  `Table[Orders]()` is a column-access handle whose `t.a` / `t.name` reflect
  each field's dtype into `NumericColumn[T]` / `StringColumn` (numeric vs string
  is dispatched by a `where` clause on the reflected field type). Replaces the
  previous `var a: NumericColumn[Orders, "a", Int64Type]`-style fields +
  hand-written `__init__` boilerplate. The named columns carry only a runtime
  `name` (the sole type parameter is the dtype), so a query with N same-dtype
  columns instantiates one column type, not N — the name never affects the
  generated SIMD compute, and the position is resolved by name against the batch
  schema at execution. The positional and named numeric column nodes are renamed
  `Column` → `NumericColumn` to pair with `StringColumn` per type family, and
  both named leaves share a new `Column` base trait exposing `to_array()`, so
  `Project[*Es: Column]` assembles a projection with no numeric-vs-string
  branching.
- **Polars-style `col(name, dtype)` column factory** (`marrow.aot.relations`):
  `col("a", int64)` / `col("name", string)` reference a column by name without a
  schema struct or handle — overloaded on the dtype's trait so the numeric case
  returns `NumericColumn[T]` and the string case `StringColumn`, both fully
  composable (`Add(col("a", int64), col("b", int64))`, `Project`/`Filter`).
  Produces the same name-carrying leaf as `Table[Tbl]()`; the two differ only in
  whether the dtype is read off a struct or spelled explicitly.

### Build Infrastructure

- PyPI wheel packaging via cibuildwheel and hatchling; Mojo runtime dylibs bundled
  with delocate (macOS) / auditwheel (Linux). Local build: `pixi run -e wheel build_wheel`.
- `python/__init__.py` added; `marrow.so` renamed to `libmarrow.so` to distinguish
  the internal Mojo extension from the `marrow` Python package.
- `build_python` now explicitly passes `-O3 -g0` to `mojo build`.

### Refactors

- **`marrow.expr` split into two independent top-level packages, `marrow.aot`
  and `marrow.dyn`** (previously `marrow/expr/{values,typed}.mojo` and
  `marrow/expr/{runtime,relations,executor}.mojo` under one `marrow.expr`
  parent): `marrow.aot` is the comptime-typed, fully-monomorphized
  implementation; `marrow.dyn` is the type-erased, runtime-dispatched one
  (what the Python bindings drive). `comptime` is a reserved Mojo keyword and
  can't be a module name; `aot`/`dyn` were chosen to match the existing
  `docs/aot-*-design.md` / `docs/dynamic-dispatch-design.md` naming. One-way
  dependency: `dyn.values` imports `NumericValue`/`BoolValue` from `aot.values`
  to declare its `Expr(value)` boxing constructors' generic bounds; `aot`
  imports nothing from `dyn`. Tests relocated to `marrow/aot/tests/` and
  `marrow/dyn/tests/` to mirror.
- **`JOIN_*` kind/strictness/algorithm-hint constants moved from
  `marrow.dyn.relations` to `marrow.kernels.join`**: the join kernel owns
  this vocabulary; the relational-plan layer is a consumer, not the owner.
  `relations.mojo` now imports and re-exports them for existing callers.
- **`NumericValue.execute()` is now a single default trait implementation**
  (`marrow/expr/values.mojo`): the identical fused-vectorize-loop body
  previously copy-pasted into `Column`, `Add`, `Sub`, and `Length` now lives
  once in the trait itself (calling `self.core[W]()`, matching the existing
  pattern already used for `dtype()`); each node only defines its own
  `core[W]()`.
- **Boxing a comptime node into `Expr` is now an `Expr(value)` constructor**
  (`marrow/expr/runtime.mojo`), replacing the `NumericValue.to_expr()` method;
  the fused-node trampolines moved alongside it into `runtime.mojo`. Also
  slimmed `Expr`'s operator overloads (shared `_binary`/`_unary` builders) and
  `write_to()` (single `_op_name()` lookup + generic arg formatting), dropping
  ~160 lines with no behavior change.
- **`marrow/expr` split into two clear layers** — `values.mojo` now holds the
  comptime-typed expression layer (`Column[T]`, `Add[L, R]`, `Sub[L, R]`,
  renamed from `FusedColumn`/`FusedAdd`/`FusedSub` since these are the default
  expression layer, not an alternate "fused" one), while `runtime.mojo` holds
  the type-erased `Expr` used to build/execute plans without concrete comptime
  types (what the Python bindings drive). The two share a single canonical
  `Value` trait (previously duplicated verbatim in both files).
- **Collapsed the value-processor hierarchy into `Expr.eval()`**
  (`marrow/expr/executor.mojo`): removed `ValueProcessor`, `AnyValueProcessor`,
  `ColumnProcessor`, `LiteralProcessor`, `BinaryProcessor`, `UnaryProcessor`,
  `IsNullProcessor`, `IfElseProcessor`, `FusedProcessor[T]`, and
  `Planner.build(Expr)` — `Expr` already carries its own tag and args, so it
  now dispatches its own execution directly. A boxed comptime-typed node
  (`FUSED` tag, produced by `to_expr()`) delegates straight back to its own
  fused `execute()` via a new trampoline, so `Expr.eval()` can now run a
  boxed fused subtree end-to-end through the runtime path — previously
  `Planner.build` raised on `FUSED` and required calling `FusedProcessor`
  directly. Relation processors (`FilterProcessor`, `ProjectProcessor`,
  `AggregateProcessor`) now hold `Expr`/`List[Expr]` fields directly instead
  of `AnyValueProcessor`/`List[AnyValueProcessor]`.
- Removed the dead `Expr.dispatch` field (always `0`, never read) and the
  `col(name)` sentinel that wrapped `-1` into a `UInt8` `kind_data`. Fixed
  `Expr.write_to()` to actually call the fused-node write trampoline for
  `FUSED` nodes instead of printing a fixed placeholder string.
- **Renamed `NumericTypedValue` → `NumericValue` and merged `TypedValue` into
  `Value`** (`marrow/expr/values.mojo`): `TypedValue` only re-stated `Value`'s
  bounds without adding new behavior, so it's gone — `Value` itself now
  carries the `Copyable`/`Writable`/etc. bounds every conformer already needed.
- **`Expr` literals now store an `AnyScalar` instead of a length-1 `AnyArray`**
  (`marrow/expr/runtime.mojo`): `lit[T]()` builds a `PrimitiveScalar[T]`
  directly rather than round-tripping through a one-element
  `PrimitiveBuilder`. Broadcasting a literal to a full batch now goes through
  the new `marrow.scalars.repeat(AnyScalar, times) -> AnyArray` free function.
- **`PrimitiveBuilder[T].extend(scalar, n)` and `AnyBuilder.extend(scalar,
  times)`** (`marrow/builders.mojo`): append the same scalar value `n` times
  in one call. `PrimitiveBuilder[T]`'s version is a single generic method
  (parameterized over every numeric type at once); `AnyBuilder`'s dispatches
  to it by dtype, mirroring the existing `AnyBuilder(dtype, capacity)`
  constructor's dispatch style. `marrow.scalars.repeat()` is now a thin
  3-line wrapper (`AnyBuilder(dtype, times).extend(scalar, times).finish()`)
  instead of an 11-branch per-dtype broadcast loop.

### Features

- **`marrow.aot` — a fully-monomorphized (AOT) relational layer**:
  `Schema.from_struct[T]()` (`marrow/schema.mojo`) derives a `Schema` from a
  marker struct via compile-time reflection; `Table`, `Column[Tbl, name, T]`,
  `StringColumn[Tbl, name]` (`marrow/aot/relations.mojo`) resolve a column's
  position as a `comptime` constant via `reflect[Tbl].field_index[name]()` —
  no runtime `Schema` lookup, ever; `BoolValue` + `Lt`/`Gt`/`Eq`
  (`marrow/aot/values.mojo`) give fused, bit-packed-`BoolArray` comparisons;
  `Project[*Es]`/`Filter[Input, Pred]` (`marrow/aot/relations.mojo`) compile a
  `SELECT`/`WHERE`-shaped query into fused SIMD loops with no tag dispatch.
  See `docs/aot-relations-design.md`.
- **`Expr`'s `FUSED` boxing constructor now also accepts `BoolValue` nodes**
  (`marrow/dyn/values.mojo`), not just `NumericValue` — lets a comptime
  `Lt`/`Gt`/`Eq` predicate drive a runtime `AnyRelation.filter()` plan.
- **Binary-size benchmark** (`benchmarks/binary_size/`): three files
  implement the identical query via `marrow.aot`, `marrow.dyn`, and a hybrid
  (runtime plan + AOT-fused predicate), showing the fully-monomorphized
  version compiles ~33x smaller (stripped). `pixi run binary_size` runs
  `compare.py`, which builds, strips, and reports a size/symbol-count table
  plus a per-module symbol breakdown.
- **String `Length` expression node + `.length()`** (`marrow/expr/values.mojo`,
  `marrow/expr/runtime.mojo`, `marrow/kernels/string.mojo`): computes
  per-element string byte lengths through both expression layers. Adds a
  `StringValue` trait (mirrors `NumericValue` but resolves to a `StringArray`
  instead of a per-lane SIMD `core[W]()`) and a `StringColumn` leaf node; the
  comptime `Length[S: StringValue]` node implements `NumericValue` with a
  SIMD-vectorized `core[W]()` that loads `W+1` contiguous string offsets and
  subtracts the shifted-by-one lanes, so it composes into a fused pass with
  other numeric nodes. The runtime `Expr` gains a `LENGTH` tag and `.length()`
  method that dispatch to a new type-erased `string_lengths(AnyArray)`
  overload, matching the existing typed-overload-plus-`AnyArray`-blanket
  kernel pattern.

- **`Schema[Field[...]]` with `__getattr_param__`** (`marrow/faszom.mojo`): compile-time
  schema type that enables Ibis-style `t.data.where(t.a + t.b > t.c).execute(batch)`
  syntax without per-field boilerplate. `Schema[Field['a', Int32Type], Field['data', Float32Type]]`
  returns `ColumnRef['a', Int32Type]` for `t.a` via `__getattr_param__`, using a
  `@staticmethod def _name_matches` trait + compile-time recursive index lookup
  (`_schema_find_idx`) to resolve the field type at compile time.

- **ColumnRef / Pipeline / FilterPipeline** (`marrow/faszom.mojo`): named column
  placeholders (`ColumnRef['name', T]`) resolved from a `RecordBatch` at execute
  time via `bind()`. Enables reusable AOT-compiled query pipelines that are defined
  once and called per batch. Convenience factories: `col['name'](dtype)`,
  `filter_pipeline['data_col'](pred, dtype)`. `FilterPipeline` and `Pipeline`
  wrappers bind all `ColumnRef` nodes in `O(cols)` and execute the fused loop in
  `O(N)`. The AOT specialization property is preserved — each distinct
  `(name, T)` pair remains a unique compile-time type.

- **`PrimitiveArray.__eq__` correctness fix** (`marrow/arrays.mojo`): the fast
  path now compares only the valid `length` elements instead of the full allocated
  buffer, preventing spurious mismatches for filtered arrays whose backing buffer
  is over-allocated.

- **Sort kernel — `argsort` and `sort`** (`marrow/kernels/sort.mojo`):
  single-column sort for all array types. Primitive arrays use LSD radix sort
  (O(N), 8-bit passes, UInt64-encoded keys, float NaN/sign-bit transform) for
  N ≥ 32 768, with parallel histogram + scatter for N ≥ 524 288. PDQsort for
  N < 32 768 (faster on Apple M-series up to ~28K elements); insertion-sort
  leaf for N < 32. `BoolArray` uses O(N) counting sort; `StringArray` uses the
  Mojo stdlib comparison sort. Null partitioning (pre-sort bitmap scan) with
  `nulls_first`/`nulls_last` placement. `sort(StructArray, key_indices,
  ascending)` wraps `argsort` + `take` for multi-column sort.

- **Large binary, string, and list types** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  added `LargeBinaryType`, `LargeStringType`, `LargeListType` (64-bit offsets);
  `BinaryLikeType` trait with `comptime offset: DType` and `StringLikeType` sub-trait
  for UTF-8 kernels; unified `BinaryArray[T: BinaryLikeType]` and
  `BinaryBuilder[T: BinaryLikeType]` with aliases `StringArray`, `LargeBinaryArray`,
  `LargeStringArray`, `StringBuilder`, `LargeBinaryBuilder`, `LargeStringBuilder`;
  IPC type codes 19/20/21 for large binary/utf8/list; C Data format codes `Z`/`U`/`+L`.

- **IPC support for dictionary-encoded columns** (`marrow/ipc.mojo`): the IPC
  file and stream writer now emits a `DictionaryBatch` message (header type 2)
  for each dictionary column before its first `RecordBatch`, encoding the
  column's value array as a separate body. The `RecordBatch` body carries only
  the integer indices. Dictionary blocks are registered in the IPC file footer so
  C++ / Rust / Go readers can locate them. The IPC reader detects
  `DictionaryEncoding` at schema-field slot 4, reconstructs `DictionaryType`
  (index type + value type + ordered flag), loads `DictionaryBatch` messages via
  footer-registered block offsets, and wires the decoded values back into
  `DictionaryArray` instances when reading record batches. Validated across all
  Arrow implementations (`dictionary` and `dictionary_unsigned` pass 14/14
  integration phases with C++, Rust, and Go).

- **Arrow interval types** (`marrow/{dtypes,scalars,arrays,builders,ipc,c_data}.mojo`, `python/`):
  added `IntervalType` trait and three concrete types — `YearMonthIntervalType` (int32, months),
  `DayTimeIntervalType` (int64, days+millis), `MonthDayNanoIntervalType` (int128, months+days+nanos).
  `AnyDataType` gains `is_interval()`, `is_year_month_interval()`, `is_day_time_interval()`,
  `is_month_day_nano_interval()` predicates and matching `as_*` accessors. Array, builder, and
  scalar aliases (`YearMonthIntervalArray/Builder/Scalar`, etc.) are fully wired into the
  `AnyArray`, `AnyBuilder`, and `AnyScalar` type-erased containers. C Data Interface uses
  format codes `tiM`, `tiD`, `tin`; IPC uses the `Interval` flatbuffer type with unit field.
  Python bindings expose `year_month_interval()`, `day_time_interval()`,
  `month_day_nano_interval()` factory functions.

- **Dictionary-encoded Arrow type** (`marrow/{dtypes,scalars,arrays,builders,
  c_data}.mojo`): added `DictionaryType` (index type + value type + ordered
  flag), `DictionaryScalar`, `DictionaryArray`, and `DictionaryBuilder`.
  `DictionaryArray.from_arrays(indices, values)` constructs from an integer
  indices array and an arbitrary values array; `__getitem__` decodes to the
  underlying value scalar; `slice()` is zero-copy. The C Data Interface emits
  the index type's format string and stores the value schema in the `dictionary`
  field of `CArrowSchema`, with `ARROW_FLAG_DICT_ORDERED = 1` when ordered;
  import detects a non-null `dictionary` field and reconstructs the type.
  Enables zero-copy exchange of PyArrow `DictionaryArray` via the Arrow C Data
  Interface (`__arrow_c_array__` / `__arrow_c_schema__` protocol).

- **Arrow Null type** (`marrow/{arrays,scalars,builders,ipc,c_data}.mojo`,
  `python/arrays.mojo`): added `NullArray`, `NullScalar`, `NullBuilder`
  (registered in the `AnyArray`, `AnyScalar`, `AnyBuilder` variants); IPC
  writer emits `Type.Null = 1` with zero body buffers; IPC reader skips the
  validity slot for null fields; C Data Interface uses `n_buffers = 0` for null
  per the spec; Python factory `ma.array(seq, type=ma.null())` builds a
  `NullArray` of the given length.

- **Fixed-size binary type** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  added `FixedSizeBinaryType`, `FixedSizeBinaryArray`, `FixedSizeBinaryBuilder`;
  C Data format code `"w:<n>"`; IPC type code 15 (FixedSizeBinary).

- **Temporal array types** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  `Date32Array`, `Date64Array`, `Time32Array`, `Time64Array`, `TimestampArray`,
  `DurationArray` with matching builders and type singletons; C Data format
  codes (`"tdD"`, `"tdm"`, `"tts"`, `"ttu"`, `"tsn:"`, `"tDn"`, etc.); IPC
  type codes and unit serialisation. Python constructors `ma.date32()`,
  `ma.date64()`, `ma.time32(unit)`, `ma.time64(unit)`, `ma.timestamp(unit)`,
  `ma.duration(unit)`.

- **Decimal types in C Data Interface and IPC**
  (`marrow/c_data.mojo`, `marrow/ipc.mojo`): wired `Decimal32Type`,
  `Decimal64Type`, `Decimal128Type`, `Decimal256Type` into schema export/import
  and IPC flatbuffer serialisation (precision, scale, bit-width).

- **Custom metadata round-trip via the C Data Interface**
  (`marrow/c_data.mojo`): `CArrowSchema.from_field` / `from_schema` now
  encode `Field.metadata` and `Schema.metadata` into the spec-defined
  metadata blob; `to_field` / `to_schema` decode it back. New
  `_encode_c_metadata` / `_decode_c_metadata` helpers handle the
  `int32 num_pairs ; (int32 key_len, key_bytes, int32 val_len, val_bytes)*`
  layout. `from_schema` now takes a full `Schema` rather than `List[Field]`
  so schema-level metadata flows through.

- **Per-field metadata** (`marrow/dtypes.mojo`, `python/dtypes.mojo`):
  `Field` carries an optional `metadata: Dict[String, String]`; the Python
  factory `ma.field(name, type, metadata={…})` accepts a dict; the C Data
  Interface and IPC flatbuffer encoder/decoder round-trip field-level
  key-value metadata.

- **Preserve nested-field names in IPC reader and C Data Interface**
  (`marrow/ipc.mojo`, `marrow/c_data.mojo`): the IPC `_read_field`
  decoder and the `CArrowSchema` list / fixed_size_list importer now preserve
  child Field names as-is, so Arrow files written by other implementations
  round-trip with the original schema.

- **Arrow IPC reader/writer** (`marrow/ipc.mojo`): `read_ipc_file()`,
  `write_ipc_file()`, `read_ipc_stream()`, `write_ipc_stream()`,
  `read_ipc_file_schema()`, `read_ipc_stream_schema()`, and streaming struct
  variants `RecordBatchFileReader`, `RecordBatchStreamReader`,
  `RecordBatchFileWriter`, `RecordBatchStreamWriter`. Supports all implemented
  Arrow types (bool, int8–64, uint8–64, float16/32/64, binary, utf8, list,
  fixed_size_list, struct, dictionary, null, temporal, decimal) with full
  nested and nullable column support. FlatBuffer encoding/decoding is a
  self-contained Rust-faithful port with correct soffset sign convention and
  `MetadataVersion::V5`.

- **GPU aggregate reductions** (`marrow/kernels/aggregate.mojo`):
  `sum_`, `min_`, `max_`, `product`, `any_`, `all_` now accept an
  `ExecutionContext`; when `.is_gpu()` is true the reduction runs as a
  single-pass GPU kernel via `_reduce_generator_wrapper`.

- **`ExecutionContext`** (`marrow/kernels/execution.mojo`): new struct bundling
  `num_threads` for CPU stripe parallelism and `device: Optional[DeviceContext]`
  for GPU. Implicit conversions from `Optional[DeviceContext]` and
  `DeviceContext` keep existing callers working. Factories: `.serial()`,
  `.parallel(num_threads=0)` (0 = `num_physical_cores()`), `.gpu(device)`.
  Wired through all kernels: arithmetic, aggregate, compare, filter, join, sort.

- **Partition-parallel hash join** (`marrow/kernels/join.mojo`,
  `marrow/kernels/hashtable.mojo`): `HashJoin` and `hash_join()` gain a
  `num_threads` argument. The parallel path radix-partitions both sides by the
  top bits of their hash into independent `SwissHashTable` instances, builds and
  probes them concurrently via `sync_parallelize`, and concatenates per-partition
  index pairs. No atomics on the hot path. At 10M×10M INNER join: **330 ms
  (serial) → 67 ms (parallel, 4.9× speedup)** — faster than Polars (97 ms),
  PyArrow (111 ms), and DuckDB (122 ms).

- **`RadixPartitioner`** (`marrow/kernels/hashtable.mojo`): partitions hashes +
  row indices by the top `num_bits` (default 6 → 64 partitions). Per-thread
  histogram → partition-major prefix sum → parallel scatter into shared flat
  buffers, then per-partition zero-copy slice via `ArcPointer`-shared immutable
  buffers.

- **Parallel per-column `take()`** (`marrow/kernels/filter.mojo`):
  `take[T](PrimitiveArray, indices, ctx)` and the `AnyArray` dispatcher
  accept an `ExecutionContext` and stripe the no-null fast path across workers.
  End-to-end 10M inner join assembly: **143 ms → 67 ms**.

- **Variant-based dispatch for `DataType`, `AnyArray`, and `Builder`**
  (`marrow/dtypes.mojo`, `marrow/arrays.mojo`, `marrow/builders.mojo`):
  Replaced integer-code dispatch with `Variant`-backed types using `comptime
  for` loops. Eliminates runtime `if`/`elif` chains across kernels, Python
  bindings, and the expression system.

- **`BoolArray` dedicated type** (`marrow/arrays.mojo`): bit-packed boolean
  arrays backed by a `Bitmap`, with `.values() -> BitmapView`, GPU transfer,
  and a matching `BoolBuilder`.

- **`BufferView` / `BitmapView` abstractions** (`marrow/views.mojo`):
  type-safe, non-owning views with `apply` dispatch, `compressed_store`,
  `pext`, and GPU-aware access.

- **`SwissHashTable`** (`marrow/kernels/hashtable.mojo`): open-addressing hash
  table with 7-bit control stamps, CSR chain storage, vectorised SIMD group
  matching, and a batch-build API.

- **Hash join** (`marrow/kernels/join.mojo`): `hash_join` kernel using
  `SwissHashTable` with separate build and probe phases.

- **`TestSuite` and `BenchSuite` framework** (`marrow/testing`):
  auto-discovery of `test_*` / `bench_*` functions via
  `__functions_in_module()`, with pytest harness integration, competition
  tables, and per-element throughput metrics.

- **AddressSanitizer support**: `pytest --asan` compiles test runners with ASAN
  instrumentation via `libcompiler-rt`.

- **GPU `BitmapView` and GPU rapidhash** (`marrow/kernels/`): `BitmapView`
  supports device-resident bitmaps; `rapidhash` ported to Metal/CUDA with
  128-bit multiply emulation.

- **Bounds checking** (`marrow/buffers.mojo`): `Buffer`, `Bitmap`, and
  `BufferView` accessors assert bounds in debug builds.

- **Unary math kernels** (`marrow/kernels/arithmetic.mojo`): `sign`, `sqrt`,
  `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `trunc`,
  `round`, `sin`, `cos` (floating-point), plus binary `pow_`, `floordiv`, `mod`.

- **Scalar types** (`marrow/scalars.mojo`): `PrimitiveScalar[T]`,
  `StringScalar`, `ListScalar`, `StructScalar`, `AnyScalar` — typed and
  type-erased scalar values mirroring the array hierarchy.

- **Group-by kernel** (`marrow/kernels/groupby.mojo`): fused
  `groupby(keys, values, aggregations)` that hashes, groups, and aggregates in
  a single pass. Supports `"sum"`, `"min"`, `"max"`, `"count"`, `"mean"`.
  Single-key (any primitive/string `AnyArray`) and multi-key (`StructArray`)
  grouping.

- **Hashing kernel** (`marrow/kernels/hashing.mojo`): `hash_` for primitive,
  string, and struct arrays; `hash_identity` for bool/uint8/int8.

- **Expression execution system** (`marrow/expr/`): pull-based streaming query
  executor with `col()`, `lit()`, `if_else()`, relational plan nodes
  (`InMemoryTable`, `Filter`, `Project`, `ParquetScan`, `Aggregate`), and
  `execute()` to collect `RecordBatch` results.

- **Parquet I/O** (`marrow/parquet.mojo`): `read_table(path)` and
  `write_table(table, path)` via the Arrow C Stream Interface.

- **Comparison kernels** (`marrow/kernels/compare.mojo`): `equal`,
  `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` for typed and
  runtime-typed arrays; null-propagating; GPU variants available.

- **String kernels** (`marrow/kernels/string.mojo`): `string_lengths` returns
  byte lengths for each element.

- **RecordBatch column operations** (`marrow/tabular.mojo`): `slice`,
  `select`, `rename_columns`, `add_column`, `append_column`, `remove_column`,
  `set_column`, `to_struct_array`.

- **Table enhancements** (`marrow/tabular.mojo`): `Table.from_batches`,
  `Table.to_batches`, `Table.combine_chunks`.

- **Schema enhancements** (`marrow/schema.mojo`): `get_field_index`, `field`
  lookup by name, `names()`, equality operators, Python interop via Arrow C
  Data Interface.

- **Self-contained archery integration suite** (`integration/`, `pixi.toml`):
  `pixi run integration` clones apache/arrow + arrow-rs + arrow-go, builds all
  reference implementations, and runs cross-implementation tests against C++,
  Rust, Go, and Mojo. All four implementations pass: 119 cases across 14
  directional phases.
