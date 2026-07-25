# Changelog

## [Unreleased]

### Refactors

- **`filter`, `take`, `sort` and `hashing` no longer reinterpret temporal columns.**
  Every typed kernel leaf is bound on `PrimitiveType`, which the temporal, interval
  and decimal types already satisfy — so the dispatch layer now hands the column
  straight to the leaf via `dispatch_temporal` / `dispatch_primitive` instead of
  reinterpreting it to an integer backing and relabelling the result. For
  `filter`/`take` this is also *more* correct: the output is a `PrimitiveArray[T]`
  whose dtype is preserved by construction, rather than stripped and reattached.
  `AnyDataType.storage_type()` is deleted (no callers remain).

  Cost is monomorphization — ~15 logical primitive types now instantiate the
  kernels rather than 4 integer widths — but it lands on the runtime binary, not
  the AOT one: the **fused `query_streaming` binary shrank 6.1%** (1,357,176 →
  1,274,584 bytes stripped) since it no longer links the reinterpret machinery,
  while `query_dynvalue` grew 5.2%. Fused size is what the small-binary property
  is about, so this is a win on the metric that matters.


### Refactors

- **`dispatch_over_*` are now methods on `AnyDataType`** (`dt.dispatch_numeric[f]()`
  rather than `dispatch_over_numeric[f](dt)`), across 47 call sites. They existed
  only to read `AnyDataType._v` — a private field — from another module, so as
  methods the accessor is unnecessary and `_v` is private again. This also removed
  the last import cycle: `utils.mojo` imported nine dtype names solely for these,
  while `dtypes.mojo` imported `variant_dispatch` back. `utils.mojo` now has no
  local imports at all, and the module graph is fully acyclic.


### Fixes

- **`sort_indices` and `rapidhash` now accept every Arrow dtype** — temporal
  (date/time/timestamp/duration), interval, `large_string`, `large_binary`,
  `binary`, decimal (32/64/128/256), and dictionary. Both were hand-written
  `if dtype == … elif …` ladders that stopped at bool/numeric/string and then
  raised, so `GROUP BY` on any temporal key and `ORDER BY <timestamp>` failed
  outright (`docs/code-quality-review.md` D5; ClickBench Q8, Q19, Q24–27, Q35,
  Q36, Q40, Q43 — the `hits` table is keyed on `EventDate`/`EventTime`).
  Supersedes FU-1, which tracked only the `large_string` half.

  Behavioural notes: a dictionary column now hashes and orders by its *decoded*
  values, so it matches the equivalent plain column; and `decimal128`/`decimal256`
  hash by folding both 64-bit limbs instead of truncating to the low one — a
  truncating hash would have merged distinct groups, since group-by buckets on
  the hash alone.

### Refactors

- **`hashing.mojo` → `RapidHash`, `sort.mojo` → `SortIndices`** — the last two
  kernel modules with no `Kernel` struct, and (not coincidentally) exactly the
  two with dtype-coverage gaps: with no single `dispatch`, each new dtype had to
  be remembered in a hand-written ladder, and wasn't. Both now follow the
  three-tier pattern (typed `apply` leaves, one `dispatch`), with thin free
  delegators kept for the call sites that bind a hasher as a comptime function
  value (`SwissHashTable[hasher]`) or match PyArrow (`sort_indices`, `sort`).
  `sort`'s multi-key body moved to `SortIndices.multi`, which returns the
  permutation, leaving `sort` as `take ∘ SortIndices.multi`. The public free
  function literally named `array()` in `sort.mojo` — which forced
  `import array as _primitive_array` in its own file — is now
  `SortIndices._assemble`. Verified-dead `hash_identity` (×3) deleted.

- **Completed the `dispatch_over_*` family** (`marrow/utils.mojo`): added
  `dispatch_over_primitive`, `dispatch_over_integer`, `dispatch_over_temporal`,
  and `dispatch_over_listlike` alongside the existing numeric / floating /
  string-like / binary-like members, so there is one member per dtype-family
  trait and a kernel never has to spell out its own ladder. They now reach the
  variant through the new `AnyDataType.variant()` accessor instead of the
  private `dt._v`.

- **`AnyDataType.storage_type()`** (`marrow/dtypes.mojo`) — the same-width
  signed-integer dtype a fixed-width logical type is stored as. Generalizes
  `temporal_backing_dtype` to interval and decimal32/64, and lets the
  value- and order-preserving kernels route those columns through the
  already-instantiated integer path rather than growing one arm (and one kernel
  instantiation) per logical dtype.

- **Broke the `dtypes` ⇄ `arrays`/`scalars` circular import.** `DataType` declared
  `comptime ScalarType` / `comptime ArrayType` companion types on the trait and on
  ~30 concrete types, which is what forced `dtypes.mojo` to import `arrays` and
  `scalars` — while those modules import `dtypes` back. **Nothing consumed them**
  (the only `.ScalarType`/`.ArrayType` uses in the tree are `Self`-qualified
  members of the separate `Array` and `Builder` traits), so removing them costs
  nothing and deletes 63 lines. `dtypes.mojo` now imports only `.utils`, making
  the module graph a DAG.

  The cycle had caused three distinct build failures, all with the same
  signature — the same source compiling or failing depending on which file the
  build entered through: a `Scalar` trait shadowing the builtin `Scalar[dtype]`,
  `BoolArray` resolving along one import path but not another, and a rewrite that
  moved errors 2 → 10 by shifting the ambiguity rather than removing it.


### Fixes

- **Fused mixed-width numeric comparison no longer truncates.** `NumericCompare`
  took the *left* operand's native type and cast the right operand down into it,
  so `col("a", int32) > col("b", int64)` silently narrowed every int64 value.
  It now compares in the promoted domain of both operands — the same
  `promote[L, R]` rule `NumericBinary` already used, so `a > b` and `a + b` can
  no longer disagree about widening. The SIMD lane width is a separate question
  (a narrower dtype yields a *larger* `W`, which would overflow a wider
  operand's load), so that is now expressed by a dedicated `wider[L, R]` alias
  which `BoolBinary` also reuses in place of its hand-inlined copy.

- **Fused `Any` / `All` no longer ignore nulls.** Two `AnyKernel`/`AllKernel`
  pairs existed — one in `kernels/boolean.mojo` that counted set bits with **no
  validity mask**, and the null-correct pair in `kernels/aggregate.mojo` that
  `kernels/__init__.mojo` re-exports. The expression layer imported the former,
  so a null slot whose data bit happened to be set read as `True`. The duplicate
  is deleted and the expression layer now binds the correct pair.

- **`DynValue.name()` no longer leaks operator payloads as column names.**
  `_name` is overloaded to carry the LIKE/ILIKE pattern and the `date_trunc`
  unit, and `name()` returned it unconditionally — so a `LIKE` node reported
  `"%foo%"` and a `DATE_TRUNC` node `"day"` as its output column name. It now
  returns an empty string unless the node is a `LOAD`.

- **`BoolScalar.repeat` added.** `AnyScalar.repeat` raised
  `unsupported dtype bool`, so broadcasting a boolean scalar to an array was
  impossible — surfaced by the new `Any`/`All` parity tests.


### Refactors

- **Renamed the `Scalar` trait to `ArrowScalar`** (`marrow/scalars.mojo`). The
  old name shadowed the builtin `Scalar[dtype]` alias, and because `arrays.mojo`
  and `dtypes.mojo` wildcard-import each other, the bare name resolved to the
  trait along that cycle and `Scalar[T.native]` stopped parsing. Both existing
  workarounds — `Scalar as ScalarTrait` in `arrays`/`dtypes` and
  `Scalar as _Scalar` in `scalars` — are gone with it.

### Build

- **Upgraded Mojo to `1.0.0b3.dev2026072406`** (pinned exactly in all three
  places in `pixi.toml`). This **fixes a long-standing heap corruption**: boxing
  a `DynValue` into an `AnyValue` wrote a `Variant` discriminant one byte past
  its `ArcPointer` allocation, silently corrupting the heap until it hit live
  allocator metadata. It accounted for *every* failure in the full test suite
  (`parquet/tests/test_reader.mojo`, `expr/tests/test_streaming.mojo`, 59 in
  total); both files now pass, and ASAN reports 0 `heap-buffer-overflow` hits
  where it previously reported 86.

  API migrations required by the upgrade (all per the upstream changelog):
  `Span(ptr=)` → `Span(unsafe_ptr=)`; `memset`/`memset_zero` →
  `unsafe_memset`/`unsafe_memset_zero`; `SIMDSize` → `SIMDLength`;
  `OwnedPointer[T]` and `Optional[T]` now conform to `ImplicitlyDeletable` only
  conditionally (explicit `__del__` on `DictionaryType`, and an
  `ImplicitlyDeletable` bound on `RadixPartitioner.map_partitions`'s result
  type); and `Variant` adopted *interior origins*, so `bitmap_and` call sites
  pass `.copy()`. Dropped the `_accelerator_arch()` GPU-arch validation.

  Adds a `max` dependency, which **should not be required** — GPU kernels
  compiled without it before this release. Tracked as a toolchain regression in
  `pixi.toml`; retry dropping it on the next upgrade.

- **New build-only tasks `check` and `check_lib`** for fast compile-error
  iteration (~4 s for one file, ~10 s for the whole library, versus minutes for
  the equivalent `pytest` run). Documented in `CLAUDE.md`.

### Features

- **Relational `Aggregate` completeness + `HAVING`.** `AnyRelation.aggregate`
  now accepts **computed group keys** (arithmetic, `CASE`/`if_else`,
  `year()`/`date_trunc()`, literals) and **computed aggregate inputs**
  (`SUM(x + 1)`, `AVG(length(s))`) — key and value dtypes are inferred by probing
  the expression against a 0-row batch, the same trick `project` uses.
  `count_distinct` / `approx_count_distinct` are wired through as aggregate
  functions, `min`/`max` work over string and date/timestamp value columns,
  `count` works over any dtype, and an optional `names=` argument aliases the
  aggregate output fields (so `mean(a), mean(b)` no longer collide). A temporal
  group key is grouped through its integer backing and relabelled on emit
  (`rapidhash` has no temporal case). `HAVING` needs no node of its own —
  `rel.aggregate(...).filter(pred)` resolves `pred` against the aggregate output
  schema.
- **Relational operators: `Sort`, `Limit`/`Offset`/`TopK`, computed `Project`.**
  `Sort` (pipeline breaker) reuses the sort kernel over a key+data struct; `Limit`
  slices exactly across morsels; a `Limit`-over-`Sort` folds into the kernel's
  top-K `limit`; `Project` now evaluates arbitrary expressions with output names.
  Plan builders `.sort/.limit/.project` on `AnyRelation`.
- **Wave 1 kernels wired into both expression drivers.** The fused comptime
  `Value` layer (F2) and the runtime `DynValue` interpreter (F1) both expose
  string ordering compares, `like`/`ilike`, `is_in`, `coalesce`/`nullif`/
  `case_when`, and temporal extraction/`date_trunc` — the two drivers agree
  element-for-element (parity harness). (Known gap FU-5: fused `is_in` composed
  under boolean logic; the dynamic path is correct.)
- **Temporal `filter`/`take`.** `Filter`/`Take`/`drop_null` now handle
  date/time/timestamp/duration columns by reinterpreting to the integer backing
  around the numeric path — unblocking filter/sort/join over temporal columns.
- **New compute kernels for analytical queries (ClickBench-driven):**
  - `marrow/kernels/conditional.mojo` — `case_when` (multi-branch, all types),
    `coalesce`, `nullif`, `fill_null`, all via a shared `concat`+`take`
    multiplexer (type-agnostic, null-correct).
  - `marrow/kernels/membership.mojo` — `is_in` over numeric/bool/string, reusing
    `SwissHashTable`/`rapidhash` (the join/distinct basis).
  - `marrow/kernels/temporal.mojo` — `year/month/day/hour/minute/second/`
    `day_of_week/quarter/day_of_year` extraction (Hinnant civil-date) +
    `date_trunc(unit)`, over date/time/timestamp (UTC).
  - `compare.mojo` — string/large_string ordering comparisons (`< <= > >=`);
    `string.mojo` — `like`/`ilike` (SQL `%`/`_`, PyArrow `match_like` semantics).
  - `aggregate.mojo`/`groupby.mojo` — `min`/`max` over string and temporal
    (whole-table and grouped); grouped `count_distinct` as a first-class agg.
- **The fused comptime `Value` lane tracks validity (nulls).** Previously the
  fused numeric/bool lane computed data only and emitted
  `BoolArray`/`PrimitiveArray` with `nulls=0, bitmap=None`, diverging from the
  null-correct dynamic path. Each node now exposes `validity(batch)`: propagating
  ops (arithmetic/compare/unary/cast/length) AND-combine children validity via
  the same `bitmap_and` helper the eager kernels use; Kleene `and_`/`or_` reuse
  the null-correct `AndKernel`/`OrKernel` apply; `is_null`/`not_null`/literals are
  never-null. The fused and dynamic drivers now agree element-for-element on
  nullable input (enforced by the parity harness), and the small-binary DCE
  property is preserved.
- **`referenced_columns()` / `is_deterministic()` on the expression layer.**
  Both the fused comptime `Value` tower (every concrete node) and the runtime
  `DynValue`, surfaced through `AnyValue` via the existing fn-pointer
  trampolines. `referenced_columns()` returns the deduped set of column names an
  expression reads; `is_deterministic()` is an overridable predicate (`True` for
  all current nodes). These are the prerequisite metadata for projection and
  predicate pushdown.
- **`DynValue` gains `mod`/`floordiv`/`xor`/`not_null`** (tags `MOD`,
  `FLOORDIV`, `XOR`, `NOT_NULL`), each wired to its existing kernel, with `%`,
  `//`, `^`, and `.not_null()` builders — narrowing the runtime interpreter's
  gap to the fused algebra.
- **Parquet reader `ByteSource` seam.** A `ByteSource` trait (`size()` +
  zero-copy `read_at(offset, length)`) in `marrow/parquet/source.mojo`;
  `MappedFile` implements it (all mmap syscalls now live there) and
  `ParquetFile[S: ByteSource = MappedFile]` reads through it while keeping the
  `ParquetFile(path)` convenience. Pure refactor toward streaming and remote
  (OpenDAL) scans.
- **LIKE/ILIKE scalar-pattern kernels (compile the pattern once).**
  `LikeKernel.apply(array, pattern)` / `.dispatch(array, pattern)` and the
  `ILikeKernel` equivalents take the pattern as a `StringSlice` instead of a
  broadcast array, so `col LIKE '%const%'` compiles the pattern once per call
  rather than once per row. A new `LikePattern[ignore_case]` carries the
  compiled form and is the single matching implementation behind both the
  scalar and the array × array overloads: it classifies the pattern into the
  literal shapes `foo` / `foo%` / `%foo` / `%foo%` (which match through the
  optimized string primitives, escapes included — `%\%%` compiles to
  `contains('%')`) and otherwise runs a backtracking wildcard matcher directly
  over each row's UTF-8 bytes, with no per-row allocation. `ILIKE` classifies
  each row with one byte scan and skips the microsecond-scale Unicode `lower()`
  whenever the row is ASCII. Measured on ClickBench-style URLs
  (`marrow/kernels/tests/bench_string.mojo`): `LIKE '%google%'` over 1M rows
  799 ms → 7.6 ms (104x), `ILIKE '%GOOGLE%'` over 100k rows 1235 ms → 3.0 ms
  (417x); the array × array path is 3-6x faster too.

### Tests

- **Cross-driver parity harness** (`marrow/expr/tests/test_parity.mojo`):
  `assert_parity` runs a fused `Value` and an equivalent `DynValue` against one
  batch and asserts equal results, so the runtime interpreter can never silently
  diverge from the fused algebra. Covers arithmetic, comparisons, cast, if_else,
  null propagation, and Kleene `and_`/`or_` over nullable masks.

### Fixes

- **Kleene 3-valued logic in the boolean kernels.** `and_`/`or_`/`xor`/`not_`
  no longer drop the validity bitmap: `TRUE OR NULL = TRUE`,
  `FALSE AND NULL = FALSE`, etc., matching `pc.and_kleene`/`pc.or_kleene`. All
  bitmap combination goes through the idiomatic `Bitmap` bitwise API.
- **`LengthKernel` / `ArrayLengthKernel` propagate nulls** — a null input
  element now yields a null output element, matching `pc.utf8_length` /
  `pc.list_value_length`.

### Refactors

- **One home for grouped aggregate dispatch and for the aggregate output-dtype
  rule.** `GroupBy.aggregate_column(gids, value, num_groups, tag)` is now the
  public per-column entry point for every runtime-tagged aggregate (distinct
  kernels, string/temporal min/max, non-numeric `count`, typed `AggState`
  folds), used by both the kernel-level multi-aggregate drivers and the
  expression layer's `AggregateProcessor` — which drops its numeric-only
  `AggState` copy and instead `concat`s each aggregate's buffered morsels once
  and calls it. The output-dtype rule moves next to the tag helpers as
  `agg_out_dtype(tag, value_dtype)` in `kernels/aggregate.mojo`, replacing
  `AggregateProcessor.out_dtype`, so plan-time schemas and the kernels can never
  disagree.
- **The staged, strategy-pluggable fusion engine is now `marrow.expr.values`
  and drives the relational engine.** The from-scratch engine (previously
  prototyped as `lane.mojo`) replaces the old `values.mojo`: `execution.mojo`
  and `relations.mojo` execute plans over its `AnyValue`, which dual-boxes
  either a comptime `Value` node or a runtime `DynValue`. `AnyValue.write_to`
  renders a boxed `DynValue`'s full expression form so plan printing is
  unchanged. The four family tests (`test_lane*`) are consolidated into a single
  `test_values`; the redundant `test_erased` / `test_relations` (old comptime
  `Table`/`AnyValue` surface) are removed, their unique `AnyValue`
  interchange/`write_to` coverage folded into `test_values`.
- **`marrow.expr.values`: family-refined `execute` eliminates all consumption
  `rebind`s.** Each value family now refines `execute`'s return to its concrete
  array — `NumericValue` → `PrimitiveArray[Self.OutType]`, `StringValue` →
  `BinaryLikeArray[Self.OutType]`, `ListValue` → `ListLikeArray[Self.OutType]`
  (matching `BoolValue` → `BoolArray`). A child's `execute()` therefore yields a
  fully typed array at every call site, so `StringLength` / `StringUnary` /
  `StringPredicate` / `Counting` / `ListContains` pass operands straight to the
  typed kernels with no `rebind`, and the fused numeric `_fused` returns its
  `PrimitiveArray` directly. `StringPredicateKernel.apply` now takes two
  independent string type params so mixed string/large_string predicates need no
  pun. `BoolReduce` is parameterized by the reduce kernel type
  (`BoolReduceKernel` — `AnyKernel`/`AllKernel`) instead of a bool flag.
- **`AggKernel` gains a fully-typed `reduce[V]`** returning
  `PrimitiveScalar[Self.AccType[V]]` (no erased `AnyScalar`, no downcast) with a
  SIMD widened fast path for `sum`/`product`, same-type SIMD for `min`/`max`, and
  metadata `count`. The expr `Reduce` node folds through it, and `Count` reads
  `len - null_count` off the typed operand directly — both drop the erase →
  dispatch → downcast round-trip.

- **`marrow.expr.ibis` merged into `marrow.expr.values`**: the ibis-designed
  comptime expression system is now the canonical `values.mojo` (the old fused
  algebra is replaced). `AnyValue` and every node evaluate via `.execute()`
  (`to_array` removed everywhere — `dynamic.DynValue`, `execution.mojo`). The
  runtime `DynValue` is bridged into the new `AnyValue` via a dedicated
  constructor (it no longer implements the comptime `Value` trait). `BoolValue`
  execution is now wired: numeric comparisons fuse (`NumericCompare` — SIMD bool
  lane, bit-packed) and boolean logic materializes + combines masks (`BoolLogic`
  / `BoolNot`). Pruning is plumbed through `Value.prune` (conservative default;
  `DynValue` keeps the real min/max rule); the old per-node comptime pruning is
  parked as a commented reference. `test_ibis` + `test_ibis_exec` merged into
  `test_values`.

### Features

- **Cross-family casts in the expr system** (`marrow.expr.values`): `.cast(target)`
  is overloaded on every value family and dispatches by the target dtype's family,
  wiring the `marrow.kernels.cast` kernels into expressions — numeric↔string↔bool
  in every direction (`NumToBool`, `StringToBool`, `NumToString`, `BoolToString`,
  `StringToString` utf8↔large, `StringToNum`, `BoolToNum`), with numeric→numeric
  staying the existing fused `Cast`. String parse casts take a comptime `safe`
  (default null-on-failure). Each cast node conforms to its *target* family and
  materializes through the kernel.
- **Prepare-then-fuse: boundary nodes re-enter the numeric lane** (`marrow.expr.values`):
  numeric execution is two phases — a one-time `prepare(batch)` where *boundary*
  nodes with no SIMD lane (string/bool→numeric casts, string/list byte-length)
  materialize their column once into a per-node cache, then the usual fused `core`
  pass where those nodes read the cache per lane like a column. So the arithmetic
  *above* a boundary fuses into a single pass: `(a.length() + b.length()) + 1`
  prepares two length arrays, then one fused loop. `StringLength` / list `Counting`
  are promoted from `Value` to `NumericValue`, so `col.length()` composes with
  arithmetic / comparisons / `.cast()` / reductions. No fusability flag — every
  numeric node fuses; the family/value type carries the whole story. Reductions
  stay non-lane `Value` (length-1 can't fuse element-wise).

- **Expr floor division and element-wise min/max** (`marrow.expr.values`):
  wire the previously-unexposed `FloordivKernel` and the binary element-wise
  `arithmetic.MinKernel`/`MaxKernel` into the numeric lane — `a // b`
  (`__floordiv__`, integer floor keeping the wider operand dtype) and
  `a.min_element_wise(b)` / `a.max_element_wise(b)` (PyArrow naming; distinct
  from the whole-column `min()`/`max()` reductions). All three fuse as
  `NumericBinary` nodes (single vectorized pass).

- **List `contains()`** (`marrow.kernels.nested`, `marrow.expr.values`):
  `ArrayContainsKernel` implements element-wise membership `elem[i] ∈ list[i]` →
  `BoolArray` (each row scans its sublist; null list rows propagate to null;
  numeric element types). The expr `ListContains` node materializes both operands
  (a literal element broadcasts) and applies it, so `col("l", list_(t)).contains(x)`
  runs end-to-end. Removes the now-dead generic `BoolBinary`/`StringBinary` nodes.

- **List `length()`** (`marrow.kernels.nested`, `marrow.expr.values`):
  `ArrayLengthKernel` counts elements per list → `Int32Array` (offset subtraction,
  vectorized — the list analogue of `string.LengthKernel`). `ListColumn.execute`
  now resolves the list column from the batch (via a new `AnyArray.as_list_like`),
  and the `Counting` node folds `.length()` over it, so `col("l", list_(t)).length()`
  runs end-to-end.

- **Fused numeric `cast`** (`marrow.expr.values`): `NumericValue.cast(target)`
  adds a `Cast` node that reinterprets the operand's SIMD lane at the target
  dtype, so `col.cast(int64) + other` stays a single vectorized pass. Truncating
  (unchecked), matching the fused cast-kernel path.

- **`any`/`all` reductions and string `==`/`!=`** (`marrow.expr.values`,
  `marrow.kernels.string`): `BoolValue` gains `.any()`/`.all()` (new `BoolReduce`
  node folding a bool column to a length-1 result via the optimized
  `kernels.aggregate` bitmap reductions). String `==`/`!=` now execute through the
  existing `StringPredicate` node via two new `StringPredicateKernel`s
  (`StringEqKernel`, `StringNeKernel`), null-propagating and working for
  `string`/`large_string`. `NumericColumn.execute` returns the resolved column
  as-is so standalone column execution (reduction/predicate operands) preserves
  the validity bitmap.

- **Boolean predicate kernels + expr wiring** (`marrow.kernels.boolean`,
  `marrow.expr.values`): the marker structs `XorKernel`, `IsNullKernel`,
  `NotNullKernel`, `IsNanKernel`, `IsInfKernel` are now real kernels. `xor`
  becomes a `BoolBinaryKernel` (bit-packed word op) routed through `BoolLogic`.
  The four unary predicates implement a new `UnaryPredicateKernel` trait
  (`dispatch(AnyArray) -> AnyArray`): `is_null`/`not_null` are family-agnostic
  (read the validity bitmap via the byte-level `views.apply`), `is_nan`/`is_inf`
  scan floating values via `views.apply` (buffer→bitmap, CPU serial/parallel +
  GPU dispatch), propagating input nulls. The expr `BoolUnary` node now executes
  by materializing the operand and applying the kernel's `dispatch`; `.isnull()`,
  `.notnull()`, `.isnan()`, `.isinf()`, and `^` all run end-to-end.

- **String compute kernels** (`marrow.kernels.string`): real implementations
  replacing the name-only markers. `LengthKernel` (byte length →
  `Int32Array`, vectorized as `offsets[i+1]-offsets[i]`), unary string→string
  ops via the `StringMapKernel` trait (`UpperKernel`, `LowerKernel`,
  `ReverseKernel`, `StripKernel`, `LStripKernel`, `RStripKernel`,
  `CapitalizeKernel`), and binary predicates via the
  `StringPredicateKernel` trait (`StartsWithKernel`, `EndsWithKernel`,
  `ContainsKernel` → `BoolArray`). Each has a typed `apply` and a runtime
  `dispatch(AnyArray)`. The free-standing `string_lengths` function is removed
  (callers use `LengthKernel`).

- **`marrow.expr.values` reduction execution**: the aggregate boundary nodes now
  execute (previously `_not_wired` stubs). `sum()`/`product()` (widen to
  int64/float64), `mean()` (float64), and `min()`/`max()` (operand dtype
  preserved) share a single `Reduce[K, A]` node whose output dtype is the
  kernel's own `AccType[A.OutType]` — each aggregate is the single source of truth
  for its result type. A new family-agnostic `count()` (`Count` → int64, on the
  base `Value` trait) works on any input dtype. All fold the operand through the
  real `marrow.kernels.aggregate` kernels and return the scalar as a length-1
  result array. Reductions are a materialization boundary: the numeric lane fuses
  up to the operand, which is computed in full, then reduced. `AnyValue` erases
  and executes them like any other node.

- **`marrow.expr.ibis` string execution**: the `StringValue` family now
  executes by materializing — `StringColumn` resolves from the batch,
  `StringConst` broadcasts, `StringUnary` applies a `StringMapKernel`,
  `Counting` (`length`) applies `LengthKernel`, and the new `StringPredicate`
  node (`startswith`/`endswith`/`contains`) applies a
  `StringPredicateKernel` → `BoolValue`. Variable-width UTF-8 has no
  fixed SIMD lane, so string ops materialize rather than fuse; `length` is the
  one op that vectorizes internally (offset subtraction).

- **dtype → scalar/array associated types** (`DataType.ScalarType` /
  `DataType.ArrayType`): every Arrow type now names its companion typed scalar
  and array (the inverse of `Array.ScalarType`), provided at the family traits
  (`NumericType` → `PrimitiveScalar[Self]` / `PrimitiveArray[Self]`,
  `StringLikeType` → `StringScalar` / `BinaryLikeArray[Self]`, …) and on the
  standalone concrete types. This lets generic code map a dtype to its concrete
  companion — e.g. a leaf can hold `T.ScalarType` and construct it via a helper
  bound on the provider trait.

- **`marrow.expr.ibis` fused execution**: the numeric family now *executes*,
  hooked to the real `marrow.kernels` — `NumericValue` **is** the numeric lane
  (refines `OutType` to `NumericType`, carries a `core[W]` SIMD primitive, and its
  `execute` vectorizes `core` across the whole tree in a **single fused pass**).
  Arithmetic nodes are parameterized by the actual `AddKernel`/`DivKernel`/… (the
  kernel supplies compute; promotion stays in the node). Dedicated per-family
  leaves (`NumericColumn`/`StringColumn`/`ListColumn`, `NumericLiteral`/
  `StringLiteral`) with `col`/`lit` overloaded by dtype family; `execute` returns
  the dtype's companion `Self.OutType.ArrayType`. Bool/string/list are the type
  architecture (execution pending); cross-family numeric-producing boundaries
  (`length`, reductions) are non-lane nodes that materialize. More ops: fused
  transcendental math (`exp2`/`log2`/`log10`/`log1p`/`sin`/`cos`, `trunc` — all
  executing via the real kernels), numeric predicates (`isnan`/`isinf`/`notnull`),
  and string transforms (`strip`/`lstrip`/`rstrip`/`capitalize`). Every op node is
  parameterized by a `marrow.kernels` kernel (real where implemented, else a
  not-implemented marker in `kernels.string`/`kernels.nested`/…) — none defined in
  the expression layer.

- **`marrow.expr.ibis` typed expression architecture**: value families are
  traits (`NumericValue` / `BoolValue` / `StringValue`), operations are node
  structs, and kernels are pure name markers — promotion lives entirely in the
  value hierarchy (one node struct per `(family, output-dtype rule)`:
  `NumericBinary` widening, `FloatBinary`, `NumericUnary`/`FloatUnary`,
  `CountingUnary`, `BoolBinary`/`BoolUnary`, `StringUnary`). `Column` and
  `Literal` are unified leaves via conditional conformance; `Literal` holds the
  dtype's companion `T.ScalarType` and `lit` is an alias for it. Ops include
  arithmetic (`+ - * / % **`, `neg`/`abs`/`ceil`/`floor`/`round`/`sign`,
  `sqrt`/`exp`/`ln`), reductions (`sum` widening to 64-bit, `mean` → float64,
  `min`/`max` preserving), comparisons, logical (`& | ^ ~`), `isnull`, string
  `length`/`startswith`/`endswith`/`contains`/`upper`/`lower`/`reverse`/`==`/`!=`,
  and a nested **`ListValue`** family (`length` → numeric, `contains` → bool).

- **Columnar selection for all array types** (`filter` / `take` / `drop_null`):
  the selection kernels now support every array type — including nested
  `list` / `large_list` / `map` / `fixed_size_list` / `struct`, plus
  `dictionary`, `binary` / `large_string` / `fixed_size_binary` and `null` —
  fully **column-wise** with no row-encoding. Nested rows are gathered by their
  contiguous child spans; struct is filtered/taken per child; dictionary shares
  its values and only selects the codes. `rapidhash` likewise gained nested
  support (`list` / `large_list` / `map` / `fixed_size_list`), so `group_by` and
  joins work on nested key columns. Performance is best-in-class on the measured
  cases: dictionary `filter` (sequential code compaction, not a gather) and
  `list` / `fsl` `take` (raw-`Int32` child-index build + one dispatched child
  gather) both beat PyArrow and Polars at every size (e.g. `list take` at 1M is
  ~1.6× faster than PyArrow, ~7× faster than Polars; `dict filter` at 1M is ~11×
  faster than PyArrow, on par with Polars).

- **Distinct-count kernels** (`marrow.kernels.distinct`, `mk.count_distinct` /
  `mk.approx_count_distinct`): whole-array cardinality reductions returning an
  `int64` scalar, both excluding nulls (SQL `COUNT(DISTINCT x)` / PyArrow
  `only_valid`). `count_distinct` is exact — it dedups the per-row hashes through
  the same `SwissHashTable` the group-by uses, so it shares that 64-bit-hash
  basis. `approx_count_distinct` is a HyperLogLog estimate (2**14 registers,
  ~0.65% standard error, fixed 16 KiB regardless of cardinality) with linear
  counting in the small-cardinality regime, mirroring
  `pyarrow.compute.approx_count_distinct`.

- **Grouped distinct counts** (`GroupBy.count_distinct` /
  `GroupBy.approx_count_distinct`, and the `"count_distinct"` /
  `"approx_count_distinct"` functions in the Python
  `rb.group_by(keys).aggregate([...])` API): per-group `COUNT(DISTINCT v)`.
  Exact grouping dedups `(group_id, value)` pairs in a single `SwissHashTable`
  (the join's table) and bumps a per-group counter on each newly-seen pair — one
  pass, `O(distinct pairs)` memory, no per-group set. Approx keeps one
  HyperLogLog sketch per group (2**11 registers, 2 KiB/group). A distinct
  aggregate can share the single grouping pass with fold aggregates (e.g.
  `[("v","sum"),("v","count_distinct")]`). Distinct aggregates are
  **radix-parallel**: partitioning by key hash keeps every group inside one
  partition, so per-partition distinct counts are final and concatenate without a
  merge (the thread-local partial-merge path can't union sets, so any distinct
  set routes to radix when parallel). ~7x over serial at 1M rows / 50k groups,
  and faster than pyarrow's `count_distinct`.

- **Python group-by** (`marrow.RecordBatch.group_by`): grouped aggregation is
  now exposed to Python with a PyArrow-compatible API —
  `rb.group_by(keys).aggregate([("v", "sum"), ("v", "mean"), ...])` returns a
  `RecordBatch` of the unique key columns plus one `<value>_<func>` column per
  aggregate (`sum`/`mean`/`min`/`max`/`count`/`product`), grouped in a single
  pass over the keys. Backed by the `GroupBy` kernel and its serial/thread-local/
  radix strategy selection. A new `python/marrow/tests/bench_groupby.py`
  benchmarks it apples-to-apples against pyarrow, polars, and duckdb (all through
  their Python APIs) across row counts and cardinalities — run with
  `pixi run -e bench pytest python/marrow/tests/bench_groupby.py --benchmark
  --competition`.

- **Scalar `mean` reduction** (`marrow.kernels.aggregate.mean`, `mk.mean`,
  `marrow.compute.mean`): arithmetic mean of the valid elements as a float64
  scalar (nulls excluded from sum and divisor; null result for empty/all-null),
  matching `pyarrow.compute.mean`.
- **Grouped `min`/`max` preserve the input dtype** (PyArrow-correct): `min(int32)`
  now returns `int32` rather than widening to `int64`. `sum` still widens
  integers to `int64`; `count` is `int64`; `mean` is `float64`.

- **Cast kernels** (`marrow.kernels.cast`, `mk.cast`): monomorphized numeric,
  bool, and temporal casts behind a two-level dispatcher — a top-level `cast`
  routes on the type family, and each family struct (`NumericCast`, `BoolCast`,
  `TemporalCast`) does the within-family typed dispatch. Numeric casts build on
  `SIMD.cast` (one `pop.cast` per lane); `safe=False` truncates/wraps like
  `numpy.astype`, while `safe=True` (the default, matching PyArrow) raises on any
  lossy conversion. Bool casts use `x != 0` / `True→1`; temporal casts reinterpret
  to the underlying integer or scale by the unit ratio (e.g. `date32↔date64`,
  `timestamp[s]↔[ms]`). Fused cast expression nodes (`marrow.expr.values`) —
  numeric→numeric (`Cast`), numeric→bool (`NumToBoolValue`), and bool→numeric
  (`BoolToNumValue`), all reached through a single `.cast(dtype)` method on the
  `NumericValue`/`BoolValue` nodes — plus a `DynValue.cast(to)` runtime node let
  casts fuse into AOT-compiled expressions (`Cast(Add(a, b), int64)` collapses to
  a single vectorized pass, `(a < b).cast(int8)` bit-unpacks in place), and a
  PyArrow-style `marrow.compute.cast(arr, target_type, safe=…)` exposes it to
  Python. Also **string ↔ numeric/bool** (per-element `atol`/`atof` parse and
  format; `safe=True` raises on an unparseable value, `safe=False` nulls it) and
  **null → any** (all-null array of the target type).
- **More cast families** (`marrow.kernels.cast`): the cast router now also covers
  - **binary-like** — `utf8`/`large_utf8`/`binary`/`large_binary` ↔ each other
    (zero-copy relabel when the offset width matches, else an offset rebuild;
    bytes→utf8 validates UTF-8 under `safe`) and `fixed_size_binary` ↔ binary;
    `large_utf8` now parses/formats to numeric/bool like `utf8`.
  - **decimal** (`DecimalCast`, decimal32/64/128/256) — decimal ↔ decimal
    (rescale by `10^Δscale`, widening the backing integer as needed), decimal ↔
    integer, and decimal ↔ float.
  - **nested** — `list`/`large_list` → same-kind list (recursively casting the
    child values) and `struct` → `struct` (recursively casting each field).
  - **dictionary decode** — a dictionary source is gathered by index (`take`) and
    the decoded values cast to the target type.

  Remaining designed extension points: dictionary *encode*, `string ↔ temporal`,
  cross-kind list (`list ↔ large_list`/`fixed_size_list`), and `map`.

- **`distinct_count` statistic** (`marrow.parquet`): a dictionary-encoded column
  chunk now writes `Statistics.distinct_count` (its dictionary size = the number
  of distinct non-null values); PLAIN/DELTA chunks leave it absent. `distinct_count`
  is also read back into `ColumnMetaData`.

- **Bloom filters for temporal / decimal / fixed-size-binary** (`marrow.parquet`):
  `write_bloom_filter=True` now also builds filters for temporal (date/time/
  timestamp/duration — hashed over their INT32/INT64 little-endian bytes),
  decimal (`decimal32`/`decimal64` as INT32/INT64, `decimal128`/`decimal256`
  over their big-endian FIXED_LEN_BYTE_ARRAY bytes), and `fixed_size_binary`
  (raw bytes) columns — matching each type's physical value encoding. Previously
  only integer, floating-point, and byte-array columns were covered.

- **Page CRC-32 checksums** (`marrow.parquet`): `write_table(...,
  write_page_checksum=True)` (default False, like PyArrow) attaches a standard
  CRC-32 to every data/dictionary page header — over the compressed body for v1
  and the uncompressed levels + compressed values for v2, matching the spec —
  and the reader verifies it on read, raising on a mismatch. A new
  `marrow.utils.Crc32` (incremental, ISO-3309 / zlib polynomial) backs it.

- **Key/value metadata round-trip** (`marrow.parquet`): the file footer's
  `key_value_metadata` is now read and written. On read it populates
  `schema.metadata` (matching `pyarrow.read_table(...).schema.metadata`,
  including PyArrow's `ARROW:schema` blob); on write the schema's metadata is
  emitted, except `ARROW:schema` (which pins exact Arrow types — marrow writes
  and infers types from the Parquet schema, so re-emitting a foreign copy would
  make the file self-inconsistent).

- **float16 read + write** (`marrow.parquet`): the Arrow `float16` (half-float)
  type now round-trips. Parquet stores it as `FIXED_LEN_BYTE_ARRAY(2)` with the
  `FLOAT16` logical annotation, and the 2 bytes are exactly the little-endian
  half bit pattern, so it routes through the existing primitive path (PLAIN,
  dictionary, flat and nested) with IEEE-ordered, signed-zero-normalised
  min/max statistics.

- **FIXED_LEN_BYTE_ARRAY DELTA_BYTE_ARRAY / BYTE_STREAM_SPLIT read**
  (`marrow.parquet`): decimal and fixed-size-binary columns encoded with
  `DELTA_BYTE_ARRAY` or `BYTE_STREAM_SPLIT` (both emitted by PyArrow via
  `column_encoding`) now read. A single shared `Encoding.decode_flba` decodes the
  present values into a contiguous width-byte buffer, keeping the PLAIN path a
  zero-copy read.

- **Nullable-struct write** (`marrow.parquet`): a nullable Arrow struct is now
  emitted as an `OPTIONAL` group so struct-level nulls ride in the definition
  levels (previously structs were always `REQUIRED` and their null-ness was
  silently dropped on write). The struct's null bit is pushed into its children
  before shredding/encoding so the value count matches the levels. A struct whose
  subtree contains a repeated group (list/map) still stays `REQUIRED` for now.

- **INT96 timestamp read** (`marrow.parquet`): the reader decodes the deprecated
  12-byte INT96 physical type (nanoseconds-of-day + Julian day) into a
  nanosecond `timestamp`, so legacy Impala/Spark/Hive files read back — PLAIN and
  RLE_DICTIONARY, flat and nested.

- **RLE boolean read** (`marrow.parquet`): boolean values encoded as RLE — what
  arrow/PyArrow emit in DataPage v2 — now decode (a 4-byte length prefix then a
  width-1 RLE/bit-packed hybrid), alongside the existing PLAIN bit-packed path.

- **Data page splitting** (`marrow.parquet`): the writer no longer emits a single
  data page per column chunk. Each chunk is split into data pages of at most
  ~1 MiB of encoded values or 20 000 rows (matching arrow-cpp `data_pagesize` /
  `max_rows_per_page` and arrow-rs `DEFAULT_PAGE_SIZE` /
  `DEFAULT_DATA_PAGE_ROW_COUNT_LIMIT`), always breaking on a record boundary so a
  nested (list/map) row never straddles pages. A single dictionary page is shared
  by all data pages of a chunk. The writer now produces a real multi-entry
  OffsetIndex + ColumnIndex (per-page location, `first_row_index`, and
  min/max/null-count), so page-level predicate pushdown works on marrow-written
  files and large columns are no longer one giant page.

- **Bloom filters, read + write** (`marrow.parquet`): a new `bloom` module
  implements the XXH64 value hash and the split-block bloom filter (SBBF) per the
  Parquet spec / arrow-rs. `write_table(..., write_bloom_filter=True)` builds a
  filter for every integer, floating-point, and byte-array column (sized to its
  distinct-value count) and writes it with a `BloomFilterHeader`;
  `ColumnMetaData` now carries `bloom_filter_offset`/`length`.
  `ParquetFile.bloom_filter(row_group, column)` returns a `SplitBlockBloomFilter`
  whose `might_contain(bytes)` proves a value's absence with no false negatives.
  Validated both ways against the Apache `parquet-testing` reference file
  (`check("Hello")` / `check("Hello_Not_Exists")`) and by an independent
  reference reader over marrow's own output.

- **Page index write** (`marrow.parquet`): the writer now emits an `OffsetIndex`
  and (when the chunk carries bounds or is all-null) a `ColumnIndex` for every
  column chunk, written after the page data and pointed to by the footer's
  `ColumnChunk.{offset,column}_index_offset` — closing the read/write asymmetry
  where the reader consumed a page index that the writer never produced. Marrow
  writes a single data page per chunk, so each index has one entry covering all
  rows; PyArrow prunes with it (page-level predicate pushdown) and marrow reads
  it back via `read_page_index` / `read_page_bounds`.

- **Nested temporal / decimal / fixed-size-binary read** (`marrow.parquet`): a
  list or map element of a temporal type (`date32`/`time32`/`timestamp`/
  `time64`/`date64`/`duration`), `decimal128`/`decimal256`, or
  `fixed_size_binary` now decodes — previously only primitive/string/binary
  leaves worked under a repeated group and anything else raised `unsupported
  list element type`. The leveled drives grow a builder and retag the int32/
  int64 storage to the temporal Arrow type (decimals carry their precision/scale
  directly), so nested and flat paths now cover the same type set.

- **Full compression codec coverage** (`marrow.parquet`): the writer now emits
  `GZIP` (zlib deflate, windowBits 31) and `BROTLI` (via `libbrotlienc`), and
  the reader decodes `BROTLI` (via `libbrotlidec`) — closing the read/write
  asymmetry where GZIP could only be read. The deprecated `LZ4` (code 5) now
  round-trips in both directions: writers emit a plain LZ4 block (as modern
  PyArrow does) and the reader tolerates the legacy Hadoop 8-byte frame by
  stripping it when present. `brotli` is now a runtime dependency (opened via
  `dlopen`, like the other codecs).

- **Binary & large byte-array write** (`marrow.parquet`): the writer now emits
  `binary`, `large_binary`, and `large_string` columns (previously only the
  reader handled them). Parquet has a single `BYTE_ARRAY` physical type, so
  `large_*` columns are written as `BYTE_ARRAY` (large_string carrying the
  `UTF8`/`STRING` annotation) and read back as `binary`/`string` — matching
  arrow-rs / parquet-cpp. All value encodings (PLAIN, RLE_DICTIONARY,
  DELTA_BYTE_ARRAY, DELTA_LENGTH_BYTE_ARRAY) now work over any byte-array type.

- **Temporal, decimal & byte-array statistics** (`marrow.parquet`): the writer
  now emits `min`/`max` bounds for temporal (`date32`/`time32` as INT32,
  `timestamp`/`time64`/`date64`/`duration` as INT64, signed order), decimal
  (`decimal32`/`decimal64` as INT32/INT64, `decimal128`/`decimal256` as
  big-endian two's-complement `FIXED_LEN_BYTE_ARRAY` in signed numeric order),
  `binary`/`large_binary`/`large_string` (byte-wise lexicographic), and
  `fixed_size_binary`. Previously only numeric, bool, and string columns carried
  bounds; PyArrow now reads correct statistics for every written type.

- **Decimal & fixed-size-binary read** (`marrow.parquet`): the reader now decodes
  `FIXED_LEN_BYTE_ARRAY` columns — `decimal128`/`decimal256` (big-endian two's-
  complement, sign-extended from PyArrow's minimal per-precision byte width to
  the 16/32-byte int128/int256 storage) and `fixed_size_binary` (raw bytes) —
  across PLAIN and `RLE_DICTIONARY` pages. With the temporal read already in
  place, all temporal, decimal, and fixed-size-binary types now round-trip both
  directions with PyArrow.

- **Temporal, decimal & fixed-size-binary write** (`marrow.parquet`): the writer
  now emits `date32` (INT32/`DATE`), `time32`/`time64` (INT32/INT64 with
  `TIME_MILLIS`/`TIME_MICROS` and the nanosecond `TIME` `LogicalType`),
  `timestamp` (INT64/`TIMESTAMP` carrying the time unit and the
  `isAdjustedToUTC` flag derived from the Arrow timezone), `decimal128`/
  `decimal256` (big-endian two's-complement `FIXED_LEN_BYTE_ARRAY` of 16/32
  bytes with `precision`/`scale`), `decimal32`/`decimal64` (INT32/INT64), and
  `fixed_size_binary` (`FIXED_LEN_BYTE_ARRAY`). `SchemaElement` now serializes
  the full `LogicalType` Thrift union — including the nested `TimeUnit`
  (MILLIS/MICROS/NANOS) and `DECIMAL {scale, precision}` members — plus the
  `scale`/`precision` fields. PyArrow reads every type back with the correct
  annotation, including nanosecond resolution and timezones.

- **Write-side encodings** (`marrow.parquet`): the writer no longer emits only
  PLAIN. `write_table(..., use_dictionary=True)` (the default, like PyArrow)
  dictionary-encodes numeric and string columns — a PLAIN dictionary page of
  distinct values plus an `RLE_DICTIONARY` data page of bit-packed indices
  (`Rle.encode_bitpacked`), shrinking low-cardinality columns; set
  `use_dictionary=False` for PLAIN, and columns whose dictionary page would
  exceed 1 MB fall back to PLAIN automatically. The other encodings are
  selectable: `DELTA_BINARY_PACKED` (signed ints; block/miniblock zig-zag deltas
  via `DeltaBinaryPacked.encode`), `DELTA_BYTE_ARRAY` / `DELTA_LENGTH_BYTE_ARRAY`
  (strings), and `BYTE_STREAM_SPLIT` (floats). Encoding is chosen per leaf, most
  specific first — a `column_encodings` name→encoding map, then a single global
  `encoding`, then the dictionary/PLAIN default; an override that does not fit a
  column's type is ignored. Nulls are placed by the definition levels, so every
  encoding composes with the flat and nested (Dremel) write paths; the encoders
  mirror the reader's decoders and PyArrow reads every variant back.
  `ColumnMetaData` now advertises the real encodings and the dictionary page
  offset.

- **Map type** (core): a first-class Arrow `map<k, v>` — `MapType`/`map_()` in
  `dtypes`, `MapArray`/`MapArray.from_arrays(offsets, keys, items)` in `arrays`
  (physically a list of a non-nullable `entries` struct, so it reuses
  `ListLikeArray[MapType]`), `MapBuilder` (composed over an entries-struct
  `ListBuilder`, so maps flow through `concat`/`combine_chunks`), and the Arrow C
  Data Interface `+m` format in both directions (with the `keys_sorted` flag).
  Following arrow-rs (`DataType::Map(field, sorted)`) and Arrow C++
  (`MapType::value_field()`), `MapType` stores the entries struct as a single
  `Field`, preserving key/value field names and nullability. The retag between a
  list and a map lives in one place: `ListArray.to_map()` / `MapArray.to_list()`.

- **Parquet map columns, read + write** (`marrow.parquet`): a Parquet `MAP`
  (`optional group(MAP) { repeated group key_value { required key; value } }`,
  incl. the legacy `MAP_KEY_VALUE` annotation) now reads back as a `MapArray` and
  writes from one. A map reconstructs with the exact same Dremel machinery as
  `list<struct<key, value>>` — only the final array tag differs — so it composes
  to any depth (`map<k, list<v>>`, `list<map>`, …). PyArrow reads the maps marrow
  writes and vice versa.

- **Nested write (lists + maps)** (`marrow.parquet`): the writer gained a general
  Dremel *shredding* path — it emits `LIST`/`MAP` schema groups with correct
  repetition/definition levels, strips a nested column into per-leaf rep/def
  level streams (`SchemaNode.shred_levels`), and encodes multi-bit RLE levels
  (v1 and v2 data pages). Flat/struct columns keep their fast path untouched;
  columns containing a repeated group are shredded. This closes the previously
  open list-write gap. A single `ColumnWriter._emit_page` now serializes every
  data page, and the map's schema geometry lives in one shared `_map_node`
  builder used by both read and write.

- **Min/max statistics** (`marrow.parquet`): the writer now computes and emits
  per-column-chunk `min_value`/`max_value` bounds (with `is_min/max_value_exact`)
  alongside the existing `null_count`, and declares `column_orders`
  (`TypeDefinedOrder`) so readers trust the logical ordering. Bounds use the
  correct comparator per type — signed vs unsigned integers, IEEE floats
  (NaN-skipped, signed-zero normalised so the bound brackets ±0.0), and
  byte-wise string ordering — and are PLAIN-encoded (little-endian numerics; raw
  bytes for strings). PyArrow reads the bounds marrow writes, including the
  unsigned-int and string cases. On read, `read_metadata(path)` exposes the raw
  footer (row groups, offsets, codecs, `null_count`, and the min/max bytes)
  mirroring `pyarrow.parquet.read_metadata`, and `read_statistics(path)` returns
  decoded typed `min`/`max` scalars per (row group, leaf column) for the numeric,
  boolean, and string types (temporal/binary bounds are a follow-up). This is the
  foundation for row-group/page skipping (predicate pushdown).

- **Arbitrarily nested read support** (`marrow.parquet`): the reader now
  reconstructs any nesting of structs and lists — `list<struct>`, `list<list<…>>`
  to any depth, `struct<list>`, lists of nullable structs, and struct-level nulls
  at any position (top level, holding a list, or as a list element). A nullable
  struct reads back as null (not a struct of null fields), with field-null and
  struct-null distinguished. Each schema node carries its Dremel geometry
  (`NodeGeom`: `present_def`, `rep_level`, `element_floor`, `entry_floor`,
  `optional`), computed once during schema parsing, so every list's offset scan
  is self-contained and the assembler composes by recursion to any depth — no
  per-level special-casing. Flat leaves under a nullable struct keep their def
  levels (`LeafColumn.carry_def`); all others stay on the fast path.

- **DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY read support** (`marrow.parquet`):
  the reader now decodes the delta string/binary encodings (PyArrow
  `use_dictionary=False`) — DELTA_LENGTH_BYTE_ARRAY (delta-packed lengths then
  concatenated bytes) and DELTA_BYTE_ARRAY (incremental prefix + suffix
  reconstruction). Handles nulls; reads compressed and uncompressed. With
  DELTA_BINARY_PACKED and BYTE_STREAM_SPLIT this completes the common
  non-dictionary encodings.

- **BYTE_STREAM_SPLIT read support** (`marrow.parquet`): the reader now decodes
  the BYTE_STREAM_SPLIT encoding (float32/float64; PyArrow
  `use_byte_stream_split=True`), reassembling each value from its strided byte
  planes. Handles nulls; reads compressed and uncompressed.

- **DELTA_BINARY_PACKED read support** (`marrow.parquet`): the reader now decodes
  the DELTA_BINARY_PACKED integer encoding (block / miniblock zigzag deltas) that
  modern writers — PyArrow with `use_dictionary=False`, and v2 defaults — emit for
  int32/int64 columns. Handles negatives, nulls, and multi-block streams;
  previously it raised "unsupported data page encoding". Reads compressed and
  uncompressed.

- **DataPage V2 write support** (`marrow.parquet`): the writer can now emit v2
  data pages (`write_table(..., version=2)` in Mojo, `data_page_version="2.0"`
  in Python) in addition to the default v1 — the reader already read both. v2
  stores the definition levels uncompressed ahead of the compressed values
  (`DataPageHeaderV2` with `is_compressed`), matching arrow-rs's
  `WriterVersion` (PARQUET_1_0 default, PARQUET_2_0 opt-in). PyArrow reads
  marrow's v2 output and marrow round-trips it, compressed and uncompressed.

- **Column projection on read** (`marrow.parquet`): `read_table(path,
  columns=[...])` reads only the named top-level columns, in the given order —
  only those columns' chunks are decoded (the rest are never touched). Works for
  flat, struct (whole subtree), and list columns; raises on an unknown name.
  Implemented by selecting the assembly nodes and remapping their leaf indices
  onto a compact decoded grid, so the parallel decode skips unselected columns
  entirely. Exposed through the Python binding as `columns=` too.

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

- **Parallel Parquet reads** (`marrow.parquet`): `read_table` decodes every
  (row group, leaf column) pair concurrently across `num_physical_cores()`
  workers — each reads a disjoint byte range of the shared read-only mmap and
  writes its own result slot, and each owns a `Codecs` (the lazy `dlopen`
  handles are not shareable). Files below 4096 rows stay single-threaded to
  avoid dispatch overhead. ~4.9× faster on a 2M×8 multi-row-group snappy file
  (54 ms → 11 ms on 16 cores), bringing multi-column reads level with polars and
  PyArrow. ASAN-clean under the concurrent path.

- **LZ4 (LZ4_RAW) write support** (`marrow.parquet`): the writer can now emit
  `CODEC_LZ4_RAW` (via `LZ4_compress_default`), joining UNCOMPRESSED/SNAPPY/ZSTD;
  LZ4_RAW was already readable. PyArrow reads marrow's LZ4 output and vice versa
  (covered by the interop suite). LZ4 is the fastest real codec through the
  reader — a 2M×8 file reads in 6.9 ms vs 10.7 (snappy) / 15.3 (zstd) at
  essentially snappy's file size (58 vs 59 MB), so LZ4 gives near-uncompressed
  read speed with compression.

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

### Refactors

- **Drop delegating kernel convenience wrappers** (`arithmetic`, `boolean`,
  `compare`, `aggregate`, `cast`): the free-standing typed (`add[T]`, `ceil[T]`,
  `equal[T]`, `sum[T]`, `cast[From, To]`, …) and type-erased (`add`, `ceil`,
  `not_equal`, `sum`, …) functions that merely delegated to the kernel structs
  are removed. Callers now use the kernel structs directly — `AddKernel.apply[T]`
  / `EqKernel.apply[T]` / `SumKernel.apply[T]` / `NumericCast.apply[From, To,
  safe]` for the typed path, and `AddKernel.dispatch` / `NeKernel.dispatch` /
  `SumKernel.dispatch` / `MeanKernel.reduce` for the runtime-typed path. The
  expression layer is the primary Mojo-side API; the Python bindings keep thin
  PyArrow-compatible wrappers. Kernels that carry real logic beyond delegation
  are retained as free functions pending a struct-first port: `equal`'s
  string / struct / `AnyArray` dispatch, `is_null` / `select`, `any` / `all`,
  and the top-level `cast` dtype router.
