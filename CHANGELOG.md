# Changelog

## [Unreleased] — 2026-04-19

### Features

- **Parallel per-column `take()`** (`marrow/kernels/filter.mojo`):
  `take[T](PrimitiveArray, indices, ctx)` and the `AnyArray` dispatcher
  now accept an `ExecutionContext`. The no-null fast path stripes its
  SIMD-gather loop across workers via `sync_parallelize`. `HashJoin`'s
  `_assemble()` uses this to fan the per-output-column gathers across
  threads — the final materialization step was the last serial piece
  of the parallel join. End-to-end 10M inner join: **143 ms → 67 ms**.
- **`ExecutionContext`** (`marrow/kernels/execution.mojo`): New struct
  bundling the two axes of kernel dispatch — `num_threads` for CPU
  stripe parallelism and `device: Optional[DeviceContext]` for GPU.
  Implicit conversion from `Optional[DeviceContext]` keeps existing
  callers working. Factories: `.serial()`, `.parallel(num_threads=0)`
  (0 = auto via `num_physical_cores()`), `.gpu(device)`. Kernels accept
  an `ExecutionContext` parameter and forward it through the call
  graph; `apply()` in `marrow/views.mojo` picks up CPU stripe-
  parallelism uniformly rather than each kernel reimplementing
  `sync_parallelize`. `rapidhash`, `SwissHashTable.build/probe/insert`,
  `HashJoin`, and `hash_join()` all take `ExecutionContext` in place of
  the old `Optional[DeviceContext]` + `num_threads` pair.
- **Partition-parallel hash join** (`marrow/kernels/join.mojo`,
  `marrow/kernels/hashtable.mojo`): `HashJoin` and top-level `hash_join()`
  gain a `num_threads` argument (`0` = auto via `num_physical_cores()`,
  `1` = serial, `>1` = partition-parallel). The parallel path radix-
  partitions both sides by the top bits of their hash into
  `2^radix_bits` independent `SwissHashTable` instances, builds and probes
  them concurrently via `sync_parallelize`, and concatenates per-partition
  index pairs into the final result. No atomics on the hot path: each
  partition is fully independent.  Serial path is unchanged —
  `build_serial` / `probe_serial` methods preserve the pre-parallel
  implementation and are used when `num_threads == 1` or the build side
  is below `_PARALLEL_THRESHOLD` (100k rows).
- **`RadixPartitioner`** (`marrow/kernels/hashtable.mojo`): Implements the
  previously-stubbed `Partitioner` trait. Partitions hashes + row indices
  by the top `num_bits` (default 6 → 64 partitions). Per-thread histogram
  → partition-major prefix sum → parallel scatter into two shared flat
  buffers, then per-partition zero-copy slice via `ArcPointer`-shared
  immutable buffers. Single allocation per output (N Int32 + N UInt64).
- **`num_threads` on `rapidhash`** (`marrow/kernels/hashing.mojo`): Bool
  and primitive overloads stripe the row range across workers via
  `sync_parallelize` when `num_threads > 1` and rows ≥ 32768. Struct
  overload forwards to per-field calls.  Struct array / `AnyArray`
  dispatch threads the parameter through.
- **Public `SwissHashTable.insert_hashes` / `build_hashes` /
  `probe_hashes`**: The previously-private `_insert_hashes` /
  `_build_hashes` / `_probe_hashes` primitives are now public. Enables
  callers that have already computed hashes (e.g. the partition-parallel
  HashJoin, which hashes once up-front and then builds per-partition
  tables against the pre-computed hashes) to skip the hasher entirely —
  no need for separate `build_with_hashes` / `probe_with_hashes`
  variants. `probe_hashes` returns raw candidate `(build_row, probe_row)`
  pairs; callers that need the equality-verified result use the
  convenience `probe()` wrapper which adds the hash + equality filter
  step.

### Benchmarks

- **Parallel join competition** (`python/tests/bench_join_parallel.py`,
  `marrow/kernels/tests/bench_join.mojo`): Multi-threaded competition
  benchmarks (marrow vs DuckDB vs Polars vs PyArrow, no forced thread=1)
  at 1M / 10M (100M gated behind `MARROW_BENCH_LARGE=1`). Mojo-side bench
  refactored around shared helpers with a 10M tier plus a build×probe
  shape matrix (100k×10M, 10M×100k, 1M×10M, 10M×1M). At 10M×10M INNER
  join: Marrow 330 ms (serial, pre-parallel) → 67 ms (parallel, **4.9×
  speedup**), now the fastest among all measured libraries — Polars
  97 ms (Marrow 1.4× faster), PyArrow 111 ms (Marrow 1.7× faster),
  DuckDB 122 ms (Marrow 1.8× faster). At 1M INNER: Marrow 7.1 ms beats
  PyArrow (7.7 ms) and DuckDB (17.2 ms) outright, within 1.2× of Polars.

## [Unreleased] — 2026-04-09

### Features

- **Variant-based dispatch for `DataType`, `AnyArray`, and `Builder`** (`marrow/dtypes.mojo`, `marrow/arrays.mojo`, `marrow/builders.mojo`): Replaced integer-code dispatch with `Variant`-backed types using `comptime for` loops throughout. Arrays and builders use variant dispatch for safer downcasting and better branch prediction. Eliminates runtime `if`/`elif` chains across kernels, Python bindings, and the expression system.

- **`BoolArray` dedicated type** (`marrow/arrays.mojo`): Bit-packed boolean arrays are now handled by a dedicated `BoolArray` backed by a `Bitmap`, with `.values() -> BitmapView`, GPU transfer (`.to_device()` / `.to_host()`), and a matching `BoolBuilder`. Removes the incorrect `PrimitiveArray[bool_]` usage throughout.

- **`BufferView` / `BitmapView` abstractions** (`marrow/views.mojo`): Type-safe, non-owning views over `Buffer` and `Bitmap` with `apply` dispatch, `compressed_store`, `pext`, and GPU-aware access. All kernel and array code now operates through views instead of raw pointers.

- **`SwissHashTable`** (`marrow/kernels/`): Open-addressing hash table with 7-bit control stamps, CSR chain storage, vectorized SIMD group matching, and a batch-build API. Supports generic hash functions and string equality.

- **Hash join** (`marrow/kernels/`): `hash_join` kernel using `SwissHashTable` with join relations and executor integration. Build and probe phases are separate for reuse across multi-join plans.

- **`TestSuite` and `BenchSuite` framework** (`marrow/testing`): Auto-discovery of `test_*` / `bench_*` functions via `__functions_in_module()`. `BenchSuite` integrates with the pytest harness for CI benchmark capture, competition tables, and per-element throughput metrics.

- **AddressSanitizer support**: `pixi run pytest --asan` compiles test runners with ASAN instrumentation via `libcompiler-rt`. Catches buffer overflows and use-after-free in Mojo kernel code.

- **GPU `BitmapView` and GPU rapidhash** (`marrow/kernels/`): `BitmapView` now supports device-resident bitmaps. `rapidhash` ported to Metal/CUDA with a 128-bit multiply emulation for compatibility with Metal's lack of 128-bit integer support.

- **Bounds checking** (`marrow/buffers.mojo`): `Buffer`, `Bitmap`, and `BufferView` accessors now assert bounds in debug builds.

- **Implicit builder conversions** (`marrow/builders.mojo`): Typed builders (`PrimitiveBuilder[T]`, `StringBuilder`, etc.) convert implicitly to `Builder` via `ArcPointer` clone, so the original typed builder remains usable after passing to a composite builder.

### Fixes

- **`PyUnicode_AsUTF8AndSize` return type** (`python/arrays.mojo`): `PyUnicode_AsUTF8AndSize` now returns `StringSlice[ImmutAnyOrigin]` directly; removed the stale `.value()` unwrap that caused a compile error against newer Mojo stdlib.

- **ASAP destruction UAF in `bench_groupby`** (`marrow/kernels/tests/bench_groupby.mojo`): Added `keep(keys)` / `keep(vals)` after `b.iter[call]()` so the `@parameter` closure's captured arrays stay live for the duration of the benchmark loop. Without them, Mojo's ASAP destruction freed `keys`/`vals` before the iteration completed, corrupting the heap and crashing the subsequent `SwissHashTable` allocation. Re-enables the `bench-mojo` CI job.

## [Unreleased] — 2026-03-18

### Features

- **Unary math kernels** (`marrow/kernels/arithmetic.mojo`): `sign`, `sqrt`, `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `trunc`, `round`, `sin`, `cos` (floating-point only where applicable), plus binary `pow_`, `floordiv`, `mod`. All available as typed `PrimitiveArray[T]` overloads and runtime-typed `AnyArray` overloads.

- **Scalar types** (`marrow/scalars.mojo`): `PrimitiveScalar[T]`, `StringScalar`, `ListScalar`, `StructScalar`, `AnyScalar` — typed and type-erased scalar values mirroring the array hierarchy. `AnyScalar` implements `ConvertibleToPython` for zero-copy Python interop.

- **Python Scalar bindings** (`python/scalars.mojo`): `ma.scalar(value)` constructs typed scalars from Python objects; scalar arithmetic and comparison operators; `__bool__`, `__repr__`, `__str__` support; round-trip with PyArrow scalars via the C Data Interface.

- **Group-by kernel** (`marrow/kernels/groupby.mojo`): Fused `groupby(keys, values, aggregations)` that hashes, groups, and aggregates in a single pass — no intermediate index arrays. Supports `"sum"`, `"min"`, `"max"`, `"count"`, `"mean"` aggregations. Single-key (any primitive/string `AnyArray`) and multi-key (`StructArray`) grouping. Returns `RecordBatch` with unique key columns + aggregated value columns.

- **Hashing kernel** (`marrow/kernels/hashing.mojo`): `hash_` computes per-element hashes for `PrimitiveArray`, `StringArray`, and `StructArray` (multi-key combining). `hash_identity` provides zero-overhead identity hash for bool/uint8/int8. Column-wise hashing follows DuckDB/DataFusion approach.

- **Expression execution system** (`marrow/expr/`): pull-based streaming query executor with a typed processor hierarchy. Value processors handle scalar expressions (`ColumnProcessor`, `LiteralProcessor`, `BinaryProcessor`, `UnaryProcessor`, `IsNullProcessor`, `IfElseProcessor`). Relation processors yield `RecordBatch` streams (`ScanProcessor`, `FilterProcessor`, `ProjectProcessor`, `ParquetScanProcessor`, `AggregateProcessor`). High-level factory API: `col()`, `lit()`, `if_else()`, `in_memory_table()`, `parquet_scan()`. `Planner` builds processor trees from expression trees; `execute()` collects results.

- **Parquet I/O** (`marrow/parquet.mojo`): `read_table(path)` reads a Parquet file into a marrow `Table`; `write_table(table, path)` writes a marrow `Table` to Parquet. Both use the Arrow C Stream Interface for zero-copy transfer via PyArrow.

- **Comparison kernels** (`marrow/kernels/compare.mojo`): `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` for `PrimitiveArray[T]` and runtime-typed `AnyArray`. Output is `PrimitiveArray[bool_]`. Null-propagating: if either input is null, the output position is null. GPU variants available when a `DeviceContext` is passed.

- **String kernels** (`marrow/kernels/string.mojo`): `string_lengths(StringArray) -> PrimitiveArray[uint32]` returns the byte length of each string element. Handles sliced arrays.

- **RecordBatch column operations** (`marrow/tabular.mojo`): `slice(offset)`, `slice(offset, length)` (zero-copy), `select(names)`, `rename_columns(names)`, `add_column(index, field, array)`, `append_column(field, array)`, `remove_column(index)`, `set_column(index, field, array)`, `to_struct_array()`.

- **Table enhancements** (`marrow/tabular.mojo`): `Table.from_batches(schema, batches)`, `Table.to_batches()`, `Table.combine_chunks() -> RecordBatch`.

- **Schema enhancements** (`marrow/schema.mojo`): `get_field_index(name)` (returns -1 if not found), `field(name)` lookup by name, `names()`, equality operators `==` / `!=`, Python interop via Arrow C Data Interface.

- **Python bindings**: expanded `python/tabular.mojo` with full `RecordBatch` and `Table` Python API; new `python/helpers.mojo` with shared conversion utilities; expanded PyArrow interop test coverage in `python/tests/test_pyarrow_interop.py`.

### Refactors

- **Unified `Buffer[mut: Bool]`** (`marrow/buffers.mojo`): Merged `Buffer` and `BufferBuilder` into a single `Buffer[mut: Bool = False]` type with parametric mutability, following the same pattern as `BufferView[mut]` and `BitmapView[mut]`. `Buffer[mut=True]` is the mutable builder; `Buffer[mut=False]` is the immutable shared-ownership view. `finish()` is a zero-cost O(1) type transfer. All call sites updated across `bitmap.mojo`, `views.mojo`, `arrays.mojo`, `builders.mojo`, `c_data.mojo`, and all kernel files.

- **Unified `Bitmap[mut: Bool]`** (`marrow/bitmap.mojo`): Merged `Bitmap` and `BitmapBuilder` into a single `Bitmap[mut: Bool = False]` type with parametric mutability, following the same pattern as `Buffer[mut]`. `Bitmap[mut=True]` is the mutable builder (use `Bitmap.alloc_zeroed(n)`); `Bitmap[mut=False]` is the immutable ref-counted view. `finish(length)` freezes to `Bitmap[]`. All call sites updated across `arrays.mojo`, `builders.mojo`, `c_data.mojo`, and all kernel/test files.

- **Array trait + AnyArray rename** (`marrow/arrays.mojo`): Introduced `Array` trait (`type()`, `null_count()`, `is_valid()`, `as_any()`) implemented by all typed arrays. Renamed the type-erased `Array` struct to `AnyArray`, aligning with the existing `Builder`/`AnyArray` and `Value`/`AnyValue` naming convention. All kernel signatures updated accordingly.

- **Scalar types hold native values** (`marrow/scalars.mojo`): `PrimitiveScalar[T]` now holds `SIMD[T.native, 1]` + `Bool` validity directly instead of a length-1 `PrimitiveArray`. `StringScalar` holds `String` + `Bool`. `ListScalar` holds `AnyArray` (child elements) + `Bool`. `StructScalar` holds `List[AnyArray]` (one per field) + `DataType` + `Bool`. `AnyScalar` remains a type-erased container backed by a length-1 `AnyArray` for uniform storage. Added a `Scalar` trait mirroring the `Array` trait.

- **Filter kernel** (`marrow/kernels/filter.mojo`): Rewrote the inner loop using run-length encoding for high-selectivity cases (>80% pass rate: bulk `memcopy`) and a bit-scan iterator for low-selectivity cases. Validity bitmap writes use `Bitmap.copy_from` for bulk transfers instead of per-element `set_bit` calls. Added profiling script and benchmark harness.

- **Arithmetic kernel** (`marrow/kernels/arithmetic.mojo`): Extracted from `kernels/__init__.mojo` into a dedicated module. GPU elementwise ops (`_add_gpu`, `_sub_gpu`, `_mul_gpu`, `_div_gpu`) added alongside CPU SIMD paths.

- **Aggregate kernel** (`marrow/kernels/aggregate.mojo`): Removed `sum.mojo`; `sum_` and all aggregates now live in `aggregate.mojo`. Restructured to use `_elementwise_unary`/`_elementwise_binary` helpers consistent with the arithmetic kernel.

- Moved the expression system from `marrow/kernels/` into the dedicated `marrow/expr/` module.
- Renamed `plan.mojo` → `relations.mojo` and `expr.mojo` → `values.mojo` for clarity.
- Redesigned executor with a pull-based streaming model and strict typed processor hierarchy.
- Refactored C Data Interface (`marrow/c_data.mojo`) for cleaner schema and array import/export paths.
- `binary_simd` kernel helper now supports independent input and output types, enabling `bool_` output for comparison kernels via a `comptime if` branch.

### Tests

- Modernised the full Mojo test suite: prefer `arr[i]` over `arr.unsafe_get(i)`, `arr.as_primitive[T]()` over `PrimitiveArray[T](data=arr)`, and structural equality `assert_true(a == b)` over element-by-element loops.
- Added testing guidelines to `CLAUDE.md`.

### Fixes

- Minor executor dispatch fixes for type handling in binary and unary expression nodes.
- Python binding test cleanup and interop edge-case fixes.
- Code formatting.
