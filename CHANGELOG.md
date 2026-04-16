# Changelog

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
