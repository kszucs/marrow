# Changelog

## [Unreleased] — 2026-03-18

### Features

- **Group-by kernel** (`marrow/kernels/groupby.mojo`): Fused `groupby(keys, values, aggregations)` that hashes, groups, and aggregates in a single pass — no intermediate index arrays. Supports `"sum"`, `"min"`, `"max"`, `"count"`, `"mean"` aggregations. Single-key (any primitive/string Array) and multi-key (StructArray) grouping. Returns `RecordBatch` with unique key columns + aggregated value columns.

- **Hashing kernel** (`marrow/kernels/hashing.mojo`): `hash_` computes per-element hashes for PrimitiveArray, StringArray, and StructArray (multi-key combining). `hash_identity` provides zero-overhead identity hash for bool/uint8/int8. Column-wise hashing follows DuckDB/DataFusion approach.

- **Expression execution system** (`marrow/expr/`): pull-based streaming query executor with a typed processor hierarchy. Value processors handle scalar expressions (`ColumnProcessor`, `LiteralProcessor`, `BinaryProcessor`, `UnaryProcessor`, `IsNullProcessor`, `IfElseProcessor`). Relation processors yield `RecordBatch` streams (`ScanProcessor`, `FilterProcessor`, `ProjectProcessor`, `ParquetScanProcessor`). High-level factory API: `col()`, `lit()`, `if_else()`, `in_memory_table()`, `parquet_scan()`. `Planner` builds processor trees from expression trees; `execute()` collects results.

- **Parquet I/O** (`marrow/parquet.mojo`): `read_table(path)` reads a Parquet file into a marrow `Table`; `write_table(table, path)` writes a marrow `Table` to Parquet. Both use the Arrow C Stream Interface for zero-copy transfer via PyArrow.

- **Comparison kernels** (`marrow/kernels/compare.mojo`): `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` for `PrimitiveArray[T]` and runtime-typed `Array`. Output is `PrimitiveArray[bool_]`. Null-propagating: if either input is null, the output position is null.

- **String kernels** (`marrow/kernels/string.mojo`): `string_lengths(StringArray) -> PrimitiveArray[uint32]` returns the byte length of each string element. Handles sliced arrays.

- **RecordBatch column operations** (`marrow/tabular.mojo`): `slice(offset)`, `slice(offset, length)` (zero-copy), `select(names)`, `rename_columns(names)`, `add_column(index, field, array)`, `append_column(field, array)`, `remove_column(index)`, `set_column(index, field, array)`, `to_struct_array()`.

- **Table enhancements** (`marrow/tabular.mojo`): `Table.from_batches(schema, batches)`, `Table.to_batches()`, `Table.combine_chunks() -> RecordBatch`.

- **Schema enhancements** (`marrow/schema.mojo`): `get_field_index(name)` (returns -1 if not found), `field(name)` lookup by name, `names()`, equality operators `==` / `!=`, Python interop via Arrow C Data Interface.

- **Python bindings**: expanded `python/tabular.mojo` with full `RecordBatch` and `Table` Python API; new `python/helpers.mojo` with shared conversion utilities; expanded PyArrow interop test coverage in `python/tests/test_pyarrow_interop.py`.

### Refactors

- **Scalar types hold native values** (`marrow/scalars.mojo`): `PrimitiveScalar[T]` now holds `SIMD[T.native, 1]` + `Bool` validity directly instead of a length-1 `PrimitiveArray`. `StringScalar` holds `String` + `Bool`. `ListScalar` holds `AnyArray` (child elements) + `Bool`. `StructScalar` holds `List[AnyArray]` (one per field) + `DataType` + `Bool`. `AnyScalar` remains a type-erased container backed by a length-1 `AnyArray` for uniform storage. Added a `Scalar` trait mirroring the `Array` trait.

- **Array trait + AnyArray rename**: Introduced `Array` trait (`type()`, `null_count()`, `is_valid()`, `as_any()`) implemented by all typed arrays. Renamed the type-erased `Array` struct to `AnyArray`, aligning with the existing `Builder`/`AnyBuilder` and `Value`/`AnyValue` naming convention.
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
