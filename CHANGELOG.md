# Changelog

## [Unreleased]

### Arrays, scalars and builders

- Arrays, builders and scalars for the null, boolean, numeric, string,
  binary, list, fixed-size-list, struct, dictionary and temporal layouts,
  with typed aliases throughout (`Int32Array`, `StringBuilder`,
  `TimestampArray`, `Decimal128Scalar`, ...), plus `ChunkedArray`.
- `RecordBatch` and `Table` -- schema plus column arrays or chunked
  columns, with slice, select, rename and add/remove/set column.
- Type erasure without dynamic dispatch: erased containers back an inline
  `Variant` selected with `isa[T]()`, and convert implicitly to and from
  their typed forms. Every value holds its data behind refcounted
  `Buffer`/`Bitmap` handles, so erasing or unerasing is O(1).
- `Buffer` and `Bitmap` in mutable and immutable forms over four
  allocation kinds -- owned CPU, foreign (with a release callback),
  pinned host and device -- with borrowed, offset-applied `BufferView`
  and `BitmapView` spans for kernels to compute over.
- `map` is a first-class Arrow type: arrays, builders, scalars, casts,
  hashing, selection, Parquet and IPC all carry it.

### Compute kernels

- SIMD-vectorized, null-aware kernels whose names mirror
  `pyarrow.compute`, each written as a typed overload per array type
  with a type-erased layer on top.
- Arithmetic and comparison: `add`, `subtract`, `multiply`, `divide`,
  `floordiv`, `mod`, `neg`, `abs_`, and the six comparisons.
- Aggregates: `sum`, `product`, `min`, `max`, `mean`, `any`, `all`.
- Boolean logic with Kleene three-valued semantics, plus `is_null`,
  `is_nan`, `is_inf`.
- `hash_join` -- inner, left, right, full, semi and anti -- over a
  `SwissHashTable` with a CSR index, partition-parallel.
- rapidhash over primitive, string, struct, list, large-list, map and
  fixed-size-list arrays.

### Expressions and the query engine

- One relational plan IR: immutable `Relation` nodes -- `InMemoryTable`,
  `Filter`, `Project`, `Aggregate`, `Limit`, `Sort`, `Join` -- chained
  through `.filter()`, `.select()`, `.project()`, `.aggregate()`,
  `.sort_by()`, `.limit()`, `.join()`, and run by `.execute()`.

### Query optimizer

- Statistics-based pruning: predicates push into `ParquetScan` and skip
  row groups; the page index skips pages within a group.

### Parquet

- A from-scratch reader and writer with no Arrow C++ dependency:
  `read_table(path, columns=None)` and `write_table(table, path, ...)`.
- Every encoding: plain, RLE/dictionary, DELTA_BINARY_PACKED,
  DELTA_BYTE_ARRAY, DELTA_LENGTH_BYTE_ARRAY and BYTE_STREAM_SPLIT, on
  read and on write, with per-column selection and dictionary fallback.
- Arbitrarily nested reconstruction -- any depth of list and struct,
  struct-level nulls included -- and general nested writes.
- Column statistics (min/max/null/distinct), and the ColumnIndex and
  OffsetIndex page index, on read and write.
- Reads parallelize across row groups and columns; page bodies are
  zero-copy out of the mapped file.

### Arrow IPC

- `read_ipc_file`/`write_ipc_file` and `read_ipc_stream`/
  `write_ipc_stream`, plus streaming `RecordBatchFileReader`,
  `RecordBatchStreamReader` and the matching writers.
- Round-trips every implemented type, nested, dictionary, temporal and
  null columns included, with delta dictionaries.

### Interoperability and GPU

- Arrow C Data Interface: `CArrowSchema` and `CArrowArray` for zero-copy
  exchange with PyArrow in both directions, release callbacks included.
- GPU execution through Mojo's `DeviceContext` -- Metal on Apple Silicon,
  CUDA on NVIDIA. Data movement is explicit: `to_device(ctx)` and
  `to_cpu(ctx)` on buffers, bitmaps and arrays, with kernel results
  staying device-resident.

### Python

- `import marrow as ma` -- a PyArrow-shaped API with `array()` type
  inference, full null handling, and nested structure support. Wrappers
  compose over the C extension rather than inheriting from it.
- `marrow.compute` exposes the kernel library: arithmetic, comparison,
  aggregates, `filter`/`take`/`drop_null`, `sort`/`sort_indices`, `cast`
  and the distinct counts.
- `marrow.parquet` mirrors `pyarrow.parquet`: `pq.read_table(...)` and
  `pq.write_table(...)`.

### Ahead-of-time compilation

- The comptime lane exists so a query can be compiled to a small,
  self-contained binary: the closed world is dead-code-eliminable, so a
  program links only the kernels its expressions name.
