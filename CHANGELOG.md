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
- Decimal 32/64/128/256 and fixed-size binary, end to end.
- Memory mapping is an allocation kind, so the Parquet and IPC readers
  map a file instead of copying it in.
- `ListScalar` carries its own dtype, so `large_list`, `map` and
  `fixed_size_list` elements report their own type rather than a
  reconstructed one.
- Every unchecked view and buffer access is bounds-checked under
  `-D ASSERT=all` and compiles out in release.

### Compute kernels

- SIMD-vectorized, null-aware kernels whose names mirror
  `pyarrow.compute`, each written as a typed overload per array type
  with a type-erased layer on top.
- Arithmetic and comparison: `add`, `subtract`, `multiply`, `divide`,
  `floordiv`, `mod`, `neg`, `abs_`, and the six comparisons.
- Aggregates: `sum`, `product`, `min`, `max`, `mean`, `any`, `all`.
- Boolean logic with Kleene three-valued semantics, plus `is_null`,
  `is_nan`, `is_inf`.
- `cast` across every family -- numeric, boolean, temporal, decimal
  rescale, string parse and format, binary-like relabelling, nested list
  and struct, and dictionary decode -- in checked and unchecked modes,
  with `safe` a compile-time parameter.
- Hash group-by: `HashGrouper` resolves key rows to dense group ids, and
  the aggregate operators fold through a `Grouping` placement strategy.
  Radix-partitioned across threads, with per-thread tables.
- Multi-column sort: `sort` and `sort_indices`, LSD radix above 32,768
  elements and PDQsort below, parallel radix above 524,288,
  `nulls_first`/`nulls_last`, and stable ordering honoured.
- `count_distinct` (exact) and `approx_count_distinct` (HyperLogLog),
  whole-array and grouped, both radix-parallel.
- Columnar `filter`, `take` and `drop_null` for every array type,
  operating on mask and index views.
- `hash_join` -- inner, left, right, full, semi and anti -- over a
  `SwissHashTable` with a CSR index, partition-parallel.
- rapidhash over primitive, string, struct, list, large-list, map and
  fixed-size-list arrays.
- Strings: length, case conversion, strip, reverse, capitalize, concat,
  `starts_with`/`ends_with`/`contains`, the six ordering comparisons, and
  `LIKE`/`ILIKE` whose pattern compiles once per array.
- Temporal extraction -- year, month, day, quarter, day-of-year,
  day-of-week, hour, minute, second -- and `date_trunc` down to year.
- Conditional kernels: `case_when`, `coalesce`, `nullif`, `fill_null`;
  membership: `is_in`; nested: `array_length`, `array_contains`.
- Float unaries and binaries: `sqrt`, `exp`, `exp2`, `log`, `log2`,
  `log10`, `log1p`, `floor`, `ceil`, `trunc`, `round`, `sign`, `sin`,
  `cos`, `pow_`, and row-wise `minimum`/`maximum`.
- Every parallel loop runs through one striped driver on the execution
  context, which owns the thread count.

### Expressions and the query engine

- One relational plan IR: immutable `Relation` nodes -- `InMemoryTable`,
  `Filter`, `Project`, `Aggregate`, `Limit`, `Sort`, `Join` -- chained
  through `.filter()`, `.select()`, `.project()`, `.aggregate()`,
  `.sort_by()`, `.limit()`, `.join()`, and run by `.execute()`.
- Two expression lanes behind that one API. The **comptime lane** binds
  every operand on a family trait and resolves the output dtype at
  compile time, so a subtree fuses into a single SIMD loop with no
  dispatch. The **runtime lane** resolves operand dtypes at run time and
  interprets. `col("a", int64)` gives the first, `col("a")` the second.
- The numeric, boolean, string, temporal, conditional, cast and list
  families, each with a fluent operator and method API, reaching both
  lanes.
- Aggregates written on the expression they aggregate --
  `col("amount", int64).sum().alias("total")` -- with GROUP BY,
  HAVING, computed keys and computed inputs.
- Late-bound parameters in both lanes. `param("min-a", int64)` is a
  literal whose value arrives at execution time through `Bindings`,
  carried through the execution rather than substituted into a copy of
  the plan, so two executions of one plan cannot interfere -- and the
  fused inner loop is unchanged, so a parameter costs nothing per row.
- `with_columns`, `drop` and `rename` on the plan layer.

### Query optimizer

- Statistics-based pruning: predicates push into `ParquetScan` and skip
  row groups; the page index skips pages within a group.
- Projection pushdown into `ParquetScan`, so a scan decodes only the
  columns its parent reads.

### Parquet

- A from-scratch reader and writer with no Arrow C++ dependency:
  `read_table(path, columns=None)` and `write_table(table, path, ...)`.
- Every encoding: plain, RLE/dictionary, DELTA_BINARY_PACKED,
  DELTA_BYTE_ARRAY, DELTA_LENGTH_BYTE_ARRAY and BYTE_STREAM_SPLIT, on
  read and on write, with per-column selection and dictionary fallback.
- Arbitrarily nested reconstruction -- any depth of list and struct,
  struct-level nulls included -- and general nested writes.
- Codecs opened at run time by `dlopen`, so there is no link-time
  dependency: snappy, zstd, gzip, brotli, lz4 and lz4-raw.
- Column statistics (min/max/null/distinct), and the ColumnIndex and
  OffsetIndex page index, on read and write.
- Split-block bloom filters on read and write, including for temporal,
  decimal and fixed-size-binary columns.
- Data page v1 and v2, optional page CRC-32 checksums, file key/value
  metadata, and INT96 and float16 columns.
- Reads parallelize across row groups and columns; page bodies are
  zero-copy out of the mapped file.
- The scan streams in bounded row-group windows rather than materializing
  the file.

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
- Device codegen is opt-in behind `-D MARROW_GPU=true`, so a default
  build compiles out every device path.

### Python

- `import marrow as ma` -- a PyArrow-shaped API with `array()` type
  inference, full null handling, and nested structure support. Wrappers
  compose over the C extension rather than inheriting from it.
- `marrow.compute` exposes the kernel library: arithmetic, comparison,
  aggregates, `filter`/`take`/`drop_null`, `sort`/`sort_indices`, `cast`
  and the distinct counts.
- `marrow.parquet` mirrors `pyarrow.parquet`: `pq.read_table(...)` and
  `pq.write_table(...)`.
- A lazy query frontend: the plan layer is bound as `Plan`, the
  expression lane as `Expr`/`Agg`, and `LazyTable` builds and collects
  queries, with `num_threads` exposed on the collect path.

### Ahead-of-time compilation

- The comptime lane exists so a query can be compiled to a small,
  self-contained binary: the closed world is dead-code-eliminable, so a
  program links only the kernels its expressions name.
- `marrow compile` builds such a program and bundles the transitive
  dylib closure with a relocatable rpath, staging the dlopen'd codec
  libraries alongside it. Marrow's Mojo source ships in the wheel, so it
  works from an installed package.

### Tooling

- A pytest harness that compiles one driver per test selection, selects
  CPU/GPU and Mojo/Python suites, and splits the unit when the compiler
  crashes so the offending case reports it.
- A binary-size gate measuring the `__text` section across the operator
  families, run by `pixi run binary_size`.
- A golden query corpus: one file per case, run by both expression lanes
  and checked against a reference, with the unsupported surface recorded
  as skipped cases in `golden/COVERAGE.md`.
- A Quarto documentation site with guides, tutorials and examples.
