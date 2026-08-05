![marrow](logo.png)

# marrow

An implementation of [Apache Arrow](https://arrow.apache.org) in [Mojo](https://www.modular.com/mojo). The initial motivation was to learn Mojo while doing something useful, and since I've been involved in Apache Arrow for a while it seemed a natural fit. The project has grown beyond a prototype: it now has a full Python binding layer, SIMD compute kernels, GPU acceleration, and benchmarks showing it outperforms PyArrow on array construction for common numeric and string workloads.

### What is Arrow?

Apache Arrow is a cross-language development platform for in-memory data. It specifies a standardized, language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations on modern hardware like CPUs and GPUs.

### What is Mojo?

[Mojo](https://www.modular.com/mojo) is a new programming language built on MLIR that combines Python expressiveness with the performance of systems programming languages.

### Why Arrow in Mojo?

Arrow should be a first-class citizen in Mojo's ecosystem. This implementation provides zero-copy interoperability with PyArrow via the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html), and serves as a foundation for high-performance data processing in Mojo.

## Features

**Array types**
- `PrimitiveArray[T]` — numeric and boolean arrays with type aliases: `BoolArray`, `Int8Array` … `Int64Array`, `UInt8Array` … `UInt64Array`, `Float32Array`, `Float64Array`
- `StringArray` — UTF-8 variable-length strings
- `ListArray` — variable-length nested arrays
- `FixedSizeListArray` — fixed-size nested arrays (embedding vectors, coordinates)
- `StructArray` — named-field structs
- `ChunkedArray` — array split across multiple chunks
- `DictionaryArray` — dictionary-encoded arrays (integer indices + arbitrary values array)
- `NullArray` — all-null arrays
- `FixedSizeBinaryArray` — fixed-width opaque byte blobs
- `LargeBinaryArray`, `LargeStringArray`, `LargeListArray` — 64-bit offset variants
- Temporal arrays: `Date32Array`, `Date64Array`, `Time32Array`, `Time64Array`, `TimestampArray`, `DurationArray`
- `AnyArray` — type-erased immutable array container (O(1) copy via `ArcPointer`)
- `RecordBatch` — schema + column arrays, with slice, select, rename, add/remove/set column operations
- `Table` — schema + chunked columns; `from_batches()`, `to_batches()`, `combine_chunks()`

**Scalar types**
- `PrimitiveScalar[T]`, `StringScalar`, `ListScalar`, `StructScalar` — typed scalars holding native values
- `AnyScalar` — type-erased scalar backed by a length-1 `AnyArray`

**Builders** — incrementally build immutable arrays
- `PrimitiveBuilder[T]`, `StringBuilder`, `ListBuilder`, `FixedSizeListBuilder`, `StructBuilder`
- `AnyBuilder` — type-erased builder using function-pointer vtable dispatch (O(1) copy via `ArcPointer`)

**Compute kernels** (SIMD-vectorized, null-aware; names mirror `pyarrow.compute`)
- Arithmetic: `add`, `subtract`, `multiply`, `divide`, `floordiv`, `mod`, `neg`, `abs_`, `min_element_wise`, `max_element_wise`
- Math (unary): `sign`, `sqrt`, `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `trunc`, `round`, `sin`, `cos`
- Math (binary): `pow_`
- Comparisons: `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` → `BoolArray` (CPU + GPU)
- Aggregates: `sum`, `product`, `min`, `max`, `mean`, `any`, `all` (null-skipping)
- Distinct counts: `count_distinct` (exact), `approx_count_distinct` (HyperLogLog); whole-array and grouped
- Cast: `cast` — numeric/bool/temporal/decimal families, string↔numeric parse/format, dictionary decode; safe (checked) and unsafe modes
- Group-by: `GroupBy(keys).sum(values)` / `.min` / `.max` / typed `.aggregate[K]`; radix-partition-parallel, returns `RecordBatch`
- Hashing: `hash_` for primitive, string, and struct arrays
- Selection: `filter`, `drop_null`, `take`
- Sort: `sort_indices` (returns index array), `sort`; LSD radix for N ≥ 32 768, PDQsort below; parallel radix for N ≥ 524 288; `nulls_first`/`nulls_last`; multi-column `SortIndices.multi(StructArray, key_indices, ascending)`
- Join: `hash_join` — inner, left, right, full, semi, anti; partition-parallel
- Strings: length, upper/lower, strip/lstrip/rstrip, reverse, capitalize, concat, starts_with/ends_with/contains, the six comparisons, and `LIKE`/`ILIKE` with a pre-compiled pattern
- Temporal: year, month, day, quarter, day_of_year, day_of_week, hour, minute, second, `date_trunc`
- Conditional: `case_when`, `coalesce`, `nullif`, `fill_null`; membership: `is_in`; nested: `array_length`, `array_contains`

**Expression execution** (`marrow.expr`) — two lanes behind one relational API
- **Runtime lane** — the `DynValue` node (`marrow/expr/dynamic.mojo`). Its operand *dtypes* are resolved at run time, but its operation is not: each node carries a pointer to its evaluator, so a binary links exactly the kernels its expressions mention. Build expression trees with `col()`, `lit()`, `if_else()` and operator overloads (`+`, `-`, `*`, `/`, `>`, `<`, `==`, `&`, `|`, …).
- **AOT lane** — the comptime-typed algebra (`marrow/expr/values.mojo`), fully monomorphized: every operand is bound on a family trait, the output dtype is a comptime type, and a subtree fuses into one SIMD loop with no dispatch.
- **One plan IR over both.** `BoxedValue` (`marrow/expr/relations.mojo`) is the single box both lanes erase into, so each relational operator compiles exactly once. Plan nodes `InMemoryTable`, `ParquetScan`, `Filter`, `Project`, `Limit`, `Sort`, `Aggregate`, `Join` chain via `.filter()`, `.select()`, `.aggregate()`, `.sort()`, `.limit()`, `.join()`; `plan.execute()` opens a pull-based processor tree. A `.filter` directly above a `ParquetScan` pushes its predicate into the scan for row-group and page pruning.
- The AOT lane's whole point is that the closed world is dead-code-eliminable: the fused gate binary is several times smaller in `__text` than the runtime equivalent. `benchmarks/binary_size/` is the live gate — trust it over any ratio quoted in prose.

**Parquet I/O** (`marrow/parquet`) — a from-scratch reader/writer, no PyArrow at runtime
- `read_table(path, columns=None)` — decode a Parquet file into a marrow `Table`, with optional column projection
- `write_table(table, path, compression=..., data_page_version=...)` — encode a `Table`; snappy / zstd / lz4 / none, data page v1 & v2
- Column statistics (min/max/null/distinct), page index, and bloom filters on write; row-group & page pruning on read
- Python API mirrors `pyarrow.parquet`: `import marrow.parquet as pq; pq.read_table(...)` / `pq.write_table(...)`

**Python bindings** — `import marrow as ma`
- `array(values, type=None)` — create any array type from Python lists with type inference
- `marrow.compute` exposes 26 kernels — arithmetic, comparison, aggregates, `filter`/`take`/`drop_null`, `sort`/`sort_indices`, `cast`, distinct counts. The string, temporal, boolean and conditional families are implemented in Mojo but **not yet bound**.
- Full null handling, type coercion, nested structure support
- The relational engine is **not** bound yet — there is no lazy/expression frontend in Python

**Arrow IPC** (`marrow/ipc`)
- `read_ipc_file(path)` / `write_ipc_file(path, schema, batches)` — IPC file format
- `read_ipc_stream(bytes)` / `write_ipc_stream(schema, batches)` — IPC stream format
- `RecordBatchFileReader`, `RecordBatchStreamReader` — streaming readers
- `RecordBatchFileWriter`, `RecordBatchStreamWriter` — streaming writers
- Full round-trip for all implemented types including nested, dictionary, temporal, and null columns

**Interoperability**
- Arrow C Data Interface — zero-copy exchange with PyArrow
- GPU acceleration via Mojo's `DeviceContext` (Metal on Apple Silicon, CUDA on NVIDIA)

## Python Quick Start

```bash
pixi run -e dev build_python   # compile libmarrow.so
```

```python
import marrow as ma

# ── Array construction ────────────────────────────────────────────────────────

# Primitive arrays — type inference
a = ma.array([1, 2, 3, None, 5])           # int64 with one null
f = ma.array([1.0, 2.5, None, 4.0])        # float64

# Explicit types
a = ma.array([1, 2, 3, None, 5], type=ma.int64())

# Strings
s = ma.array(["hello", None, "world"])

# Nested lists
nested = ma.array([[1, 2], [3, 4, 5], None])

# Struct arrays — automatic type inference from dict keys
structs = ma.array([{"x": 1, "y": 1.5}, {"x": 2, "y": 2.5}])

# With explicit schema
t = ma.struct([ma.field("x", ma.int64()), ma.field("y", ma.float64())])
structs = ma.array([{"x": 1, "y": 1.5}, {"x": 2, "y": 2.5}], type=t)

# ── Arithmetic (null-propagating) ─────────────────────────────────────────────

b = ma.array([10, 20, 30, None, 50])
result = ma.add(a, b)          # null where either input is null
result = ma.subtract(a, b)
result = ma.multiply(a, b)
result = ma.divide(a, b)

# ── Aggregates (null-skipping) ────────────────────────────────────────────────

ma.sum(a)        # → 11   (skips the null at index 3)
ma.product(a)    # → 30
ma.min(a)        # → 1
ma.max(a)        # → 5
ma.mean(a)       # → 2.75 (float64)
ma.count_distinct(a)          # → 4
ma.approx_count_distinct(a)   # → ~4 (HyperLogLog)
ma.any(ma.array([False, True, None]))   # → True
ma.all(ma.array([True, True, None]))    # → True

# ── Selection ─────────────────────────────────────────────────────────────────

mask = ma.array([True, False, True, False, True])
ma.filter(a, mask)     # [1, 3, 5]
ma.drop_null(a)        # [1, 2, 3, 5]  (removes index 3)

# ── Casting (marrow.compute mirrors pyarrow.compute) ──────────────────────────

from marrow import compute as mc
mc.cast(a, ma.float64())               # int64 → float64
mc.cast(f, ma.int32(), safe=False)     # truncating float → int

# ── Tables: group-by, aggregate, sort, join ───────────────────────────────────

rb  = ma.record_batch({"k": ma.array([1, 2, 1]), "v": ma.array([10, 20, 30])})
dim = ma.record_batch({"k": ma.array([1, 2]), "label": ma.array(["a", "b"])})
rb.group_by("k").aggregate([("v", "sum"), ("v", "count_distinct")])
rb.aggregate([("v", "sum")])                       # whole-table, one row
rb.sort_by([("k", "ascending"), ("v", "descending")])
rb.join(dim, ["k"], join_type="inner")

# ── Array methods ─────────────────────────────────────────────────────────────

len(a)             # 5
a.null_count()     # 1
a.type()           # int64
a.slice(1, 3)      # [2, 3, None]  — zero-copy
a[0]               # 1
str(a)             # "Int64Array([1, 2, 3, NULL, 5])"

# Struct field access
structs.field(0)           # Int64Array — field "x"
structs.field("y")         # Float64Array — field "y"
```

## Mojo API

### Creating arrays

```mojo
from marrow.arrays import array, PrimitiveArray, StringArray, BoolArray
from marrow.dtypes import int8, int32, int64, bool_, list_

# Factory function — list of optionals
var a = array[int32]([1, 2, 3, 4, 5])
var b = array[int64]([1, None, 3, None, 5])   # nulls at index 1 and 3
var c = array[bool_]([True, False, True])
```

### Builders

```mojo
from marrow.builders import PrimitiveBuilder, StringBuilder, ListBuilder

# Primitive
var pb = PrimitiveBuilder[int64](capacity=4)
pb.append(10)
pb.append(20)
pb.append_null()
pb.append(40)
var arr: Int64Array = pb.finish()

# String
var sb = StringBuilder()
sb.append("hello")
sb.append_null()
sb.append("world")
var strs: StringArray = sb.finish()

# List of int32 — append child elements, then commit each list element
var child = PrimitiveBuilder[int32]()
child.append(1)
child.append(2)
var lb = ListBuilder(child^)       # moves child into the builder
lb.append(True)                    # [1, 2] is the first list element
lb.values().append(3)              # child element for the next list
lb.append(True)                    # [3] is the second list element
lb.append_null()                   # null third element
var lists: ListArray = lb.finish()
```

### Display

All arrays implement `Writable` so they print directly:

```mojo
print(arr)    # Int64Array([10, 20, NULL, 40])
print(strs)   # StringArray([hello, NULL, world])
```

### Compute kernels

```mojo
from marrow.kernels.arithmetic import add, subtract, multiply, divide, sqrt, log, sin
from marrow.kernels.aggregate import sum, min, max, mean, any, all
from marrow.kernels.filter import filter, drop_null, take
from marrow.kernels.compare import equal, less, greater_equal
from marrow.kernels.groupby import GroupBy

var x = array[int64]([1, 2, 3, 4])
var y = array[int64]([10, 20, 30, 40])

var z = add(x, y)               # Int64Array([11, 22, 33, 44])
var total = sum(x)              # 10  (type inferred from x)
var filtered = filter(x, array[bool_]([True, False, True, False]))

var a = array[int64]([1, 2, 3, 4])
var b = array[int64]([1, 3, 2, 4])
var eq = equal(a, b)            # BoolArray([true, false, false, true])
var lt = less(a, b)             # BoolArray([false, true, false, false])

# Unary math (floating-point)
var f = array[float64]([1.0, 4.0, 9.0, 16.0])
var s = sqrt(f)                 # Float64Array([1.0, 2.0, 3.0, 4.0])
var l = log(f)                  # natural log

# Group-by — GroupBy(keys).sum/min/max(values) → RecordBatch
var keys = array[int64]([1, 2, 1, 2, 1])
var vals = array[float64]([10.0, 20.0, 30.0, 40.0, 50.0])
var result = GroupBy(keys).sum(vals)   # RecordBatch: key=[1,2], sum=[90.0, 60.0]
```

### Expression execution

```mojo
from marrow.expr import col, lit, in_memory_table, execute
from marrow.tabular import record_batch

var batch = record_batch(
    [array[int64]([25, 35, 45]), array[String](["Alice", "Bob", "Carol"])],
    names=["age", "name"],
)
var plan = in_memory_table(batch)
    .filter(col("age") > lit(30))
    .select("name", "age")

var result = execute(plan)   # a single RecordBatch
```

### Parquet I/O

```mojo
from marrow.parquet import read_table, write_table

var tbl = read_table("data.parquet")
write_table(tbl, "output.parquet")   # native encoder — no PyArrow at runtime
```

### Zero-copy PyArrow interop (C Data Interface)

```mojo
from std.python import Python
from marrow.c_data import CArrowArray, CArrowSchema

var pa = Python.import_module("pyarrow")
var pyarr = pa.array([1, 2, 3, 4, 5], mask=[False, False, False, False, True])

var capsules = pyarr.__arrow_c_array__()
var dtype = CArrowSchema.from_pycapsule(capsules[0]).to_dtype()  # int64
var data = CArrowArray.from_pycapsule(capsules[1])^.to_array(dtype)
var typed = data.as_int64()

print(typed.is_valid(0))   # True
print(typed.is_valid(4))   # False  (null)
print(typed.unsafe_get(0)) # 1
```

## Benchmarks

Python array construction vs PyArrow (n=100,000 elements, Apple M-series, mean time):

| Array type               | marrow  | PyArrow | speedup        |
|--------------------------|--------:|--------:|---------------|
| int64 (explicit type)    | 0.30 ms | 0.92 ms | **3.0x faster** |
| int64 + nulls (explicit) | 0.30 ms | 0.91 ms | **3.0x faster** |
| float64 (explicit)       | 0.28 ms | 0.48 ms | **1.7x faster** |
| float64 + nulls          | 0.28 ms | 0.52 ms | **1.8x faster** |
| string (explicit)        | 0.81 ms | 1.07 ms | **1.3x faster** |
| string + nulls           | 0.80 ms | 1.04 ms | **1.3x faster** |
| struct, primitive fields | 4.64 ms | 6.35 ms | **1.4x faster** |
| int64 (inferred)         | 1.58 ms | 1.28 ms | 1.2x slower    |
| string (inferred)        | 0.92 ms | 1.01 ms | ~parity        |
| nested list (inferred)   | 0.61 ms | 2.37 ms | **3.9x faster** |

When the array type is provided explicitly, marrow's builder path is faster than PyArrow's for numeric and string types. Type inference involves a Python-side scan to detect the type, which adds overhead; this gap will narrow as the inference path is optimized.

Run the benchmarks yourself:

```bash
pixi run -e bench bench_python       # Python array construction vs PyArrow
pixi run -e bench bench              # CPU SIMD arithmetic benchmarks

# Side-by-side comparison table: marrow vs polars vs pyarrow vs duckdb
pixi run -e bench pytest --benchmark --no-mojo python/marrow/tests/bench_compute.py --competition
pixi run -e bench pytest --benchmark --no-mojo python/marrow/tests/bench_join.py --competition
```

## GPU Execution (experimental)

A few element-wise kernels — arithmetic, comparisons, and hashing — can dispatch
to the GPU (Metal on Apple Silicon, CUDA on NVIDIA) when an `ExecContext`
carries a `DeviceContext`. The GPU and CPU paths share the same kernel source.

```mojo
from gpu.host import DeviceContext
from marrow.execution import ExecContext
from marrow.kernels.arithmetic import add

var result = add(a, b, ExecContext.gpu(DeviceContext()))
```

These kernels are all low arithmetic intensity (~1 op per element), so for most
workloads CPU SIMD is competitive or faster — device transfer dominates. Treat
GPU dispatch as infrastructure for data that is already device-resident. Sort,
join, aggregate and filter run on the CPU only, and GPU binary kernels do not yet
propagate null bitmaps.

## Known Limitations

1. **C Data Interface**: Release callbacks are not invoked (Mojo cannot pass a callback to a C function yet). Consuming Arrow data from PyArrow works; producing data back to PyArrow via the release mechanism is not fully implemented.

2. **Testing**: Conformance against the Arrow specification is verified through PyArrow since Mojo has no JSON library yet. Full integration testing requires a Mojo JSON reader.

3. **Type coverage**: Boolean, numeric, string, binary, fixed-size binary, list, fixed-size list, large binary/string/list, struct, dictionary, null, temporal (date32/64, time32/64, timestamp, duration), and decimal (32/64/128/256) types are implemented. Union types are not yet supported.

4. **Parquet I/O**: Marrow reads and writes Parquet natively — it decodes and encodes the format itself with no PyArrow at runtime (snappy / zstd / lz4 compression, data page v1 & v2, statistics, page index, bloom filters). Modular (Parquet) encryption and some rarer encodings are not yet supported.

5. **GPU null handling**: Binary arithmetic kernels on the GPU do not propagate null bitmaps (GPU `bitmap_and` is not yet implemented). Null-aware GPU arithmetic is CPU-only for now.

## Development

Install [pixi](https://pixi.sh/latest/installation/). The project uses pixi
environments to keep optional dependencies out of the default install:

| Environment | Activate with | What it includes |
|---|---|---|
| `dev` | `-e dev` | pyarrow, pytest, ruff — daily dev and testing |
| `asan` | `-e asan` | dev + `libcompiler-rt` for AddressSanitizer runs |
| `bench` | `-e bench` | dev + polars, duckdb, rich for comparison benchmarks |
| `format` | `-e format` | ruff only |
| `docs` | `-e docs` | jupyter, quarto |

```bash
# testing
pixi run -e dev test              # all tests (Mojo + Python)
pixi run -e dev test_mojo         # Mojo unit tests only
pixi run -e dev test_python       # Python binding tests only

# benchmarks
pixi run -e bench bench           # all benchmarks
pixi run -e bench bench_mojo      # Mojo benchmarks only
pixi run -e bench bench_python    # Python vs PyArrow benchmarks only

# formatting
pixi run -e dev fmt               # format all code (Mojo + Python)

# AddressSanitizer
pixi run -e asan test_mojo_asan   # Mojo tests under ASAN
```

The Python shared library (`python/libmarrow.so`) is built automatically before
each test run — no manual `build_python` step required.

### Running individual tests

Use `pytest` directly to run a single test file or a specific test case:

```bash
# entire file
pixi run -e dev pytest marrow/kernels/tests/test_join.mojo

# single test
pixi run -e dev pytest marrow/kernels/tests/test_join.mojo::test_collision_left_join

# verbose output
pixi run -e dev pytest -v marrow/tests/test_arrays.mojo
```

### Pytest options

| Option | Effect |
|---|---|
| `--mojo` / `--no-mojo` | Select or exclude Mojo tests |
| `--python` / `--no-python` | Select or exclude Python tests |
| `--gpu` / `--no-gpu` | Select or exclude GPU tests |
| `--benchmark` | Include benchmark files (`bench_*.mojo` / `bench_*.py`); also switches to `-O3` |
| `--asan` | Enable AddressSanitizer (use `-e asan` environment) |
| `--competition` | After benchmarks, print a side-by-side comparison table across all measured libraries |

### Writing Mojo tests

Test files (`test_*.mojo`) use `TestSuite` from `marrow.testing`:

```mojo
from marrow.testing import TestSuite

def test_something() raises:
    assert_true(1 + 1 == 2)

def main():
    TestSuite.run[__functions_in_module()]()
```

`TestSuite.run` auto-discovers every `test_*` function in the module. No
registration needed — just name the function with the `test_` prefix.

### Writing Mojo benchmarks

Benchmark files (`bench_*.mojo`) use `BenchSuite` and `Benchmark` from
`marrow.testing`:

```mojo
from marrow.testing import BenchSuite, Benchmark, BenchMetric

def bench_my_kernel(mut b: Benchmark) raises:
    var data = _prepare_data(N)
    b.throughput(BenchMetric.elements, N)
    @always_inline
    @parameter
    def call():
        keep(my_kernel(data))
    b.iter[call]()

def main():
    BenchSuite.run[__functions_in_module()]()
```

`BenchSuite.run` auto-discovers every `bench_*` function. For multiple sizes,
define a shared helper and one thin wrapper per size:

```mojo
def _bench_kernel(mut b: Benchmark, n: Int) raises:
    ...

def bench_kernel_10k(mut b: Benchmark) raises: _bench_kernel(b, 10_000)
def bench_kernel_100k(mut b: Benchmark) raises: _bench_kernel(b, 100_000)
def bench_kernel_1m(mut b: Benchmark) raises: _bench_kernel(b, 1_000_000)
```

### Build caching

The test harness compiles each Mojo test runner to a binary in
`.test_runners/` using `mojo build`.  Runner files are named by a content
hash of the selected tests, so the binary path is stable across runs with
the same test selection.  On the second run `mojo build` detects the
existing binary and skips recompilation, reducing cold-start time from ~5 s
to ~1 s.  Up to 10 runner/binary pairs are kept; older ones are pruned
automatically.

If the project matures, the goal is to contribute it upstream to the Apache Arrow project.

### Common problems

If compilation fails on MacOS make sure you have the metal toolchain:

```
xcodebuild -downloadComponent MetalToolchain
```

## References

- [Arrow columnar format specification](https://arrow.apache.org/docs/format/Columnar.html)
- [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
- [Another effort to implement Arrow in Mojo](https://github.com/mojo-data/arrow.mojo)
