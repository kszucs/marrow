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
- `AnyArray` — type-erased immutable array container (O(1) copy via `ArcPointer`)
- `RecordBatch` — schema + column arrays, with slice, select, rename, add/remove/set column operations
- `Table` — schema + chunked columns; `from_batches()`, `to_batches()`, `combine_chunks()`

**Scalar types**
- `PrimitiveScalar[T]`, `StringScalar`, `ListScalar`, `StructScalar` — typed scalars holding native values
- `AnyScalar` — type-erased scalar backed by a length-1 `AnyArray`

**Builders** — incrementally build immutable arrays
- `PrimitiveBuilder[T]`, `StringBuilder`, `ListBuilder`, `FixedSizeListBuilder`, `StructBuilder`
- `AnyBuilder` — type-erased builder using function-pointer vtable dispatch (O(1) copy via `ArcPointer`)

**Compute kernels** (SIMD-vectorized, null-aware)
- Arithmetic: `add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `neg`, `abs_`, `min_`, `max_`
- Math (unary): `sign`, `sqrt`, `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `trunc`, `round`, `sin`, `cos`
- Math (binary): `pow_`
- Comparisons: `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` → `BoolArray` (CPU + GPU)
- Aggregates: `sum_`, `product`, `min_`, `max_`, `any_`, `all_` (null-skipping)
- Group-by: `groupby(keys, values, aggregations)` — fused hash+aggregate, returns `RecordBatch`
- Hashing: `hash_` for primitive, string, and struct arrays
- Selection: `filter_`, `drop_nulls`
- Strings: `string_lengths`
- Similarity: `cosine_similarity` (batch N-vectors vs 1 query, CPU SIMD + GPU)

**Expression execution** (`marrow/expr`)
- Build lazy expression trees with `col()`, `lit()`, `if_else()` and operator overloads (`+`, `-`, `*`, `/`, `>`, `<`, `==`, `&`, `|`, …)
- Relational plan nodes: `InMemoryTable`, `Filter`, `Project`, `ParquetScan`, `Aggregate` with `.filter()`, `.select()`, `.aggregate()` chaining
- Pull-based streaming executor: `Planner` compiles a plan into typed processor trees; `execute()` collects `RecordBatch` results

**Parquet I/O** (`marrow/parquet`)
- `read_table(path)` — read a Parquet file into a marrow `Table`
- `write_table(table, path)` — write a marrow `Table` to Parquet

**Python bindings** — `import marrow as ma`
- `array(values, type=None)` — create any array type from Python lists with type inference
- All compute kernels exposed as free functions
- Full null handling, type coercion, nested structure support

**Interoperability**
- Arrow C Data Interface — zero-copy exchange with PyArrow
- GPU acceleration via Mojo's `DeviceContext` (Metal on Apple Silicon, CUDA on NVIDIA)

## Python Quick Start

```bash
pixi run build_python   # compile marrow.so
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
result = ma.add(a, b)      # null where either input is null
result = ma.sub(a, b)
result = ma.mul(a, b)
result = ma.div(a, b)

# ── Aggregates (null-skipping) ────────────────────────────────────────────────

ma.sum_(a)       # → 11.0  (skips the null at index 3)
ma.product(a)    # → 30.0
ma.min_(a)       # → 1.0
ma.max_(a)       # → 5.0
ma.any_(ma.array([False, True, None]))   # → True
ma.all_(ma.array([True, True, None]))   # → True

# ── Selection ─────────────────────────────────────────────────────────────────

mask = ma.array([True, False, True, False, True])
ma.filter_(a, mask)    # [1, 3, 5]
ma.drop_nulls(a)       # [1, 2, 3, 5]  (removes index 3)

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
var arr: Int64Array = pb.finish_typed()

# String
var sb = StringBuilder()
sb.append("hello")
sb.append_null()
sb.append("world")
var strs: StringArray = sb.finish_typed()

# List of int32 — append child elements, then commit each list element
var child = PrimitiveBuilder[int32]()
child.append(1)
child.append(2)
var lb = ListBuilder(child^)       # moves child into the builder
lb.append(True)                    # [1, 2] is the first list element
lb.values().append(3)              # child element for the next list
lb.append(True)                    # [3] is the second list element
lb.append_null()                   # null third element
var lists: ListArray = lb.finish_typed()
```

### Display

All arrays implement `Writable` so they print directly:

```mojo
print(arr)    # Int64Array([10, 20, NULL, 40])
print(strs)   # StringArray([hello, NULL, world])
```

### Compute kernels

```mojo
from marrow.kernels.arithmetic import add, sub, mul, div, sqrt, log, sin
from marrow.kernels.aggregate import sum_, min_, max_, any_, all_
from marrow.kernels.filter import filter_, drop_nulls
from marrow.kernels.compare import equal, less, greater_equal
from marrow.kernels.groupby import groupby

var x = array[int64]([1, 2, 3, 4])
var y = array[int64]([10, 20, 30, 40])

var z = add(x, y)               # Int64Array([11, 22, 33, 44])
var total = sum_[int64](x)      # 10
var filtered = filter_[int64](x, array[bool_]([True, False, True, False]))

var a = array[int64]([1, 2, 3, 4])
var b = array[int64]([1, 3, 2, 4])
var eq = equal(a, b)            # BoolArray([true, false, false, true])
var lt = less(a, b)             # BoolArray([false, true, false, false])

# Unary math (floating-point)
var f = array[float64]([1.0, 4.0, 9.0, 16.0])
var s = sqrt(f)                 # Float64Array([1.0, 2.0, 3.0, 4.0])
var l = log(f)                  # natural log

# Group-by
var keys = array[int64]([1, 2, 1, 2, 1])
var vals = array[float64]([10.0, 20.0, 30.0, 40.0, 50.0])
var result = groupby(keys, [vals], ["sum"])   # RecordBatch: key=[1,2], sum=[90.0, 60.0]
```

### Expression execution

```mojo
from marrow.expr import col, lit, in_memory_table, execute, ExecutionContext
from marrow.tabular import record_batch

var batch = record_batch(
    [array[int64]([25, 35, 45]), array[String](["Alice", "Bob", "Carol"])],
    names=["age", "name"],
)
var plan = in_memory_table(batch)
    .filter(col("age") > lit(30))
    .select(col("name"), col("age"))

var ctx = ExecutionContext()
var results = execute(plan, ctx)   # List[RecordBatch]
```

### Parquet I/O

```mojo
from marrow.parquet import read_table, write_table

var tbl = read_table("data.parquet")
write_table(tbl, "output.parquet")
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
pixi run bench_python       # Python array construction vs PyArrow
pixi run bench              # CPU SIMD arithmetic benchmarks
pixi run bench_similarity   # cosine similarity: CPU vs GPU
```

## GPU Acceleration

GPU kernels are available for compute-intensive operations when a `DeviceContext` is provided. Benchmarked on Apple Silicon (M-series, Metal, unified memory):

**Cosine similarity** (batch N-vectors vs 1 query, dim=768):

| Vectors | CPU SIMD | GPU (upload per call) | GPU (pre-loaded) |
|--------:|---------:|----------------------:|-----------------:|
|    10 K |  baseline |         2–3x slower   |    ~1x (crossover) |
|   100 K |  baseline |          ~1x           |      ~3x faster  |
|   500 K |  baseline |           —            |     ~13x faster  |

The key pattern: upload data to the GPU once, run multiple kernels, download results at the end. The crossover vs CPU SIMD is around 10K vectors at dim≥384.

Element-wise arithmetic (`add`, `mul`, etc.) is faster on CPU SIMD — data transfer overhead dominates for low arithmetic-intensity operations.

```mojo
from std.gpu.host import DeviceContext
from marrow.kernels.similarity import cosine_similarity

# Pre-load data onto the GPU once
var ctx = DeviceContext()
var vectors_gpu = vectors.to_device(ctx)
var query_gpu = query.to_device(ctx)

# Run many similarity searches without re-uploading
var scores = cosine_similarity(vectors_gpu, query_gpu, ctx)
```

## Known Limitations

1. **C Data Interface**: Release callbacks are not invoked (Mojo cannot pass a callback to a C function yet). Consuming Arrow data from PyArrow works; producing data back to PyArrow via the release mechanism is not fully implemented.

2. **Testing**: Conformance against the Arrow specification is verified through PyArrow since Mojo has no JSON library yet. Full integration testing requires a Mojo JSON reader.

3. **Type coverage**: Only boolean, numeric, string, list, fixed-size list, and struct types are implemented. Date/time, dictionary, union, decimal, and binary types are not yet supported.

4. **Parquet I/O**: Parquet support currently bridges through PyArrow. Native Mojo Parquet reading is planned for a future release.

5. **GPU null handling**: Binary arithmetic kernels on the GPU do not propagate null bitmaps (GPU `bitmap_and` is not yet implemented). Null-aware GPU arithmetic is CPU-only for now.

## Development

Install [pixi](https://pixi.sh/latest/installation/), then:

```bash
pixi run test              # run all tests (Mojo + Python)
pixi run test_mojo         # Mojo unit tests only
pixi run test_python       # Python binding tests only
pixi run bench             # CPU/GPU arithmetic benchmarks
pixi run bench_python      # Python vs PyArrow array construction benchmarks
pixi run bench_similarity  # cosine similarity: CPU vs GPU vs GPU preloaded
pixi run fmt               # format all code (Mojo + Python)
```

If the project matures, the goal is to contribute it upstream to the Apache Arrow project.

## References

- [Arrow columnar format specification](https://arrow.apache.org/docs/format/Columnar.html)
- [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
- [Another effort to implement Arrow in Mojo](https://github.com/mojo-data/arrow.mojo)
