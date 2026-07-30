# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marrow is an implementation of Apache Arrow in Mojo. Apache Arrow is a cross-language development platform for in-memory data with a standardized columnar memory format. This implementation is in early/experimental stages as Mojo itself is under heavy development.

For information about the Mojo programming language and the standard library see https://github.com/modular/modular

## Build System & Commands

This project uses **pixi** as the package manager. Commands are scoped to environments:

| Environment | Purpose | Key command |
|-------------|---------|-------------|
| `dev`       | Tests + formatting (default for development) | `pixi run -e dev test` |
| `asan`      | AddressSanitizer test runs | `pixi run -e asan test_mojo_asan` |
| `bench`     | Benchmarks (polars, duckdb for comparison) | `pixi run -e bench bench` |
| `format`    | Formatting only (no test deps) | `pixi run -e format fmt` |
| `docs`      | Documentation generation | `pixi run -e docs docs` |
| `examples`  | Runnable examples | `pixi run -e examples datafusion_udf` |

```bash
# Run all tests
pixi run -e dev test

# Format code
pixi run -e dev fmt

# Build package
pixi run package
```

### Fast build-error checking (use this while a build is broken)

When the tree does not compile — e.g. after a Mojo upgrade — do **not** iterate
with `pytest`: it has to elaborate all of marrow before it can report anything.
Use the build-only task:

```bash
pixi run -e dev precompile                       # everything under marrow/, ~18 s
```

`precompile` is a plain `mojo precompile marrow -o .test_runners/marrow.mojoc`.
It compiles every module under `marrow/` — tests and benches included, since
they live in `marrow/**/tests/` and are ordinary importable modules — so it
catches errors in code no selected test happens to import, and surfaces **all**
errors and warnings in a single pass.

A single test file **cannot** be compiled on its own: without a `main()` there
is nothing to build (`mojo build` errors with `module does not contain a 'main'
function`). Run it through `pytest` instead.

**Never leave a `marrow.mojoc` where a runner can see it.** Such an artifact
*shadows the entire `marrow/` source tree*, so every import resolves against the
stale package instead of the files you just edited. Mojo puts a source file's own
directory on the import search path, which makes **two** locations dangerous: the
repo root (runners compile with `-I .`) and **`.test_runners/`**, where the
generated driver lives.

`precompile` therefore writes to `.precompile/`, which is on neither path. It used
to write to `.test_runners/`, and that silently broke every following `pytest`
run — symptom (observed 2026-07-27): a case you just added reports
`module 'test_<x>' does not contain 'test_<your_new_case>'` and the run fails in
well under a second, because nothing is compiled at all. Stale artifacts fail
differently again, with `unable to locate module 'tests'`. If you ever hit either,
`rm -f .test_runners/marrow.mojoc marrow.mojoc` first.

**The library must stay warning-clean.** Keep `mojo precompile marrow` at 0
errors rather than letting warnings accumulate until the output is unreadable.

Switch back to `pytest` once it compiles: building is not passing.

### Running Individual Tests

Always use `pytest` to run tests — never `mojo test` or `mojo run` directly.
The pytest harness generates the runner, selects cases, parses output, and
handles ASAN.

```bash
# single file
pixi run -e dev pytest marrow/tests/test_dtypes.mojo

# single test case
pixi run -e dev pytest marrow/tests/test_arrays.mojo::test_primitive_slice

# verbose (shows PASS/FAIL per test)
pixi run -e dev pytest -v marrow/kernels/tests/test_join.mojo
```

Useful options:

```bash
--benchmark              # include bench_*.mojo files; also enables -O3
--asan                   # AddressSanitizer (requires asan environment)
--gpu                    # include GPU tests (requires Metal/CUDA device)
--no-python              # skip Python binding tests
--competition            # print a side-by-side comparison table after benchmarks
```

### Fast iteration — one selection, one compilation unit

Compilation dominates, and almost all of it is elaborating **marrow**, not the
test bodies. The compiler takes a single input file per invocation, so building
per test file paid that elaboration once per file. The harness instead generates
**one driver** for the whole selection — `.test_runners/_test_driver_<hash>.mojo`,
which imports the selected cases and hands them to `TestSuite.run` as a tuple —
and compiles that. The name is the hash of the driver's own source, so
concurrent sessions with different selections never overwrite each other, while
the same selection always resolves to the same path and the same cached
artifact. Measured on `marrow/expr/tests` (9 files, 280 cases):

| | wall | peak RSS |
|---|---|---|
| one generated driver, all 9 files | 4 min 43 s | 17.0 GB |
| separately: `test_aggregates.mojo` | 204 s | 11.7 GB |
| separately: `test_join.mojo` | 198 s | 19.6 GB |

Two files on their own cost more than all nine together, and the aggregate peaks
*below* a single file — N files in one unit cost about what 1 file costs.

Consequences worth knowing:

- **Selecting fewer *files* is what saves time; selecting fewer *cases* is not.**
  A single `::test_name` builds the same unit as its whole file.
- **Blast radius is the selection — for compile *errors*.** A diagnostic fails every
  case in the run, not just its file, because there is one unit.
- **A compiler *crash* is different: the harness splits the unit and retries.**
  The Mojo compiler dies (bug-report dump, no diagnostic) on some units simply
  because of how much they elaborate — the same cases build in smaller units. On
  a crash `MojoRunner.collect` halves the selection and compiles each half, down
  to a single case; a case that still cannot be built reports the crash as *its
  own* failure instead of failing everything selected alongside it. Ordinary
  `error:` output never splits, since it would be identical in every half.
- **Peak memory scales with the unit**, so a full-suite run is a single very
  large compile. Narrow the selection if memory is tight.
- **Optimization level follows the kind, not the session.** `bench_*.mojo` builds
  at `-O3` in its own driver; `test_*.mojo` at `-O1` with `-D ASSERT=all`. They
  cannot share a unit. `-O0` is not an option: the masked-gather intrinsic in
  `filter`/`take` fails to lower.
- **Case names must stay unique across the whole suite** — the runner reports by
  name, and that is how results map back to pytest items.
- **The driver is generated deterministically** (modules sorted, cases in source
  order) so re-running an unchanged selection produces byte-identical source and
  hits the Mojo compiler's own artifact cache. That cache only ever helps
  *identical* rebuilds: it does nothing across different files.
- Ordinary runs use **`mojo run`** — compile and execute in one step, no artifact
  left behind. **`--asan` runs use `mojo build`**, because the sanitizer runtime
  has to be linked into a real binary.
- Always go through `pytest`, never `mojo test` or a hand-written `mojo run`:
  the harness owns the driver, the flags, and the JSON parsing.

### Only run the tests the change could have broken

Do **not** run the full suite to validate a scoped change. Select the test
directories that import the code you touched; it tells you nothing extra about
modules the diff cannot reach.

For example, after editing `marrow/expr/*` and `marrow/kernels/*`:

```bash
pixi run -e dev pytest marrow/expr/tests marrow/kernels/tests
```

Narrow further when the change is narrower — the win comes from dropping
*files* from the unit, so a directory or a file helps; a single case does not:

```bash
pixi run -e dev pytest marrow/kernels/tests/test_groupby.mojo
```

### Checking everything after a big change

```bash
pixi run -e dev test               # pytest -v, everything
```

`pytest-xdist` does not help *within* a run: the selection is a single
compilation unit, so splitting it across workers makes each worker build a unit
of its own. Running several independent `pytest` invocations concurrently is
safe, though — drivers are content-addressed, so their files and (under
`--asan`) their binaries never collide.

The Python shared library (`python/libmarrow.so`) is rebuilt automatically by
`conftest.py` before each test session — no manual `build_python` step needed.

### Writing Mojo Tests

Test files live next to the code they cover (`marrow/tests/`,
`marrow/kernels/tests/`, …) and are **plain importable modules** — no `main()`,
no `TestSuite` import. The harness generates the runner:

```mojo
from std.testing import assert_true
from ..dtypes import int64          # relative: absolute `marrow.x` imports
                                    # break when compiled as part of the package

def test_something() raises:
    assert_true(1 + 1 == 2)
```

Rules that the single-runner harness depends on:

- **No `def main()`** — `mojo precompile marrow` rejects `main()` inside a
  package, and the generated driver supplies the only one. A standalone program
  belongs in `benchmarks/` instead (see `benchmarks/profiles/`).
- **Relative imports only** for `marrow.*` — `..x` from `marrow/tests/`, `...x`
  from `marrow/<sub>/tests/`. Absolute `from marrow.x import` fails with
  `unable to locate module 'marrow'` when the file is compiled as part of the
  package.
- **Test names must be unique across the entire suite**, not just per file — the
  runner reports results by name.
- Each tests directory needs an `__init__.mojo`; `marrow/` is a package, so its
  subdirectories must be declared packages to be importable.

### Writing Mojo Benchmarks

Benchmark files (`bench_*.mojo`) live beside their tests, follow the same rules
(no `main()`, relative imports), and use `Benchmark` from `marrow.testing`:

```mojo
from marrow.testing import Benchmark, BenchMetric

def bench_my_kernel(mut b: Benchmark) raises:
    var data = _prepare_data(N)
    b.throughput(BenchMetric.elements, N)
    @always_inline
    @parameter
    def call():
        keep(my_kernel(data))
    b.iter[call]()
    keep(data)  # prevent ASAP destruction (see note below)
```

**Important — `keep(data)` after `b.iter[call]()`**: Mojo's ASAP (As-Soon-As-Possible) destruction frees values as early as the compiler believes their last use has passed. When a `@parameter` closure captures a variable (e.g. `data`) and is passed to `b.iter[call]()`, ASAP may determine that `data` is no longer needed *after* the closure is registered but *before* it actually runs, causing a heap-use-after-free inside the iteration loop. Adding `keep(data)` after `b.iter[call]()` forces `data` to remain live through the entire benchmark. This applies to all non-trivial captured values: `StructArray`, `PrimitiveArray[T]`, `SwissHashTable`, `HashJoin`, etc.

For multiple sizes, define a shared helper and one thin wrapper per size:

```mojo
def _bench_kernel(mut b: Benchmark, n: Int) raises:
    ...

def bench_kernel_10k(mut b: Benchmark) raises: _bench_kernel(b, 10_000)
def bench_kernel_100k(mut b: Benchmark) raises: _bench_kernel(b, 100_000)
def bench_kernel_1m(mut b: Benchmark) raises: _bench_kernel(b, 1_000_000)
```

## Core Architecture

### Type-Erased Containers

Mojo lacks dynamic dispatch, so the codebase uses **type-erased containers** with **implicit conversions** to/from typed wrappers. Implicit conversions are cheap (O(1) ref-count bumps).

#### Arrays (`marrow/arrays.mojo`)

- **`Array`** - Trait that all typed arrays implement. Provides the common read-only interface: `type()`, `null_count()`, `is_valid()`, `as_any()`. Also extends `Sized`, `Writable`, `Equatable`, `Copyable`, `Movable`.
- **`DynArray`** - Type-erased, immutable array container (analogous to `ArrayData` in C++ Arrow). Holds `dtype`, `length`, `nulls`, `bitmap`, `buffers`, `children`, `offset`. Copying is O(1) via `ArcPointer` ref-counting inside `Buffer`/`Bitmap`.
- **Typed arrays** implement the `Array` trait and convert implicitly to/from `DynArray`:
  - `PrimitiveArray[T]` - numeric/boolean types
  - `StringArray` - UTF-8 strings
  - `ListArray` - variable-length nested lists
  - `FixedSizeListArray` - fixed-size nested lists (e.g. embedding vectors)
  - `StructArray` - nested structs
  - `ChunkedArray` - array split across multiple chunks (does NOT implement `Array` trait)

Usage: `var arr: DynArray = my_primitive_array` and `var prim: PrimitiveArray[int64] = some_array` both work transparently.

#### Builders (`marrow/builders.mojo`)

- **`Builder`** - Trait every typed builder implements.
- **`DynBuilder`** - Type-erased builder. Can be constructed from a `DataType` at runtime. `finish()` returns `DynArray`.
- **Typed builders** convert implicitly to `DynBuilder` by cloning the `ArcPointer`, so the original typed builder remains usable after passing to a composite builder:
  - `PrimitiveBuilder[T]` → `PrimitiveArray[T]`
  - `StringBuilder` → `StringArray`
  - `ListBuilder` → `ListArray`
  - `FixedSizeListBuilder` → `FixedSizeListArray`
  - `StructBuilder` → `StructArray`

### Key Abstractions

**Buffer** (`marrow/buffers.mojo`):
- `Buffer[mut=False]` — immutable, ref-counted via `ArcPointer[Allocation]`
- `Buffer[mut=True]` — mutable counterpart (replaces the former `BufferBuilder`); `finish()` freezes to `Buffer[mut=False]`
- Allocation kinds: CPU (owned heap), FOREIGN (external with release callback), HOST (pinned GPU host memory), DEVICE (GPU memory)
- All buffers are 64-byte aligned and padded. Prefer `Buffer`/`Bitmap` for owned values and `BufferView`/`BitmapView` for computation. Avoid naked pointer arithmetic — do not use raw pointer types directly in kernel or array code.
- **`unsafe_ptr()` is restricted to `buffers.mojo`, `views.mojo`, and `c_data.mojo` only.** All other files (kernels, arrays, tests, etc.) must not call `unsafe_ptr()` directly. Kernels and array code should operate through `BufferView`/`BitmapView` abstractions instead.
- **Avoid `AnyOrigin` types (`MutAnyOrigin`, `ImmutAnyOrigin`) and `unsafe_origin_cast`.** Use parametric origins instead (e.g. `out_o: Origin[mut=True]` / `src_o: Origin[mut=False]`) and pass views directly without origin casts.

**Bitmap** (`marrow/buffers.mojo` — *not* a `bitmap.mojo`; it lives beside `Buffer`):
- `Bitmap[mut=False]` — immutable, bit-packed validity buffer wrapping a `Buffer`
- Copying is O(1) (ref-count bump)
- `Bitmap[mut=True]` is the mutable counterpart; `finish()` freezes to `Bitmap[mut=False]`

**Views** (`marrow/views.mojo`):
- `BufferView` / `BitmapView` — borrowed, offset-applied spans over a `Buffer`/`Bitmap`. These are what
  kernels compute over; see the `unsafe_ptr()` restriction above.

**DataType** (`marrow/dtypes.mojo`):
- Struct-based type system matching Arrow specification
- Supports primitive types (bool, int8-64, uint8-64, float32/64)
- Nested types via `list_(DataType)`, `fixed_size_list_(DataType, size)`, and `struct_(Field, ...)`
- Uses `code` field for type identification and optional `native` field for DType mapping

**Runtime → comptime dispatch** (`marrow/dtypes.mojo`, `marrow/utils.mojo`):
- There is no visitor module. Runtime dtype dispatch is the `DynType.dispatch_*` family —
  `dispatch_primitive` / `dispatch_numeric` / `dispatch_integer` / `dispatch_floating` /
  `dispatch_temporal` / `dispatch_stringlike` / `dispatch_binarylike` / `dispatch_listlike` —
  each resolving a runtime `DataType` to a comptime type parameter for a `capturing` job.
- **Dispatch on the widest family the typed leaf accepts** (see the Kernel Implementation Pattern
  below): a leaf bound on `PrimitiveType` needs one `dispatch_primitive` arm, not one per family.
- `variant_dispatch*` (`marrow/utils.mojo`) is the comptime adapter over stdlib `Variant`.

**C Data Interface** (`marrow/c_data.mojo`):
- `CArrowSchema` and `CArrowArray` for zero-copy data exchange
- Import: `CArrowSchema.from_pycapsule()` + `.to_dtype()`, `CArrowArray.from_pycapsule()` + `.to_array(dtype)`
- Export: `CArrowSchema.from_dtype(dtype).to_pycapsule()`, `CArrowArray.from_array(arr).to_pycapsule()`
- Python arrays expose `__arrow_c_array__()` and `__arrow_c_schema__()` protocol methods for zero-copy exchange with PyArrow

**Tabular** (`marrow/tabular.mojo`):
- `RecordBatch` - schema + column arrays

### Directory Structure

```
marrow/
├── dtypes.mojo           # Type system (DataType, Field, DynType.dispatch_*)
├── buffers.mojo          # Memory management (Buffer[mut], Bitmap[mut], Allocation)
├── views.mojo            # BufferView, BitmapView — what kernels compute over
├── arrays.mojo           # Array, PrimitiveArray, StringArray, ListArray,
│                         # FixedSizeListArray, StructArray, ChunkedArray
├── builders.mojo         # Builder, DynBuilder, PrimitiveBuilder, StringBuilder,
│                         # ListBuilder, FixedSizeListBuilder, StructBuilder
├── scalars.mojo          # Scalar trait, DynScalar, PrimitiveScalar
├── utils.mojo            # variant_dispatch*, GPU_ENABLED
├── kernels/
│   ├── arithmetic.mojo   # Element-wise add, subtract, multiply, divide, math
│   ├── aggregate.mojo    # Sum, mean, min, max, count, product
│   ├── boolean.mojo      # Logical operations
│   ├── compare.mojo      # Comparisons
│   ├── cast.mojo         # Type casting
│   ├── filter.mojo       # Array filtering / take / drop_null
│   ├── groupby.mojo      # Hash group-by
│   ├── join.mojo         # Hash join
│   ├── sort.mojo         # Sort / sort_indices
│   ├── string.mojo       # String kernels
│   └── tests/            # test_*.mojo + bench_*.mojo for the kernels
├── expr/
│   └── tests/            # test_*.mojo for the expression layer
├── parquet/
│   └── tests/            # test_*.mojo + bench_*.mojo for parquet
├── c_data.mojo           # Arrow C Data Interface
├── ipc.mojo              # Arrow IPC file / stream reader + writer
├── schema.mojo           # Schema with Fields and metadata
├── tabular.mojo          # RecordBatch, Table
└── tests/                # test_*.mojo + bench_*.mojo for the core modules
python/                   # The Python module top level
└── marrow/tests/         # Python test_*.py and bench_*.py
benchmarks/               # Standalone programs (they own a `main()`, so they
                          # cannot live inside the package)
```

Tests and benchmarks sit **inside** the package, next to the code they cover.
That works because they carry no `main()` — see "Writing Mojo Tests".

## Implementation Patterns

### Type Constraints

Mojo lacks dynamic dispatch, so the codebase uses:
- Type-erased containers (`DynArray`, `DynBuilder`) with implicit conversions to/from typed wrappers
- Compile-time parameterization (`PrimitiveArray[int64]`)
- The `DynType.dispatch_*` family for runtime dtype → comptime type dispatch
- Runtime type checking via `DataType.code` comparison

## GPU Compute

### GPU codegen is opt-in — `-D MARROW_GPU=true`

**Every device path is compiled out by default.** `marrow.utils.GPU_ENABLED`
(`marrow/utils.mojo`) is the single switch, defaulting to False, and it gates
all of it: the device allocations in `cast`/`arithmetic`/`hashing`/`boolean`/
`compare`, the accelerator arms of `_apply_dispatch`, and
`has_accelerator_support`, which answers False so a GPU `ExecutionContext`
raises `"apply: no GPU accelerator available"` at the dispatch site rather than
silently taking a CPU path.

```bash
mojo build -D MARROW_GPU=true ...      # opt in
pixi run -e dev pytest --gpu           # the harness passes it for you
```

Anything touching device code must add `comptime if GPU_ENABLED:` around it —
a *runtime* `if ctx.is_gpu()` alone is not enough, since it cannot be
eliminated at elaboration time. `mojo precompile` rejects `-D` outright, so the
`precompile` task always builds the CPU-only configuration.

This is the **largest single compile-time lever** in the tree. Cold builds
(fresh `MODULAR_CACHE_DIR`; a repeated identical compile just hits the Mojo
transform cache and measures nothing):

| | GPU off (default) | GPU on |
|---|---|---|
| `cast`, numeric x numeric | **14.6 s** | 40.1 s |
| `cast` + `sort_indices` | **43.7 s** | 85.0 s |

**Both halves must be gated or you get none of it.** The allocations need
`comptime if GPU_ENABLED` *and* `has_accelerator_support` must answer False.
Gating either alone measures as no change at all (45.2 s and 84.4 s against
42.5 s / 84.1 s baselines) — which is exactly why this looked like a dead end
for a long time.

It does not shed the `libmax`/AsyncRT runtime dependency, though: a binary
built with GPU off still links it.

### Architecture

GPU kernels live in `marrow/kernels/` and are imported lazily from CPU-side modules (e.g. `arithmetic.mojo`) only when a `DeviceContext` is passed. This avoids requiring GPU compilation tools for CPU-only usage.

The `Buffer` struct has an optional `device` field (`Optional[DeviceBuffer]`). When set, the buffer has a GPU-resident copy. GPU kernel orchestration functions (e.g. `_add_gpu`, `_cosine_similarity_gpu`) check `buffer.has_device()` to skip uploads when data is already on the GPU.

### Device Transfer

- `PrimitiveArray[T].to_device(ctx)` / `.to_host(ctx)` — upload/download array data
- `FixedSizeListArray.to_device(ctx)` — uploads child values and bitmap
- `Buffer.to_device(ctx)` / `Bitmap.to_device(ctx)` — low-level transfer
- GPU kernel results are device-only by default (null host ptr, device buffer set) — call `.to_host(ctx)` to read on CPU

### Performance Guidelines

Benchmarked on Apple Silicon (M-series, Metal GPU, unified memory):

- **Low arithmetic intensity ops (e.g. element-wise add)**: CPU SIMD is faster. The data transfer overhead dominates when there's only ~1 FLOP per element. Don't GPU-accelerate these.
- **High arithmetic intensity ops (e.g. cosine similarity, ~3×dim FLOPs per vector)**: GPU wins at scale with pre-loaded data.
- **Data transfer is the bottleneck**: Raw GPU path (upload every call) is 2-3x slower than CPU even for compute-intensive kernels. Pre-loading data on the GPU is critical.
- **Crossover point**: ~10K vectors for cosine similarity with dim≥384. Below that, CPU SIMD wins.
- **At scale (500K-1M vectors, dim 768)**: GPU preloaded is ~13x faster than CPU SIMD.
- **Guideline**: Keep data device-resident across operations. Upload once, run multiple kernels, download results at the end.

### Benchmarks

```bash
pixi run bench_similarity   # CPU vs GPU vs GPU-preloaded cosine similarity
pixi run bench              # CPU arithmetic benchmarks
pixi run bench_gpu          # GPU arithmetic benchmarks
```

## Known Limitations

1. **Type system**: Variant elements must be copyable; references/lifetimes still evolving
2. **C callbacks**: Release callbacks in C Data Interface not called (Mojo limitation)
3. **Testing**: Relies on PyArrow for conformance testing until Mojo has JSON library
4. **Coverage**: bool, numeric (int/uint/float), string/large_string, binary/large_binary, fixed_size_binary, list/large_list/fixed_size_list, struct, map, dictionary, decimal (32/64/128/256), and temporal (date/time/timestamp/duration/interval) types are implemented; union, run-end-encoded, and view layouts are not
5. **Table**: `RecordBatch` and `Table` (schema + chunked columns) are implemented in `tabular.mojo`

## Dependencies

- Mojo `<1.0.0` (nightly builds from conda-forge and modular channels)
- PyArrow `>=19.0.1, <21` (for testing and C Data Interface validation)

## Coding Guidelines

- **Always use `def` for function definitions, never `fn`.** The `fn` keyword is deprecated in Mojo in favour of `def`. All functions, methods, and trait requirements must use `def`.
- **Never use `alias` — always use `comptime` instead.** `alias` is deprecated in Mojo. Use `comptime var` or `comptime` parameters everywhere a compile-time value is needed.
- **Never call `_underscore_prefixed` methods outside of the type/struct that defines them.** They are private implementation details. Use the public factory methods and APIs instead (e.g. use `Buffer.alloc_uninit[T](n)` directly rather than computing `Buffer._aligned_size[T](n)` and passing bytes manually).
- **Do not use `PrimitiveArray[bool_]` or `as_primitive[bool_]()`.**  Boolean arrays are bit-packed and require `BoolArray` for correct values access. Use `BoolArray` and `as_bool()` directly everywhere booleans are handled. Likewise, use `BoolBuilder` instead of `PrimitiveBuilder[bool_]`.
- **Prefer typed shorthand accessors over `.as_primitive[T]()`** when dispatching on a concrete type. `DynArray`, `DynScalar`, and `DynBuilder` all expose `.as_int8()`, `.as_int16()`, `.as_int32()`, `.as_int64()`, `.as_uint8()`, `.as_uint16()`, `.as_uint32()`, `.as_uint64()`, `.as_float16()`, `.as_float32()`, `.as_float64()` — use these instead of `.as_primitive[Int32Type]()` etc. Mojo can then infer the type parameter on the kernel call too, so no explicit `kernel[Int32Type](arr.as_int32())` is needed — write `kernel(arr.as_int32())`. Exception: when the type is a generic parameter `T` (e.g. inside a parameterized function), `.as_primitive[T]()` is the only option.
- **Prefer typed array aliases over `PrimitiveArray[XxxType]`**. The aliases `Int8Array`, `Int16Array`, `Int32Array`, `Int64Array`, `UInt8Array`, `UInt16Array`, `UInt32Array`, `UInt64Array`, `Float16Array`, `Float32Array`, `Float64Array` are defined in `arrays.mojo` and exported. Use `UInt64Array` instead of `PrimitiveArray[UInt64Type]` everywhere a concrete type is known. Exception: when the type is a generic parameter `T`, `PrimitiveArray[T]` is the only option.
- **Prefer typed builder aliases over `PrimitiveBuilder[XxxType]`**. The aliases `Int8Builder`, `Int16Builder`, `Int32Builder`, `Int64Builder`, `UInt8Builder`, `UInt16Builder`, `UInt32Builder`, `UInt64Builder`, `Float16Builder`, `Float32Builder`, `Float64Builder` are defined in `builders.mojo` and exported. Use `Int32Builder(n)` instead of `PrimitiveBuilder[Int32Type](n)` everywhere a concrete type is known. Exception: when the type is a generic parameter `T`, `PrimitiveBuilder[T]` is the only option.
- **Prefer typed scalar aliases over `PrimitiveScalar[XxxType]`**. The aliases `Int8Scalar`, `Int16Scalar`, `Int32Scalar`, `Int64Scalar`, `UInt8Scalar`, `UInt16Scalar`, `UInt32Scalar`, `UInt64Scalar`, `Float16Scalar`, `Float32Scalar`, `Float64Scalar` are defined in `scalars.mojo` and exported. Use `Int32Scalar(42)` instead of `PrimitiveScalar[Int32Type](42)` everywhere a concrete type is known. Exception: when the type is a generic parameter `T`, `PrimitiveScalar[T]` is the only option.
- **Do not wrap typed builders in explicit `DynBuilder(...)` calls.** `DynBuilder` has an `@implicit` conversion from any type implementing `Builder`, so passing a typed builder directly where `DynBuilder` is expected works without explicit wrapping. Write `ListBuilder(Int32Builder())` not `ListBuilder(DynBuilder(Int32Builder()))`. Exception: when constructing `DynBuilder` from a runtime `DataType` (e.g. `DynBuilder(dtype)`) or with explicit capacity, the explicit call is required.
- **Do not wrap typed arrays or scalars in explicit `DynArray(...)`/`DynScalar(...)` calls.** Both have `@implicit` conversion from their respective `Array`/`Scalar` traits, so typed values can be passed or assigned directly. Write `var a: DynArray = array([1, 2], int32)` not `var a = DynArray(array([1, 2], int32))`, and pass typed arrays directly as function arguments where `DynArray` is expected.
- **Prefer `.values()` over `.buffer.view[native](array.offset)`.**  `PrimitiveArray[T].values()` and `BoolArray.values()` return a properly offset `BufferView` / `BitmapView` in one call. Call `.buffer.view[native]()` only inside `buffers.mojo` or when constructing a view with explicit parameters not covered by `.values()`.
- Prefer explicit `if/else` over early-return `if + return` guard clauses. Keep the control flow flat and readable with `if/else` branches.
- Prefer PyArrow's API naming everywhere — both in the Mojo core types and in the Python bindings. When in doubt, match PyArrow's method names and signatures.
- **Python API must closely follow PyArrow's design.** Method names, signatures, and default values on `Array`, `Scalar`, `RecordBatch`, and `Table` must match PyArrow's equivalents so users can apply PyArrow muscle memory and potentially switch between libraries with minimal code changes. Diverge only where Arrow semantics genuinely differ. The Python wrapper classes in `python/marrow/` use **composition** (a `_Wrapper` base class with a `._binding` slot holding the C extension object) — not inheritance. All optional arguments, default values, and convenience methods live in pure Python. The Mojo binding files stay minimal and strict: no optional args, no sugar.
- Use **conventional commits** for all commit messages (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`, etc.), with an optional scope in parentheses (e.g. `feat(kernels): add concat`).
- Add an entry to **`CHANGELOG.md`** for every meaningful change (new feature, behaviour change, notable fix). Group under `### Features`, `### Refactors`, `### Tests`, or `### Fixes` inside the `## [Unreleased]` section. Trivial changes (formatting, typos, test-only fixes) do not need an entry.
- Avoid `ImplicitlyCopyable` on array and scalar types. Copies should be explicit (`.copy()`) so ownership is always visible at the call site.
- **`.as_<type>()` returns a reference** (`ref self` + `-> ref[self._data[]] T`) — zero-cost borrow tied to the heap allocation inside the `ArcPointer`, with no ownership transfer. Callers use `ref x = val.as_type()` to borrow or `.copy()` to take ownership explicitly.
- **`.to_<type>()` transfers ownership** — use this name for methods that convert a value to a new type or allocate a new representation (e.g. `.to_python_object()`, `.to_device()`, `.to_host()`).
- **Keep the `marrow.aot`/`marrow.expr` layers small-binary** — preserve the closed-erasure/DCE property (no open dispatchers, fused-only value boxes, closed per-dtype kernels) and gate changes on `benchmarks/binary_size/` (`pixi run binary_size`).
- **The box is the erasure boundary — a node never needs an erased variant.** In
  `marrow.expr`, `DynValue` erases; the nodes do not. Before adding a `Dyn*`
  node, check the existing one: either its type is known where it is constructed
  (a literal, a cast target — resolve a runtime dtype with `dispatch_*` and box
  each arm), or it does not depend on the type at all (a column read by name).
  `DynColumn`/`DynLiteral`/`DynCast` were all added and all removed for this
  reason. `DynValue` conforms to every value family, so the fused nodes take it
  as an operand with no bound relaxed; a node keys off `comptime IsErased`
  (propagated, not defaulted) to pick dispatch over fusion.
- Try to use existing building blocks instead of reimplementing them from scratch. Like do not have a handwritten loop to bitwise and/or bitmaps when bitmaps do support bitwise operations using idiomatic API.

### Prior Art — Consult C++ and Rust Implementations First

Before adding a new feature, kernel, or test suite, inspect the reference implementations in the sibling repositories:

- **Arrow C++**: `../arrow/cpp` — the canonical implementation; use it for algorithmic details, edge cases, and test vectors. If not available locally, use `https://github.com/apache/arrow`.
- **Arrow Rust** (`arrow-rs`): `../arrow-rs/` — often has cleaner, more modern API design; useful for naming and API shape. If not available locally, use `https://github.com/apache/arrow-rs`.

This applies especially to: new kernels, array type behaviour, validity/null handling, offset semantics, and test coverage.

### Testing Guidelines

When writing or modifying tests:

- **Prefer `arr[i]` over `arr.unsafe_get(i)`** for indexed element access. Use `unsafe_get` only when the explicit point of the test is to exercise the unsafe API.
- **Prefer typed shorthands** (`x.as_int32()`, `x.as_float64()`, etc.) over `x.as_primitive[Int32Type]()` when the concrete type is known — this applies to `DynArray`, `DynScalar`, and `DynBuilder`. Fall back to `x.as_primitive[T]()` only when `T` is a generic parameter.
- **Prefer typed aliases** (`Int32Array`, `Int32Builder`, `Int32Scalar`) over the generic form (`PrimitiveArray[Int32Type]`, `PrimitiveBuilder[Int32Type]`, `PrimitiveScalar[Int32Type]`) when the concrete type is known at the call site.
- **Prefer `assert_true(arr1 == arr2)`** over element-by-element loops when asserting that two typed arrays have equal contents. `PrimitiveArray[T].__eq__` returns `Bool` (structural equality), so a single `assert_true(result == expected)` replaces the whole loop.

### Kernel Implementation Pattern

Kernels in `marrow/kernels/` are implemented as typed overloads first, with a type-erased `DynArray` overload as a thin dispatch layer:

1. **Typed overloads** — one per concrete array type (`PrimitiveArray[T]`, `StringArray`, `ListArray`, etc.). These contain all the actual logic.
2. **Type-erased overload** — accepts `List[DynArray]` (or `DynArray` for unary/binary kernels), converts to the appropriate typed list/value, and delegates to the typed overload. This is the "blanket" implementation that makes kernels usable from runtime-typed code.

See `marrow/kernels/concat.mojo` and `marrow/kernels/filter.mojo` for examples.

**Dispatch on the widest family the typed leaf accepts.** A leaf bound on `PrimitiveType` takes
temporal, interval and decimal columns as-is, so it needs one `dt.dispatch_primitive[...]` arm —
not one per family, and never a reinterpret to an integer backing. Two arms differing only by
trait bound usually means the narrower bound is masking a defect rather than encoding a
constraint: `filter`/`take` had separate numeric and temporal arms that silently left decimal and
interval unsupported, hiding a SIMD gather width that computed to 0 for types wider than a
register.

## Releasing to prefix.dev

Marrow is published to [prefix.dev](https://prefix.dev/channels/marrow) as a conda package via rattler-build. The release is triggered automatically by pushing a git tag.

### Steps to cut a release

1. **Update the version in two places** — they must stay in sync:
   - `pixi.toml`: set `version = "X.Y.Z"` in the `[workspace]` table
   - `recipe/recipe.yaml`: set `version: "X.Y.Z"` under `context:`

2. **Commit the version bump:**
   ```bash
   git add pixi.toml recipe/recipe.yaml
   git commit -m "chore: bump version to X.Y.Z"
   ```

3. **Tag and push** — the `release.yml` workflow fires on `v*` tags:
   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```

The workflow will:
- Run the full test suite
- Build `marrow.mojopkg` with `pixi run package`
- Build the conda package with `rattler-build` via `pixi run -e package package-conda`
- Upload the `.conda` artifact to the `marrow` channel on prefix.dev (requires `PREFIX_API_TOKEN` secret in the repo settings)
- Create a GitHub release with auto-generated notes and both artifacts attached

### Local conda build (optional)

```bash
pixi run -e package package-conda
# output lands in output/noarch/marrow-X.Y.Z-*.conda
```

## Mojo Version Notes

Mojo is a moving target with very frequent breaking changes. On confusing compile errors, check the changelog: https://docs.modular.com/mojo/changelog/

- Use `var ^` for move semantics
- Use `deinit` for consuming parameters
- ArcPointer is used for shared ownership of buffers/bitmaps
- Many methods use `raises` for error propagation
- **Mojo resolves circular imports between modules in the same package** — do not reorganize code or move types between files to avoid circular imports; Mojo handles them correctly. **But import explicitly, never `from .x import *`.** A wildcard re-exports whatever `x` itself imported, so a name resolves or not depending on which file you entered through — the signature of three separate incidents (a trait shadowing the builtin `Scalar`; `BoolArray` resolving along one path and not another; a "fix" that took errors 2 → 10). The remaining wildcards were replaced with explicit lists; keep it that way.
- **ASAN can *hide* a heap bug rather than reveal it.** A one-byte `Variant`
  overflow (Q0.0, since fixed upstream) passed 35/35 under ASAN and failed
  without it, and a Mojo *build* failure emits no ASAN output at all — so a
  clean ASAN run is not evidence on its own. Verify without ASAN too, and
  confirm the tests actually ran rather than the build having died.

### Associated-type & trait gotchas (learned the hard way)

These are non-obvious compiler behaviours; each below is reproduced by a minimal
example. They mostly bite generic trait hierarchies (e.g. `marrow.expr.values`).

- **A *chained* associated-type projection (`Self.OutType.ArrayType`) does not
  reduce at a call/return site**, even when the concrete type is statically
  determined. A *single* projection off a **direct trait-bound parameter**
  (`T.ArrayType` for `T: SomeTrait`) *does* reduce. So to return / consume a
  concrete companion type without a `rebind` or a reducer helper, expose it as a
  **direct associated member** of the trait (e.g. `Value.ArrayType`, which each
  node fixes concretely), not as `Self.OtherAssoc.Member`.
- **An associated-type *default* that references a sibling associated type**
  (`comptime ArrayType = Foo[Self.OutType]` in a trait or sub-trait), together
  with a method returning it (`def execute -> Self.ArrayType`), errors with
  **`attempt to resolve a recursive reference to declaration 'execute'`**. A
  *concrete* default with no `Self.X` reference (e.g. `= BoolArray`) does not
  recurse. Declare such members **per concrete struct** instead.
- **Re-defaulting a base trait's abstract method in a sub-trait recurses** if that
  method returns `Self.ArrayType` and a conforming node's `ArrayType` transitively
  references another trait-member child (the compiler loops elaborating the child's
  own copy of that method). The trigger is *the abstract-then-re-defaulted method
  specifically* — a **differently named** sibling method with the same body and
  return type does **not** recurse. Fix: keep the base method abstract, have each
  node override it with a one-liner that delegates to a helper method under a new
  name (see `NumericValue.execute` abstract → `NumericValue._fused`).
- **A family-trait-provided associated-type default may not satisfy the base
  trait's abstract requirement** for conforming structs (you'll see `does not
  implement all requirements for <BaseTrait>`) — declare the member on each
  concrete struct even if a parent trait "provides" it.
- **A comptime *conditional type* is usable as a type, but carries no trait
  conformance and does not reduce at a return site** — not even inside a
  `comptime if` that has already selected the branch. `comptime C[To, V] = V
  if (V.OutType.native == To.native) else Wrapper[To, V]` resolves fine as an
  annotation, but a function returning `C[To, V]` cannot return either branch:
  `cannot implicitly convert 'V' value to 'V if (…) else Wrapper[To, V]'`. The
  usual `rebind` escape hatch does **not** help — `rebind[C[To, V]](x)` fails
  with `value of type '<the conditional>' cannot be implicitly copied, it does
  not conform to 'ImplicitlyCopyable'`, because an unreduced conditional
  conforms to nothing at all. Note `promote[L, R]` (`expr/values.mojo`) is the
  same shape and works — the difference is that it is only ever *used* as a
  type annotation, never returned from a function. Consequence: "wrap this
  operand only when it needs converting" is not expressible; either always
  wrap or do the selection somewhere the concrete type is known. This blocked
  Q0.4's promote-at-construction design.

# How to identify leaky abstractions

Analyze the dependency relations and responsobilities of each type in the requested marrow packages, like marrow.expr and marrow.kernels which are tightly coupled. Then generate a couple word summary of what each type does and what its responsibility, if we cannot identify a single responsibility that highlights a leaky abstraction. Regarding the dependencies we need a clear one directional directed tree without any cycles aggregate this knowledge then get back to me with your findings.
