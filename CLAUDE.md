# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Marrow is an implementation of Apache Arrow in Mojo. Apache Arrow is a cross-language development platform for in-memory data with a standardized columnar memory format. This implementation is in early/experimental stages as Mojo itself is under heavy development.

For information about the Mojo programming language and the standard library see https://github.com/modular/modular

## Build System & Commands

This project uses **pixi** as the package manager. All commands are run through pixi:

```bash
# Run all tests
pixi run test

# Format code
pixi run fmt

# Build package
pixi run package
```

### Running Individual Tests

To run tests for a specific module:
```bash
mojo test marrow/tests/test_dtypes.mojo -I .
mojo test marrow/arrays/tests/test_primitive.mojo -I .
```

The `-I .` flag is important as it adds the current directory to the import path.

## Core Architecture

### Type-Erased Containers

Mojo lacks dynamic dispatch, so the codebase uses **type-erased containers** with **implicit conversions** to/from typed wrappers. Implicit conversions are cheap (O(1) ref-count bumps).

#### Arrays (`marrow/arrays.mojo`)

- **`Array`** - Trait that all typed arrays implement. Provides the common read-only interface: `type()`, `null_count()`, `is_valid()`, `as_any()`. Also extends `Sized`, `Writable`, `Equatable`, `Copyable`, `Movable`.
- **`AnyArray`** - Type-erased, immutable array container (analogous to `ArrayData` in C++ Arrow). Holds `dtype`, `length`, `nulls`, `bitmap`, `buffers`, `children`, `offset`. Copying is O(1) via `ArcPointer` ref-counting inside `Buffer`/`Bitmap`.
- **Typed arrays** implement the `Array` trait and convert implicitly to/from `AnyArray`:
  - `PrimitiveArray[T]` - numeric/boolean types
  - `StringArray` - UTF-8 strings
  - `ListArray` - variable-length nested lists
  - `FixedSizeListArray` - fixed-size nested lists (e.g. embedding vectors)
  - `StructArray` - nested structs
  - `ChunkedArray` - array split across multiple chunks (does NOT implement `Array` trait)

Usage: `var arr: AnyArray = my_primitive_array` and `var prim: PrimitiveArray[int64] = some_array` both work transparently.

#### Builders (`marrow/builders.mojo`)

- **`BuilderData`** - Internal mutable builder state, always accessed through `ArcPointer[BuilderData]` for shared ownership.
- **`Builder`** - Type-erased builder wrapping `ArcPointer[BuilderData]`. Can be constructed from a `DataType` at runtime. `finish()` returns `AnyArray`.
- **Typed builders** convert implicitly to `Builder` by cloning the `ArcPointer`, so the original typed builder remains usable after passing to a composite builder:
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

**Bitmap** (`marrow/bitmap.mojo`):
- Immutable, bit-packed validity buffer wrapping a `Buffer`
- Copying is O(1) (ref-count bump)
- `BitmapBuilder` is the mutable counterpart; `finish()` transfers to immutable `Bitmap`

**DataType** (`marrow/dtypes.mojo`):
- Struct-based type system matching Arrow specification
- Supports primitive types (bool, int8-64, uint8-64, float32/64)
- Nested types via `list_(DataType)`, `fixed_size_list_(DataType, size)`, and `struct_(Field, ...)`
- Uses `code` field for type identification and optional `native` field for DType mapping

**Visitor** (`marrow/visitor.mojo`):
- `DataTypeVisitor` - runtime dispatch based on `DataType` value
- `ArrayVisitor` - runtime dispatch from `Array` to typed array overloads

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
├── dtypes.mojo           # Type system (DataType, Field)
├── buffers.mojo          # Memory management (Buffer[mut], Allocation)
├── bitmap.mojo           # Bitmap, BitmapBuilder
├── arrays.mojo           # Array, PrimitiveArray, StringArray, ListArray,
│                         # FixedSizeListArray, StructArray, ChunkedArray
├── builders.mojo         # Builder, BuilderData, PrimitiveBuilder, StringBuilder,
│                         # ListBuilder, FixedSizeListBuilder, StructBuilder
├── kernels/
│   ├── arithmetic.mojo   # Element-wise add, subtract, multiply, divide
│   ├── similarity.mojo   # Batch cosine similarity (CPU SIMD + GPU dispatch)
│   ├── aggregate.mojo    # Sum, mean, min, max
│   ├── boolean.mojo      # Logical operations
│   ├── filter.mojo       # Array filtering
│   ├── sum.mojo          # Specialized sum kernel
│   └── tests/            # Benchmarks and GPU tests
├── c_data.mojo           # Arrow C Data Interface
├── schema.mojo           # Schema with Fields and metadata
├── tabular.mojo          # RecordBatch
├── visitor.mojo          # DataTypeVisitor, ArrayVisitor traits
└── tests/                # Core module tests
python/                   # The Python module top level
```

## Implementation Patterns

### Type Constraints

Mojo lacks dynamic dispatch, so the codebase uses:
- Type-erased containers (`AnyArray`, `AnyBuilder`) with implicit conversions to/from typed wrappers
- Compile-time parameterization (`PrimitiveArray[int64]`)
- Visitor pattern (`ArrayVisitor`, `DataTypeVisitor`) for runtime type dispatch
- Runtime type checking via `DataType.code` comparison

## GPU Compute

### Architecture

GPU kernels live in `marrow/kernels/` and are imported lazily from CPU-side modules (e.g. `similarity.mojo`, `arithmetic.mojo`) only when a `DeviceContext` is passed. This avoids requiring GPU compilation tools for CPU-only usage.

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
4. **Coverage**: Only bool, numeric, string, list, fixed-size list, struct types implemented
5. **Table**: Not yet implemented (RecordBatch is available)

## Dependencies

- Mojo `<1.0.0` (nightly builds from conda-forge and modular channels)
- PyArrow `>=19.0.1, <21` (for testing and C Data Interface validation)

## Coding Guidelines

- **Always use `def` for function definitions, never `fn`.** The `fn` keyword is deprecated in Mojo in favour of `def`. All functions, methods, and trait requirements must use `def`.
- **Never call `_underscore_prefixed` methods outside of the type/struct that defines them.** They are private implementation details. Use the public factory methods and APIs instead (e.g. use `Buffer.alloc_uninit[T](n)` directly rather than computing `Buffer._aligned_size[T](n)` and passing bytes manually).
- **Do not use `PrimitiveArray[bool_]` or `as_primitive[bool_]()`.** Boolean arrays are bit-packed and require `BoolArray` for correct values access. Use `BoolArray` and `as_bool()` directly everywhere booleans are handled.
- Prefer explicit `if/else` over early-return `if + return` guard clauses. Keep the control flow flat and readable with `if/else` branches.
- Prefer PyArrow's API naming everywhere — both in the Mojo core types and in the Python bindings. When in doubt, match PyArrow's method names and signatures.
- Use **conventional commits** for all commit messages (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`, etc.), with an optional scope in parentheses (e.g. `feat(kernels): add concat`).
- Add an entry to **`CHANGELOG.md`** for every meaningful change (new feature, behaviour change, notable fix). Group under `### Features`, `### Refactors`, `### Tests`, or `### Fixes` inside the `## [Unreleased]` section. Trivial changes (formatting, typos, test-only fixes) do not need an entry.
- Avoid `ImplicitlyCopyable` on array and scalar types. Copies should be explicit (`.copy()`) so ownership is always visible at the call site.
- **`.as_<type>()` returns a reference** (`ref self` + `-> ref[self._data[]] T`) — zero-cost borrow tied to the heap allocation inside the `ArcPointer`, with no ownership transfer. Callers use `ref x = val.as_type()` to borrow or `.copy()` to take ownership explicitly.
- **`.to_<type>()` transfers ownership** — use this name for methods that convert a value to a new type or allocate a new representation (e.g. `.to_python_object()`, `.to_device()`, `.to_host()`).

### Prior Art — Consult C++ and Rust Implementations First

Before adding a new feature, kernel, or test suite, inspect the reference implementations in the sibling repositories:

- **Arrow C++**: `../arrow/cpp` — the canonical implementation; use it for algorithmic details, edge cases, and test vectors. If not available locally, use `https://github.com/apache/arrow`.
- **Arrow Rust** (`arrow-rs`): `../arrow-rs/` — often has cleaner, more modern API design; useful for naming and API shape. If not available locally, use `https://github.com/apache/arrow-rs`.

This applies especially to: new kernels, array type behaviour, validity/null handling, offset semantics, and test coverage.

### Testing Guidelines

When writing or modifying tests:

- **Prefer `arr[i]` over `arr.unsafe_get(i)`** for indexed element access. Use `unsafe_get` only when the explicit point of the test is to exercise the unsafe API.
- **Prefer `x.as_primitive[T]()`** over `PrimitiveArray[T](data=x)` for obtaining a typed view of a type-erased `AnyArray`.
- **Prefer `assert_true(arr1 == arr2)`** over element-by-element loops when asserting that two typed arrays have equal contents. `PrimitiveArray[T].__eq__` returns `Bool` (structural equality), so a single `assert_true(result == expected)` replaces the whole loop.

### Kernel Implementation Pattern

Kernels in `marrow/kernels/` are implemented as typed overloads first, with a type-erased `AnyArray` overload as a thin dispatch layer:

1. **Typed overloads** — one per concrete array type (`PrimitiveArray[T]`, `StringArray`, `ListArray`, etc.). These contain all the actual logic.
2. **Type-erased overload** — accepts `List[AnyArray]` (or `AnyArray` for unary/binary kernels), converts to the appropriate typed list/value, and delegates to the typed overload. This is the "blanket" implementation that makes kernels usable from runtime-typed code.

See `marrow/kernels/concat.mojo` and `marrow/kernels/filter.mojo` for examples.

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
- **Mojo resolves circular imports between modules in the same package** — do not reorganize code or move types between files to avoid circular imports; Mojo handles them correctly
