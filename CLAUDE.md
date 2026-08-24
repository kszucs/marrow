# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**Don't be sloppy — be precise and pedantic.**

## Project Overview

Marrow is an implementation of Apache Arrow in Mojo: a columnar in-memory
format plus kernels, an expression/relational layer, Parquet and IPC readers and
writers, and Python bindings. Both Arrow and Mojo are moving targets, so the
implementation is experimental.

For the Mojo language and standard library see
https://github.com/modular/modular; for confusing compile errors after an
upgrade, https://docs.modular.com/mojo/changelog/.

Dependencies (pinned in `pixi.toml`):

- `mojo >=1.1.0.dev2026082305,<2` and `max ==26.6.0.dev2026082305` — MAX is
  pinned to the matching version line, and is **load-bearing**: GPU codegen
  resolves `max.package_root`, *and* `DeviceContext`/`DeviceBuffer`/
  `HostBuffer` live in the `max.gpu.host` Mojo package while
  `sync_parallelize`/`elementwise`/`_reduce_generator` live in
  `max.algorithm.*`. `get_gpu_target` and `vectorize` stayed in `std`.
- `python >=3.14,<3.15` — Mojo nightlies are built against one CPython minor;
  bump it together with `mojo`.
- `pyarrow >=23.0.1,<24` (dev/test only) — the PyPI wheel, not conda-forge; see
  the comment in `pixi.toml` for why.
- `zstd` / `snappy` / `lz4-c` / `brotli` — Parquet page codecs, opened at runtime
  via `dlopen` (no link-time dependency).

## Build, Test, Benchmark

**pixi** is the package manager; commands are scoped to environments.

| Environment | Purpose | Key command |
|-------------|---------|-------------|
| `dev`         | Tests + formatting (the default) | `pixi run -e dev test` |
| `asan`        | AddressSanitizer test runs | `pixi run -e asan test_asan_core` |
| `bench`       | Benchmarks (polars, duckdb for comparison) | `pixi run -e bench bench` |
| `format`      | Formatting only (no test deps) | `pixi run -e format fmt` |
| `docs`        | Quarto documentation | `pixi run -e docs docs` |
| `examples`    | Runnable examples | `pixi run -e examples datafusion_udf` |
| `integration` | Arrow conformance via archery | `pixi run -e integration integration` |
| `wheel`       | Local macOS wheel build | `pixi run -e wheel wheel` |

```bash
pixi run -e dev test         # everything (pytest -v)
pixi run -e dev fmt          # mojo format + ruff format
pixi run package             # package/marrow.mojoc
pixi run binary_size         # AOT/hybrid/runtime binary-size gate
```

### While the tree does not compile — `precompile`

When the tree does not build — e.g. after a Mojo upgrade — do **not** iterate
with `pytest`: it has to elaborate all of marrow before it reports anything.

```bash
pixi run -e dev precompile   # everything under marrow/, ~18 s
```

`precompile` is `mojo precompile marrow -o .precompile/marrow.mojoc`. It
compiles every module under `marrow/` — tests and benches included, since they
live in `marrow/**/tests/` and are ordinary importable modules — so it catches
errors no selected test happens to import and surfaces **all** errors and
warnings in one pass. `mojo precompile` rejects `-D`, so it always builds the
CPU-only configuration.

A single test file **cannot** be compiled on its own: with no `main()` there is
nothing to build (`mojo build` reports `module does not contain a 'main'
function`). Run it through `pytest`, which generates a driver for the selection.

**The library must stay warning-clean.** Keep `mojo precompile marrow` at 0
errors and 0 warnings rather than letting output accumulate until it is
unreadable. Building is not passing: switch back to `pytest` once it compiles.

### Running tests

Always go through `pytest` — never `mojo test`, `mojo run`, or a hand-written
driver. The harness owns driver generation, case selection, flags, and JSON
parsing.

```bash
pixi run -e dev pytest marrow/tests/test_dtypes.mojo                  # one file
pixi run -e dev pytest marrow/tests/test_arrays.mojo::test_primitive_slice
pixi run -e dev pytest -v marrow/kernels/tests/test_join.mojo         # PASS/FAIL per case
```

Options: `--mojo` / `--python` / `--cpu` / `--gpu` and their `--no-*` inverses
select suites (GPU needs a Metal/CUDA device); `--benchmark` includes
`bench_*.mojo` and enables `-O3`; `--asan` runs under AddressSanitizer (needs the
`asan` environment); `--competition` prints a side-by-side comparison table after
benchmarks; `--save-benchmarks DIR` / `--benchmark-history FILE` persist results.

**Only run the tests the change could have broken.** Select the directories that
import the code you touched; the rest tells you nothing extra.

```bash
pixi run -e dev pytest marrow/exprold/tests marrow/kernels/tests   # after editing expr + kernels
pixi run -e dev pytest marrow/kernels/tests/test_groupby.mojo   # narrower still
```

`python/marrow/libmarrow.so` is rebuilt automatically by `conftest.py` before any
session that runs Python tests — no manual `build_python` step needed.

**Only `pytest` rebuilds it.** `pixi run -e dev python some_script.py` does
**not** — it imports whatever `.so` is on disk. So the obvious way to measure a
change (`git checkout` the old tree, run a script, `git checkout` back, run it
again) measures the *new* library twice and reports no difference. This has
produced a confident "no improvement, revert it" reading on a change that was
in fact a 20x win, and separately made a fixed bug look unfixed. Rebuild
explicitly with `pixi run build_python` between variants, or drive the
comparison through `pytest`.

### One selection = one compilation unit

Compilation dominates, and almost all of it is elaborating **marrow**, not the
test bodies. The harness generates **one driver** for the whole selection
(`.test_runners/_test_driver_<hash>.mojo`, content-addressed) and compiles that.
N files in one unit cost about what 1 file costs — measured on
`marrow/exprold/tests`: all 9 files together took 4 min 43 s / 17.0 GB peak, while
`test_join.mojo` alone took 3 min 18 s / 19.6 GB.

- **Selecting fewer *files* saves time; selecting fewer *cases* does not.**
- **A compile error fails every case in the run**, since there is one unit.
- **A compiler *crash* is different** — the harness halves the selection and
  retries down to a single case, so the offending case reports the crash as its
  own failure. Ordinary `error:` output never splits.
- **Peak memory scales with the unit.** Narrow the selection if memory is tight.
- **`bench_*.mojo` builds at `-O3` in its own driver**, `test_*.mojo` at `-O1`
  with `-D ASSERT=all`; they cannot share a unit. `-O0` is not an option — the
  masked-gather intrinsic in `filter`/`take` fails to lower.
- `pytest-xdist` does not help *within* a run. Several concurrent `pytest`
  invocations are safe — drivers are content-addressed.

### Writing Mojo tests

Test files live next to the code they cover (`marrow/tests/`,
`marrow/kernels/tests/`, …) and are **plain importable modules** — no `main()`,
no `TestSuite` import. The harness generates the runner.

```mojo
from std.testing import assert_true
from ..dtypes import int64          # relative: absolute `marrow.x` imports
                                    # break when compiled as part of the package

def test_something() raises:
    assert_true(1 + 1 == 2)
```

- **No `def main()`** — `mojo precompile marrow` rejects `main()` inside a
  package, and the generated driver supplies the only one. A standalone program
  belongs in `benchmarks/` instead (see `benchmarks/profiles/`).
- **Relative imports only** for `marrow.*` — `..x` from `marrow/tests/`, `...x`
  from `marrow/<sub>/tests/`. Absolute `from marrow.x import` fails with
  `unable to locate module 'marrow'` when compiled as part of the package.
- **Case names must be unique across the entire suite**, not just per file — the
  runner reports by name, and that is how results map back to pytest items.
- Each tests directory needs an `__init__.mojo`; `marrow/` is a package, so its
  subdirectories must be declared packages to be importable.

### Writing Mojo benchmarks

`bench_*.mojo` files sit beside their tests, follow the same rules (no `main()`,
relative imports), and use `Benchmark` from `marrow.utils.testing`:

```mojo
from ...utils.testing import Benchmark      # `..` from marrow/tests/
from std.benchmark import BenchMetric, keep

def bench_my_kernel(mut b: Benchmark) raises:
    var data = _prepare_data(N)
    b.throughput(BenchMetric.elements, N)
    @always_inline
    def call() {imm}:
        keep(my_kernel(data))
    b.iter(call)
    keep(data)  # prevent ASAP destruction — see below
```

**`call` is a *unified closure passed by value*, so it carries an explicit
capture list.** The list sits after the effects and before the return arrow:
`def call() raises {imm}:`, `def call() {mut builder, imm}:`. `{imm}` is the
default — it borrows every free outer name immutably, which is what a read-only
benchmark body wants. A body that *mutates* outer state must name it
(`{mut builder, imm}`); under a bare `{imm}` the compiler rejects the write
rather than silently freezing it. Never reach for `@__parameter` to avoid
writing the list.

**`keep(data)` after `b.iter(call)` is mandatory for non-trivial captures.**
ASAP destruction can free a captured value after the closure is registered and
before it runs — a heap-use-after-free inside the iteration loop.

Three rules, each of which has cost a wrong conclusion:

- **Read the unit on every row.** pytest-benchmark scales each benchmark
  independently, so adjacent rows may be ns / us / ms. Strip colour and read the
  headers: `sed 's/\x1b\[[0-9;]*m//g' out.log | grep -E "Name \(time|^bench_"`.
- **Normalise against untouched benchmarks before attributing a delta.** This
  machine drifts up to ±8% per case and ~±5% per batch, so a whole run can read
  as a uniform regression. Always include rows the change cannot touch and
  subtract their median.
- **An `assignment to 'x' was never used` warning on a value the closure *does*
  use means the capture was not made** — the body reads garbage. Never report a
  number from a run that emitted it.

For multiple sizes, share a helper and add one thin wrapper per size:

```mojo
def _bench_kernel(mut b: Benchmark, n: Int) raises: ...

def bench_kernel_10k(mut b: Benchmark) raises: _bench_kernel(b, 10_000)
def bench_kernel_100k(mut b: Benchmark) raises: _bench_kernel(b, 100_000)
def bench_kernel_1m(mut b: Benchmark) raises: _bench_kernel(b, 1_000_000)
```

## Architecture

### Type erasure

Mojo has no dynamic dispatch, so the codebase leans on three things: **type-erased
containers** with **implicit conversions** to and from typed wrappers, **comptime
parameterization** (`PrimitiveArray[Int64Type]`), and **runtime → comptime
dispatch** off `DataType.code`. Erasure is cheap: every typed value holds its data
behind ref-counted `Buffer`/`Bitmap` handles, so a conversion is O(1) ref-count
bumps.

`var arr: DynArray = my_primitive_array` and
`var prim: PrimitiveArray[Int64Type] = some_array` both work transparently.

**The erased containers do not conform to the traits they erase.** `DynArray`,
`DynScalar`, `DynBuilder` and `DynType` expose the same surface as `Array`,
`ArrowScalar`, `Builder` and `DataType`, but as their own API — they are not
substitutable for a typed value in generic code, and nothing in the tree asks
them to be: every `[T: Array]`-style bound lives inside a box's own `_dispatch`.
They did conform until the `Dyn*` conformance removal; the four were held up
only by each other's companion members (`ArrayType` → `ScalarType`,
`Value.OutType` → `DynType`), and dropping them changed no behaviour and no
binary size. Keep it that way: a box may *hold* trait-bound values, it should
not *be* one.

The exception is `DynValue: Value`, and it earns it — `Value` is a trait of
runtime methods plus `OutShape`, which `DynValue` answers truthfully, and the
conformance is what lets a runtime leaf be an operand of the fused
`NullPredicate`/`IsIn`/`WindowFunction` nodes.

### Arrays, builders, scalars

**Arrays** (`marrow/arrays.mojo`):

- **`Array`** — the trait every typed array implements: `type()`, `null_count()`,
  `is_valid()` / `is_null()`, `to_dyn()`, `to_data()`, `slice()`, plus
  `to_device()` / `to_cpu()`. Extends `Sized`, `Writable`, `Equatable`,
  `Copyable`, `Movable`, and fixes an associated `ScalarType`.
- **`DynArray`** — the type-erased, immutable handle, backed by an inline
  `Variant`. Runtime dispatch iterates the variant members at comptime and
  selects the active one with `isa[T]()` — no `rebind` casts, no function-pointer
  trampolines. Copies are O(1).
- **`ArrayData`** — the generic flat layout (dtype, length, nulls, bitmap,
  buffers, children, offset), produced *on demand* by `to_data()` for interop
  (C Data Interface, nested array construction). It is **not** stored inside
  `DynArray`.
- Concrete types: `NullArray`, `BoolArray` (bit-packed), `PrimitiveArray[T]`,
  `BinaryLikeArray[T]`, `ListLikeArray[T]`, `FixedSizeListArray`,
  `FixedSizeBinaryArray`, `StructArray`, `DictionaryArray`, and `ChunkedArray`
  (multiple chunks; does **not** implement `Array`).
- Aliases fix the parameter: `StringArray`/`LargeStringArray`/`BinaryArray`/
  `LargeBinaryArray` over `BinaryLikeArray`; `ListArray`/`LargeListArray`/
  `MapArray` over `ListLikeArray`; `Int32Array`, `Float64Array`, `TimestampArray`,
  `Decimal128Array`, … over `PrimitiveArray`.

**Builders** (`marrow/builders.mojo`): `Builder` is the trait, `DynBuilder` the
type-erased builder — constructible from a runtime `DataType`, `finish()` returns
`DynArray`. Typed builders convert implicitly to `DynBuilder` by cloning the
`ArcPointer`, so the original stays usable after being handed to a composite
builder. Same alias scheme (`StringBuilder`, `ListBuilder`, `Int32Builder`, …).

**Scalars** (`marrow/scalars.mojo`): the trait is **`ArrowScalar`** (not `Scalar`
— that name is the builtin), erased by `DynScalar`. `NullScalar`, `BoolScalar`,
`PrimitiveScalar[T]`, `StringScalar`, `FixedSizeBinaryScalar`, `ListScalar`,
`StructScalar`, `DictionaryScalar`, plus the usual aliases.

### Buffers, bitmaps, views

**`Buffer`** (`marrow/buffers.mojo`):

- `Buffer[mut=False]` — immutable, ref-counted via `ArcPointer[Allocation]`.
  `Buffer[mut=True]` is the mutable counterpart (it replaced `BufferBuilder`);
  `finish()` freezes it.
- Allocation kinds: CPU (owned heap), FOREIGN (external, with release callback),
  HOST (pinned GPU host memory), DEVICE (GPU memory). `is_cpu()` / `is_host()` /
  `is_device()` forward to `Allocation`.
- All buffers are 64-byte aligned, and their **size** is rounded up to a
  multiple of 64 — `Buffer._aligned_size` is `align_up(bytes, 64)`, the same
  rule as Arrow C++'s `PoolBuffer::RoundCapacity`. **That is not slack.** When
  the logical byte count is already a multiple of 64 the allocation ends at the
  last live byte, so nothing may read or write past a buffer's logical end
  "because Arrow buffers are padded". A one-element overrun on exactly that
  boundary corrupted tcmalloc's freelist and cost ~4 hours to trace; the
  bounds are now enforced by `debug_assert` on every `BufferView` /
  `BitmapView` write path. FOREIGN (imported) buffers carry no padding
  guarantee at all — the C Data Interface spec makes even alignment
  "recommended, but not required". See
  `docs/alpha-findings/g1-buffer-invariants.md`.

**`Bitmap`** (also `marrow/buffers.mojo` — there is no `bitmap.mojo`): bit-packed
validity wrapping a `Buffer`, same `mut` pair, same O(1) copies.

**`BufferView` / `BitmapView`** (`marrow/views.mojo`): borrowed, offset-applied
spans. These are what kernels compute over; `views.mojo` also owns `apply` and
`_apply_dispatch`, the CPU/GPU element-wise driver.

Rules:

- Prefer `Buffer`/`Bitmap` for owned values and `BufferView`/`BitmapView` for
  computation. No naked pointer arithmetic in kernel or array code.
- **`unsafe_ptr()` is restricted to `buffers.mojo`, `views.mojo`,
  `c_data.mojo`, `utils/byteorder.mojo` and the Parquet codec layer**
  (`parquet/reader.mojo`, `parquet/codecs.mojo`, which `dlopen` the C codecs
  and hand them raw pointers). Everything else goes through the view
  abstractions. `utils/byteorder.mojo` joined the list on 2026-08-17:
  `LittleEndian.fixed` is *the* byte-order primitive, and confining the
  unaligned wide load to it is what keeps raw pointers out of every decoder
  that reads a scalar. It had been copying `W` bytes into an `Array` one at a
  time and calling `SIMD.from_bytes` — about 8 loads and a stack temporary per
  64-bit read, which cost 14-38x on the hash kernel's string path and taxed
  every Parquet and IPC decode. `marrow/parquet/utils.mojo` left the list
  because it moved to `marrow/utils/compression.mojo`.
- **Avoid `AnyOrigin` types (`MutAnyOrigin`, `ImmutAnyOrigin`) and
  `unsafe_origin_cast`.** Use parametric origins (`out_o: Origin[mut=True]`,
  `src_o: Origin[mut=False]`) and pass views directly.
- Prefer `.values()` over `.buffer.view[native](array.offset)` —
  `PrimitiveArray[T].values()` and `BoolArray.values()` return a properly offset
  view in one call. Reach for `.buffer.view[native]()` only inside `buffers.mojo`
  or when constructing a view with parameters `.values()` does not cover.

### Types and runtime → comptime dispatch

`marrow/dtypes.mojo` holds a struct-based type system mirroring the Arrow spec:
`DataType` is the trait, refined by `PrimitiveType` → `NumericType` →
`IntegerType`/`FloatingType`, plus `TemporalType`, `IntervalType`, `DecimalType`,
`BinaryLikeType` → `StringLikeType`, and `ListLikeType`. A type is identified by
its `code` field and carries an optional `native` `DType`. Nested types are built
with `list_(DataType)`, `fixed_size_list_(DataType, size)`, `struct_(Field, ...)`.

There is **no visitor module**. Runtime dispatch is the `DynType.dispatch_*`
family — `dispatch_primitive` / `dispatch_numeric` / `dispatch_integer` /
`dispatch_floating` / `dispatch_temporal` / `dispatch_decimal` /
`dispatch_stringlike` / `dispatch_binarylike` / `dispatch_listlike` — each
resolving a runtime `DataType` to a comptime type parameter and running a job
passed **as a value**: `dt.dispatch_numeric(job)`, not `dt.dispatch_numeric[job]()`.

Each of the nine writes out its own `comptime for` over `DynType.VariantType.Ts`,
guarded by `comptime if conforms_to(T, Family)`, and calls `func` directly.
`DynArray`, `DynScalar` and `DynBuilder` each have one `_dispatch` of the same
shape at their own trait.

**This duplication is deliberate and measured.** Factoring the ladder into one
generic `variant_dispatch(v, func)` helper needs a narrowing closure between the
caller and the loop — a closure type cannot be generic over its own trait bound,
so the helper must bind `func` on `Movable` and let the caller narrow. That
adapter is inlined into *every* arm of *every* instantiation: it cost
**+662,740 bytes (+31.9% of `__text`)** on `query_streaming_agg_fused`, and
removing it is what recovered the regression. Prefer the local ladder; do not
reintroduce the shared helper.

### Kernels

Kernels (`marrow/kernels/`) are **typed first**, with a type-erased `DynArray`
layer on top:

1. **Typed overloads** — one per concrete array type (`PrimitiveArray[T]`,
   `StringArray`, `ListArray`, …). All the logic lives here.
2. **Type-erased overload** — takes `DynArray` (or `List[DynArray]`), converts to
   the typed form, and delegates. This is what makes a kernel usable from
   runtime-typed code.

See `marrow/kernels/concat.mojo` and `marrow/kernels/filter.mojo`.

**Dispatch on the widest family the typed leaf accepts.** A leaf bound on
`PrimitiveType` already takes temporal, interval and decimal columns, so it needs
one `dt.dispatch_primitive[...]` arm — not one per family, and never a
reinterpret to an integer backing. Two arms differing only by trait bound usually
means the narrower bound masks a defect rather than encoding a constraint:
`filter`/`take` had separate numeric and temporal arms that silently left decimal
and interval unsupported, hiding a SIMD gather width that computed to 0 for types
wider than a register.

### Expression layer

`marrow/exprold/` is split into **two lanes that no longer share node types**:

- **The AOT lane** (`values.mojo`, `aggregates.mojo`) — every node's operands are
  bound on a family trait (`L: NumericValue`), its output dtype is a comptime
  type, and a subtree fuses into one SIMD loop. Nothing is erased.
- **The runtime lane** (`dynamic.mojo`) — `DynValue` is one struct holding its
  children, an optional payload, and a pointer to the evaluating function. What
  stays runtime is the *dtype* of the operands, not the operation.

`relations.mojo` holds the plan IR: `Relation` nodes are pure, immutable
descriptions (`kind`/`schema`/`to_processor`), erased by `DynRelation` behind an
`ArcPointer` so copying a plan is an O(1) share. `execution.mojo` is the
executing counterpart: `Relation.to_processor(ctx)` builds a `Processor` owning
all mutable state (scan offset, hash index, grouper, children), erased by
`DynProcessor`, drained by `collect()`. `pruning.mojo` does conservative
statistics-based predicate pruning for row groups and pages.

Two standing constraints:

- **Keep `marrow.expr` small-binary — for the *comptime* lane.** Preserve the
  closed-erasure/DCE property (no open dispatchers, fused-only value boxes,
  closed per-dtype kernels) and gate changes on `benchmarks/binary_size/`
  (`pixi run binary_size`).

  **The constraint is about the AOT lane, not the runtime one.** A program
  built from `col("a", int64)` and the fused nodes is a size-critical AOT
  binary and must not pay for kernels it never names. A program that builds
  expressions *at run time* has already accepted an interpreter — it cannot
  know its kernels at compile time, and a frontend that constructs queries
  dynamically will reach most of them anyway. So `runtime/values.mojo`
  interprets by switching on `_tag`, and that is deliberate: it costs size only
  in binaries that use the runtime lane at all, and the comptime lane never
  reaches it. Do not reintroduce a per-node function pointer to shave that
  cost — the previous one put a thin fn field in a self-referential struct and
  the compiler miscompiled it (see `docs/backlog.md`).
- **The box is the erasure boundary — a node never needs an erased variant.**
  `DynValue` erases; the nodes do not. Before adding a `Dyn*` node, check the
  existing one: either its type is known where it is constructed (a literal, a
  cast target — resolve a runtime dtype with `dispatch_*` and box each arm), or
  it does not depend on the type at all (a column read by name).
  `DynColumn`/`DynLiteral`/`DynCast` were all added and all removed for this
  reason. **`DynValue` conforms to `Value` and to nothing else** — its own docstring
  says so, and the struct declares `(Copyable, Movable, Value, Writable)`.
  This entry used to claim it "conforms to every value family", which is
  false. Fused nodes take a runtime operand because the three that accept
  one bind on `Value` itself — `IsIn[A: Value]`,
  `NullPredicate[K, A: Value]`, `WindowFunction[Func, A: Value]` — not
  because the families are satisfied; a node keys off `comptime IsErased` (propagated,
  not defaulted) to pick dispatch over fusion.

### Interop and tabular

**C Data Interface** (`marrow/c_data.mojo`): `CArrowSchema` / `CArrowArray` for
zero-copy exchange. Import via `CArrowSchema.from_pycapsule()` + `.to_dtype()`
and `CArrowArray.from_pycapsule()` + `.to_array(dtype)`; export via
`CArrowSchema.from_dtype(dtype).to_pycapsule()` and
`CArrowArray.from_array(arr).to_pycapsule()`. Python arrays expose
`__arrow_c_array__()` / `__arrow_c_schema__()` for PyArrow.

**Tabular** (`marrow/tabular.mojo`): `RecordBatch` (schema + column arrays) and
`Table` (schema + chunked columns). `marrow/schema.mojo` holds `Schema`, `Field`
and metadata. `marrow/ipc.mojo` is the Arrow IPC file/stream reader and writer.

### Directory structure

```
marrow/
├── dtypes.mojo           # DataType traits, Field, DynType.dispatch_*
├── buffers.mojo          # Buffer[mut], Bitmap[mut], Allocation
├── views.mojo            # BufferView, BitmapView, apply/_apply_dispatch
├── arrays.mojo           # Array trait, DynArray, ArrayData, typed arrays
├── builders.mojo         # Builder trait, DynBuilder, typed builders
├── scalars.mojo          # ArrowScalar trait, DynScalar, typed scalars
├── schema.mojo           # Schema, Field, metadata
├── tabular.mojo          # RecordBatch, Table
├── c_data.mojo           # Arrow C Data Interface
├── ipc.mojo              # Arrow IPC file / stream reader + writer
├── execution.mojo        # ExecutionContext — threads, device, `stripe`
├── utils/                # byteorder, checksum, hashing, compression
│   └── testing.mojo      # TestSuite + Benchmark used by the generated driver
├── kernels/
│   ├── numeric.mojo      # arithmetic + comparison kernels (Add/Sub/…/Eq/Lt/…)
│   ├── boolean.mojo      # and/or/not/xor, is_null, is_nan, is_inf
│   ├── cast.mojo         # cast() and the per-family cast kernels
│   ├── conditional.mojo  # case_when, coalesce, nullif, fill_null
│   ├── aggregate.mojo    # sum, product, min/max, count, mean, any/all
│   ├── distinct.mojo     # (approx_)count_distinct, grouped variants
│   ├── filter.mojo       # filter / take / drop_null
│   ├── sort.mojo         # sort / sort_indices
│   ├── groupby.mojo      # hash group-by
│   ├── join.mojo         # hash join
│   ├── hashtable.mojo    # SwissHashTable
│   ├── hashing.mojo      # rapidhash
│   ├── partition.mojo    # radix partitioning
│   ├── membership.mojo   # is_in
│   ├── string.mojo       # string kernels incl. LIKE/ILIKE
│   ├── temporal.mojo     # date/time field extraction, date_trunc
│   ├── nested.mojo       # array_length, array_contains
│   ├── concat.mojo       # concat
│   └── tests/            # test_*.mojo + bench_*.mojo
├── expr/
│   ├── values.mojo       # AOT lane: fused, comptime-typed value nodes
│   ├── dynamic.mojo      # runtime lane: DynValue
│   ├── aggregates.mojo   # AggFunction → resolved aggregate
│   ├── relations.mojo    # Relation / DynRelation plan IR
│   ├── execution.mojo    # Processor / DynProcessor pull engine
│   ├── pruning.mojo      # statistics-based predicate pruning
│   └── tests/
├── parquet/              # reader, writer, schema, format, codecs, bloom,
│   └── tests/            # statistics, source
└── tests/                # test_*.mojo + bench_*.mojo for the core modules
python/                   # Python package + bindings (python/marrow/libmarrow.so)
└── marrow/tests/         # Python test_*.py and bench_*.py
benchmarks/               # standalone programs (they own a `main()`, so they
                          # cannot live inside the package) + binary_size gate
docs/                     # Quarto site, design documents, and backlog.md
```

Tests and benchmarks sit **inside** the package, next to the code they cover.
That works because they carry no `main()` — see "Writing Mojo tests".

**`docs/backlog.md` is the only place that says what is open.**

---

## GPU Compute

### GPU codegen is opt-in — `-D MARROW_GPU=true`

**Every device path is compiled out by default.** `marrow.utils.GPU_ENABLED`
(`comptime GPU_ENABLED = get_defined_bool["MARROW_GPU", False]()`) is the single
switch and gates all of it: the device allocations in `kernels/cast.mojo`,
`kernels/numeric.mojo`, `kernels/hashing.mojo`, `kernels/boolean.mojo`, and the
accelerator arm of `views._apply_dispatch` — plus `has_accelerator_support`, which
answers False so a GPU `ExecutionContext` raises
`"apply: no GPU accelerator available"` at the dispatch site instead of silently
taking a CPU path.

```bash
mojo build -D MARROW_GPU=true ...      # opt in
pixi run -e dev pytest --gpu           # the harness passes it for you
```

**macOS needs the Metal toolchain installed separately.** Xcode 26 moved the
Metal compiler out of the default install, so `xcrun metal` fails with
*"cannot execute tool 'metal' due to missing Metal Toolchain"* and every GPU
test dies at compile time with `Metal Compiler failed to compile metallib.
Please submit a bug report.` — which reads like a Mojo bug and is not one. A
marrow-free three-line `elementwise` program fails identically; that is the
fastest way to tell the two apart. Fix:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Anything touching device code needs `comptime if GPU_ENABLED:` around it — a
*runtime* `if ctx.is_gpu()` alone cannot be eliminated at elaboration time.

This is the **largest single compile-time lever** in the tree: a cold `cast` +
`sort_indices` build is 43.7 s with GPU off versus 85.0 s with it on. **Both
halves must be gated or you get none of it** — the allocations need
`comptime if GPU_ENABLED` *and* `has_accelerator_support` must answer False.
Gating either alone measures as no change at all. It does not shed the
`libmax`/AsyncRT runtime dependency: a GPU-off binary still links it.

### How device execution is wired

One kernel serves both targets. Every `apply` overload writes a **lane** —
`def lane[W: Int](i: Int)` — and hands it to one of five functions in
`views.mojo`, each with a single job:

| | |
|---|---|
| `_cpu_striped` | `vectorize` handed to `ctx.stripe` (calling thread or `ctx.resolved_num_threads()` workers; thread count is owned by `ctx`) |
| `_cpu_serial` | `vectorize` on the calling thread only — what a bit-packed destination requires, since a whole-byte stride is the only thing keeping workers off each other's read-modify-write |
| `_gpu_launch` | the only caller of `elementwise`, and the only place paying for its `Coord`-shaped signature |
| `_apply_dispatch` | buffer destination: `_gpu_launch` or `_cpu_striped` |
| `_apply_packed_dispatch` | bitmap destination: padded `_gpu_launch` or `_cpu_serial` |

`gpu_ok` is the caller's `has_accelerator_support[In, Out]()` result, a comptime
`Bool` so the GPU branch is dead-code-eliminated when unsupported. A lane that
closes over host state (an expression node, a `RecordBatch`) cannot satisfy
`elementwise`'s `RegisterPassable & ImplicitlyCopyable` bound — that is exactly
the set of lanes that cannot run on a device, so those overloads call
`_cpu_striped` / `_cpu_serial` directly instead of passing a false `gpu_ok`.

Data movement is explicit and is a *kind change*, not a shadow copy: a `Buffer`
is CPU/FOREIGN/HOST **or** DEVICE.

- `Buffer.to_device(ctx)` / `.to_cpu(ctx)`, same on `Bitmap`.
- `Array.to_device(ctx)` / `.to_cpu(ctx)` — implemented for `BoolArray`,
  `PrimitiveArray[T]`, `FixedSizeListArray`, and `DynArray`; the trait default
  raises for the rest.
- Kernel results stay device-resident; call `.to_cpu(ctx)` to read on the CPU.

```mojo
var a = array([1, 2, 3, 4], int32).to_device(ctx)
var b = array([10, 20, 30, 40], int32).to_device(ctx)
var result = AddKernel.apply[Int32Type](a, b, ctx).to_cpu(ctx)
```

### Performance guidance

Measured on Apple Silicon (Metal, unified memory), from a kernel no longer in
the tree; the shape still holds. **Transfer cost dominates**: do not
GPU-accelerate low-intensity element-wise work, and never upload per call —
uploading is 2-3x slower than CPU even for compute-heavy kernels. Upload once,
run several kernels device-resident, download at the end. Crossover was ~10K
vectors at dim >= 384.

## Coding Guidelines

### Language

- **Always use `def`, never `fn`.** `fn` is deprecated — functions, methods, and
  trait requirements all use `def`.
- **Never use `alias` — use `comptime`.** `alias` is deprecated; use
  `comptime var` or `comptime` parameters wherever a compile-time value is needed.
- **Never call `_underscore_prefixed` members from outside the defining type.**
  Use the public API (e.g. `Buffer.alloc_uninit[T](n)`, not
  `Buffer._aligned_size[T](n)` plus a byte count).
- Prefer explicit `if/else` over early-return guard clauses; keep control flow
  flat.
- **Use existing building blocks.** Do not hand-roll a loop to AND/OR two bitmaps
  when `Bitmap` already exposes bitwise operations.

### Types and naming

- **Prefer the typed aliases over the parameterized spelling** when the concrete
  type is known: `Int32Array` over `PrimitiveArray[Int32Type]`, `Int32Builder`
  over `PrimitiveBuilder[Int32Type]`, `Int32Scalar` over
  `PrimitiveScalar[Int32Type]` — likewise `StringArray`, `ListArray`,
  `TimestampArray`, `Decimal128Array`, and their builder/scalar counterparts.
  The only exception is a generic parameter `T`, where `PrimitiveArray[T]` is the
  sole option.
- **Prefer the typed shorthand accessors over `.as_primitive[T]()`.** `DynArray`,
  `DynScalar` and `DynBuilder` all expose `.as_int8()` … `.as_uint64()`,
  `.as_float16()` … `.as_float64()`, `.as_bool()`, `.as_string()`, `.as_list()`.
  Mojo then infers the kernel's type parameter too — write `kernel(arr.as_int32())`,
  not `kernel[Int32Type](arr.as_primitive[Int32Type]())`. Same exception for a
  generic `T`.
- **Never use `PrimitiveArray[bool_]` or `as_primitive[bool_]()`.** Booleans are
  bit-packed: use `BoolArray` / `.as_bool()` / `BoolBuilder`.
- **Do not wrap values in explicit `DynArray(...)`, `DynScalar(...)`, or
  `DynBuilder(...)`.** All three have `@implicit` conversions from their traits:
  write `var a: DynArray = array([1, 2], int32)` and
  `ListBuilder(Int32Builder())`. The explicit call is only required when
  constructing a `DynBuilder` from a runtime `DataType` or with explicit capacity.
- **`.as_<type>()` returns a reference** (`ref self` → `ref[self._data[]] T`) — a
  zero-cost borrow tied to the allocation inside the `ArcPointer`, no ownership
  transfer. Use `ref x = val.as_int32()` to borrow, `.copy()` to own.
- **`.to_<type>()` transfers ownership** — use that name for conversions and new
  allocations (`.to_dyn()`, `.to_python_object()`, `.to_device()`, `.to_cpu()`).
- Avoid `ImplicitlyCopyable` on array and scalar types: copies stay explicit
  (`.copy()`) so ownership is visible at the call site.

### Python API

- **Follow PyArrow's design closely**, in the Mojo core types and in the bindings:
  method names, signatures and defaults on `Array`, `Scalar`, `RecordBatch` and
  `Table` should match, so PyArrow muscle memory carries over. Diverge only where
  Arrow semantics genuinely differ.
- The wrappers in `python/marrow/` use **composition** — a `_Wrapper` base with a
  `._binding` slot holding the C extension object — not inheritance. Optional
  arguments, defaults and convenience methods live in pure Python; the Mojo
  binding files stay minimal and strict: no optional args, no sugar.

### Testing

The rules above apply to tests too (typed aliases, typed shorthands, `BoolArray`).
In addition:

- **Prefer `arr[i]` over `arr.unsafe_get(i)`** — use `unsafe_get` only when
  exercising the unsafe API is the point of the test.
- **Prefer `assert_true(result == expected)`** over element-by-element loops;
  `PrimitiveArray[T].__eq__` is structural equality and replaces the whole loop.
- **Write aggregates with the fluent API** — `col("amount", int64).sum()`,
  `.mean()`, `.min()`, `.max()`, `.count()` on `NumericValue`/`StringValue`/
  `TemporalValue`, named with `.alias("total")`, and no key list at all for a
  no-`GROUP BY` aggregate (`rel.aggregate(aggs=[...])`). Never spell one
  `AggExpr.of[NumericAgg[SumKernel, Int64Type]](...)` — that constructor is
  for the kernel layer and for the benches that deliberately measure comptime
  versus runtime resolution (`bench_aggregate_aot.mojo`,
  `benchmarks/binary_size/`), not for tests.
- Use standard pytest assertions and fixtures (`tmp_path`) on the Python side.

### Process

- **Consult the reference implementations first** — before adding a kernel, an
  array behaviour, or a test suite. **Arrow C++** (`../arrow/cpp`, else
  https://github.com/apache/arrow) is canonical for algorithmic detail, edge
  cases and test vectors; **arrow-rs** (`../arrow-rs/`, else
  https://github.com/apache/arrow-rs) usually has the cleaner API shape and
  naming. This matters most for validity/null handling, offset semantics, and
  coverage.
- **Conventional commits** (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`,
  `test:`), optional scope in parens: `feat(kernels): add concat`.
- **Add a `CHANGELOG.md` entry for every meaningful change** under
  `### Features` / `### Refactors` / `### Tests` / `### Fixes` inside
  `## [Unreleased]`. Formatting, typos and test-only fixes do not need one.

## Mojo Gotchas

### General

- Use `var ^` for move semantics and `deinit` for consuming parameters. Many
  methods `raises`.
- `ArcPointer` provides shared ownership of buffers and bitmaps.
- **Mojo resolves circular imports between modules in the same package** — do not
  reorganize code or move types between files to avoid them. **But import
  explicitly, never `from .x import *`.** A wildcard re-exports whatever `x`
  itself imported, so a name resolves or not depending on which file you entered
  through — the signature of three separate incidents (a trait shadowing the
  builtin `Scalar`; `BoolArray` resolving along one path and not another; a "fix"
  that took errors 2 → 10). All wildcards were replaced with explicit lists; keep
  it that way. **`golden/prelude.mojo` is the one sanctioned exception**: it
  defines nothing and imports nothing for its own use, so its wildcard surface
  is exactly the curated list written in it, and case files are leaves with no
  second entry path for a name to resolve along. Cases import from it, never
  from `golden/helpers.mojo`, whose own imports would otherwise leak into every
  case.
- **A `List` whose element is a `Variant` where the *largest* member is not the
  *most-aligned* member loses every other element when it grows.** Elements at
  odd indices come back holding the variant's **first** member -- discriminant
  reads 0. Reserving capacity up front avoids it, which places the fault in
  `List`'s reallocation path. Reproducers: `docs/repros/`.

  **Both properties are required, and neither explains it alone.** A variant of
  96 bytes / align 32 whose largest member *is* the 32-aligned one is correct;
  an all-8-aligned variant is correct; the *same* 96/32 layout with a 72-byte
  align-8 largest member and a 64-byte align-32 member is corrupt. Identical
  size and alignment, different answer.

  In marrow this is **`DynScalar`** (96/32): `StructScalar` is 72 bytes at
  align 8, `Decimal256Scalar` is 64 bytes at align 32. Drop `Decimal256Scalar`
  and the variant is 80/8 and correct. `DynArray` (144/8) and `ArrayData`
  (152/8) are immune -- every member is pointer-backed. `RuntimeValue` is hit
  only because its payload variant holds a `DynScalar`, which is why
  `case_when` and `StructArray.__getitem__` pre-allocate.

  **It is not a memory-safety bug**: under ASAN the values are equally wrong and
  there is *no* diagnostic -- no overflow, no use-after-free. It is a miscompile
  of the move, and ASAN perturbs it (one shape passes under ASAN, fails
  without). Do not reach for ASAN to confirm this class.

  Ruled out by experiment, so do not re-derive: recursion (`ArrayData` is
  self-referential and fine), the explicit `__deinit__` (`DynArray` has an
  identical one and is fine), `IsTriviallyDeinitable` (both `False`), the
  `Writable` reflection defaults, and variant member count.

- **A recursive type must spell out *both* `write_to` and `write_repr_to`.**
  `Writable` gives both a **reflection-based default** that walks every field at
  comptime. The trait's own docstring says that for mutually recursive types —
  "struct `A` has a field of type `List[B]` and struct `B` has a field of type
  `A`" — the default "creates an infinite monomorphization cycle that causes the
  compiler to hang", and that at least one type in the cycle must override it.
  Marrow is full of exactly that shape: `DynScalar -> StructScalar ->
  List[DynScalar]`, `DynArray -> StructArray -> List[DynArray]`, `DynType ->
  DictionaryType -> OwnedPointer[DynType]`, `ArrayData -> List[ArrayData]`.
  `write_repr_to` has its own default and its own cycle, so overriding only
  `write_to` does not cover it. `ArrayData` is exempt because it is not
  `Writable` at all — the hazard needs the conformance, not just the recursion.

  **Measured 2026-08-24: it does not currently fire here, and the tree
  deliberately leaves it that way.** 26 of `dtypes.mojo`'s 28 structs override
  only `write_to` and inherit the reflection `write_repr_to`; `repr()` on a
  recursive dtype returns rather than hanging, because the walk stops at the
  `OwnedPointer`/`List` boundary. Implementing all 26 was tried and reverted —
  it bought nothing but 26 near-identical methods and made `repr()` identical
  to `str()`. The cost of leaving it is cosmetic: `repr(list_(int64))` prints
  `ListType(item=OwnedPointer[Field](...))` and never shows the element type.
  Revisit only if a genuine hang appears.

  **This entry once blamed the reflection defaults for the `DynScalar` element
  loss. That was wrong** — the cause was `Variant`'s size/stride disagreement
  (see the entry above). Adding the 26 overrides did not change the corruption
  at all.

- **An `__eq__` that compares *elements* of an erased container deadlocks the
  compiler.** Fixed 2026-08-19; recorded because the shape recurs. Comparing a
  nested array element by element materialises a `DynArray` per element, so
  `DynArray.__eq__` and the nested arrays' `__eq__` become mutually recursive
  at instantiation, and the elaborator never resolves it: ~9 s of CPU, then
  parked in `semaphore_wait_trap` at **0% CPU** forever with no diagnostic.
  That is a *deadlock*, not an infinite loop — a runaway recursion burns CPU
  and grows memory, so `%cpu=0.0` with flat RSS is the tell. It kept all 165
  cases of `test_arrays.mojo` from ever compiling.
  **Equality compares fields, never elements** — dtype, length, null count,
  validity views, buffers, children — and element-wise value comparison lives
  in `EqKernel`. Two narrower fixes were measured and neither works: narrowing
  `DynArray.__eq__` to one `isa` (O(N) arms instead of `Variant.__eq__`'s O(N²))
  and `@no_inline` on it. The cycle has to be removed, not made cheaper.
  The general lesson: **`precompile` being clean is not evidence that a test
  file will build** — it compiles the library, not the test's instantiations.
- **ASAN can *hide* a heap bug rather than reveal it.** A one-byte `Variant`
  overflow (since fixed upstream) passed 35/35 under ASAN and failed without it,
  and a Mojo *build* failure emits no ASAN output at all — so a clean ASAN run is
  not evidence on its own. Verify without ASAN too, and confirm the tests
  actually ran rather than the build having died.

### Closures

Use **unified** closures — a value with an explicit capture list
(`def body(i: Int) raises {mut out, imm} -> None`), passed as an argument.
`@__parameter` / `capturing[_]` is the legacy comptime-parameter form; the two
do not convert, so an API and its call sites flip together. Syntax and capture
conventions are in the Mojo docs.

Two project-specific traps, neither of which produces a diagnostic:

- **A closure handed to a GPU kernel must be captured by value, never `{imm}`** —
  an `imm` capture points into the host stack frame, so the device silently
  computes garbage. `views._gpu_launch` copies before capturing.
- **A closure type cannot be generic over its own trait bound.** So a shared
  dispatch loop cannot name the caller's trait; it would have to bind `func` on
  `Movable` and let the caller narrow through an extra closure. That adapter
  inlines into every arm — measured at **+662,740 bytes** on one gate — which is
  why each erased box writes its own `isa` ladder instead (`DynArray._dispatch`).

### Associated types, traits, reflection

Undocumented compiler limits, each hit here and each invalidating an approach
that looks obvious. Terse on purpose — the reproductions are in git history.

- A **chained** associated-type projection (`Self.OutType.ArrayType`) does not
  reduce at a call/return site; a single projection off a direct trait-bound
  parameter does. Expose companions as direct members (`Value.ArrayType`).
- An associated-type **default referencing a sibling associated type**, plus a
  method returning it, errors with *"recursive reference"*. Declare such members
  per concrete struct. Composing out of *type parameters'* associated types is
  fine (`Tuple[Self.L.State, Self.R.State]`) — but a raising `prepare` needs
  `comptime State: Copyable & Deinitable`.
- **Re-defaulting a base trait's abstract method in a sub-trait recurses** when
  it returns `Self.ArrayType`. Keep the base abstract and override under a new
  name (`NumericValue.execute` -> `_fused`).
- **A trait-level default method cannot return `Self.AssocType`** unless that
  type is `ImplicitlyCopyable` — and marrow's array types deliberately are not.
  So a protocol carrying non-implicitly-copyable state cannot be rolled out
  behind defaults; every conformer must implement it in the same commit.
- **A conditional comptime *associated type* reduces, and carries its trait
  bound** — provided **both branches are always well-formed**. Verified
  2026-08-21: `comptime Simplified = Self.A.Inner.Simplified if
  Self.A.KIND == 1 else Neg[Self.A.Simplified]` recurses to a fixpoint
  (`Neg^4[Leaf]` -> `Leaf`, `Neg^3[Leaf]` -> `Neg`) and its members are
  reachable. The enabling trick is **totality**: every node declares every
  structural projection, leaves answering with `Self`, so neither arm can
  name a type that does not exist — the type-level form of `bound_column`
  returning `-1`. An earlier attempt concluded "conditionality does not
  reduce"; the real blocker was an ill-formed branch. This makes a
  compile-time plan rewrite possible in principle; its compile-time and
  binary-size cost at marrow's scale is **unmeasured**.
- **A comptime conditional type *does* reduce at a return site, and *does*
  carry its trait bound — when both branches are always well-formed.** Verified
  2026-08-22: `def __getattr_param__[name: String](self) -> Column[Int64Type if
  s.codes[s.index_of(name)] == 0 else Float64Type]` compiles, satisfies
  `Column[T: NumericType]`, and gives `t.a` and `t.b` genuinely different types.
  This entry previously said the opposite — "carries no trait conformance and
  does not reduce at a return site" — which is **false**, and false for the same
  reason an earlier "conditionality does not reduce" claim about associated
  types was: the blocker in both original incidents was an **ill-formed
  branch**, not conditionality. Totality is the enabling condition, exactly as
  in the associated-type entry above. What genuinely does not work is bridging a
  conditional type to a *different* representation — `rebind` cannot, which is
  why `promote[L, R]` works and "wrap this operand only when it needs
  converting" does not.
- **A reflected field type is opaque inside the generic function that reflects
  it.** Route construction through a separately-instantiated generic bound on
  the trait — `_construct_default[D: Defaultable & DataType]()`
  (`marrow/schema.mojo:12`), which is what makes `Schema.from_struct[T]()` work.
  The same limit is why **`Table[T]` is deferred**; `col("a", int64)` is the
  working API.
- **The constraint solver will not evaluate a non-builtin function in a `where`
  clause**, ruling out pack-based schema surfaces that dispatch on column type.
- **A `VariadicPack` captured by one function's `*args` cannot be forwarded to
  another function's variadic parameter.** Every "build a heterogeneous
  collection from variadic args and hand it on" API here hits this — take the
  pre-built collection instead.
- **A `comptime name: T` trait requirement does not resolve reliably as `E.name`
  off an externally-bound generic parameter** (though `Self.name` inside the
  concrete type works, and `Self.K.name` on a kernel parameter does resolve).
  Expose the constant through a method.
- **A trait default whose return type a conformer must change cannot be
  overridden** — the two become competing overloads and every call reports
  `ambiguous call to 'x'`, at the call site rather than at either definition.
  So a trait default that returns a *concrete node type* dictates
  representation to every conformer. `Value` carried `isnull`/`notnull`
  returning the fused `NullPredicate`, which made `DynValue.is_null() -> Self`
  unwritable; they moved onto `DynValue`, whose callers they all already were.
  **This was once written as the broader "a struct method does not override a
  trait default", and that is false.** A *same-signature* override is ordinary
  and works — `Value.prune`, `name` and `bound_column` are each a trait default
  overridden by conformers throughout `values.mojo`. The incident above had a
  differing return type, which is the actual trigger; verified again in
  `docs/superpowers/specs/2026-08-21-aot-rewrite-research.md`.
- **A trait default method's parameter name must not collide with a
  *conformer's* struct parameter**, or that struct fails with `name conflict
  between parameter 'R' in the default trait method and a parameter in the
  struct`. This is why every binary operator on `NumericValue` names its
  parameter `Rhs`: `NumericBinary`/`FloatBinary`/`ConditionalBinary` already use
  `L`/`R`.
- **`constrained` is gone; the compile-time assertion is `comptime assert`.**
  `comptime assert i >= 0, "unknown column: " + String(name)` fails the
  build with `constraint failed: unknown column: nope`, message included.
- **A struct holding heap-allocated fields can be a comptime parameter.**
  `comptime SCHEMA = MiniSchema([...])` passed as `[s: MiniSchema]` works,
  so a schema can be a compile-time value and a bad column reference can
  be a compile error. That is *not* the reflection limit below, which is
  about reflecting field types and still stands.
- **A function that can `raise` cannot run at comptime.** Marrow mandates
  `def`, which is fine — a `def` without `raises` is comptime-eligible.
  But it decides which analysis methods are reusable at compile time:
  `referenced_columns`, `name` and `render` are non-raising and can be;
  `bound_column` and `prune` declare `raises` and cannot, until the
  not-found case becomes a `comptime assert` instead of an `Error`.
  **One `def` can serve both worlds** — verified: the same `index_of` ran
  at comptime and at runtime, which is how one analysis can back both
  expression lanes.
- **`comptime` is a reserved keyword, so a module named `comptime` must be
  escaped with backticks at every import site** — `from pkg.`comptime`.x
  import y` compiles and runs (verified 2026-08-21). This entry used to say
  it "cannot be a module name", which is false; the cost is the escaping,
  not impossibility. **And the cost is one line, not every import.** Inside
  `expr/comptime/` the package name never appears, so its own relative imports
  (`from .leaves import Column`) are plain; consumers import from `marrow.expr`,
  which re-exports. Only the boundary crossing escapes — the parent
  `__init__.mojo`, plus whatever imports both lanes directly (`builders.mojo`,
  `aggregates.mojo`). Verified end to end: a backticked subpackage with an
  `__init__.mojo` re-exported through its parent compiles and runs.
- There is **no runtime `__getattr__`**; the comptime `__getattr_param__` hook
  fires only for missing attributes and needs a handle type.
- Keep recursive and nested ops **out of kernel structs** — a binding-compiler
  crash was once observed on mutually-recursive nested-type static methods.
  Struct equality belongs at the composition layer (`equal_any`).

## Releasing

Marrow is published to prefix.dev as a conda package. A push of a `v*` tag fires
`.github/workflows/release.yml`.

1. Bump `version` under `[package]` in `pixi.toml`.
2. `git commit -m "chore: bump version to X.Y.Z"`
3. `git tag vX.Y.Z && git push origin main vX.Y.Z`

The workflow then runs the full test suite, builds `package/marrow.mojoc`
(`pixi run package`) and the conda package (`pixi build`), uploads it with
`pixi run publish` (needs the `PREFIX_API_TOKEN` secret), and creates a GitHub
release with both artifacts attached.

## Known Limitations

1. **Testing**: conformance testing leans on PyArrow until Mojo has a JSON
   library.
2. **Layout coverage**: bool, numeric, string/large_string, binary/large_binary,
   fixed_size_binary, list/large_list/fixed_size_list, struct, map, dictionary,
   decimal (32/64/128/256) and temporal (date/time/timestamp/duration/interval)
   are implemented; union, run-end-encoded and view layouts are not. `map` has
   no gaps left: `cast` gained its arm, and an element taken from a `MapArray`
   now reports `map<…>` — `ListScalar` carries its own dtype rather than
   rebuilding it (see §3). **This used to say
   `map` did not go through IPC "in either direction" because "type code 17 is
   absent from `ipc.mojo`" — that is false**: `comptime _TYPE_MAP: UInt8 = 17` is
   there, and the archery integration suite now runs `map`, `map_non_canonical` and
   `interval_mdn` at **14/14** against C++, Rust and Go — both directions.
   `interval` (YEAR_MONTH / DAY_TIME) is still skipped there, but that is a
   pyarrow limit: it has no type for either unit, and the harness bridges
   through pyarrow. Marrow consumes all three from the other implementations.
3. **Scalar fidelity**: six types have no dedicated scalar — `binary`,
   `large_binary` and `large_string` share `StringScalar`; `large_list`, `map`
   and `fixed_size_list` share `ListScalar`. **Only the `StringScalar` half is
   still a fidelity bug**: `StringScalar.type()` hard-returns `string`, so a
   `binary` element reports `string`. `ListScalar` no longer has that problem —
   it carries its own `DynType` instead of rebuilding `list_(child.dtype())`,
   so `large_list`, `map` and `fixed_size_list` elements report their own type,
   `keysSorted` and the entries field name included. Sharing a struct is not
   the defect; reconstructing the type from the child was.

Release callbacks in the C Data Interface **are** implemented and invoked — this
was listed here as a Mojo limitation long after it stopped being true. See
`c_data.mojo`'s four release paths plus the three PyCapsule destructors; the
double-free guard is the spec's null-release handshake.

## How to Identify Leaky Abstractions

Analyze the dependency relations and responsibilities of each type in the
requested marrow packages — e.g. `marrow.expr` and `marrow.kernels`, which are
tightly coupled. Produce a few-word summary of what each type does and what it is
responsible for; if a single responsibility cannot be identified, that is a leaky
abstraction. The dependencies must form a one-directional tree with no cycles.
Aggregate this and report the findings.
