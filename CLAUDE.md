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

- `mojo >=1.0.0b3.dev2026072406,<2` and `max ==26.5.0.dev2026072406` — MAX is
  pinned to the matching version line because GPU codegen resolves
  `max.package_root` since that Mojo release.
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

**Never leave a `marrow.mojoc` where a runner can see it.** Such an artifact
*shadows the entire `marrow/` source tree*, so every import resolves against the
stale package instead of the files you just edited. Mojo puts a source file's own
directory on the import search path, which makes **two** locations dangerous: the
repo root (runners compile with `-I .`) and `.test_runners/`, where the generated
driver lives. `precompile` therefore writes to `.precompile/`, which is on
neither path. It used to write to `.test_runners/`, which silently broke every
following `pytest` run — symptom (observed 2026-07-27): a case you just added
reports `module 'test_<x>' does not contain 'test_<your_new_case>'` and the run
fails in well under a second, because nothing is compiled at all. Stale artifacts
fail differently again, with `unable to locate module 'tests'`. On either,
`rm -f .test_runners/marrow.mojoc marrow.mojoc` first.

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
pixi run -e dev pytest marrow/expr/tests marrow/kernels/tests   # after editing expr + kernels
pixi run -e dev pytest marrow/kernels/tests/test_groupby.mojo   # narrower still
```

`python/marrow/libmarrow.so` is rebuilt automatically by `conftest.py` before any
session that runs Python tests — no manual `build_python` step needed.

### One selection = one compilation unit

Compilation dominates, and almost all of it is elaborating **marrow**, not the
test bodies. The compiler takes a single input file per invocation, so building
per test file paid that elaboration once per file. The harness instead generates
**one driver** for the whole selection — `.test_runners/_test_driver_<hash>.mojo`,
which imports the selected cases and hands them to `TestSuite.run` as a tuple —
and compiles that. The name is the hash of the driver's own source, so concurrent
sessions with different selections never overwrite each other, while the same
selection always resolves to the same path and the same cached artifact. Measured
on `marrow/expr/tests` (9 files, 280 cases):

| | wall | peak RSS |
|---|---|---|
| one generated driver, all 9 files | 4 min 43 s | 17.0 GB |
| separately: `test_aggregates.mojo` | 204 s | 11.7 GB |
| separately: `test_join.mojo` | 198 s | 19.6 GB |

Two files on their own cost more than all nine together, and the aggregate peaks
*below* a single file — N files in one unit cost about what 1 file costs.

- **Selecting fewer *files* is what saves time; selecting fewer *cases* is not.**
  A single `::test_name` builds the same unit as its whole file.
- **Blast radius is the selection — for compile *errors*.** A diagnostic fails
  every case in the run, not just its file, because there is one unit.
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
- **The driver is generated deterministically** (modules sorted, cases in source
  order) so re-running an unchanged selection produces byte-identical source and
  hits the Mojo compiler's own artifact cache. That cache only ever helps
  *identical* rebuilds; it does nothing across different files.
- Ordinary runs use **`mojo run`** — compile and execute in one step, no artifact
  left behind. **`--asan` runs use `mojo build`**, because the sanitizer runtime
  has to be linked into a real binary.
- `pytest-xdist` does not help *within* a run: one selection is one unit, so each
  worker would build a unit of its own. Several independent `pytest` invocations
  concurrently are safe — drivers are content-addressed, so their files and
  (under `--asan`) their binaries never collide.

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
relative imports), and use `Benchmark` from `marrow.testing`:

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
    keep(data)  # prevent ASAP destruction — see below
```

**`keep(data)` after `b.iter[call]()` is mandatory for non-trivial captures.**
Mojo's ASAP destruction frees values as soon as the compiler believes their last
use has passed. When a `@parameter` closure captures a variable and is handed to
`b.iter[call]()`, ASAP may free it *after* the closure is registered but *before*
it runs — a heap-use-after-free inside the iteration loop. This applies to
`StructArray`, `PrimitiveArray[T]`, `SwissHashTable`, `HashJoin`, and friends.

**Two rules that each cost a wrong conclusion on 2026-08-05:**

- **Read the unit on every row before comparing two numbers.** pytest-benchmark
  scales each benchmark independently, so one row prints `Name (time in ns)`,
  the next `(time in us)` and the next `(time in ms)`. Comparing the bare figures
  reported a 25x speedup where there was a 40x slowdown, and the filed bug said
  the opposite of the truth. Strip the colour codes and read the headers:
  `sed 's/\x1b\[[0-9;]*m//g' out.log | grep -E "Name \(time|^bench_"`.
- **An `assignment to 'x' was never used` warning on a value the closure *does*
  use means the capture was not made** — the body reads garbage and the timing is
  meaningless. Same tell as the `sync_parallelize` miscompile noted under
  `ExecContext.stripe`. Add the missing `keep(x)` and re-measure; never report a
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
- All buffers are 64-byte aligned and padded.

**`Bitmap`** (also `marrow/buffers.mojo` — there is no `bitmap.mojo`): bit-packed
validity wrapping a `Buffer`, same `mut` pair, same O(1) copies.

**`BufferView` / `BitmapView`** (`marrow/views.mojo`): borrowed, offset-applied
spans. These are what kernels compute over; `views.mojo` also owns `apply` and
`_apply_dispatch`, the CPU/GPU element-wise driver.

Rules:

- Prefer `Buffer`/`Bitmap` for owned values and `BufferView`/`BitmapView` for
  computation. No naked pointer arithmetic in kernel or array code.
- **`unsafe_ptr()` is restricted to `buffers.mojo`, `views.mojo`, and
  `c_data.mojo`.** Everything else goes through the view abstractions.
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
resolving a runtime `DataType` to a comptime type parameter for a `capturing`
job. `variant_dispatch*` (`marrow/utils.mojo`) is the comptime adapter over
stdlib `Variant`.

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

`marrow/expr/` is split into **two lanes that no longer share node types**:

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

- **Keep `marrow.expr` small-binary** — preserve the closed-erasure/DCE property
  (no open dispatchers, fused-only value boxes, closed per-dtype kernels) and
  gate changes on `benchmarks/binary_size/` (`pixi run binary_size`).
- **The box is the erasure boundary — a node never needs an erased variant.**
  `DynValue` erases; the nodes do not. Before adding a `Dyn*` node, check the
  existing one: either its type is known where it is constructed (a literal, a
  cast target — resolve a runtime dtype with `dispatch_*` and box each arm), or
  it does not depend on the type at all (a column read by name).
  `DynColumn`/`DynLiteral`/`DynCast` were all added and all removed for this
  reason. `DynValue` conforms to every value family, so fused nodes take it as an
  operand with no bound relaxed; a node keys off `comptime IsErased` (propagated,
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
├── utils.mojo            # variant_dispatch*, GPU_ENABLED, has_accelerator_support
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
├── testing/              # TestSuite + Benchmark used by the generated driver
└── tests/                # test_*.mojo + bench_*.mojo for the core modules
python/                   # Python package + bindings (python/marrow/libmarrow.so)
└── marrow/tests/         # Python test_*.py and bench_*.py
benchmarks/               # standalone programs (they own a `main()`, so they
                          # cannot live inside the package) + binary_size gate
docs/                     # Quarto site, design documents, and backlog.md
```

Tests and benchmarks sit **inside** the package, next to the code they cover.
That works because they carry no `main()` — see "Writing Mojo tests".

**`docs/backlog.md` is the only place that says what is open.** It replaced seven
`docs/tasks-*.md` files on 2026-08-03. Its §0 carries the standing constraints and
measurement traps — read them before planning anything, because each one
invalidates an approach that looks obvious. Its status lines have been wrong
before (an audit that day found 18 wrong statuses across the files it replaced),
so **verify by `grep`, not by reading a header** — and exclude
`.claude/worktrees/`, which holds stale pre-refactor code that produces false
positives from the repo root.

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

Anything touching device code needs `comptime if GPU_ENABLED:` around it — a
*runtime* `if ctx.is_gpu()` alone cannot be eliminated at elaboration time.

This is the **largest single compile-time lever** in the tree. Cold builds (fresh
`MODULAR_CACHE_DIR`; a repeated identical compile just hits the Mojo transform
cache and measures nothing):

| | GPU off (default) | GPU on |
|---|---|---|
| `cast`, numeric x numeric | **14.6 s** | 40.1 s |
| `cast` + `sort_indices` | **43.7 s** | 85.0 s |

**Both halves must be gated or you get none of it.** The allocations need
`comptime if GPU_ENABLED` *and* `has_accelerator_support` must answer False.
Gating either alone measured as no change at all (45.2 s and 84.4 s against
42.5 s / 84.1 s baselines) — which is why this looked like a dead end for a long
time. It does not shed the `libmax`/AsyncRT runtime dependency, though: a binary
built with GPU off still links it.

### How device execution is wired

One kernel serves both targets. `views._apply_dispatch[Out, gpu_ok, process]`
picks the path from the `ExecutionContext`: `ctx.is_gpu()` takes a single grid
launch via `elementwise` at the GPU SIMD width; otherwise one `vectorize` body is
handed to `ctx.stripe`, which runs it on the calling thread or across
`ctx.resolved_num_threads()` workers. Thread count is owned by `ctx` — no
Mojo-internal heuristic. `gpu_ok` is the caller's
`has_accelerator_support[In, Out]()` result, passed as a comptime `Bool` so the
GPU branch is dead-code-eliminated when unsupported.

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

Measured on Apple Silicon (M-series, Metal, unified memory). The numbers below
come from a cosine-similarity kernel that no longer lives in the tree; the
conclusions still hold.

- **Low arithmetic intensity (element-wise add)**: CPU SIMD wins — transfer cost
  dominates at ~1 FLOP per element. Do not GPU-accelerate these.
- **High arithmetic intensity (~3×dim FLOPs per vector)**: GPU wins at scale,
  *with data already resident*. Uploading per call is 2-3× slower than CPU even
  for compute-heavy kernels.
- **Crossover**: ~10K vectors at dim ≥ 384. At 500K-1M vectors, dim 768, a
  preloaded GPU run was ~13× faster than CPU SIMD.
- **Guideline**: upload once, run several kernels device-resident, download at
  the end.

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
  it that way.
- **ASAN can *hide* a heap bug rather than reveal it.** A one-byte `Variant`
  overflow (since fixed upstream) passed 35/35 under ASAN and failed without it,
  and a Mojo *build* failure emits no ASAN output at all — so a clean ASAN run is
  not evidence on its own. Verify without ASAN too, and confirm the tests
  actually ran rather than the build having died.

### Associated types and traits (learned the hard way)

Non-obvious compiler behaviours, each reproduced by a minimal example. They
mostly bite generic trait hierarchies such as `marrow.expr.values`.

- **A *chained* associated-type projection (`Self.OutType.ArrayType`) does not
  reduce at a call/return site**, even when the concrete type is statically
  determined. A *single* projection off a **direct trait-bound parameter**
  (`T.ArrayType` for `T: SomeTrait`) *does* reduce. To return or consume a
  concrete companion type without a `rebind` or a reducer helper, expose it as a
  **direct associated member** of the trait (e.g. `Value.ArrayType`, which each
  node fixes concretely), not as `Self.OtherAssoc.Member`.
- **An associated-type *default* that references a sibling associated type**
  (`comptime ArrayType = Foo[Self.OutType]`), together with a method returning it
  (`def execute -> Self.ArrayType`), errors with `attempt to resolve a recursive
  reference to declaration 'execute'`. A *concrete* default with no `Self.X`
  reference (e.g. `= BoolArray`) does not recurse. Declare such members **per
  concrete struct** instead.
- **But composing an associated type out of *type parameters'* associated types
  is fine** — the hazard above is specifically a reference to a sibling of
  `Self`. Verified 2026-08-03: on a struct `Bin[L: Value, R: Value]`,
  `comptime State = Tuple[Self.L.State, Self.R.State]` together with
  `def prepare(self, …) raises -> Self.State` compiles, reduces, and composes to
  at least depth 4 with mixed shapes — including a node whose `State` is an
  owning container, and one such node nested inside another. It also survives
  capture by a `@parameter` closure. The one non-obvious requirement: because
  `prepare` raises, the associated type needs `ImplicitlyDeletable`
  (`comptime State: Copyable & ImplicitlyDeletable`), else every call site fails
  with *"abandoned without being explicitly destroyed"* on the throw path.
- **Re-defaulting a base trait's abstract method in a sub-trait recurses** if that
  method returns `Self.ArrayType` and a conforming node's `ArrayType` transitively
  references another trait-member child (the compiler loops elaborating the
  child's own copy of that method). The trigger is *the abstract-then-re-defaulted
  method specifically* — a **differently named** sibling with the same body and
  return type does **not** recurse. Fix: keep the base method abstract and have
  each node override it with a one-liner delegating to a helper under a new name
  (`NumericValue.execute` abstract → `NumericValue._fused`).
- **A family-trait-provided associated-type default may not satisfy the base
  trait's abstract requirement** for conforming structs (`does not implement all
  requirements for <BaseTrait>`) — declare the member on each concrete struct even
  if a parent trait "provides" it.
- **A trait-level *default method* cannot return `Self.AssocType` unless that
  associated type's bound is `ImplicitlyCopyable`.** Declaring
  `comptime State: Copyable & ImplicitlyDeletable = DynArray` alongside a
  defaulted `def state(self, …) -> Self.State` fails: the compiler will not
  reduce `Self.State` to its own declared default at the return site
  (`cannot implicitly convert 'DynArray' value to '_Self.State'`), and `rebind`
  cannot bridge it — `rebind[Self.State](x)` reports *"value of type
  '_Self.State' cannot be implicitly copied"*, since `rebind` itself requires
  `ImplicitlyCopyable`. Transferring with `^` does not help, and widening the
  bound is not available when the default is an array type, because `DynArray`
  and friends deliberately are not implicitly copyable. **Consequence:** a
  protocol whose associated type carries non-implicitly-copyable state cannot be
  rolled out incrementally behind defaults — every conformer must implement it in
  the same commit. Found 2026-08-05 attempting exactly that for A1 (see
  `docs/backlog.md`); same family as the conditional-type limitation below.
- **A comptime *conditional type* is usable as a type, but carries no trait
  conformance and does not reduce at a return site** — not even inside a
  `comptime if` that has already selected the branch.
  `comptime C[To, V] = V if (V.OutType.native == To.native) else Wrapper[To, V]`
  resolves fine as an annotation, but a function returning `C[To, V]` cannot
  return either branch: `cannot implicitly convert 'V' value to 'V if (…) else
  Wrapper[To, V]'`. `rebind` does **not** help — `rebind[C[To, V]](x)` fails with
  `value of type '<the conditional>' cannot be implicitly copied, it does not
  conform to 'ImplicitlyCopyable'`, because an unreduced conditional conforms to
  nothing at all. Note `promote[L, R]` (`expr/values.mojo`) is the same shape and
  works — the difference is that it is only ever *used* as an annotation, never
  returned. Consequence: "wrap this operand only when it needs converting" is not
  expressible; either always wrap, or do the selection where the concrete type is
  known. This blocked the promote-at-construction design recorded in
  `docs/backlog.md`.

### Reflection, packs, and comptime aliases

Each confirmed by triggering the actual compiler error on the pinned toolchain.

- **A reflected field type is opaque inside the generic function that reflects
  it.** `reflect[T].field_at[i].T` reads as a type and works even when `T` is a
  generic parameter, but a bare `FieldT()` call inside a `comptime for` over it
  fails to resolve — during generic-mode checking the compiler sees only an
  opaque type with no visible constructor. Route construction through a
  *separately-instantiated* generic bound on the trait, so the zero-arg
  constructor arrives via the trait witness: `def _construct_default[D:
  Defaultable & DataType]() -> D: return D()` (`marrow/schema.mojo:12`). That
  helper is what makes `Schema.from_struct[T]()` work.
- **This is why `Table[T]` is deferred.** The `t.a` sugar needs a parametric
  `comptime _dtype[name] = reflect[T].field[name].T`, which hits the same limit
  (`marrow/expr/values.mojo:2416-2421`). `col("a", int64)` is the working API.
- **The constraint solver refuses to evaluate a non-builtin function inside a
  `where` clause.** `reflect[T].field_index[name]()` folds to a builtin KGEN
  attribute and *can* be proven during overload selection; a recursive `def` over
  a variadic pack cannot. This rules out pack-based schema surfaces that dispatch
  numeric-versus-string columns. Returning a `comptime`-branched type from a
  helper is likewise rejected ("dynamic type values not permitted yet").
- **A `VariadicPack` captured by one function's `*args` cannot be forwarded to a
  different function's variadic parameter** — `"assigning 1 operand to an
  unresolvable variadic pack argument"`. `Tuple(a, b, c)` from fresh args is
  fine; from an already-captured pack it is not. `Tuple`/`Variant`/`UnsafeUnion`
  sidestep this by owning their pack storage with raw `__mlir_op` calls, which
  this project restricts to `buffers.mojo`, `views.mojo` and `c_data.mojo`.
  **Every "build a heterogeneous collection from variadic args and hand it to
  another type" API here will hit this** — take the pre-built collection instead.
- **A `comptime name: T` trait requirement does not resolve reliably when read as
  `E.name` from a function generic over `E: SomeTrait`**, though `Self.name`
  inside the concrete type's own method works. Expose the constant through a
  method. Narrower than it sounds: `Self.K.name` on a *kernel parameter* does
  resolve (`NumericCompare.prune`, `values.mojo:965`, branches on it at
  elaboration). The failure is reading a trait-declared alias off an
  externally-bound generic parameter.
- **`comptime` is a reserved keyword and cannot be a module name** —
  `import marrow.expr.comptime` fails to parse.
- **A binding-compiler crash was once observed on mutually-recursive nested-type
  static methods on a reflected kernel struct.** The construct was never pinned
  down. The design avoids it by keeping recursive and nested ops *out* of kernel
  structs — struct equality is a recursive `AND` over child comparisons and
  belongs at the composition layer reusing `EqKernel.dispatch` for leaves, which
  is what `equal_any` (`kernels/numeric.mojo`) is. Treat that as a layering rule
  first and a crash workaround second.
- There is **no runtime `__getattr__`** on ordinary structs. The compile-time
  hook `__getattr_param__[name: StaticString]()` exists and its return type may
  depend on the name, but a handle type is required (a real field shadows it — it
  fires only for *missing* attributes) and `var` is mandatory on field
  declarations.

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

1. **Type system**: Variant elements must be copyable; references and lifetimes
   are still evolving.
2. **Testing**: conformance testing leans on PyArrow until Mojo has a JSON
   library.
3. **Layout coverage**: bool, numeric, string/large_string, binary/large_binary,
   fixed_size_binary, list/large_list/fixed_size_list, struct, map, dictionary,
   decimal (32/64/128/256) and temporal (date/time/timestamp/duration/interval)
   are implemented; union, run-end-encoded and view layouts are not. **`map` is
   the exception to "implemented"** — it works in dtypes, arrays, builders, the C
   Data Interface and Parquet, but **not through IPC in either direction** (type
   code 17 is absent from `ipc.mojo`), it has no `MapScalar` (a scalar taken from
   a `MapArray` reports `list<…>`), and `cast` has no arm for it.
4. **Scalar fidelity**: six types have no dedicated scalar — `binary`,
   `large_binary` and `large_string` collapse to `StringScalar`; `large_list`,
   `map` and `fixed_size_list` collapse to `ListScalar`. `StringScalar.type()`
   hard-returns `string` and `ListScalar.type()` hard-returns `list_(child)`.

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
