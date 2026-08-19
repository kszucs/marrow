# `marrow compile` — AOT query binaries with late-bound parameters

Status: design approved, not implemented. 2026-08-18.

## Summary

A CLI, `marrow compile <file> [-o out]`, that turns a Mojo file containing a
marrow AOT expression into a small standalone binary. The query *shape* is
frozen at compile time; the values it needs — source paths and scalar constants
— are supplied on the command line at run time via a new `param()` placeholder
that sits alongside `col()` and `lit()`.

Three things make this worth building: it is the one capability marrow has that
polars and duckdb structurally cannot match, it gives the AOT lane a reason to
exist for users rather than only for the size gate, and it closes backlog item
**M1.6** ("AOT DSL docs — one runnable example") as a side effect.

## User-facing surface

```mojo
from marrow.expr.builders import col, param
from marrow.dtypes import int64, string

def main() raises:
    var src   = param("src", string, help="input parquet")
    var min_a = param("min-a", int64, default=0)

    var plan = DynRelation(ParquetScan[...](path=src, schema=sch))
        .filter(BoxedValue(col("a", int64) > min_a))

    plan.execute_cli()
```

```
$ marrow compile q.mojo -o q
$ ./q --src data.parquet --min-a 5
$ ./q --src data.parquet --min-a 5 -o result.parquet
$ ./q --help          # rendered from the declarations
$ ./q --describe      # the same, as JSON
```

There is **no code generation** and **no CLI object** in the user's face.
`param()` is an ordinary library call that returns an expression node, and its
placement next to `col()` and `lit()` is deliberate: `col` reads from data,
`lit` is a constant, `param` is a constant supplied later.

## Design

### 1. Parameter nodes

`param("min-a", int64)` returns a `NumericParam[Int64Type]` — structurally
`NumericLiteral[T]` (`values.mojo:831`) with one change: the scalar lives behind
an `ArcPointer[ParamCell[T]]` rather than inline.

```mojo
struct NumericParam[T: NumericType](NumericValue):
    comptime OutType = Self.T
    comptime OutShape = 0
    comptime State = Scalar[Self.T.native]   # resolved once per batch
    var _cell: ArcPointer[ParamCell[Self.T]]

    def state(self, batch: RecordBatch) raises -> Self.State:
        return self._cell[].get()            # raises if unbound

    @always_inline
    def lane[W: Int](self, state: Self.State, idx: Int) -> SIMD[Self.OutType.native, W]:
        return SIMD[Self.OutType.native, W](state)
```

Resolving the cell in `state()` rather than `lane()` is what makes this free:
the inner loop is byte-identical to `NumericLiteral`'s splat, so a parameter
costs **nothing per row**. `prune()` reads the cell too, so a bound parameter
still prunes row groups — a parameterised date filter prunes exactly as a
literal one does.

Node set:

| node | family | notes |
|---|---|---|
| `NumericParam[T]` | `NumericValue` | mirrors `NumericLiteral[T]` |
| `StringParam[T]` | `StringValue` | mirrors `StringLiteral[T]` (`values.mojo:1793`) |
| `TemporalParam[T]` | `TemporalValue` | date/timestamp filters |
| a `DynValue` param leaf | runtime lane | **required by invariant 2** |
| `PathSpec` | not a value node | `ParquetScan.path` |

### 2. `ParquetScan.path: String` -> `PathSpec`

`ParquetScan.path` is an eager `String` (`relations.mojo:1033`), set at
construction. `PathSpec` replaces it: either a literal path or a cell, with an
`@implicit` conversion from `String` so **every existing call site is
untouched**.

This is allowed under the "do not change layout" constraint, which names array,
scalar and builder layout specifically. `ParquetScan` is a relation node. The
backlog explicitly warns that this constraint has been read too broadly before
(the B13 case), so the reading is deliberate and recorded here.

### 3. Discovery: a drained registry, not a trait method

`execute_cli()` takes no arguments, so it must find the parameters some other
way. Two options were weighed:

- **A `parameters()` method on the `Value` trait**, mirroring
  `referenced_columns()`. Clean, no global state, fully isolates two plans in
  one process. Costs **40 new implementations** (39 in `values.mojo`, 1 in
  `dynamic.mojo`) and a *second* recursive traversal in every node.
- **A module-level registry that `param()` writes to and `execute_cli()`
  drains.** Zero new node code, zero size cost.

**The registry wins.** Invariant 1 is a merge gate, and this tree has already
measured a single shared adapter at **+662,740 bytes**; adding a second
recursion to 40 nodes to buy multi-plan isolation — which is explicitly out of
scope — is not a trade worth making.

Draining the registry in `execute_cli()` means *sequential* plans in one process
work correctly. The known limit is *interleaved* construction of two plans,
which would confuse the registry. `parameters()` is recorded here as the upgrade
path if the extension-module work ever needs it.

### 4. `execute_cli()`

1. Parse `argv` against the drained declarations.
2. `--help` / `--describe` short-circuit and exit **before** executing. Because
   the cells are declared by the time `execute_cli()` runs, no dummy values are
   needed: the plan is built with unbound cells and simply never executed.
3. Bind the cells.
4. `execute(ctx)` -> `RecordBatch` (`relations.mojo:390`).
5. Write the output.

Output contract: no `-o` pretty-prints to stdout (what the size gates do today);
`-o r.parquet` writes Parquet; `-o r.arrow` writes Arrow IPC; `--format`
overrides the extension. **The Parquet and IPC writers must be opt-in if the
size gate says they are expensive** — measure before making both unconditional.

### 5. The `marrow compile` CLI

Pure Python in `python/marrow/compile.py`, exposed as a console script from
`python/pyproject.toml`. It shells out to:

```
mojo build -O3 -g0 -I <marrow-source> <file> -o <out>   &&   strip <out>
```

which is what `benchmarks/binary_size/compare.py` already does, productized.

Include-path resolution, in order:

1. `--marrow-path <dir>`
2. `$MARROW_MOJO_PATH`
3. the bundled `marrow/_mojo/` inside the installed package
4. repo-root autodetect (development)

It verifies `mojo --version` against the pinned range up front and fails with an
actionable message rather than letting an opaque compiler error surface.

### 6. Packaging

The wheel ships marrow's **Mojo source**, tests excluded, under
`marrow/_mojo/`, plus an extra that pulls the compiler:

```toml
[project.optional-dependencies]
compile = ["mojo>=1.1,<2"]
```

Source rather than `.mojoc` because it is **1,678,678 B against 5,625,382 B**
(3.4x smaller) and strictly more tolerant: a `.mojoc` is pegged to the *exact*
compiler build and hard-errors on any skew, whereas source fails only on real
language changes. Both are architecture-independent.

One wheel, not two. Extras add dependencies, not files, so gating the payload
would need a second `marrow-mojo` distribution — 1.68 MB on an already-21 MB
wheel does not justify a second artifact, lockstep releases, and a new skew
failure mode.

### 7. `--bundle` — static linking was tried, and is impossible

**Measured on 2026-08-18, not assumed.** Four approaches were probed:

| attempt | result |
|---|---|
| `mojo build -Xlinker -static` | `ld: library 'System' not found` — macOS ships no `libSystem.a`; Apple does not support fully static executables |
| `mojo build -Xlinker -Bstatic` | `ld: unknown options: -Bstatic` — a GNU ld flag with no Apple equivalent |
| `mojo build -Xlinker -dead_strip_dylibs` | **works**, drops `libAsyncRTMojoBindings` from a trivial binary (2 deps -> 1) — but changes **nothing** for a real marrow query |
| build a `.a` from the shipped `.dylib`s | impossible — they are fully linked Mach-O images with no embedded bitcode (`otool -l` shows no `__LLVM` section) and no extractable relocatable objects |

There is also no static-link flag in `mojo build` at all (`--help`,
`--help-hidden` expose only `-Xlinker` and `--lld-path`);
`--relocation-model static` is a codegen addressing model, not static linking.
Modular's own build imports every runtime lib as
`cc_import(shared_library = ...)` (`bazel/modular_wheel_repository.bzl:85-111`),
and `static_library` appears in that repo only for third-party prebuilts (ucx,
gperftools). Mojo's docs contain no standalone-binary deployment story.

`-dead_strip_dylibs` fails on a real query because the symbols are genuinely
referenced. `query_scan_typed` has **47 undefined symbols**:

| origin | count | notes |
|---|---:|---|
| libc (`_memcpy`, `_open`, `_dlopen`, ...) | 22 | `/usr/lib/libSystem.B.dylib`, always dynamic |
| `_KGEN_CompilerRT_*` | 15 | allocator and runtime support |
| `_AsyncRT_*` | 10 | **all `DeviceBuffer` / `DeviceContext`** |

**A quantified follow-up falls out of this.** All 10 AsyncRT symbols are GPU
device calls — in a **GPU-off** build. CLAUDE.md already records that a GPU-off
binary still links AsyncRT; this pinpoints why: 10 device symbols survive DCE.
Eliminate them and `-dead_strip_dylibs` would drop
`libAsyncRTMojoBindings.dylib` outright — **1,156,592 B, 42% of the runtime
closure**. Out of scope here; worth a backlog card.

So `--bundle <dir>` produces a self-contained **directory**: the stripped binary
plus its transitive dylib closure, with the rpath rewritten to `@loader_path`
(macOS, `install_name_tool`) or `$ORIGIN` (Linux, `patchelf`). This is the right
shape anyway — a Lambda deployment unit is a zipped directory, not a single file.

Without it the artifact does not run off the build machine: `LC_RPATH` is baked
to an absolute path inside the local pixi env.

**The closure must be walked, not hardcoded.** The binary links 2 dylibs
directly but depends on 4 transitively, and Modular's own
`INDIRECT_DEPENDENCIES` list is `AsyncRTMojoBindings`, `AsyncRTRuntimeGlobals`,
`KGENCompilerRTShared`, `MGPRT`, `MSupportGlobals` — `MGPRT` being the GPU
runtime, which a `-D MARROW_GPU=true` build would likely add as a 5th.

Static linking is **blocked upstream, not rejected**: if Modular ships static
archives it becomes a `--static` flag over the same code path. On Linux the
question is genuinely open — glibc/musl static linking exists there — and
`-Xlinker -static` is worth one probe, but it cannot be tested on the darwin
development box.

## Measured evidence

All figures measured 2026-08-18 on osx-arm64 unless noted.

| quantity | value | how |
|---|---:|---|
| `marrow.mojoc` | 5,625,382 B | `.precompile/marrow.mojoc` |
| Mojo source, no tests | 1,678,678 B | `find marrow -name '*.mojo' -not -path '*/tests/*'` |
| Mojo source, all | 2,832,303 B | 140 files |
| `query_scan_typed` stripped | 2,085,496 B | `benchmarks/binary_size/` |
| `query_streaming` stripped | 1,522,408 B | the in-memory floor |
| `libAsyncRTMojoBindings.dylib` | 1,156,592 B | |
| `libKGENCompilerRTShared.dylib` | 1,138,960 B | |
| `libAsyncRTRuntimeGlobals.dylib` | 339,488 B | transitive |
| `libMSupportGlobals.dylib` | 123,472 B | transitive |
| **runtime closure** | **2,758,512 B** | 4 dylibs |
| **deployable Parquet query** | **~4,844,008 B (4.6 MB)** | binary + closure |
| local `mojo` | 1.1.0.dev2026081705 | `mojo --version` |
| PyPI `mojo` latest stable | 1.0.0 (2026-08-11) | pypi.org JSON API |
| `referenced_columns` impls | 40 | 39 `values.mojo` + 1 `dynamic.mojo` |
| undefined syms, `query_scan_typed` | 47 | 22 libc + 15 KGEN + 10 AsyncRT |
| trivial binary, baseline | 34,216 B | 2 `@rpath` deps |
| trivial binary, `-dead_strip_dylibs` | — | 1 `@rpath` dep (AsyncRT dropped) |

## Benefits and drawbacks, by angle

**User value — strong.** No Python, no interpreter, no PyArrow. 4.6 MB against
Lambda's 250 MB limit, with process-start cold start rather than
Python-import cold start.

**Architecture — low risk, and it pays back.** A parameter node is the smallest
possible addition to the fused lane: a literal plus an indirection resolved once
per batch. It also closes M1.6 and gives the AOT lane a user-facing purpose.

**Compile time — the worst drawback.** Every `marrow compile` elaborates all of
marrow at `-O3`. The `binary_size` sweep is 11 builds in "ten to twenty
minutes", so roughly **1-2 minutes per invocation**. That is a bad edit-run
loop and must be measured and documented, not discovered by users. Mitigations
to measure: a `--fast` mode at `-O1`, and whether `.mojoc` helps at all — it
skips parsing, not elaboration, so a null result is the expected outcome.

**Versioning — the sharpest external risk.** marrow pins a Mojo *nightly*
(1.1.0.dev2026081705) while PyPI stable is 1.0.0, and a wheel cannot force an
`--extra-index-url`. So `pip install marrow[compile]` cannot resolve marrow's
actual compiler until marrow rides a stable Mojo. v1 pins a range and checks at
run time with a clear error.

**Maintenance — modest but real.** Two node families to keep at parity across
lanes, a size gate, an argv parser in Mojo, and an integration test that needs a
compiler in CI — where CI is currently dark and would need a multi-minute `-O3`
build.

**Honest counter-case.** The query *shape* is frozen at compile time; only
scalars and paths are late-bound. Anyone whose shape varies needs the Python
lazy frontend. If most users are in that bucket, this is a demo rather than a
product — worth deciding deliberately rather than discovering.

## Invariants this must satisfy

1. **Small-binary DCE (invariant 1).** A new `query_param` gate in
   `benchmarks/binary_size/`, measured against `query_scan_typed`.

   **MEASURED 2026-08-19 — the "expected near-zero" prediction was wrong.**

   | configuration | `query_param` `__text` | delta vs `query_scan_typed` (2,025,432) |
   |---|---:|---:|
   | writers linked | 2,794,420 | **+768,988** |
   | writers gated out | 2,222,132 | **+196,700** |

   So Parquet + Arrow IPC output alone costs **572,288 bytes**, and
   `execute_cli`'s own argv / `--help` / `--describe` / `parse_params`
   machinery costs the residual ~196,700. The parameter *node* is
   near-free as predicted; the *entry point* is not.

   Consequences, both binding:
   - The writers sit behind a `CLI_WRITERS_ENABLED` comptime flag (set by
     `-D MARROW_CLI_WRITERS=true`), off by default, so the gate measures
     the floor. With it off, `-o out.parquet` raises a named error rather
     than silently writing nothing — verified by review.
   - **`marrow compile` passes `-D MARROW_CLI_WRITERS=true` by default**,
     because a binary that cannot honour its own documented `-o` flag is
     worse than a larger one. An opt-out flag produces the minimum-size
     build. Both numbers are published so the trade is visible.
2. **One engine, two drivers (invariant 2).** The runtime-lane param leaf is not
   optional, and it needs a case in `marrow/expr/tests/test_parity.mojo`.
3. **Warning-clean.** `mojo precompile marrow` stays at 0 errors, 0 warnings.
4. **CHANGELOG entry** under `## [Unreleased]`.

## Non-goals

- AWS Lambda packaging, handler shim, or deployment tooling.
- The Python extension module. Noted as a small follow-on:
  `python/bindings/lib.mojo` is already the exact recipe (`PyInit_*` +
  `PythonModuleBuilder`, `--emit shared-lib`), so it is one `def_function` over
  the same bound plan.
- Multi-query artifacts, or any late binding of the query *shape*.
- Cross-compilation. `--target-triple` exists but is unprobed here.

## Open questions for implementation

1. Does `.mojoc` measurably beat source for `marrow compile` wall-clock? Expect
   no; measure before assuming either way.
2. What do the Parquet and IPC writers cost on the size gate? If either is
   expensive, make it opt-in rather than always linked.
3. Does `-Xlinker -static` work on Linux? Untestable on the darwin box.
4. Does `-D MARROW_GPU=true` add `libMGPRT` to the closure? Affects `--bundle`
   only, and is handled correctly by walking rather than hardcoding.
