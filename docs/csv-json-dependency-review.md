<!--
Copyright 2024 Szűcs Krisztián
SPDX-License-Identifier: Apache-2.0
-->

# CSV and JSON: buy, bind, or build

marrow has no CSV reader and no JSON reader. `backlog.md` §1.2 calls that the
highest ratio of user value to engineering novelty on the page, and a first
user arrives with a CSV, not a Parquet file. This is the review of how to get
one: take a Mojo package, `dlopen` a C library the way the Parquet page codecs
are opened, or write it.

Everything below was measured on 2026-09-22, on `osx-arm64`, with the compiler
in this repository's `dev` environment — **Mojo 1.2.0.dev2026092105**
(`pixi.toml` pins `mojo >=1.2.0.dev2026092105,<2`). Nothing here was measured
on Linux; where a conclusion could plausibly differ there it is marked. Claims
taken from a README or a search result rather than run are marked
*unverified*.

## The finding that decides most of it

**A Mojo library cannot be a dependency of this repository in the ordinary
sense. It can only be a git source dependency, rebuilt by marrow's own
compiler.**

The `modular-community` channel publishes 60 packages for `osx-arm64`, of which
exactly two are relevant: `emberjson` and `mojo_csv`. `emberjson` 0.3.4
declares `mojo-compiler >=1.0.0,<2.0a0`, which marrow's pin satisfies, so a
solver will happily install it. Building against it then fails:

    error: Mojo precompiled file is incompatible with the current version of
    the Mojo compiler. Precompiled file '.../emberjson.mojoc' version 1.0.0 is
    older than compiler version 1.2.0.dev2026092105.

`mojo_csv` 1.6.4 is worse — it still ships the older `.mojopkg` format, and the
1.2 compiler does not locate the module at all. A published Mojo package is a
precompiled artifact keyed to the compiler that produced it, and its declared
version range is not that key; `randyzwitch/dataframe_mojo`'s README states the
rule outright ("a compiled Mojo package only loads in the Mojo version that
produced it"), which is why it ships one git tag per Mojo minor.

So the conda channel buys nothing at a nightly pin. Every candidate below is
evaluated as source, compiled here, which is also how EmberJson consumes its
own dependency (`emberserde = { git = ..., rev = ... }`).

## The candidates

Compiled with `mojo precompile <pkg>` against 1.2.0.dev2026092105 unless noted.
`__text` is `size -m` on a stripped `mojo build -O3 -g0` program that calls the
library, against a 1,180-byte baseline program that calls nothing.

| Candidate | Kind | Last commit | Targets | Packaged | Licence | Builds here | Verdict |
|---|---|---|---|---|---|---|---|
| `std.json` / `std.csv` | stdlib | — | — | — | — | **absent** | `from std.json import *` → *unable to locate module 'json'*; same for `csv`. A `std.collections` control compiles. |
| [bgreni/EmberJson](https://github.com/bgreni/EmberJson) @ main (0.4.0) | pure Mojo JSON | 2026-09-21 | pins `mojo ==1.1.0` | `emberjson` 0.3.4, unusable (see above) | Apache-2.0 | **0 errors, 0 warnings**; a `from_json[Document]` program builds and runs | The only serious JSON candidate. +68,520 B `__text`. Drags `EmberSerde` as a second git dep. |
| [Mojo-Mania/mm_csv](https://github.com/Mojo-Mania/mm_csv) @ main | pure Mojo CSV | 2026-09-16 | `mojo >=1.1.0.dev2026091105,<2` | no — install is `{ git = ... }` | MIT | **0 errors, 0 warnings**; a `CsvTable` program builds and runs | The best CSV code found. 1,561 lines. Repo is six days old, 0 stars, 1 watcher. +4,908 B `__text`. |
| [millfolio/csv.mojo](https://github.com/millfolio/csv.mojo) @ main | pure Mojo CSV | 2026-08-29 | pins `mojo ==1.0.0` | "mojoshelf tin" (git submodule) | Apache-2.0 | 0 errors, 0 warnings | 85 lines; `parse(text) -> List[List[String]]`. A `String` per field is the worst possible shape for Arrow. Not a dependency — an afternoon. |
| [Phelsong/mojo_csv](https://github.com/Phelsong/mojo_csv) @ main (1.6.4) | pure Mojo CSV | 2026-09-08 | `max >=26.5` | `mojo_csv` on modular-community, `.mojopkg`, unusable | MIT | **3 errors, 5 warnings** — `no matching function in call to 'parallelize'` ×2, the legacy parametric-closure API | Dead against 1.2. `__getitem__ -> String` anyway. |
| [mzaks/mojo-csv](https://github.com/mzaks/mojo-csv) @ main | pure Mojo CSV | 2025-08-03 | `modular >=25.5` | no | MIT | **35 errors** — pre-1.0 `fn`/`inout` syntax | Dead. Superseded by the same author's `mm_csv`. |
| [randyzwitch/dataframe_mojo](https://github.com/randyzwitch/dataframe_mojo) | whole dataframe engine | 2026-09-21 | Mojo 1.2 | git tag per Mojo minor | MIT | not compiled here | Not a dependency candidate — it is a rival engine with its own storage. Used below as the measurement of what a CSV reader costs. |
| lightbug_http | HTTP framework | — | — | — | — | — | Its JSON *is* EmberJson (*unverified*, from its docs). Not a separate candidate. |
| scottcgi/MojoJson | — | 2021-04-17 | — | — | MIT | — | Predates the Mojo language; unrelated to it despite the name. |
| yyjson | C, `dlopen` candidate | — | — | **not on conda-forge** | MIT | — | The anaconda API has no `yyjson` package on conda-forge (only `r-yyjsonr`). Rules it out as a `[package.run-dependencies]` line. |
| simdjson | C++ | — | — | conda-forge 4.6.11, all three marrow platforms | Apache-2.0 | — | `nm -gU libsimdjson.33.0.0.dylib` exports **zero** unmangled symbols — every export is `__ZN8simdjson…`. No C ABI to `dlopen`. |
| RapidJSON | C++ headers | — | — | conda-forge 1.1.0.post20250205 | MIT | — | The package contains **no** `.dylib`, `.so` or `.a` — headers only. Cannot be `dlopen`ed at all. |
| `arrow-c-glib` | C ABI over Arrow C++ | — | — | conda-forge 24.0.0 | Apache-2.0 | — | Technically the only candidate that returns Arrow. Practically disqualified; see below. |

## Option 2, the `dlopen`ed C library, in detail

This repository already binds seven C libraries through `utils/dylib.mojo` —
six for the five page codecs (brotli is `brotlienc` plus `brotlidec`) and
`libopendal_c` — so the machinery is a `LibSpec` line, not a project. The cost
is not the binding — it is the table on the Python side.
`python/marrow/compile.py` carries a second copy of every soname because the
wheel staging runs where no Mojo compiler exists,
`devkit/tests/test_libspecs.py` exists purely to stop the two drifting, and
`python/build.py` force-includes every library in that table into the wheel
because `delocate`/`auditwheel` walk load commands and a `dlopen`ed library
has none. An eighth library is a line in each of the two soname tables and a
conda dependency in `pixi.toml`'s `[package.run-dependencies]` and
`[dependencies]`. That is affordable — if there is a library worth binding.

There is not, for JSON. **yyjson** is the one that would have fit the pattern
exactly — pure C, MIT, one file, a stable C ABI — and it is not packaged for
conda, so it would arrive the way `libopendal_c` does: vendored, built from
source, `publish = false` in spirit, opt-in. §1.8b already records that
`libopendal_c` "cannot follow" the codecs into link-time linking for exactly
this reason, and that this is "the honest cost" of keeping two mechanisms.
Adding a second such library doubles that cost for a format marrow can parse
itself. **simdjson** and **RapidJSON** are C++: simdjson exports nothing but
mangled symbols and RapidJSON exports nothing at all, so either one needs a C
shim — a new compiled artifact marrow would have to build, ship and version for
three platforms, which is a different and much larger commitment than a
`LibSpec`.

**`arrow-c-glib` deserves its own paragraph, because on the criteria as stated
it wins and should still be refused.** It exports a genuine C ABI —
`garrow_csv_reader_new`, the whole `garrow_csv_read_options_*` family,
`garrow_json_reader_new`/`garrow_json_reader_read` for NDJSON — *and* the C
Data Interface: `garrow_record_batch_export`, `garrow_schema_export`,
`garrow_record_batch_reader_export`. It would hand marrow real Arrow arrays
through `marrow/c_data.mojo`, which already imports them, and it would answer
CSV and JSON in one binding. The disqualifier is its runtime closure.
`arrow-c-glib` depends on `libarrow`, `libarrow-acero`, `libarrow-dataset`,
`libarrow-flight`, `libarrow-flight-sql`, `libarrow-gandiva`,
`libarrow-substrait`, `libparquet` and `glib`; `libarrow` in turn depends on
`aws-crt-cpp`, `aws-sdk-cpp`, three `azure-*-cpp`, `libgoogle-cloud`,
`libprotobuf`, `orc`, `glog` and **`libopentelemetry-cpp`** — the package
`pixi.toml` already documents as crashing at process exit on macOS and causing
every test in the same binary to be reported as failed, which is why the dev
environment takes pyarrow from PyPI rather than conda-forge. `libarrow-gandiva`
pulls `libllvm21`. That is ~13.4 MB of direct downloads before the transitive
closure, LLVM and an AWS SDK, to read a comma-separated file, in a project
whose AOT lane is gated at kilobyte granularity. Refuse it.

## Option 3, write it, measured rather than asserted

The premise in the brief — that the tokenizer is not the hard part and
inference and widening are — **does not survive contact with a real Mojo
implementation.** `dataframe_mojo/csv.mojo` is 1,848 lines and targets Mojo
1.2, so it is the closest thing to a controlled measurement available. Its
inference pass, `infer_dtype`, is **70 lines**, plus `_is_integer_text` and
`_has_leading_zero` at about ten each. It is a straight-line scan over a sample
that clears three flags (`boolean`, `integer`, `floating`), bails to String on
a leading zero or an Int64 overflow, and then tries ISO date and
`datetime[us]`. The widening is not a lattice at all: `_CsvColumn` (183 lines)
holds a `kind` tag over `INT / TEMPORAL / FLOAT / BOOL / STRING` and a builder,
and promotion is a monotone walk down that order. Inference plus widening is
roughly 15% of the module. The other 85% is the tokenizer, the SIMD
`record_splits`, the mmap and parallel range jobs, the options surface, the
projection and the writer. `mm_csv`, which is tokenizer, index and writer and
has **no** inference at all, is 1,561 lines.

So the shape of the work is the opposite of the premise, and that is good news,
because **the 15% is the part marrow already owns.** `marrow/kernels/cast.mojo`
has `StringToNumKernel` (per-element `atol`/`atof`, over a `BinaryLikeArray`)
and `StringToBoolKernel`, both parameterised on a comptime `safe` flag: the
default `safe=True` raises on an unparseable value, `safe=False` nulls it —
which is what a lenient reader does with a field its inferred type rejects.
`marrow/expr/bindings.mojo` has `numeric_from_text[T]`, `bool_from_text` and
`string_from_text[T]` for the command line. `DynBuilder`
already constructs from a runtime `DynType`, which is the inference-to-builder
seam. What marrow does not have is the tokenizer — and that is the part no
candidate gives it in a form it can use.

Two corrections to `backlog.md` §1.2 fall out of this, and are applied there:

- It names `LittleEndian.fixed` among the hard parts marrow already has. That
  is the primitive for *binary* decoders; CSV and JSON are text and will never
  call it. The assets the entry should name instead are `StringToNumKernel` /
  `StringToBoolKernel` with `safe=False`, and `DynBuilder(dtype)`.
- "NDJSON is the same shape with a different tokenizer" is wrong in the part
  that matters. CSV's schema is positional and rectangular, so inference is per
  column over a sample of rows. NDJSON's schema is the union of key sets across
  records, with per-key type unification, absent keys meaning null, and nested
  values producing struct and list columns rather than flat ones. The tokenizer
  is the *least* different part of the two. CSV should be scheduled first and
  JSON costed separately, not as a variant of it.

A third correction, to `CLAUDE.md`'s Known Limitations, also applied: its
"conformance testing leans on PyArrow until Mojo has a JSON library" overstates
what a JSON library would buy. `devkit/integration.py` already parses the Arrow
JSON integration format with Python's stdlib `json`. What leans on PyArrow is
the step after that — the `_json_*` functions there convert the parsed columns
into pyarrow arrays, which reach marrow over the C Data Interface — and
`validate()`, which uses pyarrow as the comparison oracle. Both are Python
inside a Python harness. A Mojo JSON parser unblocks neither; at most it makes
a rewrite of working Python code possible, so conformance testing is no
argument for writing one.

## One thing neither candidate fits

`marrow/io/` exists because "a `ByteSource` is not always a memory map"
(`CLAUDE.md`), and both format readers go through `read_at`. `mm_csv`'s `CsvTable` takes
the whole document as a `String` and keeps it, because every field is a slice
of it — that is what makes reading free, and it also caps a document at 2 GiB
(*unverified*, from its README) and forbids a source that is a round trip.
EmberJson's `read_lines` takes a `FileHandle`. Neither reads through
`marrow/io`, so either one arrives as a reader that bypasses the seam every
other format in the tree goes through. (For EmberJson that turned out to be the
wrong entry point: its `Parser` takes a byte span — see the JSON
recommendation.)

## Recommendation — CSV: write it

Three facts decide it.

**No candidate returns Arrow, and the two that compile return exactly what
marrow's own tokenizer would produce.** `mm_csv` hands back a borrowed
`StringSlice` per field with an `is_quoted` flag; `millfolio/csv.mojo` hands
back `List[List[String]]`, a heap allocation per cell. Against the stated
criterion — does the candidate produce Arrow arrays, or something needing a
full copy — the answer is the same for both, and the half marrow would write
regardless (inference, widening, builders, the `Relation` node, the
`ByteSource` integration) is the half with all the marrow-specific design in
it.

**The dependency cannot be a package, so it is maintenance without a solver.**
It would be a git rev of a six-day-old, zero-star repository, rebuilt by
marrow's CI on every Mojo bump, with no channel to pin against and no version
range that means anything.

**The thing worth taking from `mm_csv` is a design, and a design can be read.**
Its index-then-index-arithmetic shape, its SIMD scan and its measured numbers
(0.8 ns per field; the SIMD scan 2.0-8.9x the scalar walk on an M4 — both
*unverified*) are a good template, and its 33-case RFC-4180 suite is a good
oracle to test marrow's reader against. If the SIMD scan turns out to be worth
not re-deriving, vendoring `csv_table.mojo` under its MIT licence with
attribution is a copy, not a dependency: it costs no solve, no wheel staging
and no drift test, and it stays inside the size gate. Prefer that to a
dependency edge.

The build is not small — call it `mm_csv`-sized for the tokenizer plus
`dataframe_mojo`-sized for inference, so roughly 1,000-1,500 lines — but it is
ordinary work with a known answer, against an RFC and a DuckDB oracle the
golden corpus already knows how to consult.

## Recommendation — JSON: take EmberJson's tokenizer, through a fork

*Revised 2026-09-25, after the first version of this section — "write a narrow
NDJSON reader; use EmberJson as the oracle, not the dependency" — was checked
again and found to rest on a wrong premise. This is what was built.*

**The premise was that EmberJson's value is its DOM.** It is not.
`from_json[T]` picks one of three paths at compile time — the mutable `Value`
tree, the `Document` tape, or, for any other `T`, EmberSerde's reflection
driving EmberJson's hand-written `Parser` directly, with no tree at all — and
`Parser` is exported. Its token methods (`expect_open`, `expect_string`,
`expect_int[dt]`, `expect_float[dt]`, `expect_bool`, `expect_null`, `peek`,
`skip_value`, `expect_float_bytes`) are exactly the pull tokenizer a columnar
reader wants, and they take a byte span, so a block read through `ByteSource`
parses where it lies. The objection in "One thing neither candidate fits" was
about `read_lines`, the wrong entry point.

**So the choice was "take the tokenizer" against "write roughly 5,000 lines of
one"**, and the lines in question are the ones with a single right answer:
UTF-8 validation, escape and surrogate decoding, a correctly rounded float
parser (its slow path alone is 1,737 lines), validated skipping. What marrow
writes either way — inference, the key-set union, builders, blocks — it wrote.

**The version problem is solved by forking, not by luck.** Upstream pins a
release (`mojo ==1.1.0`) and a `.mojoc` loads only under the compiler that
built it, so `kszucs/EmberJson` and `kszucs/EmberSerde` carry one commit each,
on top of upstream, that pins our exact nightly. pixi builds both from git into
the environment; both upstream suites pass unchanged on it (552 and 301 tests
on 2026-09-27). A `mojo` bump is now three bumps — see `CLAUDE.md`. One
upstream defect turned up on the way — `significant_digits` read one byte past
an unpadded input ending in an all-zero mantissa of 20+ characters, reported
by ASAN — and upstream has since fixed it independently; the fork carries the
test that pins it.

**Where it lives.** Two measurements on the 1.2 compiler decided the layout:
an import is resolved while parsing, even inside a `comptime if False`, so no
`-D` switch can make a dependency optional; and a precompiled package needs
every package *any* of its modules imports, even for a consumer that imports
an unrelated module. So the reader lives in `marrow/json/`, and nothing else
in `marrow` imports it — `marrow.expr` reaches it only through `ExternalScan`,
a node holding one function pointer — which keeps a program built from source
free of EmberJson unless it reads JSON. A precompiled marrow needs EmberJson
either way, which is what publishing the forks is for.

**The size is paid only by binaries that read JSON.** A prototype reading
NDJSON into typed columns through `Parser` was +50,032 bytes of `__text`
against the 1,180-byte baseline (+36,024 without `skip_value`, which validates
what it skips); the `Value` tree path was +68,788, the `Document` path
+77,736.

## Summary

| | Take a Mojo package | `dlopen` a C library | Write it |
|---|---|---|---|
| **CSV** | no — `mm_csv` is good code in the wrong shape, unpackageable, six days old | no — nothing exists but `arrow-c-glib`, which drags LLVM and a crashing opentelemetry | **yes** — read `mm_csv` for the design, borrow its RFC-4180 cases |
| **JSON** | **yes** — EmberJson's `Parser`, from a fork pinned to our nightly, in `marrow/json/` and imported nowhere else | no — yyjson unpackaged, simdjson and RapidJSON have no C ABI, `arrow-c-glib` as above | the reader around the tokenizer: inference, builders, blocks |
