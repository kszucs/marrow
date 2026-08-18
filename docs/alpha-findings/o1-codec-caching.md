# O1 — the Parquet codec libraries were opened once per *read*

**Status:** fixed on `worktree-agent-aa75075295de78834` (commit `perf(parquet):
open the codec libraries once per process`). Not merged.

## The defect

`marrow/utils/compression.mojo`'s `CompressionLibs` held six
`Optional[OwnedDLHandle]` — zstd, snappy, lz4, zlib, brotlienc, brotlidec —
opened lazily on first use and `dlclose`d by the destructor. Its own docstring
said "cached per read/write", and that is what it did: `ParquetFile` allocated
one per worker in `read` (`reader.mojo:2187`) and `FileWriter` one per file
(`writer.mojo:795`), and every one of them was destroyed with its file.

Nothing else in a marrow-only process holds those libraries, so the `dlclose`
dropped the last reference and dyld genuinely unmapped the image. The next read
re-mapped, re-bound and re-initialised it. **One open/close cycle costs about
0.9 ms.**

That is a per-*read* cost, so it is invisible on a big file and dominates a
small one — and a query engine opening the same file for every query pays it
every query.

## The benchmark (written before the fix)

`marrow/parquet/tests/bench_parquet.mojo`, two new cases:

- `bench_read_small_snappy` — `read_table` on a 1,000-row Snappy file, in a
  loop. Three tiny pages decode in microseconds, so what is left is the fixed
  per-read cost.
- `bench_read_small_uncompressed` — the same file written uncompressed. It never
  touches a compression library, so it is both the *baseline* the snappy case
  should approach and a drift control for the box.

```
pixi run -e dev pytest marrow/parquet/tests/bench_parquet.mojo --benchmark -k small
```

| case (median, us)              | before | after | delta |
|--------------------------------|-------:|------:|------:|
| `bench_read_small_snappy`        |  921.4 |  31.7 | **-96.6% (29x)** |
| `bench_read_small_uncompressed`  |   24.0 |  23.5 | -2.1% (control, within drift) |

897 us of the 921 us was `dlopen` + `dlclose`. Afterwards the snappy case sits
8.3 us above the uncompressed control, which is the actual decompression.

## The fix

Split the struct along the line the old docstring was really drawing — *"held by
one worker at a time; the `dlopen` handles and the reused size cell are not
thread-safe to share"* conflated two very different things.

- **`_CodecHandles`** — all six libraries, opened together, stored behind the
  stdlib's `_Global` (`std/ffi/__init__.mojo:1023`, the mechanism
  `std/utils/_ansi.mojo` and `std/random/_rng.mojo` use). Allocated by the
  runtime on the first `get_or_create_ptr()` and never `dlclose`d before process
  teardown.
- **`CompressionLibs`** — unchanged in name and API, but now only the block
  calls plus snappy's reused size out-param. Still one per worker, because that
  cell is genuinely per-call mutable state.
- **`_Library`** — one handle plus, on failure, the error `_try_find_dylib`
  raised. `get()` returns a non-owning `_DLHandle` borrow, which is sound only
  because the owner is a process-lifetime global.

### Never `dlclose`

Deliberate. Process-lifetime handles are what any library that `dlopen`s its
optional dependencies does, there is nothing to reclaim (six images, a few
hundred KB), and the alternative — reference-counting readers — reintroduces
exactly the unmap/remap cycle this fixes. `_Global` registers a deinit with the
runtime, so the handles are released at teardown without marrow doing anything.

### What is shared, and why that is safe

`_CodecHandles` is **written exactly once**, inside the `_Global` initializer,
and only read afterwards. `OwnedDLHandle.call` takes `self` immutably and
resolves the symbol with `dlsym`, which is thread-safe, so a concurrent
decompress is a concurrent *read* of an immutable structure — the property the
old comment attributed to the whole struct but which only ever held for the
handles.

The one thing that is not obviously safe is *creating* the global. `_Global`
carries an explicit warning that it vends shared mutable pointers without
locking, and `KGEN_CompilerRT_GetOrCreateGlobal` is closed-source, so racing
first-touch cannot be ruled out by inspection. `ParquetFile.read` therefore
calls `CompressionLibs.preload()` on the calling thread before it dispatches
workers. After that every worker's use is a pure read. The writer is
single-threaded and needs nothing.

### Errors are recorded, not raised

A `_Global` initializer cannot raise, and eagerly opening all six means a box
missing `libbrotlienc` would fail while opening a set it only needs zstd from.
So `_Library.open` catches and stores the error text, and `get()` re-raises it —
same message, same place (the first page that needs that codec). Missing codecs
still produce an error, never an abort.

### Laziness is preserved at the process level

`_libs()` resolves the global on first *codec call*, not in `CompressionLibs
.__init__` — every read constructs one whether or not its pages are compressed.
`ParquetFile.read`'s `preload()` is likewise guarded on some selected chunk
having a non-zero codec. Verified with `DYLD_PRINT_LIBRARIES=1` around a
`ma.read_parquet` of a 1,000-row file:

```
compression=none     -> (nothing)
compression=snappy   -> libzstd.1.5.7 libsnappy.1.2.2 liblz4.1.10.0
                        libbrotlienc.1.2.0 libbrotlicommon.1.2.0 libbrotlidec.1.2.0
```

## End-to-end, and a correction to the profile that started this

The brief reported, from `pixi run profile --sample clickbench-q1`, that 3,228
of 7,828 main-thread samples were in `_dlopen` and 3,093 more in the matching
`CompressionLibs` teardown — ~80% of `SELECT COUNT(*) FROM hits`. Re-recorded
here at 300 repeats, the shape reproduces exactly: **9,488 samples, 3,953 in
`_dlopen` (41.7%) and 3,783 in the `ArcPointer[List[CompressionLibs]]` deinit
(39.9%) — 81.6%.** After the fix the same profile is **2,391 samples, 482 of
them (20.2%) in a single `_Global::get_or_create_ptr` that runs once for all 300
runs, and zero `dlclose`.**

**But that 80% is a profiler artifact and the real end-to-end win is ~9%, not
~80%.** The same `-O1 -g` build of `libmarrow.so`, same 300 repeats, differs by
5x depending only on whether `sample` is attached:

| q1, pre-fix `-O1 -g` build | ms/run |
|---|---:|
| under `sample` | 37.83 |
| not under `sample` | 7.85 |

Attaching a sampler makes every `dlopen`/`dlclose` far more expensive (each
image load and unload has to be reported to the sampling task), so a workload
whose hot spot *is* image loading is inflated out of all proportion. The profile
was right about **where** and wrong about **how much**.

The honest end-to-end number, `-O3`, marrow-only process, three interleaved
pairs of 300 repeats:

| q1 | before | after |
|---|---:|---:|
| run 1 | 9.03 ms | 8.05 ms |
| run 2 | 9.25 ms | 8.49 ms |
| run 3 | 9.42 ms | 8.43 ms |
| median | **9.25 ms** | **8.43 ms** (-8.9%) |

Every pair favours the fix, which is what makes -8.9% readable at all on a box
that drifts ±5-8%.

`q34` (`GROUP BY URL`), the same protocol at 40 repeats, does **not** move:
104.5 ms -> 103.9 ms, medians, with individual runs straddling in both
directions. The brief's "~22% of q34" is the same profiler inflation. This is
consistent, not contradictory: the saving is a fixed ~0.9 ms per query, which is
9% of a 9 ms query and 0.9% of a 104 ms one.

The three-engine table (`pixi run -e bench pytest
python/marrow/tests/bench_clickbench.py --benchmark`) shows q1 at 9.78 ms after
the fix against a pre-fix 10.1 ms — a smaller and noisier gap, because that
process also imports polars and duckdb, which keep the codec libraries resident
and so were already absorbing most of the `dlclose` cost. It is the marrow-only
process that pays, and that is the process a marrow user runs.

## Where this actually matters

Not on ClickBench, where one query reads one large file. It matters wherever the
number of `read_table` calls is large relative to the bytes each returns: a
partitioned dataset of many small files, a repeated point query, a test suite.
The microbenchmark is the honest measure of that shape, and it is 29x.

## Verification

| gate | result |
|---|---|
| `pixi run -e dev precompile` | 0 errors, 0 warnings |
| `pixi run -e dev pytest marrow/parquet/tests/test_codecs.mojo` | 19 passed |
| `pixi run -e dev pytest marrow/parquet/tests` | 208 passed, 6 skipped |
| `pixi run -e bench pytest python/marrow/tests/test_clickbench.py` | 85 passed, 1 skipped |
| `pixi run python3 benchmarks/binary_size/check_gate.py` | OK — no gate grew more than 0.5% (largest move +0.464% on `query_dynvalue`) |
