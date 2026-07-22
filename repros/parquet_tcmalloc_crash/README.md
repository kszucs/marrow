# Parquet read → `statistics()` tcmalloc crash

A heap-state-dependent crash inside the Mojo runtime's bundled tcmalloc,
triggered by a specific sequence of Parquet reads. It is **not** a correctness
bug in any single decode path — every step passes in isolation, and the whole
test suite is clean under AddressSanitizer. It surfaces only in a normal
(tcmalloc) build.

This is the root cause behind the 34 "failures" in
`marrow/parquet/tests/test_reader.mojo`: one crash aborts the whole `TestSuite`
runner, so every test in the file is reported as failed.

## Reproduce

```bash
# Crashes (SIGSEGV inside tcmalloc during an allocation):
pixi run -e dev mojo run -I . -D ASSERT=all repros/parquet_tcmalloc_crash/repro.mojo

# Clean — no error — under AddressSanitizer (different allocator):
pixi run -e asan pytest --mojo --asan marrow/parquet/tests/test_reader.mojo
```

Observed with **Mojo 1.0.0b3.dev2026072217** on macOS/arm64.

## The minimal trigger

The reproducer performs, in order:

1. Read a flat `int64 / float64 / string` table and extract its columns
   (`to_batches()` → `columns[i].copy()` → `as_int64()/as_float64()/as_string()`).
2. Read a 6000-row **dictionary-encoded string** table (many small data pages).
3. Read a 40-column **wide** `int64` table.
4. Call `ParquetFile.statistics()` — **crashes here**, in
   `tcmalloc::SizeMap::num_objects_to_move` under `_alloc_bytes`.

Removing **any** of steps 1–3 makes the crash disappear. All three distinct
allocation/free patterns are required to arrange tcmalloc's freelists such that
step 4's allocation corrupts.

## Why it looks the way it does

- **Heap-state-dependent, not logic.** Each step is correct on its own; only the
  precise combination triggers it. A 60× loop of the *same* read does **not**
  crash — it needs the *diversity* of the three decode paths, not volume.
- **tcmalloc-specific / ASAN-invisible.** ASAN's allocator never reproduces it,
  and ASAN reports zero errors across the suite. The crash is in the runtime's
  tcmalloc size-class machinery.

## Ruled out

- **Optimization level** — crashes at both `-O0` and `-O1`.
- **`munmap`** (the `MappedFile` unmap) — disabling it does not help.
- **Decode buffer over-read** — a real DataPage-v2 word-load overrun *was* found
  and fixed separately (see the `fix(parquet): pad DataPage v2 buffer` commit);
  this crash persists after that fix and is a distinct issue.
- **Aligned alloc/free mismatch** — `_malloc(alignment=64)` is paired with
  `_free`'s `pop.aligned_free`; they match.

## Status

The evidence points below marrow, at the Mojo runtime's allocator interacting
with this allocation pattern. This directory exists to carry a self-contained
reproducer for an upstream (`modular/modular`) report.
