# Performance baseline — before the optimisation wave

The fixed reference point for attributing each optimisation's contribution.
Every number here was measured on `alpha` @ `9dc12a2`, after projection
pushdown (P1) landed and **before** O1–O4. Each optimisation is measured in its
own worktree against its own microbenchmark; this file is what lets the
*cumulative* effect be attributed at the end.

Machine: 16 cores, 128 GB. `hits_0.parquet`, 1,000,000 rows x 105 columns.
marrow built `-O3` (`build_python`). Medians of 5 interleaved repeats.

## End-to-end, marrow vs polars vs duckdb

marrow runs **single-threaded** (`ExecContext` defaults to `num_threads=1` and
the binding passes no context). polars defaults to 16 threads, so both columns
are given: the 1-thread column is the like-for-like comparison, the 16-thread
column is what a user gets out of the box today.

| query | marrow | polars 16T | polars 1T | duckdb | vs polars 1T |
|---|---:|---:|---:|---:|---:|
| Q1 `count(*)` | 9.2 ms | 0.4 ms | 0.3 ms | 0.9 ms | 27.0x |
| Q3 sum+count+avg | 12.3 ms | 1.4 ms | 1.9 ms | 2.4 ms | 6.3x |
| Q5 `nunique(UserID)` | 12.9 ms | 3.9 ms | 10.1 ms | 6.3 ms | 1.25x |
| Q7 min/max EventDate | 9.1 ms | — | — | 1.6 ms | — |
| Q8 group int + order | 5.6 ms | 1.6 ms | 1.1 ms | 1.7 ms | 4.9x |
| Q9 group + nunique top10 | 12.7 ms | 6.8 ms | 12.1 ms | 8.3 ms | **1.04x** |
| Q17 group 2 keys hi-card | 39.6 ms | 12.2 ms | 28.5 ms | 9.9 ms | 1.33x |
| Q21 `LIKE '%google%'` | 334.5 ms | 41.1 ms | 58.6 ms | — | 5.6x |
| Q31 group 2 keys + 3 agg | 22.8 ms | 10.3 ms | 24.8 ms | 13.1 ms | **0.88x** |
| Q34 `GROUP BY URL` | 101.9 ms | 58.5 ms | 87.7 ms | 36.4 ms | 1.12x |
| **TOTAL** | **560.7 ms** | **136.1 ms** | **225.0 ms** | **82.8 ms** | **2.42x** |

Serial against serial, marrow is already at parity or ahead on real group-by
work (Q31 0.88x, Q9 1.04x, Q34 1.12x). The aggregate 2.42x is carried by a few
outliers, and the profiles below say what they are.

Full-suite reference (`bench_clickbench.py`, 41 queries all three engines run):
marrow 13,935 ms / polars 790 ms / duckdb 1,595 ms — **17.6x** and 8.7x. That
table's marrow column predates projection pushdown; the per-query numbers above
do not.

## Where the time goes — profiles

`pixi run profile --sample clickbench-qN`, main-thread samples. The profile
build is `-O1` (`-O3` inlines the frames away), so these attribute *where*, not
*how much*; absolute numbers come from the table above.

**Q1 `COUNT(*)` — 7,828 samples**

| share | frame |
|---|---|
| 41.2% | `std::ffi::_DLHandle::_dlopen` |
| 39.5% | `ArcPointer.__deinit__` of `CompressionLibs` (the matching `dlclose`) |
| 5.4% | `PrimitiveScalar::repeat` — `count_star()` materialising an N-element constant column |
| 3.3% | `AggState::update` |

~80% is opening and closing the Parquet codec shared libraries.
`reader.mojo:2187` constructs `CompressionLibs` **per read**.

**Q21 `LIKE '%google%'` — 8,233 samples** (largest single query cost)

| share | frame |
|---|---|
| 25.8% | `std::memory::alloc::_alloc_bytes` |
| 15.7% | `std::collections::list::List::_realloc` |
| 9.8% | `string.mojo:774` — the actual `StringSpan` comparison |
| 5.3% | `string_span.mojo:1867` |
| 3.8% | `cast.mojo:813` — `_check_utf8` |

~41% allocating and growing, ~15% doing the string matching it exists to do.

**Q34 `GROUP BY URL` — 2,596 samples**

| share | frame |
|---|---|
| 11.2% | `dlopen` + 11.2% `dlclose` |
| 11.5% | `cast.mojo:813` — `_check_utf8` |

## The four targets

| # | target | evidence | agent |
|---|---|---|---|
| O1 | `CompressionLibs` opened *and* closed per read | ~80% of Q1, ~22% of Q34 | codec caching via `_Global` |
| O2 | `_check_utf8` scalar-loops 1M rows per `cast(string)`; the cast itself is already zero-copy | 11.5% of Q34, 3.8% of Q21 | expose `safe=`, vectorise |
| O3 | string kernel allocation churn | ~41% of Q21 | reserve capacity |
| O4 | 16 cores idle — `num_threads=1`, unreachable from Python | whole suite | expose `num_threads` |

A fifth is recorded but not yet scheduled: `count_star()` allocates a
million-element column of `1` to count rows the grouper already counted (5.4% of
Q1). It is sequenced after O4 because both touch `marrow/expr/`.
