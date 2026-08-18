# G3 — regression check: `alpha` vs `mojo-1.0-upgrade`

**Verdict: one real regression, small and localised.**

- **Binary size: yes, a regression.** Every one of the 11 gates grew `__text` by
  **+10 KB to +18 KB** (+0.33% to +1.28%). ~93% of it is one change — the C1
  builder type-resolution fix, which turned `BinaryLikeBuilder::extend`'s two
  instantiations into a six-way set and grew `DynBuilder::_dispatch_mut` by
  7,184 bytes. The CI gate still passes, but only because `baseline.json` is
  recorded well above the current tree; `query_dynvalue`, the one gate whose
  recorded baseline *is* current, went from ±0.000% to **+0.335%** — two thirds
  of the 0.5% budget consumed by one branch.
- **Performance: yes, small.** `filter` is **~2.3% slower** after normalisation,
  and the F1 `compressed_store` fix is the cause. All 9 filter cases moved the
  same direction; no untouched benchmark moved systematically.
- **ClickBench: no regression.** 42/43 PASS in 18.1 s, max query 0.98 s, every
  per-query median within ±0.05 s of `docs/alpha-clickbench-coverage.md`.

Measured 2026-08-18 on osx-arm64 (Apple Silicon), in worktree
`/Users/kszucs/Workspace/marrow/.claude/worktrees/agent-a9337eb747afe1a8d`
(`alpha` @ `557fc34`) against `/tmp/marrow-baseline` (`mojo-1.0-upgrade` @
`fb31b2d`, detached). `git merge-base alpha mojo-1.0-upgrade` = `fb31b2d`,
confirmed — the base branch tip *is* the fork point, so every delta below is
`alpha`'s own work and nothing else's.

`pixi run -e dev precompile` on `alpha`: 17.2 s, **0 errors, 0 warnings**.

---

## 1. Binary size

`pixi run binary_size` on each branch, full 11-gate sweep, `-O3 -g0` then
`strip`. **All numbers are the `__text` section**, per
`benchmarks/binary_size/README.md` — the stripped file column is page-granular
(16 KB) and cannot resolve these deltas. `benchmarks/` is byte-identical
between the two branches (`git diff --stat fb31b2d..alpha -- benchmarks/` is
empty), so the gate sources and `baseline.json` are the same on both sides and
`baseline.json` was not modified.

| gate | `mojo-1.0-upgrade` | `alpha` | Δ `__text` | % |
|---|---:|---:|---:|---:|
| `query_streaming` | 1,423,236 | 1,439,756 | **+16,520** | +1.161% |
| `query_arith` | 1,429,608 | 1,446,096 | **+16,488** | +1.153% |
| `query_exprs` | 1,508,540 | 1,524,156 | **+15,616** | +1.035% |
| `query_sort` | 4,387,572 | 4,402,996 | **+15,424** | +0.352% |
| `query_join` | 1,454,504 | 1,464,616 | **+10,112** | +0.695% |
| `query_scan` | 2,516,712 | 2,532,916 | **+16,204** | +0.644% |
| `query_scan_typed` | 1,989,616 | 2,005,820 | **+16,204** | +0.814% |
| `query_streaming_agg_fused` | 1,371,312 | 1,388,848 | **+17,536** | +1.279% |
| `query_streaming_agg` | 1,886,260 | 1,903,988 | **+17,728** | +0.940% |
| `query_dynvalue` | 4,871,156 | 4,887,476 | **+16,320** | +0.335% |
| `query_runtime` | 4,870,068 | 4,886,260 | **+16,192** | +0.332% |

Symbol counts rose by 6–10 in every binary. The growth is *uniform*, which is
the tell that it is shared core code, not any one gate's feature.

### Gate verdict — PASSES

`pixi run python3 benchmarks/binary_size/check_gate.py` on `alpha`, exit 0:

```
gate                             baseline     measured      delta      pct
query_streaming                 1,484,652    1,439,756    -44,896  -3.024%
query_join                      1,507,836    1,464,616    -43,220  -2.866%
query_streaming_agg_fused       1,417,476    1,388,848    -28,628  -2.020%
query_streaming_agg             1,932,404    1,903,988    -28,416  -1.470%
query_dynvalue                  4,871,156    4,887,476    +16,320  +0.335%

OK: no gate grew more than 0.5%.
```

Every `measured` value reproduces the sweep exactly (`query_streaming`
1,439,756 in both), so the builds are deterministic and the two tools agree.

**Reconciling this with the earlier "everything shrinking" run.** That run
(`query_streaming −4.137%`, `query_join −3.537%`, `query_streaming_agg_fused
−3.257%`, `query_streaming_agg −2.388%`, `query_dynvalue ±0.000%`) is
*confirmed* as a correct reading of what `check_gate.py` measures — and it does
not contradict the table above, because **`check_gate.py` compares against the
recorded `baseline.json`, not against `mojo-1.0-upgrade`**. Four of the five
recorded values sit above the current tree on *both* branches, so the gate
reports a shrink from either side. Only `query_dynvalue`'s recorded 4,871,156
is current — it is exactly `mojo-1.0-upgrade`'s measured value — and that is
the gate that moved: **±0.000% then, +0.335% now**. The C1, F1 and
Exhausted-fix merges are the difference, as expected.

Reading the gate as "alpha shrinks the binary" would be wrong. Against the
branch it forked from, `alpha` grows every gate.

### Where the ~16 KB went

Per-symbol attribution on `query_streaming` (the floor gate, and the one whose
own source is unchanged), by differencing `nm -n` text-symbol extents. Total
attributed: **+16,544** against a measured `__text` delta of +16,520.

| Δ bytes | symbol | change |
|---:|---|---|
| **+7,184** | `builders::DynBuilder::_dispatch_mut` | grew |
| **+4,800** | `builders::BinaryLikeBuilder::extend(…, DynArray)` ×2 | new |
| **+3,312** | `builders::BinaryLikeBuilder::extend[T](…, BinaryLikeArray[U])` | 4 new (full `T`×`U` cross product) − 2 removed |
| +396 | `query_streaming::main()` | grew (inlining) |
| +116 | `expr::relations::Project::to_processor…_closure_0` | new |
| +108 | `std::builtin::error::Error::write_to` ×3 | new — the new `raise` paths |
| **+700** | `views::BufferView::filter` ×13 instantiations, +20…+60 each | grew |

So **~15.3 KB of the ~16.5 KB is the C1 builder fix alone**: replacing the two
mistyped `extend` instantiations with a correctly-resolved six-way set, plus
the `DynBuilder::_dispatch_mut` ladder that now reaches them. The three plan
verbs and five null ops cost ~220 bytes between them in this gate; the F1
`compressed_store` fix costs ~700 bytes spread over the 13 `filter`
instantiations.

`query_join` is the outlier at +10,112 rather than ~+16,300 — `equal_any`'s
rewrite (dropping `StringEqKernel` for a `dispatch_binarylike` arm over
`_bytes_equal`) evidently recovers ~6 KB there. Not investigated further.

---

## 2. Performance

### What was run

```
pixi run -e dev pytest --benchmark --mojo-timeout 3600 \
    marrow/kernels/tests/bench_filter.mojo marrow/kernels/tests/bench_sort.mojo
```

23 cases, one `-O3` driver, **`23 passed` in all five runs**. Both files in one
selection deliberately: same compilation unit, same batch, so the untouched
rows are the tightest possible normaliser for the touched ones.

- **Touched (9):** `bench_filter*` — the only benchmarks that reach
  `BufferView.compressed_store`, which F1 changed.
- **Anchors (14):** `bench_take*` (4) and `bench_sort*` (10). `take` is a
  gather and never calls `compressed_store`; `sort`/`sort_indices` are
  untouched by this branch entirely. Neither can be affected by anything in
  the diff.

**Interleaved A/B/A/B/A**, in this order: alpha, base, alpha, base, alpha —
five runs, so machine drift over the measurement window is bracketed rather
than aliased onto the branch. Per-case value = median of the run medians
(3 alpha, 2 base).

**Rule-3 check:** `grep -ci 'never used'` = 0 on all five logs. No capture was
dropped; every number below came from a body that actually read its data.

### Normalisation

| set | n | median alpha-vs-base | range |
|---|---:|---:|---|
| anchors (`take`, `sort` — untouched) | 14 | **+0.30%** | −10.04% … +7.74% |
| filter (touched) | 9 | **+2.55%** | +0.68% … +4.03% |

**Batch offset = +0.30%**, subtracted from every raw delta below. The anchor
*range* is the honest picture of this box: ±8–10% on a single case, exactly as
CLAUDE.md warns. **No individual case below is significant on its own.**

### Touched — `filter`

| benchmark | alpha (median) | base (median) | raw Δ | **normalised Δ** |
|---|---:|---:|---:|---:|
| `bench_filter50pct_100k` | 7.616 us | 7.321 us | +4.03% | **+3.73%** |
| `bench_filter50pct_1m` | 73.221 us | 70.816 us | +3.40% | **+3.10%** |
| `bench_filter10pct_1m` | 15.831 us | 15.352 us | +3.12% | **+2.82%** |
| `bench_filter50pct_nulls_1m` | 79.957 us | 77.860 us | +2.69% | **+2.40%** |
| `bench_filter50pct_nulls_100k` | 8.389 us | 8.180 us | +2.55% | **+2.25%** |
| `bench_filter90pct_1m` | 128.926 us | 125.744 us | +2.53% | **+2.23%** |
| `bench_filter90pct_100k` | 13.152 us | 12.849 us | +2.36% | **+2.06%** |
| `bench_filter50pct_10k` | 712.4 ns | 702.1 ns | +1.47% | +1.17% |
| `bench_filter10pct_100k` | 1.155 us | 1.147 us | +0.68% | +0.38% |

Units differ per row (ns / us) — pytest-benchmark scales each row
independently; all values above are normalised to a common unit before the
percentage is taken.

### Anchors — `take`, `sort`

| benchmark | alpha (median) | base (median) | raw Δ | normalised Δ |
|---|---:|---:|---:|---:|
| `bench_sort_multi_3col_1m` | 32.862 ms | 30.503 ms | +7.74% | +7.44% |
| `bench_sort_int32_10k` | 221.278 us | 215.036 us | +2.90% | +2.61% |
| `bench_take_100k` | 64.923 us | 63.320 us | +2.53% | +2.23% |
| `bench_sort_multi_2col_10k` | 224.961 us | 219.431 us | +2.52% | +2.22% |
| `bench_sort_multi_2col_1m` | 18.897 ms | 18.505 ms | +2.12% | +1.82% |
| `bench_take_1m` | 653.289 us | 643.657 us | +1.50% | +1.20% |
| `bench_sort_float64_100k` | 2.128 ms | 2.115 ms | +0.60% | +0.31% |
| `bench_sort_int64_100k` | 2.025 ms | 2.025 ms | −0.01% | −0.31% |
| `bench_sort_int32_1m` | 11.139 ms | 11.157 ms | −0.16% | −0.46% |
| `bench_sort_float64_1m` | 22.220 ms | 22.298 ms | −0.35% | −0.65% |
| `bench_sort_int64_1m` | 21.836 ms | 21.999 ms | −0.74% | −1.04% |
| `bench_sort_int32_100k` | 1.085 ms | 1.095 ms | −0.92% | −1.22% |
| `bench_take_nulls_1m` | 1.723 ms | 1.762 ms | −2.21% | −2.50% |
| `bench_take_parallel_1m` | 309.214 us | 343.719 us | −10.04% | −10.34% |

### Conclusion: a real ~2.3% regression in `filter`

No single filter case clears the ~8% noise bar, and this report does not claim
any of them does. The finding rests on the **group**, and specifically on the
sign distribution:

- **9 of 9 filter cases are slower.** Under a null of pure noise that is
  p = 2⁻⁹ ≈ **0.002**.
- **7 of 14 anchors are slower, 7 faster** — a textbook null result, from the
  same five runs, the same driver, the same batches.
- **7 of the 9 filter cases land inside +2.06% … +3.73%**, a 1.7-point band.
  The anchors span 17.8 points. Noise does not cluster; a constant multiplier
  does.

**Best estimate: `filter` is ~2.3% slower on `alpha`** (median normalised
+2.25%; +2.32% over the eight 100k/1m cases alone, dropping the noisiest 10k
row). The two cases nearest zero (`filter10pct_100k` +0.38%,
`filter50pct_10k` +1.17%) are also the two smallest and noisiest — 1.1 us and
0.7 ns-scale working sets, where fixed overhead dominates the loop the change
touches.

**Cause: the F1 fix, as predicted in the brief.** `compressed_store`'s
dispatch condition went from `cnt <= sparse_threshold` to
`cnt <= sparse_threshold or self._length <= cnt`, adding a second term — and a
load of `self._length` — to the branch that runs once per 64-bit selection
word. The +20…+60 bytes added to each of the 13 `BufferView::filter`
instantiations (§1) is the codegen footprint of exactly that. Note the actual
*path change* is negligible on its own: only the final word of a filter falls
back to the sparse route, 1 word in 15,625 at 1M rows. The cost is the
per-word branch and its effect on the hot loop's codegen, not the sparse
fallback.

**This is the correctness fix working as intended** — it closes a one-element
heap overflow. ~2.3% on `filter` is the price; the branch is not reported here
as something to revert, only as something now measured.

### Not measured

- **`bench_join`** was not run, so `equal_any`'s rewrite is unmeasured at
  runtime. Reasoning from the source only: it adds one `DataType` comparison
  per `equal_any` *call* (array-level, not element-level), so the per-row cost
  is zero and the change should not be observable. That is an inference, not a
  measurement. Its −6 KB `__text` effect on `query_join` *is* measured.
- **`bench_groupby`** was not run. `marrow/kernels/tests/test_groupby.mojo` is
  the file with the known all-cases hang, and the C1 builder fix that it would
  cover is a correctness fix whose cost already shows up in the size table.
- **Load.** The box was shared with at least two other agents compiling
  throughout. The A/B/A/B/A interleave plus the 14-case anchor set is the
  defence; the anchor spread (−10% … +8%) is the measured evidence of how much
  contention there was.

---

## 3. ClickBench end-to-end — no regression

```
pixi run -e dev python python/marrow/tests/clickbench_alpha.py
```

Run **three times**. **42/43 PASS in all three.** Sum of per-query medians
**18.14 s**; slowest query **0.98 s** (q23). Median of three runs vs the `s`
column in `docs/alpha-clickbench-coverage.md` (recorded at `8365395`, 39/43):

- **41 of 42 timed queries within ±0.05 s**, most of them slightly *faster*.
- Largest move: **q43 −0.15 s** (`DATE_TRUNC` group key), an improvement.
- **Three queries newly PASS** that the doc records as ABORT: **q11, q12**
  (0.39 s each — the C1 `binary` filter/group-by crash) and **q24** (0.62 s —
  the `SELECT *` sort/take over `binary` columns). That is 39/43 → 42/43.
- q29 remains UNSUPPORTED (no regex kernel).

**One trap worth recording.** In run 1, q33/q34/q35/q36 read +0.19/+0.12/+0.07/
+0.13 s against the doc — a clean-looking "group-by over the whole table got
slower" story. Runs 2 and 3 put all four back at 0.39–0.41 s. It was a sibling
agent's compile. A single ClickBench run is not evidence.

There is nothing to compare against on `mojo-1.0-upgrade`: it has no Python
query API at all. That absence is the headline of the branch, not a gap in
this report.

---

## Method notes / reproduction

| what | command | logs |
|---|---|---|
| size sweep | `pixi run binary_size` (both branches) | `/tmp/g3/size_{alpha,base}.log` |
| gate | `pixi run python3 benchmarks/binary_size/check_gate.py` | `/tmp/g3/gate_alpha.log` |
| perf ×5 | `pixi run -e dev pytest --benchmark --mojo-timeout 3600 marrow/kernels/tests/bench_filter.mojo marrow/kernels/tests/bench_sort.mojo` | `/tmp/g3/perf_{alpha_1,base_1,alpha_2,base_2,alpha_3}.log` |
| clickbench ×3 | `pixi run -e dev python python/marrow/tests/clickbench_alpha.py` | `/tmp/g3/clickbench_{1,2,3}.log` |

Baseline built in a throwaway worktree at `/tmp/marrow-baseline`
(`git worktree add /tmp/marrow-baseline mojo-1.0-upgrade --detach`), `.pixi`
symlinked to the main checkout's. Neither the main checkout nor any sibling
agent worktree was touched, and `benchmarks/binary_size/baseline.json` was not
modified.
