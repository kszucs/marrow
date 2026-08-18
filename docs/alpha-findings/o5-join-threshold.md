# Join probe: one row count answering two questions

Branch `worktree-agent-a0c4f0fc844027f88`, worktree
`.claude/worktrees/agent-a0c4f0fc844027f88`, from `alpha` at `f3162e0`
(the merge that made `ExecContext.auto()` the plan default).

## The defect

`HashJoin.probe` chose serial vs partitioned by re-asking
`worth_parallel(self._left_rows, _PARALLEL_THRESHOLD)` — the **build**-side row
count. `JoinProcessor` streams the **probe** side in 8192-row morsels, so a
1M-row build put ~122 tiny probe calls through the radix-partitioned path,
paying the partitioning cost once per call instead of once per join.

The diagnosis handed to this task was correct about the cause. It was wrong
about the remedy, and the microbenchmark is what caught that — see
"What the fan-out threshold would have cost" below.

## The benchmark

`marrow/kernels/tests/bench_join.mojo`. Every pre-existing probe bench delivers
the probe side as **one big batch**, which is precisely the shape that hides
this defect: it amortizes one dispatch over 1M rows where the plan layer pays
it 122 times over 8192 rows each. Added:

- `bench_join_morsel_1m_t{1,2,4,8}` — 1M build, 1M probe in 8192-row morsels.
  The regressing shape, and the only one that matters.
- `bench_join_single_1m_t{1,2,4,8}` — same data, same build, one big probe.
  Parallel should win here, and does.
- `bench_join_morsel_10k_t{1,8}` — build below the threshold, must stay serial.
- `bench_join_drift_control_sum_1m` — `SumKernel` over 1M int64. Touches
  nothing in `join.mojo` / `partition.mojo` / `hashtable.mojo`, so it cannot
  move for any reason attributable to the fix.

`t1` uses `ExecContext.serial()`, which is what the plan layer constructed
before `auto()` — so the `t1` row is the baseline every other row must beat.

The drift control earned its keep twice. The first baseline batch and the
final one differ by **26%** on the control alone, so the two are not
comparable; and one `radix_bits` sweep batch was thermally degraded (its
untouched `1M t1` control read 25.82 ms against 17.23/17.38/17.76 ms in every
neighbouring batch, StdDev 1.88 against 0.14). That batch would have inverted
the parameter choice. All figures below come from **two adjacent runs of an
identical selection**, normalised by the control.

## Before / after

Adjacent runs, identical selection. Control: 258.97 us before, 251.02 us
after, so the after batch ran 3.2% fast; every after figure is scaled by
**1.0317** onto the before scale. Medians.

| case | before | after (normalised) | delta |
|---|---:|---:|---:|
| `drift_control_sum_1m` | 258.97 us | 258.97 us | — (control) |
| `morsel_10k_t1` | 79.01 us | 79.13 us | +0.2% |
| `morsel_10k_t8` | 123.45 us | 121.16 us | -1.9% |
| `morsel_1m_t1` | 9.650 ms | 10.278 ms | +6.5% |
| **`morsel_1m_t2`** | **25.747 ms** | **16.645 ms** | **-35.4%** |
| **`morsel_1m_t4`** | **25.309 ms** | **15.810 ms** | **-37.5%** |
| **`morsel_1m_t8`** | **26.374 ms** | **17.163 ms** | **-34.9%** |
| `single_1m_t1` | 10.049 ms | 10.354 ms | +3.0% |
| `single_1m_t2` | 5.899 ms | 6.080 ms | +3.1% |
| `single_1m_t4` | 4.008 ms | 4.206 ms | +4.9% |
| `single_1m_t8` | 3.125 ms | 3.403 ms | +8.9% |

`morsel_1m_t1` (+6.5%) and the `single_1m` row at t1/t2 sit inside this box's
±8% per-case drift. The `single_1m_t8` +8.9% is real and is the deliberate
trade described under `radix_bits` below; it remains a 3.0x win over serial.

## Where the target is met, and where it is not

The stated target is "no case slower than the old serial default".

**At the level the regression was reported — the whole join — it is met.** A
separate experiment timed build + morselized probe together, which is what
`JoinProcessor` actually runs:

| rows | serial (bar) | 8 workers, before | 8 workers, after |
|---|---:|---:|---:|
| 1M | 17.38 ms | 30.07 ms | **16.84 ms** |
| 4M | 101.76 ms | 93.54 ms | **68.52 ms** |
| 10M | 360.52 ms | 238.92 ms | **181.13 ms** |

**At the level of the probe alone, it is not.** `morsel_1m_t8` is 17.16 ms
against a 10.28 ms serial probe. The partitioned probe repeats its
partitioning on every call, and no per-call decision can recover that: the
only thing that would is a single-table build, and a single table cannot be
built in parallel without atomics. The parallel *build* is what pays for the
slower probe, and above 1M it more than does.

This is the honest boundary of the fix. The join is faster than serial at
every size end to end; the probe in isolation is not, and closing that
requires changing the build layout, not a threshold.

## What the fan-out threshold would have cost

The brief asked for a fan-out threshold sized by probe rows. **Measurement
says there should not be one.** Within the partitioned layout, fanning the
partitions out beats running them serially at *every* batch size tested —
1M build, 8 workers, single probe call:

| probe rows | fan-out | serial partitions | fan-out wins by |
|---:|---:|---:|---|
| 8,192 | 194.3 us | 388.1 us | 2.0x |
| 32,768 | 285.0 us | 700.6 us | 2.5x |
| 131,072 | 574.2 us | 1.940 ms | 3.4x |
| 262,144 | 871.8 us | 3.582 ms | 4.1x |
| 524,288 | 1.546 ms | 7.978 ms | 5.2x |
| 1,048,576 | 3.153 ms | 17.80 ms | 5.6x |

A `probe_parallel_min_rows` threshold was implemented, measured, and
**removed**. Routing an 8192-row morsel to serial partitions would have made
the reported regression twice as bad. The expensive thing is the partitioning,
not the `sync_parallelize`.

## What actually changed

1. **`_built_parallel`** records which layout `build` produced, and `probe`
   follows it. This is the decoupling the brief asked for: the correctness
   constraint is now stated once, as a fact, instead of being re-derived by
   asking a throughput predicate a second time and trusting two calls to agree.

2. **`_probe_ctx(probe_rows)`** sizes the probe's hashing from *this call's*
   rows. `ExecContext.parallel(n)` is a forced count and `stripe` reads a
   forced count as an instruction, which is right for a caller who sized the
   work and wrong for a kernel splitting whatever batch it was handed.
   `worth_parallel` reads it as a budget. Threshold `32_768`, `stripe`'s own
   `min_parallel_size` — the same crossover, asked about a probe batch.
   Measured: an 8192-row probe hashed across 8 forced workers costs **1213
   us/morsel** against **76 us** on the calling thread. This is also why
   `morsel_10k_t8` was 45% slower than `morsel_10k_t1` before the fix.

3. **`_DEFAULT_RADIX_BITS` 6 → 4** (64 → 16 partitions). This is the change
   that carries the win. The 64-partition default came from a sweep of a
   *one-shot* 10M join, where 32/64/128 landed within ~1 ms because a single
   call over 10M rows amortizes any fanout. Under morsel streaming the cost is
   paid per call. Re-swept on the morselized shape (whole join, 8 workers):

   | bits | partitions | 1M | 4M | 10M |
   |---|---|---:|---:|---:|
   | 3 | 8 | 20.30 | **61.74** | **158.85** |
   | 4 | 16 | **16.84** | 68.52 | 181.13 |
   | 6 | 64 | 30.07 | 93.54 | 238.92 |
   | — | serial bar | 17.38 | 101.76 | 360.52 |

   3 is faster at 4M and 10M but **loses to serial at 1M**; 4 is the only
   setting that beats the bar everywhere, and at 8 workers it is also the
   principled choice — 2 partitions per worker, enough to balance skew without
   8x oversubscription. This is the trade that costs `single_1m_t8` 8.9%.

## Plan-layer observation — reported, not implemented

`marrow/expr/execution.mojo` is owned by another agent, so this is a finding
only.

**`JoinProcessor`'s 8192-row probe morsel is far below the size at which the
partitioned probe amortizes.** Per-row probe cost, 1M build, 8 workers:

| probe rows/call | per-row cost |
|---:|---:|
| 8,192 | 23.7 ns |
| 32,768 | 8.7 ns |
| 131,072 | 4.4 ns |
| 262,144 | 3.3 ns |
| 524,288 | 3.0 ns |
| 1,048,576 | 3.0 ns |

The curve flattens around 256k. Raising the join's probe batch from 8192 to
~256k rows would cut the partitioned probe's per-row cost about **7x** — a
larger effect than anything available inside the kernel, and it would also
make the probe-only shortfall noted above disappear, since the per-call
partitioning would finally be amortized. It costs peak memory proportional to
the batch, which is why it belongs to whoever owns the plan layer's memory
budget rather than here.

## Gates

- `pixi run -e dev pytest marrow/kernels/tests/test_join.mojo` — **35 passed**
  (29 pre-existing + 6 new).
- `pixi run -e bench pytest python/marrow/tests/test_clickbench.py` —
  **85 passed, 1 skipped**, unchanged.
- `pixi run python3 benchmarks/binary_size/check_gate.py` — **byte-for-byte
  identical** before and after; largest gate delta `+0.453%`
  (`query_dynvalue`), unchanged from the pre-fix tree. `baseline.json` untouched.
- `pixi run -e dev precompile` — 0 errors, 0 warnings.
- `pixi run -e bench pytest python/marrow/tests/bench_join_parallel.py
  --benchmark` — 8 passed. Marrow medians 8.47 → 7.83 ms at 1M and 78.26 →
  77.86 ms at 10M. Marrow is fastest at 10M (77.86 ms vs polars 96.18,
  pyarrow 112.67, duckdb 121.74) and second at 1M (polars 6.79 ms). Small
  because this bench issues one-shot joins, not morselized ones — which is the
  same reason it never showed the regression.

## The new tests

`test_join_paths_agree_{inner,left,semi,anti}_morsels` plus two single-probe
variants. They build at 150k rows (above `_PARALLEL_THRESHOLD`) with 4 workers
and assert `built_parallel()` first, so the test cannot pass by silently
comparing the serial path against itself.

The two paths emit the same rows in **different orders** — partition order
versus probe order — so they compare an order-insensitive fingerprint: row
count, plus per column the sum of values and the null count. A row count
alone would miss a drop and a duplicate that cancel out.

Mutation-checked: corrupting the partitioned probe's row remap
(`Take.apply(rows, pairs[1])` → `pairs[1]`) fails 4 of the 6. The two that
survive are SEMI and ANTI, which emit only build-side rows and so cannot
observe a probe-index defect — correct, not a gap.
