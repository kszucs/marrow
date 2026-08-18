# O4 — Thread control for the lazy query path, and what is actually behind it

Branch: `worktree-agent-a930845fcb514b1b0` (off `alpha` @ `9dc12a2`).

## The problem, restated

`ExecContext.__init__` defaults to `num_threads: Int = 1` — **forced serial**.
`DynRelation.execute(ctx: ExecContext = ExecContext())` therefore defaulted to
serial, and `_plan_execute` in `python/bindings/plan.mojo` called `.execute()`
with no context at all. So every query built through `marrow.read_parquet(...)`
/ `memtable(...)` ran single-threaded on a 16-core machine, and there was no
spelling — anywhere in the Python surface — that could ask for anything else.

The eager surface has had one the whole time: `RecordBatch.group_by(...,
num_threads=0)`, `.join(..., num_threads=0)`, `.sort_by(..., num_threads=0)`.

## What was done

| | |
|---|---|
| **API** | `LazyTable.collect(num_threads=0)` (and `to_pyarrow(num_threads=0)`) |
| **Default** | `0` = **auto**, matching the eager surface exactly |
| **Mojo default** | `DynRelation.execute` / `to_processor` now default to `ExecContext.auto()`, not `ExecContext()` |
| **Contexts fixed** | `FilterProcessor` and `JoinProcessor` were dropping it entirely |

`collect` rather than a constructor argument or a module-level default because
`collect` is the only place a plan *runs*. A `LazyTable` is an immutable plan
and every verb returns a fresh one, so a stored worker count would have to
survive `join`, where two tables carrying different settings have no defensible
winner. `num_threads=` rather than `threads=`/`n_jobs=` because the eager
surface already spells it that way, and the sentinel set is the same one
(`0` auto, `1` serial, `N` forced).

**The default changed, so every existing lazy benchmark number is now taken
under `auto`.** The ClickBench table is re-baselined below — and the answer is
that it did not move, for reasons that are the real content of this finding.

## Scaling — `python/marrow/tests/bench_lazy_parallel.py`

Medians, 1M rows unless noted. Before the change all four thread columns were
the same measurement four times, because `num_threads` had nowhere to go.

**`num_threads=1` after the change is bit-for-bit the old behaviour** — every
site that now reads the context previously fell through to a `num_threads == 1`
default — so the `t1` column *is* the before number, measured in the same batch
rather than against a box that drifts ±8%. The separate "before" run agrees
after subtracting its drift-control delta (115.9 us → 136.5 us, +17.8% batch).

| case | t1 (= before) | t2 | t4 | t8 | t8 / t1 |
|---|---|---|---|---|---|
| `groupby` 100 groups | 3.654 ms | 3.646 | 3.648 | 3.655 | **1.00x** |
| `groupby` 500k groups | 19.708 ms | 19.698 | 19.805 | 19.622 | **1.00x** |
| `filter` (33% selective) | 2.464 ms | 2.434 | 2.417 | 2.404 | **1.03x** |
| `sort` | 11.522 ms | 7.566 | 5.554 | 5.330 | **2.16x** |
| `join` 1M x 1M inner | 43.287 ms | 59.844 | 57.933 | 56.526 | **0.77x** ← regression |
| `join` 10M x 10M inner | 759.1 ms | 654.9 | 630.3 | 665.6 | **1.14x** |
| drift control (numpy sum) | 136.5 us | — | — | — | — |

Three results, one of them a win:

### Sort scales — 2.16x at 8 threads

`SortProcessor` already held an `ExecContext` and passed it to `sort_by_keys`;
the one it held was always the serial default. It is a pipeline breaker that
collects its whole input, so it hands the kernel one large array rather than a
morsel, and `sort_indices`' parallel path applies. Best case in the tree.

### The join **regresses** at 1M and only wins past it

`HashJoin.probe` decides serial vs radix-partition-parallel from
**`self._left_rows`** — the *build* side row count — not from the batch it is
handed:

```mojo
if not self._ctx.worth_parallel(self._left_rows, _PARALLEL_THRESHOLD):
    return self.probe_serial(...)
return self.probe_parallel(...)
```

It has to: `probe_parallel` reads the per-partition tables that only
`build_parallel` populates, so the two must reach the same verdict. But
`JoinProcessor` **streams** the probe side morsel by morsel, and a morsel is
8192 rows. A 1M-row build side therefore forces `probe_parallel` — a
`sync_parallelize` over 2^`radix_bits` partitions — on **122 consecutive
8192-row morsels**. The dispatch overhead is paid 122 times against a build
saving paid once, and at 1M the dispatch wins: 43.3 ms serial → 56.5 ms at 8
threads, a **31% regression**. At 10M the build cost finally dominates and it
turns into a modest 1.14-1.20x win, so the crossover for this shape is
somewhere between 1M and 10M — well above the kernel's 100k `_PARALLEL_THRESHOLD`.

This is a defect in `marrow/kernels/join.mojo` (owned elsewhere), not in the
wiring: the fix is for `probe` to keep using the partitioned tables while
running the per-partition loop serially when the *probe batch* is small, which
decouples the layout decision from the dispatch decision. Until then, a lazy
join over a build side in the 100k-few-million range is faster at
`num_threads=1`, and `auto` picks the slower path.

### Group-by does not scale at all — the ctx never reaches a grouping strategy

This is the one that matters, and it is not a wiring gap that threading a
context fixes. `AggregateProcessor` does **not** use the `GroupBy` kernel whose
serial / thread-local / radix strategy selection this task's brief describes.
It groups *incrementally*, morsel by morsel:

```mojo
gid_chunks.append(self._grouper.consume_keys(key_struct))   # HashGrouper
...
cols.append(self.aggs[i].grouped(Grouping(gids, num_groups), value))  # AggFunc
```

Neither `HashGrouper.consume_keys(keys, hashes)` nor `AggFunc.grouped(groups,
value)` takes an `ExecContext`, and `AggFunc._grouped[A]` bottoms out in
`A.grouped(...)` — a serial scatter. `_PARALLEL_MIN_ROWS` and
`_PARALLEL_ALWAYS_ROWS` are consulted by `GroupBy._choose_strategy`, which the
relational plan never constructs. The context `AggregateProcessor` stores is
used for `concat` and nothing else.

Routing the plan through `GroupBy` + `AggFuncSet.grouped` would reach those
strategies, but `AggFuncSet` carries an `AggFold` per member and the
`AggregateProcessor` docstring records that pulling the fold machinery onto this
path measured **+3.2 MB (+24%)** on the aggregate binary-size gate — plus it
would trade incremental grouping for buffering every key column to emit. That is
a real design decision with a measured price tag, not a follow-on to this change,
so it was left alone.

### Filter is flat because the morsel is 8192 rows

`FilterProcessor` now passes the context to `filter`, but a morsel is 8192 rows
and `stripe`'s auto threshold is 32768, so `auto` correctly stays serial; even
forced striping of 8192 rows across 8 workers is a wash (2.464 → 2.404 ms).
Nothing here is broken — it is the structural consequence of morsel-at-a-time
execution. **In a pull-based morsel engine the only operators with enough work
in hand to parallelise are the pipeline breakers** (sort, join build, a
whole-table group-by). Per-morsel element-wise work needs a bigger morsel or
inter-morsel parallelism, neither of which is a thread-count argument.

## Contexts that are still dropped

| site | takes a ctx? | consequence |
|---|---|---|
| `ParquetScanProcessor` | **no** — `marrow/parquet/reader.mojo` has no `ExecContext` anywhere | every decode is single-threaded; this is most of ClickBench |
| `Value.execute(batch)` / `DynValue` | **no** — the signature is `(self, batch: RecordBatch)` | every projection and every predicate evaluates serially; adding one changes `values.mojo`, `dynamic.mojo` and every kernel call inside them |
| `AggregateProcessor` grouping | ctx stored, but `HashGrouper` / `AggFunc.grouped` take none | see above |
| `ProjectProcessor`, `InMemoryTableProcessor`, `LimitProcessor` | n/a | expression eval / slicing only |
| `JoinProcessor` | **fixed here** | was `HashJoin[RapidHash64]()` → the constructor's serial default |
| `FilterProcessor` | **fixed here** | was `filter(col, mask)` → the kernel's `ExecContext.serial()` default |

## ClickBench — re-baselined, and unmoved

`bench_clickbench.py` now runs a fourth **interleaved** engine, `marrow1` =
`collect(num_threads=1)`, so before and after are measured in the same batch
against the same page cache. 43 queries, 5 repeats, medians, `hits_0.parquet`.

| | marrow (auto) | marrow1 (= before) | polars | duckdb |
|---|---|---|---|---|
| polars default (16 threads) | **4237.5 ms** | 4246.4 ms | 871.3 ms | 1821.4 ms |
| `POLARS_MAX_THREADS=1` | **4436.6 ms** | 4506.6 ms | 1787.7 ms | 1877.0 ms |

Matched thread counts:

* **1 thread each: marrow 4506.6 / polars 1787.7 = 2.52x**
* **all cores each: marrow 4237.5 / polars 871.3 = 4.86x**

Before the change, measured in two separate runs on a quieter box: 2.5x and
4.8x. **The change moves the ClickBench total by 0.2%** — within noise, and in
the "worse" direction on the polars-1-thread run only because the two engines'
totals are 1.5% apart there.

Only one query moves outside noise: **Q24** (`SELECT * ... ORDER BY EventTime
LIMIT 10`), 681.8 → 626.0 ms, **1.09x** — the sort. Everything else is Parquet
decode plus a serial group-by, and neither has a context to give.

So the honest summary of the end-to-end result is: **the knob is now reachable,
and on ClickBench there is almost nothing behind it.** The 4x gap versus
16-thread polars is not "marrow runs serial"; at matched single-thread it is
still 2.5x, and closing it needs a parallel Parquet decode and a parallel
grouping path, not an argument on `collect`.

## Verification

* `pixi run -e bench pytest python/marrow/tests/test_clickbench.py` — **85 passed, 1 skipped**
* `pixi run -e bench pytest python/marrow/tests/test_lazy.py` — passes, including
  seven new cases asserting identical group-by / join / filter results at 1 vs
  2/4/8 threads over 250k rows (past both parallel thresholds, so the parallel
  strategies actually run). A race here returns a wrong answer, not an error, so
  they compare the whole sorted result rather than a row count.
* `pixi run -e dev pytest marrow/expr/tests/test_plan.mojo` — **43 passed**,
  including `test_execute_defaults_to_auto_not_serial`
* `pixi run -e dev precompile` — 0 errors, 0 warnings
* `pixi run python3 benchmarks/binary_size/check_gate.py` — **OK**, largest
  growth `query_dynvalue` +0.453%; three gates shrank 1.3-2.6%
