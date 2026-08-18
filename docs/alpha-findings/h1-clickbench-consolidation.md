# H1 — consolidating the ClickBench suite

**Verdict: five files became three, 42/43 still passes, and the one number worth
acting on is that marrow has no projection pushdown.**

- **Consolidation.** Five files became three, and each query is defined **once**
  instead of two-and-a-half times. It is **not** a line-count win: 1 694 lines
  became 2 061 (+367). The 43 polars thunks and the three-engine timing harness
  are new capability, and they cost more than the removed duplication saved.
  Anyone expecting consolidation to shrink the tree should read that number
  first.
- **Correctness unchanged at 42/43**, verified by the consolidated suite:
  `85 passed, 1 skipped` (42 marrow queries + 43 polars thunks; Q29 skipped as
  UNSUPPORTED).
- **New: the polars baseline is itself checked.** All 43 polars spellings are
  diffed against the same DuckDB reference before any of them is used as a
  timing denominator.
- **New: a measured marrow-vs-polars-vs-duckdb table.** marrow totals **17.6x
  polars** and **8.7x duckdb** across the 41 queries all three engines run.
- **Cause: no projection pushdown.** Not the kernels. Detailed below.

Measured 2026-08-18 on osx-arm64 (Apple Silicon), branch `mojo-1.0-upgrade`
merged fast-forward to `alpha` (`43cc0f6`), worktree
`/Users/kszucs/Workspace/marrow/.claude/worktrees/agent-a3541ed8e4153c566`.
Dataset `~/Workspace/ClickBench/data/hits_0.parquet` (1 000 000 rows, 105
columns). polars 1.43.2, DuckDB 1.5.5, pyarrow 23.

---

## 1. What the sprawl was, and what replaced it

| before | lines | after | lines |
|---|---|---|---|
| `clickbench.py` | 300 | `clickbench.py` — the registry, the reference, the comparison logic, the isolation | 1 734 |
| `test_clickbench.py` | 37 | `test_clickbench.py` — 43 marrow queries + 43 polars thunks vs the reference | 74 |
| `bench_clickbench.py` | 60 | `bench_clickbench.py` — 3 engines, interleaved, medians | 253 |
| `clickbench_alpha.py` | 1 113 | *(absorbed)* — its 43 queries, statuses, tie logic and subprocess isolation | — |
| `clickbench_reference.py` | 184 | *(absorbed)* — see §2 | — |
| | **1 694** | | **2 061** |

The old `clickbench.py` is *replaced*, not extended: its 11 queries were written
against an eager `group_by` API nothing else in the tree uses, and all 11 shapes
are already in the 43.

The old trio had the right *shape* (one definition, two consumers, multi-engine)
and the wrong *content*; the new pair had the right content and only one
consumer. Taking the shape from one and the content from the other is the whole
change.

### The one design decision worth recording

**`clickbench_reference.py` existed because of a belief that was false.** Its
docstring said the two modules "need different pixi environments — DuckDB lives
in `bench`, the marrow extension in `dev`", so the answers had to be produced by
one interpreter, serialised to `.benchmarks/clickbench-reference.json` (~1 MB,
gitignored), and read back by the other.

`pixi.toml` line 231 says `bench = { features = ["dev", "bench"] }`. **`bench` is
a superset of `dev`.** marrow, pyarrow, polars and duckdb all import into one
interpreter. There was never a need for the file hand-off, the JSON encoding of
bytes through `latin1`, or the two-command workflow. The reference is now a
lazily-memoised session object (`clickbench.Reference`) computed in-process.

That deleted three things at once: the JSON round trip, the `_jsonable` /
`_from_json` latin1 pair on the reference side, and the "run this command first"
instruction that made the suite non-self-contained.

### What was preserved deliberately

- **The two DuckDB reference corrections** (§2). These are the difference
  between scoring 25/43 and 38/43 and are the easiest thing in the whole suite
  to lose.
- **The top-N tie argument.** Ten queries pass on it.
- **Subprocess isolation.** `run_isolated` still spawns one interpreter per
  query. No query aborts today, but the isolation is what made a number
  reportable while three did, and it costs ~0.4 s per query.
- **The four probes.** They record spellings that do not work. Three of the four
  changed behaviour since the alpha measured them (§4) — which is the argument
  for keeping them.

---

## 2. The two reference traps, restated because they are load-bearing

Both made marrow look wrong when it was right.

1. **`CAST(blob AS VARCHAR)` escapes non-printable bytes as `\xNN`.** Every
   string-looking column in the file is `binary`, which DuckDB reads as BLOB,
   and BLOB binds to neither `like` nor `length` nor `min`. The cast is not the
   fix: it makes `length()` count four characters per byte and `ORDER BY` sort
   escaped text. The reference reads the file with PyArrow and `view()`s the 28
   binary columns to `string` — zero-copy reinterpret, no validation — so DuckDB
   sees real VARCHAR holding the original bytes; results come back through Arrow
   and are `view()`-ed back to `binary`. `fetchall()` would decode VARCHAR to
   `str` and choke on data that is not all UTF-8.
2. **`length` means bytes.** DuckDB's `length(VARCHAR)` counts characters;
   ClickHouse's `length()` — what ClickBench means — and marrow's `LengthKernel`
   both count bytes. `strlen` is DuckDB's byte-length function. Without the
   rewrite Q28 reads as a marrow mismatch (76.44 vs 73.97).

**These now apply to the *timing* connection differently, and that is a new
wrinkle.** Correctness uses an in-memory table with the zero-copy view; timing
needs DuckDB to scan the Parquet file lazily, where the view trick is not
available. `duckdb_lazy_connection()` therefore builds
`CREATE VIEW hits AS SELECT * REPLACE ("URL"::VARCHAR AS "URL", …) FROM
read_parquet(...)` — the escaping cast, which is *wrong for answers and fine for
timing*, and is why the two connections are separate. Stated in the code at both
sites.

---

## 3. Fairness traps designed around

Each of these produces a number that is not a comparison:

- **`pl.scan_parquet(...).select(pl.len())` answers `COUNT(*)` from the Parquet
  footer in 0.58 ms without scanning.** DuckDB does the same. Q1 is flagged
  `metadata_shortcut=True`, printed with the note, and excluded from the totals.
- **`pl.col("EventDate").min()` and `.max()` in one `select` collide on output
  name** — polars raises `DuplicateError`. Q7's thunk aliases them.
- **DuckDB `LIKE` rejects a BLOB.** Handled by the view above.
- **Lazy against lazy.** `ma.read_parquet` vs `pl.scan_parquet` vs DuckDB over
  `read_parquet(...)`. Comparing marrow's lazy plan against a pre-materialised
  in-memory DuckDB table would have flattered nobody usefully.
- **Q24 is `SELECT *`.** Its *comparison* is narrowed to three columns, so
  `duckdb_timing_sql` deliberately differs from `duckdb_sql`: timing the
  three-column probe would measure a different query than marrow runs.
- **A wrong polars thunk is free performance for marrow.**
  `test_polars_thunk_matches_reference` diffs all 43 against the DuckDB answers.
  It caught two of my own thunks — Q25 dropped `EventTime` before sorting on it,
  and Q24 returned 105 columns against a 3-column reference.

### Measurement discipline

This box drifts ±8% per case and ~±5% per batch. So: **engines interleaved per
repeat** (marrow, polars, duckdb, marrow, polars, duckdb, …) rather than each run
to completion, one **untimed warm-up** per (query, engine), and **medians of 5**.
Two independent runs were taken; see §6.

---

## 4. What changed in marrow since the alpha measured this

Three of the four probes now behave differently, and one gap-list item is gone.
None of this was found by looking for it — the probes reported it.

| probe | alpha | now |
|---|---|---|
| `p_binary_group_key` | hard-aborted the process above `_PARALLEL_ALWAYS_ROWS` rows | **runs** |
| `p_isin_untyped` | returned 0 rows silently | **raises** `is_in: dtype mismatch: int16 vs int64` |
| `p_binary_literal_ne` | returned 0 rows silently | **raises** `dispatch_primitive: dtype is not primitive` |

Also: **`count_star()` is bound to Python.** The alpha's gap-list item 2 said
`marrow.expr.builders.count_star()` existed in Mojo but not in Python, and 28 of
the 43 queries opened with a hand-rolled `lit(1).count()`. `marrow.count_star()`
now exists (`python/marrow/_expr_column.py:428`), and all 28 use it. The
`COUNT_STAR_QUERIES` list that tracked the workaround is deleted.

And Q11, Q12 and Q24 — the three ABORTs — pass. The `arrays.mojo` wrong-variant
downcast is fixed.

**`docs/alpha-clickbench-coverage.md` said 39/43 and listed three ABORTs.** It is
now 42/43 with the gap list re-cut. That document had been stale in the
optimistic-to-pessimistic direction, which is the safe one, but it was stale.

---

## 5. The headline number

```
query    marrow   polars   duckdb   vs polars
q01      262.2 ms   0.58     2.73      455x   <- metadata-only, excluded
q04      267.3 ms   1.47     4.97      182x
q21      552.1 ms  41.1     85.8        13x
q34      311.3 ms  70.7    100.5       4.4x
TOTAL   13 935 ms  790     1 595       17.6x
```

**marrow sits at a flat ~260–300 ms floor on every query, including `COUNT(*)`.**
The cause is not the kernels:

```
eager read_table(columns=["UserID"])   1.4 ms
eager read_table()  (all 105 cols)   212.8 ms
lazy select("UserID").aggregate()    254.5 ms   <- no improvement
```

**`ParquetScan`'s schema *is* its projection, and nothing narrows it.** Every
lazy query reads all 105 columns. The reader is fine; the plan never tells it
what to read.

Two things fall out of that, both visible in the table:

- **The ratio is an artefact of column count, not of kernel speed.** It collapses
  from 455x to 4.4x as the query's own column footprint grows. Q34 (`GROUP BY
  URL`) at 4.4x is the honest end — polars must read that column too. The 100x+
  rows are measuring the 104 columns marrow read for nothing.
- **Exactly one query is compute-bound: Q30**, 623 ms against a 260 ms floor —
  90 fused sums cost ~360 ms of real arithmetic. Everything else is scan.

**This was not fixed here, deliberately.** Pushing projection from the plan into
`ParquetScan` is a separate and larger piece of work than consolidating a test
suite, and doing it inside this change would have made the before/after
unreadable. It is now the top item on the coverage document's gap list.

A secondary observation worth one line: **marrow's *relative* position against
DuckDB is much better than against polars on the string-heavy queries** — Q28 is
2.5x DuckDB (and 8.6x polars), Q40 2.5x, Q37 3.2x. DuckDB pays for the escaping
`::VARCHAR` cast in the timing view there, so treat those three as an upper bound
on DuckDB rather than a marrow win.

---

## 6. Reproducibility

Two independent 5-repeat runs were taken. A single run is not evidence — a
sibling agent watched four queries "regress" in one run and evaporate in the next
two, and it was another agent's compile.

Run-to-run agreement is recorded in §7 below. Everything in
`docs/alpha-clickbench-coverage.md` is from run 1; run 2 is the check, not a
second dataset.

```
pixi run -e bench pytest python/marrow/tests/test_clickbench.py   # 85 passed, 1 skipped
pixi run -e bench python python/marrow/tests/bench_clickbench.py  # the table
pixi run -e bench python python/marrow/tests/clickbench.py        # 42/43
```

`bench` is the environment for all three. Under `dev` the whole correctness
module skips (no duckdb) and the standalone report runs unverified. Without the
dataset everything skips.

---

## 7. Run-to-run agreement

Two 5-repeat runs, back to back, same tree, nothing else running.

| | run 1 total | run 2 total | delta | median per-query \|delta\| | worst |
|---|---|---|---|---|---|
| marrow | 13 935 ms | 14 245 ms | +2.2% | 2.7% | 6.2% (q27) |
| polars | 790 ms | 809 ms | +2.3% | 3.1% | 12.5% (q12) |
| duckdb | 1 595 ms | 1 655 ms | +3.8% | 2.5% | 12.1% (q26) |

A **third** run, taken through the pytest entry point at 2 repeats instead of 5,
totalled 13 962 / 795 / 1 596 ms — 17.6x and 8.7x again.

**The ratios are identical to the printed precision across all three runs: 17.6x
polars, 8.7x / 8.6x / 8.7x duckdb.** All three engines drifted the same direction by about the
same amount between runs, which is exactly the ~±5% per-batch drift CLAUDE.md
warns about — and exactly why interleaving matters: a whole-batch shift that
lands on all three engines equally cancels in the ratio, which it would not have
if each engine had been run to completion in turn.

Worst per-query disagreement is on the *fast* rows (q12 at 2.3 ms, q26 at
11 ms), where a 1 ms absolute wobble is a 12% relative one. No marrow row moved
more than 6.2%, and no conclusion in this document rests on a difference smaller
than 2x.
