# ClickBench coverage for the alpha — 42 / 43

Measured 2026-08-18 on `alpha` (`43cc0f6`) against
`~/Workspace/ClickBench/data/hits_0.parquet` — one partition, **1,000,000 rows,
105 columns**. Every query is expressed through the lazy Python frontend
(`marrow.read_parquet` → `LazyTable`), executed in its own subprocess, and
diffed against DuckDB 1.5.5. Timings are medians of 5 interleaved repeats,
marrow vs polars 1.43.2 vs DuckDB 1.5.5, all three lazy over the same file.

```
pixi run -e bench pytest python/marrow/tests/test_clickbench.py   # correctness
pixi run -e bench python python/marrow/tests/bench_clickbench.py  # comparison
pixi run -e bench python python/marrow/tests/clickbench.py        # the report
```

**42 / 43 PASS · 1 UNSUPPORTED.** Whole correctness suite: 17.7 s, no query over
1.0 s. Two queries are marked DEVIATED — the deviation is spelled out per row and
neither changes what the query computes.

The three ABORTs the previous revision of this document recorded (Q11, Q12, Q24)
are **gone**: the `arrays.mojo` wrong-variant downcast was fixed, and so was the
`_PARALLEL_ALWAYS_ROWS` binary group-by abort. The `p_binary_group_key` probe,
written to reproduce that abort deliberately, now simply runs.

---

## How the numbers were verified

Getting the *reference* right turned out to be harder than getting marrow right,
and two reference bugs initially read as marrow failures. Correcting them moved
the score from 25/43 to 38/43. Both are worth stating because anyone re-running
this will hit them:

- **DuckDB's `CAST(blob AS VARCHAR)` escapes non-printable bytes as `\xNN`.** So
  `length()` counts four characters per byte and `ORDER BY` sorts the escaped
  text. The reference instead reads the file with PyArrow and `view()`s the 28
  binary columns to `string` (zero-copy, no validation), so DuckDB sees real
  VARCHAR holding the original bytes, and results come back through Arrow and
  are `view()`-ed back to `binary`. Byte-exact both ways.
- **DuckDB's `length(VARCHAR)` counts characters; ClickHouse's `length()` — what
  ClickBench means — and marrow's `LengthKernel` both count bytes.** Q28 read as
  a marrow mismatch (76.44 vs 73.97) until the reference switched to `strlen`.
  marrow was right.

`ORDER BY <measure> DESC LIMIT 10` has no unique answer when the boundary value
is tied, and most of these queries order by a count with hundreds of tied groups.
On a row-set mismatch the harness therefore falls back to comparing the
**multiset of ORDER BY values**: same row count plus an identical multiset of
ordering values means both engines returned a valid top-N and only the tie-break
differs. Ten queries pass on that argument and each says so in its report line.

Three dataset facts drive the query code: `EventDate` is `uint16` epoch days (so
the date predicates are integer predicates), `EventTime` is `int64` unix seconds
(cast to `timestamp('s')` for Q19/Q43), and all 28 string-looking columns are
`binary` (cast to `string` at the point of use, because marrow's string kernels
are bound on `StringLikeType`). polars and DuckDB need the same cast, so all
three sides are comparing the same thing.

---

## Per-query results

`ms` columns are medians of 5 interleaved repeats. `vs pl` is marrow ÷ polars.

| Q | SQL feature exercised | status | marrow ms | polars ms | duckdb ms | vs pl | notes |
|---|---|---|---|---|---|---|---|
| 1 | `COUNT(*)` | PASS | 262 | 0.58 | 2.7 | 455x | **polars/duckdb answer from Parquet metadata without scanning** — excluded from the totals |
| 2 | `COUNT(*)` + `WHERE <>` | PASS | 276 | 1.2 | 3.6 | 238x | |
| 3 | `SUM` + `COUNT(*)` + `AVG`, no `GROUP BY` | PASS | 269 | 1.6 | 4.5 | 171x | |
| 4 | `AVG` over int64 | PASS | 267 | 1.5 | 5.0 | 182x | |
| 5 | `COUNT(DISTINCT)` over int64 | PASS | 279 | 4.1 | 8.9 | 67x | |
| 6 | `COUNT(DISTINCT)` over string | PASS | 273 | 11.1 | 17.1 | 25x | |
| 7 | `MIN`/`MAX` | PASS | 266 | 0.88 | 3.5 | 303x | |
| 8 | `GROUP BY` int + `ORDER BY COUNT(*) DESC` | PASS | 275 | 2.1 | 3.7 | 133x | |
| 9 | grouped `COUNT(DISTINCT)` + `ORDER BY` + `LIMIT` | PASS | 270 | 6.7 | 11.5 | 41x | |
| 10 | four heterogeneous aggregates over one grouping | PASS | 282 | 8.3 | 16.8 | 34x | |
| 11 | `GROUP BY` string key + `COUNT(DISTINCT)` | PASS | 281 | 2.3 | 6.2 | 120x | was ABORT; fixed |
| 12 | `GROUP BY` (int, string) | PASS | 283 | 2.7 | 6.6 | 107x | was ABORT; fixed. tie |
| 13 | `GROUP BY` high-cardinality string | PASS | 298 | 6.8 | 14.0 | 44x | |
| 14 | `GROUP BY` string + `COUNT(DISTINCT int64)` | PASS | 296 | 7.2 | 16.7 | 41x | |
| 15 | `GROUP BY` (int, string) | PASS | 294 | 7.3 | 15.2 | 40x | |
| 16 | `GROUP BY` very-high-cardinality int64 | PASS | 280 | 4.7 | 9.7 | 60x | |
| 17 | `GROUP BY` (int64, string), very high cardinality | PASS | 277 | 15.3 | 21.9 | 18x | |
| 18 | `GROUP BY` + `LIMIT`, **no** `ORDER BY` | PASS | 281 | 10.6 | 21.4 | 27x | shape-only: the SQL is nondeterministic, so 10 rows × 3 columns is all that can be asserted |
| 19 | `extract(minute FROM ts)` as a group key | PASS | 299 | 32.1 | 30.6 | 9x | tie |
| 20 | point lookup on int64 (empty result) | PASS | 269 | 0.95 | 3.0 | 282x | passes only because the harness bypasses PyArrow for zero-row batches — `pa.record_batch()` on an empty result raises `SystemError`. Findings §1.3 |
| 21 | `LIKE '%…%'` + `COUNT(*)` | PASS | 552 | 41.1 | 85.8 | 13x | |
| 22 | `LIKE` + `MIN(string)` + `GROUP BY` string | PASS | 564 | 41.4 | 91.6 | 14x | tie |
| 23 | `LIKE` + `NOT LIKE` + two `MIN(string)` + `COUNT(DISTINCT)` | PASS | 900 | 52.5 | 178.6 | 17x | slowest query in the suite |
| 24 | `SELECT *` (105 cols) + `ORDER BY` + `LIMIT` | PASS | 555 | 105.5 | 158.7 | 5.3x | was ABORT; fixed. **DEVIATED:** only the *comparison* is narrowed to `WatchID`/`EventTime`/`URL`; the query is the real `SELECT *` |
| 25 | top-N on an int64 sort key | PASS | 303 | 10.7 | 16.8 | 28x | tie; verified against a variant that also projects `EventTime`, since the SQL sorts on a column it does not select |
| 26 | top-N on a string sort key | PASS | 315 | 5.7 | 10.7 | 55x | |
| 27 | top-N on a two-column (int, string) sort key | PASS | 304 | 6.4 | 17.2 | 47x | tie; same variant treatment as Q25 |
| 28 | `AVG(length(str))` + `HAVING` | PASS | 345 | 39.9 | 135.7 | 8.6x | `HAVING` is just `aggregate(...).filter(...)`; marrow beats DuckDB's ratio here (2.5x) |
| 29 | `REGEXP_REPLACE` | **UNSUPPORTED** | — | 684.7 | 439.5 | — | **No regex kernel exists anywhere in marrow** — not in `marrow/kernels/string.mojo`, not on `DynValue` — and the Mojo standard library has no regex engine to bind to. See `alpha-findings/g2-regex-evaluation.md` |
| 30 | 90 `SUM`s over computed expressions in one pass | PASS | 623 | 3.8 | 8.8 | 163x | the one query where marrow's *compute* is the cost, not the scan: +360 ms over the floor for 90 fused sums |
| 31 | `GROUP BY` (int16, int32) + 3 aggregates | PASS | 302 | 9.3 | 10.7 | 33x | tie |
| 32 | `GROUP BY` (int64, int32), high cardinality | PASS | 306 | 7.2 | 11.8 | 43x | tie |
| 33 | `GROUP BY` (int64, int32) over the whole table | PASS | 328 | 28.3 | 31.0 | 12x | tie |
| 34 | `GROUP BY` high-cardinality string, whole table | PASS | 311 | 70.7 | 100.5 | 4.4x | **the honest end of the range** — polars must read `URL` too |
| 35 | `GROUP BY` constant + string | PASS | 315 | 74.9 | 100.1 | 4.2x | |
| 36 | `GROUP BY` four computed integer keys | PASS | 282 | 6.9 | 13.0 | 41x | |
| 37 | multi-predicate filter + `GROUP BY` string | PASS | 366 | 40.6 | 114.0 | 9x | |
| 38 | multi-predicate filter + `GROUP BY` string | PASS | 362 | 31.3 | 80.6 | 12x | |
| 39 | `LIMIT … OFFSET 1000` | PASS | 294 | 20.6 | 23.6 | 14x | tie |
| 40 | `CASE WHEN` as a group key, five keys | PASS | 390 | 51.6 | 153.4 | 7.6x | tie |
| 41 | `IN (…)` + int64 equality + `OFFSET` | PASS | 301 | 5.8 | 10.5 | 52x | tie. **DEVIATED (spelling only):** `IN (-1, 6)` is written `isin(ma.array([-1, 6], ma.int16()))`; the plain-list spelling now raises `is_in: dtype mismatch: int16 vs int64` rather than silently returning zero rows, which is an improvement but still not the SQL spelling |
| 42 | `OFFSET 10000` past the end of the result | PASS | 288 | 4.7 | 9.3 | 62x | same empty-batch caveat as Q20 |
| 43 | `DATE_TRUNC('minute', ts)` as a group key + `ORDER BY` it | PASS | 316 | 4.3 | 13.1 | 73x | has to `order_by("key0")` — a computed group key has no name |
| | **TOTAL** (41 queries all three ran, Q1 excluded) | | **13 935** | **790** | **1 595** | **17.6x** | |

"tie" means the row set matched on the multiset-of-ORDER-BY-values argument
described above, not on an exact row-for-row diff.

### Extra probes (not part of the 43)

Each of these was *silently wrong* when the alpha first measured it. Three have
since changed behaviour, which is the reason to keep them in the report.

| probe | what it shows now | what it showed before |
|---|---|---|
| `p_binary_group_key` | **runs** | hard-aborted the process (`_PARALLEL_ALWAYS_ROWS`) |
| `p_isin_untyped` | **raises** `is_in: dtype mismatch: int16 vs int64` | returned 0 rows silently |
| `p_isin_typed` | 727,932 — correct | unchanged |
| `p_binary_literal_ne` | **raises** `dispatch_primitive: dtype is not primitive` | returned 0 rows silently |

---

## The headline: there is no projection pushdown

**marrow sits at a flat ~260–300 ms floor on every query, including
`COUNT(*)`.** That is 17.6x polars and 8.7x DuckDB in total. The cause is
diagnosed and it is **not** the kernels:

```
eager read_table(columns=["UserID"])   1.4 ms
eager read_table()  (all 105 cols)   212.8 ms
lazy select("UserID").aggregate()    254.5 ms   <-- no improvement
```

`ParquetScan`'s schema *is* its projection, but nothing narrows it, so every
lazy query reads all 105 columns. The reader is fine; the plan never tells it
what to read.

Two consequences visible in the table above:

- **The ratio collapses as real work grows.** Q34 (`GROUP BY URL`) is 4.4x, not
  455x, because polars has to read that column too. The 100x+ rows are all
  queries that touch one or two narrow columns — i.e. they are measuring the 104
  columns marrow read for nothing.
- **Only Q30 is compute-bound.** 623 ms against a 260 ms floor: 90 fused sums
  cost ~360 ms of actual arithmetic. Everything else on the list is scan.

Fixing this is a separate and larger piece of work than this suite — it needs
projection to be pushed from the plan into `ParquetScan` — and it was
deliberately **not** attempted here. It is the single largest performance item
open against the lazy frontend.

`COUNT(*)` (Q1) is excluded from the totals because
`pl.scan_parquet(...).select(pl.len())` answers it from the Parquet footer in
0.58 ms **without scanning anything**; DuckDB does the same. Timing a metadata
read against a full scan is not a comparison. The row is printed, marked, rather
than dropped.

---

## Remaining gap list

Ordered by (queries unblocked) × (how small the change looks). Items 1, 2, 3 and
6 of the previous revision's list are **done** — the wrong-variant downcast is
fixed, `count_star()` is bound to Python, the dtype mismatches raise, and
`with_columns` exists.

### 1. Projection pushdown into `ParquetScan` — 0 queries, all 43 timings — **large**

See above. Nothing else on this list is worth more.

### 2. Make a zero-row `RecordBatch` exportable over the C Data Interface — 2 queries honestly (Q20, Q42) — **small**

`pa.record_batch(empty_batch)` raises `SystemError: … returned NULL without
setting an exception`. `collect()` is fine, only the export path fails. Q20 and
Q42 legitimately return no rows, so this is 5% of ClickBench, and the harness
carries a workaround (`clickbench._batch_rows`) that should be deleted with the
fix.

### 3. Name computed group keys — ~12 queries' readability — **small**

`aggregate(by=[expr])` names the output `key0`. Every string group key in this
dataset must be a `.cast(string)`, so every string-grouping query loses its key
column name, and Q43 has to `order_by("key0")`. `Column.alias()` now exists;
`aggregate` does not honour it on a *key*.

### Below the line — not blocking the alpha

- **Regex — unblocks Q29 only; large.** `REGEXP_REPLACE` needs a regex engine,
  which the Mojo standard library does not have. Writing or binding one for a
  single query is not a good trade; state the exclusion in the release notes and
  ship 42/43 as the ceiling. Q29's particular pattern
  (`^https?://(?:www\.)?([^/]+)/.*$`) is a host extraction and would be served
  by a `url_host()` scalar function at a fraction of the cost — but that answers
  a different question than ClickBench asked.
- **`ma.timestamp("s")` without an explicit `None` timezone; small.** The dtype
  constructors are re-exported raw from `libmarrow`, so the binding layer's "no
  optional arguments" rule leaks to users. Diverges from PyArrow.
- **`Schema` is not iterable and exposes no field types to Python; small.**
- **`len(record_batch)`; tiny.** `num_rows` is in the `repr` but unreachable.
- **A `(func, column)` tuple aggregate cannot take an expression; small.** The
  `Aggregate` object form covers it, but the two spellings look interchangeable
  and are not (Q30).
- **Repeating `.cast(string)` at every mention of a binary column; small.** A
  `read_parquet(..., cast={...})` removes it.

---

## Files

One module holds the queries; three consume it.

- `python/marrow/tests/clickbench.py` — **the single source of truth.** All 43
  queries, each carrying its canonical SQL (the docstring, which is also the
  DuckDB text after the documented rewrites), a marrow lazy-API thunk, a polars
  thunk, and its declared status. Also the DuckDB reference, the normalisation
  and comparison logic, and the subprocess isolation. `python clickbench.py`
  prints the PASS/FAIL/ABORT report.
- `python/marrow/tests/test_clickbench.py` — correctness. Every query against
  the DuckDB reference, plus every polars thunk against the same reference, so
  the benchmark's baseline is known to compute the same answers.
- `python/marrow/tests/bench_clickbench.py` — timing. marrow vs polars vs
  duckdb, engines interleaved per repeat, medians of N.
- `docs/alpha-findings/e1-clickbench.md` — the API findings, with the crash
  bisection tables (historical; the crashes are fixed).
- `docs/alpha-findings/h1-clickbench-consolidation.md` — this consolidation.

All three need the `bench` environment (`["dev", "bench"]`, so marrow + pyarrow
+ polars + duckdb in one interpreter) and the dataset, which is not vendored.
Under `dev` the cross-engine checks skip cleanly.
