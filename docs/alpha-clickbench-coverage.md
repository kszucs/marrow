# ClickBench coverage for the alpha — 39 / 43

Measured on `alpha` (`8365395`) against
`~/Workspace/ClickBench/data/hits_0.parquet` — one partition, **1,000,000 rows,
105 columns**. Every query is expressed through the lazy Python frontend
(`marrow.read_parquet` → `LazyTable`), executed in its own subprocess, and
diffed against DuckDB 1.5.5.

```
pixi run -e bench python python/marrow/tests/clickbench_reference.py  # answers
pixi run -e dev   python python/marrow/tests/clickbench_alpha.py      # the report
```

**39 / 43 PASS · 3 ABORT · 1 UNSUPPORTED.** Run three times end to end; every
query returned the same status in all three, so the number is not a lucky
sample. Whole suite: **20 s**, no query over 1.1 s.

Two queries are marked DEVIATED — the deviation is spelled out per row and
neither changes what the query computes.

---

## How the numbers were verified

Getting the *reference* right turned out to be harder than getting marrow right,
and two reference bugs initially read as marrow failures. Both are worth stating
because anyone re-running this will hit them:

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
differs. Nine queries pass on that argument and each says so in its report line.

Three dataset facts drive the query code: `EventDate` is `uint16` epoch days (so
the date predicates are integer predicates), `EventTime` is `int64` unix seconds
(cast to `timestamp('s')` for Q19/Q43), and all 28 string-looking columns are
`binary` (cast to `string` at the point of use, because marrow's string kernels
are bound on `StringLikeType`).

---

## Per-query results

| Q | SQL feature exercised | status | s | if not PASS, the exact missing capability |
|---|---|---|---|---|
| 1 | `COUNT(*)` | PASS | 0.37 | |
| 2 | `COUNT(*)` + `WHERE <>` | PASS | 0.37 | |
| 3 | `SUM` + `COUNT(*)` + `AVG`, no `GROUP BY` | PASS | 0.35 | |
| 4 | `AVG` over int64 | PASS | 0.34 | |
| 5 | `COUNT(DISTINCT)` over int64 | PASS | 0.35 | |
| 6 | `COUNT(DISTINCT)` over string | PASS | 0.36 | |
| 7 | `MIN`/`MAX` | PASS | 0.35 | |
| 8 | `GROUP BY` int + `ORDER BY COUNT(*) DESC` | PASS | 0.36 | |
| 9 | grouped `COUNT(DISTINCT)` + `ORDER BY` + `LIMIT` | PASS | 0.34 | |
| 10 | four heterogeneous aggregates over one grouping | PASS | 0.35 | |
| 11 | `GROUP BY` string key + `COUNT(DISTINCT)` | **ABORT** | — | **SIGSEGV.** `filter(cast(MobilePhoneModel,string) <> '')` over a multi-morsel parquet scan with a `binary` column in the projection faults. Not the known group-by abort — it fires with no grouping at all. See findings §2.2. |
| 12 | `GROUP BY` (int, string) | **ABORT** | — | Same crash as Q11 (same filter). |
| 13 | `GROUP BY` high-cardinality string | PASS | 0.41 | |
| 14 | `GROUP BY` string + `COUNT(DISTINCT int64)` | PASS | 0.40 | |
| 15 | `GROUP BY` (int, string) | PASS | 0.40 | |
| 16 | `GROUP BY` very-high-cardinality int64 | PASS | 0.37 | |
| 17 | `GROUP BY` (int64, string), very high cardinality | PASS | 0.42 | tie |
| 18 | `GROUP BY` + `LIMIT`, **no** `ORDER BY` | PASS | 0.41 | shape-only: the SQL is nondeterministic, so 10 rows × 3 columns is all that can be asserted |
| 19 | `extract(minute FROM ts)` as a group key | PASS | 0.42 | tie |
| 20 | point lookup on int64 (empty result) | PASS | 0.36 | passes only because the harness bypasses PyArrow for zero-row batches — `to_pyarrow()` on an empty result raises `SystemError`. Findings §1.3. |
| 21 | `LIKE '%…%'` + `COUNT(*)` | PASS | 0.67 | |
| 22 | `LIKE` + `MIN(string)` + `GROUP BY` string | PASS | 0.68 | tie |
| 23 | `LIKE` + `NOT LIKE` + two `MIN(string)` + `COUNT(DISTINCT)` | PASS | 1.05 | |
| 24 | `SELECT *` (105 cols) + `ORDER BY` + `LIMIT` | **ABORT** | — | `ABORT arrays.mojo:2572 get: wrong variant type` — sort/take over a table containing `binary` columns. Also aborts at three columns, so it is not table width. DEVIATED: only the comparison is narrowed to `WatchID`/`EventTime`/`URL`; the query is the real `SELECT *`. |
| 25 | top-N on an int64 sort key | PASS | 0.41 | tie; verified against a variant that also projects `EventTime`, since the SQL sorts on a column it does not select |
| 26 | top-N on a string sort key | PASS | 0.44 | |
| 27 | top-N on a two-column (int, string) sort key | PASS | 0.43 | tie; same variant treatment as Q25 |
| 28 | `AVG(length(str))` + `HAVING` | PASS | 0.45 | `HAVING` is just `aggregate(...).filter(...)` |
| 29 | `REGEXP_REPLACE` | **UNSUPPORTED** | — | **No regex kernel exists anywhere in marrow** — not in `marrow/kernels/string.mojo`, not on `DynValue` — and the Mojo standard library has no regex engine to bind to. |
| 30 | 90 `SUM`s over computed expressions in one pass | PASS | 0.78 | |
| 31 | `GROUP BY` (int16, int32) + 3 aggregates | PASS | 0.40 | tie |
| 32 | `GROUP BY` (int64, int32), high cardinality | PASS | 0.42 | tie |
| 33 | `GROUP BY` (int64, int32) over the whole table | PASS | 0.43 | tie |
| 34 | `GROUP BY` high-cardinality string, whole table | PASS | 0.43 | |
| 35 | `GROUP BY` constant + string | PASS | 0.44 | |
| 36 | `GROUP BY` four computed integer keys | PASS | 0.39 | |
| 37 | multi-predicate filter + `GROUP BY` string | PASS | 0.47 | |
| 38 | multi-predicate filter + `GROUP BY` string | PASS | 0.47 | |
| 39 | `LIMIT … OFFSET 1000` | PASS | 0.41 | tie |
| 40 | `CASE WHEN` as a group key, five keys | PASS | 0.51 | tie |
| 41 | `IN (…)` + int64 equality + `OFFSET` | PASS | 0.40 | tie. **DEVIATED (spelling only):** `IN (-1, 6)` is written `isin(ma.array([-1, 6], ma.int16()))`. `isin([-1, 6])` builds an int64 value set and silently matches **zero rows** against an int16 column — findings §1.1. Same semantics, different spelling. |
| 42 | `OFFSET 10000` past the end of the result | PASS | 0.45 | same empty-batch caveat as Q20 |
| 43 | `DATE_TRUNC('minute', ts)` as a group key + `ORDER BY` it | PASS | 0.55 | has to `order_by("key0")` — a computed group key has no name. Findings §3.2. |

"tie" means the row set matched on the multiset-of-ORDER-BY-values argument
described above, not on an exact row-for-row diff.

### Extra probes (not part of the 43)

| probe | what it shows |
|---|---|
| `p_binary_group_key` | `GROUP BY` a raw `binary` key aborts — the known `_PARALLEL_ALWAYS_ROWS` bug, reproduced deliberately |
| `p_isin_untyped` | `isin([-1, 6])` against an int16 column returns **0** — silently wrong |
| `p_isin_typed` | the same query with a dtype-matched `ma.array` returns 727,932 — correct |
| `p_binary_literal_ne` | `binary_col != lit(b"")` returns **0**; the cast spelling returns 69,354 — silently wrong |

---

## Prioritised gap list

Ordered by (queries unblocked) × (how small the change looks). Everything above
the line is worth doing before the alpha ships.

### 1. Fix the `arrays.mojo:2572` wrong-variant downcast — unblocks 3 queries (Q11, Q12, Q24), plus 1 known bug — **medium**

All three ABORTs, and the already-known binary-group-by abort, report at the
same line: `DynArray.as_type[T]` in `marrow/arrays.mojo:2572`, a downcast whose
`debug_assert` does not fire in release. Four unrelated code paths — parallel
group-by, filtered multi-morsel parquet scan, sort/take, and whole-table
`count()` over a `binary` column — all pick the wrong variant arm for `binary`.
That reads as one defect with four symptoms, not four defects.

The filter-path variant additionally manifests as a **SIGSEGV rather than a
clean abort**, and is **partly nondeterministic** (one shape crashed 3/3, a
narrower one crashed once and passed once), so a race or use-after-free is in
play. Bisection table and eleven reproducers in the findings doc, §2.2.

This is the single biggest item: it is the only thing standing between 39/43 and
42/43, and it is a memory-safety bug, which matters more than the query count.

### 2. Bind `count_star()` to Python — unblocks 0 queries, improves 28 — **tiny**

`marrow.expr.builders.count_star()` already exists and is already tested; there
is no `libmarrow.expr_count_star`. 28 of the 43 queries currently open with a
hand-rolled `lit(1).count()`. One `def_function` plus one line in
`_expr_column.py`. Highest leverage per line of code on this list.

### 3. Raise on a dtype mismatch between an operand and a literal / value set — prevents silent wrong answers — **small**

`isin([-1, 6])` against `int16` returns 0 rows; `binary_col != lit(b"")` returns
0 rows. Neither raises. Comparisons already promote (`col("x") == lit(-1)` on an
int16 column is fine), so the fix is to make `is_in` and the binary/string
comparison path either promote the same way or raise. Raising is enough and is
the smaller change. This is the only category on the list that produces *wrong
numbers* rather than a crash or a missing feature.

### 4. Make a zero-row `RecordBatch` exportable over the C Data Interface — unblocks 2 queries honestly (Q20, Q42) — **small**

`pa.record_batch(empty_batch)` raises `SystemError: … returned NULL without
setting an exception`. `collect()` is fine, only the export path fails. Q20 and
Q42 legitimately return no rows, so this is 5% of ClickBench, and the harness
carries a workaround that should be deleted with the fix.

### 5. Name computed group keys — improves ~12 queries' readability — **small**

`aggregate(by=[expr])` names the output `key0`. Every string group key in this
dataset must be a `.cast(string)`, so every string-grouping ClickBench query
loses its key column name, and Q43 has to `order_by("key0")`. Either accept a
mapping (`by={"URL": col("URL").cast(STR)}`) or add `Column.alias()` — the
`Aggregate` half already exists.

### 6. `with_columns` on `LazyTable` — improves ~8 queries — **small**

`project()` replaces the projection, so "derive one column, keep the rest" needs
every passthrough spelled out (Q25, Q27, Q28). A `with_columns(**named)` that
projects `[*existing, *named]` is a handful of lines in `expr.py`.

### 7. Validate column names in Python — no queries, big usability win — **tiny**

`order_by(("nosuch", False))` produces `SystemError: <cyfunction record_batch>
returned NULL without setting an exception`, which names something unrelated to
the mistake. `LazyTable` knows `column_names` at every step; a `KeyError` with
the available names, raised in `expr.py`, costs nothing.

---

### Below the line — not blocking the alpha

- **Regex — unblocks Q29 only; large.** `REGEXP_REPLACE` needs a regex engine,
  which the Mojo standard library does not have. Writing one (or binding one)
  for a single query is not a good trade; state the exclusion in the release
  notes and ship 42/43 as the ceiling. If it is ever wanted, note that Q29's
  particular pattern (`^https?://(?:www\.)?([^/]+)/.*$`) is a host extraction
  and would be served by a `url_host()` scalar function at a fraction of the
  cost — but that would be answering a different question than ClickBench asked.
- **`ma.timestamp("s")` without an explicit `None` timezone; small.** The dtype
  constructors are re-exported raw from `libmarrow`, so the binding layer's "no
  optional arguments" rule leaks to users. Diverges from PyArrow.
- **`Schema` is not iterable and exposes no field types to Python; small.**
  Makes building a narrower scan projection awkward — you have to re-declare
  every dtype by hand.
- **`len(record_batch)`; tiny.** `num_rows` is in the `repr` but unreachable.
- **A `(func, column)` tuple aggregate cannot take an expression; small.** The
  `Aggregate` object form covers it, but the two spellings look interchangeable
  and are not (Q30).
- **Repeating `.cast(string)` at every mention of a binary column; small.** A
  `read_parquet(..., cast={...})` or the `with_columns` above removes it.
- **The eight aggregate-function names are undocumented on the Python side.**
  `sum`, `product`, `mean`, `min`, `max`, `count`, `count_distinct`,
  `approx_count_distinct` — found by reading `resolve_agg`.

---

## Files

- `python/marrow/tests/clickbench_alpha.py` — the 43 queries and the runner.
  Every query carries its SQL in a docstring; every deviation is on the
  decorator, not in prose here.
- `docs/alpha-findings/e1-clickbench.md` — the API findings, with the crash
  bisection tables.
- `python/marrow/tests/clickbench_reference.py` — the DuckDB reference
  generator. A separate module because it needs the `bench` environment, where
  DuckDB lives and the marrow extension does not. Writes
  `.benchmarks/clickbench-reference.json` (gitignored).

Neither module is named `test_*`, so pytest does not collect them — the same
convention the existing eager `clickbench.py` follows. Both need the dataset,
which is not vendored.
