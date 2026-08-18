# ClickBench notes for the alpha

Measured against `~/Workspace/ClickBench/data/hits_0.parquet` on 2026-08-18, not
read off the ClickBench README. Everything here was verified by running it.

## The dataset

One partition, **1,000,000 rows / 105 columns / 2 row groups**, 122 MB.
`marrow.parquet.read_table` reads it (0.06 s for an 8-column projection), and
`to_batches()` yields **one batch per row group** — 450,560 rows in the first.

Column types are **not** what the ClickBench SQL implies:

| Column | SQL implies | Actually in the file |
|---|---|---|
| `EventDate` | `Date` | **`uint16`** (days since epoch) |
| `EventTime` | `DateTime` | **`int64`** (unix seconds) |
| `SearchPhrase`, `URL`, `Title`, `Referer`, `MobilePhoneModel` | `String` | **`binary`** |

Whole-file type census: 48 × `int16`, 28 × `binary`, 19 × `int32`, 9 × `int64`,
1 × `uint16`.

## What that changes

### 1. String kernels do not accept these columns
`BinaryType` conforms to `BinaryLikeType` **only**; `StringType` conforms to
`StringLikeType` (a sub-trait). The string kernels — `upper`/`lower`/`like`/
`ilike`/`length` and friends in `marrow/kernels/string.mojo` — dispatch through
`dispatch_stringlike`, which will never match `binary`.

**Path:** cast first. `BinaryLikeCast` (`marrow/kernels/cast.mojo:747`) already
covers binary/large_binary/utf8/large_utf8 in every direction, so
`col("URL").cast(string)` is the supported spelling. Verified present; not yet
verified for cost — it may rebuild through a builder rather than being a
metadata-only reinterpret, which would be an O(n) tax per query.

### 2. Date predicates are integer predicates
Q37–Q42 filter `EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'`. With a
`uint16` column this is either an integer comparison against the epoch-day
number, or a `cast(EventDate, date32)` followed by a date comparison. The
string→date cast work is therefore **not** on the ClickBench critical path —
it is general usability work, and must not be treated as a blocker for the
query suite.

Q19 (`extract(minute FROM EventTime)`) and Q43 (`DATE_TRUNC('minute', EventTime)`)
likewise need `EventTime` (int64 unix seconds) cast to a timestamp first.

### 3. `to_batches()` is per-row-group
Any harness that aggregates one batch is aggregating one row group, not the
file. Drive the whole file through the plan layer (`parquet_scan`), or
concatenate, but do not silently benchmark 45% of the data.

## Crash found while probing

Grouping by a `binary` key **hard-aborts** (process abort, not an exception)
once the row count crosses into the parallel grouping path:

```
ABORT: ./marrow/arrays.mojo:2572:23: get: wrong variant type
```

| rows | key type | result |
|---|---|---|
| 1,000 | binary | ok |
| 50,000 | binary | ok |
| 200,000 | binary | **abort** |
| 200,000 | string | ok |

The boundary is `_PARALLEL_ALWAYS_ROWS = 200_000` in
`marrow/kernels/groupby.mojo`; 50,000 sits below `_PARALLEL_MIN_ROWS = 60_000`
and takes the serial path. So the defect is in the parallel/radix path, and
`arrays.mojo:2572` is `DynArray.as_type[T]` — a wrong-variant downcast whose
`debug_assert` does not fire in release.

This is being fixed on its own branch. It matters because ClickBench groups by
`SearchPhrase`, `URL`, `Title`, `MobilePhoneModel` and `Referer` — all binary
in this file — in Q11–Q15, Q22–Q23, Q34, Q37–Q38 and Q40.

## Query-suite guidance

- Cast the binary columns to `string` once, in the scan projection, rather than
  per predicate.
- `COUNT(*)` is not `count(col)`: `CountKernel` counts **valid** values. On
  nullable columns the two differ.
- Q29 (`REGEXP_REPLACE`) has no kernel and is out of scope for the alpha — 42 of
  43 is the target, and the exclusion should be stated in the release notes
  rather than quietly dropped.
- Q24 is `SELECT *` over 105 columns; check it is actually expressible before
  counting it as covered.
