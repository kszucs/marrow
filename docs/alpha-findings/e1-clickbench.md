# E1 — Writing 43 real queries against the lazy Python frontend

Findings from expressing all 43 ClickBench queries through `marrow.expr`'s
`LazyTable` on `alpha` (`8365395`), against
`~/Workspace/ClickBench/data/hits_0.parquet` (1,000,000 rows, 105 columns).
Everything below was observed by running it, not inferred from the source.

The headline is that the frontend is **good**. 39 of the 43 queries express
naturally, three of the four that do not are memory-safety crashes rather than
missing features, and only one query (Q29, `REGEXP_REPLACE`) is a genuine
capability gap. What follows is the friction, in descending order of how much it
cost me.

---

## 1. Correctness bugs found while writing the queries

### 1.1 `isin()` with a Python list silently returns zero rows — **wrong answers, no error**

Q41 filters `TraficSourceID IN (-1, 6)`. `TraficSourceID` is `int16`.

```python
col("TraficSourceID").isin([-1, 6])                       # -> 0 rows
col("TraficSourceID").isin(ma.array([-1, 6], ma.int16())) # -> 727,932 rows
```

`Column.isin` builds the value set with `array(list(values))`, which infers
`int64`. Membership against a differently-typed value set then matches nothing
and **raises nothing**. This is the worst class of bug in the whole surface: it
turns a correct query into a plausible, silent zero.

Two fixes, and both are worth doing:

- `Column.isin` cannot know the column's dtype (that is the point of the runtime
  lane), so it cannot pick the right literal type. The *kernel* can: `is_in`
  should either promote the value set to the column's type or raise
  `is_in: value set is int64 but the column is int16`. Raising is enough.
- Longer term, the same promotion the comparison operators already do.
  `col("TraficSourceID") == lit(-1)` works fine — an `int16`/`int64` compare
  promotes. Only `isin` does not, which makes the inconsistency the surprise.

Reproduced as `p_isin_untyped` / `p_isin_typed` in
`python/marrow/tests/clickbench_alpha.py`.

### 1.2 `binary_col != lit("")` silently returns zero rows — same shape

```python
col("SearchPhrase") != lit(b"")                    # -> 0 rows
col("SearchPhrase").cast(ma.string()) != lit("")   # -> 69,354 rows  (correct)
```

Comparing a `binary` column against a literal built by `lit(b"")` matches
nothing rather than raising. Same root cause as 1.1 — a type mismatch between
operand and literal resolves to "no rows" instead of an error. Reproduced as
`p_binary_literal_ne`.

Together, 1.1 and 1.2 are one finding: **a dtype mismatch between an operand and
a literal/value-set is not an error anywhere in the runtime lane; it is an empty
result.** Every one of these cost me a debugging cycle where the query looked
right and the number looked plausible.

### 1.3 A zero-row `RecordBatch` cannot cross the C Data Interface

```python
batch = t.filter(col("UserID") == lit(435090932899640449)).select("UserID").collect()
batch                    # RecordBatch(num_rows=0, schema=Schema(fields=[UserID: int64]))
pa.record_batch(batch)   # SystemError: <cyfunction record_batch> returned NULL
                         #               without setting an exception
```

`collect()` is fine; `to_pyarrow()` is not. Two of the 43 queries (Q20, Q42)
legitimately return no rows, so an empty result is not an edge case — it is 5%
of ClickBench. The harness works around it by short-circuiting on
`num_rows == 0`; that workaround should be deleted when this is fixed.

The `SystemError: … returned NULL without setting an exception` wording means
the export path returned a failure without setting a Python exception, so
CPython invents one. Whatever raises inside `c_data.mojo` for a zero-length
array needs to set the error properly, and ideally not fail at all.

### 1.4 An unknown sort-key name produces the same `SystemError`, not a `KeyError`

```python
t.aggregate(by=["AdvEngineID"], c=lit(1).count()).order_by(("nosuch", False)).collect()
# SystemError: <cyfunction record_batch> returned NULL without setting an exception
```

I hit this by aliasing an aggregate `c` and then ordering by `count`. The
message names `record_batch`, which is nowhere near the mistake. `LazyTable`
knows its own `column_names` at every step, so `order_by` / `select` / `filter`
could validate names in Python and raise `KeyError: no column 'nosuch'; have
[...]` for free. That single check would have saved me twenty minutes.

---

## 2. Crashes (memory safety, not features)

These take the process down. They are the reason the harness runs every query in
a subprocess.

### 2.1 Grouping by a raw `binary` key aborts above ~200,000 rows

Already known (`_PARALLEL_ALWAYS_ROWS` in `marrow/kernels/groupby.mojo`,
`ABORT: ./marrow/arrays.mojo:2572:23: get: wrong variant type`). Casting the key
to `string` avoids it, which is what every grouped query in the suite does.
Reproduced deliberately as `p_binary_group_key`.

### 2.2 A new one: a *filter* touching a `binary` column can SIGSEGV or abort

This is **not** 2.1 — no grouping is involved, and it fires on plain whole-table
counts. It is what kills Q11 and Q12.

```python
t = ma.read_parquet(HITS)                                    # all 105 columns
t.filter(col("MobilePhone") != lit(0)).aggregate(by=[], c=lit(1).count()).collect()
# SIGSEGV (exit -11), reproducibly
```

What I established by bisection:

| shape | result |
|---|---|
| `MobilePhone != 0`, full 105-column projection | **SIGSEGV** |
| `MobilePhone != 0`, projection = `{MobilePhone: int16, WatchID: int64}` | ok |
| `MobilePhone != 0`, projection = `{MobilePhone: int16, MobilePhoneModel: binary}` | **SIGSEGV** (flaky — passed on one later run) |
| `MobilePhone != 0`, `morsel_size=1_000_000` (one morsel) | ok |
| `MobilePhone != 0`, `morsel_size=64` or `8192` | **SIGSEGV** |
| same predicate, same columns, `ma.scan()` of an in-memory batch | ok |
| `MobilePhone != 0` then `.limit(3)` (no aggregate, no drain) | ok |
| `MobilePhone == 0` (matches ~97% of rows) | ok |
| `MobilePhone == 1`, `MobilePhone > 0` | **SIGSEGV** |
| `AdvEngineID != 0`, `IsRefresh != 0`, `IsMobile != 0`, `IsLink != 0`, `Sex = 2` | ok |
| `Age = 31` | **SIGSEGV** |
| `aggregate(by=[], c=("count", "MobilePhoneModel"))` — count a binary column | `ABORT arrays.mojo:2572 get: wrong variant type` |

So:

- It needs a **binary column in the scan projection** and a **multi-morsel
  parquet scan**. One morsel, or an in-memory source, or a numeric-only
  projection, and it goes away.
- It is **not** selectivity: `Age = 31` (433,099 rows) crashes while
  `IsRefresh <> 0` (256,201 rows) does not, and `MobilePhone <> 0` (27,160)
  crashes while `IsLink <> 0` (31,048) does not.
- It is **not** predicate pushdown: replacing the predicate with
  `(col("MobilePhone") + lit(0)) != lit(0)`, which `pruning.mojo` cannot use,
  crashes identically.
- It is **partly flaky**. `{MobilePhone, MobilePhoneModel}` crashed on one run
  and passed on a later identical one, while the full-projection form crashed
  three times out of three. A nondeterministic memory fault, i.e. a race or a
  use-after-free, fits better than a deterministic index error.
- `count()` over a `binary` column is a separate, deterministic abort at the
  same `arrays.mojo:2572` wrong-variant downcast as 2.1 — `CountValid.resolve`
  presumably resolves to the wrong array arm for `binary`.

### 2.3 `SELECT *` (105 columns) + `ORDER BY` + `LIMIT` aborts

Q24. `ABORT: ./marrow/arrays.mojo:2572:23: get: wrong variant type`, the same
wrong-variant downcast. Sorting a wide table whose columns include `binary`; the
narrower `select("WatchID", "EventTime", "URL")` form of the same query also
crashed, so it is not purely about width.

All three of 2.1/2.2/2.3 report at the same line, `DynArray.as_type[T]` in
`marrow/arrays.mojo:2572`. Whatever is choosing that downcast is picking the
wrong arm for `binary` in at least three unrelated code paths (parallel group-by,
filtered parquet scan, sort/take). That smells like one bug, not three.

---

## 3. API ergonomics

### 3.1 `count_star()` exists in Mojo but is not bound to Python

`marrow.expr.builders.count_star()` is there; `libmarrow` has no
`expr_count_star` and `marrow.count_star` does not exist. **28 of the 43
queries** need real `COUNT(*)` semantics, so every one of them opens with the
same incantation:

```python
def count_star():
    return lit(1).count()      # a literal is valid on every row
```

This is a one-line binding away and it is the single highest-leverage item on
the list purely by how many queries touch it.

### 3.2 A computed group key has no name

```python
t.aggregate(by=[col("URL").cast(ma.string())], c=count_star())
# -> schema: key0: string, c: int64
```

Bare-name keys keep their name (`by=["SearchEngineID", …]` yields
`SearchEngineID`); anything computed becomes `key0`, `key1`, …. Since every
string group key in this dataset *must* be a cast, **every string-grouping
query in ClickBench loses its key column name**. Q43 then has to
`order_by("key0")`, which reads like a typo.

`aggregate(by=...)` should take the same keyword form the aggregates do —
`t.aggregate(by={"URL": col("URL").cast(STR)}, PageViews=count_star())` — or at
minimum honour `Column.alias()` on a key. There is an `Aggregate.alias` but no
`Column.alias`.

### 3.3 `project()` is `SELECT`, so there is no `with_columns`

`project(**named)` replaces the projection entirely. Q25/Q27 sort on `EventTime`
but select only `SearchPhrase`, so the cast of `SearchPhrase` and the passthrough
of `EventTime` both have to be spelled out and then re-`select`ed:

```python
t.filter(...)
 .project(SearchPhrase=s("SearchPhrase"), EventTime=col("EventTime"))
 .order_by("EventTime").limit(10).select("SearchPhrase")
```

With a `with_columns(SearchPhrase=...)` that is one verb. The docstring is honest
about the design ("it is `SELECT <these>`, not `with_columns`") — the point is
that `with_columns` is the one that comes up in practice, because the common
shape is "derive one column, keep the rest".

### 3.4 The binary/string cast has to be repeated at every mention

Q28 casts `URL` twice (once for `<> ''`, once for `length()`), Q23 casts
`SearchPhrase`, `URL` and `Title` and mentions two of them twice. There is no way
to say "read this column as a string for the whole query" other than an extra
`project` node. Combined with 3.3 this is the single most repetitive thing about
writing marrow queries against this file. A `read_parquet(..., cast={"URL":
string})` or a `with_columns` would both fix it.

### 3.5 `Schema` is not iterable and has no `field()` accessor from Python

```python
[f for f in t.schema]      # TypeError: 'Schema' object is not iterable
```

`LazyTable.column_names` exists, but there is no way to get a column's *type*
without going through PyArrow. That makes "build a narrower projection schema
for `read_parquet`" — which is how you scope a scan — surprisingly awkward: you
have to re-declare every field's dtype by hand. `Schema.__iter__`,
`Schema.__getitem__` and `Schema.types` would all be cheap.

### 3.6 `ma.timestamp("s")` needs an explicit `None` timezone

```python
ma.timestamp("s")        # TypeError: <mojo function>() missing 1 required positional argument
ma.timestamp("s", None)  # ok
```

`marrow/__init__.py` re-exports the dtype constructors straight out of
`libmarrow`, so the strict "no optional args" rule of the binding layer leaks all
the way to the user. Every other dtype in the file follows PyArrow's
`pa.timestamp("s")` spelling, so this one reads like a bug. A three-line Python
wrapper — the same treatment `Array`/`RecordBatch` get — fixes it.

### 3.7 There is no `RecordBatch.__len__`

```python
len(batch)   # TypeError: object of type 'RecordBatch' has no len()
```

`num_rows` is in the `repr`, so it exists; it is just not reachable. PyArrow has
both `len()` and `.num_rows`.

### 3.8 A tuple aggregate can only name a bare column

`t.aggregate(by=[...], s=("sum", "ResolutionWidth"))` marshals to
`DynValue.column(name)`, so the tuple form cannot aggregate an *expression*.
Q30's `SUM(ResolutionWidth + 1)` has to switch spelling mid-dict:

```python
aggs = {"s0": ("sum", "ResolutionWidth")}
for k in range(1, 90):
    aggs[f"s{k}"] = (col("ResolutionWidth") + lit(k)).sum()
```

Not a blocker — the `Aggregate` object form covers everything — but the two
spellings look interchangeable and are not. Accepting a `Column` in the tuple's
second slot would remove the seam.

---

## 4. What worked better than expected

Worth recording, because a findings doc that is all complaints is misleading.

- **`HAVING` needs no special support.** `aggregate(...).filter(col("c") > lit(100000))`
  just works, because an aggregate is a relation like any other. Q28 was the one
  query I expected to be blocked and it was the easiest.
- **Grouping by arbitrary expressions works** — `if_else` (Q40), a literal (Q35),
  `ClientIP - 1` (Q36), `date_trunc` (Q43), `extract(minute …)` (Q19). Five keys
  of mixed type in one `aggregate` (Q40) worked first try.
- **90 aggregates in one pass** (Q30) works and takes 0.78 s.
- **Ninety percent of the queries are one fluent chain** and read like the SQL.
  The `filter → aggregate → order_by → limit` shape maps onto ClickBench
  essentially 1:1.
- **`.cast(ma.string())` on a `binary` column is cheap** — the cast-then-group
  queries run in 0.40-0.45 s against 0.35 s for the numeric ones, so the
  workaround for the binary columns costs about 15%, not a rewrite.
- **`limit(n, offset)` is correct**, including `OFFSET 10000` past the end of the
  result (Q42).
- **Every query runs in under 1.1 s** on a 1M-row, 105-column file, most in under
  0.5 s.

---

## 5. The expression system and aggregation specifically

- `DynValue`'s operator coverage is genuinely complete for this workload:
  arithmetic, comparison, boolean, `~`, `like`/`ilike`, `length`, `min`/`max`
  over strings, `cast`, `date_trunc`, `minute`, `if_else`, `coalesce`, `isin`.
  Nothing in ClickBench except regex needed a kernel that was not there.
- The aggregate set — `sum`, `product`, `mean`, `min`, `max`, `count`,
  `count_distinct`, `approx_count_distinct` — covers every ClickBench aggregate.
  `count_distinct` being *exact* is worth advertising; ClickBench's reference
  answers assume it.
- The **name→aggregate resolution is stringly typed and unlisted**. There is no
  `marrow.aggregates` or a docstring enumerating the eight names; I found them by
  reading `resolve_agg` in `marrow/expr/aggregates.mojo`. `Column.aggregate("nope")`
  raises with a good message, but a user should not have to guess.
- **`Column.alias()` does not exist**, only `Aggregate.alias()`. That is what
  makes 3.2 unfixable from the user's side.
- The one asymmetry that bit me: `Column.count()` is *not* `COUNT(*)`, and there
  is nothing in the Python docstring that says so — `_expr_column.py` just says
  `return self.aggregate("count", alias=alias)`. Given how much of ClickBench
  depends on the distinction, that docstring should carry the warning even after
  `count_star` is bound.
- `is_null` / `is_valid` / `fill_null` are still absent from `Column` (the TODO
  is in `_expr_column.py`). No ClickBench query needs them, but every real query
  suite does.
