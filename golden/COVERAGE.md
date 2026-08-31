# Golden corpus coverage

What the corpus asks, what it deliberately records as *unsupported*, and what
it consciously leaves out. One case is one **distinct semantic question**; a
case that only re-asks another one with different constants does not earn its
place, and the redundancies still in the tree are listed at the end.

Counts are of files in `golden/cases/`. Regenerate the expectations — always
from DuckDB, never from marrow — with:

```bash
pixi run -e bench python golden/runner.py
pixi run -e dev pytest golden
```

## The three markers

| Marker | Compiled? | Meaning |
|---|---|---|
| *(none)* | yes | marrow answers this, and the answer matches DuckDB |
| `-- xfail <reason>` | **yes** | marrow answers this **wrongly**. Strict, so a fix turns the case red |
| `-- skip mojo` | **no** | marrow has no API for it. The SQL and the DuckDB answer are recorded; the body is the spelling we would want |

The distinction is load-bearing, not stylistic. **The whole suite is one
compilation unit**, so a single compile error fails every case in the run
(CLAUDE.md, "One selection = one compilation unit"). `xfail` still compiles and
runs the body, so it is only usable when the API exists; `skip mojo` removes
the case from the generated `test_cases.mojo` entirely, which is the only safe
marker for a feature that cannot be spelled.

A `skip mojo` body is therefore **never compiled and never verified**. It is a
proposal, written in the shape the verb would most likely take, so the case
documents an intended API as well as a missing answer. Turning one on means
implementing the feature and then deleting one line.

## Fixtures

Ten tables, each with a stated reason for every value it holds
(`runner.TABLES`). Nulls appear in every column that can carry one; float
columns hold only exactly-representable values so a float aggregate compares
two implementations rather than two roundings.

| Fixture | For |
|---|---|
| `basic` | grouping with a repeated and a NULL key; two numeric columns whose nulls are on different rows |
| `nulls` | an all-null column, a null-free one, exact means and variances |
| `kleene` | the 3x3 three-valued table through *derived* predicates |
| `flags` | the same table through actual bool columns |
| `emp` / `dept` | join keys: unique match, duplicated match, unmatched on both sides, NULL key |
| `sales` / `regions` | the type matrix (int32, float64, bool, string) plus a string-keyed join with unmatched rows both ways |
| `words` | case, surrounding whitespace, the empty string, a multi-byte character, a null |
| `floats` | NaN, both infinities, `-0.0`, and an integer column with 0 and negatives |
| `events` | naive timestamps and dates: a leap day, the last microsecond of a year, a repeat, a null |
| `nums` | cast inputs: values where truncation and rounding disagree, a string that does not parse |
| `text` | string *structure*: a separator, a string without one, the empty string, mixed case with padding, multi-byte; plus a count column holding 0 and -1 |
| `lists` | a list, an empty list, a null list, a list with a null element, a one-element list |
| `nested` | a struct with a null field vs. a null struct; a map with a present key, an empty map, a null map |
| `edges` | int64 max and min, a zero divisor, a negative divisor |

`text`, `lists`, `nested` and `edges` are new. `nested` is reached only by
`skip mojo` cases, so no compiled case depends on struct or map IPC.

## What the corpus covers

### Relational verbs

`filter`, `select` (reorder, subset), `project`, `with_columns` (append **and**
replace-in-place), `drop`, `rename`, `limit`/`offset` (zero, beyond the end,
with an offset, *followed by* a filter), `sort_by` (asc, desc, nulls first,
nulls last, mixed directions, three keys of mixed types, an all-null key, a
**computed** key), `aggregate` with and without keys, `join` in all seven
kinds, and the compositions: filter→aggregate, join→aggregate,
aggregate→join, aggregate→aggregate, filter→join→sort→limit.

### Predicates and three-valued logic

Every comparison operator on int64, int32, float64, string and bool; `AND`,
`OR`, `NOT` and `XOR` over both derived predicates and real bool columns, with
the complete Kleene table for each; `IS NULL` / `IS NOT NULL` on numeric and
string columns; `x = x` being **null** on the null row; a predicate that
excludes nulls without mentioning them.

### Arithmetic and math

`+ - * / // %`, unary minus, `abs`, `sign`, `floor`, `ceil`, `round`, `trunc`,
`sqrt`, `exp`, `ln`, `pow`, `is_nan`, `is_inf`; type **promotion** in both
arithmetic and comparison (int32 + float64 → double); a three-level fused
expression; null propagation through every level.

### Aggregates

`sum` (int32 widening to int64, int64, float64), `product`, `mean`, `min`/`max`
(int64, int32 *not* widening, float64, string, date, timestamp, at the int64
extremes), `count` (column, `count(*)`), `count(DISTINCT)` (string, int64,
grouped), `variance` (population), `stddev` (sample), grouped variance with
`ddof=1` on singleton groups, `HAVING`, aggregates over a computed input, a
computed group key, two group keys, an aggregate of an aggregate; and the
empty-input family: an all-null column, zero rows ungrouped, zero rows grouped.

Group keys of every type: string, int32, float64, bool, date, a computed
string, a computed boolean, and NULL.

### Strings

`upper`, `lower`, `strip`/`lstrip`/`rstrip`, `reverse`, `capitalize`,
byte `length`, `starts_with` (literal **and** column pattern), `ends_with`,
`contains`, `LIKE` (prefix, contains, `_` single-character, and a general
backtracking pattern), `ILIKE` (ASCII and a non-ASCII fold), equality and
ordering comparisons, the empty string as a value, and bytewise sort order over
multi-byte data.

### Casts

int64 → int32 / float64 / bool / string; bool → int; float → int; string → int
/ double; float → string.

### Temporal

`year`, `month`, `day`, `hour`, `minute`, `second`, `quarter`, `day_of_week`,
`day_of_year` on both timestamp and date columns; `date_trunc` at `day`,
`month`, `quarter`, `second` and (through a filter) `year`; min/max; sort;
filter; group by a date and by a truncated timestamp.

### Nested

`array_length` over a list column — the one list verb the expression layer has.

### Window functions

Seven cases over `OVER (PARTITION BY ... ORDER BY ...)`. `window_row_number`
takes the no-partition, no-frame base case; `window_rank_and_dense_rank`
separates the two functions that differ only on ties;
`window_partitioned_running_sum` is an aggregate under the default `RANGE`
frame, restarting per partition and answering NULL for an all-null one;
`window_explicit_rows_frame` is the `ROWS` counterpart, which agrees with it
only when the order key has no duplicates; `window_lag_and_lead` reads
neighbouring rows and both partition edges; `window_first_and_last_value` is
the frame trap — `last_value` is the *current* row, not the partition's last;
and `window_qualify` filters on a window function's output.

Not covered here and not implemented: `RANGE` frames with explicit numeric
bounds, `EXCLUDE`, and `NTILE`/`PERCENT_RANK`/`CUME_DIST`/`NTH_VALUE`. All
seven keep `-- skip python` — the window surface is Mojo-only.

## Recorded as unsupported

78 cases carry `-- skip mojo`. Each names, in its prose, what is missing.

**Set operations** (4) — `setop_union_all`, `setop_union_distinct`,
`setop_except`, `setop_intersect`. There is no set-operation node in
`logical.mojo`. `UNION`/`EXCEPT`/`INTERSECT` also treat NULL as equal to
itself, which no other part of the corpus does.

**DISTINCT ON** (1) — `distinct_on_first_row_per_key`. `SELECT DISTINCT` is an
`aggregate` with keys and no aggregates; `DISTINCT ON` keeps whole rows and
cannot be.

**GROUPING SETS / ROLLUP / CUBE** (3) — `grouping_rollup`, `grouping_cube`,
`grouping_sets_explicit`. `Aggregate` carries one key list. `grouping_rollup`
also asserts `GROUPING(k)`, without which the subtotal row's NULL key is
indistinguishable from the genuine NULL group.

**Aggregates** (13) — `agg_median`, `agg_quantile_continuous`, `agg_mode`,
`agg_arg_min_and_arg_max`, `agg_first_and_last_ordered`,
`agg_string_agg_ordered`, `agg_bool_and_or`, `agg_corr_and_covar`,
`agg_count_distinct_multi_column`, `agg_filter_clause`, `agg_sum_distinct`,
`agg_skewness_and_kurtosis`, `agg_bitwise`. Three different kinds of gap:
missing *kernels* (median, quantile, mode, skewness, kurtosis, bitwise),
missing *nodes* over kernels marrow already has (`bool_and`/`bool_or` over
`AnyKernel`/`AllKernel`), and missing *shapes* — `Aggregate[Agg, A]` binds one
operand, so `arg_min`, `corr`, an ordered `first`, a `FILTER` clause and a
`DISTINCT` modifier have nowhere to go.

**String functions** (16) — `string_substr`,
`string_concat_operator_propagates_null`, `string_concat_function_skips_null`,
`string_concat_ws`, `string_replace`, `string_split_part`,
`string_regexp_matches`, `string_regexp_replace`, `string_regexp_extract`,
`string_lpad`, `string_position`, `string_repeat`, `string_left_and_right`,
`string_trim_characters`, `string_length_counts_characters`,
`string_ascii_code_point`. The three-way NULL split across `||`, `concat` and
`concat_ws` is deliberate: they are one operation with three answers.
`string_length_counts_characters` is recorded here rather than as an `xfail` on
`string_length_counts_bytes` because character length and byte length are
different functions and marrow's answer to the one it implements is correct.

**Math functions** (8) — `math_greatest_and_least`, `math_round_to_digits`,
`math_round_half_to_even`, `math_log_bases`, `math_trigonometry`,
`math_bitwise_operators`, `math_gcd_and_lcm`,
`math_signbit_separates_negative_zero`. `greatest`/`least` *skip* nulls, the
opposite of every arithmetic operator. `signbit` is the only function that can
tell `-0.0` from `0.0`, which is why the `floats` fixture carries both.
The left shift in `math_bitwise_operators` takes `abs(n)` because DuckDB
refuses to shift a negative at all — that one operation is not comparable.

**Temporal** (13) — `temporal_literal_comparison`, `temporal_date_diff`,
`temporal_add_month_interval`, `temporal_interval_between_timestamps`,
`temporal_last_day`, `temporal_epoch_seconds`, `temporal_iso_week_and_year`,
`temporal_strftime`, `temporal_strptime`, `temporal_make_date`,
`temporal_day_and_month_name`, `temporal_age`, `temporal_timezone_attach`.
`temporal_literal_comparison` is the smallest and most load-bearing: `lit` has
numeric and string-like overloads only, so no date or timestamp constant can
be written, which is why `temporal_filter_timestamp` compares against a
derived column instead.

**Nested types** (8) — `nested_list_element`, `nested_list_contains`,
`nested_list_slice_length`, `nested_unnest`, `nested_list_sum`,
`nested_struct_field`, `nested_map_lookup`, `nested_map_cardinality`.
`nested_list_contains` is another node-over-existing-kernel gap
(`kernels/nested.mojo` has `array_contains`). Results are kept flat — a length,
a sum, an element, an unnested column — because the expectation block cannot
render a list, struct or map.

**Decimals** (3) — `decimal_sum_keeps_scale`, `decimal_multiply_widens_scale`,
`decimal_division_rounds`. `Column[T]` binds `T: NumericType` and decimal is
not one, so no decimal column can enter an expression at all, though
`Decimal128Array` and the dtypes exist. All three render the result as
**text** so that the scale is asserted and not merely the value: `7.00` and
`7.0` are the same number and different answers.

**Predicates and subqueries** (5) — `filter_in_literal_list`,
`filter_not_in_list_with_null`, `filter_is_distinct_from`, `subquery_scalar`,
`subquery_correlated_scalar`. `IN` is a node-over-existing-kernel gap
(`kernels/membership.mojo` has `is_in`), and `NOT IN` with a NULL in the list
matching *nothing* is the trap a hash-set implementation gets wrong in the
natural way.

**Joins** (4) — `join_cross`, `join_non_equi`, `join_asof`,
`join_left_with_residual_condition`. The last is the semantic trap of the
group: an outer join's non-key `ON` predicate must be applied *before* the
null-widening, so moving it to a `WHERE` — the rewrite that looks equivalent —
drops the row instead of widening it.

## Recorded as wrong — `xfail`

Four cases, all strict, so fixing any of them turns the case red and forces
the marker's removal.

| Case | marrow | DuckDB |
|---|---|---|
| `group_by_float_key` | `-1.25` and `0.5` collapse into one group | three distinct float keys, three groups |
| `math_floordiv_truncates_toward_zero` | `-1 // 3` is `-1` — `//` floors, Mojo/Python semantics | `0` — SQL's `//` truncates toward zero |
| `math_modulo_sign_follows_dividend` | `-1 % 3` is `2` — the remainder takes the *divisor*'s sign | `-1` — SQL's takes the *dividend*'s |
| `math_integer_division_by_zero` | `10 // 0` is `10` and `10 % 0` is `0`: the kernels substitute 1 for a zero divisor | `NULL` for both |

The last three are one root cause with three faces. `FloordivKernel` and
`ModKernel` are `a // b` and `a % b` with `b == 0` replaced by 1, evaluated
inside a SIMD lane that can neither raise nor produce a null. Getting SQL
semantics needs a sign correction on the result *and* a validity mask derived
from the divisor — the same shape `NumericBinary.validity` already computes
from its operands.

`math_mod_int64` deliberately asks the *agreeing* form of the same question,
`((n % 3) + 3) % 3`, so the corpus records both the working case and the
divergence.

## Consciously omitted

Not gaps in the corpus — questions it cannot or should not ask.

- **Anything non-deterministic.** `random`, `setseed`, `uuid`, `SAMPLE`,
  `now()`, `current_date`, hash order, and `approx_count_distinct` — which
  marrow *has*, but whose HyperLogLog will not agree with DuckDB's on a
  sketch-by-sketch basis. Likewise unordered `first`/`last`/`arbitrary`; the
  corpus asks only the `ORDER BY`-carrying form.
- **`LIMIT`/`OFFSET` without an `ORDER BY`.** No total order, so no assertable
  answer.
- **NaN and infinity as *results*.** `render_value` writes floats with `repr`
  and `parse_value` reads them with `ast.literal_eval`, and `nan`/`inf` are not
  Python literals. `math_round` and friends filter to the finite rows; the
  values are still reachable as *inputs*, and `math_is_nan` / `math_is_inf`
  assert them as booleans.
- **List-, struct- and map-valued results.** The expectation block is typed TSV
  over `runner.TYPES`, which holds seven flat types. Every nested case projects
  something flat instead — which is why `array_agg` is absent while
  `string_agg` is present.
- **Integer overflow.** DuckDB *raises*; marrow wraps. A golden case cannot
  express "raises", and `TRY(...)` would assert DuckDB's null against marrow's
  wrapped value — a divergence better recorded here in prose than as a case
  whose expectation is an artefact of the twin's error handling. The `edges`
  fixture carries int64 max and min so the inputs exist when marrow grows a
  checked-arithmetic mode.
- **`BETWEEN`.** Sugar over two comparisons; `filter_and` already asks it.
- **CTEs.** Syntax, not semantics: a `WITH` is a `DynRelation` bound to a name,
  and copying a plan is already an O(1) share.
- **`PIVOT`/`UNPIVOT`, `UNION BY NAME`, `SELECT * EXCLUDE/REPLACE`,
  `LATERAL`.** DuckDB-specific or reshaping surface, far outside anything
  marrow plans to answer.
- **`param` / `Bindings`.** A marrow feature with no DuckDB counterpart — a
  parameter with a default is indistinguishable from a literal in the twin.
  Covered by `marrow/expr/tests/test_params.mojo`.
- **`JOIN_ALL`.** The constant is in the prelude, but `JOIN_ALL` and
  `JOIN_INNER` are the same value, so a case over it would duplicate
  `join_inner`.
- **The `-0.0` vs `0.0` distinction in results.** `values_equal` compares
  through `Float64Array.__eq__`, where the two are equal, so an expectation of
  `-0.0` would pass against `0.0`. `math_signbit_separates_negative_zero`
  records the only function that could tell them apart, and it is unsupported.

## Redundancies in the existing corpus

Flagged, not deleted.

- **`aggregate_sum_by_key` and `agg_sum_int32`/`aggregate_no_keys`** overlap
  heavily with `filter_then_aggregate` and `aggregate_several`; the `aggregate_*`
  and `agg_*` prefixes are two names for one family and could be merged under
  one prefix.
- **`subquery_exists` and `join_semi`** are the same plan under two SQL
  spellings, as are **`subquery_not_exists` and `join_anti`**. The pairs exist
  to document that `EXISTS` is a semi-join, which is worth one case each, not
  two — the `subquery_*` twins carry no additional question.
- **`math_is_nan` / `math_is_nan_filter`** and **`math_is_inf` /
  `math_is_inf_filter`** each ask one predicate twice, once projected and once
  in a `WHERE`. The filter path is exercised by dozens of other cases.
- **`order_by_asc` and `sort_float_key`/`sort_int32_key_desc`** differ only in
  the key's type, which does change the sort kernel, so these are defensible;
  **`order_by_two_keys` and `sort_three_keys_mixed_types`** are not — the
  second subsumes the first.
- **`cond_coalesce` and `cond_fill_null_with_literal`** reach the same
  `ConditionalBinary` node; only the operand's shape differs (column vs.
  scalar), which is a real difference but a thin one.
