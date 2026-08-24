# Changelog

## [Unreleased]

> **Entries below cite `docs/alpha-findings/*.md`.** Those twenty logs were
> folded into `docs/backlog.md` and deleted — open items became its `A`-IDs,
> measurement traps went to its §0, ruled-out designs to §7, and defend-this
> findings to §8. To read an original: `git show c0831f5:docs/alpha-findings/README.md`,
> which indexes all twenty.

### Tests

- **26 new golden cases, covering sixteen scalar expression methods that had
  none.** The numeric maps (`abs`, `sign`, `floor`, `ceil`, `round`, `sqrt`,
  `exp`, `ln`), the `product` aggregate, the operators `/`, `%`, `**` and unary
  `-`, the float predicates `is_nan` / `is_inf`, and the string maps
  `capitalize`, `reverse`, `lstrip`, `rstrip`, `contains` and `endswith` were
  all reachable from both lanes and asserted by neither. The numeric cases run
  on the `floats` fixture, so NaN, both infinities, `-0.0` and a null are in
  scope; the string cases run on `words`. Every case is written with the
  *method* spelling both lanes already share, so none of them needed a new
  shim — the corpus's convergence metric is unchanged.

  **One defect fell out of it, and is recorded as an `-- xfail` rather than
  fixed**: marrow's `%` floors the way Python's does, so `-1 % 3` is 2, while
  SQL truncates toward zero and DuckDB, PostgreSQL and the standard all answer
  -1. `ModKernel` is `a % b` over Mojo SIMD, which carries Python's convention
  into a SQL-facing engine. Both lanes agree with each other and differ from
  the twin, so this is one kernel's semantics and not a lane divergence.

  Three constraints shaped the set, and each is recorded in the prose of the
  cases it touches. `exp` and `ln` are not correctly rounded under IEEE 754, so
  those two cases filter down to inputs whose result is exact (`exp(0)` is 1,
  `ln(1)` is 0) rather than comparing two implementations' last bits; `sqrt` is
  correctly rounded and takes ordinary inputs. DuckDB's `sign` returns TINYINT,
  which the expectation block's type map does not carry, so those twins cast to
  the type marrow returns. And DuckDB has neither `capitalize` nor `initcap`,
  so that twin spells the operation out as `upper(s[1:1]) || lower(s[2:])`.

- **A golden expectation cannot hold a NaN or an infinity.** `render_value`
  writes a float with `repr`, and `nan` / `inf` are not Python literals, so
  `parse_value`'s `ast.literal_eval` rejects them — and because that happens in
  `load_cases()`, a single such case takes down the whole corpus rather than
  just itself. Even past the parser, `pa.Table.equals` holds NaN unequal to
  NaN, so the case could never pass. The numeric cases over `floats` therefore
  either filter to the finite rows or pick an operation that maps the edge
  values to finite ones — `sign` is the one that covers NaN and both infinities
  without a filter. Worth lifting in the expectation format; until then it is a
  standing limit on what the corpus can assert about floats.

- **16 new golden cases covering the temporal expression family**, which was
  the largest wholly-untested area of the expression layer, and the `events`
  fixture (timestamp[us] + date32, with nulls, a leap day and the last
  microsecond of a year) now has cases reading it. They cover the nine field
  extractions (`year` `month` `day` `hour` `minute` `second` `quarter`
  `day_of_week` `day_of_year`) over both a timestamp and a date column,
  `date_trunc` to month and to day, temporal `min`/`max`, grouping by a date
  and by a truncated timestamp, sorting on a timestamp, and a timestamp
  comparison in a predicate.

  Two conventions had to be read off marrow rather than assumed, and each
  twin asks DuckDB marrow's question rather than the reverse: the extractions
  return **int32** where DuckDB returns int64 (so every twin casts), and
  `day_of_week` is the **ISO weekday with Monday = 0**, matching PyArrow's
  defaults, where DuckDB's `dayofweek` is Sunday = 0 and its `isodow` is
  Monday = 1 (so the twin is `isodow(ts) - 1`). `date_trunc` over a date
  column is a third: marrow preserves the input type, DuckDB widens to
  TIMESTAMP.

- **The expectation format understands dates and timestamps.** `runner.TYPES`
  gained `date32` and `timestamp` (microseconds, matching DuckDB's
  `TIMESTAMP`), rendered as quoted ISO-8601 — `'2021-06-15'`,
  `'2021-12-31T23:59:59.999999'` — for the reason the string cells are
  quoted: `mojo format` strips trailing whitespace inside a docstring, so no
  cell may leave a line ending in one. `golden/helpers.mojo`'s `values_equal`
  gained the matching arms, keyed on `is_date32()` / `is_timestamp()` rather
  than on a unit-pinned dtype equality.

- **A lane divergence, recorded as a strict xfail:** the AOT lane loses a
  temporal group key's name. `Relation.aggregate` names a key after its
  source column when `bound_column` finds one and `key<i>` when it does not;
  `NumericColumn`, `BoolColumn` and `StringColumn` override `bound_column`,
  but `TemporalColumn` and `ListColumn` inherit the `Value` default that
  always answers -1. So `GROUP BY d` on a date column yields `key0` in the
  AOT lane and `d` in the runtime lane. `temporal_group_by_date` holds the
  correct (DuckDB) expectation and carries the marker.

- **43 new golden cases, aimed at query shapes and the types they accept.**
  The corpus tested one key type per operation: every join keyed on `int64`,
  every group-by on `string` or `int64`, every sort on `int64`/`string`. It now
  covers subquery forms that lower to semi/anti joins (`IN`, `EXISTS`,
  `NOT EXISTS` — the twin is `NOT EXISTS` rather than `NOT IN`, because an anti
  join keeps a NULL-keyed row and `NOT IN` does not), self-joins, multi-key
  joins, `SELECT DISTINCT`, join-then-aggregate, aggregate-then-join, and a
  four-operator pipeline; and it asks each of `join`, `aggregate`, `sort` and
  `filter` the same question of `string`, `int32`, `float64` and `bool`. Two
  new fixtures carry the matrix: `sales` (int32/float64/bool/string, nulls
  throughout, prices chosen as exact binary fractions so a float sum is
  order-independent) and `regions` (a string join partner unmatched in both
  directions). **Four defects fell out of it** — see Fixes.

- **`-- xfail <reason>` in a case docstring.** Records a query marrow does not
  yet answer correctly: the SQL is right and the expectation is DuckDB's, so
  the corpus states the intended behaviour and stays green. The mark is
  **strict** and applies to both lanes, so a fix turns the case red and forces
  the marker out — and a lane divergence surfaces as an xpass, which is how
  the `SELECT DISTINCT` defect below was found.


- **The golden corpus is now one file per case, and both lanes run the same
  text.** A case used to be spread across three files — the SQL and the
  runtime-lane expression in `test_<area>.py`, the AOT-lane expression in
  `test_<area>.mojo`, and the expectation in `test_<area>.exp` — matched only
  by name, with nothing holding the two expressions to the same query. They
  drifted once already, when the AOT lane had no boolean column leaf and the
  Python twin silently tested something else. Each of the 69 cases is now a
  single `golden/cases/<name>.mojo`: SQL and expected table in the docstring,
  one expression below it. It is **Mojo source**, and Mojo is the source of
  truth — the constrained lane (no `**kwargs`, a dtype required at comptime)
  is the one whose spelling the other can always accept. The Mojo lane
  *imports* each case — the file is a standalone module exposing
  `def plan() raises -> DynRelation`, and the generated `test_cases.mojo` is
  one import plus a three-line wrapper per case, so a query exists in exactly
  one place. The case's identity is its file name; nothing inside repeats it,
  and `check` is called by the wrapper, which is where the name comes from.
  The Python lane runs the same body through a one-rule transpile (drop `var`)
  against `golden/helpers.py`. Measured: 69 separately-compiled case modules
  cost 161s cold against 160s for one concatenated module.
  `expfmt.py`, `fixtures.py` and `regenerate.py` are folded into
  `golden/runner.py`, whose `import duckdb` sits inside `regenerate()` so the
  duckdb-free `dev` environment still runs the corpus. Regeneration is now
  `pixi run -e bench python golden/runner.py`. Design in
  `docs/superpowers/specs/2026-08-20-golden-single-file-cases-design.md`.

- **`helpers.SHIMS` counts what the two lanes still spell differently, and a
  test keeps the count honest.** The fused lane's internal vocabulary —
  `AggExpr.of[NumericAgg[SumKernel, Int64Type]](x)`, `NumericCast[Float64Type]`,
  `Upper(x)` — is bridged by golden-local shims rather than by growing
  marrow's public Python API, which should never require writing
  `AggExpr.of[NumericAgg[...]]` where `.sum()` exists. 39 shims today;
  `test_golden_shims_are_declared` fails both when a vocabulary name is added
  without being declared and when a shim has quietly become real API.

- **Aggregates use the fluent API, which marrow already had.** The corpus
  spelled every aggregate `AggExpr.of[NumericAgg[SumKernel, Int64Type]](x)`,
  and so did `marrow/exprold/tests/test_aggregates.mojo` — but `.sum()`,
  `.mean()`, `.min()`, `.max()` and `.count()` have been on `NumericValue`,
  `StringValue` and `TemporalValue` all along, documented in
  `Relation.aggregate`'s own docstring. Every case now reads
  `col("v", int64).sum().alias("total")`, which deleted nine shims at a
  stroke. `Relation` gained a keys-less `aggregate(aggs)` overload so a
  no-`GROUP BY` aggregate needs no empty key list, matching
  `df.select(pl.col("v").sum())` in polars and `t.aggregate(total=t.v.sum())`
  in ibis. The low-level constructors stay where they belong: the kernel
  layer, `bench_aggregate_aot.mojo` and `benchmarks/binary_size/`, which exist
  to measure comptime versus runtime resolution and would stop measuring it if
  rewritten. CLAUDE.md records the rule for new tests.

- **A case file has one import line.** `golden/prelude.mojo` re-exports the
  case vocabulary and every case opens `from golden.prelude import *`,
  replacing three to six explicit imports per file. This is the repository's
  one sanctioned wildcard, and CLAUDE.md records why it is safe here: the
  prelude defines nothing and imports nothing for its own use, so its wildcard
  surface is exactly the curated list in it, and case files are leaves with no
  second entry path for a name to resolve along. Cases import from the prelude
  rather than `helpers.mojo`, whose own `DynArray` / `RecordBatch` imports
  would otherwise leak into every case.

- **The generated wrappers moved to a gitignored `golden/generated/`.** They
  are build output and no longer sit beside the sources. A tmpdir was tried
  and is not reachable — pytest applies a conftest's `pytest_collect_file`
  only to files under that conftest's own directory, so a `.mojo` outside the
  repository is never recognised as a test file. `-I` would satisfy the
  compiler, but collection happens first. Lifting this means moving Mojo
  collection out of `conftest.py` into a real pytest plugin, which is a
  separate change to shared infrastructure.

### Features

- **`expr2` gains the string family** — `StringValue`, `StringColumn[T]`,
  `StringLiteral[T]`, and the four comparisons, plus `col(name, string)` and
  `lit(value, string)`.

  This is the family that **cannot vectorise**, and it is worth stating why
  rather than treating it as an exception. `NumericValue.lane[W]` and
  `BoolValue.lane[W]` answer `W` elements because their storage is fixed-width;
  UTF-8 is not, since a string's position depends on every string before it. So
  `StringValue.lane` takes no `W` and answers one `String`. **Fusion survives
  it**: fusion removes *dispatch*, not width, so `name > 'b' AND a > 1` is
  still one pass with no intermediate column — and that composition has a test.

  Everything else is unchanged, which is the point: `bind` still resolves once
  per batch, `validity` is still structural and still reads the `Bound` rather
  than the batch, and the output still bit-packs, so a string predicate feeds
  `And`/`Or` and `Filter` without either knowing it came from strings.
  `StringColumn` is parameterised on `StringLikeType`, so `large_string` is the
  same leaf with a different offset width rather than a second node type.

  Costs **+600 bytes** on the streaming gate and nothing on a query that uses
  no strings — the family is dead-code-eliminated exactly as the lane design
  intends.

- **`expr2` gains `Limit` and `Sort`**, taking it to 6 of `expr/`'s 8
  relations.

  `Limit` forced a real gap in the push engine and closed it. A pull engine
  gets early termination for free — a consumer simply stops calling `pull`. In
  a push engine the *source* drives, so nothing downstream can halt a scan, and
  `LIMIT 10` over a billion rows would read a billion rows. `Operator` gains
  `done()`, defaulting False, and the driver stops pulling as soon as any stage
  answers True.

  `Sort` is a pipeline breaker that needs no new machinery: it buffers in
  `push` and orders once in `finish`, expressed with the same two methods a
  filter uses. Multiple keys are decomposed into stable passes applied
  **last key first**, each one *permuting* the previous order rather than
  replacing it — dropping that composition is the classic multi-key bug where
  the last key wins and the rest are silently discarded, and it has its own
  test.

- **`Fold[K, A, G: Grouping]`** replaces
  `NumericAggregateState[K, A, grouped: Bool]`. Three axes, all comptime: the
  algebra, the whole input subtree, and now the **placement**. A sorted or
  partitioned placement arrives as another conformer rather than another
  `Bool`.

  `G` is a *phantom* parameter — the fold reads `G.scatters` and never holds a
  `G`. The grouping instance belongs to the operator above, which assigns every
  row once and shares the result with every aggregate in the query; a fold that
  owned its own grouping would re-hash the keys once per aggregate. Measured
  **byte-for-byte size-neutral**, since the two instantiations already existed
  under the `Bool`.

- **`Grouping` — the placement axis, as a trait.** `kernels/groupby.mojo` gains
  `trait Grouping` with `ScalarGrouping` and `HashGrouping` conformers, so
  window partitions and a sorted or radix placement can arrive as *conformers*
  rather than as further branches inside `GroupBy`. It takes already-evaluated
  key columns rather than a `RecordBatch`, because `kernels` must not depend on
  the expression layer — and because that is what lets one grouping serve every
  aggregate in a query, hashing the keys once instead of once per aggregate.

  `ScalarGrouping` allocates nothing and returns **no ids at all**: a fold whose
  `scatters` is False never reads them, and materialising one `Int32` per row to
  communicate a constant is the cost it exists to avoid.

  **Placement is deliberately *not* a parameter of `AggregateOperator`**, and
  that is measured rather than assumed. Making it one instantiates the operator
  once per conformer for **+24,432 bytes (+2.439% on the gate)** and buys
  nothing: the operator's branch runs once per batch, while the 14.6x
  register-fold win lives in `NumericAggregateState`, which is already
  monomorphised on whether it scatters. Comptime placement belongs in the fold,
  which is where `Fold[K, A, G]` will put it.

- **`Operator` carries an associated `Out`.** `push`/`finish` answer
  `Optional[Self.Out]` rather than `Optional[RecordBatch]`, and `DynOperator`
  is parameterised on it.

  The push engine was specified on the premise that one trait covers relations
  and values alike. It does not: a relational stage produces a batch, a value's
  stage produces a **column**. Fixing `Out = RecordBatch` would make every
  value wrap its column in a one-column `RecordBatch` — a `Schema` allocated
  per value per batch — only for `ProjectOperator` to unwrap N of them and
  reassemble one. `DynOperator[RecordBatch]` and `DynOperator[Datum]` are two
  instantiations of one definition, so the erasure surface stays single.
  Measured **byte-for-byte size-neutral** on both `expr2` gates.

- **`expr2`'s engine pushes.** `Operator{push, finish}` replaces
  `Processor{schema, pull}`, and `Exhausted` is deleted — end of stream is
  `Source.next()` answering `None`, not an exception, so the
  `String(e) == "Exhausted"` comparison in `collect` is gone too.

  The point is not the direction of dataflow but that **blocking stops being a
  type distinction and becomes *when you return `Some`***. `Filter` and
  `Project` answer from `push`; an aggregate answers `None` through the whole
  stream and produces its result from `finish`. Under the pull design those
  were two shapes needing two traits and two erased boxes.

  Sources stay pull and drive, as in DuckDB: `Source.next()` generates and
  `DynProcessor.collect` pushes through the chain. `DynProcessor` is now the
  assembled *pipeline* — a source plus an operator list — which is what lets
  `Relation.to_processor` stay compositional without a `children()` walk, since
  `DynRelation` deliberately exposes none.

  The flush is a **cascade**, not a loop of independent `finish` calls: when
  stage *i* finally produces its result, that batch has never been seen by
  stages *i+1..*, so it is pushed through them before stage *i+1* is finished.
  `SELECT total * 2 FROM (… GROUP BY g)` returns nothing at all without this,
  and `test_the_flush_cascade_feeds_the_stages_above` is the guard.

  All 63 existing expr2 tests pass **unchanged** — they describe behaviour, not
  the engine. Binary size, measured on the gates added for exactly this:
  `query_expr2_streaming` +0.433%, `query_expr2_agg_fused` +0.589%. The second
  is over the 0.5% threshold and is **not yet paid back**: this step adds
  `DynSource` and `DynOperator` while `DynAggregateState` still exists, and the
  box collapse that motivates the engine comes with `Value.to_processor`.

- **`expr2` can aggregate.** `Aggregate` (logical) and `AggregateProcessor`
  (physical) complete the operator set the package needed to run a `GROUP BY`
  end to end, and `HAVING` falls out for free — a `Filter` above the aggregate
  reads its *output* batch, so no node is needed for it.

  It **buffers nothing**. `AggregateState.update` takes the whole
  `RecordBatch` rather than a computed column, so each state binds its own
  input subtree and folds lanes straight out of the morsel: `sum(a + b)` never
  materialises `a + b`. `expr/`'s processor keeps one evaluated column per
  aggregate per morsel and `concat`s them at emit time; this one keeps only
  the grouper's key builders, which grow with the number of *groups* rather
  than the number of rows. DataFusion, ClickHouse and Polars all hand an
  aggregate an already-computed array and cannot express this.

  An empty key list is not a separate node — it is one implicit group, and the
  only thing it changes is which fold `to_state(grouped)` starts. That is
  chosen at plan-build time because it is known there; running the grouped
  loop over a single group measured 14.6x worse.

### Refactors

- **`AggKernel.AccType` is bound on `PrimitiveType`, and `TemporalMinMax` is
  gone.** The bound used to be `NumericType` at both ends, which was the
  *intersection* of four unrelated requirements — `count` needs nothing of the
  type, `min`/`max` need an ordering, `sum` needs addition, `mean` needs
  division — so the most permissive kernel was constrained by the least, and
  `min`/`max` over a timestamp needed its own `Aggregation` that reinterpreted
  the column through its integer backing and relabelled the result.

  Temporal `min`/`max` is now the same `NumericAgg[MinKernel, T]` fold every
  numeric column takes, with no view, no relabel and no second code path; it is
  also mergeable now, which `TemporalMinMax` was not. What each kernel actually
  requires is stated by the domain markers that were already declared and
  inert (`OrderedAgg` / `ArithmeticAgg` / `IntegralAgg`) and is enforced by
  `_check_domain`, so `sum(date)` is a **build error** rather than a runtime
  raise. `StringMinMax` stays: it keeps a per-group *index* rather than a
  scalar accumulator, which is a state-shape difference, not a domain one.

  The enabler is `AggKernel.acc_dtype(input_dtype)`, the new one-line
  companion to `AccType`: `AccType` names the accumulator's *type*, `acc_dtype`
  names its *value*. `NumericType` is `Defaultable` and `TemporalType` /
  `DecimalType` are not, so an accumulator that keeps the input's type can only
  get its unit, timezone, precision and scale from the column. Every caller
  that builds an `AggState` now goes through it.

- **Fixed while widening: `min`/`max` of an all-null or empty column answered
  its identity sentinel, not NULL.** `MinMax.reduce` took the SIMD fast path,
  which folds nulls to `identity` (`MAX_FINITE` / `MIN_FINITE`) and returns a
  *valid* scalar — so `SELECT min(x)` over an all-null column answered
  9223372036854775807. The grouped path was always right, because its valid
  count says the group was never touched; the whole-array path had no count and
  now asks the column directly. It surfaced as `min(date)` regressing when
  temporal folding moved onto the numeric path, and it was wrong for numeric
  columns the whole time. `Reduction`'s own docstring already claimed the fixed
  behaviour.

- **Toolchain moved to Mojo 1.1.0.dev2026082305 / MAX 26.6.0.dev2026082305**
  (from `dev2026081705`). One migration: upstream deleted
  `max.algorithm.reduction._reduce_generator_wrapper` in `327e2cc25e` as
  "unused", having looked only inside `modular` — `views._reduce_dispatch`
  was calling it. The wrapper only refined dtypes around `_reduce_generator`,
  whose lambdas are parameterised on their own `dtype` while everything in
  `_reduce_dispatch` is fixed to `T`, so the three shims it provided are now
  written at the call site.

  That arm is behind `comptime if`, so a default CPU build does **not**
  type-check it: verified separately with
  `mojo build -D MARROW_GPU=true`.

- **`Value.params()` is gone**, with `DynParam` and the `merged` overload that
  keyed on it. Sixteen implementations, fifteen of them pure plumbing, plus two
  thin-fn slots on `DynValue` and `DynRelation` — and every caller outside a
  node's own recursion was a test. It was built for a `--help` surface that
  does not exist. Reintroducing it costs the same sixteen methods, no more.

- **`expr2`'s `to_processor` slots are named `to_operator`,** matching the
  method they dispatch to. `DynProcessor` was renamed to `DynOperator` earlier;
  the trampoline slots, the trampolines themselves, and eight prose references
  kept the old word. The stale `# -- Analyzable` / `# -- Executable` section
  headers now name the traits that exist (`Value`, `Evaluable`).

- **`kernels.core.Grouping` is now `Groups`.** It holds the *result* of
  grouping — dense ids plus a group count — and the name was needed for the
  placement *strategy* the aggregation architecture specifies as
  `trait Grouping` with `ScalarGrouping` / `HashGrouping` conformers. Two
  `Grouping` names in one package is precisely the ambiguity the wildcard-import
  ban exists to prevent, and this tree has three recorded incidents of it.

  The resulting pair reads the way it should: `HashGrouping.group(batch)`
  answers `Groups`. 47 references across 10 files, including shipped `expr/`
  and two golden cases; verified by 75 kernel cases and 435 `expr/` cases.

- **The value-level aggregate trait is `AggValue`, not `Aggregate`.** Its own
  docstring already said it was named "rather than `Aggregate`, which the
  relational node wants", and `DynAggregate`'s docstring described itself as
  "an `AggValue`" — both were left stale by an earlier rename, and the name
  the relational node was promised was occupied. Renamed to `AggValue` /
  `DynAggValue`, which is what the prose says and what frees `Aggregate` for
  the plan node added above. Contained entirely within `marrow/expr/`, which
  nothing outside the package imports.

### Fixes

- **A join reached the type-erased `filter`, and it cost 450 KB.**
  `SwissHashTable.probe` verified candidate pairs with
  `filter(build_indices.to_dyn(), verifier)`. The indices are `Int32Array`, but
  `filter` exists only as a `DynArray` free function, so `.to_dyn()`
  instantiated the per-dtype ladder and made it reachable from **every binary
  that joins** — `marrow::kernels::filter` 98 → 121 symbols, dragging
  `marrow::views` 112 → 148, `marrow::arrays` 289 → 315 and
  `marrow::execution` 237 → 258.

  The null-key semantics that call introduced are correct and unchanged: a NULL
  key still matches nothing, not even another NULL. The rule is simply *data
  AND valid*, so it is now applied by ANDing validity into the mask and
  selecting through the typed `Filter.apply`, which instantiates for
  `Int32Type` alone.

  **`query_join` 1,967,052 → 1,527,820, recovering 439,232 bytes**; +30.455%
  over its baseline becomes +1.325%. The residue is the bool-key arm added by
  the same commit, which is new functionality rather than drift.

  Found by the `expr2` gates added in this cycle — they failed the *existing*
  gates on their first run, and the culprit was bisected over 183 commits to
  `6c570eb`. It is the third instance of one pattern, after hashing reaching
  `cast` (~2.4 MB) and the closure adapter in `variant_dispatch` (+662,740
  bytes): **one call site making a per-dtype family reachable from everything.**

- **`expr2`'s fused aggregate could not be instantiated at all.** Every one of
  the 13 cases in `expr2/comptime/tests/test_aggregates.mojo` failed to build,
  and the handoff plan recorded the cause as bisected "to the *body*, not the
  plumbing", to be fixed during a later rewrite. It was neither: the grouped,
  no-validity arm splatted a mask with `SIMD[DType.bool, W](True)`, and that
  positional constructor is `comptime assert Self.size == 1` — *"must be a
  scalar; use the `fill` keyword instead for explicit splatting"*. One line,
  `fill=True`, and all 13 pass.

  The trap worth recording is the near-miss fix: `fill=` is declared **only**
  for `SIMD[DType.bool, size]`. Numeric splats go through the positional
  `Scalar[Self.dtype]` constructor and were already correct, so applying
  `fill=` uniformly to the accumulator and count vectors traded one error for
  two. Only the bool mask was ever broken.

- **A temporal or list column was not recognised as a column by name.**
  `NumericColumn`, `BoolColumn` and `StringColumn` each override
  `Value.bound_column`; `TemporalColumn` and `ListColumn` did not, so they
  inherited the default that answers -1 — "not a bare column". `GROUP BY d`
  therefore produced a key named `key0` in the fused lane while the runtime
  lane answered `d`: one query, two spellings, two different output schemas.
  The golden corpus caught it on the temporal family's first outing;
  `ListColumn` had the same bug with nothing exercising it.


- **`BitmapView.load[W]` read up to 3 bytes past the allocation at a bitmap's
  tail.** The load takes an unconditional 4-byte `UInt32`, and a buffer's
  padding is `(-extent) mod 64` — so an extent of 62, 63 or 0 mod 64 leaves
  too little, and an extent that is already a multiple of 64 leaves none at
  all. A nullable 150,000-element column (18,750 bytes, 62 mod 64) tripped it
  through the masked `apply` lane; 512 bits is the smallest case. This is the
  one-byte-overrun class that once corrupted tcmalloc's freelist. The window
  now slides back to finish on the view's last live byte, with the distance it
  slid added to the shift — correct rather than merely in-bounds, because the
  caller only consumes lanes inside the view, so the clamped word always holds
  the bits asked for. Two ALU ops, no branch: measured against
  `bench_count_set_bits` as a control, `bench_load_*` and the filter path are
  unchanged. `marrow/tests/bench_bitmap.mojo` gains `bench_load_*` so the
  primitive has a benchmark of its own.

  **`load_bits[T]` has the same defect and is not fixed here.** It reads
  `size_of[T]()` bytes unconditionally, so eight residues fall short for a
  `UInt64`, and it is reachable (an offset-4 view of a 512-bit bitmap wants
  bytes 57..64 of 64). The sliding trick cannot work there — a `Scalar[T]`
  cannot hold NBITS live bits after a shift — and a tail branch was measured
  at **+5% on `bench_filter50pct_nulls_1m`**, since `filter.mojo` calls it in
  six places on the hot path. Fixing it wants a different approach: guarantee
  the padding at allocation instead, which costs 64 bytes per bitmap and
  nothing at runtime.

- **A NULL join key matched another NULL.** SQL's rule — and Arrow C++'s
  default `JoinKeyCmp::EQ`, whose probe loop routes any row with a null key
  column straight to no-match — is that a NULL key matches nothing, not even
  another NULL. marrow paired them: `SwissHashTable.probe` verified candidate
  pairs with `Filter.apply(indices, mask.values())`, and `values()` is the
  comparison's *data* bits alone. A comparison kernel evaluates every SIMD lane
  whatever the validity says, so two null keys sharing a payload — two zeroed
  `int64` slots, which is what a null slot usually holds — left the data bit
  set, and only the validity bitmap recorded that the bit was meaningless. The
  verification now goes through `filter`, which already owns the "a null entry
  does not select" rule. Dropping the pair there is enough for every join kind,
  since every candidate passes through it: INNER and SEMI lose the null-keyed
  row, ANTI keeps it (it matched nothing), and LEFT / RIGHT / FULL keep it
  null-widened. Multi-column keys follow from Kleene AND: a NULL in *any* key
  column makes the row unmatchable.

- **A `bool` join key raised `dispatch_primitive: dtype is not primitive`.**
  Booleans are bit-packed, so a bool column is a `BoolArray` and not a
  `PrimitiveArray`, and `equal`'s numeric arm could not reach it — the same
  shape as the `binary`-key failure fixed above it. `equal` gains a bool arm
  spelled with the existing boolean kernels: `Not(Xor(a, b))` is XNOR over the
  packed data bits, with Arrow's validity (valid only where both operands are),
  and it works a word at a time rather than a lane at a time. `nullif` over
  bool arrays, which shares that primitive, works for the same reason.

- **`LazyTable.aggregate` rejected `SELECT DISTINCT`.** Keys with no
  aggregates raised `ValueError: aggregate: needs at least one aggregate`,
  although the plan layer has always executed the keys-only form — the Mojo
  lane accepted it and Python did not, so one query shape was reachable from
  only one frontend. The guard now fires only when there is neither a key nor
  an aggregate. Found by a strict `-- xfail` xpassing in the Mojo lane.

- **`Column.count_distinct` and `Column.approx_count_distinct`.** Both
  aggregates resolved through the binding already and had no Python method, so
  `count(DISTINCT x)` was Mojo-only.


- **Expected-table cells are quoted, because `mojo format` strips trailing
  whitespace inside docstrings.** The `words` fixture holds `"  pad  "`, so an
  unquoted expectation block silently lost its trailing spaces and asserted a
  different string — caught by running the formatter over the new corpus and
  watching two cases go red. Quoting keeps every line ending in a printable
  character, and lets string data contain a tab or a newline, which the bare
  format could not represent at all. `golden/cases` and `golden/helpers.mojo`
  joined the `fmt` tasks now that formatting them is safe.

### Docs

- **`docs/guide/compile.qmd` — a guide to `param()`, `execute_cli()` and
  `marrow compile`, with a genuinely runnable example.** Closes backlog item
  M1.6, which flagged that `docs/guide/expressions.qmd`'s AOT blocks were
  illustrative only (plain ` ```python `, never executed) and could name
  types that did not exist without the docs build ever going red. The new
  page's example is `benchmarks/binary_size/query_param.mojo` — the same
  file the binary-size gate compiles — pulled in verbatim with Quarto's
  `include` shortcode rather than pasted, so the page cannot drift out of
  sync with the source the way a copy could; the `mojo build` itself is not
  run by the docs build (it takes 1-2 minutes and the `docs` pixi
  environment carries no Mojo toolchain), so the page states plainly that
  its transcript — `--help`, `--describe`, a filtered run, `-o
  result.parquet` — was captured by hand rather than executed by Quarto.
  Cross-linked from `expressions.qmd`. Filed a follow-up, S19 in
  `docs/backlog.md`, for the static-linking finding from the same work.
  **That finding has since been re-measured and the saving is not available**
  — see the Fixes section below.

### Fixes

- **The AOT lane had no boolean column leaf.** `col` had overloads for
  numeric, string, list and temporal dtypes and none for `BoolType`, and
  `values.mojo` had no `BoolColumn` node — so a fused expression could not
  reference a `bool` column at all, while the runtime lane could. That is an
  invariant-2 violation (a feature present in only one lane), and it forced
  three-valued-logic tests to synthesise their operands from comparisons.
  Adds `BoolColumn` and the matching `col(name, bool_)` overload; booleans
  are bit-packed, so its `State` is the `BoolArray` and the lane loads
  through `values()`, the offset-applied `BitmapView`. **Zero bytes** on the
  size gate — `query_streaming_agg_fused` measures 1,397,432 of `__text`
  with and without it, since an unused fused node is eliminated.

- **An empty result carried a schema but no columns.** `collect()` returned
  `RecordBatch(schema=..., columns=[])` when no morsel survived, so
  `num_columns()` was 0 while the schema named its fields — anything walking
  columns by schema index ran off the end, and the C Data export returned
  NULL without setting an exception, surfacing as a bare `SystemError` from
  `to_pyarrow()`. Now builds one zero-length column per field.

- **Array equality compares fields, not elements — which is what let
  `marrow/tests/test_arrays.mojo` compile for the first time.** All 167 cases
  now pass in 49 s; the file had never built. The cause was a compiler
  *deadlock*, not a slow compile or an infinite loop: `ListLikeArray.__eq__`
  compared elements via `unsafe_get(i)`, which returns a `DynArray`, so
  `DynArray.__eq__` (`self._v == other._v`, resolving the active variant member
  on *both* sides) and the nested arrays' `__eq__` were mutually recursive at
  instantiation time. The elaborator burned ~9 s of CPU and then parked in
  `semaphore_wait_trap` at 0% CPU indefinitely with no diagnostic. Because the
  cycle is in the *type* graph, it needed no nested data — a `list<int64>`
  whose child is a plain leaf deadlocked identically, and it took down all 165
  cases in the file, not just the nested ones.

  `ArrayData` is now `Equatable` and owns the comparison — dtype, length, null
  count, validity views, children, value buffers — recursing through
  `List[ArrayData]`, a concrete self-recursive type with no generic
  instantiation behind it. `DynArray.__eq__` no longer touches the variant
  ladder; it compares fields through one `to_data()` per array, never per
  element. `ListLikeArray` and `FixedSizeListArray` compare their child array
  once as a whole. Value buffers are compared by each array's own type through
  the *dtype* ladder, which has no path back to `DynArray.__eq__` — chosen over
  comparing `Buffer`s directly because `Buffer.__eq__` compares whole
  allocations, and an array's buffer is neither exactly sized (filtered output
  is over-allocated) nor read from the start (a slice has an offset).

  Element-wise value comparison belongs to `EqKernel` and stays there. Two
  narrower fixes were measured and neither works, so neither should be retried:
  narrowing `DynArray.__eq__` to a single `isa` (O(N) arms instead of
  `Variant.__eq__`'s O(N²)) still deadlocks, and so does `@no_inline` on it.
  Nested dictionaries reached through the erased path are now compared
  structurally (indices plus dictionary) rather than decoded, because
  `DictionaryArray.__eq__` compares `DynScalar`s whose `ListScalar` arm holds a
  `DynArray` and re-enters the cycle; a direct `DictionaryArray` comparison is
  unchanged and still treats permuted dictionaries as equal. Two cases were
  added for nulls inside a list's *child*, which nothing covered.

- **`project` no longer drops `nullable` and metadata on a pass-through
  column (A-1).** It rebuilt a bare `Field` from the probed dtype for every
  output column, so `select("x")` and `project(["x"], [col("x")])` produced
  *different schemas for the same column* — reproduced from Python, with a
  non-nullable Parquet field coming back nullable. A new `_projected_field`
  helper copies the source `Field` whole when the expression is a bare column
  reference (`bound_column >= 0`) and probes only genuinely computed columns;
  `with_columns` routes its two probe sites through it as well, so all four
  projecting verbs now agree. `test_select_preserves_the_source_field`
  previously asserted the lossy behaviour on purpose and now pins the fix,
  alongside a new `test_project_still_probes_a_computed_column` confirming a
  computed column is still probed and still nullable.

### Refactors

- **`DynAgg` is deleted; the runtime lane returns `AggExpr` directly (A-2).**
  `DynAgg` duplicated `AggExpr` field for field, and `AggExpr.__init__(DynAgg)`
  was a copy constructor that *also* applied the "empty alias ⇒ use the
  function name" rule — a rule `DynAgg` applied in neither its own `__init__`
  nor its own `alias`, so the binding layer's `_agg_name` wrote it a **third**
  time. `DynValue.sum()`/`mean()`/`aggregate(name)`/… now return `AggExpr`,
  which already carried both shapes (a name to resolve, or a comptime
  `Aggregation`); the default is applied once, where the name is first known,
  and `_agg_name` is a plain field read. `AggExpr` gains `function()` and
  `dyn_input()` for the binding layer, and its `write_to` now appends
  `" as <name>"` only when the output name says something the function name
  did not — so every rendering is byte-identical to before. Python-visible
  behaviour is unchanged: `render()`, `repr()`, `name()`, `function()` and
  `input()` all verified against the existing assertions.

- **The second `DynRelation.aggregate` overload is gone (A-3).** The
  `(keys, inputs, aggs, names)` spelling was a third convergence point for a
  split already resolved by `AggExpr`, contradicted overload 1's own
  docstring, and had exactly two callers — both tests, both rewritten onto
  `aggregate(keys, aggs)` with `AggExpr.of[A](...)`. Nothing in the library,
  the benchmarks or the bindings used it.

- **`marrow.kernels`' re-export boundary is stated, and the compute surface
  completed (S13).** The package docstring promised direct use of every
  submodule while seven of nineteen were re-exported, so `mk.cast` worked and
  `mk.concat` did not with nothing saying why. The element-wise kernels,
  folds and free-function verbs are now all re-exported — `boolean`, `string`,
  `temporal`, `conditional`, `nested`, `concat`, the unary/transcendental
  half of `numeric`, `CountKernel`, `Kernel`, `Grouping`. Two things stay
  behind their submodule and the docstring now says why: the hash/partition
  machinery and the pruning `interval` algebra, which are not kernels a caller
  applies to an array; and `MinKernel`/`MaxKernel`, because the name means two
  different things — `numeric.MinKernel` is an element-wise binary minimum,
  `aggregate.MinKernel` is `MinMax[MinOp]`, a whole-array fold — so a flat
  namespace cannot hold both and neither is re-exported.

- **`Partition.original_row` deleted (S18).** Zero callers repo-wide;
  `Partition` is not re-exported from `kernels/__init__`, so the surface was
  internal.

- **Four dead imports removed from `expr/relations.mojo` (A-12).** `Interval`,
  `PruneStats`, `DynArray` and `DynValue` each had exactly one occurrence in
  the file — the import itself — and were overstating the module's dependency
  edges in the backlog's §8 graph.

### Features

- **`safe=` on the expression-level cast (A-7).** `marrow.kernels.cast` and
  the array-level `compute.cast` both took it; the expression-level one did
  not, so a caller who knew their data could not skip validation.
  `DynValue.cast(to, safe=True)` selects between two evaluators rather than
  widening `DynPayload` with a `Bool` arm — the payload is size-critical, and
  the node name stays `"cast"` either way, so no parity entry changes.
  Surfaced as `Column.cast(target_type, *, safe=True)` in Python.

- **The wheel ships marrow's own Mojo source, so `marrow compile` works for a
  pip-installed user.** `python/build.py`'s hatchling hook now walks
  `marrow/**/*.mojo` and force-includes each file at `marrow/_mojo/marrow/...`
  in the wheel — the layout `resolve_marrow_path`'s third resolution step
  (the bundled-copy fallback) already looked for. Files under any `tests/`
  directory and `bench_*.mojo`/`profile_*.mojo` files are excluded — 1.68 MB
  of source (58 files) versus 5.6 MB for a precompiled `.mojoc`, and, unlike a
  `.mojoc`, tolerant of a compiler version that has drifted from marrow's
  exact pin (a `.mojoc` hard-errors on any skew). One wheel, not two: an
  extra adds a dependency, not files, so gating this behind `compile` would
  need a second `marrow-mojo` distribution — not justified by 1.68 MB on an
  already multi-MB wheel.

- **`marrow compile` — a CLI that compiles a `.mojo` query file (one ending
  in `plan.execute_cli()`) into a standalone binary.** `python/marrow/compile.py`,
  registered as the `marrow` console script (`python/pyproject.toml`'s
  `[project.scripts]`, plus a `compile` extra pinning `mojo>=1.1,<2`).
  Reuses `benchmarks/binary_size/compare.py:build_and_strip`'s recipe
  (`mojo build -O3 -g0 -I <marrow> <src> -o <out>`, then `strip`).
  `resolve_marrow_path` finds the `marrow/` package via `--marrow-path` ->
  `$MARROW_MOJO_PATH` -> the bundled `marrow/_mojo/` -> repo-root
  autodetection, raising `FileNotFoundError` listing all four when none
  resolve. `check_mojo_version` fails fast with an actionable message when
  `mojo` is missing or out of range, naming marrow's pinned nightly and that
  PyPI's stable wheel cannot reach it (a wheel cannot force
  `--extra-index-url`), instead of letting an opaque compiler error surface.
  **Passes `-D MARROW_CLI_WRITERS=true` by default** — the Parquet/IPC output
  writers are gated behind that define since linking them costs 572,288 bytes
  of `__text`, but the CLI's documented `-o result.parquet` /
  `-o result.arrow` contract has to work out of the box; `--no-writers` opts
  back out for the ~572 KB smaller binary, with `-o` disabled in that build.

- **`marrow compile --bundle DIR` — turn a compiled query binary into a
  self-contained, relocatable directory.** `dylib_closure(binary)` walks
  `otool -L` (macOS) / `ldd` (Linux) **transitively** — not a hardcoded
  list — since a marrow query binary links `libKGENCompilerRTShared.dylib`
  and `libAsyncRTMojoBindings.dylib` directly but depends on
  `libAsyncRTRuntimeGlobals.dylib` and `libMSupportGlobals.dylib`
  transitively (a `-D MARROW_GPU=true` build would likely add
  `libMGPRT.dylib` as a fifth); `/usr/lib`, `/System` and `/lib*` are
  excluded as always-present on the target machine. `bundle(binary, dest)`
  copies the binary and its closure into `dest` and rewrites the rpath to
  `@loader_path` (macOS, `install_name_tool`) / `$ORIGIN` (Linux,
  `patchelf`), so the directory runs without the build machine's pixi
  environment on `LC_RPATH` — static linking is not possible on macOS
  (`ld: library 'System' not found`; Apple ships no `libSystem.a`), so a
  relocatable directory is the achievable self-contained artifact. The
  Linux path is implemented but untested (no Linux dev machine available).

- **`DynRelation.execute_cli()` — argv binding, `--help`, `--describe`, and
  the output contract for a compiled query binary.** The last call in a
  `main()` that built a plan with `param(...)` cells: drains the parameter
  registry, short-circuits to a rendered `--help`/`--describe` **before**
  binding or executing (so a plan built with unbound cells never needs a
  dummy value to satisfy them), then `params.parse_params()` binds every
  cell from `--name value` pairs or applies its declared default (a named
  error for a missing required parameter or an unrecognized flag), then
  `execute(ctx)`, then writes the result: no `-o` pretty-prints to stdout
  (unchanged from what `benchmarks/binary_size`'s gate programs already do),
  `-o r.parquet` / `-o r.arrow` pick Parquet/IPC by extension, and
  `--format parquet|ipc|table` overrides that. `parse_params` takes an
  explicit `List[String]` rather than reading `argv` directly, so it is
  tested without spawning a process; `execute_cli` is the only caller that
  supplies the real `argv`. The two format writers
  (`_write_parquet_output`/`_write_ipc_output` in `relations.mojo`) are each
  one line calling into `marrow.parquet.write_table` /
  `marrow.ipc.write_ipc_file`, kept separate from the dispatch that picks
  between them — and that separation is what let both be **gated behind
  `-D MARROW_CLI_WRITERS`, which is off by default**, once the binary-size
  gate measured them at 572,288 bytes of `__text`. So a plain
  `mojo build` of a query binary raises on `-o r.parquet` / `-o r.arrow`
  rather than writing; `--format table` and no-`-o` stdout output always
  work. `marrow compile` passes the define, so the documented CLI contract
  holds for a binary built through it. `execute_cli` runs **once per
  process** (`claim_cli_invocation()`), and its argv splitting is
  `split_cli_args()` — a `List[String]` in, a `CliArgs` out, testable without
  spawning a process, the same factoring `parse_params` already had.

- **`ParquetScan.path` accepts a late-bound `PathSpec`, not just a literal
  `String`.** `ParquetScan(path=param("src", string), ...)` now works: `PathSpec`
  gained a `StringParam[T]` constructor overload (`@implicit`, so the plan
  builder needs no explicit wrapping) that shares the parameter's
  `ArcPointer[ParamCell]` rather than copying its value, so binding the
  parameter after the plan is built — the whole point of a late-bound path —
  is visible through the scan too. `StringParam` gained a public `cell()`
  accessor so `PathSpec`, defined in a different module, can reach it without
  touching its private field. Every internal read of the path now goes through
  `PathSpec.resolve()` (raising, used from `to_processor()`, which runs after
  parameters are bound) or the new non-raising `PathSpec.describe()` (used from
  `write_to()`, which renders `param(name)` for an unresolved path exactly as
  `StringParam.render()` does). Existing `ParquetScan(path=String(...))` call
  sites are untouched — `PathSpec`'s `@implicit` `String` constructor already
  covered them. Adding the `StringParam` overload to `PathSpec` makes
  `params.mojo` and `values.mojo` import each other, an expected cycle within
  the package (see `CLAUDE.md`).

- **`DynValue.param()` / `param(name, DynType)` — the runtime lane's param
  leaf, at parity with the fused `NumericParam`/`StringParam`/`TemporalParam`.**
  Invariant 2 (no feature in only one lane) required a runtime-lane
  counterpart, but `DynPayload` is size-critical and gained no new variant for
  it: the payload carries only the parameter's name, the same shape
  `DynValue.column` already uses, and `_param` resolves the cell by name at
  evaluate time via `params.lookup_param`. That needed a name-keyed registry
  lookup, which did not exist — `_REGISTRY` only holds declarations
  in-flight between a plan's construction and its drain, and is empty
  afterwards, exactly when `_param` runs. `lookup_param` reads a second
  module-level table, `_LOOKUP`, that `register_param` inserts into and that
  `drain_params()` **repopulates** — not clears — from exactly the
  declarations it just drained. That reset is asymmetric on purpose:
  `_REGISTRY` empties so the next plan starts clean, `_LOOKUP` repopulates so
  the runtime lane can still resolve a parameter by name after the drain,
  while a later, unrelated plan's drain replaces the scope rather than leaking
  an earlier plan's cells into it. An *empty* drain skips the repopulation
  entirely, so re-entering the drain does not strand names already in the
  table.

- **`marrow.exprold.params` — late-bound query parameter cells and a registry.**
  `ParamCell` is a shared, mutable box for a scalar that starts unbound and is
  filled in after a plan is built; `ParamDecl` is the declaration (name,
  dtype, optional help/default) plus the `ArcPointer[ParamCell]` expression
  nodes will close over. `register_param`/`drain_params` are a module-level
  registry rather than a `parameters()` trait method — the latter would need
  40 implementations and a second recursive plan traversal, against a size
  gate where one shared dispatch adapter already cost +662,740 bytes.
  `PathSpec` is the first consumer: a `ParquetScan` path that is either a
  literal string or a cell resolved at execution time. No expression nodes
  yet — those land in later tasks.

- **`NumericParam[T]` and `param()` — the first fused expression node for a
  late-bound query parameter.** Structurally `NumericLiteral[T]` with the
  scalar behind an `ArcPointer[ParamCell]`: the cell is resolved in `state()`,
  once per batch, and `lane()` splats a plain `Scalar`, byte-identical to a
  literal's — so a bound parameter costs nothing per row. `prune()` reads the
  cell too, so a bound parameter prunes row groups exactly as a literal does;
  reading an unbound cell raises, which is correct since pruning cannot run
  before binding. `param("min-a", int64)` builds the node and registers a
  `ParamDecl` in the same call, alongside `col`/`lit` in
  `marrow/exprold/builders.mojo`.

- **`StringParam[T]`/`TemporalParam[T]` and their `param()` overloads — the
  string and temporal counterparts of `NumericParam`.** `StringParam` mirrors
  `NumericValue`'s fused-lane shape adapted to the string family: `State =
  String`, resolved once per batch in `state()`; `lane()` returns a plain
  `String` copy, no `[W]` — variable-width UTF-8 has no SIMD lane, same as
  every other `StringValue` leaf. `param("src", string)` is what a Parquet
  scan path will bind its file path against. `TemporalParam` is
  materialize-only instead, matching `TemporalColumn`: the temporal family has
  no fused lane at all (`TemporalValue` declares no `state`/`lane`), so
  `materialize()` reads the cell once per pass rather than splatting through a
  lane. Both add a `PrimitiveScalar`/`StringScalar`-typed `default` to
  `param()`, and `StringScalar` gained `.value() -> String`, matching
  `PrimitiveScalar.value()`, so a bound string default can be read back
  without going through `.to_string()`.

- **`LazyTable.collect(num_threads=0)` — the lazy query path can ask for
  parallelism.** `ExecContext.__init__` defaults to `num_threads=1`, so
  `DynRelation.execute()`'s `ExecContext()` default was forced serial, and the
  Python binding called `.execute()` with no context at all: every lazy query
  ran single-threaded on every machine, and nothing in the Python surface could
  say otherwise. `collect` (and `to_pyarrow`) now take `num_threads`, spelled
  and defaulted exactly as on the eager surface — `0` auto, `1` serial, `N`
  forced. `collect` rather than a constructor argument because it is the only
  place a plan runs, and a plan is immutable and shared.

  **The default changed**: `DynRelation.execute` / `to_processor` now default to
  `ExecContext.auto()`. Pass `ExecContext.serial()` for the old behaviour.

  Measured (`python/marrow/tests/bench_lazy_parallel.py`, 1M rows, medians, 1 →
  8 threads): `sort` **2.16x**, `join` at 10M rows 1.14x, `join` at 1M rows
  **0.77x — a regression**, `group_by` and `filter` unchanged. ClickBench moves
  0.2%, i.e. not at all. The reasons are structural and are written up in
  `docs/alpha-findings/o4-parallel-exec.md`: the Parquet reader takes no
  `ExecContext` at all, `Value.execute(batch)` has nowhere to put one so every
  projection and predicate is serial, and `AggregateProcessor` groups through
  `HashGrouper`/`AggFunc.grouped` — neither of which takes a context — so the
  `GroupBy` kernel's thread-local and radix strategies are not on the plan path.
  In a morsel-at-a-time engine only the pipeline breakers have enough work in
  hand to parallelise, which is why `sort` is the one that moved.
- **Projection pushdown — a lazy query stops reading every column of a Parquet
  file.** `ParquetScan`'s schema *is* its projection, but nothing ever narrowed
  it, so `read_parquet(hits).aggregate(n=count_star())` decoded all 105 columns
  of a 1M-row file to count rows. `DynRelation.optimize()` now walks the plan
  from the root, carrying down the set of columns each node's parent actually
  reads — widened at a `Filter` by its predicate, at a `Sort` by its keys, at an
  `Aggregate` by its keys *and* aggregate inputs, and narrowed at a `Project` to
  the outputs that survive — and rewrites the scan's schema to what is left.
  `execute()` calls it, so every plan built through the verbs and every query
  through the Python `LazyTable` gets it.

  Measured on ClickBench `hits_0.parquet` (1M x 105), 5 interleaved repeats,
  medians: the 41-query total went from **13 913 ms to 3 870 ms** and from
  **17.7x polars to 5.0x**, with polars and duckdb steady to within 1.5% across
  the two runs. `COUNT(*)` went 271 ms -> 9.9 ms; `GROUP BY URL` 317 ms ->
  104 ms, which is 1.5x polars. `SELECT *` is unchanged, as it must be.

  The rewrite never changes a plan's output schema or its rows: the root seeds
  the column set with its own columns, so a node can only ever narrow to a
  subset its parent asked for, and a plan that emits everything still reads
  everything. A scan never narrows to *nothing* — `COUNT(*)` references no
  column and a zero-column read yields zero-row batches, which the scan's
  streaming loop reads as end-of-file — so the empty case keeps the narrowest
  fixed-width column. `Join` is deliberately excluded: its key indices are
  positions into its children's schemas, so pushing through one means
  recomputing them, which is a separate rewrite.
- **`Relation.children()` / `DynRelation.children()` — the plan IR is walkable.**
  A plan could not be traversed at all, which is why the layer had two
  incompatible ad-hoc rewrite mechanisms and no `EXPLAIN`. `children()` is the
  read-only half (a *method* may mention the erased `DynRelation`; only a
  *field* whose function type does makes the struct recursive), and
  `with_projection` is the rewriting half, following the erased-pointer protocol
  `with_predicate` established rather than inventing a third. It is
  `children()`, not `inputs()`, because `Aggregate.inputs` already names the
  aggregate value expressions.
- **`BufferView` and `BitmapView` bounds are enforced by `debug_assert` instead
  of asserted in prose.** Both views already carried a `_check_bounds` helper;
  the unsafe and SIMD paths never called it, which is why `-D ASSERT=all`
  stayed silent through the heap overflow in F1. `unsafe_get`/`unsafe_set`,
  `load[W]`/`store[W]`, `gather[W]`, all three `compressed_store` overloads and
  the bitmap byte accessors now check, and `BufferView.filter` /
  `BitmapView.filter` assert that what they wrote equals what they allocated —
  the postcondition that found the `load_bits` bug above. Reads and writes get
  different bounds on purpose: a write must stay inside the view, while
  `load[W]` and `load_bits` take a deliberately wide load and are bounded by the
  allocation (`align_up(extent, 64)`). `debug_assert` compiles out in release,
  and the full surface measured within noise on a fixed 141-case selection
  (135.5 s -> 134.1 s, compilation-dominated). `Buffer.unsafe_set` and
  `Buffer.unsafe_get` were unchecked one layer down and now call
  `Buffer._check_bounds[T]` too. That bound is the *allocated* element count,
  which `_aligned_size` rounds up to a 64-byte multiple, so `filter`/`take`
  destinations sized from a computed count now pass their logical length to
  `Buffer.view[T]()` instead of taking the padded default, with
  `pos == total` postconditions behind them. See
  `docs/alpha-findings/g1-buffer-invariants.md`.

- **The Python expression and plan surface closes the gap the binding agents
  deferred: null handling, `COUNT(*)`, and `with_columns`/`drop`/`rename`.**
  Three `# TODO(alpha)` markers pointed at methods that did not exist when the
  bindings were written and do now.

  - `Column.is_null()` / `.is_valid()` / `.is_nan()` / `.is_inf()` /
    `.fill_null(other)` bind the runtime lane's own null predicates (not
    `Value`'s fused defaults), so the result is another `Expr` and combines with
    `&` / `|` / `~` like any other predicate. Cross-checked against
    `pyarrow.compute`, null propagation included: `is_null` answers for every
    row, `is_nan`/`is_inf` return null on a null input.
  - `marrow.count_star()` exposes `expr.builders.count_star` as an `Aggregate`
    with no input column — `t.aggregate(by=["k"], n=marrow.count_star())`. It is
    a free function, not a `Column` method, because `COUNT(*)` counts rows while
    `col("x").count()` counts non-null values; the two disagree on every
    nullable column and ~30 of ClickBench's 43 queries want the former.
  - `LazyTable.with_columns(**named)` (aliased `mutate`) is new to the Python
    surface, and `LazyTable.drop`/`rename` now call the bound plan nodes instead
    of rebuilding a full projection list in Python.
- **`DynRelation.select(names: List[String])`** — a list overload beside the
  variadic `select(*names: String)`, which a runtime frontend cannot splat into.
  The Python `Plan.select` routed through `project` for want of it, and that is
  a fidelity bug rather than a detour: `project` probes each expression's dtype
  and builds a fresh `Field`, so selecting a non-nullable Parquet column widened
  it to nullable and dropped its metadata. `select` copies the source field
  whole.
- **`DynRelation.with_columns` / `.drop` / `.rename` — the plan builder is
  usable on a wide table now.** `project(names, values)` replaces the whole
  output schema, so adding one derived column to a 105-column table meant
  re-listing 105 columns; there was no way to remove or rename one at all.

  - `with_columns(names, values)` appends the named expressions to the input
    schema. A name already in the schema **replaces that column at its original
    position** — polars `with_columns` and ibis `mutate` semantics, both checked
    rather than assumed (polars empirically; ibis builds
    `ops.Project(self, {**node.fields, **values})`, a position-preserving dict
    merge). As in both, every expression in one call is evaluated against the
    *input* batch, so a replacement is invisible to its siblings; chain two
    calls for sequential semantics.
  - `drop(names)` removes columns, raising on an unknown name.
  - `rename(names, new_names)` renames by parallel lists — the shape the rest of
    the file already uses — raising on an unknown name, a column renamed twice,
    or a duplicate output name.

  All three lower to the existing `Project` node; no new `Relation` node types.
  Computed columns get their dtype probed against a 0-row batch exactly as
  `project` does, while pass-through columns carry their input `Field` across
  whole, so `nullable` and field metadata survive a `with_columns`, `drop` or
  `rename` (`project` still drops both — recorded in
  `docs/alpha-findings/a2-relations.md`).
- **The relational plan layer is bound to Python, with a lazy `LazyTable` on
  top.** `DynRelation` is exposed as the Python type `Plan`
  (`python/bindings/plan.mojo`) with `select`/`project`/`filter`/`aggregate`/
  `sort`/`limit`/`join`/`execute`/`schema`/`column_names` and a rendering
  `__str__`/`__repr__`, plus the leaf constructors `in_memory_table` and
  `parquet_scan`. Until now the plan engine had no Python bindings at all — no
  Python user could build or run a query plan.

  `python/marrow/expr.py` adds the ibis-flavoured lazy frontend (no `ibis`
  dependency): `marrow.read_parquet(path)` and `marrow.scan(batch)` return a
  `LazyTable` carrying `filter`/`select`/`drop`/`rename`/`project`/`aggregate`/
  `order_by`/`limit`/`head`/`join`, executed by `collect()` or `to_pyarrow()`.
  Grouped aggregation takes a keyword surface —
  `t.aggregate(by=["k"], total=("sum", "v"))`.

  The eager, PyArrow-shaped `marrow.Table` is unchanged; the lazy type is
  deliberately a different name. See `docs/alpha-findings/b2-plan-bindings.md`
  for the naming recommendation and for what binding this layer exposed about
  the plan IR and the aggregate cluster — notably that `Relation` has no
  `inputs()`, so a plan cannot be traversed, and that no node's `write_to`
  renders its children, so there is no real EXPLAIN.

  Marshalling accepts a plain `str` anywhere a bare column reference is wanted
  (sort keys, join keys, select names), so the common path needs no expression
  objects.
- **The expression system is reachable from Python.** `marrow.expr`'s runtime
  lane now has bindings: `python/bindings/expressions.mojo` exposes `DynValue`
  as the Python type `Expr` and `DynAgg` as `Agg`, and
  `python/marrow/_expr_column.py` wraps them as `Column` / `Aggregate` with the
  usual `_Wrapper` composition. `marrow.col`, `marrow.lit` and `marrow.if_else`
  are the entry points:

  ```python
  from marrow import col, lit
  ((col("a") > 10) & col("s").startswith("x")).execute(batch)
  col("amount").sum().alias("total")
  ```

  Bound: arithmetic, comparison and boolean operators; `abs`/`sign`/`floor`/
  `ceil`/`round`/`sqrt`/`exp`/`ln`; the string kernels including `like`/`ilike`;
  the temporal extractors and `date_trunc`; `isin`/`cast`/`coalesce`/`nullif`/
  `if_else`; the six aggregates plus `alias`; and `execute(batch)` for eager
  evaluation. `is_null`/`is_valid`/`is_nan`/`fill_null` do not exist on
  `DynValue` yet and carry a marked TODO.

  Two constraints shaped the result and are documented in
  `docs/alpha-findings/b1-expr-bindings.md`. `add_type[T]` derives `tp_repr`
  by reflecting over `T`'s fields, which rejects `DynValue._eval_fn`
  (a function pointer), so the binding owns two one-field boxes that override
  `write_repr_to`. And `def_method` fills `tp_dict`, not CPython slots, so
  operator dunders cannot work at the Mojo layer — `Expr` exposes named methods
  (`add`, `lt`, `and_`) and `Column` maps them onto the operators.
- **Null handling reaches the runtime expression lane, and `fill_null` reaches
  the AOT lane.** `DynValue` gained `is_null()`, `is_valid()`, `is_nan()`,
  `is_inf()` and `fill_null(other)`. The three predicates share one
  `_predicate[K: UnaryPredicateKernel]` evaluator, so a program still links only
  the kernels its expressions name; `fill_null` widens mixed numeric operands
  first, the same rule `_binary` and `_compare` already use, so
  `col("x").fill_null(lit[Int64Type](0))` works over an int32 column.

  The AOT lane needed no new node: `FillNullKernel` gained the `combine` that
  `BinaryConditionalKernel` asks for, and `FillNull` is the existing
  `ConditionalBinary` breaker under a new alias, beside `Coalesce` and `Nullif`.
  `is_nan`/`is_inf`/`is_null`/`is_valid` were already expressible there.

- **`count_star()`** (`marrow/exprold/builders.mojo`) — `COUNT(*)`, the row count,
  as distinct from `count(col)`, which skips nulls. It needs no new kernel: a
  literal is valid on every row, so the valid-count of a constant column is the
  row count. Verified against a nullable column rather than assumed. See
  `docs/alpha-findings/a1-null-ops.md` §2 for why the implementation behind the
  name should not stay this way.
- **`LIKE`/`ILIKE` compile their pattern once per array, not once per row.**
  `StringPredicateKernel.apply` (array x array) calls `predicate` per element,
  and `LikeKernel.predicate` has to build a whole `LikePattern` -- a token
  list, a literal buffer and a `String` -- before it can match. The runtime
  expression lane evaluates a literal by `DynScalar.repeat(num_rows)`, so
  `URL LIKE '%google%'` arrived as n identical right-hand rows and paid n
  identical compilations. `LikeKernel`/`ILikeKernel` now override the array x
  array `apply` with `_match_arrays`, which remembers the last pattern text and
  recompiles only when it changes -- collapsing the constant case to one
  compile without special-casing it, while a genuinely varying right operand
  still works.

  Measured on `bench_string.mojo`, min of 5 rounds, two runs per side, with
  `length`, `contains`, `upper` and the scalar-pattern LIKE cases as drift
  controls (all flat to within 5%): `bench_like_array_1m` **250.9 ms ->
  12.0 ms (20.9x)**, `bench_like_array_dense_1m` 242.4 -> 9.8 ms,
  `bench_like_array_sparse_1m` 239.2 -> 11.4 ms, `bench_ilike_array_100k`
  38.1 -> 3.5 ms. End to end on ClickBench `hits_0.parquet` at `-O3`,
  normalised against duckdb (flat to within 5% on the untouched q01/q13):
  **q21 781 -> 92 ms, q22 807 -> 102 ms, q23 1659 -> 162 ms** -- 3.9x, 3.6x and
  4.6x. q23 now runs faster than duckdb. See
  `docs/alpha-findings/o3-string-alloc.md`.

### Fixes

- **One parameter name is now one cell, so a name declared in both expression
  lanes cannot give two different answers.** `register_param` appended to the
  registry unconditionally while upserting `_LOOKUP` last-wins, so
  `param("min-a", int64)` (fused) plus `param("min-a", DynType(int64))`
  (runtime) in one plan produced *two* declarations with *two* cells;
  `parse_params` keys by name, so only the last one received the CLI value and
  the fused node silently read its default instead — or, with no default,
  `--min-a 5` still aborted with `missing required parameter '--min-a'`.
  `register_param` now deduplicates by name and **returns the authoritative
  cell**, which the `param()` builders bind their node to, so every mention of
  a name observes the one value its flag binds. Redeclaring a name with a
  conflicting dtype raises naming both dtypes; `default`/`help` are first-wins
  and documented as such. `test_parity.mojo`'s param case was rebuilt on the
  corrected semantics — it now asserts the two lanes agree on a shared name
  bound once **through `parse_params`**, which is exactly the path the old
  hand-bound-cells test never exercised.

- **A temporal parameter can be bound from the command line.**
  `param("cutoff", timestamp(second))` was declarable but not bindable:
  `_parse_scalar` handled bool / string-like / integer / floating only, and
  `DynType.is_integer()` is variant-based so a timestamp never reached the
  integer arm — a required temporal parameter always aborted and one with a
  default could never be overridden. Added a `dispatch_temporal` arm taking
  the epoch integer in the dtype's own unit, the same spelling `default=`
  already used.

- **The runtime lane's `param` leaf prunes row groups again.**
  `DynValue.prune` had arms for `"column"` and `"literal"` but not `"param"`,
  so it fell through to `Interval.unknown()` and a parameterised predicate
  decoded every row group — while all three fused param nodes implement
  `prune()` and skipped them. An invariant-2 divergence, now closed: a bound
  parameter prunes as the point interval a literal of the same value does.

- **The runtime lane's `param()` accepts `default` and `help`.** All three
  fused overloads took them and `ParamDecl` already carried them, so a
  runtime-lane parameter was unavoidably required and invisible to `--help` /
  `--describe`. `default` is spelled as a `DynScalar` there, since there is no
  comptime dtype to build one from.

- **Re-entering the parameter registry no longer corrupts the runtime lane's
  name table.** `drain_params()` cleared and repopulated `_LOOKUP` even when
  the drained list was empty, so a second `execute_cli()` in one process left
  fused cells bound while `lookup_param` raised `unknown parameter`. An empty
  drain now leaves `_LOOKUP` alone, and `execute_cli` raises a named error if
  invoked twice (`claim_cli_invocation()`) rather than re-executing against
  the previous invocation's values. Two related limitations are now documented
  in `params.mojo` rather than fixed: an undrained plan leaves its
  declarations in the registry for the next plan's `--help` to list, and the
  registry has no synchronisation, so concurrent plan *construction* is a data
  race.

- **`marrow compile --bundle DIR` now ships a Parquet file compressed with
  any of marrow's codecs, not just uncompressed ones.** Found by actually
  running a compiled binary against a snappy-compressed file (pyarrow's
  default `pq.write_table` compression) — it raised `Failed to load snappy
  from libsnappy.dylib or libsnappy.so or libsnappy.so.1`. Two independent
  gaps, both needed:

  - `marrow/utils/compression.mojo` opened its codecs with `dlopen` on a
    bare soname (`"libsnappy.dylib"`), which the dynamic loader resolves
    through its own default search paths, never `@loader_path` — so a copy
    sitting right next to the binary was never found regardless of whether
    it was bundled. `_exe_dir()`/`_with_exe_dir()` now derive the running
    executable's own directory from `argv()[0]` and try
    `<that dir>/<candidate>` before falling back to the original bare-name
    list, so a bundled codec library is found without `DYLD_LIBRARY_PATH`,
    `LD_LIBRARY_PATH`, or any other external state.
  - `python/marrow/compile.py`'s `bundle()` only ever copied `binary`'s
    *linked* dependencies (`dylib_closure`, from `otool -L`/`ldd`) — the
    codecs are `dlopen`-ed, not linked, so they never appeared there at all.
    `codec_lib_dir()` resolves the active pixi/conda environment's `lib/`
    (via `$CONDA_PREFIX`, else two directories up from `mojo` on `PATH` —
    never a hardcoded pixi path) and `stage_codec_libs()` copies zstd,
    snappy, lz4, zlib and both brotli libraries (plus their own transitive
    deps — `libbrotlienc`/`libbrotlidec` each pull in `libbrotlicommon`)
    into the bundle. A codec missing from the environment is skipped with a
    warning rather than failing the whole bundle.

  Fixing the second gap surfaced a latent bug in `_resolve_macos_dep`: it
  called `.resolve()` on every `@rpath`/`@loader_path`/`@executable_path`
  dependency, collapsing a conda-forge version-symlink chain
  (`libbrotlicommon.dylib` -> `.1.dylib` -> `.1.2.0.dylib`) down to the real
  file's name — but the dependent Mach-O looks the library up by the
  *symlink's* name (`@rpath/libbrotlicommon.1.dylib`), so the copy landed
  under the wrong filename and `dlopen` of `libbrotlienc.dylib` failed at
  the next hop. `_resolve_macos_dep` no longer resolves past the candidate
  it found; `shutil.copy2`'s default `follow_symlinks=True` still
  dereferences it for the actual bytes. The same un-resolved candidate list
  now also surfaces the version-symlink itself as a same-content alias
  (e.g. both `libzstd.dylib` and `libzstd.1.dylib` resolve to one real
  file) — `_copy_deduped()` writes the real bytes once and symlinks every
  other name to it, instead of doubling the codec footprint.

  Bundle size grew from 5.5 MB (no codecs) to roughly 8.8 MB for
  `query_param` with every codec present; see `docs/guide/compile.qmd`.

- **`marrow compile <file> [out]` now works, matching the UX originally
  requested for `marrow compile`.** The CLI had shipped with no `compile`
  subcommand — the only working invocation was the bare `marrow query.mojo
  -o query` — which the docs task caught and flagged rather than papering
  over. `python/marrow/compile.py`'s argument parser is now built with
  `argparse` subparsers (`_build_arg_parser` -> `_add_compile_subparser`),
  so `marrow compile query.mojo`, `marrow compile query.mojo out` and every
  existing flag (`-o/--output`, `--marrow-path`, `--bundle`, `--no-writers`,
  `--fast`, `--no-strip`, `-v`) all keep working under the subcommand, and
  `marrow --help` / `marrow compile --help` list them. The bare form
  (`marrow query.mojo`) is no longer accepted: nothing outside this repo
  depended on it (the CLI has never been released), and requiring the
  subcommand is what avoids a file literally named `compile` becoming
  ambiguous. `main()` now dispatches on `args.command` through a
  `_SUBCOMMANDS` table, so a second subcommand is a new `_add_*_subparser`
  plus a table entry, not another redesign.

### Refactors

- **The fluent null predicates follow PyArrow's spelling.** `isnull`/`notnull`
  on `Value` and `isnan`/`isinf` on `NumericValue` are now
  `is_null`/`is_valid`/`is_nan`/`is_inf` — the names PyArrow exposes, and the
  ones the kernels' own `comptime name` strings already used.

  `is_null`/`is_valid` also moved off the `Value` trait onto `DynValue`. As
  trait defaults they returned the *fused* `NullPredicate`, which made the
  erased lane's version unwritable — a struct method does not override a trait
  default in Mojo, the two become competing overloads
  (`ambiguous call to 'is_null'`). That mattered:
  `col("a").is_null() | (col("b") > lit(1))` did not compile, because a
  predicate that left the lane cannot be recombined with one that stayed. The
  move costs nothing — every caller in the tree was already a `DynValue`, since
  the AOT lane spells the node directly as `IsNull(col("a", int64))`.

### Fixes

- **`filter` ignored the mask's validity bitmap**, so a comparison against a
  null selected on the raw payload underneath. The comparison kernels
  evaluate every SIMD lane whatever the validity says — a null input compares
  its payload (0), writes the result, and only then marks the lane invalid —
  so `WHERE v < 4` returned the row where `v` was NULL while `WHERE v > 3` was
  accidentally correct (`0 > 3` is False, so the stray bit was clear). Now
  intersects data with validity through `Bitmap.intersect_views`, which is
  offset-applied and so stays correct for a sliced mask. Found by the golden
  corpus, which failed it identically on both lanes.

- **Joining on differently named keys always raised.** `StructArray.select`
  keeps each field's name, a struct's dtype includes those names, and the
  `EqKernel.apply` that filters hash collisions rejects mismatched dtypes — so
  `left_on="dept", right_on="did"` died with
  `equal: dtype mismatch: struct<dept: int64> vs struct<did: int64>` and only
  joins whose keys happened to share a name worked. Key structs are now
  renamed positionally, which is how join keys are matched.

### Tests

- **Golden query corpus** (`golden/`) — 28 queries run through both the
  runtime and AOT lanes against expectations generated by DuckDB, with an
  in-memory source so what is under test is the engine rather than a scan.
  Found three defects on its first run: `filter` ignores the mask's validity
  bitmap, an empty result carries a schema but no columns, and exporting a
  zero-row batch through the C Data interface fails.
  Now 69 cases across ten areas — basic, nulls, kleene, aggregate, sort,
  join, string, conditional and cast — passing on both lanes.

- **`benchmarks/binary_size/query_param.mojo` gates what a late-bound
  parameter costs**, and the writer-gating decision it forced. Against
  `query_scan_typed` (identical query, literal-bound scan and predicate,
  `print(...execute())`), `query_param` — `param("src", string)` for the scan
  path, `param("min-a", int64)` in the predicate, `plan.execute_cli()` for the
  tail — first measured **+768,988 bytes** of `__text`, almost all of it the
  Parquet writer, the IPC writer, and their codec layer, linked in
  unconditionally by `_write_parquet_output`/`_write_ipc_output` even though
  neither gate program calls either format. Both are now gated behind
  `comptime CLI_WRITERS_ENABLED = get_defined_bool["MARROW_CLI_WRITERS",
  False]()` in `marrow/exprold/relations.mojo` — off by default, the same
  posture as `GPU_ENABLED`: `-o out.parquet`/`-o out.arrow` raise unless the
  binary is built with `-D MARROW_CLI_WRITERS=true`, `--format table`/stdout
  is unaffected. Re-measured with the flag off: **+196,700 bytes**, all of it
  `execute_cli`'s own argv/`--help`/`--describe`/`parse_params` machinery and
  a second `DynArray`/`DynScalar` dispatch-ladder instantiation from
  `_write_cli_output`'s `print(result)` being a different call site than
  `query_scan_typed`'s inline `print(...execute())` — not writer linkage, and
  out of scope for this task. See `benchmarks/binary_size/README.md` for the
  full breakdown.
- **The five ClickBench files are one query registry with three consumers.**
  `python/marrow/tests/clickbench.py` now holds all 43 canonical queries exactly
  once, each carrying its SQL (the docstring, which is also the DuckDB text
  after the documented dataset rewrites), a marrow lazy-API thunk, a polars
  thunk, and its declared status. `test_clickbench.py` checks marrow *and* every
  polars thunk against the DuckDB reference; `bench_clickbench.py` times all
  three engines lazily, interleaved per repeat, and reports medians with a
  marrow/polars ratio column; `python clickbench.py` prints the PASS/FAIL report.
  `clickbench_alpha.py` and `clickbench_reference.py` are deleted — the latter
  existed only because of a false belief that duckdb and the marrow extension
  needed separate environments, when `bench` is `["dev", "bench"]` and has both.
  42/43 unchanged; see `docs/alpha-clickbench-coverage.md` and
  `docs/alpha-findings/h1-clickbench-consolidation.md`.
- **A systematic bounds matrix for `BufferView`/`BitmapView`** — `test_views.mojo`
  goes from 69 to 82 cases, varying selection shape (including the degenerate
  ends that provably cannot overstep), the sparse/dense threshold at 23/24/25,
  element width across int8..float64, destination slack, multi-word filters
  sized to a 64-byte multiple, every bit offset 0-7 crossed with counts 1-16,
  ragged bit lengths, and sub-byte-offset views. Each was verified to fail
  against the pre-fix behaviour by reverting the three fixes in turn.
  `test_filter.mojo` gains two multi-word sliced-array cases, the shape none of
  its five existing sliced tests could reach.
- 13 cases for the above across `test_runtime.mojo`, `test_parity.mojo` and
  `test_aggregates.mojo`, including the cross-lane parity cases for `fill_null`,
  `is_valid` and `is_nan`, and the five new ops added to the op-name axis.
- `test_parity_coalesce` and `test_parity_nullif` now assert real cross-lane
  parity. They used `assert_fused` — the one-lane placeholder for "ops the
  runtime lane does not yet expose" — long after `DynValue.coalesce`/`.nullif`
  shipped, so two live ops had no parity assertion at all.

### Fixes

- **`pixi run -e wheel wheel` failed on every invocation with `ValueError:
  Readme path must be within the project directory: ../README.md`,
  independent of and pre-existing this branch's changes** (confirmed by
  reproducing the failure with the wheel-payload change stashed out). A
  conda-forge `hatchling` bump (1.29.0 -> 1.32.0, see `pixi.lock`) added
  strict validation that `project.readme` resolve inside the build root —
  `python/pyproject.toml` pointed at `../README.md`, one level above
  `python/`, which every prior hatchling version accepted. Fixed with
  `python/README.md` as a symlink to `../README.md` (single source of truth
  kept at the repo root) and `readme = "README.md"` in
  `python/pyproject.toml`.

- **Hash join sized its probe from the build side, so the plan layer's 8192-row
  morsels each paid a full radix partitioning.** `HashJoin.probe` picked serial
  vs partitioned by re-asking `worth_parallel` about `self._left_rows`, letting
  one row count answer two unrelated questions: which layout `build` produced
  (a correctness constraint) and whether *this call* is big enough to
  parallelise (a throughput decision). `JoinProcessor` streams the probe side
  in 8192-row morsels, so a 1M-row build sent ~122 tiny probes down the
  partitioned path and the 1M join ran 1.7x slower than serial once
  `ExecContext.auto()` became the plan default.

  The layout is now recorded by `build` in `_built_parallel` and simply
  followed. The throughput levers moved to where the per-call cost actually is:
  `_probe_ctx` sizes the probe's hashing from that call's rows via
  `worth_parallel` (which reads a forced thread count as a budget, not an
  instruction — striping an 8192-row probe across 8 forced workers cost
  ~1213 us against ~76 us on the calling thread), and `_DEFAULT_RADIX_BITS`
  drops 6 -> 4. The 64-partition default came from a *one-shot* 10M sweep that
  amortized partitioning over a single call; re-swept on the morselized shape,
  16 partitions is the only setting that beats the serial baseline at every
  size.

  Measured (`marrow/kernels/tests/bench_join.mojo`, 1M build, 1M probe in
  8192-row morsels, medians, normalised against an untouched `SumKernel`
  control): **-35.4% / -37.5% / -34.9%** at 2 / 4 / 8 workers. Whole join
  including build, 8 workers: 30.07 -> 16.84 ms at 1M, 93.54 -> 68.52 ms at 4M,
  238.92 -> 181.13 ms at 10M, against a serial bar of 17.38 / 101.76 /
  360.52 ms. One-shot big probes give up 8.9% at 8 workers to the smaller
  fanout and remain 3.0x faster than serial.

  No fan-out threshold was added: measurement showed fanning the partitions out
  beats running them serially at every batch size tested, 8192 rows included,
  so the expensive thing is the partitioning and not the `sync_parallelize`.
  Details and the plan-layer morsel-size recommendation in
  `docs/alpha-findings/o5-join-threshold.md`.


- **`FilterProcessor` and `JoinProcessor` no longer drop the execution
  context.** `JoinProcessor` built its index with `HashJoin[RapidHash64]()`,
  falling through to that constructor's serial default, so a plan-driven join
  could never reach `build_parallel` / `probe_parallel` however many workers the
  caller had asked for. `FilterProcessor` called `filter(col, mask)` and got the
  kernel's `ExecContext.serial()` default the same way. Both now carry the
  context `to_processor` was given.
- **The Parquet codec libraries are opened once per process, not once per read
  (O1).** `CompressionLibs` was built per read and per write, and its destructor
  `dlclose`d whatever it had opened. Nothing else in a marrow-only process holds
  `libsnappy` / `libzstd` / `liblz4` / `libbrotli*`, so that `dlclose` really
  unmapped them and the next read re-mapped and re-linked them — about 0.9 ms of
  pure set-up per `read_table`.

  The handles now live in one process-wide `_CodecHandles` behind the stdlib's
  `_Global`, opened on first codec use and never `dlclose`d before teardown.
  They are immutable after initialization, so the workers a read dispatches
  share a read-only structure; the mutable half — snappy's reused size
  out-param — stays in the per-worker `CompressionLibs`, which is now just that
  scratch plus the block calls. `ParquetFile.read` opens the set on the calling
  thread before dispatching, and only when a chunk it is about to decode is
  actually compressed, so an all-uncompressed program still loads no codec at
  all (verified with `DYLD_PRINT_LIBRARIES`).

  A `dlopen` that cannot find its library is recorded rather than raised — the
  set is opened inside a global initializer that cannot raise — and re-raised
  with the original text from the first page that needs that codec, so a box
  without `libbrotlienc` still gets working zstd exactly as before.

  New `bench_read_small_snappy` / `bench_read_small_uncompressed` isolate the
  per-read set-up on a 1,000-row file: **921.4 us -> 31.7 us** (29x), against an
  untouched uncompressed control that moved 24.0 -> 23.5 us. End to end,
  ClickBench Q1 in a marrow-only process goes **9.25 ms -> 8.43 ms** (-8.9%,
  three interleaved pairs at 300 repeats); queries that do real work absorb the
  same fixed ~0.9 ms and do not move measurably.
- **`filter` over a *sliced* column silently dropped rows.**
  `BitmapView.load_bits[T]` issues one unaligned `size_of[T]()`-byte load and
  shifts it down by the view's sub-byte bit offset, so the run's top
  `offset & 7` bits — which live in the byte after the load — came back as
  zeros. Every `BitmapView` over a sliced array carries such an offset, and both
  `BufferView.filter` and `BitmapView.filter` read their selection this way, so
  a filter over a slice returned fewer rows than the predicate selected, with no
  error. The five existing sliced-filter tests all used 3-5 element arrays,
  which fit entirely inside `filter`'s tail block where the tail mask discards
  the corrupted bits; reproducing it needs a slice longer than 64 elements.
  `load_bits` now folds in the following byte when the offset is sub-byte and
  that byte is inside the view — the `offset == 0` case short-circuits, so the
  hot loop is unchanged.

- **`BitmapView.compressed_store` no longer read-modify-writes up to 7 bytes
  past the bitmap.** It always stored a full 8 bytes (plus a 9th when the run
  straddled) and justified the overshoot with "Arrow buffers are 64-byte
  padded". They are padded to a 64-byte *multiple*, which is not slack: a
  512-bit bitmap is exactly 64 bytes, so the last block of a filter wrote 7
  bytes of the neighbouring allocation. The written-back value was unchanged
  (`x | 0`), so it never corrupted a heap, but it was a lost-update race between
  threads filtering into adjacent allocations and an out-of-bounds write to any
  sanitizer. It now writes exactly `ceildiv(bit_offset % 8 + count, 8)` bytes;
  the wide path still serves every full 64-bit block, so only the short final
  block changes.
- **`binary` → `string` no longer re-validates UTF-8 one element at a time.**
  Parquet `BYTE_ARRAY` columns arrive as `binary` and the string kernels are
  bound on `StringLikeType`, so every string query casts — 28 of ClickBench's
  105 columns. `BinaryLikeCast.apply` relabels with no allocation when the
  offset widths match, so the cast itself is free; the whole cost was the
  `safe` guard's `_check_utf8`, a scalar loop building a `StringSlice` per row
  and calling `_is_valid_utf8` on it, over an immutable buffer it had already
  validated. It ran at 4.2 GB/s on ASCII and 5.5 GB/s on ClickBench's `URL`.

  `_check_utf8` now tries two whole-buffer conditions first. `_all_ascii` is a
  four-accumulator SIMD OR-reduction — every byte `< 0x80` makes each element
  valid however the offsets cut the bytes — reduced once per 4 KiB so a buffer
  that fails bails near the first non-ASCII byte rather than after a full pass.
  Failing that, `_validate_utf8_window` validates the window while skipping
  pure-ASCII SIMD blocks, and the element starts are scanned for continuation
  bytes. Both are sufficient conditions only: either failing falls through to
  the original loop, so the accept/reject decision is unchanged. `safe=True`
  stays the default and still rejects malformed input.

  The block skipping is what makes this work on real data rather than only on
  synthetic ASCII. ClickBench's `URL` is 4.5% non-ASCII by byte so it never
  takes the all-ASCII path — but those bytes are clustered, and 92.7% of
  16-byte blocks are pure ASCII. A pure-ASCII block cannot hold part of a
  multi-byte sequence, so every region handed to `_is_valid_utf8` begins and
  ends on a character boundary.

  The boundary scan is load-bearing: `"é"` is `0xC3 0xA9`, and split across two
  elements the concatenation validates while each half is malformed, so a
  window-only check would be strictly weaker than the loop it replaces. So is
  the fall-through — a null slot may hold arbitrary bytes, and falling back
  rather than raising keeps that from becoming a false rejection. Both are
  pinned by tests.

  Measured through the Python API, rebuilding `libmarrow.so` per variant:
  72.7 MB of pure ASCII **17.19 ms → 0.83 ms (x20.7)**, and ClickBench's 88.5 MB
  `URL` column **16.01 ms → 6.14 ms (x2.6)**. End to end on `bench_clickbench.py`
  with both builds run back to back, the 27 string-cast queries moved by a
  median **-4.5%** while the 15 queries with no string cast — the drift control
  — sat at +0.4%: **q21 346.5 → 328.7 ms (-5.1%)**, **q34 106.9 → 96.7 ms
  (-9.5%)**, q28 -18.4%, q37 -13.3%, 41-query total 4 052 → 3 905 ms.

- **`filter` no longer writes one element past its output buffer (F1).** This is
  the SIGSEGV behind ClickBench Q11, Q12 and Q24 — a bare `exit -11` with no
  message, always reported from inside tcmalloc at some later, innocent
  allocation.

  `BufferView.compressed_store_dense` is branchless by design: it stores a lane
  before it inspects that lane's selection bit, so an unselected lane writes a
  value the next selected lane overwrites. Every lane *above* the highest set
  bit has no next selected lane, so its write survives — at element index
  `popcount(sel_bits)`, one past the packed output. `BufferView.filter` sizes
  its destination to the pre-counted set-bit total, so on the last selection
  word of a filter that element is outside the buffer, and
  `Buffer._aligned_size` aligns to 64 bytes without padding beyond it — an
  `int64` output of length divisible by 8 therefore has no slack at all and the
  store lands in the neighbouring heap block, corrupting tcmalloc's freelist.

  The adaptive `compressed_store` now takes the dense path only when the view is
  longer than the popcount, which is false for exactly one word per `filter`
  call. Over-allocating in `filter` instead was rejected: it would leave
  `compressed_store_dense` a trap for the next caller.

  Neither `COUNT(DISTINCT)`, the group-by, the aggregate layer nor Parquet page
  skipping is involved, despite every crash report naming
  `AggregateProcessor::pull`; see `docs/alpha-findings/f1-distinct-segfault.md`
  for the bisection and the ruled-out list.
- **Grouping, concatenating or joining on a `binary` key no longer aborts the
  process (C1).**

  `BinaryLikeBuilder`'s erased `extend(DynArray)` reconstructed the *source*
  array's type from the **builder's own offset width** — `as_string()` when
  `Self.T.offset` was int32, `as_large_string()` otherwise. Offset width does
  not identify a type: `BinaryType` and `StringType` are both 32-bit-offset
  `BinaryLikeType`s. So a `BinaryBuilder` handed a `binary` array asked the
  variant for `BinaryLikeArray[StringType]`, and since `DynArray.as_type`'s
  `debug_assert` is compiled out under release, `Variant.__getitem__`
  **aborted the process**: `get: wrong variant type`. The typed leaf one line
  below already accepted any `U: BinaryLikeType`; only the erased wrapper was
  narrow. Both it and `ListLikeBuilder.extend` (same defect — `MapType` is a
  32-bit-offset `ListLikeType`) now resolve the concrete type from the
  *source* dtype.

  Both are explicit `is_…()` ladders rather than `dispatch_binarylike` /
  `dispatch_listlike`, and have to be: capturing `mut self` in a dispatch
  closure miscompiles here. `ListLikeBuilder`'s version fails the pass manager
  (`'kgen.call' op callee argument #1 expected type …`) because its child is a
  `DynBuilder`; `BinaryLikeBuilder`'s codegens and then crashes the binary at
  startup. **`mojo precompile` reports both clean** — it elaborates without
  running codegen — so only a real test-driver build catches them.

  Two reachable callers:

  - **`group_by` on a `binary` key**, but only above ~60,000 rows and only at
    low cardinality. Of the three grouping strategies, `GROUP_THREAD_LOCAL` is
    the only one that materializes unique keys through a builder — serial and
    radix gather theirs with `take` — so the abort read as a size threshold
    rather than a type bug. The expression layer's streaming group-by shares
    the `HashGrouper` and shared the abort.
  - **`concat()`**, which is entirely `DynBuilder.extend`-driven, and therefore
    **`ChunkedArray.combine_chunks()`** — a `binary` column in a multi-chunk
    `Table` aborted on combine, at any size.

  Nothing caught it because the suite exercised `BinaryLikeBuilder` only
  through `StringBuilder`, the alias that happens to work: `BinaryBuilder` and
  `LargeBinaryBuilder` appeared in no test, and none of `test_concat.mojo`'s 16
  cases used `binary`.

  Found while auditing: **`equal_any` picked its kernel family with
  `is_string() or is_large_string()`**, so `binary` fell into the numeric arm
  and `dispatch_primitive` raised — hash-join row verification is built on
  `equal_any`, so joining on a `binary` key was impossible while the identical
  join on `string` worked. It now dispatches `binarylike` into a `_bytes_equal`
  leaf. `StringPredicateKernel` is deliberately *not* widened: `LIKE`, `upper`
  and `startswith` are text operations and their `is_string_like` guards should
  keep rejecting `binary`.

  `filter`, `take`, `sort`, `rapidhash`, `count_distinct` and the Parquet
  writer were audited and are correct. Tests are now parameterized over
  `binary` / `large_binary` / `string` / `large_string` rather than exercising
  the shared generic through `StringBuilder` alone. Write-up in
  `docs/alpha-findings/c1-binary-groupby.md`.

- **The Python extension builds again.** `python/bindings/arrays.mojo` still
  called `dt.variant_dispatch_raises`, which `e5509c3` deleted when it replaced
  the shared dispatch helper with per-box `isa` ladders — so `libmarrow.so` had
  not compiled, and `python/marrow/tests` had not run, for three commits. The
  two call sites now write out the ladder like every other box. 416 passed.

  Nothing reported this: the binary-size gate does not build the extension, and
  CI has not run since 2026-05-11.

- **`cast()` no longer drops `safe` and `ctx` on the way to a kernel, and two
  casts that silently corrupted data now raise (S4).**

  `cast(array, to, safe, ctx)` delegated to fifteen kernels across **six
  different signatures**, so its ladder handed each arm only the arguments that
  arm happened to accept. Six kernels took no `ctx`; seven took no `safe`. Two
  of those seven could genuinely fail, and did, under the default that promises
  to raise:

  - `cast(x, decimal128(38, 2), safe=True)` **wrapped on overflow**, and the
    value did not survive — a test asserting the corruption positively passes
    against the old code.
  - `TemporalCast` — not previously reported — **discarded its remainder** on a
    unit downscale and **overflowed int64** on an upscale, so 1500 ms became
    1 s and 10^10 s became garbage. Arrow C++'s `ShiftTime` raises on both,
    under `allow_time_truncate` and `allow_time_overflow`.

  The suite encoded the temporal defect: `test_timestamp_unit_downscale`
  asserted the truncation under the default and now passes `safe=False`.

  All fifteen kernels conform to a new `trait CastKernel(Kernel)` with one
  `dispatch(array, to, safe, ctx)`. The trait buys signature uniformity, not
  code reuse and not a shared dispatcher — `cast()` stays a hand-written ladder,
  because a closure generic over its own trait bound needs a narrowing adapter
  that inlines into every arm (+662,740 bytes, measured previously).

- **The decimal cast family is five kernels instead of one, removing 1,268,480
  bytes (−20.7% of `__text`) from `query_dynvalue`.**

  `DecimalCast` did decimal↔decimal, decimal↔float and decimal↔integer behind a
  single `_convert[FromN, ToN]` that resolved *both* sides over "decimal or
  numeric" — roughly 16 × 16 monomorphizations, every line inside stamped into
  all of them. That stayed cheap only while the bodies were three-line unchecked
  casts. Adding the checks `safe` requires cost **+371,584 bytes for 1,476 bytes
  of source, a 250× multiplier** and 85% of the change's total size regression.

  `DecimalRescale`, `IntToDecimal`, `DecimalToInt`, `FloatToDecimal` and
  `DecimalToFloat` each dispatch only over the families they can see, with
  `DecimalCast` kept as a router so `cast()` has one decimal arm. Two of the
  five lose a runtime branch outright: an integer source is always scale 0, so
  `IntToDecimal` only scales up, and an integer target is always scale 0, so
  `DecimalToInt` only scales down — the fat kernel had to test `delta` on every
  path. Net effect is 1.27 MB *below* the pre-change baseline.

  Attribution, `__text` on `query_dynvalue`, single-variable A/B:

  | variant | `__text` | Δ |
  |---|---:|---:|
  | base (`8cc9723`) | 6,139,636 | — |
  | `_map`'s op made `raises` | 6,139,636 | 0 |
  | trait + widened signatures + `TemporalCast` | 6,203,124 | +63,488 |
  | + `DecimalCast` checks (fat kernel) | 6,574,708 | +435,072 |
  | + the split (shipped) | 4,871,156 | **−1,268,480** |

  Making the map's op raising is free; the error-message interpolation was only
  54,912 bytes of the regression. The cross product was the whole story.

- **`DynArray.slice` is no longer `raises`, which removes 46,164 bytes
  (−3.3% of `__text`) from `query_streaming_agg_fused`.** It was raising only
  for as long as `DynArray` conformed to `Array`, and outlived that conformance
  by several commits; every typed `slice` is total, so the erased one is too.
  The cost was error-handling machinery threaded through a 37-variant ladder.

- **Three null-counting and equality defects in `arrays.mojo`.**
  - `DictionaryArray.slice` copied the *parent's* null count onto every
    sub-range, so a 10-element slice of a 1000-element array with 500 nulls
    reported 500 — a count larger than the length. A dictionary's validity is
    its indices', so the count now comes from slicing those, which the previous
    point makes possible in one non-raising call.
  - Every `slice()` passed its raw `length` argument — `-1` on the defaulted
    call — into the null count, where `Bitmap.view` reads `-1` as "to the end of
    the bitmap". A slice of a slice therefore counted validity bits it does not
    own, and was correct only when an array happened to own its whole bitmap.
  - `DictionaryArray.__eq__` compared `_indices` and `_values` whole and never
    looked at `_offset`, so two *different* slices of one parent with equal
    length compared equal. It now compares decoded values, which also makes two
    arrays encoding the same column against differently ordered dictionaries
    equal. `DictionaryScalar.__eq__` stops comparing `_index` for the same
    reason: the slot a value was read from is not part of its identity.

- **Validity comparison in `__eq__` no longer scans.** The shared helper
  popcounted to answer a question its callers had already settled in O(1), then
  compared bitmaps bit by bit where `BitmapView.__eq__` does it word-at-a-time.
  All seven implementations now compare null counts and, only when those are
  nonzero, the two offset-applied views.

- **Erased dispatch no longer routes through a shared `variant_dispatch`,
  recovering 662,740 bytes (−31.9% of `__text`) on `query_streaming_agg_fused`.**

  The closure migration factored the four erased boxes' `isa` ladders into one
  generic `variant_dispatch(v, func)` in `marrow/utils/dispatch.mojo`. Because a
  closure type cannot be generic over its own trait bound, that helper has to
  bind `func` on `Movable` and let each caller narrow through an extra `narrow`
  closure — and *that adapter is inlined into every arm of every
  instantiation*. It cost +739,316 bytes (+54.8%) on the fused-aggregation
  gate, where `variant_dispatch` instantiations alone were 1,186,564 bytes,
  57% of `__text`.

  `DynArray`, `DynScalar`, `DynBuilder` and `DynType` now each write out their
  own `comptime for` over `Self.VariantType.Ts`, and `DynType.dispatch_*` do the
  same at their dtype family instead of layering over `_dispatch`. The helper
  module is deleted; it had no other callers. **The value-taking closure form is
  untouched** — every `func` is still a value with an explicit capture list, and
  no `@__parameter` was reintroduced.

  Attributed by single-variable A/B, `__text` on the gate:

  | variant | `__text` | Δ |
  |---|---:|---:|
  | base (`84c8d4a`) | 2,077,396 | — |
  | `R: Movable` → `R: AnyType` | 2,077,396 | 0 |
  | in-loop `conforms_to` elision at the box trait | 2,077,396 | 0 |
  | `dispatch_*` bypass `_dispatch` (one layer) | 1,974,972 | −102,424 |
  | …plus elision at the dtype family | 1,972,628 | −2,344 |
  | `DynArray._dispatch` ladder written out | 1,839,876 | −132,752 |
  | all four boxes written out | 1,414,656 | −425,220 |

  So the cost was the *interposed closure layer*, not the `Movable` return
  bound (0 bytes) and not lost arm elision (2,344 bytes). `DynType.VariantType.Ts`
  is reachable at comptime, which is what makes the local ladder writable.

  **Every gate improves, and the erasure-heavy ones improve most** — measured
  base-vs-fix on the same machine and toolchain:

  | gate | before | after | Δ |
  |---|---:|---:|---:|
  | `query_dynvalue` | 10,957,172 | 6,198,516 | −4,758,656 (−43.4%) |
  | `query_runtime` | 10,956,020 | 6,197,300 | −4,758,720 (−43.4%) |
  | `query_streaming` | 2,607,180 | 1,481,896 | −1,125,284 (−43.2%) |
  | `query_exprs` | 2,268,196 | 1,558,948 | −709,248 (−31.3%) |
  | `query_streaming_agg` | 2,625,876 | 1,929,648 | −696,228 (−26.5%) |
  | `query_streaming_agg_fused` | 2,077,396 | 1,414,656 | −662,740 (−31.9%) |

  Cold build times roughly halve with them (the fused gate ~45 s → ~29 s,
  `query_dynvalue` ~189 s → ~91 s). `benchmarks/binary_size/baseline.json` is
  left alone deliberately — it is owned by another branch and predates both
  this and the Mojo 1.1 upgrade, so re-baselining is a separate step.

### Refactors

- **`query_dynvalue` joins the binary-size gate.** The four gates it watched
  link **zero** symbols from `marrow.kernels.cast` — checked with `nm -C` across
  all eleven benchmark binaries — so the whole cast family was ungated, and a
  change that added +7.09% to `query_dynvalue` measured 0.00% on every gate CI
  actually reads. `query_runtime` and `query_sort` also link the ladder and stay
  ungated; one gate was judged enough signal for the extra sweep time.

- **`marrow.expr` is down from three import cycles to one, and the survivor is
  structural.** Pure code motion, no signature or API change: `col("a")` and
  `col("a", int64)` both still resolve from `marrow.expr`.

  - **`expr/core.mojo`** (new, a leaf) holds `Datum`, `into_array` and
    `_union_columns` — the vocabulary both lanes speak. `dynamic.mojo` no longer
    imports `values.mojo` for `Datum`.
  - **`expr/builders.mojo`** (new) holds the **entire** `col` / `lit` /
    `if_else` / `coalesce` / `case_when` overload set, typed and untyped
    together. An overload set cannot span modules — splitting it is what made
    `__init__.mojo` re-export `col` from two places and trip *"importing 'col'
    from multiple modules is deprecated"*.
  - **`BoxedValue` moved from `relations.mojo` to `values.mojo`**, beside the
    `Value` it erases. That is what kills both `values -> relations` and
    `execution -> relations`.
  - Removed an unused `_promote_operands` import in `values.mojo`.

  `values <-> relations` and `relations <-> execution` are gone. `values <->
  dynamic` remains and is not a placement accident: `DynValue` conforms to
  `Value`, `Value` defaults `count_distinct` to an `AggExpr`, `Value` defaults
  `isnull`/`notnull` to the fused `NullPredicate`, and `AggExpr` holds an
  unresolved `DynValue` and converts implicitly from a `Reduction`. Those form
  one strongly connected component that no placement of `Value` or `AggExpr`
  can break, which is why neither moved and why `core.mojo` does not hold them.

- **`trait Join` is gone; `HashJoin` is free-standing.** The kernel-side join
  algorithm trait (`kernels/join.mojo`) declared four abstract methods over
  exactly one conformer. Its own docstring said it carried no dispatch —
  "operators use concrete types directly" — so it constrained nothing and the
  compiler derives `HashJoin`'s `Movable` conformance without it. The
  commented-out `SortMergeJoin` field list went with it: it existed only to
  argue the trait was not hash-specific, and sort-merge join (backlog M3.1) will
  be designed fresh. Unrelated to `struct Join(Relation)`, the plan node in
  `expr/relations.mojo`, which is untouched. No behaviour, signature or binary
  size change.

- **The hashing kernel is pluggable, and the aHash string fallback is gone.**

  `marrow/kernels/hashing.mojo` hashed numeric columns with rapidhash and
  string columns with `std.hashlib.hash` (aHash), because rapidhash's
  multi-branch byte-string path had never been ported. Both halves are fixed:
  `RapidHash64.hash` is that path, and the kernel is `HashKernel[H: Hasher]`, so
  the algorithm is a comptime parameter rather than a hard-coded call.

  - **`Hasher`** (`utils/hashing.mojo`) is three methods: `hash(span, seed)`,
    `hash_lanes[byte_width, W]`, and a defaulted `combine_lanes[W]`.
    `std.hashlib.Hasher` is deliberately not reused — it folds a whole SIMD
    vector into *one* accumulator and its `finish` consumes the hasher, so a
    per-row digest would need one instance per lane, which is the opposite of
    what the kernel's hot loop needs.
  - **Three conformers.** `RapidHash64` (v3, ported and validated against the
    upstream C), `XxHash64` (its lane form is new), and `AHash64`, whose span
    hash delegates to `std.hashlib` so there is no second aHash here to drift.
    `RapidHashKernel` / `XxHashKernel` / `AHashKernel` alias the kernel.
  - **`SwissHashTable` and `HashJoin` take the hash as a type**, not a callable
    (`SwissHashTable[Hash: Hasher = RapidHash64]`). `test_join`'s degenerate
    collision hash became a 15-line `Hasher` conformer, which is the smallest
    demonstration that the parameter is open.
  - **`RapidSecret`** ports wyhash's `make_secret` for optional per-process
    randomised secrets (HashDoS resistance), with all four upstream invariants
    — odd, every byte popcount 4, pairwise XOR popcount 32, prime — pinned by
    tests. Off by default; `hash` uses the canonical secret. Measured free on
    the lane path (-0.33%), so the cost is startup only.

  **`LittleEndian.fixed` now does one unaligned wide load** instead of copying
  `W` bytes into an `Array` one at a time — about 8 loads and a stack temporary
  per 64-bit read. Rapidhash string hashing got **47.9x faster**, xxhash 20x;
  every Parquet and IPC decode was paying it too. Found by the three-way kernel
  comparison, where aHash was 14-38x faster purely because `std.hashlib` loads
  wide.

  On 100k keys, rapidhash now wins every workload: numeric lanes 123 µs (xxhash
  1.09x, ahash 1.31x), 16-byte strings 185 µs (1.21x / 1.01x), 64-byte strings
  220 µs (2.50x / 1.32x). So replacing the fallback is a 1.3x speedup on long
  string keys rather than the 14x regression it was before the load fix.

  **No regression on the kernel**: three interleaved runs against the
  pre-refactor commit gave medians -0.53%, -0.44% and -0.05% across the nine
  `bench_rapidhash_*` cases. `HashKernel[H]` resolves at comptime, so the
  generated code is what a hand-written kernel for one algorithm would be.

  Reference vectors were generated by compiling upstream `rapidhash.h`, covering
  every branch of `rapidhash_internal` (the <=16 sub-cases, the 17-112 ladder,
  and the 112/113 and 224/225 loop boundaries). `python-xxhash` is a new dev
  dependency and is the oracle for XXH64; the PyPI `rapidhash` package is *not*
  usable as one — it implements a different revision and disagrees with the
  canonical header on all 27 vectors.

- **`marrow/utils.mojo` became a `marrow/utils/` package**, and the two other
  modules that were really format-agnostic primitives moved into it.

  The old file was four unrelated things behind one name — variant dispatch (its
  documented purpose), byte order, CRC-32 and GPU capability — and
  `marrow/parquet/utils.mojo` was a *second* module called `utils` holding the
  codec bindings. Now: `dispatch` · `byteorder` · `checksum` · `hashing` ·
  `compression` · `testing`, each depending on `std` alone. `__init__.mojo`
  re-exports the names, so `from ..utils import LittleEndian` is unchanged at
  every call site.

  - **`CompressionLibs` moved out of `marrow/parquet/`.** Nothing in it is
    Parquet-specific — it is `dlopen`ed zstd/snappy/lz4/zlib/brotli. The
    format-specific half (codec codes, the legacy Hadoop LZ4 frame, the
    bit-unpacker scratch slack) stays as `Compression` in `parquet/codecs.mojo`.
    Arrow IPC is the waiting second consumer: it currently *refuses* compressed
    bodies.
  - **Both hash functions now live together** as `RapidHash64` and `XxHash64` in
    `utils/hashing.mojo`, behind a `Hasher` trait (`name` + a static
    `hash(span, seed) -> UInt64`) so the algorithm is swappable. `std.hashlib`
    was surveyed first: it has `AHasher` and `Fnv1a` behind a *streaming*
    `Hasher` protocol, and no CRC-32, XXH64, rapidhash or LEB128 anywhere — none
    of these six is replaceable by std. rapidhash's six free functions and three
    module-level secrets became `RapidHash64`'s static methods.
  - **`GPU_ENABLED` and `has_accelerator_support` went to `marrow/execution.mojo`
    instead**, the latter as `ExecContext.has_accelerator_support` — device
    capability is `ExecContext`'s job, not a utility's. This also removed
    `views.mojo`'s only absolute `from marrow.…` import.
  - **`marrow/testing/` collapsed into `utils/testing.mojo`**, dropping the
    byte-identical `CLIFlags` and `_print_json_array` the two files each carried.
    It is the one submodule not re-exported from `__init__.mojo`: every module
    imports `marrow.utils`, and none should pull `std.benchmark` in behind it.

  **Measured, because rapidhash is on the group-by/join hot path.** Extracting
  the primitives is free: **-0.03% median, range -1.36% to +0.36%** across the
  nine `bench_rapidhash_*` cases, interleaved four rounds each against the
  original inline version. A separate interleaved A/B put the struct wrapper
  itself at **-0.07%**, so the encapsulation costs nothing. An earlier *nested*
  baseline-then-change comparison showed +2% on int64 — that was the artifact
  `docs/backlog.md` §0 warns nesting invents, and it disappeared under
  interleaving.

  New: 40 tests over `marrow/utils/tests/` and 12 benchmarks in
  `utils/tests/bench_hashing.mojo`. The XXH64 vectors come from an independent
  Python implementation of the canonical algorithm, validated against the two
  values already pinned in `test_bloom.mojo` and covering every branch of `hash`;
  CRC-32 is checked against `zlib.crc32`; and `mum_wide` is asserted equal to
  `mum` lane for lane, which nothing previously covered.

- **Cleared the unambiguous items from the duplication audit**
  (`docs/duplication-audit.md`), ~215 lines net.

  - The three-line comment above all seven `_sliced_null_count` calls in
    `arrays.mojo` is gone — `_sliced_null_count`'s own body comment already said
    it, and the copies were most of the apparent `slice()` duplication.
  - `python/marrow/__init__.py` gained a `_Tabular(_Wrapper)` base holding the
    15 methods `RecordBatch` and `Table` had written out twice. It sits *between*
    `_Wrapper` and the two classes, so `Array` and `Scalar` do not inherit a
    tabular surface.
  - `bench_bitmap.mojo`'s three `_bench_pack_bools_w{8,32,64}` helpers became one
    `_bench_pack_bools[W]`; the alternating pattern is built rather than spelled
    out as a 64-element literal. Built once in setup, so the timed closure is
    unchanged.
  - `marrow/tests/test_parquet.mojo` moved to `marrow/parquet/tests/`, beside the
    nine other Parquet test files.
  - `load_word_le` moved from `views.mojo` to its only caller,
    `parquet/codecs.mojo` — which already uses raw pointers, so the "so decode
    kernels need not call `unsafe_ptr()` themselves" rationale no longer applied.
  - Six orphaned section banners: the `dispatch_*` family banner moved from the
    end of `utils.mojo` into `DynType` where the adapters actually live; the
    `AggFunction` banner moved from the end of `kernels/aggregate.mojo` up to the
    trait it describes; four `# Main` banners left in `bench_*.mojo` from when
    benches carried a `main()` deleted.
- **The erased containers no longer conform to the traits they erase.**
  `DynBuilder`, `DynArray`, `DynScalar` and `DynType` dropped `Builder`,
  `Array`, `ArrowScalar` and `DataType`. Each exposes the same surface as its
  own API; nothing is generic over those traits except the boxes' own
  `_dispatch` closures, so the conformances had no consumer. They were added in
  `8334bf0` as step 1 of a lane unification that `7d57398` then abandoned.

  The four were load-bearing only for each other, in two closed loops:
  `DynBuilder.ArrayType` required `DynArray: Array`, which required
  `DynArray.ScalarType`, which required `DynScalar: ArrowScalar`; and
  `Value.OutType: DataType` — read by no `[V: Value]` code anywhere — required
  `DynValue.OutType = DynType`, which required `DynType: DataType`. Removing
  `Value.OutType` collapsed the second loop entirely.

  What went with them: `DynArray.type()` (a forward to `dtype()`, which ~200
  call sites use), `DynArray.__init__(ArrayData)` (a delegate to `from_data`),
  the `ArrayType`/`ScalarType`/`OutType` companion members, and `DynType`'s
  `offset` placeholder. `Array.slice` and `Builder.reset` are non-raising again
  — they were widened only to accommodate the erased implementations' variant
  dispatch. The `pymethod[DynScalar.X]()` compiler crash that `8334bf0` worked
  around with three hand-written binding wrappers does not recur, confirming the
  conformance caused it.

  `DynValue: Value` stays. It is the one erasure with a consumer outside its own
  loop — `BoxedValue` boxes it, and `NullPredicate`/`IsIn`/`WindowFunction` are
  bound on plain `Value`, so a runtime leaf can feed a fused tree.

  Binary size: **0 bytes on all four gates**, byte-identical to the branch
  point. `8334bf0` had recorded +13,428 for adding the raising requirements, but
  a `raises` on a trait requirement only costs anything at generic call sites,
  and there are none.

- **Migrated 288 of 292 closures off parametric `@__parameter` onto
  value-taking unified closures.** `Benchmark.iter`, `ExecContext.stripe`, the
  `views.apply` family, `DynType.dispatch_*`, the erased
  `DynArray`/`DynScalar`/`DynBuilder` methods, the Parquet leaf decoders and the
  aggregate/partition/cast helpers all take their closure as an argument.
  Explicit capture lists surfaced state the implicit form hid — `{mut
  write_offsets, imm}` in `sort`'s radix scatter, `{mut writer, imm}` on every
  `write_to`.

  Three structural changes did the real work, each forced by a compiler limit:

  - **`variant_dispatch` binds `func` on `Movable`, not on a trait parameter.**
    A closure type cannot be generic over its own trait bound, so the dispatch
    loop cannot name the caller's trait. Binding on `Movable` — which `Variant`
    already requires — removes the parameter and leaves narrowing to the caller.
    `DynType`, `DynArray`, `DynScalar` and `DynBuilder` each wrap it in one
    adapter instead of hand-rolling the loop. Free: `bench_cast` median +0.2%.
  - **`parquet`'s `_drive_leveled` takes a `LeveledSink` trait**, not four
    closures. Sibling closure arguments may not both capture mutably, and all
    four worked on one builder; passing the state as a parameter makes the
    callbacks its methods. A state *struct* handed to four closures does not
    work — it makes them parametric over the leaf's dtype.
  - **`views.apply` split into five single-purpose functions** —
    `_cpu_striped` / `_cpu_serial` / `_gpu_launch` / `_apply_dispatch` /
    `_apply_packed_dispatch`. The distinction is not "is the output packed" but
    "can this lane run on a device", which `elementwise`'s
    `RegisterPassable & ImplicitlyCopyable` bound already states.

  `sync_parallelize`'s value form takes a non-raising worker, so the three
  raising workers (`groupby`, `partition`, Parquet row-groups) collect errors
  into a per-worker slot — sized like the result slots beside them — and raise
  after the join. Each worker still unwinds at its own first error;
  `sync_parallelize` offers no cancellation, so sibling workers cannot be
  stopped, which was equally true of the parametric form this replaces.

  **Performance is neutral, measured against a drift reference.** This machine
  drifts up to ±8% per benchmark, so deltas were normalised against 24 untouched
  `count_set_bits`/`set_range` rows: median −0.0% over 30 sort/expr/scan
  benchmarks, worst +1.5% against a ±7.7% same-config floor, with
  `bench_sort_int32_10k` a reproducible −25%. 2033 tests pass, 15/15 on GPU.

  **This entry claimed "binary size max +0.36%", and that was wrong.** The
  sweep behind it did not cover the gate the change actually hit: measured
  later by a single-variable A/B against the parent commit on the identical
  pinned toolchain, `query_streaming_agg_fused` went 1,349,768 → 2,089,084
  bytes of `__text` — **+739,316 (+54.8%)**. The cause was the shared
  `variant_dispatch` helper introduced here, not the value-taking closure form;
  see "Erased dispatch no longer routes through a shared `variant_dispatch`"
  under Unreleased, which recovers 662,740 of it.

  Two changes were tried and reverted on evidence: **`std.memory.pack_bits`**
  (+12-15% on all 18 `bench_pack_bools_*` — ARM has no mask-move instruction;
  the old comment blamed x86 `pmovmskb`, which is no longer what it is), and
  `BitmapView.load_bytes`' alignment, which turned out to be drift.

  Four `@__parameter` remain, in `views._reduce_dispatch`. They feed
  `max.algorithm.reduction._reduce_generator_wrapper`, which has exactly one
  definition upstream and it is comptime-only — there is no value overload to
  migrate to.

- **`views.mojo` lost 95 lines** and its raw pointer arithmetic: the ten
  hand-built `Coord` closures collapsed to one, and the byte-level bitmap
  applies dropped 15 `unsafe_load` chains and 13 private-field reach-ins in
  favour of `BitmapView.load_bytes` and a new `BitmapView.offset()`.

- **Upgraded to Mojo 1.1.0.dev2026081605 / MAX 26.6.0.dev2026081605**, from the
  `1.0.0b3.dev2026072406` nightly, via stable 1.0.0. The tree is warning-clean:
  `mojo precompile marrow` and `build_python` both report 0 errors and
  0 warnings.

  Measured against the pre-migration toolchain: the binary-size gate moved
  −0.57% (`query_dynvalue`), −0.45% (`query_sort`) and +0.39% (`query_arith`)
  of `__text`, and 132 benchmarks moved a median of −0.3% (mean +0.0%). Every
  delta above 10% was re-measured and has run-to-run spread on the *unchanged*
  baseline wider than the delta itself.

  The breaking half of the 1.0 stabilization push:

  - `DeviceContext`, `DeviceBuffer` and `HostBuffer` moved from `std.gpu.host`
    to the **`max.gpu.host`** Mojo package, and `sync_parallelize`,
    `elementwise` and `_reduce_generator_wrapper` from `std.algorithm.*` to
    **`max.algorithm.*`**. `get_gpu_target` and `vectorize` stayed in `std`.
    MAX is now a load-bearing *import* dependency, not just a codegen one.
  - A method's `self` must have type `Self`. The 37 methods that spelled a
    custom `self` type to select the mutable or immutable instantiation of
    `Buffer` / `Bitmap` / `BufferView` / `BitmapView` now use a trailing
    `where Self.mut` (or `where not Self.mut`) clause instead, and the two
    `__eq__` pairs take an explicit `[m: Bool]` parameter.
  - A `where Self.mut` clause constrains the instantiation but does **not**
    refine `Self.mut` to `True` inside the body (upstream MOCO-4220), so a
    write through an origin-typed field still sees a symbolic origin. Each
    write site now casts with `unsafe_mut_cast[True]()`, the same workaround
    the standard library uses in `Span.fill`. `Buffer.resize` and
    `Bitmap.resize` swap whole buffer *values* rather than writing through a
    pointer, which the cast cannot express; they keep a custom `self` type
    behind the temporary `@__allow_legacy_custom_self_type` escape hatch, as
    `Span.unsafe_swap_elements` does upstream.
  - Static `String.write(...)` was removed; all 15 call sites use the
    equivalent `String(...)` constructor.
  - A list expression now builds an `Array`, not a `List`, so the nine
    unannotated list literals that fed a `List` parameter are now spelled
    `var x: List[T] = [...]`. `Array` also lost `ImplicitlyCopyable`, so the
    bloom filter's comptime salt table is read at comptime rather than
    materialized per call.

  The 1.1 nightly then *removes* what 1.0 deprecated, so the rest followed:

  - `ImplicitlyDeletable` → `Deinitable`, `InlineArray` → `Array`,
    `as_immutable()` → `as_imm()`, `std.builtin.type_aliases` → the prelude,
    `std.ffi._CPointer` → `OptionalPointer`.
  - `sort` moved to unified closures: the comparator is a runtime argument
    with a capture list, not a comptime function parameter.
  - `alloc` → `unsafe_alloc`, which is not re-exported from `std.memory`, so
    six files import it from `std.memory.alloc`.
  - `UnsafePointer` → `Pointer`, `@parameter` → `@__parameter`, `__del__` →
    `__deinit__`, and the raw pointer operations gained their `unsafe_` prefix
    and keyword offset: `ptr[i]` → `ptr[unsafe_offset=i]`, `a + b` →
    `a.unsafe_offset(b)`, `bitcast`/`free`/`load`/`store`/`gather`/
    `take_pointee` → `unsafe_*`. These were applied from the compiler's own
    diagnostics — the underline in each message gives the exact extent of the
    expression — so marrow's own same-named methods on `BufferView`, `Rle` and
    `page` were left untouched.
  - Implicit variable declarations now need `var`, walrus bindings included,
    and `Schema`'s `deinit take` move constructor is now `deinit move`.
  - `Scalar[DType.bool](someBool)` no longer compiles: since 1.0 any integer
    scalar constructs from an `Intable`, `Bool` is `Intable`, and that
    overload rejects `bool` as non-integral.

- **`Partition.row_indices` is a plain `Int32Array`.** It had been an
  `Optional[Int32Array]` defaulting to `None`, encoding "all rows in order" for
  the unpartitioned case — but `NoPartition` was removed, so the only
  construction site (`RadixPartitioner.partition`) always supplies the array and
  `map_partitions` unwrapped it unconditionally. The `Optional`, its `None`
  default and the `if` guard in `original_row` are gone; no behaviour or API
  change.

### Fixes

- **A scalar taken from a `map`, `large_list` or `fixed_size_list` array
  reported the wrong type.** `ListScalar` backs all four list-shaped layouts and
  rebuilt its type as `list_(child.dtype())`, which reports the *shape* and not
  the type: a `MapArray` element answered `list<struct<key, value>>`, a
  `LargeListArray` element answered `list<…>`, and a `FixedSizeListArray`
  element lost its size.

  The fix is a `_dtype: DynType` field carrying the array's own type, set by
  `ListLikeArray.__getitem__` and `FixedSizeListArray.__getitem__` — one field
  for all three cases. A 1-byte kind tag rebuilding `list_`/`large_list_`/
  `map_` from the child was considered and rejected: it cannot carry
  `keysSorted` or the entries field name. Regression tests:
  `test_list_scalar_type_from_{list,large_list,fixed_size_list,map}_array`.

  This is the second and last half of `map` support (the `cast` arm below is the
  first). It costs **+2,820 bytes of `__text` (+0.136%) on
  `query_streaming_agg_fused`** — +2,756 on `query_streaming` and
  `query_streaming_agg`, 0 on `query_join` — which is why it landed together
  with the binary-size re-baseline recorded under Refactors. That reproduces the
  +0.137% measured when the same change was written and reverted on 2026-08-16.

- **`cast` had no map arm**, so `map<string, int64> → map<string, int32>` raised
  "unsupported cast". It needed no kernel of its own: a map is physically a list
  whose single child is the non-nullable `entries` struct, so `ListCast` casts
  that struct and `StructCast` casts the fields. Only the *target child type*
  had to be read differently — from `entries_field()` rather than
  `value_type()`. No binary-size cost.


- **A bit-packed mask was read byte-wise, so `mask1 & mask2` could return wrong
  rows.** `BitmapView.load[DType.bool, W]` bitcasts to `Scalar[DType.bool]` —
  which is a *byte* — so it walked a bit-packed validity/data mask one byte per
  element and reported true for any non-zero byte. The mask `[F, F, T]` is the
  byte `0b100`, which reads as `True` at element 0. Five fused bool lanes used
  it: `NullPredicate`, `StringToBool`, `StringPredicate`, `IsIn` and
  `ListContains`. Wrong answers appeared whenever a bool binary had **two**
  materialized-stage operands — `s.startswith(x) & s.endswith(y)`,
  `a.isin(…) & b.isin(…)`, `x.isnull() & y.isnull()`. Numeric breakers were
  never affected; they read a plain buffer, never a bitmap.

  The fix is `mask[W]`, the bit-wise reader that already sat directly above
  `load` in the same struct ("expand W consecutive bits starting at logical
  index"). `BitmapView.load`'s docstring now warns that the `bool` instantiation
  is almost never what a caller wants.

  **This predates the A1 refactor below** — the same five lanes spelled it
  `ctx.get_ref[BoolArray](i).values().load[DType.bool, W](idx)` beforehand, and
  three of the four regression tests added here also fail when checked out
  against the pre-A1 tree. A1 changed which expressions happened to hit it,
  which is how it surfaced. Regression tests: `test_or_over_two_bool_breakers`,
  `test_and_over_two_isin_breakers`,
  `test_and_over_two_breakers_of_different_types`,
  `test_and_over_breaker_and_fused_unary`, and the numeric control
  `test_add_over_two_numeric_breakers`.

### Fixes

- **A string predicate pruned nothing in the fused lane, so an AOT plan decoded
  every row group.** `prune` and `bound_column` were defined on `NumericColumn`
  alone. `StringColumn` inherited `Value`'s conservative defaults, which meant a
  fused `s > "z"` could not skip a row group whose `s` maxes out at `"p"`, and a
  string column could not be a join key. The runtime lane had neither problem —
  its `prune` keys on `_tag == "column"` regardless of dtype — so the two lanes
  disagreed about the same predicate. `Interval.compare` has ordered strings the
  whole time; only the overrides were missing. Sound, therefore silent: nothing
  failed, the scan just did more work. Pinned by
  `test_string_predicate_prunes_in_both_lanes`, which asserts both lanes.

### Fixes

- **FU-7a: `coalesce`, `nullif` and `case_when` ran their kernel twice per fused
  pass.** 1M rows, `coalesce(a, b) + 1`: **150.72 ms → 81.31 ms, 1.85x.** The
  driver asked a node for its `state` and, separately, for its `validity`, and
  for a conditional breaker both calls ran the whole selection kernel. The same
  shape cost `StringToNum` a second parse.

  Nodes now answer `state_validity(batch, state)`, and a breaker reads the
  bitmap straight off the array it already materialized.

  **The first version of this fix measured no gain — 149.78 ms, inside the noise
  — and that measurement is what found the real defect.** Hooking only the
  breaker leaves the chain stopping at the first composite above it: a fused
  parent's default `state_validity` falls back to `validity(batch)`, which
  recurses into the child's expensive `validity` and re-runs the kernel anyway.
  Validity has to ride the *same state tree* the lane does, so the four `Pair`
  binaries and the eight delegating unaries propagate it as well. Pinned by
  `bench_fu7_coalesce_fused_1m`.

### Refactors

- **Q4.1: `Grouping` names the `(gids, num_groups)` pair.** The two were threaded
  separately through ~22 signatures across `groupby`, `aggregate`, `distinct`,
  `expr/aggregates` and `expr/execution`. They are one fact — `ids[i]` is row
  `i`'s group and `num_groups` sizes the accumulator those ids scatter into —
  and passing them apart let a caller size an accumulator from one grouping and
  index it with another's ids. That is an out-of-bounds scatter rather than a
  type error, and silent whenever the mismatched count happened to be larger.

  `struct Grouping { ids, num_groups }` lives in `kernels/core.mojo`: neutral
  ground both `aggregate` and `distinct` can import without a cycle, since
  `groupby` already imports `aggregate`. Three *function types* carried the pair
  as well — `def(Int, Int32Array, DynArray, Int)` and two
  `def(Int32Array, DynArray, Int)` — so a callback with a mismatched arity
  type-checked; they name `Grouping` now too. Sibling of the `JoinIndex` half of
  the same card.

  Size cost worth watching: `query_streaming_agg_fused` +0.449% against the 0.5%
  ceiling (from +0.416%), and `query_streaming_agg` +0.335%. The aggregate
  binaries are where a `Grouping` parameter lands, and the headroom on that gate
  is nearly gone.


- **Q4.3: the Arrow → Parquet physical-type mapping is stated once instead of
  three times.** `writer._encode_values`, `writer._bloom_hashes` and
  `statistics.decode` each carried a hand-written dtype ladder — 25, 25 and 22
  arms — restating the same relation: narrow ints widen to INT32, date32 and
  time32 and decimal32 ride on INT32, time64/timestamp/decimal64 on INT64,
  decimal128/256 on a big-endian FIXED_LEN_BYTE_ARRAY. Three copies, free to
  drift, and `marrow/parquet/` used the `dispatch_*` family nowhere at all.

  `schema.mojo` now names the mapping (`physical_type`, `is_wide_decimal`,
  `flba_width`) and each ladder collapses to five arms over
  `DynType.dispatch_primitive` / `dispatch_binarylike`, which resolve a runtime
  dtype to its comptime type. The statistics decoder no longer names
  `Time32Scalar`, `Decimal64Scalar` and the rest one at a time either: the
  dispatch witness carries the unit or precision/scale, so one
  `PrimitiveScalar[T](value, witness)` retags them all.

  `has_plain_physical` guards the widening: `dispatch_primitive` covers every
  fixed-width dtype including `date64`, `duration` and the intervals, which this
  layer has never written or read, and turning a ladder into a dispatch must not
  silently start accepting them. The reader's 28-arm comptime-gated ladder is
  deliberate and untouched. Net −54 lines; all four size gates unchanged.


- **A5: pruning is a kernel family, not a match on a display string.** Both
  lanes selected their interval rule by comparing `Kernel.name` against
  hand-maintained literals — `values.mojo` against `"and_"`/`"or_"` and five
  comparison names, `dynamic.mojo` against its own set. `Kernel.name` is
  documented as "for display and diagnostics, **never dispatch**", and the
  consequence of ignoring that was invisible: renaming a kernel dropped through
  to `Interval.unknown()`, which is *sound*, so pruning switched itself off with
  no error and no failing test.

  `marrow/kernels/interval.mojo` now holds the interval reading of each operator
  — `LtInterval`, `AndInterval`, and so on, each a one-line naming of an
  `Interval` method, exactly as `LtKernel.core` is a one-line naming of `a < b`
  over SIMD. The expression node pairs the two (`NumericCompare[LtKernel,
  LtInterval, L, R]`), which is where the codebase already decided this belongs:
  `NumericCompareKernel` used to carry a `comptime StringKernel` and it was
  removed because "which family `a < b` means is a question about the operands,
  and it belongs to whoever is interpreting the operator, not to the SIMD
  kernel". Both string ladders are deleted, and the runtime lane matches its tag
  against the *kernel's* own `name`, so rule and spelling cannot drift apart.

  `PruneBound` moves to that module as `Interval` — the interval domain is a
  value concept, and `kernels/` must not take an inbound edge from `expr/`.
  `expr/pruning.mojo` keeps `PruneStats`, the plan-level half.

- **Aggregate parity across the two lanes, the last A5 axis.** An aggregate is
  not a `Value`, so `assert_parity` cannot reach it: `col("x").sum()` on a
  `DynValue` yields a `DynAgg` (a group-by spec) while `col("x", int64).sum()`
  on a fused node yields a scalar `Reduction`. They converge only at `AggExpr`,
  so parity is observed through a plan. `sum`, `mean`, `min`, `max`, `product`,
  `count` and `count_distinct` now run through a keyless plan in each lane —
  keyless so there is one output row and the comparison cannot be confused by
  group order, which follows the key hash — plus one grouped `sum`, since the
  grouped path is a separate implementation from the whole-table fold. All
  passed on arrival and were mutation-checked.

- **Value parity for the 25 ops that had no cross-lane assertion.** `<= >= !=
  ** ^`, `neg`/`abs`/`sign`/`floor`/`ceil`/`round`, `sqrt`/`exp`/`ln`, the seven
  string maps, `length`, and `startswith`/`endswith`/`contains` are now asserted
  to compute the same answer in both lanes, over batches chosen so the ops are
  distinguishable from each other (signed fractional floats for the rounding
  family, whitespace and mixed case for the string maps). All 25 passed on
  arrival — this axis found no divergence, unlike naming and pruning. The
  assertions were verified to bite by mutation: swapping the fused side of `ln`
  for `exp` and `rstrip` for `lstrip` failed exactly those two cases.

  Aggregates stay uncovered on purpose. `DynValue.sum()` returns a `DynAgg`, a
  group-by spec, while the fused `Sum(col)` is a scalar `Reduction`, so their
  parity is a plan-level comparison rather than two expressions over one batch.

- **The two lanes named the same operator differently, and now cannot.** Eight
  pairs had drifted: `mod`/`modulo`, `pow_`/`power`, `neg`/`negate`,
  `abs_`/`abs`, `and_`/`and`, `or_`/`or`, `not_`/`not`, `log`/`ln` — so
  `render()` printed a different plan per lane for the same expression. The
  kernels adopt the spelling Arrow uses (which is what the runtime lane already
  had); the trailing underscores were Mojo identifier artifacts leaking into
  display strings, and a `comptime name: String` need not match its struct's
  identifier. The 47 runtime factories now read `XKernel.name` instead of
  repeating a literal, so there is one source of truth.

  `test_op_names_agree_across_lanes` enumerates the op set and reads every
  expected name off the kernel — it never spells one — so adding an op to one
  lane, or renaming a kernel, fails loudly.


- **`BitmapView`'s bit-addressed accessors are now the default pair; the
  byte-addressed ones say so in their names.** The struct had six accessors
  split across two families that a call site could not tell apart: `test`,
  `mask[W]`, `load_bits[T]` and `store[W]` are bit-addressed and apply
  `_offset`; `load[T, W]` and `store[T, W]` were byte-addressed and silently
  ignored it. Two traps followed. `store` was overloaded on *whether a DType was
  passed*, and `load` had no bit-wise overload at all — so the pairing that
  looks matched, `store[W]` to write and `load[...]` to read, was write-bits /
  read-bytes. That is exactly how the fused bool lanes came to read a mask a
  byte at a time (see Fixes). And on a *sliced* bitmap the offset divergence is
  wrong on its own, independent of the bit/byte confusion.

  `mask[W]` is now `load[W]`, mirroring `BufferView.load[W]` — both take a
  logical element index and return W elements — and pairing with the existing
  bit-wise `store[W]`. The byte-addressed forms become `load_bytes` /
  `store_bytes`, under a comment marking where the two families divide. `store`
  is no longer overloaded, and `load[DType.bool, W]` is no longer expressible.

  No caller outside `views.mojo` used the byte forms: the remaining ones are its
  own bitwise and/or/xor kernels doing deliberate whole-byte arithmetic with an
  explicitly computed `byte_idx`. Behaviour unchanged; all four size gates
  unchanged.


- **A1: typed per-node `State` for the fused expression lane — `a + 1` over 1M
  rows goes from 2.04 ms to 70.9 µs.** Every fused node now implements a
  two-method protocol: a comptime `State`, `state(batch)` which resolves the
  whole subtree once per pass, and `lane(state, idx)` which reads nothing else.
  A column leaf's `State` is its typed column, a literal's is nothing, a unary
  node's is its operand's, a binary node's is `Pair[L.State, R.State]`, and a
  pipeline breaker's *is* its materialized stage.

  The old lane re-resolved everything on every SIMD chunk — a schema lookup by
  name, a `Variant` unwrap and a `BufferView` reconstruction, 250,000 times over
  a million rows to read a column that never moves. Hoisting that into `state()`
  lands within 2.4% of a hand-written loop with the view resolved once
  (69.2 µs), a **28.8x** win, and cost no longer scales with column-leaf count:
  `a + a` now costs 70.4 µs, the same as `a + 1`, where it used to cost exactly
  double.

  The `Context` of positionally-addressed slots goes with it, and with it the
  invariant that a `prepare` walk and a `core` walk had to visit breakers in the
  same DFS order for a bare integer `slot` to match reads to writes. Six methods
  collapse to two: `Context.get`/`get_ref`/`append`/`size`, `prepare`, and the
  `mut slot` threading are all deleted, along with ~84 hand-written recursion
  bodies. `execute(batch, ctx)` is gone; `execute(batch)` is `materialize(batch)`.

  A trait default returning `Self.State` cannot be written — the bound would have
  to be `ImplicitlyCopyable`, which array states deliberately are not — so every
  conformer had to land in one commit. The three fused drivers are free functions
  rather than trait defaults for a related reason, recorded at their definition.

- **The `Breaker` trait is deleted.** It existed to answer one question —
  does this node materialize into a `Context` slot instead of running the fused
  loop? — asked by `conforms_to(Self, Breaker)` in `Value.execute` and
  `Value.prepare`. With per-node `State` there is no slot and no pre-pass: a
  breaker is simply a node whose `state()` does the work and whose `lane()` is a
  load or a splat, which its own `State` declaration already says. The marker
  drove no dispatch once A1 landed, so its 16 conformances went with it.

  The `Context` positional-slot invariant — "the single most dangerous thing in
  `marrow/expr`", where `prepare` and `vectorwise` had to walk the tree in
  identical DFS order by hand across 15 and 29 sites with nothing enforcing it —
  is gone with the mechanism rather than mitigated. Its worst failure mode
  returned the wrong column silently. `DateTrunc` was latently in another of its
  three modes, unreachable only because no temporal breaker existed to sit under
  it.

### Docs

- **`CountAgg`'s docstring claimed to be the grouped `count` for numeric columns
  too; it is not.** `CountValid.resolve` (`marrow/exprold/aggregates.mojo`) hands
  numeric columns `NumericAgg[CountKernel, V]`, the typed `AggState` fold —
  `CountAgg` serves non-numeric columns on that lane, plus the AOT lane's
  `K.Grouped`. Measured before deciding whether to converge them (1M rows,
  g100k, both implementations in one binary): nullable input, `AggState`
  1.7159 ms (sd 0.0702) vs `CountAgg` 8.3555 ms (sd 0.2331) — `CountAgg` pays an
  erased `variant_dispatch` `is_valid` check per row; null-free input, `AggState`
  1.4124 ms (sd 0.0098) vs `CountAgg` 1.3710 ms (sd 0.0117), where `CountAgg`'s
  `has_null` guard skips validity entirely. The nullable gap is decisive, so the
  split stays — the docstring now says so instead of claiming a unification
  that never happened. A test pins the two implementations to agree on the same
  input (`test_grouped_count_implementations_agree_on_nulls`).

- **One backlog, `docs/backlog.md`, replacing seven task files.** A five-agent audit
  verified every task ID against the code and found **18 wrong statuses**: eight marked
  open that were done (Q2.1, L3, Q1.2/Q1.3, Q5.3, Q2.4, FU-1–FU-4, T2.4, Q3.4), four
  marked done that were not (Q4.5's test, T3.4, T3.5, Q6.1's baseline), and six whose
  premise the two-lane split had destroyed. A consolidation pass four days earlier had
  not caught any of it, because nothing external was contradicting the claims — CI has
  not run since 2026-05-11, and `test.yml` still invokes a `test_parallel` task deleted
  in `2aa1954`.

  The new file is sequenced as waves: nine silent-wrong-answer defects, then CI and the
  docs build, then the ClickBench milestone, then M2/M3 enablers. Arrow-parity gaps are
  listed once as deferred rather than scheduled. §0 collects the standing constraints
  and measurement traps that were previously scattered across the files it replaces.

  Deleted as superseded: the seven `tasks-*.md` files, plus `dynamic-dispatch-design.md`
  (specifies fn-pointer vtables; the tree uses inline `Variant`), `aot-query-compilation.md`
  (its central `comptime if op == ADD` construct is now a *measured* anti-pattern here),
  `unified-plan-hierarchy.md` (its mechanism is what the two-lane split disproved),
  `expr-unification-plan.md`, `ibis-fusion-design.md`, and `lane-shape-window-skeleton.mojo`
  (401 lines prototyping three rejected designs — a runnable artifact of a rejected design
  is a trap). `docs/architecture.md` is new and describes the shipped design.

- **Corrected claims that had drifted into being false.** CLAUDE.md listed "release
  callbacks in the C Data Interface are never invoked" as a Mojo limitation; they are
  implemented and invoked on four paths plus three PyCapsule destructors. Its layout
  coverage claimed `map` is implemented — true everywhere except IPC, where it round-trips
  in neither direction. README.md claimed the runtime lane "is what the Python bindings
  drive" (nothing in the expression layer is bound), that "all compute kernels" are exposed
  to Python (26 of roughly 90 are), and described the AOT lane in terms of `Table`,
  `Column[Tbl, name, T]` and `Project` — all deleted.

### Fixes

- **`__eq__` called logically-equal arrays unequal (B26).** It compared the
  bitmaps themselves -- presence against presence, then whole bitmap against
  whole bitmap -- so an all-valid array carrying a bitmap was unequal to one
  carrying none, and two slices whose logical validity matched were unequal
  whenever their offsets differed. Six array types shared the shape.

  Equality is a question about null *positions*, not about how the validity is
  stored: a missing bitmap means all-valid, which is a value rather than a
  distinguishing representation. New `_validity_equal` compares positions through
  offset-applied views.

  This reached further than equality. CLAUDE.md tells you to write
  `assert_true(result == expected)` rather than an element loop, and every kernel
  that intersects validity emits an array with a bitmap while `array([...])`
  emits one without -- so the recommended assertion was unreliable for exactly
  the outputs a kernel test wants to check.

- **Kernels shifted a sliced input's nulls (Q2.3).** Arithmetic, comparison and the
  string predicates took their values from `left.values()` -- offset-applied --
  but built the output bitmap from the raw `left.bitmap`. The result is always
  `offset=0`, so on a sliced input the data came from the slice and the validity
  came from the parent: every null landed `offset` positions from where it
  belonged. `Bitmap.intersect`'s one-sided path was worse, returning the lone
  operand whole rather than offsetting it at all.

  New `Bitmap.intersect_views` combines two `Optional[BitmapView]` and resolves
  the one-sided cases through `to_owned()`; both overloads now say which is for
  what. `BinaryLikeArray` gained the `validity()` accessor every other array type
  already had -- its absence is why the string kernels reached for `.bitmap`.

- **A hung Mojo compile no longer blocks the test run forever.** `run_with_progress`
  called `proc.communicate()` with no timeout, so a process that stops making
  progress and never exits is indistinguishable from a slow compile -- and the
  harness waits. `MojoRunner.collect` already recovers from a compiler *crash* by
  splitting the selection, because a crash produces a signal; a hang produces
  none. New `--mojo-timeout SECONDS` (default 1800, 0 disables) kills the process
  and reports it as an ordinary failure, with a message pointing at `ps -o
  etime,time` to distinguish a hang from slowness. Verified against the known
  deadlock in `test_join.mojo`, which previously ran 7 hours on 10.4s of CPU.

- **`DynType.is_fixed_size()` removed.** It answered `is_bool() or is_primitive()`,
  so it covered neither `fixed_size_binary` nor `fixed_size_list` -- the two dtypes
  its name promises -- and it had no production caller. Arrow C++ spells the real
  predicate `is_fixed_width` = `is_primitive || is_dictionary ||
  is_fixed_size_binary` (`type_traits.h:1400`), which marrow's matched in no
  respect. Deleted rather than widened: both places that want it already write
  `is_bool() or is_primitive()` inline. `dtypes.mojo` records Arrow's definition
  for whoever needs it next.

- **B22 was a test bug, not a device-transfer defect.** `Buffer.unsafe_set[T]`
  infers `T` from its value, so a bare literal widens and the store strides by the
  wider type, while `unsafe_get[T]` defaults to `uint8`. The write landed eight
  bytes from the read. `to_device`/`to_cpu` are correct; `unsafe_set` now
  documents the asymmetry.

- **`is_primitive()` said bool, and `byte_width()` aborted on it.** `byte_width()`
  guards on `is_primitive()` then dispatches over `variant_dispatch[PrimitiveType]`;
  `BoolType` does not conform -- correctly, bool is bit-packed -- so bool aborted the
  process rather than answering. All six callers already peeled bool off before
  reaching `is_primitive()`, because it exists to guard exactly that dispatch, so the
  predicate was the defect, not `byte_width()`. `is_primitive()` now narrows to the
  set conforming to `PrimitiveType`; `is_fixed_size()` re-adds bool, which is the
  Arrow-spec notion. This diverges from PyArrow and Arrow C++, which both count bool
  as primitive -- deliberate, since neither has a `PrimitiveType` trait to stay
  consistent with.

- **A MARK join built a `StructArray` whose dtype declared more fields than it had
  columns.** "Does this kind emit the right side's columns?" was answered inline at
  four places with three different memberships: `output_dtype` said MARK does,
  `_assemble` said it does not, `relations.Join.schema` agreed with the first, and
  `tabular.join` re-parsed strings. The first two disagreeing is what made the
  result malformed, and nothing checked.

  `JoinKind` now answers it once, as `emits_right_columns()`, with
  `emits_unmatched_left()`/`_right()` collapsing the LEFT/FULL and RIGHT/FULL pairs
  and `JoinKind.parse` owning the `how=` spelling. Both references do it this way --
  polars has `JoinType::is_semi_anti()`, ClickHouse a set of `constexpr isLeft(kind)`
  free functions -- and neither answers it at a use site. marrow follows polars' flat
  model where SEMI and ANTI are kinds; ClickHouse files them under strictness instead.

  `CROSS`, `MARK` and `SINGLE` have constants and no implementation. They used to fall
  through to the outer-join arm; `hash_join` now rejects them by name.

- **A GPU context lost its device at six places, one of them reachable from Python.**
  `ma.compute.sum(arr, ctx)` and its siblings passed `ctx.resolved_num_threads()`
  into `Aggregation.whole`, so a Python-supplied GPU context arrived as a bare
  worker count and the aggregate ran on the CPU. The same shape existed at five
  internal sites that rebuilt `ExecContext.parallel(n)` -- a *factory*, which sets
  `device=None`, used where a *modification* was meant. `HashJoin` had already
  found and fixed this once; the fix had not reached `GroupBy`, `Aggregation.whole`,
  or `RecordBatch.join`/`group_by`/`sort_by`.

  `ExecContext.with_threads(n)` is the missing operation -- change the worker count,
  keep the device -- and no internal site rebuilds a context any more. The three
  Python bindings still construct one from an `Int`, which is correct there: an
  `Int` really is all the information a Python caller sends.

### Features

- **`DynType.num_buffers()`, and `ArrayData` now validates its own layout.**
  How many data buffers a type owns was re-derived by every codec; `ipc.mojo`'s
  buffer walk was a two-branch ladder asking exactly that, and it gave `map`
  **zero** buffers, silently shifting every buffer read after it. That walk is
  now `for _ in range(dtype.num_buffers())`.

  Counts data buffers only -- marrow's `ArrayData` carries `bitmap` as its own
  field, unlike Arrow C++ and arrow-rs where `buffers[0]` is validity -- so there
  is no bitmap buffer kind to model and no per-buffer `BufferSpec`, just a count.

  `ArrayData` replaces `@fieldwise_init` with a constructor that asserts the
  count, so none of its ~39 construction sites can bypass the invariant, plus a
  raising `validate()` used by the C Data Interface importer, where the buffers
  are a foreign producer's memory and a wrong count reads past the end of
  somebody else's allocation.

  Costs **+1.27%** of the AOT size gate (`query_streaming` __text 1,309,032 ->
  1,325,672), accepted deliberately. Measured split: +8,832 for the explicit
  constructor alone -- `DynType` is dispatched over a 37-member variant, and
  per-arm work there costs ~8 KB, the lever B12 found -- and +7,808 for the
  check, which `-O3` does not eliminate and `@no_inline` recovers only 128 bytes
  of. Both references decline this trade (arrow-rs pairs `try_new` with
  `new_unchecked`; Arrow C++ leaves `ValidateFull` opt-in); marrow takes the
  safety. `num_buffers()`, `validate()` and the IPC rewrite on their own cost 0.

- **`StringPredicateKernel.apply_scalar`**, a scalar-pattern entry point for the eleven
  string predicate kernels (`StartsWith`/`EndsWith`/`Contains`/`StrEq`/`StrNe`/`Like`/
  `ILike`/the ordering compares), and `StringPredicate.prepare` in the expression layer
  now calls it whenever the right operand is a constant (`Self.R.OutShape == 0`).
  Previously `s LIKE 'foo%'` materialized the literal into an n-row array and handed it
  to the array x array kernel, which for LIKE/ILIKE recompiled the same `LikePattern`
  on every row; now it evaluates the pattern once. `StrEq`/`StrNe`/ordering share the
  same fix through the trait's defaulted `apply_scalar` body. The branch resolves at
  elaboration off `OutShape`, so it adds no new node and no runtime check.

- **`ExecContext.worth_parallel(n, min_parallel_size)`**, the predicate four kernels
  had each written by hand. It differs from `wants_parallel` on exactly one input,
  `parallel(N)` below the threshold, and that difference is the point: a forced
  thread count is an *instruction* to `stripe`, where going parallel costs one
  dispatch, but only a *budget* to a chooser between algorithms, where it costs
  radix partitioning and N hash tables. `parallel(4)` on 1,000 rows must still take
  the serial path. It also answers False on a GPU context, which none of the four
  copies did -- they asked `resolved_num_threads()`, which knows nothing about the
  device. The threshold is a required argument: the four callers measured different
  crossovers (60k group-by, 100k join, 200k distinct) and none is `stripe`'s 32768.

  One behaviour change, in the intended direction: `auto()` now stays auto down the
  call chain instead of being resolved to a forced count and back. Previously
  `whole()` destructured `auto()` into `parallel(num_physical_cores())`, which
  bypasses every downstream size threshold; now a small input under `auto()` can
  correctly choose a serial hash.

### Refactors

- **Dead constants, imports and scaffolding removed (Q7.5).** Five `JOIN_ALGO_*`
  hints, `JOIN_ASOF`, five `RAPID_SECRET*`, `comptime lo32`, four unused imports,
  and `trait Partitioner` / `NoPartition` -- the latter had no caller and would
  have crashed `map_partitions`, which unwraps `row_indices` unconditionally.
  Two stale banners went with them: one titled a section of "temporal reinterpret
  helpers" that documented a design the single `is_primitive()` dispatch arm
  replaced, and one describing a `sync_parallelize` loop that is now `ctx.stripe`.
  No size change -- the dead code was already being eliminated, so this buys
  readability, not bytes.

- **`ExecutionContext` is now `ExecContext`, and lives in `marrow/execution.mojo`.**
  It was filed under `kernels/`, which made it the tree's only `core -> kernels`
  import edge — `views.apply` and `tabular` both need it, so core reached *up* into
  a leaf package to get it. It imports nothing from marrow and is a pure
  thread-count/device/`stripe` policy object, so the move is free: `query_streaming`
  `__text` is byte-identical at 1,309,024, and all 1,942 tests are unchanged. The
  new name matches `arrow::compute::ExecContext`, which is the same object in the
  same position. `marrow.kernels` still re-exports it. Python-visible name changes
  with it: `ma.ExecutionContext` -> `ma.ExecContext`.

- **The expression layer is two lanes that share no node types.** A fused node used to
  carry a second `_erased` body selected by a hand-propagated `comptime IsErased`, and
  the value box claimed `NumericValue`/`BoolValue`/`StringValue`/`TemporalValue` so those
  nodes would accept it as an operand. That conformance was **unsound** — those traits
  promise a comptime `OutType: NumericType` and a `vectorwise` lane, and the box supplied
  a placeholder `native = DType.bool` and a stub returning zero. The compiler reported it
  as `attempt to resolve a recursive reference to declaration 'DynValue.__gt__'`, which is
  what forced the fluent surface into a `NumericOps` sub-trait. Now: `marrow.exprold.values`
  is the AOT lane, every operand bound on its family trait; `marrow.exprold.dynamic` is the
  runtime lane, where `DynValue` is a tag, its children and an optional payload; and
  `BoxedValue` (`marrow.exprold.relations`) is the one box both erase into, so each
  relational operator still compiles exactly once. `IsErased`, all 14 `_erased` methods,
  and `NumericOps` are deleted, and `DynType` drops the 8 family-trait conformances that
  told the same lie about `native`. `values.mojo` loses 595 lines.

  **API change:** `marrow.exprold.DynValue` now names the *runtime expression node*, not the
  box. Plan APIs that took a `DynValue` — `filter`, `project`, `aggregate`, `sort`, `join`,
  and the `Processor`s — take a `BoxedValue`; both a fused node and a `DynValue` convert
  implicitly. `Value.is_deterministic` is removed (a default returning True, with no
  caller).

  **A runtime node's operation stays comptime.** `DynValue` carries a pointer to its
  evaluator, so `__gt__` names `_compare[GtKernel, StringGtKernel]` and a binary links
  exactly the kernels its expressions mention; the tag string that remains drives only
  `render`/`prune`/`name`. Written first as one `_eval` switch over ~70 tags, it cost
  **+1,807,168 bytes of `__text` (+45.7%)** on `query_dynvalue`, because every arm was
  reachable from every node — the whole win from deleting the old 41-tag interpreter.
  Binary size now, `__text` against HEAD rebuilt from source: `query_streaming` (AOT)
  1,303,028 -> 1,302,900; `query_dynvalue` (runtime) 3,956,596 -> 3,984,756 (+0.71%).

### Fixes

- **`coalesce`, `nullif` and the string comparisons aborted instead of computing.**
  `StringPredicate` and `ConditionalBinary` read a pipeline-breaker slot in `vectorwise`
  but had lost their `Breaker` conformance, so nothing ever filled it and every such
  expression tripped `index 0 is out of bounds` inside `Context.get`.

- **Runtime-lane expressions now match the fused lane.** `+` over string columns
  concatenates rather than raising; `/` and `**` are float64, so `5 / 2 == 2.5` and not
  `2`; `sqrt`/`exp`/`ln` accept an integer column by casting up rather than raising out of
  `dispatch_floating`; `length()` and `isin()` exist again; and `prune` evaluates columns,
  literals, comparisons and `and`/`or` against statistics instead of answering "unknown",
  which is what lets a Parquet scan skip row groups and pages for a predicate built at run
  time.

### Features

- **The Parquet scan streams row groups and projects into the read (T2.4).**
  `ParquetScanProcessor` used to decode the whole file into one `Table` on the first pull
  and slice it, so resident memory scaled with file size. It now opens the file once,
  fixes the pushdown plan once, and decodes **a bounded window of row groups at a time**
  (64 MB of decoded data), dropping the previous window; groups are handed out one at a
  time so morsels never straddle a row-group boundary, which changes no rows. The
  scan's schema is now also its **projection** — only its columns are read, so unselected
  column chunks are never decoded, and `parquet_scan(path, schema, morsel_size=...)`
  narrowing the schema is how a projection gets pushed down. Pruning was decoupled from
  column position to make that safe: statistics are indexed by file leaf, the predicate by
  the scan's schema, and a nested file or an unmatched name turns pruning off rather than
  misaligning the two. Net effect on throughput, measured against reading the file whole:
  **0.75x–0.93x** (i.e. faster) on every multi-row-group case, flat on single-group.

### Fixes

- **`DynType.is_list_like()` admits map, matching `ListLikeType`.** `MapType` conforms to
  `ListLikeType` and `dispatch_listlike` walks it, but the predicate guarding that dispatch
  returned False for map — so a kernel that guarded with `is_list_like()` rejected map
  columns its own typed leaf accepts. `rapidhash` was already writing
  `is_list_like() or is_map()` by hand to work around it; that workaround is gone, and
  `array_length` over a map now returns its entry count instead of raising.

- **Parquet reads no longer re-`dlopen` a codec library per call.** `ParquetFile.read`
  built a fresh `CompressionLibs` in each worker, and the first decompress for a codec
  opens its shared library. That was invisible while a scan issued one `read` for the
  whole file; once it issued one per row group it dominated — a Snappy 16-row-group scan
  ran **4.7x slower**. The handles are now pooled on the `ParquetFile` (shared ownership,
  so `read` stays a borrow — `_read_at` hands out spans whose origin is `self` and
  `ColumnReader` needs an immutable one) and reused across calls, one slot per worker.

### Refactors

- **`rapidhash` and `sort_indices` dispatched the same arm four times (−66 KB).** Both
  ladders carried separate `is_numeric`, `is_decimal128`, `is_decimal256` and
  `is_primitive` arms whose bodies were identical: `Decimal128Array` *is*
  `PrimitiveArray[Decimal128Type]` and `dispatch_primitive` already walks every decimal
  width, so three of the four were spellings of the fourth. `cast`'s `_on_native`
  likewise folds onto a new `DynType.dispatch_decimal`, completing the `dispatch_*`
  family. Measured on `__text`: **−65,856 to −66,176 bytes on all seven gates** that link
  the erased dispatch path, and **exactly 0** on the fused and typed gates, which never
  link it.

- **Every named kernel now conforms to `Kernel`.** Twenty-five kernels carried a
  `comptime name` without the conformance that gives the name meaning, so
  `Self.error`/`expect_same_length`/`expect_same_dtype` were unavailable to them and
  each spelled its own diagnostics. `Filter`'s `_require_len` was `expect_same_length`
  under another name, at nine call sites; `SortIndices.multi` prefixed its errors
  `sort:` while the kernel is named `sort_indices`. The aggregate family traits
  (`WideningOp`, `MinMaxOp`, `Aggregation`, `AggFunction`) descend from `Kernel` rather
  than re-declaring `comptime name` themselves, which is the drift the root trait
  exists to prevent.

- **Temporal and nested kernels dispatch through the `dispatch_*` family.**
  `TemporalExtractKernel` and the two `nested.mojo` kernels hand-wrote if/elif ladders
  over type families that `dispatch_temporal`/`dispatch_listlike` already cover. One had
  drifted: the temporal ladder omitted `DurationType`, so `date_trunc` and the
  extractors rejected duration columns their typed leaves accept.

- **`membership.mojo` and `conditional.mojo` are kernels, not free functions (Q3.1).**
  Neither had a struct, so `is_in` was five free functions (three of them byte-identical
  typed delegators) and the four conditional kernels each re-implemented the same
  validate / fill-a-selector / multiplex sequence with its own error vocabulary. Now
  `IsInKernel`, and `CaseWhenKernel`/`CoalesceKernel`/`NullifKernel`/`FillNullKernel` over a
  shared `Selection` engine that owns the candidates, the per-row branch selector, and the
  `concat`+`take` gather. Diagnostics come from `Kernel.error`/`expect_same_dtype`/
  `expect_same_length` instead of four hand-rolled phrasings, and `date_trunc` takes a
  `CalendarUnit` instead of a `String`, parsed at plan-build time by the frontends so an
  unsupported unit can no longer reach the kernel. Ten temporal free functions that
  forwarded to exactly one kernel each are gone.
- **`PageReader` reads one column chunk, not the file.** It took a whole-file span and
  seeked to absolute footer offsets; it now takes exactly the chunk's bytes and starts at
  0. `ColumnMetaData.byte_range()` (new) is the single place the "start at the dictionary
  page if there is one" rule lives. `ParquetFile._span()` is gone — the page index and
  bloom filter fetch their own recorded extents through `_read_at`/`_metadata_at`, and
  `page_index()`/`page_bounds()` now share one `_chunk_page_index` instead of decoding the
  same two Thrift structs twice. Nothing in the decode path asks for the whole file any
  more, which is the precondition for a streaming or remote `ByteSource`.

### Fixes

- **The two expression lanes agree on mixed-dtype arithmetic again (Q0.4).** The fused
  algebra promoted mixed operands (`promote[L, R]`) while the interpreted lane demanded
  identical dtypes, so `int64 + float64` executed in one lane and raised
  `add: dtype mismatch: int64 vs float64` in the other — and `.project()` surfaced it at
  *plan-build* time, since it probes each expression against a 0-row batch. `DynValue` now
  widens the narrower numeric operand before handing it to a kernel, by the same rule the
  fused lane uses at comptime (every float outranks every integer). It is done in the
  expression layer, beside the `_compare` that already picks a kernel family from operand
  dtypes — kernels stay array-in/array-out and strict, and `expect_same_dtype` still means
  what it says for `nullif` and `case_when`'s candidates. Costs +16,528 bytes on
  `query_dynvalue` and nothing on the fused gates; the promotion is written inline in each
  of `eval`'s twelve binary arms because folding them into one generic helper measured far
  larger (⚠️ BINSIZE note at the call sites). Note the byte figures previously quoted here
  came from stripped *file* size, which is quantized to 16 KB pages on Apple Silicon — see
  Q0.8; code-size deltas should be read from `size -m` → `Section __text`.
- **`precompile` no longer breaks every following `pytest` run.** It wrote its package
  artifact to `.test_runners/`, and Mojo puts a source file's own directory on the import
  search path — so a `marrow.mojoc` sitting next to the generated test driver *shadowed the
  whole `marrow/` source tree*. Every subsequent run resolved imports against the stale
  package: a case you had just added reported `module 'test_x' does not contain
  'test_your_new_case'` and the run failed in well under a second without compiling
  anything. The artifact now goes to `.precompile/`, which is on no include path. CLAUDE.md
  claimed `.test_runners/` was chosen "precisely to avoid this".
- **A named column no longer renders as `input(0)`.** `DynValue.write_to` printed
  `input(_kind_data)` for every `LOAD` node, but an *unresolved* named reference carries
  `_kind_data == 0` — so `col("x") < col("y")` displayed as `less(input(0), input(0))`,
  reporting positions neither column had yet. It now prints the name until `resolve_names`
  binds it, which also makes a rendered plan distinguish bound from unbound references.
- **Restored warning-clean.** `groupby.mojo` and `join.mojo` reached `take` implicitly
  through the package `__init__` in 5 places (deprecated); both now import it explicitly.

### Fixes

- **`test_cross_check_temporal_pyarrow` was silently broken by an over-applied rename.**
  The Q3.1 pass that turned `filter`/`take` into `Filter.apply`/`Take.apply` also rewrote
  the **PyArrow** reference calls in the cross-check — `pc.filter` became
  `pc.Filter.apply`, `pc.take` became `pc.Take.apply` — and moved a
  `.cast(pa.int64())` off the *result* onto the *filter mask*, where a boolean mask was
  required. The test compared marrow against an API PyArrow does not have, so it could
  only ever fail. Swept the tree for other casualties; this was the only one.

### Refactors

- **`arithmetic.mojo` + `compare.mojo` → `numeric.mojo`.** They were the same kind of
  thing: numeric-only, three-tier (`core`/`apply`/`dispatch`), both resolving runtime
  dtypes through `AnyDataType.dispatch_numeric`. What separates an `AddKernel` from an
  `LtKernel` is the `core` functor and the output layout. Public names are unchanged —
  `kernels/__init__.mojo` re-exports the same set. Tests stay split as `test_arithmetic`
  / `test_compare` on purpose: they cover different operations and smaller files keep
  failures legible. This is organisation, not deduplication — the identical `dispatch`
  bodies it puts side by side are tracked separately as Q0.6. Fused gate unchanged at
  1,307,624.
- **`equal_any` names the cross-family equality primitive once.** Hash-join row
  verification (a key row is an arbitrary schema) and `nullif` (defined for any dtype with
  an equality) both need equality *over an arbitrary dtype*, as opposed to interpreting a
  user's `==`. Both used to reach it through the numeric kernel's string branch; after
  that branch was removed they each grew their own copy. Now there is one.
- **Comparison is two kernel families, not one with a dtype branch.**
  `BinaryCompareKernel` is now `NumericCompareKernel` and is numeric-only. It used to
  carry `comptime StringKernel: StringPredicateKernel`, so every numeric comparison named
  a string counterpart it never used and its `dispatch` chose between two unrelated
  implementations at run time — SIMD over fixed-width lanes versus an elementwise walk
  over variable-width data. Which family `a < b` means is now decided by whoever
  interprets the operator (`DynValue._compare[N, S]`, plus an explicit route in `nullif`,
  the one kernel legitimately spanning both). The string kernels were already first-class
  `StringPredicateKernel` conformers; they were just unreachable except through the
  numeric kernel's associated type.

### Features

- **`in_memory_table(batch, morsel_size=…)`** — the morsel size was reachable only by
  constructing `InMemoryTable` directly, which is the sole reason several streaming tests
  built nodes by hand.
- **`AnyRelation.project` and `.aggregate` accept already-bound `AnyValue`s**, so a fused
  comptime plan uses the same plan-building API as an interpreted one instead of assembling
  `Project`/`Aggregate` nodes with a caller-written schema. `project` now takes
  `List[AnyValue]` only: `AnyValue` converts implicitly from `DynValue`, so a second
  `List[DynValue]` overload was shadowed at every list-literal call site — reachable only by
  spelling the conversion out, which is not an API. Column references bind when the value
  executes, exactly as `filter` already did; an unknown column still fails at plan-build
  time because the dtype probe executes the expression.

### Tests

- **`expr/tests/{test_plan,test_streaming,test_aggregates}.mojo` build their plans through
  the plan-building API.** The `benchmarks/binary_size/*.mojo` gates are now the only place
  that constructs plan nodes directly, and deliberately so: schema derivation probes each
  expression against a 0-row batch, and converting the gates measured **+16,528 bytes
  (+1.26 %)** on `query_streaming` stripped (1,307,624 -> 1,324,152). They were reverted and
  their docstring records why; tracked as Q0.5, whose fix is to answer a *fused* value's
  dtype statically instead of by execution.
- **`expr/tests/test_plan.mojo` builds its plans through the plan-building API** —
  `parquet_scan(...).filter(...).select(...)`/`.project(...)` — instead of constructing
  `AnyRelation(ParquetScan(...))` / `Filter(input=…)` / `Project(input=…, schema=…)` by hand.
  Two consequences beyond style:
  - **The five long-standing compiler crashes in this file are gone.** They were attributed
    to an upstream bug ("filtering with a comparison predicate under `TestSuite`"); they were
    caused by the hand-built plan nodes. The whole file now compiles as **one** unit and
    passes 21/21 in 105 s, where before the harness had to bisect to single cases and five
    still died (390 s). The same cap was believed to block Q1.2/Q1.3, L6 and Q4.5.
  - **It exposed a real lane divergence** (now tracked as Q0.4, and asserted by
    `test_project_mixed_dtype_arithmetic_raises`): the fused algebra promotes mixed operand
    widths, the interpreted lane raises `dtype mismatch`. The old test could not catch it —
    it declared `schema=[field("z", int64)]` for an `int64 + float64` expression it never
    evaluated, so it asserted the arithmetic it had itself written down, wrongly.

- **`Buffer.resize` no longer corrupts DEVICE memory.** It probed `Allocation._host` to
  decide between a pinned and a heap reallocation, so a DEVICE buffer took the heap arm
  and then `memcpy`'d through a pointer it does not have. `Allocation` now answers
  `is_host()` / `is_device()` itself and resize raises for device memory, which has to go
  through the device API.
- **The `binary_size` gate builds again.** `query_streaming_agg_fused.mojo` still used the
  removed `List.append[A](dtype)` spelling, so the whole gate errored out.
- **A compiler crash no longer takes down the whole test selection.** The compiler dies
  with a bug-report dump — no diagnostic — on units that elaborate too much, which is a
  function of unit size, not of the code being wrong: 17 cases in one driver crashed where
  2 of the same cases built fine. The harness now halves the selection on a crash and
  compiles each half, down to a single case, so a case that genuinely cannot be built
  reports the crash as *its own* failure instead of failing every case selected with it.
  Ordinary `error:` diagnostics never split — they would be identical in each half.
  `test_aggregates.mojo` went from 0/17 to 16/17, `test_plan.mojo` from 0/22 to 17/22.

### Refactors

- **The C-ABI double-free guard has a name.** `UnsafePointer(to=x.release).bitcast[UInt64]()[0]`
  — read as "has this been released?" and written as "mark it released" — was open-coded at
  eleven sites across `c_data.mojo`. `CArrowSchema`, `CArrowArray` and `CArrowArrayStream`
  now answer `is_released()` and `mark_released()`, so the one handshake the C Data
  Interface has for ownership transfer is stated once instead of re-derived per call site.
- **`filter` / `take` / `drop_null` have three free functions, not twenty-three.** Each
  typed array shape had its own one-line delegator forwarding to `Filter.apply` /
  `Take.apply`, so adding an array type meant editing six places. The kernels are the
  typed surface now (`Filter.apply`, `Filter.drop_null[T]`, `Take.apply`) and only the
  three erased `pc.*` entry points remain free. The verified-dead `_drop_null_bool` is
  deleted with them.
- **`execute` is a method on the plan, and `lit` is one verb.** Draining a plan was the
  only plan verb that was a free function — and it collided with `Value.execute`, the
  fused lane's per-node verb, so `execute` meant two different things depending on the
  argument. It is now `plan.execute(ctx)`. Likewise `slit("x")` is gone: `lit` overloads
  on the argument type, and a new `lit(3.5, float64)` overload makes fractional constants
  representable — the only spelling took an `Int` and silently truncated.
- **Validity has one owner.** `BitmapView.to_owned()` replaces the three ad-hoc "copy a
  view into an owned bitmap" idioms (a `difference` against a zero scratch, `v.union(v)`,
  and an identity SIMD functor) — and it goes through `Bitmap.extend`, so a byte-aligned
  view costs a `memcpy` instead of a pass over every bit. `Bitmap.unset_count()` replaces
  the `length - …count_set_bits()` incantation at eight sites, and
  `ArrayData.owned_validity()` replaces `_column_validity` / `_result_validity` /
  `_view_to_owned` / `_nulls_of` in the fused expression layer.
- **The five byte-level SIMD functors are private to `BitmapView`.** They were module-level
  free functions in `views.mojo` (`_and`, `_or`, `_xor`, `_and_not`, `_invert`) usable by
  anything; only the bitmap set operations ever wanted them.
- **The erased boxes expose their downcast instead of having it reached into.**
  `AnyDataType`, `AnyArray`, `AnyBuilder` and `AnyScalar` each had a private `_as[T]`
  with public one-liners over it, so generic callers went through the *variant field* —
  `data.dtype._v[Self.T]`, `slot[AnyArray]._v[A]`, `variant_dispatch_raises[...](t._v)`.
  It is now `as_type[T]()` on all four, and the last three private reach-ins are gone
  (`PrimitiveArray` reading its own dtype, `Context.get[A]`, and statistics comparison,
  which now uses `AnyDataType.dispatch_numeric`). `PrimitiveBuilder.append_nulls(n)`
  likewise replaces `nulls()` writing the builder's `_null_count` by hand.
- **No wildcard imports left in the core modules.** `arrays`, `scalars`, `builders`,
  `c_data` and `kernels/aggregate` pulled `from .dtypes import *`, which also re-exported
  whatever `dtypes` had imported — `variant_dispatch`, `DeviceContext` and the dtype
  vocabulary all arrived in three namespaces by accident, and a test was importing
  `StringType` from `kernels.aggregate`. Each is now an explicit list, so a module's
  dependencies are readable at its head and adding a name to `dtypes` cannot silently land
  somewhere else. CLAUDE.md's standing advice is updated to match.
- **String comparison has one implementation.** `LtKernel`/`LeKernel`/`GtKernel`/`GeKernel`/
  `EqKernel`/`NeKernel` each name their string counterpart as `comptime StringKernel`
  (`StringLtKernel`, … — new in `string.mojo` alongside the existing `StringEqKernel`), so
  `dispatch` routes `string`/`large_string` to the *same* `StringPredicateKernel` the fused
  expression layer uses. The parallel `str_predicate`/`apply_string` pair in `compare.mojo`
  is gone, and with it the second copy of the element-wise compare loop. `StringCompare` in
  `expr/values.mojo` collapsed into `StringPredicate` for the same reason: the two nodes
  differed only by which kernel they held.
- **`Kernel` owns the argument checks every kernel family was repeating.** `error`,
  `expect_same_length` and `expect_same_dtype` are defaults on the base trait, so the message
  a caller sees is one sentence written once instead of ten copies drifting apart — and the
  trait now constrains behaviour rather than being a name-only marker.
- **The legacy free functions in `compare.mojo`/`boolean.mojo` are gone.** `equal` existed
  three times (a `StringArray` overload that allocated a `String` per element, a
  `StructArray` one, and an erased one) beside `EqKernel`; struct row equality is now
  `EqKernel.apply(StructArray, StructArray)` — the shape the hash table verifies key rows
  with — and everything else goes through `EqKernel.dispatch`. The numeric-only `is_null`
  and the validity-dropping `select` are deleted; the interpreter reaches `IsNullKernel` and
  `case_when` instead, which is also what fixes the two defects below.

### Fixes

- **`is_null` works on every dtype from the interpreter.** `DynValue`'s `IS_NULL` called the
  numeric-only free function, so `is_null` over a string or temporal column raised.
- **`if_else` no longer drops validity.** The interpreter's `IF_ELSE` used the legacy
  `select`, which was numeric-only and ignored the condition's nulls; it now routes through
  `case_when`, where a null condition counts as false (Arrow semantics).
- **A fused string predicate reports its nulls.** `StringPredicate` (`startswith`, `like`,
  `contains`, string `==`/`!=`) inherited the default all-valid `validity()`, so a null
  operand produced a *valid* `false` in the fused lane while the dynamic lane returned null.
  It now ANDs its operands' validity, as `StringCompare` already did for `<`/`<=`/`>`/`>=`.
- **`Bitmap.__eq__` compares words, not bits.** It re-implemented the comparison bit-by-bit
  against the backing pointer instead of forwarding to `BitmapView.__eq__` (~64x slower for
  the same answer).

### Features

- **One compilation unit per test selection.** The harness used to build one runner per
  `.mojo` file, paying marrow's elaboration once per file. `conftest.py` now generates a
  single driver (`.test_runners/_test_driver_<hash>.mojo`) that imports every selected case
  and hands it to `TestSuite.run`, then compiles that once. Measured on `marrow/exprold/tests`
  (9 files, 280 cases): **4m43s for all nine together against ~200s *each* separately**, and
  the aggregate peaks *below* a single file's memory — the cost is elaborating marrow, not
  the test bodies. Consequences: narrowing the selection by *file* saves time, narrowing to a
  single `::case` does not; one compile error fails the whole selection; case names must be
  unique suite-wide (12 collisions were renamed). Driver names are content-addressed so
  parallel `pytest` invocations cannot clobber each other. `--pkg`/`--no-pkg` and the
  precompiled-`marrow.mojoc` build path are gone with it; `--asan` is unchanged.
- **Tests and benchmarks live next to the code they cover again** (`marrow/tests/`,
  `marrow/kernels/tests/`, …, `python/marrow/tests/`). They carry no `main()` — the generated
  driver owns the only one — which is what lets them sit inside the package. Their
  `marrow.*` imports had to become relative: absolute self-imports break `mojo precompile
  marrow` and `mojo package marrow`. The two standalone `profile_*.mojo` programs moved to
  `benchmarks/profiles/` since `main()` is illegal inside a package.
- **`-D MARROW_GPU=true` opts into GPU code generation; it is off by default.**
  `marrow.utils.GPU_ENABLED` is the single switch, gating the kernels' device allocations and
  `has_accelerator_support`. This is the largest compile-time lever in the tree: `cast`'s
  numeric x numeric dispatch goes **40.1s -> 14.6s**, `cast` + `sort_indices` **85.0s ->
  43.7s** (cold builds). Both halves must be gated to get any of it — gating the allocations
  or the capability check alone measures as no change. A GPU `ExecutionContext` raises at the
  dispatch site in a GPU-off build rather than silently taking a CPU path; `pytest --gpu`
  passes the flag automatically.

- **Aggregates are written on the expression they aggregate.** `rel.aggregate(...)`
  took three positionally-correlated lists (`values`, `funcs`, `names`); it now takes one:

  ```mojo
  rel.aggregate(keys=[col("region")],
                aggs=[col("amount").sum().alias("total"),
                      col("amount").max().alias("biggest")])
  ```

  `col("x").sum()` on a `DynValue` yields a `DynAgg` — the function's *name* plus its input —
  and on a fused node it is the existing `Reduction`, which converts to the same `AggExpr`
  with its `Aggregation` already named. So the fused (AOT) spelling is the identical call, one
  lane down: `col("amount", int64).sum()` resolves nothing at run time. Both lanes mix in one
  list. The distinct counts, which have no fused reduction, are `.count_distinct()` /
  `.approx_count_distinct()` on any value. A whole class of caller mistake (lists out of step)
  is now unrepresentable, so the test that guarded it is gone.
- **`SELECT agg(x)` with no `GROUP BY` executes as a plan.** Leaving `keys` empty used to
  raise inside the grouper; it is now one implicit group. It routes through the same
  per-column entry point as a keyed query rather than the vectorized whole-column reduce:
  reaching the latter from a plan makes the whole name→aggregate catalog reachable from
  *every* plan, which measured **+13%** on the fused binary-size gate. The eager
  `RecordBatch.aggregate` binding still takes the fast route.

### Fixes

- **`count` of an all-null group is 0, not NULL** — over any dtype. It was 0 for non-numeric
  columns (the validity scan) and NULL for numeric ones (the `AggState` fold), which SQL says
  is wrong. Stated on the kernel as `AggKernel.empty_is_null` rather than special-cased at the
  call site, so a group with nothing to fold is NULL unless the aggregate says otherwise.

### Refactors

- **`marrow/kernels` no longer knows what an aggregate is called.** The `Aggregation`
  implementations (`NumericAgg`, `TemporalMinMax`, `StringMinMax`, `CountAgg`, `DistinctAgg`)
  and the `AggFunction` catalog (`Sum`, `Min`, `Count`, …) live in `kernels/aggregate.mojo`
  with the algebra they execute; the delegating `StringMinMaxKernel`/`CountValidKernel` layer
  is gone. This also removed a cycle: `expr.aggregates ↔ expr.dynamic`.
- **The group-by driver is one function.** `GroupPartitioner` / `WholeRows` / `ByKeyHash` are
  deleted — inverting control flow through a `work` callback bought nothing for two cases, so
  `_by_partition[col_agg](..., partition: Bool)` is the whole thing. `slice_struct` is gone
  too: `RapidHash.apply(StructArray)` now honours a struct's own offset, so `keys.slice(...)`
  works. `AggFold` lost its `whole` pointer (a box resolved and called once buys nothing) and
  the thread-local path's three parallel `List[Optional[...]]` arrays indexed `[t * na + j]`
  became one `ThreadPartials` value.

### Tests

- `marrow/exprold/tests/test_aggregates.mojo` — aggregation through the expression API only
  (plan-build + `execute`), so the machinery underneath stays refactorable, plus AOT/fused
  cases that must agree with the dynamic ones column for column.
- `bench_groupby.mojo` gains `g1k` / `g100k` cases: cardinality is what picks the execution
  strategy, and the radix path had no Mojo-level benchmark at all.

### Refactors

- **Aggregates are types all the way down — `Aggregation` replaces every name/tag
  dispatch.** An aggregate used to be a comptime `AggKernel` whose *behaviour* lived in
  free functions that re-identified it by name (`comptime if K.name == "min"`). Now
  resolving an aggregate against an input dtype yields one **`Aggregation`** type —
  `NumericAgg[K, V]`, `TemporalMinMax[Op, T]`, `StringMinMax[Op, T]`, `CountAgg`,
  `DistinctAgg[exact]` — that names its own `InArray`/`OutArray` and carries the whole
  per-column implementation (`grouped` / `whole` / `partials` / `merge`). The routing that
  was a name comparison (bytewise string min/max, the temporal fold, the validity-only
  count, the distinct sketches) *is* which type was chosen. `AggFunction` (`Sum`, `Min`,
  `Count`, …) states which input dtypes each aggregate supports, so a new aggregate cannot
  forget the rule and no central ladder has to know every aggregate that will ever exist.
  18 module-level functions and every `K.name ==` comparison are gone.
- **The type erasure moved to the edges.** The aggregate path is typed end to end:
  `Aggregation.grouped` takes a `PrimitiveArray[V]` / `BinaryLikeArray[T]` and returns a
  typed column; `AnyArray` appears only where a caller genuinely holds a heterogeneous
  list, via one `from_any`/`to_any` pair of O(1) handle conversions. `AggKernel`'s erased
  `reduce(AnyArray)` / `dispatch` overloads, `agg_out_dtype`, `_acc_dtype`,
  `_fold_grouped`, `_partials`, `_merge_partials` and their `_typed` halves, and the
  `_reduce_widened` pair are all deleted; `Reduction` in the fused value tower now calls
  the typed `K.reduce[V]`.
- **The runtime→comptime boundary is one switch, in the dynamic layer.**
  `marrow.exprold.dynamic.resolve_agg(name, value_dtype)` is the only place a string is
  compared, and it happens once per aggregate at plan-build time — the aggregate
  counterpart of `DynValue`'s tag switch. It resolves the *dtype* at the same moment, so
  the plan holds a pointer to a fully monomorphized aggregation and no `dispatch_numeric`
  ladder runs per batch. An aggregate that is not defined for a column's type (`mean` of a
  string) is now rejected when the plan is built rather than when it executes.
- **Layering: `marrow/kernels` executes, `marrow/expr` orchestrates.**
  `kernels/aggregate.mojo` keeps the fold algebra (`AggKernel`, `AggState`), the two scans
  that are not folds (`StringMinMaxKernel`, `CountValidKernel`), and the `Aggregation` /
  `AggFunction` contracts. The aggregations, the function catalog, the erased boxes and
  the drivers live in `expr/aggregates.mojo`, next to the fused value expressions.
- **`Aggregates` — the aggregate *set* owns the multi-aggregate drivers.** The standalone
  `aggregate_grouped` / `aggregate_whole` / `_thread_local_multi` functions are replaced by
  `Aggregates.grouped(gb, values)` / `.whole(values)`, so "group once, apply N aggregates"
  is a property of the set rather than three free functions over parallel lists. The
  `Aggregate` relation node and `AggregateProcessor` hold one `Aggregates` (plus the value
  expressions) instead of two parallel lists.
- **`GroupBy` is factored on a `GroupPartitioner`.** The serial and radix strategies were
  the same algorithm over a different number of partitions, written out four times (single-
  vs multi-column × serial vs radix). They are now one driver, `GroupBy._by_partition[P,
  col_agg]`, parameterized on `WholeRows` (one partition, no gather) or `ByKeyHash` (radix
  partitioning, parallel). Only the thread-local path stays separate, because it splits by
  row range rather than by key and therefore has to merge. `GroupBy(keys, ctx, strategy)`
  can now force a strategy — what tests and benchmarks need to compare the paths on the
  same input.
- **Temporal columns no longer need reinterpreting by their callers.** `take`, `concat` and
  `rapidhash` handle temporal dtypes directly, so `reinterpret_array` /
  `temporal_backing_dtype` and the reinterpret-and-relabel dance in the group-by drivers and
  in `AggregateProcessor`'s key path are deleted; `TemporalMinMax` folds over the integer
  backing internally and relabels on the way out, carrying unit and timezone.
- **`GroupBy`'s `sum`/`min`/`max`/`count`/`mean`/`count_distinct` shorthands are removed**
  in favour of `aggregate[A]` (typed) and `apply[F]` (runtime dtype).

### Features

- **The relational layer can express a *comptime* aggregate — `AggFunc`.** An aggregate is
  no longer a `String` in a plan node: `Aggregate` (and `AggregateProcessor`) now carry
  `List[AggFunc]`, a closed erasure over a comptime `AggKernel`. Three ways to build one,
  all landing on the same kernels:
  `AggFunc.typed[SumKernel, Int64Type]()` (fused/AOT — kernel *and* input dtype comptime,
  so the plan holds a direct pointer to `AggState[K, V]` with no dispatch left),
  `AggFunc.of[SumKernel]()` (kernel comptime, dtype resolved per column), and
  `AggFunc("sum")` (the dynamic frontend's entry). This closes the F1/F2 gap that
  `benchmarks/binary_size/query_streaming_agg.mojo` documented by construction: a fused plan
  now contains no function-name switch at all. `AnyRelation.aggregate(keys, values, funcs,
  names)` and the Python `group_by(...).aggregate([...])` binding are unchanged in signature
  and behaviour; they resolve names to `AggFunc` once, at plan-build time.
- **`benchmarks/binary_size/query_streaming_agg_fused.mojo`** — the same
  `SELECT name, sum(a), min(b) ... GROUP BY name` as `query_streaming_agg.mojo` but through
  the comptime-kernel spec. The delta between the two is the cost of a runtime aggregate
  identity.

### Refactors

- **The aggregate tag vocabulary is deleted.** `AGG_SUM`…`AGG_APPROX_COUNT_DISTINCT`,
  `agg_tag_from_name`, `agg_name_from_tag`, `agg_is_distinct`, `for_agg_tag`, the
  tag-keyed `agg_out_dtype`/`aggregate_column`/`_agg_over_gids`/`_whole_col` are gone. What
  replaces them is one generic body per concern, parameterised on the kernel
  (`agg_out_dtype[K]` / `agg_grouped[K]` / `agg_whole[K]`), and one name switch,
  `dispatch_agg[job](name)`, keyed on the kernels' own `name`. The output-dtype rule now
  *is* the kernel's accumulator algebra (`AggKernel.AccType`) for numeric inputs, with a
  rule of its own only for the non-numeric cases; and the temporal-`min`/`max` reinterpret in
  `aggregate_grouped` is derived from `out_dtype` rather than re-listed. Aggregate results and
  output dtypes are unchanged.
- **`AggFold`** splits the eager `GroupBy` drivers' extra capabilities (whole-array reduce,
  thread-local partial + merge) out of `AggFunc`. Every field of an erased box is live code
  for every kernel its name switch can produce, so carrying them on the box a *plan* holds
  cost the aggregate binary-size gate **+3.2 MB (+24 %)** for capabilities a relational plan
  never calls. Both boxes resolve through the same `dispatch_agg`, so there is still exactly
  one list of kernels.
- **Runtime aggregate routing left the kernel layer.** `marrow/kernels/aggregate.mojo`
  documented `for_agg_tag` as "the one place a runtime function name resolves to a comptime
  `AggKernel`" while `AggKernel`'s own docstring says such selection "lives in the expression
  layer, never here". The `AGG_*` tags, `agg_tag_from_name`, `agg_is_distinct`, `for_agg_tag`
  and `agg_out_dtype` now live in the new **`marrow/exprold/aggregates.mojo`**, together with the
  runtime multi-aggregate drivers that were `GroupBy.aggregate_runtime` / `aggregate_column` /
  `aggregate_whole` / `_agg_name` / `_serial_multi` / `_radix_multi` / `_thread_local_multi`.
  No `UInt8` aggregate tag crosses into `marrow/kernels/` any more. `GroupBy` keeps only
  comptime-kernel-parameterised entry points: `aggregate[K]`, the
  `sum`/`min`/`max`/`count`/`mean`/`product` shorthands, and the new tag-free
  `aggregate_columns[col_agg]` — a multi-column driver over a caller-supplied *comptime*
  aggregator, which the expression layer instantiates with its tag routing. String and
  temporal `min`/`max` and the distinct counts no longer detour through the runtime surface:
  temporal folds over its integer backing around `aggregate[K]` and relabels; string min/max
  and the distinct counts go through `aggregate_columns`. Results, output dtypes and
  parallel-strategy selection are unchanged.
- **The four fold aggregates are two parameterised kernels.** `MinKernel`/`MaxKernel`
  differed only in `identity`, `combine` and an `is_min` flag; `SumKernel`/`ProductKernel`
  only in `identity` and `combine`. They are now `MinMax[Op]` and `Widening[Op]` with
  small op structs, the same shape already used by `ConditionalBinary[K]`. The old names
  survive as `comptime` aliases, so no call site changed.
- **`_reduce_widened` delegates to `_reduce_widened_typed`** instead of re-implementing
  its body — the erased entry point now only resolves the runtime dtype.


### Refactors

- **`AnyArray.view(dtype)`** — zero-copy reinterpretation of an array's buffers under
  a same-layout dtype, mirroring `pyarrow.Array.view`. `parquet`'s `_retag` and
  `kernels/aggregate`'s `reinterpret_array` were the same function written twice in
  different layers; both now delegate to it.
- **`Bitmap.intersect(a, b)`** — the optional-validity AND algebra (`None` = all-valid,
  so it is the identity). `kernels/helpers.bitmap_and` delegates to it.


### Breaking

- **The compute functions live only in `marrow.compute` now** — the 24 duplicates at
  package top level (`marrow.add`, `marrow.sum`, …) are removed, matching PyArrow,
  which exposes them only under `pyarrow.compute`. This resolves a real hazard
  rather than tidying: `filter` / `take` / `sort_indices` each existed three times
  (top level, `compute.py`, and as `Array` methods) with **contradictory
  `null_placement` defaults**, so the same call meant different things depending on
  the spelling. It also stops `min`/`max`/`sum`/`any`/`all`/`filter` shadowing
  Python builtins at package scope. Migration is mechanical:
  `ma.sum(x)` → `ma.compute.sum(x)`.

### Fixes

- **`marrow.compute` no longer silently ignores keyword arguments.** `skip_nulls=False`,
  `mode=` on `count_distinct`, `boundscheck=False` on `take`, and multi-key `sort_keys`
  were all accepted and dropped, so e.g. `pc.sum(a, skip_nulls=False)` returned the
  *skip-nulls* answer — a wrong result, not a missing feature. They now raise
  `NotImplementedError`.

- **`marrow.compute` can use a parallel/GPU context.** Every function hard-wired
  `_serial()`; they now take and honour `ctx`.

### Refactors

- Removed four dead public functions (`list_array_from_arrays`,
  `fixed_size_list_array_from_arrays`, `struct_array_from_arrays`,
  `read_ipc_stream_schema`) — no callers anywhere. PyArrow spells the survivors as
  `ListArray.from_arrays` classmethods, not free functions.


### Fixes

- **`filter` and `take` now support decimal and interval columns.** Their dispatch
  had separate `is_numeric()` and `is_temporal()` arms — identical apart from the
  trait bound — and `is_numeric()` is integer-or-float only, so decimal and
  interval columns fell through to `unsupported dtype`. Both arms collapse into a
  single `dispatch_primitive` one, since the typed leaf is bound on
  `PrimitiveType` and accepts all of them directly.

  This was masked by a latent bug: `Take.apply` computed its gather width as
  `simd_byte_width() // size_of[...]`, which is **0** for types wider than a SIMD
  register (decimal256 at 32 bytes) and is not a legal store width. The width is
  now floored at 1, where the vector gather degenerates to a scalar one — which
  is what those types want anyway. The narrow arms had been hiding this by never
  instantiating the wide types.


### Refactors

- **`filter`, `take`, `sort` and `hashing` no longer reinterpret temporal columns.**
  Every typed kernel leaf is bound on `PrimitiveType`, which the temporal, interval
  and decimal types already satisfy — so the dispatch layer now hands the column
  straight to the leaf via `dispatch_temporal` / `dispatch_primitive` instead of
  reinterpreting it to an integer backing and relabelling the result. For
  `filter`/`take` this is also *more* correct: the output is a `PrimitiveArray[T]`
  whose dtype is preserved by construction, rather than stripped and reattached.
  `AnyDataType.storage_type()` is deleted (no callers remain).

  Cost is monomorphization — ~15 logical primitive types now instantiate the
  kernels rather than 4 integer widths — but it lands on the runtime binary, not
  the AOT one: the **fused `query_streaming` binary shrank 6.1%** (1,357,176 →
  1,274,584 bytes stripped) since it no longer links the reinterpret machinery,
  while `query_dynvalue` grew 5.2%. Fused size is what the small-binary property
  is about, so this is a win on the metric that matters.


### Refactors

- **`dispatch_over_*` are now methods on `AnyDataType`** (`dt.dispatch_numeric[f]()`
  rather than `dispatch_over_numeric[f](dt)`), across 47 call sites. They existed
  only to read `AnyDataType._v` — a private field — from another module, so as
  methods the accessor is unnecessary and `_v` is private again. This also removed
  the last import cycle: `utils.mojo` imported nine dtype names solely for these,
  while `dtypes.mojo` imported `variant_dispatch` back. `utils.mojo` now has no
  local imports at all, and the module graph is fully acyclic.


### Fixes

- **`sort_indices` and `rapidhash` now accept every Arrow dtype** — temporal
  (date/time/timestamp/duration), interval, `large_string`, `large_binary`,
  `binary`, decimal (32/64/128/256), and dictionary. Both were hand-written
  `if dtype == … elif …` ladders that stopped at bool/numeric/string and then
  raised, so `GROUP BY` on any temporal key and `ORDER BY <timestamp>` failed
  outright (`docs/code-quality-review.md` D5; ClickBench Q8, Q19, Q24–27, Q35,
  Q36, Q40, Q43 — the `hits` table is keyed on `EventDate`/`EventTime`).
  Supersedes FU-1, which tracked only the `large_string` half.

  Behavioural notes: a dictionary column now hashes and orders by its *decoded*
  values, so it matches the equivalent plain column; and `decimal128`/`decimal256`
  hash by folding both 64-bit limbs instead of truncating to the low one — a
  truncating hash would have merged distinct groups, since group-by buckets on
  the hash alone.

### Refactors

- **`hashing.mojo` → `RapidHash`, `sort.mojo` → `SortIndices`** — the last two
  kernel modules with no `Kernel` struct, and (not coincidentally) exactly the
  two with dtype-coverage gaps: with no single `dispatch`, each new dtype had to
  be remembered in a hand-written ladder, and wasn't. Both now follow the
  three-tier pattern (typed `apply` leaves, one `dispatch`), with thin free
  delegators kept for the call sites that bind a hasher as a comptime function
  value (`SwissHashTable[hasher]`) or match PyArrow (`sort_indices`, `sort`).
  `sort`'s multi-key body moved to `SortIndices.multi`, which returns the
  permutation, leaving `sort` as `take ∘ SortIndices.multi`. The public free
  function literally named `array()` in `sort.mojo` — which forced
  `import array as _primitive_array` in its own file — is now
  `SortIndices._assemble`. Verified-dead `hash_identity` (×3) deleted.

- **Completed the `dispatch_over_*` family** (`marrow/utils.mojo`): added
  `dispatch_over_primitive`, `dispatch_over_integer`, `dispatch_over_temporal`,
  and `dispatch_over_listlike` alongside the existing numeric / floating /
  string-like / binary-like members, so there is one member per dtype-family
  trait and a kernel never has to spell out its own ladder. They now reach the
  variant through the new `AnyDataType.variant()` accessor instead of the
  private `dt._v`.

- **`AnyDataType.storage_type()`** (`marrow/dtypes.mojo`) — the same-width
  signed-integer dtype a fixed-width logical type is stored as. Generalizes
  `temporal_backing_dtype` to interval and decimal32/64, and lets the
  value- and order-preserving kernels route those columns through the
  already-instantiated integer path rather than growing one arm (and one kernel
  instantiation) per logical dtype.

- **Broke the `dtypes` ⇄ `arrays`/`scalars` circular import.** `DataType` declared
  `comptime ScalarType` / `comptime ArrayType` companion types on the trait and on
  ~30 concrete types, which is what forced `dtypes.mojo` to import `arrays` and
  `scalars` — while those modules import `dtypes` back. **Nothing consumed them**
  (the only `.ScalarType`/`.ArrayType` uses in the tree are `Self`-qualified
  members of the separate `Array` and `Builder` traits), so removing them costs
  nothing and deletes 63 lines. `dtypes.mojo` now imports only `.utils`, making
  the module graph a DAG.

  The cycle had caused three distinct build failures, all with the same
  signature — the same source compiling or failing depending on which file the
  build entered through: a `Scalar` trait shadowing the builtin `Scalar[dtype]`,
  `BoolArray` resolving along one import path but not another, and a rewrite that
  moved errors 2 → 10 by shifting the ambiguity rather than removing it.


### Fixes

- **Fused mixed-width numeric comparison no longer truncates.** `NumericCompare`
  took the *left* operand's native type and cast the right operand down into it,
  so `col("a", int32) > col("b", int64)` silently narrowed every int64 value.
  It now compares in the promoted domain of both operands — the same
  `promote[L, R]` rule `NumericBinary` already used, so `a > b` and `a + b` can
  no longer disagree about widening. The SIMD lane width is a separate question
  (a narrower dtype yields a *larger* `W`, which would overflow a wider
  operand's load), so that is now expressed by a dedicated `wider[L, R]` alias
  which `BoolBinary` also reuses in place of its hand-inlined copy.

- **Fused `Any` / `All` no longer ignore nulls.** Two `AnyKernel`/`AllKernel`
  pairs existed — one in `kernels/boolean.mojo` that counted set bits with **no
  validity mask**, and the null-correct pair in `kernels/aggregate.mojo` that
  `kernels/__init__.mojo` re-exports. The expression layer imported the former,
  so a null slot whose data bit happened to be set read as `True`. The duplicate
  is deleted and the expression layer now binds the correct pair.

- **`DynValue.name()` no longer leaks operator payloads as column names.**
  `_name` is overloaded to carry the LIKE/ILIKE pattern and the `date_trunc`
  unit, and `name()` returned it unconditionally — so a `LIKE` node reported
  `"%foo%"` and a `DATE_TRUNC` node `"day"` as its output column name. It now
  returns an empty string unless the node is a `LOAD`.

- **`BoolScalar.repeat` added.** `AnyScalar.repeat` raised
  `unsupported dtype bool`, so broadcasting a boolean scalar to an array was
  impossible — surfaced by the new `Any`/`All` parity tests.


### Refactors

- **Renamed the `Scalar` trait to `ArrowScalar`** (`marrow/scalars.mojo`). The
  old name shadowed the builtin `Scalar[dtype]` alias, and because `arrays.mojo`
  and `dtypes.mojo` wildcard-import each other, the bare name resolved to the
  trait along that cycle and `Scalar[T.native]` stopped parsing. Both existing
  workarounds — `Scalar as ScalarTrait` in `arrays`/`dtypes` and
  `Scalar as _Scalar` in `scalars` — are gone with it.

### Build

- **Upgraded Mojo to `1.0.0b3.dev2026072406`** (pinned exactly in all three
  places in `pixi.toml`). This **fixes a long-standing heap corruption**: boxing
  a `DynValue` into an `AnyValue` wrote a `Variant` discriminant one byte past
  its `ArcPointer` allocation, silently corrupting the heap until it hit live
  allocator metadata. It accounted for *every* failure in the full test suite
  (`parquet/tests/test_reader.mojo`, `expr/tests/test_streaming.mojo`, 59 in
  total); both files now pass, and ASAN reports 0 `heap-buffer-overflow` hits
  where it previously reported 86.

  API migrations required by the upgrade (all per the upstream changelog):
  `Span(ptr=)` → `Span(unsafe_ptr=)`; `memset`/`memset_zero` →
  `unsafe_memset`/`unsafe_memset_zero`; `SIMDSize` → `SIMDLength`;
  `OwnedPointer[T]` and `Optional[T]` now conform to `ImplicitlyDeletable` only
  conditionally (explicit `__del__` on `DictionaryType`, and an
  `ImplicitlyDeletable` bound on `RadixPartitioner.map_partitions`'s result
  type); and `Variant` adopted *interior origins*, so `bitmap_and` call sites
  pass `.copy()`. Dropped the `_accelerator_arch()` GPU-arch validation.

  Adds a `max` dependency, which **should not be required** — GPU kernels
  compiled without it before this release. Tracked as a toolchain regression in
  `pixi.toml`; retry dropping it on the next upgrade.

- **New build-only tasks `check` and `check_lib`** for fast compile-error
  iteration (~4 s for one file, ~10 s for the whole library, versus minutes for
  the equivalent `pytest` run). Documented in `CLAUDE.md`.

### Features

- **Relational `Aggregate` completeness + `HAVING`.** `AnyRelation.aggregate`
  now accepts **computed group keys** (arithmetic, `CASE`/`if_else`,
  `year()`/`date_trunc()`, literals) and **computed aggregate inputs**
  (`SUM(x + 1)`, `AVG(length(s))`) — key and value dtypes are inferred by probing
  the expression against a 0-row batch, the same trick `project` uses.
  `count_distinct` / `approx_count_distinct` are wired through as aggregate
  functions, `min`/`max` work over string and date/timestamp value columns,
  `count` works over any dtype, and an optional `names=` argument aliases the
  aggregate output fields (so `mean(a), mean(b)` no longer collide). A temporal
  group key is grouped through its integer backing and relabelled on emit
  (`rapidhash` has no temporal case). `HAVING` needs no node of its own —
  `rel.aggregate(...).filter(pred)` resolves `pred` against the aggregate output
  schema.
- **Relational operators: `Sort`, `Limit`/`Offset`/`TopK`, computed `Project`.**
  `Sort` (pipeline breaker) reuses the sort kernel over a key+data struct; `Limit`
  slices exactly across morsels; a `Limit`-over-`Sort` folds into the kernel's
  top-K `limit`; `Project` now evaluates arbitrary expressions with output names.
  Plan builders `.sort/.limit/.project` on `AnyRelation`.
- **Wave 1 kernels wired into both expression drivers.** The fused comptime
  `Value` layer (F2) and the runtime `DynValue` interpreter (F1) both expose
  string ordering compares, `like`/`ilike`, `is_in`, `coalesce`/`nullif`/
  `case_when`, and temporal extraction/`date_trunc` — the two drivers agree
  element-for-element (parity harness). (Known gap FU-5: fused `is_in` composed
  under boolean logic; the dynamic path is correct.)
- **Temporal `filter`/`take`.** `Filter`/`Take`/`drop_null` now handle
  date/time/timestamp/duration columns by reinterpreting to the integer backing
  around the numeric path — unblocking filter/sort/join over temporal columns.
- **New compute kernels for analytical queries (ClickBench-driven):**
  - `marrow/kernels/conditional.mojo` — `case_when` (multi-branch, all types),
    `coalesce`, `nullif`, `fill_null`, all via a shared `concat`+`take`
    multiplexer (type-agnostic, null-correct).
  - `marrow/kernels/membership.mojo` — `is_in` over numeric/bool/string, reusing
    `SwissHashTable`/`rapidhash` (the join/distinct basis).
  - `marrow/kernels/temporal.mojo` — `year/month/day/hour/minute/second/`
    `day_of_week/quarter/day_of_year` extraction (Hinnant civil-date) +
    `date_trunc(unit)`, over date/time/timestamp (UTC).
  - `compare.mojo` — string/large_string ordering comparisons (`< <= > >=`);
    `string.mojo` — `like`/`ilike` (SQL `%`/`_`, PyArrow `match_like` semantics).
  - `aggregate.mojo`/`groupby.mojo` — `min`/`max` over string and temporal
    (whole-table and grouped); grouped `count_distinct` as a first-class agg.
- **The fused comptime `Value` lane tracks validity (nulls).** Previously the
  fused numeric/bool lane computed data only and emitted
  `BoolArray`/`PrimitiveArray` with `nulls=0, bitmap=None`, diverging from the
  null-correct dynamic path. Each node now exposes `validity(batch)`: propagating
  ops (arithmetic/compare/unary/cast/length) AND-combine children validity via
  the same `bitmap_and` helper the eager kernels use; Kleene `and_`/`or_` reuse
  the null-correct `AndKernel`/`OrKernel` apply; `is_null`/`not_null`/literals are
  never-null. The fused and dynamic drivers now agree element-for-element on
  nullable input (enforced by the parity harness), and the small-binary DCE
  property is preserved.
- **`referenced_columns()` / `is_deterministic()` on the expression layer.**
  Both the fused comptime `Value` tower (every concrete node) and the runtime
  `DynValue`, surfaced through `AnyValue` via the existing fn-pointer
  trampolines. `referenced_columns()` returns the deduped set of column names an
  expression reads; `is_deterministic()` is an overridable predicate (`True` for
  all current nodes). These are the prerequisite metadata for projection and
  predicate pushdown.
- **`DynValue` gains `mod`/`floordiv`/`xor`/`not_null`** (tags `MOD`,
  `FLOORDIV`, `XOR`, `NOT_NULL`), each wired to its existing kernel, with `%`,
  `//`, `^`, and `.not_null()` builders — narrowing the runtime interpreter's
  gap to the fused algebra.
- **Parquet reader `ByteSource` seam.** A `ByteSource` trait (`size()` +
  zero-copy `read_at(offset, length)`) in `marrow/parquet/source.mojo`;
  `MappedFile` implements it (all mmap syscalls now live there) and
  `ParquetFile[S: ByteSource = MappedFile]` reads through it while keeping the
  `ParquetFile(path)` convenience. Pure refactor toward streaming and remote
  (OpenDAL) scans.
- **LIKE/ILIKE scalar-pattern kernels (compile the pattern once).**
  `LikeKernel.apply(array, pattern)` / `.dispatch(array, pattern)` and the
  `ILikeKernel` equivalents take the pattern as a `StringSlice` instead of a
  broadcast array, so `col LIKE '%const%'` compiles the pattern once per call
  rather than once per row. A new `LikePattern[ignore_case]` carries the
  compiled form and is the single matching implementation behind both the
  scalar and the array × array overloads: it classifies the pattern into the
  literal shapes `foo` / `foo%` / `%foo` / `%foo%` (which match through the
  optimized string primitives, escapes included — `%\%%` compiles to
  `contains('%')`) and otherwise runs a backtracking wildcard matcher directly
  over each row's UTF-8 bytes, with no per-row allocation. `ILIKE` classifies
  each row with one byte scan and skips the microsecond-scale Unicode `lower()`
  whenever the row is ASCII. Measured on ClickBench-style URLs
  (`marrow/kernels/tests/bench_string.mojo`): `LIKE '%google%'` over 1M rows
  799 ms → 7.6 ms (104x), `ILIKE '%GOOGLE%'` over 100k rows 1235 ms → 3.0 ms
  (417x); the array × array path is 3-6x faster too.

### Tests

- **Cross-driver parity harness** (`marrow/exprold/tests/test_parity.mojo`):
  `assert_parity` runs a fused `Value` and an equivalent `DynValue` against one
  batch and asserts equal results, so the runtime interpreter can never silently
  diverge from the fused algebra. Covers arithmetic, comparisons, cast, if_else,
  null propagation, and Kleene `and_`/`or_` over nullable masks.

### Fixes

- **Kleene 3-valued logic in the boolean kernels.** `and_`/`or_`/`xor`/`not_`
  no longer drop the validity bitmap: `TRUE OR NULL = TRUE`,
  `FALSE AND NULL = FALSE`, etc., matching `pc.and_kleene`/`pc.or_kleene`. All
  bitmap combination goes through the idiomatic `Bitmap` bitwise API.
- **`LengthKernel` / `ArrayLengthKernel` propagate nulls** — a null input
  element now yields a null output element, matching `pc.utf8_length` /
  `pc.list_value_length`.

### Refactors

- **One home for grouped aggregate dispatch and for the aggregate output-dtype
  rule.** `GroupBy.aggregate_column(gids, value, num_groups, tag)` is now the
  public per-column entry point for every runtime-tagged aggregate (distinct
  kernels, string/temporal min/max, non-numeric `count`, typed `AggState`
  folds), used by both the kernel-level multi-aggregate drivers and the
  expression layer's `AggregateProcessor` — which drops its numeric-only
  `AggState` copy and instead `concat`s each aggregate's buffered morsels once
  and calls it. The output-dtype rule moves next to the tag helpers as
  `agg_out_dtype(tag, value_dtype)` in `kernels/aggregate.mojo`, replacing
  `AggregateProcessor.out_dtype`, so plan-time schemas and the kernels can never
  disagree.
- **The staged, strategy-pluggable fusion engine is now `marrow.exprold.values`
  and drives the relational engine.** The from-scratch engine (previously
  prototyped as `lane.mojo`) replaces the old `values.mojo`: `execution.mojo`
  and `relations.mojo` execute plans over its `AnyValue`, which dual-boxes
  either a comptime `Value` node or a runtime `DynValue`. `AnyValue.write_to`
  renders a boxed `DynValue`'s full expression form so plan printing is
  unchanged. The four family tests (`test_lane*`) are consolidated into a single
  `test_values`; the redundant `test_erased` / `test_relations` (old comptime
  `Table`/`AnyValue` surface) are removed, their unique `AnyValue`
  interchange/`write_to` coverage folded into `test_values`.
- **`marrow.exprold.values`: family-refined `execute` eliminates all consumption
  `rebind`s.** Each value family now refines `execute`'s return to its concrete
  array — `NumericValue` → `PrimitiveArray[Self.OutType]`, `StringValue` →
  `BinaryLikeArray[Self.OutType]`, `ListValue` → `ListLikeArray[Self.OutType]`
  (matching `BoolValue` → `BoolArray`). A child's `execute()` therefore yields a
  fully typed array at every call site, so `StringLength` / `StringUnary` /
  `StringPredicate` / `Counting` / `ListContains` pass operands straight to the
  typed kernels with no `rebind`, and the fused numeric `_fused` returns its
  `PrimitiveArray` directly. `StringPredicateKernel.apply` now takes two
  independent string type params so mixed string/large_string predicates need no
  pun. `BoolReduce` is parameterized by the reduce kernel type
  (`BoolReduceKernel` — `AnyKernel`/`AllKernel`) instead of a bool flag.
- **`AggKernel` gains a fully-typed `reduce[V]`** returning
  `PrimitiveScalar[Self.AccType[V]]` (no erased `AnyScalar`, no downcast) with a
  SIMD widened fast path for `sum`/`product`, same-type SIMD for `min`/`max`, and
  metadata `count`. The expr `Reduce` node folds through it, and `Count` reads
  `len - null_count` off the typed operand directly — both drop the erase →
  dispatch → downcast round-trip.

- **`marrow.exprold.ibis` merged into `marrow.exprold.values`**: the ibis-designed
  comptime expression system is now the canonical `values.mojo` (the old fused
  algebra is replaced). `AnyValue` and every node evaluate via `.execute()`
  (`to_array` removed everywhere — `dynamic.DynValue`, `execution.mojo`). The
  runtime `DynValue` is bridged into the new `AnyValue` via a dedicated
  constructor (it no longer implements the comptime `Value` trait). `BoolValue`
  execution is now wired: numeric comparisons fuse (`NumericCompare` — SIMD bool
  lane, bit-packed) and boolean logic materializes + combines masks (`BoolLogic`
  / `BoolNot`). Pruning is plumbed through `Value.prune` (conservative default;
  `DynValue` keeps the real min/max rule); the old per-node comptime pruning is
  parked as a commented reference. `test_ibis` + `test_ibis_exec` merged into
  `test_values`.

### Features

- **Cross-family casts in the expr system** (`marrow.exprold.values`): `.cast(target)`
  is overloaded on every value family and dispatches by the target dtype's family,
  wiring the `marrow.kernels.cast` kernels into expressions — numeric↔string↔bool
  in every direction (`NumToBool`, `StringToBool`, `NumToString`, `BoolToString`,
  `StringToString` utf8↔large, `StringToNum`, `BoolToNum`), with numeric→numeric
  staying the existing fused `Cast`. String parse casts take a comptime `safe`
  (default null-on-failure). Each cast node conforms to its *target* family and
  materializes through the kernel.
- **Prepare-then-fuse: boundary nodes re-enter the numeric lane** (`marrow.exprold.values`):
  numeric execution is two phases — a one-time `prepare(batch)` where *boundary*
  nodes with no SIMD lane (string/bool→numeric casts, string/list byte-length)
  materialize their column once into a per-node cache, then the usual fused `core`
  pass where those nodes read the cache per lane like a column. So the arithmetic
  *above* a boundary fuses into a single pass: `(a.length() + b.length()) + 1`
  prepares two length arrays, then one fused loop. `StringLength` / list `Counting`
  are promoted from `Value` to `NumericValue`, so `col.length()` composes with
  arithmetic / comparisons / `.cast()` / reductions. No fusability flag — every
  numeric node fuses; the family/value type carries the whole story. Reductions
  stay non-lane `Value` (length-1 can't fuse element-wise).

- **Expr floor division and element-wise min/max** (`marrow.exprold.values`):
  wire the previously-unexposed `FloordivKernel` and the binary element-wise
  `arithmetic.MinKernel`/`MaxKernel` into the numeric lane — `a // b`
  (`__floordiv__`, integer floor keeping the wider operand dtype) and
  `a.min_element_wise(b)` / `a.max_element_wise(b)` (PyArrow naming; distinct
  from the whole-column `min()`/`max()` reductions). All three fuse as
  `NumericBinary` nodes (single vectorized pass).

- **List `contains()`** (`marrow.kernels.nested`, `marrow.exprold.values`):
  `ArrayContainsKernel` implements element-wise membership `elem[i] ∈ list[i]` →
  `BoolArray` (each row scans its sublist; null list rows propagate to null;
  numeric element types). The expr `ListContains` node materializes both operands
  (a literal element broadcasts) and applies it, so `col("l", list_(t)).contains(x)`
  runs end-to-end. Removes the now-dead generic `BoolBinary`/`StringBinary` nodes.

- **List `length()`** (`marrow.kernels.nested`, `marrow.exprold.values`):
  `ArrayLengthKernel` counts elements per list → `Int32Array` (offset subtraction,
  vectorized — the list analogue of `string.LengthKernel`). `ListColumn.execute`
  now resolves the list column from the batch (via a new `AnyArray.as_list_like`),
  and the `Counting` node folds `.length()` over it, so `col("l", list_(t)).length()`
  runs end-to-end.

- **Fused numeric `cast`** (`marrow.exprold.values`): `NumericValue.cast(target)`
  adds a `Cast` node that reinterprets the operand's SIMD lane at the target
  dtype, so `col.cast(int64) + other` stays a single vectorized pass. Truncating
  (unchecked), matching the fused cast-kernel path.

- **`any`/`all` reductions and string `==`/`!=`** (`marrow.exprold.values`,
  `marrow.kernels.string`): `BoolValue` gains `.any()`/`.all()` (new `BoolReduce`
  node folding a bool column to a length-1 result via the optimized
  `kernels.aggregate` bitmap reductions). String `==`/`!=` now execute through the
  existing `StringPredicate` node via two new `StringPredicateKernel`s
  (`StringEqKernel`, `StringNeKernel`), null-propagating and working for
  `string`/`large_string`. `NumericColumn.execute` returns the resolved column
  as-is so standalone column execution (reduction/predicate operands) preserves
  the validity bitmap.

- **Boolean predicate kernels + expr wiring** (`marrow.kernels.boolean`,
  `marrow.exprold.values`): the marker structs `XorKernel`, `IsNullKernel`,
  `NotNullKernel`, `IsNanKernel`, `IsInfKernel` are now real kernels. `xor`
  becomes a `BoolBinaryKernel` (bit-packed word op) routed through `BoolLogic`.
  The four unary predicates implement a new `UnaryPredicateKernel` trait
  (`dispatch(AnyArray) -> AnyArray`): `is_null`/`not_null` are family-agnostic
  (read the validity bitmap via the byte-level `views.apply`), `is_nan`/`is_inf`
  scan floating values via `views.apply` (buffer→bitmap, CPU serial/parallel +
  GPU dispatch), propagating input nulls. The expr `BoolUnary` node now executes
  by materializing the operand and applying the kernel's `dispatch`; `.isnull()`,
  `.notnull()`, `.isnan()`, `.isinf()`, and `^` all run end-to-end.

- **String compute kernels** (`marrow.kernels.string`): real implementations
  replacing the name-only markers. `LengthKernel` (byte length →
  `Int32Array`, vectorized as `offsets[i+1]-offsets[i]`), unary string→string
  ops via the `StringMapKernel` trait (`UpperKernel`, `LowerKernel`,
  `ReverseKernel`, `StripKernel`, `LStripKernel`, `RStripKernel`,
  `CapitalizeKernel`), and binary predicates via the
  `StringPredicateKernel` trait (`StartsWithKernel`, `EndsWithKernel`,
  `ContainsKernel` → `BoolArray`). Each has a typed `apply` and a runtime
  `dispatch(AnyArray)`. The free-standing `string_lengths` function is removed
  (callers use `LengthKernel`).

- **`marrow.exprold.values` reduction execution**: the aggregate boundary nodes now
  execute (previously `_not_wired` stubs). `sum()`/`product()` (widen to
  int64/float64), `mean()` (float64), and `min()`/`max()` (operand dtype
  preserved) share a single `Reduce[K, A]` node whose output dtype is the
  kernel's own `AccType[A.OutType]` — each aggregate is the single source of truth
  for its result type. A new family-agnostic `count()` (`Count` → int64, on the
  base `Value` trait) works on any input dtype. All fold the operand through the
  real `marrow.kernels.aggregate` kernels and return the scalar as a length-1
  result array. Reductions are a materialization boundary: the numeric lane fuses
  up to the operand, which is computed in full, then reduced. `AnyValue` erases
  and executes them like any other node.

- **`marrow.exprold.ibis` string execution**: the `StringValue` family now
  executes by materializing — `StringColumn` resolves from the batch,
  `StringConst` broadcasts, `StringUnary` applies a `StringMapKernel`,
  `Counting` (`length`) applies `LengthKernel`, and the new `StringPredicate`
  node (`startswith`/`endswith`/`contains`) applies a
  `StringPredicateKernel` → `BoolValue`. Variable-width UTF-8 has no
  fixed SIMD lane, so string ops materialize rather than fuse; `length` is the
  one op that vectorizes internally (offset subtraction).

- **dtype → scalar/array associated types** (`DataType.ScalarType` /
  `DataType.ArrayType`): every Arrow type now names its companion typed scalar
  and array (the inverse of `Array.ScalarType`), provided at the family traits
  (`NumericType` → `PrimitiveScalar[Self]` / `PrimitiveArray[Self]`,
  `StringLikeType` → `StringScalar` / `BinaryLikeArray[Self]`, …) and on the
  standalone concrete types. This lets generic code map a dtype to its concrete
  companion — e.g. a leaf can hold `T.ScalarType` and construct it via a helper
  bound on the provider trait.

- **`marrow.exprold.ibis` fused execution**: the numeric family now *executes*,
  hooked to the real `marrow.kernels` — `NumericValue` **is** the numeric lane
  (refines `OutType` to `NumericType`, carries a `core[W]` SIMD primitive, and its
  `execute` vectorizes `core` across the whole tree in a **single fused pass**).
  Arithmetic nodes are parameterized by the actual `AddKernel`/`DivKernel`/… (the
  kernel supplies compute; promotion stays in the node). Dedicated per-family
  leaves (`NumericColumn`/`StringColumn`/`ListColumn`, `NumericLiteral`/
  `StringLiteral`) with `col`/`lit` overloaded by dtype family; `execute` returns
  the dtype's companion `Self.OutType.ArrayType`. Bool/string/list are the type
  architecture (execution pending); cross-family numeric-producing boundaries
  (`length`, reductions) are non-lane nodes that materialize. More ops: fused
  transcendental math (`exp2`/`log2`/`log10`/`log1p`/`sin`/`cos`, `trunc` — all
  executing via the real kernels), numeric predicates (`isnan`/`isinf`/`notnull`),
  and string transforms (`strip`/`lstrip`/`rstrip`/`capitalize`). Every op node is
  parameterized by a `marrow.kernels` kernel (real where implemented, else a
  not-implemented marker in `kernels.string`/`kernels.nested`/…) — none defined in
  the expression layer.

- **`marrow.exprold.ibis` typed expression architecture**: value families are
  traits (`NumericValue` / `BoolValue` / `StringValue`), operations are node
  structs, and kernels are pure name markers — promotion lives entirely in the
  value hierarchy (one node struct per `(family, output-dtype rule)`:
  `NumericBinary` widening, `FloatBinary`, `NumericUnary`/`FloatUnary`,
  `CountingUnary`, `BoolBinary`/`BoolUnary`, `StringUnary`). `Column` and
  `Literal` are unified leaves via conditional conformance; `Literal` holds the
  dtype's companion `T.ScalarType` and `lit` is an alias for it. Ops include
  arithmetic (`+ - * / % **`, `neg`/`abs`/`ceil`/`floor`/`round`/`sign`,
  `sqrt`/`exp`/`ln`), reductions (`sum` widening to 64-bit, `mean` → float64,
  `min`/`max` preserving), comparisons, logical (`& | ^ ~`), `isnull`, string
  `length`/`startswith`/`endswith`/`contains`/`upper`/`lower`/`reverse`/`==`/`!=`,
  and a nested **`ListValue`** family (`length` → numeric, `contains` → bool).

- **Columnar selection for all array types** (`filter` / `take` / `drop_null`):
  the selection kernels now support every array type — including nested
  `list` / `large_list` / `map` / `fixed_size_list` / `struct`, plus
  `dictionary`, `binary` / `large_string` / `fixed_size_binary` and `null` —
  fully **column-wise** with no row-encoding. Nested rows are gathered by their
  contiguous child spans; struct is filtered/taken per child; dictionary shares
  its values and only selects the codes. `rapidhash` likewise gained nested
  support (`list` / `large_list` / `map` / `fixed_size_list`), so `group_by` and
  joins work on nested key columns. Performance is best-in-class on the measured
  cases: dictionary `filter` (sequential code compaction, not a gather) and
  `list` / `fsl` `take` (raw-`Int32` child-index build + one dispatched child
  gather) both beat PyArrow and Polars at every size (e.g. `list take` at 1M is
  ~1.6× faster than PyArrow, ~7× faster than Polars; `dict filter` at 1M is ~11×
  faster than PyArrow, on par with Polars).

- **Distinct-count kernels** (`marrow.kernels.distinct`, `mk.count_distinct` /
  `mk.approx_count_distinct`): whole-array cardinality reductions returning an
  `int64` scalar, both excluding nulls (SQL `COUNT(DISTINCT x)` / PyArrow
  `only_valid`). `count_distinct` is exact — it dedups the per-row hashes through
  the same `SwissHashTable` the group-by uses, so it shares that 64-bit-hash
  basis. `approx_count_distinct` is a HyperLogLog estimate (2**14 registers,
  ~0.65% standard error, fixed 16 KiB regardless of cardinality) with linear
  counting in the small-cardinality regime, mirroring
  `pyarrow.compute.approx_count_distinct`.

- **Grouped distinct counts** (`GroupBy.count_distinct` /
  `GroupBy.approx_count_distinct`, and the `"count_distinct"` /
  `"approx_count_distinct"` functions in the Python
  `rb.group_by(keys).aggregate([...])` API): per-group `COUNT(DISTINCT v)`.
  Exact grouping dedups `(group_id, value)` pairs in a single `SwissHashTable`
  (the join's table) and bumps a per-group counter on each newly-seen pair — one
  pass, `O(distinct pairs)` memory, no per-group set. Approx keeps one
  HyperLogLog sketch per group (2**11 registers, 2 KiB/group). A distinct
  aggregate can share the single grouping pass with fold aggregates (e.g.
  `[("v","sum"),("v","count_distinct")]`). Distinct aggregates are
  **radix-parallel**: partitioning by key hash keeps every group inside one
  partition, so per-partition distinct counts are final and concatenate without a
  merge (the thread-local partial-merge path can't union sets, so any distinct
  set routes to radix when parallel). ~7x over serial at 1M rows / 50k groups,
  and faster than pyarrow's `count_distinct`.

- **Python group-by** (`marrow.RecordBatch.group_by`): grouped aggregation is
  now exposed to Python with a PyArrow-compatible API —
  `rb.group_by(keys).aggregate([("v", "sum"), ("v", "mean"), ...])` returns a
  `RecordBatch` of the unique key columns plus one `<value>_<func>` column per
  aggregate (`sum`/`mean`/`min`/`max`/`count`/`product`), grouped in a single
  pass over the keys. Backed by the `GroupBy` kernel and its serial/thread-local/
  radix strategy selection. A new `python/marrow/tests/bench_groupby.py`
  benchmarks it apples-to-apples against pyarrow, polars, and duckdb (all through
  their Python APIs) across row counts and cardinalities — run with
  `pixi run -e bench pytest python/marrow/tests/bench_groupby.py --benchmark
  --competition`.

- **Scalar `mean` reduction** (`marrow.kernels.aggregate.mean`, `mk.mean`,
  `marrow.compute.mean`): arithmetic mean of the valid elements as a float64
  scalar (nulls excluded from sum and divisor; null result for empty/all-null),
  matching `pyarrow.compute.mean`.
- **Grouped `min`/`max` preserve the input dtype** (PyArrow-correct): `min(int32)`
  now returns `int32` rather than widening to `int64`. `sum` still widens
  integers to `int64`; `count` is `int64`; `mean` is `float64`.

- **Cast kernels** (`marrow.kernels.cast`, `mk.cast`): monomorphized numeric,
  bool, and temporal casts behind a two-level dispatcher — a top-level `cast`
  routes on the type family, and each family struct (`NumericCast`, `BoolCast`,
  `TemporalCast`) does the within-family typed dispatch. Numeric casts build on
  `SIMD.cast` (one `pop.cast` per lane); `safe=False` truncates/wraps like
  `numpy.astype`, while `safe=True` (the default, matching PyArrow) raises on any
  lossy conversion. Bool casts use `x != 0` / `True→1`; temporal casts reinterpret
  to the underlying integer or scale by the unit ratio (e.g. `date32↔date64`,
  `timestamp[s]↔[ms]`). Fused cast expression nodes (`marrow.exprold.values`) —
  numeric→numeric (`Cast`), numeric→bool (`NumToBoolValue`), and bool→numeric
  (`BoolToNumValue`), all reached through a single `.cast(dtype)` method on the
  `NumericValue`/`BoolValue` nodes — plus a `DynValue.cast(to)` runtime node let
  casts fuse into AOT-compiled expressions (`Cast(Add(a, b), int64)` collapses to
  a single vectorized pass, `(a < b).cast(int8)` bit-unpacks in place), and a
  PyArrow-style `marrow.compute.cast(arr, target_type, safe=…)` exposes it to
  Python. Also **string ↔ numeric/bool** (per-element `atol`/`atof` parse and
  format; `safe=True` raises on an unparseable value, `safe=False` nulls it) and
  **null → any** (all-null array of the target type).
- **More cast families** (`marrow.kernels.cast`): the cast router now also covers
  - **binary-like** — `utf8`/`large_utf8`/`binary`/`large_binary` ↔ each other
    (zero-copy relabel when the offset width matches, else an offset rebuild;
    bytes→utf8 validates UTF-8 under `safe`) and `fixed_size_binary` ↔ binary;
    `large_utf8` now parses/formats to numeric/bool like `utf8`.
  - **decimal** (`DecimalCast`, decimal32/64/128/256) — decimal ↔ decimal
    (rescale by `10^Δscale`, widening the backing integer as needed), decimal ↔
    integer, and decimal ↔ float.
  - **nested** — `list`/`large_list` → same-kind list (recursively casting the
    child values) and `struct` → `struct` (recursively casting each field).
  - **dictionary decode** — a dictionary source is gathered by index (`take`) and
    the decoded values cast to the target type.

  Remaining designed extension points: dictionary *encode*, `string ↔ temporal`,
  cross-kind list (`list ↔ large_list`/`fixed_size_list`), and `map`.

- **`distinct_count` statistic** (`marrow.parquet`): a dictionary-encoded column
  chunk now writes `Statistics.distinct_count` (its dictionary size = the number
  of distinct non-null values); PLAIN/DELTA chunks leave it absent. `distinct_count`
  is also read back into `ColumnMetaData`.

- **Bloom filters for temporal / decimal / fixed-size-binary** (`marrow.parquet`):
  `write_bloom_filter=True` now also builds filters for temporal (date/time/
  timestamp/duration — hashed over their INT32/INT64 little-endian bytes),
  decimal (`decimal32`/`decimal64` as INT32/INT64, `decimal128`/`decimal256`
  over their big-endian FIXED_LEN_BYTE_ARRAY bytes), and `fixed_size_binary`
  (raw bytes) columns — matching each type's physical value encoding. Previously
  only integer, floating-point, and byte-array columns were covered.

- **Page CRC-32 checksums** (`marrow.parquet`): `write_table(...,
  write_page_checksum=True)` (default False, like PyArrow) attaches a standard
  CRC-32 to every data/dictionary page header — over the compressed body for v1
  and the uncompressed levels + compressed values for v2, matching the spec —
  and the reader verifies it on read, raising on a mismatch. A new
  `marrow.utils.Crc32` (incremental, ISO-3309 / zlib polynomial) backs it.

- **Key/value metadata round-trip** (`marrow.parquet`): the file footer's
  `key_value_metadata` is now read and written. On read it populates
  `schema.metadata` (matching `pyarrow.read_table(...).schema.metadata`,
  including PyArrow's `ARROW:schema` blob); on write the schema's metadata is
  emitted, except `ARROW:schema` (which pins exact Arrow types — marrow writes
  and infers types from the Parquet schema, so re-emitting a foreign copy would
  make the file self-inconsistent).

- **float16 read + write** (`marrow.parquet`): the Arrow `float16` (half-float)
  type now round-trips. Parquet stores it as `FIXED_LEN_BYTE_ARRAY(2)` with the
  `FLOAT16` logical annotation, and the 2 bytes are exactly the little-endian
  half bit pattern, so it routes through the existing primitive path (PLAIN,
  dictionary, flat and nested) with IEEE-ordered, signed-zero-normalised
  min/max statistics.

- **FIXED_LEN_BYTE_ARRAY DELTA_BYTE_ARRAY / BYTE_STREAM_SPLIT read**
  (`marrow.parquet`): decimal and fixed-size-binary columns encoded with
  `DELTA_BYTE_ARRAY` or `BYTE_STREAM_SPLIT` (both emitted by PyArrow via
  `column_encoding`) now read. A single shared `Encoding.decode_flba` decodes the
  present values into a contiguous width-byte buffer, keeping the PLAIN path a
  zero-copy read.

- **Nullable-struct write** (`marrow.parquet`): a nullable Arrow struct is now
  emitted as an `OPTIONAL` group so struct-level nulls ride in the definition
  levels (previously structs were always `REQUIRED` and their null-ness was
  silently dropped on write). The struct's null bit is pushed into its children
  before shredding/encoding so the value count matches the levels. A struct whose
  subtree contains a repeated group (list/map) still stays `REQUIRED` for now.

- **INT96 timestamp read** (`marrow.parquet`): the reader decodes the deprecated
  12-byte INT96 physical type (nanoseconds-of-day + Julian day) into a
  nanosecond `timestamp`, so legacy Impala/Spark/Hive files read back — PLAIN and
  RLE_DICTIONARY, flat and nested.

- **RLE boolean read** (`marrow.parquet`): boolean values encoded as RLE — what
  arrow/PyArrow emit in DataPage v2 — now decode (a 4-byte length prefix then a
  width-1 RLE/bit-packed hybrid), alongside the existing PLAIN bit-packed path.

- **Data page splitting** (`marrow.parquet`): the writer no longer emits a single
  data page per column chunk. Each chunk is split into data pages of at most
  ~1 MiB of encoded values or 20 000 rows (matching arrow-cpp `data_pagesize` /
  `max_rows_per_page` and arrow-rs `DEFAULT_PAGE_SIZE` /
  `DEFAULT_DATA_PAGE_ROW_COUNT_LIMIT`), always breaking on a record boundary so a
  nested (list/map) row never straddles pages. A single dictionary page is shared
  by all data pages of a chunk. The writer now produces a real multi-entry
  OffsetIndex + ColumnIndex (per-page location, `first_row_index`, and
  min/max/null-count), so page-level predicate pushdown works on marrow-written
  files and large columns are no longer one giant page.

- **Bloom filters, read + write** (`marrow.parquet`): a new `bloom` module
  implements the XXH64 value hash and the split-block bloom filter (SBBF) per the
  Parquet spec / arrow-rs. `write_table(..., write_bloom_filter=True)` builds a
  filter for every integer, floating-point, and byte-array column (sized to its
  distinct-value count) and writes it with a `BloomFilterHeader`;
  `ColumnMetaData` now carries `bloom_filter_offset`/`length`.
  `ParquetFile.bloom_filter(row_group, column)` returns a `SplitBlockBloomFilter`
  whose `might_contain(bytes)` proves a value's absence with no false negatives.
  Validated both ways against the Apache `parquet-testing` reference file
  (`check("Hello")` / `check("Hello_Not_Exists")`) and by an independent
  reference reader over marrow's own output.

- **Page index write** (`marrow.parquet`): the writer now emits an `OffsetIndex`
  and (when the chunk carries bounds or is all-null) a `ColumnIndex` for every
  column chunk, written after the page data and pointed to by the footer's
  `ColumnChunk.{offset,column}_index_offset` — closing the read/write asymmetry
  where the reader consumed a page index that the writer never produced. Marrow
  writes a single data page per chunk, so each index has one entry covering all
  rows; PyArrow prunes with it (page-level predicate pushdown) and marrow reads
  it back via `read_page_index` / `read_page_bounds`.

- **Nested temporal / decimal / fixed-size-binary read** (`marrow.parquet`): a
  list or map element of a temporal type (`date32`/`time32`/`timestamp`/
  `time64`/`date64`/`duration`), `decimal128`/`decimal256`, or
  `fixed_size_binary` now decodes — previously only primitive/string/binary
  leaves worked under a repeated group and anything else raised `unsupported
  list element type`. The leveled drives grow a builder and retag the int32/
  int64 storage to the temporal Arrow type (decimals carry their precision/scale
  directly), so nested and flat paths now cover the same type set.

- **Full compression codec coverage** (`marrow.parquet`): the writer now emits
  `GZIP` (zlib deflate, windowBits 31) and `BROTLI` (via `libbrotlienc`), and
  the reader decodes `BROTLI` (via `libbrotlidec`) — closing the read/write
  asymmetry where GZIP could only be read. The deprecated `LZ4` (code 5) now
  round-trips in both directions: writers emit a plain LZ4 block (as modern
  PyArrow does) and the reader tolerates the legacy Hadoop 8-byte frame by
  stripping it when present. `brotli` is now a runtime dependency (opened via
  `dlopen`, like the other codecs).

- **Binary & large byte-array write** (`marrow.parquet`): the writer now emits
  `binary`, `large_binary`, and `large_string` columns (previously only the
  reader handled them). Parquet has a single `BYTE_ARRAY` physical type, so
  `large_*` columns are written as `BYTE_ARRAY` (large_string carrying the
  `UTF8`/`STRING` annotation) and read back as `binary`/`string` — matching
  arrow-rs / parquet-cpp. All value encodings (PLAIN, RLE_DICTIONARY,
  DELTA_BYTE_ARRAY, DELTA_LENGTH_BYTE_ARRAY) now work over any byte-array type.

- **Temporal, decimal & byte-array statistics** (`marrow.parquet`): the writer
  now emits `min`/`max` bounds for temporal (`date32`/`time32` as INT32,
  `timestamp`/`time64`/`date64`/`duration` as INT64, signed order), decimal
  (`decimal32`/`decimal64` as INT32/INT64, `decimal128`/`decimal256` as
  big-endian two's-complement `FIXED_LEN_BYTE_ARRAY` in signed numeric order),
  `binary`/`large_binary`/`large_string` (byte-wise lexicographic), and
  `fixed_size_binary`. Previously only numeric, bool, and string columns carried
  bounds; PyArrow now reads correct statistics for every written type.

- **Decimal & fixed-size-binary read** (`marrow.parquet`): the reader now decodes
  `FIXED_LEN_BYTE_ARRAY` columns — `decimal128`/`decimal256` (big-endian two's-
  complement, sign-extended from PyArrow's minimal per-precision byte width to
  the 16/32-byte int128/int256 storage) and `fixed_size_binary` (raw bytes) —
  across PLAIN and `RLE_DICTIONARY` pages. With the temporal read already in
  place, all temporal, decimal, and fixed-size-binary types now round-trip both
  directions with PyArrow.

- **Temporal, decimal & fixed-size-binary write** (`marrow.parquet`): the writer
  now emits `date32` (INT32/`DATE`), `time32`/`time64` (INT32/INT64 with
  `TIME_MILLIS`/`TIME_MICROS` and the nanosecond `TIME` `LogicalType`),
  `timestamp` (INT64/`TIMESTAMP` carrying the time unit and the
  `isAdjustedToUTC` flag derived from the Arrow timezone), `decimal128`/
  `decimal256` (big-endian two's-complement `FIXED_LEN_BYTE_ARRAY` of 16/32
  bytes with `precision`/`scale`), `decimal32`/`decimal64` (INT32/INT64), and
  `fixed_size_binary` (`FIXED_LEN_BYTE_ARRAY`). `SchemaElement` now serializes
  the full `LogicalType` Thrift union — including the nested `TimeUnit`
  (MILLIS/MICROS/NANOS) and `DECIMAL {scale, precision}` members — plus the
  `scale`/`precision` fields. PyArrow reads every type back with the correct
  annotation, including nanosecond resolution and timezones.

- **Write-side encodings** (`marrow.parquet`): the writer no longer emits only
  PLAIN. `write_table(..., use_dictionary=True)` (the default, like PyArrow)
  dictionary-encodes numeric and string columns — a PLAIN dictionary page of
  distinct values plus an `RLE_DICTIONARY` data page of bit-packed indices
  (`Rle.encode_bitpacked`), shrinking low-cardinality columns; set
  `use_dictionary=False` for PLAIN, and columns whose dictionary page would
  exceed 1 MB fall back to PLAIN automatically. The other encodings are
  selectable: `DELTA_BINARY_PACKED` (signed ints; block/miniblock zig-zag deltas
  via `DeltaBinaryPacked.encode`), `DELTA_BYTE_ARRAY` / `DELTA_LENGTH_BYTE_ARRAY`
  (strings), and `BYTE_STREAM_SPLIT` (floats). Encoding is chosen per leaf, most
  specific first — a `column_encodings` name→encoding map, then a single global
  `encoding`, then the dictionary/PLAIN default; an override that does not fit a
  column's type is ignored. Nulls are placed by the definition levels, so every
  encoding composes with the flat and nested (Dremel) write paths; the encoders
  mirror the reader's decoders and PyArrow reads every variant back.
  `ColumnMetaData` now advertises the real encodings and the dictionary page
  offset.

- **Map type** (core): a first-class Arrow `map<k, v>` — `MapType`/`map_()` in
  `dtypes`, `MapArray`/`MapArray.from_arrays(offsets, keys, items)` in `arrays`
  (physically a list of a non-nullable `entries` struct, so it reuses
  `ListLikeArray[MapType]`), `MapBuilder` (composed over an entries-struct
  `ListBuilder`, so maps flow through `concat`/`combine_chunks`), and the Arrow C
  Data Interface `+m` format in both directions (with the `keys_sorted` flag).
  Following arrow-rs (`DataType::Map(field, sorted)`) and Arrow C++
  (`MapType::value_field()`), `MapType` stores the entries struct as a single
  `Field`, preserving key/value field names and nullability. The retag between a
  list and a map lives in one place: `ListArray.to_map()` / `MapArray.to_list()`.

- **Parquet map columns, read + write** (`marrow.parquet`): a Parquet `MAP`
  (`optional group(MAP) { repeated group key_value { required key; value } }`,
  incl. the legacy `MAP_KEY_VALUE` annotation) now reads back as a `MapArray` and
  writes from one. A map reconstructs with the exact same Dremel machinery as
  `list<struct<key, value>>` — only the final array tag differs — so it composes
  to any depth (`map<k, list<v>>`, `list<map>`, …). PyArrow reads the maps marrow
  writes and vice versa.

- **Nested write (lists + maps)** (`marrow.parquet`): the writer gained a general
  Dremel *shredding* path — it emits `LIST`/`MAP` schema groups with correct
  repetition/definition levels, strips a nested column into per-leaf rep/def
  level streams (`SchemaNode.shred_levels`), and encodes multi-bit RLE levels
  (v1 and v2 data pages). Flat/struct columns keep their fast path untouched;
  columns containing a repeated group are shredded. This closes the previously
  open list-write gap. A single `ColumnWriter._emit_page` now serializes every
  data page, and the map's schema geometry lives in one shared `_map_node`
  builder used by both read and write.

- **Min/max statistics** (`marrow.parquet`): the writer now computes and emits
  per-column-chunk `min_value`/`max_value` bounds (with `is_min/max_value_exact`)
  alongside the existing `null_count`, and declares `column_orders`
  (`TypeDefinedOrder`) so readers trust the logical ordering. Bounds use the
  correct comparator per type — signed vs unsigned integers, IEEE floats
  (NaN-skipped, signed-zero normalised so the bound brackets ±0.0), and
  byte-wise string ordering — and are PLAIN-encoded (little-endian numerics; raw
  bytes for strings). PyArrow reads the bounds marrow writes, including the
  unsigned-int and string cases. On read, `read_metadata(path)` exposes the raw
  footer (row groups, offsets, codecs, `null_count`, and the min/max bytes)
  mirroring `pyarrow.parquet.read_metadata`, and `read_statistics(path)` returns
  decoded typed `min`/`max` scalars per (row group, leaf column) for the numeric,
  boolean, and string types (temporal/binary bounds are a follow-up). This is the
  foundation for row-group/page skipping (predicate pushdown).

- **Arbitrarily nested read support** (`marrow.parquet`): the reader now
  reconstructs any nesting of structs and lists — `list<struct>`, `list<list<…>>`
  to any depth, `struct<list>`, lists of nullable structs, and struct-level nulls
  at any position (top level, holding a list, or as a list element). A nullable
  struct reads back as null (not a struct of null fields), with field-null and
  struct-null distinguished. Each schema node carries its Dremel geometry
  (`NodeGeom`: `present_def`, `rep_level`, `element_floor`, `entry_floor`,
  `optional`), computed once during schema parsing, so every list's offset scan
  is self-contained and the assembler composes by recursion to any depth — no
  per-level special-casing. Flat leaves under a nullable struct keep their def
  levels (`LeafColumn.carry_def`); all others stay on the fast path.

- **DELTA_BYTE_ARRAY / DELTA_LENGTH_BYTE_ARRAY read support** (`marrow.parquet`):
  the reader now decodes the delta string/binary encodings (PyArrow
  `use_dictionary=False`) — DELTA_LENGTH_BYTE_ARRAY (delta-packed lengths then
  concatenated bytes) and DELTA_BYTE_ARRAY (incremental prefix + suffix
  reconstruction). Handles nulls; reads compressed and uncompressed. With
  DELTA_BINARY_PACKED and BYTE_STREAM_SPLIT this completes the common
  non-dictionary encodings.

- **BYTE_STREAM_SPLIT read support** (`marrow.parquet`): the reader now decodes
  the BYTE_STREAM_SPLIT encoding (float32/float64; PyArrow
  `use_byte_stream_split=True`), reassembling each value from its strided byte
  planes. Handles nulls; reads compressed and uncompressed.

- **DELTA_BINARY_PACKED read support** (`marrow.parquet`): the reader now decodes
  the DELTA_BINARY_PACKED integer encoding (block / miniblock zigzag deltas) that
  modern writers — PyArrow with `use_dictionary=False`, and v2 defaults — emit for
  int32/int64 columns. Handles negatives, nulls, and multi-block streams;
  previously it raised "unsupported data page encoding". Reads compressed and
  uncompressed.

- **DataPage V2 write support** (`marrow.parquet`): the writer can now emit v2
  data pages (`write_table(..., version=2)` in Mojo, `data_page_version="2.0"`
  in Python) in addition to the default v1 — the reader already read both. v2
  stores the definition levels uncompressed ahead of the compressed values
  (`DataPageHeaderV2` with `is_compressed`), matching arrow-rs's
  `WriterVersion` (PARQUET_1_0 default, PARQUET_2_0 opt-in). PyArrow reads
  marrow's v2 output and marrow round-trips it, compressed and uncompressed.

- **Column projection on read** (`marrow.parquet`): `read_table(path,
  columns=[...])` reads only the named top-level columns, in the given order —
  only those columns' chunks are decoded (the rest are never touched). Works for
  flat, struct (whole subtree), and list columns; raises on an unknown name.
  Implemented by selecting the assembly nodes and remapping their leaf indices
  onto a compact decoded grid, so the parallel decode skips unselected columns
  entirely. Exposed through the Python binding as `columns=` too.

- **Native Parquet reader/writer** (`marrow.parquet`): a from-scratch Parquet
  implementation that reads and writes Arrow directly, replacing the PyArrow
  bridge (`read_table`/`write_table` are now native; PyArrow is only a test
  oracle). Includes a hand-written Thrift Compact Protocol codec
  (`thrift.mojo`) and metadata structs (`format.mojo`) — no Thrift runtime or
  code generator — modelled on arrow-rs's `parquet_thrift.rs`; page/level
  decoding via the RLE/bit-packed hybrid and PLAIN encodings (`encoding.mojo`);
  and Snappy/Zstd compression through runtime `dlopen` FFI (`compression.mojo`,
  new `zstd`/`snappy` conda deps). Covers flat columns (all common primitives,
  string/binary), definition-level nullability, dictionary (RLE_DICTIONARY /
  PLAIN_DICTIONARY) and PLAIN encodings, v1 and v2 data pages, multiple row
  groups, and struct nesting. The reader additionally handles int8/16 &
  uint8/16, temporal (date32, timestamp incl. nanosecond, time32/64),
  binary/large variants, GZIP/LZ4_RAW compression, and single-level
  List/LargeList columns (Dremel repetition levels). The writer emits multiple
  row groups, per-column null-count statistics, and widens narrow ints. Map
  columns, struct-level nulls, list/temporal writing, dictionary-encoding on
  write, and min/max statistics are follow-ups (all raise a clear error where
  unsupported). The reader mmaps the file, decodes
  fixed-width PLAIN pages straight into the output buffer (memcpy fast path),
  counts definition levels without materializing them for no-null columns, and
  SIMD-unpacks RLE/dictionary index streams eight values at a time (one 64-bit
  load per lane, then a vector shift + mask) — matching single-threaded PyArrow
  on PLAIN data and beating both PyArrow (~3.1×) and polars (~1.6×) on
  dictionary-encoded columns.

- **Parallel Parquet reads** (`marrow.parquet`): `read_table` decodes every
  (row group, leaf column) pair concurrently across `num_physical_cores()`
  workers — each reads a disjoint byte range of the shared read-only mmap and
  writes its own result slot, and each owns a `Codecs` (the lazy `dlopen`
  handles are not shareable). Files below 4096 rows stay single-threaded to
  avoid dispatch overhead. ~4.9× faster on a 2M×8 multi-row-group snappy file
  (54 ms → 11 ms on 16 cores), bringing multi-column reads level with polars and
  PyArrow. ASAN-clean under the concurrent path.

- **LZ4 (LZ4_RAW) write support** (`marrow.parquet`): the writer can now emit
  `CODEC_LZ4_RAW` (via `LZ4_compress_default`), joining UNCOMPRESSED/SNAPPY/ZSTD;
  LZ4_RAW was already readable. PyArrow reads marrow's LZ4 output and vice versa
  (covered by the interop suite). LZ4 is the fastest real codec through the
  reader — a 2M×8 file reads in 6.9 ms vs 10.7 (snappy) / 15.3 (zstd) at
  essentially snappy's file size (58 vs 59 MB), so LZ4 gives near-uncompressed
  read speed with compression.

- **AOT typed tables declared as plain dtype-tag structs** (`marrow.aot.relations`):
  a plain struct declares its columns as bare dtype fields (`var a: Int64Type`,
  `var name: StringType`) with no column-node wrappers and no `__init__`, and
  `Table[Orders]()` is a column-access handle whose `t.a` / `t.name` reflect
  each field's dtype into `NumericColumn[T]` / `StringColumn` (numeric vs string
  is dispatched by a `where` clause on the reflected field type). Replaces the
  previous `var a: NumericColumn[Orders, "a", Int64Type]`-style fields +
  hand-written `__init__` boilerplate. The named columns carry only a runtime
  `name` (the sole type parameter is the dtype), so a query with N same-dtype
  columns instantiates one column type, not N — the name never affects the
  generated SIMD compute, and the position is resolved by name against the batch
  schema at execution. The positional and named numeric column nodes are renamed
  `Column` → `NumericColumn` to pair with `StringColumn` per type family, and
  both named leaves share a new `Column` base trait exposing `to_array()`, so
  `Project[*Es: Column]` assembles a projection with no numeric-vs-string
  branching.
- **Polars-style `col(name, dtype)` column factory** (`marrow.aot.relations`):
  `col("a", int64)` / `col("name", string)` reference a column by name without a
  schema struct or handle — overloaded on the dtype's trait so the numeric case
  returns `NumericColumn[T]` and the string case `StringColumn`, both fully
  composable (`Add(col("a", int64), col("b", int64))`, `Project`/`Filter`).
  Produces the same name-carrying leaf as `Table[Tbl]()`; the two differ only in
  whether the dtype is read off a struct or spelled explicitly.

- **`marrow.aot` — a fully-monomorphized (AOT) relational layer**:
  `Schema.from_struct[T]()` (`marrow/schema.mojo`) derives a `Schema` from a
  marker struct via compile-time reflection; `Table`, `Column[Tbl, name, T]`,
  `StringColumn[Tbl, name]` (`marrow/aot/relations.mojo`) resolve a column's
  position as a `comptime` constant via `reflect[Tbl].field_index[name]()` —
  no runtime `Schema` lookup, ever; `BoolValue` + `Lt`/`Gt`/`Eq`
  (`marrow/aot/values.mojo`) give fused, bit-packed-`BoolArray` comparisons;
  `Project[*Es]`/`Filter[Input, Pred]` (`marrow/aot/relations.mojo`) compile a
  `SELECT`/`WHERE`-shaped query into fused SIMD loops with no tag dispatch.
  See `docs/aot-relations-design.md`.
- **`Expr`'s `FUSED` boxing constructor now also accepts `BoolValue` nodes**
  (`marrow/dyn/values.mojo`), not just `NumericValue` — lets a comptime
  `Lt`/`Gt`/`Eq` predicate drive a runtime `AnyRelation.filter()` plan.
- **Binary-size benchmark** (`benchmarks/binary_size/`): three files
  implement the identical query via `marrow.aot`, `marrow.dyn`, and a hybrid
  (runtime plan + AOT-fused predicate), showing the fully-monomorphized
  version compiles ~33x smaller (stripped). `pixi run binary_size` runs
  `compare.py`, which builds, strips, and reports a size/symbol-count table
  plus a per-module symbol breakdown.
- **String `Length` expression node + `.length()`** (`marrow/exprold/values.mojo`,
  `marrow/exprold/runtime.mojo`, `marrow/kernels/string.mojo`): computes
  per-element string byte lengths through both expression layers. Adds a
  `StringValue` trait (mirrors `NumericValue` but resolves to a `StringArray`
  instead of a per-lane SIMD `core[W]()`) and a `StringColumn` leaf node; the
  comptime `Length[S: StringValue]` node implements `NumericValue` with a
  SIMD-vectorized `core[W]()` that loads `W+1` contiguous string offsets and
  subtracts the shifted-by-one lanes, so it composes into a fused pass with
  other numeric nodes. The runtime `Expr` gains a `LENGTH` tag and `.length()`
  method that dispatch to a new type-erased `string_lengths(AnyArray)`
  overload, matching the existing typed-overload-plus-`AnyArray`-blanket
  kernel pattern.

- **`Schema[Field[...]]` with `__getattr_param__`** (`marrow/faszom.mojo`): compile-time
  schema type that enables Ibis-style `t.data.where(t.a + t.b > t.c).execute(batch)`
  syntax without per-field boilerplate. `Schema[Field['a', Int32Type], Field['data', Float32Type]]`
  returns `ColumnRef['a', Int32Type]` for `t.a` via `__getattr_param__`, using a
  `@staticmethod def _name_matches` trait + compile-time recursive index lookup
  (`_schema_find_idx`) to resolve the field type at compile time.

- **ColumnRef / Pipeline / FilterPipeline** (`marrow/faszom.mojo`): named column
  placeholders (`ColumnRef['name', T]`) resolved from a `RecordBatch` at execute
  time via `bind()`. Enables reusable AOT-compiled query pipelines that are defined
  once and called per batch. Convenience factories: `col['name'](dtype)`,
  `filter_pipeline['data_col'](pred, dtype)`. `FilterPipeline` and `Pipeline`
  wrappers bind all `ColumnRef` nodes in `O(cols)` and execute the fused loop in
  `O(N)`. The AOT specialization property is preserved — each distinct
  `(name, T)` pair remains a unique compile-time type.

- **`PrimitiveArray.__eq__` correctness fix** (`marrow/arrays.mojo`): the fast
  path now compares only the valid `length` elements instead of the full allocated
  buffer, preventing spurious mismatches for filtered arrays whose backing buffer
  is over-allocated.

- **Sort kernel — `argsort` and `sort`** (`marrow/kernels/sort.mojo`):
  single-column sort for all array types. Primitive arrays use LSD radix sort
  (O(N), 8-bit passes, UInt64-encoded keys, float NaN/sign-bit transform) for
  N ≥ 32 768, with parallel histogram + scatter for N ≥ 524 288. PDQsort for
  N < 32 768 (faster on Apple M-series up to ~28K elements); insertion-sort
  leaf for N < 32. `BoolArray` uses O(N) counting sort; `StringArray` uses the
  Mojo stdlib comparison sort. Null partitioning (pre-sort bitmap scan) with
  `nulls_first`/`nulls_last` placement. `sort(StructArray, key_indices,
  ascending)` wraps `argsort` + `take` for multi-column sort.

- **Large binary, string, and list types** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  added `LargeBinaryType`, `LargeStringType`, `LargeListType` (64-bit offsets);
  `BinaryLikeType` trait with `comptime offset: DType` and `StringLikeType` sub-trait
  for UTF-8 kernels; unified `BinaryArray[T: BinaryLikeType]` and
  `BinaryBuilder[T: BinaryLikeType]` with aliases `StringArray`, `LargeBinaryArray`,
  `LargeStringArray`, `StringBuilder`, `LargeBinaryBuilder`, `LargeStringBuilder`;
  IPC type codes 19/20/21 for large binary/utf8/list; C Data format codes `Z`/`U`/`+L`.

- **IPC support for dictionary-encoded columns** (`marrow/ipc.mojo`): the IPC
  file and stream writer now emits a `DictionaryBatch` message (header type 2)
  for each dictionary column before its first `RecordBatch`, encoding the
  column's value array as a separate body. The `RecordBatch` body carries only
  the integer indices. Dictionary blocks are registered in the IPC file footer so
  C++ / Rust / Go readers can locate them. The IPC reader detects
  `DictionaryEncoding` at schema-field slot 4, reconstructs `DictionaryType`
  (index type + value type + ordered flag), loads `DictionaryBatch` messages via
  footer-registered block offsets, and wires the decoded values back into
  `DictionaryArray` instances when reading record batches. Validated across all
  Arrow implementations (`dictionary` and `dictionary_unsigned` pass 14/14
  integration phases with C++, Rust, and Go).

- **Arrow interval types** (`marrow/{dtypes,scalars,arrays,builders,ipc,c_data}.mojo`, `python/`):
  added `IntervalType` trait and three concrete types — `YearMonthIntervalType` (int32, months),
  `DayTimeIntervalType` (int64, days+millis), `MonthDayNanoIntervalType` (int128, months+days+nanos).
  `AnyDataType` gains `is_interval()`, `is_year_month_interval()`, `is_day_time_interval()`,
  `is_month_day_nano_interval()` predicates and matching `as_*` accessors. Array, builder, and
  scalar aliases (`YearMonthIntervalArray/Builder/Scalar`, etc.) are fully wired into the
  `AnyArray`, `AnyBuilder`, and `AnyScalar` type-erased containers. C Data Interface uses
  format codes `tiM`, `tiD`, `tin`; IPC uses the `Interval` flatbuffer type with unit field.
  Python bindings expose `year_month_interval()`, `day_time_interval()`,
  `month_day_nano_interval()` factory functions.

- **Dictionary-encoded Arrow type** (`marrow/{dtypes,scalars,arrays,builders,
  c_data}.mojo`): added `DictionaryType` (index type + value type + ordered
  flag), `DictionaryScalar`, `DictionaryArray`, and `DictionaryBuilder`.
  `DictionaryArray.from_arrays(indices, values)` constructs from an integer
  indices array and an arbitrary values array; `__getitem__` decodes to the
  underlying value scalar; `slice()` is zero-copy. The C Data Interface emits
  the index type's format string and stores the value schema in the `dictionary`
  field of `CArrowSchema`, with `ARROW_FLAG_DICT_ORDERED = 1` when ordered;
  import detects a non-null `dictionary` field and reconstructs the type.
  Enables zero-copy exchange of PyArrow `DictionaryArray` via the Arrow C Data
  Interface (`__arrow_c_array__` / `__arrow_c_schema__` protocol).

- **Arrow Null type** (`marrow/{arrays,scalars,builders,ipc,c_data}.mojo`,
  `python/arrays.mojo`): added `NullArray`, `NullScalar`, `NullBuilder`
  (registered in the `AnyArray`, `AnyScalar`, `AnyBuilder` variants); IPC
  writer emits `Type.Null = 1` with zero body buffers; IPC reader skips the
  validity slot for null fields; C Data Interface uses `n_buffers = 0` for null
  per the spec; Python factory `ma.array(seq, type=ma.null())` builds a
  `NullArray` of the given length.

- **Fixed-size binary type** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  added `FixedSizeBinaryType`, `FixedSizeBinaryArray`, `FixedSizeBinaryBuilder`;
  C Data format code `"w:<n>"`; IPC type code 15 (FixedSizeBinary).

- **Temporal array types** (`marrow/{dtypes,arrays,builders,ipc,c_data}.mojo`):
  `Date32Array`, `Date64Array`, `Time32Array`, `Time64Array`, `TimestampArray`,
  `DurationArray` with matching builders and type singletons; C Data format
  codes (`"tdD"`, `"tdm"`, `"tts"`, `"ttu"`, `"tsn:"`, `"tDn"`, etc.); IPC
  type codes and unit serialisation. Python constructors `ma.date32()`,
  `ma.date64()`, `ma.time32(unit)`, `ma.time64(unit)`, `ma.timestamp(unit)`,
  `ma.duration(unit)`.

- **Decimal types in C Data Interface and IPC**
  (`marrow/c_data.mojo`, `marrow/ipc.mojo`): wired `Decimal32Type`,
  `Decimal64Type`, `Decimal128Type`, `Decimal256Type` into schema export/import
  and IPC flatbuffer serialisation (precision, scale, bit-width).

- **Custom metadata round-trip via the C Data Interface**
  (`marrow/c_data.mojo`): `CArrowSchema.from_field` / `from_schema` now
  encode `Field.metadata` and `Schema.metadata` into the spec-defined
  metadata blob; `to_field` / `to_schema` decode it back. New
  `_encode_c_metadata` / `_decode_c_metadata` helpers handle the
  `int32 num_pairs ; (int32 key_len, key_bytes, int32 val_len, val_bytes)*`
  layout. `from_schema` now takes a full `Schema` rather than `List[Field]`
  so schema-level metadata flows through.

- **Per-field metadata** (`marrow/dtypes.mojo`, `python/dtypes.mojo`):
  `Field` carries an optional `metadata: Dict[String, String]`; the Python
  factory `ma.field(name, type, metadata={…})` accepts a dict; the C Data
  Interface and IPC flatbuffer encoder/decoder round-trip field-level
  key-value metadata.

- **Preserve nested-field names in IPC reader and C Data Interface**
  (`marrow/ipc.mojo`, `marrow/c_data.mojo`): the IPC `_read_field`
  decoder and the `CArrowSchema` list / fixed_size_list importer now preserve
  child Field names as-is, so Arrow files written by other implementations
  round-trip with the original schema.

- **Arrow IPC reader/writer** (`marrow/ipc.mojo`): `read_ipc_file()`,
  `write_ipc_file()`, `read_ipc_stream()`, `write_ipc_stream()`,
  `read_ipc_file_schema()`, `read_ipc_stream_schema()`, and streaming struct
  variants `RecordBatchFileReader`, `RecordBatchStreamReader`,
  `RecordBatchFileWriter`, `RecordBatchStreamWriter`. Supports all implemented
  Arrow types (bool, int8–64, uint8–64, float16/32/64, binary, utf8, list,
  fixed_size_list, struct, dictionary, null, temporal, decimal) with full
  nested and nullable column support. FlatBuffer encoding/decoding is a
  self-contained Rust-faithful port with correct soffset sign convention and
  `MetadataVersion::V5`.

- **GPU aggregate reductions** (`marrow/kernels/aggregate.mojo`):
  `sum_`, `min_`, `max_`, `product`, `any_`, `all_` now accept an
  `ExecutionContext`; when `.is_gpu()` is true the reduction runs as a
  single-pass GPU kernel via `_reduce_generator_wrapper`.

- **`ExecutionContext`** (`marrow/kernels/execution.mojo`): new struct bundling
  `num_threads` for CPU stripe parallelism and `device: Optional[DeviceContext]`
  for GPU. Implicit conversions from `Optional[DeviceContext]` and
  `DeviceContext` keep existing callers working. Factories: `.serial()`,
  `.parallel(num_threads=0)` (0 = `num_physical_cores()`), `.gpu(device)`.
  Wired through all kernels: arithmetic, aggregate, compare, filter, join, sort.

- **Partition-parallel hash join** (`marrow/kernels/join.mojo`,
  `marrow/kernels/hashtable.mojo`): `HashJoin` and `hash_join()` gain a
  `num_threads` argument. The parallel path radix-partitions both sides by the
  top bits of their hash into independent `SwissHashTable` instances, builds and
  probes them concurrently via `sync_parallelize`, and concatenates per-partition
  index pairs. No atomics on the hot path. At 10M×10M INNER join: **330 ms
  (serial) → 67 ms (parallel, 4.9× speedup)** — faster than Polars (97 ms),
  PyArrow (111 ms), and DuckDB (122 ms).

- **`RadixPartitioner`** (`marrow/kernels/hashtable.mojo`): partitions hashes +
  row indices by the top `num_bits` (default 6 → 64 partitions). Per-thread
  histogram → partition-major prefix sum → parallel scatter into shared flat
  buffers, then per-partition zero-copy slice via `ArcPointer`-shared immutable
  buffers.

- **Parallel per-column `take()`** (`marrow/kernels/filter.mojo`):
  `take[T](PrimitiveArray, indices, ctx)` and the `AnyArray` dispatcher
  accept an `ExecutionContext` and stripe the no-null fast path across workers.
  End-to-end 10M inner join assembly: **143 ms → 67 ms**.

- **Variant-based dispatch for `DataType`, `AnyArray`, and `Builder`**
  (`marrow/dtypes.mojo`, `marrow/arrays.mojo`, `marrow/builders.mojo`):
  Replaced integer-code dispatch with `Variant`-backed types using `comptime
  for` loops. Eliminates runtime `if`/`elif` chains across kernels, Python
  bindings, and the expression system.

- **`BoolArray` dedicated type** (`marrow/arrays.mojo`): bit-packed boolean
  arrays backed by a `Bitmap`, with `.values() -> BitmapView`, GPU transfer,
  and a matching `BoolBuilder`.

- **`BufferView` / `BitmapView` abstractions** (`marrow/views.mojo`):
  type-safe, non-owning views with `apply` dispatch, `compressed_store`,
  `pext`, and GPU-aware access.

- **`SwissHashTable`** (`marrow/kernels/hashtable.mojo`): open-addressing hash
  table with 7-bit control stamps, CSR chain storage, vectorised SIMD group
  matching, and a batch-build API.

- **Hash join** (`marrow/kernels/join.mojo`): `hash_join` kernel using
  `SwissHashTable` with separate build and probe phases.

- **`TestSuite` and `BenchSuite` framework** (`marrow/testing`):
  auto-discovery of `test_*` / `bench_*` functions via
  `__functions_in_module()`, with pytest harness integration, competition
  tables, and per-element throughput metrics.

- **AddressSanitizer support**: `pytest --asan` compiles test runners with ASAN
  instrumentation via `libcompiler-rt`.

- **GPU `BitmapView` and GPU rapidhash** (`marrow/kernels/`): `BitmapView`
  supports device-resident bitmaps; `rapidhash` ported to Metal/CUDA with
  128-bit multiply emulation.

- **Bounds checking** (`marrow/buffers.mojo`): `Buffer`, `Bitmap`, and
  `BufferView` accessors assert bounds in debug builds.

- **Unary math kernels** (`marrow/kernels/arithmetic.mojo`): `sign`, `sqrt`,
  `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `floor`, `ceil`, `trunc`,
  `round`, `sin`, `cos` (floating-point), plus binary `pow_`, `floordiv`, `mod`.

- **Scalar types** (`marrow/scalars.mojo`): `PrimitiveScalar[T]`,
  `StringScalar`, `ListScalar`, `StructScalar`, `AnyScalar` — typed and
  type-erased scalar values mirroring the array hierarchy.

- **Group-by kernel** (`marrow/kernels/groupby.mojo`): fused
  `groupby(keys, values, aggregations)` that hashes, groups, and aggregates in
  a single pass. Supports `"sum"`, `"min"`, `"max"`, `"count"`, `"mean"`.
  Single-key (any primitive/string `AnyArray`) and multi-key (`StructArray`)
  grouping.

- **Hashing kernel** (`marrow/kernels/hashing.mojo`): `hash_` for primitive,
  string, and struct arrays; `hash_identity` for bool/uint8/int8.

- **Expression execution system** (`marrow/exprold/`): pull-based streaming query
  executor with `col()`, `lit()`, `if_else()`, relational plan nodes
  (`InMemoryTable`, `Filter`, `Project`, `ParquetScan`, `Aggregate`), and
  `execute()` to collect `RecordBatch` results.

- **Parquet I/O** (`marrow/parquet.mojo`): `read_table(path)` and
  `write_table(table, path)` via the Arrow C Stream Interface.

- **Comparison kernels** (`marrow/kernels/compare.mojo`): `equal`,
  `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` for typed and
  runtime-typed arrays; null-propagating; GPU variants available.

- **String kernels** (`marrow/kernels/string.mojo`): `string_lengths` returns
  byte lengths for each element.

- **RecordBatch column operations** (`marrow/tabular.mojo`): `slice`,
  `select`, `rename_columns`, `add_column`, `append_column`, `remove_column`,
  `set_column`, `to_struct_array`.

- **Table enhancements** (`marrow/tabular.mojo`): `Table.from_batches`,
  `Table.to_batches`, `Table.combine_chunks`.

- **Schema enhancements** (`marrow/schema.mojo`): `get_field_index`, `field`
  lookup by name, `names()`, equality operators, Python interop via Arrow C
  Data Interface.

- **Self-contained archery integration suite** (`integration/`, `pixi.toml`):
  `pixi run integration` clones apache/arrow + arrow-rs + arrow-go, builds all
  reference implementations, and runs cross-implementation tests against C++,
  Rust, Go, and Mojo. All four implementations pass: 119 cases across 14
  directional phases.

### Refactors

- **Drop delegating kernel convenience wrappers** (`arithmetic`, `boolean`,
  `compare`, `aggregate`, `cast`): the free-standing typed (`add[T]`, `ceil[T]`,
  `equal[T]`, `sum[T]`, `cast[From, To]`, …) and type-erased (`add`, `ceil`,
  `not_equal`, `sum`, …) functions that merely delegated to the kernel structs
  are removed. Callers now use the kernel structs directly — `AddKernel.apply[T]`
  / `EqKernel.apply[T]` / `SumKernel.apply[T]` / `NumericCast.apply[From, To,
  safe]` for the typed path, and `AddKernel.dispatch` / `NeKernel.dispatch` /
  `SumKernel.dispatch` / `MeanKernel.reduce` for the runtime-typed path. The
  expression layer is the primary Mojo-side API; the Python bindings keep thin
  PyArrow-compatible wrappers. Kernels that carry real logic beyond delegation
  are retained as free functions pending a struct-first port: `equal`'s
  string / struct / `AnyArray` dispatch, `is_null` / `select`, `any` / `all`,
  and the top-level `cast` dtype router.
