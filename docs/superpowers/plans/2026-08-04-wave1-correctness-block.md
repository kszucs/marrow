# Wave 1 Correctness Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Q7.4 (scalar-RHS string predicates splat and recompile per row), Q7.3 (`count` has two grouped implementations and a false documented invariant), and B8 (`is_primitive()` disagrees with the `PrimitiveType` trait, so `byte_width()` aborts for bool).

**Architecture:** Three independent changes, one commit each, in the order Q7.4 → Q7.3 → B8. Q7.4 goes first because it adds a trait method and so needs its own binary-size reading, isolated from Q7.3, which moves the aggregate size gates. Q7.3 is measurement-gated: a decision rule is fixed in this plan *before* any number is seen.

**Tech Stack:** Mojo (pinned `>=1.0.0b3.dev2026072406,<2`), pixi environments (`dev` for tests/format, default for `binary_size`), pytest harness with a generated Mojo test driver.

**Spec:** `docs/superpowers/specs/2026-08-04-wave1-correctness-block-design.md`

## Global Constraints

These apply to **every** task below.

- **Always use `def`, never `fn`.** `fn` is deprecated.
- **Never use `alias` — use `comptime`.**
- **Relative imports only** for `marrow.*` (`...kernels.x` from `marrow/<sub>/tests/`). Absolute `from marrow.x import` fails inside the package.
- **Test case names must be unique across the entire suite**, not just per file.
- **Never call `_underscore_prefixed` members from outside the defining type.**
- **`pixi run -e dev precompile` must stay at 0 errors and 0 warnings.** Run it after every code change, before running tests.
- **`benchmarks/` is outside `marrow/`**, so `precompile marrow` does **not** compile it. Only `pixi run binary_size` does. A clean precompile is not sufficient evidence for a change to any public signature.
- **Never select `marrow/kernels/tests/test_join.mojo` alone** — it deadlocks the toolchain (B23). Select the directory instead.
- **Never leave a `marrow.mojoc` where a runner can see it.** If a run fails in under a second with `module 'test_x' does not contain 'test_y'`, run `rm -f .test_runners/marrow.mojoc marrow.mojoc` first.
- **A/B benchmarks must be interleaved**, never all-after-then-all-before. Every task here achieves that by putting both variants in one binary.
- **Size gate baselines** (`__text`, from `pixi run binary_size`):
  - `query_streaming` **1,309,024**
  - `query_streaming_agg_fused` **3,748,532**
  - `query_streaming_agg` **4,122,292**
  - `query_join` **3,780,276**
- **Test baseline to beat:** 1,951 passing (523 `marrow/kernels/tests`, 807 `marrow/expr/tests` + `marrow/tests`, 621 `marrow/parquet/tests` + `python/marrow/tests`).
- **Add a `CHANGELOG.md` entry** under `## [Unreleased]` for each of the three changes.
- **Conventional commits**, and end each commit message with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `marrow/kernels/string.mojo` | `StringPredicateKernel` trait gains `apply_scalar`; `LikeKernel`/`ILikeKernel` override it | 1 |
| `marrow/kernels/tests/test_string.mojo` | Kernel-level agreement between `apply` and `apply_scalar` | 1 |
| `marrow/expr/values.mojo` | `StringPredicate.prepare` takes the scalar branch when `R.OutShape == 0` | 2 |
| `marrow/expr/tests/test_strings.mojo` | Expr-level agreement, literal vs column RHS | 2 |
| `marrow/kernels/tests/bench_groupby.mojo` | Grouped `count` benchmarks for both implementations | 3 |
| `marrow/expr/aggregates.mojo` | `CountValid.resolve` — which implementation numeric columns get | 4 |
| `marrow/kernels/aggregate.mojo` | `CountAgg` docstring — made true either way | 4 |
| `marrow/dtypes.mojo` | `is_primitive()` excludes bool; `is_fixed_size()` re-adds it | 5 |
| `marrow/tests/test_dtypes.mojo` | Pins the new predicate contract and the divergence | 5 |

---

## Task 1: `StringPredicateKernel.apply_scalar`

**Files:**
- Modify: `marrow/kernels/string.mojo:297-345` (trait), `:725-780` (`LikeKernel`, `ILikeKernel`)
- Test: `marrow/kernels/tests/test_string.mojo`

**Interfaces:**
- Consumes: existing `LikePattern[ignore_case]`, `_match_pattern[T, ignore_case]`, `Bitmap.alloc_zeroed`, `BinaryLikeArray[T]`.
- Produces: `StringPredicateKernel.apply_scalar[T: StringLikeType](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray`. Task 2 calls exactly this.

- [ ] **Step 1: Write the failing test**

Append to `marrow/kernels/tests/test_string.mojo`:

```mojo
def test_apply_scalar_agrees_with_apply_for_like() raises:
    """The scalar-pattern path must produce exactly what the array x array path
    does when every row of the right operand is the same pattern.

    This is the invariant that makes the expr-layer optimisation safe: it is a
    pure performance change, so any disagreement here is a correctness bug.
    """
    var b = StringBuilder(capacity=5)
    b.append("hello")
    b.append("help")
    b.append_null()
    b.append("world")
    b.append("")
    var arr = b.finish()

    var pb = StringBuilder(capacity=5)
    for _ in range(5):
        pb.append("hel%")
    var pats = pb.finish()

    var via_array = LikeKernel.apply(arr, pats)
    var via_scalar = LikeKernel.apply_scalar(arr, "hel%")
    assert_true(via_array == via_scalar)


def test_apply_scalar_agrees_with_apply_for_equality() raises:
    """The default `apply_scalar` body — the one every non-LIKE predicate
    inherits — must match its array x array counterpart too."""
    var b = StringBuilder(capacity=4)
    b.append("x")
    b.append_null()
    b.append("y")
    b.append("x")
    var arr = b.finish()

    var pb = StringBuilder(capacity=4)
    for _ in range(4):
        pb.append("x")
    var pats = pb.finish()

    assert_true(StringEqKernel.apply(arr, pats) == StringEqKernel.apply_scalar(arr, "x"))
    assert_true(StartsWithKernel.apply(arr, pats) == StartsWithKernel.apply_scalar(arr, "x"))


def test_apply_scalar_empty_pattern_and_all_wildcards() raises:
    """Two shapes `LikePattern` special-cases, pinned through the new entry."""
    var b = StringBuilder(capacity=3)
    b.append("")
    b.append("a")
    b.append_null()
    var arr = b.finish()

    var empty = LikeKernel.apply_scalar(arr, "")
    assert_true(empty[0].value())
    assert_true(not empty[1].value())
    assert_true(empty.is_null(2))

    var any = LikeKernel.apply_scalar(arr, "%")
    assert_true(any[0].value())
    assert_true(any[1].value())
    assert_true(any.is_null(2))
```

Add to that file's import block whatever of `StringBuilder`, `LikeKernel`, `StringEqKernel`, `StartsWithKernel`, `assert_true` is not already imported — check the existing block first rather than assuming.

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest marrow/kernels/tests/test_string.mojo
```

Expected: a compile error naming `apply_scalar` as unresolved. Because one selection is one compilation unit, **every** case in the file will report as failed — that is the expected red, not a sign of wider breakage.

- [ ] **Step 3: Add the trait method**

In `marrow/kernels/string.mojo`, inside `trait StringPredicateKernel`, after the existing `apply`:

```mojo
    @staticmethod
    def apply_scalar[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        """`array × one constant pattern`, without materialising the constant.

        The peer of `apply` for the case where the right operand is a scalar.
        `apply` needs a `BinaryLikeArray` on both sides, so the expression layer
        had to splat a literal into an n-row array first — n copies of the same
        string, allocated per morsel.

        The default body is the same loop as `apply` with the right operand
        hoisted. `LikeKernel`/`ILikeKernel` override it to compile their pattern
        once instead of per row.
        """
        var n = len(array)
        var data = Bitmap.alloc_zeroed(n)
        for i in range(n):
            if array.is_valid(i) and Self.predicate(
                array.unsafe_get(UInt(i)), pattern
            ):
                data.set(i)
        # Validity is the left operand's: a constant right operand is never
        # null, so `Bitmap.intersect(l, None)` reduces to `l`.
        var vbm: Optional[Bitmap[mut=False]] = None
        if array.bitmap:
            var v = array.bitmap.value().view(array.offset, n)
            vbm = v.union(v).to_immutable()
        return BoolArray(
            length=n,
            nulls=array.null_count(),
            offset=0,
            bitmap=vbm^,
            buffer=data.to_immutable(),
        )
```

- [ ] **Step 4: Override it in `LikeKernel` and `ILikeKernel`**

In `marrow/kernels/string.mojo`, add to `struct LikeKernel`:

```mojo
    @staticmethod
    def apply_scalar[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        # Compile once, not once per row — which is the whole point of
        # `LikePattern`, and had no non-test caller before this.
        return _match_pattern(array, LikePattern[False](pattern))
```

and to `struct ILikeKernel`:

```mojo
    @staticmethod
    def apply_scalar[
        T: StringLikeType
    ](array: BinaryLikeArray[T], pattern: StringSlice) raises -> BoolArray:
        return _match_pattern(array, LikePattern[True](pattern))
```

Update `LikeKernel`'s class docstring: it currently says "use the latter whenever the pattern is a constant, which is the dominant case" — which was aspirational. Replace that clause with a statement that `apply_scalar` is what the expression layer calls for a constant pattern.

- [ ] **Step 5: Precompile**

```bash
pixi run -e dev precompile
```
Expected: no output (0 errors, 0 warnings).

- [ ] **Step 6: Run tests to verify they pass**

```bash
pixi run -e dev pytest marrow/kernels/tests/test_string.mojo
```
Expected: PASS, including all pre-existing cases in the file.

- [ ] **Step 7: Commit**

```bash
git add marrow/kernels/string.mojo marrow/kernels/tests/test_string.mojo
git commit -m "feat(kernels): give string predicates a scalar-pattern entry point

\`apply\` takes a \`BinaryLikeArray\` on both sides, so a caller with a constant
right operand had to splat it into an n-row array first. \`apply_scalar\` is the
peer that does not, with the default body hoisting the right operand out of
\`apply\`'s loop.

\`LikeKernel\`/\`ILikeKernel\` override it to compile one \`LikePattern\` instead of
one per row. That machinery already existed and had no non-test caller.

No behaviour change yet — nothing calls this until the expression layer does.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: route scalar-RHS string predicates through `apply_scalar`

**Files:**
- Modify: `marrow/expr/values.mojo` — `StringPredicate.prepare` (around `:1753`)
- Test: `marrow/expr/tests/test_strings.mojo`

**Interfaces:**
- Consumes: `StringPredicateKernel.apply_scalar` from Task 1; `Self.R.OutShape` (0 = scalar, 1 = columnar); `Context()`; `StringValue.prepare` and `StringValue.elementwise`.
- Produces: no new names. Behaviour only.

**Why `elementwise` and not the scalar `Datum`:** `StringValue.materialize` for `OutShape == 0` already does `self.prepare(...)` then `elementwise(..., idx=0)` and wraps the result in a `StringScalar`. Reading it back out of the scalar would need `StringScalar._value`, which is private. Calling `prepare` + `elementwise` directly is the same computation using only public API.

- [ ] **Step 1: Write the failing test**

Append to `marrow/expr/tests/test_strings.mojo`:

```mojo
def test_like_with_literal_matches_like_with_column() raises:
    """A constant pattern and a column of that same constant must agree.

    The literal route now skips materialising the pattern per row; this is what
    proves that is only a performance difference.
    """
    var sb = StringBuilder(capacity=4)
    sb.append("hello")
    sb.append("help")
    sb.append_null()
    sb.append("world")

    var pb = StringBuilder(capacity=4)
    for _ in range(4):
        pb.append("hel%")

    var batch = record_batch(
        [sb.finish().to_dyn(), pb.finish().to_dyn()], names=["s", "p"]
    )

    var by_literal = col("s", string).like(lit("hel%", string)).execute(batch)
    var by_column = col("s", string).like(col("p", string)).execute(batch)
    assert_true(
        into_array(by_literal, 4).as_bool() == into_array(by_column, 4).as_bool()
    )


def test_string_eq_with_literal_matches_eq_with_column() raises:
    """The same agreement for the default `apply_scalar` body, which the other
    five predicates inherit. `WHERE url = 'x'` is this shape."""
    var sb = StringBuilder(capacity=4)
    sb.append("x")
    sb.append_null()
    sb.append("y")
    sb.append("x")

    var pb = StringBuilder(capacity=4)
    for _ in range(4):
        pb.append("x")

    var batch = record_batch(
        [sb.finish().to_dyn(), pb.finish().to_dyn()], names=["s", "p"]
    )

    var by_literal = (col("s", string) == lit("x", string)).execute(batch)
    var by_column = (col("s", string) == col("p", string)).execute(batch)
    assert_true(
        into_array(by_literal, 4).as_bool() == into_array(by_column, 4).as_bool()
    )


def test_like_literal_preserves_null_rows() raises:
    """A null input row stays null, not False. The array path gets this from
    `Bitmap.intersect`; the scalar path must carry the left operand's validity
    itself."""
    var sb = StringBuilder(capacity=3)
    sb.append("hello")
    sb.append_null()
    sb.append("nope")
    var batch = record_batch([sb.finish().to_dyn()], names=["s"])

    var out = into_array(
        col("s", string).like(lit("hel%", string)).execute(batch), 3
    ).as_bool()
    assert_true(out[0].value())
    assert_true(out.is_null(1))
    assert_true(not out[2].value())
```

Check the file's existing import block and add only what is missing — likely `lit`, `into_array`, `StringBuilder`, `record_batch`, `string`. Follow the spellings already used in that file.

**One spec case is deliberately absent: a null literal RHS.** The spec asks for
it, but it is not constructible — `StringLiteral` holds a plain `String` with no
validity flag, so no `OutShape == 0` string node in the current set can be null.
The scalar path's `apply_scalar` therefore takes validity from the left operand
alone, which is what `Bitmap.intersect(l, None)` already reduced to. If a
nullable string literal is ever added, that assumption breaks and this is where
the test belongs — say so in a comment next to the scalar branch rather than
leaving it implicit.

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest marrow/expr/tests
```

Expected: these three cases FAIL if the imports resolve but behaviour differs, or the whole selection fails to compile if a name is missing. Either is an acceptable red — but read the message. If it is a missing import, fix the import and re-run so the red is a *behavioural* one before proceeding.

Note: at this point the tests may already PASS, because the array path is still correct. That is fine and expected — they are regression pins for the change in Step 3, not a demonstration of a current bug. If they pass here, say so plainly rather than pretending a red was observed.

- [ ] **Step 3: Add the comptime branch**

In `marrow/expr/values.mojo`, replace `StringPredicate.prepare`:

```mojo
    def prepare(self, batch: RecordBatch, mut ctx: Context) raises:
        var n = batch.num_rows()
        var la = into_array(self.l.execute(batch), n).as_string().copy()
        comptime if Self.R.OutShape == 0:
            # A scalar right operand: evaluate it once. `into_array` would
            # broadcast it into n copies of the same string, and the array x
            # array kernel would then re-read — and for LIKE, recompile — that
            # constant on every row.
            var rctx = Context()
            self.r.prepare(batch, rctx)
            var rslot = 0
            var pat = self.r.elementwise(batch, rctx, rslot, 0)
            ctx.append(Self.K.apply_scalar(la, pat).to_dyn())
        else:
            var ra = into_array(self.r.execute(batch), n).as_string().copy()
            ctx.append(Self.K.apply(la, ra).to_dyn())
```

- [ ] **Step 4: Precompile**

```bash
pixi run -e dev precompile
```
Expected: 0 errors, 0 warnings.

- [ ] **Step 5: Run the expr and kernel suites**

```bash
pixi run -e dev pytest marrow/expr/tests
pixi run -e dev pytest marrow/kernels/tests
```
Expected: PASS. Kernel count 523 + Task 1's 3 = 526; expr count unchanged plus the 3 added here.

- [ ] **Step 6: Take the size reading**

```bash
pixi run binary_size > /tmp/size_after_q74.log 2>&1
grep -a "query_streaming\|query_join " /tmp/size_after_q74.log
```

Compare `__text` against the Global Constraints baselines. **Report the numbers whether or not they moved.** A defaulted trait method can instantiate per conforming kernel (six kernels × two string widths).

**If `query_streaming` regresses by more than ~1%:** stop and fall back. The fallback is to revert the trait method and instead branch only in `StringPredicate` for `Like`/`ILike`, calling the already-existing `LikeKernel.apply(array, pattern)` scalar overload at `string.mojo:746`/`:769`. That needs no trait change and so cannot fan out.

- [ ] **Step 7: Commit**

```bash
git add marrow/expr/values.mojo marrow/expr/tests/test_strings.mojo CHANGELOG.md
git commit -m "perf(expr): stop splatting a constant string operand across every row

\`StringPredicate.prepare\` materialised both operands unconditionally, so
\`s LIKE 'foo%'\` allocated an n-row array holding the same pattern n times and
then handed it to the array x array kernel, whose per-row \`predicate\` compiled a
fresh \`LikePattern\` for each one. Two independent wastes: the splat affects all
six predicates sharing this node, the recompilation only LIKE/ILIKE.

\`OutShape\` already distinguishes a scalar operand from a columnar one, so the
branch resolves at elaboration — no new node, no runtime check.

The other five predicates matter too: \`StrEq\` against a literal is what
\`WHERE url = '...'\` compiles to.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Add the `CHANGELOG.md` entry under `## [Unreleased]` → `### Features` before committing.

---

## Task 3: grouped `count` benchmarks for both implementations

**Files:**
- Modify: `marrow/kernels/tests/bench_groupby.mojo`

**Interfaces:**
- Consumes: existing `_bench_group_by[A: Aggregation](mut b, n, num_groups)`, `_make_keys`, `Benchmark`.
- Produces: `_make_vals_nulls(n)`, and four bench entry points named in Step 1. Task 4 reads their numbers.

**Why this task exists:** no grouped `count` benchmark exists — `bench_groupby.mojo` covers `sum`, `mean`, `min`, `max` only. Task 4's decision cannot be made without one.

- [ ] **Step 1: Add the benchmarks**

In `marrow/kernels/tests/bench_groupby.mojo`, extend the import from `...kernels.aggregate` to include `CountKernel` and `CountAgg`, then append:

```mojo
def _make_vals_nulls(n: Int) raises -> DynArray:
    """Values with every third row null — enough nulls that the validity check
    cannot be branch-predicted away."""
    var b = Float64Builder(n)
    for i in range(n):
        if i % 3 == 0:
            b.append_null()
        else:
            b.append(Scalar[float64.native](Float64(i)))
    return b.finish()


def _bench_group_by_nulls[
    A: Aggregation
](mut b: Benchmark, n: Int, num_groups: Int = 10) raises:
    var keys = _make_keys(n, num_groups)
    var vals = A.from_any(_make_vals_nulls(n))
    b.throughput(BenchMetric.elements, n)

    @always_inline
    @parameter
    def call() raises:
        keep(GroupBy(keys).aggregate[A](vals))

    b.iter[call]()
    keep(keys)
    keep(vals)


# ---------------------------------------------------------------------------
# grouped count — the A/B for Q7.3.
#
# Two implementations exist and the two expression lanes disagree about which
# to use: `CountValid.resolve` picks `NumericAgg[CountKernel, V]` for numeric
# columns, while the AOT lane uses `CountKernel.Grouped`, which is `CountAgg`.
#
# They are not obviously ordered. `CountAgg` takes a `DynArray` and calls
# `values.is_valid(i)` per row — erased dispatch — but skips it entirely when
# the column is null-free, in which case it never loads a value at all.
# `AggState` pays a typed validity check and does load the value. So null-free
# should favour `CountAgg` and nullable may favour `AggState`.
#
# Both live in this one binary so the harness interleaves them: measuring one
# variant, rebuilding, then measuring the other invents regressions that are not
# there.
#
# g100k, never g10 — cardinality picks the execution strategy.
# ---------------------------------------------------------------------------


def bench_groupby_count_1m_g100k_aggstate(mut b: Benchmark) raises:
    _bench_group_by[NumericAgg[CountKernel, Float64Type]](b, 1_000_000, 100_000)


def bench_groupby_count_1m_g100k_countagg(mut b: Benchmark) raises:
    _bench_group_by[CountAgg](b, 1_000_000, 100_000)


def bench_groupby_count_nulls_1m_g100k_aggstate(mut b: Benchmark) raises:
    _bench_group_by_nulls[NumericAgg[CountKernel, Float64Type]](
        b, 1_000_000, 100_000
    )


def bench_groupby_count_nulls_1m_g100k_countagg(mut b: Benchmark) raises:
    _bench_group_by_nulls[CountAgg](b, 1_000_000, 100_000)
```

- [ ] **Step 2: Precompile**

```bash
pixi run -e dev precompile
```
Expected: 0 errors, 0 warnings. (`precompile` compiles `bench_*.mojo` too — they are ordinary modules under `marrow/`.)

- [ ] **Step 3: Run the benchmarks**

```bash
pixi run -e dev pytest marrow/kernels/tests/bench_groupby.mojo --benchmark
```

Expected: four new results. Record all four numbers verbatim — they are the input to Task 4's decision rule.

- [ ] **Step 4: Commit**

```bash
git add marrow/kernels/tests/bench_groupby.mojo
git commit -m "test(kernels): benchmark grouped count, both implementations

No grouped \`count\` benchmark existed — bench_groupby covered sum/mean/min/max
only, which is a gap given how much of ClickBench is COUNT(*).

Both implementations run in one binary so the harness interleaves them. Q7.3
cannot be decided without this: \`CountAgg\` skips the value load entirely on a
null-free column but pays erased per-row \`is_valid\` on a nullable one, so which
wins is input-dependent and not readable off the source.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: apply the Q7.3 decision rule

**Files:**
- Modify (conditionally): `marrow/expr/aggregates.mojo` — `CountValid.resolve` (around `:152-170`)
- Modify (always): `marrow/kernels/aggregate.mojo` — `CountAgg` docstring (around `:1040-1048`)
- Test: `marrow/kernels/tests/test_groupby.mojo` or `marrow/expr/tests/test_aggregates.mojo`

**Interfaces:**
- Consumes: Task 3's four measurements.
- Produces: no new names.

**Decision rule — fixed before the numbers were seen. Do not renegotiate it after reading them.**

| Outcome | Action |
|---|---|
| `CountAgg` wins or ties on **both** shapes | Converge: in `CountValid.resolve`, delete the numeric branch so every dtype gets `job[CountAgg]()`. The documented invariant becomes true. |
| `AggState` wins clearly on nullable | Keep both. Rewrite the `CountAgg` docstring to state the split and its reason. |
| Mixed, or inside run-to-run noise | Keep both. Rewrite the docstring, and record the erased `is_valid` as why merging is not free. |

- [ ] **Step 1: Write the agreement test**

Both implementations must already produce identical results; nothing currently pins that across the numeric/non-numeric boundary. Append to `marrow/kernels/tests/test_groupby.mojo`:

```mojo
def test_grouped_count_implementations_agree_on_nulls() raises:
    """`CountAgg` and `NumericAgg[CountKernel, _]` must be interchangeable.

    Q7.3 is a choice between them, so any disagreement here makes that choice a
    correctness question rather than a performance one. `count` excludes nulls
    (SQL `COUNT(x)`), and an empty group counts 0, never null.
    """
    var kb = Int32Builder(6)
    for i in range(6):
        kb.append(Scalar[int32.native](i % 2))
    var keys = kb.finish()

    var vb = Float64Builder(6)
    vb.append(Float64(1))
    vb.append_null()
    vb.append(Float64(3))
    vb.append_null()
    vb.append(Float64(5))
    vb.append(Float64(6))
    var vals = vb.finish()

    var via_state = GroupBy(keys).aggregate[
        NumericAgg[CountKernel, Float64Type]
    ](NumericAgg[CountKernel, Float64Type].from_any(vals))
    var via_scan = GroupBy(keys).aggregate[CountAgg](
        CountAgg.from_any(vals)
    )
    assert_true(via_state == via_scan)
```

Add `CountKernel`, `CountAgg`, `NumericAgg`, `Float64Type`, `Float64Builder` to that file's imports if absent.

- [ ] **Step 2: Run it**

```bash
pixi run -e dev pytest marrow/kernels/tests
```
Expected: PASS. **If it FAILS, stop.** Q7.3 is then a correctness bug, not a performance choice, and the decision rule does not apply — report the disagreement and re-plan.

- [ ] **Step 3: Apply the rule**

Read the four numbers from Task 3 and pick the matching row of the table above.

**If converging** — in `marrow/expr/aggregates.mojo`, `CountValid.resolve` becomes:

```mojo
    @staticmethod
    def resolve[
        job: def[A: Aggregation]() raises capturing[_] -> None
    ](value_dtype: DynType) raises:
        # One implementation for every dtype. `count` reads validity and
        # nothing else, so there is nothing to monomorphize on — the numeric
        # branch that used to take `NumericAgg[CountKernel, V]` loaded a value
        # it then discarded.
        job[CountAgg]()
```

and delete the now-unused `NumericAgg`/`CountKernel` imports from that file if nothing else uses them. Update `CountValid`'s own docstring, which currently says "Numeric columns take the mergeable `AggState` fold; everything else the validity-only scan."

**If keeping both** — leave `CountValid.resolve` alone and rewrite the `CountAgg` docstring so it stops claiming to be the numeric implementation. Replace the sentence *"That makes it the grouped form of `count` for numeric columns too (`CountKernel.Grouped` names it), so there is one implementation rather than a fold for numbers and a scan for everything else"* with the measured reason, quoting the numbers.

- [ ] **Step 4: Precompile and test**

```bash
pixi run -e dev precompile
pixi run -e dev pytest marrow/kernels/tests
pixi run -e dev pytest marrow/expr/tests
```
Expected: 0 diagnostics; both suites PASS.

- [ ] **Step 5: Size gate**

```bash
pixi run binary_size > /tmp/size_after_q73.log 2>&1
grep -a "query_streaming" /tmp/size_after_q73.log
```

If you converged, `query_streaming_agg` should shrink (one fewer instantiation family). Report the numbers either way.

- [ ] **Step 6: Commit**

Use whichever of these two messages matches the outcome, appending the four
measurements verbatim in both cases so the next reader sees what the decision
rested on instead of re-deriving it.

**If you converged:**

```bash
git add marrow/expr/aggregates.mojo marrow/kernels/aggregate.mojo marrow/kernels/tests/test_groupby.mojo CHANGELOG.md
git commit -m "fix(aggregate): count had two grouped implementations; now it has one

\`CountAgg\`'s docstring claimed twice to be the grouped \`count\` for numeric
columns too. \`CountValid.resolve\` contradicted it, handing numeric columns
\`NumericAgg[CountKernel, V]\` while the AOT lane used \`CountKernel.Grouped\` —
which is \`CountAgg\`. Two lanes, two implementations, one false invariant.

Measured before choosing, both implementations in one binary at 1M rows and
g100k so the harness interleaves them:

  <paste the four numbers here>

The runtime lane now uses \`CountAgg\` like the AOT lane. The numeric fold it
replaces loaded a value it then discarded — \`count\` reads validity and nothing
else, so there was never anything to monomorphize on.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

**If you kept both:**

```bash
git add marrow/kernels/aggregate.mojo marrow/kernels/tests/test_groupby.mojo CHANGELOG.md
git commit -m "docs(aggregate): count's two grouped implementations are real; say so

\`CountAgg\`'s docstring claimed twice to be the grouped \`count\` for numeric
columns too, while \`CountValid.resolve\` handed those columns
\`NumericAgg[CountKernel, V]\`. The split is real and worth keeping; the claim
was not.

Measured before choosing, both implementations in one binary at 1M rows and
g100k so the harness interleaves them:

  <paste the four numbers here>

\`CountAgg\` skips the value load entirely on a null-free column but pays erased
per-row \`is_valid\` on a nullable one, which is why merging the two is not free.
Docstring now states that instead of claiming a unification that never happened.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: `is_primitive()` excludes bool

**Files:**
- Modify: `marrow/dtypes.mojo:1057-1069` (`is_primitive`), `:1191-1192` (`is_fixed_size`)
- Test: `marrow/tests/test_dtypes.mojo:10-18`, `:114-126`

**Interfaces:**
- Consumes: nothing new.
- Produces: no new names. `is_primitive()` narrows; `is_fixed_size()` keeps its current answer for every dtype.

**Background:** `byte_width()` guards on `is_primitive()` then dispatches over `variant_dispatch[PrimitiveType]`. `BoolType` does not conform to `PrimitiveType` — correctly, since bool is bit-packed — so that dispatch **aborts the process**. All six callers already peel bool off first (`kernels/filter.mojo:85`, `:607`, `kernels/hashing.mojo:317`, `kernels/sort.mojo:454`, `c_data.mojo:1001`, `ipc.mojo:1941`), so narrowing the predicate changes no call site's behaviour and removes the abort without a special case.

- [ ] **Step 1: Write the failing test**

In `marrow/tests/test_dtypes.mojo`, change the bool case at `:16` from `assert_true(t.is_primitive())` to:

```mojo
    # marrow deliberately diverges from PyArrow and Arrow C++ here, both of
    # which call bool primitive. marrow has something neither has: a
    # `PrimitiveType` *trait* that generic code dispatches on, which `BoolType`
    # cannot conform to because bool is bit-packed. `is_primitive()` exists to
    # guard `variant_dispatch[PrimitiveType]`, so a True here is a trap — it
    # aborted `byte_width()`. Do not "fix" this back.
    assert_false(t.is_primitive())
    assert_true(t.is_fixed_size())
```

and append a new case:

```mojo
def test_bool_byte_width_is_zero_not_an_abort() raises:
    """bool has no byte width — it is one bit.

    This used to abort the process: `is_primitive()` answered True, so
    `byte_width()` fell through to `variant_dispatch[PrimitiveType]`, which
    `BoolType` cannot enter. `c_data.mojo:1001` is one branch-ordering mistake
    away from reaching it on the C Data import path.
    """
    assert_equal(DynType(dt.bool_).byte_width(), 0)
    assert_true(DynType(dt.bool_).is_fixed_size())
    assert_false(DynType(dt.bool_).is_primitive())
```

Check the file's existing import block for `assert_false` and `assert_equal` and add if missing.

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest marrow/tests/test_dtypes.mojo
```

Expected: FAIL. The `byte_width` case fails as an **abort**, not an assertion — it kills the whole `TestSuite` runner, so pytest reports every case in the file as failed. Read the inner runner summary, not pytest's file-level rollup. That mass-failure appearance is the expected red.

- [ ] **Step 3: Narrow the predicate**

In `marrow/dtypes.mojo`, replace `is_primitive`:

```mojo
    def is_primitive(self) -> Bool:
        """True for the fixed-**byte**-width, buffer-backed types: numeric,
        temporal, interval, decimal.

        Exactly the set that conforms to the `PrimitiveType` trait, which is
        what this predicate is for — every caller uses it to guard a
        `variant_dispatch[PrimitiveType]`.

        **Deliberately narrower than PyArrow and Arrow C++**, which both count
        bool (`pa.types.is_primitive(pa.bool_())` is True; Arrow C++'s
        `is_primitive` lists `Type::BOOL` first). Neither has a `PrimitiveType`
        trait to stay consistent with; marrow does, and `BoolType` cannot
        conform to it because bool is bit-packed rather than fixed-byte-width.
        Answering True here made `byte_width()` abort. Use `is_fixed_size()` for
        the Arrow-spec notion, or `is_bool() or is_primitive()` for PyArrow
        parity — which is what `ipc.mojo` already writes.
        """
        return (
            self.is_numeric()
            or self.is_temporal()
            or self.is_interval()
            or self.is_decimal()
        )
```

and `is_fixed_size`:

```mojo
    def is_fixed_size(self) -> Bool:
        """True for every type with a fixed width, bool included.

        This is the Arrow-spec notion, and it is *wider* than `is_primitive()`
        by exactly bool: boolean is fixed-width at one bit, it just has no fixed
        *byte* width.
        """
        return self.is_bool() or self.is_primitive()
```

Leave `byte_width()` untouched — bool now fails its `is_primitive()` guard and returns 0 through the existing path.

- [ ] **Step 4: Precompile**

```bash
pixi run -e dev precompile
```
Expected: 0 errors, 0 warnings.

- [ ] **Step 5: Run the suites the predicate reaches**

```bash
pixi run -e dev pytest marrow/tests marrow/kernels/tests
pixi run -e dev pytest marrow/expr/tests marrow/parquet/tests
pixi run -e dev pytest python/marrow/tests
```

Expected: PASS. `filter`, `take`, `sort`, `hashing`, `c_data` and `ipc` all call `is_primitive()`; their bool cases must still work, since each peels bool off before reaching it. A failure in any of them means a caller relied on bool being primitive — investigate rather than widening the predicate back.

**Before relying on that, confirm the coverage exists.** The spec calls for one
caller pinned end-to-end. Check that `filter` and `take` over a `BoolArray` are
actually tested:

```bash
rtk proxy grep -n "bool" marrow/kernels/tests/test_filter.mojo | head -20
```

If neither has a bool case, add one to `marrow/kernels/tests/test_filter.mojo`
before Step 6 — a green suite proves nothing about a path it never exercises:

```mojo
def test_filter_bool_array_after_primitive_narrowing() raises:
    """`Filter.dispatch` peels bool off before `is_primitive()`. Narrowing that
    predicate must not disturb the bool arm."""
    var bb = BoolBuilder(capacity=4)
    bb.append(True)
    bb.append(False)
    bb.append_null()
    bb.append(True)
    var arr = bb.finish()

    var mb = BoolBuilder(capacity=4)
    mb.append(True)
    mb.append(True)
    mb.append(True)
    mb.append(False)
    var mask = mb.finish()

    var out = filter(arr.to_dyn(), mask).as_bool()
    assert_equal(len(out), 3)
    assert_true(out[0].value())
    assert_true(not out[1].value())
    assert_true(out.is_null(2))
```

- [ ] **Step 6: Commit**

```bash
git add marrow/dtypes.mojo marrow/tests/test_dtypes.mojo CHANGELOG.md
git commit -m "fix(dtypes): is_primitive() said bool, and byte_width() aborted on it

\`byte_width()\` guards on \`is_primitive()\` then dispatches over
\`variant_dispatch[PrimitiveType]\`. \`BoolType\` does not conform — correctly,
bool is bit-packed — so bool aborted the process rather than answering.

The defect is the predicate, not \`byte_width()\`. All six callers already peel
bool off before reaching \`is_primitive()\`, because it exists to guard exactly
that dispatch; \`ipc.mojo\`'s \`is_bool() or is_primitive()\` is redundant only
because of this bug, and \`c_data.mojo\` would abort on the C Data import path if
its branch order ever changed. The docstring's own prose already excluded bool.

So \`is_primitive()\` narrows to the set conforming to \`PrimitiveType\`, and
\`byte_width()\` needs no special case — the abort disappears as a consequence.
\`is_fixed_size()\` re-adds bool, which is the Arrow-spec notion and its actual
meaning.

This diverges from PyArrow and Arrow C++, which both count bool. That is
deliberate: neither has a \`PrimitiveType\` trait to stay consistent with, and a
runtime predicate that disagrees with the comptime trait is a trap. Recorded in
the docstring and pinned by a test comment.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Add the `CHANGELOG.md` entry under `### Fixes` before committing.

---

## Block sign-off

- [ ] **Full verification pass**

```bash
rm -f .test_runners/marrow.mojoc marrow.mojoc
pixi run -e dev precompile
pixi run -e dev pytest marrow/tests marrow/kernels/tests
pixi run -e dev pytest marrow/expr/tests marrow/parquet/tests
pixi run -e dev pytest python/marrow/tests
pixi run binary_size
```

Expected: 0 diagnostics; ≥1,951 + 9 new cases passing; all four size gates reported against the Global Constraints baselines.

- [ ] **Update `docs/backlog.md`**

- Remove the Q7.3 and Q7.4 rows.
- **Reword the B8 row** rather than deleting it silently: it is carded as a one-line `byte_width()` guard, and the real fix was the `is_primitive()` predicate. Say so, so the next reader learns the same thing.
- Record the size-gate readings on I4 if any moved.

- [ ] **Commit the backlog update**

```bash
git add docs/backlog.md
git commit -m "docs: close Q7.3, Q7.4 and B8

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
