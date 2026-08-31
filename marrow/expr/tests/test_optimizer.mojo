"""Every rule, twice: that it fires, and that firing changes no answer.

**The second half is the one that matters.** A rule that fires and is wrong
looks identical to a rule that fires and is right, from the plan alone. So each
rule gets a pair: one case asserting the rewritten plan's shape — which is
possible at all only because `optimize` returns a plan you can render — and one
asserting `optimize[AllRules]()` and `optimize[NoRules]()` produce equal
results on real data.

`NoRules` is the control arm rather than "call `execute` directly", so both
sides of the comparison travel the same code path and differ in exactly one
variable: whether any rule was allowed to fire.
"""

from std.testing import assert_equal, assert_false, assert_true

from ...builders import array
from ...dtypes import int64, string
from ...execution import ExecContext
from ...tabular import RecordBatch, record_batch
from ..builders import col, lit, table
from ..logical import DynRelation, DynValue
from ..optimizer import (
    AllRules,
    MergeLimits,
    NoRules,
    PushFilterBelowProject,
    PushFilterBelowSort,
    PushLimitBelowProject,
    RemoveNoOpProject,
    RemoveRedundantSort,
    TopN,
)


def _batch() raises -> RecordBatch:
    """Six rows, with `a` deliberately unsorted so ordering is observable and
    `b` distinct so a projection cannot accidentally collide."""
    return record_batch(
        [
            array([3, 1, 4, 1, 5, 9], int64).copy(),
            array([10, 20, 30, 40, 50, 60], int64).copy(),
        ],
        names=["a", "b"],
    )


def _occurrences(haystack: String, needle: String) -> Int:
    """How many times `needle` appears in `haystack`.

    Implemented with `split` rather than a `find`-with-offset loop, which is
    **not** a style preference: the loop form spun forever and left a test
    driver at 98% CPU for over an hour, because it assumed `find`'s second
    argument advances the search and never verified it. `split` cannot loop —
    it either finds the pieces or it does not.
    """
    return len(haystack.split(needle)) - 1


def _col(batch: RecordBatch, index: Int) raises -> List[Int]:
    """One int64 column as plain values, nulls rendered as `_NULL`.

    Results are compared **by extracted value**, not by `RecordBatch.__eq__`.
    That is deliberate on two counts: comparing values is what proves an answer
    *correct* rather than merely self-consistent, and whole-batch equality is
    itself under suspicion — see `test_executing_one_plan_twice_agrees`.
    """
    ref col = batch.columns[index].as_int64()
    var out = List[Int](capacity=len(col))
    for i in range(len(col)):
        if col.is_valid(i):
            out.append(Int(col[i].value()))
        else:
            out.append(_NULL)
    return out^


comptime _NULL = -999_999
"""Sentinel for a null cell, chosen outside every value the fixtures use."""


def _check(plan: DynRelation, expected: List[List[Int]]) raises:
    """Both plans return exactly `expected` — not merely the same thing.

    **The soundness contract of this file.** Asserting only that the optimized
    and unoptimized plans agree proves nothing when both are wrong, and two
    wrong plans agree whenever a rule drops the same rows on both sides — which
    is precisely how an optimizer fails. So the expected rows are written by
    hand and each side is checked against them independently.

    Takes a column-per-entry list rather than two fixed columns, so rules over
    `Project` and `Aggregate` — which change the output shape — are testable at
    all. The first version of this helper compared exactly two `int64` columns,
    which is why five of eleven rules went untested.
    """
    var ctx = ExecContext()
    var before = plan.optimize[NoRules]().execute(ctx)
    var after = plan.optimize[AllRules]().execute(ctx)

    assert_equal(
        before.num_columns(), len(expected), "unoptimized: wrong column count"
    )
    assert_equal(
        after.num_columns(), len(expected), "OPTIMIZED: wrong column count"
    )
    for i in range(len(expected)):
        assert_equal(
            _col(before, i), expected[i], "unoptimized plan is wrong"
        )
        assert_equal(_col(after, i), expected[i], "OPTIMIZED plan is wrong")


def _fires(plan: DynRelation) raises:
    """Some rule actually rewrote this plan.

    Pairs with `_check`: without it an equivalence assertion passes trivially
    whenever no rule fires, which would let a rule silently stop working and
    take its own test down with it.
    """
    assert_true(
        String(plan) != String(plan.optimize[AllRules]()),
        "no rule fired on: " + String(plan),
    )


def _inert(plan: DynRelation) raises:
    """No rule rewrote this plan — for the cases asserting a rule is *blocked*.
    """
    assert_equal(
        String(plan),
        String(plan.optimize[AllRules]()),
        "a rule fired that should not have",
    )


# ---------------------------------------------------------------------------
# Is the comparison itself trustworthy?
# ---------------------------------------------------------------------------
def test_executing_one_plan_twice_agrees() raises:
    """One unchanged plan, executed twice, must return the same rows.

    **A control on the whole file.** If this fails, every equivalence assertion
    below is measuring engine nondeterminism rather than the optimizer, and the
    six failures that motivated this rewrite were exactly that shape — plans on
    which *no rule fires* still reported a disagreement, which no optimizer bug
    can explain.

    It compares extracted values, and then whole batches, so a failure says
    which of the two is at fault.
    """
    var ctx = ExecContext()
    var plan = table(_batch()).limit(3).filter(col("b", int64) > lit(20, int64))
    var first = plan.execute(ctx)
    var second = plan.execute(ctx)
    assert_equal(_col(first, 0), _col(second, 0), "values differ across runs")
    assert_equal(_col(first, 1), _col(second, 1), "values differ across runs")
    assert_true(
        first == second,
        "values agree but RecordBatch.__eq__ does not — the equality is at"
        " fault, not the engine",
    )


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
def test_optimizer_no_rules_is_the_identity() raises:
    """The control arm has to be inert, or every case below compares two
    rewritten plans and proves nothing."""
    var plan = table(_batch()).filter(col("a", int64) > lit(1, int64)).limit(2)
    assert_equal(String(plan), String(plan.optimize[NoRules]()))


def test_optimizer_reaches_a_fixpoint() raises:
    var plan = table(_batch()).sort_by([col("a", int64)], [True]).limit(2)
    var once = plan.optimize[AllRules]()
    assert_equal(String(once), String(once.optimize[AllRules]()))


# ---------------------------------------------------------------------------
# TopN
# ---------------------------------------------------------------------------
def test_optimizer_topn_bounds_the_sort() raises:
    var plan = table(_batch()).sort_by([col("a", int64)], [True]).limit(2)
    _fires(plan)
    assert_true("top 2" in String(plan.optimize[AllRules]()))
    # a sorted = [1,1,3,4,5,9] with b following: [20,40,10,30,50,60]
    _check(plan, [[1, 1], [20, 40]])


def test_optimizer_topn_keeps_the_limit() raises:
    """The `Limit` survives — it applies the offset and stops the source."""
    var plan = table(_batch()).sort_by([col("a", int64)], [True]).limit(2)
    assert_true("Limit(" in String(plan.optimize[AllRules]()))


def test_optimizer_topn_bound_covers_the_offset() raises:
    """`limit(2, offset=3)` must retain five ordered rows, not two.

    Bounding at `length` would discard exactly the rows the offset skips to and
    the query would come back empty — so both the rendered bound and the rows
    are asserted.
    """
    var plan = table(_batch()).sort_by([col("a", int64)], [True]).limit(2, 3)
    assert_true("top 5" in String(plan.optimize[AllRules]()))
    _check(plan, [[4, 5], [30, 50]])


def test_optimizer_topn_does_not_fire_through_a_filter() raises:
    """`Limit(Filter(Sort(x)))` is the silent-wrong-answer shape.

    `PushFilterBelowSort` relocates the filter first, after which `TopN` is
    adjacent and may legitimately fire — so this asserts the rows, which is the
    property that actually matters.
    """
    var plan = (
        table(_batch())
        .sort_by([col("a", int64)], [True])
        .filter(col("b", int64) > lit(20, int64))
        .limit(2)
    )
    # sorted by a: b = [20,40,10,30,50,60]; b > 20 keeps [40,30,50,60]
    _check(plan, [[1, 4], [40, 30]])


# ---------------------------------------------------------------------------
# Filter movement
# ---------------------------------------------------------------------------
def test_optimizer_pushes_a_filter_below_a_sort() raises:
    var plan = (
        table(_batch())
        .sort_by([col("a", int64)], [True])
        .filter(col("b", int64) > lit(20, int64))
    )
    _fires(plan)
    var out = String(plan.optimize[AllRules]())
    assert_true(out.find("Sort(") < out.find("Filter("), out)
    _check(plan, [[1, 4, 5, 9], [40, 30, 50, 60]])


def test_optimizer_does_not_push_a_filter_below_a_limit() raises:
    """`limit(3)` then `filter(p)` means "of the first three rows, those
    matching" — filtering first would yield three *matching* rows, a different
    and larger answer."""
    var plan = table(_batch()).limit(3).filter(col("b", int64) > lit(20, int64))
    _inert(plan)
    # first three rows: a=[3,1,4], b=[10,20,30]; b > 20 keeps only b=30
    _check(plan, [[4], [30]])


# ---------------------------------------------------------------------------
# Limit rules
# ---------------------------------------------------------------------------
def test_optimizer_merges_stacked_limits() raises:
    var plan = table(_batch()).limit(4).limit(2)
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Limit("), 1)
    _check(plan, [[3, 1], [10, 20]])


def test_optimizer_merged_limit_composes_offsets() raises:
    """The outer window is relative to the inner, so offsets add."""
    var plan = table(_batch()).limit(4, 1).limit(2, 1)
    _fires(plan)
    # inner: rows 1..4 -> a=[1,4,1,5]; outer: skip 1 take 2 -> a=[4,1]
    _check(plan, [[4, 1], [30, 40]])


def test_optimizer_merged_limit_clamps_to_the_inner_window() raises:
    """An outer limit asking for more than the inner left gets what is there.
    """
    var plan = table(_batch()).limit(2).limit(5)
    _fires(plan)
    _check(plan, [[3, 1], [10, 20]])


def test_optimizer_zero_length_limit_becomes_empty() raises:
    """`LIMIT 0` discards the whole subtree — how a frontend asks for a schema
    without data."""
    var plan = table(_batch()).sort_by([col("a", int64)], [True]).limit(0)
    _fires(plan)
    assert_true("Empty(" in String(plan.optimize[AllRules]()))
    _check(plan, [List[Int](), List[Int]()])


def test_optimizer_empty_propagates_through_a_filter() raises:
    """Emptiness travels upward, so no operator is built above it."""
    var plan = (
        table(_batch())
        .limit(0)
        .filter(col("b", int64) > lit(20, int64))
    )
    var out = String(plan.optimize[AllRules]())
    assert_equal(_occurrences(out, "Filter("), 0)
    assert_true("Empty(" in out, out)
    _check(plan, [List[Int](), List[Int]()])


# ---------------------------------------------------------------------------
# Sort rules
# ---------------------------------------------------------------------------
def test_optimizer_removes_a_redundant_sort() raises:
    var plan = (
        table(_batch())
        .sort_by([col("b", int64)], [True])
        .sort_by([col("a", int64)], [True])
    )
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Sort("), 1)
    _check(plan, [[1, 1, 3, 4, 5, 9], [20, 40, 10, 30, 50, 60]])


# ---------------------------------------------------------------------------
# Projection rules
#
# These five rules had **no coverage at all** until 2026-08-31: the fixture was
# two int64 columns and `_check` compared exactly two, so every rule that
# changes the output shape was untestable and silently went untested. The gap
# tracked what was cheap to assert, not what was risky.
# ---------------------------------------------------------------------------
def test_optimizer_removes_a_no_op_projection() raises:
    """`select` of every column, in order, is the input."""
    var plan = table(_batch()).select("a", "b")
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Project("), 0)
    _check(plan, [[3, 1, 4, 1, 5, 9], [10, 20, 30, 40, 50, 60]])


def test_optimizer_keeps_a_narrowing_projection() raises:
    """`select("a")` changes the schema, so it is **not** a no-op.

    The negative case matters more than the positive one: matching on
    expressions rather than on the resulting schema would delete this and
    silently return an extra column.
    """
    var plan = table(_batch()).select("a")
    assert_true("Project(" in String(plan.optimize[AllRules]()))
    _check(plan, [[3, 1, 4, 1, 5, 9]])


def test_optimizer_merges_stacked_projections() raises:
    """`Project(Project(x))` collapses when the outer only selects."""
    var plan = table(_batch()).select("a", "b").select("a")
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Project("), 1)
    _check(plan, [[3, 1, 4, 1, 5, 9]])


def test_optimizer_pushes_a_filter_below_a_projection() raises:
    """A predicate on a pass-through column moves under the projection."""
    var plan = table(_batch()).select("a", "b").filter(
        col("a", int64) > lit(3, int64)
    )
    _check(plan, [[4, 5, 9], [30, 50, 60]])


def test_optimizer_pushes_a_limit_below_a_projection() raises:
    """A projection is row- and order-preserving, so the window commutes."""
    var plan = table(_batch()).select("a").limit(2)
    _fires(plan)
    _check(plan, [[3, 1]])


# ---------------------------------------------------------------------------
# Aggregate rules
# ---------------------------------------------------------------------------
def test_optimizer_removes_a_sort_before_an_aggregate() raises:
    """Every fold marrow has is order-insensitive, so the sort is wasted work.

    Asserted on the answer as well as the shape: `sum` over a reordered input
    must still be 23, and if the rule ever fires where it should not — say a
    `first`/`last` aggregate is added — this is what catches it.
    """
    var plan = (
        table(_batch())
        .sort_by([col("a", int64)], [True])
        .aggregate([col("a", int64).sum().alias("total")])
    )
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Sort("), 0)
    # 3 + 1 + 4 + 1 + 5 + 9 = 23
    _check(plan, [[23]])


def test_optimizer_keeps_a_topn_sort_before_an_aggregate() raises:
    """A bounded sort **drops rows**, so it changes which rows are aggregated
    and may not be removed.

    Built as `Limit(Sort)` so `TopN` bounds the sort first; the aggregate then
    sees a sort that is load-bearing rather than cosmetic.
    """
    var plan = (
        table(_batch())
        .sort_by([col("a", int64)], [True])
        .limit(3)
        .aggregate([col("a", int64).sum().alias("total")])
    )
    # the three smallest values of a are 1, 1, 3
    _check(plan, [[5]])


def test_an_aggregate_above_a_limit_emits_one_row() raises:
    """An ungrouped aggregate always emits exactly one row — even over a limit.

    **A probe for an engine defect, not for a rule.** It runs the plan through
    `optimize[NoRules]`, so no rewrite is involved; it exists because
    `test_optimizer_keeps_a_topn_sort_before_an_aggregate` found the
    *unoptimized* plan returning zero rows, and the failure had to be pinned to
    the engine rather than to the optimizer.

    The likely shape: `LimitOperator` reports `done` once it has its rows, and
    in a push engine that stops the source — so if `done` also skips `drain` on
    the operators above, the aggregate never gets the call it emits from. An
    aggregate has nothing to push and answers only from `drain`.
    """
    var ctx = ExecContext()
    var plan = table(_batch()).limit(3).aggregate(
        [col("a", int64).sum().alias("total")]
    )
    var out = plan.optimize[NoRules]().execute(ctx)
    assert_equal(out.num_rows(), 1, "an ungrouped aggregate must emit one row")
    _ = ctx^
