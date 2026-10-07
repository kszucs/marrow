# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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

from std.collections import Set
from std.testing import assert_equal, assert_false, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import bool_, field, int64, string
from ...execution import ExecContext
from ...tabular import RecordBatch, record_batch
from ...scalars import BoolScalar, Int64Scalar
from ..logical import Bindings
from ..builders import (
    col,
    count_star,
    dense_rank,
    lit,
    param,
    rank,
    row_number,
    scan,
    table,
)
from ..runtime.values import and_, column, gt, literal, not_
from ...kernels.join import (
    BUILD_LEFT,
    BUILD_RIGHT,
    JOIN_ALL,
    JOIN_ANTI,
    JOIN_ANY,
    JOIN_CROSS,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    JoinKind,
)
from ..physical import JoinOrder, PlannedJoin
from ..logical import (
    Aggregate,
    DynRelation,
    DynValue,
    Filter,
    JoinChain,
    JoinLink,
    JoinPricing,
    JoinRules,
    ParquetScan,
    Project,
    Sort,
)
from ...schema import schema
from ..optimizer import (
    AllRules,
    JoinOrdering,
    MergeLimits,
    NoRules,
    Optimizer,
    PushFilterBelowProject,
    PushFilterBelowSort,
    PushFilterIntoJoin,
    PushFilterIntoScan,
    PushLimitBelowProject,
    RemoveNoOpProject,
    RemoveRedundantSort,
    RuleSet,
    ScanPruning,
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
        assert_equal(_col(before, i), expected[i], "unoptimized plan is wrong")
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
        (
            "values agree but RecordBatch.__eq__ does not — the equality is at"
            " fault, not the engine"
        ),
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
    """An outer limit asking for more than the inner left gets what is there."""
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
    var plan = table(_batch()).limit(0).filter(col("b", int64) > lit(20, int64))
    _fires(plan)
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


def test_optimizer_narrowing_projection_dissolves_into_the_source() raises:
    """`select("a")` ends up with **no** projection, and that is correct.

    The projection is not "deleted because it looked redundant" — column
    pruning narrows the source to `a` first, which *makes* it redundant, and
    `RemoveNoOpProject` then removes a genuine no-op. Two rules composing.

    This case previously asserted the projection survived, which was true
    before pruning existed and is now simply a worse plan.
    """
    var plan = table(_batch()).select("a")
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Project("), 0)
    _check(plan, [[3, 1, 4, 1, 5, 9]])


def test_optimizer_keeps_a_reordering_projection() raises:
    """A projection that **reorders** columns is not a no-op, however much its
    field set matches.

    The negative that matters: `RemoveNoOpProject` compares schemas, and a
    schema carries order. Matching on the field *set* instead would delete this
    and silently return the columns the other way round — which no assertion on
    row values alone would catch, since both columns contain the right data.
    """
    var plan = table(_batch()).select("b", "a")
    assert_true("Project(" in String(plan.optimize[AllRules]()))
    _check(plan, [[10, 20, 30, 40, 50, 60], [3, 1, 4, 1, 5, 9]])


def test_optimizer_merges_stacked_projections() raises:
    """`Project(Project(x))` collapses when the outer only selects — and then
    disappears: the merged projection reads `a` alone, so the second pruning
    pass in `AllRules.finish` narrows the source to `a`, and a projection
    reproducing its input is `RemoveNoOpProject`'s."""
    var plan = table(_batch()).select("a", "b").select("a")
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Project("), 0)
    _check(plan, [[3, 1, 4, 1, 5, 9]])


def test_optimizer_pushes_a_filter_below_a_projection() raises:
    """A predicate on a pass-through column moves under the projection."""
    var plan = (
        table(_batch()).select("a", "b").filter(col("a", int64) > lit(3, int64))
    )
    _fires(plan)
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

    **What actually stops the rule here is the `Limit` between them**, not the
    bound — `RemoveSortBeforeAggregate` matches `Aggregate(Sort(x))` and this
    plan is `Aggregate(Limit(Sort(x)))`. `sort_by` exposes no bound, and `TopN`
    only ever sets one directly under a `Limit`, so `Aggregate(Sort(top k))` is
    not reachable from the public verbs at all and the rule's own `sort.limit`
    guard is belt-and-braces. The surviving `Sort` is asserted because that is
    what this case's name claims and the rows alone do not show it.
    """
    var plan = (
        table(_batch())
        .sort_by([col("a", int64)], [True])
        .limit(3)
        .aggregate([col("a", int64).sum().alias("total")])
    )
    var out = String(plan.optimize[AllRules]())
    assert_equal(_occurrences(out, "Sort("), 1, out)
    assert_true("top 3" in out, out)
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
    var plan = (
        table(_batch())
        .limit(3)
        .aggregate([col("a", int64).sum().alias("total")])
    )
    var out = plan.optimize[NoRules]().execute(ctx)
    assert_equal(out.num_rows(), 1, "an ungrouped aggregate must emit one row")
    _ = ctx^


# ---------------------------------------------------------------------------
# Column pruning — the downward pass
# ---------------------------------------------------------------------------
def _wide() raises -> RecordBatch:
    """Four columns, so pruning has something to remove and the survivors can
    be checked by position."""
    return record_batch(
        [
            array([3, 1, 4], int64).copy(),
            array([10, 20, 30], int64).copy(),
            array([100, 200, 300], int64).copy(),
            array([7, 8, 9], int64).copy(),
        ],
        names=["a", "b", "c", "d"],
    )


def test_pruning_narrows_a_source_to_the_projected_columns() raises:
    """`select("a")` over four columns reads one."""
    var plan = table(_wide()).select("a")
    _fires(plan)
    _check(plan, [[3, 1, 4]])


def test_pruning_keeps_columns_a_filter_reads() raises:
    """A predicate's columns are needed below even though the output drops
    them — this is the case a naive "keep what the output names" gets wrong,
    and it fails as a missing column rather than as wrong rows."""
    var plan = (
        table(_wide()).filter(col("b", int64) > lit(15, int64)).select("a")
    )
    _check(plan, [[1, 4]])


def test_pruning_keeps_columns_a_sort_reads() raises:
    var plan = table(_wide()).sort_by([col("c", int64)], [False]).select("a")
    _check(plan, [[4, 1, 3]])


def test_pruning_keeps_columns_an_aggregate_groups_by() raises:
    var plan = table(_wide()).aggregate(
        [col("b", int64).sum().alias("total")], [col("a", int64)]
    )
    _check(plan, [[3, 1, 4], [10, 20, 30]])


def test_pruning_never_leaves_a_source_with_no_columns() raises:
    """`count_star()` reads no columns at all.

    A `RecordBatch` carries its row count in its columns, so pruning to the
    empty set would make the source report zero rows and the count come back
    `0` — a wrong answer, not a slow one. The source keeps its first column.
    """
    var plan = table(_wide()).aggregate([count_star()])
    _check(plan, [[3]])


def test_pruning_preserves_source_column_order() raises:
    """A source keeps *its* order, not the order the consumer named.

    Reordering here would move every positional reference above it.
    """
    var plan = table(_wide()).select("d", "a")
    _check(plan, [[7, 8, 9], [3, 1, 4]])


def test_pruning_is_off_without_the_rule_set() raises:
    """`NoRules.prepare` is the identity, which is what makes it the control
    arm every `_check` above depends on."""
    var plan = table(_wide()).select("a")
    assert_equal(String(plan), String(plan.optimize[NoRules]()))


# ---------------------------------------------------------------------------
# Filter pushdown through a join
# ---------------------------------------------------------------------------
def _left_table() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, 3], int64).copy(),
            array([10, 20, 30], int64).copy(),
        ],
        names=["id", "lval"],
    )


def _right_table() raises -> RecordBatch:
    return record_batch(
        [
            array([1, 2, 3], int64).copy(),
            array([100, 200, 300], int64).copy(),
        ],
        names=["rid", "rval"],
    )


def test_optimizer_pushes_a_filter_into_the_left_side_of_a_join() raises:
    """A predicate reading only left columns shrinks the left input first."""
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_INNER)
        .filter(col("lval", int64) > lit(15, int64))
    )
    _fires(plan)
    var out = String(plan.optimize[AllRules]())
    assert_true(out.find("Join(") < out.find("Filter("), out)
    _check(plan, [[2, 3], [20, 30], [2, 3], [200, 300]])


def test_optimizer_pushes_a_filter_into_the_right_side_of_a_join() raises:
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_INNER)
        .filter(col("rval", int64) > lit(150, int64))
    )
    _fires(plan)
    _check(plan, [[2, 3], [20, 30], [2, 3], [200, 300]])


def test_optimizer_does_not_push_a_filter_below_an_outer_join() raises:
    """The rule that would silently return nothing.

    An outer join manufactures NULL rows for non-matches, and a predicate
    evaluated before that step never sees them. `LEFT JOIN ... WHERE r IS NULL`
    is the anti-join idiom; pushing its predicate into the right side answers
    empty. Asserted on where the predicate lands rather than only on rows,
    because a wrong answer here depends on the data happening to contain a
    non-match: the chain takes it, and evaluates it after the LEFT join.
    """
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_LEFT)
        .filter(col("rval", int64) > lit(150, int64))
    )
    var optimized = plan.optimize[AllRules]()
    ref chain = optimized.get[JoinChain]()
    assert_false(chain.inputs[1][].isa[Filter](), String(optimized))
    assert_equal(_landing(chain), chain.planned_order().root())


def test_optimizer_pushes_a_filter_on_a_name_both_sides_carry() raises:
    """Both sides carry `lval`, and the join renames the right one
    `lval_right`, so a predicate names exactly one side: `lval` moves into the
    left input, and `lval_right` runs on the right one under the name it was
    written with."""
    var both = record_batch(
        [array([1, 2, 3], int64).copy(), array([9, 9, 9], int64).copy()],
        names=["id", "lval"],
    )
    var plan = (
        table(both.copy())
        .join(table(_left_table()), [0], [0], JOIN_INNER)
        .filter(col("lval", int64) > lit(5, int64))
    )
    var out = plan.optimize[AllRules]()
    assert_true(out.isa[JoinChain](), String(out))
    assert_true(out.get[JoinChain]().inputs[0][].isa[Filter](), String(out))
    _check(
        plan.sort_by([col("id", int64)], [True]),
        [[1, 2, 3], [9, 9, 9], [10, 20, 30]],
    )

    var right = (
        table(both^)
        .join(table(_left_table()), [0], [0], JOIN_INNER)
        .filter(col("lval_right", int64) > lit(15, int64))
    )
    _fires(right)
    _check(
        right.sort_by([col("id", int64)], [True]),
        [[2, 3], [9, 9], [20, 30]],
    )


def test_a_select_over_an_absorbed_filter_folds_into_the_chain() raises:
    """SQL's `SELECT ... FROM l JOIN r WHERE ...`: the filter sits between
    the chain and the projection until `PushFilterIntoJoin` takes it in, and
    then the projection is the chain's own output — one node."""
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_INNER)
        .filter(col("lval", int64) + col("rval", int64) > lit(250, int64))
        .select(["rval", "id"])
    )
    var out = plan.optimize[AllRules]()
    assert_true(out.isa[JoinChain](), String(out))
    _check(plan, [[300], [3]])


# ---------------------------------------------------------------------------
# Negative cases for the rules that had none
# ---------------------------------------------------------------------------
def test_optimizer_does_not_merge_limits_across_a_filter() raises:
    """`Limit(Filter(Limit(x)))` is not two adjacent limits, and merging them
    would skip the filter's row reduction entirely."""
    var plan = (
        table(_batch())
        .limit(4)
        .filter(col("b", int64) > lit(15, int64))
        .limit(2)
    )
    _check(plan, [[1, 4], [20, 30]])


def test_optimizer_does_not_remove_a_sort_across_a_limit() raises:
    """`Sort(Limit(Sort(x)))` keeps both: the inner sort decides *which* rows
    the limit takes, so discarding it changes the row set, not just the
    order."""
    var plan = (
        table(_batch())
        .sort_by([col("a", int64)], [True])
        .limit(3)
        .sort_by([col("b", int64)], [True])
    )
    _check(plan, [[3, 1, 1], [10, 20, 40]])


def test_optimizer_empty_propagates_through_sort_and_project() raises:
    """The `Sort` and `Project` arms of `PropagateEmpty`, which the `Filter`
    case did not exercise.

    The `Project` arm is the one with a rule of its own: it must keep the
    *projection's* schema, not the input's, or an empty result comes back with
    the wrong columns.
    """
    var plan = (
        table(_batch()).limit(0).sort_by([col("a", int64)], [True]).select("b")
    )
    var out = String(plan.optimize[AllRules]())
    assert_equal(_occurrences(out, "Sort("), 0)
    assert_true("Empty(" in out, out)
    _check(plan, [List[Int]()])


# ---------------------------------------------------------------------------
# Constant folding, done in the value constructors
# ---------------------------------------------------------------------------
def test_folding_collapses_a_conjunction_with_true() raises:
    """`x AND TRUE` is `x`, folded where the operands are still concrete."""
    var p = and_(
        gt(column("a"), literal(Int64Scalar(1).to_dyn())),
        literal(BoolScalar(True).to_dyn()),
    )
    assert_equal(
        String(p), String(gt(column("a"), literal(Int64Scalar(1).to_dyn())))
    )


def test_folding_annihilates_a_conjunction_with_false() raises:
    """`x AND FALSE` is `FALSE` — true in Kleene logic even against a null."""
    var p = and_(
        gt(column("a"), literal(Int64Scalar(1).to_dyn())),
        literal(BoolScalar(False).to_dyn()),
    )
    assert_equal(String(p), String(literal(BoolScalar(False).to_dyn())))


def test_folding_leaves_a_null_literal_alone() raises:
    """`x AND NULL` is neither `x` nor `NULL`, so it must not fold.

    It is `FALSE` when `x` is false and `NULL` otherwise. Folding a null as if
    it were false is the one way this rewrite changes answers.
    """
    var p = and_(
        gt(column("a"), literal(Int64Scalar(1).to_dyn())),
        literal(BoolScalar.null().to_dyn()),
    )
    assert_true("and" in String(p), String(p))


def test_folding_cancels_double_negation() raises:
    assert_equal(String(not_(not_(col("a")))), String(col("a")))


def test_a_folded_false_filter_collapses_the_plan() raises:
    """Why folding matters: it is what lets `PropagateEmpty` reach a real
    query. Nobody writes `LIMIT 0`, but predicates fold to `FALSE` often.
    """
    var plan = table(_batch()).filter(
        and_(
            gt(column("a"), literal(Int64Scalar(1).to_dyn())),
            literal(BoolScalar(False).to_dyn()),
        )
    )
    assert_true("Empty(" in String(plan.optimize[AllRules]()), String(plan))


# ---------------------------------------------------------------------------
# Filter below an aggregate
# ---------------------------------------------------------------------------
def test_optimizer_pushes_a_filter_below_a_group_key() raises:
    """A predicate on a group key filters before grouping, not after."""
    var plan = (
        table(_batch())
        .aggregate([col("b", int64).sum().alias("total")], [col("a", int64)])
        .filter(col("a", int64) > lit(3, int64))
    )
    _fires(plan)
    var out = String(plan.optimize[AllRules]())
    assert_true(out.find("Aggregate(") < out.find("Filter("), out)


def test_optimizer_does_not_push_a_filter_on_an_aggregate_output() raises:
    """`HAVING sum(...) > n` names a column the aggregate computes, so it
    cannot move below the node that computes it."""
    var plan = (
        table(_batch())
        .aggregate([col("b", int64).sum().alias("total")], [col("a", int64)])
        .filter(col("total", int64) > lit(30, int64))
    )
    _inert(plan)


def test_optimizer_does_not_push_a_filter_below_a_keyless_aggregate() raises:
    """A keyless aggregate emits one row; a filter above it asks a different
    question than one below."""
    var plan = (
        table(_batch())
        .aggregate([col("a", int64).sum().alias("total")])
        .filter(col("total", int64) > lit(1, int64))
    )
    _inert(plan)


# ---------------------------------------------------------------------------
# Empty through a join, per kind
# ---------------------------------------------------------------------------
def test_optimizer_inner_join_with_an_empty_side_is_empty() raises:
    var plan = (
        table(_left_table())
        .limit(0)
        .join(table(_right_table()), [0], [0], JOIN_INNER)
    )
    _fires(plan)
    assert_true("Empty(" in String(plan.optimize[AllRules]()))


def test_optimizer_left_join_with_an_empty_right_is_not_empty() raises:
    """The case that would delete rows: a `LEFT JOIN` with an empty right
    still emits every left row, padded with NULLs."""
    var plan = table(_left_table()).join(
        table(_right_table()).limit(0), [0], [0], JOIN_LEFT
    )
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Join("), 1)


# ---------------------------------------------------------------------------
# EliminateFilter and SplitConjunction
# ---------------------------------------------------------------------------
def test_optimizer_eliminates_a_constant_true_filter() raises:
    var plan = table(_batch()).filter(literal(BoolScalar(True).to_dyn()))
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Filter("), 0)
    _check(plan, [[3, 1, 4, 1, 5, 9], [10, 20, 30, 40, 50, 60]])


def test_optimizer_a_constant_false_filter_becomes_empty() raises:
    var plan = table(_batch()).filter(literal(BoolScalar(False).to_dyn()))
    _fires(plan)
    assert_true("Empty(" in String(plan.optimize[AllRules]()))
    _check(plan, [List[Int](), List[Int]()])


def test_optimizer_a_null_predicate_is_not_a_constant() raises:
    """A filter keeps rows where the predicate is `TRUE`. A null predicate is
    not `FALSE` — it merely fails to select — so it must not be folded into
    either branch."""
    var plan = table(_batch()).filter(literal(BoolScalar.null().to_dyn()))
    _inert(plan)


def test_optimizer_splits_a_conjunction_into_stacked_filters() raises:
    var plan = table(_batch()).filter(
        (col("a", int64) > lit(1, int64)) & (col("b", int64) < lit(50, int64))
    )
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Filter("), 2)
    # a > 1 keeps rows 0,2,4,5; b < 50 keeps 0,1,2,3; both keep 0 and 2.
    _check(plan, [[3, 4], [10, 30]])


def test_optimizer_splits_a_conjunction_recursively() raises:
    """`a AND (b AND c)` reaches three filters, terminating at the first
    operand that is not an `AND`."""
    var plan = table(_batch()).filter(
        (col("a", int64) > lit(0, int64))
        & (
            (col("b", int64) < lit(50, int64))
            & (col("a", int64) < lit(5, int64))
        )
    )
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Filter("), 3)
    _check(plan, [[3, 1, 4, 1], [10, 20, 30, 40]])


def test_optimizer_does_not_split_a_disjunction() raises:
    """`OR` is not a conjunction; splitting it would keep rows neither half
    selects."""
    var plan = table(_batch()).filter(
        (col("a", int64) > lit(8, int64)) | (col("b", int64) < lit(15, int64))
    )
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Filter("), 1)
    _check(plan, [[3, 9], [10, 60]])


def test_optimizer_split_conjuncts_reach_both_sides_of_a_join() raises:
    """The payoff: `a AND b` spanning both inputs used to stay above the join.

    Split, each half lands in the side it names — which is why splitting runs
    before the pushdown rules rather than after.
    """
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_INNER)
        .filter(
            (col("lval", int64) > lit(15, int64))
            & (col("rval", int64) > lit(150, int64))
        )
    )
    var out = String(plan.optimize[AllRules]())
    assert_true(out.find("Join(") < out.find("Filter("), out)
    _check(plan, [[2, 3], [20, 30], [2, 3], [200, 300]])


def test_optimizer_reaches_a_fixpoint_with_splitting_and_pushdown() raises:
    """Splitting rebuilds through `.filter()`, which re-enters the loop.

    If splitting and any pushdown rule ever disagreed, the driver would spin
    until `MAX_PASSES` truncated it — silently returning a half-optimized plan
    rather than hanging. This asserts they converge on the shape most likely to
    expose it: multiple conjuncts over a join.
    """
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_INNER)
        .filter(
            (col("lval", int64) > lit(15, int64))
            & (col("rval", int64) > lit(150, int64))
        )
    )
    var once = plan.optimize[AllRules]()
    assert_equal(String(once), String(once.optimize[AllRules]()))


# ---------------------------------------------------------------------------
# Window — the node only MergeWindows reads
# ---------------------------------------------------------------------------
def test_no_rule_moves_a_node_into_a_window() raises:
    """No rule moves a node into a `Window`'s input, and that must stay true
    by test.

    `MergeWindows` is the one rule that matches a `Window`, and it folds
    stacked windows together without moving a row, so a lone window is inert
    to it. Rules still fire *inside* a window's subtree —
    `Optimizer._rewritten_children` walks through `Window.traverse` — and
    `ColumnPruning` is the pass that actually stops, by falling through every
    `isa` to `return node.copy()`. A future rule that pattern-matched a
    `Filter` above a window would push a predicate below it, and a window
    computes over its whole partition: it would then compute over a pruned
    one. Wrong numbers, no error, and `precompile` cannot see it because
    nothing fails to compile.

    Both shapes are asserted: a filter above a window, which is the `QUALIFY`
    plan the golden corpus exercises, and a limit above one, which `TopN` and
    `MergeLimits` would otherwise be candidates to move.
    """
    var b = record_batch(
        [array([3, 1, 4], int64).copy(), array([10, 20, 30], int64).copy()],
        names=["a", "b"],
    )
    var windowed = table(b^).with_columns(
        ["rn"], [row_number().over(order_by=[col("a", int64)])]
    )
    _inert(windowed.filter(col("a", int64) > lit(1, int64)))
    _inert(windowed.limit(2))

    # A `Sort` *above* the window is a different matter: `TopN` rewrites
    # `Limit(Sort(x))` to a bounded sort whatever `x` is, and that is sound
    # here because the window has already run below it. What must hold is
    # not that nothing fires but that nothing reaches *into* the window, so
    # the assertion is on the subtree rather than on the whole plan.
    var stacked = windowed.sort_by([col("a", int64)], [True]).limit(2)
    var optimized = String(stacked.optimize[AllRules]())
    # Counted with `_occurrences`, not `in`: see that helper's own note about
    # a search loop in this file that spun for an hour. The window must render
    # exactly once and still sit directly on its source, wherever `TopN` moved
    # the bound above it.
    # One substring spanning the boundary, not two counted separately:
    # `Window(...)` and `InMemoryTable(...)` each appearing once is also true
    # of `Window(Limit(InMemoryTable(...)))`, which is exactly the rewrite
    # this is meant to forbid. `Window.write_to` emits its input immediately
    # after the paren, so adjacency is expressible.
    assert_equal(
        _occurrences(optimized, "Window(InMemoryTable(3 rows)"), 1, optimized
    )


def _ties() raises -> RecordBatch:
    """`a` ties once, so `rank` and `dense_rank` differ."""
    return record_batch(
        [
            array([3, 1, 4, 1], int64).copy(),
            array([10, 20, 30, 40], int64).copy(),
        ],
        names=["a", "b"],
    )


def test_merge_windows_folds_independent_windows_into_one_node() raises:
    """`with_columns` builds a `Window` per value; independent values fold
    into one node, which sorts once per distinct window — twice here, not
    three times — and neither the answers nor the column order change."""
    var plan = table(_ties()).with_columns(
        ["r", "lg", "d"],
        [
            rank().over(order_by=[col("a", int64)]),
            col("b", int64).lag().over(order_by=[col("b", int64)]),
            dense_rank().over(order_by=[col("a", int64)]),
        ],
    )
    _fires(plan)
    var optimized = plan.optimize[AllRules]()
    var rendered = String(optimized)
    assert_equal(_occurrences(rendered, "Window("), 1, rendered)
    assert_equal(
        String(optimized.optimize[AllRules]()), rendered, "not idempotent"
    )
    _check(
        plan,
        [
            [3, 1, 4, 1],
            [10, 20, 30, 40],
            [3, 1, 4, 1],
            [_NULL, 10, 20, 30],
            [2, 1, 3, 1],
        ],
    )


def test_merge_windows_keeps_a_value_above_the_column_it_reads() raises:
    """`lag(rn)` reads `rn`, so it cannot be computed beside it."""
    var plan = (
        table(_ties())
        .with_columns(["rn"], [row_number().over(order_by=[col("a", int64)])])
        .with_columns(
            ["prev"],
            [col("rn", int64).lag().over(order_by=[col("a", int64)])],
        )
    )
    _inert(plan)
    _check(
        plan,
        [[3, 1, 4, 1], [10, 20, 30, 40], [3, 1, 4, 2], [2, _NULL, 3, 1]],
    )


# ---------------------------------------------------------------------------
# PushFilterIntoScan
# ---------------------------------------------------------------------------
def _never_read() raises -> DynRelation:
    """A scan of a file that does not exist.

    Legal, and the reason plan-shape cases here need no fixture: a `Relation`
    is a description, so building and optimizing a plan touches no I/O. The
    end-to-end half — that the installed pruner really skips row groups — is in
    `test_scan_pruning.mojo`, over a file pyarrow wrote.
    """
    return scan(
        String("/tmp/marrow_optimizer_never_read.parquet"),
        schema([field("a", int64), field("b", int64)]),
    )


def _pruners(plan: DynRelation) raises -> Int:
    """How many pruners the scan under this plan's top `Filter` carries."""
    assert_true(
        plan.isa[Filter](), "expected a Filter on top of: " + String(plan)
    )
    ref below = plan.get[Filter]().input[]
    assert_true(below.isa[ParquetScan](), "expected a scan under the Filter")
    return len(below.get[ParquetScan]().pruners)


def test_push_filter_into_scan_moves_the_pruner_onto_the_scan() raises:
    """The scan gains the pruner **and keeps the filter above it** — pruning is
    conservative, so the exact predicate still has to run."""
    var plan = _never_read().filter(col("a", int64) > lit(150, int64))
    assert_equal(_pruners(plan), 0)

    var out = plan.optimize[AllRules]()
    assert_equal(_pruners(out), 1)
    assert_equal(_occurrences(String(out), String("pruned by 1")), 1)
    assert_equal(_occurrences(String(out), String("Filter(")), 1)


def test_push_filter_into_scan_is_idempotent() raises:
    """Optimizing twice installs one pruner, not two.

    The driver runs to a fixpoint, so a rule that leaves its own precondition
    standing duplicates its work on every later pass — and this one does leave
    it standing, deliberately: the `Filter` has to survive because pruning is
    conservative. What stops it is the scan answering "already carried" to a
    predicate it holds, which is also what keeps the two renderings equal so
    the driver can see it has converged.
    """
    var plan = _never_read().filter(col("a", int64) > lit(150, int64))
    var once = plan.optimize[AllRules]()
    assert_equal(_pruners(once), 1)
    assert_equal(_pruners(once.optimize[AllRules]()), 1)


def test_push_filter_into_scan_reaches_a_scan_below_a_sort() raises:
    """`Filter(Sort(scan))` prunes: `PushFilterBelowSort` moves the filter down
    first, and it rebuilds with `with_input`, so the pruner arrives with it."""
    var plan = (
        _never_read()
        .sort_by([col("b", int64)], [True])
        .filter(col("a", int64) > lit(150, int64))
    )
    assert_equal(_occurrences(String(plan), String("pruned by")), 0)
    assert_equal(
        _occurrences(String(plan.optimize[AllRules]()), String("pruned by 1")),
        1,
    )


def test_push_filter_into_scan_reaches_a_scan_below_a_project() raises:
    """The reach the descent does not have.

    The retired `to_operator` descent threaded a pushdown down the plan and
    `Project` **cleared** it — a predicate names output columns, which may be computed, so
    the descent cannot tell a pass-through from a rename and gives up on all of
    them. `PushFilterBelowProject` can: it asks `passes_through_all` by name,
    and once it has moved the filter the scan is adjacent and this rule fires.
    The same argument covers `Aggregate` and `Join`, which the descent also
    clears at.
    """
    var plan = (
        _never_read()
        .project(
            ["a", "sum"], [col("a", int64), col("a", int64) + col("b", int64)]
        )
        .filter(col("a", int64) > lit(150, int64))
    )
    assert_equal(
        _occurrences(String(plan.optimize[AllRules]()), String("pruned by 1")),
        1,
    )


def test_push_filter_into_scan_conjoins_stacked_filters() raises:
    """`Filter(a, Filter(b, scan))` lands **two** pruners, not one.

    The parity case for the retired `to_operator` descent, which conjoined a
    filter chain on the way down. Plain adjacency would prune only `b`: after
    the inner filter is rewritten the outer one still has a `Filter` beneath
    it, and never becomes adjacent to anything. `_grown` walks the chain for
    exactly this shape.
    """
    var plan = (
        _never_read()
        .filter(col("a", int64) > lit(60, int64))
        .filter(col("a", int64) < lit(140, int64))
    )
    var out = plan.optimize[AllRules]()
    assert_equal(_occurrences(String(out), String("pruned by 2")), 1)


def test_scan_pruning_rule_set_prunes_with_two_rules() raises:
    """`ScanPruning` is the rule set a compiled query names to prune
    (`QueryCli(plan.optimize[ScanPruning]()).run()`), so it has to reach a scan
    on its own — through a filter chain and under an unbounded sort, which is
    the whole of what the descent it replaced could reach."""
    var plan = (
        _never_read()
        .sort_by([col("b", int64)], [True])
        .filter(col("a", int64) > lit(150, int64))
    )
    assert_equal(
        _occurrences(
            String(plan.optimize[ScanPruning]()), String("pruned by 1")
        ),
        1,
    )


def test_scan_pruning_rule_set_leaves_everything_else_alone() raises:
    """It is two rules, not sixteen: a no-op `Project` that `AllRules` deletes
    survives `ScanPruning` untouched. That is what keeps an AOT binary from
    linking the other fourteen."""
    var plan = _never_read().select(["a", "b"])
    assert_equal(
        _occurrences(String(plan.optimize[ScanPruning]()), String("Project(")),
        1,
    )
    assert_equal(
        _occurrences(String(plan.optimize[AllRules]()), String("Project(")), 0
    )


def test_push_filter_into_scan_does_not_reach_below_a_limit() raises:
    """The one shape that would change the answer, and the reason the rule
    matches on adjacency.

    `filter(p)` above `limit(3)` means "the first three rows, then `p`". A scan
    that skipped a row group would hand `Limit` a different first three, and
    rows the correct query returns would disappear. No rule moves a filter
    below a `Limit`, so the filter never becomes adjacent and this one never
    matches.
    """
    var plan = _never_read().limit(3).filter(col("a", int64) > lit(150, int64))
    assert_equal(
        _occurrences(String(plan.optimize[AllRules]()), String("pruned by")), 0
    )


def test_push_filter_into_scan_prunes_a_boxed_predicate() raises:
    """A predicate that arrived already boxed prunes exactly as well as a
    typed one.

    It did not, for a release: pruning lived in a second box built at
    `.filter()` where the concrete type was still visible, so a predicate the
    caller had already erased reached the scan with nothing to say. `mask` is a
    slot on `DynValue` now, so the box carries it like any other method and
    this case is ordinary rather than special.

    Boxing has to be forced with a typed local: `.filter()` has two overloads
    and the typed one wins wherever the concrete type is still visible.
    """
    var boxed: DynValue = col("a", int64) > lit(150, int64)
    var plan = _never_read().filter(boxed^)
    assert_equal(
        _occurrences(String(plan.optimize[AllRules]()), String("pruned by 1")),
        1,
    )


def test_push_filter_into_scan_prunes_a_runtime_lane_predicate() raises:
    """The runtime lane prunes too — `RuntimeValue` implements `mask` itself,
    reading the same zone maps through the erased comparison kernels.

    Worth pinning: "runtime lane" and "erased" read as the same thing and are
    not. A `RuntimeValue` is an interpreted node with its own `mask`; a
    `DynValue` is a box that forwards to whatever it holds.
    """
    var plan = _never_read().filter(gt(column("a"), literal(Int64Scalar(150))))
    assert_equal(
        _occurrences(String(plan.optimize[AllRules]()), String("pruned by 1")),
        1,
    )


def test_push_filter_into_scan_lands_each_conjunct_separately() raises:
    """Ordering, made visible.

    `SplitConjunction` runs first and `PushFilterIntoScan` lands **both**
    halves: two filters above the scan, two pruners on it. That the halves are
    `DynValue`s no longer costs anything, which is what reversed the order —
    the compound predicate used to be the only one with a pruning method, so
    the rule had to catch it before the split and settle for one blunt entry.

    Two sharp bounds beat one blunt `AND`: `a > 60 AND a < 140` as a single
    predicate skips a chunk only when neither bound can, where the halves
    separately skip everything outside `[60, 140]`.
    """
    var plan = _never_read().filter(
        (col("a", int64) > lit(60, int64)) & (col("a", int64) < lit(140, int64))
    )
    var out = plan.optimize[AllRules]()
    assert_equal(_occurrences(String(out), String("Filter(")), 2)
    assert_equal(_occurrences(String(out), String("pruned by 2")), 1)


# ---------------------------------------------------------------------------
# Build sides — the cost `JoinOrdering` spends on every join
#
# Every rule here is true by inspection; the join search is true by
# arithmetic. So each case says which of the two numbers it is relying on, and
# the pair `_fires` / `_inert` is load-bearing in both directions: a pass that
# never fires and a pass that fires on everything both pass an equivalence
# check.
# ---------------------------------------------------------------------------
def _big_left() raises -> RecordBatch:
    """Twelve rows against `_right_table`'s three.

    Four times the rows is well clear of any rounding in `Cost`: indexing this
    side charges twelve rows of `hash_build` — twelve comparisons and 192
    bytes held — where indexing the other charges three.
    """
    var ids = List[Optional[Int]](capacity=12)
    var vals = List[Optional[Int]](capacity=12)
    for i in range(12):
        ids.append(i + 1)
        vals.append((i + 1) * 10)
    return record_batch(
        [array(ids, int64).copy(), array(vals, int64).copy()],
        names=["id", "lval"],
    )


def test_join_ordering_builds_the_smaller_input() raises:
    """Twelve rows on the left, three on the right: index the right.

    A chain prints a step's non-default build side, so the flip is visible in
    the plan rather than only in the operator.
    """
    var plan = table(_big_left()).join(
        table(_right_table()), [0], [0], JOIN_INNER
    )
    assert_false(String(plan).find("build=right") >= 0, String(plan))
    _fires(plan)

    var out = String(plan.optimize[AllRules]())
    assert_true(out.find("build=right") >= 0, out)


def test_join_ordering_leaves_the_smaller_left_alone() raises:
    """Three rows on the left, twelve on the right: already right.

    A tie keeps the current side and so does a loss, which is what makes the
    pass idempotent — a second run prices the same two sides and declines.
    """
    var plan = table(_right_table()).join(
        table(_big_left()), [0], [0], JOIN_INNER
    )
    _inert(plan)


def test_join_ordering_is_a_no_op_when_nothing_is_known() raises:
    """An unknown is not a number that can lose a comparison.

    A scan nobody read a footer for estimates unknown, so `Cost.total()` is
    unknown on both sides and `known()` answers `None`. Treating that as zero
    would make the unestimable plan win every comparison, which is the failure
    `Approx`'s absorbing unknown exists to prevent one layer down.
    """
    var s = schema([field("id", int64), field("v", int64)])
    var plan = scan("/nonexistent-left.parquet", s).join(
        scan("/nonexistent-right.parquet", s), [0], [0], JOIN_INNER
    )
    _inert(plan)


def test_join_ordering_declines_a_kind_it_cannot_mirror() raises:
    """`commutes` is the guard, and the pass declines where a kernel raises.

    `JOIN_CROSS` has a constant and no implementation, so `JoinKind.mirror`
    raises for it. Reaching `mirror` from inside the pass would turn an
    optimization into an error on a plan that was merely unexecutable.
    """
    var plan = table(_big_left()).join(
        table(_right_table()), [0], [0], JOIN_CROSS
    )
    _inert(plan)


def test_join_ordering_does_not_move_the_schema() raises:
    """Bit-identical, for every kind the kernel implements.

    The property the whole rewrite rests on: a chain's schema is its output
    projection, which no step reads, so flipping a side cannot rename or
    reorder a column. Compared as
    whole `Schema`s — `Schema.__eq__` compares `Field`s, which compare all four
    of their members — rather than through a rendering, which is how a join
    dropping `nullable` went unnoticed in the first place.
    """
    var kinds: List[JoinKind] = [
        JOIN_INNER,
        JOIN_LEFT,
        JOIN_RIGHT,
        JOIN_FULL,
        JOIN_SEMI,
        JOIN_ANTI,
        JOIN_RIGHT_SEMI,
        JOIN_RIGHT_ANTI,
    ]
    for ref k in kinds:
        var plan = table(_big_left()).join(table(_right_table()), [0], [0], k)
        var optimized = plan.optimize[AllRules]()
        assert_true(
            plan.schema() == optimized.schema(),
            String("kind ", k, ": the schema moved"),
        )


def test_join_ordering_returns_the_same_rows() raises:
    """Same rows, same columns, after the flip.

    Sorted, because row order *does* move with the build side — it follows the
    probe side — and that is the one thing the rewrite does not promise. The
    multiset is what it promises, and a trailing sort is how `_check`'s
    positional comparison is made to test it.
    """
    var plan = (
        table(_big_left())
        .join(table(_right_table()), [0], [0], JOIN_INNER)
        .sort_by([col("id", int64)], [True])
    )
    _fires(plan)
    _check(plan, [[1, 2, 3], [10, 20, 30], [1, 2, 3], [100, 200, 300]])


def test_join_ordering_returns_the_same_rows_for_a_left_join() raises:
    """The kind whose *physical* kind changes when the side flips.

    A LEFT join built on its right input is a physical RIGHT join, which is
    also the case where `JoinOperator._blocks_on_probe_side` has to ask the
    physical kind: a logical LEFT buffers its probe side, the physical RIGHT
    it becomes streams. Getting that backwards is a wrong answer, not a slow
    one, and only an end-to-end case sees it.
    """
    var plan = (
        table(_big_left())
        .join(table(_right_table()), [0], [0], JOIN_LEFT)
        .sort_by([col("id", int64)], [True])
    )
    _fires(plan)
    _check(
        plan,
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            [
                1,
                2,
                3,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
            ],
            [
                100,
                200,
                300,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
                _NULL,
            ],
        ],
    )


def test_join_ordering_is_absent_from_scan_pruning() raises:
    """The rule set is a comptime parameter, so a binary links exactly what it
    names — and `ScanPruning` exists so an AOT program can name row-group
    pruning and nothing else."""
    var plan = table(_big_left()).join(
        table(_right_table()), [0], [0], JOIN_INNER
    )
    assert_equal(String(plan), String(plan.optimize[ScanPruning]()))
    assert_equal(String(plan), String(plan.optimize[NoRules]()))


# ---------------------------------------------------------------------------
# Join order on small hand-checked chains
#
# The rows here are written by hand, so these are the cases that say *sound*
# rather than merely consistent; the search's own section below covers what it
# chooses. A declining search is asserted on `_shape` — the tree by participant
# and kind — since a build side may still change where the tree does not.
# ---------------------------------------------------------------------------
def _shape_of(order: JoinOrder, node: Int) -> String:
    if not order.is_join(node):
        return String("#", node)
    ref j = order.join(node)
    return String(
        "(",
        _shape_of(order, j.left),
        " ",
        j.link.kind,
        " ",
        _shape_of(order, j.right),
        ")",
    )


def _shape(plan: DynRelation) raises -> String:
    """A chain's planned tree by participant and kind — no keys, no build
    sides."""
    var order = plan.get[JoinChain]().planned_order()
    return _shape_of(order, order.root())


def _landing(chain: JoinChain) raises -> Int:
    """The node of `chain`'s planned tree its first filter lands on."""
    var order = chain.planned_order()
    return order.landing(order.participants(), chain.rules(), chain.filters[0])


def _facts() raises -> RecordBatch:
    """Twelve rows carrying two foreign keys — the big input."""
    var did = List[Optional[Int]](capacity=12)
    var eid = List[Optional[Int]](capacity=12)
    var amt = List[Optional[Int]](capacity=12)
    for i in range(12):
        did.append((i % 3) + 1)
        eid.append((i % 2) + 1)
        amt.append(i + 1)
    return record_batch(
        [
            array(did, int64).copy(),
            array(eid, int64).copy(),
            array(amt, int64).copy(),
        ],
        names=["fdid", "feid", "amt"],
    )


def _dim() raises -> RecordBatch:
    """`A` — three rows, two columns. Its `dval` is `_mid`'s key."""
    return record_batch(
        [array([1, 2, 3], int64).copy(), array([11, 22, 33], int64).copy()],
        names=["did", "dval"],
    )


def _mid() raises -> RecordBatch:
    """`B` — three rows, **one** column, joined to `A` on one side and to `C`
    on the other.

    One column on purpose: the two shapes differ only in how wide a row the
    intermediate they build holds, so the fixture has to make that width
    differ. Right-deep builds `B` alone (8 bytes a row); left-deep builds
    `A|B` (24). An unanalysed in-memory table records no distinct count, so
    the *cardinality* term, which would normally decide this, is blind here.
    """
    return record_batch([array([11, 22, 33], int64).copy()], names=["mid"])


def _leaf() raises -> RecordBatch:
    """`C` — six rows, two columns, referencing `_mid` three ways over.

    Six rather than three because at three the two shapes tie exactly, and a
    tie keeps the plan as written.
    """
    return record_batch(
        [
            array([11, 22, 33, 11, 22, 33], int64).copy(),
            array([1, 2, 3, 4, 5, 6], int64).copy(),
        ],
        names=["ck", "cv"],
    )


def _other() raises -> RecordBatch:
    """Two rows keyed by `eid`, which `_facts.feid` references."""
    return record_batch(
        [array([1, 2], int64).copy(), array([100, 200], int64).copy()],
        names=["eid", "eval"],
    )


def _three_way() raises -> DynRelation:
    """`(A ⋈ B) ⋈ C`, keyed on `mid` — a column of `B`.

    The inner join's schema is `did, dval, mid`, so index 2 is `B`'s only
    column: the outer predicate reads `B` and `C` and never `A`, which is
    exactly the condition that makes the two associations agree.
    """
    var inner = table(_dim()).join(table(_mid()), [1], [0], JOIN_INNER)
    return inner.join(table(_leaf()), [2], [0], JOIN_INNER)


def test_join_order_three_way_returns_the_same_rows() raises:
    """Hand-written rows on both sides — the assertion that says *sound*
    rather than merely different.

    Sorted on `cv`, which is unique, so neither the join's probe order nor the
    build side it ends up with can reorder the comparison.
    """
    var plan = _three_way().sort_by([col("cv", int64)], [True])
    _check(
        plan,
        [
            [1, 2, 3, 1, 2, 3],
            [11, 22, 33, 11, 22, 33],
            [11, 22, 33, 11, 22, 33],
            [11, 22, 33, 11, 22, 33],
            [1, 2, 3, 4, 5, 6],
        ],
    )


def test_join_order_declines_an_outer_join() raises:
    """Associativity is a property of the *inner* join.

    An outer join manufactures rows for non-matches, and when it does so
    depends on the association, so it keeps its place and its children.
    Checked in both positions, because a guard that read only one of the two
    kinds would pass the other.
    """
    var outer_inner = table(_dim()).join(table(_mid()), [1], [0], JOIN_LEFT)
    var plan = outer_inner.join(table(_leaf()), [2], [0], JOIN_INNER)
    var optimized = plan.optimize[AllRules]()
    assert_equal(_shape(optimized), _shape(plan), String(optimized))

    var inner = table(_dim()).join(table(_mid()), [1], [0], JOIN_INNER)
    var plan2 = inner.join(table(_leaf()), [2], [0], JOIN_LEFT)
    var optimized2 = plan2.optimize[AllRules]()
    assert_equal(_shape(optimized2), _shape(plan2), String(optimized2))


def test_join_order_is_a_no_op_when_nothing_is_known() raises:
    """Three unestimable scans: no tree can be shown cheaper, so none is
    chosen — and no build side is either, which is why `_inert` still holds
    here and nowhere else in this section."""
    var s = schema([field("k", int64), field("v", int64)])
    var t = schema([field("k2", int64), field("v2", int64)])
    var plan = (
        scan("/nonexistent-a.parquet", s)
        .join(scan("/nonexistent-b.parquet", t), [0], [0], JOIN_INNER)
        .join(scan("/nonexistent-c.parquet", s), [2], [0], JOIN_INNER)
    )
    _inert(plan)


# ---------------------------------------------------------------------------
# The passes together — fixpoint and oscillation
#
# The rewrite loop moves filters into chains and folds projections; the join
# search runs in `finish`, after it, and reads the estimates they leave. A
# rule that undid what the search chose, or a search that re-ordered what a
# second run handed it, would oscillate under a driver that converges on a
# *rendered* comparison and answer a half-optimized plan with no diagnostic
# anywhere.
# ---------------------------------------------------------------------------
def _tail() raises -> RecordBatch:
    """`D` — three rows keyed on `_leaf.cv`, so a four-way chain is
    expressible."""
    return record_batch(
        [
            array([1, 2, 3], int64).copy(),
            array([1000, 2000, 3000], int64).copy(),
        ],
        names=["tk", "tv"],
    )


def _four_way() raises -> DynRelation:
    """`((A ⋈ B) ⋈ C) ⋈ D`, left-deep, every join INNER: one region of four
    leaves, the largest the hand-checked rows cover."""
    return (
        table(_dim())
        .join(table(_mid()), [1], [0], JOIN_INNER)
        .join(table(_leaf()), [2], [0], JOIN_INNER)
        .join(table(_tail()), [4], [0], JOIN_INNER)
    )


def _right_deep_three_way() raises -> DynRelation:
    """`A ⋈ (B ⋈ C)` written by hand, a bushy shape rather than a left-deep
    one: the search must reach one fixpoint from either spelling."""
    var inner = table(_mid()).join(table(_leaf()), [0], [0], JOIN_INNER)
    return table(_dim()).join(inner^, [1], [0], JOIN_INNER)


def _mixed_kinds() raises -> DynRelation:
    """LEFT, then INNER, then SEMI over one chain.

    Three regions of one step each: the search takes the build side of each
    kind that commutes and re-trees nothing, so a chain that mixes kinds
    exercises every arm of the planner on one plan.
    """
    return (
        table(_dim())
        .join(table(_mid()), [1], [0], JOIN_LEFT)
        .join(table(_leaf()), [2], [0], JOIN_INNER)
        .join(table(_other()), [0], [0], JOIN_SEMI)
    )


def _partly_estimable() raises -> DynRelation:
    """A three-way chain whose middle input nobody read a footer for.

    The search may not spend a number here, and the interesting part is that
    the other two inputs are exactly estimable: a cost model that let a known
    side stand in for an unknown one would fire, and a fixpoint is the cheapest
    place to notice that it did.
    """
    var s = schema([field("k", int64), field("v", int64)])
    return (
        table(_dim())
        .join(scan("/nonexistent-middle.parquet", s), [0], [0], JOIN_INNER)
        .join(table(_leaf()), [1], [0], JOIN_INNER)
    )


def _filtered_three_way() raises -> DynRelation:
    """`(A ⋈ B) ⋈ C` with a filter under the first join and one above the last.

    The lower one is already where it belongs and the upper one is pushed into
    the participant it names, so the search sees cardinalities that changed
    under it — which is why it runs in `finish`, after the rewrite loop.
    """
    return (
        table(_dim())
        .filter(col("did", int64) > lit(1, int64))
        .join(table(_mid()), [1], [0], JOIN_INNER)
        .join(table(_leaf()), [2], [0], JOIN_INNER)
        .filter(col("cv", int64) > lit(2, int64))
    )


def _join_shapes() raises -> List[DynRelation]:
    """The join plans the cost-rule tests run over."""
    return [
        _three_way(),
        _right_deep_three_way(),
        _four_way(),
        _filtered_three_way(),
        _mixed_kinds(),
        _partly_estimable(),
        _natural_key(),
        _self_join(),
        table(_big_left()).join(table(_right_table()), [0], [0], JOIN_INNER),
    ]


def _settles(plan: DynRelation) raises:
    """Re-optimizing an optimized plan changes nothing, `prepare` included.
    `optimize` is deterministic, so one re-application proves it."""
    var once = plan.optimize[AllRules]()
    var twice = once.optimize[AllRules]()
    assert_equal(
        String(twice),
        String(once),
        String("re-optimizing moved the plan:\n", once, "\n", twice),
    )


def _single_passes_settle(plan: DynRelation) raises:
    """One bottom-up pass at a time: the plan stops changing and stays stopped.

    **`optimize()` cannot make this assertion.** `Optimizer.run` caps at
    `MAX_PASSES` and answers whatever the sixteenth pass produced, so two rules
    alternating a plan between two shapes return the *same* answer every time
    `optimize` is called — an even-period oscillation is invisible from outside
    the driver, and re-optimizing an already-optimized plan agrees with itself
    forever. Stepping the driver a pass at a time is the only way to see one.

    The steps are what `run` does: `AllRules.prepare`, passes of
    `Optimizer.rewrite` to a fixpoint, `AllRules.finish`, and passes again —
    the second run must settle too, and stay settled, with the search's
    output under it.
    """
    var current = _passes(AllRules.prepare(plan))
    _ = _passes(AllRules.finish(current))


def _passes(plan: DynRelation) raises -> DynRelation:
    """Twelve single passes over `plan`: it stops changing and stays stopped.
    Answers the settled plan."""
    var current = plan.copy()
    var rendered = String(current)
    var settled = -1
    for i in range(12):
        var next = Optimizer[AllRules].rewrite(current)
        var next_rendered = String(next)
        if next_rendered == rendered:
            if settled < 0:
                settled = i + 1
        else:
            assert_true(
                settled < 0,
                String(
                    "pass ",
                    i + 1,
                    " moved a plan that settled at pass ",
                    settled,
                    ":\n",
                    rendered,
                    "\n",
                    next_rendered,
                ),
            )
        current = next^
        rendered = next_rendered^
    assert_true(settled >= 0, "no fixpoint in twelve passes: " + rendered)
    return current^


def test_optimizer_is_idempotent_on_join_plans() raises:
    var shapes = _join_shapes()
    for ref plan in shapes:
        _settles(plan)


def test_the_cost_rules_do_not_oscillate() raises:
    """`optimize` stops at a fixed pass count and answers whatever that pass
    produced, so an oscillation is invisible from outside it. Step single
    passes instead, and require the plan to stop changing and stay stopped."""
    var shapes = _join_shapes()
    for ref plan in shapes:
        _single_passes_settle(plan)


def test_a_four_way_join_returns_the_same_rows_optimized() raises:
    """The rows, on both arms, for the largest region written by hand.

    `A ⋈ B` is three rows on `dval = mid`, each matching two rows of `C`, of
    which `D` keeps the three whose `cv` is 1, 2 or 3. Sorted on `cv`, which is
    unique in the result, so neither probe order nor a chosen build side can
    reorder the comparison.
    """
    var plan = _four_way().sort_by([col("cv", int64)], [True])
    _fires(plan)
    _check(
        plan,
        [
            [1, 2, 3],
            [11, 22, 33],
            [11, 22, 33],
            [11, 22, 33],
            [1, 2, 3],
            [1, 2, 3],
            [1000, 2000, 3000],
        ],
    )


def test_a_filtered_three_way_join_returns_the_same_rows_optimized() raises:
    """A filter below the chain and a filter above it, with both cost rules
    free to act on what they left behind."""
    var plan = _filtered_three_way().sort_by([col("cv", int64)], [True])
    _fires(plan)
    _check(
        plan,
        [
            [3, 2, 3],
            [33, 22, 33],
            [33, 22, 33],
            [33, 22, 33],
            [3, 5, 6],
        ],
    )


def test_a_four_way_join_keeps_its_schema_through_the_optimizer() raises:
    """Bit-identical, not merely equivalent — `Schema.__eq__` compares every
    member of every `Field`, which is how a join that dropped `nullable` stayed
    hidden through two reimplementations."""
    var plan = _four_way()
    assert_true(
        plan.schema() == plan.optimize[AllRules]().schema(),
        String("the schema moved: ", plan.optimize[AllRules]()),
    )
    var mixed = _mixed_kinds()
    assert_true(
        mixed.schema() == mixed.optimize[AllRules]().schema(),
        String("the schema moved: ", mixed.optimize[AllRules]()),
    )


# ---------------------------------------------------------------------------
# Join ordering — the search over a region's trees
#
# The dangerous pass in this file: a wrong tree is a silent wrong answer. So
# every case that asserts a tree changed is paired with the same rows through
# `NoRules`, and the cases that assert it declines name the condition they
# decline on. A chain renders its tree with every key as `#participant.column`
# and every non-default build side, so its rendering is its fingerprint.
#
# The fixtures are analysed (`DynRelation.analyze`), so the search sees bounds
# and distinct counts; an unanalysed in-memory table estimates every join at
# the smaller side and gives the search little to choose with.
# ---------------------------------------------------------------------------
def _keyed(
    names: List[String], rows: Int, modulus: List[Int]
) raises -> DynRelation:
    """An analysed table of `rows` rows, column `j` holding `i % modulus[j]`."""
    var columns = List[DynArray](capacity=len(names))
    for j in range(len(names)):
        var values = List[Optional[Int]](capacity=rows)
        for i in range(rows):
            values.append(i % modulus[j])
        columns.append(array(values^, int64).to_dyn())
    return table(record_batch(columns^, names=names.copy())).analyze()


def _find_join_of(plan: DynRelation, a: String, b: String) raises -> Bool:
    """Does the planned tree join the participant holding `a` directly to the
    one holding `b`, whichever way round?"""
    ref chain = plan.get[JoinChain]()
    var p = chain.ref_of(a).input
    var q = chain.ref_of(b).input
    for ref j in chain.planned_order().joins:
        if (j.left == p and j.right == q) or (j.left == q and j.right == p):
            return True
    return False


def _agree(plan: DynRelation) raises:
    """`NoRules` and `AllRules` return the same rows, compared column by
    column after sorting on every column — so neither a probe order nor a
    build side can reorder the comparison — and the same schema."""
    var keys = List[DynValue]()
    var ascending = List[Bool]()
    for ref f in plan.schema().fields:
        keys.append(column(f.name.copy()))
        ascending.append(True)
    var sorted = plan.sort_by(keys^, ascending^)
    var ctx = ExecContext()
    var before = sorted.optimize[NoRules]().execute(ctx)
    var after = sorted.optimize[AllRules]().execute(ctx)
    assert_true(
        sorted.optimize[AllRules]().schema() == plan.schema(), "schema moved"
    )
    assert_equal(before.num_rows(), after.num_rows(), "row count differs")
    assert_true(before.num_rows() > 0, "the fixture joins to nothing")
    for i in range(before.num_columns()):
        assert_equal(_col(before, i), _col(after, i), "OPTIMIZED rows differ")


def _chain() raises -> DynRelation:
    """`A - B - C - D`, written left to right, where joining `C` early is
    what blows up: `B ⋈ C` fans out 250-fold, `C ⋈ D` shrinks to a thirtieth."""
    return (
        _keyed(["ax"], 1_000, [50])
        .join(_keyed(["bx", "by"], 200, [50, 20]), [0], [0], JOIN_INNER)
        .join(_keyed(["cy", "cz"], 5_000, [20, 1_000]), [2], [0], JOIN_INNER)
        .join(_keyed(["dz"], 30, [1_000]), [4], [0], JOIN_INNER)
    )


def _star() raises -> DynRelation:
    """A fact table and three dimensions, the largest dimension joined first
    and the one a filter makes smallest joined last."""
    var fact = _keyed(["f1", "f2", "f3"], 3_000, [300, 20, 7])
    return (
        fact.join(_keyed(["d1"], 300, [300]), [0], [0], JOIN_INNER)
        .join(_keyed(["d2"], 20, [20]), [1], [0], JOIN_INNER)
        .join(
            _keyed(["d3"], 7, [7]).filter(col("d3", int64) < lit(2, int64)),
            [2],
            [0],
            JOIN_INNER,
        )
    )


def _cycle() raises -> DynRelation:
    """`A - B - C - D - A`: the last join closes the cycle on two key pairs."""
    var ab = _keyed(["ab", "ad"], 400, [40, 10]).join(
        _keyed(["ba", "bc"], 400, [40, 25]), [0], [0], JOIN_INNER
    )
    var abc = ab.join(_keyed(["cb", "cd"], 100, [25, 5]), [3], [0], JOIN_INNER)
    return abc.join(
        _keyed(["dc", "da"], 50, [5, 10]), [5, 1], [0, 1], JOIN_INNER
    )


def _clique() raises -> DynRelation:
    """Four tables on one key, written as a chain: every pair of them is
    joinable through the class, the implied pairs included."""
    return (
        _keyed(["k1"], 2_000, [100])
        .join(_keyed(["k2"], 50, [50]), [0], [0], JOIN_INNER)
        .join(_keyed(["k3"], 900, [100]), [1], [0], JOIN_INNER)
        .join(_keyed(["k4"], 10, [10]), [2], [0], JOIN_INNER)
    )


def _join_order_shapes() raises -> List[DynRelation]:
    return [_chain(), _star(), _cycle(), _clique()]


def _natural_key() raises -> DynRelation:
    """Three analysed tables all keyed `k`, one `k` in the output: the
    shape ibis's name rules make reorderable, since no input's `k` is ever
    confused with another's."""
    return (
        _keyed(["k", "av"], 1_000, [50, 7])
        .join(_keyed(["k", "bv"], 1_000, [50, 3]), [0], [0], JOIN_INNER)
        .join(_keyed(["k", "cv"], 10, [5, 2]), [0], [0], JOIN_INNER)
    )


def _self_join() raises -> DynRelation:
    """A table joined to itself, then to a small one on the same key."""
    var big = _keyed(["k", "v"], 2_000, [100, 9])
    return big.join(big.copy(), [0], [0], JOIN_INNER).join(
        _keyed(["k", "w"], 10, [10, 3]), [0], [0], JOIN_INNER
    )


def _shifted(join: PlannedJoin, inputs: Int, by: Int) -> PlannedJoin:
    """`join` moved `by` places up a list of joins over `inputs`
    participants."""
    var out = join.copy()
    if out.left >= inputs:
        out.left += by
    if out.right >= inputs:
        out.right += by
    return out^


def _root_of(
    s: Set[Int], joins: List[PlannedJoin], at: Int, inputs: Int
) -> Int:
    """The root of a tree over inputs `s` whose joins start at `at`."""
    if len(joins) == 0:
        for p in s:
            return p
    return inputs + at + len(joins) - 1


def _every_tree(
    rules: JoinRules, inputs: Int, s: Set[Int]
) raises -> List[List[PlannedJoin]]:
    """Every cross-product-free bushy tree over the inputs in `s` of a chain's
    one multi-join, both build sides of every join, keyed by its own rule —
    the space the search claims to be exact over, spelled out the slow way. A
    tree is its joins, root last; a single input is a tree of none."""
    var out = List[List[PlannedJoin]]()
    if len(s) == 1:
        out.append(List[PlannedJoin]())
        return out^
    var low = 0
    while low not in s:
        low += 1
    # Every subset holding the lowest input: each other input doubles them.
    var subs: List[Set[Int]] = [{low}]
    for i in s:
        if i != low:
            for k in range(len(subs)):
                var grown = subs[k].copy()
                grown.add(i)
                subs.append(grown^)
    for ref sub in subs:
        if len(sub) < len(s):
            var other = s - sub
            var keys = rules.keys(sub, other)
            if len(keys[0]) > 0:
                var lefts = _every_tree(rules, inputs, sub)
                var rights = _every_tree(rules, inputs, other)
                for ref l in lefts:
                    for ref r in rights:
                        for side in [BUILD_LEFT, BUILD_RIGHT]:
                            var joins = l.copy()
                            for ref j in r:
                                joins.append(_shifted(j, inputs, len(l)))
                            joins.append(
                                PlannedJoin(
                                    _root_of(sub, l, 0, inputs),
                                    _root_of(other, r, len(l), inputs),
                                    JoinLink(
                                        JOIN_INNER,
                                        JOIN_ALL,
                                        side,
                                        keys[0].copy(),
                                        keys[1].copy(),
                                    ),
                                )
                            )
                            out.append(joins^)
    return out^


def test_join_order_is_the_brute_force_optimum() raises:
    """Over chain, star, cycle, a one-key clique and a chain with a filter
    between two of its joins, the tree the planner picks costs exactly the
    cheapest of every tree the multi-join can take — which holds only because
    every node over one set of inputs estimates it the same way, and a filter
    is priced where it lands, so the search prices a tree as `cost()` does."""
    var prepared_shapes = List[DynRelation]()
    for ref plan in _join_order_shapes():
        prepared_shapes.append(AllRules.prepare(plan))
    prepared_shapes.append(
        PushFilterIntoJoin.apply(
            AllRules.prepare(
                _chain().filter(col("bx", int64) < col("cy", int64))
            )
        )
    )
    assert_equal(
        len(prepared_shapes[4].get[JoinChain]().filters),
        1,
        String(prepared_shapes[4]),
    )
    for ref prepared in prepared_shapes:
        ref chain = prepared.get[JoinChain]()
        var rules = chain.rules()
        var pricing = JoinPricing(chain)
        var best = Optional[Int](None)
        var trees = _every_tree(rules, len(chain.inputs), rules.everything())
        for ref joins in trees:
            var tree = JoinOrder(len(chain.inputs))
            for ref j in joins:
                _ = tree.add(j.copy())
            tree.verify(chain)
            var total = pricing.tree(tree).total().known()
            assert_true(Bool(total), "a tree has no cost: " + String(tree))
            if not best or total.value() < best.value():
                best = total
        var found = JoinOrdering.order(chain)
        assert_equal(
            pricing.tree(found).total().known().value(),
            best.value(),
            String(found),
        )


def test_join_order_returns_the_same_rows() raises:
    var shapes = _join_order_shapes()
    for ref plan in shapes:
        _agree(plan)


def test_join_order_improves_every_shape() raises:
    """Each fixture is written in an order the search can beat."""
    var shapes = _join_order_shapes()
    for ref plan in shapes:
        var written = plan.optimize[NoRules]().cost().total().known().value()
        var chosen = plan.optimize[AllRules]().cost().total().known().value()
        assert_true(
            chosen < written,
            String(chosen, " is not below ", written, " for ", plan),
        )


def test_join_order_joins_through_an_implied_edge() raises:
    """`k1 = k2` and `k2 = k3` imply `k1 = k3`, and joining the two small
    tables on it first is only possible because the class says so."""
    var plan = (
        _keyed(["k1"], 40, [40])
        .join(_keyed(["k2"], 5_000, [40]), [0], [0], JOIN_INNER)
        .join(_keyed(["k3"], 20, [20]), [1], [0], JOIN_INNER)
    )
    var optimized = plan.optimize[AllRules]()
    assert_true(_find_join_of(optimized, "k1", "k3"), String(optimized))
    _agree(plan)


def test_join_order_never_moves_the_output() raises:
    """The chain answers with its own projection, so a new tree needs nothing
    above it to put the columns back — under an observer or not."""
    var plan = _chain()
    var optimized = plan.optimize[AllRules]()
    assert_true(optimized.isa[JoinChain](), String(optimized))
    assert_true(_shape(optimized) != _shape(plan), String(optimized))
    assert_true(optimized.schema() == plan.schema(), String(optimized))

    var grouped = plan.aggregate(
        [col("ax", int64).count().alias("n")], [col("dz", int64)]
    )
    var optimized_grouped = grouped.optimize[AllRules]()
    ref agg = optimized_grouped.get[Aggregate]()
    assert_true(agg.input[].isa[JoinChain](), String(optimized_grouped))
    assert_true(optimized_grouped.schema() == grouped.schema())


def test_join_order_reorders_a_natural_key_chain() raises:
    """Three inputs all calling their key `k`: one class, one output `k`, and
    the small input joined first — to either large one, which tie."""
    var plan = _natural_key()
    var optimized = plan.optimize[AllRules]()
    assert_true(_shape(optimized) != _shape(plan), String(optimized))
    assert_true(
        _find_join_of(optimized, "av", "cv")
        or _find_join_of(optimized, "bv", "cv"),
        String(optimized),
    )
    _agree(plan)


def test_join_order_reorders_a_self_join() raises:
    """Two participants are the same table under the same names; every key
    and output names its participant, so the search moves them freely."""
    var plan = _self_join()
    var optimized = plan.optimize[AllRules]()
    assert_true(_shape(optimized) != _shape(plan), String(optimized))
    _agree(plan)


def test_join_order_reaches_one_cost_however_the_chain_was_written() raises:
    """`A ⋈ (B ⋈ (C ⋈ D))` nests three chains; `MergeJoinChains` splices
    them into the region `((A ⋈ B) ⋈ C) ⋈ D` is, over the same participants
    in the same order — so both spellings reach the same cheapest tree."""
    var bushy = _keyed(["ax"], 1_000, [50]).join(
        _keyed(["bx", "by"], 200, [50, 20]).join(
            _keyed(["cy", "cz"], 5_000, [20, 1_000]).join(
                _keyed(["dz"], 30, [1_000]), [1], [0], JOIN_INNER
            ),
            [1],
            [0],
            JOIN_INNER,
        ),
        [0],
        [0],
        JOIN_INNER,
    )
    assert_true(bushy.schema() == _chain().schema())
    var spliced = bushy.optimize[AllRules]()
    assert_equal(len(spliced.get[JoinChain]().inputs), 4, String(spliced))
    var from_bushy = bushy.optimize[AllRules]().cost().total().known()
    var from_chain = _chain().optimize[AllRules]().cost().total().known()
    assert_equal(from_bushy.value(), from_chain.value())
    _agree(bushy)


def test_join_order_compares_two_members_of_one_input() raises:
    """`a.x = b.k` and `a.y = b.k = c.m` put two of `a`'s columns in one
    class — `a.x = a.y`, which only a join comparing both evaluates. The key
    rule pairs every member of an input no join has compared yet, so any
    tree the search picks keeps the answer."""
    var plan = (
        _keyed(["x", "y"], 100, [10, 10])
        .join(_keyed(["k"], 50, [10]), [0], [0], JOIN_INNER)
        .join(_keyed(["m"], 20, [10]), [1, 2], [0, 0], JOIN_INNER)
    )
    var rules = plan.get[JoinChain]().rules()
    var keys = rules.keys({0}, {1})
    assert_equal(len(keys[0]), 2, String(plan))
    _agree(plan)


def test_join_order_takes_the_greedy_path_past_its_budget() raises:
    """A budget of one pair abandons the exhaustive search at once; the greedy
    ordering still beats the written chain and returns the same rows."""
    var plan = AllRules.prepare(_chain())
    ref chain = plan.get[JoinChain]()
    var pricing = JoinPricing(chain)
    var written = pricing.tree(
        JoinOrder.written(len(chain.inputs), chain.links)
    )
    var chosen = pricing.tree(JoinOrdering.order[1](chain))
    assert_true(
        chosen.total().known().value() < written.total().known().value(),
        String(chosen, " vs ", written),
    )
    # A one-pair budget leaves the dynamic program no pair to price, so a
    # tree cheaper than written is the greedy one.
    var greedy_plan = plan.with_chain(
        chain.with_order(JoinOrdering.order[1](chain))
    )
    _same_rows(greedy_plan, plan)


def test_join_order_places_a_two_leaf_filter_at_the_lowest_join() raises:
    """`bx < cy` reads two participants and cannot move into either; the
    chain keeps it and evaluates it at the lowest join holding both — below
    the root, since the chosen tree joins `B` and `C` beneath it — and the
    rows do not move."""
    var plan = _chain().filter(col("bx", int64) < col("cy", int64))
    var optimized = plan.optimize[AllRules]()
    assert_true(optimized.isa[JoinChain](), String(optimized))
    ref chain = optimized.get[JoinChain]()
    assert_equal(len(chain.filters), 1, String(optimized))
    var order = chain.planned_order()
    var landing = _landing(chain)
    assert_true(order.is_join(landing), String(optimized))
    assert_true(landing != order.root(), String(optimized))
    _agree(plan)


def test_join_order_folds_a_deep_and_a_column_free_filter() raises:
    """An eighteen-way chain: a filter on the deepest participant moves into
    it in one rewrite, however many joins sit above, and a parameter test,
    which reads no column, is kept by the chain rather than dropped."""
    var plan = _keyed(["c0"], 30, [30])
    for i in range(1, 18):
        plan = plan.join(
            _keyed([String("c", i)], 30, [30]), [i - 1], [0], JOIN_INNER
        )
    var filtered = plan.filter(col("c0", int64) < lit(10, int64)).filter(
        param("keep", bool_)
    )
    var optimized = filtered.optimize[AllRules]()
    assert_true(optimized.isa[JoinChain](), String(optimized))
    ref chain = optimized.get[JoinChain]()
    assert_true(chain.inputs[0][].isa[Filter](), String(optimized))
    assert_equal(len(chain.filters), 1, String(optimized))
    var keep: Bindings = {"keep": BoolScalar(True).to_dyn()}
    var drop: Bindings = {"keep": BoolScalar(False).to_dyn()}
    var written = filtered.optimize[NoRules]().execute(ExecContext(), keep)
    assert_equal(optimized.execute(ExecContext(), keep).num_rows(), 10)
    assert_equal(written.num_rows(), 10)
    assert_equal(optimized.execute(ExecContext(), drop).num_rows(), 0)


def test_join_order_keeps_a_join_any_attached_the_same_way() raises:
    """At most one match per probe row depends on the sides, so a `JOIN_ANY`
    step keeps its build side and the participant it attaches, while inner
    joins over its probe side may cross it. Any match is a correct one; here
    every match carries the same `b`, so the rows agree exactly."""
    var plan = (
        _keyed(["a"], 50, [10])
        .join(
            _keyed(["b"], 20, [10]), [0], [0], JOIN_INNER, BUILD_RIGHT, JOIN_ANY
        )
        .join(_keyed(["c"], 5, [10]), [0], [0], JOIN_INNER)
        .join(_keyed(["d"], 400, [10]), [2], [0], JOIN_INNER)
    )
    var optimized = plan.optimize[AllRules]()
    var found = False
    for ref j in optimized.get[JoinChain]().planned_order().joins:
        if j.link.strictness == JOIN_ANY:
            found = True
            assert_equal(j.right, 1, String(optimized))
            assert_true(j.link.build_side == BUILD_RIGHT, String(optimized))
    assert_true(found, String(optimized))
    _agree(plan)


def _attached_last(kind: JoinKind, strictness: UInt8) raises -> DynRelation:
    """`F` attached to `D1` by `kind`, then joined to `D2`, which keeps a
    tenth of `F`: cheaper the other way round."""
    var side = BUILD_RIGHT if strictness == JOIN_ANY else BUILD_LEFT
    return (
        _keyed(["a", "b"], 2_000, [200, 100])
        .join(
            _keyed(["a1", "x"], 100, [400, 7]),
            [0],
            [0],
            kind,
            side,
            strictness,
        )
        .join(_keyed(["b2"], 10, [1_000]), [1], [0], JOIN_INNER)
    )


def test_join_order_moves_an_inner_join_below_an_attaching_step() raises:
    """LEFT, SEMI, ANTI and `JOIN_ANY` over `F`'s side all commute with an
    inner join over `F` that reads nothing they attach, so the search joins
    the selective `D2` first and attaches `D1` to what is left — the same
    rows, cheaper."""
    var kinds: List[JoinKind] = [JOIN_LEFT, JOIN_SEMI, JOIN_ANTI, JOIN_INNER]
    var strictness: List[UInt8] = [JOIN_ALL, JOIN_ALL, JOIN_ALL, JOIN_ANY]
    for i in range(len(kinds)):
        var plan = _attached_last(kinds[i], strictness[i])
        var optimized = plan.optimize[AllRules]()
        var order = optimized.get[JoinChain]().planned_order()
        ref root = order.join(order.root())
        assert_true(root.link.kind == kinds[i], String(optimized))
        assert_true(root.link.strictness == strictness[i], String(optimized))
        var written = plan.optimize[NoRules]().cost().total().known().value()
        var chosen = optimized.cost().total().known().value()
        assert_true(chosen < written, String(chosen, " vs ", written))
        _agree(plan)


def test_join_order_attaches_to_an_attached_leaf_after_its_spine() raises:
    """`(#0 ⟖ #1) ⟕ #2 ⋉ #3`, every key on `#0`, which the RIGHT join
    attaches to `#1`: `#2` and `#3` attach to `#0`, so neither may join it
    before `#1` has — the search only builds trees that hold."""
    var plan = (
        _keyed(["a", "x"], 40, [8, 5])
        .join(_keyed(["b"], 20, [8]), [0], [0], JOIN_RIGHT)
        .join(_keyed(["c"], 30, [8]), [0], [0], JOIN_LEFT)
        .join(_keyed(["d"], 10, [4]), [0], [0], JOIN_SEMI)
        .filter(col("x", int64) < col("c", int64))
    )
    _agree(plan)


def test_join_order_keeps_an_inner_join_reading_an_attached_side() raises:
    """`D2` joins on `D1`'s `x`, which a LEFT join pads: compared below the
    LEFT join it would keep `F`'s unmatched rows, so the LEFT join stays
    beneath it."""
    var plan = (
        _keyed(["a", "b"], 2_000, [200, 100])
        .join(_keyed(["a1", "x"], 100, [400, 7]), [0], [0], JOIN_LEFT)
        .join(_keyed(["x2"], 3, [1_000]), [3], [0], JOIN_INNER)
    )
    var optimized = plan.optimize[AllRules]()
    var order = optimized.get[JoinChain]().planned_order()
    assert_true(
        order.join(order.root()).link.kind == JOIN_INNER, String(optimized)
    )
    _agree(plan)


def test_join_order_prices_a_string_column_nobody_measured() raises:
    """An unanalysed string column is priced at Spark's default width, so a
    tree over its table can still be priced and chosen."""
    var dim = table(
        record_batch(
            [array([1, 2, 3], int64).copy(), array(["x", "y", "z"]).copy()],
            names=["sk", "label"],
        )
    )
    var plan = (
        table(_facts())
        .join(dim^, [0], [0], JOIN_INNER)
        .join(table(_other()), [1], [0], JOIN_INNER)
    )
    assert_true(plan.optimize[AllRules]().cost().total().is_known())
    # `_agree` compares int64 columns, so the label is left out of it.
    _agree(plan.drop(["label"]))


def test_join_order_orders_both_regions_an_outer_join_separates() raises:
    """A LEFT join attaches its right side as one leaf, so the region along
    its spine and the region under the side it attaches are each ordered."""
    var plan = _chain().join(_star(), [0], [1], JOIN_LEFT)
    var optimized = plan.optimize[AllRules]()
    var written = plan.optimize[NoRules]().cost().total().known().value()
    var chosen = optimized.cost().total().known().value()
    assert_true(chosen < written, String(chosen, " vs ", written))
    assert_true(optimized.schema() == plan.schema())


def _same_rows(a: DynRelation, b: DynRelation) raises:
    """`a` and `b` return the same rows, compared after sorting on every
    column so neither a probe order nor a build side can reorder them."""
    var keys = List[DynValue]()
    var ascending = List[Bool]()
    for ref f in a.schema().fields:
        keys.append(column(f.name.copy()))
        ascending.append(True)
    var ctx = ExecContext()
    var left = a.sort_by(keys.copy(), ascending.copy()).execute(ctx)
    var right = b.sort_by(keys^, ascending^).execute(ctx)
    assert_equal(left.num_rows(), right.num_rows())
    for i in range(left.num_columns()):
        assert_equal(_col(left, i), _col(right, i))


def test_join_order_does_not_move_the_estimate() raises:
    """A set of leaves has one cardinality whatever tree joins it, so the
    chain the search picks estimates exactly as the written one — rows and
    every column's counts."""
    var shapes = _join_order_shapes()
    for ref plan in shapes:
        var optimized = plan.optimize[AllRules]()
        assert_true(
            String(_shape(optimized)) != String(_shape(plan)), String(optimized)
        )
        assert_equal(
            String(optimized.estimate()),
            String(plan.estimate()),
            String(optimized),
        )


def test_join_order_keeps_the_written_tree_on_a_tie() raises:
    """Three tables alike in every statistic, written with the second join
    hashing the small table rather than the intermediate: every tree then
    costs the same, and a tree is replaced only by a strictly cheaper one."""
    var plan = (
        _keyed(["a"], 100, [10])
        .join(_keyed(["b"], 100, [10]), [0], [0], JOIN_INNER)
        .join(_keyed(["c"], 100, [10]), [0], [0], JOIN_INNER, BUILD_RIGHT)
    )
    _inert(plan)
