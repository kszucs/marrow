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
from ...scalars import BoolScalar, Int64Scalar
from ..builders import col, count_star, lit, row_number, table
from ..runtime.values import and_, column, gt, literal, not_
from ...kernels.join import JOIN_INNER, JOIN_LEFT
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
    """`Project(Project(x))` collapses when the outer only selects."""
    var plan = table(_batch()).select("a", "b").select("a")
    _fires(plan)
    assert_equal(_occurrences(String(plan.optimize[AllRules]()), "Project("), 1)
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
    empty. Asserted as *inert* rather than only on rows, because a wrong answer
    here depends on the data happening to contain a non-match.
    """
    var plan = (
        table(_left_table())
        .join(table(_right_table()), [0], [0], JOIN_LEFT)
        .filter(col("rval", int64) > lit(150, int64))
    )
    _inert(plan)


def test_optimizer_leaves_an_ambiguous_join_predicate_alone() raises:
    """A predicate on a column both sides carry could belong to either, so the
    rule declines rather than guessing a side."""
    var both = record_batch(
        [array([1, 2, 3], int64).copy(), array([9, 9, 9], int64).copy()],
        names=["id", "lval"],
    )
    var plan = (
        table(both^)
        .join(table(_left_table()), [0], [0], JOIN_INNER)
        .filter(col("lval", int64) > lit(5, int64))
    )
    _inert(plan)


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
# Window — the node no rule knows about
# ---------------------------------------------------------------------------
def test_no_rule_rewrites_a_plan_containing_a_window() raises:
    """A `Window` is inert to every rule, and that must stay true by test.

    Today it holds for a reason that is invisible from here: `Window` is not
    in `optimizer.mojo`'s import list, so no `isa[Window]()` exists and no
    rule can *match* one. Rules still fire *inside* a window's subtree —
    `Optimizer._rewritten_children` walks through `Window.traverse` — and
    `ColumnPruning` is the pass that actually stops, by falling through every
    `isa` to `return node.copy()`. That is a real guarantee and a fragile one — a
    future rule that imports the node and pattern-matches a `Filter` above it
    would push a predicate below a window, which computes over its whole
    partition and would then compute over a pruned one. Wrong numbers, no
    error, and `precompile` cannot see it because nothing fails to compile.

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
