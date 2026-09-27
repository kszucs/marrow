# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`distinct()` and the set relations — `Union`, `Difference` (`EXCEPT`) and
`Intersection` — each with and without `ALL`.

What these pin is the part SQL gets backwards from everywhere else: NULL is
**equal to itself** in a set operation, so a NULL row deduplicates and a NULL
the right side also has is removed. Every expected answer below is sorted
first, because none of these operations promises an order.
"""

from std.testing import assert_equal, assert_raises, assert_true

from ...builders import array
from ...arrays import Int64Array
from ...dtypes import int64
from ...tabular import RecordBatch, record_batch
from ..builders import col, lit, table
from ..logical import (
    Difference,
    DynRelation,
    EmptyRelation,
    Except,
    Intersect,
    Intersection,
    Union,
)
from ..optimizer import AllRules, ColumnPruning, PropagateEmpty


def _left() raises -> RecordBatch:
    """`k` repeats 1 and 3 and holds one NULL; `v` rides along."""
    return record_batch(
        [
            array([1, 1, 2, None, 3, 3, 3], int64).copy(),
            array([10, 10, 20, 30, 40, 40, 40], int64).copy(),
        ],
        names=["k", "v"],
    )


def _right() raises -> RecordBatch:
    """Named differently from `_left`, to show matching is by position."""
    return record_batch(
        [
            array([1, None, 3, 3, 5], int64).copy(),
            array([10, 30, 40, 40, 50], int64).copy(),
        ],
        names=["a", "b"],
    )


def _sorted_k(plan: DynRelation) raises -> Int64Array:
    """Column `k` of `plan`'s result, sorted with NULLs first."""
    var out = plan.sort_by([col("k", int64)], [True]).execute()
    return out.columns[0].as_int64().copy()


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------
def test_distinct_keeps_one_row_per_distinct_row() raises:
    """`(1, 10)` twice and `(3, 40)` three times collapse; NULL keeps one."""
    var plan = table(_left()).distinct()
    assert_true(_sorted_k(plan) == array([None, 1, 2, 3], int64))
    assert_true(plan.schema() == _left().schema)


def test_distinct_compares_whole_rows() raises:
    """Rows equal on `k` but not on `v` are distinct rows."""
    var b = record_batch(
        [
            array([1, 1, 1], int64).copy(),
            array([10, 20, 10], int64).copy(),
        ],
        names=["k", "v"],
    )
    assert_equal(table(b^).distinct().execute().num_rows(), 2)


# ---------------------------------------------------------------------------
# UNION
# ---------------------------------------------------------------------------
def test_union_all_keeps_every_row_of_both_sides() raises:
    var plan = table(_left()).union_all(table(_right()))
    assert_equal(plan.execute().num_rows(), 12)
    assert_true(
        _sorted_k(plan)
        == array([None, None, 1, 1, 1, 2, 3, 3, 3, 3, 3, 5], int64)
    )


def test_union_takes_the_left_names_and_matches_by_position() raises:
    var plan = table(_left()).union_all(table(_right()))
    var s = plan.schema()
    assert_equal(s.fields[0].name, "k")
    assert_equal(s.fields[1].name, "v")
    assert_true(plan.execute().schema == s)


def test_union_deduplicates_with_null_equal_to_itself() raises:
    """Both sides hold a `(NULL, 30)`; the answer holds exactly one."""
    var plan = table(_left()).union(table(_right()))
    assert_true(_sorted_k(plan) == array([None, 1, 2, 3, 5], int64))


# ---------------------------------------------------------------------------
# EXCEPT
# ---------------------------------------------------------------------------
def test_except_removes_every_row_the_right_side_has() raises:
    """`(1, 10)` goes although the left has two of it; the right side's
    `(NULL, 30)` removes the left's."""
    var plan = table(_left()).except_(table(_right()))
    assert_true(_sorted_k(plan) == array([2], int64))


def test_except_all_subtracts_counts() raises:
    """Left has `(1, 10)` x2 and `(3, 40)` x3, right x1 and x2: one of each
    survives, and `(2, 20)`, which the right lacks, survives whole."""
    var plan = table(_left()).except_all(table(_right()))
    assert_true(_sorted_k(plan) == array([1, 2, 3], int64))


# ---------------------------------------------------------------------------
# INTERSECT
# ---------------------------------------------------------------------------
def test_intersect_keeps_rows_on_both_sides_once() raises:
    var plan = table(_left()).intersect(table(_right()))
    assert_true(_sorted_k(plan) == array([None, 1, 3], int64))


def test_intersect_all_keeps_the_smaller_count() raises:
    """`(3, 40)`: three on the left, two on the right -> two."""
    var plan = table(_left()).intersect_all(table(_right()))
    assert_true(_sorted_k(plan) == array([None, 1, 3, 3], int64))


def test_intersect_of_an_empty_side_is_empty() raises:
    var empty = table(_right()).filter(col("a", int64) > lit(100, int64))
    var plan = table(_left()).intersect(empty^)
    assert_equal(plan.execute().num_rows(), 0)
    assert_equal(plan.execute().num_columns(), 2)


def test_except_of_an_empty_right_side_is_the_distinct_left() raises:
    var empty = table(_right()).filter(col("a", int64) > lit(100, int64))
    var plan = table(_left()).except_(empty^)
    assert_true(_sorted_k(plan) == array([None, 1, 2, 3], int64))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_a_set_relation_rejects_a_different_column_count() raises:
    with assert_raises(contains="columns"):
        _ = table(_left()).union_all(table(_right()).select(["a"]))


def test_a_set_relation_rejects_a_different_dtype() raises:
    var strings = record_batch(
        [
            array(["x"]).copy(),
            array([1], int64).copy(),
        ],
        names=["k", "v"],
    )
    with assert_raises(contains="column 0"):
        _ = table(_left()).intersect(table(strings^))


def test_a_set_relation_over_a_limited_input_reads_the_slice() raises:
    """`LimitOperator` hands on a slice sharing its parent's children, so the
    operator must read columns through the batch's offset."""
    var plan = (
        table(_left())
        .limit(2, offset=5)
        .union_all(table(_right()).limit(1, offset=4))
    )
    assert_true(_sorted_k(plan) == array([3, 3, 5], int64))


def test_a_deduplicating_union_is_distinct_over_union_all() raises:
    """The optimizer should see the dedup as the `Aggregate` it is."""
    var plan = table(_left()).union(table(_right()))
    assert_true(String(plan).startswith("Aggregate(Union("))


def test_multiplicity_rules() raises:
    """The one thing `Intersection` and `Difference` differ by. Distinct
    `EXCEPT` is *not* `max(l - r, 0)` capped at one: a row the right side has
    at all is gone."""
    assert_equal(Intersect.copies(3, 2, True), 2)
    assert_equal(Intersect.copies(3, 2, False), 1)
    assert_equal(Intersect.copies(3, 0, False), 0)
    assert_equal(Except.copies(3, 2, True), 1)
    assert_equal(Except.copies(3, 2, False), 0)
    assert_equal(Except.copies(3, 0, False), 1)


def test_set_relations_print_their_own_names() raises:
    var intersection = table(_left()).intersect_all(table(_right()))
    assert_true(String(intersection).startswith("Intersection("))
    assert_true(String(intersection).endswith(", all)"))
    assert_true(
        String(table(_left()).except_(table(_right()))).startswith(
            "Difference("
        )
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def test_propagate_empty_collapses_an_intersect_with_an_empty_side() raises:
    var empty: DynRelation = EmptyRelation(RecordBatch.empty(_right().schema))
    var node: DynRelation = Intersection(table(_left()), empty^, False)
    assert_true(PropagateEmpty.apply(node).isa[EmptyRelation]())


def test_propagate_empty_keeps_a_union_with_one_empty_side() raises:
    var empty: DynRelation = EmptyRelation(RecordBatch.empty(_right().schema))
    var node: DynRelation = Union(table(_left()), empty^)
    assert_true(PropagateEmpty.apply(node).isa[Union]())


def test_propagate_empty_collapses_a_difference_with_an_empty_left() raises:
    var empty: DynRelation = EmptyRelation(RecordBatch.empty(_left().schema))
    var node: DynRelation = Difference(empty^, table(_right()), False)
    assert_true(PropagateEmpty.apply(node).isa[EmptyRelation]())


def test_column_pruning_keeps_every_column_of_a_set_relation() raises:
    """Pruning `v` would change which rows `EXCEPT` treats as equal."""
    var plan = table(_left()).except_(table(_right())).select(["k"])
    var pruned = ColumnPruning.apply(plan, ["k"])
    assert_true(_sorted_k(pruned) == _sorted_k(plan))
    assert_true(_sorted_k(plan) == array([2], int64))


def test_an_optimized_set_relation_gives_the_same_answer() raises:
    var plan = table(_left()).intersect_all(table(_right()))
    assert_true(
        _sorted_k(plan.optimize[AllRules]()) == array([None, 1, 3, 3], int64)
    )
