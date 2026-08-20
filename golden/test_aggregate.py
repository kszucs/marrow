"""Golden cases — aggregation depth, the runtime lane.

Beyond `test_basic`'s one-key sum: computed group keys, computed aggregate
inputs, several keys at once, `count(*)` against `count(col)`, and HAVING
(a filter above the aggregate).
"""

import marrow


def test_golden_agg_count_star_counts_rows(golden):
    """SELECT CAST(count(*) AS BIGINT) AS n FROM basic"""
    t = golden.table("basic")
    golden.check(t.aggregate(by=[], n=marrow.count_star()))


def test_golden_agg_count_column_skips_nulls(golden):
    """SELECT CAST(count(v) AS BIGINT) AS n FROM basic

    The contrast with `count(*)`: `v` has one null, so this is 6 where the
    row count is 7.
    """
    t = golden.table("basic")
    golden.check(t.aggregate(by=[], n=("count", "v")))


def test_golden_agg_over_computed_input(golden):
    """SELECT k, CAST(sum(v * 2) AS BIGINT) AS d
    FROM basic GROUP BY k ORDER BY k NULLS FIRST"""
    t = golden.table("basic")
    plan = t.aggregate(["k"], (t["v"] * 2).sum(alias="d"))
    golden.check(plan.order_by("k"))


def test_golden_agg_two_group_keys(golden):
    """SELECT k, v, CAST(sum(w) AS BIGINT) AS s FROM basic
    GROUP BY k, v ORDER BY k NULLS FIRST, v NULLS FIRST"""
    t = golden.table("basic")
    plan = t.aggregate(by=["k", "v"], s=("sum", "w"))
    golden.check(plan.order_by("k", "v"))


def test_golden_agg_computed_group_key(golden):
    """SELECT (v > 3) AS key0, CAST(sum(w) AS BIGINT) AS s FROM basic
    GROUP BY (v > 3) ORDER BY key0 NULLS FIRST

    A computed key has no source column to name it after, so the plan calls
    it `key0` (`relations.mojo`); the twin matches that rather than the other
    way round.
    """
    t = golden.table("basic")
    plan = t.aggregate(by=[t["v"] > 3], s=("sum", "w"))
    golden.check(plan.order_by("key0"))


def test_golden_agg_having(golden):
    """SELECT k, CAST(sum(v) AS BIGINT) AS total FROM basic
    GROUP BY k HAVING sum(v) > 5 ORDER BY k NULLS FIRST"""
    t = golden.table("basic")
    agg = t.aggregate(by=["k"], total=("sum", "v"))
    golden.check(agg.filter(agg["total"] > 5).order_by("k"))
