"""Golden cases — the runtime lane.

Each case is one query written twice: as SQL in the docstring, which is what
DuckDB runs to produce the expectation, and as a marrow plan in the body. The
`golden` fixture loads the fixture table, runs the plan and compares against
`test_basic.exp`.

`golden/test_basic.mojo` holds the same cases under the same names for the AOT
lane. The two name sets must match — that is invariant 2 ("one engine, two
drivers") checked at query level.

Every ORDER BY spells its null placement explicitly. marrow defaults to
`nulls_first=True` and DuckDB defaults to NULLS LAST, so a case that leaves it
implicit compares two different queries.
"""


def test_golden_select_two_columns(golden):
    """SELECT k, v FROM basic"""
    t = golden.table("basic")
    golden.check(t.select("k", "v"))


def test_golden_filter_gt(golden):
    """SELECT k, v, w FROM basic WHERE v > 3"""
    t = golden.table("basic")
    golden.check(t.filter(t["v"] > 3))


def test_golden_filter_and(golden):
    """SELECT k, v, w FROM basic WHERE v > 2 AND w < 60"""
    t = golden.table("basic")
    golden.check(t.filter((t["v"] > 2) & (t["w"] < 60)))


def test_golden_project_sum_of_columns(golden):
    """SELECT v + w AS s FROM basic"""
    t = golden.table("basic")
    golden.check(t.project(s=t["v"] + t["w"]))


def test_golden_with_columns_appends(golden):
    """SELECT k, v, w, v + w AS s FROM basic"""
    t = golden.table("basic")
    golden.check(t.with_columns(s=t["v"] + t["w"]))


def test_golden_aggregate_sum_by_key(golden):
    """SELECT k, CAST(sum(v) AS BIGINT) AS total
    FROM basic GROUP BY k ORDER BY k NULLS FIRST"""
    t = golden.table("basic")
    golden.check(t.aggregate(by=["k"], total=("sum", "v")).order_by("k"))


def test_golden_aggregate_several(golden):
    """SELECT k, CAST(sum(v) AS BIGINT) AS total, max(v) AS biggest,
           CAST(count(w) AS BIGINT) AS n
    FROM basic GROUP BY k ORDER BY k NULLS FIRST"""
    t = golden.table("basic")
    plan = t.aggregate(
        by=["k"], total=("sum", "v"), biggest=("max", "v"), n=("count", "w")
    )
    golden.check(plan.order_by("k"))


def test_golden_aggregate_no_keys(golden):
    """SELECT CAST(sum(v) AS BIGINT) AS total FROM basic"""
    t = golden.table("basic")
    golden.check(t.aggregate(by=[], total=("sum", "v")))


def test_golden_order_by_asc(golden):
    """SELECT k, v, w FROM basic ORDER BY v NULLS FIRST"""
    t = golden.table("basic")
    golden.check(t.order_by("v"))


def test_golden_order_by_desc_limit(golden):
    """SELECT k, v, w FROM basic ORDER BY v DESC NULLS FIRST LIMIT 3"""
    t = golden.table("basic")
    golden.check(t.order_by(("v", "descending")).limit(3))


def test_golden_filter_then_aggregate(golden):
    """SELECT k, CAST(sum(v) AS BIGINT) AS total FROM basic WHERE w > 20
    GROUP BY k ORDER BY k NULLS FIRST"""
    t = golden.table("basic")
    plan = t.filter(t["w"] > 20).aggregate(by=["k"], total=("sum", "v"))
    golden.check(plan.order_by("k"))


def test_golden_filter_lt(golden):
    """SELECT k, v, w FROM basic WHERE v < 4"""
    t = golden.table("basic")
    golden.check(t.filter(t["v"] < 4))


def test_golden_filter_or(golden):
    """SELECT k, v, w FROM basic WHERE v < 2 OR w > 60"""
    t = golden.table("basic")
    golden.check(t.filter((t["v"] < 2) | (t["w"] > 60)))


def test_golden_filter_not(golden):
    """SELECT k, v, w FROM basic WHERE NOT (v > 3)"""
    t = golden.table("basic")
    golden.check(t.filter(~(t["v"] > 3)))


def test_golden_project_difference(golden):
    """SELECT v - w AS d FROM basic"""
    t = golden.table("basic")
    golden.check(t.project(d=t["v"] - t["w"]))


def test_golden_project_predicate_is_boolean(golden):
    """SELECT v > 3 AS gt FROM basic"""
    t = golden.table("basic")
    golden.check(t.project(gt=t["v"] > 3))


def test_golden_order_by_two_keys(golden):
    """SELECT k, v, w FROM basic ORDER BY k NULLS FIRST, v NULLS FIRST"""
    t = golden.table("basic")
    golden.check(t.order_by("k", "v"))


def test_golden_order_by_nulls_last(golden):
    """SELECT k, v, w FROM basic ORDER BY v NULLS LAST"""
    t = golden.table("basic")
    golden.check(t.order_by("v", nulls_first=False))


def test_golden_limit_with_offset(golden):
    """SELECT k, v, w FROM basic ORDER BY v NULLS FIRST LIMIT 3 OFFSET 2"""
    t = golden.table("basic")
    golden.check(t.order_by("v").limit(3, 2))


def test_golden_aggregate_min_max(golden):
    """SELECT k, min(v) AS lo, max(w) AS hi
    FROM basic GROUP BY k ORDER BY k NULLS FIRST"""
    t = golden.table("basic")
    plan = t.aggregate(by=["k"], lo=("min", "v"), hi=("max", "w"))
    golden.check(plan.order_by("k"))
