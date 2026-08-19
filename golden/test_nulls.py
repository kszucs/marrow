"""Golden cases — null semantics, the runtime lane.

Where engines disagree most, and where Arrow's three-valued logic differs from
a naive reading: a predicate that evaluates to NULL does not select the row,
`sum` over no valid input is NULL rather than 0, `count` over the same is 0
rather than NULL, and any arithmetic touching a NULL yields NULL.

Runs against the `nulls` fixture: `a` is entirely null, `b` has none, `g` is a
string key carrying one null.
"""


def test_golden_nulls_is_null(golden):
    """SELECT a, b, g FROM nulls WHERE a IS NULL"""
    t = golden.table("nulls")
    golden.check(t.filter(t["a"].is_null()))


def test_golden_nulls_is_not_null(golden):
    """SELECT a, b, g FROM nulls WHERE b IS NOT NULL"""
    t = golden.table("nulls")
    golden.check(t.filter(t["b"].is_valid()))


def test_golden_nulls_predicate_excludes_null(golden):
    """SELECT a, b, g FROM nulls WHERE a > 0"""
    t = golden.table("nulls")
    golden.check(t.filter(t["a"] > 0))


def test_golden_nulls_arithmetic_propagates(golden):
    """SELECT a + b AS s FROM nulls"""
    t = golden.table("nulls")
    golden.check(t.project(s=t["a"] + t["b"]))


def test_golden_nulls_sum_of_all_null_is_null(golden):
    """SELECT CAST(sum(a) AS BIGINT) AS total FROM nulls"""
    t = golden.table("nulls")
    golden.check(t.aggregate(by=[], total=("sum", "a")))


def test_golden_nulls_count_of_all_null_is_zero(golden):
    """SELECT CAST(count(a) AS BIGINT) AS n FROM nulls"""
    t = golden.table("nulls")
    golden.check(t.aggregate(by=[], n=("count", "a")))


def test_golden_nulls_mean_ignores_nulls(golden):
    """SELECT avg(b) AS m FROM nulls"""
    t = golden.table("nulls")
    golden.check(t.aggregate(by=[], m=("mean", "b")))


def test_golden_nulls_group_by_null_key(golden):
    """SELECT g, CAST(sum(b) AS BIGINT) AS total
    FROM nulls GROUP BY g ORDER BY g NULLS FIRST"""
    t = golden.table("nulls")
    golden.check(t.aggregate(by=["g"], total=("sum", "b")).order_by("g"))
