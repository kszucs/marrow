from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(*) AS BIGINT) AS n FROM (SELECT k, sum(v) AS total FROM basic GROUP BY k) g

    An aggregate whose input is another aggregate. Both are pipeline breakers,
    so this asks whether the outer one drains the inner rather than reading the
    source: the answer is the number of distinct `k` values, the NULL group
    included.

    -- expected
    n:int64
    4
    """
    var t = table("basic")
    var grouped = t.aggregate(
        keys=[col("k", string)],
        aggs=[col("v", int64).sum().alias("total")],
    )
    return grouped.aggregate(aggs=[count_star().alias("n")])
