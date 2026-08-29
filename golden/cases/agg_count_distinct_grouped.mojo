from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(count(DISTINCT w) AS BIGINT) AS n FROM basic GROUP BY k ORDER BY k NULLS FIRST

    The *grouped* distinct count, which keeps one hash set per slot rather than
    one for the whole column — a different code path from the ungrouped form,
    and the one that gets a group's set confused with its neighbour's. Group
    `b` holds `{NULL, 50}` and must answer 1.

    -- expected
    k:string	n:int64
    NULL	1
    'a'	3
    'b'	1
    'c'	1
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[col("w", int64).count_distinct().alias("n")],
    )
    return agg.sort_by([col("k", string)], [True])
