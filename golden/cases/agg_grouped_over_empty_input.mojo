from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(count(*) AS BIGINT) AS n FROM basic WHERE v > 1000 GROUP BY k ORDER BY k NULLS FIRST

    The same empty input through a *grouped* aggregate, which must produce no
    rows at all rather than one row of nulls. That is the difference between
    `GROUP BY` and no `GROUP BY` on an empty relation, and the place an engine
    that seeds one implicit group gets it wrong.

    -- expected
    k:string	n:int64
    """
    var t = table("basic")
    var empty = t.filter(col("v", int64) > lit(1000, int64))
    var agg = empty.aggregate(
        keys=[col("k", string)], aggs=[count_star().alias("n")]
    )
    return agg.sort_by([col("k", string)], [True])
