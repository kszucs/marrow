from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total FROM basic GROUP BY k HAVING sum(v) > 5 ORDER BY k NULLS FIRST

    -- expected
    k:string	total:int64
    NULL	7
    'a'	10
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[col("v", int64).sum().alias("total")],
    )
    return agg.filter(col("total", int64) > lit(5, int64)).sort_by(
        [col("k", string)], [True]
    )
