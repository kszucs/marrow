from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, CAST(sum(w) AS BIGINT) AS s FROM basic GROUP BY k, v ORDER BY k NULLS FIRST, v NULLS FIRST

    -- expected
    k:string	v:int64	s:int64
    NULL	7	70
    'a'	1	10
    'a'	3	30
    'a'	6	60
    'b'	NULL	50
    'b'	2	NULL
    'c'	4	40
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string), col("v", int64)],
        aggs=[col("w", int64).sum().alias("s")],
    )
    return agg.sort_by([col("k", string), col("v", int64)], [True, True])
