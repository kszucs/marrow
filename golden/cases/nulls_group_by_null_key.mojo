from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT g, CAST(sum(b) AS BIGINT) AS total FROM nulls GROUP BY g ORDER BY g NULLS FIRST

    -- expected
    g:string	total:int64
    NULL	4
    'x'	8
    'y'	8
    """
    var t = table("nulls")
    var agg = t.aggregate(
        keys=[col("g", string)],
        aggs=[col("b", int64).sum().alias("total")],
    )
    var q = agg.sort([col("g", string)], [True])
    return q
