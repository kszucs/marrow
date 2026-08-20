from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT upper(region) AS r, CAST(count(*) AS BIGINT) AS n FROM sales GROUP BY upper(region) ORDER BY r NULLS FIRST

    The group key is an expression, not a column, so it is named
    positionally and then renamed by the projection above it.

    -- expected
    r:string	n:int64
    NULL	1
    'EAST'	1
    'NORTH'	2
    'SOUTH'	2
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[Upper(col("region", string))], aggs=[count_star().alias("n")]
    )
    var named = agg.rename(["key0"], ["r"])
    var q = named.sort([col("r", string)], [True])
    return q
