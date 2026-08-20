from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, sum(price) AS total FROM sales GROUP BY region ORDER BY region NULLS FIRST

    -- expected
    region:string	total:double
    NULL	NULL
    'east'	4.0
    'north'	2.0
    'south'	1.0
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[col("region", string)],
        aggs=[col("price", float64).sum().alias("total")],
    )
    var q = agg.sort([col("region", string)], [True])
    return q
