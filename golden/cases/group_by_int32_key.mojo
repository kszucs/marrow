from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty, CAST(count(*) AS BIGINT) AS n FROM sales GROUP BY qty ORDER BY qty NULLS FIRST

    A group key narrower than the register the grouper hashes into.

    -- expected
    qty:int32	n:int64
    NULL	1
    5	1
    10	1
    20	1
    40	1
    50	1
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[col("qty", int32)], aggs=[count_star().alias("n")]
    )
    var q = agg.sort([col("qty", int32)], [True])
    return q
