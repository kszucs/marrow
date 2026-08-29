from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a.qty AS aqty, b.qty AS bqty FROM sales a JOIN sales b ON a.active = b.active ORDER BY aqty NULLS FIRST, bqty NULLS FIRST

    A bit-packed join key. The NULL matches nothing.

    -- expected
    aqty:int32	bqty:int32
    NULL	NULL
    NULL	10
    NULL	50
    5	5
    5	20
    10	NULL
    10	10
    10	50
    20	5
    20	20
    50	NULL
    50	10
    50	50
    """
    var left = table("sales")
    var right = table("sales").rename(["qty"], ["bqty"])
    var joined = left.join(right^, [3], [3], JOIN_INNER)
    var picked = joined.project(
        ["aqty", "bqty"], [col("qty", int32), col("bqty", int32)]
    )
    return picked.sort_by(
        [col("aqty", int32), col("bqty", int32)], [True, True]
    )
