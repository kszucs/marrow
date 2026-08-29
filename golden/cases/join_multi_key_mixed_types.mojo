from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a.qty AS aqty, b.qty AS bqty FROM sales a JOIN sales b ON a.region = b.region AND a.active = b.active ORDER BY aqty NULLS FIRST, bqty NULLS FIRST

    Two keys of different types at once — a string and a bool. Rows whose
    region or active is NULL match nothing, so they drop out entirely.

    -- expected
    aqty:int32	bqty:int32
    NULL	NULL
    NULL	10
    5	5
    5	20
    10	NULL
    10	10
    20	5
    20	20
    50	50
    """
    var left = table("sales")
    var right = table("sales").rename(["qty"], ["bqty"])
    var joined = left.join(right^, [0, 3], [0, 3], JOIN_INNER)
    var picked = joined.project(
        ["aqty", "bqty"], [col("qty", int32), col("bqty", int32)]
    )
    return picked.sort_by(
        [col("aqty", int32), col("bqty", int32)], [True, True]
    )
