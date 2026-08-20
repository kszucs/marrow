from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a.qty AS aqty, b.qty AS bqty FROM sales a JOIN sales b ON a.ref = b.ref ORDER BY aqty NULLS FIRST, bqty NULLS FIRST

    A self-join. `ref` 2 appears twice, so the match is many-to-many and
    the NULL ref matches nothing — not even the other NULL.

    -- expected
    aqty:int32	bqty:int32
    NULL	NULL
    NULL	20
    5	5
    10	10
    20	NULL
    20	20
    40	40
    """
    var left = table("sales")
    var right = table("sales").rename(["qty"], ["bqty"])
    var joined = left.join(
        right,
        left_on=[col("ref", int64)],
        right_on=[col("ref", int64)],
        how=JOIN_INNER,
        strictness=JOIN_ALL,
    )
    var picked = joined.project(
        ["aqty", "bqty"], [col("qty", int32), col("bqty", int32)]
    )
    var q = picked.sort([col("aqty", int32), col("bqty", int32)], [True, True])
    return q
