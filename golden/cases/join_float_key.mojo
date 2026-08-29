from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a.qty AS aqty, b.qty AS bqty FROM sales a JOIN sales b ON a.price = b.price ORDER BY aqty NULLS FIRST, bqty NULLS FIRST

    Equality on a float key. Every price is an exact binary fraction, so
    this asks about hashing and null handling rather than about rounding.

    -- expected
    aqty:int32	bqty:int32
    NULL	NULL
    5	5
    10	10
    20	20
    50	50
    """
    var left = table("sales")
    var right = table("sales").rename(["qty"], ["bqty"])
    var joined = left.join(right^, [2], [2], JOIN_INNER)
    var picked = joined.project(
        ["aqty", "bqty"], [col("qty", int32), col("bqty", int32)]
    )
    return picked.sort_by(
        [col("aqty", int32), col("bqty", int32)], [True, True]
    )
