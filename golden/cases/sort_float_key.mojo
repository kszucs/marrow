from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty, price FROM sales ORDER BY price NULLS FIRST

    -- expected
    qty:int32	price:double
    40	NULL
    5	-1.25
    NULL	0.5
    10	1.5
    20	2.25
    50	4.0
    """
    var t = table("sales")
    var picked = t.select("qty", "price")
    var q = picked.sort([col("price", float64)], [True])
    return q
