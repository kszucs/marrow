from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty, price FROM sales WHERE price < 2.0 ORDER BY price NULLS FIRST

    -- expected
    qty:int32	price:double
    5	-1.25
    NULL	0.5
    10	1.5
    """
    var t = table("sales")
    var picked = t.select("qty", "price")
    var filtered = picked.filter(col("price", float64) < lit(2.0, float64))
    var q = filtered.sort([col("price", float64)], [True])
    return q
