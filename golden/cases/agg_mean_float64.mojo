from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT mean(price) AS m FROM sales

    -- expected
    m:double
    1.4
    """
    var t = table("sales")
    var q = t.aggregate(aggs=[col("price", float64).mean().alias("m")])
    return q
