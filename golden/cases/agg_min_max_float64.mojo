from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(price) AS lo, max(price) AS hi FROM sales

    `min`/`max` preserve the input dtype rather than widening it.

    -- expected
    lo:double	hi:double
    -1.25	4.0
    """
    var t = table("sales")
    var q = t.aggregate(
        aggs=[
            col("price", float64).min().alias("lo"),
            col("price", float64).max().alias("hi"),
        ]
    )
    return q
