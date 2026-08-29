from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(region) AS lo, max(region) AS hi FROM sales

    `min`/`max` over strings is a bytewise comparison, a different kernel
    from the numeric fold.

    -- expected
    lo:string	hi:string
    'east'	'south'
    """
    var t = table("sales")
    return t.aggregate(
        aggs=[
            col("region", string).min().alias("lo"),
            col("region", string).max().alias("hi"),
        ]
    )
