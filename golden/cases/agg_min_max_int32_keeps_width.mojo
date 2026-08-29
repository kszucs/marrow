from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(qty) AS lo, max(qty) AS hi FROM sales

    `sum` widens int32 to int64 (`agg_sum_int32`); `min` and `max` must not.
    The output dtype is the assertion — the values would compare equal either
    way.

    -- expected
    lo:int32	hi:int32
    5	50
    """
    var t = table("sales")
    return t.aggregate(
        aggs=[
            col("qty", int32).min().alias("lo"),
            col("qty", int32).max().alias("hi"),
        ]
    )
