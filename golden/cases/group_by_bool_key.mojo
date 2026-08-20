from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT active, CAST(sum(qty) AS BIGINT) AS total FROM sales GROUP BY active ORDER BY active NULLS FIRST

    A bit-packed group key, including the NULL group.

    -- expected
    active:bool	total:int64
    NULL	40
    False	25
    True	60
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[col("active", bool_)],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    var q = agg.sort([col("active", bool_)], [True])
    return q
