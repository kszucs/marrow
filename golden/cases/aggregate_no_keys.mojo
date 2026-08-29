from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(v) AS BIGINT) AS total FROM basic

    -- expected
    total:int64
    23
    """
    var t = table("basic")
    return t.aggregate(
        aggs=[col("v", int64).sum().alias("total")],
    )
