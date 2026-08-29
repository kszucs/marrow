from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(a) AS BIGINT) AS n FROM nulls

    -- expected
    n:int64
    0
    """
    var t = table("nulls")
    return t.aggregate(
        aggs=[col("a", int64).count().alias("n")],
    )
