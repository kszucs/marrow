from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(DISTINCT region) AS BIGINT) AS n FROM sales

    `count(DISTINCT ...)` skips nulls, like `count(col)`.

    -- expected
    n:int64
    3
    """
    var t = table("sales")
    return t.aggregate(aggs=[col("region", string).count_distinct().alias("n")])
