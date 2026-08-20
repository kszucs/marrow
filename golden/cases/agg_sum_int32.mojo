from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(qty) AS BIGINT) AS total FROM sales

    `sum` widens int32 to int64, so the output dtype is not the input's.

    -- expected
    total:int64
    125
    """
    var t = table("sales")
    var q = t.aggregate(aggs=[col("qty", int32).sum().alias("total")])
    return q
