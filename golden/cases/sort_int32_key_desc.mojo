from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales ORDER BY qty DESC NULLS LAST

    -- expected
    qty:int32
    50
    40
    20
    10
    5
    NULL
    """
    var t = table("sales")
    var picked = t.select("qty")
    var q = picked.sort([col("qty", int32)], [False], nulls_first=False)
    return q
