from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales WHERE qty >= 20 ORDER BY qty NULLS FIRST

    -- expected
    qty:int32
    20
    40
    50
    """
    var t = table("sales")
    var picked = t.select("qty")
    var filtered = picked.filter(col("qty", int32) >= lit(20, int32))
    var q = filtered.sort([col("qty", int32)], [True])
    return q
