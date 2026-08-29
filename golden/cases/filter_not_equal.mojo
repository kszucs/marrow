from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales WHERE qty != 20 ORDER BY qty NULLS FIRST

    `!=` against a column with a null: the null row is not selected,
    because NULL != 20 is UNKNOWN rather than true.

    -- expected
    qty:int32
    5
    10
    40
    50
    """
    var t = table("sales")
    var picked = t.select(["qty"])
    var filtered = picked.filter(col("qty", int32) != lit(20, int32))
    return filtered.sort_by([col("qty", int32)], [True])
