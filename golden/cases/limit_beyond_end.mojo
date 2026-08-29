from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales ORDER BY qty NULLS FIRST LIMIT 100 OFFSET 4

    An offset near the end and a limit past it: fewer rows come back than
    were asked for, which is not an error.

    -- expected
    qty:int32
    40
    50
    """
    var t = table("sales")
    var picked = t.select(["qty"])
    return picked.sort_by([col("qty", int32)], [True]).limit(100, 4)
