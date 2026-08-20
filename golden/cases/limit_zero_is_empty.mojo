from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales ORDER BY qty NULLS FIRST LIMIT 0

    An empty result is a normal outcome, and it still has to be a
    well-formed batch carrying the right schema.

    -- expected
    qty:int32
    """
    var t = table("sales")
    var picked = t.select("qty")
    var q = picked.sort([col("qty", int32)], [True]).limit(0)
    return q
