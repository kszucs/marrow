from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT x, y FROM kleene WHERE (x > 0) AND (y > 0)

    -- expected
    x:int64	y:int64
    1	1
    """
    var t = table("kleene")
    var q = t.filter(
        (col("x", int64) > lit(0, int64)) & (col("y", int64) > lit(0, int64))
    )
    return q
