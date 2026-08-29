from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT coalesce(v, w) AS c FROM basic

    -- expected
    c:int64
    1
    2
    3
    4
    50
    6
    7
    """
    var t = table("basic")
    return t.project(["c"], [Coalesce(col("v", int64), col("w", int64))])
