from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT v - w AS d FROM basic

    -- expected
    d:int64
    -9
    NULL
    -27
    -36
    NULL
    -54
    -63
    """
    var t = table("basic")
    return t.project(["d"], [col("v", int64) - col("w", int64)])
