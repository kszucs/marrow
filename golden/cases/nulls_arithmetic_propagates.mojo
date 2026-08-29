from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a + b AS s FROM nulls

    -- expected
    s:int64
    NULL
    NULL
    NULL
    NULL
    """
    var t = table("nulls")
    return t.project(["s"], [col("a", int64) + col("b", int64)])
