from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS INTEGER) AS c FROM nums

    -- expected
    c:int32
    1
    -2
    300
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [col("i", int64).cast(int32, safe=False)])
