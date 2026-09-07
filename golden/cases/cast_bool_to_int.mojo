from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(b AS BIGINT) AS c FROM nums

    -- expected
    c:int64
    1
    0
    1
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [col("b", bool_).cast(int64, safe=False)])
