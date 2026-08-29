from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS BOOLEAN) AS c FROM nums

    -- expected
    c:bool
    True
    True
    True
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [NumToBool(col("i", int64))])
