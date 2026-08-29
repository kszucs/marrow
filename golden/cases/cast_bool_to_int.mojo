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
    return t.project(["c"], [BoolToNum[Int64Type](col("b", bool_))])
