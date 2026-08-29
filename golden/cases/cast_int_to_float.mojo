from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS DOUBLE) AS c FROM nums

    -- expected
    c:double
    1.0
    -2.0
    300.0
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [NumericCast[Float64Type](col("i", int64))])
