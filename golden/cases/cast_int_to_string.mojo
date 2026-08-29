from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS VARCHAR) AS c FROM nums

    -- expected
    c:string
    '1'
    '-2'
    '300'
    NULL
    """
    var t = table("nums")
    return t.project(["c"], [NumToString[StringType](col("i", int64))])
