from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT -n AS m FROM floats

    Unary minus over an integer column, including a zero and the null row.

    -- expected
    m:int64
    -4
    9
    0
    -1
    -2
    -3
    1
    NULL
    """
    var t = table("floats")
    var q = t.project(["m"], [-col("n", int64)])
    return q
