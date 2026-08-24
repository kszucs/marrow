from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT abs(n) AS a FROM floats

    Integer `abs`, including the null row and a zero that has no sign to lose.

    -- expected
    a:int64
    4
    9
    0
    1
    2
    3
    1
    NULL
    """
    var t = table("floats")
    var q = t.project(["a"], [col("n", int64).abs()])
    return q
