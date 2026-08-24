from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT y ** 2.0 AS p FROM floats

    Squaring a power of two is exact, so no rounding question arises. `**`
    promotes to double on both sides.

    -- expected
    p:double
    4.0
    16.0
    64.0
    1.0
    1.0
    4.0
    4.0
    NULL
    """
    var t = table("floats")
    var q = t.project(["p"], [col("y", float64) ** lit(2.0, float64)])
    return q
