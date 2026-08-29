from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT trunc(f) AS t FROM nums

    Truncation toward zero, which `math_floor` and `math_ceil` bracket: `-2.7`
    truncates to `-2.0` where it floors to `-3.0`. `cast_float_to_int` reaches
    the same kernel through a cast; this asks about the rounding rule rather
    than the type change.

    -- expected
    t:double
    1.0
    -2.0
    0.0
    NULL
    """
    var t = table("nums")
    return t.project(["t"], [col("f", float64).trunc()])
