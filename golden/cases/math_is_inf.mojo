from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT isinf(x) AS b FROM floats

    True for both infinities and, like `is_nan`, NULL rather than false for the null row.

    -- expected
    b:bool
    False
    False
    False
    False
    False
    True
    True
    NULL
    """
    var t = table("floats")
    var q = t.project(["b"], [col("x", float64).is_inf()])
    return q
