from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT isnan(x) AS b FROM floats

    A null is not a NaN: the null row answers NULL, not false. That is the
    whole difference between `is_nan` and `is_null`.

    -- expected
    b:bool
    False
    False
    False
    False
    True
    False
    False
    NULL
    """
    var t = table("floats")
    return t.project(["b"], [col("x", float64).is_nan()])
