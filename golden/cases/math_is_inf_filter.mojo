from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n FROM floats WHERE isinf(x)

    Both infinities pass the predicate; NaN, the finite values and the null do not.

    -- expected
    n:int64
    3
    -1
    """
    var t = table("floats")
    var q = t.filter(col("x", float64).is_inf()).select("n")
    return q
