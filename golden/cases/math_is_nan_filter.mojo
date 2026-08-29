from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n FROM floats WHERE isnan(x)

    `is_nan` as a predicate rather than a projection. `n` is selected instead
    of `x` so the result stays representable in the expectation block.

    -- expected
    n:int64
    2
    """
    var t = table("floats")
    return t.filter(col("x", float64).is_nan()).select(["n"])
