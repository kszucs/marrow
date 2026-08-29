from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ln(y) AS l FROM floats WHERE y = 1

    `ln` is **not** correctly rounded under IEEE 754 -- the same hazard as
    `exp` -- so the filter keeps only the inputs whose result is exact: `ln(1)`
    is 0.0 in any conforming implementation.

    -- expected
    l:double
    0.0
    0.0
    """
    var t = table("floats")
    var ones = t.filter(col("y", float64) == lit(1.0, float64))
    return ones.project(["l"], [col("y", float64).ln()])
