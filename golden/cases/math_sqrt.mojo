from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT sqrt(y) AS r FROM floats

    `sqrt` is correctly rounded under IEEE 754, so any input is fair game and
    the irrational results compare bit for bit. `y` is positive throughout, so
    nothing here produces a NaN.

    -- expected
    r:double
    1.4142135623730951
    2.0
    2.8284271247461903
    1.0
    1.0
    1.4142135623730951
    1.4142135623730951
    NULL
    """
    var t = table("floats")
    return t.project(["r"], [col("y", float64).sqrt()])
