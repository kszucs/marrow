from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT round_even(f, 0) AS r FROM nums

    Banker's rounding. `0.5` is the discriminating input: half-away-from-zero
    gives `1.0` and half-to-even gives `0.0`. `math_round` notes that the
    corpus does not currently pin marrow's halfway rule down; this records the
    other rule as a function marrow does not have at all.

    `marrow/kernels/numeric.mojo` has no such kernel.

    -- skip mojo

    -- expected
    r:double
    2.0
    -3.0
    0.0
    NULL
    """
    var t = table("nums")
    return t.project(["r"], [col("f", float64).round_even(0)])
