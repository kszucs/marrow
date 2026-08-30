from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT round(price, 1) AS r FROM sales

    `round` with a digit count. marrow's `RoundKernel` takes no argument and
    rounds to an integer, so this is a second kernel rather than a parameter —
    and one whose answer is not exactly representable, which is why `2.25`
    rounding to one place is the interesting row.

    `marrow/kernels/numeric.mojo` has no such kernel.

    -- skip mojo

    -- expected
    r:double
    1.5
    2.3
    0.5
    NULL
    4.0
    -1.3
    """
    var t = table("sales")
    return t.project(["r"], [col("price", float64).round(1)])
