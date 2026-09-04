from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n % 3 AS r FROM floats

    The companion to `math_floordiv_truncates_toward_zero`, and the same rule:
    `%` and `//` are one convention. Python's remainder has the sign of the
    *divisor*, C's and SQL's the sign of the *dividend*, so `-1 % 3` is 2 under
    one and -1 under the other. `ModKernel` corrects Mojo's operator to SQL's,
    and `a == (a // b) * b + a % b` holds under both.

    -- expected
    r:int64
    1
    0
    0
    1
    2
    0
    -1
    NULL
    """
    var t = table("floats")
    return t.project(["r"], [col("n", int64) % lit(3, int64)])
