from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n % 3 AS m FROM floats

    The interesting row is `n = -1`. SQL truncates toward zero, so `-1 % 3` is
    -1; Python floors, so it would be 2. The divisor is a non-zero literal
    because modulo by zero is a separate question the two engines answer
    differently.

    -- xfail marrow's `%` floors like Python, so `-1 % 3` is 2; SQL truncates toward zero and answers -1.

    -- expected
    m:int64
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
    var q = t.project(["m"], [col("n", int64) % lit(3, int64)])
    return q
