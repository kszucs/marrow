from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT gcd(n, 6) AS g, lcm(n, 6) AS l FROM floats

    Integer number theory, and two edge answers worth pinning: `gcd(0, 6)` is
    6, not 0, and both functions are defined to be non-negative even for a
    negative argument.

    `marrow/kernels/numeric.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    g:int64	l:int64
    2	12
    3	18
    6	0
    1	6
    2	6
    3	6
    1	6
    NULL	NULL
    """
    var t = table("floats")
    return t.project(
        ["g", "l"],
        [
            gcd(col("n", int64), lit(6, int64)),
            lcm(col("n", int64), lit(6, int64)),
        ],
    )
