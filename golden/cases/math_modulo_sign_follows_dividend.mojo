from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT n % 3 AS r FROM floats

    The companion to `math_floordiv_truncates_toward_zero`, and the same root
    cause: `%` and `//` are one rule. Python's remainder has the sign of the
    *divisor*, C's and SQL's the sign of the *dividend*, so `-1 % 3` is 2 under
    one and -1 under the other.

    `math_mod_int64` asks the same kernel the question in the form that agrees
    — `((n % 3) + 3) % 3` is the non-negative residue under either rule — so
    the corpus states both the working case and the divergence.

    -- xfail marrow's % takes the sign of the divisor (Mojo/Python), SQL's takes the sign of the dividend: -1 % 3 is -1 in SQL and 2 here

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
