from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT greatest(v, w) AS g, least(v, w) AS l FROM basic

    The row-wise extrema, and the family's whole difficulty is nulls: SQL's
    `greatest`/`least` **skip** them, so a row with one null operand answers
    with the other one rather than null — the opposite of every arithmetic
    operator.

    `MinKernel` and `MaxKernel` in `marrow/kernels/numeric.mojo` are the binary
    kernels this needs, but no expression node exposes them.

    -- skip mojo
    -- skip python

    -- expected
    g:int64	l:int64
    10	1
    2	2
    30	3
    40	4
    50	50
    60	6
    70	7
    """
    var t = table("basic")
    return t.project(
        ["g", "l"],
        [
            greatest(col("v", int64), col("w", int64)),
            least(col("v", int64), col("w", int64)),
        ],
    )
