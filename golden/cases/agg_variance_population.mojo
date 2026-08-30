from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT var_pop(b) AS v FROM nulls

    `ddof=0` is the default, matching Arrow's `VarianceOptions`, so the plain
    `.variance()` is the *population* variance and the twin has to say
    `var_pop` rather than `variance` — DuckDB's unqualified `variance` is the
    sample one.

    `b` is `[2, 4, 6, 8]`: the running mean is exact at every step (2, 3, 4,
    5), so Welford's `M2` is exactly 20.0 and the answer is exactly 5.0
    whichever algorithm either engine uses. A column whose intermediate means
    are not representable would compare two roundings instead.

    -- expected
    v:double
    5.0
    """
    var t = table("nulls")
    return t.aggregate(aggs=[col("b", int64).variance().alias("v")])
