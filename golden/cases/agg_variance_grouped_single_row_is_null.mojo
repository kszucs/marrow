from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT g, var_samp(b) AS v FROM nulls GROUP BY g ORDER BY g NULLS FIRST

    The grouped dispersion accumulator, and the rule that separates the two
    `ddof`s: a group with one value has `n - ddof = 0` at `ddof=1`, which is
    null, not zero. Two of the three groups here are singletons and the third
    (`x = [2, 6]`) has an exact answer of 8.0.

    -- expected
    g:string	v:double
    NULL	NULL
    'x'	8.0
    'y'	NULL
    """
    var t = table("nulls")
    var agg = t.aggregate(
        keys=[col("g", string)],
        aggs=[col("b", int64).variance[1]().alias("v")],
    )
    return agg.sort_by([col("g", string)], [True])
