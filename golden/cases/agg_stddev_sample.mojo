from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT stddev_samp(b) AS s FROM nulls

    `ddof=1` — the sample standard deviation, the other half of the pair
    `agg_variance_population` covers. The divisor is `n - 1` over the *non-
    null* values, and the root is taken from the same Welford state rather than
    by aggregating twice.

    The value is irrational, so this asserts that both engines take `sqrt` of
    the same double: `M2` is exactly 20.0 (see `agg_variance_population`), 20/3
    is one rounding, and IEEE `sqrt` is correctly rounded.

    -- expected
    s:double
    2.581988897471611
    """
    var t = table("nulls")
    return t.aggregate(aggs=[col("b", int64).stddev[1]().alias("s")])
