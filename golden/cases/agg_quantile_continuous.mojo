from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT quantile_cont(v, 0.25) AS q FROM basic

    The generalisation of the median, and the one where engines genuinely
    disagree: `quantile_cont` interpolates between the two neighbouring values,
    `quantile_disc` returns one of them. A corpus that said only "quantile"
    would not pin down which.

    `marrow/kernels/aggregate.mojo` has no order-statistic kernel.

    -- skip mojo
    -- skip python

    -- expected
    q:double
    2.25
    """
    var t = table("basic")
    return t.aggregate(aggs=[col("v", int64).quantile(0.25).alias("q")])
