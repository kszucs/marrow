from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT skewness(v) AS s, kurtosis(v) AS k FROM basic

    The third and fourth central moments. marrow's Welford accumulator tracks
    count, mean and `M2` and stops there; these need `M3` and `M4`, and the
    engines that offer them disagree about the bias correction and about
    whether `kurtosis` is excess or raw.

    -- skip mojo

    -- expected
    s:double	k:double
    0.30028928507945546	-1.4177693761814532
    """
    var t = table("basic")
    return t.aggregate(
        aggs=[
            col("v", int64).skewness().alias("s"),
            col("v", int64).kurtosis().alias("k"),
        ]
    )
