from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.qty, r.country FROM sales s JOIN regions r ON s.region = r.region WHERE s.qty >= 10 ORDER BY s.qty DESC NULLS LAST LIMIT 2

    Four operators stacked: the shape a real query has, and the one where
    a morsel boundary is most likely to be mishandled.

    -- expected
    qty:int32	country:string
    20	'mx'
    10	'ca'
    """
    var left = table("sales")
    var right = table("regions")
    var joined = left.join(
        right,
        left_on=[col("region", string)],
        right_on=[col("region", string)],
        how=JOIN_INNER,
        strictness=JOIN_ALL,
    )
    var filtered = joined.filter(col("qty", int32) >= lit(10, int32))
    var picked = filtered.project(
        ["qty", "country"], [col("qty", int32), col("country", string)]
    )
    var q = picked.sort([col("qty", int32)], [False], nulls_first=False).limit(
        2
    )
    return q
