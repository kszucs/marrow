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
    var joined = left.join(table("regions"), [0], [0], JOIN_INNER)
    var filtered = joined.filter(col("qty", int32) >= lit(10, int32))
    var picked = filtered.project(
        ["qty", "country"], [col("qty", int32), col("country", string)]
    )
    return picked.sort_by(
        [col("qty", int32)], [False], nulls_first=False
    ).limit(2)
