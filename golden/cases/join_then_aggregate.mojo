from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT r.country, CAST(sum(s.qty) AS BIGINT) AS total FROM sales s JOIN regions r ON s.region = r.region GROUP BY r.country ORDER BY r.country NULLS FIRST

    An aggregate above a join, grouping on a column that came from the
    right side.

    -- expected
    country:string	total:int64
    'ca'	10
    'mx'	25
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
    var agg = joined.aggregate(
        keys=[col("country", string)],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    var q = agg.sort([col("country", string)], [True])
    return q
