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
    var joined = left.join(table("regions"), [0], [0], JOIN_INNER)
    var agg = joined.aggregate(
        keys=[col("country", string)],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    return agg.sort_by([col("country", string)], [True])
