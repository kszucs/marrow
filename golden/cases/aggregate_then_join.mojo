from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT a.region, CAST(a.total AS BIGINT) AS total, r.country FROM (SELECT region, sum(qty) AS total FROM sales GROUP BY region) a JOIN regions r ON a.region = r.region ORDER BY a.region

    A join whose left input is an aggregate — the derived-table shape.

    -- expected
    region:string	total:int64	country:string
    'north'	10	'ca'
    'south'	25	'mx'
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[col("region", string)],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    var joined = agg.join(table("regions"), [0], [0], JOIN_INNER)
    var picked = joined.project(
        ["region", "total", "country"],
        [col("region", string), col("total", int64), col("country", string)],
    )
    return picked.sort_by([col("region", string)], [True])
