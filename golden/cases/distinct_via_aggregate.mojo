from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT DISTINCT region FROM sales ORDER BY region NULLS FIRST

    `SELECT DISTINCT` is an aggregate with keys and no aggregates.

    -- expected
    region:string
    NULL
    'east'
    'north'
    'south'
    """
    var t = table("sales")
    var agg = t.aggregate(keys=[col("region", string)], aggs=[])
    var q = agg.sort([col("region", string)], [True])
    return q
