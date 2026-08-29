from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, active, CAST(sum(qty) AS BIGINT) AS total FROM sales GROUP BY CUBE (region, active) ORDER BY region NULLS FIRST, active NULLS FIRST

    `CUBE` is every subset of the key list — here four groupings at once. It is
    `ROLLUP`'s superset and the one that multiplies output rows fastest, so it
    is where an implementation that materialises each grouping separately shows
    its cost.

    There is no multi-grouping node: `Aggregate` carries one key list.

    -- skip mojo

    -- expected
    region:string	active:bool	total:int64
    NULL	NULL	40
    NULL	NULL	125
    NULL	NULL	40
    NULL	NULL	40
    NULL	False	25
    NULL	True	60
    'east'	NULL	50
    'east'	True	50
    'north'	NULL	10
    'north'	True	10
    'south'	NULL	25
    'south'	False	25
    """
    var t = table("sales")
    var agg = t.cube(
        keys=[col("region", string), col("active", bool_)],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    return agg.sort_by(
        [col("region", string), col("active", bool_)], [True, True]
    )
