from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, active, CAST(count(*) AS BIGINT) AS n FROM sales GROUP BY region, active ORDER BY region NULLS FIRST, active NULLS FIRST

    Two keys of different widths, one of them bit-packed.

    -- expected
    region:string	active:bool	n:int64
    NULL	NULL	1
    'east'	True	1
    'north'	True	2
    'south'	False	2
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[col("region", string), col("active", bool_)],
        aggs=[count_star().alias("n")],
    )
    return agg.sort_by(
        [col("region", string), col("active", bool_)], [True, True]
    )
