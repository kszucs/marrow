from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, active, CAST(sum(qty) AS BIGINT) AS total, CAST(GROUPING(region) AS BIGINT) AS gr, CAST(GROUPING(active) AS BIGINT) AS ga FROM sales GROUP BY CUBE (region, active) ORDER BY gr, ga, region NULLS FIRST, active NULLS FIRST

    `CUBE` is every subset of the key list — here four groupings at once. It is
    `ROLLUP`'s superset and the one that multiplies output rows fastest, so it
    is where an implementation that materialises each grouping separately shows
    its cost.

    `GROUPING(region)` and `GROUPING(active)` are not decoration: four of the
    output rows print as `(NULL, NULL)` — the grand total, the NULL-region
    group, the NULL-active group, and the `(NULL, NULL)` pair — and nothing
    else distinguishes them, so without the two flags the case would have no
    total order and no assertable answer.

    There is no multi-grouping node: `Aggregate` carries one key list.

    -- skip mojo
    -- skip python

    -- expected
    region:string	active:bool	total:int64	gr:int64	ga:int64
    NULL	NULL	40	0	0
    'east'	True	50	0	0
    'north'	True	10	0	0
    'south'	False	25	0	0
    NULL	NULL	40	0	1
    'east'	NULL	50	0	1
    'north'	NULL	10	0	1
    'south'	NULL	25	0	1
    NULL	NULL	40	1	0
    NULL	False	25	1	0
    NULL	True	60	1	0
    NULL	NULL	125	1	1
    """
    var t = table("sales")
    var agg = t.cube(
        keys=[col("region", string), col("active", bool_)],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    return agg.sort_by(
        [
            col("gr", int64),
            col("ga", int64),
            col("region", string),
            col("active", bool_),
        ],
        [True, True, True, True],
    )
