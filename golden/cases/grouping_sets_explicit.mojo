from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, active, CAST(sum(qty) AS BIGINT) AS total, CAST(GROUPING(region) AS BIGINT) AS gr, CAST(GROUPING(active) AS BIGINT) AS ga FROM sales GROUP BY GROUPING SETS ((region), (active), ()) ORDER BY gr, ga, region NULLS FIRST, active NULLS FIRST

    The general form `ROLLUP` and `CUBE` are shorthands for: an explicit list
    of groupings, including the empty one, which is the whole-table aggregate.
    Three groupings, and the rows of each are distinguishable only by which key
    columns are NULL.

    The two `GROUPING` flags are what make the answer assertable: three of the
    output rows print as `(NULL, NULL)` — one per grouping — and the flags
    are the only thing that tells them apart.

    There is no multi-grouping node: `Aggregate` carries one key list.

    -- skip mojo

    -- expected
    region:string	active:bool	total:int64	gr:int64	ga:int64
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
    var agg = t.grouping_sets(
        [[col("region", string)], [col("active", bool_)], []],
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
