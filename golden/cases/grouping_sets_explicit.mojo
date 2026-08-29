from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, active, CAST(sum(qty) AS BIGINT) AS total FROM sales GROUP BY GROUPING SETS ((region), (active), ()) ORDER BY region NULLS FIRST, active NULLS FIRST

    The general form `ROLLUP` and `CUBE` are shorthands for: an explicit list
    of groupings, including the empty one, which is the whole-table aggregate.
    Three groupings, and the rows of each are distinguishable only by which key
    columns are NULL.

    There is no multi-grouping node: `Aggregate` carries one key list.

    -- skip mojo

    -- expected
    region:string	active:bool	total:int64
    NULL	NULL	125
    NULL	NULL	40
    NULL	NULL	40
    NULL	False	25
    NULL	True	60
    'east'	NULL	50
    'north'	NULL	10
    'south'	NULL	25
    """
    var t = table("sales")
    var agg = t.grouping_sets(
        [[col("region", string)], [col("active", bool_)], []],
        aggs=[col("qty", int32).sum().alias("total")],
    )
    return agg.sort_by(
        [col("region", string), col("active", bool_)], [True, True]
    )
