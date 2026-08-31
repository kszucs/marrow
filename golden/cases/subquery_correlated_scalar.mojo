from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.region, s.qty, CAST((SELECT count(*) FROM regions r WHERE r.region = s.region) AS BIGINT) AS n FROM sales s ORDER BY s.region NULLS FIRST, s.qty NULLS FIRST

    A subquery that references the outer row. This is the shape that cannot be
    rewritten as a semi- or anti-join — it produces a *value* per outer row,
    not a filter — and the unmatched rows answer 0, which is `count(*)`'s
    answer over an empty relation rather than a null from an outer join.

    -- skip mojo
    -- skip python

    -- expected
    region:string	qty:int32	n:int64
    NULL	40	0
    'east'	50	0
    'north'	NULL	1
    'north'	10	1
    'south'	5	1
    'south'	20	1
    """
    var t = table("sales")
    var counts = table("regions").aggregate(
        keys=[col("region", string)], aggs=[count_star().alias("n")]
    )
    var joined = t.join(counts, [0], [0], JOIN_LEFT)
    return joined.select(["region", "qty", "n"]).sort_by(
        [col("region", string), col("qty", int32)], [True, True]
    )
