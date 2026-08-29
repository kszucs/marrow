from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.region, s.qty, r.country FROM sales s LEFT JOIN regions r ON s.region = r.region ORDER BY s.region NULLS FIRST, s.qty NULLS FIRST

    `east` and the NULL region are unmatched from the left, so both widen
    to a NULL country.

    -- expected
    region:string	qty:int32	country:string
    NULL	40	NULL
    'east'	50	NULL
    'north'	NULL	'ca'
    'north'	10	'ca'
    'south'	5	'mx'
    'south'	20	'mx'
    """
    var left = table("sales")
    var joined = left.join(table("regions"), [0], [0], JOIN_LEFT)
    var picked = joined.project(
        ["region", "qty", "country"],
        [col("region", string), col("qty", int32), col("country", string)],
    )
    return picked.sort_by(
        [col("region", string), col("qty", int32)], [True, True]
    )
