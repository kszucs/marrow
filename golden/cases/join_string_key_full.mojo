from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.qty, r.region AS rregion FROM sales s FULL JOIN regions r ON s.region = r.region ORDER BY rregion NULLS FIRST, s.qty NULLS FIRST

    `west` is unmatched from the right, so a full join has to widen in
    both directions on a string key.

    -- expected
    qty:int32	rregion:string
    40	NULL
    50	NULL
    NULL	'north'
    10	'north'
    5	'south'
    20	'south'
    NULL	'west'
    """
    var left = table("sales")
    var right = table("regions").rename(["region"], ["rregion"])
    var joined = left.join(right^, [0], [0], JOIN_FULL)
    var picked = joined.project(
        ["qty", "rregion"], [col("qty", int32), col("rregion", string)]
    )
    return picked.sort_by(
        [col("rregion", string), col("qty", int32)], [True, True]
    )
