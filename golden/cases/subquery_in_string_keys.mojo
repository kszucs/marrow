from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, qty FROM sales WHERE region IN (SELECT region FROM regions) ORDER BY region NULLS FIRST, qty NULLS FIRST

    The same shape on a **string** key, where the hash join has to compare
    bytes rather than a register-width integer.

    -- expected
    region:string	qty:int32
    'north'	NULL
    'north'	10
    'south'	5
    'south'	20
    """
    var left = table("sales")
    var right = table("regions")
    var joined = left.join(
        right,
        left_on=[col("region", string)],
        right_on=[col("region", string)],
        how=JOIN_SEMI,
        strictness=JOIN_ALL,
    )
    var picked = joined.select("region", "qty")
    var q = picked.sort(
        [col("region", string), col("qty", int32)], [True, True]
    )
    return q
