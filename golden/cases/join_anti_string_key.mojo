from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.region, s.qty, s.price, s.active, s.ref FROM sales s WHERE NOT EXISTS (SELECT 1 FROM regions r WHERE s.region = r.region) ORDER BY s.region NULLS FIRST, s.qty NULLS FIRST

    Anti-join on a *string* key, where `join_anti` uses int64. The NULL region
    is the interesting row: a NULL key matches nothing, so it survives an anti-
    join — the opposite of the semi-join case, where it is dropped.

    -- expected
    region:string	qty:int32	price:double	active:bool	ref:int64
    NULL	40	NULL	NULL	3
    'east'	50	4.0	True	NULL
    """
    var joined = table("sales").join(table("regions"), [0], [0], JOIN_ANTI)
    return joined.sort_by(
        [col("region", string), col("qty", int32)], [True, True]
    )
