from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, active, qty FROM sales ORDER BY region NULLS FIRST, active DESC NULLS FIRST, qty NULLS FIRST

    String, bool and int32 keys in one sort, with a direction change in
    the middle.

    -- expected
    region:string	active:bool	qty:int32
    NULL	NULL	40
    'east'	True	50
    'north'	True	NULL
    'north'	True	10
    'south'	False	5
    'south'	False	20
    """
    var t = table("sales")
    var picked = t.select("region", "active", "qty")
    var q = picked.sort(
        [col("region", string), col("active", bool_), col("qty", int32)],
        [True, False, True],
    )
    return q
