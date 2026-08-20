from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT active, qty FROM sales ORDER BY active NULLS FIRST, qty NULLS FIRST

    Sorting on a bit-packed column, with the NULL group first.

    -- expected
    active:bool	qty:int32
    NULL	40
    False	5
    False	20
    True	NULL
    True	10
    True	50
    """
    var t = table("sales")
    var picked = t.select("active", "qty")
    var q = picked.sort([col("active", bool_), col("qty", int32)], [True, True])
    return q
