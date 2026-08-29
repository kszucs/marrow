from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region, qty FROM sales WHERE region = 'north' ORDER BY qty NULLS FIRST

    -- expected
    region:string	qty:int32
    'north'	NULL
    'north'	10
    """
    var t = table("sales")
    var picked = t.select(["region", "qty"])
    var filtered = picked.filter(col("region", string) == lit("north", string))
    return filtered.sort_by([col("qty", int32)], [True])
