from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region FROM sales WHERE region >= 'north' ORDER BY region NULLS FIRST

    A bytewise ordering comparison, not equality — `east` sorts below
    `north` and drops out.

    -- expected
    region:string
    'north'
    'north'
    'south'
    'south'
    """
    var t = table("sales")
    var picked = t.select("region")
    var filtered = picked.filter(col("region", string) >= lit("north"))
    var q = filtered.sort([col("region", string)], [True])
    return q
