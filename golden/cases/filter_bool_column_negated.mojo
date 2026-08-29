from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty, active FROM sales WHERE NOT active ORDER BY qty NULLS FIRST

    `NOT active` is NULL where `active` is NULL, and a NULL predicate
    does not select.

    -- expected
    qty:int32	active:bool
    5	False
    20	False
    """
    var t = table("sales")
    var picked = t.select(["qty", "active"])
    var filtered = picked.filter(~col("active", bool_))
    return filtered.sort_by([col("qty", int32)], [True])
