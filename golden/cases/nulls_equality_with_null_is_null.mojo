from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT v, (v = v) AS e FROM basic

    A column compared with itself. Every valid row is true and the null row is
    **null**, not true — three-valued equality, which is what separates `=`
    from `IS NOT DISTINCT FROM` and is why a NULL join key matches nothing, not
    even another NULL.

    -- expected
    v:int64	e:bool
    1	True
    2	True
    3	True
    4	True
    NULL	NULL
    6	True
    7	True
    """
    var t = table("basic")
    return t.project(
        ["v", "e"], [col("v", int64), col("v", int64) == col("v", int64)]
    )
