from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic ORDER BY v NULLS FIRST LIMIT 3 OFFSET 2

    -- expected
    k:string	v:int64	w:int64
    'b'	2	NULL
    'a'	3	30
    'c'	4	40
    """
    var t = table("basic")
    return t.sort_by([col("v", int64)], [True]).limit(3, 2)
