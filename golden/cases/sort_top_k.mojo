from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic ORDER BY v DESC NULLS LAST LIMIT 2

    -- expected
    k:string	v:int64	w:int64
    NULL	7	70
    'a'	6	60
    """
    var t = table("basic")
    return t.sort_by([col("v", int64)], [False], nulls_first=False).limit(2)
