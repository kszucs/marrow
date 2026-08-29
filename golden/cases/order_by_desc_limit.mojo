from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic ORDER BY v DESC NULLS FIRST LIMIT 3

    -- expected
    k:string	v:int64	w:int64
    'b'	NULL	50
    NULL	7	70
    'a'	6	60
    """
    var t = table("basic")
    return t.sort_by([col("v", int64)], [False]).limit(3)
