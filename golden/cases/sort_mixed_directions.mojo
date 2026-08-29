from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic ORDER BY k ASC NULLS FIRST, v DESC NULLS FIRST

    -- expected
    k:string	v:int64	w:int64
    NULL	7	70
    'a'	6	60
    'a'	3	30
    'a'	1	10
    'b'	NULL	50
    'b'	2	NULL
    'c'	4	40
    """
    var t = table("basic")
    return t.sort_by([col("k", string), col("v", int64)], [True, False])
