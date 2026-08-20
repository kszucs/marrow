from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic WHERE NOT (v > 3)

    -- expected
    k:string	v:int64	w:int64
    'a'	1	10
    'b'	2	NULL
    'a'	3	30
    """
    var t = table("basic")
    var q = t.filter(~(col("v", int64) > lit(3, int64)))
    return q
