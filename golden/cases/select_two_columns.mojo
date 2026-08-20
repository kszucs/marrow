from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v FROM basic

    -- expected
    k:string	v:int64
    'a'	1
    'b'	2
    'a'	3
    'c'	4
    'b'	NULL
    'a'	6
    NULL	7
    """
    var t = table("basic")
    var q = t.select("k", "v")
    return q
