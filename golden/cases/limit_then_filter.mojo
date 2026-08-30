from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM (SELECT k, v, w FROM basic ORDER BY v NULLS FIRST LIMIT 4) t WHERE v > 2

    A filter *above* a limit, which must see only the four rows the limit kept.
    The composition is the assertion: pushing the predicate below the limit —
    the rewrite an optimiser reaches for first — changes the answer, because a
    different four rows survive.

    -- expected
    k:string	v:int64	w:int64
    'a'	3	30
    """
    var t = table("basic")
    var top = t.sort_by([col("v", int64)], [True]).limit(4)
    return top.filter(col("v", int64) > lit(2, int64))
