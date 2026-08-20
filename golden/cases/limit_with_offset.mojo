from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


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
    var q = t.sort([col("v", int64)], [True]).limit(3, 2)
    return q
