from golden.helpers import table
from marrow.dtypes import string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic ORDER BY k DESC NULLS LAST

    -- expected
    k:string	v:int64	w:int64
    'c'	4	40
    'b'	2	NULL
    'b'	NULL	50
    'a'	1	10
    'a'	3	30
    'a'	6	60
    NULL	7	70
    """
    var t = table("basic")
    var q = t.sort([col("k", string)], [False], nulls_first=False)
    return q
