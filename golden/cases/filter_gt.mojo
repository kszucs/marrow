from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic WHERE v > 3

    -- expected
    k:string	v:int64	w:int64
    'c'	4	40
    'a'	6	60
    NULL	7	70
    """
    var t = table("basic")
    var q = t.filter(col("v", int64) > lit(3, int64))
    return q
