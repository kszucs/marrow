from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT a, b, g FROM nulls WHERE a > 0

    -- expected
    a:int64	b:int64	g:string
    """
    var t = table("nulls")
    var q = t.filter(col("a", int64) > lit(0, int64))
    return q
