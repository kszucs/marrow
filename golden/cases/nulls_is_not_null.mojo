from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import NotNull


def plan() raises -> DynRelation:
    """
    SELECT a, b, g FROM nulls WHERE b IS NOT NULL

    -- expected
    a:int64	b:int64	g:string
    NULL	2	'x'
    NULL	4	NULL
    NULL	6	'x'
    NULL	8	'y'
    """
    var t = table("nulls")
    var q = t.filter(NotNull(col("b", int64)))
    return q
