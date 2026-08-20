from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT x, NOT (x > 0) AS r FROM kleene

    -- expected
    x:int64	r:bool
    1	False
    1	False
    1	False
    -1	True
    -1	True
    -1	True
    NULL	NULL
    NULL	NULL
    NULL	NULL
    """
    var t = table("kleene")
    var q = t.project(
        ["x", "r"], [col("x", int64), ~(col("x", int64) > lit(0, int64))]
    )
    return q
