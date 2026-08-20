from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation
from marrow.expr.values import FillNull


def plan() raises -> DynRelation:
    """
    SELECT coalesce(v, 0) AS c FROM basic

    -- expected
    c:int64
    1
    2
    3
    4
    0
    6
    7
    """
    var t = table("basic")
    var q = t.project(["c"], [FillNull(col("v", int64), lit(0, int64))])
    return q
