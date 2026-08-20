from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import Coalesce


def plan() raises -> DynRelation:
    """
    SELECT coalesce(v, w) AS c FROM basic

    -- expected
    c:int64
    1
    2
    3
    4
    50
    6
    7
    """
    var t = table("basic")
    var q = t.project(["c"], [Coalesce(col("v", int64), col("w", int64))])
    return q
