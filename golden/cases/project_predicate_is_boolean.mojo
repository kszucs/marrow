from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT v > 3 AS gt FROM basic

    -- expected
    gt:bool
    False
    False
    False
    True
    NULL
    True
    True
    """
    var t = table("basic")
    var q = t.project(["gt"], [col("v", int64) > lit(3, int64)])
    return q
