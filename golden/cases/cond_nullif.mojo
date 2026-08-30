from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT nullif(v, 3) AS c FROM basic

    The inverse of `coalesce`: it *introduces* a null where the two operands
    are equal. The row that was already null stays null — `nullif(NULL, 3)` is
    NULL, not 3 — so the output has two nulls from two different causes.

    -- expected
    c:int64
    1
    2
    NULL
    4
    NULL
    6
    7
    """
    var t = table("basic")
    return t.project(["c"], [Nullif(col("v", int64), lit(3, int64))])
