from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation
from marrow.expr.values import CaseWhen


def plan() raises -> DynRelation:
    """
    SELECT CASE WHEN v > 3 THEN v ELSE w END AS c FROM basic

    A null condition counts as false in Arrow, so the null row takes the
    ELSE branch rather than becoming null.

    -- expected
    c:int64
    10
    NULL
    30
    4
    50
    6
    7
    """
    var t = table("basic")
    var q = t.project(
        ["c"],
        [
            CaseWhen(
                col("v", int64) > lit(3, int64),
                col("v", int64),
                col("w", int64),
            )
        ],
    )
    return q
