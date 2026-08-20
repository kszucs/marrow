from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import NumToBool


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS BOOLEAN) AS c FROM nums

    -- expected
    c:bool
    True
    True
    True
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumToBool(col("i", int64))])
    return q
