from golden.helpers import table
from marrow.dtypes import Int64Type, bool_
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import BoolToNum


def plan() raises -> DynRelation:
    """
    SELECT CAST(b AS BIGINT) AS c FROM nums

    -- expected
    c:int64
    1
    0
    1
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [BoolToNum[Int64Type](col("b", bool_))])
    return q
