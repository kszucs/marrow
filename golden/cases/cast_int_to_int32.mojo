from golden.helpers import table
from marrow.dtypes import Int32Type, int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import NumericCast


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS INTEGER) AS c FROM nums

    -- expected
    c:int32
    1
    -2
    300
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumericCast[Int32Type](col("i", int64))])
    return q
