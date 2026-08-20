from golden.helpers import table
from marrow.dtypes import Float64Type, int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.expr.values import NumericCast


def plan() raises -> DynRelation:
    """
    SELECT CAST(i AS DOUBLE) AS c FROM nums

    -- expected
    c:double
    1.0
    -2.0
    300.0
    NULL
    """
    var t = table("nums")
    var q = t.project(["c"], [NumericCast[Float64Type](col("i", int64))])
    return q
