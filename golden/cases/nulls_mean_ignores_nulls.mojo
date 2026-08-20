from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT avg(b) AS m FROM nulls

    -- expected
    m:double
    5.0
    """
    var t = table("nulls")
    var q = t.aggregate(
        aggs=[col("b", int64).mean().alias("m")],
    )
    return q
