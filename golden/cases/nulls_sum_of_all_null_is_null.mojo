from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(a) AS BIGINT) AS total FROM nulls

    -- expected
    total:int64
    NULL
    """
    var t = table("nulls")
    var q = t.aggregate(
        aggs=[col("a", int64).sum().alias("total")],
    )
    return q
