from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(v) AS BIGINT) AS total FROM basic

    -- expected
    total:int64
    23
    """
    var t = table("basic")
    var q = t.aggregate(
        aggs=[col("v", int64).sum().alias("total")],
    )
    return q
