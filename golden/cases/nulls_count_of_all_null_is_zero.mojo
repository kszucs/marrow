from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(a) AS BIGINT) AS n FROM nulls

    -- expected
    n:int64
    0
    """
    var t = table("nulls")
    var q = t.aggregate(
        aggs=[col("a", int64).count().alias("n")],
    )
    return q
