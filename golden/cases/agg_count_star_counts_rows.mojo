from golden.helpers import table
from marrow.expr.builders import count_star
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(*) AS BIGINT) AS n FROM basic

    -- expected
    n:int64
    7
    """
    var t = table("basic")
    var q = t.aggregate(aggs=[count_star().alias("n")])
    return q
