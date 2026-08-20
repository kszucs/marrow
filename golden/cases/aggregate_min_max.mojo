from golden.helpers import table
from marrow.dtypes import int64, string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT k, min(v) AS lo, max(w) AS hi FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	lo:int64	hi:int64
    NULL	7	70
    'a'	1	60
    'b'	2	50
    'c'	4	40
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            col("v", int64).min().alias("lo"),
            col("w", int64).max().alias("hi"),
        ],
    )
    var q = agg.sort([col("k", string)], [True])
    return q
