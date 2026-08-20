from golden.helpers import table
from marrow.dtypes import int64, string
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(sum(v) AS BIGINT) AS total, max(v) AS biggest, CAST(count(w) AS BIGINT) AS n FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	total:int64	biggest:int64	n:int64
    NULL	7	7	1
    'a'	10	6	3
    'b'	2	2	1
    'c'	4	4	1
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            col("v", int64).sum().alias("total"),
            col("v", int64).max().alias("biggest"),
            col("w", int64).count().alias("n"),
        ],
    )
    var q = agg.sort([col("k", string)], [True])
    return q
