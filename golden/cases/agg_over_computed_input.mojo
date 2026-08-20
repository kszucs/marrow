from golden.helpers import table
from marrow.dtypes import int64, string
from marrow.expr.builders import col, lit
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(sum(v * 2) AS BIGINT) AS d FROM basic GROUP BY k ORDER BY k NULLS FIRST

    -- expected
    k:string	d:int64
    NULL	14
    'a'	20
    'b'	4
    'c'	8
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[(col("v", int64) * lit(2, int64)).sum().alias("d")],
    )
    var q = agg.sort([col("k", string)], [True])
    return q
