from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(sum(v) FILTER (WHERE v > 2) AS BIGINT) AS total FROM basic GROUP BY k ORDER BY k NULLS FIRST

    `FILTER` restricts one aggregate without restricting the query. It is not
    the same as a `WHERE`: group `a` keeps its row here even though its `v = 1`
    is excluded from the sum, and a group with *no* surviving row still
    appears, with a null total.

    marrow would have to carry a predicate on the aggregate node.

    -- skip mojo

    -- expected
    k:string	total:int64
    NULL	7
    'a'	9
    'b'	NULL
    'c'	4
    """
    var t = table("basic")
    var agg = t.aggregate(
        keys=[col("k", string)],
        aggs=[
            col("v", int64)
            .sum()
            .filter(col("v", int64) > lit(2, int64))
            .alias("total")
        ],
    )
    return agg.sort_by([col("k", string)], [True])
