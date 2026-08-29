from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT price, CAST(count(*) AS BIGINT) AS n FROM sales GROUP BY price ORDER BY price NULLS FIRST

    -- xfail float group keys collapse: -1.25 and 0.5 land in one group (repro: 3 distinct floats -> 2 groups)

    -- expected
    price:double	n:int64
    NULL	1
    -1.25	1
    0.5	1
    1.5	1
    2.25	1
    4.0	1
    """
    var t = table("sales")
    var agg = t.aggregate(
        keys=[col("price", float64)], aggs=[count_star().alias("n")]
    )
    return agg.sort_by([col("price", float64)], [True])
