from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT price, CAST(count(*) AS BIGINT) AS n FROM sales GROUP BY price ORDER BY price NULLS FIRST

    Float group keys, which the hash kernel used to collapse: it widened a
    lane with `cast[uint64]()`, a *numeric* conversion, so every value in
    (-1, 1) truncated to 0 — and grouping buckets on the hash alone, so -1.25
    and 0.5 were one group. `RapidHashKernel` now hashes the bit pattern, with
    `-0.0` and NaN canonicalised first so that "same number" and "same bits"
    cannot disagree.

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
