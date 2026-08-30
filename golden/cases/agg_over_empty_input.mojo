from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(*) AS BIGINT) AS n, CAST(sum(v) AS BIGINT) AS total, avg(v) AS m FROM basic WHERE v > 1000

    Zero *rows*, where `agg_all_null_column` has rows whose values are all
    null. The two reach the same accumulator by different routes — one never
    sees a batch, the other sees batches and skips every value — and `count(*)`
    separates them from a missing result: it is 0, while `sum` and `avg` are
    null.

    -- expected
    n:int64	total:int64	m:double
    0	NULL	NULL
    """
    var t = table("basic")
    var empty = t.filter(col("v", int64) > lit(1000, int64))
    return empty.aggregate(
        aggs=[
            count_star().alias("n"),
            col("v", int64).sum().alias("total"),
            col("v", int64).mean().alias("m"),
        ]
    )
