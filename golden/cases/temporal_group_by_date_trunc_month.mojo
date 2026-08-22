from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT date_trunc('month', ts) AS key0, CAST(count(*) AS BIGINT) AS n FROM events GROUP BY date_trunc('month', ts) ORDER BY key0 NULLS FIRST

    Groups by a *computed* temporal key, which is how a real time-series
    rollup is written. A computed key has no source column to name it after, so
    the plan calls it `key0` and the twin matches that.

    -- expected
    key0:timestamp	n:int64
    NULL	1
    '2020-02-01T00:00:00'	1
    '2021-01-01T00:00:00'	1
    '2021-06-01T00:00:00'	2
    '2021-12-01T00:00:00'	1
    """
    var t = table("events")
    var agg = t.aggregate(
        keys=[col("ts", timestamp(microsecond)).date_trunc("month")],
        aggs=[count_star().alias("n")],
    )
    var q = agg.sort([col("key0", timestamp(microsecond))], [True])
    return q
