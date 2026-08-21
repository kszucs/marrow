from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(ts) AS lo, max(ts) AS hi FROM events

    `min`/`max` over a timestamp is a third kernel again — not the numeric
    fold and not the bytewise string one. `TemporalMinMax` folds over the signed
    integer backing and relabels the result with the input dtype, unit and
    timezone intact, so `hi` must come back as timestamp[us] carrying the last
    microsecond of 2021 rather than a truncated second.

    -- expected
    lo:timestamp	hi:timestamp
    '2020-02-29T23:59:59'	'2021-12-31T23:59:59.999999'
    """
    var t = table("events")
    var q = t.aggregate(
        aggs=[
            col("ts", timestamp(microsecond)).min().alias("lo"),
            col("ts", timestamp(microsecond)).max().alias("hi"),
        ]
    )
    return q
