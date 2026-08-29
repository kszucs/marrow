from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(d) AS lo, max(d) AS hi FROM events

    The int32-backed half of temporal `min`/`max`, and the null must be skipped
    rather than sorting to one end.

    -- expected
    lo:date32	hi:date32
    '2020-02-29'	'2021-12-31'
    """
    var t = table("events")
    return t.aggregate(
        aggs=[
            col("d", date32()).min().alias("lo"),
            col("d", date32()).max().alias("hi"),
        ]
    )
