from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT date_trunc('month', ts) AS m FROM events

    `date_trunc` preserves its input type, so this is timestamp in and
    timestamp[us] out — the expectation block is the first thing in the corpus
    whose column type is `timestamp`.

    -- expected
    m:timestamp
    '2021-01-01T00:00:00'
    '2021-06-01T00:00:00'
    '2021-06-01T00:00:00'
    NULL
    '2020-02-01T00:00:00'
    '2021-12-01T00:00:00'
    """
    var t = table("events")
    var q = t.project(
        ["m"], [col("ts", timestamp(microsecond)).date_trunc("month")]
    )
    return q
