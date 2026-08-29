from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT date_trunc('day', ts) AS dd FROM events

    Truncating to a fixed-length unit rather than a calendar one — a
    division on the tick count, where `month` has to go through the civil
    calendar. The leap-day row is the one that would expose an off-by-one.

    -- expected
    dd:timestamp
    '2021-01-01T00:00:00'
    '2021-06-15T00:00:00'
    '2021-06-15T00:00:00'
    NULL
    '2020-02-29T00:00:00'
    '2021-12-31T00:00:00'
    """
    var t = table("events")
    return t.project(
        ["dd"], [col("ts", timestamp(microsecond)).date_trunc("day")]
    )
