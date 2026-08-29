from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(quarter(ts) AS INTEGER) AS q FROM events

    `(month - 1) / 3 + 1`, so 1..4 rather than 0..3.

    -- expected
    q:int32
    1
    2
    2
    NULL
    1
    4
    """
    var t = table("events")
    return t.project(["q"], [col("ts", timestamp(microsecond)).quarter()])
