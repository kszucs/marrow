from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(month(ts) AS INTEGER) AS mo, CAST(day(ts) AS INTEGER) AS dy FROM events

    Both are 1-based, and the leap day is in the fixture so `day` has a 29 to find.

    -- expected
    mo:int32	dy:int32
    1	1
    6	15
    6	15
    NULL	NULL
    2	29
    12	31
    """
    var t = table("events")
    var q = t.project(
        ["mo", "dy"],
        [
            col("ts", timestamp(microsecond)).month(),
            col("ts", timestamp(microsecond)).day(),
        ],
    )
    return q
