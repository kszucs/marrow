from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(year(d) AS INTEGER) AS y, CAST(month(d) AS INTEGER) AS mo, CAST(day(d) AS INTEGER) AS dy FROM events

    The same calendar kernels over a **date32** column rather than a
    timestamp. date32 counts days and carries no time-of-day, so this exercises
    the other half of `_extract`'s decomposition.

    -- expected
    y:int32	mo:int32	dy:int32
    2021	1	1
    2021	6	15
    NULL	NULL	NULL
    2020	2	29
    2021	12	31
    2021	6	15
    """
    var t = table("events")
    return t.project(
        ["y", "mo", "dy"],
        [
            col("d", date32()).year(),
            col("d", date32()).month(),
            col("d", date32()).day(),
        ],
    )
