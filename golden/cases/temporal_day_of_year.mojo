from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(dayofyear(ts) AS INTEGER) AS doy FROM events

    1-based. The leap day is the row that matters: 2020-02-29 is day 60,
    which a 365-day table would call March 1.

    -- expected
    doy:int32
    1
    166
    166
    NULL
    60
    365
    """
    var t = table("events")
    return t.project(["doy"], [col("ts", timestamp(microsecond)).day_of_year()])
