from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(week(ts) AS BIGINT) AS w, CAST(isoyear(ts) AS BIGINT) AS y FROM events

    The ISO week numbering, which is where a year boundary stops agreeing with
    itself: `2021-01-01` falls in ISO week 53 of **2020**, so `year` and
    `isoyear` differ on the same row. marrow extracts nine temporal fields and
    neither of these is among them.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip python

    -- expected
    w:int64	y:int64
    53	2020
    24	2021
    24	2021
    NULL	NULL
    9	2020
    52	2021
    """
    var t = table("events")
    return t.project(
        ["w", "y"],
        [
            col("ts", timestamp(microsecond)).week(),
            col("ts", timestamp(microsecond)).iso_year(),
        ],
    )
