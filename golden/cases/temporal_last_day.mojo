from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(last_day(d) AS DATE) AS l FROM events

    The last day of the month, which needs the leap rule rather than a table of
    month lengths: `2020-02-29` must answer `2020-02-29` and not `2020-02-28`.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    l:date32
    '2021-01-31'
    '2021-06-30'
    NULL
    '2020-02-29'
    '2021-12-31'
    '2021-06-30'
    """
    var t = table("events")
    return t.project(["l"], [col("d", date32).last_day()])
