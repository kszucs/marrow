from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(age(TIMESTAMP '2022-01-01 00:00:00', ts) AS VARCHAR) AS a FROM events

    `age` is calendar subtraction: it reports years, months and days rather
    than a duration, so the answer for a February date depends on the month
    lengths in between. That is the same calendar machinery
    `temporal_add_month_interval` needs, read in the other direction.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    a:string
    '1 year'
    '6 months 15 days 11:29:15'
    '6 months 15 days 11:29:15'
    NULL
    '1 year 10 months 00:00:01'
    '00:00:00.000001'
    """
    var t = table("events")
    return t.project(
        ["a"],
        [
            age(
                lit(datetime(2022, 1, 1), timestamp(microsecond)),
                col("ts", timestamp(microsecond)),
            )
        ],
    )
