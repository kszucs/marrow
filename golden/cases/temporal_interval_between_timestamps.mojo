from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(ts - TIMESTAMP '2021-01-01 00:00:00' AS VARCHAR) AS d FROM events

    Subtracting two timestamps yields an *interval*, not a number, and the
    interval carries months, days and microseconds separately rather than
    normalising to one unit. The twin renders it as text because the
    expectation block has no interval type — which is itself the honest
    statement that marrow has nowhere to put the value.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    d:string
    '00:00:00'
    '165 days 12:30:45'
    '165 days 12:30:45'
    NULL
    '-306 days -00:00:01'
    '364 days 23:59:59.999999'
    """
    var t = table("events")
    return t.project(
        ["d"],
        [
            col("ts", timestamp(microsecond))
            - lit(datetime(2021, 1, 1), timestamp(microsecond))
        ],
    )
