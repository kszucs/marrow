from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ts + INTERVAL 1 MONTH AS t FROM events

    Adding a month is not adding a fixed number of ticks, and `2020-02-29` is
    the row that says so — the target month has no 29th in some years and the
    result has to clamp. There is no interval type in the expression layer and
    no calendar arithmetic kernel.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    t:timestamp
    '2021-02-01T00:00:00'
    '2021-07-15T12:30:45'
    '2021-07-15T12:30:45'
    NULL
    '2020-03-29T23:59:59'
    '2022-01-31T23:59:59.999999'
    """
    var t = table("events")
    return t.project(
        ["t"], [col("ts", timestamp(microsecond)) + interval(months=1)]
    )
