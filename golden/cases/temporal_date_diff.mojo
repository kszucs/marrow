from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(date_diff('day', d, DATE '2021-12-31') AS BIGINT) AS n FROM events

    `date_diff` counts **boundary crossings**, not elapsed duration: the number
    of unit boundaries between the two instants. That is a different function
    from subtracting and truncating, and the two disagree whenever the times of
    day do.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    n:int64
    364
    199
    NULL
    671
    0
    199
    """
    var t = table("events")
    return t.project(
        ["n"],
        [date_diff("day", col("d", date32), lit(date(2021, 12, 31), date32))],
    )
