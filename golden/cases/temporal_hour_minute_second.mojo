from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(hour(ts) AS INTEGER) AS h, CAST(minute(ts) AS INTEGER) AS mi, CAST(second(ts) AS INTEGER) AS s FROM events

    The time-of-day kernels, which unlike the calendar ones reject a date32
    input. `second` is whole seconds: the fixture's last row is
    `23:59:59.999999`, and both engines answer 59 rather than rounding up.

    -- expected
    h:int32	mi:int32	s:int32
    0	0	0
    12	30	45
    12	30	45
    NULL	NULL	NULL
    23	59	59
    23	59	59
    """
    var t = table("events")
    var q = t.project(
        ["h", "mi", "s"],
        [
            col("ts", timestamp(microsecond)).hour(),
            col("ts", timestamp(microsecond)).minute(),
            col("ts", timestamp(microsecond)).second(),
        ],
    )
    return q
