from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT date_trunc('quarter', ts) AS q FROM events

    The quarter boundary, which is the only `date_trunc` unit that is neither a
    fixed number of ticks nor a whole month count: it floors the month to one
    of 1, 4, 7, 10. `2020-02-29` is the row that separates it from `month`.

    -- expected
    q:timestamp
    '2021-01-01T00:00:00'
    '2021-04-01T00:00:00'
    '2021-04-01T00:00:00'
    NULL
    '2020-01-01T00:00:00'
    '2021-10-01T00:00:00'
    """
    var t = table("events")
    return t.project(
        ["q"], [col("ts", timestamp(microsecond)).date_trunc("quarter")]
    )
