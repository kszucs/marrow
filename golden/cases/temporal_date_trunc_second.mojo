from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT date_trunc('second', ts) AS s FROM events

    The sub-second boundary. `2021-12-31 23:59:59.999999` is the value this
    case exists for: truncating drops the fraction and must not round the
    timestamp into the next year.

    -- expected
    s:timestamp
    '2021-01-01T00:00:00'
    '2021-06-15T12:30:45'
    '2021-06-15T12:30:45'
    NULL
    '2020-02-29T23:59:59'
    '2021-12-31T23:59:59'
    """
    var t = table("events")
    return t.project(
        ["s"], [col("ts", timestamp(microsecond)).date_trunc("second")]
    )
