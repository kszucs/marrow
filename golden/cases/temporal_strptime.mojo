from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT strptime(strftime(ts, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') AS t FROM events

    The inverse, and a round trip so the case asserts a timestamp rather than
    pinning a format. The fractional second is lost on the way out and cannot
    come back, which is why `2021-12-31 23:59:59.999999` does not return to
    itself.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    t:timestamp
    '2021-01-01T00:00:00'
    '2021-06-15T12:30:45'
    '2021-06-15T12:30:45'
    NULL
    '2020-02-29T23:59:59'
    '2021-12-31T23:59:59'
    """
    var t = table("events")
    var formatted = t.project(
        ["s"], [col("ts", timestamp(microsecond)).strftime("%Y-%m-%d %H:%M:%S")]
    )
    return formatted.project(
        ["t"], [col("s", string).strptime("%Y-%m-%d %H:%M:%S")]
    )
