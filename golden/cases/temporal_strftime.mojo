from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT strftime(ts, '%Y-%m-%dT%H:%M:%S') AS s FROM events

    Formatting a timestamp to text with an explicit pattern.
    `cast_int_to_string` covers the numeric side of "value to text"; there is
    no temporal one at all — `NumToString` binds `A: NumericValue`.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    s:string
    '2021-01-01T00:00:00'
    '2021-06-15T12:30:45'
    '2021-06-15T12:30:45'
    NULL
    '2020-02-29T23:59:59'
    '2021-12-31T23:59:59'
    """
    var t = table("events")
    return t.project(
        ["s"], [col("ts", timestamp(microsecond)).strftime("%Y-%m-%dT%H:%M:%S")]
    )
