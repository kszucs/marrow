from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(epoch(ts) AS BIGINT) AS e FROM events

    Seconds since the Unix epoch. The unit is the whole question — DuckDB's
    `epoch` is seconds while the column is microseconds, so this is a
    truncating conversion and not a reinterpretation, and `2021-12-31
    23:59:59.999999` must lose its fraction downward.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    e:int64
    1609459200
    1623760245
    1623760245
    NULL
    1583020799
    1640995200
    """
    var t = table("events")
    return t.project(["e"], [col("ts", timestamp(microsecond)).epoch()])
