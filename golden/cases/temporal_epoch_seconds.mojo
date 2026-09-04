from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(epoch(ts) AS BIGINT) AS e FROM events

    Seconds since the Unix epoch. The unit is the whole question — DuckDB's
    `epoch` is seconds while the column is microseconds, so this is a
    truncating conversion and not a reinterpretation, and `2021-12-31
    23:59:59.999999` must lose its fraction downward.

    **The kernel exists — this case still cannot be un-skipped, and the reason
    is the twin.** `EpochKernel` landed with the temporal
    surface, so the body compiles and answers; measured 2026-09-04, it differs
    on exactly one row. DuckDB's `epoch` returns a DOUBLE, so
    `2021-12-31 23:59:59.999999` is 1640995199.999999 and `CAST(... AS BIGINT)`
    **rounds** it to 1640995200 — a full second into the next year. marrow
    answers 1640995199, which is what this docstring asks for: the fraction
    lost *downward*.

    So the expectation encodes DuckDB's cast, not its `epoch`, and asserting it
    would pin the twin's rounding rather than a semantic marrow gets wrong.
    Un-skipping needs a twin written to truncate — `CAST(FLOOR(epoch(ts)) AS
    BIGINT)` — and a regenerated expectation, which is a change to the case,
    not to marrow.

    -- skip mojo
    -- skip python

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
