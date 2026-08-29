from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ts, label FROM events WHERE ts > date_trunc('year', ts)

    A predicate comparing two timestamps, which drops the null and the one
    row that sits exactly on its own year boundary.

    The right-hand side is a truncation of the column rather than a constant
    because `lit` has no temporal overload in either lane, so a timestamp
    literal cannot be written down at all. Comparing a column against a
    truncation of itself is still a genuine timestamp comparison.

    It is spelled in the **comptime** lane. It could not be until
    `TemporalValue` grew its six comparison dunders: `TemporalCompare` and its
    aliases existed with no callers, so the only way to reach one was to name
    `TemporalGt(...)` by hand. Both operands are `timestamp[us]`, which is what
    the node's per-batch dtype check requires — cross-unit comparison raises
    rather than silently coercing.

    -- expected
    ts:timestamp	label:string
    '2021-06-15T12:30:45'	'b'
    '2021-06-15T12:30:45'	'a'
    '2020-02-29T23:59:59'	'c'
    '2021-12-31T23:59:59.999999'	'b'
    """
    var t = table("events")
    var kept = t.filter(
        col("ts", timestamp(microsecond))
        > col("ts", timestamp(microsecond)).date_trunc("year")
    )
    return kept.select(["ts", "label"])
