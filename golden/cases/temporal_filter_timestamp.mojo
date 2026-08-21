from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ts, label FROM events WHERE ts > date_trunc('year', ts)

    A predicate comparing two timestamps, which drops the null and the one
    row that sits exactly on its own year boundary.

    The comparison is spelled through the **runtime** lane — `col("ts")` with no
    dtype — because the AOT lane cannot express a temporal comparison at all:
    `TemporalValue` (`marrow/expr/values.mojo`) carries the extraction,
    truncation and min/max methods but no relational operators, and `lit` has no
    temporal overload, so neither `col("ts", timestamp(microsecond)) > ...` nor a
    timestamp constant compiles. Comparing the column against a truncation of
    itself is a genuine timestamp comparison that both lanes can spell.

    -- expected
    ts:timestamp	label:string
    '2021-06-15T12:30:45'	'b'
    '2021-06-15T12:30:45'	'a'
    '2020-02-29T23:59:59'	'c'
    '2021-12-31T23:59:59.999999'	'b'
    """
    var t = table("events")
    var kept = t.filter(col("ts") > col("ts").date_trunc("year"))
    var q = kept.select("ts", "label")
    return q
