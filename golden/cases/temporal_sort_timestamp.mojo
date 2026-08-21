from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ts, label FROM events ORDER BY ts NULLS FIRST

    Sorting on a timestamp. The two equal instants keep their input order
    (the sort is stable), and the microsecond row must sort above the whole
    second before it rather than comparing equal to it.

    -- expected
    ts:timestamp	label:string
    NULL	NULL
    '2020-02-29T23:59:59'	'c'
    '2021-01-01T00:00:00'	'a'
    '2021-06-15T12:30:45'	'b'
    '2021-06-15T12:30:45'	'a'
    '2021-12-31T23:59:59.999999'	'b'
    """
    var t = table("events")
    var picked = t.select("ts", "label")
    var q = picked.sort([col("ts", timestamp(microsecond))], [True])
    return q
