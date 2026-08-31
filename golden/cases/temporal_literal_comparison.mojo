from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT ts, d, label FROM events WHERE d >= DATE '2021-01-01'

    A temporal column compared with a temporal **constant**. `lit` has numeric
    and string-like overloads only, so a date or timestamp literal cannot be
    spelled at all — `temporal_filter_timestamp` compares a column against a
    *derived* column for exactly that reason.

    This is the smallest missing piece in the temporal family and the one every
    other case here needs.

    -- skip mojo
    -- skip python

    -- expected
    ts:timestamp	d:date32	label:string
    '2021-01-01T00:00:00'	'2021-01-01'	'a'
    '2021-06-15T12:30:45'	'2021-06-15'	'b'
    '2020-02-29T23:59:59'	'2021-12-31'	'c'
    '2021-12-31T23:59:59.999999'	'2021-06-15'	'b'
    """
    var t = table("events")
    return t.filter(col("d", date32) >= lit(date(2021, 1, 1), date32))
