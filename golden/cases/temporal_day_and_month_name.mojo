from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT dayname(d) AS dn, monthname(d) AS mn FROM events

    Field extraction that answers with a *name* rather than a number, so the
    result is locale-shaped text and the answer depends on a table the engine
    carries. marrow's nine extractions all return int32.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    dn:string	mn:string
    'Friday'	'January'
    'Tuesday'	'June'
    NULL	NULL
    'Saturday'	'February'
    'Friday'	'December'
    'Tuesday'	'June'
    """
    var t = table("events")
    return t.project(
        ["dn", "mn"],
        [col("d", date32).day_name(), col("d", date32).month_name()],
    )
