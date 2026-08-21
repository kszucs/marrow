from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(date_trunc('month', d) AS DATE) AS m FROM events

    A recorded divergence in the *type*, not the value. DuckDB's
    `date_trunc` returns TIMESTAMP whatever it is given, so `date_trunc('month',
    d)` over a DATE column widens to timestamp[us]. marrow's `DateTruncKernel`
    returns the input dtype unchanged, so a date32 column truncates to date32.

    The twin casts back to DATE to ask for marrow's type. Both engines agree on
    the instant; only DuckDB's return type is wider.

    -- expected
    m:date32
    '2021-01-01'
    '2021-06-01'
    NULL
    '2020-02-01'
    '2021-12-01'
    '2021-06-01'
    """
    var t = table("events")
    var q = t.project(["m"], [col("d", date32()).date_trunc("month")])
    return q
