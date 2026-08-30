from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(make_date(year(d), 1, 1) AS DATE) AS j FROM events

    Building a date from components, the inverse of the extraction family
    marrow does have. The null row is the rule worth stating: a null component
    makes a null date rather than an error.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo

    -- expected
    j:date32
    '2021-01-01'
    '2021-01-01'
    NULL
    '2020-01-01'
    '2021-01-01'
    '2021-01-01'
    """
    var t = table("events")
    return t.project(
        ["j"],
        [make_date(col("d", date32).year(), lit(1, int32), lit(1, int32))],
    )
