from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(timezone('UTC', ts) AS VARCHAR) AS t FROM events

    Interpreting a naive timestamp in a zone. marrow's `timestamp` dtype
    carries a timezone slot, but nothing in the expression layer reads or
    changes it, so attaching one and converting between two are both absent —
    and they are different operations, which is the distinction this records.

    The twin renders the result as text: the expectation block's `timestamp` is
    zone-free by construction.

    `marrow/kernels/temporal.mojo` has no such kernel.

    -- skip mojo
    -- skip python

    -- expected
    t:string
    '2021-01-01 01:00:00+01'
    '2021-06-15 14:30:45+02'
    '2021-06-15 14:30:45+02'
    NULL
    '2020-03-01 00:59:59+01'
    '2022-01-01 00:59:59.999999+01'
    """
    var t = table("events")
    return t.project(
        ["t"], [col("ts", timestamp(microsecond)).assign_timezone("UTC")]
    )
