from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT DISTINCT ON (region) region, qty FROM sales ORDER BY region NULLS FIRST, qty NULLS FIRST

    `DISTINCT ON` keeps one whole row per key — the first under the `ORDER BY`
    — where `SELECT DISTINCT` keeps one *tuple* per distinct combination. The
    difference is that the non-key columns come along unaggregated.

    There is no distinct-on node; `distinct_via_aggregate` shows the shape
    marrow does have, an `aggregate` with keys and no aggregates, which cannot
    carry `qty` through.

    -- skip mojo
    -- skip python

    -- expected
    region:string	qty:int32
    NULL	40
    'east'	50
    'north'	NULL
    'south'	5
    """
    var t = table("sales")
    return t.distinct_on([col("region", string)], [col("qty", int32)], [True])
