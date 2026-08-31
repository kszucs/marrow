from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, w FROM basic WHERE v > (SELECT avg(b) FROM nulls) ORDER BY v NULLS FIRST

    An uncorrelated scalar subquery — one relation's aggregate used as a
    constant in another's predicate. It needs no correlation machinery, only
    the ability to evaluate a plan to one value and broadcast it, which is the
    smallest step beyond the join-shaped subqueries `subquery_in` and
    `subquery_exists` cover.

    -- skip mojo
    -- skip python

    -- expected
    k:string	v:int64	w:int64
    'a'	6	60
    NULL	7	70
    """
    var threshold = table("nulls").aggregate(
        aggs=[col("b", int64).mean().alias("m")]
    )
    var t = table("basic")
    return t.filter(col("v", int64) > scalar_of(threshold)).sort_by(
        [col("v", int64)], [True]
    )
