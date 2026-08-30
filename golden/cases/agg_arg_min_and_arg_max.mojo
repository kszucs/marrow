from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT arg_min(k, v) AS lo, arg_max(k, v) AS hi FROM basic

    Two columns, one aggregate: return `k` from the row where `v` is smallest
    and largest. Every aggregate marrow has reads a single operand, so this is
    a shape gap and not only a missing kernel — `Aggregate[Agg, A]` has one
    `A`.

    The rows where `v` is null are skipped rather than winning the minimum.

    -- skip mojo

    -- expected
    lo:string	hi:string
    'a'	'a'
    """
    var t = table("basic")
    return t.aggregate(
        aggs=[
            col("k", string).arg_min(col("v", int64)).alias("lo"),
            col("k", string).arg_max(col("v", int64)).alias("hi"),
        ]
    )
