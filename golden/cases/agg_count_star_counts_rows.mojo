from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(*) AS BIGINT) AS n FROM basic

    -- expected
    n:int64
    7
    """
    var t = table("basic")
    var q = t.aggregate(aggs=[count_star().alias("n")])
    return q
