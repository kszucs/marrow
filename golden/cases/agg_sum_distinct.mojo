from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(DISTINCT k) AS BIGINT) AS n FROM (SELECT length(k) AS k FROM basic) t

    `DISTINCT` inside an aggregate that is not `count`. marrow supports it for
    `count` only, through a dedicated `DistinctCount` kernel; the general form
    needs the distinct set and the fold to compose.

    The inner query reduces `k` to its length so that the deduplication has
    something to do: seven rows, two distinct lengths, one of them null.

    -- skip mojo
    -- skip python

    -- expected
    n:int64
    1
    """
    var t = table("basic")
    var lengths = t.project(["k"], [StringLength(col("k", string))])
    return lengths.aggregate(
        aggs=[col("k", int32).sum(distinct=True).alias("n")]
    )
