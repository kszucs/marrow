from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(DISTINCT v) AS BIGINT) AS n FROM basic

    The numeric half of `agg_count_distinct_string`: a different accumulator,
    since `CountDistinct` hashes a `PrimitiveArray[T]` where the string form
    hashes a `BinaryLikeArray[T]`. Nulls are skipped, so the answer is 6 and
    not 7.

    -- expected
    n:int64
    6
    """
    var t = table("basic")
    return t.aggregate(aggs=[col("v", int64).count_distinct().alias("n")])
