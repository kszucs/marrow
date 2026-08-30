from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(i) AS lo, max(i) AS hi FROM edges

    `min` and `max` at the type's limits. An accumulator seeded with 0, or one
    that tracks the extremum in a wider signed type and narrows on the way out,
    answers this wrongly while passing every other min/max case in the corpus.

    -- expected
    lo:int64	hi:int64
    -9223372036854775808	9223372036854775807
    """
    var t = table("edges")
    return t.aggregate(
        aggs=[
            col("i", int64).min().alias("lo"),
            col("i", int64).max().alias("hi"),
        ]
    )
