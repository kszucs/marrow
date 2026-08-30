from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT min(a) AS lo, max(a) AS hi, avg(a) AS m FROM nulls

    Three aggregates over zero valid rows, in one case because they answer one
    question: an aggregate with no input is null, not the identity element and
    not the accumulator's seed. `nulls_sum_of_all_null_is_null` asks it of
    `sum`; these are the three that could each get it wrong differently — a
    `min` seeded with the type's maximum, a `max` seeded with its minimum, a
    `mean` dividing by zero.

    -- expected
    lo:int64	hi:int64	m:double
    NULL	NULL	NULL
    """
    var t = table("nulls")
    return t.aggregate(
        aggs=[
            col("a", int64).min().alias("lo"),
            col("a", int64).max().alias("hi"),
            col("a", int64).mean().alias("m"),
        ]
    )
