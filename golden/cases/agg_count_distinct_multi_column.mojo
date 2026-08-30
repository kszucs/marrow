from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(DISTINCT (region, active)) AS BIGINT) AS n FROM sales

    Distinct over a *tuple*. marrow's `count_distinct` hashes one typed array;
    counting distinct pairs needs the composite key encoding the group-by
    already has, applied to an aggregate's operand.

    `(NULL, NULL)` counts as a value here — set semantics, not comparison
    semantics — which is the same rule `setop_union_distinct` states.

    -- skip mojo

    -- expected
    n:int64
    4
    """
    var t = table("sales")
    return t.aggregate(
        aggs=[
            count_distinct([col("region", string), col("active", bool_)]).alias(
                "n"
            )
        ]
    )
