from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT v, (v IS DISTINCT FROM 3) AS b FROM basic

    The null-safe comparison: two-valued where `=` is three-valued, so the null
    row answers **true** rather than null. It is the operator that makes NULL
    behave the way set operations and `GROUP BY` already treat it, and marrow
    has only the three-valued form (`nulls_equality_with_null_is_null`).

    -- skip mojo
    -- skip python

    -- expected
    v:int64	b:bool
    1	True
    2	True
    3	False
    4	True
    NULL	True
    6	True
    7	True
    """
    var t = table("basic")
    return t.project(
        ["v", "b"],
        [
            col("v", int64),
            col("v", int64).is_distinct_from(lit(3, int64)),
        ],
    )
