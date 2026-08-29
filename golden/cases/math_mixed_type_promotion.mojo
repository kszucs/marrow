from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty + price AS s FROM sales

    int32 + float64. The output dtype is the assertion: `promote[L, R]` decides
    the *value domain*, where a float outranks any integer, so the result is
    double and not a truncated int32. `qty` is null on one row and `price` on
    another, so the intersection of the two validity maps is also exercised.

    -- expected
    s:double
    11.5
    22.25
    NULL
    NULL
    54.0
    3.75
    """
    var t = table("sales")
    return t.project(["s"], [col("qty", int32) + col("price", float64)])
