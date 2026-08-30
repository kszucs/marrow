from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT list_contains(l, 5) AS b FROM lists

    Membership within a list. `marrow/kernels/nested.mojo` **has**
    `array_contains`, so this is a missing expression node rather than a
    missing kernel — the same shape as `agg_bool_and_or`.

    The null list answers null and the empty list answers false, which are
    different answers to "is 5 in there".

    -- skip mojo

    -- expected
    b:bool
    False
    False
    NULL
    True
    False
    """
    var t = table("lists")
    return t.project(["b"], [col("l", list_(int64)).contains(lit(5, int64))])
