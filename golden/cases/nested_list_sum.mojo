from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(list_sum(l) AS BIGINT) AS s FROM lists

    An aggregate *within* a row's list rather than across rows — the same fold
    marrow already has, applied along the other axis. Nulls inside the list are
    skipped, so `[NULL, 5]` sums to 5, while the empty list sums to null and
    not to zero.

    -- skip mojo

    -- expected
    s:int64
    6
    NULL
    NULL
    5
    7
    """
    var t = table("lists")
    return t.project(["s"], [col("l", list_(int64)).sum()])
