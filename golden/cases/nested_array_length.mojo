from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(len(l) AS INTEGER) AS n FROM lists

    The one list verb the expression layer has. Four answers in five rows: 3
    for an ordinary list, 0 for the empty one, NULL for the null one — not 0 —
    and 2 for `[NULL, 5]`, because a null *element* still occupies a position.

    `ListLength` is int32, like the temporal extractions, so the twin casts.

    -- expected
    n:int32
    3
    0
    NULL
    2
    1
    """
    var t = table("lists")
    return t.project(["n"], [array_length(col("l", list_(int64)))])
