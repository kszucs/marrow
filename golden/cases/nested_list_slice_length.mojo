from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(len(list_slice(l, 2, 3)) AS BIGINT) AS n FROM lists

    Slicing, asserted through its length so the result stays flat. The bounds
    are clamped rather than checked: `[7]` sliced from 2 to 3 is empty, not an
    error, and the null list stays null.

    -- skip mojo
    -- skip python

    -- expected
    n:int64
    2
    0
    NULL
    1
    0
    """
    var t = table("lists")
    return t.project(["n"], [array_length(col("l", list_(int64)).slice(2, 3))])
