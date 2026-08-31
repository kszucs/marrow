from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT unnest(l) AS e FROM lists

    The relational verb of the list family: one output row per element, so the
    row count changes and this is a plan node rather than an expression. The
    empty list and the null list both contribute **no** rows — not one row of
    null — and a null *element* does contribute one.

    -- skip mojo
    -- skip python

    -- expected
    e:int64
    1
    2
    3
    NULL
    5
    7
    """
    var t = table("lists")
    return t.unnest(col("l", list_(int64)), "e").select(["e"])
