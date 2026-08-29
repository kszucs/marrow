from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT p, q FROM flags WHERE p

    A null predicate does not select, so only the three TRUE rows survive.

    -- expected
    p:bool	q:bool
    True	True
    True	False
    True	NULL
    """
    var t = table("flags")
    return t.filter(col("p", bool_))
