from golden.helpers import table
from marrow.dtypes import bool_
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT p, q, p OR q AS r FROM flags

    -- expected
    p:bool	q:bool	r:bool
    True	True	True
    True	False	True
    True	NULL	True
    False	True	True
    False	False	False
    False	NULL	NULL
    NULL	True	True
    NULL	False	NULL
    NULL	NULL	NULL
    """
    var t = table("flags")
    var q = t.project(
        ["p", "q", "r"],
        [col("p", bool_), col("q", bool_), col("p", bool_) | col("q", bool_)],
    )
    return q
