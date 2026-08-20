from golden.helpers import table
from marrow.dtypes import bool_
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation


def plan() raises -> DynRelation:
    """
    SELECT p, q, p AND q AS r FROM flags

    Reads bool *columns*, not derived predicates. The AOT lane could not
    express this at all until `col` gained a `BoolType` overload — `col` had
    numeric, string, list and temporal leaves and no boolean one, so a fused
    expression had no way to reference a bool column.

    -- expected
    p:bool	q:bool	r:bool
    True	True	True
    True	False	False
    True	NULL	NULL
    False	True	False
    False	False	False
    False	NULL	False
    NULL	True	NULL
    NULL	False	False
    NULL	NULL	NULL
    """
    var t = table("flags")
    var q = t.project(
        ["p", "q", "r"],
        [col("p", bool_), col("q", bool_), col("p", bool_) & col("q", bool_)],
    )
    return q
