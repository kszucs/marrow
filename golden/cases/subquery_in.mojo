from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp WHERE dept IN (SELECT did FROM dept) ORDER BY eid

    `IN (subquery)` is a semi join: each left row appears once however
    many right rows it matches, and a NULL key matches nothing.

    -- expected
    eid:int64	dept:int64
    1	10
    2	20
    3	20
    """
    var left = table("emp")
    var joined = left.join(table("dept"), [1], [0], JOIN_SEMI)
    return joined.sort_by([col("eid", int64)], [True])
