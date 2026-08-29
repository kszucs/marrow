from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept FROM emp e WHERE NOT EXISTS (SELECT 1 FROM dept d WHERE e.dept = d.did) ORDER BY e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64
    4	99
    5	NULL
    """
    var left = table("emp")
    var joined = left.join(table("dept"), [1], [0], JOIN_ANTI)
    return joined.sort_by([col("eid", int64)], [True])
