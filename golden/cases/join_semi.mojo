from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept FROM emp e WHERE EXISTS (SELECT 1 FROM dept d WHERE e.dept = d.did) ORDER BY e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64
    1	10
    2	20
    3	20
    """
    var left = table("emp")
    var joined = left.join(table("dept"), [1], [0], JOIN_SEMI)
    return joined.sort_by([col("eid", int64)], [True])
