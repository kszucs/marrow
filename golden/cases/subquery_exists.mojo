from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp e WHERE EXISTS (SELECT 1 FROM dept d WHERE d.did = e.dept) ORDER BY eid

    `EXISTS` and `IN` agree here because the correlation is the join
    condition itself.

    -- expected
    eid:int64	dept:int64
    1	10
    2	20
    3	20
    """
    var left = table("emp")
    var joined = left.join(table("dept"), [1], [0], JOIN_SEMI)
    return joined.sort_by([col("eid", int64)], [True])
