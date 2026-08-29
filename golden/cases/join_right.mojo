from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e RIGHT JOIN dept d ON e.dept = d.did ORDER BY d.did NULLS FIRST, e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    1	10	10	'eng'
    2	20	20	'sales'
    3	20	20	'sales'
    NULL	NULL	30	'ops'
    """
    var left = table("emp")
    var joined = left.join(table("dept"), [1], [0], JOIN_RIGHT)
    return joined.sort_by([col("did", int64), col("eid", int64)], [True, True])
