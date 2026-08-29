from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e LEFT JOIN dept d ON e.dept = d.did ORDER BY e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    1	10	10	'eng'
    2	20	20	'sales'
    3	20	20	'sales'
    4	99	NULL	NULL
    5	NULL	NULL	NULL
    """
    var left = table("emp")
    var joined = left.join(table("dept"), [1], [0], JOIN_LEFT)
    return joined.sort_by([col("eid", int64)], [True])
