from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e LEFT JOIN (SELECT * FROM dept WHERE did > 1000) d ON e.dept = d.did ORDER BY e.eid NULLS FIRST

    An outer join whose right side is empty. Every left row survives with nulls
    widening the right columns — including the row whose key is NULL, which
    would match nothing anyway. An engine that short-circuits an empty build
    side loses the left rows entirely.

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    1	10	NULL	NULL
    2	20	NULL	NULL
    3	20	NULL	NULL
    4	99	NULL	NULL
    5	NULL	NULL	NULL
    """
    var joined = table("emp").join(
        table("dept").filter(col("did", int64) > lit(1000, int64)),
        [1],
        [0],
        JOIN_LEFT,
    )
    return joined.sort_by([col("eid", int64)], [True])
