from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.kernels.join import JOIN_ALL, JOIN_FULL


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e FULL OUTER JOIN dept d ON e.dept = d.did ORDER BY e.eid NULLS FIRST, d.did NULLS FIRST

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    NULL	NULL	30	'ops'
    1	10	10	'eng'
    2	20	20	'sales'
    3	20	20	'sales'
    4	99	NULL	NULL
    5	NULL	NULL	NULL
    """
    var left = table("emp")
    var right = table("dept")
    var joined = left.join(
        right,
        left_on=[col("dept", int64)],
        right_on=[col("did", int64)],
        how=JOIN_FULL,
        strictness=JOIN_ALL,
    )
    var q = joined.sort([col("eid", int64), col("did", int64)], [True, True])
    return q
