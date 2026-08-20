from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.kernels.join import JOIN_ALL, JOIN_INNER


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e JOIN dept d ON e.dept = d.did ORDER BY e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    1	10	10	'eng'
    2	20	20	'sales'
    3	20	20	'sales'
    """
    var left = table("emp")
    var right = table("dept")
    var joined = left.join(
        right,
        left_on=[col("dept", int64)],
        right_on=[col("did", int64)],
        how=JOIN_INNER,
        strictness=JOIN_ALL,
    )
    var q = joined.sort([col("eid", int64)], [True])
    return q
