from golden.helpers import table
from marrow.dtypes import int64
from marrow.expr.builders import col
from marrow.expr.relations import DynRelation
from marrow.kernels.join import JOIN_ALL, JOIN_ANTI


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept FROM emp e WHERE NOT EXISTS (SELECT 1 FROM dept d WHERE e.dept = d.did) ORDER BY e.eid NULLS FIRST

    -- expected
    eid:int64	dept:int64
    4	99
    5	NULL
    """
    var left = table("emp")
    var right = table("dept")
    var joined = left.join(
        right,
        left_on=[col("dept", int64)],
        right_on=[col("did", int64)],
        how=JOIN_ANTI,
        strictness=JOIN_ALL,
    )
    var q = joined.sort([col("eid", int64)], [True])
    return q
