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
    var right = table("dept")
    var joined = left.join(
        right,
        left_on=[col("dept", int64)],
        right_on=[col("did", int64)],
        how=JOIN_SEMI,
        strictness=JOIN_ALL,
    )
    var q = joined.sort([col("eid", int64)], [True])
    return q
