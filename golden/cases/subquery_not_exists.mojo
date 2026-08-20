from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp e WHERE NOT EXISTS (SELECT 1 FROM dept d WHERE d.did = e.dept) ORDER BY eid

    The twin is `NOT EXISTS`, **not** `NOT IN`, and the difference is the
    point. An anti join keeps a row whose key is NULL, because NULL
    matches nothing; `NOT IN` would drop it, because `NULL NOT IN (...)`
    is UNKNOWN rather than true. `NOT EXISTS` has the anti-join meaning,
    so it is the twin that asks marrow's question. eid 5 (NULL dept) is
    the row that separates them.

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
