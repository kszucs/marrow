from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, d.dname FROM emp e LEFT JOIN dept d ON e.dept = d.did AND d.dname <> 'eng' ORDER BY e.eid NULLS FIRST

    An outer join whose `ON` clause carries a non-key predicate. The predicate
    has to be applied **before** the null-widening: `eid = 1` matches `did =
    10` on the key but fails `dname <> 'eng'`, so it comes back null-widened
    rather than disappearing. Moving that predicate to a `WHERE` — the rewrite
    that looks equivalent — would drop the row instead, and that is the bug
    this records.

    -- skip mojo
    -- skip python

    -- expected
    eid:int64	dept:int64	did:int64	dname:string
    1	10	NULL	NULL
    2	20	20	'sales'
    3	20	20	'sales'
    4	99	NULL	NULL
    5	NULL	NULL	NULL
    """
    var joined = table("emp").join_on(
        table("dept"),
        (col("dept", int64) == col("did", int64))
        & (col("dname", string) != lit("eng", string)),
        JOIN_LEFT,
    )
    return joined.sort_by([col("eid", int64)], [True])
