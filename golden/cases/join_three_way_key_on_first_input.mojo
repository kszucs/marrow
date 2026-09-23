from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, s.ref, s.qty FROM emp e JOIN dept d ON e.dept = d.did JOIN sales s ON e.eid = s.ref ORDER BY e.eid NULLS FIRST, s.qty NULLS FIRST

    Three inner joins again, and this time the rewriter must leave them alone:
    the outer predicate reads `eid`, a column of the *first* input, which has
    nowhere to go in `dept ⋈ sales`. Every other condition holds and the cost
    model prefers the right-deep form, so the guard is the only thing standing
    between this plan and a join on a column that is not in scope.

    -- skip python

    -- expected
    eid:int64	dept:int64	did:int64	ref:int64	qty:int32
    1	10	10	1	10
    2	20	20	2	NULL
    2	20	20	2	20
    3	20	20	3	40
    """
    var left = table("emp")
    var staffed = left.join(table("dept"), [1], [0], JOIN_INNER)
    var sold = staffed.join(table("sales"), [0], [4], JOIN_INNER)
    var picked = sold.project(
        ["eid", "dept", "did", "ref", "qty"],
        [
            col("eid", int64),
            col("dept", int64),
            col("did", int64),
            col("ref", int64),
            col("qty", int32),
        ],
    )
    return picked.sort_by(
        [col("eid", int64), col("qty", int32)], [True, True]
    ).optimize[AllRules]()
