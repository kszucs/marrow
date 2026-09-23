from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.ref, s.qty, e.eid, e.dept, d.did, m.eid AS meid FROM sales s JOIN emp e ON s.ref = e.eid JOIN dept d ON e.dept = d.did JOIN emp m ON d.did = m.dept ORDER BY s.ref NULLS FIRST, s.qty NULLS FIRST, meid NULLS FIRST

    Four inputs, so the rewriter has two associations to choose between rather
    than one and has to reach a fixpoint instead of a single rewrite. A
    left-deep chain of three joins is `Join(Join(Join(...)))`, and
    `JoinReassociation` only ever matches a join whose *left* input is a join —
    so what it can reach at all is the question this case asks, and the answer
    it has to keep giving while it reaches more.

    -- skip python

    -- expected
    ref:int64	qty:int32	eid:int64	dept:int64	did:int64	meid:int64
    1	10	1	10	10	1
    2	NULL	2	20	20	2
    2	NULL	2	20	20	3
    2	20	2	20	20	2
    2	20	2	20	20	3
    3	40	3	20	20	2
    3	40	3	20	20	3
    """
    var left = table("sales")
    var employed = left.join(table("emp"), [4], [0], JOIN_INNER)
    var staffed = employed.join(table("dept"), [6], [0], JOIN_INNER)
    var mates = table("emp").rename(["eid", "dept"], ["meid", "mdept"])
    var paired = staffed.join(mates^, [7], [1], JOIN_INNER)
    var picked = paired.project(
        ["ref", "qty", "eid", "dept", "did", "meid"],
        [
            col("ref", int64),
            col("qty", int32),
            col("eid", int64),
            col("dept", int64),
            col("did", int64),
            col("meid", int64),
        ],
    )
    return picked.sort_by(
        [col("ref", int64), col("qty", int32), col("meid", int64)],
        [True, True, True],
    ).optimize[AllRules]()
