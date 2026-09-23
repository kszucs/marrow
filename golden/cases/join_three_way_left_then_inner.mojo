from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, m.eid AS meid FROM emp e LEFT JOIN dept d ON e.dept = d.did JOIN emp m ON d.did = m.dept ORDER BY e.eid NULLS FIRST, meid NULLS FIRST

    The mirror of `join_three_way_inner_then_left`, and the one that says why
    the guard is not paranoia. `JoinReassociation` declines on the *inner*
    join's kind, and reassociating anyway would change the answer rather than
    the cost: written this way, `emp` rows 4 and 5 — department 99 and NULL —
    are widened to a null `did` by the LEFT join and then dropped by the inner
    one, for five rows. As `emp LEFT JOIN (dept ⋈ emp)` they would find no
    partner and be widened *again*, for seven.

    -- skip python

    -- expected
    eid:int64	dept:int64	did:int64	meid:int64
    1	10	10	1
    2	20	20	2
    2	20	20	3
    3	20	20	2
    3	20	20	3
    """
    var left = table("emp")
    var staffed = left.join(table("dept"), [1], [0], JOIN_LEFT)
    var mates = table("emp").rename(["eid", "dept"], ["meid", "mdept"])
    var paired = staffed.join(mates^, [2], [1], JOIN_INNER)
    var picked = paired.project(
        ["eid", "dept", "did", "meid"],
        [
            col("eid", int64),
            col("dept", int64),
            col("did", int64),
            col("meid", int64),
        ],
    )
    return picked.sort_by(
        [col("eid", int64), col("meid", int64)], [True, True]
    ).optimize[AllRules]()
