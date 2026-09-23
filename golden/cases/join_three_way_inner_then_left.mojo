from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.ref, s.qty, e.eid, e.dept, g.i, g.j FROM sales s JOIN emp e ON s.ref = e.eid LEFT JOIN edges g ON e.eid = g.i ORDER BY s.ref NULLS FIRST, s.qty NULLS FIRST

    Two kinds in one chain. `JoinReassociation` declines on the outer join's
    kind before it looks at anything else: associativity is a property of the
    inner join as a multiset operation, and a LEFT join manufactures rows for
    non-matches at a moment the association would move.

    `SelectBuildSide` is not blocked by any of that and flips the LEFT join to
    index its right input — `edges` is five rows of sixteen bytes against an
    intermediate the model puts at five rows of twenty-eight — so the plan
    that runs here is still not the plan that was written.

    Only `eid` 1 finds a partner in `edges.i`; the other three rows widen, so
    the case also pins that the kind survived the rewrite.

    -- skip python

    -- expected
    ref:int64	qty:int32	eid:int64	dept:int64	i:int64	j:int64
    1	10	1	10	1	3
    2	NULL	2	20	NULL	NULL
    2	20	2	20	NULL	NULL
    3	40	3	20	NULL	NULL
    """
    var left = table("sales")
    var employed = left.join(table("emp"), [4], [0], JOIN_INNER)
    var edged = employed.join(table("edges"), [5], [0], JOIN_LEFT)
    var picked = edged.project(
        ["ref", "qty", "eid", "dept", "i", "j"],
        [
            col("ref", int64),
            col("qty", int32),
            col("eid", int64),
            col("dept", int64),
            col("i", int64),
            col("j", int64),
        ],
    )
    return picked.sort_by(
        [col("ref", int64), col("qty", int32)], [True, True]
    ).optimize[AllRules]()
