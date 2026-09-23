from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.ref, s.qty, s.price, e.eid, e.dept, d.did FROM sales s JOIN emp e ON s.ref = e.eid JOIN dept d ON e.dept = d.did ORDER BY s.ref NULLS FIRST, s.qty NULLS FIRST

    The corpus's first query with more than one join. Left-deep as written,
    and run exactly as written — `execute()` applies no rules — so this is the
    answer every rewrite of the same query has to reproduce, and
    `join_three_way_reassociated` is that same query after the rewriter has
    turned it inside out.

    -- expected
    ref:int64	qty:int32	price:double	eid:int64	dept:int64	did:int64
    1	10	1.5	1	10	10
    2	NULL	0.5	2	20	20
    2	20	2.25	2	20	20
    3	40	NULL	3	20	20
    """
    var left = table("sales")
    var employed = left.join(table("emp"), [4], [0], JOIN_INNER)
    var staffed = employed.join(table("dept"), [6], [0], JOIN_INNER)
    var picked = staffed.project(
        ["ref", "qty", "price", "eid", "dept", "did"],
        [
            col("ref", int64),
            col("qty", int32),
            col("price", float64),
            col("eid", int64),
            col("dept", int64),
            col("did", int64),
        ],
    )
    return picked.sort_by([col("ref", int64), col("qty", int32)], [True, True])
