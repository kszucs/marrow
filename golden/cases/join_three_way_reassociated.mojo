from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.ref, s.qty, s.price, e.eid, e.dept, d.did FROM sales s JOIN emp e ON s.ref = e.eid JOIN dept d ON e.dept = d.did ORDER BY s.ref NULLS FIRST, s.qty NULLS FIRST

    `join_three_way_chain`'s query, handed to the rewriter, and the only case
    in the corpus that runs a plan the optimizer touched. `JoinReassociation`
    fires: written `(sales ⋈ emp) ⋈ dept`, it comes back as
    `sales ⋈ (emp ⋈ dept)`, and `SelectBuildSide` then indexes the right input
    of both joins where the plan as written indexed the left. Three physical
    decisions change; the expectation is `join_three_way_chain`'s, character
    for character, because none of them may change the answer.

    The rewrite is legal because the second predicate reads `e.dept` — a
    column of the middle input and of neither of the other two.
    `join_three_way_key_on_first_input` is the same shape with exactly that
    condition broken.

    -- skip python

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
    return picked.sort_by(
        [col("ref", int64), col("qty", int32)], [True, True]
    ).optimize[AllRules]()
