# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.ref, s.qty, s.price, e.eid, e.dept, d.did FROM sales s JOIN emp e ON s.ref = e.eid JOIN dept d ON e.dept = d.did ORDER BY s.ref NULLS FIRST, s.qty NULLS FIRST

    `join_three_way_chain`'s query, handed to the rewriter. The join search
    may take the region apart — `(sales ⋈ emp) ⋈ dept` has a second tree,
    `sales ⋈ (emp ⋈ dept)` — and may index either input of each join; the
    expectation is `join_three_way_chain`'s, character for character, because
    none of those decisions may change the answer.

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
