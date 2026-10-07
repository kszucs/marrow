# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept, d.did, s.ref, s.qty FROM emp e JOIN dept d ON e.dept = d.did JOIN sales s ON e.eid = s.ref ORDER BY e.eid NULLS FIRST, s.qty NULLS FIRST

    Three inner joins again, and a star this time: both key on `emp`, so
    `dept ⋈ sales` has no key at all. A reassociation into it would join on a
    column that is not in scope; the join search only ever joins sets of
    leaves a key connects, so its trees here are the ones that take `emp`
    first.

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
