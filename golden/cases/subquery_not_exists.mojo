# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

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

    -- skip python

    -- expected
    eid:int64	dept:int64
    4	99
    5	NULL
    """
    var matching = table("dept").filter(eq(col("did"), outer("dept")))
    var plan = table("emp").filter(not_(exists(matching)))
    return plan.optimize[SubqueryRules]().sort_by([col("eid", int64)], [True])
