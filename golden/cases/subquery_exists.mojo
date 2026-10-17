# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp e WHERE EXISTS (SELECT 1 FROM dept d WHERE d.did = e.dept) ORDER BY eid

    `EXISTS` and `IN` agree here because the correlation is the join
    condition itself.

    -- skip python

    -- expected
    eid:int64	dept:int64
    1	10
    2	20
    3	20
    """
    var matching = table("dept").filter(eq(col("did"), outer("dept")))
    var plan = table("emp").filter(exists(matching))
    return plan.optimize[SubqueryRules]().sort_by([col("eid", int64)], [True])
