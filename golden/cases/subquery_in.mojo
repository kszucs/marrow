# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp WHERE dept IN (SELECT did FROM dept) ORDER BY eid

    `IN (subquery)` is a semi join: each left row appears once however
    many right rows it matches, and a NULL key matches nothing.

    -- skip python

    -- expected
    eid:int64	dept:int64
    1	10
    2	20
    3	20
    """
    var plan = table("emp").filter(
        is_in(col("dept"), table("dept").select(["did"]))
    )
    return plan.optimize[SubqueryRules]().sort_by([col("eid", int64)], [True])
