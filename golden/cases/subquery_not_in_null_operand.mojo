# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp WHERE dept NOT IN (SELECT did FROM dept) ORDER BY eid

    `subquery_not_exists`'s question asked with `NOT IN`: eid 5's NULL
    dept is UNKNOWN against a non-empty subquery, so the row that
    `NOT EXISTS` keeps is dropped here.

    -- skip python

    -- expected
    eid:int64	dept:int64
    4	99
    """
    var plan = table("emp").filter(
        not_(is_in(col("dept"), table("dept").select(["did"])))
    )
    return plan.optimize[SubqueryRules]().sort_by([col("eid", int64)], [True])
