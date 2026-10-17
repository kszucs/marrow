# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT eid, dept FROM emp WHERE dept IN (SELECT did FROM dept WHERE did IN (SELECT dept FROM emp WHERE eid > 2)) ORDER BY eid

    A subquery inside a subquery. The innermost answers 20, 99 and NULL;
    only department 20 is among them, and the NULL makes every other
    department's membership UNKNOWN rather than true.

    -- skip python

    -- expected
    eid:int64	dept:int64
    2	20
    3	20
    """
    var late = table("emp").filter(col("eid", int64) > lit(2, int64))
    var depts = table("dept").filter(is_in(col("did"), late.select(["dept"])))
    var plan = table("emp").filter(is_in(col("dept"), depts.select(["did"])))
    return plan.optimize[SubqueryRules]().sort_by([col("eid", int64)], [True])
