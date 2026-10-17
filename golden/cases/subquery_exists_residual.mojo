# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT e.eid, e.dept FROM emp e WHERE EXISTS (SELECT 1 FROM emp o WHERE o.dept = e.dept AND o.eid <> e.eid) ORDER BY e.eid

    A correlation that is not only an equality: another employee of the
    same department. The equality is the join key and the inequality is
    evaluated per matching pair, so a department of one does not count
    itself.

    -- skip python

    -- expected
    eid:int64	dept:int64
    2	20
    3	20
    """
    var others = (
        table("emp")
        .filter(eq(col("dept"), outer("dept")))
        .filter(ne(col("eid"), outer("eid")))
    )
    var plan = table("emp").filter(exists(others))
    return plan.optimize[SubqueryRules]().sort_by([col("eid", int64)], [True])
