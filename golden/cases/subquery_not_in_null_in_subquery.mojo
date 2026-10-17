# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT did, dname FROM dept WHERE did NOT IN (SELECT dept FROM emp) ORDER BY did

    A NULL among the subquery's rows makes `NOT IN` UNKNOWN for every value
    it does not equal, so nothing survives: did 30 is in no department list
    and is still dropped. An anti join alone would keep it.

    -- skip python

    -- expected
    did:int64	dname:string
    """
    var plan = table("dept").filter(
        not_(is_in(col("did"), table("emp").select(["dept"])))
    )
    return plan.optimize[SubqueryRules]().sort_by([col("did", int64)], [True])
