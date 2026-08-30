from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(CASE WHEN v > 4 THEN 1 WHEN v > 2 THEN 2 ELSE 3 END AS BIGINT) AS c FROM basic

    A multi-branch `CASE`, which the fused lane spells as nested two-armed
    nodes. The first matching branch wins, so the ordering of the two
    conditions is observable: `v = 6` satisfies both and must take the first.

    The null row takes the `ELSE` branch, because a null condition counts as
    false at *every* level — the rule `cond_case_when` states for one branch,
    applied twice here.

    -- expected
    c:int64
    3
    3
    2
    2
    3
    1
    1
    """
    var t = table("basic")
    return t.project(
        ["c"],
        [
            CaseWhen(
                col("v", int64) > lit(4, int64),
                lit(1, int64),
                CaseWhen(
                    col("v", int64) > lit(2, int64),
                    lit(2, int64),
                    lit(3, int64),
                ),
            )
        ],
    )
