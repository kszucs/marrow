# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.region, s.qty, CAST((SELECT count(*) FROM regions r WHERE r.region = s.region) AS BIGINT) AS n FROM sales s ORDER BY s.region NULLS FIRST, s.qty NULLS FIRST

    A subquery that references the outer row. This is the shape that cannot be
    rewritten as a semi- or anti-join — it produces a *value* per outer row,
    not a filter — and the unmatched rows answer 0, which is `count(*)`'s
    answer over an empty relation rather than a null from an outer join.

    -- skip python

    -- expected
    region:string	qty:int32	n:int64
    NULL	40	0
    'east'	50	0
    'north'	NULL	1
    'north'	10	1
    'south'	5	1
    'south'	20	1
    """
    var counts = (
        table("regions")
        .filter(eq(col("region"), outer("region")))
        .aggregate(aggs=[count_star().alias("n")])
    )
    var plan = table("sales").with_columns(["n"], [scalar(counts)])
    return (
        plan.select(["region", "qty", "n"])
        .optimize[SubqueryRules]()
        .sort_by([col("region", string), col("qty", int32)], [True, True])
    )
