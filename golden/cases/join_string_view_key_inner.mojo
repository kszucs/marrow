# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT s.qty, r.country FROM sales_view s JOIN regions_view r ON s.region = r.region ORDER BY s.qty NULLS FIRST

    A hash join keyed on `string_view` on both sides.

    -- expected
    qty:int32	country:string
    NULL	'ca'
    5	'mx'
    10	'ca'
    20	'mx'
    """
    var left = table("sales_view")
    var joined = left.join(table("regions_view"), [0], [0], JOIN_INNER)
    var picked = joined.project(
        ["qty", "country"], [col("qty", int32), col("country", string)]
    )
    return picked.sort_by([col("qty", int32)], [True])
