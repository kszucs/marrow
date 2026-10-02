# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales_view ORDER BY region DESC NULLS LAST, qty NULLS LAST

    -- expected
    qty:int32
    5
    20
    10
    NULL
    50
    40
    """
    var t = table("sales_view")
    var sorted = t.sort_by(
        [col("region", string_view), col("qty", int32)],
        [False, True],
        nulls_first=False,
    )
    return sorted.project(["qty"], [col("qty", int32)])
