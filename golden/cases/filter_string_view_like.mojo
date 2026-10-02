# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales_view WHERE region LIKE 's%' ORDER BY qty

    -- expected
    qty:int32
    5
    20
    """
    var t = table("sales_view")
    var filtered = t.filter(col("region", string_view).like(lit("s%", string)))
    var picked = filtered.project(["qty"], [col("qty", int32)])
    return picked.sort_by([col("qty", int32)], [True])
