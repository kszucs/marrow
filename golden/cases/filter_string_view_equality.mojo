# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT qty FROM sales_view WHERE region = 'north' ORDER BY qty NULLS FIRST

    `region` is a `string_view` column compared with a `string` literal.

    -- expected
    qty:int32
    NULL
    10
    """
    var t = table("sales_view")
    var filtered = t.filter(col("region", string_view) == lit("north", string))
    var picked = filtered.project(["qty"], [col("qty", int32)])
    return picked.sort_by([col("qty", int32)], [True])
