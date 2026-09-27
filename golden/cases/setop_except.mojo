# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT region FROM sales EXCEPT SELECT region FROM regions ORDER BY region NULLS FIRST

    Set difference, also deduplicating and also with NULL equal to itself: the
    NULL region survives because `regions` has none, and `north` disappears
    although `sales` has two of it.

    -- expected
    region:string
    NULL
    'east'
    """
    var t = table("sales").select(["region"])
    var d = t.except_(table("regions").select(["region"]))
    return d.sort_by([col("region", string)], [True])
