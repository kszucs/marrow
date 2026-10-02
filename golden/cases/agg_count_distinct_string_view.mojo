# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(count(DISTINCT region) AS BIGINT) AS n FROM sales_view

    -- expected
    n:int64
    3
    """
    var t = table("sales_view")
    return t.aggregate(
        aggs=[col("region", string_view).count_distinct().alias("n")]
    )
