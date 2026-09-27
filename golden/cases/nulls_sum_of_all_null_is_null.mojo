# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(a) AS BIGINT) AS total FROM nulls

    -- expected
    total:int64
    NULL
    """
    var t = table("nulls")
    return t.aggregate(
        aggs=[col("a", int64).sum().alias("total")],
    )
