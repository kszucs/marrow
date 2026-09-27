# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT avg(b) AS m FROM nulls

    -- expected
    m:double
    5.0
    """
    var t = table("nulls")
    return t.aggregate(
        aggs=[col("b", int64).mean().alias("m")],
    )
