# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(sum(CAST(price AS DECIMAL(10, 2))) AS VARCHAR) AS total FROM sales

    Summing a decimal is exact and keeps the scale — `1.50 + 2.25 + ...` is not
    a float sum that happens to look right. The twin renders the result as text
    so that the *scale* is asserted and not just the value: `7.00` and `7.0`
    are the same number and different answers.

    -- skip python

    -- expected
    total:string
    '7.00'
    """
    var t = table("sales")
    var dec = decimal128(10, 2)
    return (
        t.project(["d"], [col("price", float64).cast(dec)])
        .aggregate(aggs=[col("d", dec).sum().alias("total")])
        .project(["total"], [col("total", decimal128(38, 2)).cast(string)])
    )
