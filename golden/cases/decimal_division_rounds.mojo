# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT CAST(CAST(price AS DECIMAL(10, 2)) / 3 AS VARCHAR) AS d FROM sales

    Decimal division cannot be exact, so the engine has to choose a result
    scale and round to it — the one place in decimal arithmetic where
    information is lost, and where two engines that agree on `+`, `-` and `*`
    can still disagree.

    marrow follows Arrow C++ instead: a decimal divided by an integer stays a
    decimal, at scale `max(4, s1 + p2 - s2 + 1)` = 13 here, truncated —
    `'0.5000000000000'`. DuckDB answers `DOUBLE`, and so do these
    expectations.

    -- xfail decimal division is Arrow C++'s exact decimal, DuckDB's is DOUBLE
    -- skip python

    -- expected
    d:string
    '0.5'
    '0.75'
    '0.16666666666666666'
    NULL
    '1.3333333333333333'
    '-0.4166666666666667'
    """
    var t = table("sales")
    var dec = decimal128(10, 2)
    return t.project(["a"], [col("price", float64).cast(dec)]).project(
        ["d"],
        [(col("a", dec) / lit(3, decimal128(10, 0))).cast(string)],
    )
