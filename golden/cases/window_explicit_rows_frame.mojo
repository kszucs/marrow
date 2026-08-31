from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT v, CAST(sum(v) OVER (ORDER BY v NULLS FIRST ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS BIGINT) AS s FROM basic ORDER BY v NULLS FIRST

    An explicit `ROWS` frame, which is a different question from the default
    `RANGE` one: `ROWS` counts rows and `RANGE` counts *peers*, so the two
    agree only when the order key has no duplicates. A two-row sliding sum is
    the smallest frame that shows the window moving.

    -- skip python

    -- expected
    v:int64	s:int64
    NULL	NULL
    1	1
    2	3
    3	5
    4	7
    6	10
    7	13
    """
    var t = table("basic")
    var framed = t.with_columns(
        ["s"],
        [col("v", int64).sum().over(order_by=[col("v", int64)], rows=(-1, 0))],
    )
    return framed.select(["v", "s"]).sort_by([col("v", int64)], [True])
