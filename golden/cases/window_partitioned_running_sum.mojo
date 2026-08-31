from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, CAST(sum(v) OVER (PARTITION BY k ORDER BY v NULLS FIRST) AS BIGINT) AS running FROM basic ORDER BY k NULLS FIRST, v NULLS FIRST

    An aggregate used as a window function, with a partition and an order but
    no explicit frame — so the default `RANGE UNBOUNDED PRECEDING TO CURRENT
    ROW` applies and the column is a running total that restarts at each `k`.
    The NULL partition is its own group, exactly as in `GROUP BY`.

    -- skip python

    -- expected
    k:string	v:int64	running:int64
    NULL	7	7
    'a'	1	1
    'a'	3	4
    'a'	6	10
    'b'	NULL	NULL
    'b'	2	2
    'c'	4	4
    """
    var t = table("basic")
    var running = t.with_columns(
        ["running"],
        [
            col("v", int64)
            .sum()
            .over(partition_by=[col("k", string)], order_by=[col("v", int64)])
        ],
    )
    return running.select(["k", "v", "running"]).sort_by(
        [col("k", string), col("v", int64)], [True, True]
    )
