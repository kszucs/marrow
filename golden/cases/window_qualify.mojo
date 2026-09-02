from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v FROM basic QUALIFY row_number() OVER (PARTITION BY k ORDER BY v NULLS FIRST) = 1 ORDER BY k NULLS FIRST

    `QUALIFY` filters on a window function's output, standing to `OVER` as
    `HAVING` stands to `GROUP BY`. It is the idiomatic "one row per group,
    chosen by an order" — the same answer `DISTINCT ON` gives, reached the
    other way.

    -- skip python

    -- expected
    k:string	v:int64
    NULL	7
    'a'	1
    'b'	NULL
    'c'	4
    """
    var t = table("basic")
    var numbered = t.with_columns(
        ["rn"],
        [
            row_number().over(
                partition_by=[col("k", string)], order_by=[col("v", int64)]
            )
        ],
    )
    var first = numbered.filter(col("rn", int64) == lit(1, int64))
    return first.select(["k", "v"]).sort_by([col("k", string)], [True])
