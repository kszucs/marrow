from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, v, CAST(row_number() OVER (ORDER BY v NULLS FIRST, w NULLS FIRST) AS BIGINT) AS rn FROM basic ORDER BY rn

    The simplest window: a dense position in a total order, with no partition
    and no frame. The second sort key makes the numbering deterministic, which
    a window over a non-total order would not be.

    -- skip python

    -- expected
    k:string	v:int64	rn:int64
    'b'	NULL	1
    'a'	1	2
    'b'	2	3
    'a'	3	4
    'c'	4	5
    'a'	6	6
    NULL	7	7
    """
    var t = table("basic")
    var numbered = t.with_columns(
        ["rn"],
        [row_number().over(order_by=[col("v", int64), col("w", int64)])],
    )
    return numbered.sort_by([col("rn", int64)], [True])
