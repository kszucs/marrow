from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT k, CAST(rank() OVER (ORDER BY k NULLS FIRST) AS BIGINT) AS r, CAST(dense_rank() OVER (ORDER BY k NULLS FIRST) AS BIGINT) AS d FROM basic ORDER BY k NULLS FIRST

    The two ranking functions differ only on ties, which is the whole question:
    `rank` leaves gaps after a tied run and `dense_rank` does not. `basic.k`
    has three `a`s and two `b`s, so the two columns separate immediately.

    There is no window node in `marrow/expr/logical.mojo` and no windowed
    operator in `physical.mojo`.

    -- skip mojo

    -- expected
    k:string	r:int64	d:int64
    NULL	1	1
    'a'	2	2
    'a'	2	2
    'a'	2	2
    'b'	5	3
    'b'	5	3
    'c'	7	4
    """
    var t = table("basic")
    var ranked = t.with_columns(
        ["r", "d"],
        [
            rank().over(order_by=[col("k", string)]),
            dense_rank().over(order_by=[col("k", string)]),
        ],
    )
    return ranked.select(["k", "r", "d"]).sort_by([col("k", string)], [True])
