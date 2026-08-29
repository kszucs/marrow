from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT v, lag(v) OVER (ORDER BY v NULLS FIRST) AS prev, lead(v) OVER (ORDER BY v NULLS FIRST) AS next FROM basic ORDER BY v NULLS FIRST

    Reading a neighbouring row. The two ends are the interesting rows: `lag` is
    null on the first and `lead` on the last, and neither of those nulls means
    "the value there was null" — which the null `v` in the middle of the
    ordering does.

    There is no window node in `marrow/expr/logical.mojo` and no windowed
    operator in `physical.mojo`.

    -- skip mojo

    -- expected
    v:int64	prev:int64	next:int64
    NULL	NULL	1
    1	NULL	2
    2	1	3
    3	2	4
    4	3	6
    6	4	7
    7	6	NULL
    """
    var t = table("basic")
    var shifted = t.with_columns(
        ["prev", "next"],
        [
            col("v", int64).lag().over(order_by=[col("v", int64)]),
            col("v", int64).lead().over(order_by=[col("v", int64)]),
        ],
    )
    return shifted.select(["v", "prev", "next"]).sort_by(
        [col("v", int64)], [True]
    )
