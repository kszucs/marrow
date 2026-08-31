from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT v, first_value(v) OVER (ORDER BY v NULLS LAST) AS f, last_value(v) OVER (ORDER BY v NULLS LAST) AS l FROM basic ORDER BY v NULLS LAST

    The frame gotcha in its shortest form. Under the default frame
    `first_value` is constant — the frame always starts at the partition's
    first row — while `last_value` is the *current* row, because the frame ends
    there. Everyone expects `last_value` to be the partition's last value and
    it is not.

    Nulls sort last so the constant `first_value` is a real value rather than a
    null, which would have made the two columns harder to tell apart.

    There is no window node in `marrow/expr/logical.mojo` and no windowed
    operator in `physical.mojo`.

    -- skip mojo
    -- skip python

    -- expected
    v:int64	f:int64	l:int64
    1	1	1
    2	1	2
    3	1	3
    4	1	4
    6	1	6
    7	1	7
    NULL	1	NULL
    """
    var t = table("basic")
    var edges = t.with_columns(
        ["f", "l"],
        [
            col("v", int64)
            .first_value()
            .over(order_by=[col("v", int64)], nulls_first=False),
            col("v", int64)
            .last_value()
            .over(order_by=[col("v", int64)], nulls_first=False),
        ],
    )
    return edges.select(["v", "f", "l"]).sort_by(
        [col("v", int64)], [True], nulls_first=False
    )
