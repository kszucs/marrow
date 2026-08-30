from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT first(k ORDER BY v NULLS FIRST) AS f, last(k ORDER BY v NULLS FIRST) AS l FROM basic

    `first` and `last` are only meaningful with an `ORDER BY` inside the
    aggregate — without one they return an arbitrary row and no golden case
    could assert them. That ordered form is a second shape gap: an aggregate
    that carries its own sort.

    -- skip mojo

    -- expected
    f:string	l:string
    'b'	NULL
    """
    var t = table("basic")
    return t.aggregate(
        aggs=[
            col("k", string).first(order_by=[col("v", int64)]).alias("f"),
            col("k", string).last(order_by=[col("v", int64)]).alias("l"),
        ]
    )
