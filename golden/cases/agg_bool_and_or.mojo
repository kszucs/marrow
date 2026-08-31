from golden.prelude import *


def plan() raises -> DynRelation:
    """
    SELECT bool_and(active) AS a, bool_or(active) AS o FROM sales

    The boolean reductions. `marrow/kernels/aggregate.mojo` **has** `AnyKernel`
    and `AllKernel`, so this is a missing expression node rather than a missing
    kernel — `BoolValue` exposes no aggregate methods at all.

    Nulls are skipped, which is what makes `bool_and` false here (there is a
    `False` among the non-null values) rather than null.

    -- skip mojo
    -- skip python

    -- expected
    a:bool	o:bool
    False	True
    """
    var t = table("sales")
    return t.aggregate(
        aggs=[
            col("active", bool_).all().alias("a"),
            col("active", bool_).any().alias("o"),
        ]
    )
